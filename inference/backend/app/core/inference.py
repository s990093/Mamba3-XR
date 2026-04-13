from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import AsyncIterator, Callable, TypedDict

from app.core.inference_status import inference_status
from app.schemas.settings import InferenceSettings


class InferenceCancelled(Exception):
    pass


class InferenceEvent(TypedDict, total=False):
    type: str
    value: str | dict[str, object]


def _metrics(
    ttft: float,
    tpot: float,
    decode_tps: float,
    *,
    prefill_tps: float = 0.0,
) -> dict[str, object]:
    return {
        "architecture": {
            "precision": "mlx",
            "total_params_m": 546.1,
            "active_params_m": 234.08,
            "active_percentage": 42.86,
        },
        "performance": {
            "ttft_s": round(ttft, 4),
            "tpot_s": round(tpot, 4),
            "prefill_tps": round(prefill_tps, 2),
            "prefill_tflops": 0.0,
            "decode_tps": round(decode_tps, 2),
            "decode_tflops": 0.0,
        },
        "memory": {
            "framework": "MLX",
            "active_mem_mb": 0.0,
            "peak_mem_mb": 0.0,
        },
    }


class _Done:
    pass


class _InfRuntime:
    def __init__(self) -> None:
        self._ready = False
        self._mx = None
        self._model = None
        self._tokenizer = None
        self._router_temp = None
        self._run_prefill = None
        self._stream_mod = None
        self._target_dtype = None
        self._kv_dtype = None
        self._base_cache_len = int(os.getenv("INFERENCE_MAX_CACHE_LEN", "8192"))
        self._no_eos_stop = os.getenv("INFERENCE_NO_EOS_STOP", "0").strip() == "1"

    def ensure_ready(self) -> None:
        if self._ready:
            return

        inference_status.set_loading("Loading MLX runtime…")
        repo_root = Path(__file__).resolve().parents[4]
        inf_dir = repo_root / "inference"
        if str(inf_dir) not in sys.path:
            sys.path.insert(0, str(inf_dir))

        import mlx.core as mx
        import mlx.nn as nn
        from transformers import AutoTokenizer

        import stream_mlx as stream_mod
        from mlx_hybrid_infer import (
            Mamba3Config,
            Mamba3LanguageModel,
            attach_decode_compilation,
            maybe_export_npz_sidecar_after_pt_load,
            resolve_mlx_checkpoint,
            strict_load_and_convert,
        )

        tokenizer_path = os.getenv("INFERENCE_TOKENIZER_PATH", str(inf_dir / "tokenizer"))
        ckpt = os.getenv("INFERENCE_CKPT", str(repo_root / "checkpoint.pt"))
        npz_cache = os.getenv("INFERENCE_NPZ_CACHE", "")
        force_pt = os.getenv("INFERENCE_FORCE_PT", "0").strip() == "1"
        dtype_key = os.getenv("INFERENCE_DTYPE", "bf16").lower()
        kv_key = os.getenv("INFERENCE_KV_DTYPE", "auto").lower()
        quant_bits = int(os.getenv("INFERENCE_QUANT_BITS", "4"))
        quant_bits = quant_bits if quant_bits in (0, 4, 8) else 4
        router_temp_val = float(os.getenv("INFERENCE_ROUTER_TEMP", "0.5"))
        full_decode_compile = os.getenv("INFERENCE_FULL_DECODE_COMPILE", "1").strip() == "1"
        compile_prefill = os.getenv("INFERENCE_COMPILE_PREFILL", "1").strip() == "1"

        inference_status.set_loading("Loading tokenizer…")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        vocab_size = len(tokenizer)

        dtype_map = {"fp32": mx.float32, "bf16": mx.bfloat16, "fp16": mx.float16}
        target_dtype = dtype_map.get(dtype_key, mx.bfloat16)
        kv_map = {"bf16": mx.bfloat16, "fp16": mx.float16, "fp32": mx.float32}
        kv_dtype = target_dtype if kv_key == "auto" else kv_map.get(kv_key, mx.bfloat16)

        inference_status.set_loading("Building model architecture…")
        config = Mamba3Config(
            d_model=768,
            d_state=64,
            d_head=64,
            expand=2,
            num_layers=6,
            mimo_rank=4,
            num_kv_heads=4,
            use_parallel_scan=True,
            chunk_size=64,
            use_kmoe=True,
            kmoe_num_experts=8,
            kmoe_top_k=2,
            kmoe_r1=32,
            kmoe_r2=512,
            kmoe_r3=256,
            ffn_expand=6,
        )
        model = Mamba3LanguageModel(config, vocab_size)

        resolved, kind = resolve_mlx_checkpoint(
            ckpt,
            repo_root=str(repo_root),
            npz_cache=npz_cache,
            force_pt=force_pt,
        )
        if resolved is not None and kind != "none":
            strict_load_and_convert(model, resolved)
            if kind == "pt":
                maybe_export_npz_sidecar_after_pt_load(model, resolved, force_refresh=force_pt)

        model.apply(lambda x: x.astype(target_dtype))
        if quant_bits > 0:
            nn.quantize(model, group_size=64, bits=quant_bits)
        mx.eval(model.parameters())
        stream_mod._invalidate_tucker_caches(model)

        router_temp = mx.array(router_temp_val, dtype=target_dtype)
        attach_decode_compilation(
            model,
            max_cache_len=self._base_cache_len,
            kv_dtype=kv_dtype,
            compile_decode=not full_decode_compile,
        )

        def prefill_forward(x: mx.array, rt: mx.array):
            return model(x, caches=None, seq_pos=None, router_temp=rt)

        run_prefill = mx.compile(prefill_forward) if compile_prefill else prefill_forward

        self._mx = mx
        self._model = model
        self._tokenizer = tokenizer
        self._router_temp = router_temp
        self._run_prefill = run_prefill
        self._stream_mod = stream_mod
        self._target_dtype = target_dtype
        self._kv_dtype = kv_dtype
        self._ready = True
        inference_status.set_ready("Model loaded — ready for prompts")

    def run_generate_streaming(
        self,
        prompt: str,
        settings: InferenceSettings,
        on_chunk: Callable[[str], None],
        should_stop: Callable[[], bool],
    ) -> dict[str, object]:
        self.ensure_ready()
        print(
            "[inference] settings_applied:",
            {
                "temperature": settings.temperature,
                "top_k": settings.top_k,
                "top_p": settings.top_p,
                "min_p": settings.min_p,
                "rep_pen": settings.rep_pen,
                "pres_pen": settings.pres_pen,
                "freq_pen": settings.freq_pen,
                "max_tokens": settings.max_tokens,
                "no_eos_stop": settings.no_eos_stop,
            },
        )
        inference_status.set_generating("Running stream_mlx generator…")
        mx = self._mx
        tokenizer = self._tokenizer
        stream_mod = self._stream_mod

        prompt_ids = stream_mod._build_prompt_ids(tokenizer, prompt, 0)
        x_prefill = mx.array([prompt_ids], dtype=mx.int32)
        max_cache_len = max(self._base_cache_len, len(prompt_ids) + settings.max_tokens + 8)

        eos_id = getattr(tokenizer, "eos_token_id", None)
        if isinstance(eos_id, (list, tuple)):
            eos_id = eos_id[0] if eos_id else None
        if settings.no_eos_stop or self._no_eos_stop:
            eos_id = None

        sample_args = SimpleNamespace(
            temp=float(settings.temperature),
            top_k=int(settings.top_k),
            top_p=float(settings.top_p),
            min_p=float(settings.min_p),
            rep_pen=float(settings.rep_pen),
            pres_pen=float(settings.pres_pen),
            freq_pen=float(settings.freq_pen),
            fast_sample=False,
            no_materialize_caches=False,
        )

        t_prefill0 = time.perf_counter()
        logits, caches = self._run_prefill(x_prefill, self._router_temp)
        mx.eval(logits, caches)
        prefill_s = max(time.perf_counter() - t_prefill0, 1e-9)
        prefill_tps = len(prompt_ids) / prefill_s

        t0 = time.perf_counter()
        first_t = None
        emitted = 0
        token_ids_for_text: list[int] = []
        prev_text = ""
        special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])

        for tok in stream_mod.generate_token_stream(
            model=self._model,
            run_prefill=self._run_prefill,
            x_prefill=x_prefill,
            prompt_ids=prompt_ids,
            max_cache_len=max_cache_len,
            router_temp=self._router_temp,
            max_new_tokens=int(settings.max_tokens),
            sample_args=sample_args,
            eos_token_id=eos_id,
            use_compiled_decode_step=True,
            prefill_outputs=(logits, caches),
        ):
            if should_stop():
                raise InferenceCancelled()
            mx.eval(tok)
            tid = int(tok.item())
            emitted += 1
            now = time.perf_counter()
            if first_t is None:
                first_t = now

            if tid in special_ids:
                try:
                    chunk = f"<{tokenizer.convert_ids_to_tokens(tid)}>"
                except Exception:
                    chunk = f"<{tid}>"
            else:
                token_ids_for_text.append(tid)
                full_text = tokenizer.decode(
                    token_ids_for_text,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                chunk = full_text[len(prev_text) :] if full_text.startswith(prev_text) else full_text
                prev_text = full_text
                if chunk == "":
                    continue
            on_chunk(chunk)

        t1 = time.perf_counter()
        ttft = (first_t - t0) if first_t is not None else 0.0
        decode_time = max(t1 - (first_t or t0), 1e-9)
        tpot = decode_time / max(emitted, 1)
        decode_tps = emitted / decode_time
        inference_status.set_ready("Model ready")
        return _metrics(ttft, tpot, decode_tps, prefill_tps=prefill_tps)


_runtime = _InfRuntime()
_runtime_lock = threading.Lock()


def warmup_runtime() -> None:
    with _runtime_lock:
        _runtime.ensure_ready()


def warmup_runtime_async() -> threading.Thread:
    thread = threading.Thread(target=warmup_runtime, name="inf-runtime-warmup", daemon=True)
    thread.start()
    return thread


def _run_inf_streaming(
    prompt: str,
    settings: InferenceSettings,
    on_chunk: Callable[[str], None],
    should_stop: Callable[[], bool],
) -> dict[str, object]:
    with _runtime_lock:
        return _runtime.run_generate_streaming(prompt, settings, on_chunk=on_chunk, should_stop=should_stop)


async def mock_stream(prompt: str, stop_event: asyncio.Event) -> AsyncIterator[InferenceEvent]:
    response = f"Mock response for: {prompt}. This stream is ready for real model integration."
    t0 = time.perf_counter()
    first = None
    count = 0
    for token in response.split(" "):
        if stop_event.is_set():
            raise InferenceCancelled()
        await asyncio.sleep(0.03)
        if first is None:
            first = time.perf_counter()
        count += 1
        yield {"type": "token", "value": token + " "}
    t1 = time.perf_counter()
    ttft = (first - t0) if first else 0.0
    decode_time = max(t1 - (first or t0), 1e-6)
    yield {
        "type": "metrics",
        "value": _metrics(
            ttft,
            decode_time / max(count, 1),
            max(count, 1) / decode_time,
            prefill_tps=0.0,
        ),
    }


async def inf_main_stream(prompt: str, settings: InferenceSettings, stop_event: asyncio.Event) -> AsyncIterator[InferenceEvent]:
    if stop_event.is_set():
        raise InferenceCancelled()
    queue: asyncio.Queue[InferenceEvent | Exception | _Done] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def on_chunk(chunk: str) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, {"type": "token", "value": chunk})

    def should_stop() -> bool:
        return stop_event.is_set()

    def worker() -> None:
        try:
            metrics = _run_inf_streaming(prompt, settings, on_chunk=on_chunk, should_stop=should_stop)
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "metrics", "value": metrics})
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, _Done())

    thread = threading.Thread(target=worker, name="inf-stream-worker", daemon=True)
    thread.start()

    while True:
        item = await queue.get()
        if isinstance(item, _Done):
            break
        if isinstance(item, Exception):
            raise item
        if stop_event.is_set():
            raise InferenceCancelled()
        yield item


async def stream_inference(prompt: str, settings: InferenceSettings, stop_event: asyncio.Event) -> AsyncIterator[InferenceEvent]:
    backend = os.getenv("INFERENCE_BACKEND", "inf").strip().lower()
    seen_any = False
    try:
        if backend in {"inf", "mamba3", "turboquant"}:
            iterator = inf_main_stream(prompt, settings, stop_event)
        else:
            iterator = mock_stream(prompt, stop_event)
        async for event in iterator:
            seen_any = True
            yield event
    except InferenceCancelled:
        raise
    except Exception:
        if seen_any or backend in {"inf", "mamba3", "turboquant"}:
            raise
        async for event in mock_stream(prompt, stop_event):
            yield event
