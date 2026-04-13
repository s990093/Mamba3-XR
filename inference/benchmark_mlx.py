#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLX prefill / decode throughput for train.py-compatible hybrid model.

Usage (from repo root):
  python inference/benchmark_mlx.py --checkpoint checkpoint.pt \\
      --tokenizer inference/tokenizer --seq-len 512 --decode-tokens 128

If ``checkpoint.pt`` exists next to ``checkpoint.npz`` (or repo ``model.npz``), the npz is loaded
unless ``--force-pt`` is set. The first successful load from a ``.pt`` auto-writes ``<stem>.npz``;
``--save-npz`` still writes an extra path when set.

Requires: mlx, numpy, torch (for .pt only), transformers (tokenizer).

Examples:
  python inference/benchmark_mlx.py --prompt "Hello" --decode-tokens 64
  python inference/benchmark_mlx.py --inference-type safe --decode-tokens 64
  python inference/benchmark_mlx.py --dtype bf16 --kv-dtype auto
  # Full bf16: weights + KV + router/MoE in bf16 (see mlx_hybrid_infer TuckerMoE + router_temp dtype)
  python inference/benchmark_mlx.py --temp 0.8 --top_p 0.9 --rep_pen 1.05
  python inference/benchmark_mlx.py --fast-sample --no-show-io   # greedy TPS, no I/O dump
  # Near peak decode TPS: full single-step compile, no penalties, skip cache materialize
  python inference/benchmark_mlx.py --full-decode-compile --fast-sample --no-penalties \\
      --no-materialize-caches --dtype bf16 --kv-dtype auto --decode-tokens 128
  # Near peak TPS: per-layer decode + bf16 + weight quantization (less memory bandwidth)
  python inference/benchmark_mlx.py --inference-type throughput --dtype bf16 --kv-dtype bf16 \\
      --quantize 8 --decode-tokens 128 --fast-sample
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any

import mlx.core as mx
import numpy as np

# Allow `python inference/benchmark_mlx.py` without installing the package.
_INF_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_INF_DIR, ".."))
if _INF_DIR not in sys.path:
    sys.path.insert(0, _INF_DIR)

from mlx_hybrid_infer import (
    Mamba3Config,
    Mamba3LanguageModel,
    attach_decode_compilation,
    export_npz_cache,
    maybe_export_npz_sidecar_after_pt_load,
    resolve_mlx_checkpoint,
    strict_load_and_convert,
)


def _invalidate_tucker_caches(model: Mamba3LanguageModel) -> None:
    for m in model.modules():
        if hasattr(m, "invalidate_g_cache"):
            m.invalidate_g_cache()


def _pad_kv(k: mx.array, v: mx.array, max_len: int) -> tuple[mx.array, mx.array]:
    cur = k.shape[2]
    if cur >= max_len:
        return k, v
    pad = max_len - cur
    zk = mx.zeros((k.shape[0], k.shape[1], pad, k.shape[3]), dtype=k.dtype)
    zv = mx.zeros((v.shape[0], v.shape[1], pad, v.shape[3]), dtype=v.dtype)
    return mx.concatenate([k, zk], axis=2), mx.concatenate([v, zv], axis=2)


def _materialize_mx_leaf(a: mx.array) -> mx.array:
    """Detach from compiled-graph outputs without NumPy (avoids sync + host round-trip on unified memory)."""
    z = mx.zeros((), dtype=a.dtype)
    return a + z


def _materialize_cache_tree(node: Any) -> Any:
    """
    Copy MLX arrays out of compiled-graph outputs so eager decode sees stable buffers.
    Without this, caches from ``mx.compile(prefill_forward)`` can leave decode broken (e.g. argmax → token 0).
    """
    if isinstance(node, (list, tuple)):
        t = type(node)(_materialize_cache_tree(x) for x in node)
        return t
    if node is None:
        return None
    if isinstance(node, mx.array):
        return _materialize_mx_leaf(node)
    return node


def _build_prompt_ids(tokenizer, text: str, seq_len: int) -> list[int]:
    """
    Tokenize *text*; do NOT pad/repeat to ``seq_len``.

    - If *seq_len* > 0: truncate to at most *seq_len*.
    - If tokenization is empty: fall back to a single token.
    """
    ids = list(tokenizer.encode(text))
    if not ids:
        ids = list(tokenizer.encode("a"))[:1]
    if seq_len > 0 and len(ids) > seq_len:
        return ids[:seq_len]
    return ids


def _pad_transformer_caches(caches, max_len: int):
    out = []
    for c in caches:
        if c is None:
            out.append(None)
        elif isinstance(c, tuple) and len(c) == 2:
            k, v = c
            out.append(_pad_kv(k, v, max_len))
        else:
            # Mamba cache (h, prev_in, angles) — no KV padding
            out.append(c)
    return out


def _init_token_counts(prompt_ids: list[int], vocab_size: int) -> mx.array:
    """Initialize token counts once (prompt side) as MLX tensor."""
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    counts = np.bincount(np.array(prompt_ids, dtype=np.int64), minlength=vocab_size).astype(np.float32)
    return mx.array(counts)


def _apply_penalties_fast(logits: mx.array, token_counts: mx.array, args: Any) -> mx.array:
    """GPU-side penalty path using per-vocab token counts."""
    if args.rep_pen == 1.0 and args.pres_pen == 0.0 and args.freq_pen == 0.0:
        return logits
    mask = token_counts > 0
    pres = mx.array(args.pres_pen, dtype=logits.dtype)
    freq = mx.array(args.freq_pen, dtype=logits.dtype)
    token_counts_d = token_counts.astype(logits.dtype)
    penalty_val = mx.where(mask, pres + token_counts_d * freq, mx.array(0.0, dtype=logits.dtype))
    logits = logits - penalty_val
    if args.rep_pen != 1.0:
        rep = mx.array(args.rep_pen, dtype=logits.dtype)
        rep_adjusted = mx.where(logits > 0, logits / rep, logits * rep)
        logits = mx.where(mask, rep_adjusted, logits)
    return logits


def _advanced_sample(logits: mx.array, args: Any) -> mx.array:
    """Same as ``main.py`` advanced_sample — greedy, temperature, min-p, top-k, top-p, categorical."""
    if args.fast_sample:
        return mx.argmax(logits, axis=-1)
    if args.temp == 0.0:
        return mx.argmax(logits, axis=-1)
    t = mx.maximum(mx.array(args.temp, dtype=logits.dtype), mx.array(1e-8, dtype=logits.dtype))
    logits = logits / t
    probs = mx.softmax(logits, axis=-1)
    if args.min_p > 0.0:
        p_max = mx.max(probs)
        logits = mx.where(probs < (args.min_p * p_max), mx.array(-1e9, dtype=logits.dtype), logits)
    if args.top_k > 0:
        top_k_indices = mx.argsort(-logits)
        kth_val = logits[top_k_indices[args.top_k - 1]]
        logits = mx.where(logits < kth_val, mx.array(-1e9, dtype=logits.dtype), logits)
    if args.top_p < 1.0:
        probs = mx.softmax(logits, axis=-1)
        sorted_indices = mx.argsort(-probs)
        sorted_probs = probs[sorted_indices]
        cumulative_probs = mx.cumsum(sorted_probs, axis=-1)
        mask = cumulative_probs > args.top_p
        shifted_mask = mx.concatenate([mx.array([False]), mask[:-1]])
        sorted_logits = logits[sorted_indices]
        sorted_logits = mx.where(shifted_mask, mx.array(-1e9, dtype=logits.dtype), sorted_logits)
        inverse_indices = mx.argsort(sorted_indices)
        logits = sorted_logits[inverse_indices]
    return mx.random.categorical(logits)


def _apply_inference_type(args: Any) -> None:
    """
    Map --inference-type presets to compile / cache / Mamba scan flags.
    Explicit ``--no-compile-prefill`` / ``--eager-decode`` / ``--no-materialize-caches``
    in ``sys.argv`` override the preset (later wins).
    """
    presets: dict[str, dict[str, Any]] = {
        # Max TPS: compiled prefill + compiled decode + stable caches
        "throughput": {
            "no_compile_prefill": False,
            "eager_decode": False,
            "no_materialize_caches": False,
            "use_parallel_scan": True,
        },
        # Prefer correct decode + compiled prefill (decode kernels eager)
        "safe": {
            "no_compile_prefill": False,
            "eager_decode": True,
            "no_materialize_caches": False,
            "use_parallel_scan": True,
        },
        # Slowest; easiest to debug / match non-compiled numerics
        "eager": {
            "no_compile_prefill": True,
            "eager_decode": True,
            "no_materialize_caches": False,
            "use_parallel_scan": True,
        },
        # Reference-style step-wise Mamba SSM (much slower; optional numerics check)
        "sequential-ssm": {
            "no_compile_prefill": False,
            "eager_decode": False,
            "no_materialize_caches": False,
            "use_parallel_scan": False,
        },
    }
    if args.inference_type == "custom":
        args.use_parallel_scan = True
        return
    cfg = presets[args.inference_type]
    for k, v in cfg.items():
        setattr(args, k, v)
    for attr, flag in (
        ("no_compile_prefill", "--no-compile-prefill"),
        ("eager_decode", "--eager-decode"),
        ("no_materialize_caches", "--no-materialize-caches"),
    ):
        if flag in sys.argv:
            setattr(args, attr, True)


def main() -> None:
    p = argparse.ArgumentParser(description="MLX hybrid Mamba3 benchmark (prefill + decode TPS)")
    p.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Weights: .pt/.pth/.npz, or empty to try repo model.npz then checkpoint.pt",
    )
    p.add_argument(
        "--npz-cache",
        type=str,
        default="",
        help="Explicit .npz path (overrides auto sidecar / model.npz when loading from .pt)",
    )
    p.add_argument(
        "--force-pt",
        action="store_true",
        help="Load PyTorch checkpoint even if a matching .npz cache exists",
    )
    p.add_argument(
        "--save-npz",
        nargs="?",
        const="__default__",
        default=None,
        metavar="PATH",
        help="After loading a .pt file, save MLX weights as npz (default: same stem as .pt)",
    )
    p.add_argument("--tokenizer", type=str, default=os.path.join(os.path.dirname(__file__), "tokenizer"))
    p.add_argument(
        "--inference-type",
        type=str,
        default="throughput",
        choices=("throughput", "safe", "eager", "sequential-ssm", "custom"),
        help=(
            "Preset for MLX graph + Mamba scan: "
            "throughput=max compile + chunk parallel scan; "
            "safe=compiled prefill + eager decode; "
            "eager=no compile prefill + eager decode; "
            "sequential-ssm=chunk scan off (slow); "
            "custom=use only --no-compile-prefill / --eager-decode / --no-materialize-caches"
        ),
    )
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--decode-tokens", type=int, default=128)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--vocab-size", type=int, default=32007)
    p.add_argument(
        "--dtype",
        type=str,
        default="fp32",
        choices=["fp32", "bf16", "fp16"],
        help=(
            "Weight / compute dtype (applied after load). For maximum bf16 throughput use "
            "'--dtype bf16 --kv-dtype auto' so KV + MoE router temps match (no fp32 promotion)."
        ),
    )
    p.add_argument(
        "--kv-dtype",
        type=str,
        default="bf16",
        choices=["auto", "bf16", "fp16", "fp32"],
        help="KV-cache storage dtype. 'auto' matches --dtype (use with bf16/fp16 weights). Default bf16 pairs with fp32 weights (legacy).",
    )
    p.add_argument(
        "--quantize",
        type=int,
        choices=[0, 4, 8],
        default=0,
        help="Quantize Linear/Embedding weights to 4- or 8-bit (0 = off; cuts memory bandwidth). Applied after --dtype cast.",
    )
    p.add_argument("--router-temp", type=float, default=0.5, help="Matches train ROUTER_T_END for inference")
    p.add_argument(
        "--no-compile-prefill",
        action="store_true",
        help="Disable mx.compile on prefill forward (default: compile static graph for (1, seq_len))",
    )
    p.add_argument(
        "--eager-decode",
        action="store_true",
        help="Disable per-layer / LM-head mx.compile for decode (use if generation looks wrong)",
    )
    p.add_argument(
        "--full-decode-compile",
        action="store_true",
        help=(
            "Compile the entire single-token forward once with mx.compile (disables per-layer decode compile). "
            "Often best decode TPS; use with --fast-sample --no-penalties --no-materialize-caches."
        ),
    )
    p.add_argument(
        "--no-materialize-caches",
        action="store_true",
        help="Skip graph-detach copy of prefill caches (unsafe after compiled prefill; fastest benchmark)",
    )
    p.add_argument(
        "--prompt",
        type=str,
        default="Hello! Write one short sentence about MLX on Apple Silicon.",
        help="Text for prefill (tokenized / padded to --seq-len). Ignored when --synthetic-prompt is set.",
    )
    p.add_argument(
        "--synthetic-prompt",
        action="store_true",
        help="Use repeating filler tokens for pure throughput (no real --prompt text).",
    )
    p.add_argument(
        "--no-show-io",
        action="store_true",
        help="Only print prefill/decode TPS lines (hide prompt & generated text)",
    )
    p.add_argument(
        "--show-token-ids",
        type=int,
        default=24,
        metavar="N",
        help="Show first N generated token ids (0 = hide)",
    )
    # Sampling / penalties — aligned with inference/backend/app/local_inf/main.py
    # p.add_argument("--temp", type=float, default=0.8, help="Softmax temperature (0 = greedy)")
    # p.add_argument("--top_k", type=int, default=0, help="Keep only top-k logits (0 = off)")
    # p.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling p (1.0 = off)")
    # p.add_argument("--min_p", type=float, default=0.0, help="Min-p filter vs p_max (0 = off)")
    # p.add_argument("--rep_pen", type=float, default=1.05, help="Repetition penalty on logits")
    # p.add_argument("--pres_pen", type=float, default=0.0, help="Presence penalty (subtract from logits)")
    # p.add_argument("--freq_pen", type=float, default=0.05, help="Frequency penalty (per count)")
    
    
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--min_p", type=float, default=0.05)
    p.add_argument("--rep_pen", type=float, default=1.1)
    p.add_argument("--pres_pen", type=float, default=0.0)
    p.add_argument("--freq_pen", type=float, default=0.02)
  
    p.add_argument(
        "--fast-sample",
        action="store_true",
        help="Greedy argmax (fastest; ignores temp / top-k / top-p / min-p)",
    )
    p.add_argument(
        "--no-penalties",
        action="store_true",
        help="Skip repetition / presence / frequency penalty path on logits (best decode TPS with --fast-sample)",
    )
    args = p.parse_args()
    _apply_inference_type(args)
    if args.no_penalties:
        args.rep_pen = 1.0
        args.pres_pen = 0.0
        args.freq_pen = 0.0

    compute_dtype_map = {"fp32": mx.float32, "bf16": mx.bfloat16, "fp16": mx.float16}
    target_dtype = compute_dtype_map[args.dtype]
    kv_map = {"bf16": mx.bfloat16, "fp16": mx.float16, "fp32": mx.float32}
    if args.kv_dtype == "auto":
        kv_dtype = target_dtype
    else:
        kv_dtype = kv_map[args.kv_dtype]

    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise SystemExit("Please `pip install transformers` to load the tokenizer.") from e

    tok_path = args.tokenizer
    if os.path.isdir(tok_path):
        tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)

    vocab_size = len(tokenizer) if args.vocab_size <= 0 else args.vocab_size

    config = Mamba3Config(
        d_model=768,
        d_state=64,
        d_head=64,
        expand=2,
        num_layers=6,
        mimo_rank=4,
        num_kv_heads=4,
        use_parallel_scan=args.use_parallel_scan,
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
        args.checkpoint,
        repo_root=_REPO_ROOT,
        npz_cache=args.npz_cache,
        force_pt=args.force_pt,
    )
    if resolved is None or kind == "none":
        print("No checkpoint found — random weights (for graph / TPS smoke test only).")
    else:
        # Same pipeline as inference/backend/app/local_inf/tool.py strict_load_and_convert
        strict_load_and_convert(model, resolved)
        if kind == "pt":
            sidecar_written = maybe_export_npz_sidecar_after_pt_load(
                model, resolved, force_refresh=args.force_pt
            )
            if sidecar_written is not None:
                print(f"💾 已建立 npz 快取，下次將優先讀取: {sidecar_written}")
        if kind == "pt" and args.save_npz is not None:
            dest = (
                os.path.splitext(resolved)[0] + ".npz"
                if args.save_npz == "__default__"
                else args.save_npz
            )
            dest_abs = os.path.abspath(dest)
            stem_npz = os.path.abspath(os.path.splitext(resolved)[0] + ".npz")
            if dest_abs != stem_npz:
                export_npz_cache(model, dest)
                print(f"Wrote npz cache → {dest}")

    model.apply(lambda x: x.astype(target_dtype))
    if args.quantize > 0:
        import mlx.nn as nn

        print(f"Quantizing Linear/Embedding to {args.quantize}-bit (group_size=64, mode=affine)...")
        nn.quantize(model, group_size=64, bits=args.quantize)
    mx.eval(model.parameters())
    _invalidate_tucker_caches(model)
    # Match compute dtype so MoE / Tucker router path stays bf16 when weights are bf16 (was hard-coded fp32).
    router_temp = mx.array(args.router_temp, dtype=target_dtype)

    # Prefill: (1, seq_len) tokens
    if args.synthetic_prompt:
        filler = list(tokenizer.encode("benchmark " * max(1, args.seq_len // 4)))[: args.seq_len]
        if len(filler) < args.seq_len:
            filler = (filler * (args.seq_len // max(len(filler), 1) + 1))[: args.seq_len]
        prompt_ids = filler
        prompt_source = "[synthetic repeat for throughput]"
    else:
        prompt_ids = _build_prompt_ids(tokenizer, args.prompt, args.seq_len)
        prompt_source = args.prompt

    prefill_tokens = len(prompt_ids)
    max_cache_len = prefill_tokens + args.decode_tokens + 8

    per_layer_decode = not args.eager_decode and not args.full_decode_compile
    attach_decode_compilation(
        model,
        max_cache_len=max_cache_len,
        kv_dtype=kv_dtype,
        compile_decode=per_layer_decode,
    )

    compiled_full_decode = None
    if args.full_decode_compile:

        def _full_decode_step(x_in: mx.array, cur_caches: Any, pos_arr: mx.array):
            return model(x_in, caches=cur_caches, seq_pos=pos_arr, router_temp=router_temp)

        compiled_full_decode = mx.compile(_full_decode_step)

    x_prefill = mx.array([prompt_ids], dtype=mx.int32)

    def prefill_forward(x: mx.array, rt: mx.array):
        return model(x, caches=None, seq_pos=None, router_temp=rt)

    prefill_compile_mode = "eager"
    if args.no_compile_prefill:
        run_prefill = prefill_forward
    else:
        run_prefill = mx.compile(prefill_forward)
        prefill_compile_mode = "static"

    for _ in range(args.warmup):
        logits, caches = run_prefill(x_prefill, router_temp)
        mx.eval(logits, caches)

    t0 = time.perf_counter()
    logits, caches = run_prefill(x_prefill, router_temp)
    mx.eval(logits, caches)
    prefill_s = time.perf_counter() - t0
    prefill_tps = prefill_tokens / max(prefill_s, 1e-9)

    if not args.no_materialize_caches:
        caches = _materialize_cache_tree(caches)
        mx.eval(caches)

    caches = _pad_transformer_caches(caches, max_cache_len)
    mx.eval(caches)

    # Decode: first token from prefill logits, then (decode_tokens - 1) single-token forwards
    # (same token count as before: total new tokens == decode_tokens).
    pos = len(prompt_ids)
    generated_ids: list[int] = []
    t1 = time.perf_counter()
    if args.decode_tokens > 0:
        row = logits[0, -1, :]
        token_counts = _init_token_counts(prompt_ids, int(row.shape[0]))
        row = _apply_penalties_fast(row, token_counts, args)
        last = _advanced_sample(row, args)
        token_counts[last] = token_counts[last] + 1
        mx.eval(last, token_counts)
        generated_ids.append(int(last.item()))
        x_one = last.reshape(1, 1)
        for _ in range(args.decode_tokens - 1):
            pos_arr = mx.array(pos, dtype=mx.int32)
            if compiled_full_decode is not None:
                logits_d, caches = compiled_full_decode(x_one, caches, pos_arr)
            else:
                logits_d, caches = model(
                    x_one,
                    caches=caches,
                    seq_pos=pos_arr,
                    router_temp=router_temp,
                )
            row = logits_d[0, -1, :]
            row = _apply_penalties_fast(row, token_counts, args)
            last = _advanced_sample(row, args)
            token_counts[last] = token_counts[last] + 1
            # Evaluate sampled token and updated cache together to avoid splitting the graph.
            mx.eval(last, caches, token_counts)
            generated_ids.append(int(last.item()))
            x_one = last.reshape(1, 1)
            pos += 1
    decode_s = time.perf_counter() - t1
    decode_tps = args.decode_tokens / max(decode_s, 1e-9)

    print(
        f"Prefill: tokens={prefill_tokens} time={prefill_s*1000:.2f} ms  ({prefill_tps:.1f} tok/s)  "
        f"compile={prefill_compile_mode}"
    )
    print(f"Decode:  tokens={args.decode_tokens} time={decode_s*1000:.2f} ms  ({decode_tps:.1f} tok/s)")
    _dn = {mx.float32: "fp32", mx.bfloat16: "bf16", mx.float16: "fp16"}
    kv_resolved = _dn.get(kv_dtype, "?")
    kv_label = f"auto→{kv_resolved}" if args.kv_dtype == "auto" else args.kv_dtype
    if compiled_full_decode is not None:
        decode_compile_mode = "full"
    elif not args.eager_decode:
        decode_compile_mode = "per-layer"
    else:
        decode_compile_mode = "eager"
    print(
        f"Inference: type={args.inference_type}  dtype={args.dtype}  kv_dtype={kv_label}  "
        f"parallel_scan={args.use_parallel_scan}  "
        f"compile_prefill={not args.no_compile_prefill}  prefill_graph={prefill_compile_mode}  "
        f"decode_compile={decode_compile_mode}  "
        f"materialize_caches={not args.no_materialize_caches}"
    )

    if not args.no_show_io:
        print()
        print("─" * 60)
        print("I/O")
        print("─" * 60)
        if args.synthetic_prompt:
            print(f"Prompt mode: synthetic ({args.seq_len} tokens)")
        else:
            print(f"Prompt ({len(prompt_source)} chars → {len(prompt_ids)} tokens):")
            preview = prompt_source if len(prompt_source) <= 600 else prompt_source[:600] + " …"
            for line in preview.splitlines() or [preview]:
                print(f"  {line}")
        try:
            prompt_decoded = tokenizer.decode(prompt_ids, skip_special_tokens=True)
            if prompt_decoded.strip():
                pd_prev = prompt_decoded if len(prompt_decoded) <= 800 else prompt_decoded[:800] + " …"
                print("Prefill tokens decoded (sanity check):")
                print(f"  {pd_prev}")
        except Exception as e:
            print(f"  (decode prefill ids failed: {e})")

        out_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        print()
        print(f"Generated ({len(generated_ids)} new tokens):")
        ot = out_text if len(out_text) <= 2000 else out_text[:2000] + "\n  …"
        for line in ot.splitlines() or [ot]:
            print(f"  {line}")
        if args.show_token_ids > 0:
            head = generated_ids[: args.show_token_ids]
            print()
            print(f"New token ids (first {len(head)}): {head}")
        print("─" * 60)


if __name__ == "__main__":
    main()
