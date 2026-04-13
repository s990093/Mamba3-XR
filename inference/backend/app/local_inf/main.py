import os
import time
import argparse
from collections import Counter
import mlx.core as mx
import mlx.nn as nn
import mlx.utils
from transformers import AutoTokenizer

from app.local_inf.tool import save_checkpoint_numpy, load_and_compare_vocab
from app.local_inf.model import Mamba3Config, Mamba3LanguageModel, TuckerMoE, Mamba3Block, TransformerBlock


def _stream_chunk_text(tokenizer, token_ids):
    if not tokenizer:
        return " ".join(str(t) for t in token_ids)
    try:
        pieces = tokenizer.convert_ids_to_tokens(token_ids)
        if isinstance(pieces, str):
            pieces = [pieces]
        text = "".join(piece.replace("▁", " ").replace("Ġ", " ") for piece in pieces)
        # Trim only control leftovers; keep leading spaces for natural word boundaries.
        return text.replace("</s>", "").replace("<s>", "")
    except Exception:
        return tokenizer.decode(token_ids, skip_special_tokens=True)


def apply_penalties(logits, generated_tokens, args):
    if not generated_tokens:
        return logits
    if args.rep_pen == 1.0 and args.pres_pen == 0.0 and args.freq_pen == 0.0:
        return logits
    counts = Counter(generated_tokens)
    idx_arr = mx.array(list(counts.keys()))
    freq_arr = mx.array(list(counts.values()))
    target_logits = logits[idx_arr]
    if args.rep_pen != 1.0:
        target_logits = mx.where(target_logits > 0, target_logits / args.rep_pen, target_logits * args.rep_pen)
    target_logits = target_logits - (args.pres_pen + freq_arr * args.freq_pen)
    logits[idx_arr] = target_logits
    return logits


def advanced_sample(logits, args):
    if args.fast_sample:
        return mx.argmax(logits, axis=-1)
    if args.temp == 0.0:
        return mx.argmax(logits, axis=-1)
    logits = logits / args.temp
    probs = mx.softmax(logits, axis=-1)
    if args.min_p > 0.0:
        p_max = mx.max(probs)
        logits = mx.where(probs < (args.min_p * p_max), -1e9, logits)
    if args.top_k > 0:
        top_k_indices = mx.argsort(-logits)
        kth_val = logits[top_k_indices[args.top_k - 1]]
        logits = mx.where(logits < kth_val, -1e9, logits)
    if args.top_p < 1.0:
        probs = mx.softmax(logits, axis=-1)
        sorted_indices = mx.argsort(-probs)
        sorted_probs = probs[sorted_indices]
        cumulative_probs = mx.cumsum(sorted_probs, axis=-1)
        mask = cumulative_probs > args.top_p
        shifted_mask = mx.concatenate([mx.array([False]), mask[:-1]])
        sorted_logits = logits[sorted_indices]
        sorted_logits = mx.where(shifted_mask, -1e9, sorted_logits)
        inverse_indices = mx.argsort(sorted_indices)
        logits = sorted_logits[inverse_indices]
    return mx.random.categorical(logits)


def resolve_checkpoint_path(load_ckpt, save_ckpt):
    if load_ckpt and os.path.exists(load_ckpt):
        if not save_ckpt or save_ckpt == load_ckpt:
            return load_ckpt, False

        load_ext = os.path.splitext(load_ckpt)[1].lower()
        save_ext = os.path.splitext(save_ckpt)[1].lower()
        if load_ext in {".pt", ".bin"} and save_ext == ".npz":
            if os.path.exists(save_ckpt) and os.path.getmtime(save_ckpt) >= os.path.getmtime(load_ckpt):
                print(f"📦 偵測到已存在且最新的 NumPy Checkpoint，直接讀取 {save_ckpt}")
                return save_ckpt, False
            return load_ckpt, True

        return load_ckpt, False

    if save_ckpt and os.path.exists(save_ckpt):
        return save_ckpt, False

    return None, False


def calculate_model_metrics(model, config):
    flat_params = mlx.utils.tree_flatten(model.parameters())
    total_params = 0
    active_params = 0
    for name, v in flat_params:
        num_p = v.size
        if "head.weight" in name:
            continue
        total_params += num_p
        if any(k in name for k in ["U_expert", "U_in", "U_out", "core"]):
            active_params += int(num_p * (config.kmoe_top_k / config.kmoe_num_experts))
        else:
            active_params += num_p
    return total_params, active_params


def preallocate_decode_caches(caches, max_len, kv_dtype=None):
    if caches is None:
        return None
    new_caches = []
    for c in caches:
        if c is None:
            new_caches.append(None)
            continue
        if isinstance(c, tuple) and len(c) == 2:
            k, v = c
            target_dtype = kv_dtype if kv_dtype is not None else k.dtype
            if k.dtype != target_dtype:
                k = k.astype(target_dtype)
            if v.dtype != target_dtype:
                v = v.astype(target_dtype)
            cur_len = k.shape[2]
            if cur_len >= max_len:
                new_caches.append((k, v))
                continue
            k_full = mx.zeros((k.shape[0], k.shape[1], max_len, k.shape[3]), dtype=target_dtype)
            v_full = mx.zeros((v.shape[0], v.shape[1], max_len, v.shape[3]), dtype=target_dtype)
            k_full[:, :, :cur_len, :] = k
            v_full[:, :, :cur_len, :] = v
            new_caches.append((k_full, v_full))
        else:
            new_caches.append(c)
    return new_caches


def generate(model, prompt_tokens, args, config, tokenizer=None, on_chunk=None, should_stop=None):
    print(f"🧠 開始生成 (提示詞長度: {len(prompt_tokens)} tokens, 模式: inference)...")
    current_tokens = prompt_tokens.copy()
    generated_new_tokens = []
    caches = None
    output_enabled = args.stream_output and (not args.no_stream_output)
    output_chunk = max(1, args.stream_chunk_tokens)
    token_output_buffer = []
    flush_count = 0

    start_time = time.perf_counter()
    x = mx.array([prompt_tokens])
    logits, caches = model(x, caches)
    mx.eval(logits, caches)
    logits = logits[0, -1, :]
    logits = apply_penalties(logits, generated_new_tokens, args)
    next_token_arr = advanced_sample(logits, args)
    mx.eval(next_token_arr)
    prefill_time = time.perf_counter() - start_time
    ttft = prefill_time
    print(f"✅ Prefill 完畢 (TTFT: {ttft:.3f}s)")
    print("------------------------------------------------------------")

    next_token = next_token_arr.item()
    current_tokens.append(next_token)
    generated_new_tokens.append(next_token)

    if output_enabled:
        token_output_buffer.append(next_token)
        if len(token_output_buffer) >= output_chunk:
            flush_count += 1
            chunk_text = _stream_chunk_text(tokenizer, token_output_buffer)
            if tokenizer:
                print(chunk_text, end="", flush=True)
            else:
                print(chunk_text, end=" ", flush=True)
            if on_chunk:
                on_chunk(chunk_text)
            token_output_buffer = []

    decode_start_time = time.perf_counter()
    seq_pos = len(prompt_tokens)
    decode_max_len = len(prompt_tokens) + args.max_tokens + 8
    kv_dtype = model.embed.weight.dtype if args.kv_dtype == "auto" else (
        mx.bfloat16 if args.kv_dtype == "bf16" else mx.float16
    )
    caches = preallocate_decode_caches(caches, decode_max_len, kv_dtype=kv_dtype)

    for _ in range(args.max_tokens - 1):
        if should_stop and should_stop():
            break
        x = mx.array([[next_token]])
        logits, caches = model(x, caches, seq_pos=seq_pos)
        mx.eval(logits, caches)
        logits = logits[0, -1, :]
        logits = apply_penalties(logits, generated_new_tokens, args)
        next_token_arr = advanced_sample(logits, args)
        mx.eval(next_token_arr)
        next_token = next_token_arr.item()
        seq_pos += 1
        current_tokens.append(next_token)
        generated_new_tokens.append(next_token)
        if output_enabled:
            token_output_buffer.append(next_token)
            if len(token_output_buffer) >= output_chunk:
                flush_count += 1
                chunk_text = _stream_chunk_text(tokenizer, token_output_buffer)
                if tokenizer:
                    print(chunk_text, end="", flush=True)
                else:
                    print(chunk_text, end=" ", flush=True)
                if on_chunk:
                    on_chunk(chunk_text)
                token_output_buffer = []

    if output_enabled and token_output_buffer:
        flush_count += 1
        chunk_text = _stream_chunk_text(tokenizer, token_output_buffer)
        if tokenizer:
            print(chunk_text, end="", flush=True)
        else:
            print(chunk_text, end=" ", flush=True)
        if on_chunk:
            on_chunk(chunk_text)

    end_time = time.perf_counter()
    print("\n------------------------------------------------------------")

    decode_time = end_time - decode_start_time
    total_generated = args.max_tokens
    prompt_len = len(prompt_tokens)
    prefill_tps = prompt_len / prefill_time if prefill_time > 0 else 0
    decode_tps = (total_generated - 1) / decode_time if decode_time > 0 else 0
    tpot = decode_time / (total_generated - 1) if total_generated > 1 else 0

    model_dtype = model.embed.weight.dtype
    model_dtype_str = str(model_dtype).split(".")[-1]
    total_params, active_params = calculate_model_metrics(model, config)
    prefill_tflops = (2 * active_params * prompt_len) / (prefill_time * 1e12) if prefill_time > 0 else 0
    decode_tflops = (2 * active_params * 1) / (tpot * 1e12) if tpot > 0 else 0
    mlx_active_mem = mx.get_active_memory() / (1024**2)
    mlx_peak_mem = mx.get_peak_memory() / (1024**2)
    print(
        f"""
============================================================
📈 [Academic Metrics] Architecture & Efficiency
------------------------------------------------------------
⚙️ Precision          : {model_dtype_str}
📦 Total Params       : {total_params / 1e6:.2f} M
⚡ Active Params      : {active_params / 1e6:.2f} M
🎯 % Active           : {(active_params / total_params) * 100:.2f} %
------------------------------------------------------------
⏱️  Time To First Token (TTFT) : {ttft:.3f} s
⏱️  Time Per Output Token (TPOT): {tpot:.3f} s/token
------------------------------------------------------------
📊  Prefill TPS               : {prefill_tps:.2f} tokens/s
🚀  Prefill Compute           : {prefill_tflops:.4f} TFLOPS
------------------------------------------------------------
📊  Decode TPS                : {decode_tps:.2f} tokens/s
🚀  Decode Compute            : {decode_tflops:.4f} TFLOPS
🍎 MLX Active Mem: {mlx_active_mem:.2f} MB | Peak: {mlx_peak_mem:.2f} MB
============================================================
"""
    )
    return current_tokens


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mamba3 MLX Inference (Standalone)")
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temp", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--rep_pen", type=float, default=1.05)
    parser.add_argument("--pres_pen", type=float, default=0.0)
    parser.add_argument("--freq_pen", type=float, default=0.05)
    parser.add_argument("--dtype", type=str, default="fp16", choices=["32", "fp32", "16", "fp16", "bf16"])
    parser.add_argument("--save_ckpt", type=str, default="/Users/hungwei/Desktop/Proj/Mamba3-XR/model.npz")
    parser.add_argument("--load_ckpt", type=str, default="/Users/hungwei/Desktop/Proj/Mamba3-XR/checkpoint.pt")
    parser.add_argument("--tokenizer_path", type=str, default="/Users/hungwei/Desktop/Proj/Mamba3-XR/inf/tokenizer")
    parser.add_argument("--prompt", type=str, default="The future of artificial intelligence is")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--quant_bits", type=int, default=8, choices=[4, 8])
    parser.add_argument("--no_stream_output", action="store_true", help="Disable per-token stdout for accurate TPS")
    parser.add_argument("--stream_output", action="store_true", help="Enable token streaming output with buffered flush")
    parser.add_argument("--stream_chunk_tokens", type=int, default=8, help="Flush stream output every N tokens")
    parser.add_argument("--fast_sample", action="store_true", help="Use greedy sampling for maximum decode speed")
    parser.add_argument("--kv_dtype", type=str, default="auto", choices=["auto", "fp16", "bf16"], help="KV cache storage dtype")
    args = parser.parse_args()

    dtype_map = {"32": mx.float32, "fp32": mx.float32, "16": mx.float16, "fp16": mx.float16, "bf16": mx.bfloat16}
    target_dtype = dtype_map[args.dtype]

    if args.tokenizer_path:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
        vocab_size = len(tokenizer)
    else:
        tokenizer = None
        vocab_size = 32007

    config = Mamba3Config(
        d_model=768, d_state=64, d_head=64, expand=2, num_layers=6, mimo_rank=4, num_kv_heads=4,
        use_kmoe=True, kmoe_num_experts=8, kmoe_top_k=2, kmoe_r1=32, kmoe_r2=512, kmoe_r3=256, ffn_expand=6
    )
    model = Mamba3LanguageModel(config, vocab_size)

    ckpt_path, should_refresh_npz = resolve_checkpoint_path(args.load_ckpt, args.save_ckpt)
    if ckpt_path:
        load_and_compare_vocab(model, ckpt_path, vocab_size)
        if should_refresh_npz:
            save_checkpoint_numpy(model, args.save_ckpt)

    model.apply(lambda x: x.astype(target_dtype))
    if args.quantize:
        nn.quantize(
            model,
            class_predicate=lambda p, m: isinstance(m, nn.Linear) and "head" not in p and "embed" not in p,
            group_size=64,
            bits=args.quant_bits,
        )
    mx.eval(model.parameters())

    tucker_modules = [(name, mod) for name, mod in model.named_modules() if isinstance(mod, TuckerMoE)]
    if tucker_modules:
        for _, mod in tucker_modules:
            mod._get_G()

    if tokenizer:
        prompt_tokens = tokenizer.encode(args.prompt)
    else:
        prompt_tokens = [101, 2023, 1045, 2066, 2186]
    max_cache_len = len(prompt_tokens) + args.max_tokens + 8

    dtype = model.embed.weight.dtype
    kv_dtype = dtype if args.kv_dtype == "auto" else (mx.bfloat16 if args.kv_dtype == "bf16" else mx.float16)
    h_dim, p_dim, n_dim = config.n_heads, config.d_head, config.d_state
    for layer in model.backbone.layers:
        dummy_x = mx.zeros((1, 1, config.d_model), dtype=dtype)
        if isinstance(layer, Mamba3Block):
            dummy_h = mx.zeros((1, h_dim, n_dim, p_dim), dtype=dtype)
            dummy_pi = mx.zeros((1, 1, h_dim, n_dim, p_dim), dtype=dtype)
            dummy_ang = mx.zeros((1, 1, h_dim, n_dim // 2), dtype=dtype)
            out, _ = layer(dummy_x, cache=(dummy_h, dummy_pi, dummy_ang))
            mx.eval(out)

            def make_mamba_compiled(blk):
                def _decode_step(x, h_s, prev_in, prev_ang):
                    out_step, nc = blk(x, cache=(h_s, prev_in, prev_ang))
                    return out_step, nc[0], nc[1], nc[2]

                return mx.compile(_decode_step)

            layer._compiled_decode = make_mamba_compiled(layer)
        elif isinstance(layer, TransformerBlock):
            dummy_k = mx.zeros((1, layer.num_heads, max_cache_len, 64), dtype=kv_dtype)
            dummy_v = mx.zeros((1, layer.num_heads, max_cache_len, 64), dtype=kv_dtype)
            out, _ = layer(dummy_x, cache=(dummy_k, dummy_v), seq_pos=mx.array(0, dtype=mx.int32))
            mx.eval(out)

            def make_transformer_compiled(blk):
                def _decode_step(x, k_cache, v_cache, seq_pos):
                    out_step, nc = blk(x, cache=(k_cache, v_cache), seq_pos=seq_pos)
                    return out_step, nc[0], nc[1]

                return mx.compile(_decode_step)

            layer._compiled_decode = make_transformer_compiled(layer)

    generate(model, prompt_tokens, args, config, tokenizer=tokenizer)
