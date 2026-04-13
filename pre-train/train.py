%%writefile train.py

# -*- coding: utf-8 -*-
import os
import gc
import csv
import math
import time
import warnings
import shutil
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from accelerate import Accelerator

warnings.filterwarnings(
    "ignore",
    message=".*Online softmax is disabled on the fly.*",
    category=UserWarning,
)

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    MIXED_PRECISION = "bf16"
    torch.backends.cuda.matmul.fp32_precision = 'tf32'   # 新版 API，取代 allow_tf32
    torch.backends.cudnn.conv.fp32_precision  = 'tf32'
    print("🚀 高階 GPU 偵測成功，啟用 bf16 + TF32 最佳化！")
else:
    MIXED_PRECISION = "fp16"
    torch.backends.cuda.matmul.fp32_precision = 'ieee'   # 標準 IEEE 浮點數
    torch.backends.cudnn.conv.fp32_precision  = 'ieee'
    print("🐢 舊版 GPU，自動 Fallback 至 fp16。")



class PretokenizedDataset(IterableDataset):
    """
    【資料集】記憶體映射的預先 Tokenize 資料集
    ─────────────────────────────────────────
    直接從 .bin 檔（uint16 token id）以 mmap 方式串流讀取。

    優化：整個 buffer 一次性轉成 Long Tensor，再逐行 yield，
    消除了每條樣本都呼叫 torch.from_numpy 的 Python loop overhead。
    配合 DataLoader pin_memory=True 可以讓 CPU→GPU 非同步傳輸完整發揮。

    多 Worker：根據 worker id 自動分割資料範圍。
    """
    def __init__(self, data_path: str, seq_len: int, buffer_size: int = 4_000_000):
        self.data_path    = data_path
        self.seq_len      = seq_len
        self.buffer_size  = buffer_size
        self.total_tokens = len(np.memmap(data_path, dtype=np.uint16, mode="r"))

    def __iter__(self):
        worker_info = get_worker_info()
        start_idx, end_idx = 0, self.total_tokens
        if worker_info is not None:
            per       = self.total_tokens // worker_info.num_workers
            start_idx = worker_info.id * per
            end_idx   = (self.total_tokens
                         if worker_info.id == worker_info.num_workers - 1
                         else start_idx + per)

        mmap_data = np.memmap(self.data_path, dtype=np.uint16, mode="r")
        curr_idx  = start_idx

        while curr_idx + self.seq_len < end_idx:
            chunk_end = min(curr_idx + self.buffer_size, end_idx)
            # ── 整個 chunk 一次轉成 int64，再包成 Tensor（零額外拷貝）──
            buffer   = mmap_data[curr_idx:chunk_end].astype(np.int64)
            num_seqs = (len(buffer) - 1) // self.seq_len
            if num_seqs == 0:
                break

            # 一次性建立 Tensor：(num_seqs, seq_len)，避免逐條 from_numpy
            x_t = torch.from_numpy(
                buffer[: num_seqs * self.seq_len].reshape(num_seqs, self.seq_len)
            )  # shape: [N, L]
            y_t = torch.from_numpy(
                buffer[1 : num_seqs * self.seq_len + 1].reshape(num_seqs, self.seq_len)
            )  # shape: [N, L]

            # 逐行 yield：DataLoader collate 只需 stack，不必再做 from_numpy
            for i in range(num_seqs):
                yield x_t[i], y_t[i]

            curr_idx += num_seqs * self.seq_len

def unwrap_model(model: nn.Module) -> nn.Module:
    if hasattr(model, "module"):   model = model.module
    if hasattr(model, "_orig_mod"): model = model._orig_mod
    return model

def get_lr_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step):
        if step < warmup_steps: return step / max(1, warmup_steps)
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * ((step - warmup_steps) / max(1, total_steps - warmup_steps)))))
    return LambdaLR(optimizer, lr_lambda)


# ── Triton Kernels & Model Classes (synced from test_cpmoe.py) ──

@triton.jit
def tanh_approx(x):
    return tl.inline_asm_elementwise(
        "tanh.approx.f32 $0, $1;", constraints="=f,f", args=[x],
        dtype=tl.float32, is_pure=True, pack=1,
    )

@triton.jit
def silu(x):
    return x * tl.sigmoid(x)

def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ]

@triton.autotune(configs=get_cuda_autotune_config(), key=['n_elements'])
@triton.jit
def _fused_scaled_tanh_fwd(x_ptr, y_ptr, scale, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    t = tanh_approx(x * (1.0 / scale))
    tl.store(y_ptr + offsets, t * scale, mask=mask)

@triton.autotune(configs=get_cuda_autotune_config(), key=['n_elements'])
@triton.jit
def _fused_scaled_tanh_bwd(dy_ptr, x_ptr, dx_ptr, scale, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    dy = tl.load(dy_ptr + offsets, mask=mask).to(tl.float32)
    x  = tl.load(x_ptr  + offsets, mask=mask).to(tl.float32)
    t  = tanh_approx(x * (1.0 / scale))
    tl.store(dx_ptr + offsets, dy * (1.0 - t * t), mask=mask)

# ── Triton Autograd Wrappers ──────────────────────────────────────────

class _FastScaledTanh(torch.autograd.Function):
    """
    【Triton Autograd】縮放 Tanh 激活函數（Triton 加速）
    ────────────────────────────────────────────────────
    計算：y = scale * tanh(x / scale)

    用途：對 Logits 做「軟性截斷（Soft Capping）」，
    避免 Router 或最終 logits 爆炸，同時保留梯度流通。

    使用 CUDA PTX `tanh.approx.f32` 指令，
    比 PyTorch 原生 tanh 快 2–3x。
    """
    @staticmethod
    def forward(ctx, x, scale=10.0):
        ctx.save_for_backward(x); ctx.scale = scale
        y = torch.empty_like(x)
        grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)
        _fused_scaled_tanh_fwd[grid](x, y, scale, x.numel())
        return y
    @staticmethod
    def backward(ctx, dy):
        (x,) = ctx.saved_tensors
        dx = torch.empty_like(x)
        grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)
        _fused_scaled_tanh_bwd[grid](dy, x, dx, ctx.scale, x.numel())
        return dx, None

def fast_scaled_tanh(x, scale=10.0):
    return _FastScaledTanh.apply(x, scale)

@triton.autotune(configs=get_cuda_autotune_config(), key=['n_elements'])
@triton.jit
def _fused_silu_mul_fwd(gate_ptr, feat_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    gate = tl.load(gate_ptr + offsets, mask=mask).to(tl.float32)
    feat = tl.load(feat_ptr + offsets, mask=mask).to(tl.float32)
    tl.store(out_ptr + offsets, silu(gate) * feat, mask=mask)

@triton.autotune(configs=get_cuda_autotune_config(), key=['n_elements'])
@triton.jit
def _fused_silu_mul_bwd(dout_ptr, gate_ptr, feat_ptr, dgate_ptr, dfeat_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    dout = tl.load(dout_ptr + offsets, mask=mask).to(tl.float32)
    gate = tl.load(gate_ptr + offsets, mask=mask).to(tl.float32)
    feat = tl.load(feat_ptr + offsets, mask=mask).to(tl.float32)
    sig = tl.sigmoid(gate); s = gate * sig
    tl.store(dfeat_ptr + offsets, dout * s, mask=mask)
    tl.store(dgate_ptr + offsets, dout * feat * sig * (1.0 + gate * (1.0 - sig)), mask=mask)

class _FastSiluGating(torch.autograd.Function):
    """
    【Triton Autograd】融合 SiLU Gating（Triton 加速）
    ──────────────────────────────────────────────────
    計算：y = silu(gate) * feat  （GLU-style gating）

    在 Triton kernel 中融合兩個元素操作：
      1. SiLU(gate) = gate * sigmoid(gate)
      2. element-wise multiply with feat

    用於 MixtralMoEFeedForward 的 gate 路徑，
    比兩次獨立 kernel 呼叫少一次 HBM round-trip。
    """
    @staticmethod
    def forward(ctx, gate, feat):
        ctx.save_for_backward(gate, feat)
        out = torch.empty_like(gate)
        grid = lambda meta: (triton.cdiv(gate.numel(), meta["BLOCK_SIZE"]),)
        _fused_silu_mul_fwd[grid](gate, feat, out, gate.numel())
        return out
    @staticmethod
    def backward(ctx, dout):
        gate, feat = ctx.saved_tensors
        dgate, dfeat = torch.empty_like(gate), torch.empty_like(feat)
        grid = lambda meta: (triton.cdiv(gate.numel(), meta["BLOCK_SIZE"]),)
        _fused_silu_mul_bwd[grid](dout, gate, feat, dgate, dfeat, gate.numel())
        return dgate, dfeat

def fast_silu_gating(gate, feat):
    return _FastSiluGating.apply(gate, feat)

def get_router_temperature(step, warmup=500, total=10000, t_start=2.0, t_end=0.5):
    if step is None: return t_end
    if step < warmup: return t_start
    progress = min((step - warmup) / max(1, total - warmup), 1.0)
    return t_end + 0.5 * (t_start - t_end) * (1.0 + math.cos(math.pi * progress))


# ── Core Model Components ─────────────────────────────────────────────

class Mamba3Config:
    """
    【設定檔】Mamba3 模型的所有超參數集中管理
    ────────────────────────────────────────────
    負責計算並儲存所有衍生維度（d_inner、n_heads、kv_groups 等），
    確保模型各元件的參數一致。

    關鍵欄位：
        d_model         : 主幹隱藏層維度
        d_state         : SSM 狀態維度 N
        d_head          : 每個 Head 的維度 P
        n_heads         : 總 Head 數 = d_inner // d_head
        expand          : d_inner = expand * d_model
        mimo_rank       : SSM 中 B/C 矩陣的 MIMO Rank R
        kmoe_r1/r2/r3   : Tucker 分解的三個 Rank
        ffn_expand      : Transformer FFN 的擴張比例
        num_kv_heads    : GQA 的 KV-Head 數
        chunk_size      : Parallel Scan 的 Chunk 大小
    """
    def __init__(
        self, d_model=768, d_state=64, d_head=64, n_groups=1, mimo_rank=4, expand=4,
        num_layers=15, use_parallel_scan=True, use_kmoe=True,
        kmoe_num_experts=8, kmoe_top_k=2, kmoe_r1=4, kmoe_r2=1024, kmoe_r3=256, ffn_expand=6, num_kv_heads=4,
        dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, layer_scale_init=1e-2, rms_norm_eps=1e-5, chunk_size=64
    ):
        self.d_model = d_model; self.d_state = d_state; self.d_head = d_head
        self.expand = expand; self.num_layers = num_layers
        self.d_inner = int(expand * d_model); self.n_heads = self.d_inner // d_head
        self.n_groups = n_groups; self.mimo_rank = mimo_rank
        self.rms_norm_eps = rms_norm_eps; self.chunk_size = chunk_size
        self.use_parallel_scan = use_parallel_scan; self.use_kmoe = use_kmoe
        self.kmoe_num_experts = kmoe_num_experts; self.kmoe_top_k = kmoe_top_k
        self.kmoe_r1 = kmoe_r1; self.kmoe_r2 = kmoe_r2; self.kmoe_r3 = kmoe_r3
        self.ffn_expand = ffn_expand; self.num_kv_heads = num_kv_heads
        self.kv_groups = self.n_heads // num_kv_heads
        self.dt_min, self.dt_max, self.dt_init_floor = dt_min, dt_max, dt_init_floor
        self.layer_scale_init = layer_scale_init

# class RMSNorm(LigerRMSNorm):
#     """
#     【標準化】Root Mean Square 層標準化
#     ────────────────────────────────────
#     繼承 liger_kernel 的 LigerRMSNorm，
#     提供 CUDA-level fused kernel（比原生 PyTorch 約 2x 快）。
#     沒有 Bias；縮放係數 γ 可學習。

#     公式：y = x / sqrt(mean(x²) + eps) * γ
#     """
#     def __init__(self, dim, eps=1e-5):
#         super().__init__(hidden_size=dim, eps=eps)


class RMSNorm(nn.RMSNorm):
    """
    【標準化】Root Mean Square 層標準化 (原生 PyTorch)
    ────────────────────────────────────
    使用 PyTorch 內建的 nn.RMSNorm，完美支援 torch.compile 與 CUDA Graphs。
    """
    def __init__(self, dim, eps=1e-5):
        # 官方的參數名稱叫做 normalized_shape，我們把傳進來的 dim 交給它
        super().__init__(normalized_shape=dim, eps=eps)    

class LayerScale(nn.Module):
    """
    【殘差縮放】LayerScale — 每層學習一個縮放係數
    ──────────────────────────────────────────────
    對殘差分支的輸出乘上可學習純量向量 γ（初始值很小）。
    防止深層網路初期訓練時殘差分支破壞主幹特徵，
    有效改善深度模型的訓練穩定性。

    參考：CaiT (Touvron et al. 2021) — "Going deeper with Image Transformers"

    公式：LayerScale(x) = x * γ，  γ ∈ R^d_model
    """
    def __init__(self, dim, init_value=1e-2):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim) * init_value)
    def forward(self, x):
        return x * self.gamma.to(x.dtype)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_R3': 32, 'BLOCK_R2': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_R3': 32, 'BLOCK_R2': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_R3': 64, 'BLOCK_R2': 64},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_R3': 16, 'BLOCK_R2': 128}, num_warps=4, num_stages=4),
    ],
    key=['r3', 'r2'],
)
@triton.jit
def _fused_latent_moe_fwd(
    x_ptr, g_ptr, idx_ptr, prob_ptr, out_ptr,
    stride_xb, stride_xr3, stride_ge, stride_gr3, stride_gr2,
    stride_idxb, stride_idxk, stride_probb, stride_probk, stride_ob, stride_or2,
    B, r3, r2, top_k, BLOCK_R3: tl.constexpr, BLOCK_R2: tl.constexpr
):
    pid_b = tl.program_id(0); pid_r2 = tl.program_id(1)
    offs_r2 = pid_r2 * BLOCK_R2 + tl.arange(0, BLOCK_R2)
    acc = tl.zeros((BLOCK_R2,), dtype=tl.float32)
    for k in range(top_k):
        exp_idx = tl.load(idx_ptr  + pid_b * stride_idxb  + k * stride_idxk)
        prob    = tl.load(prob_ptr + pid_b * stride_probb + k * stride_probk)
        for r3_idx in range(0, r3, BLOCK_R3):
            offs_r3 = r3_idx + tl.arange(0, BLOCK_R3)
            x = tl.load(x_ptr + pid_b * stride_xb + offs_r3 * stride_xr3, mask=offs_r3 < r3, other=0.0)
            g = tl.load(g_ptr + exp_idx * stride_ge + offs_r3[:, None] * stride_gr3 + offs_r2[None, :] * stride_gr2,
                        mask=(offs_r3[:, None] < r3) & (offs_r2[None, :] < r2), other=0.0)
            acc += prob * tl.sum(x[:, None] * g, axis=0)
    tl.store(out_ptr + pid_b * stride_ob + offs_r2 * stride_or2, acc.to(out_ptr.dtype.element_ty), mask=offs_r2 < r2)

def get_dG_bwd_autotune_config():
    return [
        triton.Config({'BLOCK_R3': 32,  'BLOCK_R2': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_R3': 32,  'BLOCK_R2': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_R3': 64,  'BLOCK_R2': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_R3': 64,  'BLOCK_R2': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_R3': 128, 'BLOCK_R2': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_R3': 64,  'BLOCK_R2': 256}, num_warps=8, num_stages=4),
    ]

@triton.autotune(configs=get_dG_bwd_autotune_config(), key=['r3', 'r2', 'B'])
@triton.jit
def _fused_latent_moe_bwd_dG_kernel(
    x_ptr, dout_ptr, prob_ptr, idx_ptr, dG_ptr,
    stride_xb, stride_xr3, stride_doutb, stride_doutr2,
    stride_probb, stride_probk, stride_idxb, stride_idxk,
    stride_dGe, stride_dGr3, stride_dGr2,
    B, top_k, r3, r2, BLOCK_R3: tl.constexpr, BLOCK_R2: tl.constexpr
):
    pid_e = tl.program_id(0); pid_r3 = tl.program_id(1); pid_r2 = tl.program_id(2)
    offs_r3 = pid_r3 * BLOCK_R3 + tl.arange(0, BLOCK_R3)
    offs_r2 = pid_r2 * BLOCK_R2 + tl.arange(0, BLOCK_R2)
    mask_r3 = offs_r3 < r3; mask_r2 = offs_r2 < r2
    acc = tl.zeros((BLOCK_R3, BLOCK_R2), dtype=tl.float32)
    for b in range(B):
        for k in range(top_k):
            e = tl.load(idx_ptr + b * stride_idxb + k * stride_idxk)
            if e == pid_e:
                prob = tl.load(prob_ptr + b * stride_probb + k * stride_probk).to(tl.float32)
                x    = tl.load(x_ptr   + b * stride_xb    + offs_r3 * stride_xr3,   mask=mask_r3, other=0.0).to(tl.float32)
                dout = tl.load(dout_ptr + b * stride_doutb + offs_r2 * stride_doutr2, mask=mask_r2, other=0.0).to(tl.float32)
                acc += x[:, None] * (dout * prob)[None, :]
    tl.store(dG_ptr + pid_e * stride_dGe + offs_r3[:, None] * stride_dGr3 + offs_r2[None, :] * stride_dGr2,
             acc.to(dG_ptr.dtype.element_ty), mask=mask_r3[:, None] & mask_r2[None, :])


# ── MoE Triton Kernels ───────────────────────────────────────────────

class FusedLatentMoE(torch.autograd.Function):
    """
    【Triton Autograd】Tucker 潛在空間 MoE 專家混合層（自定義反向傳播）
    ────────────────────────────────────────────────────────
    Forward：計算純量分解的 Top-K MoE 加譄展開

        x_out[b, :] = Σ_{k=0}^{top_k-1}  prob[b,k] * G_experts[idx[b,k]] @ x_shared[b]

    其中 G_experts = U_expert ×_1 Core（Tucker 專家矩陣）。

    Backward：
        • dx_shared  : 通過 PyTorch bmm 反推
        • dG_experts : 由專屬 Triton kernel (_fused_latent_moe_bwd_dG_kernel) 計算
                       按專家分組續寫，避免鍵竭問題
        • dprobs     : softmax 路由機率的梯度

    輸入：
        x_shared     : (B, r3) 口 Token 在潛在 r3 空間的共享表示
        G_experts    : (E, r3, r2) Tucker 漗合專家矩陣
        top_k_indices: (B, top_k)  選出的專家索引
        top_k_probs  : (B, top_k)  歸一化後的路由機率
    """
    @staticmethod
    def forward(ctx, x_shared, G_experts, top_k_indices, top_k_probs):
        B, r3 = x_shared.shape; E, _, r2 = G_experts.shape; top_k = top_k_indices.size(1)
        ctx.save_for_backward(x_shared, G_experts, top_k_indices, top_k_probs)
        out = torch.empty((B, r2), device=x_shared.device, dtype=x_shared.dtype)
        _fused_latent_moe_fwd[lambda meta: (B, triton.cdiv(r2, meta['BLOCK_R2']))](
            x_shared, G_experts, top_k_indices, top_k_probs, out,
            x_shared.stride(0), x_shared.stride(1),
            G_experts.stride(0), G_experts.stride(1), G_experts.stride(2),
            top_k_indices.stride(0), top_k_indices.stride(1),
            top_k_probs.stride(0), top_k_probs.stride(1),
            out.stride(0), out.stride(1), B, r3, r2, top_k)
        return out

    @staticmethod
    def backward(ctx, dout):
        x_shared, G_experts, top_k_indices, top_k_probs = ctx.saved_tensors
        B, r3 = x_shared.shape; E, _, r2 = G_experts.shape; top_k = top_k_indices.size(1)
        dx_shared = torch.zeros_like(x_shared)
        dprobs    = torch.zeros_like(top_k_probs)

        # 🚀 核心修正：Backward 內 autocast 不作用，必須手動對齊 dtype
        # x_shared.dtype 是 fp16/bf16（forward 時由 autocast 決定）
        target_dtype = x_shared.dtype
        dout = dout.to(target_dtype)   # dout 可能從上層傳入 fp32，強制對齊

        for k in range(top_k):
            idx  = top_k_indices[:, k]
            # top_k_probs 可能是 fp32（softmax 數值穩定），強制轉換
            prob = top_k_probs[:, k].unsqueeze(1).to(target_dtype)
            G_k  = G_experts[idx].to(target_dtype)

            dout_k     = dout * prob
            dx_shared += torch.matmul(dout_k.unsqueeze(1), G_k.transpose(1, 2)).squeeze(1)

            # dprobs 的 dtype 是 top_k_probs 的原始 dtype（可能 fp32），存回前轉換
            dprobs_k        = (dout * torch.matmul(x_shared.unsqueeze(1), G_k).squeeze(1)).sum(dim=-1)
            dprobs[:, k]    = dprobs_k.to(dprobs.dtype)

        dG_experts = torch.zeros_like(G_experts)
        _fused_latent_moe_bwd_dG_kernel[lambda meta: (E, triton.cdiv(r3, meta['BLOCK_R3']), triton.cdiv(r2, meta['BLOCK_R2']))](
            x_shared, dout, top_k_probs, top_k_indices, dG_experts,
            x_shared.stride(0), x_shared.stride(1), dout.stride(0), dout.stride(1),
            top_k_probs.stride(0), top_k_probs.stride(1),
            top_k_indices.stride(0), top_k_indices.stride(1),
            dG_experts.stride(0), dG_experts.stride(1), dG_experts.stride(2), B, top_k, r3, r2)
        return dx_shared, dG_experts, None, dprobs

class TritonTuckerMoE(nn.Module):
    """
    【專家層】Tucker 張量分解 Mixture-of-Experts（核心元件）
    ─────────────────────────────────────────────────────
    將傳統 MoE 的專家矩陣 (n_experts, d_in, d_out) Tucker 分解為：

        考慮 E 個專家：  G[e] = U_expert[e] ×_1 Core  ∈ R^{r3 x r2}
        專家輸出：      y = FusedLatentMoE(U_in @ x, G, top_k_idx, probs) @ U_out + bias

    參數量比較：(vs. 密集 MLP 每專家 d_in*d_out)
        勘筣儲存： U_expert(E*r1) + U_in(d_in*r3) + U_out(r2*d_out) + Core(r1*r3*r2)
        每層專家激活：僅需 top_k / num_experts 的專家參數

    Router：
        使用 ScaledTanh-capped softmax + trainable temperature 退火。
        訓練時有 Load Balancing Loss + Z-Loss 防止 Router Collapse。

    參數：
        dim_in / dim_out  : 輸入輸出維度
        num_experts       : 總專家數 E
        top_k             : 每 token 激活的專家數
        r1, r2, r3        : Tucker Rank
    """
    def __init__(self, dim_in, dim_out, num_experts=8, top_k=2, r1=4, r2=1024, r3=256):
        super().__init__()
        self.num_experts = num_experts; self.top_k = min(top_k, num_experts)
        self.router = nn.Linear(dim_in, num_experts, bias=False)
        nn.init.normal_(self.router.weight, mean=0.0, std=0.02)
        self.U_expert = nn.Parameter(torch.empty(num_experts, r1))
        self.U_in     = nn.Parameter(torch.empty(dim_in, r3))
        self.U_out    = nn.Parameter(torch.empty(r2, dim_out))
        self.core     = nn.Parameter(torch.empty(r1, r3, r2))
        self.bias     = nn.Parameter(torch.zeros(dim_out))
        nn.init.normal_(self.U_expert, std=1.0)
        nn.init.normal_(self.U_in,  std=1.0 / math.sqrt(dim_in))
        nn.init.normal_(self.U_out, std=1.0 / math.sqrt(r2))
        nn.init.normal_(self.core,  std=1.0 / math.sqrt(r1 * r3))

    def forward(self, x, step=None):
        orig_shape = x.shape; x_flat = x.reshape(-1, orig_shape[-1]); B_flat = x_flat.size(0)
        temperature = get_router_temperature(step)
        raw_logits  = self.router(x_flat)
        capped      = fast_scaled_tanh(raw_logits, 10.0)
        z_loss = (torch.mean(torch.logsumexp(capped, dim=-1) ** 2) if self.training else 0.0)
        router_logits = capped / temperature
        router_probs  = torch.softmax(router_logits, dim=-1)
        _, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
        top_k_raw   = router_probs.gather(-1, top_k_indices)
        top_k_probs = top_k_raw / (top_k_raw.sum(-1, keepdim=True) + 1e-6)
        if self.training:
            expert_mask = torch.zeros_like(router_logits).scatter_(1, top_k_indices, 1.0)
            lb_loss = self.num_experts * torch.sum(expert_mask.mean(0) * router_probs.float().mean(0))
        else:
            lb_loss = 0.0
        x_shared  = torch.matmul(x_flat, self.U_in)
        G_experts = torch.einsum('er, rst -> est', self.U_expert, self.core)
        x_core    = FusedLatentMoE.apply(x_shared, G_experts, top_k_indices, top_k_probs).to(x.dtype)
        out = torch.matmul(x_core, self.U_out).reshape(*orig_shape[:-1], -1)
        return out + self.bias, lb_loss, z_loss

TuckerMoE = TritonTuckerMoE

class MixtralMoEFeedForward(nn.Module):
    """
    【FFN】Mixtral 式三段 MoE Feed-Forward Network
    ──────────────────────────────────────────
    在 TransformerBlock 內作為 FFN，三個投影層全部是 TuckerMoE：

        流程：
            gate  = TuckerMoE(x)              ← 潛在幾何門控
            feat  = TuckerMoE(x)              ← 主內容路徑
            y     = TuckerMoE(silu(gate)*feat) ← 下投影（压縮回 d_model）

        d_ff = ceil(ffn_expand * d_model / 256) * 256  ← 對齊到 256 的倍數

    返回：(output, lb_loss, z_loss) —— 輔助損失用於訓練
    """
    def __init__(self, config: Mamba3Config):
        super().__init__()
        d_ff = int(math.ceil(config.ffn_expand * config.d_model / 256) * 256)
        kw = dict(num_experts=config.kmoe_num_experts, top_k=config.kmoe_top_k,
                  r1=config.kmoe_r1, r2=config.kmoe_r2, r3=config.kmoe_r3)
        self.gate_proj = TuckerMoE(config.d_model, d_ff, **kw)
        self.up_proj   = TuckerMoE(config.d_model, d_ff, **kw)
        self.down_proj = TuckerMoE(d_ff, config.d_model, **kw)

    def forward(self, x, step=None):
        gate, lb_g, z_g = self.gate_proj(x, step=step)
        feat, lb_u, z_u = self.up_proj(x, step=step)
        y,    lb_d, z_d = self.down_proj(fast_silu_gating(gate, feat), step=step)
        return y, lb_g + lb_u + lb_d, z_g + z_u + z_d

@triton.jit
def first_order_combine_op(alpha_left, beta_left, alpha_right, beta_right):
    return alpha_right * alpha_left, alpha_right * beta_left + beta_right

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_D': 32},  num_warps=4),
        triton.Config({'BLOCK_D': 64},  num_warps=4),
        triton.Config({'BLOCK_D': 128}, num_warps=8),
        triton.Config({'BLOCK_D': 256}, num_warps=8),
        triton.Config({'BLOCK_D': 512}, num_warps=8),
    ],
    key=['D', 'L'],
)
@triton.jit
def _chunk_scan_kernel(
    alpha_ptr, u_ptr, h_out_ptr,
    stride_a_b, stride_a_l, stride_u_b, stride_u_l, stride_u_d,
    B_flat, L: tl.constexpr, D: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid_b = tl.program_id(0); pid_d = tl.program_id(1)
    offset_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offset_l = tl.arange(0, L)
    mask_d = offset_d < D
    alpha = tl.load(alpha_ptr + pid_b * stride_a_b + offset_l * stride_a_l)
    u     = tl.load(u_ptr + pid_b * stride_u_b + offset_l[:, None] * stride_u_l + offset_d[None, :] * stride_u_d,
                    mask=mask_d[None, :], other=0.0)
    alpha_fp32 = alpha.to(tl.float32); u_fp32 = u.to(tl.float32)
    _, h_out_fp32 = tl.associative_scan(
        (tl.broadcast_to(alpha_fp32[:, None], (L, BLOCK_D)), u_fp32),
        axis=0, combine_fn=first_order_combine_op)
    tl.store(h_out_ptr + pid_b * stride_u_b + offset_l[:, None] * stride_u_l + offset_d[None, :] * stride_u_d,
             h_out_fp32.to(u.dtype), mask=mask_d[None, :])

def fast_triton_chunk_scan(log_alpha_chunk, u_chunk):
    B, num_chunks, L, H = log_alpha_chunk.shape; D = u_chunk.shape[-1] * u_chunk.shape[-2]
    alpha_flat = torch.exp(log_alpha_chunk).transpose(2, 3).reshape(-1, L).contiguous()
    u_flat     = u_chunk.transpose(2, 3).reshape(-1, L, D).contiguous()
    h_out_flat = torch.empty_like(u_flat)
    B_flat = alpha_flat.shape[0]
    _chunk_scan_kernel[lambda meta: (B_flat, triton.cdiv(D, meta['BLOCK_D']))](
        alpha_flat, u_flat, h_out_flat,
        alpha_flat.stride(0), alpha_flat.stride(1),
        u_flat.stride(0), u_flat.stride(1), u_flat.stride(2),
        B_flat=B_flat, L=L, D=D)
    return h_out_flat.reshape(B, num_chunks, H, L, u_chunk.shape[-2], u_chunk.shape[-1]).transpose(2, 3)


# ── Main Architecture Blocks ──────────────────────────────────────────

class Mamba3Block(nn.Module):
    """
    【核心區塊】Mamba-3 SSM 區塊（含 MIMO + RoPE + TuckerMoE 投影）
    ───────────────────────────────────────────────────────────
    本區塊包含兩個殘差分支：

    (1) Mamba SSM 分支：
        in_proj → x_up (TuckerMoE) → Chunk Parallel Scan (SSD) →
        y_down_proj → mamba_dense_proj * SiLU(z) + D*x_prime
        加上 RoPE 旋轉的 B/C 矩陣、可學習频率 theta_log。
        LSTM-like lambda gate 控制新舊輸入的混合比例。

    (2) Out-Proj 分支斯：
        norm_out_proj → out_proj (TuckerMoE) → LayerScale

    兩層獨立的 LayerScale 、RMSNorm 、LayerScale·ls_mamba / ls_out_proj
    對殘差分支做縮放。

    返回： (hidden, lb_loss, z_loss)
    """
    def __init__(self, config: Mamba3Config):
        super().__init__()
        self.config = config
        d_in, H, G, P, N, R = config.d_model, config.n_heads, config.n_groups, config.d_head, config.d_state, config.mimo_rank
        self.ratio, self.dim_z, self.dim_x = H // G, H * P, H * P
        self.dim_B, self.dim_C, self.dim_dt, self.dim_A, self.dim_lambda = G*N*R, G*N*R, G, G, G
        self.in_proj = nn.Linear(d_in, self.dim_z+self.dim_x+self.dim_B+self.dim_C+self.dim_dt+self.dim_A+self.dim_lambda, bias=True)
        if config.use_kmoe:
            kw = dict(num_experts=config.kmoe_num_experts, top_k=config.kmoe_top_k, r1=config.kmoe_r1, r2=config.kmoe_r2, r3=config.kmoe_r3)
            self.x_up_proj = TuckerMoE(H*P, H*P*R, **kw)
            self.out_proj  = TuckerMoE(d_in, d_in, **kw)
        else:
            self.x_up_proj = nn.Linear(P, P*R, bias=False)
            self.out_proj  = nn.Linear(d_in, d_in, bias=False)
        self.y_down_proj      = nn.Linear(P*R, P, bias=False)
        self.theta_log        = nn.Parameter(torch.randn(G, N//2))
        self.D                = nn.Parameter(torch.ones(H))
        self.norm_B           = RMSNorm(N*R, eps=config.rms_norm_eps)
        self.norm_C           = RMSNorm(N*R, eps=config.rms_norm_eps)
        self.bias_B           = nn.Parameter(torch.zeros(G, N, R))
        self.bias_C           = nn.Parameter(torch.zeros(G, N, R))
        self.mamba_dense_proj = nn.Linear(config.d_inner, d_in, bias=False)
        self.pre_gate_norm    = RMSNorm(H*P)
        self.act              = nn.SiLU()
        self.norm_mamba       = RMSNorm(config.d_model)
        self.norm_out_proj    = RMSNorm(config.d_model)
        self.ls_mamba         = LayerScale(config.d_model, init_value=config.layer_scale_init)
        self.ls_out_proj      = LayerScale(config.d_model, init_value=config.layer_scale_init)
        with torch.no_grad():
            self.bias_B.fill_(1.0); self.bias_C.fill_(1.0)
            dt = torch.clamp(torch.exp(torch.rand(G) * (math.log(config.dt_max) - math.log(config.dt_min)) + math.log(config.dt_min)), min=config.dt_init_floor)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            dt_start = self.dim_z + self.dim_x + self.dim_B + self.dim_C
            dt_end = dt_start + self.dim_dt; A_end = dt_end + self.dim_A
            self.in_proj.bias[dt_start:dt_end].copy_(inv_dt)
            self.in_proj.bias[dt_end:A_end].uniform_(1, 16).log_()
            self.in_proj.bias[A_end:].fill_(-3.0)

    def apply_rope(self, x, angles):
        N_half = angles.shape[-1]
        x_reshaped = x.float().view(*x.shape[:-2], N_half, 2, x.shape[-1])
        x1, x2 = x_reshaped[..., 0, :], x_reshaped[..., 1, :]
        sin_a, cos_a = torch.sin(angles).unsqueeze(-1), torch.cos(angles).unsqueeze(-1)
        return torch.stack([x1*cos_a - x2*sin_a, x2*cos_a + x1*sin_a], dim=-2).reshape_as(x).type_as(x)

    def segsum(self, x):
        x_cumsum = torch.cumsum(x, dim=-1)
        mask = torch.tril(torch.ones(x.size(-1), x.size(-1), device=x.device, dtype=torch.bool))
        return (x_cumsum[..., :, None] - x_cumsum[..., None, :]).masked_fill(~mask, -float("inf"))

    def chunk_parallel_scan(self, u, dt, A, C, chunk_size=128):
        B, L, H, N, P = u.shape; R = C.shape[-1]; input_dtype = u.dtype; L_orig = L
        if L % chunk_size != 0:
            pad = chunk_size - (L % chunk_size)
            u = F.pad(u, (0,0,0,0,0,0,0,pad)); dt = F.pad(dt,(0,0,0,pad))
            C = F.pad(C, (0,0,0,0,0,0,0,pad)); A = F.pad(A, (0,0,0,pad)); L += pad
        nc = L // chunk_size
        log_alpha = dt * A
        u_c  = u.view(B, nc, chunk_size, H, N, P)
        la_c = log_alpha.view(B, nc, chunk_size, H)
        C_c  = C.view(B, nc, chunk_size, H, N, R)
        h_intra = fast_triton_chunk_scan(la_c, u_c)
        y_diag  = torch.einsum("bclhnp, bclhnr -> bclhpr", h_intra, C_c)
        decay   = torch.exp(torch.sum(la_c, dim=2))
        h_prev  = torch.zeros(B, H, N, P, device=u.device, dtype=input_dtype)
        h_inter = torch.empty(B, nc, H, N, P, device=u.device, dtype=input_dtype)
        for c in range(nc):
            h_inter[:, c] = h_prev
            h_prev = h_prev * decay[:, c].view(B, H, 1, 1) + h_intra[:, c, -1]
        c_dec = C_c * torch.exp(torch.cumsum(la_c, dim=2)).unsqueeze(-1).unsqueeze(-1)
        y_off = torch.einsum("bchnp, bclhnr -> bclhpr", h_inter, c_dec)
        y = (y_diag + y_off).view(B, -1, H, P, R)
        return (y[:, :L_orig] if L_orig < L else y).to(input_dtype), h_prev.to(input_dtype)

    def forward(self, x, step=None):
        B_sz, L, _ = x.shape
        H, G, P, N, R, ratio = self.config.n_heads, self.config.n_groups, self.config.d_head, self.config.d_state, self.config.mimo_rank, self.ratio
        residual_mamba, u = x, self.norm_mamba(x)
        z, x_prime, B_param, C_param, dt, A_param, lambda_param = torch.split(
            self.in_proj(u), [self.dim_z, self.dim_x, self.dim_B, self.dim_C, self.dim_dt, self.dim_A, self.dim_lambda], dim=-1)
        x_prime = x_prime.view(B_sz, L, H, P)
        dt = F.softplus(dt); A = -torch.exp(A_param); theta = torch.exp(self.theta_log)
        bg = lambda t: t.repeat_interleave(ratio, dim=2)
        dt_b = bg(dt.unsqueeze(-1)).squeeze(-1); A_b = bg(A.unsqueeze(-1)).squeeze(-1)
        angles = torch.cumsum(torch.einsum("blh, hn -> blhn", dt_b, theta.repeat_interleave(ratio, dim=0)), dim=1)
        B_rotated = self.apply_rope(bg(self.norm_B(B_param.reshape(B_sz,L,G,N*R)).view(B_sz,L,G,N,R) + self.bias_B), angles)
        C_rotated = self.apply_rope(bg(self.norm_C(C_param.reshape(B_sz,L,G,N*R)).view(B_sz,L,G,N,R) + self.bias_C), angles)
        if self.config.use_kmoe:
            x_up, lb_up, z_up = self.x_up_proj(x_prime.view(B_sz, L, -1), step=step)
            x_ssm = x_up.view(B_sz, L, H, P, R)
        else:
            x_ssm, lb_up, z_up = self.x_up_proj(x_prime).view(B_sz,L,H,P,R), 0.0, 0.0
        input_signal = torch.einsum("blhnr, blhpr -> blhnp", B_rotated, x_ssm)
        lv = F.sigmoid(bg(lambda_param.unsqueeze(-1)).squeeze(-1)).view(B_sz,L,H,1,1)
        dv = dt_b.view(B_sz,L,H,1,1); av = torch.exp(dt_b*A_b).view(B_sz,L,H,1,1)
        ip = torch.roll(input_signal,1,1); ip[:,0] = 0
        u_ssm = lv*dv*input_signal + (1-lv)*dv*av*ip

        
        if self.config.use_parallel_scan:
            y_stack, _ = self.chunk_parallel_scan(u_ssm, dt_b, A_b, C_rotated, chunk_size=self.config.chunk_size)
        else:
            h_s = torch.zeros(B_sz,H,N,P,device=x.device); y_list=[]
            for t in range(L):
                h_s = h_s * av[:,t] + u_ssm[:,t]
                y_list.append(torch.einsum("bhnp,bhnr->bhpr", h_s, C_rotated[:,t]))
            y_stack = torch.stack(y_list, dim=1)
        y = self.y_down_proj(y_stack.view(B_sz,L,H,P*R)).view(B_sz,L,H*P)
        y = y + x_prime.reshape(B_sz,L,H*P) * self.D.repeat_interleave(P,dim=0)
        mamba_out = self.mamba_dense_proj(self.pre_gate_norm(y) * self.act(z))
        mid_x = residual_mamba + self.ls_mamba(mamba_out)
        residual_proj, normed_mid = mid_x, self.norm_out_proj(mid_x)
        if self.config.use_kmoe:
            proj_out, lb_out, z_out = self.out_proj(normed_mid, step=step)
        else:
            proj_out, lb_out, z_out = self.out_proj(normed_mid), 0.0, 0.0
        return residual_proj + self.ls_out_proj(proj_out), lb_up + lb_out, z_up + z_out

class TransformerBlock(nn.Module):
    """
    【注意力區塊】Grouped Query Attention (GQA) + MoE FFN
    ────────────────────────────────────────────────
    區塊內部包含兩個殘差分支：

    (1) 自注意力：
        - Q-Head: n_heads = d_model // 64 組
        - KV-Head: num_kv_heads 組（GQA — kv_groups = n_heads // num_kv_heads）
        - FlashAttention 透過 F.scaled_dot_product_attention（因果遅燬注意力遮罩）
        - out_proj: 羮集層輸出

    (2) FFN：若 use_kmoe=True 兩段使用 MixtralMoEFeedForward，
              否則使用駔滿式 GLU-FFN (SwiGLU)。

    兩層獨立的 RMSNorm + LayerScale。

    返回： (hidden, lb_loss, z_loss)
    """
    def __init__(self, config: Mamba3Config):
        super().__init__()
        self.head_dim=64; self.num_heads=config.d_model//64
        self.num_kv_heads=config.num_kv_heads; self.kv_groups=self.num_heads//config.num_kv_heads
        self.q_proj  = nn.Linear(config.d_model, self.num_heads*64, bias=False)
        self.k_proj  = nn.Linear(config.d_model, self.num_kv_heads*64, bias=False)
        self.v_proj  = nn.Linear(config.d_model, self.num_kv_heads*64, bias=False)
        self.o_proj  = nn.Linear(config.d_model, config.d_model, bias=True)
        self.norm_attn = RMSNorm(config.d_model); self.use_kmoe = config.use_kmoe
        if config.use_kmoe:
            self.ffn = MixtralMoEFeedForward(config)
        else:
            d_ff = int(math.ceil(8*config.d_model/3/256)*256)
            self.ffn_gate = nn.Linear(config.d_model, d_ff, bias=False)
            self.ffn_up   = nn.Linear(config.d_model, d_ff, bias=False)
            self.ffn_down = nn.Linear(d_ff, config.d_model, bias=False)
        self.norm_ffn = RMSNorm(config.d_model)
        self.ls_attn = LayerScale(config.d_model, init_value=config.layer_scale_init)
        self.ls_ffn  = LayerScale(config.d_model, init_value=config.layer_scale_init)

    def forward(self, x, step=None):
        B, L, D = x.shape; residual, nx = x, self.norm_attn(x)
        q = self.q_proj(nx).view(B,L,self.num_heads,64).transpose(1,2)
        k = self.k_proj(nx).view(B,L,self.num_kv_heads,64).transpose(1,2)
        v = self.v_proj(nx).view(B,L,self.num_kv_heads,64).transpose(1,2)
        k = k.unsqueeze(2).expand(B,self.num_kv_heads,self.kv_groups,L,64).reshape(B,self.num_heads,L,64)
        v = v.unsqueeze(2).expand(B,self.num_kv_heads,self.kv_groups,L,64).reshape(B,self.num_heads,L,64)
        attn = F.scaled_dot_product_attention(q,k,v,dropout_p=0.0,is_causal=True)
        x = residual + self.ls_attn(self.o_proj(attn.transpose(1,2).contiguous().view(B,L,D)))
        residual, h = x, self.norm_ffn(x)
        if self.use_kmoe:
            ffn_out, lb, z = self.ffn(h, step=step)
        else:
            ffn_out = self.ffn_down(fast_silu_gating(self.ffn_gate(h), self.ffn_up(h))); lb=0.0; z=0.0
        return residual + self.ls_ffn(ffn_out), lb, z

class TrueHybridMamba(nn.Module):
    """
    【Backbone】混合型 Mamba3 主幹（Mamba 區塊 + Attention 區塊 交採堆疊）
    ────────────────────────────────────────────────────────────
    每個 Macro Block = mamba_ratio 個 Mamba3Block + 1 個 TransformerBlock。
    共堆疊 num_layers 層，總層數 = num_layers * (mamba_ratio + 1)。

    重要設計小節：
      • 全部區塊除了結構外一薇相同，方便 checkpoint 和 compile。
      • 使用 torch.utils.checkpoint 將每個 Block 包複，
        成倒倒就活化歸一化梯度檢查點技術（Gradient Checkpointing）。
      • 累積 Load Balancing Loss 和 Z-Loss。

    返回： (hidden_states, total_lb_loss, total_z_loss)
    """
    def __init__(self, config: Mamba3Config, mamba_ratio=4):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(config.num_layers):
            for _ in range(mamba_ratio):
                self.layers.append(nn.ModuleDict({"block": Mamba3Block(config)}))
            self.layers.append(nn.ModuleDict({"block": TransformerBlock(config)}))

    def forward(self, x, step=None):
        total_lb, total_z = 0.0, 0.0
        for ld in self.layers:
            x, lb, z = checkpoint(ld["block"], x, step, use_reentrant=False)
            if isinstance(lb, torch.Tensor): total_lb = total_lb + lb; total_z = total_z + z
        return x, total_lb, total_z

class Mamba3LanguageModel(nn.Module):
    """
    【頂層模型】Mamba3 語言模型（含 Causal LM 損失計算）
    ───────────────────────────────────────────────────
    結構：
        embed (Embedding)   --設計 tied weights  --> head (Linear)
            ↓
        TrueHybridMamba (Backbone)
            ↓
        RMSNorm
            ↓
        ScaledTanh-capped Logits (scale=30)

    Forward 輸入/輸出：
        輸入： input_ids (B, L)、labels (B, L)、step (int 或 None)
        輸出 (labels 不為 None)：5-tuple
            (loss, lb_tensor, ce_loss.detach(), lb_contrib.detach(), z_contrib.detach())
        輸出 (labels 為 None)： logits (B, L, Vocab)

    損失公式：
        loss = CE_loss + (0.1/n_moe) * LB_loss + (1e-3/n_moe) * Z_loss

    注意：
        self._last_loss_terms 已將註解挎去，避免 CUDA Graph breaks。
        訓練時請從返回的 5-tuple 取得分量損失。
    """
    def __init__(self, config: Mamba3Config, vocab_size: int, **kwargs):
        super().__init__()
        self.config = config
        self.embed    = nn.Embedding(vocab_size, config.d_model)
        self.backbone = TrueHybridMamba(config)
        self.norm     = RMSNorm(config.d_model)
        self.head     = nn.Linear(config.d_model, vocab_size, bias=False)
        self.head.weight = self.embed.weight
        self.ce_loss_fn  = nn.CrossEntropyLoss()
        self._last_loss_terms = None
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None, step=None):
        backbone_out = self.backbone(self.embed(input_ids), step=step)
        hidden = self.norm(backbone_out[0])
        total_lb_loss, total_z_loss = backbone_out[1], backbone_out[2]
        logits = fast_scaled_tanh(self.head(hidden / math.sqrt(self.config.d_model)), 30.0)
        if labels is not None:
            ce_loss = self.ce_loss_fn(logits.float().view(-1, logits.size(-1)), labels.view(-1))
            if isinstance(total_lb_loss, torch.Tensor):
                total_lb_loss = total_lb_loss.mean(); total_z_loss = total_z_loss.mean()
            n = self.config.num_layers * (4*2 + 1*3)
            lb_contrib = (0.1 / max(1, n)) * total_lb_loss
            z_contrib  = (1e-3 / max(1, n)) * total_z_loss
            loss = ce_loss + lb_contrib + z_contrib
            # NOTE: self._last_loss_terms mutation removed to prevent CUDA Graph breaks
            # Return (loss, lb_contrib, z_contrib) as a 3-tuple for logging
            return (
                loss.unsqueeze(0),
                total_lb_loss.detach().unsqueeze(0) if isinstance(total_lb_loss, torch.Tensor) else loss.unsqueeze(0),
                ce_loss.detach(), lb_contrib.detach(), z_contrib.detach(),
            )
        return logits


# ┌──────────────────────────────────────────────────────────────────┐
# │  §10  train() — 完整訓練迴圈                                       │
# └──────────────────────────────────────────────────────────────────┘



def print_model_analysis(model, config, vocab_size):
    """顯示模型參數分類、TuckerMoE 分析及所有超參數。"""
    total_params = 0; trainable_params = 0; active_params = 0
    bucket = {"embed_head": 0, "mamba_ssm": 0, "cpmoe_router": 0, "cpmoe_U_expert": 0,
              "cpmoe_bias": 0, "layer_scale": 0, "norm": 0, "attn_proj": 0, "other": 0}

    for name, p in model.named_parameters():
        num_p = p.numel(); total_params += num_p
        if not p.requires_grad: continue
        trainable_params += num_p
        if any(k in name for k in ["U_expert", "U_in", "U_out", "core"]):
            active_params += int(num_p * (config.kmoe_top_k / config.kmoe_num_experts))
            bucket["cpmoe_U_expert"] += num_p
        else:
            active_params += num_p
            if "embed" in name or "head.weight" in name:     bucket["embed_head"] += num_p
            elif "router" in name:                           bucket["cpmoe_router"] += num_p
            elif ".bias" in name and any(k in name for k in ("gate_proj","up_proj","down_proj","out_proj","x_up_proj")): bucket["cpmoe_bias"] += num_p
            elif "ls_" in name or "gamma" in name:          bucket["layer_scale"] += num_p
            elif any(k in name for k in ("norm_","norm.")): bucket["norm"] += num_p
            elif any(k in name for k in ("q_proj","k_proj","v_proj","o_proj")): bucket["attn_proj"] += num_p
            elif any(k in name for k in ("in_proj","y_down_proj","mamba_dense","theta_log",".D","bias_B","bias_C")): bucket["mamba_ssm"] += num_p
            else: bucket["other"] += num_p

    num_cpmoe = len([m for _, m in model.named_modules() if "TuckerMoE" in type(m).__name__])
    active_cpmoe = int(bucket["cpmoe_U_expert"] * config.kmoe_top_k / config.kmoe_num_experts)
    W = 68
    print("═" * W)
    print("🚀  Mamba3 Hybrid TuckerMoE  ·  Model Analysis")
    print("═" * W)
    print("📦  【總參數覽表】")
    print(f"   {'總參數量  (Total)':.<42} {total_params/1e6:>8.2f} M")
    print(f"   {'可訓練    (Trainable)':.<42} {trainable_params/1e6:>8.2f} M")
    print(f"   {'⚡ 實際激活 (Active / step)':.<42} {active_params/1e6:>8.2f} M  ({active_params/max(1,trainable_params)*100:.1f}%)")
    print(f"\n{'─'*W}")
    print("🔬  【參數分類明細】")
    labels = {"embed_head": "Embedding + LM-Head (tied)", "mamba_ssm": "Mamba SSM 核心",
              "cpmoe_router": "TuckerMoE Router", "cpmoe_U_expert": "TuckerMoE 核心張量 & 模式矩陣",
              "cpmoe_bias": "TuckerMoE Bias", "layer_scale": "LayerScale γ",
              "norm": "RMSNorm", "attn_proj": "Attention Q/K/V/O", "other": "其他"}
    for key, label in labels.items():
        val = bucket[key]
        if val == 0: continue
        pct = val / max(1, trainable_params) * 100
        bar = "█" * int(pct/2) + "░" * (25 - int(pct/2))
        print(f"   {label:<40} {val/1e6:>7.2f} M  {pct:>5.1f}%  {bar}")
    print(f"\n{'─'*W}")
    print("🧩  【TuckerMoE 分析】")
    print(f"   {'模組總數':.<42} {num_cpmoe:>4} 個")
    print(f"   {'全 Tucker 參數 (全儲存)':.<42} {bucket['cpmoe_U_expert']/1e6:>7.2f} M")
    print(f"   {'每步實際激活':.<42} {active_cpmoe/1e6:>7.2f} M")
    print(f"   {'Tucker Rank (r1, r2, r3)':.<42} {config.kmoe_r1}, {config.kmoe_r2}, {config.kmoe_r3}")
    print(f"   {'FFN 擴張倍率':.<42} {config.ffn_expand}")
    print(f"   {'Vocab Size':.<42} {vocab_size}")
    print("═" * W)


def validate_config(**kw):
    """
    🛡️ 防呆驗證：在訓練開始前檢查所有超參數、路徑、GPU。
    ANY 錯誤都會直接 raise，讓你在燒 GPU 前知道問題在哪。
    """
    errors   = []
    warnings = []
    ok_tag   = "  ✅"
    warn_tag = "  ⚠️"
    err_tag  = "  ❌"

    W = 68
    print("═" * W)
    print("🛡️  Config Validation — 訓練前安全檢查")
    print("═" * W)

    # ── 1. GPU / CUDA ─────────────────────────────────────────────
    if not torch.cuda.is_available():
        errors.append("CUDA 不可用！請確認 PyTorch 已安裝 GPU 版本且驅動正常。")
        print(f"{err_tag} CUDA Not Available")
    else:
        n_gpu = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(n_gpu)]
        print(f"{ok_tag} CUDA OK — {n_gpu} GPU(s): {', '.join(gpu_names)}")
        # 檢查目標精度是否支援
        cap = torch.cuda.get_device_capability(0)
        if cap[0] < 8 and MIXED_PRECISION == "bf16":
            warnings.append(f"GPU Compute Capability {cap} < 8.0，BF16 可能不受支援，建議改為 fp16。")
            print(f"{warn_tag} BF16 on sm_{cap[0]}{cap[1]} — 可能 fallback 到 fp32")
        else:
            print(f"{ok_tag} Mixed Precision '{MIXED_PRECISION}' OK (sm_{cap[0]}{cap[1]})")

    # ── 2. 模型結構一致性 ─────────────────────────────────────────
    D_MODEL       = kw["D_MODEL"]
    D_HEAD        = kw["D_HEAD"]
    EXPAND        = kw["EXPAND"]
    NUM_KV_HEADS  = kw["NUM_KV_HEADS"]
    KMOE_TOP_K    = kw["KMOE_TOP_K"]
    KMOE_NUM_EXPERTS = kw["KMOE_NUM_EXPERTS"]
    CHUNK_SIZE    = kw["CHUNK_SIZE"]
    SEQ_LEN       = kw["SEQ_LEN"]
    BATCH_SIZE    = kw["BATCH_SIZE"]

    d_inner = int(EXPAND * D_MODEL)
    n_heads = d_inner // D_HEAD

    if d_inner % D_HEAD != 0:
        errors.append(f"d_inner ({d_inner}) 必須能被 D_HEAD ({D_HEAD}) 整除。")
        print(f"{err_tag} d_inner % D_HEAD != 0  →  {d_inner} / {D_HEAD}")
    else:
        print(f"{ok_tag} d_inner={d_inner}, n_heads={n_heads}, D_HEAD={D_HEAD}  ✓ 整除")

    if n_heads % NUM_KV_HEADS != 0:
        errors.append(f"n_heads ({n_heads}) 必須能被 NUM_KV_HEADS ({NUM_KV_HEADS}) 整除（GQA 要求）。")
        print(f"{err_tag} n_heads % NUM_KV_HEADS != 0  →  {n_heads} / {NUM_KV_HEADS}")
    else:
        print(f"{ok_tag} GQA: n_heads={n_heads} / NUM_KV_HEADS={NUM_KV_HEADS} = {n_heads//NUM_KV_HEADS}  ✓")

    if D_MODEL % D_HEAD != 0:
        errors.append(f"D_MODEL ({D_MODEL}) 必須能被 D_HEAD ({D_HEAD}) 整除（Attention）。")
        print(f"{err_tag} D_MODEL % D_HEAD != 0  →  {D_MODEL} / {D_HEAD}")
    else:
        print(f"{ok_tag} D_MODEL={D_MODEL} / D_HEAD={D_HEAD}  ✓ (Attn heads: {D_MODEL//D_HEAD})")

    if KMOE_TOP_K > KMOE_NUM_EXPERTS:
        errors.append(f"KMOE_TOP_K ({KMOE_TOP_K}) 不能大於 KMOE_NUM_EXPERTS ({KMOE_NUM_EXPERTS})。")
        print(f"{err_tag} TOP_K > NUM_EXPERTS  →  {KMOE_TOP_K} > {KMOE_NUM_EXPERTS}")
    else:
        print(f"{ok_tag} MoE: top_k={KMOE_TOP_K} / num_experts={KMOE_NUM_EXPERTS}  ✓")

    if SEQ_LEN < CHUNK_SIZE:
        errors.append(f"SEQ_LEN ({SEQ_LEN}) 不能小於 CHUNK_SIZE ({CHUNK_SIZE})。")
        print(f"{err_tag} SEQ_LEN < CHUNK_SIZE  →  {SEQ_LEN} < {CHUNK_SIZE}")
    else:
        chunks = SEQ_LEN // CHUNK_SIZE
        print(f"{ok_tag} SEQ_LEN={SEQ_LEN} / CHUNK_SIZE={CHUNK_SIZE} = {chunks} chunks  ✓")

    # ── 3. 路徑防呆 ───────────────────────────────────────────────
    DATA_PATH = kw["DATA_PATH"]
    OUTPUT_DIR = kw["OUTPUT_DIR"]
    PRETRAINED_EMBED_PATH = kw["PRETRAINED_EMBED_PATH"]

    if not os.path.isfile(DATA_PATH):
        errors.append(f"DATA_PATH 不存在：'{DATA_PATH}'")
        print(f"{err_tag} DATA_PATH 不存在  →  {DATA_PATH}")
    else:
        size_gb = os.path.getsize(DATA_PATH) / 1e9
        print(f"{ok_tag} DATA_PATH OK  →  {DATA_PATH}  ({size_gb:.2f} GB)")

    if PRETRAINED_EMBED_PATH and not os.path.isfile(PRETRAINED_EMBED_PATH):
        warnings.append(f"PRETRAINED_EMBED_PATH 指定但不存在：'{PRETRAINED_EMBED_PATH}'，將略過。")
        print(f"{warn_tag} PRETRAINED_EMBED_PATH 不存在，將略過  →  {PRETRAINED_EMBED_PATH}")
    elif PRETRAINED_EMBED_PATH:
        print(f"{ok_tag} PRETRAINED_EMBED_PATH OK  →  {PRETRAINED_EMBED_PATH}")
    else:
        print(f"  ➖  PRETRAINED_EMBED_PATH 為空，從頭初始化")

    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        test_file = os.path.join(OUTPUT_DIR, ".write_test")
        with open(test_file, "w") as f: f.write("test")
        os.remove(test_file)
        print(f"{ok_tag} OUTPUT_DIR 可寫入  →  {OUTPUT_DIR}")
    except Exception as e:
        errors.append(f"OUTPUT_DIR 無法寫入：'{OUTPUT_DIR}'  ({e})")
        print(f"{err_tag} OUTPUT_DIR 無法寫入  →  {OUTPUT_DIR}")

    # ── 4. 記憶體估算 (粗略) ───────────────────────────────────────
    if torch.cuda.is_available():
        VOCAB_SIZE = kw["VOCAB_SIZE"]
        MIMO_RANK  = kw["MIMO_RANK"]
        NUM_LAYERS = kw["NUM_LAYERS"]
        d_state    = kw["D_STATE"]
        # 粗估參數量：Embedding + Backbone(近似) + Head
        est_embed  = D_MODEL * VOCAB_SIZE * 2  # fp16 bytes (tied)
        est_mamba  = (NUM_LAYERS * 4) * (d_inner * 2 + d_inner * d_state * MIMO_RANK * 4) * 2
        est_total_bytes = (est_embed + est_mamba) * BATCH_SIZE / 1e9
        gpu_free_gb = (torch.cuda.get_device_properties(0).total_memory -
                       torch.cuda.memory_allocated()) / 1e9
        print(f"  💾 粗估每批模型記憶體佔用：{est_total_bytes:.1f} GB  |  GPU 可用：{gpu_free_gb:.1f} GB")
        if est_total_bytes > gpu_free_gb * 0.8:
            warnings.append(f"估計記憶體 ({est_total_bytes:.1f}GB) 接近 GPU 可用量 ({gpu_free_gb:.1f}GB)，考慮縮小 BATCH_SIZE 或 SEQ_LEN。")
            print(f"{warn_tag} 記憶體可能不足，請考慮縮小 BATCH_SIZE 或 SEQ_LEN")
        else:
            print(f"{ok_tag} 記憶體估算充足")

    # ── 5. 訓練超參數合理性 ────────────────────────────────────────
    LR         = kw["LR"]
    WARMUP     = kw["STEPS"]
    STEPS      = kw["STEPS"]
    WARMUP_VAL = kw["WARMUP"]

    if LR > 1e-2:
        warnings.append(f"LR={LR} 異常偏高，通常應 < 1e-3，請確認是否正確。")
        print(f"{warn_tag} LR={LR} 異常偏高！")
    elif LR < 1e-6:
        warnings.append(f"LR={LR} 異常偏低，學習可能停滯。")
        print(f"{warn_tag} LR={LR} 異常偏低！")
    else:
        print(f"{ok_tag} LR={LR}  ✓")

    if WARMUP_VAL >= STEPS:
        errors.append(f"WARMUP ({WARMUP_VAL}) 必須小於 STEPS ({STEPS})。")
        print(f"{err_tag} WARMUP >= STEPS  →  {WARMUP_VAL} >= {STEPS}")
    else:
        print(f"{ok_tag} WARMUP={WARMUP_VAL} / STEPS={STEPS}  ✓")

    # ── 結果彙總 ──────────────────────────────────────────────────
    print(f"\n{'─'*W}")
    if warnings:
        print(f"⚠️  {len(warnings)} 項警告（不阻止訓練，但請注意）：")
        for w in warnings: print(f"   • {w}")
    if errors:
        print(f"\n❌  發現 {len(errors)} 項錯誤，訓練無法啟動：")
        for e in errors: print(f"   • {e}")
        print("═" * W)
        raise ValueError(f"Config 驗證失敗，共 {len(errors)} 項錯誤，請修正後重啟。")
    else:
        print(f"✅  所有檢查通過！準備啟動訓練 🚀")
    print("═" * W + "\n")


def train(
    # ── 模型超參數
    D_MODEL=768, D_STATE=64, D_HEAD=64, EXPAND=2, NUM_LAYERS=6,
    MIMO_RANK=4, NUM_KV_HEADS=4, CHUNK_SIZE=64,
    # ── TuckerMoE
    KMOE_NUM_EXPERTS=8, KMOE_TOP_K=2,
    KMOE_R1=32, KMOE_R2=512, KMOE_R3=256, FFN_EXPAND=6,
    # ── 資料集與路徑
    DATA_PATH="/kaggle/input/datasets/s990093/fineweb-edu-tokenized-32007/fineweb_tokenized.bin",
    OUTPUT_DIR="output/",
    LOG_FILE="output/train_log.csv",
    CHECKPOINT_SAVE_PATH="output/checkpoint.pt",
    PRETRAINED_EMBED_PATH="",
    VOCAB_SIZE=32007,
    SEQ_LEN=512,
    # ── 訓練超參數
    BATCH_SIZE=2,
    GRADIENT_ACCUMULATION_STEPS=8,
    LR=8e-5, WARMUP=400, STEPS=60000, CHECKPOINT_EVERY=100,
    # ── Router 退火
    ROUTER_T_START=2.0, ROUTER_T_END=0.5,
    # ── 模式
    TRAIN_MODE=True,
    # ── torch.compile
    ENABLE_RESUME_COMPILE_WARMUP=True,  # True = 續訓時先做 dummy pass 預熱 compile，再載 optimizer state
    COMPILE_MODE="default",       # "default" | "reduce-overhead" | "max-autotune"
    COMPILE_FULLGRAPH=False,      # True = 單體圖（需模型無 Python mutation）
    # ── 診斷
    GRAD_CHECK_INTERVAL=50,       # 每 N steps 印出梯度診斷
):
   # 🛡️ 防呆驗證 — 在任何 GPU / 模型初始化之前執行
    validate_config(
        D_MODEL=D_MODEL, D_STATE=D_STATE, D_HEAD=D_HEAD, EXPAND=EXPAND,
        NUM_LAYERS=NUM_LAYERS, MIMO_RANK=MIMO_RANK, NUM_KV_HEADS=NUM_KV_HEADS,
        CHUNK_SIZE=CHUNK_SIZE, KMOE_NUM_EXPERTS=KMOE_NUM_EXPERTS, KMOE_TOP_K=KMOE_TOP_K,
        VOCAB_SIZE=VOCAB_SIZE, DATA_PATH=DATA_PATH, OUTPUT_DIR=OUTPUT_DIR,
        PRETRAINED_EMBED_PATH=PRETRAINED_EMBED_PATH,
        LR=LR, WARMUP=WARMUP, STEPS=STEPS, BATCH_SIZE=BATCH_SIZE, SEQ_LEN=SEQ_LEN,
    )

    accelerator = Accelerator(
        mixed_precision=MIXED_PRECISION,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    )

    config = Mamba3Config(
        d_model=D_MODEL, d_state=D_STATE, d_head=D_HEAD, expand=EXPAND,
        num_layers=NUM_LAYERS, use_parallel_scan=True, chunk_size=CHUNK_SIZE, use_kmoe=True,
        kmoe_num_experts=KMOE_NUM_EXPERTS, kmoe_top_k=KMOE_TOP_K,
        kmoe_r1=KMOE_R1, kmoe_r2=KMOE_R2, kmoe_r3=KMOE_R3,
        ffn_expand=FFN_EXPAND, mimo_rank=MIMO_RANK, num_kv_heads=NUM_KV_HEADS,
        layer_scale_init=1e-2,
    )
    model = Mamba3LanguageModel(config, VOCAB_SIZE)

    if accelerator.is_main_process:
        print_model_analysis(model, config, VOCAB_SIZE)

    optimizer = AdamW(
        model.parameters(), lr=LR, weight_decay=0.1,
        betas=(0.9, 0.95), fused=True,   # 🚀 fused=True：梯度更新融合成單一 CUDA Kernel，比 foreach 更快
    )
    scheduler = get_lr_scheduler(optimizer, WARMUP, STEPS)

    ckpt_cache = None
    start_step = 0
    if os.path.exists(CHECKPOINT_SAVE_PATH):
        accelerator.print(f"📂 發現 checkpoint，先僅載入模型權重：{CHECKPOINT_SAVE_PATH}")
        ckpt_cache = torch.load(CHECKPOINT_SAVE_PATH, map_location="cpu")
        model.load_state_dict(ckpt_cache["model"])
        start_step = ckpt_cache.get("step", 0)
        accelerator.print(f"✅ 模型權重載入完成，準備從 step {start_step} 續訓。")
        if ENABLE_RESUME_COMPILE_WARMUP and TRAIN_MODE:
            accelerator.print("   🔥 將先執行 compile warmup，再載入 optimizer/scheduler 狀態以降低峰值顯存。")
        else:
            accelerator.print("   ℹ️ 本次不使用 resume compile warmup，optimizer/scheduler 狀態將照常載入。")
    elif os.path.isfile(PRETRAINED_EMBED_PATH):
        accelerator.print(f"🌟 嘗試掛載預處理 Embedding：{PRETRAINED_EMBED_PATH}")
        pretrained_embed = torch.load(PRETRAINED_EMBED_PATH, map_location="cpu")
        expected_shape = model.embed.weight.shape
        if pretrained_embed.shape == expected_shape:
            model.embed.weight.data.copy_(pretrained_embed)
            model.head.weight.data.copy_(pretrained_embed)
            accelerator.print(f"✅ 成功將預訓練 Embedding 載入！維度: {expected_shape}")
        else:
            accelerator.print(
                f"⚠️ 預訓練 Embedding 維度 {pretrained_embed.shape} "
                f"與模型設定 {expected_shape} 不符，略過載入。"
            )
    else:
        accelerator.print("🌱 沒有 Checkpoint 也沒有預訓練 Embedding，從頭隨機初始化。")

    compile_active = False
    if TRAIN_MODE:
        if accelerator.is_main_process:
            print(f"🔥 [TRAIN] 啟動 torch.compile (mode='{COMPILE_MODE}', fullgraph={COMPILE_FULLGRAPH})...")
        try:
            import torch._dynamo as dynamo
            dynamo.config.suppress_errors = False
            model = torch.compile(model, mode=COMPILE_MODE, fullgraph=COMPILE_FULLGRAPH)
            compile_active = True
            if accelerator.is_main_process:
                fg_label = "單體圖加速" if COMPILE_FULLGRAPH else "分段圖編譯加速"
                print(f"✅ torch.compile 成功，進入{fg_label}模式 (mode='{COMPILE_MODE}')。")
        except Exception as e:
            if accelerator.is_main_process:
                print(f"⚠️ torch.compile 啟動失敗，退回 eager 模式: {e}")
    else:
        if accelerator.is_main_process:
            print("🐛 [DEBUG] 跳過 torch.compile，保留完整 Python 追蹤棧。")

    dataset   = PretokenizedDataset(DATA_PATH, SEQ_LEN)

    _dl_workers = 2 if TRAIN_MODE else 0
    _dl_pin_mem = True if TRAIN_MODE else False
    dataloader  = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=_dl_workers,
        pin_memory=_dl_pin_mem,
    )
    if accelerator.is_main_process:
        print(f"📦 DataLoader: num_workers={_dl_workers}, pin_memory={_dl_pin_mem}")

    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    # ── Checkpoint & Embedding Resume ────────────────────────────
    if ckpt_cache is not None:
        if TRAIN_MODE and compile_active and ENABLE_RESUME_COMPILE_WARMUP:
            accelerator.print("🔥 執行 Dummy Pass 預熱 torch.compile，暫時不載入 optimizer state...")
            _amp_dtype = torch.bfloat16 if MIXED_PRECISION == "bf16" else torch.float16
            dummy_step = start_step
            dummy_x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=accelerator.device)
            dummy_y = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=accelerator.device)

            with torch.autocast(device_type="cuda", dtype=_amp_dtype):
                dummy_out = model(dummy_x, labels=dummy_y, step=dummy_step)

            dummy_loss = dummy_out[0].mean()
            accelerator.backward(dummy_loss)
            optimizer.zero_grad(set_to_none=True)
            del dummy_x, dummy_y, dummy_out, dummy_loss
            gc.collect()
            torch.cuda.empty_cache()
            accelerator.print("✅ Dummy Pass 完成，已清理暫存梯度與 allocator cache。")

        accelerator.print("📦 開始載入 optimizer / scheduler 狀態...")
        optimizer.load_state_dict(ckpt_cache["optimizer"])
        scheduler.load_state_dict(ckpt_cache["scheduler"])
        old_lr = ckpt_cache["optimizer"]["param_groups"][0]["lr"]
        new_lr = scheduler.get_last_lr()[0]
        accelerator.print(f"✅ 從 step {start_step} 繼續訓練。")
        accelerator.print(f"   📉 [LR 轉換確認] 原本舊規則 LR: {old_lr:.2e} ➡️ 目前 LR: {new_lr:.2e}")
        del ckpt_cache
        gc.collect()
        torch.cuda.empty_cache()

    # ── 準備 Log ─────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_fp     = open(LOG_FILE, "a", newline="", encoding="utf-8")
    log_writer = csv.writer(log_fp)
    if not os.path.exists(LOG_FILE) or os.stat(LOG_FILE).st_size == 0:
        log_writer.writerow([
            "step", "loss", "ce_loss", "lb_contrib", "z_contrib",
            "router_temp",          # ← 新增：當前 Router 溫度，方便觀察退火曲線
            "lr", "grad_norm", "loss_scale",
            "tokens_seen", "elapsed_s", "step_time_s",
        ])

    if accelerator.is_main_process:
        print_model_analysis(unwrap_model(model), config, VOCAB_SIZE)

        # ── 訓練超參數摘要（rich 美化版）───────────────────────
        eff_batch = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * accelerator.num_processes
      

        # Fallback: 原始 print（與修改前相同）
        W = 68
        print("═" * W);  print("⚙️   Training Config Summary");  print("═" * W)
        print(f"  【D】訓練超參數")
        print(f"  {'SEQ_LEN':.<38} {SEQ_LEN}")
        print(f"  {'BATCH_SIZE (per GPU)':.<38} {BATCH_SIZE}")
        print(f"  {'GRADIENT_ACCUMULATION_STEPS':.<38} {GRADIENT_ACCUMULATION_STEPS}")
        print(f"  {'→ Effective Batch':.<38} {eff_batch}  ({BATCH_SIZE}×{GRADIENT_ACCUMULATION_STEPS}×{accelerator.num_processes} GPU)")
        print(f"  {'LR':.<38} {LR:.2e}")
        print(f"  {'WARMUP':.<38} {WARMUP} steps")
        print(f"  {'STEPS':.<38} {STEPS}")
        print(f"  {'CHECKPOINT_EVERY':.<38} {CHECKPOINT_EVERY} steps")
        print(f"  {'GRAD_CHECK_INTERVAL':.<38} {GRAD_CHECK_INTERVAL} steps")
        print(f"  {'Mixed Precision':.<38} {MIXED_PRECISION}")
        print("─" * W);  print("  【E】Router 退火")
        print(f"  {'ROUTER_T_START':.<38} {ROUTER_T_START}")
        print(f"  {'ROUTER_T_END':.<38} {ROUTER_T_END}")
        print("─" * W);  print("  【F】模式切換")
        print(f"  {'TRAIN_MODE':.<38} {'✅ 正式訓練' if TRAIN_MODE else '🐛 Debug'}")
        print("─" * W);  print("  【G】torch.compile")
        print(f"  {'ENABLE_RESUME_COMPILE_WARMUP':.<38} {ENABLE_RESUME_COMPILE_WARMUP}")
        print(f"  {'COMPILE_MODE':.<38} {COMPILE_MODE}")
        print(f"  {'COMPILE_FULLGRAPH':.<38} {COMPILE_FULLGRAPH}")
        print("─" * W);  print("  【路徑】")
        print(f"  {'DATA_PATH':.<38} {DATA_PATH}")
        print(f"  {'OUTPUT_DIR':.<38} {OUTPUT_DIR}")
        print(f"  {'LOG_FILE':.<38} {LOG_FILE}")
        print(f"  {'CHECKPOINT_SAVE_PATH':.<38} {CHECKPOINT_SAVE_PATH}")
        print("═" * W)

        accelerator.print(f"🚂 開始訓練，目標 {STEPS} steps...")

        if not TRAIN_MODE:
            accelerator.print("🔬 [DEBUG] 梯度診斷 hooks 永久掛載")
        else:
            accelerator.print("🔬 [TRAIN] 延遲掛載策略：hooks 僅在 grad_check_interval 時臨時啟用")

    # ── 訓練迴圈 ─────────────────────────────────────────────────
    model.train()
    global_step = start_step
    tokens_seen = global_step * BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * SEQ_LEN
    t_start     = time.time()
    data_iter   = iter(dataloader)

    while global_step < STEPS:
        step_start = time.time()
        acc_loss   = 0.0
        acc_ce     = 0.0
        acc_lb     = 0.0
        acc_z      = 0.0
        optimizer.zero_grad()

        for _ in range(GRADIENT_ACCUMULATION_STEPS):
            try:
                x_batch, y_batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                x_batch, y_batch = next(data_iter)

            with accelerator.accumulate(model):
                # 🚀 核心修正：告訴 CUDA Graphs 這是一個全新的執行步，防止梯度累積時記憶體被覆寫
                if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                    torch.compiler.cudagraph_mark_step_begin()

                _amp_dtype = torch.bfloat16 if MIXED_PRECISION == "bf16" else torch.float16
                with torch.autocast(device_type="cuda", dtype=_amp_dtype):
                    outputs = model(x_batch, labels=y_batch, step=global_step)
                loss    = outputs[0].mean()
                # 🚨 修正：只跳過 backward，不 zero_grad（否則會清掉其他正常微批次已累積的梯度！）
                if torch.isnan(loss) or torch.isinf(loss):
                    accelerator.print("⚠️ 偵測到 Loss NaN/Inf，跳過此微批次 (Micro-batch)！")
                    continue
                accelerator.backward(loss)
                acc_loss += loss.detach().float()
                # Model returns 5-tuple: (loss, lb_tensor, ce_d, lb_d, z_d)
                if len(outputs) >= 5:
                    acc_ce += outputs[2].item() if isinstance(outputs[2], torch.Tensor) else float(outputs[2])
                    acc_lb += outputs[3].item() if isinstance(outputs[3], torch.Tensor) else float(outputs[3])
                    acc_z  += outputs[4].item() if isinstance(outputs[4], torch.Tensor) else float(outputs[4])

        # ---------- 梯度裁剪與 NaN 防火牆 ----------
        grad_norm = 0.0
        if accelerator.sync_gradients:
            norm_val  = accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            grad_norm = norm_val.item() if isinstance(norm_val, torch.Tensor) else norm_val

        # 🚨 核心防護：攔截 NaN / Inf 梯度，絕不讓毒化的梯度更新權重
        if math.isnan(grad_norm) or math.isinf(grad_norm):
            if accelerator.is_main_process:
                print(f"🚨 [Step {global_step}] 攔截到異常梯度爆炸 (Grad Norm: {grad_norm})！"
                      f" 放棄本次權重更新，清空梯度繼續訓練。")
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            continue  # 跳過 optimizer.step() 和 scheduler.step()，直接進入下一個 batch

        # 只有梯度正常時才更新權重與學習率
        optimizer.step()
        scheduler.step()
        global_step += 1

        step_tokens  = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * SEQ_LEN
        tokens_seen += step_tokens

        # ---------- Logging ----------
        if accelerator.is_main_process:
            step_time = time.time() - step_start
            avg_loss  = acc_loss / GRADIENT_ACCUMULATION_STEPS
            cur_lr    = scheduler.get_last_lr()[0]
            elapsed   = time.time() - t_start

            # 當前 Router 溫度（純 Python float，不進入計算圖）
            cur_router_temp = get_router_temperature(global_step)

            if hasattr(accelerator, "scaler") and accelerator.scaler is not None:
                current_loss_scale = accelerator.scaler.get_scale()
            else:
                current_loss_scale = 1.0

            # Read loss breakdown from accumulated sums (no _last_loss_terms mutation needed)
            n_accum = GRADIENT_ACCUMULATION_STEPS
            ce_val = acc_ce / n_accum if acc_ce > 0 else float(avg_loss)
            lb_val = acc_lb / n_accum
            z_val  = acc_z  / n_accum

            try:
                current_ppl = math.exp(ce_val) if ce_val < 20 else float('inf')
            except OverflowError:
                current_ppl = float('inf')

            # 寫入 CSV（含 router_temp 欄位）
            log_writer.writerow([
                global_step,
                f"{avg_loss:.5f}", f"{ce_val:.5f}", f"{lb_val:.5f}", f"{z_val:.5f}",
                f"{cur_router_temp:.4f}",   # ← 新增欄位
                f"{cur_lr:.2e}",
                f"{grad_norm:.4f}",
                f"{current_loss_scale:.1f}",
                tokens_seen,
                f"{elapsed:.1f}", f"{step_time:.3f}",
            ])
            log_fp.flush()

            step_tok_per_s = step_tokens / (step_time + 1e-8)

            if TRAIN_MODE:
                print(
                    f"  step {global_step:>6}/{STEPS} | "
                    f"Loss: {avg_loss:.4f} (CE:{ce_val:.4f}, LB:{lb_val:.4f}, Z:{z_val:.4f}) | "
                    f"PPL: {current_ppl:>6.1f} | Grad: {grad_norm:>5.2f} | "
                    f"T_router: {cur_router_temp:.3f} | "   # ← 每步顯示當前溫度
                    f"LR: {cur_lr:.2e} | Time: {step_time:.2f}s | Tok/s: {step_tok_per_s:.0f}"
                )
            else:
                _ls_flag = ""
                if current_loss_scale < 256:
                    _ls_flag = " ⚠️ Scale↓"
                elif current_loss_scale > 32768:
                    _ls_flag = " ✅ Scale↑"
                print(
                    f"  🐛 step {global_step:>6}/{STEPS} | "
                    f"Loss: {avg_loss:.4f} (CE:{ce_val:.4f}, LB:{lb_val:.4f}, Z:{z_val:.4f}) | "
                    f"PPL: {current_ppl:>6.1f} | Grad: {grad_norm:>5.2f} | "
                    f"T_router: {cur_router_temp:.3f} | "   # ← DEBUG 模式也顯示
                    f"Scale: {current_loss_scale:.0f}{_ls_flag} | "
                    f"LR: {cur_lr:.2e} | Time: {step_time:.2f}s | Tok/s: {step_tok_per_s:.0f}"
                )

            # ── 梯度診斷區塊 ─────────────────────────────────────
            if global_step % GRAD_CHECK_INTERVAL == 0:
                print(f"  {'='*65}")
                print(f"  🔍 [Step {global_step}] 基本梯度狀態")
                print(f"      - 梯度總範數 (Grad Norm)  : {grad_norm:.4f}")
                print(f"      - 當前 Router 溫度        : {cur_router_temp:.4f}  "
                      f"({'Warmup 階段' if global_step < WARMUP else '退火中'})")

                if math.isnan(grad_norm) or math.isinf(grad_norm):
                    print("      ⚠️  警告: 梯度出現 NaN 或 Inf！")
                elif grad_norm > 5.0:
                    print("      ⚠️  警告: 梯度異常偏高，有爆炸風險 (Clip 保護中)。")
                elif grad_norm < 1e-4:
                    print("      ⚠️  警告: 梯度幾近消失。")
                else:
                    print("      ✅  狀態: 梯度流動正常。")



        # ---------- Checkpoint ----------
        if global_step % CHECKPOINT_EVERY == 0 and accelerator.is_main_process:
            ckpt_dict = {
                "step":         global_step,
                "model":        unwrap_model(model).state_dict(),
                "optimizer":    optimizer.state_dict(),
                "scheduler":    scheduler.state_dict(),
                "config":       config.__dict__,
                "train_mode":   TRAIN_MODE,
                "router_t_start": ROUTER_T_START,   # ← 儲存退火設定，方便 resume 核對
                "router_t_end":   ROUTER_T_END,
            }
            torch.save(ckpt_dict, CHECKPOINT_SAVE_PATH)
            del ckpt_dict
            gc.collect()
            torch.cuda.empty_cache()

            if TRAIN_MODE:
                print(f"  💾 Checkpoint 已儲存（step {global_step}, "
                      f"T_router={cur_router_temp:.4f}）→ {CHECKPOINT_SAVE_PATH}")
            else:
                print(f"  💾 [DEBUG] Checkpoint 已儲存（step {global_step}, "
                      f"T_router={cur_router_temp:.4f}, "
                      f"loss_scale={current_loss_scale:.1f}）→ {CHECKPOINT_SAVE_PATH}")

    log_fp.close()

    if accelerator.is_main_process:
        mode_label = "TRAIN" if TRAIN_MODE else "DEBUG"
        print(f"🎉 [{mode_label}] 訓練完成！共 {global_step} steps，"
              f"累計 {tokens_seen:,} tokens。")
        print(f"   最終 Router 溫度: {get_router_temperature(global_step):.4f} "
              f"（目標 T_end={ROUTER_T_END}）")

        accelerator.end_training()


# ┌──────────────────────────────────────────────────────────────────┐
# │  §11  Entry Point — 所有超參數集中在此                          │
# └──────────────────────────────────────────────────────────────────┘

if __name__ == "__main__":

    # ════════════════════════════════════════════════
    # 【A】模型超參數
    # ════════════════════════════════════════════════
    D_MODEL      = 768       # 模型隱藏層維度
    D_STATE      = 64        # SSM 狀態維度
    D_HEAD       = 64        # 每個 Head 的維度
    EXPAND       = 2         # d_inner = EXPAND * D_MODEL
    NUM_LAYERS   = 6         # Macro Block 數量 (每個含 4 Mamba + 1 Attn)
    MIMO_RANK    = 4         # SSM MIMO Rank R
    NUM_KV_HEADS = 4         # GQA KV-Head 數
    CHUNK_SIZE   = 64        # Parallel Scan Chunk Size

    # ════════════════════════════════════════════════
    # 【B】TuckerMoE 超參數
    # ════════════════════════════════════════════════
    KMOE_NUM_EXPERTS = 8     # 專家數量 E
    KMOE_TOP_K       = 2     # 每 token 激活 top-k 個專家
    KMOE_R1          = 32    # 與 checkpoint 相容的 Tucker 專家 Rank
    KMOE_R2          = 512   # 與 checkpoint 相容的 Tucker 輸出 Rank
    KMOE_R3          = 256   # Tucker 輸入壓縮 Rank
    FFN_EXPAND       = 6     # Transformer FFN 擴展比


    VOCAB_SIZE            = 32007


    # ════════════════════════════════════════════════
    # 【C】資料集與路徑
    # ════════════════════════════════════════════════
    DATA_PATH             = "/kaggle/input/datasets/s990093/fineweb-edu-tokenized-32007/fineweb_tokenized.bin"  # 與 checkpoint 相容的 32007 vocab 資料
    OUTPUT_DIR            = "/kaggle/working/"
    LOG_FILE              = "/kaggle/working/train_log.csv"
    CHECKPOINT_SAVE_PATH  = "/kaggle/working/checkpoint.pt"
    PRETRAINED_EMBED_PATH = ""                         # 留空則不載入

    # ════════════════════════════════════════════════
    # 【D】訓練超參數
    # ════════════════════════════════════════════════
    SEQ_LEN                  = 512
    BATCH_SIZE               = 2     # Per-GPU batch size
    GRADIENT_ACCUMULATION_STEPS = 8  # Effective batch = BATCH * ACCUM * n_gpu
    LR                       = 8e-5
    WARMUP                   = 400   # Warmup steps
    STEPS                    = 60000 # 總訓練 steps
    CHECKPOINT_EVERY         = 100   # 每 N steps 存一次

    # ════════════════════════════════════════════════
    # 【E】Router 退火設定
    # ════════════════════════════════════════════════
    ROUTER_T_START = 2.0     # 初始路由溫度（高 → 更均勻分配）
    ROUTER_T_END   = 0.5     # 最終路由溫度（低 → 更集中）

    # ════════════════════════════════════════════════
    # 【F】模式切換
    # ════════════════════════════════════════════════
    TRAIN_MODE = True        # True = 正式訓練 | False = Debug 模式

    # ════════════════════════════════════════════════
    # 【G】torch.compile 設定
    # ════════════════════════════════════════════════
    ENABLE_RESUME_COMPILE_WARMUP = True
    COMPILE_MODE      = "default"  # "default" | "reduce-overhead" | "max-autotune"
    COMPILE_FULLGRAPH = False      # True = 單體圖（需模型無 Python mutation）

    # ════════════════════════════════════════════════
    # 【H】診斷設定
    # ════════════════════════════════════════════════
    GRAD_CHECK_INTERVAL = 50       # 每 N steps 印出梯度診斷

    # ── 啟動訓練 ────────────────────────────────────
    train(
        D_MODEL=D_MODEL, D_STATE=D_STATE, D_HEAD=D_HEAD,
        EXPAND=EXPAND, NUM_LAYERS=NUM_LAYERS,
        MIMO_RANK=MIMO_RANK, NUM_KV_HEADS=NUM_KV_HEADS, CHUNK_SIZE=CHUNK_SIZE,
        KMOE_NUM_EXPERTS=KMOE_NUM_EXPERTS, KMOE_TOP_K=KMOE_TOP_K,
        KMOE_R1=KMOE_R1, KMOE_R2=KMOE_R2, KMOE_R3=KMOE_R3, FFN_EXPAND=FFN_EXPAND,
        DATA_PATH=DATA_PATH, OUTPUT_DIR=OUTPUT_DIR,
        LOG_FILE=LOG_FILE, CHECKPOINT_SAVE_PATH=CHECKPOINT_SAVE_PATH,
        PRETRAINED_EMBED_PATH=PRETRAINED_EMBED_PATH, VOCAB_SIZE=VOCAB_SIZE,
        SEQ_LEN=SEQ_LEN,
        BATCH_SIZE=BATCH_SIZE,
        GRADIENT_ACCUMULATION_STEPS=GRADIENT_ACCUMULATION_STEPS,
        LR=LR, WARMUP=WARMUP, STEPS=STEPS, CHECKPOINT_EVERY=CHECKPOINT_EVERY,
        ROUTER_T_START=ROUTER_T_START, ROUTER_T_END=ROUTER_T_END,
        TRAIN_MODE=TRAIN_MODE,
        ENABLE_RESUME_COMPILE_WARMUP=ENABLE_RESUME_COMPILE_WARMUP,
        COMPILE_MODE=COMPILE_MODE, COMPILE_FULLGRAPH=COMPILE_FULLGRAPH,
        GRAD_CHECK_INTERVAL=GRAD_CHECK_INTERVAL,
    )