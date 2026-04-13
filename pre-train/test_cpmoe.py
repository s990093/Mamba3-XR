%%writefile train.py


# -*- coding: utf-8 -*-
import os
import gc
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import triton
import triton.language as tl
from torch.utils.checkpoint import checkpoint
from liger_kernel.transformers import LigerRMSNorm, LigerCrossEntropyLoss

# ── 1. Fused PTX & Triton Kernels ──

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
    x_scaled = x * (1.0 / scale)
    t = tanh_approx(x_scaled)
    y = t * scale
    tl.store(y_ptr + offsets, y, mask=mask)

@triton.autotune(configs=get_cuda_autotune_config(), key=['n_elements'])
@triton.jit
def _fused_scaled_tanh_bwd(dy_ptr, x_ptr, dx_ptr, scale, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    dy = tl.load(dy_ptr + offsets, mask=mask).to(tl.float32)
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    x_scaled = x * (1.0 / scale)
    t = tanh_approx(x_scaled)
    dx = dy * (1.0 - t * t)
    tl.store(dx_ptr + offsets, dx, mask=mask)

class _FastScaledTanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float = 10.0) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.scale = scale
        y = torch.empty_like(x)
        n = x.numel()
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        _fused_scaled_tanh_fwd[grid](x, y, scale, n)
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        (x,) = ctx.saved_tensors
        dx = torch.empty_like(x)
        n = x.numel()
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        _fused_scaled_tanh_bwd[grid](dy, x, dx, ctx.scale, n)
        return dx, None

def fast_scaled_tanh(x: torch.Tensor, scale: float = 10.0) -> torch.Tensor:
    return _FastScaledTanh.apply(x, scale)

@triton.autotune(configs=get_cuda_autotune_config(), key=['n_elements'])
@triton.jit
def _fused_silu_mul_fwd(gate_ptr, feat_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    gate = tl.load(gate_ptr + offsets, mask=mask).to(tl.float32)
    feat = tl.load(feat_ptr + offsets, mask=mask).to(tl.float32)
    out = silu(gate) * feat
    tl.store(out_ptr + offsets, out, mask=mask)

@triton.autotune(configs=get_cuda_autotune_config(), key=['n_elements'])
@triton.jit
def _fused_silu_mul_bwd(dout_ptr, gate_ptr, feat_ptr, dgate_ptr, dfeat_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    dout = tl.load(dout_ptr + offsets, mask=mask).to(tl.float32)
    gate = tl.load(gate_ptr + offsets, mask=mask).to(tl.float32)
    feat = tl.load(feat_ptr + offsets, mask=mask).to(tl.float32)
    sig = tl.sigmoid(gate)
    s = gate * sig
    dfeat = dout * s
    tl.store(dfeat_ptr + offsets, dfeat, mask=mask)
    dsilu = sig * (1.0 + gate * (1.0 - sig))
    dgate = dout * feat * dsilu
    tl.store(dgate_ptr + offsets, dgate, mask=mask)

class _FastSiluGating(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(gate, feat)
        out = torch.empty_like(gate)
        n = gate.numel()
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        _fused_silu_mul_fwd[grid](gate, feat, out, n)
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        gate, feat = ctx.saved_tensors
        dgate = torch.empty_like(gate)
        dfeat = torch.empty_like(feat)
        n = gate.numel()
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        _fused_silu_mul_bwd[grid](dout, gate, feat, dgate, dfeat, n)
        return dgate, dfeat

def fast_silu_gating(gate: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
    return _FastSiluGating.apply(gate, feat)

def get_router_temperature(step: int | None, warmup: int = 500, total: int = 10000, t_start: float = 2.0, t_end: float = 0.5) -> float:
    if step is None: return t_end
    if step < warmup: return t_start
    progress = min((step - warmup) / max(1, total - warmup), 1.0)
    return t_end + 0.5 * (t_start - t_end) * (1.0 + math.cos(math.pi * progress))

class Mamba3Config:
    def __init__(
        self, d_model=768, d_state=64, d_head=64, n_groups=1, mimo_rank=4, expand=4,
        num_layers=15, use_parallel_scan=True, use_kmoe=True,
        kmoe_num_experts=8, kmoe_top_k=2, kmoe_r1=4, kmoe_r2=1024, kmoe_r3=256, ffn_expand=6, num_kv_heads=4,
        dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, layer_scale_init=1e-2, rms_norm_eps=1e-5, chunk_size=64
    ):
        self.d_model = d_model
        self.d_state = d_state
        self.d_head = d_head
        self.expand = expand
        self.num_layers = num_layers
        self.d_inner = int(expand * d_model)
        self.n_heads = self.d_inner // d_head
        self.n_groups = n_groups
        self.mimo_rank = mimo_rank
        self.rms_norm_eps = rms_norm_eps
        self.chunk_size = chunk_size
        self.use_parallel_scan = use_parallel_scan

        self.use_kmoe = use_kmoe
        self.kmoe_num_experts = kmoe_num_experts
        self.kmoe_top_k = kmoe_top_k
        self.kmoe_r1 = kmoe_r1
        self.kmoe_r2 = kmoe_r2
        self.kmoe_r3 = kmoe_r3
        self.ffn_expand = ffn_expand

        self.num_kv_heads = num_kv_heads
        self.kv_groups = self.n_heads // num_kv_heads
        self.dt_min, self.dt_max, self.dt_init_floor = dt_min, dt_max, dt_init_floor
        self.layer_scale_init = layer_scale_init

class RMSNorm(LigerRMSNorm):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__(hidden_size=dim, eps=eps)

class LayerScale(nn.Module):
    def __init__(self, dim: int, init_value: float = 1e-2):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim) * init_value)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma.to(x.dtype)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_R3': 32, 'BLOCK_R2': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_R3': 32, 'BLOCK_R2': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_R3': 64, 'BLOCK_R2': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_R3': 16, 'BLOCK_R2': 128}, num_warps=4, num_stages=4),
    ],
    key=['r3', 'r2'],
)
@triton.jit
def _fused_latent_moe_fwd(
    x_ptr, g_ptr, idx_ptr, prob_ptr, out_ptr,
    stride_xb, stride_xr3,
    stride_ge, stride_gr3, stride_gr2,
    stride_idxb, stride_idxk,
    stride_probb, stride_probk,
    stride_ob, stride_or2,
    B, r3, r2, top_k,
    BLOCK_R3: tl.constexpr, BLOCK_R2: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_r2 = tl.program_id(1)

    offs_r2 = pid_r2 * BLOCK_R2 + tl.arange(0, BLOCK_R2)

    acc = tl.zeros((BLOCK_R2,), dtype=tl.float32)

    for k in range(top_k):
        idx_p = idx_ptr + pid_b * stride_idxb + k * stride_idxk
        prob_p = prob_ptr + pid_b * stride_probb + k * stride_probk
        exp_idx = tl.load(idx_p)
        prob = tl.load(prob_p)

        # 🚀 分塊計算以節省 Shared Memory
        for r3_idx in range(0, r3, BLOCK_R3):
            offs_r3 = r3_idx + tl.arange(0, BLOCK_R3)

            x_ptrs = x_ptr + pid_b * stride_xb + offs_r3 * stride_xr3
            x = tl.load(x_ptrs, mask=offs_r3 < r3, other=0.0) # [BLOCK_R3]

            g_ptrs = g_ptr + exp_idx * stride_ge + offs_r3[:, None] * stride_gr3 + offs_r2[None, :] * stride_gr2
            g = tl.load(g_ptrs, mask=(offs_r3[:, None] < r3) & (offs_r2[None, :] < r2), other=0.0) # [BLOCK_R3, BLOCK_R2]

            acc += prob * tl.sum(x[:, None] * g, axis=0)

    out_ptrs = out_ptr + pid_b * stride_ob + offs_r2 * stride_or2
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=offs_r2 < r2)



def get_dG_bwd_autotune_config():
    return [
        triton.Config({'BLOCK_R3': 32, 'BLOCK_R2': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_R3': 32, 'BLOCK_R2': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_R3': 64, 'BLOCK_R2': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_R3': 64, 'BLOCK_R2': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_R3': 128, 'BLOCK_R2': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_R3': 64, 'BLOCK_R2': 256}, num_warps=8, num_stages=4),
    ]

@triton.autotune(
    configs=get_dG_bwd_autotune_config(),
    key=['r3', 'r2', 'B']
)
@triton.jit
def _fused_latent_moe_bwd_dG_kernel(
    x_ptr, dout_ptr, prob_ptr, idx_ptr, dG_ptr,
    stride_xb, stride_xr3,
    stride_doutb, stride_doutr2,
    stride_probb, stride_probk,
    stride_idxb, stride_idxk,
    stride_dGe, stride_dGr3, stride_dGr2,
    B, top_k, r3, r2,
    BLOCK_R3: tl.constexpr, BLOCK_R2: tl.constexpr
):
    pid_e = tl.program_id(0)
    pid_r3 = tl.program_id(1)
    pid_r2 = tl.program_id(2)

    offs_r3 = pid_r3 * BLOCK_R3 + tl.arange(0, BLOCK_R3)
    offs_r2 = pid_r2 * BLOCK_R2 + tl.arange(0, BLOCK_R2)
    
    mask_r3 = offs_r3 < r3
    mask_r2 = offs_r2 < r2

    acc = tl.zeros((BLOCK_R3, BLOCK_R2), dtype=tl.float32)

    for b in range(B):
        for k in range(top_k):
            e = tl.load(idx_ptr + b * stride_idxb + k * stride_idxk)
            
            if e == pid_e:
                prob = tl.load(prob_ptr + b * stride_probb + k * stride_probk).to(tl.float32)
                x = tl.load(x_ptr + b * stride_xb + offs_r3 * stride_xr3, mask=mask_r3, other=0.0).to(tl.float32)
                dout = tl.load(dout_ptr + b * stride_doutb + offs_r2 * stride_doutr2, mask=mask_r2, other=0.0).to(tl.float32)
                
                # 改成這樣：完美對齊 PyTorch 的操作順序
                dout_scaled = dout * prob
                acc += x[:, None] * dout_scaled[None, :]

    dG_ptrs = dG_ptr + pid_e * stride_dGe + offs_r3[:, None] * stride_dGr3 + offs_r2[None, :] * stride_dGr2
    tl.store(dG_ptrs, acc.to(dG_ptr.dtype.element_ty), mask=mask_r3[:, None] & mask_r2[None, :])

class FusedLatentMoE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_shared, G_experts, top_k_indices, top_k_probs):
        B, r3 = x_shared.shape
        E, _, r2 = G_experts.shape
        top_k = top_k_indices.size(1)

        ctx.save_for_backward(x_shared, G_experts, top_k_indices, top_k_probs)

        out = torch.empty((B, r2), device=x_shared.device, dtype=x_shared.dtype)
        grid = lambda meta: (B, triton.cdiv(r2, meta['BLOCK_R2']))

        _fused_latent_moe_fwd[grid](
            x_shared, G_experts, top_k_indices, top_k_probs, out,
            x_shared.stride(0), x_shared.stride(1),
            G_experts.stride(0), G_experts.stride(1), G_experts.stride(2),
            top_k_indices.stride(0), top_k_indices.stride(1),
            top_k_probs.stride(0), top_k_probs.stride(1),
            out.stride(0), out.stride(1),
            B, r3, r2, top_k
        )
        return out

    @staticmethod
    def backward(ctx, dout):
        x_shared, G_experts, top_k_indices, top_k_probs = ctx.saved_tensors
        B, r3 = x_shared.shape
        E, _, r2 = G_experts.shape
        top_k = top_k_indices.size(1)

        # === 1. 安全的 dx_shared 與 dprobs 計算 (避免 8.5GB Memory Bomb) ===
        dx_shared = torch.zeros_like(x_shared)
        dprobs = torch.zeros_like(top_k_probs)
        
        for k in range(top_k):
            idx = top_k_indices[:, k]               # [B]
            prob = top_k_probs[:, k].unsqueeze(1)    # [B, 1]
            G_k = G_experts[idx]                    # [B, r3, r2]
            dout_k = dout * prob                    # [B, r2]
            
            # 使用安全的迴圈內 bmm，最高只佔用 [B, r3, r2] 的記憶體
            dx_shared += torch.matmul(dout_k.unsqueeze(1), G_k.transpose(1, 2)).squeeze(1)
            out_k = torch.matmul(x_shared.unsqueeze(1), G_k).squeeze(1)
            dprobs[:, k] = (dout * out_k).sum(dim=-1)
            
        # === 2. 高效 Triton Kernel 運算 dG_experts (完全零額外記憶體開銷) ===
        dG_experts = torch.zeros_like(G_experts)
        grid = lambda meta: (E, triton.cdiv(r3, meta['BLOCK_R3']), triton.cdiv(r2, meta['BLOCK_R2']))
        
        _fused_latent_moe_bwd_dG_kernel[grid](
            x_shared, dout, top_k_probs, top_k_indices, dG_experts,
            x_shared.stride(0), x_shared.stride(1),
            dout.stride(0), dout.stride(1),
            top_k_probs.stride(0), top_k_probs.stride(1),
            top_k_indices.stride(0), top_k_indices.stride(1),
            dG_experts.stride(0), dG_experts.stride(1), dG_experts.stride(2),
            B, top_k, r3, r2
        )

        return dx_shared, dG_experts, None, dprobs


class NativeTuckerMoE(nn.Module):
    """
    基於論文 TD-MoE (Tucker Decomposition) 的 Parameter-Efficient MoE。
    引入 Core Tensor (核心張量)，解決 CP 分解過度壓縮、表達能力不足的問題。
    """
    def __init__(self, dim_in, dim_out, num_experts=8, top_k=2,
                 r1=4,    # 專家維度的 Rank (壓縮專家冗餘)
                 r2=1024, # 輸出/中間層的 Rank (決定虛擬容量)
                 r3=256   # 輸入維度的 Rank
                 ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k  = min(top_k, num_experts)

        # Router
        self.router   = nn.Linear(dim_in, num_experts, bias=False)
        nn.init.normal_(self.router.weight, mean=0.0, std=0.02)

        # 🌟 Tucker 分解的四大核心元件 🌟
        # 1. 專家模式矩陣 U_1 (對應論文的 U_E)
        self.U_expert = nn.Parameter(torch.empty(num_experts, r1))
        # 2. 輸入模式矩陣 U_3 (對應論文的 U_in)
        self.U_in = nn.Parameter(torch.empty(dim_in, r3))
        # 3. 輸出模式矩陣 U_2 (對應論文的 U_out)
        self.U_out = nn.Parameter(torch.empty(r2, dim_out))
        # 4. 核心張量 Core Tensor G (這就是大腦！)
        self.core = nn.Parameter(torch.empty(r1, r3, r2))

        self.bias = nn.Parameter(torch.zeros(dim_out))

        # 權重初始化
        nn.init.normal_(self.U_expert, std=1.0)
        nn.init.normal_(self.U_in,  std=1.0 / math.sqrt(dim_in))
        nn.init.normal_(self.U_out, std=1.0 / math.sqrt(r2))
        nn.init.normal_(self.core,  std=1.0 / math.sqrt(r1 * r3))

    def forward(self, x, step=None):
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])
        B_flat = x_flat.size(0)

        # === Router 邏輯 ===
        temperature   = get_router_temperature(step)
        raw_logits    = self.router(x_flat)
        capped_logits = fast_scaled_tanh(raw_logits, 10.0)
        z_loss = (torch.mean(torch.logsumexp(capped_logits, dim=-1) ** 2) if self.training else 0.0)

        router_logits = capped_logits / temperature
        router_probs  = torch.softmax(router_logits, dim=-1)
        _, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)

        top_k_raw   = router_probs.gather(-1, top_k_indices)
        top_k_probs = top_k_raw / (top_k_raw.sum(-1, keepdim=True) + 1e-6)

        if self.training:
            expert_mask = torch.zeros_like(router_logits).scatter_(1, top_k_indices, 1.0)
            lb_loss = self.num_experts * torch.sum(expert_mask.mean(0) * router_probs.float().mean(0))
        else:
            lb_loss = 0.0

        # === 優化版 Tucker 核心運算 ===
        # 1. 輸入模式 (U_in) 降維 -> [B, r3]
        x_shared = torch.matmul(x_flat, self.U_in)

        # 2. 預計算專家的 Core Tensor (提取到迴圈外)
        G_experts = torch.einsum('er, rst -> est', self.U_expert, self.core)

        # 3. 迴圈進行低維度路由
        x_core_accum = torch.zeros((B_flat, G_experts.size(2)), device=x.device, dtype=x.dtype)

        for k in range(self.top_k):
            idx_k = top_k_indices[:, k]
            prob_k = top_k_probs[:, k].unsqueeze(1)
            G_k = G_experts[idx_k]
            x_core = torch.bmm(x_shared.unsqueeze(1), G_k).squeeze(1)
            x_core_accum += x_core * prob_k

        # 4. 輸出模式 (U_out) 升維 -> [Batch, dim_out]
        out = torch.matmul(x_core_accum, self.U_out)

        out = out.reshape(*orig_shape[:-1], -1)
        return out + self.bias, lb_loss, z_loss


class TritonTuckerMoE(nn.Module):
    """
    Triton 版本的優化 TuckerMoE
    """
    def __init__(self, dim_in, dim_out, num_experts=8, top_k=2,
                 r1=4, r2=1024, r3=256):
        super().__init__()
        self.num_experts = num_experts
        self.top_k  = min(top_k, num_experts)

        self.router   = nn.Linear(dim_in, num_experts, bias=False)
        nn.init.normal_(self.router.weight, mean=0.0, std=0.02)

        self.U_expert = nn.Parameter(torch.empty(num_experts, r1))
        self.U_in = nn.Parameter(torch.empty(dim_in, r3))
        self.U_out = nn.Parameter(torch.empty(r2, dim_out))
        self.core = nn.Parameter(torch.empty(r1, r3, r2))
        self.bias = nn.Parameter(torch.zeros(dim_out))

        nn.init.normal_(self.U_expert, std=1.0)
        nn.init.normal_(self.U_in,  std=1.0 / math.sqrt(dim_in))
        nn.init.normal_(self.U_out, std=1.0 / math.sqrt(r2))
        nn.init.normal_(self.core,  std=1.0 / math.sqrt(r1 * r3))

    def forward(self, x, step=None):
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])
        B_flat = x_flat.size(0)

        temperature   = get_router_temperature(step)
        raw_logits    = self.router(x_flat)
        capped_logits = fast_scaled_tanh(raw_logits, 10.0)
        z_loss = (torch.mean(torch.logsumexp(capped_logits, dim=-1) ** 2) if self.training else 0.0)

        router_logits = capped_logits / temperature
        router_probs  = torch.softmax(router_logits, dim=-1)
        _, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)

        top_k_raw   = router_probs.gather(-1, top_k_indices)
        top_k_probs = top_k_raw / (top_k_raw.sum(-1, keepdim=True) + 1e-6)

        if self.training:
            expert_mask = torch.zeros_like(router_logits).scatter_(1, top_k_indices, 1.0)
            lb_loss = self.num_experts * torch.sum(expert_mask.mean(0) * router_probs.float().mean(0))
        else:
            lb_loss = 0.0

        x_shared = torch.matmul(x_flat, self.U_in)
        G_experts = torch.einsum('er, rst -> est', self.U_expert, self.core)

        x_core_accum = FusedLatentMoE.apply(
            x_shared,
            G_experts,
            top_k_indices,
            top_k_probs
        ).to(x.dtype)

        out = torch.matmul(x_core_accum, self.U_out)
        out = out.reshape(*orig_shape[:-1], -1)
        return out + self.bias, lb_loss, z_loss

TuckerMoE = TritonTuckerMoE # We set TuckerMoE to default to Triton

class MixtralMoEFeedForward(nn.Module):
    def __init__(self, config: Mamba3Config):
        super().__init__()
        d_model = config.d_model
        d_ff = int(math.ceil(config.ffn_expand * d_model / 256) * 256)

        kw = dict(
            num_experts=config.kmoe_num_experts,
            top_k=config.kmoe_top_k,
            r1=config.kmoe_r1,
            r2=config.kmoe_r2,
            r3=config.kmoe_r3,
        )

        self.gate_proj = TuckerMoE(d_model, d_ff, **kw)
        self.up_proj   = TuckerMoE(d_model, d_ff, **kw)
        self.down_proj = TuckerMoE(d_ff, d_model, **kw)

    def forward(self, x: torch.Tensor, step: int | None = None):
        gate, lb_g, z_g = self.gate_proj(x, step=step)
        feat, lb_u, z_u = self.up_proj(x, step=step)
        h = fast_silu_gating(gate, feat)
        y, lb_d, z_d = self.down_proj(h, step=step)
        return y, lb_g + lb_u + lb_d, z_g + z_u + z_d

@triton.jit
def first_order_combine_op(alpha_left, beta_left, alpha_right, beta_right):
    alpha_out = alpha_right * alpha_left
    beta_out = alpha_right * beta_left + beta_right
    return alpha_out, beta_out

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_D': 32}, num_warps=4),
        triton.Config({'BLOCK_D': 64}, num_warps=4),
        triton.Config({'BLOCK_D': 128}, num_warps=8),
        triton.Config({'BLOCK_D': 256}, num_warps=8),
        triton.Config({'BLOCK_D': 512}, num_warps=8), 
    ],
    key=['D', 'L'],
)
@triton.jit
def _chunk_scan_kernel(
    alpha_ptr, u_ptr, h_out_ptr,
    stride_a_b, stride_a_l,
    stride_u_b, stride_u_l, stride_u_d,
    B_flat, L: tl.constexpr, D: tl.constexpr,
    BLOCK_D: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)

    offset_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offset_l = tl.arange(0, L)
    mask_d = offset_d < D

    a_ptrs = alpha_ptr + pid_b * stride_a_b + offset_l * stride_a_l
    alpha = tl.load(a_ptrs)

    u_ptrs = u_ptr + pid_b * stride_u_b + offset_l[:, None] * stride_u_l + offset_d[None, :] * stride_u_d
    u = tl.load(u_ptrs, mask=mask_d[None, :], other=0.0)

    # 🚀 關鍵修復：把資料在 SRAM / 暫存器中升級到 FP32 進行高精度運算
    alpha_fp32 = alpha.to(tl.float32)
    u_fp32 = u.to(tl.float32)

    alpha_exp = tl.broadcast_to(alpha_fp32[:, None], (L, BLOCK_D))
    
    # 這裡的 combined_fn 就會吃 fp32 並且回傳 fp32
    _, h_out_fp32 = tl.associative_scan((alpha_exp, u_fp32), axis=0, combine_fn=first_order_combine_op)

    # 🚀 算完之後，無縫降級回原本的 dtype (FP16 或 BF16) 準備存回 Global Memory
    h_out = h_out_fp32.to(u.dtype)

    h_out_ptrs = h_out_ptr + pid_b * stride_u_b + offset_l[:, None] * stride_u_l + offset_d[None, :] * stride_u_d
    tl.store(h_out_ptrs, h_out, mask=mask_d[None, :])

def fast_triton_chunk_scan(log_alpha_chunk, u_chunk):
    B, num_chunks, L, H = log_alpha_chunk.shape
    _, _, _, _, N, P = u_chunk.shape
    D = N * P
    
    alpha_chunk = torch.exp(log_alpha_chunk)
    alpha_flat = alpha_chunk.transpose(2, 3).reshape(-1, L).contiguous()
    u_flat = u_chunk.transpose(2, 3).reshape(-1, L, D).contiguous()
    
    h_out_flat = torch.empty_like(u_flat)
    B_flat = alpha_flat.shape[0]

    # 🚀 交給 meta 動態取得 autotune 測試出來的 BLOCK_D
    grid = lambda meta: (B_flat, triton.cdiv(D, meta['BLOCK_D']))

    _chunk_scan_kernel[grid](
        alpha_flat, u_flat, h_out_flat,
        alpha_flat.stride(0), alpha_flat.stride(1),
        u_flat.stride(0), u_flat.stride(1), u_flat.stride(2),
        B_flat=B_flat, L=L, D=D
        # 注意：不要在這裡傳遞 BLOCK_D，Autotuner 會自動把它當作 kwargs 塞進去
    )
    
    return h_out_flat.reshape(B, num_chunks, H, L, N, P).transpose(2, 3)

class Mamba3Block(nn.Module):
    def __init__(self, config: Mamba3Config):
        super().__init__()
        self.config = config
        d_in, H, G, P, N, R = config.d_model, config.n_heads, config.n_groups, config.d_head, config.d_state, config.mimo_rank
        self.ratio, self.dim_z, self.dim_x = H // G, H * P, H * P
        self.dim_B, self.dim_C, self.dim_dt, self.dim_A, self.dim_lambda = G * N * R, G * N * R, G, G, G
        d_proj_total = self.dim_z + self.dim_x + self.dim_B + self.dim_C + self.dim_dt + self.dim_A + self.dim_lambda
        self.in_proj = nn.Linear(d_in, d_proj_total, bias=True)

        if config.use_kmoe:
            kw = dict(num_experts=config.kmoe_num_experts, top_k=config.kmoe_top_k, r1=config.kmoe_r1, r2=config.kmoe_r2, r3=config.kmoe_r3)
            self.x_up_proj = TuckerMoE(H * P, H * P * R, **kw)
            self.out_proj  = TuckerMoE(d_in, d_in, **kw)
        else:
            self.x_up_proj = nn.Linear(P, P * R, bias=False)
            self.out_proj  = nn.Linear(d_in, d_in, bias=False)

        self.y_down_proj = nn.Linear(P * R, P, bias=False)
        self.theta_log = nn.Parameter(torch.randn(G, N // 2))
        self.D         = nn.Parameter(torch.ones(H))
        self.norm_B = RMSNorm(N * R, eps=config.rms_norm_eps)
        self.norm_C = RMSNorm(N * R, eps=config.rms_norm_eps)
        self.bias_B = nn.Parameter(torch.zeros(G, N, R))
        self.bias_C = nn.Parameter(torch.zeros(G, N, R))

        self.mamba_dense_proj = nn.Linear(config.d_inner, d_in, bias=False)
        self.pre_gate_norm    = RMSNorm(H * P)
        self.act              = nn.SiLU()
        self.norm_mamba       = RMSNorm(config.d_model)
        self.norm_out_proj    = RMSNorm(config.d_model)

        self.ls_mamba    = LayerScale(config.d_model, init_value=config.layer_scale_init)
        self.ls_out_proj = LayerScale(config.d_model, init_value=config.layer_scale_init)

        with torch.no_grad():
            self.bias_B.fill_(1.0)
            self.bias_C.fill_(1.0)
            dt = torch.clamp(torch.exp(torch.rand(G) * (math.log(config.dt_max) - math.log(config.dt_min)) + math.log(config.dt_min)), min=config.dt_init_floor)
            inv_dt   = dt + torch.log(-torch.expm1(-dt))
            dt_start = self.dim_z + self.dim_x + self.dim_B + self.dim_C
            dt_end   = dt_start + self.dim_dt
            A_end    = dt_end   + self.dim_A
            self.in_proj.bias[dt_start:dt_end].copy_(inv_dt)
            self.in_proj.bias[dt_end:A_end].uniform_(1, 16).log_()
            self.in_proj.bias[A_end:].fill_(-3.0)

    def apply_rope(self, x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        # Avoid torch.polar and view_as_complex to prevent torch.compile ATen fallbacks
        N_half = angles.shape[-1]
        x_reshaped = x.float().view(*x.shape[:-2], N_half, 2, x.shape[-1])
        x1 = x_reshaped[..., 0, :]  # Real
        x2 = x_reshaped[..., 1, :]  # Imag

        sin_angles = torch.sin(angles).unsqueeze(-1)
        cos_angles = torch.cos(angles).unsqueeze(-1)

        out1 = x1 * cos_angles - x2 * sin_angles
        out2 = x2 * cos_angles + x1 * sin_angles

        return torch.stack([out1, out2], dim=-2).reshape_as(x).type_as(x)

    def segsum(self, x: torch.Tensor) -> torch.Tensor:
        x_cumsum = torch.cumsum(x, dim=-1)
        mask = torch.tril(torch.ones(x.size(-1), x.size(-1), device=x.device, dtype=torch.bool), diagonal=0)
        return (x_cumsum[..., :, None] - x_cumsum[..., None, :]).masked_fill(~mask, -float("inf"))

    def chunk_parallel_scan(self, u, dt, A, C, chunk_size=128):
        B, L, H, N, P = u.shape
        R, device, input_dtype = C.shape[-1], u.device, u.dtype
        L_orig = L
        if L % chunk_size != 0:
            pad_len = chunk_size - (L % chunk_size)
            u  = F.pad(u,  (0,0,0,0,0,0,0,pad_len))
            dt = F.pad(dt, (0,0,0,pad_len))
            C  = F.pad(C,  (0,0,0,0,0,0,0,pad_len))
            A  = F.pad(A,  (0,0,0,pad_len))
            L  = L + pad_len
        num_chunks = L // chunk_size
        log_alpha  = dt * A
        u_chunk         = u.view(B, num_chunks, chunk_size, H, N, P)
        log_alpha_chunk = log_alpha.view(B, num_chunks, chunk_size, H)
        C_chunk         = C.view(B, num_chunks, chunk_size, H, N, R)
        
        # 🚀 使用 Triton 加速的平行掃描！
        h_intra = fast_triton_chunk_scan(log_alpha_chunk, u_chunk)
        
        y_diag   = torch.einsum("bclhnp, bclhnr -> bclhpr", h_intra, C_chunk)
        decay_chunk    = torch.exp(torch.sum(log_alpha_chunk, dim=2))
        h_prev = torch.zeros(B, H, N, P, device=device, dtype=input_dtype)
        h_inter_tensor = torch.empty(B, num_chunks, H, N, P, device=device, dtype=input_dtype)
        for c in range(num_chunks):
            h_inter_tensor[:, c] = h_prev
            h_prev = h_prev * decay_chunk[:, c].view(B, H, 1, 1) + h_intra[:, c, -1]
        c_decayed   = C_chunk * torch.exp(torch.cumsum(log_alpha_chunk, dim=2)).unsqueeze(-1).unsqueeze(-1)
        y_off       = torch.einsum("bchnp, bclhnr -> bclhpr", h_inter_tensor, c_decayed)
        y_total = (y_diag + y_off).view(B, -1, H, P, R)
        return y_total[:, :L_orig].to(input_dtype) if L_orig < L else y_total.to(input_dtype), h_prev.to(input_dtype)

    def forward(self, x: torch.Tensor, step: int | None = None):
        B_sz, L, _ = x.shape
        H, G, P, N, R, ratio = self.config.n_heads, self.config.n_groups, self.config.d_head, self.config.d_state, self.config.mimo_rank, self.ratio
        residual_mamba, u = x, self.norm_mamba(x)
        z, x_prime, B_param, C_param, dt, A_param, lambda_param = torch.split(
            self.in_proj(u), [self.dim_z, self.dim_x, self.dim_B, self.dim_C, self.dim_dt, self.dim_A, self.dim_lambda], dim=-1
        )
        x_prime = x_prime.view(B_sz, L, H, P)
        dt = F.softplus(dt)
        A  = -torch.exp(A_param)
        theta = torch.exp(self.theta_log)

        broadcast_group = lambda t: t.repeat_interleave(ratio, dim=2)
        dt_b    = broadcast_group(dt.unsqueeze(-1)).squeeze(-1)
        A_b     = broadcast_group(A.unsqueeze(-1)).squeeze(-1)
        angles  = torch.cumsum(torch.einsum("blh, hn -> blhn", dt_b, theta.repeat_interleave(ratio, dim=0)), dim=1)

        B_rotated = self.apply_rope(broadcast_group((self.norm_B(B_param.reshape(B_sz, L, G, N * R)).view(B_sz, L, G, N, R) + self.bias_B)), angles)
        C_rotated = self.apply_rope(broadcast_group((self.norm_C(C_param.reshape(B_sz, L, G, N * R)).view(B_sz, L, G, N, R) + self.bias_C)), angles)

        if self.config.use_kmoe:
            x_up, lb_up, z_up = self.x_up_proj(x_prime.view(B_sz, L, -1), step=step)
            x_ssm = x_up.view(B_sz, L, H, P, R)
        else:
            x_ssm, lb_up, z_up = self.x_up_proj(x_prime).view(B_sz, L, H, P, R), 0.0, 0.0

        input_signal = torch.einsum("blhnr, blhpr -> blhnp", B_rotated, x_ssm)
        lambda_view  = F.sigmoid(broadcast_group(lambda_param.unsqueeze(-1)).squeeze(-1)).view(B_sz, L, H, 1, 1)
        dt_view    = dt_b.view(B_sz, L, H, 1, 1)
        alpha_view = torch.exp(dt_b * A_b).view(B_sz, L, H, 1, 1)
        input_prev = torch.roll(input_signal, shifts=1, dims=1)
        input_prev[:, 0] = 0
        u_ssm = lambda_view * dt_view * input_signal + (1 - lambda_view) * dt_view * alpha_view * input_prev

        if self.config.use_parallel_scan:
            y_stack, _ = self.chunk_parallel_scan(u_ssm, dt_b, A_b, C_rotated, chunk_size=self.config.chunk_size)
        else:
            h_state, y_list = torch.zeros(B_sz, H, N, P, device=x.device), []
            for t in range(L):
                h_state = h_state * alpha_view[:, t] + u_ssm[:, t]
                y_list.append(torch.einsum("bhnp, bhnr -> bhpr", h_state, C_rotated[:, t]))
            y_stack = torch.stack(y_list, dim=1)

        y = self.y_down_proj(y_stack.view(B_sz, L, H, P * R)).view(B_sz, L, H * P)
        y = y + x_prime.reshape(B_sz, L, H * P) * self.D.repeat_interleave(P, dim=0)
        mamba_out = self.mamba_dense_proj(self.pre_gate_norm(y) * self.act(z))

        mid_x = residual_mamba + self.ls_mamba(mamba_out)
        residual_proj, normed_mid = mid_x, self.norm_out_proj(mid_x)

        if self.config.use_kmoe:
            proj_out, lb_out, z_out = self.out_proj(normed_mid, step=step)
        else:
            proj_out, lb_out, z_out = self.out_proj(normed_mid), 0.0, 0.0

        return residual_proj + self.ls_out_proj(proj_out), lb_up + lb_out, z_up + z_out

class TransformerBlock(nn.Module):
    def __init__(self, config: Mamba3Config):
        super().__init__()
        self.head_dim     = 64
        self.num_heads    = config.d_model // 64
        self.num_kv_heads = config.num_kv_heads
        self.kv_groups    = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(config.d_model, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=True)
        self.norm_attn = RMSNorm(config.d_model)
        self.use_kmoe  = config.use_kmoe

        if config.use_kmoe:
            self.ffn = MixtralMoEFeedForward(config)
        else:
            d_ff = int(math.ceil(8 * config.d_model / 3 / 256) * 256)
            self.ffn_gate = nn.Linear(config.d_model, d_ff, bias=False)
            self.ffn_up   = nn.Linear(config.d_model, d_ff, bias=False)
            self.ffn_down = nn.Linear(d_ff, config.d_model, bias=False)

        self.norm_ffn = RMSNorm(config.d_model)
        self.ls_attn = LayerScale(config.d_model, init_value=config.layer_scale_init)
        self.ls_ffn  = LayerScale(config.d_model, init_value=config.layer_scale_init)

    def forward(self, x: torch.Tensor, step: int | None = None):
        B, L, D = x.shape
        residual, normed_x = x, self.norm_attn(x)

        # Q: [B, num_heads, L, head_dim]
        q = self.q_proj(normed_x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # K/V 原始形狀轉換為: [B, num_kv_heads, L, head_dim]
        k = self.k_proj(normed_x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(normed_x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # 🚀 優化點：使用 expand 取代 repeat_interleave
        # 1. 增加一個維度: [B, num_kv_heads, 1, L, head_dim]
        # 2. 擴展該維度: [B, num_kv_heads, kv_groups, L, head_dim] (Zero-copy)
        # 3. Reshape 壓平對齊 Q: [B, num_heads, L, head_dim]
        k = k.unsqueeze(2).expand(B, self.num_kv_heads, self.kv_groups, L, self.head_dim).reshape(B, self.num_heads, L, self.head_dim)
        v = v.unsqueeze(2).expand(B, self.num_kv_heads, self.kv_groups, L, self.head_dim).reshape(B, self.num_heads, L, self.head_dim)

        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)

        x = residual + self.ls_attn(self.o_proj(attn_out))
        residual, h_norm = x, self.norm_ffn(x)
        if self.use_kmoe:
            ffn_out, ffn_lb, ffn_z = self.ffn(h_norm, step=step)
        else:
            ffn_out = self.ffn_down(fast_silu_gating(self.ffn_gate(h_norm), self.ffn_up(h_norm)))
            ffn_lb, ffn_z = 0.0, 0.0
        return residual + self.ls_ffn(ffn_out), ffn_lb, ffn_z

class TrueHybridMamba(nn.Module):
    def __init__(self, config: Mamba3Config, mamba_ratio: int = 4):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(config.num_layers):
            for _ in range(mamba_ratio):
                self.layers.append(nn.ModuleDict({"block": Mamba3Block(config)}))
            self.layers.append(nn.ModuleDict({"block": TransformerBlock(config)}))

    def forward(self, x: torch.Tensor, step: int | None = None):
        total_lb_loss, total_z_loss = 0.0, 0.0
        for layer_dict in self.layers:
            x, lb, z = checkpoint(layer_dict["block"], x, step, use_reentrant=True)
            if isinstance(lb, torch.Tensor):
                total_lb_loss = total_lb_loss + lb
                total_z_loss  = total_z_loss  + z
        return x, total_lb_loss, total_z_loss

class Mamba3LanguageModel(nn.Module):
    def __init__(self, config: Mamba3Config, vocab_size: int, **kwargs):
        super().__init__()
        self.config = config
        self.embed    = nn.Embedding(vocab_size, config.d_model)
        self.backbone = TrueHybridMamba(config)
        self.norm     = RMSNorm(config.d_model)
        self.head     = nn.Linear(config.d_model, vocab_size, bias=False)
        self.head.weight = self.embed.weight
        self.ce_loss_fn = nn.CrossEntropyLoss()
        self._last_loss_terms = None
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None, step=None):
        embeds = self.embed(input_ids)
        backbone_out  = self.backbone(embeds, step=step)
        hidden        = self.norm(backbone_out[0])
        total_lb_loss = backbone_out[1]
        total_z_loss  = backbone_out[2]

        hidden_scaled = hidden / math.sqrt(self.config.d_model)
        raw_logits    = self.head(hidden_scaled).float()
        logits = fast_scaled_tanh(raw_logits, 30.0)

        if labels is not None:
            ce_loss = self.ce_loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            if isinstance(total_lb_loss, torch.Tensor):
                total_lb_loss = total_lb_loss.mean()
                total_z_loss  = total_z_loss.mean()

            num_moe_layers = self.config.num_layers * (4 * 2 + 1 * 3)
            lb_contrib = (0.1 / max(1, num_moe_layers)) * total_lb_loss
            z_contrib  = (1e-3  / max(1, num_moe_layers)) * total_z_loss
            loss = ce_loss + lb_contrib + z_contrib

            # if torch.isnan(loss) or torch.isinf(loss):
            #     return loss.detach().unsqueeze(0), loss.detach().unsqueeze(0)

            # Commented out to prevent CUDA Graph breaks during fullgraph=True compilation
            # self._last_loss_terms = {
            #     "ce_loss":    ce_loss.detach(),
            #     "lb_contrib": lb_contrib.detach(),
            #     "z_contrib":  z_contrib.detach(),
            # }
            return (loss.unsqueeze(0), total_lb_loss.detach().unsqueeze(0) if isinstance(total_lb_loss, torch.Tensor) else loss.unsqueeze(0))
        return logits


# ── 3. Model Parameters and Profile Printout ──


# ── 2. Precision Checking ──

def run_precision_check():
    print("=' '*64")
    print("🔍  [Precision Check] Triton vs Optimized Native PyTorch TuckerMoE")
    print("=' '*64")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("⚠️ No CUDA device found. Precision check requires CUDA for Triton.")
        return

    B, seq_len = 1, 128
    D, d_ff = 768, 4608
    num_experts, top_k, r1, r2, r3 = 8, 2, 4, 1024, 256

    torch.manual_seed(42)
    native_moe = NativeTuckerMoE(D, d_ff, num_experts, top_k, r1, r2, r3).to(device)
    triton_moe = TritonTuckerMoE(D, d_ff, num_experts, top_k, r1, r2, r3).to(device)

    # Sync weights
    triton_moe.load_state_dict(native_moe.state_dict())

    x = torch.randn(B, seq_len, D, device=device, requires_grad=True)
    x_triton = x.clone().detach().requires_grad_(True)

    # Forward
    out_native, lb_native, z_native = native_moe(x)
    out_triton, lb_triton, z_triton = triton_moe(x_triton)

    fw_diff = (out_native - out_triton).abs().max().item()
    print(f"   ▶ Forward Max Diff:  {fw_diff:.2e}")

    if fw_diff < 1e-4:
        print("   ✅ Forward Pass: Precision Matches!")
    else:
        print("   ❌ Forward Pass: Precision Mismatch!")

    # Backward
    dout = torch.randn_like(out_native)
    out_native.backward(dout)
    native_grads = [p.grad.clone() for p in native_moe.parameters()]
    native_x_grad = x.grad.clone()

    out_triton.backward(dout)
    triton_grads = [p.grad.clone() for p in triton_moe.parameters()]
    triton_x_grad = x_triton.grad.clone()

    x_bw_diff = (native_x_grad - triton_x_grad).abs().max().item()
    print(f"   ▶ Backward (x) Max Diff: {x_bw_diff:.2e}")

    all_matched = True
    for (name, p), n_grad, t_grad in zip(native_moe.named_parameters(), native_grads, triton_grads):
        diff = (n_grad - t_grad).abs().max().item()
        if diff > 5e-4:
            all_matched = False
        print(f"      - {name:<12} Grad Diff: {diff:.2e}")

    if all_matched and x_bw_diff < 5e-4:
        print("   ✅ Backward Pass: Precision Matches!")
    else:
        print("   ❌ Backward Pass: Precision Mismatch!")
    print()

    print("=" * 64)
    print("🔍  [Precision Check] Segsum + Einsum vs Triton Chunk Scan (Multiple Configs)")
    print("=" * 64)

    # Native Segsum calculation
    def segsum_fn(x):
        x_cumsum = torch.cumsum(x, dim=-1)
        mask = torch.tril(torch.ones(x.size(-1), x.size(-1), device=x.device, dtype=torch.bool), diagonal=0)
        return (x_cumsum[..., :, None] - x_cumsum[..., None, :]).masked_fill(~mask, -float("inf"))

    configs = [
        {"dtype": torch.float32, "chunk_size": 32, "N": 64, "P": 64},
        {"dtype": torch.float32, "chunk_size": 128, "N": 16, "P": 64},
        {"dtype": torch.float16, "chunk_size": 64, "N": 64, "P": 64},
        {"dtype": torch.bfloat16, "chunk_size": 128, "N": 64, "P": 64},
    ]

    B_val, num_chunks_val, H_val = 2, 2, 4
    
    for cfg in configs:
        dtp = cfg["dtype"]
        cz = cfg["chunk_size"]
        n_val = cfg["N"]
        p_val = cfg["P"]
        
        # ⚠️ Mamba's log_alpha is strictly negative (dt * A, where A < 0). 
        # Must use negative values to prevent exp() from exploding to 10^30+ during accumulation and destroying FP32 precision.
        log_alpha_chunk = -torch.abs(torch.randn(B_val, num_chunks_val, cz, H_val, device=device)) - 0.1
        u_chunk = torch.randn(B_val, num_chunks_val, cz, H_val, n_val, p_val, device=device)
        
        # 轉換 dtype
        log_alpha_chunk = log_alpha_chunk.to(dtp)
        u_chunk = u_chunk.to(dtp)
        
        # Native
        # Native Segsum 必須在 fp32 下執行以防止 mask 的 -inf 溢位，然後轉回
        L_mask_native = torch.exp(segsum_fn(log_alpha_chunk.transpose(-1, -2).float())).to(dtp)
        h_intra_native = torch.einsum("bchls, bcshnp -> bclhnp", L_mask_native, u_chunk)
        
        # Triton
        h_intra_triton = fast_triton_chunk_scan(log_alpha_chunk, u_chunk)
        
        diff = (h_intra_native - h_intra_triton).abs().max().item()
        
        # 根據不同 dtype 放寬容忍度
        tol = 1e-4 if dtp == torch.float32 else (1e-2 if dtp == torch.float16 else 5e-2)
        
        status = "✅" if diff < tol else "❌"
        print(f"   [{str(dtp).split('.')[-1].upper():>8} | L={cz:<3} | D={n_val*p_val:<4}] Max Diff: {diff:.2e}  {status}")

    print()

    print("=" * 64)
    print("🔍  [Precision Check] PyTorch Loop vs Triton Expert-Centric dG")
    print("=" * 64)

    configs_dg = [
        {"dtype": torch.float32, "B": 1024, "E": 8, "top_k": 2},
        {"dtype": torch.float16, "B": 2048, "E": 4, "top_k": 2},
        {"dtype": torch.bfloat16, "B": 4096, "E": 8, "top_k": 2},
    ]

    r3_val, r2_val = 64, 128

    for cfg in configs_dg:
        dtp = cfg["dtype"]
        B_val = cfg["B"]
        E_val = cfg["E"]
        top_k_val = cfg["top_k"]

        x_shared_fp32 = torch.randn(B_val, r3_val, device=device)
        dout_fp32 = torch.randn(B_val, r2_val, device=device)
        G_experts_fp32 = torch.randn(E_val, r3_val, r2_val, device=device)
        top_k_probs_fp32 = torch.rand(B_val, top_k_val, device=device)
        top_k_probs_fp32 = top_k_probs_fp32 / top_k_probs_fp32.sum(dim=-1, keepdim=True)
        top_k_indices_val = torch.randint(0, E_val, (B_val, top_k_val), device=device)

        x_shared_t = x_shared_fp32.to(dtp)
        dout_t = dout_fp32.to(dtp)
        G_experts_t = G_experts_fp32.to(dtp)
        top_k_probs_t = top_k_probs_fp32.to(dtp)

        # Convert the truncated `dtp` tensors BACK to `fp32` to simulate exact Triton behavior
        # Triton loads `dtp` from VRAM, casts to `fp32`, multiplies, and accumulates.
        x_shared_math = x_shared_t.to(torch.float32)
        dout_math = dout_t.to(torch.float32)
        top_k_probs_math = top_k_probs_t.to(torch.float32)
        G_experts_math = G_experts_t.to(torch.float32)

        # Native Loop (FP32 baseline precision tracker for fair mathematical validation)
        dG_experts_native_fp32 = torch.zeros_like(G_experts_math)
        for k in range(top_k_val):
            idx = top_k_indices_val[:, k]               
            prob = top_k_probs_math[:, k].unsqueeze(1)    
            G_k = G_experts_math[idx]                    
            dout_k = dout_math * prob                    
            dG_k = torch.bmm(x_shared_math.unsqueeze(2), dout_k.unsqueeze(1)) 
            dG_experts_native_fp32.index_add_(0, idx, dG_k)

        # Target expected result mathematically unhindered by repeated sequential BFLOAT16 truncation
        dG_experts_native = dG_experts_native_fp32.to(dtp)

        # Triton Kernel
        dG_experts_triton = torch.zeros_like(G_experts_t)
        grid_dg = lambda meta: (E_val, triton.cdiv(r3_val, meta['BLOCK_R3']), triton.cdiv(r2_val, meta['BLOCK_R2']))
        
        _fused_latent_moe_bwd_dG_kernel[grid_dg](
            x_shared_t, dout_t, top_k_probs_t, top_k_indices_val, dG_experts_triton,
            x_shared_t.stride(0), x_shared_t.stride(1),
            dout_t.stride(0), dout_t.stride(1),
            top_k_probs_t.stride(0), top_k_probs_t.stride(1),
            top_k_indices_val.stride(0), top_k_indices_val.stride(1),
            dG_experts_triton.stride(0), dG_experts_triton.stride(1), dG_experts_triton.stride(2),
            B_val, top_k_val, r3_val, r2_val
        )

        diff = (dG_experts_native - dG_experts_triton).abs().max().item()
        # BF16 on large accumulations generates natural numerical drift (since Triton natively loops in FP32 inside the kernel, but PyTorch cuts accuracy natively over multiple ops) - adjusting tolerance accordingly
        tol = 1e-4 if dtp == torch.float32 else (1e-2 if dtp == torch.float16 else 5e-1)

        status = "✅" if diff < tol else "❌"
        print(f"   [{str(dtp).split('.')[-1].upper():>8} | B={B_val:<4} | E={E_val:<2}] Max Diff: {diff:.2e}  {status}")

    print()

def memory_footprint_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0

def print_model_analysis(model, config, vocab_size):
    total_params = 0
    trainable_params = 0
    active_params = 0

    bucket = {
        "embed_head": 0,
        "mamba_ssm": 0,
        "cpmoe_router": 0,
        "cpmoe_U_in": 0,
        "cpmoe_U_expert": 0,
        "cpmoe_U_out": 0,
        "cpmoe_bias": 0,
        "layer_scale": 0,
        "norm": 0,
        "attn_proj": 0,
        "other": 0,
    }

    for name, p in model.named_parameters():
        num_p = p.numel()
        total_params += num_p
        if not p.requires_grad:
            continue
        trainable_params += num_p

        # 👑 修正這裡：只要是 Tucker 專家的參數，就只算 Top-K 的比例
        if any(k in name for k in ["U_expert", "U_in", "U_out", "core", "expert_A", "expert_B"]):
            active_params += int(num_p * (config.kmoe_top_k / config.kmoe_num_experts))
            bucket["cpmoe_U_expert"] += num_p
        else:
            active_params += num_p
            if "embed" in name or "head.weight" in name:
                bucket["embed_head"] += num_p
            elif "router" in name:
                bucket["cpmoe_router"] += num_p
            elif ".bias" in name and any(k in name for k in ("gate_proj", "up_proj", "down_proj", "out_proj", "x_up_proj")):
                bucket["cpmoe_bias"] += num_p
            elif "ls_" in name or "gamma" in name:
                bucket["layer_scale"] += num_p
            elif any(k in name for k in ("norm_", "norm.", "rms_norm")):
                bucket["norm"] += num_p
            elif any(k in name for k in ("q_proj", "k_proj", "v_proj", "o_proj")):
                bucket["attn_proj"] += num_p
            elif any(k in name for k in ("in_proj", "y_down_proj", "mamba_dense", "theta_log", ".D", "bias_B", "bias_C")):
                bucket["mamba_ssm"] += num_p
            else:
                bucket["other"] += num_p

    cpmoe_modules = [mod for name, mod in model.named_modules() if "TuckerMoE" in type(mod).__name__]
    num_cpmoe_modules = len(cpmoe_modules)
    total_cpmoe_params = bucket["cpmoe_U_expert"]

    # Active TuckerMoE params per token: U_expert * (top_k / num_experts)
    active_cpmoe_params = int(bucket["cpmoe_U_expert"] * config.kmoe_top_k / config.kmoe_num_experts)

    W = 64
    print("═" * W)
    print("🚀  Mamba3 Hybrid TuckerMoE  ·  Testing & Profiling Configuration")
    print("═" * W)

    print("📦  【總參數覽表】")
    print(f"   {'總參數量  (Total) ':.<40} {total_params/1e6:>8.2f} M")
    print(f"   {'可訓練    (Trainable) ':.<40} {trainable_params/1e6:>8.2f} M")
    print(f"   {'⚡ 實際激活 (Active / step) ':.<40} {active_params/1e6:>8.2f} M  ({active_params/trainable_params*100:.1f}%)")

    print(f"\n{'─'*W}")
    print("🔬  【參數分類明細】")
    cat_labels = {
        "embed_head":     "Embedding + LM-Head (tied)",
        "mamba_ssm":      "Mamba SSM 核心 (in_proj/D/θ…)",
        "cpmoe_router":   "TuckerMoE Router (所有層合計)",
        "cpmoe_U_expert": "TuckerMoE 核心張量與模式矩陣",
        "cpmoe_bias":     "TuckerMoE Bias / out_norm",
        "layer_scale":    "LayerScale γ",
        "norm":           "RMSNorm (所有層)",
        "attn_proj":      "Attention Q/K/V/O proj",
        "other":          "其他",
    }
    for key, label in cat_labels.items():
        val = bucket[key]
        if val == 0: continue
        pct = val / trainable_params * 100
        bar_len = int(pct / 2)
        bar = "█" * bar_len + "░" * (25 - bar_len)
        print(f"   {label:<38} {val/1e6:>7.2f} M  {pct:>5.1f}%  {bar}")

    print(f"\n{'─'*W}")
    print("🧩  【TuckerMoE 獨立專家分析】")
    print(f"   {'TuckerMoE 模組總數':.<38} {num_cpmoe_modules:>4} 個")
    print(f"   {'獨立專家降維/升維總參數 (全儲存)':.<38} {total_cpmoe_params/1e6:>7.2f} M")
    print(f"   {'每步實際激活專家參數':.<38} {active_cpmoe_params/1e6:>7.2f} M")
    print(f"   {'Router 參數 (所有模組)':.<38} {bucket['cpmoe_router']/1e6:>7.2f} M")
    print(f"   {'Tucker Rank (r1, r2, r3)':.<38} {config.kmoe_r1}, {config.kmoe_r2}, {config.kmoe_r3}")
    print(f"   {'FFN 擴張倍率 (虛擬 d_ff)':.<38} {config.ffn_expand:>4}")
    print("\n")


# ── 4. Profiling Full Forward + Backward ──

def profile_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("⚠️ Profiling requires CUDA GPU.")
        return

    B, seq_len = 1, 256
    vocab_size = 50280

    config = Mamba3Config(
        d_model=768, d_state=64, expand=2, num_layers=6,
        use_parallel_scan=True, chunk_size=64, use_kmoe=True,
        kmoe_num_experts=8, kmoe_top_k=2, kmoe_r1=4, kmoe_r2=1024, kmoe_r3=256, ffn_expand=6,
        mimo_rank=4, num_kv_heads=4, layer_scale_init=1e-2,
    )

    model = Mamba3LanguageModel(config, vocab_size).to(device)

    print_model_analysis(model, config, vocab_size)

    # x = torch.randint(0, vocab_size, (B, seq_len), device=device)
    # y = torch.randint(0, vocab_size, (B, seq_len), device=device)

    # print(f"   [Memory] Initial allocated: {memory_footprint_mb():.1f} MB")

    # # Native Warmup
    # torch.cuda.synchronize()
    # loss, _ = model(x, labels=y)
    # if isinstance(loss, tuple):
    #     loss = loss[0]
    # loss.backward()

    # print(f"   [Memory] After Uncompiled FWD/BWD: {memory_footprint_mb():.1f} MB")

    # # Setup torch.compile
    # print("=' '*64")
    # print("🔥  Compiling Model with mode='reduce-overhead'...")
    # try:
    #     import torch._dynamo as dynamo
    #     dynamo.config.suppress_errors = True
    #     compiled_model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
    # except Exception as e:
    #     print(f"⚠️ Torch compile failed: {e}")
    #     compiled_model = model

    # # Warmup Compile
    # torch.cuda.synchronize()
    # start = time.time()
    # loss, _ = compiled_model(x, labels=y)
    # if isinstance(loss, tuple):
    #     loss = loss[0]
    # loss.backward()
    # torch.cuda.synchronize()
    # print(f"   [Compile Warmup] Took: {time.time() - start:.2f} s")
    # print(f"   [Memory] After compiled Warmup: {memory_footprint_mb():.1f} MB")

    # # Benchmark Speed
    # iters = 5
    # torch.cuda.synchronize()
    # start = time.time()
    # for _ in range(iters):
    #     compiled_model.zero_grad(set_to_none=True)
    #     loss, _ = compiled_model(x, labels=y)
    #     if isinstance(loss, tuple):
    #         loss = loss[0]
    #     loss.backward()
    # torch.cuda.synchronize()
    # avg_t = (time.time() - start) / iters

    # print(f"   [Benchmark] FWD+BWD Avg Time/step: {avg_t * 1000:.1f} ms")
    # print(f"   [Memory] Peak memory usage check complete.")
    # print("=' '*64\n")


# ── 3. Mathematical Visualization ──

def plot_feature_correlation(x_input):
    """
    x_input: shape [Batch * Seq, D_in]
    """
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
    except ImportError:
        print("Please install seaborn to view the feature correlation heatmap.")
        return

    x_centered = x_input - x_input.mean(dim=0)
    cov_matrix = torch.matmul(x_centered.T, x_centered) / (x_input.shape[0] - 1)

    std_dev = torch.sqrt(torch.diag(cov_matrix))
    corr_matrix = cov_matrix / torch.outer(std_dev, std_dev)

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix[:100, :100].cpu().detach().numpy(), cmap='coolwarm', center=0)
    plt.title("Input Feature Correlation Heatmap")
    plt.show()

    eigenvalues = torch.linalg.eigvalsh(cov_matrix)
    print(f"最大特徵值 (Max Eigenvalue): {eigenvalues.max().item():.2f}")
    print(f"最小特徵值 (Min Eigenvalue): {eigenvalues.min().item():.2f}")
    print(f"條件數 Condition Number (Max/Min): {eigenvalues.max() / (eigenvalues.min() + 1e-6):.2f}")

def plot_core_tensor_energy(tucker_moe_layer):
    """
    Core Shape: [r1 (expert), r3 (input), r2 (output)]
    """
    import matplotlib.pyplot as plt
    core = tucker_moe_layer.core.detach().cpu()

    energy_r3 = torch.linalg.norm(core, dim=(0, 2))
    energy_r2 = torch.linalg.norm(core, dim=(0, 1))

    energy_r3_sorted, _ = torch.sort(energy_r3, descending=True)
    energy_r2_sorted, _ = torch.sort(energy_r2, descending=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(energy_r3_sorted.numpy(), marker='o', markersize=3)
    axes[0].set_title("Energy Distribution across Input Mode (r3)")
    axes[0].set_xlabel("Latent Dimension Index")
    axes[0].set_ylabel("L2 Norm (Energy)")

    axes[1].plot(energy_r2_sorted.numpy(), marker='o', markersize=3, color='orange')
    axes[1].set_title("Energy Distribution across Output Mode (r2)")
    axes[1].set_xlabel("Latent Dimension Index")

    plt.tight_layout()
    plt.show()

def plot_compression_landscape(d_in=768, d_out=4608, num_experts=8, r1=4):
    import numpy as np
    import matplotlib.pyplot as plt

    r2_vals = np.linspace(128, 2048, 50)
    r3_vals = np.linspace(64, 512, 50)
    R2, R3 = np.meshgrid(r2_vals, r3_vals)

    # 壓縮參數公式: Core Tensor + Mode Matrices
    P_tucker = r1 * R2 * R3 + (num_experts * r1 + d_out * R2 + d_in * R3)
    P_baseline = num_experts * (d_in * d_out)

    compression_ratio = P_tucker / P_baseline

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(R2, R3, compression_ratio, levels=20, cmap='coolwarm')
    plt.colorbar(contour, label='Compression Ratio')

    cs = plt.contour(R2, R3, compression_ratio, levels=[0.1, 0.2, 0.3, 0.4, 0.5], colors='white', linestyles='dashed')
    plt.clabel(cs, inline=True, fontsize=10, fmt='%1.2f')

    plt.title(f"Parameter Compression Landscape Contours (r1={r1})")
    plt.xlabel("r2 (Output Latent Dim)")
    plt.ylabel("r3 (Input Latent Dim)")
    plt.show()

def run_mathematical_tests():
    print("=' '*64")
    print("📐  [Mathematical Verification] Tucker Decomposition Properties")
    print("=' '*64")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, seq_len, d_in = 4, 128, 768
    x_input = torch.randn(B * seq_len, d_in, device=device)

    print("1. Feature Correlation & Condition Number:")
    x_centered = x_input - x_input.mean(dim=0)
    cov_matrix = torch.matmul(x_centered.T, x_centered) / (x_input.shape[0] - 1)
    eigenvalues = torch.linalg.eigvalsh(cov_matrix)
    print(f"   - Max Eigenvalue: {eigenvalues.max().item():.2f}")
    print(f"   - Min Eigenvalue: {eigenvalues.min().item():.2f}")
    print(f"   - Condition Number: {eigenvalues.max() / (eigenvalues.min() + 1e-6):.2f}")
    print("   -> Plotting feature correlation heatmap...")
    plot_feature_correlation(x_input)
    print()

    print("2. Core Tensor Energy Distribution (r3=256, r2=1024):")
    moe = TritonTuckerMoE(dim_in=d_in, dim_out=4608, num_experts=8, top_k=2, r1=4, r2=1024, r3=256).to(device)
    core = moe.core.detach().cpu()
    energy_r3 = torch.linalg.norm(core, dim=(0, 2))
    energy_r2 = torch.linalg.norm(core, dim=(0, 1))
    print(f"   - Top 5 Energy on r3 (Input Mode):  {torch.sort(energy_r3, descending=True)[0][:5].numpy()}")
    print(f"   - Top 5 Energy on r2 (Output Mode): {torch.sort(energy_r2, descending=True)[0][:5].numpy()}")
    print("   -> Plotting core tensor energy distribution...")
    plot_core_tensor_energy(moe)
    print()

    print("3. Compression Landscape Contours:")
    print("   -> Plotting compression landscape contours...")
    plot_compression_landscape()
    print()

def benchmark_compiled_training(dtype=torch.bfloat16):
    print("=" * 64)
    print(f"🚀  [Training Benchmark] {str(dtype).split('.')[-1].upper()} Compiled Model Speed & Convergence")
    print("=" * 64)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("⚠️ Profiling requires CUDA GPU.")
        return

    B, seq_len = 1, 256
    vocab_size = 50280

    config = Mamba3Config(
        d_model=768, d_state=64, expand=2, num_layers=6,
        use_parallel_scan=True, chunk_size=64, use_kmoe=True,
        kmoe_num_experts=8, kmoe_top_k=2, kmoe_r1=4, kmoe_r2=1024, kmoe_r3=256, ffn_expand=6,
        mimo_rank=4, num_kv_heads=4, layer_scale_init=1e-2,
    )

    model = Mamba3LanguageModel(config, vocab_size).to(device)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=True)

    print("🔥  Compiling Model with mode='reduce-overhead'...")
    try:
        import torch._dynamo as dynamo
        # dynamo.config.suppress_errors = True
        dynamo.config.suppress_errors = False # 不要壓抑錯誤，印出來！
        compiled_model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
        # compiled_model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
    except Exception as e:
        print(f"⚠️ Torch compile failed: {e}")
        compiled_model = model

    torch.manual_seed(42)
    x = torch.randint(0, vocab_size, (B, seq_len), device=device)
    y = torch.randint(0, vocab_size, (B, seq_len), device=device)

    scaler = torch.cuda.amp.GradScaler() if dtype == torch.float16 else None

    print(f"⏳  Warming up (Compilation might take a few minutes, Precision: {dtype})...")
    torch.cuda.synchronize()
    import time
    start = time.time()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        
        with torch.autocast(device_type="cuda", dtype=dtype):
            loss, _ = compiled_model(x, labels=y)
            if isinstance(loss, tuple): loss = loss[0]

        if torch.isnan(loss) or torch.isinf(loss):
            print("⚠️ 偵測到 NaN/Inf，跳過 Warmup 這一小步！")
            optimizer.zero_grad(set_to_none=True)
            continue

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
    torch.cuda.synchronize()
    print(f"✅  Warmup complete! Took {time.time() - start:.2f} s\n")

    print("📈  Starting Benchmark & Loss Check (10 steps):")
    iters = 10
    total_time = 0.0
    for i in range(iters):
        torch.cuda.synchronize()
        start = time.time()

        optimizer.zero_grad(set_to_none=True)
        
        with torch.autocast(device_type="cuda", dtype=dtype):
            loss, _ = compiled_model(x, labels=y)
            if isinstance(loss, tuple): loss = loss[0]

        if torch.isnan(loss) or torch.isinf(loss):
            print("⚠️ 偵測到 NaN/Inf，跳過這步更新！")
            optimizer.zero_grad(set_to_none=True)
            continue

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        torch.cuda.synchronize()
        step_time = time.time() - start
        total_time += step_time

        print(f"   Step {i+1}: Loss = {loss.item():.4f} | Time = {step_time*1000:.2f} ms")

    print("-" * 40)
    print(f"🚀  Average Time per Batch: {(total_time / iters)*1000:.2f} ms")

    print("\n🔍  Running PyTorch Profiler for 1 step to analyze bottlenecks...")
    from torch.profiler import profile, record_function, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=dtype):
            loss, _ = compiled_model(x, labels=y)
            if isinstance(loss, tuple): loss = loss[0]
        
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()

    print("=" * 80)
    print("📊  [Profiler Report] Top 15 Time-Consuming Operations (CUDA Time)")
    print("=" * 80)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
    print("=' '*80\n")

if __name__ == "__main__":
    import os
    # os.environ["TORCH_LOGS"] = "+dynamo"
    run_precision_check()
    # Choose precision: torch.float32, torch.float16, or torch.bfloat16
    # Recommendation: Use BFLOAT16 for fullgraph=True to avoid GradScaler DeviceCopy sync points.
    benchmark_compiled_training(torch.bfloat16)
    # run_mathematical_tests()
    # profile_model()
