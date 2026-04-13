%%writefile train.py

import os

import warnings

warnings.filterwarnings(
    "ignore",
    message=".*Online softmax is disabled on the fly.*",
    category=UserWarning,
)

import torch
import triton
import triton.language as tl



# ============================================================
# 🚀 最佳化版本：Flash Kronecker MoE (Triton Forward + Broadcast Backward)
# ============================================================

from torch.amp import custom_fwd, custom_bwd

@triton.jit
def flash_kron_moe_fwd_kernel(
    # 指標 (Pointers)
    X_ptr, A_ptr, B_ptr,
    expert_ids_ptr, expert_probs_ptr,
    Y_ptr,

    # 步幅 (Strides)
    stride_xb, stride_xi, stride_xj,
    stride_ae, stride_ao, stride_ai,
    stride_be, stride_bo, stride_bi,
    stride_yb, stride_yo, stride_yop,

    # 矩陣維度
    IN1, IN2, OUT1, OUT2, TOPK,

    # 區塊大小 (必須是 2 的次方，如 16, 32, 64，觸發 Tensor Core)
    BLOCK_OUT1: tl.constexpr, 
    BLOCK_IN2: tl.constexpr, 
    BLOCK_K1: tl.constexpr,    # 👈 第一個 GEMM 的 Reduction 維度分塊
    BLOCK_OUT2: tl.constexpr,
):
    # 每個 Program 負責處理 1 個 Token
    token_id = tl.program_id(0)

    # 建立輸出維度與中間維度的座標偏移
    offs_out1 = tl.arange(0, BLOCK_OUT1)
    offs_in2 = tl.arange(0, BLOCK_IN2)
    offs_out2 = tl.arange(0, BLOCK_OUT2)

    # 最終輸出的 Accumulator (使用 FP32 確保精度)
    acc_y = tl.zeros((BLOCK_OUT1, BLOCK_OUT2), dtype=tl.float32)

    for k_exp in range(TOPK):
        expert_idx = tl.load(expert_ids_ptr + token_id * TOPK + k_exp)
        prob = tl.load(expert_probs_ptr + token_id * TOPK + k_exp)

        # ==========================================
        # 🛡️ 步驟 A: Tiled GEMM - 計算 M = A_k @ X
        # M shape: (BLOCK_OUT1, BLOCK_IN2)
        # 這裡沿著 IN1 (K1) 維度切塊，避免 SRAM 爆滿並觸發 Tensor Core
        # ==========================================
        m = tl.zeros((BLOCK_OUT1, BLOCK_IN2), dtype=tl.float32)

        for k1 in range(0, tl.cdiv(IN1, BLOCK_K1)):
            offs_k1 = k1 * BLOCK_K1 + tl.arange(0, BLOCK_K1)

            # 載入 A_k 的分塊 (OUT1, K1)
            a_ptrs = A_ptr + expert_idx * stride_ae + offs_out1[:, None] * stride_ao + offs_k1[None, :] * stride_ai
            mask_a = (offs_out1[:, None] < OUT1) & (offs_k1[None, :] < IN1)
            a = tl.load(a_ptrs, mask=mask_a, other=0.0)

            # 載入 X 的分塊 (K1, IN2)
            x_ptrs = X_ptr + token_id * stride_xb + offs_k1[:, None] * stride_xi + offs_in2[None, :] * stride_xj
            mask_x = (offs_k1[:, None] < IN1) & (offs_in2[None, :] < IN2)
            x = tl.load(x_ptrs, mask=mask_x, other=0.0)

            # 累加至 M，允許 TF32 (如果硬體支援)
            m += tl.dot(a, x, allow_tf32=True)

        # 必須將 M 轉回 FP16/BF16 才能餵給下一個 Tensor Core 矩陣乘法
        m_fp16 = m.to(X_ptr.dtype.element_ty)

        # ==========================================
        # 🛡️ 步驟 B: 計算 Y_k = M @ B_k^T
        # 因為 M 已經在暫存器且 IN2 通常很小，我們可以直接 One-shot 乘完
        # ==========================================
        b_T_ptrs = B_ptr + expert_idx * stride_be + offs_in2[:, None] * stride_bi + offs_out2[None, :] * stride_bo
        mask_b = (offs_in2[:, None] < IN2) & (offs_out2[None, :] < OUT2)
        b_T = tl.load(b_T_ptrs, mask=mask_b, other=0.0)

        # Y_k = M @ B_k^T
        y_k = tl.dot(m_fp16, b_T, allow_tf32=True)

        # 乘上 Router 機率並累加到最終輸出
        acc_y += y_k * prob

    # ==========================================
    # 寫回 HBM (Global Memory)
    # ==========================================
    y_ptrs = Y_ptr + token_id * stride_yb + offs_out1[:, None] * stride_yo + offs_out2[None, :] * stride_yop
    mask_y = (offs_out1[:, None] < OUT1) & (offs_out2[None, :] < OUT2)
    tl.store(y_ptrs, acc_y.to(Y_ptr.dtype.element_ty), mask=mask_y)


def flash_kron_moe_forward(x_sub, A_experts, B_experts, top_k_indices, top_k_probs):
    B_sz, IN1, IN2 = x_sub.shape
    _, OUT1, _ = A_experts.shape
    _, OUT2, _ = B_experts.shape
    TOPK = top_k_indices.shape[1]

    Y = torch.empty((B_sz, OUT1, OUT2), device=x_sub.device, dtype=x_sub.dtype)

    # 確保記憶體連續
    x_sub = x_sub.contiguous()
    A_experts = A_experts.contiguous()
    B_experts = B_experts.contiguous()

    # Triton Block 大小計算 (最小 16 以觸發 Tensor Core)
    def triton_block_size(dim):
        return max(16, 2 ** (dim - 1).bit_length())

    BLOCK_OUT1 = triton_block_size(OUT1)
    BLOCK_IN2  = triton_block_size(IN2)
    BLOCK_OUT2 = triton_block_size(OUT2)

    # 💡 決定 K 維度的 Tiling 切塊大小 (避免大矩陣把 SRAM 撐爆)
    BLOCK_K1 = min(32, triton_block_size(IN1))

    grid = (B_sz,)

    flash_kron_moe_fwd_kernel[grid](
        x_sub, A_experts, B_experts,
        top_k_indices, top_k_probs, Y,
        x_sub.stride(0), x_sub.stride(1), x_sub.stride(2),
        A_experts.stride(0), A_experts.stride(1), A_experts.stride(2),
        B_experts.stride(0), B_experts.stride(1), B_experts.stride(2),
        Y.stride(0), Y.stride(1), Y.stride(2),
        IN1, IN2, OUT1, OUT2, TOPK,
        BLOCK_OUT1=BLOCK_OUT1, BLOCK_IN2=BLOCK_IN2, 
        BLOCK_K1=BLOCK_K1, BLOCK_OUT2=BLOCK_OUT2,
    )

    return Y


class FlashKroneckerMoEFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(ctx, x_sub, A_experts, B_experts, top_k_indices, top_k_probs):
        A_experts = A_experts.to(x_sub.dtype)
        B_experts = B_experts.to(x_sub.dtype)
        top_k_probs = top_k_probs.to(x_sub.dtype)

        ctx.save_for_backward(x_sub, A_experts, B_experts, top_k_indices, top_k_probs)
        return flash_kron_moe_forward(x_sub, A_experts, B_experts, top_k_indices, top_k_probs)

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        x_sub, A_experts, B_experts, top_k_indices, top_k_probs = ctx.saved_tensors
        B_sz, top_k = top_k_indices.shape

        grad_output = grad_output.contiguous()

        # 直接取出當下 Token 使用的專家，無需複製整個 x_sub
        # A_gathered: (B_sz, top_k, OUT1, IN1)
        # B_gathered: (B_sz, top_k, OUT2, IN2)
        A_gathered = A_experts[top_k_indices]
        B_gathered = B_experts[top_k_indices]

        # 🚀 終極優化：利用 unsqueeze 創建 View，讓 PyTorch 的 matmul 自動廣播 (Broadcasting)
        # 徹底消滅 repeat_interleave，做到零記憶體複製！
        x_view = x_sub.unsqueeze(1)                          # (B_sz, 1, IN1, IN2)
        gY_view = grad_output.unsqueeze(1)                   # (B_sz, 1, OUT1, OUT2)
        probs_view = top_k_probs.unsqueeze(-1).unsqueeze(-1) # (B_sz, top_k, 1, 1)

        # ----------------------------------------------------
        # 計算 grad_x = sum_k [ (A_k^T @ grad_Y @ B_k) * prob_k ]
        # ----------------------------------------------------
        # M1 = A_k^T @ grad_Y  ->  (B_sz, top_k, IN1, OUT2)
        M1 = torch.matmul(A_gathered.transpose(-1, -2), gY_view)

        # dX = M1 @ B_k        ->  (B_sz, top_k, IN1, IN2)
        dX = torch.matmul(M1, B_gathered)

        # 套用機率並把 top_k 維度壓平加總  ->  (B_sz, IN1, IN2)
        grad_x = (dX * probs_view).sum(dim=1)

        # ----------------------------------------------------
        # 計算 grad_A = (grad_Y @ B_k @ X^T) * prob_k
        # ----------------------------------------------------
        # M2 = grad_Y @ B_k    ->  (B_sz, top_k, OUT1, IN2)
        M2 = torch.matmul(gY_view, B_gathered)

        # dA = M2 @ X^T        ->  (B_sz, top_k, OUT1, IN1)
        dA = torch.matmul(M2, x_view.transpose(-1, -2)) * probs_view

        # 使用 scatter_add_ 累加回原本形狀的 grad_A (解決同一個 Batch 選到相同專家的衝突)
        grad_A = torch.zeros_like(A_experts)
        flat_dA = dA.view(-1, dA.shape[-2], dA.shape[-1])
        flat_indices_A = top_k_indices.view(-1, 1, 1).expand_as(flat_dA)
        grad_A.scatter_add_(0, flat_indices_A, flat_dA)

        # ----------------------------------------------------
        # 計算 grad_B = (grad_Y^T @ A_k @ X) * prob_k
        # ----------------------------------------------------
        # M3 = grad_Y^T @ A_k  ->  (B_sz, top_k, OUT2, IN1)
        M3 = torch.matmul(gY_view.transpose(-1, -2), A_gathered)

        # dB = M3 @ X          ->  (B_sz, top_k, OUT2, IN2)
        dB = torch.matmul(M3, x_view) * probs_view

        # 累加回 grad_B
        grad_B = torch.zeros_like(B_experts)
        flat_dB = dB.view(-1, dB.shape[-2], dB.shape[-1])
        flat_indices_B = top_k_indices.view(-1, 1, 1).expand_as(flat_dB)
        grad_B.scatter_add_(0, flat_indices_B, flat_dB)

        return grad_x, grad_A, grad_B, None, None

import os
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from accelerate import Accelerator
import torch
import gc

import numpy as np

import warnings


warnings.filterwarnings(
    "ignore",
    message=".*Online softmax is disabled on the fly.*",
    category=UserWarning
)
# 減少 CUDA 記憶體碎片（新變數名稱；舊版 PYTORCH_CUDA_ALLOC_CONF 已棄用）
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# 讓 PyTorch 自動判斷要用 bf16(A100) 還是 fp16(T4)
import torch

if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    # A100 / RTX 3090 等級以上
    MIXED_PRECISION = "bf16"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("🚀 偵測到高階 GPU，啟用 bf16 與 TF32 最佳化！")
else:
    # T4 / V100 等級
    MIXED_PRECISION = "fp16" # T4 只能用 fp16
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print("🐢 偵測到舊版 GPU (如 T4)，自動 Fallback 至 fp16。")

# 之後在初始化 Accelerator 時改成：
# accelerator = Accelerator(mixed_precision=MIXED_PRECISION)

# ==========================================
# 🚀 新增：Router 穩定度黑科技 (Triton & Annealing)
# ==========================================
@triton.jit
def tanh_approx(x):
    return tl.inline_asm_elementwise(
        "tanh.approx.f32 $0, $1;", constraints="=f,f", args=[x],
        dtype=tl.float32, is_pure=True, pack=1,
    )

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

class _FastScaledTanh(torch.autograd.Function):
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

def get_router_temperature(step, warmup=500, total=10000, t_start=2.0, t_end=0.5):
    if step is None: return t_end
    if step < warmup: return t_start
    progress = min((step - warmup) / max(1, total - warmup), 1.0)
    return t_end + 0.5 * (t_start - t_end) * (1.0 + math.cos(math.pi * progress))


# ==========================================
# 1. K-MoE Mamba-3 Config
# ==========================================
class Mamba3Config:
    def __init__(
        self,
        d_model=768,
        d_state=64,
        d_head=64,
        n_groups=1,
        mimo_rank=4,
        expand=4,
        num_layers=15,
        use_conv=False,
        d_conv=4,
        rms_norm_eps=1e-5,
        chunk_size=65,
        use_parallel_scan=True,

        # === K-MoE Configs ===
        use_kmoe=True,
        kmoe_num_experts=1024,
        kmoe_top_k=2,

        # === Mamba-2 Initialization ===
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        A_init_range=(1, 16),
    ):
        self.d_model = d_model
        self.d_state = d_state
        self.d_head = d_head
        self.expand = expand
        self.num_layers = num_layers

        self.d_inner = int(expand * d_model)
        assert self.d_inner % d_head == 0, "d_inner must be divisible by d_head"
        self.n_heads = self.d_inner // d_head

        assert self.n_heads % n_groups == 0, f"n_heads ({self.n_heads}) must be divisible by n_groups"
        self.n_groups = n_groups
        self.mimo_rank = mimo_rank
        self.use_conv = use_conv
        self.d_conv = d_conv
        self.rms_norm_eps = rms_norm_eps
        self.chunk_size = chunk_size
        self.use_parallel_scan = use_parallel_scan

        self.use_kmoe = use_kmoe
        self.kmoe_num_experts = kmoe_num_experts
        self.kmoe_top_k = kmoe_top_k

        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init_floor = dt_init_floor
        self.dt_limit = dt_limit
        self.A_init_range = A_init_range

# ==========================================
# 2. RMSNorm
# ==========================================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x, step=None):
        rrms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rrms * self.weight

# ==========================================
# 3. KroneckerMoE
# ==========================================
class KroneckerMoE(nn.Module):
    def __init__(self, dim_in1, dim_in2, dim_out1, dim_out2, num_experts=1024, top_k=2):
        super().__init__()
        self.dim_in1 = dim_in1
        self.dim_in2 = dim_in2
        self.dim_out1 = dim_out1
        self.dim_out2 = dim_out2
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)

        self.router = nn.Linear(dim_in1 * dim_in2, num_experts, bias=False)
        self.A_experts = nn.Parameter(torch.randn(num_experts, dim_out1, dim_in1))
        self.B_experts = nn.Parameter(torch.randn(num_experts, dim_out2, dim_in2))

        std_A = (1.0 / math.sqrt(dim_in1 * dim_out1)) ** 0.5
        std_B = (1.0 / math.sqrt(dim_in2 * dim_out2)) ** 0.5

        nn.init.normal_(self.A_experts, mean=0.0, std=std_A)
        nn.init.normal_(self.B_experts, mean=0.0, std=std_B)
        nn.init.normal_(self.router.weight, mean=0.0, std=0.02)

        self.register_parameter('scale', nn.Parameter(torch.ones(1)))
        self.bias = nn.Parameter(torch.zeros(dim_out1 * dim_out2))

    def forward(self, x, step=None): # 🚀 新增 step 參數
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.dim_in1 * self.dim_in2)
        B_flat = x_flat.size(0)

        # 🚀 1. Router 黑科技：溫度退火與 Soft Capping
        temperature = get_router_temperature(step)
        raw_logits = self.router(x_flat)
        capped_logits = fast_scaled_tanh(raw_logits, 10.0)
        
        # 🚀 2. 計算 Z-loss
        if self.training:
            z_loss = torch.mean(torch.logsumexp(capped_logits, dim=-1) ** 2)
        else:
            z_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        # 🚀 3. 套用溫度
        router_logits = capped_logits / temperature
        router_probs = torch.softmax(router_logits, dim=-1)

        top_k_vals, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
        top_k_probs = torch.softmax(top_k_vals, dim=-1)

        # 計算原本的 Load Balancing Aux Loss
        if self.training:
            expert_mask = torch.zeros(B_flat, self.num_experts, device=x.device, dtype=torch.float32)
            expert_mask.scatter_(1, top_k_indices, 1.0)
            epoch_f_i = expert_mask.mean(dim=0)
            epoch_P_i = router_probs.float().mean(dim=0)
            aux_loss = self.num_experts * torch.sum(epoch_f_i * epoch_P_i)
            aux_loss = aux_loss.to(x.dtype)
        else:
            aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        x_sub = x_flat.reshape(B_flat, self.dim_in1, self.dim_in2)

        # ==========================================
        # 🚀 終極效能優化區塊：呼叫 Flash Autograd Function
        # ==========================================
        # 這裡會自動走 Triton Forward，並且在 backward 時呼叫我們寫的公式
        output = FlashKroneckerMoEFunction.apply(
            x_sub, 
            self.A_experts, 
            self.B_experts, 
            top_k_indices, 
            top_k_probs
        )
        # ==========================================

        output = output.reshape(*orig_shape[:-1], -1)
        output = output * self.scale + self.bias
        
        # 🚀 回傳新增 z_loss
        return output, aux_loss, z_loss



# ==========================================
# 4. K-MoE Mamba-3 Block
# ==========================================
class Mamba3Block(nn.Module):
    def __init__(self, config: Mamba3Config):
        super().__init__()
        self.config = config
        d_in, H, G, P, N, R = config.d_model, config.n_heads, config.n_groups, config.d_head, config.d_state, config.mimo_rank
        self.ratio = H // G
        self.dim_z = H * P
        self.dim_x = H * P
        self.dim_B = G * N * R
        self.dim_C = G * N * R
        self.dim_dt = G
        self.dim_A = G
        self.dim_lambda = G

        d_proj_total = self.dim_z + self.dim_x + self.dim_B + self.dim_C + self.dim_dt + self.dim_A + self.dim_lambda
        self.in_proj = nn.Linear(d_in, d_proj_total, bias=True)

        if config.use_kmoe:
            def get_factors(n):
                for i in range(int(math.sqrt(n)), 0, -1):
                    if n % i == 0: return i, n // i
                return 1, n
            p1, p2 = get_factors(P)
            q1, q2 = get_factors(P * R)
            self.x_up_proj = KroneckerMoE(p1, p2, q1, q2, config.kmoe_num_experts, config.kmoe_top_k)
        else:
            self.x_up_proj = nn.Linear(P, P * R, bias=False)

        self.y_down_proj = nn.Linear(P * R, P, bias=False)

        self.theta_log = nn.Parameter(torch.randn(G, N // 2))
        self.D = nn.Parameter(torch.ones(H))

        self.norm_B = RMSNorm(N * R, eps=config.rms_norm_eps)
        self.norm_C = RMSNorm(N * R, eps=config.rms_norm_eps)
        self.bias_B = nn.Parameter(torch.zeros(G, N, R))
        self.bias_C = nn.Parameter(torch.zeros(G, N, R))

        if config.use_kmoe:
            d_inner_f1, d_inner_f2 = get_factors(config.d_inner)
            d_in_f1, d_in_f2 = get_factors(d_in)
            self.out_proj = KroneckerMoE(d_inner_f1, d_inner_f2, d_in_f1, d_in_f2, config.kmoe_num_experts, config.kmoe_top_k)
        else:
            self.out_proj = nn.Linear(config.d_inner, d_in, bias=False)

        self.pre_gate_norm = RMSNorm(H * P)
        self.act = nn.SiLU()

        with torch.no_grad():
            if not config.use_kmoe:
                nn.init.xavier_uniform_(self.x_up_proj.weight, gain=1.0 / math.sqrt(R) if R > 1 else 1.0)
            nn.init.xavier_uniform_(self.y_down_proj.weight, gain=1.0 / math.sqrt(R) if R > 1 else 1.0)
            self.bias_B.fill_(1.0)
            self.bias_C.fill_(1.0)
            A_min, A_max = config.A_init_range
            dt = torch.clamp(torch.exp(torch.rand(G) * (math.log(config.dt_max) - math.log(config.dt_min)) + math.log(config.dt_min)), min=config.dt_init_floor)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            dt_start = self.dim_z + self.dim_x + self.dim_B + self.dim_C
            dt_end = dt_start + self.dim_dt
            A_end = dt_end + self.dim_A
            self.in_proj.bias[dt_start:dt_end].copy_(inv_dt)
            self.in_proj.bias[dt_end:A_end].uniform_(A_min, A_max).log_()
            self.in_proj.bias[A_end:].fill_(-3.0)

    # 🚀 最佳化寫法 (使用 torch.view_as_complex)：
    def apply_rope(self, x, angles):
        # x 形狀: (B, L, H, N, R) 
        # angles 形狀: (B, L, H, N/2)
        N_half = angles.shape[-1]

        # 1. 針對 N 維度拆出實部與虛部
        # 形狀變為: (B, L, H, N/2, 2, R)
        x_reshaped = x.float().view(*x.shape[:-2], N_half, 2, x.shape[-1])

        # 2. 將 2 這個維度換到最後面，並確保記憶體連續，才能觸發底層複數轉換
        # 形狀變為: (B, L, H, N/2, R, 2)
        x_transposed = x_reshaped.transpose(-1, -2).contiguous()

        # 3. 轉換為複數張量
        # 形狀變為: (B, L, H, N/2, R)
        x_complex = torch.view_as_complex(x_transposed)

        # 4. 準備旋轉頻率
        # angles 擴展為: (B, L, H, N/2, 1) 以便跟 R 維度廣播
        freqs_complex = torch.polar(
            torch.ones_like(angles, dtype=torch.float32), 
            angles.float()
        ).unsqueeze(-1)

        # 5. 複數相乘完成旋轉 (GPU 內部會自動處理 R 維度的廣播)
        x_rotated = x_complex * freqs_complex

        # 6. 轉回實數，再把維度搬回原樣
        # 轉回實數形狀: (B, L, H, N/2, R, 2)
        x_rotated_real = torch.view_as_real(x_rotated)

        # 轉置回來 (..., N/2, 2, R)，最後 reshape 復原 (..., N, R)
        return x_rotated_real.transpose(-1, -2).reshape_as(x).type_as(x)

    def segsum(self, x):
        T = x.size(-1)
        x_cumsum = torch.cumsum(x, dim=-1)
        x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=0)
        return x_segsum.masked_fill(~mask, -float('inf'))

    def chunk_parallel_scan(self, u, dt, A, C, chunk_size=128):
        B, L, H, N, P = u.shape
        R, device, input_dtype = C.shape[-1], u.device, u.dtype
        L_orig = L

        # 1. 處理 Padding (保持不變，這是必要的)
        if L % chunk_size != 0:
            pad_len = chunk_size - (L % chunk_size)
            u = F.pad(u, (0, 0, 0, 0, 0, 0, 0, pad_len))
            dt = F.pad(dt, (0, 0, 0, pad_len))
            C = F.pad(C, (0, 0, 0, 0, 0, 0, 0, pad_len))
            L = L + pad_len

        num_chunks = L // chunk_size
        log_alpha = dt * A

        # 2. 建立 View (全部都是 O(1) 的指標操作，不觸發記憶體複製)
        u_chunk = u.view(B, num_chunks, chunk_size, H, N, P)
        log_alpha_chunk = log_alpha.view(B, num_chunks, chunk_size, H)
        C_chunk = C.view(B, num_chunks, chunk_size, H, N, R)

        # 3. 計算 Intra-chunk Mask (只做一次 transpose，且只用於 segsum)
        # log_a_t: (B, num_chunks, H, chunk_size)
        log_a_t = log_alpha_chunk.transpose(-1, -2)
        L_mask = torch.exp(self.segsum(log_a_t)) # (B, num_chunks, H, chunk_size, chunk_size)

        # ==========================================
        # 🔥 優化核心 1：利用 Einsum 徹底消滅 Permute + Reshape
        # ==========================================
        # 舊寫法：permute -> reshape -> matmul -> reshape -> permute
        # 新寫法：直接宣告輸入與輸出的維度對應，讓 Inductor 自動生成最優 Triton 核心
        # L_mask: (b, c, h, l, s), u_chunk: (b, c, s, h, n, p) -> h_intra: (b, c, l, h, n, p)
        h_intra = torch.einsum('bchls, bcshnp -> bclhnp', L_mask, u_chunk)

        # 計算 Y_diag (對角線/Chunk內輸出)
        # 舊寫法：轉置 -> 攤平 -> matmul -> reshape
        # 新寫法：沿著 N 維度做內積 (h^T @ C)
        # h_intra: (b, c, l, h, n, p), C_chunk: (b, c, l, h, n, r) -> y_diag: (b, c, l, h, p, r)
        y_diag = torch.einsum('bclhnp, bclhnr -> bclhpr', h_intra, C_chunk)

        # ==========================================
        # 🔥 優化核心 2：預先分配連續記憶體 (取代 List + torch.stack)
        # ==========================================
        decay_chunk = torch.exp(torch.sum(log_alpha_chunk, dim=2)) # (B, num_chunks, H)
        h_chunk_final = h_intra[:, :, -1, :, :, :]                 # (B, num_chunks, H, N, P)

        h_prev = torch.zeros(B, H, N, P, device=device, dtype=input_dtype)

        # 預先分配一整塊連續記憶體，避免 stack 時的記憶體碎片與複製
        h_inter_tensor = torch.empty(B, num_chunks, H, N, P, device=device, dtype=input_dtype)

        for c in range(num_chunks):
            h_inter_tensor[:, c] = h_prev
            decay = decay_chunk[:, c].view(B, H, 1, 1)
            h_prev = h_prev * decay + h_chunk_final[:, c]

        # ==========================================
        # 🔥 優化核心 3：跨 Chunk 的 Offset 計算同樣改用 Einsum
        # ==========================================
        decay_intra = torch.exp(torch.cumsum(log_alpha_chunk, dim=2)) # (B, num_chunks, chunk_size, H)
        c_decayed = C_chunk * decay_intra.unsqueeze(-1).unsqueeze(-1) # (B, num_chunks, chunk_size, H, N, R)

        # 舊寫法：unsqueeze -> transpose -> matmul
        # 新寫法：沿著 N 維度內積，將 h_inter_tensor 廣播到 L (chunk_size)
        # h_inter: (b, c, h, n, p), c_decayed: (b, c, l, h, n, r) -> y_off: (b, c, l, h, p, r)
        y_off = torch.einsum('bchnp, bclhnr -> bclhpr', h_inter_tensor, c_decayed)

        # 4. 最終組合
        y_total = y_diag + y_off

        # 安全地轉回原始序列長度
        y_total = y_total.view(B, -1, H, P, R)
        if L_orig < L:
            y_total = y_total[:, :L_orig]

        return y_total.to(input_dtype), h_prev.to(input_dtype)

    def forward(self, u, step=None):
        B_sz, L, _ = u.shape
        H, G, P, N, R = self.config.n_heads, self.config.n_groups, self.config.d_head, self.config.d_state, self.config.mimo_rank
        ratio = self.ratio

        projected = self.in_proj(u)
        z, x_prime, B_param, C_param, dt, A_param, lambda_param = torch.split(projected, [self.dim_z, self.dim_x, self.dim_B, self.dim_C, self.dim_dt, self.dim_A, self.dim_lambda], dim=-1)

        x_prime = x_prime.view(B_sz, L, H, P)

        dt = F.softplus(dt)
        A = -torch.exp(A_param)
        theta = torch.exp(self.theta_log)

        broadcast_group = lambda t, _: t.repeat_interleave(ratio, dim=2)
        dt = broadcast_group(dt.unsqueeze(-1), None).squeeze(-1)
        A_broadcast, theta_broadcast = broadcast_group(A.unsqueeze(-1), None).squeeze(-1), theta.repeat_interleave(ratio, dim=0)

        angles = torch.cumsum(torch.einsum('blh, hn -> blhn', dt, theta_broadcast), dim=1)
        B_param_normed = self.norm_B(B_param.reshape(B_sz, L, G, N * R)).view(B_sz, L, G, N, R) + self.bias_B
        C_param_normed = self.norm_C(C_param.reshape(B_sz, L, G, N * R)).view(B_sz, L, G, N, R) + self.bias_C

        B_rotated = self.apply_rope(broadcast_group(B_param_normed, None), angles)
        C_rotated = self.apply_rope(broadcast_group(C_param_normed, None), angles)

        if self.config.use_kmoe:
            # KroneckerMoE 的 x_up 路徑輸入維度是 P（最後一維），保留 Head 維度避免 shape 錯位
            x_up, aux_loss_up, z_loss_up = self.x_up_proj(x_prime, step=step)
            x = x_up.view(B_sz, L, H, P, R)
        else:
            x = self.x_up_proj(x_prime).view(B_sz, L, H, P, R)
            aux_loss_up, z_loss_up = 0.0, 0.0

        input_signal = torch.einsum('blhnr, blhpr -> blhnp', B_rotated, x)
        lambda_view = F.sigmoid(broadcast_group(lambda_param.unsqueeze(-1), None).squeeze(-1)).view(B_sz, L, H, 1, 1)
        dt_view = dt.view(B_sz, L, H, 1, 1)
        alpha_view = torch.exp(dt * A_broadcast).view(B_sz, L, H, 1, 1)

        input_signal_prev = torch.roll(input_signal, shifts=1, dims=1)
        input_signal_prev[:, 0] = 0

        # Memory-friendly rewrite to avoid building multiple giant temporaries.
        # Original:
        #   u_ssm = lambda*dt*input + (1-lambda)*dt*alpha*input_prev
        u_ssm = input_signal * lambda_view
        u_ssm.mul_(dt_view)

        input_signal_prev.mul_(alpha_view)
        input_signal_prev.mul_(1.0 - lambda_view)
        input_signal_prev.mul_(dt_view)
        u_ssm.add_(input_signal_prev)

        if self.config.use_parallel_scan:
            y_stack, h_state = self.chunk_parallel_scan(u_ssm, dt, A_broadcast, C_rotated, chunk_size=self.config.chunk_size)
        else:
            h_state = torch.zeros(B_sz, H, N, P, device=u.device)
            y_stack_list = []
            for t in range(L):
                h_state = h_state * alpha_view[:, t] + u_ssm[:, t]
                y_stack_list.append(torch.einsum('bhnp, bhnr -> bhpr', h_state, C_rotated[:, t]))
            y_stack = torch.stack(y_stack_list, dim=1)

        y = self.y_down_proj(y_stack.view(B_sz, L, H, P * R)).view(B_sz, L, H * P)
        y = y + x_prime.reshape(B_sz, L, H * P) * self.D.repeat_interleave(P, dim=0)

        # 🌟 論文強推的改進點：在乘上 act(z) 之前做 Norm
        y = self.pre_gate_norm(y)
        y = y * self.act(z)

        if self.config.use_kmoe:
            # 🚀 這裡要改
            out_y, aux_loss_out, z_loss_out = self.out_proj(y, step=step)
        else:
            out_y = self.out_proj(y)
            aux_loss_out, z_loss_out = 0.0, 0.0

        block_aux_loss = aux_loss_up + aux_loss_out
        block_z_loss = z_loss_up + z_loss_out
        return out_y, block_aux_loss, block_z_loss

# ==========================================
# 5. Transformer Block (1 in every 5 layers)
# ==========================================
class TransformerBlock(nn.Module):
    """A single causal Transformer block with optional K-MoE FFN."""
    def __init__(self, config: Mamba3Config):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.d_model // 64,  # auto-scale heads
            dropout=0.0,
            batch_first=True
        )
        self.norm_attn = RMSNorm(config.d_model)

        self.use_kmoe = config.use_kmoe
        if config.use_kmoe:
            self.ffn = KMoEFeedForward(config)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(config.d_model, config.d_model * 4),
                nn.GELU(),
                nn.Linear(config.d_model * 4, config.d_model)
            )
        self.norm_ffn = RMSNorm(config.d_model)

    def forward(self, x):
        B, L, D = x.shape
        # Causal attention
        causal_mask = nn.Transformer.generate_square_subsequent_mask(L, device=x.device)
        attn_out, _ = self.attn(
            self.norm_attn(x),
            self.norm_attn(x),
            self.norm_attn(x),
            attn_mask=causal_mask,
            is_causal=True,
            need_weights=False  # 👈 這是觸發 FlashAttention 的關鍵
        )
        x = x + attn_out

        # FFN
        h_norm = self.norm_ffn(x)
        if self.use_kmoe:
            ffn_out, ffn_loss, ffn_z_loss = self.ffn(h_norm, step=step)
        else:
            ffn_out = self.ffn(h_norm)
            ffn_loss, ffn_z_loss = 0.0, 0.0
        x = x + ffn_out
        return x, ffn_loss, ffn_z_loss

# ==========================================
# 5.5 K-MoE Transformer FeedForward
# ==========================================
class KMoEFeedForward(nn.Module):
    def __init__(self, config: Mamba3Config):
        super().__init__()
        def get_factors(n):
            for i in range(int(math.sqrt(n)), 0, -1):
                if n % i == 0: return i, n // i
            return 1, n

        d_model = config.d_model
        d_ff = d_model * 4

        d1, d2 = get_factors(d_model)
        f1, f2 = get_factors(d_ff)

        self.up_proj = KroneckerMoE(d1, d2, f1, f2, config.kmoe_num_experts, config.kmoe_top_k)
        self.down_proj = KroneckerMoE(f1, f2, d1, d2, config.kmoe_num_experts, config.kmoe_top_k)
        self.act = nn.GELU()

    def forward(self, x, step=None):
        h, loss_up, z_up = self.up_proj(x, step=step)
        h = self.act(h)
        y, loss_down, z_down = self.down_proj(h, step=step)
        return y, loss_up + loss_down, z_up + z_down

# ==========================================
# 6. True 4:1 Interleaved Hybrid Backbone
# ==========================================
class TrueHybridMamba(nn.Module):
    """
    Full-sequence 4:1 interleaved Mamba-Transformer backbone.
    No chunking, no .detach() - full gradient flow over entire sequence.
    Each macro-block = [Mamba x4, Transformer x1].
    config.num_layers = number of macro-blocks.
    """
    def __init__(self, config: Mamba3Config, mamba_ratio: int = 4):
        super().__init__()
        self.config = config
        self.mamba_ratio = mamba_ratio
        self.num_macro_blocks = config.num_layers
        self.layer_types = []

        # Build flat interleaved layer list
        self.layers = nn.ModuleList()
        for macro in range(self.num_macro_blocks):
            # 4 Mamba Layers
            for _ in range(self.mamba_ratio):
                self.layer_types.append('mamba')
                self.layers.append(nn.ModuleDict({
                    'norm': RMSNorm(config.d_model),
                    'block': Mamba3Block(config)
                }))
            # 1 Transformer Layer
            self.layer_types.append('transformer')
            self.layers.append(nn.ModuleDict({
                'block': TransformerBlock(config)
            }))

    def forward(self, x, step=None):
        # x: (B, L, d_model) - full sequence, no chunking
        total_aux_loss = 0.0
        total_z_loss = 0.0 # 🚀 新增

        for i, layer_dict in enumerate(self.layers):
            l_type = self.layer_types[i]

            if l_type == 'mamba':
                # Pre-norm + residual；啟用 checkpoint 以節省記憶體
                normed_x = layer_dict['norm'](x)

                # 注意：使用 non-reentrant 版本以符合 PyTorch 2.x 要求
                # 🚀 傳入 step，並接住三個回傳值
                out, aux, z_loss = checkpoint(layer_dict['block'], normed_x, step, use_reentrant=False)
                if isinstance(aux, torch.Tensor):
                    total_aux_loss = total_aux_loss + aux
                    total_z_loss = total_z_loss + z_loss # 🚀 累加
                x = x + out

            elif l_type == 'transformer':
                # TransformerBlock: causal attn over full L, K-MoE FFN
                # Block 本身已處理殘差，這裡直接用 checkpoint 包起來
                out, aux, z_loss = checkpoint(layer_dict['block'], x, step, use_reentrant=False)
                if isinstance(aux, torch.Tensor):
                    total_aux_loss = total_aux_loss + aux
                    total_z_loss = total_z_loss + z_loss # 🚀 累加
                x = out

        return x, total_aux_loss, total_z_loss

# ==========================================
# 7. Language Model
# ==========================================
class Mamba3LanguageModel(nn.Module):
    def __init__(self, config: Mamba3Config, vocab_size: int, **kwargs):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, config.d_model)
        self.backbone = TrueHybridMamba(config)
        self.norm = RMSNorm(config.d_model)
        self.head = nn.Linear(config.d_model, vocab_size, bias=False)
        self.head.weight = self.embed.weight  # weight tying
        self.config = config
        # for logging breakdown of CE loss vs aux loss from MoE
        self._last_loss_terms = None

        # 標準 Transformer 初始化：std=0.02 防止初始 logits 數值爆炸（解決 Loss 564 問題）
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None, step=None):
        x = self.embed(input_ids)
        # 🚀 接住 z_loss
        x, aux_loss, z_loss = self.backbone(x, step=step)
        x = self.norm(x)
        
        # 🚀 新版的 Logits 也要 Capping 防止爆掉 (可選，但強烈建議)
        raw_logits = self.head(x)
        logits = fast_scaled_tanh(raw_logits, 30.0)

        if labels is not None:
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            if isinstance(aux_loss, torch.Tensor):
                aux_loss = aux_loss.mean()
                z_loss = z_loss.mean()
            
            num_moe_layers = self.config.num_layers * (self.config.expand * 2 + 2)
            raw_aux = aux_loss.detach()
            
            # 🚀 計算兩種 Aux 貢獻
            aux_contrib = (0.01 / max(1, num_moe_layers)) * aux_loss
            z_contrib = (5e-3 / max(1, num_moe_layers)) * z_loss 
            
            # 🚀 總 Loss
            loss = ce_loss + aux_contrib + z_contrib
            
            try:
                self._last_loss_terms = {
                    "ce_loss": ce_loss.detach(),
                    "aux_loss": aux_contrib.detach(),
                    "z_loss": z_contrib.detach(), # 🚀 記錄 z_loss
                    "raw_aux": raw_aux.detach(),
                }
            except Exception:
                self._last_loss_terms = None

            loss = loss.unsqueeze(0)
            return loss, raw_aux.unsqueeze(0)

        return logits

# ==========================================
# 7. Memmapped Binary Dataset (🚀 NumPy 大矩陣切片優化版)
# ==========================================

class PretokenizedDataset(IterableDataset):
    def __init__(self, data_path, seq_len, buffer_size=4_000_000):
        """
        🚀 終極優化版：大區塊緩存 (Chunked Prefetching) + NumPy 矩陣化切片
        消滅 Python 內層迴圈，極大化 DataLoader 吞吐量ㄉ。
        """
        if not os.path.exists(data_path):
             raise FileNotFoundError(f"找不到預處理檔案！請確認路徑: {data_path}")

        self.data_path = data_path
        self.seq_len = seq_len
        self.buffer_size = buffer_size

        # 僅獲取長度資訊，不預載入全部資料到 RAM
        data_info = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.total_tokens = len(data_info)
        del data_info

    def __iter__(self):
        # 處理多進程 (Multi-processing) 時的資料切分，避免多個 worker 讀到重複資料
        worker_info = get_worker_info()
        if worker_info is None:
            start_idx, end_idx = 0, self.total_tokens
        else:
            per_worker = self.total_tokens // worker_info.num_workers
            start_idx = worker_info.id * per_worker
            # 確保最後一個 worker 能讀到結尾
            end_idx = start_idx + per_worker if worker_info.id != worker_info.num_workers - 1 else self.total_tokens

        # 每個 worker 獨立掛載 memmap
        mmap_data = np.memmap(self.data_path, dtype=np.uint16, mode='r')
        curr_idx = start_idx

        while curr_idx + self.seq_len < end_idx:
            # 決定這一連串的高速緩取區間 (Buffer)
            chunk_end = min(curr_idx + self.buffer_size, end_idx)

            # 從硬碟連續讀取一大塊資料到 RAM，並轉換為 int64 (PyTorch Embedding 需要 LongTensor)
            buffer = mmap_data[curr_idx : chunk_end].astype(np.int64)

            # 計算這塊 buffer 可以完整切出多少條長度為 seq_len 的序列
            num_seqs = (len(buffer) - 1) // self.seq_len 

            if num_seqs > 0:
                # 🚀 魔法就在這裡：使用 NumPy reshape 一次性產生矩陣，徹底消滅 while 迴圈
                x_arr = buffer[:num_seqs * self.seq_len].reshape(num_seqs, self.seq_len)
                y_arr = buffer[1 : num_seqs * self.seq_len + 1].reshape(num_seqs, self.seq_len)

                # 雖然這裡還是有 for 迴圈，但只是 yield 已經建好的視圖(View)，開銷極低
                for i in range(num_seqs):
                    yield torch.from_numpy(x_arr[i]), torch.from_numpy(y_arr[i])

                # 更新指標，準備讀取下一塊 buffer
                curr_idx += num_seqs * self.seq_len
            else:
                # 如果剩下的資料長度連一條 seq_len 都湊不齊，就直接跳出，避免無窮迴圈
                break

        # 釋放資源
        del mmap_data

# ==========================================
# 8. Main Training Loop
# ==========================================
def main():
    import time as _t
    _init_start = _t.time()

    # ── Step 1/8: Google Drive 掛載 ──
    print("\n" + "="*60)
    print("[1/8] 📁 掛載 Google Drive...  -> pass!!")


    # ── Step 2/8: 啟用 Accelerate ──
    print("[2/8] ⚙️  初始化 Accelerator (mixed_precision=bf16)...")
    accelerator = Accelerator(mixed_precision=MIXED_PRECISION)
    print(f"      ✅ Device: {accelerator.device} | Processes: {accelerator.num_processes}")
    print(f"      🎯 Mixed Precision: {accelerator.mixed_precision} | TF32: {torch.backends.cuda.matmul.allow_tf32}")

    # === 路徑與輸出 (來自底部全域設定) ===
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # === 模型架構 (參數來自底部全域設定) ===
    config = Mamba3Config(
        d_model=D_MODEL,
        d_state=D_STATE,
        expand=EXPAND,
        num_layers=NUM_LAYERS,
        use_parallel_scan=True,
        chunk_size=CHUNK_SIZE,
        use_kmoe=True,
        kmoe_num_experts=KMOE_NUM_EXPERTS,
        mimo_rank=MIMO_RANK,
        kmoe_top_k=KMOE_TOP_K
    )

    # ── Step 3/8: 初始化模型 ──
    print("[3/8] 🧠 初始化模型 (Mamba3LanguageModel)...")
    _t3 = _t.time()
    model = Mamba3LanguageModel(config, vocab_size=VOCAB_SIZE)
    print(f"      ✅ 模型建立完成！({_t.time()-_t3:.1f}s)")

    # ── Step 4/8: torch.compile ──
    if hasattr(torch, "compile"):
        print("[4/8] 🔥 編譯模型 (torch.compile, mode=reduce-overhead)...")
        print("      ⚠️ 第一次執行需要等待幾分鐘編譯，後續 step 會大幅加速。")
        _t4 = _t.time()
        # 1. 全域強制關閉 Inductor 的 CUDA Graphs
        # 1. 直接編譯模型，透過 options 關閉 CUDA graphs
        model = torch.compile(
            model,
            dynamic=False,
        )


        print(f"      ✅ 編譯完成！({_t.time()-_t4:.1f}s)")
    else:
        print("[4/8] ⚠️ torch.compile 不可用，略過。")

    print("=== Model Architecture ===")
    total_params = sum(p.numel() for p in model.parameters())
    moe_params = sum(p.numel() for m in model.modules() if isinstance(m, KroneckerMoE) for p in m.parameters(recurse=False))
    # 分析不同 KroneckerMoE 模組的「每個專家」參數量
    expert_shapes = {}
    for m in model.modules():
        if isinstance(m, KroneckerMoE):
            key = (m.dim_in1, m.dim_in2, m.dim_out1, m.dim_out2, m.num_experts)
            if key not in expert_shapes:
                # A_experts: (num_experts, dim_out1, dim_in1)
                # B_experts: (num_experts, dim_out2, dim_in2)
                per_expert_A = m.A_experts.shape[1] * m.A_experts.shape[2]
                per_expert_B = m.B_experts.shape[1] * m.B_experts.shape[2]
                per_expert_total = per_expert_A + per_expert_B
                expert_shapes[key] = per_expert_total
    mamba_params = sum(
        p.numel() for i, ld in enumerate(model.backbone.layers)
        if model.backbone.layer_types[i] == 'mamba'
        for p in ld.parameters()
    )
    transformer_params = sum(
        p.numel() for i, ld in enumerate(model.backbone.layers)
        if model.backbone.layer_types[i] == 'transformer'
        for p in ld.parameters()
    )
    embed_params = sum(p.numel() for p in model.embed.parameters()) + \
                   sum(p.numel() for p in model.norm.parameters())
    num_mac = config.num_layers
    print(f"Total Parameters: {total_params/1e6:.2f}M")
    print(f"  ├─ Mamba Layers ({num_mac*4} x K-MoE Mamba3Block): {mamba_params/1e6:.2f}M")
    print(f"  ├─ Transformer Layers ({num_mac} x Causal Attn + K-MoE FFN): {transformer_params/1e6:.2f}M")
    print(f"  ├─ Embeddings & Head: {embed_params/1e6:.2f}M")
    print(f"  └─ MoE Routing & Expert Params: {moe_params/1e6:.2f}M")
    if expert_shapes:
        print("      └─ Per-Expert Param Counts (A,B Kronecker mats only):")
        for (din1, din2, dout1, dout2, n_exp), per_exp in expert_shapes.items():
            print(
                f"         • Experts shape ({din1}x{din2} → {dout1}x{dout2}), "
                f"{n_exp} experts: {per_exp/1e3:.2f}K params / expert"
            )

    # Visual full stack display
    mamba_ratio = 4
    total_layers = num_mac * (mamba_ratio + 1)
    print(f"\n=== Layer Stack (4:1 Mamba-Transformer Interleaved, {total_layers} total layers) ===")
    print(f"  [Embedding]  d_model={config.d_model}, vocab={VOCAB_SIZE}")
    layer_num = 1
    for mb in range(num_mac):
        print(f"  --- Macro Block {mb+1}/{num_mac} ---")
        for k in range(mamba_ratio):
            print(f"  Layer {layer_num:02d} │ 🔵 Mamba3 KMoE  (d_model={config.d_model}, d_state={config.d_state}, K-MoE experts={config.kmoe_num_experts})")
            layer_num += 1
        print(f"  Layer {layer_num:02d} │ 🟠 Transformer  (Causal Attn {config.d_model//64} heads + K-MoE FFN experts={config.kmoe_num_experts})")
        layer_num += 1
    print(f"  [LM Head]    tied weights")
    print(f"      ✅ 架構顯示完成。")

    device = accelerator.device
    num_gpus = accelerator.num_processes
    print(f"Using {num_gpus} processes with Accelerate (device: {device})")
    print(f"🎯 Mixed Precision: {accelerator.mixed_precision}  |  TF32: {torch.backends.cuda.matmul.allow_tf32}")

    # ── Step 5/8: 初始化優化器 ──
    print("[5/8] ⚙️  初始化 Fused AdamW 優化器...")
    use_fused = torch.cuda.is_available()
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.1, fused=use_fused)
    print(f"      ✅ AdamW (fused={use_fused}, lr={LR}, warmup={WARMUP})")

    def lr_lambda(current_step: int):
        if current_step < WARMUP:
            return float(current_step) / float(max(1, WARMUP))
        progress = float(current_step - WARMUP) / float(max(1, STEPS - WARMUP))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = LambdaLR(optimizer, lr_lambda)

    # ── Step 6/8: 載入 Checkpoint ──
    print("[6/8] 💾 檢查 Checkpoint...")
    start_step = 0

    def unwrap_model(model):
        if hasattr(model, "module"):
            model = model.module
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        return model

    if os.path.exists(CHECKPOINT_SAVE_PATH):
        print(f"      🔄 發現 Checkpoint: {CHECKPOINT_SAVE_PATH}")
        try:
            _t6 = _t.time()
            ckpt = torch.load(CHECKPOINT_SAVE_PATH, map_location="cpu")

            real_model = unwrap_model(model)
            real_model.load_state_dict(ckpt["model"])

            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])

            start_step = ckpt["step"]

            print(f"      ✅ 成功恢復！Step {start_step} ({_t.time()-_t6:.1f}s)")
        except Exception as e:
            print(f"      ⚠️ 讀取失敗: {e}，重新訓練")
            start_step = 0
    else:
        print("      🆕 找不到 Checkpoint — 從頭開始訓練。")
    _t7 = _t.time()
    dataset = PretokenizedDataset(DATA_PATH, seq_len=SEQ_LEN)
    _num_workers = min(8, os.cpu_count())
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=_num_workers,
        prefetch_factor=4,
        pin_memory=True
    )
    print(f"      ✅ 資料集已載入！({dataset.total_tokens:,} tokens, {_num_workers} workers, {_t.time()-_t7:.1f}s)")

    # ── Step 8/8: Accelerate Prepare ──
    print("[8/8] 🚀 Accelerate Prepare (分配模型/資料到設備)...")
    _t8 = _t.time()
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )
    print(f"      ✅ Prepare 完成！({_t.time()-_t8:.1f}s)")
    print(f"\n{'='*60}")
    print(f"✅ 初始化全部完成！總共花費 {_t.time()-_init_start:.1f} 秒。")
    print(f"{'='*60}\n")

    model.train()
    data_iter = iter(dataloader)
    optimizer.zero_grad()

    global_step = start_step
    batch_idx = 0
    running_loss = 0.0
    running_aux = 0.0
    running_z = 0.0
    running_raw_aux = 0.0

    if accelerator.is_main_process and not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            f.write("step,loss,aux_loss,z_loss,raw_aux,ppl,lr,mem_gb,step_time\n")

    world_size = num_gpus
    if accelerator.is_main_process:
        print(
            f"🚀 開始訓練！Global Batch Size = {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * world_size} "
            f"({BATCH_SIZE}/GPU × {GRADIENT_ACCUMULATION_STEPS} accum × {world_size} GPUs)"
        )

    if accelerator.is_main_process:
        step_start_time = time.time()

    while global_step < STEPS:
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            try:
                x, y = next(data_iter)
            except StopIteration:
                print("Dataset empty! Ensure .bin file is at", DATA_PATH)
                break

        # x, y = x.to(device), y.to(device)
        # 讓資料傳輸與 GPU 計算在背景重疊進行
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with accelerator.autocast():
            # 訓練時只需要 loss，避免把巨量 logits gather 回單卡
            loss, raw_aux = model(x, labels=y, step=global_step)
            # 轉成 CPU scalar 做 logging；多卡時各進程各自記錄本地平均

            # 從模型中取出剛才存下來的 scalar 字典
            if hasattr(model, "module") and getattr(model.module, "_last_loss_terms", None) is not None:
                loss_terms = model.module._last_loss_terms
            elif getattr(model, "_last_loss_terms", None) is not None:
                loss_terms = model._last_loss_terms
            else:
                loss_terms = {"ce_loss": 0.0, "aux_loss": 0.0, "z_loss": 0.0, "raw_aux": raw_aux.detach()}

            loss_for_log = loss.detach().float().mean().item()
            def get_float(val):
                return val.float().mean().item() if isinstance(val, torch.Tensor) else val

            aux_for_log = get_float(loss_terms.get("aux_loss", 0.0))
            z_for_log = get_float(loss_terms.get("z_loss", 0.0)) # 取出 z_loss
            raw_aux_log = get_float(loss_terms.get("raw_aux", raw_aux.detach()))

            loss = loss / GRADIENT_ACCUMULATION_STEPS

        accelerator.backward(loss)
        running_loss += loss_for_log  # track pre-scaled loss for accurate reporting
        running_aux += aux_for_log
        running_z += z_for_log
        running_raw_aux += raw_aux_log
        batch_idx += 1

        # Show batch-level progress within each accumulation window
        step_within = batch_idx % GRADIENT_ACCUMULATION_STEPS or GRADIENT_ACCUMULATION_STEPS
        if step_within % 8 == 0 or step_within == GRADIENT_ACCUMULATION_STEPS:
            cur_loss = running_loss / step_within  # running avg
            print(f"  ⏳ GS {global_step+1} | Accum [{step_within:2d}/{GRADIENT_ACCUMULATION_STEPS}] | Avg Loss: {cur_loss:.4f}")

        if batch_idx % GRADIENT_ACCUMULATION_STEPS == 0:
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            loss_val = running_loss / GRADIENT_ACCUMULATION_STEPS
            aux_val = running_aux / GRADIENT_ACCUMULATION_STEPS
            z_val = running_z / GRADIENT_ACCUMULATION_STEPS
            raw_aux_val = running_raw_aux / GRADIENT_ACCUMULATION_STEPS
            running_loss = 0.0
            running_aux = 0.0
            running_z = 0.0
            running_raw_aux = 0.0
            ppl = math.exp(min(loss_val, 20))
            lr_val = scheduler.get_last_lr()[0]
            # GPU memory usage (in GB) on當前進程的 device
            if torch.cuda.is_available():
                mem_gb_0 = torch.cuda.memory_allocated(device) / 1e9
                mem_gb_1 = 0.0
            else:
                mem_gb_0 = 0.0
                mem_gb_1 = 0.0

            if accelerator.is_main_process:
                step_time = time.time() - step_start_time
                log_line = (
                    f"Step {global_step:05d}/{STEPS} | Time {step_time:.2f}s | "
                    f"Loss: {loss_val:.4f} (aux_scaled {aux_val:.4f}, z_scaled {z_val:.4f}, aux_raw {raw_aux_val:.4f}) | "
                    f"PPL: {ppl:.2f} | LR: {lr_val:.2e} | "
                    f"Mem[GB] GPU={mem_gb_0:.2f}"
                )
                print(f"✅ {log_line}")

                # Save every step to CSV file
                with open(LOG_FILE, "a") as f:
                    f.write(f"{global_step},{loss_val:.6f},{aux_val:.6f},{z_val:.6f},{raw_aux_val:.6f},{ppl:.4f},{lr_val:.2e},{mem_gb_0:.3f},{mem_gb_1:.3f},{step_time:.3f}\n")


                step_start_time = time.time()

                # 每 CHECKPOINT_EVERY 步存一次到 Google Drive
                if global_step % CHECKPOINT_EVERY == 0:
                    # only kaggle fixed????
                    unwrapped = model.module if hasattr(model, "module") else model
                    if hasattr(unwrapped, "_orig_mod"):
                        unwrapped = unwrapped._orig_mod

                    torch.save({
                        'step': global_step,
                        'total_steps': STEPS,
                        'last_loss': round(loss_val, 4),
                        'model': unwrapped.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                    }, CHECKPOINT_SAVE_PATH)
                    print(f"💾 Checkpoint → Google Drive | Step {global_step}/{STEPS} | Loss: {loss_val:.4f}")

    if accelerator.is_main_process:
        print("🎉 Training Completed.")
        # only kaggle fixed????
        unwrapped = model.module if hasattr(model, "module") else model
        if hasattr(unwrapped, "_orig_mod"):
            unwrapped = unwrapped._orig_mod

        final_path = os.path.join(OUTPUT_DIR, "mamba3_colab_final.pt")
        torch.save({
            'step': STEPS,
            'model': unwrapped.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, final_path)
        print(f"✅ 最終模型儲存到 Google Drive: {final_path}")

# ==========================================
# 9. ⭐ 全域超參數設定 (在這裡統一調整所有旋鈕)
# ==========================================

OUTPUT_DIR = "/kaggle/working/"
CHECKPOINT_SAVE_PATH = os.path.join(OUTPUT_DIR, "mamba3_colab_checkpoint.pt")
LOG_FILE = os.path.join(OUTPUT_DIR, "colab_training_log.csv")
DATA_PATH = "/kaggle/input/datasets/s990093/fineweb-edu-tokenized/fineweb_edu_tokenized.bin"

# --- 訓練超參數 ---
BATCH_SIZE                  = 8       # 單 GPU 每次餵入的樣本數
GRADIENT_ACCUMULATION_STEPS = 2     # 有效 Global Batch = BATCH_SIZE × 這個值 = 32
SEQ_LEN                     = 512     # 每條訓練序列的長度 (A100 可 setting 到 2048)
STEPS                       = 10000   # 總訓練步數
LR                          = 2.5e-4    # 學習率 (AdamW)
WARMUP                      = 1000     # 前 N 步從 0 線性上升到 LR
CHECKPOINT_EVERY            = 100     # 每幾步存一次到 Drive (避免 Drive Rate Limit)

# --- 模型架構 ---
VOCAB_SIZE                  = 32000   # 與 prepare_fineweb_data.py 的 SentencePiece 一致
D_MODEL                     = 768    # 必須是 64 的倍數 (MultiheadAttention 需要整除)
D_STATE                     = 64      # Mamba SSM 狀態維度
EXPAND                      = 4       # SSM 展開率
NUM_LAYERS                  = 5      # Macro Block 數量 (每個 = 4 Mamba + 1 Transformer)
CHUNK_SIZE                  = 64      # Parallel Scan Chunk 大小
KMOE_NUM_EXPERTS            = 256     # K-MoE 專家數量
MIMO_RANK                   = 4       # Kronecker 分解 Rank
KMOE_TOP_K                  = 4       # 每個 Token 選用的專家數


def print_config():
    global_batch = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS

    print("\n" + "="*60)
    print("🚀 Mamba3 Training Configuration")
    print("="*60)

    print("\n📦 Training Hyperparameters")
    print("-"*60)
    print(f"{'Batch Size (per GPU)':30}: {BATCH_SIZE}")
    print(f"{'Gradient Accumulation':30}: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"{'Effective Global Batch':30}: {global_batch}")
    print(f"{'Sequence Length':30}: {SEQ_LEN}")
    print(f"{'Total Steps':30}: {STEPS}")
    print(f"{'Learning Rate':30}: {LR}")
    print(f"{'Warmup Steps':30}: {WARMUP}")
    print(f"{'Checkpoint Every':30}: {CHECKPOINT_EVERY}")

    print("\n🧠 Model Architecture")
    print("-"*60)
    print(f"{'Vocab Size':30}: {VOCAB_SIZE}")
    print(f"{'Model Dimension (d_model)':30}: {D_MODEL}")
    print(f"{'SSM State Dim':30}: {D_STATE}")
    print(f"{'SSM Expand Ratio':30}: {EXPAND}")
    print(f"{'Macro Layers':30}: {NUM_LAYERS}")
    print(f"{'Parallel Scan Chunk':30}: {CHUNK_SIZE}")

    print("\n🧩 Advanced Modules")
    print("-"*60)
    print(f"{'K-MoE Experts':30}: {KMOE_NUM_EXPERTS}")
    print(f"{'K-MoE Top-K':30}: {KMOE_TOP_K}")
    print(f"{'MIMO Rank':30}: {MIMO_RANK}")

    print("\n📊 Derived Stats")
    print("-"*60)
    tokens_per_step = global_batch * SEQ_LEN
    tokens_total = tokens_per_step * STEPS

    print(f"{'Tokens / Step':30}: {tokens_per_step:,}")
    print(f"{'Total Training Tokens':30}: {tokens_total:,}")

    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    gc.collect()  # 強制回收 Python 中未使用的變數
    print_config()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # 釋放 PyTorch 佔用的 CUDA 快取
        torch.cuda.reset_peak_memory_stats()  # 重置峰值記憶體統計，方便後續觀察
        print("      ✅ GPU 記憶體清理完成！")
    else:
        print("      ⚠️ 找不到 GPU，略過清理。")

    main()


