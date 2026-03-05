# ============================================================================
# CRITICAL FIX for Multi-GPU in Notebooks
# ============================================================================
# MUST be set BEFORE any imports to prevent CUDA context initialization
import sys
import os

# Configuration: Set to True for dual-GPU training
# If True, Triton will be FORCEFULLY DISABLED to prevent fork crashes
USE_MULTI_GPU = True  # Change to False for single-GPU + Triton acceleration

if USE_MULTI_GPU:
    os.environ['DISABLE_TRITON'] = '1'
    print("🔒 Dual-GPU Mode: Triton is FORCEFULLY DISABLED to prevent crash.")
else:
    os.environ['DISABLE_TRITON'] = '0'
    print("⚡ Single-GPU Mode: Triton enabled (if available).")

# ============================================================================
# CRITICAL FIX for Kaggle/Colab PyTorch circular import issue
# ============================================================================
# Disable torch.compile to avoid circular import
os.environ['PYTORCH_JIT'] = '0'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Must import torch FIRST before anything else
import torch

# Disable torch.compile and dynamo completely
torch.compiler.disable()
if hasattr(torch, '_dynamo'):
    torch._dynamo.config.suppress_errors = True

# Now safe to import other torch modules
import torch.nn as nn
import torch.nn.functional as F
import math
import requests
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import time

# ============================================================================
# Multi-GPU Support with Accelerate
# ============================================================================
try:
    from accelerate import Accelerator
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False
    print("⚠️  Warning: accelerate not installed. Running in single-GPU mode.")
    print("   To enable multi-GPU: pip install accelerate")

# ============================================================================
# Progress Bar Support
# ============================================================================
try:
    from tqdm.auto import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("⚠️  Warning: tqdm not installed. Progress bars disabled.")
    print("   To enable progress bars: pip install tqdm")

# ============================================================================
# Triton Support for GPU Acceleration
# ============================================================================
# CRITICAL: Triton kernels are incompatible with DDP's fork-based multiprocessing
# Set DISABLE_TRITON=1 environment variable to disable Triton (required for multi-GPU)
import os
TRITON_DISABLED_BY_ENV = os.environ.get('DISABLE_TRITON', '0') == '1'

if TRITON_DISABLED_BY_ENV:
    TRITON_AVAILABLE = False
    print("⚠️  Triton disabled via DISABLE_TRITON environment variable")
    print("   Using PyTorch fallback (required for multi-GPU training)")
else:
    try:
        import triton
        import triton.language as tl
        
        # Check if we're in a DDP subprocess
        if 'RANK' in os.environ or 'LOCAL_RANK' in os.environ:
            TRITON_AVAILABLE = False
            print("⚠️  Triton disabled in DDP subprocess")
        else:
            TRITON_AVAILABLE = True
            
    except ImportError:
        TRITON_AVAILABLE = False
        print("⚠️  Warning: triton not installed. Using PyTorch fallback.")
        print("   To enable Triton acceleration: pip install triton")

# ============================================================================
# Part 1: Triton Kernels for Acceleration
# ============================================================================

if TRITON_AVAILABLE:
    @triton.jit
    def inter_chunk_scan_kernel_fwd(
        H_ptr,      # Output [B*H, C, N, P]
        X_ptr,      # Input  [B*H, C, N, P] (h_chunk_final)
        Decay_ptr,  # Decay  [B*H, C]
        
        stride_h_bh, stride_h_c, stride_h_n, stride_h_p,
        stride_x_bh, stride_x_c, stride_x_n, stride_x_p,
        stride_d_bh, stride_d_c,
        
        n_chunks: tl.constexpr,
        dim_n: tl.constexpr,
        dim_p: tl.constexpr,
        
        BLOCK_N: tl.constexpr,
        BLOCK_P: tl.constexpr
    ):
        """
        Triton kernel for inter-chunk recurrence scan.
        Computes: h[c] = h[c-1] * decay[c] + x[c] for each chunk c.
        
        Grid: (num_bh, cdiv(N, BLOCK_N))
        """
        # Program IDs
        pid_bh = tl.program_id(0)  # Batch*Head index
        pid_n = tl.program_id(1)   # N dimension block
        
        # N dimension offsets
        off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = off_n < dim_n
        
        # P dimension offsets (assume P <= BLOCK_P)
        off_p = tl.arange(0, BLOCK_P)
        mask_p = off_p < dim_p
        
        # Combined mask [BLOCK_N, BLOCK_P]
        mask = mask_n[:, None] & mask_p[None, :]
        
        # Initialize accumulator [BLOCK_N, BLOCK_P] in FP32 for stability
        h_acc = tl.zeros((BLOCK_N, BLOCK_P), dtype=tl.float32)
        
        # Sequential loop over chunks (this is the recurrence dependency)
        for c in range(n_chunks):
            # 1. Load decay scalar for this chunk
            decay_ptr_loc = Decay_ptr + pid_bh * stride_d_bh + c * stride_d_c
            decay_val = tl.load(decay_ptr_loc)
            
            # 2. Load input contribution X [BLOCK_N, BLOCK_P]
            x_ptr_base = X_ptr + pid_bh * stride_x_bh + c * stride_x_c
            x_ptrs = x_ptr_base + (off_n[:, None] * stride_x_n) + (off_p[None, :] * stride_x_p)
            x_val = tl.load(x_ptrs, mask=mask, other=0.0)
            
            # 3. Recurrence: h = h * decay + x
            h_acc = h_acc * decay_val + x_val
            
            # 4. Store result for this chunk
            h_out_base = H_ptr + pid_bh * stride_h_bh + c * stride_h_c
            h_out_ptrs = h_out_base + (off_n[:, None] * stride_h_n) + (off_p[None, :] * stride_h_p)
            tl.store(h_out_ptrs, h_acc, mask=mask)


    def triton_inter_chunk_scan(x, decay):
        """
        Triton-accelerated inter-chunk scan.
        
        Args:
            x: [B, C, H, N, P] - Contribution from each chunk
            decay: [B, C, H] - Decay factor per chunk
        
        Returns:
            h: [B, C, H, N, P] - Accumulated state after each chunk
        """
        B, C, H, N, P = x.shape
        
        # Reshape to merge B and H for simpler grid mapping
        # [B, C, H, N, P] -> [B, H, C, N, P] -> view as [B*H, C, N, P]
        x_view = x.permute(0, 2, 1, 3, 4).contiguous()
        x_view = x_view.view(B * H, C, N, P)
        
        # [B, C, H] -> [B, H, C] -> view as [B*H, C]
        decay_view = decay.permute(0, 2, 1).contiguous()
        decay_view = decay_view.view(B * H, C)
        
        # Allocate output
        out = torch.empty_like(x_view)
        
        # Grid configuration
        BLOCK_N = 32
        BLOCK_P = triton.next_power_of_2(P)
        BLOCK_P = min(BLOCK_P, 128)  # Cap at 128 for register pressure
        
        grid = (B * H, triton.cdiv(N, BLOCK_N))
        
        # Launch kernel
        inter_chunk_scan_kernel_fwd[grid](
            out, x_view, decay_view,
            # Strides for Out [B*H, C, N, P]
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            # Strides for X [B*H, C, N, P]
            x_view.stride(0), x_view.stride(1), x_view.stride(2), x_view.stride(3),
            # Strides for Decay [B*H, C]
            decay_view.stride(0), decay_view.stride(1),
            # Shape info
            n_chunks=C,
            dim_n=N,
            dim_p=P,
            BLOCK_N=BLOCK_N,
            BLOCK_P=BLOCK_P
        )
        
        # Restore original shape [B*H, C, N, P] -> [B, H, C, N, P] -> [B, C, H, N, P]
        out = out.view(B, H, C, N, P)
        out = out.permute(0, 2, 1, 3, 4)
        
        return out

else:
    # Fallback: Triton not available
    def triton_inter_chunk_scan(x, decay):
        raise RuntimeError("Triton not available. Please install triton or disable Triton acceleration.")


# ============================================================================
# Part 2: Core Mamba-3 Components (from model.py)
# ============================================================================

class Mamba3Config:
    def __init__(
        self, 
        d_model=256, 
        d_state=64, 
        d_head=64, 
        n_groups=1, 
        mimo_rank=4,
        expand=2,
        use_conv=False,
        d_conv=4,
        rms_norm_eps=1e-5,
        chunk_size=256,
        use_parallel_scan=True,
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
        
        self.d_inner = int(expand * d_model)
        assert self.d_inner % d_head == 0
        self.n_heads = self.d_inner // d_head
        
        assert self.n_heads % n_groups == 0
        self.n_groups = n_groups
        self.mimo_rank = mimo_rank
        
        self.use_conv = use_conv
        self.d_conv = d_conv
        self.rms_norm_eps = rms_norm_eps
        self.chunk_size = chunk_size
        self.use_parallel_scan = use_parallel_scan
        
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init_floor = dt_init_floor
        self.dt_limit = dt_limit
        self.A_init_range = A_init_range


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rrms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rrms * self.weight


class Mamba3Block(nn.Module):
    """Mamba-3 Block with Grouped SSM and MIMO"""
    
    def __init__(self, config: Mamba3Config):
        super().__init__()
        self.config = config
        
        H = config.n_heads
        G = config.n_groups
        P = config.d_head
        N = config.d_state
        R = config.mimo_rank

        self.d_inner = H * P
        self.ratio = H // G

        # Projections
        self.dim_z = H * P
        self.dim_x = H * P
        self.dim_B = G * N * R
        self.dim_C = G * N * R
        self.dim_dt = G
        self.dim_lambda = G
        
        d_proj_total = self.dim_z + self.dim_x + self.dim_B + self.dim_C + self.dim_dt + self.dim_lambda
        self.in_proj = nn.Linear(config.d_model, d_proj_total, bias=True)

        if config.use_conv:
            self.conv = nn.Conv1d(
                self.dim_x, self.dim_x, bias=True,
                kernel_size=config.d_conv, groups=self.dim_x,
                padding=config.d_conv - 1
            )
        
        # MIMO Projections
        self.x_up_proj = nn.Linear(P, P * R, bias=False)
        self.y_down_proj = nn.Linear(P * R, P, bias=False)

        # Parameters
        A_min, A_max = config.A_init_range
        self.A_log = nn.Parameter(torch.empty(G).uniform_(A_min, A_max).log())
        self.A_log._no_weight_decay = True
        
        self.theta_log = nn.Parameter(torch.randn(G, N // 2))
        self.D = nn.Parameter(torch.ones(H))

        # Norms and Biases
        self.norm_B = RMSNorm(N * R, eps=config.rms_norm_eps)
        self.norm_C = RMSNorm(N * R, eps=config.rms_norm_eps)
        self.bias_B = nn.Parameter(torch.zeros(G, N, R))
        self.bias_C = nn.Parameter(torch.zeros(G, N, R))

        self.out_proj = nn.Linear(self.d_inner, config.d_model, bias=False)
        self.act = nn.SiLU()
        
        # Initialization
        with torch.no_grad():
            rank_scale = 1.0 / math.sqrt(R) if R > 1 else 1.0
            nn.init.xavier_uniform_(self.x_up_proj.weight, gain=rank_scale)
            nn.init.xavier_uniform_(self.y_down_proj.weight, gain=rank_scale)
            self.bias_B.fill_(1.0)
            self.bias_C.fill_(1.0)
            
            # dt initialization
            dt = torch.exp(
                torch.rand(G) * (math.log(config.dt_max) - math.log(config.dt_min))
                + math.log(config.dt_min)
            )
            dt = torch.clamp(dt, min=config.dt_init_floor)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            
            dt_start = self.dim_z + self.dim_x + self.dim_B + self.dim_C
            dt_end = dt_start + self.dim_dt
            self.in_proj.bias[dt_start:dt_end].copy_(inv_dt)
            
            lambda_start = dt_end
            self.in_proj.bias[lambda_start:].fill_(-3.0)
    
    def apply_rope(self, x, angles):
        N_half = angles.shape[-1]
        x_reshaped = x.view(*x.shape[:-2], N_half, 2, x.shape[-1])
        real_part = x_reshaped[..., 0, :]
        imag_part = x_reshaped[..., 1, :]

        w_cos = torch.cos(angles).unsqueeze(-1)
        w_sin = torch.sin(angles).unsqueeze(-1)

        real_rot = real_part * w_cos - imag_part * w_sin
        imag_rot = real_part * w_sin + imag_part * w_cos
        
        x_out = torch.stack([real_rot, imag_rot], dim=-2)
        return x_out.flatten(-3, -2)

    def segsum(self, x):
        T = x.size(-1)
        x_cumsum = torch.cumsum(x, dim=-1)
        x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=0)
        x_segsum = x_segsum.masked_fill(~mask, -float('inf'))
        return x_segsum
    
    def chunk_parallel_scan(self, u, dt, A, C, chunk_size=128):
        B, L, H, N, P = u.shape
        R = C.shape[-1]
        device = u.device
        input_dtype = u.dtype
        
        # Padding
        L_orig = L
        if L % chunk_size != 0:
            pad_len = chunk_size - (L % chunk_size)
            u = F.pad(u, (0, 0, 0, 0, 0, 0, 0, pad_len))
            dt = F.pad(dt, (0, 0, 0, pad_len))
            C = F.pad(C, (0, 0, 0, 0, 0, 0, 0, pad_len))
            L = L + pad_len
        
        log_alpha = dt * A.view(1, 1, H)
        num_chunks = L // chunk_size
        
        # Reshape
        u_chunk = u.view(B, num_chunks, chunk_size, H, N, P)
        dt_chunk = dt.view(B, num_chunks, chunk_size, H)
        log_alpha_chunk = log_alpha.view(B, num_chunks, chunk_size, H)
        C_chunk = C.view(B, num_chunks, chunk_size, H, N, R)
        
        # Intra-chunk (keep PyTorch - already optimized with Tensor Cores)
        log_alpha_perm = log_alpha_chunk.permute(0, 1, 3, 2)
        L_mask = torch.exp(self.segsum(log_alpha_perm))
        
        # Ensure FP16 for Tensor Core utilization
        L_mask = L_mask.to(u.dtype)
        
        h_intra = torch.einsum('bchij, bcjhnp -> bcihnp', L_mask, u_chunk)
        y_diag = torch.einsum('bclhnp, bclhnr -> bclhpr', h_intra, C_chunk)
        
        # Inter-chunk - USE TRITON ACCELERATION
        decay_chunk = torch.exp(torch.sum(log_alpha_chunk, dim=2))  # [B, num_chunks, H]
        h_chunk_final = h_intra[:, :, -1].contiguous()  # [B, num_chunks, H, N, P]
        
        # Try Triton acceleration if available and on CUDA
        # 🔴 SAFETY: Never use Triton in multi-GPU mode (causes DDP hangs)
        use_triton = TRITON_AVAILABLE and device.type == 'cuda'
        
        if use_triton:
            try:
                # Triton kernel expects [B, C, H, N, P] and [B, C, H]
                h_states_inter = triton_inter_chunk_scan(
                    h_chunk_final, 
                    decay_chunk
                )
                
                # Shift: current chunk uses previous chunk's output
                # h_prevs for chunk `i` is h_states_inter `i-1`
                h_prev = torch.roll(h_states_inter, shifts=1, dims=1)
                h_prev[:, 0, ...] = 0  # First chunk has no predecessor
                
                # Final state for return
                final_h_prev = h_states_inter[:, -1, ...].contiguous()
                
            except Exception as e:
                # Fallback to PyTorch if Triton fails
                print(f"⚠️  Triton kernel failed: {e}. Falling back to PyTorch.")
                use_triton = False
        
        if not use_triton:
            # PyTorch fallback (original implementation)
            h_prev_state = torch.zeros(B, H, N, P, device=device, dtype=input_dtype)
            h_states_inter = []
            
            for c in range(num_chunks):
                h_states_inter.append(h_prev_state)
                decay = decay_chunk[:, c].view(B, H, 1, 1)
                contrib = h_chunk_final[:, c]
                h_prev_state = h_prev_state * decay + contrib
            
            h_states_inter = torch.stack(h_states_inter, dim=1)
            h_prev = h_states_inter
            final_h_prev = h_prev_state
        
        # Combine
        log_alpha_cumsum = torch.cumsum(log_alpha_chunk, dim=2)
        decay_intra = torch.exp(log_alpha_cumsum)
        h_effect = torch.einsum('bchnp, bclh -> bclhnp', h_prev, decay_intra)
        y_off = torch.einsum('bclhnp, bclhnr -> bclhpr', h_effect, C_chunk)
        
        y_total = y_diag + y_off
        y_total = y_total.view(B, L, H, P, R)
        
        if L_orig < L:
            y_total = y_total[:, :L_orig]
        
        return y_total.to(input_dtype), final_h_prev.to(input_dtype)


    def forward(self, u, return_states=False, return_internals=False):
        B_sz, L, _ = u.shape
        H, G, P, N, R = self.config.n_heads, self.config.n_groups, self.config.d_head, self.config.d_state, self.config.mimo_rank
        ratio = self.ratio

        # Projection & Split
        projected = self.in_proj(u)
        split_sections = [self.dim_z, self.dim_x, self.dim_B, self.dim_C, self.dim_dt, self.dim_lambda]
        z, x_prime, B_param, C_param, dt, lambda_param = torch.split(projected, split_sections, dim=-1)

        if self.config.use_conv:
            x_prime_conv = x_prime.transpose(1, 2)
            x_prime_conv = self.conv(x_prime_conv)
            x_prime_conv = x_prime_conv[:, :, :L]
            x_prime = x_prime_conv.transpose(1, 2)
        
        x_prime = x_prime.view(B_sz, L, H, P)
        
        # Activations
        dt = F.softplus(dt)
        min_dt, max_dt = self.config.dt_limit
        if self.config.dt_limit != (0.0, float("inf")):
            dt = dt.clamp(min=min_dt, max=max_dt)
        
        A = -torch.exp(self.A_log)
        theta = torch.exp(self.theta_log)
        
        # Broadcasting
        def broadcast_group(tensor, target_shape_suffix):
            return tensor.repeat_interleave(ratio, dim=2)
            
        dt = broadcast_group(dt.unsqueeze(-1), (1,)).squeeze(-1)
        A_broadcast = A.repeat_interleave(ratio, dim=0)
        theta_broadcast = theta.repeat_interleave(ratio, dim=0)
        
        # RoPE
        dt_theta = torch.einsum('blh, hn -> blhn', dt, theta_broadcast)
        angles = torch.cumsum(dt_theta, dim=1)
        
        B_param = B_param.view(B_sz, L, G, N, R)
        C_param = C_param.view(B_sz, L, G, N, R)
        
        B_param = self.norm_B(B_param.flatten(-2, -1)).view(B_sz, L, G, N, R) + self.bias_B
        C_param = self.norm_C(C_param.flatten(-2, -1)).view(B_sz, L, G, N, R) + self.bias_C

        B_rotated_input = broadcast_group(B_param, (N, R))
        C_rotated_input = broadcast_group(C_param, (N, R))
        
        B_rotated = self.apply_rope(B_rotated_input, angles)
        C_rotated = self.apply_rope(C_rotated_input, angles)
        
        # MIMO
        x = self.x_up_proj(x_prime).view(B_sz, L, H, P, R)
        input_signal = torch.einsum('blhnr, blhpr -> blhnp', B_rotated, x)

        # Trapezoidal Rule
        lambda_view = F.sigmoid(broadcast_group(lambda_param.unsqueeze(-1), (1,)).squeeze(-1))
        alpha_val = torch.exp(torch.einsum('blh, h -> blh', dt, A_broadcast))
        
        dt_view = dt.view(B_sz, L, H, 1, 1)
        lambda_view = lambda_view.view(B_sz, L, H, 1, 1)
        alpha_view = alpha_val.view(B_sz, L, H, 1, 1)

        term_curr = lambda_view * dt_view * input_signal
        input_signal_prev = torch.roll(input_signal, shifts=1, dims=1)
        input_signal_prev[:, 0] = 0
        term_prev = (1 - lambda_view) * dt_view * alpha_view * input_signal_prev
        u_ssm = term_curr + term_prev

        # SSM Scan
        if self.config.use_parallel_scan:
            y_stack, h_state = self.chunk_parallel_scan(u_ssm, dt, A_broadcast, C_rotated, 
                                                        chunk_size=self.config.chunk_size)
        else:
            h_state = torch.zeros(B_sz, H, N, P, device=u.device)
            y_stack_list = []
            
            for t in range(L):
                h_state = h_state * alpha_view[:, t] + u_ssm[:, t]
                y_t = torch.einsum('bhnp, bhnr -> bhpr', h_state, C_rotated[:, t])
                y_stack_list.append(y_t)
                
            y_stack = torch.stack(y_stack_list, dim=1)

        if return_internals:
             return {
                 'h_states': h_state,
                 'inputs': u,
                 'angles': angles,
                 'alpha': alpha_val
             }

        # Output
        y_down = self.y_down_proj(y_stack.view(B_sz, L, H, P * R))
        y = y_down.view(B_sz, L, H * P)

        # Skip Connection
        x_prime_view = x_prime.reshape(B_sz, L, H * P)
        y = y + x_prime_view * self.D.repeat_interleave(P, dim=0)
        
        # Gate
        z_act = self.act(z)
        y = y * z_act

        if return_states:
             return self.out_proj(y), h_state

        return self.out_proj(y)


# ============================================================================
# Part 2: Mamba-3 Language Model
# ============================================================================

class Mamba3LM(nn.Module):
    """Mamba-3 Language Model"""
    
    def __init__(
        self,
        vocab_size=50257,
        d_model=512,
        n_layers=12,
        d_state=64,
        d_head=64,
        n_groups=4,
        mimo_rank=4,
        expand=2,
        max_seq_len=2048,
        dropout=0.0,
        tie_embeddings=True,
        use_rope=False,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.tie_embeddings = tie_embeddings
        
        # Token Embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional Embedding (optional)
        if use_rope:
            self.pos_embedding = None
        else:
            self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
            nn.init.normal_(self.pos_embedding, std=0.02)
        
        # Mamba-3 Config
        self.config = Mamba3Config(
            d_model=d_model,
            d_state=d_state,
            d_head=d_head,
            n_groups=n_groups,
            mimo_rank=mimo_rank,
            expand=expand,
            use_conv=False,
            d_conv=4,
            chunk_size=256,
            use_parallel_scan=True,
        )
        
        # Mamba-3 Blocks
        self.layers = nn.ModuleList([
            Mamba3Block(self.config) for _ in range(n_layers)
        ])
        
        # Layer Norms
        self.norms = nn.ModuleList([
            RMSNorm(d_model) for _ in range(n_layers)
        ])
        
        # Final Norm
        self.final_norm = RMSNorm(d_model)
        
        # LM Head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie embeddings
        if tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, targets=None, return_loss=True):
        batch_size, seq_len = input_ids.shape
        
        # Token Embedding
        x = self.token_embedding(input_ids)
        
        # Add Positional Embedding
        if self.pos_embedding is not None:
            x = x + self.pos_embedding[:, :seq_len, :]
        
        x = self.dropout(x)
        
        # Mamba-3 Blocks
        for i, layer in enumerate(self.layers):
            normed_x = self.norms[i](x)
            out = layer(normed_x)
            x = x + out
        
        # Final Norm
        x = self.final_norm(x)
        
        # LM Head
        logits = self.lm_head(x)
        
        # Compute Loss
        loss = None
        if targets is not None and return_loss:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = targets[:, 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_targets.view(-1),
                ignore_index=-100
            )
        
        if return_loss and targets is not None:
            return loss, logits
        else:
            return logits
    
    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens=100,
        temperature=1.0,
        top_k=None,
        top_p=None,
        eos_token_id=None,
    ):
        self.eval()
        
        for _ in range(max_new_tokens):
            if input_ids.size(1) > self.max_seq_len:
                input_ids = input_ids[:, -self.max_seq_len:]
            
            logits = self.forward(input_ids, return_loss=False)
            logits = logits[:, -1, :]
            logits = logits / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        
        return input_ids
    
    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_embedding.weight.numel()
            if not self.tie_embeddings:
                n_params -= self.lm_head.weight.numel()
        return n_params


def create_mamba3_tiny(vocab_size=50257, mimo_rank=4, dropout=0.2):
    """Tiny model for Shakespeare testing"""
    return Mamba3LM(
        vocab_size=vocab_size,
        d_model=256,
        n_layers=4,
        d_state=32,
        d_head=64,
        n_groups=2,
        mimo_rank=mimo_rank,
        expand=2,
        max_seq_len=512,
        dropout=dropout,  # Add dropout to prevent overfitting
    )


# ============================================================================
# Part 3: Shakespeare Dataset and Training
# ============================================================================

class CharDataset(Dataset):
    """Character-level dataset"""
    
    def __init__(self, text, block_size):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.block_size = block_size
        
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        
        self.data = torch.tensor([self.stoi[ch] for ch in text], dtype=torch.long)
        
        print(f"Dataset: {len(text)} characters, {self.vocab_size} unique")
        print(f"Vocabulary: {''.join(chars[:50])}...")
    
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y
    
    def decode(self, tokens):
        return ''.join([self.itos[int(t)] for t in tokens])


def get_shakespeare_data(cache_dir='data'):
    """Download Shakespeare dataset"""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)
    
    data_path = cache_dir / 'shakespeare.txt'
    
    if not data_path.exists():
        print("Downloading Shakespeare dataset...")
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        response = requests.get(url)
        with open(data_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f"✓ Downloaded to {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    return text


@torch.no_grad()
def evaluate(model, val_loader, device, eval_iters):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    
    for i, (x, y) in enumerate(val_loader):
        if i >= eval_iters:
            break
        
        x, y = x.to(device), y.to(device)
        loss, _ = model(x, targets=y, return_loss=True)
        total_loss += loss.item()
    
    return total_loss / min(eval_iters, len(val_loader))


@torch.no_grad()
def generate_sample(model, dataset, device, prompt="\n", max_tokens=200):
    """Generate a sample"""
    model.eval()
    
    tokens = torch.tensor(
        [dataset.stoi[ch] for ch in prompt],
        dtype=torch.long
    ).unsqueeze(0).to(device)
    
    generated = model.generate(
        tokens,
        max_new_tokens=max_tokens,
        temperature=0.8,
        top_k=40
    )
    
    text = dataset.decode(generated[0])
    return text


def train_shakespeare(
    model,
    train_loader,
    val_loader,
    dataset,
    epochs=10,
    lr=3e-4,
    eval_interval=100,
    eval_iters=20,
    use_accelerate=True,
    mixed_precision="fp16",  # Enable FP16 for T4 GPUs (2x faster, 50% memory)
):
    """
    Train on Shakespeare with optional multi-GPU support
    
    Args:
        use_accelerate: If True and accelerate is available, use multi-GPU training
        mixed_precision: "no", "fp16", or "bf16" (fp16 recommended for T4)
    """
    
    # Save batch size before prepare() (it becomes None after)
    original_batch_size = train_loader.batch_size
    
    # ========================================================================
    # Multi-GPU Setup with Accelerate
    # ========================================================================
    if use_accelerate and ACCELERATE_AVAILABLE:
        try:
            # Check if AcceleratorState already exists
            from accelerate.state import AcceleratorState
            
            if AcceleratorState._shared_state != {}:
                # State already initialized, reuse it
                print("⚠️  Accelerator already initialized, reusing existing state...")
                accelerator = Accelerator()
            else:
                # Fresh initialization with mixed precision
                accelerator = Accelerator(mixed_precision=mixed_precision)
                
        except Exception as e:
            print(f"⚠️  Accelerator initialization issue: {e}")
            print("   Trying without mixed_precision parameter...")
            try:
                accelerator = Accelerator()
            except:
                print("   Failed to initialize Accelerator, falling back to single-GPU")
                accelerator = None
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                model = model.to(device)
        
        if accelerator is not None:
            device = accelerator.device
            
            # Print GPU info (only on main process)
            if accelerator.is_main_process:
                print("\n" + "=" * 80)
                print("🚀 Multi-GPU Training with Accelerate")
                print("=" * 80)
                print(f"Number of GPUs: {accelerator.num_processes}")
                print(f"Device: {device}")
                print(f"Mixed Precision: {accelerator.mixed_precision}")
                print(f"Distributed Type: {accelerator.distributed_type}")
                print("=" * 80 + "\n")
    else:
        accelerator = None
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        
        if use_accelerate and not ACCELERATE_AVAILABLE:
            print("⚠️  Accelerate not available, falling back to single-GPU mode")
    
    # ========================================================================
    # Optimizer and Scheduler
    # ========================================================================
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)  # Increased from 0.01 to 0.1
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(train_loader)
    )
    
    # ========================================================================
    # Prepare for Distributed Training (KEY STEP!)
    # ========================================================================
    if accelerator is not None:
        model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, val_loader, scheduler
        )
    
    # ========================================================================
    # Training Info
    # ========================================================================
    if accelerator is None or accelerator.is_main_process:
        print("\n" + "=" * 80)
        print("Training Mamba-3 on Shakespeare")
        print("=" * 80)
        
        # Get model params (handle DDP wrapper)
        if accelerator is not None:
            unwrapped_model = accelerator.unwrap_model(model)
            n_params = unwrapped_model.get_num_params()
        else:
            n_params = model.get_num_params()
        
        print(f"Model parameters: {n_params / 1e6:.2f}M")
        print(f"Device: {device}")
        print(f"Epochs: {epochs}")
        print(f"Learning rate: {lr}")
        
        if accelerator is not None:
            # Use saved batch size
            effective_batch_size = original_batch_size * accelerator.num_processes
            print(f"Batch size per GPU: {original_batch_size}")
            print(f"Effective batch size: {effective_batch_size}")
        else:
            print(f"Batch size: {original_batch_size}")
        
        print("=" * 80 + "\n")
    
    # ========================================================================
    # Training Loop
    # ========================================================================
    global_step = 0
    best_val_loss = float('inf')
    total_samples = 0
    
    # Training history for logging
    training_history = {
        'config': {
            'mimo_rank': model.config.mimo_rank if hasattr(model, 'config') else 'unknown',
            'd_model': model.d_model if hasattr(model, 'd_model') else 'unknown',
            'n_layers': model.n_layers if hasattr(model, 'n_layers') else 'unknown',
            'vocab_size': dataset.vocab_size,
            'epochs': epochs,
            'lr': lr,
            'batch_size': original_batch_size,
            'num_gpus': accelerator.num_processes if accelerator else 1,
            'mixed_precision': accelerator.mixed_precision if accelerator else 'no',
        },
        'epochs': [],
        'steps': [],
    }
    
    # Create epoch progress bar (only on main process)
    if TQDM_AVAILABLE and (accelerator is None or accelerator.is_main_process):
        epoch_pbar = tqdm(
            range(epochs),
            desc="🚀 Training",
            position=0,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
    else:
        epoch_pbar = range(epochs)
    
    for epoch in epoch_pbar:
        model.train()
        epoch_loss = 0
        epoch_start = time.time()
        num_batches = len(train_loader)
        
        epoch_data = {
            'epoch': epoch + 1,
            'train_losses': [],
            'val_loss': None,
            'learning_rates': [],
        }
        
        # Create batch progress bar (only on main process)
        if TQDM_AVAILABLE and (accelerator is None or accelerator.is_main_process):
            batch_pbar = tqdm(
                train_loader,
                desc=f"📊 Epoch {epoch + 1}/{epochs}",
                position=1,
                leave=False,
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
            )
        else:
            batch_pbar = train_loader
        
        for batch_idx, (x, y) in enumerate(batch_pbar):
            # Data is automatically moved to correct device by accelerator
            if accelerator is None:
                x, y = x.to(device), y.to(device)
            
            # Forward pass
            loss, logits = model(x, targets=y, return_loss=True)
            
            # Backward pass
            optimizer.zero_grad()
            
            if accelerator is not None:
                # Accelerator handles gradient synchronization across GPUs
                accelerator.backward(loss)
            else:
                loss.backward()
            
            # Gradient clipping
            if accelerator is not None:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            global_step += 1
            total_samples += x.size(0)
            
            # Record metrics
            current_lr = scheduler.get_last_lr()[0]
            avg_loss_so_far = epoch_loss / (batch_idx + 1)
            
            epoch_data['train_losses'].append(loss.item())
            epoch_data['learning_rates'].append(current_lr)
            
            # Update progress bar with detailed metrics
            if TQDM_AVAILABLE and (accelerator is None or accelerator.is_main_process):
                # Calculate samples/sec
                elapsed = time.time() - epoch_start
                samples_per_sec = total_samples / elapsed if elapsed > 0 else 0
                
                batch_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg': f'{avg_loss_so_far:.4f}',
                    'lr': f'{current_lr:.2e}',
                    'samp/s': f'{samples_per_sec:.0f}'
                })
            
            # Evaluation
            if global_step % eval_interval == 0:
                # --- CRITICAL FIX START: 解決 DDP 死鎖問題 ---
                
                # 1. 加上同步鎖，確保所有 GPU 都完成了目前的訓練步驟
                if accelerator is not None:
                    accelerator.wait_for_everyone()
                
                # Evaluate (only on main process to avoid redundancy)
                if accelerator is None or accelerator.is_main_process:
                    # 2. 【重要】解包模型 (Unwrap Model)
                    # 這是為了避免在 evaluate 呼叫 model() 時觸發 DDP 的同步機制
                    # 如果不解包，Rank 0 呼叫 forward 會等待 Rank 1 的梯度，導致死鎖
                    if accelerator is not None:
                        unwrapped_model = accelerator.unwrap_model(model)
                    else:
                        unwrapped_model = model
                    
                    # 使用解包後的模型進行評估
                    val_loss = evaluate(unwrapped_model, val_loader, device, eval_iters)
                    
                    # Record step metrics
                    training_history['steps'].append({
                        'step': global_step,
                        'epoch': epoch + 1,
                        'train_loss': loss.item(),
                        'val_loss': val_loss,
                        'lr': current_lr,
                        'samples': total_samples,
                    })
                    
                    print(f"\n{'='*80}")
                    print(f"📈 Step {global_step:5d} Evaluation")
                    print(f"{'='*80}")
                    print(f"  Train Loss: {loss.item():.4f}")
                    print(f"  Val Loss:   {val_loss:.4f}")
                    print(f"  LR:         {current_lr:.2e}")
                    print(f"  Samples:    {total_samples:,}")
                    print(f"{'='*80}\n")
                    
                    # Save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        
                        print(f"💾 New best model! Val loss: {val_loss:.4f}")
                        
                        # 這裡已經有 unwrapped_model 了
                        if accelerator is not None:
                            torch.save(unwrapped_model.state_dict(), 'shakespeare_best.pt')
                        else:
                            torch.save(model.state_dict(), 'shakespeare_best.pt')
                
                # 3. 再次同步，確保 Rank 1 等待 Rank 0 評估完成後才一起進入下一個 batch
                if accelerator is not None:
                    accelerator.wait_for_everyone()
                
                # 確保切回訓練模式
                model.train()
                # --- CRITICAL FIX END ---
        
        # Epoch summary
        avg_loss = epoch_loss / len(train_loader)
        epoch_time = time.time() - epoch_start
        samples_per_sec = total_samples / epoch_time
        
        # Final validation for this epoch
        # Add synchronization barrier before epoch-end evaluation
        if accelerator is not None:
            accelerator.wait_for_everyone()
        
        if accelerator is None or accelerator.is_main_process:
            # Unwrap model for evaluation to avoid DDP deadlock
            if accelerator is not None:
                unwrapped_model = accelerator.unwrap_model(model)
            else:
                unwrapped_model = model
            
            val_loss = evaluate(unwrapped_model, val_loader, device, eval_iters)
            epoch_data['val_loss'] = val_loss
            epoch_data['avg_train_loss'] = avg_loss
            epoch_data['time'] = epoch_time
            epoch_data['samples_per_sec'] = samples_per_sec
            
            training_history['epochs'].append(epoch_data)
            
            # Update epoch progress bar with comprehensive metrics
            if TQDM_AVAILABLE:
                epoch_pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'val': f'{val_loss:.4f}',
                    'time': f'{epoch_time:.1f}s',
                    'samp/s': f'{samples_per_sec:.0f}'
                })
            
            print(f"\n{'='*80}")
            print(f"✅ Epoch {epoch + 1}/{epochs} Complete")
            print(f"{'='*80}")
            print(f"  Avg Loss:      {avg_loss:.4f}")
            print(f"  Val Loss:      {val_loss:.4f}")
            print(f"  Best Val Loss: {best_val_loss:.4f}")
            print(f"  Time:          {epoch_time:.1f}s")
            print(f"  Throughput:    {samples_per_sec:.0f} samples/sec")
            print(f"{'='*80}\n")
            
            # Generate sample
            print("-" * 80)
            print("📝 Sample Generation:")
            print("-" * 80)
            sample_text = generate_sample(model, dataset, device, max_tokens=200)
            print(sample_text)
            print("-" * 80 + "\n")
    
    # Save training history
    if accelerator is None or accelerator.is_main_process:
        import json
        
        # Save as JSON
        with open('training_history.json', 'w') as f:
            json.dump(training_history, f, indent=2)
        
        # Save as CSV for easy analysis
        import csv
        
        # Epoch-level CSV
        with open('training_epochs.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'avg_train_loss', 'val_loss', 'time', 'samples_per_sec'])
            for ep in training_history['epochs']:
                writer.writerow([
                    ep['epoch'],
                    ep['avg_train_loss'],
                    ep['val_loss'],
                    ep['time'],
                    ep['samples_per_sec']
                ])
        
        # Step-level CSV
        if training_history['steps']:
            with open('training_steps.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['step', 'epoch', 'train_loss', 'val_loss', 'lr', 'samples'])
                for step in training_history['steps']:
                    writer.writerow([
                        step['step'],
                        step['epoch'],
                        step['train_loss'],
                        step['val_loss'],
                        step['lr'],
                        step['samples']
                    ])
        
        print(f"\n{'='*80}")
        print(f"🎉 Training Complete!")
        print(f"{'='*80}")
        print(f"  Best Val Loss: {best_val_loss:.4f}")
        print(f"  Total Steps:   {global_step:,}")
        print(f"  Total Samples: {total_samples:,}")
        print(f"\n📊 Data saved:")
        print(f"  - training_history.json")
        print(f"  - training_epochs.csv")
        print(f"  - training_steps.csv")
        print(f"  - shakespeare_best.pt")
        print(f"{'='*80}\n")
    
    return model


# ============================================================================
# Part 4: Main Execution
# ============================================================================

def main():
    # Configuration
    BLOCK_SIZE = 256
    BATCH_SIZE = 48            # Per-GPU batch size (effective = 48 x num_GPUs = 96 total)
    EPOCHS = 5                 # Reduced from 10 to prevent overfitting
    LR = 1e-4                  # Reduced from 3e-4 to slow down learning
    DROPOUT = 0.2              # Added dropout to prevent overfitting
    WEIGHT_DECAY = 0.1         # Increased from 0.01 for stronger regularization
    MIMO_RANK = 4
    USE_MULTI_GPU = True       # Set to False to force single-GPU mode
    USE_COMPILE = False        # ⚠️ WARNING: Conflicts with Triton! Auto-disabled if Triton available
    
    # 🔴 CRITICAL FIX: Force disable Triton in multi-GPU mode
    # This prevents zombie Triton kernels in DDP subprocesses
    if USE_MULTI_GPU:
        global TRITON_AVAILABLE
        TRITON_AVAILABLE = False
        print("🛡️  Safety: TRITON_AVAILABLE forcefully set to False for Multi-GPU stability.")
    
    # Note: Device is automatically handled by Accelerate
    # Don't manually set device when using multi-GPU
    
    print("=" * 80)
    print("🚀 Mamba-3 Shakespeare Training")
    print("=" * 80)
    print(f"  Batch size per GPU: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning Rate: {LR}")
    print(f"  Dropout: {DROPOUT}")
    print(f"  Weight Decay: {WEIGHT_DECAY}")
    print(f"  MIMO Rank: {MIMO_RANK}")
    print(f"  Multi-GPU: {USE_MULTI_GPU and ACCELERATE_AVAILABLE}")
    print(f"  Triton: {'Disabled (Multi-GPU)' if not TRITON_AVAILABLE else 'Enabled'}")
    print(f"  torch.compile: {USE_COMPILE}")
    print("=" * 80 + "\n")
    
    # Load data
    text = get_shakespeare_data()
    
    # Train/val split
    split_idx = int(0.9 * len(text))
    train_text = text[:split_idx]
    val_text = text[split_idx:]
    
    # Create datasets
    train_dataset = CharDataset(train_text, BLOCK_SIZE)
    val_dataset = CharDataset(val_text, BLOCK_SIZE)
    
    # --- CRITICAL FIX START: 修正驗證集編碼 ---
    # 1. 強制讓驗證集使用訓練集的字典
    val_dataset.stoi = train_dataset.stoi
    val_dataset.itos = train_dataset.itos
    val_dataset.vocab_size = train_dataset.vocab_size
    
    # 2. 【重要】必須用訓練集的 stoi 重新編碼驗證集資料！
    # 原本的 val_dataset.data 是用錯誤的字典編碼的，必須覆蓋掉
    print("🔄 Re-encoding validation data with training vocabulary...")
    val_dataset.data = torch.tensor(
        [train_dataset.stoi.get(ch, 0) for ch in val_text],  # 使用 get 避免 key error
        dtype=torch.long
    )
    print(f"✅ Validation dataset re-encoded: {len(val_dataset.data)} tokens")
    # --- CRITICAL FIX END ---
    
    # Create dataloaders
    # CRITICAL: num_workers=0 to avoid NCCL conflicts in multi-GPU training
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,  # Accelerator will handle distributed sampling
        num_workers=0,  # Must be 0 for DDP to avoid zombie processes
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,  # Must be 0 for DDP to avoid zombie processes
        pin_memory=True
    )
    
    # Create model
    model = create_mamba3_tiny(
        vocab_size=train_dataset.vocab_size,
        mimo_rank=MIMO_RANK,
        dropout=DROPOUT  # Pass dropout parameter
    )
    
    print(f"\n📦 Model Configuration")
    print(f"{'='*80}")
    print(f"  MIMO Rank: {MIMO_RANK}")
    print(f"  Vocabulary size: {train_dataset.vocab_size}")
    print(f"  Model parameters: {model.get_num_params() / 1e6:.2f}M")
    
    # Optional: torch.compile for acceleration (PyTorch 2.0+)
    # IMPORTANT: Disable if Triton is available to avoid conflicts
    if USE_COMPILE and TRITON_AVAILABLE:
        print(f"\n⚠️  torch.compile disabled: conflicts with Triton kernels")
        print(f"   Using Triton acceleration instead (faster on T4 GPUs)")
        USE_COMPILE = False
    
    if USE_COMPILE:
        try:
            print(f"\n⚡ Compiling model with torch.compile...")
            model = torch.compile(model, mode='reduce-overhead')
            print(f"✅ Model compiled successfully!")
        except Exception as e:
            print(f"⚠️  torch.compile failed: {e}")
            print(f"   Continuing without compilation...")
    
    print(f"{'='*80}\n")
    
    # Train (device is handled by Accelerate)
    model = train_shakespeare(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        dataset=train_dataset,
        epochs=EPOCHS,
        lr=LR,
        eval_interval=100,
        eval_iters=20,
        use_accelerate=USE_MULTI_GPU,
    )
    
    # Final generation test (only on main process)
    # In multi-GPU mode, this will only run on GPU 0
    print("\n" + "=" * 80)
    print("🎭 Final Generation Test")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    prompts = [
        "ROMEO:",
        "First Citizen:",
        "KING HENRY:",
    ]
    
    for prompt in prompts:
        print(f"\n📝 Prompt: '{prompt}'")
        print("-" * 80)
        text = generate_sample(model, train_dataset, device, prompt=prompt, max_tokens=300)
        print(text)
        print("-" * 80)


# ============================================================================
# Part 5: Dual GPU Launcher (Kaggle/Colab Fix)
# ============================================================================

def main_launcher():
    """
    Wrapper for notebook_launcher
    This function will be called by each GPU process
    """
    main()


def launch_training(num_gpus=2):
    """
    Launch multi-GPU training using notebook_launcher
    
    Args:
        num_gpus: Number of GPUs to use (2 for Kaggle T4 x2)
    
    CRITICAL: Do NOT call torch.cuda APIs here!
    Any torch.cuda call initializes CUDA in the main process,
    which will crash forked subprocesses.
    """
    
    # CRITICAL: Set MASTER_PORT to avoid port conflicts from zombie processes
    # Use a non-default port to avoid conflicts with previous crashed runs
    # Note: DISABLE_TRITON is already set at module level based on USE_MULTI_GPU flag
    import os
    os.environ['MASTER_PORT'] = '29500'
    
    print("\n" + "=" * 80)
    print("� 啟動雙卡並行訓練 (Launching Multi-GPU Training)")
    print("=" * 80)
    print(f"Target GPUs: {num_gpus}")
    print(f"Mixed Precision: FP16 (enabled via Accelerate)")
    print(f"Triton: Disabled (incompatible with DDP fork)")
    print("\n⚠️  IMPORTANT: If you see CUDA errors, restart the Notebook runtime!")
    print("   Kaggle: Run → Restart Session")
    print("=" * 80 + "\n")
    
    try:
        from accelerate import notebook_launcher
        
        # Launch with notebook_launcher for proper DDP initialization
        notebook_launcher(main_launcher, args=(), num_processes=num_gpus)
        
    except ImportError:
        print("⚠️  notebook_launcher not available, running in single-process mode")
        main()


if __name__ == "__main__":
    # ⚠️ IMPORTANT: If you see "Cannot re-initialize CUDA" errors:
    #    1. Restart Notebook runtime (Run → Restart Session)
    #    2. Re-run all cells
    #    3. Then run this cell again
    
    # 默認使用雙卡訓練
    # 如果只想用單卡，改為 main()
    launch_training(num_gpus=2)

