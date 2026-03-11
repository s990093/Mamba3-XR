
import os
import glob
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import sentencepiece as spm
from accelerate import Accelerator

import torch
from accelerate import notebook_launcher

import multiprocessing


# 減少 CUDA 記憶體碎片，允許 allocator 使用 expandable segments
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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

    def forward(self, x):
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

    def forward(self, x):
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.dim_in1 * self.dim_in2)
        B_flat = x_flat.size(0)

        router_logits = self.router(x_flat)
        router_probs = torch.softmax(router_logits, dim=-1)

        if self.training:
            router_logits = router_logits + torch.randn_like(router_logits) * 0.1

        top_k_vals, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
        top_k_probs = torch.softmax(top_k_vals, dim=-1)

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
        # 🚀 效能優化區塊：Token-to-Expert Dispatching
        # ==========================================
        output = torch.zeros(B_flat, self.dim_out1, self.dim_out2, device=x.device, dtype=x.dtype)

        # 找出這個 Batch 中，真正有被分配到 Token 的專家 (排除掉閒置的專家)
        active_experts = torch.unique(top_k_indices)

        for expert_idx in active_experts:
            # 找出哪些 Token (batch_idx) 選擇了這位專家，以及是第幾個志願 (k_idx)
            token_mask = (top_k_indices == expert_idx)
            batch_idx, k_idx = torch.where(token_mask)

            # 如果剛好沒有 Token 選到，直接跳過
            if len(batch_idx) == 0:
                continue

            # 1. 整理 Token 與對應的機率 (Gather)
            tokens_e = x_sub[batch_idx]  # 形狀: (Tokens數量, dim_in1, dim_in2)
            probs_e = top_k_probs[batch_idx, k_idx].unsqueeze(1).unsqueeze(2).to(x.dtype) # 形狀: (Tokens數量, 1, 1)

            # 2. 取出該名專家的權重 (只發生一次，減少記憶體頻寬消耗)
            A_e = self.A_experts[expert_idx]  # 形狀: (dim_out1, dim_in1)
            B_e = self.B_experts[expert_idx]  # 形狀: (dim_out2, dim_in2)

            # 3. 透過 Einsum 一次性對這批 Token 進行 Kronecker 乘法
            # 這裡完美取代了原本的兩次 bmm 與 transpose 操作
            # 'oi' 是 A矩陣, 'nij' 是 Token, 'pj' 是 B矩陣 -> 輸出 'nop'
            Y_e = torch.einsum('oi, nij, pj -> nop', A_e, tokens_e, B_e)

            # 4. 乘上 Router 權重並加回原本的 output (Scatter Add)
            output[batch_idx] += Y_e * probs_e

            # 🚀 新增這行：強制立刻釋放計算圖的暫存節點，把 VRAM 還給 GPU
            del tokens_e, probs_e, A_e, B_e, Y_e
        # ==========================================

        output = output.reshape(*orig_shape[:-1], -1)
        output = output * self.scale + self.bias
        return output, aux_loss

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

    def apply_rope(self, x, angles):
        N_half = angles.shape[-1]
        x_reshaped = x.view(*x.shape[:-2], N_half, 2, x.shape[-1])
        real_part, imag_part = x_reshaped[..., 0, :], x_reshaped[..., 1, :]
        w_cos, w_sin = torch.cos(angles).unsqueeze(-1), torch.sin(angles).unsqueeze(-1)
        real_rot = real_part * w_cos - imag_part * w_sin
        imag_rot = real_part * w_sin + imag_part * w_cos
        return torch.stack([real_rot, imag_rot], dim=-2).flatten(-3, -2)

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
        if L % chunk_size != 0:
            pad_len = chunk_size - (L % chunk_size)
            u = F.pad(u, (0, 0, 0, 0, 0, 0, 0, pad_len))
            dt = F.pad(dt, (0, 0, 0, pad_len))
            C = F.pad(C, (0, 0, 0, 0, 0, 0, 0, pad_len))
            L = L + pad_len
        log_alpha = dt * A
        num_chunks = L // chunk_size
        u_chunk = u.view(B, num_chunks, chunk_size, H, N, P)
        dt_chunk = dt.view(B, num_chunks, chunk_size, H)
        log_alpha_chunk = log_alpha.view(B, num_chunks, chunk_size, H)
        C_chunk = C.view(B, num_chunks, chunk_size, H, N, R)

        log_alpha_perm = log_alpha_chunk.permute(0, 1, 3, 2)
        L_mask = torch.exp(self.segsum(log_alpha_perm))

        BCH = B * num_chunks * H
        L_mask_flat = L_mask.reshape(BCH, chunk_size, chunk_size)
        u_chunk_flat = u_chunk.permute(0, 1, 3, 2, 4, 5).reshape(BCH, chunk_size, N * P)
        h_intra = torch.matmul(L_mask_flat, u_chunk_flat).reshape(B, num_chunks, H, chunk_size, N, P).permute(0, 1, 3, 2, 4, 5)

        batch_dims = B * num_chunks * chunk_size * H
        h_trans = h_intra.permute(0, 1, 2, 3, 5, 4).reshape(batch_dims, P, N)
        c_for_mat = C_chunk.reshape(batch_dims, N, R)
        y_diag = torch.matmul(h_trans, c_for_mat).reshape(B, num_chunks, chunk_size, H, P, R)

        decay_chunk = torch.exp(torch.sum(log_alpha_chunk, dim=2))
        h_chunk_final = h_intra[:, :, -1]
        h_prev = torch.zeros(B, H, N, P, device=device, dtype=input_dtype)
        h_states_inter = []
        for c in range(num_chunks):
            h_states_inter.append(h_prev)
            h_prev = h_prev * decay_chunk[:, c].view(B, H, 1, 1) + h_chunk_final[:, c]
        h_states_inter = torch.stack(h_states_inter, dim=1)

        decay_intra = torch.exp(torch.cumsum(log_alpha_chunk, dim=2))
        c_decayed = C_chunk * decay_intra.unsqueeze(-1).unsqueeze(-1)

        # 2) 將中間 hidden 狀態轉成 [..., P, N]，讓 matmul 自動廣播
        # h_states_inter: [B, num_chunks, H, N, P] → [B, num_chunks, 1, H, P, N]
        h_inter_trans = h_states_inter.unsqueeze(2).transpose(-1, -2)

        # 3) 廣播 matmul：[..., P, N] @ [..., N, R] → [..., P, R]
        y_off = torch.matmul(h_inter_trans, c_decayed)
        y_total = y_diag + y_off

        y_total = y_total.view(B, L, H, P, R)
        if L_orig < L:
            y_total = y_total[:, :L_orig]
        return y_total.to(input_dtype), h_prev.to(input_dtype)

    def forward(self, u):
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
            x_up, aux_loss_up = self.x_up_proj(x_prime)
            x = x_up.view(B_sz, L, H, P, R)
        else:
            x = self.x_up_proj(x_prime).view(B_sz, L, H, P, R)
            aux_loss_up = 0.0

        input_signal = torch.einsum('blhnr, blhpr -> blhnp', B_rotated, x)
        lambda_view = F.sigmoid(broadcast_group(lambda_param.unsqueeze(-1), None).squeeze(-1)).view(B_sz, L, H, 1, 1)
        dt_view = dt.view(B_sz, L, H, 1, 1)
        alpha_view = torch.exp(dt * A_broadcast).view(B_sz, L, H, 1, 1)

        input_signal_prev = torch.roll(input_signal, shifts=1, dims=1)
        input_signal_prev[:, 0] = 0 
        u_ssm = lambda_view * dt_view * input_signal + (1 - lambda_view) * dt_view * alpha_view * input_signal_prev

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
            out_y, aux_loss_out = self.out_proj(y)
        else:
            out_y = self.out_proj(y)
            aux_loss_out = 0.0

        block_aux_loss = aux_loss_up + aux_loss_out
        return out_y, block_aux_loss

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
            ffn_out, ffn_loss = self.ffn(h_norm)
        else:
            ffn_out = self.ffn(h_norm)
            ffn_loss = 0.0
        x = x + ffn_out
        return x, ffn_loss

# ==========================================
# 5.5 Block Recurrent Mamba3 (Mamba-only, per-chunk state passing)
# ==========================================
class BlockRecurrentMamba3(nn.Module):
    def __init__(self, config: Mamba3Config, block_size: int = 64):
        super().__init__()
        self.config = config
        self.block_size = block_size
        self.num_layers = config.num_layers
        self.layers = nn.ModuleList([Mamba3Block(config) for _ in range(self.num_layers)])
        self.norms = nn.ModuleList([RMSNorm(config.d_model) for _ in range(self.num_layers)])
        self.initial_state_tokens = nn.Parameter(torch.zeros(self.num_layers, 1, 1, config.d_model))

    def forward(self, x, return_memory_bank=False):
        B, L, D = x.shape
        out_chunks = []
        memory_bank = []
        total_aux_loss = 0.0
        prev_state_tokens = [self.initial_state_tokens[i].expand(B, 1, D) for i in range(self.num_layers)]

        for i in range(0, L, self.block_size):
            chunk = x[:, i : i + self.block_size, :]
            chunk_out = chunk
            new_prev_state_tokens = []

            for j, layer in enumerate(self.layers):
                normed_chunk = self.norms[j](chunk_out)
                chunk_with_context = torch.cat([prev_state_tokens[j], normed_chunk], dim=1)

                layer_out_with_context, aux_loss = layer(chunk_with_context)

                if isinstance(aux_loss, torch.Tensor):
                    total_aux_loss = total_aux_loss + aux_loss

                layer_out = layer_out_with_context[:, 1:, :] 
                new_state = layer_out_with_context[:, -1:, :].detach() 
                new_prev_state_tokens.append(new_state)
                chunk_out = chunk_out + layer_out

            prev_state_tokens = new_prev_state_tokens
            out_chunks.append(chunk_out)
            memory_bank.append(new_prev_state_tokens[-1].squeeze(1))

        out = torch.cat(out_chunks, dim=1)
        if return_memory_bank:
            memory_bank_tensor = torch.stack(memory_bank, dim=1)
            return out, memory_bank_tensor, total_aux_loss
        return out, total_aux_loss

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

    def forward(self, x):
        h, loss_up = self.up_proj(x)
        h = self.act(h)
        y, loss_down = self.down_proj(h)
        return y, loss_up + loss_down

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

    def forward(self, x):
        # x: (B, L, d_model) - full sequence, no chunking
        total_aux_loss = 0.0

        for i, layer_dict in enumerate(self.layers):
            l_type = self.layer_types[i]

            if l_type == 'mamba':
                # Pre-norm + residual；啟用 checkpoint 以節省記憶體
                normed_x = layer_dict['norm'](x)

                # 注意：使用 non-reentrant 版本以符合 PyTorch 2.x 要求
                out, aux = checkpoint(layer_dict['block'], normed_x, use_reentrant=False)
                if isinstance(aux, torch.Tensor):
                    total_aux_loss = total_aux_loss + aux
                x = x + out

            elif l_type == 'transformer':
                # TransformerBlock: causal attn over full L, K-MoE FFN
                # Block 本身已處理殘差，這裡直接用 checkpoint 包起來
                out, aux = checkpoint(layer_dict['block'], x, use_reentrant=False)
                if isinstance(aux, torch.Tensor):
                    total_aux_loss = total_aux_loss + aux
                x = out

        return x, total_aux_loss

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

    def forward(self, input_ids, labels=None):
        x = self.embed(input_ids)
        x, aux_loss = self.backbone(x)
        x = self.norm(x)
        logits = self.head(x)

        if labels is not None:
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            if isinstance(aux_loss, torch.Tensor):
                aux_loss = aux_loss.mean()
            # Normalize aux_loss by total MoE layer count to prevent it dominating CE loss
            num_moe_layers = self.config.num_layers * (self.config.expand * 2 + 2)
            # raw_aux: 未縮水的 MoE routing 負載，用於觀察路由是否崩潰
            raw_aux = aux_loss.detach()
            aux_contrib = (0.01 / max(1, num_moe_layers)) * aux_loss
            loss = ce_loss + aux_contrib
            # record detached scalars for logging
            try:
                self._last_loss_terms = {
                    "ce_loss": ce_loss.detach(),
                    "aux_loss": aux_contrib.detach(),
                    "raw_aux": raw_aux.detach(),
                }
            except Exception:
                self._last_loss_terms = None
            # unsqueeze(0) → DataParallel gather 時能正確 cat 成 (num_gpus,) 再做 .mean()
            loss = loss.unsqueeze(0)
            # 訓練路徑回傳 loss 與未縮水 raw_aux，方便在訓練 loop 中直接記錄 Aux Loss
            return loss, raw_aux.unsqueeze(0)

        # 推論或評估時才回傳 logits
        return logits


# ==========================================
# 7. SentencePiece Tokenizer Training & Dataset
# ==========================================
def train_sentencepiece(subset_file_path, vocab_size=32000, model_prefix="spm_tokenizer", output_dir="."):
    """
    直接讀取準備好的 txt 檔案來訓練 SentencePiece 模型
    """
    # 確保輸出目錄存在 (例如 /kaggle/working)
    os.makedirs(output_dir, exist_ok=True)

    # 組合完整的模型路徑前綴
    full_model_prefix = os.path.join(output_dir, model_prefix)
    model_file = f"{full_model_prefix}.model"

    # 1. 如果模型已經訓練過並存在，直接讀取
    if os.path.exists(model_file):
        print(f"Loading existing SentencePiece model: {model_file}")
        return spm.SentencePieceProcessor(model_file=model_file)

    # 2. 確認你提供的 txt 檔案真的存在
    if not os.path.exists(subset_file_path):
        raise FileNotFoundError(f"找不到訓練檔！請確認路徑: {subset_file_path}")

    print(f"Training SentencePiece Tokenizer using file: {subset_file_path}")

    # 3. 直接開始訓練
    try:
        spm.SentencePieceTrainer.train(
            input=subset_file_path,          # 👈 直接吃你準備好的檔案
            model_prefix=full_model_prefix,  # 👈 存到指定路徑
            vocab_size=vocab_size,
            model_type="bpe",
            character_coverage=0.9995,
            input_sentence_size=50000,       # 避免記憶體爆掉，最多隨機抽 5 萬行來算 BPE
            shuffle_input_sentence=True,
            num_threads=4,
            pad_id=0, unk_id=1, bos_id=2, eos_id=3
        )
    except Exception as e:
        print(f"SPM Training Error: {e}")
        raise e

    print("Tokenizer training done! Files successfully saved to:")
    print(f"  👉 Model 檔: {model_file}")
    print(f"  👉 Vocab 檔: {full_model_prefix}.vocab")

    return spm.SentencePieceProcessor(model_file=model_file)



def load_sentencepiece(model_path):
    """直接讀取已經訓練好的 Tokenizer 模型"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到 Tokenizer 檔案，請檢查路徑：{model_path}")

    print(f"✅ 成功載入現有的 SentencePiece 模型: {model_path}")
    return spm.SentencePieceProcessor(model_file=model_path)


class FineWebIterableDataset(IterableDataset):
    def __init__(self, data_dir, tokenizer, seq_len):
        self.files = glob.glob(f"{data_dir}/*.txt")
        if not self.files: 
            self.files = glob.glob(f"{data_dir}/**/*.txt", recursive=True)
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            files_to_process = self.files
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            files_to_process = self.files[worker_id::num_workers]

        buffer = []
        for file in files_to_process:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                for line in lines:
                    line = line.strip()
                    if not line: continue
                    tokens = self.tokenizer.encode(line)
                    buffer.extend(tokens)

                    while len(buffer) >= self.seq_len + 1:
                        x = buffer[:self.seq_len]
                        y = buffer[1:self.seq_len+1]
                        yield torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
                        buffer = buffer[self.seq_len:]
            except Exception as e:
                print(f"Skipping file {file} due to {e}")

# ==========================================
# 8. Main Training Loop
# ==========================================
def main():
    # 啟用 Accelerate，統一管理多卡與自動混合精度
    accelerator = Accelerator()
    DATA_DIR = "/kaggle/input/datasets/nameonlu/fineweb-edu"
    TOKENIZER_PATH = "/kaggle/input/datasets/s990093/tokenizer/fineweb_tokenizer.model"
    # 先用較小 batch 與較少累積步數，降低顯存壓力
    BATCH_SIZE = 1
    GRADIENT_ACCUMULATION_STEPS = 16

    SEQ_LEN = 512
    STEPS = 10000
    LR = 3e-4 
    WARMUP = 1000
    VOCAB_SIZE = 32000

    SUBSET_FILE = "/kaggle/input/datasets/s990093/train-sentencepiece/spm_train_subset.txt"
    OUTPUT_DIR = "/kaggle/working" # 確保存下來的模型你可以下載

    print("Initializing Tokenizer...")
    # tokenizer = train_sentencepiece(
    #     subset_file_path=SUBSET_FILE, 
    #     vocab_size=VOCAB_SIZE,
    #     model_prefix="fineweb_tokenizer",
    #     output_dir=OUTPUT_DIR
    # )
    tokenizer = load_sentencepiece(TOKENIZER_PATH)
    actual_vocab_size = tokenizer.vocab_size()
    print(f"Actual Vocab Size: {actual_vocab_size}")


    config = Mamba3Config(
        # d_model 必須是 64 的倍數，MultiheadAttention 才能整除成 num_heads
        d_model=1024,
        d_state=64,
        expand=4,           
        num_layers=7,       # num_layers = num macro-blocks; each = 4 Mamba + 1 Transformer → 15 total blocks
        use_parallel_scan=True,
        chunk_size=64,
        use_kmoe=True,
        kmoe_num_experts=256,
        mimo_rank=4,
        kmoe_top_k=8
    )

    print("Initializing Model...")
    model = Mamba3LanguageModel(config, vocab_size=actual_vocab_size)

    # 🚀 新增這段：啟用計算圖編譯
    if hasattr(torch, "compile"):
        print("🔥 Compiling model with torch.compile for extra speed...")
        # 這裡不影響 state_dict，Checkpoint 依然能無縫讀取！
        model = torch.compile(model)

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
    print(f"  [Embedding]  d_model={config.d_model}, vocab={actual_vocab_size}")
    layer_num = 1
    for mb in range(num_mac):
        print(f"  --- Macro Block {mb+1}/{num_mac} ---")
        for k in range(mamba_ratio):
            print(f"  Layer {layer_num:02d} │ 🔵 Mamba3 KMoE  (d_model={config.d_model}, d_state={config.d_state}, K-MoE experts={config.kmoe_num_experts})")
            layer_num += 1
        print(f"  Layer {layer_num:02d} │ 🟠 Transformer  (Causal Attn {config.d_model//64} heads + K-MoE FFN experts={config.kmoe_num_experts})")
        layer_num += 1
    print(f"  [LM Head]    tied weights")

    device = accelerator.device
    num_gpus = accelerator.num_processes
    print(f"Using {num_gpus} processes with Accelerate (device: {device})")

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.1)

    def lr_lambda(current_step: int):
        if current_step < WARMUP:
            return float(current_step) / float(max(1, WARMUP))
        progress = float(current_step - WARMUP) / float(max(1, STEPS - WARMUP))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = LambdaLR(optimizer, lr_lambda)

    start_step = 0
    checkpoint_load_path = "/kaggle/input/datasets/s990093/checkpoint/mamba3_kmoe_checkpoint.pt"
    checkpoint_save_path = "mamba3_kmoe_checkpoint.pt"

    if os.path.exists(checkpoint_load_path):
        print(f"🔄 Found checkpoint in Kaggle Input: {checkpoint_load_path} — Resuming training...")
        ckpt = torch.load(checkpoint_load_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_step = ckpt["step"]
        print(
            f"✅ Resumed from Step {start_step} / {ckpt.get('total_steps', '?')} | "
            f"Last Loss: {ckpt.get('last_loss', 'N/A')}"
        )
    elif os.path.exists(checkpoint_save_path):
        print(f"🔄 Found checkpoint in local working dir: {checkpoint_save_path} — Resuming training...")
        ckpt = torch.load(checkpoint_save_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_step = ckpt["step"]
        print(
            f"✅ Resumed from Step {start_step} / {ckpt.get('total_steps', '?')} | "
            f"Last Loss: {ckpt.get('last_loss', 'N/A')}"
        )
    else:
        print("🆕 No checkpoint found — Starting from scratch.")

    dataset = FineWebIterableDataset(DATA_DIR, tokenizer, SEQ_LEN)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE,
        num_workers=2,          
        prefetch_factor=4,      
        pin_memory=True         
    ) 

    # 交給 Accelerate 將模型 / 優化器 / dataloader / scheduler 分配到多卡
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    model.train()
    data_iter = iter(dataloader)
    optimizer.zero_grad() 

    global_step = start_step  
    batch_idx = 0
    running_loss = 0.0
    running_aux = 0.0
    running_raw_aux = 0.0  

    LOG_FILE = "training_log.csv"
    if accelerator.is_main_process and not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            f.write("step,loss,aux_loss,raw_aux,ppl,lr,mem_gb_0,mem_gb_1,step_time\n")

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
                print("Dataset empty! Ensure txt files are in", DATA_DIR)
                break

        x, y = x.to(device), y.to(device)

        with accelerator.autocast():
            # 訓練時只需要 loss，避免把巨量 logits gather 回單卡
            loss, raw_aux = model(x, labels=y)
            # 轉成 CPU scalar 做 logging；多卡時各進程各自記錄本地平均

            # 從模型中取出剛才存下來的 scalar 字典
            if hasattr(model, "module") and getattr(model.module, "_last_loss_terms", None) is not None:
                loss_terms = model.module._last_loss_terms
            elif getattr(model, "_last_loss_terms", None) is not None:
                loss_terms = model._last_loss_terms
            else:
                loss_terms = {"ce_loss": 0.0, "aux_loss": 0.0, "raw_aux": raw_aux.detach()}

            loss_for_log = loss.detach().float().mean().item()
            def get_float(val):
                return val.float().mean().item() if isinstance(val, torch.Tensor) else val

            aux_for_log = get_float(loss_terms.get("aux_loss", 0.0))
            raw_aux_log = get_float(loss_terms.get("raw_aux", raw_aux.detach()))

            loss = loss / GRADIENT_ACCUMULATION_STEPS 

        accelerator.backward(loss)
        running_loss += loss_for_log  # track pre-scaled loss for accurate reporting
        running_aux += aux_for_log
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
            raw_aux_val = running_raw_aux / GRADIENT_ACCUMULATION_STEPS
            running_loss = 0.0
            running_aux = 0.0
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
                    f"Loss: {loss_val:.4f} (aux_scaled {aux_val:.4f}, aux_raw {raw_aux_val:.4f}) | "
                    f"PPL: {ppl:.2f} | LR: {lr_val:.2e} | "
                    f"Mem[GB] GPU={mem_gb_0:.2f}"
                )
                print(f"✅ {log_line}")

                # Save every step to CSV file
                with open(LOG_FILE, "a") as f:
                    f.write(f"{global_step},{loss_val:.6f},{aux_val:.6f},{raw_aux_val:.6f},{ppl:.4f},{lr_val:.2e},{mem_gb_0:.3f},{mem_gb_1:.3f},{step_time:.3f}\n")

                step_start_time = time.time()

                if global_step % 200 == 0:
                    unwrapped = accelerator.unwrap_model(model)
                    state_dict = unwrapped.state_dict()
                    torch.save({
                        'step': global_step,
                        'total_steps': STEPS,
                        'last_loss': round(loss_val, 4),
                        'model': state_dict,
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                    }, checkpoint_save_path)
                    print(f"💾 Checkpoint saved  →  Step {global_step}/{STEPS} | Loss: {loss_val:.4f}")

    if accelerator.is_main_process:
        print("🎉 Training Completed.")
        unwrapped = accelerator.unwrap_model(model)
        state_dict = unwrapped.state_dict()
        torch.save({
            'step': STEPS,
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, "mamba3_kmoe_final.pt")

if __name__ == "__main__":
    main()
