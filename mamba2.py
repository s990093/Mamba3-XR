import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================
# 1. Mamba-2 Config
# ==========================================
class Mamba2Config:
    def __init__(
        self, 
        d_model=256, 
        d_state=64, 
        d_head=64, 
        n_groups=1, 
        expand=2,
        use_conv=False,
        d_conv=4,
        rms_norm_eps=1e-5,
        chunk_size=256,
        use_parallel_scan=True,
        
        # === Mamba-2 Initialization Hyperparameters ===
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        A_init_range=(1, 16),
    ):
        """
        Configuration for Mamba-2 Block (pure SISO SSD model).
        """
        self.d_model = d_model
        self.d_state = d_state
        self.d_head = d_head
        self.expand = expand
        
        self.d_inner = int(expand * d_model)
        assert self.d_inner % d_head == 0, "d_inner must be divisible by d_head"
        self.n_heads = self.d_inner // d_head
        assert self.n_heads % n_groups == 0, f"n_heads ({self.n_heads}) must be divisible by n_groups ({n_groups})"
        self.n_groups = n_groups
        
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

    def __repr__(self):
        return (f"Mamba2Config(d_model={self.d_model}, n_heads={self.n_heads}, "
                f"n_groups={self.n_groups}, use_conv={self.use_conv}, "
                f"parallel={self.use_parallel_scan})")


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
# 3. Mamba-2 Block
# ==========================================
class Mamba2Block(nn.Module):
    """
    純粹的 Mamba-2 Block (基於 SSD 架構)
    作為與 Mamba-3 對比的 Baseline，剝離了：
    1. MIMO (R=1)
    2. Data-Dependent RoPE (純實數)
    3. Generalized Trapezoidal Rule (回歸 Euler)
    """
    def __init__(self, config: Mamba2Config):
        super().__init__()
        self.config = config

        d_in = config.d_model
        H = config.n_heads
        G = config.n_groups
        P = config.d_head
        N = config.d_state

        self.d_inner = H * P
        self.ratio = H // G

        # === 1. Mamba-2 投影 (沒有 lambda) ===
        self.dim_z = H * P
        self.dim_x = H * P
        self.dim_B = G * N
        self.dim_C = G * N
        self.dim_dt = G

        d_proj_total = self.dim_z + self.dim_x + self.dim_B + self.dim_C + self.dim_dt
        self.in_proj = nn.Linear(d_in, d_proj_total, bias=True)

        if config.use_conv:
            self.conv = nn.Conv1d(
                self.dim_x, self.dim_x, bias=True,
                kernel_size=config.d_conv, groups=self.dim_x, padding=config.d_conv - 1
            )

        # === 2. Mamba-2 參數 (沒有 theta_log, 沒有 MIMO) ===
        A_min, A_max = config.A_init_range
        self.A_log = nn.Parameter(torch.empty(G).uniform_(A_min, A_max).log())
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(H))

        self.norm_B = RMSNorm(N, eps=config.rms_norm_eps)
        self.norm_C = RMSNorm(N, eps=config.rms_norm_eps)

        self.out_proj = nn.Linear(self.d_inner, d_in, bias=False)
        self.act = nn.SiLU()

        # === dt_bias Initialization ===
        with torch.no_grad():
            dt = torch.exp(
                torch.rand(G) * (math.log(config.dt_max) - math.log(config.dt_min))
                + math.log(config.dt_min)
            )
            dt = torch.clamp(dt, min=config.dt_init_floor)
            inv_dt = dt + torch.log(-torch.expm1(-dt))

            dt_start = self.dim_z + self.dim_x + self.dim_B + self.dim_C
            dt_end = dt_start + self.dim_dt
            self.in_proj.bias[dt_start:dt_end].copy_(inv_dt)

    def segsum(self, x):
        T = x.size(-1)
        x_cumsum = torch.cumsum(x, dim=-1)
        x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=0)
        x_segsum = x_segsum.masked_fill(~mask, -float('inf'))
        return x_segsum

    def chunk_parallel_scan(self, u, dt, A, C, chunk_size=128):
        # 沿用原有的 chunk_parallel_scan，為了對齊維度，C 輸入 (B, L, H, N, 1)，最後再將 y squeeze 掉
        B, L, H, N, P = u.shape
        R = C.shape[-1]
        device = u.device
        input_dtype = u.dtype
        
        L_orig = L
        if L % chunk_size != 0:
            pad_len = chunk_size - (L % chunk_size)
            u = F.pad(u, (0, 0, 0, 0, 0, 0, 0, pad_len))
            dt = F.pad(dt, (0, 0, 0, pad_len))
            C = F.pad(C, (0, 0, 0, 0, 0, 0, 0, pad_len))
            L = L + pad_len
        
        log_alpha = dt * A.view(1, 1, H)
        
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
        h_intra_flat = torch.matmul(L_mask_flat, u_chunk_flat)
        h_intra = h_intra_flat.reshape(B, num_chunks, H, chunk_size, N, P).permute(0, 1, 3, 2, 4, 5)
        
        batch_dims = B * num_chunks * chunk_size * H
        h_trans = h_intra.permute(0, 1, 2, 3, 5, 4).reshape(batch_dims, P, N)
        c_for_mat = C_chunk.reshape(batch_dims, N, R)
        y_diag_flat = torch.matmul(h_trans, c_for_mat)
        y_diag = y_diag_flat.reshape(B, num_chunks, chunk_size, H, P, R)
        
        decay_chunk = torch.exp(torch.sum(log_alpha_chunk, dim=2))
        h_chunk_final = h_intra[:, :, -1]
        
        h_prev = torch.zeros(B, H, N, P, device=device, dtype=input_dtype)
        h_states_inter = []
        
        for c in range(num_chunks):
            h_states_inter.append(h_prev)
            decay = decay_chunk[:, c].view(B, H, 1, 1)
            contrib = h_chunk_final[:, c]
            h_prev = h_prev * decay + contrib
        
        h_states_inter = torch.stack(h_states_inter, dim=1)
        
        log_alpha_cumsum = torch.cumsum(log_alpha_chunk, dim=2)
        decay_intra = torch.exp(log_alpha_cumsum)
        
        h_effect = h_states_inter.unsqueeze(2) * decay_intra.unsqueeze(-1).unsqueeze(-1)
        
        h_eff_trans = h_effect.permute(0, 1, 2, 3, 5, 4).reshape(batch_dims, P, N)
        y_off_flat = torch.matmul(h_eff_trans, c_for_mat)
        y_off = y_off_flat.reshape(B, num_chunks, chunk_size, H, P, R)
        
        y_total = y_diag + y_off
        y_total = y_total.view(B, L, H, P, R)
        if L_orig < L:
            y_total = y_total[:, :L_orig]
        
        return y_total.to(input_dtype), h_prev.to(input_dtype)

    def forward(self, u, return_states=False, return_internals=False):
        B_sz, L, _ = u.shape
        H, G, P, N = self.config.n_heads, self.config.n_groups, self.config.d_head, self.config.d_state
        ratio = self.ratio

        # 1. Main Projection & Split
        projected = self.in_proj(u)
        split_sections = [self.dim_z, self.dim_x, self.dim_B, self.dim_C, self.dim_dt]
        z, x_prime, B_param, C_param, dt = torch.split(projected, split_sections, dim=-1)

        if self.config.use_conv:
            x_prime_conv = x_prime.transpose(1, 2)
            x_prime_conv = self.conv(x_prime_conv)
            x_prime_conv = x_prime_conv[:, :, :L]
            x_prime = x_prime_conv.transpose(1, 2)

        x_prime = x_prime.view(B_sz, L, H, P)

        dt = F.softplus(dt)
        min_dt, max_dt = self.config.dt_limit
        if self.config.dt_limit != (0.0, float("inf")):
            dt = dt.clamp(min=min_dt, max=max_dt)

        A = -torch.exp(self.A_log)

        def broadcast_group(tensor, target_shape_suffix):
            return tensor.repeat_interleave(ratio, dim=2)

        dt = broadcast_group(dt.unsqueeze(-1), (1,)).squeeze(-1) # (B, L, H)
        A_broadcast = A.repeat_interleave(ratio, dim=0) # (H)

        B_param = B_param.view(B_sz, L, G, N)
        C_param = C_param.view(B_sz, L, G, N)

        B_param = self.norm_B(B_param).view(B_sz, L, G, N)
        C_param = self.norm_C(C_param).view(B_sz, L, G, N)

        # 廣播到各個 Head (沒有 RoPE 旋轉)
        B_state = broadcast_group(B_param, (N,)) # (B, L, H, N)
        C_state = broadcast_group(C_param, (N,)) # (B, L, H, N)

        # 2. Mamba-2 Euler 離散化 (無梯形法則)
        input_signal = torch.einsum('blhn, blhp -> blhnp', B_state, x_prime)

        dt_view = dt.view(B_sz, L, H, 1, 1)
        u_ssm = dt_view * input_signal # (B, L, H, N, P)

        # 3. SSD 並行掃描
        if self.config.use_parallel_scan:
            # 增加一個 R=1 維度以適應 chunk_parallel_scan
            C_for_scan = C_state.unsqueeze(-1) # (B, L, H, N, 1)
            y_stack, h_state = self.chunk_parallel_scan(u_ssm, dt, A_broadcast, C_for_scan,
                                                        chunk_size=self.config.chunk_size)
            y_stack = y_stack.squeeze(-1) # (B, L, H, P)
        else:
            h_state = torch.zeros(B_sz, H, N, P, device=u.device)
            y_stack_list = []
            
            alpha_view = torch.exp(torch.einsum('blh, h -> blh', dt, A_broadcast)).view(B_sz, L, H, 1, 1)
            
            for t in range(L):
                h_state = h_state * alpha_view[:, t] + u_ssm[:, t]
                y_t = torch.einsum('bhnp, bhn -> bhp', h_state, C_state[:, t])
                y_stack_list.append(y_t)
                
            y_stack = torch.stack(y_stack_list, dim=1)

        # 4. 輸出處理
        y = y_stack.view(B_sz, L, H * P)

        # Skip Connection & Gate
        x_prime_view = x_prime.reshape(B_sz, L, H * P)
        y = y + x_prime_view * self.D.repeat_interleave(P, dim=0)
        
        z_act = self.act(z)
        y = y * z_act

        if return_internals:
            return {
                'h_states': h_state,
                'inputs': u,
            }

        if return_states:
             return self.out_proj(y), h_state

        return self.out_proj(y)
