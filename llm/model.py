import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==========================================
# 1. Mamba-3 Config
# ==========================================
class Mamba3Config:
    def __init__(
        self, 
        d_model=256, 
        d_state=64, 
        d_head=64, 
        n_groups=1, 
        mimo_rank=4,
        expand=2,        # [New] Expansion factor
        use_conv=False,  # [New] Optional Convolution toggle
        d_conv=4,        # [New] Convolution kernel size
        rms_norm_eps=1e-5, # [New] Stability epsilon
        chunk_size=256,   # [New] Chunk size for chunk-wise scan (預設 256，平衡精度與效率)
        use_parallel_scan=True, # [New] Enable Parallel Scan (使用 chunk-wise 實現)
        
        # === Mamba-2 Initialization Hyperparameters (移植自官方實現) ===
        dt_min=0.001,       # dt 最小值 (決定能夠捕捉的最小時間細粒度)
        dt_max=0.1,         # dt 最大值 (決定能夠捕捉的最大時間細粒度)
        dt_init_floor=1e-4, # 防止 dt 過小的下限
        dt_limit=(0.0, float("inf")), # Forward 時限制 dt 範圍 (數值穩定性)
        A_init_range=(1, 16), # A 初始化範圍 (決定 Decay Rate 的分佈)
    ):
        """
        Configuration for Mamba-3 Block with Grouped SSM.
        """
        self.d_model = d_model
        self.d_state = d_state
        self.d_head = d_head
        self.expand = expand
        
        # Calculate d_inner and n_heads
        # Default: d_inner = expand * d_model
        self.d_inner = int(expand * d_model)
        
        assert self.d_inner % d_head == 0, "d_inner must be divisible by d_head"
        self.n_heads = self.d_inner // d_head
        
        # Group Check
        assert self.n_heads % n_groups == 0, f"n_heads ({self.n_heads}) must be divisible by n_groups ({n_groups})"
        self.n_groups = n_groups
        
        # Mamba-3 Specifics
        self.mimo_rank = mimo_rank
        
        # Architecture Toggles
        self.use_conv = use_conv
        self.d_conv = d_conv
        self.rms_norm_eps = rms_norm_eps
        self.chunk_size = chunk_size
        self.use_parallel_scan = use_parallel_scan
        
        # Mamba-2 Initialization Parameters
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init_floor = dt_init_floor
        self.dt_limit = dt_limit
        self.A_init_range = A_init_range

    def __repr__(self):
        return (f"Mamba3Config(d_model={self.d_model}, n_heads={self.n_heads}, "
                f"n_groups={self.n_groups}, mimo_rank={self.mimo_rank}, "
                f"use_conv={self.use_conv}, parallel={self.use_parallel_scan})")

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
# 2. Mamba-3 Block
# ==========================================
class Mamba3Block(nn.Module):
    """
    Mamba-3 Block with Grouped SSM Support.
    Replaces original Mamba3Block.
    
    Features:
    - Grouped SSM: Share dynamics (A, B, C) across heads.
    - Generalized Trapezoidal Rule.
    - Data-Dependent RoPE.
    - MIMO Projections.
    - [New] Optional 1D Convolution (Mamba-2 style).
    - [New] Parallel Scan (Prefix Sum) for speed.
    """
    def __init__(self, config: Mamba3Config):
        super().__init__()
        self.config = config
        
        d_in = config.d_model
        H = config.n_heads
        G = config.n_groups
        P = config.d_head
        N = config.d_state
        R = config.mimo_rank

        self.d_inner = H * P
        self.ratio = H // G

        # === 1. Projections (Mixed Dimensions) ===
        self.dim_z = H * P
        self.dim_x = H * P
        self.dim_B = G * N * R
        self.dim_C = G * N * R
        self.dim_dt = G
        self.dim_lambda = G
        
        d_proj_total = self.dim_z + self.dim_x + self.dim_B + self.dim_C + self.dim_dt + self.dim_lambda
        self.in_proj = nn.Linear(d_in, d_proj_total, bias=True)

        # [New] Optional Convolution on x/z/states? 
        if config.use_conv:
            self.conv = nn.Conv1d(
                self.dim_x, 
                self.dim_x, 
                bias=True,
                kernel_size=config.d_conv, 
                groups=self.dim_x, 
                padding=config.d_conv - 1
            )
        
        # MIMO Projections
        self.x_up_proj = nn.Linear(P, P * R, bias=False)
        self.y_down_proj = nn.Linear(P * R, P, bias=False)

        # === 2. Parameters ===
        # A_log: Decay parameter (Mamba-2 技巧: log-space uniform)
        A_min, A_max = config.A_init_range
        self.A_log = nn.Parameter(torch.empty(G).uniform_(A_min, A_max).log())
        self.A_log._no_weight_decay = True
        
        self.theta_log = nn.Parameter(torch.randn(G, N // 2))
        self.D = nn.Parameter(torch.ones(H))

        # Biases & Norms
        self.norm_B = RMSNorm(N * R, eps=config.rms_norm_eps)
        self.norm_C = RMSNorm(N * R, eps=config.rms_norm_eps)
        self.bias_B = nn.Parameter(torch.zeros(G, N, R))
        self.bias_C = nn.Parameter(torch.zeros(G, N, R))

        self.out_proj = nn.Linear(self.d_inner, d_in, bias=False)
        self.act = nn.SiLU()
        
        # Initialization
        with torch.no_grad():
            rank_scale = 1.0 / math.sqrt(R) if R > 1 else 1.0
            nn.init.xavier_uniform_(self.x_up_proj.weight, gain=rank_scale)
            nn.init.xavier_uniform_(self.y_down_proj.weight, gain=rank_scale)
            self.bias_B.fill_(1.0)
            self.bias_C.fill_(1.0)
            
            # === dt_bias Initialization (Mamba-2 核心技巧) ===
            # 目標: 讓每個 Group 初始關注不同的時間尺度
            # 方法: Inverse Softplus - 確保 softplus(bias) 落在 [dt_min, dt_max]
            
            # 1. 在 log 空間隨機採樣 dt
            dt = torch.exp(
                torch.rand(G) * (math.log(config.dt_max) - math.log(config.dt_min))
                + math.log(config.dt_min)
            )
            dt = torch.clamp(dt, min=config.dt_init_floor)
            
            # 2. Inverse Softplus: softplus^{-1}(x) = x + log(1 - exp(-x))
            #    這樣 forward 做 softplus(bias) 時，剛好得到上面的 dt 值
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            
            # 3. dt 位於 in_proj bias 的對應位置
            dt_start = self.dim_z + self.dim_x + self.dim_B + self.dim_C
            dt_end = dt_start + self.dim_dt
            self.in_proj.bias[dt_start:dt_end].copy_(inv_dt)
            
            # === lambda_bias Initialization (梯形法則混合係數) ===
            # 初期偏向 Euler (lambda ≈ 0)，讓訓練更穩定
            # sigmoid(-3.0) ≈ 0.047
            lambda_start = dt_end
            self.in_proj.bias[lambda_start:].fill_(-3.0)
    
    def apply_rope(self, x, angles):
        """
        Input:
            x: (B, L, H, N, R)
            angles: (B, L, H, N/2)
        Output:
            Rotated x with same shape
        """
        N_half = angles.shape[-1]
        x_reshaped = x.view(*x.shape[:-2], N_half, 2, x.shape[-1])
        real_part = x_reshaped[..., 0, :] # (..., N/2, R)
        imag_part = x_reshaped[..., 1, :] # (..., N/2, R)

        w_cos = torch.cos(angles).unsqueeze(-1)
        w_sin = torch.sin(angles).unsqueeze(-1)

        real_rot = real_part * w_cos - imag_part * w_sin
        imag_rot = real_part * w_sin + imag_part * w_cos
        
        # Re-stack and flatten
        x_out = torch.stack([real_rot, imag_rot], dim=-2)
        return x_out.flatten(-3, -2)

    def segsum(self, x):
        """
        穩定的分段累積求和 (Stable Segment Sum)。
        用於計算 Chunk 內的 decay 矩陣，避免直接 exp(cumsum) 導致的溢位。
        
        Args:
            x: (B, C, H, L) - log decay values within chunks
        
        Returns:
            (B, C, H, L, L) - pairwise decay matrix (causal masked)
        """
        T = x.size(-1)
        # x_cumsum: (B, C, H, L)
        x_cumsum = torch.cumsum(x, dim=-1)
        
        # Broadcasting: x_cumsum[:, :, :, i, None] - x_cumsum[:, :, :, None, j]
        # 計算每對位置之間的相對累積 (i to j)
        x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
        
        # Causal masking: 只保留下三角 (t_out >= t_in)
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=0)
        x_segsum = x_segsum.masked_fill(~mask, -float('inf'))
        
        return x_segsum
    
    def chunk_parallel_scan(self, u, dt, A, C, chunk_size=128):
        """
        Mamba-2 SSD Algorithm (Chunk-wise Parallel Scan)。
        
        核心思想：
        1. Chunking: 將長序列切成小塊 (chunk_size)
        2. Intra-chunk: 在塊內用矩陣運算並行計算 (數值穩定)
        3. Inter-chunk: 塊間用 sequential scan (步數少，穩定)
        4. Merge: 合併塊內和塊間貢獻
        
        Args:
            u: Input signal (B, L, H, N, P)
            dt: Delta time (B, L, H)
            A: Decay parameter (H)
            C: Output projection (B, L, H, N, R)
            chunk_size: 分塊大小 (預設 128)
        
        Returns:
            y: Output (B, L, H, P, R)
            final_state: (B, H, N, P)
        """
        B, L, H, N, P = u.shape
        R = C.shape[-1]
        device = u.device
        input_dtype = u.dtype
        
        # 保持輸入精度（不強制轉 FP32）
        # 這樣 FP64 輸入可以達到機器精度 (~1e-11)
        # 如需穩定性，用戶可在外部控制 dtype
        
        # Padding to make L divisible by chunk_size
        L_orig = L
        if L % chunk_size != 0:
            pad_len = chunk_size - (L % chunk_size)
            u = F.pad(u, (0, 0, 0, 0, 0, 0, 0, pad_len))
            dt = F.pad(dt, (0, 0, 0, pad_len))
            C = F.pad(C, (0, 0, 0, 0, 0, 0, 0, pad_len))
            L = L + pad_len
        
        # 計算 log_alpha = dt * A
        # A: (H) -> broadcast to (B, L, H)
        log_alpha = dt * A.view(1, 1, H)  # (B, L, H)
        
        # === 1. Chunking (重排維度) ===
        num_chunks = L // chunk_size
        
        # Reshape: (B, L, ...) -> (B, C, chunk_size, ...)
        u_chunk = u.view(B, num_chunks, chunk_size, H, N, P)
        dt_chunk = dt.view(B, num_chunks, chunk_size, H)
        log_alpha_chunk = log_alpha.view(B, num_chunks, chunk_size, H)
        C_chunk = C.view(B, num_chunks, chunk_size, H, N, R)
        
        # === 2. Intra-Chunk Computation (塊內並行計算) ===
        # 2.1 計算塊內 decay mask: L_mask (L, L) for each chunk
        # log_alpha_chunk: (B, C, L, H) -> permute to (B, C, H, L)
        log_alpha_perm = log_alpha_chunk.permute(0, 1, 3, 2)  # (B, C, H, L)
        
        # L_mask: (B, C, H, L, L) - 塊內每個時間點對後續的影響
        L_mask = torch.exp(self.segsum(log_alpha_perm))  # (B, C, H, L, L)
        
        # 2.2 計算塊內「加權隱藏狀態」(局部，不包含歷史)
        # h_intra[t] = sum_{s<=t} L[t,s] * u[s]
        # L_mask: (B, C, H, L_out, L_in)
        # u_chunk: (B, C, L_in, H, N, P)
        # 需要對齊維度: einsum 'bchij, bcjhnp -> bcihnp'
        h_intra = torch.einsum('bchij, bcjhnp -> bcihnp', L_mask, u_chunk)  # (B, C, L, H, N, P)
        
        # 2.3 投影到輸出: y_diag = C^T * h_intra
        # h_intra: (B, C, L, H, N, P)
        # C_chunk: (B, C, L, H, N, R)
        y_diag = torch.einsum('bclhnp, bclhnr -> bclhpr', h_intra, C_chunk)  # (B, C, L, H, P, R)
        
        # === 3. Inter-Chunk Recurrence (塊間遞迴) ===
        # 3.1 每個 Chunk 的總衰減 (chunk decay)
        decay_chunk = torch.exp(torch.sum(log_alpha_chunk, dim=2))  # (B, C, H)
        
        # 3.2 每個 Chunk 的「貢獻」(最終狀態，假設初始為 0)
        # h_intra[-1] 就是從 chunk 開始（狀態為0）計算到結束的狀態
        h_chunk_final = h_intra[:, :, -1]  # (B, C, H, N, P)
        
        # 3.3 Sequential scan at chunk level (數量少，穩定)
        h_prev = torch.zeros(B, H, N, P, device=device, dtype=input_dtype)
        h_states_inter = []
        
        for c in range(num_chunks):
            # 記錄「進入」當前 chunk 的狀態
            h_states_inter.append(h_prev)
            
            # 更新: h_prev = h_prev * decay + contribution
            decay = decay_chunk[:, c].view(B, H, 1, 1)  # (B, H, 1, 1)
            contrib = h_chunk_final[:, c]  # (B, H, N, P)
            h_prev = h_prev * decay + contrib
        
        h_states_inter = torch.stack(h_states_inter, dim=1)  # (B, C, H, N, P)
        
        # === 4. Output Combination (合併塊內和塊間) ===
        # 歷史狀態需要乘上「塊內隨時間的衰減」
        # decay_intra[t] = exp(sum(log_alpha[0:t]))
        log_alpha_cumsum = torch.cumsum(log_alpha_chunk, dim=2)  # (B, C, L, H)
        decay_intra = torch.exp(log_alpha_cumsum)  # (B, C, L, H)
        
        # h_effect = h_prev * decay_intra
        # h_states_inter: (B, C, H, N, P)
        # decay_intra: (B, C, L, H)
        # 需要 broadcast: einsum 'bchnp, bclh -> bclhnp'
        h_effect = torch.einsum('bchnp, bclh -> bclhnp', h_states_inter, decay_intra)
        
        # y_off = C^T * h_effect
        y_off = torch.einsum('bclhnp, bclhnr -> bclhpr', h_effect, C_chunk)
        
        # 最終輸出
        y_total = y_diag + y_off  # (B, C, L, H, P, R)
        
        # Reshape 回原始序列
        y_total = y_total.view(B, L, H, P, R)
        
        # 移除 padding
        if L_orig < L:
            y_total = y_total[:, :L_orig]
        
        return y_total.to(input_dtype), h_prev.to(input_dtype)

    def forward(self, u, return_states=False, return_internals=False):
        B_sz, L, _ = u.shape
        H, G, P, N, R = self.config.n_heads, self.config.n_groups, self.config.d_head, self.config.d_state, self.config.mimo_rank
        ratio = self.ratio

        # 1. Main Projection & Split
        projected = self.in_proj(u)
        
        # Split 
        split_sections = [self.dim_z, self.dim_x, self.dim_B, self.dim_C, self.dim_dt, self.dim_lambda]
        z, x_prime, B_param, C_param, dt, lambda_param = torch.split(projected, split_sections, dim=-1)

        # [New] Optional Convolution (on x_prime)
        # x_prime: (B, L, H*P)
        if self.config.use_conv:
            # Transpose to (B, D, L) for Conv1d
            x_prime_conv = x_prime.transpose(1, 2)
            x_prime_conv = self.conv(x_prime_conv) # Causal padding applied
            # Slice off extra padding: [..., :L]
            x_prime_conv = x_prime_conv[:, :, :L]
            x_prime = x_prime_conv.transpose(1, 2)
        
        # View x_prime to (B, L, H, P)
        x_prime = x_prime.view(B_sz, L, H, P)
        
        # 2. Activation & Param Transformation
        dt = F.softplus(dt) # (B, L, G)
        
        # === dt_limit (數值穩定性 - Mamba-2 技巧) ===
        min_dt, max_dt = self.config.dt_limit
        if self.config.dt_limit != (0.0, float("inf")):
            dt = dt.clamp(min=min_dt, max=max_dt)
        
        A = -torch.exp(self.A_log) # (G) -> neg
        theta = torch.exp(self.theta_log) # (G, N/2)
        
        # 3. Broadcasting (Group -> Head)
        def broadcast_group(tensor, target_shape_suffix):
            # tensor: (B, L, G, ...)
            # target return: (B, L, H, ...)
            # repeat_interleave: G -> G*ratio = H
            return tensor.repeat_interleave(ratio, dim=2)
            
        dt = broadcast_group(dt.unsqueeze(-1), (1,)).squeeze(-1) # (B, L, H)
        A_broadcast = A.repeat_interleave(ratio, dim=0) # (H)
        theta_broadcast = theta.repeat_interleave(ratio, dim=0) # (H, N/2)
        
        # 4. RoPE
        # Dynamic Frequency: theta * dt
        # dt: (B, L, H)
        # theta: (H, N/2)
        dt_theta = torch.einsum('blh, hn -> blhn', dt, theta_broadcast)
        angles = torch.cumsum(dt_theta, dim=1) # (B, L, H, N/2)
        
        # Broadcast B, C params
        B_param = B_param.view(B_sz, L, G, N, R)
        C_param = C_param.view(B_sz, L, G, N, R)
        
        B_param = self.norm_B(B_param.flatten(-2, -1)).view(B_sz, L, G, N, R) + self.bias_B
        C_param = self.norm_C(C_param.flatten(-2, -1)).view(B_sz, L, G, N, R) + self.bias_C

        B_rotated_input = broadcast_group(B_param, (N, R))      # (B, L, H, N, R)
        C_rotated_input = broadcast_group(C_param, (N, R))      # (B, L, H, N, R)
        
        # Apply RoPE
        # Treat N as N/2 Complex. Input N is real. 
        # B/C are (N, R). N/2 pairs.
        B_rotated = self.apply_rope(B_rotated_input, angles)
        C_rotated = self.apply_rope(C_rotated_input, angles)
        
        # 5. MIMO Preparation
        # Up-Projection X' -> X (rank R)
        x = self.x_up_proj(x_prime).view(B_sz, L, H, P, R)
        
        # Input Signal (MIMO Contraction)
        # B: (..., N, R), x: (..., P, R) -> input: (..., N, P)
        input_signal = torch.einsum('blhnr, blhpr -> blhnp', B_rotated, x)

        # Trapezoidal Rule Terms
        lambda_view = F.sigmoid(broadcast_group(lambda_param.unsqueeze(-1), (1,)).squeeze(-1))
        
        # 為了計算正確的 u_ssm，我們還是需要 alpha 的值 (用於 term_prev)
        # 這裡算一次 alpha 給 trapezoidal 用 (不是給 scan 用)
        alpha_val = torch.exp(torch.einsum('blh, h -> blh', dt, A_broadcast)) 
        
        dt_view = dt.view(B_sz, L, H, 1, 1)
        lambda_view = lambda_view.view(B_sz, L, H, 1, 1)
        alpha_view = alpha_val.view(B_sz, L, H, 1, 1)

        # Term 1 (Curr)
        term_curr = lambda_view * dt_view * input_signal
        
        # Term 2 (Prev)
        input_signal_prev = torch.roll(input_signal, shifts=1, dims=1)
        input_signal_prev[:, 0] = 0 
        term_prev = (1 - lambda_view) * dt_view * alpha_view * input_signal_prev
        
        # Final Driver u_ssm
        u_ssm = term_curr + term_prev

        # === 6. SSM Scan ===
        if self.config.use_parallel_scan:
            # Chunk-wise Parallel Scan (Mamba-2 SSD 算法，數值穩定)
            y_stack, h_state = self.chunk_parallel_scan(u_ssm, dt, A_broadcast, C_rotated, 
                                                        chunk_size=self.config.chunk_size)
        else:
            # Sequential Scan (Legacy Loop)
            h_state = torch.zeros(B_sz, H, N, P, device=u.device)
            y_stack_list = []
            
            # Re-compute alpha for loop just to be consistent (or use alpha_val)
            # alpha_view is already (B, L, H, 1, 1)
            
            for t in range(L):
                # h_t = alpha_t * h_{t-1} + u_t
                h_state = h_state * alpha_view[:, t] + u_ssm[:, t]
                
                # Output Contraction
                # h_t: (B, H, N, P)
                # C_t: (B, H, N, R) -> from C_rotated[:, t] which is (B, H, N, R)
                y_t = torch.einsum('bhnp, bhnr -> bhpr', h_state, C_rotated[:, t])
                y_stack_list.append(y_t)
                
            y_stack = torch.stack(y_stack_list, dim=1)

        if return_internals:
             return {
                 'h_states': h_state,
                 'inputs':   u,
                 'angles':   angles,  # Added for diagnostics
                 'alpha':    alpha_val # Added for diagnostics
             }

        # === 7. Output Processing (MIMO Down-Projection) ===
        # y_stack shape is (B, L, H, P, R)
        
        # Flatten H, P, R -> H, P*R for Linear layer
        y_down = self.y_down_proj(y_stack.view(B_sz, L, H, P * R))
        y = y_down.view(B_sz, L, H * P)

        # Skip Connection (D term)
        x_prime_view = x_prime.reshape(B_sz, L, H * P)
        y = y + x_prime_view * self.D.repeat_interleave(P, dim=0) # D * x
        
        # Gate
        z_act = self.act(z) # z is (B, L, H*P)
        y = y * z_act

        if return_states:
             return self.out_proj(y), h_state

        return self.out_proj(y)
