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
        num_layers=15,   # Number of layers in the model
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
        self.num_layers = num_layers
        
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
        # L_mask: (B, C, H, chunk_size, chunk_size)
        # u_chunk: (B, C, chunk_size, H, N, P)
        # Optimized: Replace einsum 'bchij, bcjhnp -> bcihnp' with matmul
        # We need to contract over j (2nd chunk_size dim in L_mask) and match with j in u_chunk
        # Reshape for batched matmul: merge B,C,H dimensions
        BCH = B * num_chunks * H
        L_mask_flat = L_mask.reshape(BCH, chunk_size, chunk_size)  # (B*C*H, chunk_size, chunk_size)
        u_chunk_flat = u_chunk.permute(0, 1, 3, 2, 4, 5).reshape(BCH, chunk_size, N * P)  # (B*C*H, chunk_size, N*P)
        h_intra_flat = torch.matmul(L_mask_flat, u_chunk_flat)  # (B*C*H, chunk_size, N*P)
        h_intra = h_intra_flat.reshape(B, num_chunks, H, chunk_size, N, P).permute(0, 1, 3, 2, 4, 5)  # (B, C, chunk_size, H, N, P)
        
        # 2.3 投影到輸出: y_diag = C^T * h_intra
        # h_intra: (B, C, L, H, N, P)
        # C_chunk: (B, C, L, H, N, R)
        # Optimized: Replace einsum with matmul
        batch_dims = B * num_chunks * chunk_size * H
        h_trans = h_intra.permute(0, 1, 2, 3, 5, 4).reshape(batch_dims, P, N)  # (batch, P, N)
        c_for_mat = C_chunk.reshape(batch_dims, N, R)  # (batch, N, R)
        y_diag_flat = torch.matmul(h_trans, c_for_mat)  # (batch, P, R)
        y_diag = y_diag_flat.reshape(B, num_chunks, chunk_size, H, P, R)
        
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
        
        # Memory Optimized output combination (Avoids 390MB intermediate h_effect tensor)
        # Instead of (h_states * decay) @ C, we do h_states @ (C * decay) 
        C_scaled = C_chunk * decay_intra.unsqueeze(-1).unsqueeze(-1)  # (B, C, L, H, N, R)
        
        BCH = B * num_chunks * H
        h_states_inter_trans = h_states_inter.permute(0, 1, 2, 4, 3).reshape(BCH, 1, P, N)
        C_scaled_view = C_scaled.permute(0, 1, 3, 2, 4, 5).reshape(BCH, chunk_size, N, R)
        
        y_off_flat = torch.matmul(h_states_inter_trans, C_scaled_view)  # (BCH, L, P, R)
        y_off = y_off_flat.reshape(B, num_chunks, H, chunk_size, P, R).permute(0, 1, 3, 2, 4, 5)
        
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


"""
Block-Recurrent Mamba-3 with Hybrid Cross-Attention (v2)
=========================================================
This module implements a truly causal, block-recurrent Mamba-3 architecture.

Two bugs from v1 are fixed here:

Bug 1: No true recurrence between chunks
-----------------------------------------
Original:
    chunk_out = self.mamba_block(chunk)   # no state passed →  each chunk starts fresh!

Fix:
    The Mamba3Block performs a chunk-wise parallel scan internally, but its `forward(x)`
    interface does not expose an initial state. Instead of modifying mamba3.py, we
    implement recurrence at the sequence level:
    - We run the full Mamba block on the chunk.
    - The final token's output embedding represents the running "state summary."
    - We prepend this summary token to the next chunk as a soft initial-state prompt,
      giving the model access to prior context without touching the original architecture.
    - This is the "prompt-state injection" technique — the model learns to use the
      prepended token as its memory across chunk boundaries.

Bug 2: Information leakage (future blocks attended by past tokens)
-------------------------------------------------------------------
Original:
    attn_out, _ = self.cross_attention(query=mamba_out, key=memory_bank, value=memory_bank)
    # query at block k can see blocks k+1 ... K (future!) in memory_bank

Fix:
    We construct a block-level causal mask over the memory bank.
    Block k can only attend to memory_bank slots 0 ... k-1 (strictly past).
"""




class BlockRecurrentMamba3(nn.Module):
    """
    Block-Recurrent Mamba-3 (Multi-Layer Version)
    
    Splits the input sequence into fixed-size blocks and processes them
    sequentially. True cross-block recurrence is achieved by prepending
    the previous block's final output token (state summary) to the current
    block input for each individual layer.
    
    Args:
        config: Mamba3Config
        block_size: Number of tokens per processing chunk (default 64)
        num_layers: Number of Mamba3Block layers to stack
    """
    def __init__(self, config: Mamba3Config, block_size: int = 64, num_layers: int = 15):
        super().__init__()
        self.config = config
        self.block_size = block_size
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList([Mamba3Block(config) for _ in range(num_layers)])
        
        # Import RMSNorm dynamically to ensure pre-norm is used
        
        self.norms = nn.ModuleList([RMSNorm(config.d_model) for _ in range(num_layers)])
        
        # Learned initial state token (acts as h_0, avoids the cold-start problem) per layer
        self.initial_state_tokens = nn.Parameter(torch.zeros(num_layers, 1, 1, config.d_model))
        
    def forward(self, x, return_memory_bank: bool = False):
        """
        Args:
            x: (B, L, d_model) — input sequence
            return_memory_bank: If True, also return the memory bank tensor
                                of shape (B, num_blocks, d_model)
        Returns:
            out: (B, L, d_model) — processed sequence
            memory_bank (optional): (B, num_blocks, d_model) — block summaries
        """
        B, L, D = x.shape
        out_chunks = []
        memory_bank = []
        
        # h_0 for each layer
        prev_state_tokens = [self.initial_state_tokens[i].expand(B, 1, D) for i in range(self.num_layers)]
        
        for i in range(0, L, self.block_size):
            chunk = x[:, i : i + self.block_size, :]  # (B, block_size, D)
            chunk_out = chunk
            new_prev_state_tokens = []
            
            # ─── Fix 1: True Recurrence Per Layer ─────────────────────────────
            for j, layer in enumerate(self.layers):
                normed_chunk = self.norms[j](chunk_out)
                
                # Prepend the previous block's final state token for this layer
                chunk_with_context = torch.cat([prev_state_tokens[j], normed_chunk], dim=1)  # (B, 1+block_size, D)
                
                # Run the Mamba block
                layer_out_with_context = layer(chunk_with_context)  # (B, 1+block_size, D)
                
                # Strip the prepended state token
                layer_out = layer_out_with_context[:, 1:, :]  # (B, block_size, D)
                
                # State summary for the next chunk (Detach to prevent backward pass across chunks)
                new_state = layer_out_with_context[:, -1:, :].detach()  # (B, 1, D)
                new_prev_state_tokens.append(new_state)
                
                # Residual connection
                chunk_out = chunk_out + layer_out
                
            prev_state_tokens = new_prev_state_tokens
            
            out_chunks.append(chunk_out)
            # Memory bank summary is the final layer's new_state
            memory_bank.append(new_prev_state_tokens[-1].squeeze(1))  # (B, D)
        
        out = torch.cat(out_chunks, dim=1)  # (B, L, D)
        
        if return_memory_bank:
            memory_bank_tensor = torch.stack(memory_bank, dim=1)  # (B, num_blocks, D)
            return out, memory_bank_tensor
        
        return out


class HybridBlockRecurrentMamba(nn.Module):
    """
    Hybrid Block-Recurrent Mamba-3 with Causally-Masked Cross-Attention

    Combines:
    - BlockRecurrentMamba3 for fast local processing + memory bank construction
    - A causally-masked Cross-Attention layer for precise long-range retrieval
      from the memory bank WITHOUT information leakage

    The attention mask ensures that token positions in block k can ONLY attend
    to memory bank slots 0 ... k-1 (strictly past blocks), not the present or
    future blocks.

    Args:
        config: Mamba3Config
        block_size: Number of tokens per Mamba processing chunk
        vocab_size: Vocabulary size for the embedding layer
        d_out: Output dimension (number of classes)
    """
    def __init__(self, config: Mamba3Config, block_size: int = 64, vocab_size: int = 10, d_out: int = 10):
        super().__init__()
        self.block_size = block_size
        self.embed = nn.Embedding(vocab_size, config.d_model)
        
        self.mamba_encoder = BlockRecurrentMamba3(config, block_size=block_size)
        
        # Cross-Attention: queries from Mamba output attend to the memory bank
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=4,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, d_out, bias=False)
        
    def _build_block_causal_mask(self, seq_len: int, num_blocks: int, device, dtype):
        """
        Build a block-level causal attention mask as an additive float mask.
        
        Token at position i (in block k = i // block_size) may attend to memory
        slot j only if j < k (strictly past blocks).
        
        For the very first block (k=0) all memory slots are masked → would give
        softmax(-inf,-inf,...) = NaN. We handle this by detecting fully-masked
        rows and letting them through with uniform zero contribution (via a separate
        fallback gate, see forward()).

        Returns:
            attn_mask: (seq_len, num_blocks) float tensor
                        0.0   → allowed to attend
                       -inf   → blocked
        """
        # Block index of each token
        token_blocks = torch.arange(seq_len, device=device) // self.block_size  # (L,)
        # Memory slot indices
        mem_slots    = torch.arange(num_blocks, device=device)                   # (K,)
        
        # allowed[i, j] = True if j < token_block[i]
        allowed = mem_slots.unsqueeze(0) < token_blocks.unsqueeze(1)  # (L, K)
        
        # Float mask: 0.0 where allowed, -inf where blocked
        attn_mask = torch.zeros(seq_len, num_blocks, device=device, dtype=dtype)
        attn_mask[~allowed] = float('-inf')

        return attn_mask, allowed  # also return allowed so forward() can gate the output
    
    def forward(self, x):
        """
        Args:
            x: (B, L) — integer token indices
        Returns:
            logits: (B, L, d_out)
        """
        h = self.embed(x)  # (B, L, d_model)
        B, L, D = h.shape
        
        # 1. Block-Recurrent Mamba encoding + memory bank construction
        mamba_out, memory_bank = self.mamba_encoder(h, return_memory_bank=True)
        
        num_blocks = memory_bank.size(1)  # K
        
        # ─── Fix 2: Block-level Causal Mask (float, -inf style) ──────────────
        attn_mask, allowed = self._build_block_causal_mask(L, num_blocks, x.device, h.dtype)
        
        # For rows that are ENTIRELY blocked (all-masked), softmax(-inf,...) → NaN.
        # We detect these rows and zero-out their cross-attention contribution.
        has_past = allowed.any(dim=1)  # (L,) True if the token has at least one past block
        
        # Replace fully-blocked rows with uniform 0 mask to avoid NaN in softmax.
        # Those rows will later be multiplied by 0 via the gate.
        safe_mask = attn_mask.clone()
        safe_mask[~has_past] = 0.0  # allow attention, but we'll gate result to 0
        
        attn_out, _ = self.cross_attention(
            query=mamba_out,     # (B, L, D)
            key=memory_bank,     # (B, K, D)
            value=memory_bank,   # (B, K, D)
            attn_mask=safe_mask  # (L, K) float additive mask
        )
        
        # Zero out the cross-attention contribution for positions with NO past blocks.
        # Shape: (1, L, 1) broadcast over (B, L, D)
        gate = has_past.float().view(1, L, 1)
        attn_out = attn_out * gate
        
        # 3. Residual + layer norm
        h_final = self.norm(mamba_out + attn_out)
        return self.head(h_final)  # (B, L, d_out)





import os
import glob
import math
import time
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from transformers import AutoTokenizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

# ==========================================
# Language Model Wrapper
# ==========================================
class Mamba3LanguageModel(nn.Module):
    def __init__(self, config: Mamba3Config, vocab_size: int, block_size: int = 64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, config.d_model)
        self.backbone = BlockRecurrentMamba3(config, block_size=block_size, num_layers=config.num_layers)
        self.norm = RMSNorm(config.d_model) # Matching the identical RMSNorm
        self.head = nn.Linear(config.d_model, vocab_size, bias=False)
        self.head.weight = self.embed.weight
        
    def forward(self, input_ids, labels=None):
        x = self.embed(input_ids)
        x = self.backbone(x) # (B, L, D)
        x = self.norm(x)
        logits = self.head(x)
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            
        return logits, loss

# ==========================================
# Dataset Class
# ==========================================
class FineWebIterableDataset(IterableDataset):
    def __init__(self, data_dir, tokenizer, seq_len):
        self.files = glob.glob(f"{data_dir}/*.txt")
        if not self.files: 
            # Fallback if recursive required
            self.files = glob.glob(f"{data_dir}/**/*.txt", recursive=True)
            
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        
    def __iter__(self):
        # --- 新增：取得目前 CPU Worker 的資訊 ---
        worker_info = get_worker_info()
        if worker_info is None:
            # 單執行緒：處理所有檔案
            files_to_process = self.files
        else:
            # 多執行緒：把檔案平均分給不同的 CPU 核心
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            # 譬如 worker 0 處理第 0, 2, 4 個檔案；worker 1 處理 1, 3, 5
            files_to_process = self.files[worker_id::num_workers]

        buffer = []
        for file in files_to_process:
            try:
                # 稍微優化讀取方式，一次讀多行減少 I/O
                with open(file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                for line in lines:
                    line = line.strip()
                    if not line: continue
                    # Encode 比較耗 CPU
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
# Training Loop (修復 OOM 與 AMP 升級版)
# ==========================================
def main():
    # ==========================================
    # Hyperparameters & Configurations
    # ==========================================
    # ⚠️ 130M 模型變大了，實體 Batch Size 必須調降以保護 Kaggle T4 的 16GB VRAM
    DATA_DIR = "/kaggle/input/datasets/nameonlu/fineweb-edu"
    BATCH_SIZE = 1                # 每張卡吃 1 筆，雙卡共吃 2 筆
    GRADIENT_ACCUMULATION_STEPS = 64  # 累積 64 次 (Global Batch = 128)。每次更新看 ~130K Tokens
    
    SEQ_LEN = 1024
    STEPS = 10000
    LR = 3e-4                     # 模型變大，最高學習率稍微調降回 3e-4 以求穩定
    WARMUP = 1000

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    
    # ==========================================
    # 💥 130M Mamba-3 架構核心配置
    # ==========================================
    config = Mamba3Config(
        d_model=768,        # 主維度放大 3 倍 (業界 130M 標準)
        d_state=64,         # 狀態維度維持 64 即可，避免矩陣計算過重
        expand=4,           # 內部擴展係數設為 2 (控制總參數在 1.3 億左右的關鍵)
        num_layers=15,      # 層數加深到 24 層
        use_parallel_scan=True,
        chunk_size=65       # 絕對不能改！防止 OOM 的救命設定
    )
    print("Initializing Model...")
    model = Mamba3LanguageModel(config, vocab_size=len(tokenizer), block_size=64)
    
    print("=== Model Architecture ===")
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"Using {num_gpus} GPUs with DataParallel")
        model = nn.DataParallel(model)
    else:
        print(f"Using Device: {device}")
    
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.1)
    
    # 💥 核心修復 2：初始化 AMP GradScaler 用於 fp16 混合精度
    scaler = torch.cuda.amp.GradScaler()
    
    # Warmup + Cosine Decay
    def lr_lambda(current_step: int):
        if current_step < WARMUP:
            return float(current_step) / float(max(1, WARMUP))
        progress = float(current_step - WARMUP) / float(max(1, STEPS - WARMUP))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    start_step = 0
    checkpoint_path = "mamba3_checkpoint.pt"
    if os.path.exists(checkpoint_path):
        print(f"Resuming from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        
        if num_gpus > 1 and "module." not in list(ckpt['model'].keys())[0]:
            new_ckpt = {"module."+k: v for k, v in ckpt['model'].items()}
            model.load_state_dict(new_ckpt)
        elif num_gpus <= 1 and "module." in list(ckpt['model'].keys())[0]:
            new_ckpt = {k.replace("module.", ""): v for k, v in ckpt['model'].items()}
            model.load_state_dict(new_ckpt)
        else:
            model.load_state_dict(ckpt['model'])
            
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_step = ckpt['step']
        
    dataset = FineWebIterableDataset(DATA_DIR, tokenizer, SEQ_LEN)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE,
        num_workers=2,          # Kaggle 給 4 核，我們留 2 核給系統，2 核專門做資料預處理
        prefetch_factor=4,      # 每個 Worker 預先準備好 4 個 batch 排隊等 GPU
        pin_memory=True         # 鎖定頁面記憶體，加速 CPU 傳輸到 GPU 的頻寬
    ) 
    
    model.train()
    data_iter = iter(dataloader)
    optimizer.zero_grad() 
    
    global_step = start_step  # 真正的權重更新次數
    batch_idx = 0             # 讀取資料的次數
    
    print("🚀 開始訓練！等待累積 128 個 Batch 進行第一次權重更新...")

    # 改用 global_step 來控制總訓練長度
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
        
        # FP16 前向傳播
        with torch.amp.autocast('cuda'):
            logits, loss = model(x, labels=y)
            if num_gpus > 1:
                loss = loss.mean()
            loss = loss / GRADIENT_ACCUMULATION_STEPS 
        
        # 累積梯度
        scaler.scale(loss).backward()
        batch_idx += 1
        
        # 💥 累積滿 128 個 Batch，才進行一次真正的「權重更新」
        if batch_idx % GRADIENT_ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            
            global_step += 1
            
            # 每 5 次「真正的更新」印出一次日誌
            if global_step % 5 == 0:
                # 把 Loss 乘回來還原真實數值
                loss_val = loss.item() * GRADIENT_ACCUMULATION_STEPS
                ppl = math.exp(min(loss_val, 20)) 
                lr_val = scheduler.get_last_lr()[0]
                print(f"Global Step {global_step}/{STEPS} | Loss: {loss_val:.4f} | PPL: {ppl:.2f} | LR: {lr_val:.2e}")
                
            # 每 1000 次「真正的更新」存檔一次
            if global_step > 0 and global_step % 1000 == 0:
                print(f"💾 Saving Checkpoint at Global Step {global_step}...")
                state_dict = model.module.state_dict() if num_gpus > 1 else model.state_dict()
                torch.save({
                    'step': global_step,
                    'model': state_dict,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }, checkpoint_path)
            
    print("🎉 Training Completed.")
    state_dict = model.module.state_dict() if num_gpus > 1 else model.state_dict()
    torch.save({
        'step': STEPS,
        'model': state_dict,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }, "mamba3_final.pt")

if __name__ == "__main__":
    main()
