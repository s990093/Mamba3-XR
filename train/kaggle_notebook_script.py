# Colab Script Pro: Mamba-3 (MIMO-Pro) CIFAR-100 Training
# Features:
# 1. Chunk-wise Parallel Scan (SSD Kernel) - Faster & More Stable
# 2. 1D Convolution - Better Local Features
# 3. Canonical Init - Inverse Softplus for dt, Trapezoidal Rule
# 4. Expand=2 - Standard Mamba expansion

import torch
import os
# [OOM Fix] Reduce fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# [Debug] Synchronous CUDA error reporting (User Request)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: No GPU detected. Training will be slow.")

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint # [OOM Fix] Gradient Checkpointing
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import math
from tqdm import tqdm
from sklearn.decomposition import PCA
import time
import gc
import datetime
from sklearn.metrics import f1_score
import copy
import sys

# ==========================================
# HYPERPARAMETERS & CONFIG are now in main() via Args class to support Notebook editing behavior
# The global constants below are placeholders/linked to Args in main execution flow 
# BUT for class definitions they need defaults or to be passed in. 
# VisionMambaPro uses arguments, so we are good.
# ==========================================

# [Kaggle Optimization]
torch.set_float32_matmul_precision('high') # Enable TensorCores (P100/T4/V100+)

# ==========================================
# PART 1: Mamba-3 Pro Modeling (from model_Lai_Mod.py)
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
        use_conv=True,   # [New] Optional Convolution toggle (Default True for Vision)
        d_conv=4,        # [New] Convolution kernel size
        rms_norm_eps=1e-5, 
        chunk_size=64,   # [OPTIMIZATION] Reduced from 256 for CIFAR (L=64)
        use_parallel_scan=True,
        
        # === Mamba-2 Initialization Hyperparameters ===
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
        
        # Calculate d_inner and n_heads
        self.d_inner = int(expand * d_model)
        
        # Ensure divisibility
        if self.d_inner % d_head != 0:
             # Adjust d_inner to be divisible
             self.d_inner = ((self.d_inner // d_head) + 1) * d_head
             
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
                f"expand={self.expand}, use_conv={self.use_conv}, mimo_rank={self.mimo_rank})")

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rrms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rrms * self.weight

class Mamba3Block(nn.Module):
    """
    Mamba-3 Block with SSD Scan and Conv1d.
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

        # [New] Optional Convolution
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
            
            # === dt_bias Initialization (Inverse Softplus) ===
            dt = torch.exp(
                torch.rand(G) * (math.log(config.dt_max) - math.log(config.dt_min))
                + math.log(config.dt_min)
            )
            dt = torch.clamp(dt, min=config.dt_init_floor)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            
            dt_start = self.dim_z + self.dim_x + self.dim_B + self.dim_C
            dt_end = dt_start + self.dim_dt
            self.in_proj.bias[dt_start:dt_end].copy_(inv_dt)
            
            # === lambda_bias Initialization ===
            lambda_start = dt_end
            self.in_proj.bias[lambda_start:].fill_(-3.0) # sigmoid(-3) approx 0.05
    
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
        
        L_orig = L
        if L % chunk_size != 0:
            pad_len = chunk_size - (L % chunk_size)
            u = F.pad(u, (0, 0, 0, 0, 0, 0, 0, pad_len))
            dt = F.pad(dt, (0, 0, 0, pad_len))
            C = F.pad(C, (0, 0, 0, 0, 0, 0, 0, pad_len))
            L = L + pad_len
        
        # [Stability] Force FP32 for Scan to prevent NaN/Inf causing Illegal Memory Access in Backward
        u = u.float()
        dt = dt.float()
        A = A.float()
        C = C.float()
        
        log_alpha = dt * A.view(1, 1, H)
        
        num_chunks = L // chunk_size
        u_chunk = u.view(B, num_chunks, chunk_size, H, N, P)
        dt_chunk = dt.view(B, num_chunks, chunk_size, H)
        log_alpha_chunk = log_alpha.view(B, num_chunks, chunk_size, H)
        C_chunk = C.view(B, num_chunks, chunk_size, H, N, R)
        
        # Intra-Chunk
        log_alpha_perm = log_alpha_chunk.permute(0, 1, 3, 2)
        L_mask = torch.exp(self.segsum(log_alpha_perm))
        h_intra = torch.einsum('bchij, bcjhnp -> bcihnp', L_mask, u_chunk)
        y_diag = torch.einsum('bclhnp, bclhnr -> bclhpr', h_intra, C_chunk)
        
        # Inter-Chunk
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
        
        # Output Combination
        log_alpha_cumsum = torch.cumsum(log_alpha_chunk, dim=2)
        decay_intra = torch.exp(log_alpha_cumsum)
        h_effect = torch.einsum('bchnp, bclh -> bclhnp', h_states_inter, decay_intra)
        y_off = torch.einsum('bclhnp, bclhnr -> bclhpr', h_effect, C_chunk)
        
        y_total = y_diag + y_off
        y_total = y_total.view(B, L, H, P, R)
        
        if L_orig < L:
            y_total = y_total[:, :L_orig]
        
        return y_total.to(input_dtype), h_prev.to(input_dtype)

    def forward(self, u, return_states=False, return_internals=False):
        B_sz, L, _ = u.shape
        H, G, P, N, R = self.config.n_heads, self.config.n_groups, self.config.d_head, self.config.d_state, self.config.mimo_rank
        ratio = self.ratio

        projected = self.in_proj(u)
        split_sections = [self.dim_z, self.dim_x, self.dim_B, self.dim_C, self.dim_dt, self.dim_lambda]
        z, x_prime, B_param, C_param, dt, lambda_param = torch.split(projected, split_sections, dim=-1)

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
        theta = torch.exp(self.theta_log)
        
        def broadcast_group(tensor, target_shape_suffix):
            return tensor.repeat_interleave(ratio, dim=2)
            
        dt = broadcast_group(dt.unsqueeze(-1), (1,)).squeeze(-1)
        A_broadcast = A.repeat_interleave(ratio, dim=0)
        theta_broadcast = theta.repeat_interleave(ratio, dim=0)
        
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
        
        x = self.x_up_proj(x_prime).view(B_sz, L, H, P, R)
        input_signal = torch.einsum('blhnr, blhpr -> blhnp', B_rotated, x)

        lambda_view = torch.sigmoid(broadcast_group(lambda_param.unsqueeze(-1), (1,)).squeeze(-1))
        alpha_val = torch.exp(torch.einsum('blh, h -> blh', dt, A_broadcast)) 
        
        dt_view = dt.view(B_sz, L, H, 1, 1)
        lambda_view = lambda_view.view(B_sz, L, H, 1, 1)
        alpha_view = alpha_val.view(B_sz, L, H, 1, 1)

        term_curr = lambda_view * dt_view * input_signal
        input_signal_prev = torch.roll(input_signal, shifts=1, dims=1)
        input_signal_prev[:, 0] = 0 
        term_prev = (1 - lambda_view) * dt_view * alpha_view * input_signal_prev
        u_ssm = term_curr + term_prev

        if self.config.use_parallel_scan:
            y_stack, h_state = self.chunk_parallel_scan(u_ssm, dt, A_broadcast, C_rotated, 
                                                        chunk_size=self.config.chunk_size)
        else:
            # Fallback (Should not be used in Pro script)
            y_stack, _ = self.chunk_parallel_scan(u_ssm, dt, A_broadcast, C_rotated, chunk_size=L) # Hack

        y_down = self.y_down_proj(y_stack.view(B_sz, L, H, P * R))
        y = y_down.view(B_sz, L, H * P)
        
        x_prime_view = x_prime.reshape(B_sz, L, H * P)
        y = y + x_prime_view * self.D.repeat_interleave(P, dim=0)
        
        z_act = self.act(z)
        y = y * z_act

        if return_states:
             return self.out_proj(y), h_state

        return self.out_proj(y)

# ==========================================
# PART 2: Vision Backbone (Modified for Pro)
# ==========================================

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1) 
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

class ConvStem(nn.Module):
    """ 
    ConvStem replaces PatchEmbedding for better early feature extraction.
    32x32 image -> 3 layers Conv -> 8x8 feature map (equivalent to patch size 4)
    """
    def __init__(self, in_chans=3, embed_dim=384):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, bias=False), # Maintain size
        )

    def forward(self, x):
        x = self.proj(x)
        # x shape: [B, C, H, W] -> Flatten to [B, L, C]
        x = x.flatten(2).transpose(1, 2)
        return x

class VisionMambaPro(nn.Module):
    def __init__(self, 
                 img_size=32, 
                 patch_size=4, 
                 depth=12, 
                 embed_dim=384, 
                 d_state=32, 
                 d_head=64, 
                 mimo_rank=8,
                 num_classes=100,
                 drop_path_rate=0.2,
                 bidirectional=True,
                 expand=2,       
                 use_conv=True): 
        super().__init__()
        self.embed_dim = embed_dim
        self.bidirectional = bidirectional
        self.img_size = img_size
        self.patch_size = patch_size
        self.drop_path_rate = drop_path_rate
        
        # [Modification] Use ConvStem instead of PatchEmbedding
        # Renamed to 'stem' to avoid any attribute naming conflicts or confusion
        self.stem = ConvStem(in_chans=3, embed_dim=embed_dim)
        
        self.config = Mamba3Config(
            d_model=embed_dim, 
            d_state=d_state, 
            d_head=d_head,
            n_groups=1,
            mimo_rank=mimo_rank,
            expand=expand,
            use_conv=use_conv,
            use_parallel_scan=True
        )
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            if bidirectional:
                self.layers.append(nn.ModuleDict({
                    'fwd': Mamba3Block(self.config),
                    'bwd': Mamba3Block(self.config)
                }))
            else:
                self.layers.append(Mamba3Block(self.config))
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)] 
        self.drop_paths = nn.ModuleList([DropPath(dpr[i]) for i in range(depth)])
        self.norms = nn.ModuleList([RMSNorm(embed_dim) for _ in range(depth)])
        self.final_norm = RMSNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        n_patches = (img_size // 4) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # [Optimization] Precompute snake indices in __init__ to avoid side-effects in forward()
        # This is friendlier for torch.compile
        H_grid = W_grid = img_size // patch_size 
        grid = torch.arange(H_grid * W_grid).view(H_grid, W_grid)
        grid[1::2] = grid[1::2].flip(1)
        self.register_buffer('snake_indices', grid.flatten())
        self.register_buffer('snake_rev_indices', torch.argsort(grid.flatten()))

    def forward(self, x):
        x = self.stem(x)
        x = x + self.pos_embed
        
        # Determine strict or flexible shape handling
        B, L, C = x.shape
        if L == self.snake_indices.shape[0]:
             x = x[:, self.snake_indices, :]
        else:
            # Fallback for unexpected shapes (e.g. if image size changes, though unlikely here)
            pass 
        
        # Forward layers
        # Forward layers
        for i, layer in enumerate(self.layers):
            norm_x = self.norms[i](x)
            
            # === [FIX] Gradient Checkpointing Helper ===
            def run_layer(module, inp):
                return module(inp)

            if self.bidirectional:
                # Use checkpointing to save VRAM
                # Note: We must pass use_reentrant=False for modern PyTorch
                out_fwd = checkpoint(run_layer, layer['fwd'], norm_x, use_reentrant=False)
                
                norm_x_rev = norm_x.flip(dims=[1])
                out_bwd = checkpoint(run_layer, layer['bwd'], norm_x_rev, use_reentrant=False)
                out_bwd = out_bwd.flip(dims=[1])
                
                out_combined = (out_fwd + out_bwd) / 2
                x = x + self.drop_paths[i](out_combined)
            else:
                out = checkpoint(run_layer, layer, norm_x, use_reentrant=False)
                x = x + self.drop_paths[i](out)
                
        x = self.final_norm(x)
        x = x.mean(dim=1) 
        logits = self.head(x)
        return logits

# ==========================================
# PART 3: Utilities & Diagnostics
# ==========================================
class MambaDiagnostics:
    def __init__(self, log_dir="results/diagnostics_pro"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.history = {
            "loss": [], "accuracy": [], "val_loss": [], "val_accuracy": [], 
            "val_acc5": [], "val_acc_ema": [], "val_acc5_ema": [], 
            "layer_stats": {}, "mimo_ranks": {}, "eigen_A": {},
            "delta_stats": {}, "delta_heatmap": {}, "layer_activations": {}, 
        }
    
    def log_metrics(self, epoch, loss, accuracy, val_loss=None, val_acc=None, val_acc5=None, val_acc_ema=None, val_acc5_ema=None):
        self.history["loss"].append(loss)
        self.history["accuracy"].append(accuracy)
        if val_loss: self.history["val_loss"].append(val_loss)
        if val_acc: self.history["val_accuracy"].append(val_acc)
        if val_acc5: self.history["val_acc5"].append(val_acc5)
        if val_acc_ema: self.history["val_acc_ema"].append(val_acc_ema)
        if val_acc5_ema: self.history["val_acc5_ema"].append(val_acc5_ema)
        
        msg = f"[Pro-Metrics] Epoch {epoch} | Train Loss: {loss:.4f} Acc: {accuracy:.2f}%"
        if val_loss: msg += f" | Val: {val_acc:.2f}% (Top5: {val_acc5:.2f}%)"
        if val_acc_ema: msg += f" | EMA: {val_acc_ema:.2f}% (Top5: {val_acc5_ema:.2f}%)"
        print(msg)

    def save_history(self):
        torch.save(self.history, os.path.join(self.log_dir, "diagnostics_history.pt"))

    def log_gradients(self, model, epoch):
        print(f"[Diagnostics] Logging Gradients for Epoch {epoch}...")
        for name, param in model.named_parameters():
            if param.grad is None: continue
            if not ("A_log" in name or "x_up_proj" in name or "in_proj" in name): continue
            
            g_mean = param.grad.mean().item()
            g_std = param.grad.std().item()
            g_norm = param.grad.norm().item()
            w_norm = param.norm().item()
            snr = abs(g_mean) / (g_std + 1e-9)
            update_ratio = (1e-3 * g_norm) / (w_norm + 1e-9) 
            
            key = f"{name}"
            if key not in self.history["layer_stats"]:
                self.history["layer_stats"][key] = {"snr": [], "update_ratio": [], "grad_norm": []}
            self.history["layer_stats"][key]["snr"].append(snr)
            self.history["layer_stats"][key]["update_ratio"].append(update_ratio)
            self.history["layer_stats"][key]["grad_norm"].append(g_norm)

    def log_eigenvalues(self, model, epoch):
        print(f"[Diagnostics] Logging Matrix A Eigenvalues for Epoch {epoch}...")
        all_taus = []
        for name, param in model.named_parameters():
            if "A_log" in name:
                A = -torch.exp(param.detach())
                taus = -1.0 / (A.cpu().numpy() + 1e-9)
                all_taus.extend(taus.flatten())
        self.history["eigen_A"][epoch] = all_taus

    def log_mimo_rank(self, model, epoch):
        print(f"[Diagnostics] Logging MIMO Ranks for Epoch {epoch}...")
        for name, module in model.named_modules():
            if hasattr(module, "x_up_proj") and isinstance(module.x_up_proj, nn.Linear):
                W = module.x_up_proj.weight.detach().cpu().float()
                try:
                    U, S, V = torch.linalg.svd(W)
                    S_norm = S / S.sum()
                    entropy = -torch.sum(S_norm * torch.log(S_norm + 1e-9))
                    eff_rank = torch.exp(entropy).item()
                    if name not in self.history["mimo_ranks"]: self.history["mimo_ranks"][name] = []
                    self.history["mimo_ranks"][name].append(eff_rank)
                except:
                    pass

class ModelEma(nn.Module):
    def __init__(self, model, decay=0.9999, device=None):
        super().__init__()
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device 
        if self.device is not None: self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None: model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

def mixup_data(x, y, alpha=1.0, device='cuda'):
    if alpha > 0: lam = np.random.beta(alpha, alpha)
    else: lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def get_cifar100_loaders(batch_size=128, num_workers=2):
    print("Preparing CIFAR-100 Data (Pro Enhancement)...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10), # [Enhanced Augmentation]
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        transforms.RandomErasing(p=0.1) # [Enhanced Augmentation]
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return trainloader, testloader

def get_dry_run_loaders(batch_size=128):
    print("Preparing Dry Run Data (Subset)...")
    transform_train = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    # Use CIFAR10 for quick dry run as it's smaller/easier or just subset CIFAR100
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    # Subset
    trainset = torch.utils.data.Subset(trainset, range(BATCH_SIZE * 2)) # 2 batches
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_train)
    testset = torch.utils.data.Subset(testset, range(BATCH_SIZE * 2))
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
    return trainloader, testloader

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, use_mixup=True, scaler=None, accum_steps=1, use_amp=False):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    optimizer.zero_grad()
    
    for i, (inputs, labels) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # [Stability] Condition AMP on USE_AMP flag
        # Note: We must use a dummy context manager or check flag
        if use_amp:
            context = torch.amp.autocast('cuda')
        else:
            context = torch.no_grad() if False else torch.enable_grad() # Dummy

        with context:
            if use_mixup:
                inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha=1.0, device=device)
                outputs = model(inputs)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
        
        # Gradient Accumulation
        loss = loss / accum_steps
        
        if use_amp:
            scaler.scale(loss).backward()
            if (i + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss.backward()
            if (i + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
        
        running_loss += loss.item() * accum_steps # Scale back for logging
        _, predicted = outputs.max(1)
        if use_mixup:
            total += labels.size(0) 
            correct += (lam * predicted.eq(targets_a).sum().float() + (1 - lam) * predicted.eq(targets_b).sum().float()).item()
        else:
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return running_loss / len(dataloader), 100. * correct / total

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    correct5 = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Top-5
            _, pred5 = outputs.topk(5, 1, True, True)
            pred5 = pred5.t()
            correct5 += pred5.eq(labels.view(1, -1).expand_as(pred5)).reshape(-1).float().sum().item()
            
    return running_loss / len(dataloader), 100. * correct / total, 100. * correct5 / total

# ==========================================
# MAIN
# ==========================================

def main():
    # [Config] Manual Configuration (Replaces argparse)
    class Args:
        rank = 8          # [Optimization] Reduced from 12 to 8 as D_HEAD is larger
        epochs = 300      # [Optimization] Increased from 100 to 300
        name = "pro_run_80p"
        batch_size = 128 * 2  # Virtual Batch Size (we use Grad Accumulation)
        lr = 1e-3
        no_conv = False
        expand = 2
        dry_run = False
        compile_mode = 'reduce-overhead' # [Fix] Default to 'none' to avoid attribute errors on T4 for now 

    args = Args()

    # ==========================================
    # HYPERPARAMETERS (Targeting 80%+)
    # ==========================================
    REAL_BATCH_SIZE = 24      # [Stability] Reduced to 24 for FP32 training
    ACCUM_STEPS = args.batch_size // REAL_BATCH_SIZE
    LR = args.lr
    EPOCHS = args.epochs
    IMG_SIZE = 32
    PATCH_SIZE = 4
    D_MODEL = 384       # [Optimization] Increased from 192
    DEPTH = 12          # [Optimization] Increased from 6
    D_STATE = 32
    D_HEAD = 64         # [Optimization] Increased from 32
    MIMO_RANK = args.rank
    NUM_CLASSES = 100 
    DROP_PATH_RATE = 0.2 # [Optimization] Increased regularization
    LABEL_SMOOTHING = 0.1
    USE_COMPILE = True
    USE_MAMBA_DIAGNOSTICS = True
    USE_EMA = True
    USE_MIXUP = True
    USE_AMP = False # [Stability] Disable AMP (Mixed Precision) to prevent illegal memory access errors


    # parser = argparse.ArgumentParser(description='Mamba-3 Pro Training')
    # parser.add_argument('--rank', type=int, default=12, help='MIMO Rank')
    # parser.add_argument('--epochs', type=int, default=100, help='Epochs')
    # parser.add_argument('--name', type=str, default="pro_run", help='Run Name')
    # parser.add_argument('--batch-size', type=int, default=64, help='Batch Size')
    # parser.add_argument('--lr', type=float, default=1e-3, help='Peak LR')
    # parser.add_argument('--no-conv', action='store_true', help='Disable 1D Conv')
    # parser.add_argument('--expand', type=int, default=2, help='Expansion Factor')
    # parser.add_argument('--dry-run', action='store_true', help='Quick syntax check with minimal data')
    # parser.add_argument('--compile-mode', type=str, default='default', choices=['default', 'reduce-overhead', 'max-autotune'], help='torch.compile mode')
    # args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting Mamba-Pro Training: Rank={args.rank}, Conv={not args.no_conv}, Expand={args.expand}")
    
    # [Kaggle Optimization]
    if torch.cuda.is_available():
        # [Stability] Disable benchmark if we are crashing. Enable if stable.
        # torch.backends.cudnn.benchmark = True 
        torch.backends.cudnn.benchmark = False
        print(f"cuDNN Benchmark: Disabled (Stability Mode)")

    # Init Data
    if args.dry_run:
        print("[DRY RUN] Using minimal dataset for syntax check...")
        trainloader, testloader = get_dry_run_loaders(batch_size=args.batch_size)
        args.epochs = 1 # Override epochs
    else:
        # [Optimization] Use REAL_BATCH_SIZE for dataloaders
        trainloader, testloader = get_cifar100_loaders(batch_size=REAL_BATCH_SIZE)
    
    # Init Model
    model = VisionMambaPro(
        img_size=IMG_SIZE,
        depth=DEPTH,
        embed_dim=D_MODEL,
        d_state=D_STATE,
        d_head=D_HEAD,
        mimo_rank=args.rank,
        num_classes=NUM_CLASSES,
        expand=args.expand,
        use_conv=not args.no_conv,
        drop_path_rate=DROP_PATH_RATE
    )
    
    # [Stability] Force Float32 (T4 does not support BFloat16 well, and Mamba kernels can break)
    model = model.float().to(device)
    
    # EMA Model (Init before DataParallel wrapping)
    ema_model = ModelEma(model, decay=0.9999, device=device)

    # [Optimization] torch.compile
    # CRITICAL FIX: DataParallel + torch.compile is unstable on T4 and often causes AttributeErrors.
    # We automatically disable compile if using multiple GPUs with DataParallel.
    if torch.cuda.device_count() > 1:
        print("[System] Multiple GPUs detected. Disabling torch.compile to avoid DataParallel compatibility issues.")
        args.compile_mode = 'none'

    if hasattr(torch, 'compile') and args.compile_mode != 'none':
        print(f"Compiling model with mode={args.compile_mode}...")
        try:
            model = torch.compile(model, mode=args.compile_mode)
        except Exception as e:
            print(f"[Warning] torch.compile failed: {e}. Continuing without compilation.")

    # [Dual GPU Support]
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    # Separate groups for weight decay
    param_groups = [
        {'params': [p for n, p in model.named_parameters() if 'A_log' in n or 'theta_log' in n or 'bias' in n], 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if not ('A_log' in n or 'theta_log' in n or 'bias' in n)], 'weight_decay': 0.05}
    ]
    optimizer = optim.AdamW(param_groups, lr=args.lr)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(trainloader), epochs=args.epochs, pct_start=0.1)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    scaler = torch.amp.GradScaler('cuda')
    
    # Diagnostics
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{timestamp}_pro_rank{args.rank}_e{args.epochs}_{args.name}"
    diagnostics = MambaDiagnostics(log_dir=f"results/{run_name}")
    
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, trainloader, criterion, optimizer, device, epoch, use_mixup=USE_MIXUP, scaler=scaler, accum_steps=ACCUM_STEPS, use_amp=USE_AMP)
        val_loss, val_acc, val_acc5 = evaluate(model, testloader, criterion, device)
        _, val_acc_ema, val_acc5_ema = evaluate(ema_model.module, testloader, criterion, device)
        
        diagnostics.log_metrics(epoch, train_loss, train_acc, val_loss, val_acc, val_acc5, val_acc_ema, val_acc5_ema)
        print(f"EMA Acc: {val_acc_ema:.2f}%")
        
        scheduler.step()
        # [Dual GPU Support] Unwrap model for EMA update
        ema_model.update(model.module if hasattr(model, 'module') else model)
        
        if val_acc_ema > best_acc:
            best_acc = val_acc_ema
            torch.save(model.state_dict(), os.path.join(diagnostics.log_dir, "best_model.pth"))
        
        # [DIAGNOSTICS] Extensive Logging
        diagnostics.log_gradients(model, epoch)
        diagnostics.log_eigenvalues(model, epoch)
        diagnostics.log_mimo_rank(model, epoch)
            
        diagnostics.save_history()

    print(f"Training Complete. Best EMA Acc: {best_acc:.2f}%")

if __name__ == "__main__":
    main()
