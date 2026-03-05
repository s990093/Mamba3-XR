"""
Mamba-3 Wikipedia Training Script for Kaggle
使用 /kaggle/input/wikipedia-structured-contents 數據集
"""

import os
import sys
import time
import torch
import torch.nn as nn
import pandas as pd
import json
from pathlib import Path
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

# ============================================================================
# 環境配置
# ============================================================================
# 🔴 CRITICAL FIX: Suppress HuggingFace tokenizers fork warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# 禁用 torch.compile (Kaggle 環境可能不穩定)
torch.compiler.disable()
if hasattr(torch, '_dynamo'):
    torch._dynamo.config.suppress_errors = True


# ============================================================================
# Part 1: Mamba-3 Model Components (Standalone)
# ============================================================================

import torch.nn.functional as F
import math

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
        
        # Intra-chunk
        log_alpha_perm = log_alpha_chunk.permute(0, 1, 3, 2)
        L_mask = torch.exp(self.segsum(log_alpha_perm))
        L_mask = L_mask.to(u.dtype)
        
        h_intra = torch.einsum('bchij, bcjhnp -> bcihnp', L_mask, u_chunk)
        y_diag = torch.einsum('bclhnp, bclhnr -> bclhpr', h_intra, C_chunk)
        
        # Inter-chunk
        decay_chunk = torch.exp(torch.sum(log_alpha_chunk, dim=2))
        h_chunk_final = h_intra[:, :, -1].contiguous()
        
        # PyTorch fallback
        h_states_inter = torch.empty(B, num_chunks, H, N, P, device=device, dtype=input_dtype)
        current_state = torch.zeros(B, H, N, P, device=device, dtype=input_dtype)
        
        for c in range(num_chunks):
            d = decay_chunk[:, c, :, None, None]
            x = h_chunk_final[:, c]
            current_state = current_state * d + x
            h_states_inter[:, c] = current_state
        
        h_prev = torch.roll(h_states_inter, shifts=1, dims=1)
        h_prev[:, 0] = 0
        final_h_prev = h_states_inter[:, -1]
        
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
            chunk_size=512,
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


# ============================================================================
# Part 2: Wikipedia 數據解析
# ============================================================================

def extract_text_from_section(section):
    """遞迴從 nested json 中提取文字"""
    if not isinstance(section, dict):
        return ""
        
    text_parts = []
    
    # 1. 標題
    if 'name' in section and section['name']:
        text_parts.append(f"\n\n=={section['name']}==\n")
        
    # 2. 段落內容
    if 'value' in section and isinstance(section['value'], str):
        text_parts.append(section['value'])
        
    # 3. 遞迴處理子部分
    if 'has_parts' in section and isinstance(section['has_parts'], list):
        for part in section['has_parts']:
            text_parts.append(extract_text_from_section(part))
            
    return "".join(text_parts)


def parse_wikipedia_record(record):
    """將單條 Wikipedia JSON 記錄轉為純文字"""
    # 跳過重定向頁面
    if record.get('abstract') and str(record['abstract']).startswith('REDIRECT'):
        return None
        
    full_text = []
    
    # 標題
    if record.get('name'):
        full_text.append(f"Title: {record['name']}\n")
        
    # 簡介
    if record.get('description'):
        full_text.append(f"{record['description']}\n")
        
    # 主要內容 (sections)
    sections = record.get('sections')
    if isinstance(sections, list):
        for sec in sections:
            if isinstance(sec, dict):
                full_text.append(extract_text_from_section(sec))
            elif isinstance(sec, str):
                full_text.append(sec)
                
    text = "\n".join(full_text)
    
    # 過濾太短的文章
    if len(text) < 200:
        return None
        
    return text


def read_jsonl_lazy(file_path):
    """Lazy loading JSONL 文件（逐行讀取，節省記憶體）"""
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                yield json.loads(line.strip())
            except json.JSONDecodeError:
                continue


# ============================================================================
# Part 2: 數據預處理
# ============================================================================

def prepare_wikipedia_data(
    data_dir='/kaggle/input/wikipedia-structured-contents/enwiki_namespace_0',
    cache_dir='data_wiki_processed',
    max_tokens=50_000_000,  # 5000萬 tokens (約 40MB 文本)
    split_ratio=0.95
):
    """
    處理 Wikipedia 數據並保存為 PyTorch tensors
    
    Args:
        data_dir: Wikipedia JSONL 文件目錄
        cache_dir: 快取目錄
        max_tokens: 最大 token 數量
        split_ratio: 訓練集比例
    """
    os.makedirs(cache_dir, exist_ok=True)
    train_path = os.path.join(cache_dir, 'train.pt')
    val_path = os.path.join(cache_dir, 'val.pt')
    
    # 檢查是否已處理
    if os.path.exists(train_path) and os.path.exists(val_path):
        print("✅ Wikipedia data already processed.")
        return train_path, val_path

    print("⚡ Processing Wikipedia data...")
    print(f"   Source: {data_dir}")
    print(f"   Target tokens: {max_tokens:,}")
    
    # 初始化 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # 收集 tokens
    token_ids = []
    total_tokens = 0
    articles_processed = 0
    
    # 找出所有 JSONL 文件
    jsonl_files = list(Path(data_dir).glob('*.jsonl'))
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found in {data_dir}")
    
    print(f"📂 Found {len(jsonl_files)} JSONL files")
    
    # Progress logging (no tqdm for Kaggle)
    
    for jsonl_file in jsonl_files:
        print(f"\n📄 Processing: {jsonl_file.name}")
        
        for record in read_jsonl_lazy(jsonl_file):
            # 解析文本
            text = parse_wikipedia_record(record)
            if text is None:
                continue
                
            # Tokenize
            ids = tokenizer.encode(text, add_special_tokens=False)
            ids.append(tokenizer.eos_token_id)  # 文章結束符
            
            token_ids.extend(ids)
            total_tokens += len(ids)
            articles_processed += 1
            # Log every 10000 articles
            if articles_processed % 10000 == 0:
                print(f"  Processed {articles_processed:,} articles, {total_tokens:,} tokens...")
            
            # 檢查是否達到目標
            if total_tokens >= max_tokens:
                break
                
        if total_tokens >= max_tokens:
            break
    
    
    print(f"\n📊 Statistics:")
    print(f"   Articles processed: {articles_processed:,}")
    print(f"   Total tokens: {total_tokens:,}")
    print(f"   Converting to tensor...")
    
    # 轉換為 tensor
    data_tensor = torch.tensor(token_ids, dtype=torch.long)
    
    # 切分訓練/驗證集
    n_train = int(len(data_tensor) * split_ratio)
    train_data = data_tensor[:n_train]
    val_data = data_tensor[n_train:]
    
    # 保存
    torch.save(train_data, train_path)
    torch.save(val_data, val_path)
    
    print(f"💾 Saved:")
    print(f"   Train: {train_path} ({len(train_data):,} tokens)")
    print(f"   Val:   {val_path} ({len(val_data):,} tokens)")
    
    return train_path, val_path


# ============================================================================
# Part 3: Dataset Class
# ============================================================================

class WikiTokenDataset(Dataset):
    """Token-level Wikipedia Dataset"""
    
    def __init__(self, data_path, block_size):
        print(f"📂 Loading {data_path}...")
        self.data = torch.load(data_path)
        self.block_size = block_size
        print(f"   Loaded {len(self.data):,} tokens")
        
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


# ============================================================================
# Part 4: 訓練函數 (簡化版，專注於 loss 記錄)
# ============================================================================

def train_wikipedia(
    model,
    train_loader,
    val_loader,
    epochs=1,
    lr=6e-4,
    eval_interval=500,
    eval_iters=50,
    save_dir='checkpoints',
    use_accelerate=True,
    mixed_precision='fp16'
):
    """
    訓練 Wikipedia 模型 (支持多GPU)
    
    Args:
        model: Mamba3LM 模型
        train_loader: 訓練 DataLoader
        val_loader: 驗證 DataLoader
        epochs: 訓練輪數
        lr: 學習率
        eval_interval: 評估間隔 (steps)
        eval_iters: 每次評估的 iterations
        save_dir: 模型保存目錄
        use_accelerate: 是否使用多GPU訓練
        mixed_precision: 混合精度 ('fp16' 或 'no')
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save batch size before prepare
    original_batch_size = train_loader.batch_size
    
    # ========================================================================
    # Multi-GPU Setup with Accelerate
    # ========================================================================
    accelerator = None
    if use_accelerate:
        try:
            from accelerate import Accelerator
            from accelerate.state import AcceleratorState
            
            if AcceleratorState._shared_state != {}:
                print("⚠️  Accelerator already initialized, reusing...")
                accelerator = Accelerator()
            else:
                accelerator = Accelerator(mixed_precision=mixed_precision)
                
            device = accelerator.device
            
            if accelerator.is_main_process:
                print("\n" + "=" * 80)
                print("🚀 Multi-GPU Training with Accelerate")
                print("=" * 80)
                print(f"GPUs: {accelerator.num_processes}")
                print(f"Device: {device}")
                print(f"Mixed Precision: {accelerator.mixed_precision}")
                print("=" * 80 + "\n")
                
        except ImportError:
            print("⚠️  Accelerate not available, falling back to single-GPU")
            accelerator = None
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = model.to(device)
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
    
    # ========================================================================
    # Optimizer and Scheduler
    # ========================================================================
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    
    total_steps = epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps
    )
    
    # ========================================================================
    # Prepare for DDP
    # ========================================================================
    if accelerator is not None:
        model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, val_loader, scheduler
        )
    
    # 訓練歷史記錄
    history = {
        'train_losses': [],
        'val_losses': [],
        'learning_rates': [],
        'steps': [],
        'best_val_loss': float('inf')
    }
    
    global_step = 0
    best_val_loss = float('inf')
    
    if accelerator is None or accelerator.is_main_process:
        print("\n" + "=" * 80)
        print("🚀 Starting Wikipedia Training")
        print("=" * 80)
        
        if accelerator is not None:
            unwrapped_model = accelerator.unwrap_model(model)
            n_params = unwrapped_model.get_num_params()
        else:
            n_params = model.get_num_params()
            
        print(f"Model parameters: {n_params / 1e6:.2f}M")
        print(f"Training steps: {total_steps:,}")
        print(f"Batch size per GPU: {original_batch_size}")
        if accelerator is not None:
            print(f"Effective batch size: {original_batch_size * accelerator.num_processes}")
        print(f"Device: {device}")
        print("=" * 80 + "\n")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        epoch_start_time = time.time()
        
        # Log epoch start (no tqdm for Kaggle)
        if accelerator is None or accelerator.is_main_process:
            print(f"\n" + "="*80)
            print(f"📖 EPOCH {epoch+1}/{epochs} STARTED")
            print(f"="*80)
            print(f"  Batches in epoch: {len(train_loader)}")
            print(f"  Learning rate: {scheduler.get_last_lr()[0]:.2e}")
            print(f"="*80 + "\n")
        
        for batch_idx, (x, y) in enumerate(train_loader):
            # Data already on device if using accelerator
            if accelerator is None:
                x, y = x.to(device), y.to(device)
            
            # Forward
            loss, _ = model(x, targets=y, return_loss=True)
            
            # Backward
            optimizer.zero_grad()
            
            if accelerator is not None:
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
            
            # 記錄
            current_loss = loss.item()
            epoch_loss += current_loss
            global_step += 1
            
            # ⚡ LOG EVERY 1000 STEPS
            if global_step % 1000 == 0:
                if accelerator is None or accelerator.is_main_process:
                    elapsed = time.time() - epoch_start_time
                    steps_per_sec = (batch_idx + 1) / elapsed if elapsed > 0 else 0
                    samples_per_sec = steps_per_sec * original_batch_size
                    avg_loss = epoch_loss / (batch_idx + 1)
                    
                    print(f"[Step {global_step:6d}] "
                          f"Epoch {epoch+1}/{epochs} | "
                          f"Batch {batch_idx+1}/{len(train_loader)} | "
                          f"Loss: {current_loss:.4f} | "
                          f"Avg Loss: {avg_loss:.4f} | "
                          f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                          f"Speed: {samples_per_sec:.0f} samp/s")

                    pbar.set_postfix({
                        'loss': f'{current_loss:.4f}',
                        'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                    })
            
            # 評估
            if global_step % eval_interval == 0:
                # 🔴 CRITICAL: Synchronization for DDP
                if accelerator is not None:
                    accelerator.wait_for_everyone()
                
                # Only main process evaluates
                if accelerator is None or accelerator.is_main_process:
                    # Unwrap model for evaluation
                    if accelerator is not None:
                        unwrapped_model = accelerator.unwrap_model(model)
                    else:
                        unwrapped_model = model
                    
                    val_loss = evaluate_model(unwrapped_model, val_loader, device, eval_iters)
                    
                    # 記錄歷史
                    history['steps'].append(global_step)
                    history['train_losses'].append(current_loss)
                    history['val_losses'].append(val_loss)
                    history['learning_rates'].append(scheduler.get_last_lr()[0])
                    
                    print(f"\n{'='*80}")
                    print(f"Step {global_step:,} Evaluation")
                    print(f"{'='*80}")
                    print(f"  Train Loss: {current_loss:.4f}")
                    print(f"  Val Loss:   {val_loss:.4f}")
                    print(f"  LR:         {scheduler.get_last_lr()[0]:.2e}")
                    print(f"{'='*80}\n")
                    
                    # 保存最佳模型
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        history['best_val_loss'] = best_val_loss
                        
                        best_path = os.path.join(save_dir, 'best_model.pt')
                        torch.save({
                            'model_state_dict': unwrapped_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'step': global_step,
                            'val_loss': val_loss,
                        }, best_path)
                        print(f"💾 Saved best model (val_loss={val_loss:.4f})")
                    
                    # 保存歷史
                    import json
                    history_path = os.path.join(save_dir, 'training_history.json')
                    with open(history_path, 'w') as f:
                        json.dump(history, f, indent=2)
                
                # Wait for main process to finish evaluation
                if accelerator is not None:
                    accelerator.wait_for_everyone()
                
                model.train()
        
        # Epoch 結束
        avg_loss = epoch_loss / len(train_loader)
        
        # Synchronize before epoch-end evaluation
        if accelerator is not None:
            accelerator.wait_for_everyone()
        
        if accelerator is None or accelerator.is_main_process:
            print(f"\n✅ Epoch {epoch+1} Complete - Avg Loss: {avg_loss:.4f}\n")
            
            # 保存 checkpoint
            if accelerator is not None:
                unwrapped_model = accelerator.unwrap_model(model)
            else:
                unwrapped_model = model
                
            ckpt_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': unwrapped_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'avg_loss': avg_loss,
            }, ckpt_path)
            print(f"💾 Saved checkpoint: {ckpt_path}")
    
    return model, history



@torch.no_grad()
def evaluate_model(model, val_loader, device, eval_iters):
    """評估模型"""
    model.eval()
    total_loss = 0
    
    for i, (x, y) in enumerate(val_loader):
        if i >= eval_iters:
            break
            
        x, y = x.to(device), y.to(device)
        loss, _ = model(x, targets=y, return_loss=True)
        total_loss += loss.item()
    
    return total_loss / min(eval_iters, len(val_loader))


# ============================================================================
# Part 5: 主程序
# ============================================================================

def main():
    """主訓練流程"""
    
    # =========================
    # 配置參數
    # =========================
    WIKI_DATA_DIR = '/kaggle/input/wikipedia-structured-contents/enwiki_namespace_0'
    CACHE_DIR = 'data_wiki_processed'
    CHECKPOINT_DIR = 'checkpoints_wikipedia'
    
    # ⚡ PRODUCTION-READY MODEL CONFIGURATION (T4-Optimized)
    BLOCK_SIZE = 512         # Context length
    BATCH_SIZE = 12          # ⚡ Reduced to 12 to ensure stability (24 total with 2 GPUs)
    EPOCHS = 2               # Train for 2 epochs
    LR = 3e-4                # Standard learning rate
    MAX_TOKENS = 100_000_000 # 100M tokens
    
    # 🚀 MODEL CONFIGURATION (Conservative for T4 16GB)
    # This configuration is tested and stable on T4 GPUs
    D_MODEL = 384            # ⚡ Reduced from 512 for stability
    N_LAYERS = 6             # 6 layers = good balance
    D_STATE = 64             # State dimension
    D_HEAD = 64              # Head dimension  
    N_GROUPS = 2             # 2 groups
    MIMO_RANK = 2            # MIMO rank
    DROPOUT = 0.1
    CHUNK_SIZE = 256         # Chunk size for parallel scan
    
    # ⚡ TRAINING MODE
    USE_MULTI_GPU = False    # ⚡ START WITH SINGLE-GPU FOR TESTING
                             # Set to True after confirming single-GPU works
    
    # =========================
    # 1. 數據準備
    # =========================
    print("=" * 80)
    print("📚 Preparing Wikipedia Data")
    print("=" * 80)
    
    train_path, val_path = prepare_wikipedia_data(
        data_dir=WIKI_DATA_DIR,
        cache_dir=CACHE_DIR,
        max_tokens=MAX_TOKENS,
        split_ratio=0.95
    )
    
    # =========================
    # 2. 創建 Dataset
    # =========================
    train_dataset = WikiTokenDataset(train_path, BLOCK_SIZE)
    val_dataset = WikiTokenDataset(val_path, BLOCK_SIZE)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,          # ⚡ MUST be 0 for Kaggle DDP (fork-based spawning)
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,          # ⚡ MUST be 0 for Kaggle DDP
        pin_memory=True
    )
    
    # =========================
    # 3. 創建模型 (使用小 chunk_size 避免 OOM)
    # =========================
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    vocab_size = tokenizer.vocab_size
    
    # 🔴 CRITICAL: Create model with proper configuration from the start
    # Do NOT manually override config after creation (causes segfault in fork mode)
    def create_mamba3_wikipedia(
        vocab_size, 
        d_model=512, 
        n_layers=6,          # ⚡ Reduced from 8 to fit T4 memory
        d_state=64,          # ⚡ Reduced from 128 to fit T4 memory
        d_head=64,           # ⚡ Reduced from 128 to fit T4 memory  
        n_groups=2,          # ⚡ Reduced from 4 to fit T4 memory
        mimo_rank=2, 
        dropout=0.1, 
        chunk_size=256,
        block_size=512
    ):
        """Create production-scale Mamba3 model for Wikipedia training
        
        CRITICAL: This function creates the model with the correct config from the start.
        DO NOT modify model.config after creation - it causes segfaults in fork mode!
        """
        
        # The Mamba3LM will create its own internal config based on these parameters
        # We don't need to manually override anything
        model = Mamba3LM(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            d_state=d_state,
            d_head=d_head,
            n_groups=n_groups,
            mimo_rank=mimo_rank,
            expand=2,
            max_seq_len=block_size,
            dropout=dropout,
            use_rope=False,  # Use positional embeddings
            tie_embeddings=True,
        )
        
        # ⚡ CRITICAL: DO NOT modify model.config.chunk_size here!
        # The config inside Mamba3LM.__init__ already sets it to 512 by default
        # If we need different chunk_size, we must modify Mamba3LM.__init__ itself
        
        return model
    
    model = create_mamba3_wikipedia(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        d_state=D_STATE,
        d_head=D_HEAD,
        n_groups=N_GROUPS,
        mimo_rank=MIMO_RANK,
        dropout=DROPOUT,
        chunk_size=CHUNK_SIZE,
        block_size=BLOCK_SIZE
    )
    
    print(f"\n" + "="*80)
    print(f"📦 PRODUCTION-SCALE MODEL CONFIGURATION")
    print(f"="*80)
    print(f"  Model Size:     {model.get_num_params() / 1e6:.1f}M parameters")
    print(f"  Vocab Size:     {vocab_size:,}")
    print(f"  d_model:        {D_MODEL}")
    print(f"  n_layers:       {N_LAYERS}")
    print(f"  d_state:        {D_STATE}")
    print(f"  d_head:         {D_HEAD}")
    print(f"  n_groups:       {N_GROUPS}")
    print(f"  MIMO Rank:      {MIMO_RANK}")
    print(f"  Block Size:     {BLOCK_SIZE}")
    print(f"  Chunk Size:     {CHUNK_SIZE}")
    print(f"  Batch Size:     {BATCH_SIZE} per GPU")
    print(f"  Learning Rate:  {LR}")
    print(f"  Dropout:        {DROPOUT}")
    print(f"="*80 + "\n")

    
    # =========================
    # 4. 訓練 (With Enhanced Logging)
    # =========================
    print(f"\n" + "="*80)
    print(f"🚀 STARTING TRAINING")
    print(f"="*80)
    print(f"  Total Tokens:    {MAX_TOKENS:,}")
    print(f"  Training Steps:  ~{len(train_loader) * EPOCHS:,}")
    print(f"  Log Interval:    Every 1000 steps")
    print(f"  Eval Interval:   Every 1000 steps")
    print(f"="*80 + "\n")
    
    model, history = train_wikipedia(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        lr=LR,
        eval_interval=1000,       # ⚡ Log every 1000 steps
        eval_iters=100,           # ⚡ More eval iterations for stability
        save_dir=CHECKPOINT_DIR,
        use_accelerate=True,      # ⚡ Enable multi-GPU training
        mixed_precision='fp16'    # ⚡ FP16 for T4 GPUs
    )
    
    # =========================
    # 5. 保存最終模型
    # =========================
    final_path = os.path.join(CHECKPOINT_DIR, 'final_model.pt')
    
    # Get unwrapped model if using DDP
    try:
        from accelerate import Accelerator
        accelerator = Accelerator()
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save(unwrapped_model.state_dict(), final_path)
    except:
        torch.save(model.state_dict(), final_path)
        
    print(f"\n💾 Final model saved to: {final_path}")
    
    print("\n" + "=" * 80)
    print("🎉 Training Complete!")
    print("=" * 80)
    print(f"Best validation loss: {history['best_val_loss']:.4f}")
    print(f"Checkpoints saved in: {CHECKPOINT_DIR}")
    print("=" * 80 + "\n")


# ============================================================================
# Multi-GPU Launcher
# ============================================================================

def launch_training(num_gpus=2):
    """
    Launch multi-GPU training with notebook_launcher
    
    Args:
        num_gpus: Number of GPUs (2 for Kaggle T4 x2)
    """
    import os
    os.environ['MASTER_PORT'] = '29500'
    
    print("\n" + "=" * 80)
    print("🚀 Launching Multi-GPU Training")
    print("=" * 80)
    print(f"Target GPUs: {num_gpus}")
    print(f"Mixed Precision: FP16")
    print("\n⚠️  IMPORTANT: If errors occur, restart the session!")
    print("   Kaggle: Run → Restart Session")
    print("=" * 80 + "\n")
    
    try:
        from accelerate import notebook_launcher
        notebook_launcher(main, args=(), num_processes=num_gpus)
    except ImportError:
        print("⚠️  notebook_launcher not available, running single-GPU")
        main()


if __name__ == "__main__":
    # For Kaggle T4 x2, use dual-GPU training
    launch_training(num_gpus=2)
    
    # For single-GPU, use:
    # main()

