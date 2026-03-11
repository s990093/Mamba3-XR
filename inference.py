import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
import argparse
import time
import math

# ==========================================
# 1. Model Architecture (Optimized with KV Cache)
# ==========================================

class Mamba3Config:
    def __init__(self, d_model=768, d_state=64, d_head=64, n_groups=1, mimo_rank=4,
                 expand=4, num_layers=5, use_kmoe=True, kmoe_num_experts=256, kmoe_top_k=2):
        self.d_model = d_model
        self.d_state = d_state
        self.d_head = d_head
        self.expand = expand
        self.num_layers = num_layers # Number of macro-blocks (4:1)
        self.d_inner = int(expand * d_model)
        self.n_heads = self.d_inner // d_head
        self.n_groups = n_groups
        self.mimo_rank = mimo_rank
        self.use_kmoe = use_kmoe
        self.kmoe_num_experts = kmoe_num_experts
        self.kmoe_top_k = kmoe_top_k
        self.rms_norm_eps = 1e-5
        self.use_parallel_scan = False 
        self.use_conv = False
        self.A_init_range = (1, 16)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        rrms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rrms * self.weight

class KroneckerMoE(nn.Module):
    def __init__(self, dim_in1, dim_in2, dim_out1, dim_out2, num_experts=256, top_k=2):
        super().__init__()
        self.dim_in1, self.dim_in2 = dim_in1, dim_in2
        self.dim_out1, self.dim_out2 = dim_out1, dim_out2
        self.num_experts, self.top_k = num_experts, top_k
        self.router = nn.Linear(dim_in1 * dim_in2, num_experts, bias=False)
        self.A_experts = nn.Parameter(torch.randn(num_experts, dim_out1, dim_in1))
        self.B_experts = nn.Parameter(torch.randn(num_experts, dim_out2, dim_in2))
        self.register_parameter('scale', nn.Parameter(torch.ones(1)))
        self.bias = nn.Parameter(torch.zeros(dim_out1 * dim_out2))

    def forward(self, x):
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.dim_in1 * self.dim_in2)
        logits = self.router(x_flat)
        top_k_vals, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        top_k_probs = torch.softmax(top_k_vals, dim=-1)
        
        x_sub = x_flat.reshape(-1, self.dim_in1, self.dim_in2)
        flat_indices = top_k_indices.flatten()
        A_gathered = self.A_experts[flat_indices]
        B_gathered = self.B_experts[flat_indices]
        tokens_expanded = x_sub.unsqueeze(1).expand(-1, self.top_k, -1, -1).reshape(-1, self.dim_in1, self.dim_in2)
        
        Y = torch.einsum('noi, nij, npj -> nop', A_gathered, tokens_expanded, B_gathered)
        Y = Y.reshape(-1, self.top_k, self.dim_out1, self.dim_out2)
        output = (Y * top_k_probs.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
        
        output = output.reshape(*orig_shape[:-1], -1)
        return output * self.scale + self.bias, None

class Mamba3Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        H, G, P, N, R = config.n_heads, config.n_groups, config.d_head, config.d_state, config.mimo_rank
        self.ratio = H // G
        d_proj = H*P*2 + G*N*R*2 + G*3
        self.in_proj = nn.Linear(config.d_model, d_proj)
        
        def get_factors(n):
            for i in range(int(n**0.5), 0, -1):
                if n % i == 0: return i, n // i
            return 1, n
        
        p1, p2 = get_factors(P)
        q1, q2 = get_factors(P * R)
        self.x_up_proj = KroneckerMoE(p1, p2, q1, q2, config.kmoe_num_experts, config.kmoe_top_k)
        self.y_down_proj = nn.Linear(P * R, P, bias=False)
        self.out_proj = KroneckerMoE(*get_factors(config.d_inner), *get_factors(config.d_model), config.kmoe_num_experts, config.kmoe_top_k)
        
        self.theta_log = nn.Parameter(torch.randn(G, N // 2))
        self.D = nn.Parameter(torch.ones(H))
        self.norm_B = RMSNorm(N * R)
        self.norm_C = RMSNorm(N * R)
        self.bias_B = nn.Parameter(torch.zeros(G, N, R))
        self.bias_C = nn.Parameter(torch.zeros(G, N, R))
        self.pre_gate_norm = RMSNorm(H * P)

    def forward(self, u, state=None):
        B, L, _ = u.shape
        H, G, P, N, R = self.config.n_heads, self.config.n_groups, self.config.d_head, self.config.d_state, self.config.mimo_rank
        
        proj = self.in_proj(u)
        z, x_p, B_p, C_p, dt, A_p, lam = torch.split(proj, [H*P, H*P, G*N*R, G*N*R, G, G, G], dim=-1)
        
        dt = F.softplus(dt)
        A = -torch.exp(A_p)
        theta = torch.exp(self.theta_log)
        
        # Norms
        B_p = self.norm_B(B_p.reshape(B, L, G, N*R)).view(B, L, G, N, R) + self.bias_B
        C_p = self.norm_C(C_p.reshape(B, L, G, N*R)).view(B, L, G, N, R) + self.bias_C
        
        # Handle state/cache in G-space to avoid redundant expansion
        if state is None:
            # angles_g shape: (B, L, G, N/2)
            angles_g = torch.cumsum(dt.unsqueeze(-1) * theta.unsqueeze(0).unsqueeze(0), dim=1)
            h = torch.zeros(B, H, N, P, device=u.device)
            
            # Broadcast everything to H-space for computation
            angles = angles_g.repeat_interleave(self.ratio, dim=2)
            dt_h = dt.unsqueeze(-1).repeat_interleave(self.ratio, dim=2)
            A_h = A.unsqueeze(-1).repeat_interleave(self.ratio, dim=2)
            B_h = B_p.repeat_interleave(self.ratio, dim=2)
            C_h = C_p.repeat_interleave(self.ratio, dim=2)
            
            cos, sin = torch.cos(angles).unsqueeze(-1), torch.sin(angles).unsqueeze(-1)
            def rotate(v):
                v_re, v_im = v[..., 0::2, :], v[..., 1::2, :]
                return torch.stack([v_re*cos - v_im*sin, v_re*sin + v_im*cos], dim=-2).reshape_as(v)
            
            B_rot, C_rot = rotate(B_h), rotate(C_h)
            x, _ = self.x_up_proj(x_p.view(B, L, H, P))
            x = x.view(B, L, H, P, R)
            
            y_list = []
            alpha = torch.exp(dt_h * A_h).unsqueeze(-1)
            for t in range(L):
                in_sig = torch.einsum('bhnr, bhpr -> bhnp', B_rot[:, t], x[:, t])
                h = h * alpha[:, t] + dt_h[:, t].unsqueeze(-1) * in_sig
                y_t = torch.einsum('bhnp, bhnr -> bhpr', h, C_rot[:, t])
                y_list.append(y_t)
            y = torch.stack(y_list, dim=1)
            new_state = (h, angles_g[:, -1:]) 
        else:
            h_prev, angles_g_prev = state
            # Update angles in G-space
            angles_g = angles_g_prev + dt.unsqueeze(-1) * theta.unsqueeze(0).unsqueeze(0)
            
            # Broadcast to H-space
            angles = angles_g.repeat_interleave(self.ratio, dim=2)
            dt_h = dt.unsqueeze(-1).repeat_interleave(self.ratio, dim=2)
            A_h = A.unsqueeze(-1).repeat_interleave(self.ratio, dim=2)
            B_h = B_p.repeat_interleave(self.ratio, dim=2)
            C_h = C_p.repeat_interleave(self.ratio, dim=2)
            
            cos, sin = torch.cos(angles).unsqueeze(-1), torch.sin(angles).unsqueeze(-1)
            def rotate(v):
                v_re, v_im = v[..., 0::2, :], v[..., 1::2, :]
                return torch.stack([v_re*cos - v_im*sin, v_re*sin + v_im*cos], dim=-2).reshape_as(v)
            
            B_rot, C_rot = rotate(B_h), rotate(C_h)
            x, _ = self.x_up_proj(x_p.view(B, L, H, P))
            x = x.view(B, L, H, P, R)
            
            alpha = torch.exp(dt_h * A_h).unsqueeze(-1)
            in_sig = torch.einsum('bhnr, bhpr -> bhnp', B_rot[:, 0], x[:, 0])
            h = h_prev * alpha[:, 0] + dt_h[:, 0].unsqueeze(-1) * in_sig
            y = torch.einsum('bhnp, bhnr -> bhpr', h, C_rot[:, 0]).unsqueeze(1)
            new_state = (h, angles_g)

        y = self.y_down_proj(y.view(B, L, H, P*R)).view(B, L, H*P)
        y = self.pre_gate_norm(y + x_p.view(B, L, H*P) * self.D.repeat_interleave(P)) * torch.sigmoid(z)
        out, _ = self.out_proj(y)
        return out, new_state

class KMoEFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        def get_factors(n):
            for i in range(int(n**0.5), 0, -1):
                if n % i == 0: return i, n // i
            return 1, n
        d1, d2 = get_factors(config.d_model)
        f1, f2 = get_factors(config.d_model * 4)
        self.up = KroneckerMoE(d1, d2, f1, f2, config.kmoe_num_experts, config.kmoe_top_k)
        self.down = KroneckerMoE(f1, f2, d1, d2, config.kmoe_num_experts, config.kmoe_top_k)
    def forward(self, x):
        h, _ = self.up(x)
        y, _ = self.down(F.gelu(h))
        return y, None

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attn = nn.MultiheadAttention(config.d_model, config.d_model // 64, batch_first=True)
        self.ffn = KMoEFeedForward(config)
        self.norm1 = RMSNorm(config.d_model)
        self.norm2 = RMSNorm(config.d_model)

    def forward(self, x, cache=None):
        B, L, D = x.shape
        nx = self.norm1(x)
        
        if cache is not None:
            # Optimized for single-token generation
            attn_out, _ = self.attn(nx, cache[0], cache[1])
            new_cache = (torch.cat([cache[0], nx], dim=1), torch.cat([cache[1], nx], dim=1))
        else:
            mask = torch.triu(torch.ones(L, L, device=x.device) * float('-inf'), diagonal=1)
            attn_out, _ = self.attn(nx, nx, nx, attn_mask=mask)
            new_cache = (nx, nx)
            
        x = x + attn_out
        f, _ = self.ffn(self.norm2(x))
        return x + f, new_cache

class Mamba3LanguageModel(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(vocab_size, config.d_model)
        self.layers = nn.ModuleList()
        for _ in range(config.num_layers):
            for _ in range(4): self.layers.append(Mamba3Block(config))
            self.layers.append(TransformerBlock(config))
        self.norm = RMSNorm(config.d_model)
        self.head = nn.Linear(config.d_model, vocab_size, bias=False)
        self.head.weight = self.embed.weight

    def get_param_stats(self):
        total_params = sum(p.numel() for p in self.parameters())
        
        # Calculate active params (Non-MoE + top-k MoE)
        active_params = 0
        
        # 1. Embeddings & Head (Weight tied)
        active_params += self.embed.weight.numel()
        active_params += self.norm.weight.numel()
        
        # 2. Layers
        for layer in self.layers:
            if isinstance(layer, Mamba3Block):
                # Non-MoE parts of Mamba3Block
                active_params += layer.in_proj.weight.numel() + layer.in_proj.bias.numel()
                active_params += layer.y_down_proj.weight.numel()
                active_params += layer.theta_log.numel() + layer.D.numel()
                active_params += layer.norm_B.weight.numel() + layer.norm_C.weight.numel()
                active_params += layer.bias_B.numel() + layer.bias_C.numel()
                active_params += layer.pre_gate_norm.weight.numel()
                
                # MoE parts (only top-k experts)
                # x_up_proj
                moe = layer.x_up_proj
                active_params += moe.router.weight.numel()
                active_params += moe.scale.numel() + moe.bias.numel()
                # Active experts: top_k * (A + B)
                active_params += moe.top_k * (moe.A_experts[0].numel() + moe.B_experts[0].numel())
                
                # out_proj
                moe = layer.out_proj
                active_params += moe.router.weight.numel()
                active_params += moe.scale.numel() + moe.bias.numel()
                active_params += moe.top_k * (moe.A_experts[0].numel() + moe.B_experts[0].numel())
                
            elif isinstance(layer, TransformerBlock):
                # Attention part
                active_params += sum(p.numel() for p in layer.attn.parameters())
                active_params += layer.norm1.weight.numel() + layer.norm2.weight.numel()
                
                # MoE FFN
                moe_up = layer.ffn.up
                active_params += moe_up.router.weight.numel()
                active_params += moe_up.scale.numel() + moe_up.bias.numel()
                active_params += moe_up.top_k * (moe_up.A_experts[0].numel() + moe_up.B_experts[0].numel())
                
                moe_down = layer.ffn.down
                active_params += moe_down.router.weight.numel()
                active_params += moe_down.scale.numel() + moe_down.bias.numel()
                active_params += moe_down.top_k * (moe_down.A_experts[0].numel() + moe_down.B_experts[0].numel())
                
        return total_params, active_params

    def forward(self, ids, states=None):
        x = self.embed(ids)
        new_states = []
        state_idx = 0
        for layer in self.layers:
            current_state = states[state_idx] if states is not None else None
            out, next_state = layer(x, current_state)
            new_states.append(next_state)
            x = x + out if isinstance(layer, Mamba3Block) else out
            state_idx += 1
        return self.head(self.norm(x)), new_states

# ==========================================
# 2. Optimized Inference Logic
# ==========================================

def get_memory_usage():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**2
    elif torch.backends.mps.is_available():
        # MPS doesn't have a direct equivalent to max_memory_allocated yet
        return 0.0 
    return 0.0

def generate(model, tokenizer, prompt, max_len=100, temp=0.7, top_k=40, top_p=0.9, 
             repetition_penalty=1.1, presence_penalty=0.0, frequency_penalty=0.0, 
             stop_sequences=None, device='cuda'):
    model.eval()
    input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
    
    # --- Pre-fill Phase ---
    start_time = time.time()
    logits, states = model(input_ids)
    prefill_time = time.time() - start_time
    
    generated_ids = input_ids[0].tolist()
    next_token_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    
    # Tracking for frequency/presence penalties
    token_counts = {}
    for tid in generated_ids:
        token_counts[tid] = token_counts.get(tid, 0) + 1
    
    # --- Incremental Generation Phase ---
    tokens_generated = 0
    gen_start_time = time.time()
    
    print("\n" + "="*30 + " GENERATION " + "="*30)
    print(tokenizer.decode(generated_ids), end='', flush=True)

    for step in range(max_len):
        with torch.no_grad():
            logits, states = model(next_token_id, states)
            next_token_logits = logits[:, -1, :] / temp
            
            # 1. Repetition Penalty
            # We apply penalty to all tokens that have appeared so far
            for tid, count in token_counts.items():
                if next_token_logits[0, tid] > 0:
                    next_token_logits[0, tid] /= repetition_penalty
                else:
                    next_token_logits[0, tid] *= repetition_penalty
                
                # 2. Presence & Frequency Penalty
                # Presence: penalty if token has appeared at least once
                # Frequency: penalty scales with how many times it appeared
                next_token_logits[0, tid] -= (presence_penalty + count * frequency_penalty)

            # Robust sampling
            next_token_logits = torch.nan_to_num(next_token_logits, nan=-100.0, posinf=-100.0, neginf=-100.0)
            
            # 3. Top-K Filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('Inf')

            # 4. Top-P (Nucleus) Filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[0, indices_to_remove] = -float('Inf')

            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            if probs.sum() <= 0:
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                next_token_id = torch.multinomial(probs, 1)
            
            tid = next_token_id.item()
            generated_ids.append(tid)
            token_counts[tid] = token_counts.get(tid, 0) + 1
            tokens_generated += 1
            
            word = tokenizer.decode([tid])
            print(word, end='', flush=True)
            
            # 5. Stop Sequence Detection
            if tid == tokenizer.eos_id(): break
            if stop_sequences:
                current_text = tokenizer.decode(generated_ids)
                if any(stop_seq in current_text for stop_seq in stop_sequences):
                    break
            
    total_time = time.time() - start_time
    gen_time = time.time() - gen_start_time
    
    # Metrics
    throughput = tokens_generated / gen_time if gen_time > 0 else 0
    mem_mb = get_memory_usage()
    total_p, active_p = model.get_param_stats()
    
    print("\n" + "="*72)
    print(f"\n🚀 Performance Summary:")
    print(f"├─ Total Time      : {total_time:.2f}s")
    print(f"├─ Pre-fill Time   : {prefill_time:.4f}s")
    print(f"├─ Inference Time  : {gen_time:.4f}s (Generation phase)")
    print(f"├─ Gen Speed       : {throughput:.2f} tokens/sec")
    print(f"├─ Tokens Generated: {tokens_generated}")
    print(f"├─ Total Params    : {total_p / 1e6:.2f}M")
    print(f"├─ Active Params   : {active_p / 1e6:.2f}M ({(active_p / total_p * 100):.1f}% of total)")
    print(f"└─ GPU Memory Use  : {mem_mb:.2f} MB" if mem_mb > 0 else "└─ Memory info N/A on this device")
    
    return tokenizer.decode(generated_ids)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--spm", type=str, default="data/spm_tokenizer.model", help="Path to .model file")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Input prompt")
    parser.add_argument("--steps", type=int, default=100, help="Max generation steps")
    parser.add_argument("--top_k", type=int, default=40, help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p (nucleus) sampling parameter")
    parser.add_argument("--temp", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--rep_penalty", type=float, default=1.1, help="Repetition penalty")
    parser.add_argument("--pres_penalty", type=float, default=0.0, help="Presence penalty")
    parser.add_argument("--freq_penalty", type=float, default=0.0, help="Frequency penalty")
    parser.add_argument("--stop", type=str, nargs="+", default=None, help="Stop sequences")
    args = parser.parse_args()

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f" usando device: {device}")

    sp = spm.SentencePieceProcessor(model_file=args.spm)
    config = Mamba3Config(num_layers=5, kmoe_num_experts=256) 
    model = Mamba3LanguageModel(config, vocab_size=len(sp))
    
    print(f"Loading checkpoint from {args.ckpt}...")
    ckpt = torch.load(args.ckpt, map_location=device)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    result = generate(model, sp, args.prompt, max_len=args.steps, temp=args.temp, 
                      top_k=args.top_k, top_p=args.top_p, 
                      repetition_penalty=args.rep_penalty, 
                      presence_penalty=args.pres_penalty, 
                      frequency_penalty=args.freq_penalty,
                      stop_sequences=args.stop, device=device)
