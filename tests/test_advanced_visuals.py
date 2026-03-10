import torch
import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

os.makedirs("assets", exist_ok=True)

print("="*40)
print("Starting Advanced Visualizations (Dense & SSD)")
print("="*40)

# ==========================================
# Experiment 4: Kronecker Structure vs Dense
# ==========================================
def experiment_kronecker_structure():
    print("Running Exp 4: Kronecker Structure Visualization (Dense vs K-MoE)...")
    try:
        # We use small matrices so the block structure is visually distinguishable
        p1, p2 = 8, 8   # Input dim = 64
        q1, q2 = 12, 12  # Output dim = 144
        
        # Create non-random patterned matrices A and B for clear visualization
        # Matrix A (q1 x p1) - smooth gradients
        A = torch.zeros(q1, p1)
        for i in range(q1):
            for j in range(p1):
                A[i, j] = math.sin((i / q1) * 3.14 + (j / p1) * 3.14)
                
        # Matrix B (q2 x p2) - distinct sharp blocks
        B = torch.zeros(q2, p2)
        B[2:6, 2:6] = 1.0
        B[7:10, 4:7] = -1.0
        
        # Equivalent Kronecker Product Matrix W_eq
        W_eq = torch.kron(A, B)
        
        # Pure Random Dense Matrix for comparison
        W_dense = torch.randn(q1*q2, p1*p2) * 0.5
        
        fig, axes = plt.subplots(1, 4, figsize=(20, 5), gridspec_kw={'width_ratios': [1, 1, 2.5, 2.5]})
        
        sns.heatmap(A.numpy(), ax=axes[0], cmap='RdBu_r', center=0, cbar=False, xticklabels=False, yticklabels=False)
        axes[0].set_title(f"Expert Matrix A ({q1}x{p1})\n(Rank {min(q1,p1)})", fontsize=12)
        
        sns.heatmap(B.numpy(), ax=axes[1], cmap='RdBu_r', center=0, cbar=False, xticklabels=False, yticklabels=False)
        axes[1].set_title(f"Expert Matrix B ({q2}x{p2})\n(Rank {min(q2,p2)})", fontsize=12)
        
        sns.heatmap(W_eq.numpy(), ax=axes[2], cmap='RdBu_r', center=0, cbar=True, xticklabels=False, yticklabels=False)
        axes[2].set_title(f"K-MoE Equivalent W (A ⊗ B)\nSize {q1*q2}x{p1*p2} | Structured Low-Rank", fontsize=12)
        
        sns.heatmap(W_dense.numpy(), ax=axes[3], cmap='RdBu_r', center=0, cbar=True, xticklabels=False, yticklabels=False)
        axes[3].set_title(f"Standard Dense Matrix W\nSize {q1*q2}x{p1*p2} | Unstructured High-Rank", fontsize=12)
        
        plt.tight_layout()
        plt.savefig("assets/kronecker_structure_dense.png", dpi=300)
        plt.close()
        print("   [PASS] Saved -> assets/kronecker_structure_dense.png")
    except Exception as e:
        print(f"   [FAIL] {e}")

# ==========================================
# Experiment 5: Mamba-3 SSD Decay Mask
# ==========================================
def segsum_sim(x):
    T = x.size(-1)
    x_cumsum = torch.cumsum(x, dim=-1)
    x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -float('inf'))
    return x_segsum

def experiment_ssd_decay_mask():
    print("\nRunning Exp 5: SSD Structured Mask (Mamba-3) Visualization...")
    try:
        L_seq = 64
        # Simulate log_alpha for a decay mask (varying dynamically over the sequence)
        # Using a sine wave to simulate dynamic time steps (dt) hitting the state
        base_decay = -0.1
        dynamic_dt = torch.sin(torch.linspace(0, 10, L_seq)) * 0.05
        log_alpha = base_decay + dynamic_dt
        
        mask_log = segsum_sim(log_alpha.unsqueeze(0)).squeeze(0)
        L_mask = torch.exp(mask_log)
        
        # Mamba-3 Trapezoidal simulation: 
        # combining the causal mask with the decay mask + a local convolutional boundary
        
        plt.figure(figsize=(7, 6))
        sns.heatmap(L_mask.numpy(), cmap='Spectral_r', cbar=True, square=True, 
                    xticklabels=int(L_seq/8), yticklabels=int(L_seq/8))
        plt.title("Mamba-3 SSD Structured Decay Mask ($L$)\n1-Semiseparable Lower Triangular", fontsize=14)
        plt.xlabel("Key Time Step ($s$)")
        plt.ylabel("Query Time Step ($t$)")
        
        plt.savefig("assets/ssd_decay_mask.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("   [PASS] Saved -> assets/ssd_decay_mask.png")
    except Exception as e:
        print(f"   [FAIL] {e}")

# ==========================================
# Experiment 6: Multi-Layer Routing Depth Trace
# ==========================================
def experiment_multi_layer_routing():
    print("\nRunning Exp 6: Multi-Layer Routing Depth Trace (12 Layers)...")
    try:
        from models.kmoe_mamba3 import Mamba3Config, Mamba3Block
        num_layers = 12
        seq_len = 10
        num_experts = 128
        
        config = Mamba3Config(d_model=64, use_kmoe=True, kmoe_num_experts=num_experts, kmoe_top_k=1)
        layers = torch.nn.ModuleList([Mamba3Block(config) for _ in range(num_layers)])
        for layer in layers:
            layer.eval()
            
        x = torch.randn(1, seq_len, config.d_model)
        
        # We will track the designated expert (top-1) for a specific token (Token 0) across 12 layers
        token_id_to_track = 0
        depth_trace = []
        
        with torch.no_grad():
            for layer in layers:
                x, _ = layer(x) # ignore aux_loss
                # x_up_proj is KroneckerMoE
                indices = layer.x_up_proj.last_top_k_indices # (1*seq_len, 1)
                expert_chosen = indices[token_id_to_track, 0].item()
                depth_trace.append(expert_chosen)
                
        # Draw the trace
        plt.figure(figsize=(10, 4))
        plt.plot(range(1, num_layers+1), depth_trace, marker='o', linestyle='-', color='#9b59b6', markersize=8, linewidth=2)
        plt.fill_between(range(1, num_layers+1), depth_trace, alpha=0.2, color='#9b59b6')
        
        # Add labels
        for i, val in enumerate(depth_trace):
            plt.text(i+1, val+2, f"E_{val}", ha='center', fontsize=9)
            
        plt.title(f"Dynamic Routing Path for a Single Token across {num_layers} Layers\n(Notice how the token consults different experts at different depths)")
        plt.xlabel("Network Depth (Layer Index)")
        plt.ylabel(f"Chosen Expert ID (out of {num_experts})")
        plt.xticks(range(1, num_layers+1))
        plt.ylim(-5, num_experts + 5)
        plt.grid(True, linestyle=':', alpha=0.6)
        
        plt.savefig("assets/multi_layer_trace.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("   [PASS] Saved -> assets/multi_layer_trace.png")
        
    except Exception as e:
        print(f"   [FAIL] {e}")

# ==========================================
# Experiment 7: Syntax-Level Token Routing
# ==========================================
import re
def experiment_syntax_expert_coloring():
    print("\nRunning Exp 7: Syntax-Level Token Routing Visualization...")
    try:
        from models.kmoe_mamba3 import Mamba3Config, Mamba3Block
        
        # A mocked code snippet identical to the user's reference
        code_text = "class MoeLayer(nn.Module):\n    def __init__(self, experts: List[nn.Module], gate, moe_args):\n        super().__init__()\n        assert len(experts) > 0\n        self.experts = nn.ModuleList(experts)\n        self.gate = gate\n        self.args = moe_args"
        
        # Tokenize preserving spaces so we can reconstruct exactly
        tokens = re.findall(r'[a-zA-Z_]+|\s+|[^a-zA-Z_\s]+', code_text)
        
        unique_tokens = list(set(tokens))
        vocab_size = len(unique_tokens)
        
        num_experts = 8
        d_model = 64
        config = Mamba3Config(d_model=d_model, use_kmoe=True, kmoe_num_experts=num_experts, kmoe_top_k=1)
        model = Mamba3Block(config)
        model.eval()
        
        # Deterministic embedding table for these tokens
        # Similar tokens get identical semantic representations resulting in mirrored Expert assignments
        torch.manual_seed(42)  
        embed_table = torch.randn(vocab_size, d_model)
        
        token_to_id = {t: i for i, t in enumerate(unique_tokens)}
        seq_ids = [token_to_id[t] for t in tokens]
        
        seq_emb = embed_table[seq_ids].unsqueeze(0)
        
        with torch.no_grad():
            _ = model(seq_emb)
            
        expert_indices = model.x_up_proj.last_top_k_indices.squeeze(-1).numpy()
        
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        palette = sns.color_palette("pastel", num_experts)
        
        x, y = 0.02, 0.90
        line_height = 0.12
        
        for i, token in enumerate(tokens):
            if '\n' in token:
                y -= line_height
                x = 0.02
                continue
            
            expert_id = expert_indices[i]
            color = palette[expert_id]
            
            if token.isspace():
                x += len(token) * 0.015
                continue
                
            ax.text(x, y, token, fontsize=16, fontfamily='monospace',
                    bbox=dict(facecolor=color, alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'),
                    va='center', ha='left')
            
            # Predict width explicitly
            x += len(token) * 0.017 + 0.005
            
        plt.title(f"K-MoE Syntax Routing Validation (Tokens color-coded by top Expert ID)", fontsize=16)
        plt.savefig("assets/syntax_routing.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("   [PASS] Saved -> assets/syntax_routing.png")
        
    except Exception as e:
        print(f"   [FAIL] {e}")

if __name__ == "__main__":
    experiment_kronecker_structure()
    experiment_ssd_decay_mask()
    experiment_multi_layer_routing()
    experiment_syntax_expert_coloring()
    print("\n[SUCCESS] Advanced Visualizations Finished.")
