import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import io
import sys
import traceback

# Add project root to sys path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.kmoe_mamba3 import Mamba3Config, KroneckerMoE, Mamba3Block

# Ensure assets directory exists
os.makedirs("assets", exist_ok=True)

print("="*40)
print("Starting K-MoE Visualization Experiments")
print("="*40)

# ==========================================
# Experiment 1: Compression Ratio Comparison
# ==========================================
def experiment_compression_ratio():
    print("Running Experiment 1: Compression Ratio...")
    try:
        # Simulate an LLM model width
        D = 4096 
        expand_factor = 4
        # Dense Project (D -> D*expand)
        dense_params = D * (D * expand_factor)
        
        # K-MoE configuration matching the Dense expansion
        # D -> 4096
        p1, p2 = 64, 64 # p1*p2 = 4096
        
        # D*4 -> 16384
        q1, q2 = 128, 128 # q1*q2 = 16384
        
        # Assume huge number of experts
        N = 1024
        
        # Each expert parameter count
        expert_params = (q1 * p1) + (q2 * p2)
        total_kmoe_params = N * expert_params
        
        names = ["Dense Linear Layer", f"K-MoE ({N} Experts)"]
        values = [dense_params / 1e6, total_kmoe_params / 1e6] # Convert to Millions
        
        plt.figure(figsize=(8, 6))
        bars = plt.bar(names, values, color=['#e74c3c', '#3498db'])
        plt.title(f"Parameter Compression\n(D_in=4096, D_out=16384, N={N})")
        plt.ylabel("Parameters (Millions)")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f"{yval:.1f}M", ha='center', va='bottom', fontweight='bold')
            
        plt.savefig("assets/compression_ratio.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[PASS] Experiment 1 Completed. Saved to assets/compression_ratio.png")
        print(f"       Dense: {dense_params/1e6:.1f}M vs K-MoE: {total_kmoe_params/1e6:.1f}M")
    except Exception as e:
        print(f"[FAIL] Experiment 1 failed: {str(e)}")
        traceback.print_exc()

# ==========================================
# Experiment 2: MIMO / SVD Effective Rank
# ==========================================
def experiment_svd_rank():
    print("\nRunning Experiment 2: SVD Effective Rank Analysis...")
    try:
        # Create a single expert's A and B matrices
        p1, p2 = 32, 32   # D_in = 1024
        q1, q2 = 64, 64   # D_out = 4096
        
        std_A = (1.0 / math.sqrt(p1 * q1)) ** 0.5
        std_B = (1.0 / math.sqrt(p2 * q2)) ** 0.5
        
        A = torch.randn(q1, p1) * std_A
        B = torch.randn(q2, p2) * std_B
        
        # Calculate equivalent mathematical linear matrix: W_eq = A \otimes B 
        # Using built in torch kron for purely analytical purposes:
        W_eq = torch.kron(A, B) # (D_out, D_in) = (4096, 1024)
        
        # Run SVD
        print(f"       Computing SVD for equivalent matrix of size {W_eq.shape}...")
        U, S, V = torch.svd(W_eq)
        
        # Plot Top Singular Values
        S_np = S.numpy()
        
        plt.figure(figsize=(10, 5))
        plt.plot(S_np, color='#2ecc71', linewidth=2, label="Singular Values (Kronecker)")
        
        # The theoretical rank of Kron(A, B) is rank(A) * rank(B).
        # Since A is 64x32 (rank 32 usually), and B is 64x32 (rank 32), the Kron rank is limited exactly to 32*32=1024
        rank_A = min(q1, p1)
        rank_B = min(q2, p2)
        theoretical_rank = rank_A * rank_B
        
        plt.axvline(x=theoretical_rank, color='#e74c3c', linestyle='--', label=f"Theoretical Max Rank ({theoretical_rank})")
        plt.title(f"SVD Effective Rank Decay of a K-MoE Expert (D_in=1024, D_out=4096)")
        plt.xlabel("Singular Value Index")
        plt.ylabel("Magnitude")
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend()
        
        plt.savefig("assets/svd_rank_decay.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("[PASS] Experiment 2 Completed. Saved to assets/svd_rank_decay.png")
    except Exception as e:
        print(f"[FAIL] Experiment 2 failed: {str(e)}")
        traceback.print_exc()

# ==========================================
# Experiment 3: MoE Routing Distribution Heatmap
# ==========================================
def experiment_routing_heatmap():
    print("\nRunning Experiment 3: MoE Routing Heatmap...")
    try:
        # Create a small Mamba configuration
        config = Mamba3Config(d_model=128, use_kmoe=True, kmoe_num_experts=128, kmoe_top_k=2)
        model = Mamba3Block(config)
        model.eval() # Must put in eval to capture indices
        
        # Feed some random sequence data (simulating 4 long phrases)
        B_sz = 4
        Seq_Len = 64
        x = torch.randn(B_sz, Seq_Len, config.d_model)
        
        with torch.no_grad():
            _ = model(x)
            
        # Extract indices. x_up_proj is an instance of KroneckerMoE
        indices = model.x_up_proj.last_top_k_indices # (B_flat, K)
        
        # Construct Heatmap Matrix (Experts vs Tokens in sequence)
        # We will plot the first sequence in the batch for simplicity
        seq_idx = 0
        starts_at = seq_idx * Seq_Len
        ends_at = starts_at + Seq_Len
        seq_indices = indices[starts_at:ends_at, :] # (Seq_Len, K)
        
        heatmap_data = np.zeros((config.kmoe_num_experts, Seq_Len))
        
        # Populate hits
        for t in range(Seq_Len):
            expert1 = seq_indices[t, 0].item()
            expert2 = seq_indices[t, 1].item()
            heatmap_data[expert1, t] += 1
            heatmap_data[expert2, t] += 1
            
        # Draw Heatmap
        plt.figure(figsize=(14, 6))
        sns.heatmap(heatmap_data, cmap="viridis", cbar=False, 
                    xticklabels=int(Seq_Len/8), yticklabels=16)
        
        plt.title(f"Expert Routing Heatmap (Sequence Length {Seq_Len}, Top-2 over {config.kmoe_num_experts} Experts)")
        plt.xlabel("Token Index in Sequence")
        plt.ylabel("Expert ID")
        
        plt.savefig("assets/routing_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("[PASS] Experiment 3 Completed. Saved to assets/routing_heatmap.png")
    except Exception as e:
        print(f"[FAIL] Experiment 3 failed: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    experiment_compression_ratio()
    experiment_svd_rank()
    experiment_routing_heatmap()
    print("\n[SUCCESS] Visual Experiments Finished.")
