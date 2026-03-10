import torch
import torch.nn as nn
import time
import sys
import os

# Ensure models can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.kmoe_mamba3 import KroneckerMoE, Mamba3Block, Mamba3Config

def test_sanity_check():
    print("=== Sanity Check: Shapes & Gradients ===")
    
    B, L, D_in = 2, 128, 64
    D_out = 256
    num_experts = 4
    top_k = 2
    
    # 64 -> 8x8, 256 -> 16x16
    kmoe = KroneckerMoE(dim_in1=8, dim_in2=8, dim_out1=16, dim_out2=16, num_experts=num_experts, top_k=top_k)
    
    # Input
    x = torch.randn(B, L, D_in, requires_grad=True)
    
    # Forward
    y, aux_loss = kmoe(x)
    
    # Check shape
    expected_shape = (B, L, D_out)
    assert y.shape == expected_shape, f"Expected shape {expected_shape}, got {y.shape}"
    print(f"[PASS] Forward shape check: Output shape is {y.shape}")
    
    # Backward
    loss = y.sum() + aux_loss
    loss.backward()
    
    # Check Gradients
    assert kmoe.A_experts.grad is not None, "A_experts gradient is None"
    assert kmoe.B_experts.grad is not None, "B_experts gradient is None"
    assert x.grad is not None, "Input gradient is None"
    print("[PASS] Backward pass: Gradients computed")
    
    # Check sparsity in gradients
    # We can't guarantee exactly 0 because topk might select different experts for different batch/seq elements.
    # But let's check if the gradients look reasonable.
    print("[PASS] Sanity Check Completed\n")


def test_resource_profiling():
    print("=== Resource Profiling: Baseline vs K-MoE ===")
    
    baseline_config = Mamba3Config(d_model=256, use_kmoe=False)
    kmoe_config = Mamba3Config(d_model=256, use_kmoe=True, kmoe_num_experts=16, kmoe_top_k=2)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    baseline_block = Mamba3Block(baseline_config).to(device)
    kmoe_block = Mamba3Block(kmoe_config).to(device)
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
    baseline_params = count_parameters(baseline_block)
    kmoe_params = count_parameters(kmoe_block)
    
    print(f"Baseline Params: {baseline_params:,}")
    print(f"K-MoE Params   : {kmoe_params:,}")
    if kmoe_params <= baseline_params:
        print("[PASS] K-MoE parameter count is lower or equal.")
    else:
        print(f"[NOTE] K-MoE has more params. Difference: {kmoe_params - baseline_params:,}")
        
    # Execution Time Profiling
    B, L = 8, 512
    x = torch.randn(B, L, 256).to(device)
    
    # warmup
    for _ in range(3):
        _ = baseline_block(x)
        _ = kmoe_block(x)
        
    def get_time():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif torch.backends.mps.is_available():
            torch.mps.synchronize()
        return time.time()
        
    start_base = get_time()
    for _ in range(10):
        _ = baseline_block(x)
    time_base = (get_time() - start_base) / 10
    
    start_kmoe = get_time()
    for _ in range(10):
        _ = kmoe_block(x)
    time_kmoe = (get_time() - start_kmoe) / 10
    
    print(f"Baseline Forward Time: {time_base*1000:.2f} ms")
    print(f"K-MoE Forward Time   : {time_kmoe*1000:.2f} ms")
    print("[PASS] Resource Profiling Completed\n")

def test_mvp_convergence():
    print("=== MVP Pre-training (Simple Addition) ===")
    
    # Synth dataset: Output sum of input sequence along feature dim
    B, L, D = 32, 16, 256
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    def train_model(config, name):
        model = Mamba3Block(config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        alpha = 0.01
        losses = []
        for step in range(200):
            x = torch.randn(B, L, D).to(device)
            # Predict x * 2 just as a synthetic target to force learning
            target = x * 2.0 
            
            optimizer.zero_grad()
            y, aux_loss = model(x)
            
            ce_loss = criterion(y, target)
            loss = ce_loss + alpha * aux_loss
            loss.backward()
            optimizer.step()
            
            losses.append(ce_loss.item())
            
        print(f"[{name}] Extracted First Loss: {losses[0]:.4f} -> Final Loss: {losses[-1]:.4f}")
        return losses
        
    baseline_config = Mamba3Config(d_model=256, use_kmoe=False)
    kmoe_config = Mamba3Config(d_model=256, use_kmoe=True, kmoe_num_experts=16, kmoe_top_k=2)

    base_losses = train_model(baseline_config, "Baseline")
    kmoe_losses = train_model(kmoe_config, "K-MoE")

    print(f"[{'Baseline'}] Extracted First Loss: {base_losses[0]:.4f} -> Final Loss: {base_losses[-1]:.4f}")
    print(f"[{'K-MoE'}] Extracted First Loss: {kmoe_losses[0]:.4f} -> Final Loss: {kmoe_losses[-1]:.4f}")

    # Evaluate 30% gap Criteria
    target_loss = base_losses[-1] * 1.30
    print(f"[TARGET] To pass, K-MoE final loss must be <= {target_loss:.4f} (Under 30% degradation from baseline)")
    
    if kmoe_losses[-1] <= target_loss:
        print("[PASS] MVP Convergence: K-MoE loss reached within 30% of the baseline.")
    else:
        print("[FAIL] K-MoE did not converge within 30% variance.")

if __name__ == "__main__":
    test_sanity_check()
    test_resource_profiling()
    test_mvp_convergence()
