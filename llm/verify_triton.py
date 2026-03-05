#!/usr/bin/env python3
"""
Triton Kernel Verification Script

Tests the correctness and performance of the Triton-accelerated 
chunk_parallel_scan implementation.
"""

import torch
import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import from the main script
print("Loading Mamba-3 implementation...")
exec(open('mamba3_shakespeare_kaggle.py').read())

def test_triton_correctness():
    """Test that Triton kernel produces same results as PyTorch"""
    print("\n" + "="*80)
    print("TEST 1: Correctness Verification")
    print("="*80)
    
    if not TRITON_AVAILABLE:
        print("❌ Triton not available. Skipping test.")
        return False
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Skipping test.")
        return False
    
    device = 'cuda'
    dtype = torch.float16
    
    # Test parameters
    B, C, H, N, P = 2, 4, 8, 32, 64
    
    print(f"Test shape: B={B}, C={C}, H={H}, N={N}, P={P}")
    
    # Generate random test data
    torch.manual_seed(42)
    x = torch.randn(B, C, H, N, P, device=device, dtype=dtype)
    decay = torch.rand(B, C, H, device=device, dtype=dtype) * 0.5 + 0.5  # [0.5, 1.0]
    
    # Triton version
    print("Running Triton kernel...")
    h_triton = triton_inter_chunk_scan(x, decay)
    
    # PyTorch reference (manual computation)
    print("Running PyTorch reference...")
    h_pytorch = torch.zeros(B, C, H, N, P, device=device, dtype=dtype)
    h_acc = torch.zeros(B, H, N, P, device=device, dtype=dtype)
    
    for c in range(C):
        h_acc = h_acc * decay[:, c, :].view(B, H, 1, 1) + x[:, c, :, :, :]
        h_pytorch[:, c, :, :, :] = h_acc
    
    # Compare
    max_diff = (h_triton - h_pytorch).abs().max().item()
    mean_diff = (h_triton - h_pytorch).abs().mean().item()
    
    print(f"\nResults:")
    print(f"  Max difference:  {max_diff:.6e}")
    print(f"  Mean difference: {mean_diff:.6e}")
    
    # Check tolerance (relaxed for FP16)
    rtol, atol = 1e-3, 1e-3
    passed = torch.allclose(h_triton, h_pytorch, rtol=rtol, atol=atol)
    
    if passed:
        print(f"✅ PASSED (rtol={rtol}, atol={atol})")
    else:
        print(f"❌ FAILED (rtol={rtol}, atol={atol})")
        print(f"\nSample values:")
        print(f"  Triton:  {h_triton[0, 0, 0, 0, :5]}")
        print(f"  PyTorch: {h_pytorch[0, 0, 0, 0, :5]}")
    
    return passed


def benchmark_performance():
    """Benchmark Triton vs PyTorch performance"""
    print("\n" + "="*80)
    print("TEST 2: Performance Benchmark")
    print("="*80)
    
    if not TRITON_AVAILABLE or not torch.cuda.is_available():
        print("❌ Triton or CUDA not available. Skipping benchmark.")
        return
    
    device = 'cuda'
    dtype = torch.float16
    
    # Realistic parameters for Shakespeare training
    B, C, H, N, P = 32, 2, 8, 32, 64  # 2 chunks of 256 tokens
    
    print(f"Benchmark shape: B={B}, C={C}, H={H}, N={N}, P={P}")
    
    x = torch.randn(B, C, H, N, P, device=device, dtype=dtype)
    decay = torch.rand(B, C, H, device=device, dtype=dtype) * 0.5 + 0.5
    
    # Warmup
    for _ in range(10):
        _ = triton_inter_chunk_scan(x, decay)
    torch.cuda.synchronize()
    
    # Benchmark Triton
    print("\nBenchmarking Triton kernel...")
    num_iters = 100
    start = time.time()
    for _ in range(num_iters):
        h_triton = triton_inter_chunk_scan(x, decay)
    torch.cuda.synchronize()
    triton_time = (time.time() - start) / num_iters * 1000  # ms
    
    # Benchmark PyTorch
    print("Benchmarking PyTorch reference...")
    start = time.time()
    for _ in range(num_iters):
        h_acc = torch.zeros(B, H, N, P, device=device, dtype=dtype)
        for c in range(C):
            h_acc = h_acc * decay[:, c, :].view(B, H, 1, 1) + x[:, c, :, :, :]
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start) / num_iters * 1000  # ms
    
    print(f"\nResults:")
    print(f"  Triton:  {triton_time:.3f} ms")
    print(f"  PyTorch: {pytorch_time:.3f} ms")
    print(f"  Speedup: {pytorch_time / triton_time:.2f}x")
    
    if triton_time < pytorch_time:
        print(f"✅ Triton is faster!")
    else:
        print(f"⚠️  PyTorch is faster (may be due to small problem size)")


def test_chunk_parallel_scan():
    """Test the full chunk_parallel_scan with Triton integration"""
    print("\n" + "="*80)
    print("TEST 3: Full chunk_parallel_scan Integration")
    print("="*80)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Skipping test.")
        return False
    
    device = 'cuda'
    
    # Create a small Mamba-3 config
    config = Mamba3Config(
        d_model=128,
        d_state=32,
        d_head=64,
        n_groups=2,
        mimo_rank=4,
        chunk_size=128
    )
    
    # Create Mamba3Block
    block = Mamba3Block(config).to(device).half()
    
    # Test input
    B, L = 4, 256
    u = torch.randn(B, L, config.d_model, device=device, dtype=torch.float16)
    
    print(f"Testing with input shape: {u.shape}")
    
    # Forward pass
    try:
        output = block(u)
        print(f"✅ Forward pass successful!")
        print(f"   Output shape: {output.shape}")
        
        # Check for NaNs
        if torch.isnan(output).any():
            print(f"❌ Output contains NaNs!")
            return False
        
        print(f"   Output range: [{output.min():.4f}, {output.max():.4f}]")
        return True
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*80)
    print("Triton Kernel Verification Suite")
    print("="*80)
    
    print(f"\nEnvironment:")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"  Triton available: {TRITON_AVAILABLE}")
    if TRITON_AVAILABLE:
        print(f"  Triton version: {triton.__version__}")
    
    # Run tests
    results = []
    
    results.append(("Correctness", test_triton_correctness()))
    benchmark_performance()
    results.append(("Integration", test_chunk_parallel_scan()))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for name, result in results:
        if result is None:
            status = "⏭️  SKIPPED"
        elif result:
            status = "✅ PASSED"
        else:
            status = "❌ FAILED"
        print(f"  {name:20s} {status}")
    
    print("="*80)
    
    # Exit code
    if all(r is not False for r in [r[1] for r in results]):
        print("\n🎉 All tests passed or skipped!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
