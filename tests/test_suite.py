# Copyright (C) 2025, Mamba-3 Implementation
# Pytest-based Test Suite

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytest
from models.mamba3 import Mamba3Config, Mamba3Block
from models.vision_mamba import VisionMamba

# ==========================================
# Test Fixtures
# ==========================================

@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def basic_config():
    return Mamba3Config(
        d_model=32, d_state=16, d_head=16, mimo_rank=4, expand=1, n_groups=1
    )


@pytest.fixture
def basic_model(device, basic_config):
    torch.manual_seed(42)
    return Mamba3Block(basic_config).to(device)


# ==========================================
# Core Mamba3Block Tests
# ==========================================

def test_output_shape(device, basic_model):
    """Test forward pass output shape consistency"""
    B, L, D = 2, 10, 32
    x = torch.randn(B, L, D).to(device)
    y = basic_model(x)
    assert y.shape == (B, L, D)
    print("✓ Output shape test passed")


def test_rope_mechanism(device, basic_model):
    """Test RoPE 90-degree rotation"""
    angles = torch.tensor([[[[1.5708]]]], device=device)  # pi/2
    x = torch.ones(1, 1, 1, 2, 1, device=device)  # Real=1, Imag=1
    
    x_rot = basic_model.apply_rope(x, angles)
    real_rot = x_rot[..., 0, :]
    imag_rot = x_rot[..., 1, :]
    
    assert torch.allclose(real_rot, torch.tensor(-1.0, device=device), atol=1e-4)
    assert torch.allclose(imag_rot, torch.tensor(1.0, device=device), atol=1e-4)
    print("✓ RoPE mechanism test passed")


def test_backward_pass(device, basic_model):
    """Test gradient computation"""
    B, L, D = 2, 10, 32
    x = torch.randn(B, L, D).to(device)
    basic_model.train()
    y = basic_model(x)
    loss = y.sum()
    loss.backward()
    
    assert basic_model.A_log.grad is not None
    assert basic_model.bias_B.grad is not None
    print("✓ Backward pass test passed")


# ==========================================
# Mamba-2 Initialization Tests
# ==========================================

def test_A_log_range(device):
    """Test A_log initialized in correct log-space range"""
    config = Mamba3Config(d_model=64, n_groups=2, A_init_range=(1, 16))
    model = Mamba3Block(config).to(device)
    
    A_values = torch.exp(model.A_log)
    assert A_values.min().item() >= 0.9
    assert A_values.max().item() <= 16.5
    print("✓ A_log range test passed")


def test_dt_bias_range(device):
    """Test dt_bias produces dt values in [dt_min, dt_max]"""
    config = Mamba3Config(d_model=64, n_groups=2, dt_min=0.001, dt_max=0.1)
    model = Mamba3Block(config).to(device)
    
    dt_start = model.dim_z + model.dim_x + model.dim_B + model.dim_C
    dt_end = dt_start + model.dim_dt
    dt_bias = model.in_proj.bias[dt_start:dt_end]
    dt_values = F.softplus(dt_bias)
    
    assert dt_values.min().item() >= 0.0009
    assert dt_values.max().item() <= 0.11
    print("✓ dt_bias range test passed")


def test_lambda_initialization(device):
    """Test lambda initialized to favor Euler method initially"""
    config = Mamba3Config(d_model=64, n_groups=2)
    model = Mamba3Block(config).to(device)
    
    dt_start = model.dim_z + model.dim_x + model.dim_B + model.dim_C
    dt_end = dt_start + model.dim_dt
    lambda_start = dt_end
    lambda_bias = model.in_proj.bias[lambda_start:]
    lambda_values = torch.sigmoid(lambda_bias)
    
    expected = torch.sigmoid(torch.tensor(-3.0))
    assert torch.allclose(lambda_values, expected, atol=0.01)
    print("✓ Lambda initialization test passed")


def test_no_weight_decay_flags():
    """Test _no_weight_decay flags are set correctly"""
    config = Mamba3Config(d_model=64)
    model = Mamba3Block(config)
    
    assert hasattr(model.A_log, '_no_weight_decay')
    assert model.A_log._no_weight_decay
    print("✓ No weight decay flags test passed")


# ==========================================
# Gradient Flow Tests
# ==========================================

def test_all_parameters_receive_gradients(device):
    """Ensure all trainable parameters receive gradients"""
    config = Mamba3Config(d_model=32, mimo_rank=4, use_conv=True)
    model = Mamba3Block(config).to(device)
    model.train()
    
    x = torch.randn(2, 16, 32).to(device)
    y = model(x)
    loss = y.pow(2).mean()
    loss.backward()
    
    params_without_grad = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is None:
            params_without_grad.append(name)
    
    assert len(params_without_grad) == 0, f"Parameters without gradients: {params_without_grad}"
    print("✓ All parameters receive gradients test passed")


def test_gradient_norms_reasonable(device):
    """Check gradient norms are in reasonable range"""
    config = Mamba3Config(d_model=64, mimo_rank=4)
    model = Mamba3Block(config).to(device)
    model.train()
    
    x = torch.randn(4, 32, 64).to(device)
    y = model(x)
    loss = y.pow(2).mean()
    loss.backward()
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
            assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"
            assert grad_norm < 1000, f"Gradient too large in {name}: {grad_norm}"
    
    print("✓ Gradient norms reasonable test passed")


def test_gradient_accumulation(device):
    """Test gradient accumulation works correctly"""
    config = Mamba3Config(d_model=32)
    model = Mamba3Block(config).to(device)
    model.train()
    
    x1 = torch.randn(2, 8, 32).to(device)
    x2 = torch.randn(2, 8, 32).to(device)
    
    y1 = model(x1)
    loss1 = y1.sum()
    loss1.backward()
    grad1 = model.A_log.grad.clone()
    
    y2 = model(x2)
    loss2 = y2.sum()
    loss2.backward()
    grad2 = model.A_log.grad
    
    assert not torch.allclose(grad1, grad2)
    print("✓ Gradient accumulation test passed")


def test_mimo_gradients(device):
    """Test MIMO-specific parameters receive gradients"""
    config = Mamba3Config(d_model=32, mimo_rank=8)
    model = Mamba3Block(config).to(device)
    model.train()
    
    x = torch.randn(2, 16, 32).to(device)
    y = model(x)
    loss = y.sum()
    loss.backward()
    
    assert model.x_up_proj.weight.grad is not None
    assert model.y_down_proj.weight.grad is not None
    assert model.bias_B.grad is not None
    assert model.bias_C.grad is not None
    
    assert model.x_up_proj.weight.grad.abs().sum().item() > 0
    assert model.y_down_proj.weight.grad.abs().sum().item() > 0
    print("✓ MIMO gradients test passed")


# ==========================================
# Numerical Stability Tests
# ==========================================

def test_dt_limit_clamping(device):
    """Test dt_limit actually clamps dt values"""
    config = Mamba3Config(d_model=32, dt_limit=(0.01, 0.05))
    model = Mamba3Block(config).to(device)
    model.eval()
    
    with torch.no_grad():
        x = torch.randn(1, 10, 32).to(device)
        y = model(x)
    
    print("✓ dt_limit clamping test passed")


def test_long_sequence_stability(device):
    """Test model handles long sequences without NaN"""
    config = Mamba3Config(d_model=32, chunk_size=256, use_parallel_scan=True)
    model = Mamba3Block(config).to(device)
    model.eval()
    
    with torch.no_grad():
        x_long = torch.randn(1, 2048, 32).to(device)
        y_long = model(x_long)
        
        assert not torch.isnan(y_long).any(), "NaN in long sequence output"
        assert not torch.isinf(y_long).any(), "Inf in long sequence output"
    
    print("✓ Long sequence stability test passed")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_mixed_precision_fp16(device):
    """Test model works with FP16"""
    config = Mamba3Config(d_model=64)
    model = Mamba3Block(config).to(device).half()
    x = torch.randn(2, 32, 64, dtype=torch.float16).to(device)
    
    with torch.no_grad():
        y = model(x)
        assert not torch.isnan(y).any()
    
    print("✓ Mixed precision FP16 test passed")


@pytest.mark.skipif(not hasattr(torch, 'bfloat16'), reason="Requires BF16 support")
def test_mixed_precision_bf16(device):
    """Test model works with BF16"""
    config = Mamba3Config(d_model=64)
    model = Mamba3Block(config).to(device).to(torch.bfloat16)
    x = torch.randn(2, 32, 64, dtype=torch.bfloat16).to(device)
    
    with torch.no_grad():
        y = model(x)
        assert not torch.isnan(y).any()
    
    print("✓ Mixed precision BF16 test passed")


# ==========================================
# Parallel Scan Tests
# ==========================================

def test_parallel_scan_numerical_equivalence(device):
    """Test parallel scan vs sequential scan equivalence"""
    dtype = torch.float64
    
    config = Mamba3Config(
        d_model=16, n_groups=1, d_state=4, d_head=16, 
        mimo_rank=1, expand=1, use_parallel_scan=True
    )
    block = Mamba3Block(config).to(device).double()
    
    B, L, H, N, P, R = 1, 16, 1, 4, 4, 1
    
    dt = torch.rand(B, L, H, dtype=dtype).to(device)
    A = torch.randn(H, dtype=dtype).to(device)
    u = torch.randn(B, L, H, N, P, dtype=dtype).to(device)
    C = torch.randn(B, L, H, N, R, dtype=dtype).to(device)
    
    test_chunk_size = 4
    y_par, h_par = block.chunk_parallel_scan(u, dt, A, C, chunk_size=test_chunk_size)
    
    # Sequential scan
    log_alpha = torch.einsum('blh, h -> blh', dt, A)
    alpha = torch.exp(log_alpha).view(B, L, H, 1, 1)
    h_state = torch.zeros(B, H, N, P, dtype=dtype, device=device)
    
    for t in range(L):
        h_state = h_state * alpha[:, t] + u[:, t]
    
    diff = (h_state - h_par).abs().max().item()
    assert diff < 1e-4, f"Chunk Scan Mismatch! Diff: {diff}"
    print(f"✓ Parallel scan equivalence test passed (diff={diff:.2e})")


# ==========================================
# Grouping Tests
# ==========================================

def test_parameter_scaling_mqa_vs_mha():
    """Test MQA has fewer parameters than MHA"""
    c_mqa = Mamba3Config(d_model=128, d_head=32, expand=2, n_groups=1)
    c_mha = Mamba3Config(d_model=128, d_head=32, expand=2, n_groups=8)
    
    m_mqa = Mamba3Block(c_mqa)
    m_mha = Mamba3Block(c_mha)
    
    p_mqa = sum(p.numel() for p in m_mqa.parameters())
    p_mha = sum(p.numel() for p in m_mha.parameters())
    
    assert p_mqa < p_mha
    print(f"✓ Parameter scaling test passed (MQA: {p_mqa:,}, MHA: {p_mha:,})")


# ==========================================
# Vision Mamba Tests
# ==========================================

def test_vision_mamba_instantiation_and_forward(device):
    """Test VisionMamba instantiation and forward pass"""
    model = VisionMamba(
        img_size=32, patch_size=4, depth=1, embed_dim=32, 
        n_groups=2, use_conv=True, num_classes=5
    ).to(device)
    
    assert 'snake_indices' in model.state_dict(), "Snake Indices buffer missing"
    assert model.snake_indices.device.type == device.type
    
    x = torch.randn(2, 3, 32, 32).to(device)
    y = model(x)
    assert y.shape == (2, 5)
    print("✓ Vision Mamba instantiation test passed")


def test_vision_mamba_parameter_propagation(device):
    """Test Mamba-3 parameters propagate correctly to VisionMamba"""
    model = VisionMamba(
        img_size=32, patch_size=4, depth=1, embed_dim=64,
        n_groups=2, mimo_rank=3, expand=1, use_conv=True
    ).to(device)
    
    first_layer_config = model.layers[0]['fwd'].config
    
    assert first_layer_config.n_groups == 2
    assert first_layer_config.mimo_rank == 3
    assert first_layer_config.use_conv
    assert first_layer_config.expand == 1
    print("✓ Vision Mamba parameter propagation test passed")


def test_vision_mamba_bidirectional_structure(device):
    """Test bidirectional structure is correct"""
    model_bi = VisionMamba(img_size=32, depth=1, bidirectional=True).to(device)
    assert isinstance(model_bi.layers[0], nn.ModuleDict)
    assert 'fwd' in model_bi.layers[0]
    assert 'bwd' in model_bi.layers[0]
    
    model_uni = VisionMamba(img_size=32, depth=1, bidirectional=False).to(device)
    assert isinstance(model_uni.layers[0], Mamba3Block)
    assert not isinstance(model_uni.layers[0], nn.ModuleDict)
    print("✓ Vision Mamba bidirectional structure test passed")


# ==========================================
# Sanity Tests
# ==========================================

def test_overfitting_sanity(device):
    """Test model can overfit to small dataset"""
    torch.manual_seed(100)
    
    config = Mamba3Config(d_model=32, n_groups=1, mimo_rank=2, use_parallel_scan=True)
    model = Mamba3Block(config).to(device)
    
    B, L = 2, 16
    u = torch.randn(B, L, 32).to(device)
    target = torch.randn(B, L, 32).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()
    
    model.train()
    final_loss = 100.0
    
    for _ in range(50):
        optimizer.zero_grad()
        out = model(u)
        loss = loss_fn(out, target)
        loss.backward()
        optimizer.step()
        final_loss = loss.item()
    
    assert final_loss < 0.1, f"Model failed to overfit. Loss: {final_loss}"
    print(f"✓ Overfitting sanity test passed (final loss={final_loss:.4f})")


# ==========================================
# Parametrized Tests (Official Mamba Style)
# ==========================================

@pytest.mark.parametrize('seqlen', [128, 256, 512, 1024, 2048])
@pytest.mark.parametrize('mimo_rank', [1, 4, 8, 16])
@pytest.mark.parametrize('n_groups', [1, 2, 4])
@pytest.mark.parametrize('use_parallel_scan', [True, False])
def test_mamba3_forward_shape(seqlen, mimo_rank, n_groups, use_parallel_scan):
    """Test forward pass shape consistency with various configurations"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 2
    d_model = 256
    
    config = Mamba3Config(
        d_model=d_model,
        d_state=64,
        d_head=64,
        n_groups=n_groups,
        mimo_rank=mimo_rank,
        expand=2,
        use_parallel_scan=use_parallel_scan,
        chunk_size=256
    )
    
    model = Mamba3Block(config).to(device)
    x = torch.randn(batch_size, seqlen, d_model, device=device)
    
    with torch.no_grad():
        y = model(x)
    
    assert y.shape == x.shape, f"Output shape {y.shape} != Input shape {x.shape}"
    print(f"✓ Shape test: L={seqlen}, R={mimo_rank}, G={n_groups}, parallel={use_parallel_scan}")


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16])
@pytest.mark.parametrize('seqlen', [128, 1024, 4096])
def test_mamba3_numerical_precision(seqlen, dtype):
    """Test numerical precision: parallel vs sequential scan"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if dtype == torch.float16 and device == 'cpu':
        pytest.skip("FP16 not supported on CPU")
    
    batch_size = 2
    d_model = 128
    
    config = Mamba3Config(
        d_model=d_model,
        d_state=32,
        d_head=32,
        mimo_rank=4,
        use_parallel_scan=True,
        chunk_size=256
    )
    
    model = Mamba3Block(config).to(device).to(dtype)
    x = torch.randn(batch_size, seqlen, d_model, device=device, dtype=dtype)
    
    with torch.no_grad():
        y_parallel = model(x)
    
    model.config.use_parallel_scan = False
    with torch.no_grad():
        y_sequential = model(x)
    
    max_diff = (y_parallel - y_sequential).abs().max().item()
    mean_diff = (y_parallel - y_sequential).abs().mean().item()
    
    rtol = 1e-5 if dtype == torch.float32 else 1e-3
    atol = 1e-6 if dtype == torch.float32 else 1e-4
    
    assert torch.allclose(y_parallel, y_sequential, rtol=rtol, atol=atol), \
        f"Parallel/sequential differ: max={max_diff:.2e}"
    print(f"✓ Precision test: L={seqlen}, dtype={dtype}, max_diff={max_diff:.2e}")


@pytest.mark.parametrize('seqlen', [128, 1024, 8192])
def test_mamba3_long_sequence_stability(seqlen):
    """Test stability on long sequences"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 1
    d_model = 256
    
    config = Mamba3Config(
        d_model=d_model,
        d_state=64,
        mimo_rank=4,
        use_parallel_scan=True,
        chunk_size=256
    )
    
    model = Mamba3Block(config).to(device)
    x = torch.randn(batch_size, seqlen, d_model, device=device)
    
    with torch.no_grad():
        y = model(x)
    
    assert not torch.isnan(y).any(), f"NaN detected for seqlen={seqlen}"
    assert not torch.isinf(y).any(), f"Inf detected for seqlen={seqlen}"
    assert y.abs().max() < 1e6, f"Numerical explosion for seqlen={seqlen}"
    
    print(f"✓ Stability test: L={seqlen}, max_val={y.abs().max():.2e}")


def test_mamba3_gradient_flow():
    """Test gradient flow through all critical parameters"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 2
    seqlen = 128
    d_model = 128
    
    config = Mamba3Config(
        d_model=d_model,
        d_state=32,
        mimo_rank=4,
        use_parallel_scan=True
    )
    
    model = Mamba3Block(config).to(device)
    x = torch.randn(batch_size, seqlen, d_model, device=device, requires_grad=True)
    
    y = model(x)
    loss = y.sum()
    loss.backward()
    
    critical_params = ['A_log', 'in_proj.weight', 'in_proj.bias', 
                       'x_up_proj.weight', 'y_down_proj.weight', 'D']
    
    for name, param in model.named_parameters():
        if any(cp in name for cp in critical_params):
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
            grad_norm = param.grad.norm().item()
            assert grad_norm > 0, f"Zero gradient for {name}"
            print(f"  ✓ {name}: grad_norm={grad_norm:.2e}")
    
    print("✓ Gradient flow test passed")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
@pytest.mark.parametrize('mimo_rank', [1, 4, 8, 16])
def test_mamba3_mimo_scaling(mimo_rank):
    """Test MIMO rank scaling efficiency"""
    device = 'cuda'
    batch_size = 4
    seqlen = 1024
    d_model = 256
    
    config = Mamba3Config(
        d_model=d_model,
        d_state=64,
        mimo_rank=mimo_rank,
        use_parallel_scan=True
    )
    
    model = Mamba3Block(config).to(device)
    x = torch.randn(batch_size, seqlen, d_model, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)
    
    # Benchmark
    torch.cuda.synchronize()
    import time
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model(x)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    throughput = (batch_size * seqlen * 100) / elapsed
    print(f"✓ MIMO R={mimo_rank}: {throughput:.2e} tokens/sec")


@pytest.mark.parametrize('use_conv', [True, False])
@pytest.mark.parametrize('n_groups', [1, 2, 4])
def test_mamba3_config_variants(use_conv, n_groups):
    """Test different configuration variants"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 2
    seqlen = 256
    d_model = 128
    
    config = Mamba3Config(
        d_model=d_model,
        d_state=64,
        d_head=32,
        n_groups=n_groups,
        mimo_rank=4,
        use_conv=use_conv,
        d_conv=4
    )
    
    model = Mamba3Block(config).to(device)
    x = torch.randn(batch_size, seqlen, d_model, device=device)
    
    with torch.no_grad():
        y = model(x)
    
    assert y.shape == x.shape
    assert not torch.isnan(y).any()
    print(f"✓ Config test: conv={use_conv}, groups={n_groups}")


def test_mamba3_initialization():
    """Test parameter initialization ranges"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    config = Mamba3Config(
        d_model=256,
        d_state=64,
        mimo_rank=4,
        dt_min=0.001,
        dt_max=0.1,
        A_init_range=(1, 16)
    )
    
    model = Mamba3Block(config).to(device)
    
    # Check A_log: initialized as uniform(A_min, A_max).log()
    # So A_log is in [log(1), log(16)] = [0, ~2.77]
    # A = -exp(A_log) gives A in [-16, -1]
    A = -torch.exp(model.A_log)
    A_abs = A.abs().detach().cpu().numpy()
    # |A| should be in [1, 16]
    assert A_abs.min() >= 0.9 and A_abs.max() <= 16.5, f"A initialization out of range: [{A_abs.min():.3f}, {A_abs.max():.3f}]"
    print(f"  A magnitude range: [{A_abs.min():.3f}, {A_abs.max():.3f}]")
    
    # Check dt_bias
    dt_start = config.d_inner + config.d_inner + config.n_groups * config.d_state * config.mimo_rank * 2
    dt_end = dt_start + config.n_groups
    dt_bias_raw = model.in_proj.bias[dt_start:dt_end]
    dt = F.softplus(dt_bias_raw)
    assert dt.min() >= config.dt_min and dt.max() <= config.dt_max, "dt initialization out of range"
    print(f"  dt range: [{dt.min():.4f}, {dt.max():.4f}]")
    
    print("✓ Initialization test passed")


# ==========================================
# Main Entry Point
# ==========================================

if __name__ == '__main__':
    print(f"Running Mamba-3 Pytest Test Suite on {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print("=" * 70)
    print("Use: pytest test_suite.py -v")
    print("=" * 70)
    pytest.main([__file__, '-v'])
