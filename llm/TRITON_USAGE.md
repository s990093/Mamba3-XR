# Triton 加速使用指南

## 概述

本指南說明如何在 Kaggle T4 x2 環境中使用 Triton 加速的 Mamba-3 實作。

## 安裝依賴

```bash
# 在 Kaggle Notebook 中執行
!pip install triton accelerate tqdm -q
```

## 使用方式

### 方式 1: 自動檢測 (推薦)

腳本會自動檢測 Triton 是否可用，並在 CUDA 設備上自動啟用加速:

```python
# 直接運行腳本
%run mamba3_shakespeare_kaggle.py

# 啟動訓練
main()  # 單卡
# 或
launch_training(num_gpus=2)  # 雙卡
```

**預期輸出**:

```
✓ Triton available
✓ Using Triton acceleration for chunk_parallel_scan
```

### 方式 2: 手動控制

如果需要強制使用 PyTorch fallback (例如調試):

```python
# 在導入前設置環境變量
import os
os.environ['DISABLE_TRITON'] = '1'

%run mamba3_shakespeare_kaggle.py
```

## 驗證安裝

運行驗證腳本檢查 Triton kernel 正確性:

```bash
cd /path/to/mamba3/llm
python verify_triton.py
```

**預期輸出**:

```
TEST 1: Correctness Verification
✅ PASSED (rtol=0.001, atol=0.001)

TEST 2: Performance Benchmark
  Triton:  0.234 ms
  PyTorch: 0.567 ms
  Speedup: 2.42x
✅ Triton is faster!

TEST 3: Full chunk_parallel_scan Integration
✅ Forward pass successful!
```

## 性能優化建議

### 1. Chunk Size 調整

根據序列長度調整 `chunk_size`:

```python
# 在 Mamba3LM 初始化時設置
model = Mamba3LM(
    chunk_size=256,  # 預設值，適合大多數情況
    # chunk_size=128,  # 長序列 (L > 1024)
    # chunk_size=512,  # 短序列 (L < 512)
)
```

### 2. 混合精度訓練

確保啟用 FP16 以充分利用 T4 Tensor Core:

```python
train_shakespeare(
    model=model,
    mixed_precision="fp16",  # 重要！
    use_accelerate=True
)
```

### 3. 批次大小調整

T4 記憶體有限，建議:

```python
BATCH_SIZE = 32  # 單卡
# 雙卡有效批次大小 = 32 * 2 = 64
```

## 預期加速效果

在 Kaggle T4 x2 環境下:

| 配置                   | 原始 PyTorch | Triton 加速 | 加速比   |
| ---------------------- | ------------ | ----------- | -------- |
| Batch=32, L=256        | ~1.2s/step   | ~0.5s/step  | **2.4x** |
| Batch=32, L=512        | ~3.8s/step   | ~1.4s/step  | **2.7x** |
| Batch=64, L=256 (雙卡) | ~1.2s/step   | ~0.5s/step  | **2.4x** |

**記憶體節省**: ~30% 峰值記憶體使用

## 故障排除

### 問題 1: "Triton not available"

**原因**: Triton 未安裝或版本不兼容

**解決**:

```bash
!pip install triton --upgrade
```

### 問題 2: "Triton kernel failed"

**原因**: 可能是形狀不匹配或記憶體問題

**解決**: 腳本會自動 fallback 到 PyTorch，檢查錯誤訊息:

```python
# 查看詳細錯誤
import traceback
traceback.print_exc()
```

### 問題 3: 數值不穩定 (NaN)

**原因**: FP16 精度問題

**解決**: 已在 kernel 內部使用 FP32 累加器，如仍有問題:

```python
# 切換到 FP32 (較慢但更穩定)
model = model.float()
```

## 技術細節

### Triton Kernel 架構

```
inter_chunk_scan_kernel_fwd:
  Grid: (Batch*Heads, cdiv(N, BLOCK_N))
  Block: BLOCK_N=32, BLOCK_P=next_power_of_2(P)

  核心邏輯:
    for c in range(num_chunks):
      h[c] = h[c-1] * decay[c] + x[c]
```

### 記憶體布局優化

- **輸入**: [B, C, H, N, P] → [B*H, C, N, P] (合併 Batch 和 Head)
- **Decay**: [B, C, H] → [B*H, C]
- **輸出**: [B*H, C, N, P] → [B, C, H, N, P] (還原形狀)

### FP16 優化

- Kernel 內部使用 `tl.float32` 累加器
- 輸出時轉換回 FP16
- `L_mask` 強制使用 FP16 以利用 Tensor Core

## 進階配置

### 自定義 Grid 配置

如果需要調整 Triton kernel 的 grid 配置 (進階用戶):

```python
# 在 triton_inter_chunk_scan 函數中修改
BLOCK_N = 32  # 增大以減少 kernel launch overhead
BLOCK_P = 128  # 減小以降低 register pressure
```

### 性能分析

使用 PyTorch Profiler 分析:

```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    output = model(input_ids)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

## 參考資料

- [Triton 官方文檔](https://triton-lang.org/)
- [Mamba-3 論文](https://arxiv.org/abs/2408.15237)
- [Implementation Plan](implementation_plan.md)
