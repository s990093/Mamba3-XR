# CUDA Fork 問題解決方案

## 問題描述

在 Jupyter/Kaggle Notebook 中使用多 GPU 訓練時，遇到錯誤：

```
RuntimeError: Cannot re-initialize CUDA in forked subprocess.
To use CUDA with multiprocessing, you must use the 'spawn' start method
```

## 根本原因

- Jupyter Notebook 默認使用 `fork` 方法啟動子進程
- CUDA 不支持在 forked 子進程中重新初始化
- 必須使用 `spawn` 方法

## 解決方案

### 方案 1：使用更新後的腳本（推薦）✅

最新版本的 `mamba3_shakespeare_kaggle.py` 已經自動處理這個問題：

```python
# 腳本會自動設置 spawn 方法
launch_training(num_gpus=2)
```

### 方案 2：手動設置 spawn 方法

如果使用舊版本，在訓練前添加：

```python
import multiprocessing as mp

# 設置 spawn 方法
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # 已經設置過了

# 然後運行訓練
launch_training(num_gpus=2)
```

### 方案 3：重啟 Notebook Runtime

如果問題仍然存在：

1. 重啟 Kaggle Runtime
2. 重新運行所有 cells
3. 使用最新版本的腳本

## 技術細節

### Fork vs Spawn

| 方法  | 優點             | 缺點           | CUDA 支持 |
| ----- | ---------------- | -------------- | --------- |
| fork  | 快速，共享記憶體 | 不安全（CUDA） | ❌ 不支持 |
| spawn | 安全，獨立進程   | 較慢           | ✅ 支持   |

### 為什麼 Fork 不行？

1. Fork 複製父進程的記憶體空間
2. CUDA 上下文已經在父進程中初始化
3. 子進程嘗試重新初始化 CUDA → 錯誤

### Spawn 如何解決？

1. Spawn 創建全新的 Python 進程
2. 每個進程獨立初始化 CUDA
3. 沒有衝突 ✅

## 驗證修復

成功的輸出應該包含：

```
✅ Multiprocessing start method set to 'spawn' for CUDA compatibility

================================================================================
🔍 GPU Detection
================================================================================
  Available GPUs: 2
  GPU 0: Tesla T4 (14.7 GB)
  GPU 1: Tesla T4 (14.7 GB)
================================================================================

================================================================================
🚀 啟動雙卡並行訓練 (Launching Multi-GPU Training)
================================================================================
Target GPUs: 2
Mixed Precision: FP16 (enabled)
Start Method: spawn (CUDA compatible)  ← 確認使用 spawn
================================================================================

Launching training on 2 CUDAs.

[訓練開始...]
```

## 常見問題

### Q: 為什麼不在腳本開頭就設置 spawn？

A: 因為 `set_start_method()` 只能調用一次。如果在 import 時調用，可能會與其他代碼衝突。所以在 `launch_training()` 中使用 `force=True` 來確保設置。

### Q: 單卡訓練需要 spawn 嗎？

A: 不需要。單卡訓練（`main()`）不使用多進程，所以沒有這個問題。

### Q: 為什麼 Accelerate 不自動處理？

A: Accelerate 依賴 PyTorch 的 `torch.multiprocessing`，而 Jupyter 環境的限制導致默認使用 fork。我們需要手動覆蓋。

## 其他注意事項

### Kaggle 特定問題

Kaggle Notebook 環境可能有額外限制：

1. **Runtime 狀態**：如果之前運行過 CUDA 代碼，必須重啟 Runtime
2. **Cell 執行順序**：確保按順序執行所有 cells
3. **依賴安裝**：確保 `accelerate` 已安裝

### 最佳實踐

```python
# Cell 1: 安裝依賴
!pip install accelerate tqdm -q

# Cell 2: 重啟 Runtime（如果需要）
# 點擊 Runtime → Restart Session

# Cell 3: 導入並運行
%run mamba3_shakespeare_kaggle.py

# 腳本會自動處理 spawn 設置
```

## 參考

- [PyTorch Multiprocessing](https://pytorch.org/docs/stable/notes/multiprocessing.html)
- [CUDA and Multiprocessing](https://pytorch.org/docs/stable/notes/cuda.html#use-nn-parallel-distributeddataparallel-instead-of-multiprocessing-or-nn-dataparallel)
- [Accelerate Documentation](https://huggingface.co/docs/accelerate)

---

**狀態**: ✅ 已修復  
**版本**: mamba3_shakespeare_kaggle.py (最新)  
**測試**: Kaggle T4 x2 環境
