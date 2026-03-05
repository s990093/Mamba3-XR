# Mamba-3 Multi-GPU Training - Installation & Usage

## 📦 安裝 Accelerate

在 Kaggle Notebook 第一個 cell 運行：

```python
!pip install accelerate -q
```

## 🚀 使用方法

### 方案 1：直接運行（推薦）

```python
# 上傳 mamba3_shakespeare_kaggle.py 後直接運行
%run mamba3_shakespeare_kaggle.py
```

Accelerate 會自動檢測並使用所有可用的 GPU。

### 方案 2：自定義配置

修改 `main()` 函數中的參數：

```python
BATCH_SIZE = 32       # 每張卡的 batch size
EPOCHS = 10           # 訓練輪數
MIMO_RANK = 4         # MIMO 秩
USE_MULTI_GPU = True  # 設為 False 強制單卡模式
```

### 方案 3：強制單卡模式

如果只想用一張卡：

```python
USE_MULTI_GPU = False
```

## 📊 雙卡 vs 單卡

| 指標       | 單卡 (16GB) | 雙卡 (16GB x 2) |
| ---------- | ----------- | --------------- |
| Batch Size | 32          | 32 x 2 = 64     |
| 訓練速度   | 1x          | ~1.8x           |
| 記憶體     | 16GB        | 32GB (總計)     |
| 可訓練模型 | 較小        | 較大            |

## 🔧 Accelerate 配置

### 自動配置（推薦）

Accelerate 會自動檢測環境並配置：

- GPU 數量
- Mixed Precision (FP16/BF16)
- Distributed 類型 (DDP)

### 手動配置（進階）

如果需要手動配置：

```bash
accelerate config
```

然後選擇：

- Multi-GPU
- Number of GPUs: 2
- Mixed Precision: fp16 或 bf16

## 📈 預期輸出

```
================================================================================
🚀 Multi-GPU Training with Accelerate
================================================================================
Number of GPUs: 2
Device: cuda:0
Mixed Precision: no
Distributed Type: MULTI_GPU
================================================================================

================================================================================
Training Mamba-3 on Shakespeare
================================================================================
Model parameters: 2.37M
Device: cuda:0
Epochs: 10
Learning rate: 0.0003
Batch size per GPU: 32
Effective batch size: 64
================================================================================
```

## 💡 優化建議

### 1. 增加 Batch Size

雙卡模式下可以增加每張卡的 batch size：

```python
BATCH_SIZE = 64  # 總 batch size = 64 x 2 = 128
```

### 2. 啟用 Mixed Precision

在訓練函數中啟用 FP16：

```python
accelerator = Accelerator(mixed_precision='fp16')
```

可節省約 50% 記憶體，訓練速度提升 2-3x。

### 3. Gradient Accumulation

如果記憶體不足：

```python
accelerator = Accelerator(gradient_accumulation_steps=4)
```

## 🐛 故障排除

### 問題 1：Accelerate 未安裝

```
⚠️  Warning: accelerate not installed. Running in single-GPU mode.
```

**解決**：

```python
!pip install accelerate
```

### 問題 2：CUDA Out of Memory

**解決**：

1. 減小 batch size
2. 啟用 gradient checkpointing
3. 使用 mixed precision

### 問題 3：只用了一張卡

**檢查**：

- Kaggle 是否選擇了 "GPU T4 x2"
- `USE_MULTI_GPU = True`
- Accelerate 是否正確安裝

## 📝 Rank 實驗建議

### 實驗矩陣（雙卡 16GB x 2）

| Rank | Batch Size | 預估訓練時間 | 記憶體使用 |
| ---- | ---------- | ------------ | ---------- |
| 1    | 64         | ~30 min      | ~8GB       |
| 4    | 64         | ~35 min      | ~10GB      |
| 8    | 64         | ~40 min      | ~12GB      |
| 16   | 48         | ~50 min      | ~15GB      |

### 運行多個實驗

```python
for mimo_rank in [1, 4, 8, 16]:
    print(f"\n{'='*80}")
    print(f"Training with MIMO Rank {mimo_rank}")
    print(f"{'='*80}\n")

    model = create_mamba3_tiny(
        vocab_size=train_dataset.vocab_size,
        mimo_rank=mimo_rank
    )

    model = train_shakespeare(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        dataset=train_dataset,
        epochs=10,
        lr=3e-4,
        use_accelerate=True,
    )

    # Save results
    torch.save(model.state_dict(), f'shakespeare_rank{mimo_rank}.pt')
```

## 🎯 與 CV 結果對比

記錄以下指標以便與 CIFAR-100 結果對比：

1. **最終驗證損失**
2. **訓練時間**
3. **生成文本質量**（主觀評分）
4. **參數量**
5. **記憶體使用**

---

**文件**: `mamba3_shakespeare_kaggle.py` (920 行)  
**支持**: 單卡 / 雙卡自動切換  
**依賴**: `torch`, `accelerate`
