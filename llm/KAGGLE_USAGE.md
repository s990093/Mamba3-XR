# Mamba-3 Shakespeare - Kaggle 單文件版本

## 📄 文件說明

**`mamba3_shakespeare_kaggle.py`** - 完整的獨立文件（812 行）

包含所有必要組件：

1. ✅ Mamba3Config
2. ✅ RMSNorm
3. ✅ Mamba3Block（完整 SSM 實現）
4. ✅ Mamba3LM（語言模型）
5. ✅ CharDataset（字符級數據集）
6. ✅ 訓練循環
7. ✅ 生成函數

## 🚀 Kaggle 使用方法

### 1. 上傳文件

- 將 `mamba3_shakespeare_kaggle.py` 上傳到 Kaggle Notebook

### 2. 直接運行

```python
# 在 Kaggle Notebook 中
!python mamba3_shakespeare_kaggle.py
```

或者在 Notebook cell 中：

```python
%run mamba3_shakespeare_kaggle.py
```

### 3. 自定義配置

修改 `main()` 函數中的參數：

```python
BLOCK_SIZE = 256      # 上下文長度
BATCH_SIZE = 32       # 批次大小
EPOCHS = 10           # 訓練輪數
LR = 3e-4             # 學習率
MIMO_RANK = 4         # MIMO 秩（1, 4, 8, 16）
```

## 📊 預期輸出

```
Using device: cuda
Downloading Shakespeare dataset...
✓ Downloaded to data/shakespeare.txt
Dataset: 1003854 characters, 65 unique
Vocabulary:
 !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZ...

================================================================================
Training Mamba-3 on Shakespeare
================================================================================
Model parameters: 2.37M
Device: cuda
Epochs: 10
Learning rate: 0.0003
================================================================================

Step   100 | Train Loss: 2.1234 | Val Loss: 2.3456 | LR: 2.95e-04
Step   200 | Train Loss: 1.8765 | Val Loss: 2.0123 | LR: 2.85e-04
...

Epoch 1/10 | Avg Loss: 2.0123 | Time: 45.2s

--------------------------------------------------------------------------------
Sample Generation:
--------------------------------------------------------------------------------
ROMEO:
What light through yonder window breaks?
It is the east, and Juliet is the sun.
...
--------------------------------------------------------------------------------
```

## 🎯 實驗建議

### Rank 對比實驗

修改 `MIMO_RANK` 並多次運行：

```python
# 實驗 1: Rank 1 (基準)
MIMO_RANK = 1

# 實驗 2: Rank 4 (CV 最優)
MIMO_RANK = 4

# 實驗 3: Rank 8
MIMO_RANK = 8

# 實驗 4: Rank 16
MIMO_RANK = 16
```

### 記錄指標

- 最終驗證損失
- 訓練時間
- 生成文本質量
- 模型參數量

## 💾 保存模型

模型會自動保存到：

- `shakespeare_best.pt` - 最佳驗證損失的模型

加載模型：

```python
model = create_mamba3_tiny(vocab_size=65, mimo_rank=4)
model.load_state_dict(torch.load('shakespeare_best.pt'))
```

## 🔧 故障排除

### CUDA Out of Memory

減小批次大小：

```python
BATCH_SIZE = 16  # 或 8
```

### 訓練太慢

減少 epochs 或使用更小的模型：

```python
EPOCHS = 5
BLOCK_SIZE = 128
```

## 📈 與 CV 結果對比

| 指標        | CV (CIFAR-100) | LLM (Shakespeare) |
| ----------- | -------------- | ----------------- |
| Rank 4 飽和 | 87.5% (28/32)  | ? (待測試)        |
| Rank 8 提升 | +0.48%         | ? (待測試)        |
| 最優配置    | Rank 4         | ? (待測試)        |

## 🎓 研究問題

1. **MIMO 秩飽和是否右移？**

   - 語言是否需要更高的秩？

2. **飽和點在哪裡？**

   - Rank 4, 8, 還是 16？

3. **性能 vs. 參數效率**
   - 最佳性價比配置是什麼？

---

**文件大小**: 812 行  
**無需外部依賴**: 所有代碼都在一個文件中  
**即插即用**: 上傳到 Kaggle 即可運行
