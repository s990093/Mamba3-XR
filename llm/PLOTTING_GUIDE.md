# 訓練數據可視化使用指南

## 📊 自動生成的數據文件

訓練完成後會自動生成：

1. **training_history.json** - 完整訓練歷史（JSON 格式）
2. **training_epochs.csv** - 每個 epoch 的統計（CSV 格式）
3. **training_steps.csv** - 每個評估步驟的統計（CSV 格式）
4. **shakespeare_best.pt** - 最佳模型權重

## 🎨 使用繪圖腳本

### 方法 1：直接運行

```bash
python plot_training.py
```

### 方法 2：在 Jupyter/Kaggle Notebook 中

```python
%run plot_training.py
```

### 方法 3：自定義繪圖

```python
import pandas as pd
import matplotlib.pyplot as plt

# 讀取 CSV 數據
epochs_df = pd.read_csv('training_epochs.csv')
steps_df = pd.read_csv('training_steps.csv')

# 繪製損失曲線
plt.figure(figsize=(10, 6))
plt.plot(epochs_df['epoch'], epochs_df['avg_train_loss'], label='Train Loss')
plt.plot(epochs_df['epoch'], epochs_df['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('my_loss_curve.png', dpi=300)
plt.show()
```

## 📈 生成的圖表

運行 `plot_training.py` 會生成以下圖表：

### 1. training_loss.png

- 訓練損失和驗證損失曲線
- 標註最佳驗證損失點

### 2. learning_rate.png

- 學習率變化曲線（對數尺度）
- 顯示 Cosine Annealing 調度

### 3. throughput.png

- 每個 epoch 的訓練吞吐量（samples/sec）
- 平均吞吐量基準線

### 4. step_losses.png

- 每個評估步驟的詳細損失
- 訓練損失和驗證損失對比

### 5. training_summary.png

- 綜合 4 合 1 圖表
- 包含所有關鍵指標

## 📊 CSV 數據格式

### training_epochs.csv

| 列名            | 說明           |
| --------------- | -------------- |
| epoch           | Epoch 編號     |
| avg_train_loss  | 平均訓練損失   |
| val_loss        | 驗證損失       |
| time            | 訓練時間（秒） |
| samples_per_sec | 吞吐量         |

### training_steps.csv

| 列名       | 說明         |
| ---------- | ------------ |
| step       | 全局步驟編號 |
| epoch      | 所屬 epoch   |
| train_loss | 訓練損失     |
| val_loss   | 驗證損失     |
| lr         | 當前學習率   |
| samples    | 已處理樣本數 |

## 🔄 比較多個實驗

### 為不同 Rank 保存數據

```python
# 訓練完成後，重命名文件
import shutil

rank = 4  # 當前實驗的 rank

shutil.move('training_epochs.csv', f'rank{rank}_epochs.csv')
shutil.move('training_steps.csv', f'rank{rank}_steps.csv')
shutil.move('shakespeare_best.pt', f'rank{rank}_best.pt')
```

### 比較不同 Rank 的結果

```python
import pandas as pd
import matplotlib.pyplot as plt

# 讀取不同 rank 的數據
ranks = [1, 4, 8, 16]
results = {}

for rank in ranks:
    try:
        df = pd.read_csv(f'rank{rank}_epochs.csv')
        results[rank] = df
    except:
        pass

# 繪製對比圖
plt.figure(figsize=(12, 6))

for rank, df in results.items():
    plt.plot(df['epoch'], df['val_loss'],
             label=f'Rank {rank}', linewidth=2, marker='o')

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Validation Loss', fontsize=12)
plt.title('MIMO Rank Comparison', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.savefig('rank_comparison.png', dpi=300)
plt.show()

# 打印最終結果
print("=" * 60)
print("MIMO Rank Comparison Results")
print("=" * 60)
for rank, df in results.items():
    final_val_loss = df['val_loss'].iloc[-1]
    best_val_loss = df['val_loss'].min()
    print(f"Rank {rank:2d}: Final={final_val_loss:.4f}, Best={best_val_loss:.4f}")
print("=" * 60)
```

## 📥 在 Kaggle 下載數據

### 下載單個文件

```python
from IPython.display import FileLink

# 下載 CSV
FileLink('training_epochs.csv')
```

### 打包下載所有文件

```python
import shutil
from IPython.display import FileLink

# 打包所有訓練數據
files_to_pack = [
    'training_history.json',
    'training_epochs.csv',
    'training_steps.csv',
    'shakespeare_best.pt',
    'training_loss.png',
    'training_summary.png',
]

# 創建壓縮包
shutil.make_archive('mamba3_training_results', 'zip', '.',
                   base_dir=None)

# 提供下載鏈接
FileLink('mamba3_training_results.zip')
```

## 🎯 進階分析示例

### 計算改進百分比

```python
import pandas as pd

# 讀取不同 rank 的數據
rank1_df = pd.read_csv('rank1_epochs.csv')
rank4_df = pd.read_csv('rank4_epochs.csv')
rank8_df = pd.read_csv('rank8_epochs.csv')

# 計算最佳驗證損失
rank1_best = rank1_df['val_loss'].min()
rank4_best = rank4_df['val_loss'].min()
rank8_best = rank8_df['val_loss'].min()

# 計算改進
improvement_4_vs_1 = (rank1_best - rank4_best) / rank1_best * 100
improvement_8_vs_4 = (rank4_best - rank8_best) / rank4_best * 100

print(f"Rank 4 vs Rank 1: {improvement_4_vs_1:.2f}% improvement")
print(f"Rank 8 vs Rank 4: {improvement_8_vs_4:.2f}% improvement")

# 與 CV 結果對比
print("\n與 CIFAR-100 對比：")
print(f"CV (Rank 8 vs 4): +0.48% accuracy")
print(f"LLM (Rank 8 vs 4): {improvement_8_vs_4:.2f}% loss reduction")
```

### 分析訓練穩定性

```python
import pandas as pd
import numpy as np

df = pd.read_csv('training_epochs.csv')

# 計算損失的標準差（穩定性指標）
train_loss_std = df['avg_train_loss'].std()
val_loss_std = df['val_loss'].std()

# 計算吞吐量的變異係數
throughput_cv = df['samples_per_sec'].std() / df['samples_per_sec'].mean()

print(f"訓練損失標準差: {train_loss_std:.4f}")
print(f"驗證損失標準差: {val_loss_std:.4f}")
print(f"吞吐量變異係數: {throughput_cv:.4f}")
```

---

**自動化**: 訓練完成後自動保存 CSV  
**靈活性**: 支持 JSON 和 CSV 兩種格式  
**易用性**: 一鍵生成所有可視化圖表
