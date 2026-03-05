# 訓練數據保存說明

## 📊 自動保存的文件

訓練完成後，會自動保存以下文件：

### 1. `training_history.json`

完整的訓練歷史（JSON 格式）

```json
{
  "config": {
    "mimo_rank": 4,
    "d_model": 256,
    "n_layers": 4,
    "vocab_size": 65,
    "epochs": 10,
    "lr": 0.0003,
    "batch_size": 32,
    "num_gpus": 2,
    "mixed_precision": "fp16"
  },
  "epochs": [
    {
      "epoch": 1,
      "avg_train_loss": 2.1234,
      "val_loss": 2.3456,
      "time": 45.2,
      "samples_per_sec": 1280
    },
    ...
  ],
  "steps": [
    {
      "step": 100,
      "epoch": 1,
      "train_loss": 2.0123,
      "val_loss": 2.2345,
      "lr": 0.000295,
      "samples": 12800
    },
    ...
  ]
}
```

### 2. `training_epochs.csv`

每個 epoch 的統計數據

| epoch | avg_train_loss | val_loss | time | samples_per_sec |
| ----- | -------------- | -------- | ---- | --------------- |
| 1     | 2.1234         | 2.3456   | 45.2 | 1280            |
| 2     | 1.8765         | 2.0123   | 43.8 | 1320            |
| ...   | ...            | ...      | ...  | ...             |

### 3. `training_steps.csv`

每個評估步驟的詳細數據

| step | epoch | train_loss | val_loss | lr       | samples |
| ---- | ----- | ---------- | -------- | -------- | ------- |
| 100  | 1     | 2.0123     | 2.2345   | 0.000295 | 12800   |
| 200  | 1     | 1.9456     | 2.1234   | 0.000290 | 25600   |
| ...  | ...   | ...        | ...      | ...      | ...     |

### 4. `shakespeare_best.pt`

最佳模型權重（驗證損失最低）

## 📈 數據分析示例

### 使用 Python 分析

```python
import json
import pandas as pd
import matplotlib.pyplot as plt

# 讀取 JSON
with open('training_history.json', 'r') as f:
    history = json.load(f)

# 讀取 CSV
epochs_df = pd.read_csv('training_epochs.csv')
steps_df = pd.read_csv('training_steps.csv')

# 繪製損失曲線
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs_df['epoch'], epochs_df['avg_train_loss'], label='Train Loss')
plt.plot(epochs_df['epoch'], epochs_df['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Progress')

plt.subplot(1, 2, 2)
plt.plot(steps_df['step'], steps_df['val_loss'])
plt.xlabel('Step')
plt.ylabel('Validation Loss')
plt.title('Validation Loss over Steps')

plt.tight_layout()
plt.savefig('training_curves.png')
```

### 比較不同 Rank

```python
import glob
import json

# 收集所有實驗結果
results = {}

for rank in [1, 4, 8, 16]:
    try:
        with open(f'rank{rank}_training_history.json', 'r') as f:
            results[rank] = json.load(f)
    except:
        pass

# 比較最終性能
comparison = []
for rank, data in results.items():
    final_epoch = data['epochs'][-1]
    comparison.append({
        'rank': rank,
        'final_val_loss': final_epoch['val_loss'],
        'avg_samples_per_sec': sum(e['samples_per_sec'] for e in data['epochs']) / len(data['epochs']),
        'total_params': data['config'].get('total_params', 'N/A')
    })

df = pd.DataFrame(comparison)
print(df)
```

## 🔄 多次實驗管理

### 為不同 Rank 保存數據

```python
# 在訓練完成後重命名文件
import shutil

rank = 4  # 當前實驗的 rank

shutil.move('training_history.json', f'rank{rank}_training_history.json')
shutil.move('training_epochs.csv', f'rank{rank}_epochs.csv')
shutil.move('training_steps.csv', f'rank{rank}_steps.csv')
shutil.move('shakespeare_best.pt', f'shakespeare_rank{rank}_best.pt')
```

### 批量實驗腳本

```python
for rank in [1, 4, 8, 16]:
    print(f"\n{'='*80}")
    print(f"Training with MIMO Rank {rank}")
    print(f"{'='*80}\n")

    # 修改配置並訓練
    # ... (訓練代碼)

    # 重命名保存的文件
    import shutil
    shutil.move('training_history.json', f'rank{rank}_history.json')
    shutil.move('training_epochs.csv', f'rank{rank}_epochs.csv')
    shutil.move('shakespeare_best.pt', f'rank{rank}_best.pt')
```

## 📊 與 CV 結果對比

將 LLM 結果與 CIFAR-100 結果對比：

```python
# LLM 結果
llm_results = {
    1: {'val_loss': 2.1, 'params': '2.1M'},
    4: {'val_loss': 1.8, 'params': '2.4M'},
    8: {'val_loss': 1.75, 'params': '2.8M'},
}

# CV 結果（從之前的實驗）
cv_results = {
    1: {'accuracy': 0.7234, 'params': '1.5M'},
    4: {'accuracy': 0.7812, 'params': '1.6M'},
    8: {'accuracy': 0.7860, 'params': '1.7M'},
}

# 對比分析
print("MIMO Rank Saturation Comparison")
print("="*50)
print("CV (CIFAR-100):")
print(f"  Rank 4 vs 1: +{(cv_results[4]['accuracy'] - cv_results[1]['accuracy'])*100:.2f}%")
print(f"  Rank 8 vs 4: +{(cv_results[8]['accuracy'] - cv_results[4]['accuracy'])*100:.2f}%")
print("\nLLM (Shakespeare):")
print(f"  Rank 4 vs 1: {((llm_results[1]['val_loss'] - llm_results[4]['val_loss'])/llm_results[1]['val_loss'])*100:.2f}% improvement")
print(f"  Rank 8 vs 4: {((llm_results[4]['val_loss'] - llm_results[8]['val_loss'])/llm_results[4]['val_loss'])*100:.2f}% improvement")
```

## 💾 下載數據（Kaggle）

在 Kaggle Notebook 中下載所有數據：

```python
from IPython.display import FileLink, FileLinks

# 顯示所有可下載的文件
FileLinks('.')
```

或打包下載：

```python
import shutil

# 打包所有結果
shutil.make_archive('mamba3_llm_results', 'zip', '.',
                   base_dir=None,
                   verbose=True)

# 下載
FileLink('mamba3_llm_results.zip')
```

---

**自動保存**: 訓練完成後自動保存所有數據  
**格式**: JSON（完整）+ CSV（易分析）  
**用途**: Rank 對比、性能分析、論文圖表
