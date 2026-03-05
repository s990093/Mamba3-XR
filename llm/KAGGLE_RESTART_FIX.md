# Kaggle Notebook 重啟提示

如果遇到以下錯誤：

```
ValueError: AcceleratorState has already been initialized and cannot be changed
```

## 🔧 解決方案

### 方案 1：重啟 Notebook Runtime（推薦）

在 Kaggle Notebook 中：

1. 點擊右上角的 "⋮" (三個點)
2. 選擇 "Restart Session"
3. 重新運行所有 cells

### 方案 2：清除 Accelerator 狀態

在運行訓練之前，先運行這個 cell：

```python
# 清除 Accelerator 狀態
try:
    from accelerate.state import AcceleratorState
    AcceleratorState._reset_state()
    print("✅ Accelerator state cleared")
except:
    print("⚠️  Could not clear state, please restart runtime")
```

### 方案 3：使用單卡模式

如果只是想快速測試，可以暫時使用單卡：

```python
# 在 main() 中設置
USE_MULTI_GPU = False
```

## 📝 最佳實踐

### 首次運行

```python
# Cell 1: 安裝依賴
!pip install accelerate tqdm -q

# Cell 2: 導入並運行
%run mamba3_shakespeare_kaggle.py

# Cell 3: 啟動訓練（雙卡）
launch_training(num_gpus=2)
```

### 重新運行

如果需要重新運行訓練：

1. **重啟 Runtime**（最安全）
2. 重新運行所有 cells

或者：

```python
# 清除狀態後直接運行
from accelerate.state import AcceleratorState
AcceleratorState._reset_state()

# 重新啟動
launch_training(num_gpus=2)
```

## 🐛 為什麼會發生這個錯誤？

Accelerate 使用全局狀態來管理分散式訓練。在 Notebook 環境中：

1. 第一次運行 `Accelerator()` 時，它會初始化全局狀態
2. 如果再次運行但參數不同（如 `mixed_precision`），就會報錯
3. 這是為了防止配置不一致

## ✅ 已修復

最新版本的代碼已經包含自動處理：

```python
# 自動檢測並重用現有狀態
if AcceleratorState._shared_state != {}:
    print("⚠️  Accelerator already initialized, reusing...")
    accelerator = Accelerator()
else:
    accelerator = Accelerator(mixed_precision="fp16")
```

但最安全的方式仍然是重啟 Runtime。
