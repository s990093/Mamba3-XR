#!/usr/bin/env python3
"""
檢查可用的狀態相關指標
Check available state-related metrics
"""
import torch
from pathlib import Path

diag_path = Path('/Users/hungwei/Desktop/Proj/Mamba-Orin-Nano-Custom-S6-CUDA/mamba3/2025-12-28_14-52-14_rank1_e100_lr0.001_OneCycle_sweep_100ep/diagnostics_history.pt')
data = torch.load(diag_path, map_location='cpu', weights_only=False)

print("=" * 80)
print("檢查可用的狀態相關指標")
print("=" * 80)

# 檢查所有頂層鍵
print("\n頂層鍵:")
for key in sorted(data.keys()):
    print(f"  - {key}")

# 檢查 layer_activations
print("\n\nlayer_activations 結構:")
if 'layer_activations' in data:
    first_layer = list(data['layer_activations'].keys())[0]
    print(f"  範例層: {first_layer}")
    layer_data = data['layer_activations'][first_layer]
    if isinstance(layer_data, dict):
        print(f"  包含的指標:")
        for key, value in layer_data.items():
            if isinstance(value, list):
                print(f"    - {key}: list of length {len(value)}")
                if len(value) > 0:
                    print(f"      Sample values: {value[:3]}")
            else:
                print(f"    - {key}: {type(value)}")

# 檢查 delta_stats
print("\n\ndelta_stats 可用於狀態分析:")
if 'delta_stats' in data:
    print(f"  總共 {len(data['delta_stats'])} 個層")
    first_layer = list(data['delta_stats'].keys())[0]
    print(f"  範例層: {first_layer}")
    layer_data = data['delta_stats'][first_layer]
    if isinstance(layer_data, dict):
        for key, value in layer_data.items():
            if isinstance(value, list) and len(value) > 0:
                print(f"    - {key}: mean={sum(value)/len(value):.4f}, final={value[-1]:.4f}")

# 檢查 eigen_A
print("\n\neigen_A 可用於狀態分析:")
if 'eigen_A' in data:
    print(f"  總共 {len(data['eigen_A'])} 個 epochs")
    print(f"  每個 epoch 有 {len(data['eigen_A'][0])} 個特徵值")
    
    # 計算統計
    import numpy as np
    epoch_0 = data['eigen_A'][0]
    epoch_99 = data['eigen_A'][99]
    
    print(f"\n  Epoch 0:")
    print(f"    Max eigenvalue: {max(epoch_0):.2f}")
    print(f"    Min eigenvalue: {min(epoch_0):.2f}")
    print(f"    Mean eigenvalue: {np.mean(epoch_0):.2f}")
    
    print(f"\n  Epoch 99:")
    print(f"    Max eigenvalue: {max(epoch_99):.2f}")
    print(f"    Min eigenvalue: {min(epoch_99):.2f}")
    print(f"    Mean eigenvalue: {np.mean(epoch_99):.2f}")
