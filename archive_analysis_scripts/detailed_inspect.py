#!/usr/bin/env python3
"""
深入分析 diagnostics 中的所有數據
Extract all available metrics from diagnostics
"""
import torch
from pathlib import Path
import json

diag_path = Path('/Users/hungwei/Desktop/Proj/Mamba-Orin-Nano-Custom-S6-CUDA/mamba3/2025-12-28_14-52-14_rank1_e100_lr0.001_OneCycle_sweep_100ep/diagnostics_history.pt')

data = torch.load(diag_path, map_location='cpu', weights_only=False)

print("=" * 80)
print("完整 Diagnostics 數據結構分析")
print("=" * 80)

# 1. 檢查 layer_stats 的詳細結構
print("\n1. Layer Stats 詳細結構:")
if 'layer_stats' in data:
    # 取第一個層的第一個參數來看結構
    first_layer_key = list(data['layer_stats'].keys())[0]
    first_layer_data = data['layer_stats'][first_layer_key]
    
    print(f"\n  範例層: {first_layer_key}")
    print(f"  數據類型: {type(first_layer_data)}")
    
    if isinstance(first_layer_data, dict):
        print(f"  包含的指標:")
        for key, value in first_layer_data.items():
            if isinstance(value, list):
                print(f"    - {key}: list of length {len(value)}")
                if len(value) > 0:
                    print(f"      First value: {value[0]}")
            else:
                print(f"    - {key}: {type(value)}")

# 2. 檢查 delta_stats 的詳細結構
print("\n2. Delta Stats 詳細結構:")
if 'delta_stats' in data:
    first_layer_key = list(data['delta_stats'].keys())[0]
    first_layer_data = data['delta_stats'][first_layer_key]
    
    print(f"\n  範例層: {first_layer_key}")
    print(f"  數據類型: {type(first_layer_data)}")
    
    if isinstance(first_layer_data, dict):
        print(f"  包含的指標:")
        for key, value in first_layer_data.items():
            if isinstance(value, list):
                print(f"    - {key}: list of length {len(value)}")
                if len(value) > 0:
                    print(f"      First value: {value[0]}")
            else:
                print(f"    - {key}: {type(value)}")

# 3. 檢查 eigen_A 的詳細結構
print("\n3. Eigen A 詳細結構:")
if 'eigen_A' in data:
    # 取第一個 epoch
    first_epoch_data = data['eigen_A'][0]
    
    print(f"\n  Epoch 0 數據:")
    print(f"  數據類型: {type(first_epoch_data)}")
    print(f"  長度: {len(first_epoch_data)}")
    if len(first_epoch_data) > 0:
        print(f"  第一個元素: {first_epoch_data[0]}")
        print(f"  元素類型: {type(first_epoch_data[0])}")

# 4. 檢查 layer_activations
print("\n4. Layer Activations 結構:")
if 'layer_activations' in data:
    print(f"  數據類型: {type(data['layer_activations'])}")
    if isinstance(data['layer_activations'], dict):
        first_key = list(data['layer_activations'].keys())[0]
        print(f"  範例層: {first_key}")
        print(f"  數據: {type(data['layer_activations'][first_key])}")

# 5. 檢查 delta_heatmap
print("\n5. Delta Heatmap 結構:")
if 'delta_heatmap' in data:
    print(f"  數據類型: {type(data['delta_heatmap'])}")
    if isinstance(data['delta_heatmap'], dict):
        for key, value in data['delta_heatmap'].items():
            print(f"  {key}: {type(value)}")
            if hasattr(value, 'shape'):
                print(f"    Shape: {value.shape}")

# 6. 檢查 MIMO ranks 的詳細數據
print("\n6. MIMO Ranks 詳細數據:")
if 'mimo_ranks' in data:
    print(f"  總共 {len(data['mimo_ranks'])} 個層")
    for layer_key in sorted(data['mimo_ranks'].keys())[:3]:  # 只顯示前3個
        ranks = data['mimo_ranks'][layer_key]
        print(f"\n  {layer_key}:")
        print(f"    長度: {len(ranks)}")
        print(f"    前5個值: {ranks[:5]}")
        print(f"    後5個值: {ranks[-5:]}")
        print(f"    平均: {sum(ranks)/len(ranks):.2f}")
        print(f"    最大: {max(ranks):.2f}")
        print(f"    最小: {min(ranks):.2f}")
