#!/usr/bin/env python3
"""
檢查 A_log SNR 數據結構
Check A_log SNR data structure
"""
import torch
from pathlib import Path
import numpy as np

diag_path = Path('/Users/hungwei/Desktop/Proj/Mamba-Orin-Nano-Custom-S6-CUDA/mamba3/2025-12-28_14-52-14_rank1_e100_lr0.001_OneCycle_sweep_100ep/diagnostics_history.pt')
data = torch.load(diag_path, map_location='cpu', weights_only=False)

print("=" * 80)
print("檢查 A_log 相關數據")
print("=" * 80)

# 檢查 layer_stats 中的 A_log
print("\nlayer_stats 中的 A_log 層:")
a_log_layers = [key for key in data['layer_stats'].keys() if 'A_log' in key]
print(f"找到 {len(a_log_layers)} 個 A_log 層")

for layer_key in sorted(a_log_layers)[:3]:  # 只顯示前 3 個
    print(f"\n{layer_key}:")
    layer_data = data['layer_stats'][layer_key]
    
    if isinstance(layer_data, dict):
        for metric_name, metric_values in layer_data.items():
            if isinstance(metric_values, list):
                # 檢查是否有非零值
                non_zero = [v for v in metric_values if v != 0]
                print(f"  {metric_name}:")
                print(f"    長度: {len(metric_values)}")
                print(f"    非零值數量: {len(non_zero)}")
                if len(non_zero) > 0:
                    print(f"    前5個非零值: {non_zero[:5]}")
                    print(f"    後5個非零值: {non_zero[-5:]}")
                    print(f"    平均值: {np.mean(non_zero):.6f}")
                else:
                    print(f"    所有值都是 0！")
                print(f"    前5個值: {metric_values[:5]}")
                print(f"    後5個值: {metric_values[-5:]}")

# 檢查是否有其他可用的 A_log 相關指標
print("\n\n檢查其他可能的 A_log 指標:")
for key in data.keys():
    if 'log' in key.lower() or 'a_' in key.lower():
        print(f"  - {key}: {type(data[key])}")
