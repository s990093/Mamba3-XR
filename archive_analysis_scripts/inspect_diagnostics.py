#!/usr/bin/env python3
"""
檢查更詳細的診斷數據結構
"""
import torch
from pathlib import Path

diag_path = Path('/Users/hungwei/Desktop/Proj/Mamba-Orin-Nano-Custom-S6-CUDA/mamba3/2025-12-28_14-52-14_rank1_e100_lr0.001_OneCycle_sweep_100ep/diagnostics_history.pt')

data = torch.load(diag_path, map_location='cpu', weights_only=False)

print("Nested dictionary structures:")
print("=" * 80)

# Check layer_stats
if 'layer_stats' in data:
    print("\nlayer_stats keys:")
    for key in sorted(data['layer_stats'].keys()):
        value = data['layer_stats'][key]
        if isinstance(value, list):
            print(f"  {key}: list of length {len(value)}")
        else:
            print(f"  {key}: {type(value)}")

# Check delta_stats
if 'delta_stats' in data:
    print("\ndelta_stats keys:")
    for key in sorted(data['delta_stats'].keys()):
        value = data['delta_stats'][key]
        if isinstance(value, list):
            print(f"  {key}: list of length {len(value)}")
        else:
            print(f"  {key}: {type(value)}")

# Check mimo_ranks
if 'mimo_ranks' in data:
    print("\nmimo_ranks keys:")
    for key in sorted(data['mimo_ranks'].keys()):
        value = data['mimo_ranks'][key]
        if isinstance(value, list):
            print(f"  {key}: list of length {len(value)}")
        else:
            print(f"  {key}: {type(value)}")

# Check eigen_A
if 'eigen_A' in data:
    print("\neigen_A keys:")
    for key in sorted(data['eigen_A'].keys()):
        value = data['eigen_A'][key]
        if isinstance(value, list):
            print(f"  {key}: list of length {len(value)}")
        else:
            print(f"  {key}: {type(value)}")
