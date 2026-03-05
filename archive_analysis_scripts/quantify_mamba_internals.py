#!/usr/bin/env python3
"""
深入分析 Mamba 內部狀態並量化
Quantify Mamba internal states
"""

import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from pathlib import Path
import json

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

RUNS = {
    'Rank 1': '2025-12-28_14-52-14_rank1_e100_lr0.001_OneCycle_sweep_100ep',
    'Rank 4': '2025-12-28_18-54-07_rank4_e100_lr0.001_OneCycle_sweep_100ep',
    'Rank 8': '2025-12-28_23-13-51_rank8_e100_lr0.001_OneCycle_sweep_100ep'
}

BASE_DIR = Path('/Users/hungwei/Desktop/Proj/Mamba-Orin-Nano-Custom-S6-CUDA/mamba3')
OUTPUT_DIR = Path('/Users/hungwei/.gemini/antigravity/brain/bd37de13-51b9-479f-bbe1-066937a600cf')

def analyze_mamba_internals(data):
    """深入分析 Mamba 內部狀態"""
    analysis = {}
    
    # 1. Layer activations 分析
    if 'layer_activations' in data:
        l2_norms = []
        variances = []
        means = []
        
        for layer_name, layer_data in data['layer_activations'].items():
            if isinstance(layer_data, dict):
                if 'l2' in layer_data and len(layer_data['l2']) > 0:
                    l2_norms.append(layer_data['l2'][-1])  # 最終值
                if 'var' in layer_data and len(layer_data['var']) > 0:
                    variances.append(layer_data['var'][-1])
                if 'mean' in layer_data and len(layer_data['mean']) > 0:
                    means.append(abs(layer_data['mean'][-1]))
        
        analysis['activation_stats'] = {
            'l2_mean': float(np.mean(l2_norms)) if l2_norms else 0,
            'l2_std': float(np.std(l2_norms)) if l2_norms else 0,
            'l2_max': float(np.max(l2_norms)) if l2_norms else 0,
            'l2_min': float(np.min(l2_norms)) if l2_norms else 0,
            'var_mean': float(np.mean(variances)) if variances else 0,
            'var_std': float(np.std(variances)) if variances else 0,
            'num_layers': len(l2_norms)
        }
    
    # 2. MIMO ranks 分析
    if 'mimo_ranks' in data:
        final_ranks = []
        mean_ranks = []
        rank_stds = []
        
        for layer_name, ranks in data['mimo_ranks'].items():
            if isinstance(ranks, list) and len(ranks) > 0:
                final_ranks.append(ranks[-1])
                mean_ranks.append(np.mean(ranks))
                rank_stds.append(np.std(ranks))
        
        analysis['mimo_stats'] = {
            'final_rank_mean': float(np.mean(final_ranks)) if final_ranks else 0,
            'final_rank_std': float(np.std(final_ranks)) if final_ranks else 0,
            'final_rank_max': float(np.max(final_ranks)) if final_ranks else 0,
            'final_rank_min': float(np.min(final_ranks)) if final_ranks else 0,
            'rank_stability_mean': float(np.mean(rank_stds)) if rank_stds else 0,
            'num_layers': len(final_ranks)
        }
    
    # 3. Delta stats 分析
    if 'delta_stats' in data:
        final_cvs = []
        final_means = []
        
        for layer_name, layer_data in data['delta_stats'].items():
            if isinstance(layer_data, dict):
                if 'cv' in layer_data and len(layer_data['cv']) > 0:
                    final_cvs.append(layer_data['cv'][-1])
                if 'mean' in layer_data and len(layer_data['mean']) > 0:
                    final_means.append(layer_data['mean'][-1])
        
        analysis['delta_stats'] = {
            'cv_mean': float(np.mean(final_cvs)) if final_cvs else 0,
            'cv_std': float(np.std(final_cvs)) if final_cvs else 0,
            'delta_mean_avg': float(np.mean(final_means)) if final_means else 0,
            'num_layers': len(final_cvs)
        }
    
    # 4. Eigen A 分析
    if 'eigen_A' in data:
        final_epoch = max(data['eigen_A'].keys())
        eigenvalues = data['eigen_A'][final_epoch]
        
        if isinstance(eigenvalues, list) and len(eigenvalues) > 0:
            analysis['eigen_stats'] = {
                'mean': float(np.mean(eigenvalues)),
                'std': float(np.std(eigenvalues)),
                'max': float(np.max(eigenvalues)),
                'min': float(np.min(eigenvalues)),
                'condition_number': float(np.max(eigenvalues) / np.min(eigenvalues)) if np.min(eigenvalues) > 0 else float('inf'),
                'num_eigenvalues': len(eigenvalues)
            }
    
    return analysis

def main():
    print("=" * 80)
    print("深入分析 Mamba 內部狀態")
    print("=" * 80)
    
    all_analysis = {}
    
    for name, run_dir_name in RUNS.items():
        run_dir = BASE_DIR / run_dir_name
        diag_path = run_dir / 'diagnostics_history.pt'
        
        if diag_path.exists():
            print(f"\n處理 {name}...")
            data = torch.load(diag_path, map_location='cpu', weights_only=False)
            analysis = analyze_mamba_internals(data)
            all_analysis[name] = analysis
            
            # 打印摘要
            print(f"\n  激活統計:")
            if 'activation_stats' in analysis:
                stats = analysis['activation_stats']
                print(f"    L2 範數: {stats['l2_mean']:.2f} ± {stats['l2_std']:.2f} (範圍: {stats['l2_min']:.2f} - {stats['l2_max']:.2f})")
                print(f"    變異數: {stats['var_mean']:.2f} ± {stats['var_std']:.2f}")
            
            print(f"\n  MIMO 統計:")
            if 'mimo_stats' in analysis:
                stats = analysis['mimo_stats']
                print(f"    最終秩: {stats['final_rank_mean']:.2f} ± {stats['final_rank_std']:.2f}")
                print(f"    秩範圍: {stats['final_rank_min']:.2f} - {stats['final_rank_max']:.2f}")
                print(f"    秩穩定性: {stats['rank_stability_mean']:.2f}")
            
            print(f"\n  Delta 統計:")
            if 'delta_stats' in analysis:
                stats = analysis['delta_stats']
                print(f"    變異係數: {stats['cv_mean']:.4f} ± {stats['cv_std']:.4f}")
            
            print(f"\n  特徵值統計:")
            if 'eigen_stats' in analysis:
                stats = analysis['eigen_stats']
                print(f"    平均: {stats['mean']:.4f}")
                print(f"    條件數: {stats['condition_number']:.2f}")
    
    # 保存為 JSON
    with open(OUTPUT_DIR / 'mamba_internals_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(all_analysis, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Saved: mamba_internals_analysis.json")
    
    # 生成對比表格
    print("\n" + "=" * 80)
    print("量化對比表格")
    print("=" * 80)
    
    print("\n狀態分布範圍（L2 範數範圍）:")
    print(f"{'Rank':<10} {'L2 範圍':<20} {'評級':<10}")
    print("-" * 40)
    for name in ['Rank 1', 'Rank 4', 'Rank 8']:
        if name in all_analysis and 'activation_stats' in all_analysis[name]:
            stats = all_analysis[name]['activation_stats']
            range_val = stats['l2_max'] - stats['l2_min']
            rating = "窄" if range_val < 5 else ("中等" if range_val < 10 else "廣")
            print(f"{name:<10} {range_val:<20.2f} {rating:<10}")
    
    print("\n狀態豐富度（MIMO 秩）:")
    print(f"{'Rank':<10} {'平均 MIMO 秩':<20} {'評級':<10}")
    print("-" * 40)
    for name in ['Rank 1', 'Rank 4', 'Rank 8']:
        if name in all_analysis and 'mimo_stats' in all_analysis[name]:
            stats = all_analysis[name]['mimo_stats']
            rank_val = stats['final_rank_mean']
            rating = "低" if rank_val < 20 else ("高" if rank_val < 28 else "最高")
            print(f"{name:<10} {rank_val:<20.2f} {rating:<10}")
    
    print("\n表徵能力（特徵值條件數，越小越好）:")
    print(f"{'Rank':<10} {'條件數':<20} {'評級':<10}")
    print("-" * 40)
    for name in ['Rank 1', 'Rank 4', 'Rank 8']:
        if name in all_analysis and 'eigen_stats' in all_analysis[name]:
            stats = all_analysis[name]['eigen_stats']
            cond = stats['condition_number']
            rating = "最強" if cond < 1.05 else ("良好" if cond < 1.1 else "受限")
            print(f"{name:<10} {cond:<20.2f} {rating:<10}")

if __name__ == '__main__':
    main()
