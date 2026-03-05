#!/usr/bin/env python3
"""
重新生成狀態健康度對比圖（使用實際可用的數據）
Regenerate state health comparison with actual available data
"""

import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from pathlib import Path

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

def extract_state_metrics(data):
    """提取狀態相關指標"""
    metrics = {}
    
    # 從 layer_activations 提取
    if 'layer_activations' in data:
        all_l2 = []
        all_var = []
        all_mean = []
        
        for layer_name, layer_data in data['layer_activations'].items():
            if isinstance(layer_data, dict):
                if 'l2' in layer_data:
                    all_l2.append(layer_data['l2'])
                if 'var' in layer_data:
                    all_var.append(layer_data['var'])
                if 'mean' in layer_data:
                    all_mean.append(layer_data['mean'])
        
        # 計算平均值
        if all_l2:
            num_epochs = len(all_l2[0])
            avg_l2 = []
            for epoch_idx in range(num_epochs):
                epoch_l2 = [l2[epoch_idx] for l2 in all_l2 if epoch_idx < len(l2)]
                if epoch_l2:
                    avg_l2.append(np.mean(epoch_l2))
            metrics['state_l2'] = avg_l2
        
        if all_var:
            num_epochs = len(all_var[0])
            avg_var = []
            for epoch_idx in range(num_epochs):
                epoch_var = [var[epoch_idx] for var in all_var if epoch_idx < len(var)]
                if epoch_var:
                    avg_var.append(np.mean(epoch_var))
            metrics['state_var'] = avg_var
    
    # 從 delta_stats 提取
    if 'delta_stats' in data:
        all_cv = []
        for layer_name, layer_data in data['delta_stats'].items():
            if isinstance(layer_data, dict) and 'cv' in layer_data:
                all_cv.append(layer_data['cv'])
        
        if all_cv:
            num_epochs = len(all_cv[0])
            avg_cv = []
            for epoch_idx in range(num_epochs):
                epoch_cv = [cv[epoch_idx] for cv in all_cv if epoch_idx < len(cv)]
                if epoch_cv:
                    avg_cv.append(np.mean(epoch_cv))
            metrics['delta_cv'] = avg_cv
    
    # 從 eigen_A 提取
    if 'eigen_A' in data:
        eigen_means = []
        eigen_stds = []
        eigen_max = []
        
        for epoch, eigenvalues in data['eigen_A'].items():
            if isinstance(eigenvalues, list) and len(eigenvalues) > 0:
                eigen_means.append(np.mean(eigenvalues))
                eigen_stds.append(np.std(eigenvalues))
                eigen_max.append(np.max(eigenvalues))
        
        metrics['eigen_mean'] = eigen_means
        metrics['eigen_std'] = eigen_stds
        metrics['eigen_max'] = eigen_max
    
    return metrics

def plot_state_health_fixed():
    """繪製狀態健康度對比圖（使用實際數據）"""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('內部狀態健康度對比', fontsize=16, fontweight='bold')
    
    colors = {'Rank 1': '#FF6B6B', 'Rank 4': '#4ECDC4', 'Rank 8': '#45B7D1'}
    
    # 載入所有數據
    all_metrics = {}
    for name, run_dir_name in RUNS.items():
        run_dir = BASE_DIR / run_dir_name
        diag_path = run_dir / 'diagnostics_history.pt'
        if diag_path.exists():
            data = torch.load(diag_path, map_location='cpu', weights_only=False)
            all_metrics[name] = extract_state_metrics(data)
    
    # 1. State L2 Norm
    ax = axes[0, 0]
    for name, metrics in all_metrics.items():
        if 'state_l2' in metrics:
            epochs = range(1, len(metrics['state_l2']) + 1)
            ax.plot(epochs, metrics['state_l2'], label=name, 
                   color=colors[name], linewidth=2, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('L2 Norm', fontsize=12)
    ax.set_title('狀態 L2 範數（層平均）', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # 2. State Variance
    ax = axes[0, 1]
    for name, metrics in all_metrics.items():
        if 'state_var' in metrics:
            epochs = range(1, len(metrics['state_var']) + 1)
            ax.plot(epochs, metrics['state_var'], label=name, 
                   color=colors[name], linewidth=2, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Variance', fontsize=12)
    ax.set_title('狀態變異數（層平均）', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # 3. Delta CV (Coefficient of Variation)
    ax = axes[0, 2]
    for name, metrics in all_metrics.items():
        if 'delta_cv' in metrics:
            epochs = range(1, len(metrics['delta_cv']) + 1)
            ax.plot(epochs, metrics['delta_cv'], label=name, 
                   color=colors[name], linewidth=2, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Coefficient of Variation', fontsize=12)
    ax.set_title('Delta 變異係數（層平均）', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # 4. Eigen Mean
    ax = axes[1, 0]
    for name, metrics in all_metrics.items():
        if 'eigen_mean' in metrics:
            epochs = range(1, len(metrics['eigen_mean']) + 1)
            ax.plot(epochs, metrics['eigen_mean'], label=name, 
                   color=colors[name], linewidth=2, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Mean Eigenvalue', fontsize=12)
    ax.set_title('SSM 平均特徵值', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # 5. Eigen Std
    ax = axes[1, 1]
    for name, metrics in all_metrics.items():
        if 'eigen_std' in metrics:
            epochs = range(1, len(metrics['eigen_std']) + 1)
            ax.plot(epochs, metrics['eigen_std'], label=name, 
                   color=colors[name], linewidth=2, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Std Eigenvalue', fontsize=12)
    ax.set_title('SSM 特徵值標準差', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # 6. Eigen Max
    ax = axes[1, 2]
    for name, metrics in all_metrics.items():
        if 'eigen_max' in metrics:
            epochs = range(1, len(metrics['eigen_max']) + 1)
            ax.plot(epochs, metrics['eigen_max'], label=name, 
                   color=colors[name], linewidth=2, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Max Eigenvalue', fontsize=12)
    ax.set_title('SSM 最大特徵值', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'comparison_state_health.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: comparison_state_health.png (with actual data)")
    plt.close()
    
    # 複製到 docs
    import shutil
    shutil.copy(OUTPUT_DIR / 'comparison_state_health.png', 
                BASE_DIR / 'docs' / 'comparison_state_health.png')
    print(f"✓ Copied to docs/")

if __name__ == '__main__':
    plot_state_health_fixed()
