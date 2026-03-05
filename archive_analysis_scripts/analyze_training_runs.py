#!/usr/bin/env python3
"""
完整分析三個訓練檔案的腳本
Comprehensive analysis script for three training runs
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

# 定義三個訓練目錄
RUNS = {
    'Rank 1': '2025-12-28_14-52-14_rank1_e100_lr0.001_OneCycle_sweep_100ep',
    'Rank 4': '2025-12-28_18-54-07_rank4_e100_lr0.001_OneCycle_sweep_100ep',
    'Rank 8': '2025-12-28_23-13-51_rank8_e100_lr0.001_OneCycle_sweep_100ep'
}

BASE_DIR = Path('/Users/hungwei/Desktop/Proj/Mamba-Orin-Nano-Custom-S6-CUDA/mamba3')
OUTPUT_DIR = Path('/Users/hungwei/.gemini/antigravity/brain/bd37de13-51b9-479f-bbe1-066937a600cf')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_diagnostics(run_dir):
    """載入診斷數據"""
    diag_path = run_dir / 'diagnostics_history.pt'
    if not diag_path.exists():
        print(f"Warning: {diag_path} not found")
        return None
    
    data = torch.load(diag_path, map_location='cpu', weights_only=False)
    return data

def get_model_size(run_dir):
    """獲取模型大小"""
    model_path = run_dir / 'best_model.pth'
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        return size_mb
    return None

def extract_metrics(data):
    """提取關鍵指標"""
    metrics = {}
    
    # 訓練指標 - 使用正確的鍵名
    if 'loss' in data:
        metrics['train_loss'] = data['loss']
    if 'accuracy' in data:
        metrics['train_acc'] = data['accuracy']
    if 'val_loss' in data:
        metrics['val_loss'] = data['val_loss']
    if 'val_accuracy' in data:
        metrics['val_acc'] = data['val_accuracy']
    if 'val_acc_ema' in data:
        metrics['val_acc_ema'] = data['val_acc_ema']
    if 'val_f1' in data:
        metrics['f1_score'] = data['val_f1']
    if 'val_acc5' in data:
        metrics['val_acc5'] = data['val_acc5']
    if 'val_acc5_ema' in data:
        metrics['val_acc5_ema'] = data['val_acc5_ema']
    
    # MIMO ranks - 計算平均值
    if 'mimo_ranks' in data:
        # 收集所有層的 MIMO ranks
        all_ranks = []
        for key, values in data['mimo_ranks'].items():
            if isinstance(values, list) and len(values) > 0:
                all_ranks.append(values)
        
        if all_ranks:
            # 計算每個 epoch 的平均 rank
            num_epochs = len(all_ranks[0])
            avg_ranks = []
            for epoch_idx in range(num_epochs):
                epoch_ranks = [ranks[epoch_idx] for ranks in all_ranks if epoch_idx < len(ranks)]
                if epoch_ranks:
                    avg_ranks.append(np.mean(epoch_ranks))
            metrics['mimo_rank_avg'] = avg_ranks
    
    # Layer stats - 提取梯度信息
    if 'layer_stats' in data:
        grad_norms = []
        grad_means = []
        
        for layer_name, layer_data in data['layer_stats'].items():
            if isinstance(layer_data, dict):
                if 'grad_norm' in layer_data and isinstance(layer_data['grad_norm'], list):
                    grad_norms.append(layer_data['grad_norm'])
                if 'grad_mean' in layer_data and isinstance(layer_data['grad_mean'], list):
                    grad_means.append(layer_data['grad_mean'])
        
        # 計算平均梯度範數
        if grad_norms:
            num_epochs = len(grad_norms[0])
            avg_grad_norms = []
            for epoch_idx in range(num_epochs):
                epoch_norms = [norms[epoch_idx] for norms in grad_norms if epoch_idx < len(norms)]
                if epoch_norms:
                    avg_grad_norms.append(np.mean(epoch_norms))
            metrics['grad_norm'] = avg_grad_norms
    
    return metrics

def plot_training_metrics(all_data):
    """繪製訓練指標對比圖"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('訓練指標對比 (Rank 1 vs Rank 4 vs Rank 8)', fontsize=16, fontweight='bold')
    
    colors = {'Rank 1': '#FF6B6B', 'Rank 4': '#4ECDC4', 'Rank 8': '#45B7D1'}
    
    # Loss
    ax = axes[0, 0]
    for name, data in all_data.items():
        if 'train_loss' in data:
            epochs = range(1, len(data['train_loss']) + 1)
            ax.plot(epochs, data['train_loss'], label=f'{name} (Train)', 
                   color=colors[name], linewidth=2, alpha=0.8)
        if 'val_loss' in data:
            epochs = range(1, len(data['val_loss']) + 1)
            ax.plot(epochs, data['val_loss'], label=f'{name} (Val)', 
                   color=colors[name], linewidth=2, linestyle='--', alpha=0.6)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('訓練與驗證損失', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Accuracy
    ax = axes[0, 1]
    for name, data in all_data.items():
        if 'train_acc' in data:
            epochs = range(1, len(data['train_acc']) + 1)
            ax.plot(epochs, data['train_acc'], label=f'{name} (Train)', 
                   color=colors[name], linewidth=2, alpha=0.8)
        if 'val_acc' in data:
            epochs = range(1, len(data['val_acc']) + 1)
            ax.plot(epochs, data['val_acc'], label=f'{name} (Val)', 
                   color=colors[name], linewidth=2, linestyle='--', alpha=0.6)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('訓練與驗證準確率', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # F1 Score
    ax = axes[1, 0]
    for name, data in all_data.items():
        if 'f1_score' in data:
            epochs = range(1, len(data['f1_score']) + 1)
            ax.plot(epochs, data['f1_score'], label=name, 
                   color=colors[name], linewidth=2, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('F1 分數', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Final metrics comparison (bar chart)
    ax = axes[1, 1]
    metrics_names = []
    rank1_vals = []
    rank4_vals = []
    rank8_vals = []
    
    for metric in ['val_acc', 'f1_score']:
        if metric in all_data['Rank 1']:
            metrics_names.append(metric.replace('_', ' ').title())
            rank1_vals.append(all_data['Rank 1'][metric][-1] if len(all_data['Rank 1'][metric]) > 0 else 0)
            rank4_vals.append(all_data['Rank 4'][metric][-1] if len(all_data['Rank 4'][metric]) > 0 else 0)
            rank8_vals.append(all_data['Rank 8'][metric][-1] if len(all_data['Rank 8'][metric]) > 0 else 0)
    
    x = np.arange(len(metrics_names))
    width = 0.25
    ax.bar(x - width, rank1_vals, width, label='Rank 1', color=colors['Rank 1'], alpha=0.8)
    ax.bar(x, rank4_vals, width, label='Rank 4', color=colors['Rank 4'], alpha=0.8)
    ax.bar(x + width, rank8_vals, width, label='Rank 8', color=colors['Rank 8'], alpha=0.8)
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('最終指標對比', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'comparison_training_metrics.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: comparison_training_metrics.png")
    plt.close()

def plot_gradient_health(all_data):
    """繪製梯度健康度對比圖"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('梯度健康度對比', fontsize=16, fontweight='bold')
    
    colors = {'Rank 1': '#FF6B6B', 'Rank 4': '#4ECDC4', 'Rank 8': '#45B7D1'}
    
    # Gradient Norm
    ax = axes[0]
    for name, data in all_data.items():
        if 'grad_norm' in data:
            epochs = range(1, len(data['grad_norm']) + 1)
            ax.plot(epochs, data['grad_norm'], label=name, 
                   color=colors[name], linewidth=2, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Gradient Norm', fontsize=12)
    ax.set_title('梯度範數', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Gradient SNR
    ax = axes[1]
    for name, data in all_data.items():
        if 'grad_snr' in data:
            epochs = range(1, len(data['grad_snr']) + 1)
            ax.plot(epochs, data['grad_snr'], label=name, 
                   color=colors[name], linewidth=2, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Signal-to-Noise Ratio', fontsize=12)
    ax.set_title('梯度信噪比 (SNR)', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'comparison_gradient_health.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: comparison_gradient_health.png")
    plt.close()

def plot_state_health(all_data):
    """繪製狀態健康度對比圖"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('內部狀態健康度對比', fontsize=16, fontweight='bold')
    
    colors = {'Rank 1': '#FF6B6B', 'Rank 4': '#4ECDC4', 'Rank 8': '#45B7D1'}
    
    # State L2 Norm
    ax = axes[0]
    for name, data in all_data.items():
        if 'state_l2_norm' in data:
            epochs = range(1, len(data['state_l2_norm']) + 1)
            ax.plot(epochs, data['state_l2_norm'], label=name, 
                   color=colors[name], linewidth=2, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('L2 Norm', fontsize=12)
    ax.set_title('狀態 L2 範數', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # State Variance
    ax = axes[1]
    for name, data in all_data.items():
        if 'state_variance' in data:
            epochs = range(1, len(data['state_variance']) + 1)
            ax.plot(epochs, data['state_variance'], label=name, 
                   color=colors[name], linewidth=2, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Variance', fontsize=12)
    ax.set_title('狀態變異數', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Effective Rank
    ax = axes[2]
    for name, data in all_data.items():
        if 'effective_rank' in data:
            epochs = range(1, len(data['effective_rank']) + 1)
            ax.plot(epochs, data['effective_rank'], label=name, 
                   color=colors[name], linewidth=2, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Effective Rank', fontsize=12)
    ax.set_title('有效秩 (Effective Rank)', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'comparison_state_health.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: comparison_state_health.png")
    plt.close()

def plot_a_log_stability(all_data):
    """繪製 A_log 穩定性對比圖"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('A_log 參數穩定性對比', fontsize=16, fontweight='bold')
    
    colors = {'Rank 1': '#FF6B6B', 'Rank 4': '#4ECDC4', 'Rank 8': '#45B7D1'}
    
    # A_log Mean
    ax = axes[0]
    for name, data in all_data.items():
        if 'A_log_mean' in data:
            epochs = range(1, len(data['A_log_mean']) + 1)
            ax.plot(epochs, data['A_log_mean'], label=name, 
                   color=colors[name], linewidth=2, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Mean Value', fontsize=12)
    ax.set_title('A_log 平均值', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # A_log Std
    ax = axes[1]
    for name, data in all_data.items():
        if 'A_log_std' in data:
            epochs = range(1, len(data['A_log_std']) + 1)
            ax.plot(epochs, data['A_log_std'], label=name, 
                   color=colors[name], linewidth=2, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Standard Deviation', fontsize=12)
    ax.set_title('A_log 標準差', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'comparison_a_log_stability.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: comparison_a_log_stability.png")
    plt.close()

def plot_mimo_ranks(all_data):
    """繪製 MIMO Rank 對比圖"""
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.suptitle('MIMO 有效秩對比 (Effective Rank)', fontsize=16, fontweight='bold')
    
    colors = {'Rank 1': '#FF6B6B', 'Rank 4': '#4ECDC4', 'Rank 8': '#45B7D1'}
    
    for name, data in all_data.items():
        if 'mimo_rank_avg' in data:
            epochs = range(1, len(data['mimo_rank_avg']) + 1)
            ax.plot(epochs, data['mimo_rank_avg'], label=name, 
                   color=colors[name], linewidth=2.5, alpha=0.8, marker='o', 
                   markersize=3, markevery=10)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Average MIMO Rank', fontsize=12)
    ax.set_title('各層平均 MIMO 秩隨訓練變化', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'comparison_mimo_ranks.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: comparison_mimo_ranks.png")
    plt.close()


def plot_model_size_comparison(model_sizes):
    """繪製模型大小對比圖"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = list(model_sizes.keys())
    sizes = list(model_sizes.values())
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    bars = ax.bar(names, sizes, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # 添加數值標籤
    for bar, size in zip(bars, sizes):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{size:.2f} MB',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('模型大小 (MB)', fontsize=12)
    ax.set_title('模型大小對比', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'comparison_model_size.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: comparison_model_size.png")
    plt.close()

def generate_summary_stats(all_data, model_sizes):
    """生成統計摘要"""
    summary = {}
    
    for name, data in all_data.items():
        stats = {
            'model_size_mb': model_sizes.get(name, 0),
        }
        
        # 最終指標
        if 'val_acc' in data and len(data['val_acc']) > 0:
            stats['final_val_acc'] = float(data['val_acc'][-1])
            stats['best_val_acc'] = float(max(data['val_acc']))
        
        if 'val_acc_ema' in data and len(data['val_acc_ema']) > 0:
            stats['final_val_acc_ema'] = float(data['val_acc_ema'][-1])
            stats['best_val_acc_ema'] = float(max(data['val_acc_ema']))
        
        if 'f1_score' in data and len(data['f1_score']) > 0:
            stats['final_f1'] = float(data['f1_score'][-1])
            stats['best_f1'] = float(max(data['f1_score']))
        
        if 'val_loss' in data and len(data['val_loss']) > 0:
            stats['final_val_loss'] = float(data['val_loss'][-1])
            stats['best_val_loss'] = float(min(data['val_loss']))
        
        if 'train_loss' in data and len(data['train_loss']) > 0:
            stats['final_train_loss'] = float(data['train_loss'][-1])
        
        if 'train_acc' in data and len(data['train_acc']) > 0:
            stats['final_train_acc'] = float(data['train_acc'][-1])
        
        if 'val_acc5' in data and len(data['val_acc5']) > 0:
            stats['final_val_acc5'] = float(data['val_acc5'][-1])
            stats['best_val_acc5'] = float(max(data['val_acc5']))
        
        # 梯度統計
        if 'grad_norm' in data and len(data['grad_norm']) > 0:
            stats['avg_grad_norm'] = float(np.mean(data['grad_norm']))
            stats['final_grad_norm'] = float(data['grad_norm'][-1])
        
        # MIMO rank 統計
        if 'mimo_rank_avg' in data and len(data['mimo_rank_avg']) > 0:
            stats['avg_mimo_rank'] = float(np.mean(data['mimo_rank_avg']))
            stats['final_mimo_rank'] = float(data['mimo_rank_avg'][-1])
            stats['max_mimo_rank'] = float(max(data['mimo_rank_avg']))
            stats['min_mimo_rank'] = float(min(data['mimo_rank_avg']))
        
        summary[name] = stats
    
    # 保存為 JSON
    with open(OUTPUT_DIR / 'summary_statistics.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved: summary_statistics.json")
    return summary

def main():
    print("=" * 80)
    print("開始分析三個訓練檔案...")
    print("=" * 80)
    
    # 載入所有數據
    all_diagnostics = {}
    all_metrics = {}
    model_sizes = {}
    
    for name, run_dir_name in RUNS.items():
        run_dir = BASE_DIR / run_dir_name
        print(f"\n處理 {name}: {run_dir_name}")
        
        # 載入診斷數據
        diag_data = load_diagnostics(run_dir)
        if diag_data is not None:
            all_diagnostics[name] = diag_data
            all_metrics[name] = extract_metrics(diag_data)
            print(f"  ✓ 載入診斷數據: {len(diag_data)} 個指標")
        
        # 獲取模型大小
        size = get_model_size(run_dir)
        if size is not None:
            model_sizes[name] = size
            print(f"  ✓ 模型大小: {size:.2f} MB")
    
    print("\n" + "=" * 80)
    print("生成對比圖表...")
    print("=" * 80)
    
    # 生成各種對比圖
    plot_training_metrics(all_metrics)
    plot_gradient_health(all_metrics)
    plot_state_health(all_metrics)
    plot_a_log_stability(all_metrics)
    plot_mimo_ranks(all_metrics)
    plot_model_size_comparison(model_sizes)
    
    # 生成統計摘要
    print("\n" + "=" * 80)
    print("生成統計摘要...")
    print("=" * 80)
    summary = generate_summary_stats(all_metrics, model_sizes)
    
    print("\n" + "=" * 80)
    print("分析完成！")
    print("=" * 80)
    print(f"\n所有圖表已保存至: {OUTPUT_DIR}")
    print("\n生成的檔案:")
    print("  - comparison_training_metrics.png")
    print("  - comparison_gradient_health.png")
    print("  - comparison_state_health.png")
    print("  - comparison_a_log_stability.png")
    print("  - comparison_mimo_ranks.png")
    print("  - comparison_model_size.png")
    print("  - summary_statistics.json")
    
    # 打印摘要
    print("\n" + "=" * 80)
    print("統計摘要:")
    print("=" * 80)
    for name, stats in summary.items():
        print(f"\n{name}:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

if __name__ == '__main__':
    main()
