#!/usr/bin/env python3
"""
增強版完整分析 - 包含所有深度指標
Enhanced comprehensive analysis with all deep metrics
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
        return None
    return torch.load(diag_path, map_location='cpu', weights_only=False)

def extract_layer_wise_mimo_ranks(data):
    """提取每層的 MIMO ranks"""
    if 'mimo_ranks' not in data:
        return None
    
    layer_ranks = {}
    for layer_key, ranks in data['mimo_ranks'].items():
        layer_ranks[layer_key] = {
            'values': ranks,
            'mean': np.mean(ranks),
            'std': np.std(ranks),
            'final': ranks[-1] if len(ranks) > 0 else 0,
            'max': np.max(ranks),
            'min': np.min(ranks)
        }
    return layer_ranks

def extract_a_log_stats(data):
    """提取 A_log 相關統計"""
    if 'layer_stats' not in data:
        return None
    
    a_log_stats = {}
    for layer_key, layer_data in data['layer_stats'].items():
        if 'A_log' in layer_key and isinstance(layer_data, dict):
            a_log_stats[layer_key] = {
                'snr': layer_data.get('snr', []),
                'update_ratio': layer_data.get('update_ratio', []),
                'grad_norm': layer_data.get('grad_norm', [])
            }
    return a_log_stats

def extract_delta_stats(data):
    """提取 delta 統計"""
    if 'delta_stats' not in data:
        return None
    
    delta_stats = {}
    for layer_key, layer_data in data['delta_stats'].items():
        if isinstance(layer_data, dict):
            delta_stats[layer_key] = {
                'cv': layer_data.get('cv', []),
                'mean': layer_data.get('mean', [])
            }
    return delta_stats

def extract_eigen_a(data):
    """提取 eigen_A 數據"""
    if 'eigen_A' not in data:
        return None
    
    # eigen_A 是一個字典，key 是 epoch，value 是 eigenvalues 列表
    eigen_stats = {}
    for epoch, eigenvalues in data['eigen_A'].items():
        if isinstance(eigenvalues, list) and len(eigenvalues) > 0:
            eigen_stats[int(epoch)] = {
                'values': eigenvalues,
                'max': float(np.max(eigenvalues)),
                'min': float(np.min(eigenvalues)),
                'mean': float(np.mean(eigenvalues)),
                'std': float(np.std(eigenvalues))
            }
    return eigen_stats

def plot_layer_wise_mimo_ranks(all_data):
    """繪製每層的 MIMO ranks 對比"""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('各層 MIMO 秩演化對比', fontsize=16, fontweight='bold')
    
    colors = {'Rank 1': '#FF6B6B', 'Rank 4': '#4ECDC4', 'Rank 8': '#45B7D1'}
    
    # 選擇 6 個代表性的層（3 forward + 3 backward）
    representative_layers = [
        'layers.0.fwd', 'layers.2.fwd', 'layers.4.fwd',
        'layers.0.bwd', 'layers.2.bwd', 'layers.4.bwd'
    ]
    
    for idx, layer_name in enumerate(representative_layers):
        ax = axes[idx // 3, idx % 3]
        
        for run_name, data in all_data.items():
            if data and layer_name in data:
                values = data[layer_name]['values']
                epochs = range(1, len(values) + 1)
                ax.plot(epochs, values, label=run_name, color=colors[run_name], 
                       linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel('MIMO Rank', fontsize=10)
        ax.set_title(f'{layer_name}', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'layer_wise_mimo_ranks.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: layer_wise_mimo_ranks.png")
    plt.close()

def plot_a_log_snr(all_data):
    """繪製 A_log SNR 對比"""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('A_log 參數信噪比 (SNR) 演化', fontsize=16, fontweight='bold')
    
    colors = {'Rank 1': '#FF6B6B', 'Rank 4': '#4ECDC4', 'Rank 8': '#45B7D1'}
    
    # 選擇 6 個代表性的 A_log 層
    representative_layers = [
        'layers.0.fwd.A_log', 'layers.2.fwd.A_log', 'layers.4.fwd.A_log',
        'layers.0.bwd.A_log', 'layers.2.bwd.A_log', 'layers.4.bwd.A_log'
    ]
    
    for idx, layer_name in enumerate(representative_layers):
        ax = axes[idx // 3, idx % 3]
        
        for run_name, data in all_data.items():
            if data and layer_name in data:
                snr = data[layer_name]['snr']
                if len(snr) > 0:
                    epochs = range(1, len(snr) + 1)
                    ax.plot(epochs, snr, label=run_name, color=colors[run_name], 
                           linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel('SNR', fontsize=10)
        ax.set_title(f'{layer_name}', fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'a_log_snr_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: a_log_snr_comparison.png")
    plt.close()

def plot_delta_stats(all_data):
    """繪製 Delta 統計對比"""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Delta 參數統計 (變異係數 CV)', fontsize=16, fontweight='bold')
    
    colors = {'Rank 1': '#FF6B6B', 'Rank 4': '#4ECDC4', 'Rank 8': '#45B7D1'}
    
    representative_layers = [
        'layers.0.fwd', 'layers.2.fwd', 'layers.4.fwd',
        'layers.0.bwd', 'layers.2.bwd', 'layers.4.bwd'
    ]
    
    for idx, layer_name in enumerate(representative_layers):
        ax = axes[idx // 3, idx % 3]
        
        for run_name, data in all_data.items():
            if data and layer_name in data:
                cv = data[layer_name]['cv']
                if len(cv) > 0:
                    epochs = range(1, len(cv) + 1)
                    ax.plot(epochs, cv, label=run_name, color=colors[run_name], 
                           linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel('Coefficient of Variation', fontsize=10)
        ax.set_title(f'{layer_name}', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'delta_cv_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: delta_cv_comparison.png")
    plt.close()

def plot_eigen_a_evolution(all_data):
    """繪製 Eigen A 演化"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('SSM 矩陣特徵值演化', fontsize=16, fontweight='bold')
    
    colors = {'Rank 1': '#FF6B6B', 'Rank 4': '#4ECDC4', 'Rank 8': '#45B7D1'}
    
    # 提取統計數據
    for run_name, data in all_data.items():
        if data:
            epochs = sorted(data.keys())
            max_vals = [data[e]['max'] for e in epochs]
            min_vals = [data[e]['min'] for e in epochs]
            mean_vals = [data[e]['mean'] for e in epochs]
            std_vals = [data[e]['std'] for e in epochs]
            
            epoch_nums = [e + 1 for e in epochs]
            
            # Max eigenvalue
            axes[0, 0].plot(epoch_nums, max_vals, label=run_name, 
                           color=colors[run_name], linewidth=2, alpha=0.8)
            
            # Min eigenvalue
            axes[0, 1].plot(epoch_nums, min_vals, label=run_name, 
                           color=colors[run_name], linewidth=2, alpha=0.8)
            
            # Mean eigenvalue
            axes[1, 0].plot(epoch_nums, mean_vals, label=run_name, 
                           color=colors[run_name], linewidth=2, alpha=0.8)
            
            # Std eigenvalue
            axes[1, 1].plot(epoch_nums, std_vals, label=run_name, 
                           color=colors[run_name], linewidth=2, alpha=0.8)
    
    axes[0, 0].set_title('最大特徵值', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Max Eigenvalue', fontsize=12)
    axes[0, 0].legend(loc='best')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    axes[0, 1].set_title('最小特徵值', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Min Eigenvalue', fontsize=12)
    axes[0, 1].legend(loc='best')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')
    
    axes[1, 0].set_title('平均特徵值', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Mean Eigenvalue', fontsize=12)
    axes[1, 0].legend(loc='best')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')
    
    axes[1, 1].set_title('特徵值標準差', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Std Eigenvalue', fontsize=12)
    axes[1, 1].legend(loc='best')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'eigen_a_evolution.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: eigen_a_evolution.png")
    plt.close()

def plot_mimo_rank_distribution(all_data):
    """繪製 MIMO rank 分布對比"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('MIMO 秩分布對比 (所有層)', fontsize=16, fontweight='bold')
    
    colors = {'Rank 1': '#FF6B6B', 'Rank 4': '#4ECDC4', 'Rank 8': '#45B7D1'}
    
    for idx, (run_name, data) in enumerate(all_data.items()):
        ax = axes[idx]
        
        if data:
            # 收集所有層的最終 MIMO rank
            final_ranks = [layer_data['final'] for layer_data in data.values()]
            mean_ranks = [layer_data['mean'] for layer_data in data.values()]
            layer_names = list(data.keys())
            
            # 創建箱型圖
            bp = ax.boxplot([final_ranks, mean_ranks], 
                           labels=['Final Ranks', 'Mean Ranks'],
                           patch_artist=True)
            
            for patch in bp['boxes']:
                patch.set_facecolor(colors[run_name])
                patch.set_alpha(0.7)
            
            ax.set_title(f'{run_name}', fontsize=14, fontweight='bold')
            ax.set_ylabel('MIMO Rank', fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
            
            # 添加統計信息
            final_mean = np.mean(final_ranks)
            mean_mean = np.mean(mean_ranks)
            ax.text(0.5, 0.95, f'Final Avg: {final_mean:.2f}\nMean Avg: {mean_mean:.2f}',
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'mimo_rank_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: mimo_rank_distribution.png")
    plt.close()

def generate_enhanced_stats(all_mimo, all_a_log, all_delta, all_eigen):
    """生成增強統計數據"""
    enhanced_stats = {}
    
    for run_name in RUNS.keys():
        stats = {
            'mimo_analysis': {},
            'a_log_analysis': {},
            'delta_analysis': {},
            'eigen_analysis': {}
        }
        
        # MIMO 分析
        if run_name in all_mimo and all_mimo[run_name]:
            mimo_data = all_mimo[run_name]
            final_ranks = [layer['final'] for layer in mimo_data.values()]
            mean_ranks = [layer['mean'] for layer in mimo_data.values()]
            
            stats['mimo_analysis'] = {
                'num_layers': len(mimo_data),
                'final_ranks': {
                    'mean': float(np.mean(final_ranks)),
                    'std': float(np.std(final_ranks)),
                    'min': float(np.min(final_ranks)),
                    'max': float(np.max(final_ranks))
                },
                'mean_ranks': {
                    'mean': float(np.mean(mean_ranks)),
                    'std': float(np.std(mean_ranks)),
                    'min': float(np.min(mean_ranks)),
                    'max': float(np.max(mean_ranks))
                }
            }
        
        # A_log 分析
        if run_name in all_a_log and all_a_log[run_name]:
            a_log_data = all_a_log[run_name]
            final_snrs = []
            for layer_data in a_log_data.values():
                if len(layer_data['snr']) > 0:
                    final_snrs.append(layer_data['snr'][-1])
            
            if final_snrs:
                stats['a_log_analysis'] = {
                    'num_layers': len(a_log_data),
                    'final_snr': {
                        'mean': float(np.mean(final_snrs)),
                        'std': float(np.std(final_snrs)),
                        'min': float(np.min(final_snrs)),
                        'max': float(np.max(final_snrs))
                    }
                }
        
        # Eigen 分析
        if run_name in all_eigen and all_eigen[run_name]:
            eigen_data = all_eigen[run_name]
            if len(eigen_data) > 0:
                last_epoch = max(eigen_data.keys())
                stats['eigen_analysis'] = {
                    'final_epoch': last_epoch,
                    'final_stats': eigen_data[last_epoch]
                }
        
        enhanced_stats[run_name] = stats
    
    # 保存為 JSON
    with open(OUTPUT_DIR / 'enhanced_statistics.json', 'w', encoding='utf-8') as f:
        json.dump(enhanced_stats, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved: enhanced_statistics.json")
    return enhanced_stats

def main():
    print("=" * 80)
    print("開始增強版深度分析...")
    print("=" * 80)
    
    all_mimo = {}
    all_a_log = {}
    all_delta = {}
    all_eigen = {}
    
    # 載入所有數據
    for run_name, run_dir_name in RUNS.items():
        run_dir = BASE_DIR / run_dir_name
        print(f"\n處理 {run_name}...")
        
        data = load_diagnostics(run_dir)
        if data:
            all_mimo[run_name] = extract_layer_wise_mimo_ranks(data)
            all_a_log[run_name] = extract_a_log_stats(data)
            all_delta[run_name] = extract_delta_stats(data)
            all_eigen[run_name] = extract_eigen_a(data)
            
            print(f"  ✓ MIMO layers: {len(all_mimo[run_name]) if all_mimo[run_name] else 0}")
            print(f"  ✓ A_log layers: {len(all_a_log[run_name]) if all_a_log[run_name] else 0}")
            print(f"  ✓ Delta layers: {len(all_delta[run_name]) if all_delta[run_name] else 0}")
            print(f"  ✓ Eigen epochs: {len(all_eigen[run_name]) if all_eigen[run_name] else 0}")
    
    print("\n" + "=" * 80)
    print("生成增強版圖表...")
    print("=" * 80)
    
    # 生成所有圖表
    plot_layer_wise_mimo_ranks(all_mimo)
    plot_a_log_snr(all_a_log)
    plot_delta_stats(all_delta)
    plot_eigen_a_evolution(all_eigen)
    plot_mimo_rank_distribution(all_mimo)
    
    # 生成增強統計
    print("\n" + "=" * 80)
    print("生成增強統計數據...")
    print("=" * 80)
    enhanced_stats = generate_enhanced_stats(all_mimo, all_a_log, all_delta, all_eigen)
    
    print("\n" + "=" * 80)
    print("增強分析完成！")
    print("=" * 80)
    print(f"\n所有結果已保存至: {OUTPUT_DIR}")
    print("\n新生成的檔案:")
    print("  - layer_wise_mimo_ranks.png")
    print("  - a_log_snr_comparison.png")
    print("  - delta_cv_comparison.png")
    print("  - eigen_a_evolution.png")
    print("  - mimo_rank_distribution.png")
    print("  - enhanced_statistics.json")

if __name__ == '__main__':
    main()
