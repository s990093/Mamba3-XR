#!/usr/bin/env python3
"""
完整分析訓練目錄中的所有檔案
Deep analysis of all files in training directories
"""

import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from pathlib import Path
from PIL import Image
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

def analyze_directory_structure(run_dir, run_name):
    """分析目錄結構"""
    print(f"\n{'='*80}")
    print(f"分析 {run_name}: {run_dir.name}")
    print(f"{'='*80}")
    
    files = list(run_dir.iterdir())
    
    # 分類檔案
    categories = {
        'diagnostics': [],
        'erf_plots': [],
        'dist_plots': [],
        'models': [],
        'other': []
    }
    
    for f in files:
        if f.name == 'diagnostics_history.pt':
            categories['diagnostics'].append(f)
        elif f.name.startswith('erf_epoch_'):
            categories['erf_plots'].append(f)
        elif f.name.startswith('dist_'):
            categories['dist_plots'].append(f)
        elif f.name.endswith('.pth'):
            categories['models'].append(f)
        else:
            categories['other'].append(f)
    
    print(f"\n檔案分類:")
    print(f"  診斷數據: {len(categories['diagnostics'])} 個")
    print(f"  ERF 圖表: {len(categories['erf_plots'])} 個")
    print(f"  分布圖表: {len(categories['dist_plots'])} 個")
    print(f"  模型檔案: {len(categories['models'])} 個")
    print(f"  其他檔案: {len(categories['other'])} 個")
    
    return categories

def analyze_erf_evolution(run_dir, run_name):
    """分析 ERF 演化"""
    erf_files = sorted(run_dir.glob('erf_epoch_*.png'))
    
    if not erf_files:
        print(f"  ⚠ 未找到 ERF 圖表")
        return None
    
    print(f"\n  找到 {len(erf_files)} 個 ERF 圖表")
    
    # 分析關鍵 epoch 的 ERF
    key_epochs = [0, 9, 19, 49, 99]  # epoch 0, 10, 20, 50, 100
    erf_info = {}
    
    for epoch in key_epochs:
        erf_file = run_dir / f'erf_epoch_{epoch:03d}.png'
        if erf_file.exists():
            img = Image.open(erf_file)
            erf_info[epoch] = {
                'file': erf_file.name,
                'size': img.size,
                'mode': img.mode
            }
    
    return {
        'total_erfs': len(erf_files),
        'key_epochs': erf_info,
        'first_erf': erf_files[0].name,
        'last_erf': erf_files[-1].name
    }

def analyze_dist_plots(run_dir, run_name):
    """分析分布圖表"""
    dist_plots = {
        'gradients': run_dir / 'dist_gradients.png',
        'mamba_internals': run_dir / 'dist_mamba_internals.png',
        'metrics': run_dir / 'dist_metrics.png',
        'state_health': run_dir / 'dist_state_health.png'
    }
    
    plot_info = {}
    for name, path in dist_plots.items():
        if path.exists():
            img = Image.open(path)
            plot_info[name] = {
                'exists': True,
                'size': img.size,
                'file_size_kb': path.stat().st_size / 1024
            }
        else:
            plot_info[name] = {'exists': False}
    
    return plot_info

def analyze_model_checkpoints(run_dir, run_name):
    """分析模型檢查點"""
    best_model = run_dir / 'best_model.pth'
    last_model = run_dir / 'last_model.pth'
    
    checkpoint_info = {}
    
    for name, path in [('best', best_model), ('last', last_model)]:
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            
            # 載入檢查點以獲取更多信息
            try:
                checkpoint = torch.load(path, map_location='cpu', weights_only=False)
                
                info = {
                    'size_mb': size_mb,
                    'keys': list(checkpoint.keys()) if isinstance(checkpoint, dict) else ['model_state_dict'],
                }
                
                # 如果有 epoch 信息
                if isinstance(checkpoint, dict) and 'epoch' in checkpoint:
                    info['epoch'] = checkpoint['epoch']
                
                # 如果有性能指標
                if isinstance(checkpoint, dict):
                    for key in ['val_acc', 'val_loss', 'best_acc']:
                        if key in checkpoint:
                            info[key] = float(checkpoint[key])
                
                checkpoint_info[name] = info
            except Exception as e:
                checkpoint_info[name] = {
                    'size_mb': size_mb,
                    'error': str(e)
                }
    
    return checkpoint_info

def create_erf_comparison(all_runs):
    """創建 ERF 對比圖"""
    fig, axes = plt.subplots(3, 5, figsize=(25, 15))
    fig.suptitle('ERF 演化對比 (Epoch 0, 10, 20, 50, 100)', fontsize=16, fontweight='bold')
    
    key_epochs = [0, 9, 19, 49, 99]
    run_names = ['Rank 1', 'Rank 4', 'Rank 8']
    
    for row_idx, (run_name, run_dir_name) in enumerate(RUNS.items()):
        run_dir = BASE_DIR / run_dir_name
        
        for col_idx, epoch in enumerate(key_epochs):
            ax = axes[row_idx, col_idx]
            erf_file = run_dir / f'erf_epoch_{epoch:03d}.png'
            
            if erf_file.exists():
                img = Image.open(erf_file)
                ax.imshow(img)
                ax.axis('off')
                
                if row_idx == 0:
                    ax.set_title(f'Epoch {epoch+1}', fontsize=12, fontweight='bold')
                if col_idx == 0:
                    ax.set_ylabel(run_name, fontsize=12, fontweight='bold', rotation=0, 
                                 ha='right', va='center')
            else:
                ax.text(0.5, 0.5, 'Not Found', ha='center', va='center')
                ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'erf_evolution_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: erf_evolution_comparison.png")
    plt.close()

def create_dist_plots_comparison(all_runs):
    """創建分布圖對比"""
    dist_types = ['gradients', 'mamba_internals', 'metrics', 'state_health']
    
    for dist_type in dist_types:
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))
        fig.suptitle(f'{dist_type.replace("_", " ").title()} 對比', 
                     fontsize=16, fontweight='bold')
        
        for idx, (run_name, run_dir_name) in enumerate(RUNS.items()):
            run_dir = BASE_DIR / run_dir_name
            dist_file = run_dir / f'dist_{dist_type}.png'
            
            ax = axes[idx]
            if dist_file.exists():
                img = Image.open(dist_file)
                ax.imshow(img)
                ax.set_title(run_name, fontsize=14, fontweight='bold')
                ax.axis('off')
            else:
                ax.text(0.5, 0.5, 'Not Found', ha='center', va='center')
                ax.axis('off')
        
        plt.tight_layout()
        output_file = OUTPUT_DIR / f'dist_{dist_type}_comparison.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: dist_{dist_type}_comparison.png")
        plt.close()

def generate_comprehensive_report(all_analysis):
    """生成完整分析報告"""
    report = {
        'analysis_date': '2025-12-29',
        'runs_analyzed': len(all_analysis),
        'runs': {}
    }
    
    for run_name, analysis in all_analysis.items():
        report['runs'][run_name] = {
            'directory_structure': {
                'total_files': sum(len(files) for files in analysis['structure'].values()),
                'erf_plots': len(analysis['structure']['erf_plots']),
                'dist_plots': len(analysis['structure']['dist_plots']),
                'model_files': len(analysis['structure']['models'])
            },
            'erf_analysis': analysis['erf'],
            'dist_plots': analysis['dist_plots'],
            'checkpoints': analysis['checkpoints']
        }
    
    # 保存為 JSON
    with open(OUTPUT_DIR / 'comprehensive_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"✓ Saved: comprehensive_analysis.json")
    return report

def main():
    print("=" * 80)
    print("開始完整分析所有訓練檔案...")
    print("=" * 80)
    
    all_analysis = {}
    
    # 分析每個訓練目錄
    for run_name, run_dir_name in RUNS.items():
        run_dir = BASE_DIR / run_dir_name
        
        analysis = {
            'structure': analyze_directory_structure(run_dir, run_name),
            'erf': analyze_erf_evolution(run_dir, run_name),
            'dist_plots': analyze_dist_plots(run_dir, run_name),
            'checkpoints': analyze_model_checkpoints(run_dir, run_name)
        }
        
        all_analysis[run_name] = analysis
        
        # 打印摘要
        print(f"\n  ERF 分析: {analysis['erf']['total_erfs'] if analysis['erf'] else 0} 個圖表")
        print(f"  分布圖: {sum(1 for p in analysis['dist_plots'].values() if p.get('exists', False))} / 4 個")
        print(f"  模型檢查點: {len(analysis['checkpoints'])} 個")
    
    print("\n" + "=" * 80)
    print("生成對比圖表...")
    print("=" * 80)
    
    # 創建 ERF 對比圖
    create_erf_comparison(all_analysis)
    
    # 創建分布圖對比
    create_dist_plots_comparison(all_analysis)
    
    # 生成完整報告
    print("\n" + "=" * 80)
    print("生成完整分析報告...")
    print("=" * 80)
    report = generate_comprehensive_report(all_analysis)
    
    print("\n" + "=" * 80)
    print("分析完成！")
    print("=" * 80)
    print(f"\n所有結果已保存至: {OUTPUT_DIR}")
    print("\n新生成的檔案:")
    print("  - erf_evolution_comparison.png")
    print("  - dist_gradients_comparison.png")
    print("  - dist_mamba_internals_comparison.png")
    print("  - dist_metrics_comparison.png")
    print("  - dist_state_health_comparison.png")
    print("  - comprehensive_analysis.json")

if __name__ == '__main__':
    main()
