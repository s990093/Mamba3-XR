"""
Mamba-3 訓練數據可視化腳本

使用方法：
    python plot_training.py
    
或在 Jupyter/Kaggle Notebook 中：
    %run plot_training.py
    
支持的數據格式：
    - training_history.json (完整數據)
    - training_epochs.csv + training_steps.csv (CSV 格式)
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 設置中文字體（可選）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_training_data_from_json(history_file='training_history.json'):
    """從 JSON 載入訓練歷史數據"""
    with open(history_file, 'r') as f:
        return json.load(f)

def load_training_data_from_csv(epochs_file='training_epochs.csv', 
                                steps_file='training_steps.csv'):
    """從 CSV 載入訓練歷史數據"""
    print(f"📂 從 CSV 載入數據...")
    
    # 讀取 epoch 數據
    epochs_df = pd.read_csv(epochs_file)
    
    # 轉換為與 JSON 相同的格式
    history = {
        'config': {
            'mimo_rank': 'unknown',
            'd_model': 'unknown',
            'n_layers': 'unknown',
            'vocab_size': 'unknown',
            'epochs': len(epochs_df),
            'lr': 'unknown',
            'batch_size': 'unknown',
            'num_gpus': 'unknown',
            'mixed_precision': 'unknown',
        },
        'epochs': [],
        'steps': [],
    }
    
    # 轉換 epoch 數據
    for _, row in epochs_df.iterrows():
        history['epochs'].append({
            'epoch': int(row['epoch']),
            'avg_train_loss': float(row['avg_train_loss']),
            'val_loss': float(row['val_loss']),
            'time': float(row['time']),
            'samples_per_sec': float(row['samples_per_sec']),
        })
    
    # 讀取 step 數據（如果存在）
    if Path(steps_file).exists():
        steps_df = pd.read_csv(steps_file)
        for _, row in steps_df.iterrows():
            history['steps'].append({
                'step': int(row['step']),
                'epoch': int(row['epoch']),
                'train_loss': float(row['train_loss']),
                'val_loss': float(row['val_loss']),
                'lr': float(row['lr']),
                'samples': int(row['samples']),
            })
    
    return history

def load_training_data():
    """智能載入訓練數據（優先 JSON，其次 CSV）"""
    if Path('training_history.json').exists():
        print("📂 從 JSON 載入數據...")
        return load_training_data_from_json()
    elif Path('training_epochs.csv').exists():
        return load_training_data_from_csv()
    else:
        raise FileNotFoundError(
            "找不到訓練數據文件！\n"
            "需要以下文件之一：\n"
            "  - training_history.json\n"
            "  - training_epochs.csv"
        )

def plot_loss_curves(history, save_path='training_loss.png'):
    """繪製訓練和驗證損失曲線"""
    epochs_data = history['epochs']
    
    epochs = [e['epoch'] for e in epochs_data]
    train_losses = [e['avg_train_loss'] for e in epochs_data]
    val_losses = [e['val_loss'] for e in epochs_data]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2, marker='o')
    plt.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2, marker='s')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # 標註最佳驗證損失
    best_epoch = np.argmin(val_losses)
    best_val_loss = val_losses[best_epoch]
    plt.axvline(x=epochs[best_epoch], color='g', linestyle='--', alpha=0.5)
    plt.text(epochs[best_epoch], best_val_loss, 
             f'Best: {best_val_loss:.4f}', 
             fontsize=10, ha='right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 保存: {save_path}")
    plt.close()

def plot_learning_rate(history, save_path='learning_rate.png'):
    """繪製學習率變化曲線"""
    if not history['steps']:
        print("⚠️  沒有 step 級別數據，跳過學習率圖表")
        return
    
    steps = [s['step'] for s in history['steps']]
    lrs = [s['lr'] for s in history['steps']]
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, lrs, 'g-', linewidth=2)
    
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 保存: {save_path}")
    plt.close()

def plot_throughput(history, save_path='throughput.png'):
    """繪製訓練吞吐量"""
    epochs_data = history['epochs']
    
    epochs = [e['epoch'] for e in epochs_data]
    throughput = [e['samples_per_sec'] for e in epochs_data]
    
    plt.figure(figsize=(10, 6))
    plt.bar(epochs, throughput, color='steelblue', alpha=0.7)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Samples/Second', fontsize=12)
    plt.title('Training Throughput', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    # 平均吞吐量線
    avg_throughput = np.mean(throughput)
    plt.axhline(y=avg_throughput, color='r', linestyle='--', 
                label=f'Average: {avg_throughput:.0f} samples/s')
    plt.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 保存: {save_path}")
    plt.close()

def plot_step_losses(history, save_path='step_losses.png'):
    """繪製每個評估步驟的損失"""
    if not history['steps']:
        print("⚠️  沒有 step 級別數據，跳過步驟損失圖表")
        return
    
    steps = [s['step'] for s in history['steps']]
    train_losses = [s['train_loss'] for s in history['steps']]
    val_losses = [s['val_loss'] for s in history['steps']]
    
    plt.figure(figsize=(12, 6))
    plt.plot(steps, train_losses, 'b-', label='Train Loss', linewidth=1.5, alpha=0.7)
    plt.plot(steps, val_losses, 'r-', label='Val Loss', linewidth=2)
    
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Loss per Evaluation Step', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 保存: {save_path}")
    plt.close()

def plot_all_in_one(history, save_path='training_summary.png'):
    """綜合圖表：4 個子圖"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    epochs_data = history['epochs']
    epochs = [e['epoch'] for e in epochs_data]
    train_losses = [e['avg_train_loss'] for e in epochs_data]
    val_losses = [e['val_loss'] for e in epochs_data]
    throughput = [e['samples_per_sec'] for e in epochs_data]
    
    # 1. Loss curves
    ax1 = axes[0, 0]
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2, marker='o')
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2, marker='s')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Training Progress', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Throughput
    ax2 = axes[0, 1]
    ax2.bar(epochs, throughput, color='steelblue', alpha=0.7)
    avg_throughput = np.mean(throughput)
    ax2.axhline(y=avg_throughput, color='r', linestyle='--', 
                label=f'Avg: {avg_throughput:.0f} samples/s')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Samples/Second', fontsize=11)
    ax2.set_title('Training Throughput', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Learning rate (if available)
    ax3 = axes[1, 0]
    if history['steps']:
        steps = [s['step'] for s in history['steps']]
        lrs = [s['lr'] for s in history['steps']]
        ax3.plot(steps, lrs, 'g-', linewidth=2)
        ax3.set_xlabel('Training Step', fontsize=11)
        ax3.set_ylabel('Learning Rate', fontsize=11)
        ax3.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No step data available', 
                ha='center', va='center', fontsize=12)
        ax3.set_xticks([])
        ax3.set_yticks([])
    
    # 4. Training time per epoch
    ax4 = axes[1, 1]
    times = [e['time'] for e in epochs_data]
    ax4.plot(epochs, times, 'purple', linewidth=2, marker='D')
    ax4.set_xlabel('Epoch', fontsize=11)
    ax4.set_ylabel('Time (seconds)', fontsize=11)
    ax4.set_title('Time per Epoch', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 添加配置信息
    config = history['config']
    config_text = (
        f"MIMO Rank: {config['mimo_rank']} | "
        f"Batch Size: {config['batch_size']} | "
        f"GPUs: {config['num_gpus']} | "
        f"Mixed Precision: {config['mixed_precision']}"
    )
    fig.suptitle(f'Mamba-3 Training Summary\n{config_text}', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 保存: {save_path}")
    plt.close()

def print_summary(history):
    """打印訓練摘要"""
    config = history['config']
    epochs_data = history['epochs']
    
    print("\n" + "=" * 80)
    print("📊 訓練摘要")
    print("=" * 80)
    
    print("\n配置:")
    print(f"  MIMO Rank: {config['mimo_rank']}")
    print(f"  Vocabulary Size: {config['vocab_size']}")
    print(f"  Batch Size: {config['batch_size']}")
    print(f"  GPUs: {config['num_gpus']}")
    print(f"  Mixed Precision: {config['mixed_precision']}")
    print(f"  Learning Rate: {config['lr']}")
    
    print("\n結果:")
    final_epoch = epochs_data[-1]
    best_val_loss = min(e['val_loss'] for e in epochs_data)
    best_epoch = [e['epoch'] for e in epochs_data if e['val_loss'] == best_val_loss][0]
    
    print(f"  總 Epochs: {len(epochs_data)}")
    print(f"  最佳驗證損失: {best_val_loss:.4f} (Epoch {best_epoch})")
    print(f"  最終訓練損失: {final_epoch['avg_train_loss']:.4f}")
    print(f"  最終驗證損失: {final_epoch['val_loss']:.4f}")
    print(f"  平均吞吐量: {np.mean([e['samples_per_sec'] for e in epochs_data]):.0f} samples/sec")
    print(f"  總訓練時間: {sum(e['time'] for e in epochs_data):.1f}s")
    
    print("=" * 80 + "\n")

def main():
    """主函數"""
    print("\n" + "=" * 80)
    print("🎨 Mamba-3 訓練數據可視化")
    print("=" * 80 + "\n")
    
    # 載入數據
    try:
        history = load_training_data()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    
    # 打印摘要
    print_summary(history)
    
    # 生成圖表
    print("🎨 生成圖表...")
    plot_loss_curves(history)
    plot_learning_rate(history)
    plot_throughput(history)
    plot_step_losses(history)
    plot_all_in_one(history)
    
    print("\n" + "=" * 80)
    print("✅ 完成！所有圖表已保存")
    print("=" * 80)
    print("\n生成的文件:")
    print("  - training_loss.png (損失曲線)")
    print("  - learning_rate.png (學習率)")
    print("  - throughput.png (吞吐量)")
    print("  - step_losses.png (步驟損失)")
    print("  - training_summary.png (綜合圖表)")
    print("\n")

if __name__ == "__main__":
    main()
