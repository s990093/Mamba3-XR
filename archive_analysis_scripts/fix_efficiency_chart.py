#!/usr/bin/env python3
"""
修正 MIMO 飽和效率分析圖表
Fix MIMO saturation efficiency chart with correct values
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from pathlib import Path
import shutil

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = Path('/Users/hungwei/.gemini/antigravity/brain/bd37de13-51b9-479f-bbe1-066937a600cf')

# 正確的數據
ranks = ['Rank 1', 'Rank 4', 'Rank 8']
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

# 1. 參數利用效率圖（修正版）
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('MIMO 秩飽和分析：參數效率與理論上限', fontsize=16, fontweight='bold')

# 左圖：參數量 vs MIMO 秩
params = [18.07, 39.30, 67.60]  # MB
mimo_ranks = [18.66, 27.95, 28.06]

ax1.scatter(params, mimo_ranks, s=300, c=colors, alpha=0.7, edgecolors='black', linewidth=2)
for i, (p, r, name) in enumerate(zip(params, mimo_ranks, ranks)):
    ax1.annotate(f'{name}\n秩: {r:.1f}', 
                xy=(p, r), 
                xytext=(10, 10), 
                textcoords='offset points',
                fontsize=11,
                bbox=dict(boxstyle='round,pad=0.5', facecolor=colors[i], alpha=0.3))

# 添加飽和線
ax1.axhline(y=28, color='red', linestyle='--', linewidth=2, label='飽和點 (~28)', alpha=0.7)
ax1.axhline(y=64, color='green', linestyle='--', linewidth=2, label='理論上限 (64)', alpha=0.7)

# 添加趨勢線（指數飽和曲線）
x_fit = np.linspace(0, 80, 100)
y_fit = 28 * (1 - np.exp(-0.08 * x_fit))
ax1.plot(x_fit, y_fit, 'k--', alpha=0.5, linewidth=1.5, label='飽和曲線擬合')

ax1.set_xlabel('參數量 (MB)', fontsize=12, fontweight='bold')
ax1.set_ylabel('MIMO 有效秩', fontsize=12, fontweight='bold')
ax1.set_title('參數量 vs MIMO 秩（飽和現象）', fontsize=14, fontweight='bold')
ax1.legend(loc='lower right', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 75)
ax1.set_ylim(0, 70)

# 右圖：參數利用效率（秩/參數比）- 修正版
# 正確計算：秩 / 參數量
efficiency = [
    18.66 / 18.07,  # Rank 1: 1.03
    27.95 / 39.30,  # Rank 4: 0.71
    28.06 / 67.60   # Rank 8: 0.41
]

bars = ax2.bar(ranks, efficiency, color=colors, alpha=0.8, edgecolor='black', linewidth=2)

# 添加數值標籤
for bar, eff in zip(bars, efficiency):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{eff:.2f}\n秩/MB',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax2.set_ylabel('參數利用效率 (秩/MB)', fontsize=12, fontweight='bold')
ax2.set_title('參數利用效率對比', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim(0, 1.2)

# 添加效率評級
ax2.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, linewidth=1)
ax2.text(2.5, 0.82, '高效率', fontsize=10, color='green', fontweight='bold')
ax2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, linewidth=1)
ax2.text(2.5, 0.52, '中等', fontsize=10, color='orange', fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'mimo_saturation_efficiency.png', dpi=300, bbox_inches='tight')
print("✓ Saved: mimo_saturation_efficiency.png (CORRECTED)")
plt.close()

# 複製到 docs
dst = Path('/Users/hungwei/Desktop/Proj/Mamba-Orin-Nano-Custom-S6-CUDA/mamba3/docs/mimo_saturation_efficiency.png')
shutil.copy(OUTPUT_DIR / 'mimo_saturation_efficiency.png', dst)
print(f"✓ Copied to docs/")

print("\n✅ 已修正參數效率圖表！")
print(f"   Rank 1: {efficiency[0]:.2f} 秩/MB")
print(f"   Rank 4: {efficiency[1]:.2f} 秩/MB")
print(f"   Rank 8: {efficiency[2]:.2f} 秩/MB")
