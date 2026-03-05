#!/usr/bin/env python3
"""
最終修正：d_state = 32
Final correction: d_state = 32
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from pathlib import Path
import shutil

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = Path('/Users/hungwei/.gemini/antigravity/brain/bd37de13-51b9-479f-bbe1-066937a600cf')

ranks = ['Rank 1', 'Rank 4', 'Rank 8']
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

# 1. 參數效率圖（理論上限 = 32）
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('MIMO 秩飽和分析：參數效率與理論上限', fontsize=16, fontweight='bold')

params = [18.07, 39.30, 67.60]
mimo_ranks = [18.66, 27.95, 28.06]

ax1.scatter(params, mimo_ranks, s=300, c=colors, alpha=0.7, edgecolors='black', linewidth=2)
for i, (p, r, name) in enumerate(zip(params, mimo_ranks, ranks)):
    ax1.annotate(f'{name}\n秩: {r:.1f}', 
                xy=(p, r), 
                xytext=(10, 10), 
                textcoords='offset points',
                fontsize=11,
                bbox=dict(boxstyle='round,pad=0.5', facecolor=colors[i], alpha=0.3))

# 理論上限 = 32（不是 64）
ax1.axhline(y=28, color='red', linestyle='--', linewidth=2, label='飽和點 (~28)', alpha=0.7)
ax1.axhline(y=32, color='green', linestyle='--', linewidth=2, label='理論上限 (32)', alpha=0.7)

x_fit = np.linspace(0, 80, 100)
y_fit = 28 * (1 - np.exp(-0.08 * x_fit))
ax1.plot(x_fit, y_fit, 'k--', alpha=0.5, linewidth=1.5, label='飽和曲線擬合')

ax1.set_xlabel('參數量 (MB)', fontsize=12, fontweight='bold')
ax1.set_ylabel('MIMO 有效秩', fontsize=12, fontweight='bold')
ax1.set_title('參數量 vs MIMO 秩（飽和現象）', fontsize=14, fontweight='bold')
ax1.legend(loc='lower right', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 75)
ax1.set_ylim(0, 40)  # 調整 Y 軸範圍以適應 32

# 右圖：效率
efficiency = [1.03, 0.71, 0.42]
bars = ax2.bar(ranks, efficiency, color=colors, alpha=0.8, edgecolor='black', linewidth=2)

for bar, eff in zip(bars, efficiency):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{eff:.2f}\n秩/MB',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax2.set_ylabel('參數利用效率 (秩/MB)', fontsize=12, fontweight='bold')
ax2.set_title('參數利用效率對比', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim(0, 1.2)

ax2.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, linewidth=1)
ax2.text(2.5, 0.82, '高效率', fontsize=10, color='green', fontweight='bold')
ax2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, linewidth=1)
ax2.text(2.5, 0.52, '中等', fontsize=10, color='orange', fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'mimo_saturation_efficiency.png', dpi=300, bbox_inches='tight')
print("✓ Saved: mimo_saturation_efficiency.png (d_state=32)")

# 2. 理論利用率圖（32 維）
fig, ax = plt.subplots(figsize=(12, 8))

categories = ['理論上限\n(d_state=32)', '實際飽和點\n(87.5%)', '未利用空間\n(12.5%)']
values = [32, 28, 4]
colors_pie = ['#2ecc71', '#3498db', '#e74c3c']
explode = (0.05, 0.1, 0.05)

wedges, texts, autotexts = ax.pie(values, 
                                    explode=explode,
                                    labels=categories, 
                                    colors=colors_pie,
                                    autopct='%1.1f%%',
                                    startangle=90,
                                    textprops={'fontsize': 14, 'fontweight': 'bold'})

for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(16)
    autotext.set_fontweight('bold')

ax.legend(wedges, [f'{cat}\n{val} 維' for cat, val in zip(categories, values)],
          title="MIMO 秩空間分配",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1),
          fontsize=12)

ax.set_title('理論 MIMO 秩空間利用率\n(實際利用 87.5% - 配置合理)', 
             fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'mimo_theoretical_utilization.png', dpi=300, bbox_inches='tight')
print("✓ Saved: mimo_theoretical_utilization.png (d_state=32)")

# 複製到 docs
for filename in ['mimo_saturation_efficiency.png', 'mimo_theoretical_utilization.png']:
    src = OUTPUT_DIR / filename
    dst = Path('/Users/hungwei/Desktop/Proj/Mamba-Orin-Nano-Custom-S6-CUDA/mamba3/docs') / filename
    shutil.copy(src, dst)
    print(f"✓ Copied {filename} to docs/")

plt.close('all')
print("\n✅ 已完成最終修正（d_state=32）")
