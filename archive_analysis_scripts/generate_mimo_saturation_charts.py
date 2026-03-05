#!/usr/bin/env python3
"""
生成 MIMO 秩飽和分析的視覺化圖表
Generate visualizations for MIMO rank saturation analysis
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from pathlib import Path

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = Path('/Users/hungwei/.gemini/antigravity/brain/bd37de13-51b9-479f-bbe1-066937a600cf')

# 數據
ranks = ['Rank 1', 'Rank 4', 'Rank 8']
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

# 1. 參數利用效率圖
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

# 右圖：參數利用效率（秩/參數比）
efficiency = [0.71, 0.71, 0.41]  # 秩/MB
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
ax2.set_ylim(0, 0.8)

# 添加效率評級
ax2.axhline(y=0.6, color='green', linestyle='--', alpha=0.5, linewidth=1)
ax2.text(2.5, 0.62, '高效率', fontsize=10, color='green', fontweight='bold')
ax2.axhline(y=0.4, color='orange', linestyle='--', alpha=0.5, linewidth=1)
ax2.text(2.5, 0.42, '中等', fontsize=10, color='orange', fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'mimo_saturation_efficiency.png', dpi=300, bbox_inches='tight')
print("✓ Saved: mimo_saturation_efficiency.png")
plt.close()

# 2. 理論上限 vs 實際利用圖
fig, ax = plt.subplots(figsize=(12, 8))

categories = ['理論上限\n(d_state)', '實際飽和點', '未利用空間']
values = [64, 28, 36]
colors_pie = ['#2ecc71', '#3498db', '#e74c3c']
explode = (0.05, 0.1, 0.05)

wedges, texts, autotexts = ax.pie(values, 
                                    explode=explode,
                                    labels=categories, 
                                    colors=colors_pie,
                                    autopct='%1.1f%%',
                                    startangle=90,
                                    textprops={'fontsize': 14, 'fontweight': 'bold'})

# 美化百分比文字
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(16)
    autotext.set_fontweight('bold')

# 添加圖例
ax.legend(wedges, [f'{cat}\n{val} 維' for cat, val in zip(categories, values)],
          title="MIMO 秩空間分配",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1),
          fontsize=12)

ax.set_title('理論 MIMO 秩空間利用率\n(實際利用 43.75%)', 
             fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'mimo_theoretical_utilization.png', dpi=300, bbox_inches='tight')
print("✓ Saved: mimo_theoretical_utilization.png")
plt.close()

# 3. 邊際效益遞減圖
fig, ax = plt.subplots(figsize=(12, 7))

transitions = ['Rank 1\n→\nRank 4', 'Rank 4\n→\nRank 8']
rank_increase = [49.8, 0.4]  # %
perf_increase = [1.90, 0.48]  # %

x = np.arange(len(transitions))
width = 0.35

bars1 = ax.bar(x - width/2, rank_increase, width, label='MIMO 秩提升 (%)', 
               color='#3498db', alpha=0.8, edgecolor='black', linewidth=2)
bars2 = ax.bar(x + width/2, perf_increase, width, label='性能提升 (%)', 
               color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=2)

# 添加數值標籤
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylabel('提升幅度 (%)', fontsize=12, fontweight='bold')
ax.set_title('邊際效益遞減：MIMO 秩提升 vs 性能提升', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(transitions, fontsize=12)
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3, axis='y')

# 添加註解
ax.annotate('大幅提升', xy=(0, 25), xytext=(0, 35),
            arrowprops=dict(arrowstyle='->', color='green', lw=2),
            fontsize=12, color='green', fontweight='bold',
            ha='center')
ax.annotate('微小提升\n(飽和)', xy=(1, 0.5), xytext=(1, 10),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=12, color='red', fontweight='bold',
            ha='center')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'mimo_marginal_returns.png', dpi=300, bbox_inches='tight')
print("✓ Saved: mimo_marginal_returns.png")
plt.close()

# 複製到 docs
import shutil
for filename in ['mimo_saturation_efficiency.png', 'mimo_theoretical_utilization.png', 'mimo_marginal_returns.png']:
    src = OUTPUT_DIR / filename
    dst = Path('/Users/hungwei/Desktop/Proj/Mamba-Orin-Nano-Custom-S6-CUDA/mamba3/docs') / filename
    shutil.copy(src, dst)
    print(f"✓ Copied {filename} to docs/")

print("\n✅ 所有 MIMO 飽和分析圖表已生成！")
