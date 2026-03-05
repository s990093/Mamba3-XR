#!/usr/bin/env python3
"""
重新生成 MIMO 理論利用率圖表（修正為 d_state=32）
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
import shutil

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = Path('/Users/hungwei/.gemini/antigravity/brain/bd37de13-51b9-479f-bbe1-066937a600cf')

# 修正後的數據：d_state = 32
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
print("✓ Saved: mimo_theoretical_utilization.png (updated with d_state=32)")

# 複製到 docs
dst = Path('/Users/hungwei/Desktop/Proj/Mamba-Orin-Nano-Custom-S6-CUDA/mamba3/docs/mimo_theoretical_utilization.png')
shutil.copy(OUTPUT_DIR / 'mimo_theoretical_utilization.png', dst)
print(f"✓ Copied to docs/")

plt.close()
print("\n✅ 已更新理論利用率圖表（d_state=32）")
