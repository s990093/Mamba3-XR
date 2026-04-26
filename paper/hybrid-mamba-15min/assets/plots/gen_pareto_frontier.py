#!/usr/bin/env python3
"""
Pareto Frontier: Active Parameters vs. Dense-Equivalent Model Capacity
Shows that Hybrid Mamba-TuckerMoE achieves large effective capacity
at a fraction of the per-token inference cost.

Reference data sourced from:
  - Mamba (Gu & Dao, 2023): arXiv:2312.00752, Table 1 (The Pile PPL)
  - Mamba-2 (Dao & Gu, 2024): arXiv:2405.21060
  - Pythia (Biderman et al., 2023): arXiv:2304.01373, Table 2
  - GPT-NeoX-20B (Black et al., 2022): arXiv:2204.06745
  - Mixtral 8x7B (Jiang et al., 2024): arXiv:2401.04088
  - Switch Transformer (Fedus et al., 2022): arXiv:2101.03961
  - Our model: measured values from §9.3, §9.5, §9.6
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT = Path(__file__).resolve().parent / "pareto_frontier.png"

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
# Each entry: (label, active_M, dense_equiv_M, pile_ppl, marker, color, size)
#
# For dense models: active == dense_equiv (they lie on the "dense diagonal")
# For sparse MoE:   active < dense_equiv  (off-diagonal upper-left)
# For TuckerMoE:    active << dense_equiv (far upper-left = Pareto-optimal)
#
# PPL is used only for bubble shading / annotation; not an axis.
# The Pile PPL references (300B tokens training):
#   Pythia 160M~11.68, 410M~9.65, 1B~8.82, 2.8B~8.22, 6.9B~7.67
#   Mamba 130M~10.56, 370M~8.28, 790M~7.35, 1.4B~6.80, 2.8B~6.22
#   GPT-NeoX 20B~7.50
#   Mixtral 8x7B: not directly comparable (different dataset, but state-of-art)
# ---------------------------------------------------------------------------

DENSE_COLOR     = "#4C72B0"  # blue
MAMBA_COLOR     = "#DD8452"  # orange
MAMBA2_COLOR    = "#C44E52"  # red
MOE_COLOR       = "#55A868"  # green
OURS_COLOR      = "#8172B3"  # purple

# (label, active_M, dense_equiv_M, ppl_pile, color, zorder)
models = [
    # ── Dense Transformers (Pythia / GPT-NeoX) ──────────────────────────────
    ("Pythia-160M",    160,    160,  11.68, DENSE_COLOR, 2),
    ("Pythia-410M",    410,    410,   9.65, DENSE_COLOR, 2),
    ("Pythia-1B",     1000,   1000,   8.82, DENSE_COLOR, 2),
    ("Pythia-2.8B",   2800,   2800,   8.22, DENSE_COLOR, 2),
    ("GPT-NeoX-20B", 20000,  20000,   7.50, DENSE_COLOR, 2),
    # ── Mamba-1 ─────────────────────────────────────────────────────────────
    ("Mamba-130M",    130,    130,   10.56, MAMBA_COLOR, 2),
    ("Mamba-370M",    370,    370,    8.28, MAMBA_COLOR, 2),
    ("Mamba-790M",    790,    790,    7.35, MAMBA_COLOR, 2),
    ("Mamba-1.4B",   1400,   1400,   6.80, MAMBA_COLOR, 2),
    ("Mamba-2.8B",   2800,   2800,   6.22, MAMBA_COLOR, 2),
    # ── Mamba-2 ─────────────────────────────────────────────────────────────
    ("Mamba2-370M",   370,    370,    8.24, MAMBA2_COLOR, 2),
    ("Mamba2-1.3B",  1300,   1300,   6.94, MAMBA2_COLOR, 2),
    ("Mamba2-2.7B",  2700,   2700,   6.32, MAMBA2_COLOR, 2),
    # ── Sparse MoE ──────────────────────────────────────────────────────────
    # Mixtral 8x7B: 12.9B active out of 46.7B total
    ("Mixtral-8×7B", 12900,  46700,  None, MOE_COLOR, 2),
    # Switch-Base (Fedus 2022): 7.4B active, ~26.4B total (256 experts, 1 active)
    ("Switch-Base",   7400,  26400,  None, MOE_COLOR, 2),
]

# Our model
OURS_ACTIVE      = 230      # M  (active per token)
OURS_DENSE_EQUIV = 2435     # M  (dense-equivalent before Tucker compression)
OURS_TOTAL       = 550      # M  (actual stored params after compression)

# ---------------------------------------------------------------------------
# Figure setup
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 7.5), dpi=300)
fig.patch.set_facecolor("white")
ax.set_facecolor("#F8F9FA")

# ── Pareto-frontier diagonal (dense model line: active = dense_equiv) ───────
x_diag = np.logspace(np.log10(100), np.log10(50000), 300)
ax.plot(x_diag, x_diag, "--", color="#AAAAAA", linewidth=1.2, zorder=1,
        label="Dense models (Active = Capacity)")

# ── Shaded "Pareto-superior" region ─────────────────────────────────────────
# Region where dense-equiv is high AND active params are low
# We shade above the curve from our model's position
ax.fill_between(
    [80, OURS_ACTIVE + 5],
    [OURS_DENSE_EQUIV - 50, OURS_DENSE_EQUIV - 50],
    [60000, 60000],
    alpha=0.07, color=OURS_COLOR, zorder=0,
    label="Pareto-superior region"
)

# ── Draw baseline models ─────────────────────────────────────────────────────
label_positions = {
    "Pythia-160M":  (0, -200),
    "Pythia-410M":  (0, -260),
    "Pythia-1B":    (0, -500),
    "Pythia-2.8B":  (60, -1200),
    "GPT-NeoX-20B": (200, 1000),
    "Mamba-130M":   (-80, 150),
    "Mamba-370M":   (-80, 200),
    "Mamba-790M":   (30, -450),
    "Mamba-1.4B":   (40, -800),
    "Mamba-2.8B":   (60, 900),
    "Mamba2-370M":  (30, -270),
    "Mamba2-1.3B":  (-400, 600),
    "Mamba2-2.7B":  (100, -1400),
    "Mixtral-8×7B": (200, 2000),
    "Switch-Base":  (-2000, 1500),
}

already_labeled = {"Dense": False, "Mamba-1": False, "Mamba-2": False, "MoE": False}
group_map = {
    DENSE_COLOR:  "Dense",
    MAMBA_COLOR:  "Mamba-1",
    MAMBA2_COLOR: "Mamba-2",
    MOE_COLOR:    "MoE",
}

for label, active, dense_eq, ppl, color, zo in models:
    # bubble size scales loosely with model size
    s = max(40, min(400, dense_eq / 60))
    ax.scatter(active, dense_eq, s=s, color=color, alpha=0.82,
               edgecolors="white", linewidths=0.8, zorder=zo + 2)

    # annotation
    dx, dy = label_positions.get(label, (20, 20))
    short = label
    ax.annotate(
        short, xy=(active, dense_eq),
        xytext=(active + dx, dense_eq + dy),
        fontsize=6.5, color="#333333",
        arrowprops=dict(arrowstyle="-", color="#CCCCCC", lw=0.6),
        va="center",
    )

# ── Our model ────────────────────────────────────────────────────────────────
ax.scatter(OURS_ACTIVE, OURS_DENSE_EQUIV,
           s=520, color=OURS_COLOR, marker="*",
           edgecolors="white", linewidths=1.2, zorder=10)

# Annotation with arrow from dense-equiv position to actual stored params
ax.annotate(
    "Hybrid Mamba-\nTuckerMoE (Ours)\n"
    f"Active: {OURS_ACTIVE}M  |  Stored: {OURS_TOTAL}M\n"
    f"Dense-Equiv Capacity: {OURS_DENSE_EQUIV / 1000:.1f}B",
    xy=(OURS_ACTIVE, OURS_DENSE_EQUIV),
    xytext=(480, 6500),
    fontsize=8.5, color=OURS_COLOR, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=OURS_COLOR, lw=1.5,
                    connectionstyle="arc3,rad=-0.25"),
    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=OURS_COLOR,
              lw=1.2, alpha=0.95),
    zorder=11,
)

# Horizontal dashed lines showing dense-equivalent capacity of nearby baselines
for label, active, dense_eq, *_ in models:
    if label in ("Pythia-2.8B", "Mamba-2.8B"):
        ax.axhline(dense_eq, color="#DDDDDD", linestyle=":", linewidth=0.8, zorder=1)

ax.axhline(OURS_DENSE_EQUIV, color=OURS_COLOR, linestyle=":",
           linewidth=0.9, alpha=0.5, zorder=1)

# ── Axes ─────────────────────────────────────────────────────────────────────
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(80, 35000)
ax.set_ylim(80, 60000)

ax.set_xlabel("Active Parameters per Token (M)  [inference cost ∝]",
              fontsize=11, labelpad=8)
ax.set_ylabel("Dense-Equivalent Model Capacity (M params)  [knowledge ∝]",
              fontsize=11, labelpad=8)
ax.set_title(
    "Pareto Frontier: Inference Cost vs. Effective Model Capacity\n"
    "Hybrid Mamba-TuckerMoE achieves 2.4B-class capacity at 230M active-parameter cost",
    fontsize=12, fontweight="bold", pad=14,
)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(color=DENSE_COLOR,  label="Dense Transformer (Pythia / GPT-NeoX)"),
    mpatches.Patch(color=MAMBA_COLOR,  label="Mamba-1 (SSM)"),
    mpatches.Patch(color=MAMBA2_COLOR, label="Mamba-2 (SSD)"),
    mpatches.Patch(color=MOE_COLOR,    label="Sparse MoE (Mixtral / Switch)"),
    mpatches.Patch(color=OURS_COLOR,   label="Hybrid Mamba-TuckerMoE (Ours)"),
    plt.Line2D([0], [0], color="#AAAAAA", linestyle="--", linewidth=1.2,
               label="Dense diagonal (Active = Capacity)"),
]
ax.legend(handles=legend_handles, fontsize=8, loc="upper left",
          framealpha=0.95, edgecolor="#CCCCCC")

# ── Source note ───────────────────────────────────────────────────────────────
fig.text(
    0.99, 0.01,
    "Sources: Mamba (Gu & Dao, 2023), Mamba-2 (Dao & Gu, 2024), Pythia (Biderman et al., 2023),\n"
    "Mixtral (Jiang et al., 2024), Switch (Fedus et al., 2022). Our model: §9.3 / §9.6.",
    ha="right", va="bottom", fontsize=6, color="#888888",
    transform=fig.transFigure,
)

ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.5, color="#CCCCCC")
ax.tick_params(axis="both", labelsize=9)

plt.subplots_adjust(bottom=0.10, top=0.92, left=0.09, right=0.97)
fig.savefig(OUT, dpi=300, facecolor="white")
print(f"Saved → {OUT}")
