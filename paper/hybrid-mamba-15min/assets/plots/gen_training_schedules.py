"""
Generate training_schedules.png:
  Left panel  — 3-phase LR schedule (warmup / linear-decay / cosine-anneal)
  Right panel — Router temperature cosine annealing
Both panels share the x-axis (training step 0–80000) and mark the step-38400 checkpoint.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path

OUT = Path(__file__).parent

# ── Hyper-parameters ───────────────────────────────────────────────
TOTAL_STEPS   = 80_000
WARMUP        = 500
LINEAR_END    = 60_500     # warmup + linear-decay region
# cosine from LINEAR_END → TOTAL_STEPS
LR_BASE       = 3e-4
LR_STABLE     = LR_BASE * 0.8     # linear decay target  (0.8×)
LR_FINAL      = LR_BASE * 0.05    # cosine anneal target (5%)
T_START       = 2.0
T_END         = 0.5
CKPT_STEP     = 38_400

BG   = "#F7F9FC"
GRAY = "#94A3B8"
BLUE = "#3B82F6"
ORG  = "#F59E0B"
RED  = "#EF4444"
GRN  = "#10B981"

def lr_at(s):
    if s <= WARMUP:
        return LR_BASE * s / WARMUP
    elif s <= LINEAR_END:
        frac = (s - WARMUP) / (LINEAR_END - WARMUP)
        return LR_BASE - (LR_BASE - LR_STABLE) * frac
    else:
        frac = (s - LINEAR_END) / (TOTAL_STEPS - LINEAR_END)
        cos  = 0.5 * (1 + np.cos(np.pi * frac))
        return LR_FINAL + (LR_STABLE - LR_FINAL) * cos

def temp_at(s):
    p = s / TOTAL_STEPS
    return T_END + 0.5 * (T_START - T_END) * (1 + np.cos(np.pi * p))

steps = np.arange(0, TOTAL_STEPS + 1, 100)
lrs   = np.array([lr_at(s) for s in steps])
temps = np.array([temp_at(s) for s in steps])

# ── Figure ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 5.2), facecolor=BG)
gs  = GridSpec(1, 2, figure=fig, wspace=0.38)
ax_lr  = fig.add_subplot(gs[0])
ax_tmp = fig.add_subplot(gs[1])

fig.suptitle("Training Schedules  —  Hybrid Mamba-TuckerMoE  (80,000 steps)",
             fontsize=13, fontweight="bold", color="#1E293B", y=1.01)

kw_ckpt = dict(color=RED, lw=1.4, ls="--", zorder=5)

# ── LR panel ───────────────────────────────────────────────────────
ax = ax_lr
ax.set_facecolor(BG)

# Phase regions (shaded)
ax.axvspan(0,           WARMUP,       alpha=0.08, color=GRAY)
ax.axvspan(WARMUP,      LINEAR_END,   alpha=0.08, color=BLUE)
ax.axvspan(LINEAR_END,  TOTAL_STEPS,  alpha=0.08, color=ORG)

# LR curve, coloured by phase
mask_w = steps <= WARMUP
mask_l = (steps >= WARMUP) & (steps <= LINEAR_END)
mask_c = steps >= LINEAR_END

ax.plot(steps[mask_w], lrs[mask_w] * 1e4, color=GRAY, lw=2.2, label="Warmup (linear)")
ax.plot(steps[mask_l], lrs[mask_l] * 1e4, color=BLUE, lw=2.2, label="Linear decay → 0.8×")
ax.plot(steps[mask_c], lrs[mask_c] * 1e4, color=ORG,  lw=2.2, label="Cosine → 5%")

# checkpoint line
ax.axvline(CKPT_STEP, **kw_ckpt)
ax.text(CKPT_STEP + 600, 2.75,
        f"ckpt\nstep {CKPT_STEP//1000}K",
        fontsize=8, color=RED, va="top")

# annotate phase boundaries
for xv, label, col in [(WARMUP, "500", GRAY), (LINEAR_END, "60.5K", BLUE)]:
    ax.axvline(xv, color=col, lw=0.8, ls=":", zorder=3)
    ax.text(xv, 0.05, label, ha="center", fontsize=7.5, color=col,
            transform=ax.get_xaxis_transform(), va="bottom")

# reference lines
for yv, label in [(LR_BASE*1e4, r"$3\times10^{-4}$  (base)"),
                  (LR_STABLE*1e4, r"$2.4\times10^{-4}$  (0.8×)"),
                  (LR_FINAL*1e4,  r"$1.5\times10^{-5}$  (5%)")]:
    ax.axhline(yv, color="#CBD5E1", lw=0.8, ls=":")
    ax.text(TOTAL_STEPS*0.97, yv + 0.02, label,
            ha="right", va="bottom", fontsize=7.2, color="#64748B")

ax.set_xlim(0, TOTAL_STEPS)
ax.set_ylim(-0.1, 3.25)
ax.set_xlabel("Training Step", fontsize=10)
ax.set_ylabel(r"Learning Rate ($\times 10^{-4}$)", fontsize=10)
ax.set_title("(A)  LR Schedule", fontsize=11, fontweight="bold", pad=8)
ax.legend(fontsize=8.5, framealpha=0.9, edgecolor="#CBD5E1",
          loc="upper right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_facecolor(BG)

# ── Temperature panel ──────────────────────────────────────────────
ax = ax_tmp
ax.set_facecolor(BG)

# fill: high-temp = warm, low-temp = cool
cmap = plt.get_cmap("coolwarm_r")
for i in range(len(steps)-1):
    frac = temps[i] / T_START          # 1 at start, ~0.25 at end
    ax.fill_betweenx([T_END - 0.05, temps[i]],
                     steps[i], steps[i+1],
                     color=cmap(1 - frac * 0.7), alpha=0.18, lw=0)

ax.plot(steps, temps, color="#7C3AED", lw=2.5, label="Router Temperature T(s)")

# checkpoint line
ax.axvline(CKPT_STEP, **kw_ckpt)
ax.text(CKPT_STEP + 600, T_START - 0.12,
        f"ckpt\nstep {CKPT_STEP//1000}K",
        fontsize=8, color=RED, va="top")

# checkpoint temperature value
t_ckpt = temp_at(CKPT_STEP)
ax.plot(CKPT_STEP, t_ckpt, "o", color=RED, ms=7, zorder=6)
ax.annotate(f"T={t_ckpt:.2f}", xy=(CKPT_STEP, t_ckpt),
            xytext=(CKPT_STEP + 3000, t_ckpt + 0.08),
            fontsize=8.5, color=RED,
            arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.2))

# reference lines
for yv, label, col in [
    (T_START, f"$T_{{start}}={T_START}$  (flat distribution)", "#EF4444"),
    (T_END,   f"$T_{{end  }}={T_END}$  (sharp distribution)",  "#3B82F6"),
]:
    ax.axhline(yv, color=col, lw=0.9, ls=":")
    ax.text(TOTAL_STEPS * 0.97, yv + 0.02, label,
            ha="right", va="bottom", fontsize=7.5, color=col)

# softmax illustration annotation
ax.annotate("High T:\nExperts explored broadly\n(uniform routing)",
            xy=(8000, temp_at(8000)),
            xytext=(5000, 1.05),
            fontsize=7.8, color="#7C3AED",
            bbox=dict(facecolor="#EDE9FE", edgecolor="#7C3AED",
                      boxstyle="round,pad=0.4", lw=1),
            arrowprops=dict(arrowstyle="-|>", color="#7C3AED", lw=1.2))

ax.annotate("Low T:\nExperts specialised\n(sharp routing)",
            xy=(72000, temp_at(72000)),
            xytext=(54000, 0.82),
            fontsize=7.8, color="#2563EB",
            bbox=dict(facecolor="#DBEAFE", edgecolor="#2563EB",
                      boxstyle="round,pad=0.4", lw=1),
            arrowprops=dict(arrowstyle="-|>", color="#2563EB", lw=1.2))

ax.set_xlim(0, TOTAL_STEPS)
ax.set_ylim(T_END - 0.25, T_START + 0.35)
ax.set_xlabel("Training Step", fontsize=10)
ax.set_ylabel("Router Temperature T(s)", fontsize=10)
ax.set_title("(B)  Router Temperature Annealing", fontsize=11,
             fontweight="bold", pad=8)
ax.legend(fontsize=8.5, framealpha=0.9, edgecolor="#CBD5E1", loc="upper right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
out = OUT / "training_schedules.png"
fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"saved -> {out}")
