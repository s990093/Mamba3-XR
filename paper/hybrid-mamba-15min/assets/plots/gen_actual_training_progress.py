"""
Generate actual_training_progress.png  —  v2 clean redesign
3 panels:
  (A) CE Loss  — hero panel, full width
  (B) LR + Router Temperature
  (C) Gradient Norm health
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

# ── Load data ──────────────────────────────────────────────────────
SRC = Path(__file__).parent.parent / "train_log.csv"
OUT = Path(__file__).parent / "actual_training_progress.png"

df = pd.read_csv(SRC)
# remove outlier spikes after step 50 (loss > 12)
df = df[(df.step <= 50) | (df.loss < 12)].copy()
df = df.sort_values("step").reset_index(drop=True)

TOTAL_PLAN   = 80_000
CURRENT_STEP = int(df.step.max())
CKPT_STEP    = 38_400
TOKENS_B     = df.tokens_seen.max() / 1e9
ELAPSED_H    = df.elapsed_s.max() / 3600

# ── EMA helper ─────────────────────────────────────────────────────
def ema(series, alpha=0.02):
    a = np.empty(len(series))
    a[0] = series.iloc[0]
    for i in range(1, len(series)):
        a[i] = alpha * series.iloc[i] + (1 - alpha) * a[i - 1]
    return a

df["ce_ema"]   = ema(df.ce_loss,   alpha=0.012)
df["loss_ema"] = ema(df.loss,      alpha=0.012)
df["gn_ema"]   = ema(df.grad_norm, alpha=0.025)

# thin scatter every N steps
THIN = 25
ds = df[df.step % THIN == 0].copy()

# ── Colour palette ─────────────────────────────────────────────────
BG        = "#FFFFFF"
GRID      = "#F1F5F9"
AXIS_COL  = "#475569"
CE_FILL   = "#BFDBFE"          # light blue fill under loss
CE_LINE   = "#1D4ED8"          # bold blue
LOSS_LINE = "#7C3AED"          # purple dashed
RAW_DOT   = "#93C5FD"          # pale scatter
CKPT_C    = "#DC2626"          # red dashed checkpoint
NOW_C     = "#0891B2"          # cyan current step
LR_C      = "#2563EB"
TMP_C     = "#7C3AED"
GN_C      = "#E11D48"
GN_FILL   = "#FFE4E6"

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        GRID,
    "grid.linewidth":    0.8,
    "axes.edgecolor":    "#CBD5E1",
    "xtick.color":       AXIS_COL,
    "ytick.color":       AXIS_COL,
    "xtick.labelsize":   8.5,
    "ytick.labelsize":   8.5,
})

fig = plt.figure(figsize=(15, 11), facecolor=BG)
gs  = GridSpec(2, 2, figure=fig,
               height_ratios=[2.6, 1.4],
               hspace=0.52, wspace=0.30,
               left=0.07, right=0.96, top=0.88, bottom=0.08)

ax_loss = fig.add_subplot(gs[0, :])   # full width
ax_lr   = fig.add_subplot(gs[1, 0])
ax_gn   = fig.add_subplot(gs[1, 1])

for ax in (ax_loss, ax_lr, ax_gn):
    ax.set_facecolor(BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# ── helpers ────────────────────────────────────────────────────────
def x_kstep_fmt(ax):
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}K" if x >= 1000 else str(int(x))))

def add_vert(ax, x, color, ls, lw=1.2, alpha=0.75):
    ax.axvline(x, color=color, lw=lw, ls=ls, alpha=alpha, zorder=3)

def shade_progress(ax):
    ax.axvspan(0, CURRENT_STEP, alpha=0.04, color=LR_C, zorder=0, lw=0)
    ax.axvspan(CURRENT_STEP, TOTAL_PLAN, alpha=0.015, color="#94A3B8", zorder=0, lw=0)
    ax.set_xlim(0, TOTAL_PLAN)

# ══════════════════════════════════════════════════════════════════
# (A)  Loss  — hero panel
# ══════════════════════════════════════════════════════════════════
ax = ax_loss

shade_progress(ax)

# filled area under CE EMA
ax.fill_between(df.step, df.ce_ema, alpha=0.18, color=CE_LINE, zorder=1)

# raw CE scatter (light)
ax.scatter(ds.step, ds.ce_loss,
           s=1.8, color=RAW_DOT, alpha=0.4, zorder=2, rasterized=True,
           label="CE Loss (raw, every 25 steps)")

# smoothed CE
ax.plot(df.step, df.ce_ema,   color=CE_LINE,   lw=2.2, zorder=5,
        label="CE Loss (EMA smoothed)")
# total loss
ax.plot(df.step, df.loss_ema, color=LOSS_LINE, lw=1.5, ls=(0, (5,3)),
        zorder=4, alpha=0.85, label="Total Loss  (CE + LB + Z)")

# checkpoint line
add_vert(ax, CKPT_STEP, CKPT_C, "--", lw=1.4, alpha=0.8)
yl = ax.get_ylim()
ax.text(CKPT_STEP + 300, 9.6,
        f"checkpoint\nstep {CKPT_STEP//1000}K",
        color=CKPT_C, fontsize=8, va="top")

# current step line
add_vert(ax, CURRENT_STEP, NOW_C, "-.", lw=1.5, alpha=0.9)

# ── annotate final CE value ────────────────────────────────────────
final_ce = float(df.ce_ema.iloc[-1])
ax.annotate(
    f"CE = {final_ce:.3f}\n(step {CURRENT_STEP:,})",
    xy=(CURRENT_STEP, final_ce),
    xytext=(CURRENT_STEP - 9000, final_ce + 1.1),
    fontsize=9.5, color=CE_LINE, fontweight="bold",
    arrowprops=dict(arrowstyle="-|>", color=CE_LINE, lw=1.3,
                    connectionstyle="arc3,rad=0.15"),
    bbox=dict(facecolor="#EFF6FF", edgecolor=CE_LINE,
              boxstyle="round,pad=0.4", lw=1.2),
    zorder=10,
)

# ── progress badge ─────────────────────────────────────────────────
prog_pct = CURRENT_STEP / TOTAL_PLAN * 100
badge_txt = (f"  Training in progress:  "
             f"step {CURRENT_STEP:,} / {TOTAL_PLAN:,}  ({prog_pct:.1f}%)  |  "
             f"{TOKENS_B:.2f}B tokens  |  "
             f"Elapsed: {ELAPSED_H:.1f}h  ")
bbox_props = dict(boxstyle="round,pad=0.5", facecolor="#F0FDF4",
                  edgecolor="#16A34A", lw=1.3, alpha=0.95)
ax.text(0.5, 0.97, badge_txt,
        transform=ax.transAxes, ha="center", va="top",
        fontsize=8.8, color="#15803D", bbox=bbox_props, zorder=10)

# planned-end vertical
ax.axvline(TOTAL_PLAN, color="#94A3B8", lw=0.9, ls=":", zorder=2)
ax.text(TOTAL_PLAN - 200, 2.35, "80K\n(plan)", color="#94A3B8",
        fontsize=7.5, ha="right", va="bottom")

ax.set_ylim(2.0, 11.5)
ax.set_ylabel("Loss", fontsize=10, color=AXIS_COL)
ax.set_xlabel("Training Step", fontsize=10, color=AXIS_COL)
ax.set_title("(A)  Training Loss", fontsize=12, fontweight="bold",
             loc="left", pad=8, color="#1E293B")
x_kstep_fmt(ax)

legend_elems = [
    Line2D([0],[0], color=CE_LINE,   lw=2.2,          label="CE Loss (EMA)"),
    Line2D([0],[0], color=LOSS_LINE, lw=1.5, ls=(0,(5,3)), label="Total Loss (CE+LB+Z)"),
    Line2D([0],[0], color=RAW_DOT,   lw=0, marker="o",
           ms=4, alpha=0.6,                            label="CE raw (every 25 steps)"),
    Line2D([0],[0], color=CKPT_C,    lw=1.4, ls="--", label=f"Checkpoint @ step {CKPT_STEP//1000}K"),
    Line2D([0],[0], color=NOW_C,     lw=1.5, ls="-.", label=f"Current  (step {CURRENT_STEP//1000}K)"),
]
ax.legend(handles=legend_elems, loc="upper right", fontsize=8.5,
          framealpha=0.95, edgecolor="#CBD5E1", ncol=2, columnspacing=1.2)

# ══════════════════════════════════════════════════════════════════
# (B)  LR & Router Temperature
# ══════════════════════════════════════════════════════════════════
ax = ax_lr
ax2 = ax.twinx()
ax2.set_facecolor(BG)
ax2.spines["top"].set_visible(False)

shade_progress(ax)

ax.plot(df.step, df.lr * 1e4, color=LR_C, lw=1.9, label="LR (×10⁻⁴)")
ax2.plot(df.step, df.router_temp, color=TMP_C, lw=1.8,
         ls=(0,(6,2)), alpha=0.9, label="Router Temp T(s)")

add_vert(ax, CKPT_STEP, CKPT_C, "--", lw=1.2, alpha=0.7)
add_vert(ax, CURRENT_STEP, NOW_C, "-.", lw=1.3, alpha=0.8)

ax.set_xlabel("Training Step", fontsize=9.5, color=AXIS_COL)
ax.set_ylabel(r"LR $(\times 10^{-4})$", fontsize=9.5, color=LR_C)
ax2.set_ylabel("Router Temperature", fontsize=9.5, color=TMP_C)
ax.tick_params(axis="y", labelcolor=LR_C)
ax2.tick_params(axis="y", labelcolor=TMP_C)
ax2.spines["right"].set_visible(True)
ax2.spines["right"].set_color("#CBD5E1")
ax.set_title("(B)  LR & Router Temperature",
             fontsize=11, fontweight="bold", loc="left", pad=6, color="#1E293B")
x_kstep_fmt(ax)

h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1+h2, l1+l2, fontsize=8.5, framealpha=0.95,
          edgecolor="#CBD5E1", loc="upper right")

# ══════════════════════════════════════════════════════════════════
# (C)  Gradient Norm
# ══════════════════════════════════════════════════════════════════
ax = ax_gn

shade_progress(ax)

# remove extreme outliers for display (> 99.5th percentile)
gn_cap = df.grad_norm.quantile(0.995)
ds_gn  = ds[ds.grad_norm <= gn_cap]

ax.fill_between(df.step, df.gn_ema, alpha=0.12, color=GN_C, zorder=1)
ax.scatter(ds_gn.step, ds_gn.grad_norm,
           s=1.5, color="#FCA5A5", alpha=0.35, zorder=2, rasterized=True)
ax.plot(df.step, df.gn_ema, color=GN_C, lw=2.0, zorder=5,
        label="Grad Norm (EMA)")

ax.axhline(1.0, color="#94A3B8", lw=0.8, ls=":", zorder=2)
ax.text(TOTAL_PLAN * 0.95, 1.04, "norm = 1.0",
        fontsize=7.5, color="#94A3B8", ha="right")

add_vert(ax, CKPT_STEP, CKPT_C, "--", lw=1.2, alpha=0.7)
add_vert(ax, CURRENT_STEP, NOW_C, "-.", lw=1.3, alpha=0.8)

# final grad norm annotation
final_gn = float(df.gn_ema.iloc[-1])
ax.annotate(f"norm = {final_gn:.3f}",
            xy=(CURRENT_STEP, final_gn),
            xytext=(CURRENT_STEP - 8000, final_gn + 0.06),
            fontsize=8.5, color=GN_C,
            arrowprops=dict(arrowstyle="-|>", color=GN_C, lw=1.0),
            bbox=dict(facecolor="#FFF1F2", edgecolor=GN_C,
                      boxstyle="round,pad=0.3", lw=1))

ax.set_ylim(0, min(gn_cap * 1.05, 2.0))
ax.set_xlabel("Training Step", fontsize=9.5, color=AXIS_COL)
ax.set_ylabel("Gradient L2 Norm", fontsize=9.5, color=AXIS_COL)
ax.set_title("(C)  Gradient Health",
             fontsize=11, fontweight="bold", loc="left", pad=6, color="#1E293B")
ax.legend(fontsize=8.5, framealpha=0.95, edgecolor="#CBD5E1", loc="upper left")
x_kstep_fmt(ax)

# ── Global title ───────────────────────────────────────────────────
fig.suptitle(
    "Hybrid Mamba-TuckerMoE  —  Live Training Dashboard\n"
    f"Step {CURRENT_STEP:,} / {TOTAL_PLAN:,}  |  "
    f"{TOKENS_B:.2f}B tokens seen  |  "
    f"Elapsed: {ELAPSED_H:.1f}h  |  Still training...",
    fontsize=13, fontweight="bold", color="#1E293B", y=0.97,
)

fig.savefig(OUT, dpi=160, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"saved -> {OUT}")
print(f"step={CURRENT_STEP}  CE_ema={df['ce_ema'].iloc[-1]:.4f}  "
      f"gn_ema={df['gn_ema'].iloc[-1]:.4f}  tokens={TOKENS_B:.3f}B")
