"""
Generate two publication-quality figures for TuckerMoE backward propagation:
  1. tucker_backward_flow.png   — step-by-step gradient flow diagram
  2. tucker_sparse_matrix.png   — nine-square token-expert sparse mask heatmap
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.patheffects as pe
from pathlib import Path

OUT = Path(__file__).parent

# ─────────────────────────────────────────────────────────────────
# Shared style
# ─────────────────────────────────────────────────────────────────
FONT_FAMILY = "DejaVu Sans"
plt.rcParams.update({
    "font.family": FONT_FAMILY,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

BG        = "#F7F9FC"
SHARED_C  = "#3B82F6"   # blue  – shared parameters
EXPERT_C  = "#F59E0B"   # amber – expert-specific
GATE_C    = "#10B981"   # green – gating / router
FLOW_C    = "#EF4444"   # red   – gradient flow arrow
NODE_BG   = "#FFFFFF"
SHADOW    = "#CBD5E1"


# ═══════════════════════════════════════════════════════════════════
# Figure 1 — Backward pass flow
# ═══════════════════════════════════════════════════════════════════
def make_backward_flow():
    fig, ax = plt.subplots(figsize=(13, 7.2), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    # ── title ──────────────────────────────────────────────────────
    ax.text(6.5, 6.95, "TuckerMoE Backward Pass  —  Gradient Flow (L → x)",
            ha="center", va="top", fontsize=14, fontweight="bold",
            color="#1E293B")

    # ── column header labels ────────────────────────────────────────
    cols = {"Loss / δ": 1.1,
            "U_out\n(shared)": 3.0,
            "G_e / G / U_exp\n(expert)": 5.7,
            "x_s\n(bottleneck)": 8.5,
            "U_in\n(shared)": 11.0}

    for label, cx in cols.items():
        ax.text(cx, 6.55, label, ha="center", va="top",
                fontsize=8.5, color="#475569", style="italic")

    # ── helper: draw a rounded box ─────────────────────────────────
    def node(x, y, w, h, label, sub="", color=NODE_BG, lc="#94A3B8",
             fontsize=9, bold=False):
        box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                             boxstyle="round,pad=0.06",
                             facecolor=color, edgecolor=lc, linewidth=1.4,
                             zorder=3)
        ax.add_patch(box)
        weight = "bold" if bold else "normal"
        ax.text(x, y + (0.12 if sub else 0), label,
                ha="center", va="center", fontsize=fontsize,
                fontweight=weight, color="#1E293B", zorder=4)
        if sub:
            ax.text(x, y - 0.22, sub, ha="center", va="center",
                    fontsize=7.2, color="#64748B", zorder=4)
        return (x, y)

    # ── helper: draw gradient arrow ────────────────────────────────
    def arrow(x0, y0, x1, y1, label="", color=FLOW_C, lw=1.8):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                   lw=lw, mutation_scale=14),
                    zorder=5)
        if label:
            mx, my = (x0+x1)/2, (y0+y1)/2
            ax.text(mx, my + 0.18, label, ha="center", va="bottom",
                    fontsize=7.5, color=color, zorder=6,
                    bbox=dict(facecolor=BG, edgecolor="none", pad=1.5))

    # ── helper: draw dashed fork line ──────────────────────────────
    def fork(x0, y0, targets, color="#94A3B8"):
        xs = [t[0] for t in targets]
        mid_x = (min(xs) + max(xs)) / 2
        # vertical down
        ax.plot([x0, x0], [y0, y0 - 0.35], color=color, lw=1.4, ls="--", zorder=4)
        # horizontal rail
        ax.plot([min(xs), max(xs)], [y0-0.35, y0-0.35], color=color, lw=1.4, ls="--", zorder=4)
        for (tx, ty) in targets:
            ax.plot([tx, tx], [y0-0.35, ty+0.26], color=color, lw=1.4, ls="--", zorder=4)
            ax.annotate("", xy=(tx, ty+0.26), xytext=(tx, ty+0.27),
                        arrowprops=dict(arrowstyle="-|>", color=color,
                                       lw=1.4, mutation_scale=11), zorder=5)

    # ─── Row layout (y positions) ──────────────────────────────────
    # Forward direction: left (input x) → right (output y)
    # Backward direction: right → left (drawn as bold red arrows)

    Y_MAIN = 4.8   # main path row
    Y_CORE = 3.2   # core tensor branch row
    Y_EXP  = 2.0   # expert-id branch

    # ─── Forward nodes (top half, faint) ──────────────────────────
    fwd_alpha = 0.30
    for label, cx, fcc in [
        ("x  (input)", 11.0, "#E2E8F0"),
        ("U_in", 9.3, "#DBEAFE"),
        ("x_s = RMSNorm\n(x U_in)", 8.0, "#DBEAFE"),
        ("G_e", 5.7, "#FEF3C7"),
        ("U_out", 3.0, "#DBEAFE"),
        ("y = Σ p_e x_s G_e U_out + b", 1.1, "#E2E8F0"),
    ]:
        ax.text(cx, 5.65, label, ha="center", va="center", fontsize=7.8,
                color="#94A3B8", style="italic")

    ax.annotate("", xy=(1.9, 5.65), xytext=(0.5, 5.65),
                arrowprops=dict(arrowstyle="-|>", color="#CBD5E1", lw=1.2), zorder=2)
    ax.annotate("", xy=(3.8, 5.65), xytext=(2.1, 5.65),
                arrowprops=dict(arrowstyle="-|>", color="#CBD5E1", lw=1.2), zorder=2)
    ax.annotate("", xy=(4.8, 5.65), xytext=(3.8, 5.65),
                arrowprops=dict(arrowstyle="-|>", color="#CBD5E1", lw=1.2), zorder=2)
    ax.annotate("", xy=(7.3, 5.65), xytext=(6.5, 5.65),
                arrowprops=dict(arrowstyle="-|>", color="#CBD5E1", lw=1.2), zorder=2)
    ax.annotate("", xy=(9.0, 5.65), xytext=(7.7, 5.65),
                arrowprops=dict(arrowstyle="-|>", color="#CBD5E1", lw=1.2), zorder=2)
    ax.text(5.7, 5.65, "top-k route", ha="center", va="center",
            fontsize=7.0, color="#CBD5E1", style="italic")
    ax.text(6.5, 5.82, "FORWARD", ha="center", va="bottom",
            fontsize=7.5, color="#CBD5E1", fontweight="bold")
    ax.axhline(y=5.35, color="#CBD5E1", lw=0.8, ls=":")

    # ─── BACKWARD main nodes ───────────────────────────────────────
    ax.text(6.5, 5.18, "BACKWARD", ha="center", va="top",
            fontsize=7.5, color=FLOW_C, fontweight="bold")

    # Loss node
    node(1.1, Y_MAIN, 1.6, 0.58, "Loss  L", color="#FEE2E2", lc=FLOW_C, bold=True)

    # delta node
    node(1.1, 3.7, 1.6, 0.58,
         "δ = ∂L/∂y",
         sub="upstream gradient",
         color="#FEE2E2", lc=FLOW_C)

    arrow(1.1, Y_MAIN - 0.29, 1.1, 3.99,
          label="backprop", color=FLOW_C)

    # ── Step 1: dL/dG_e ────────────────────────────────────────────
    node(5.7, Y_MAIN, 2.0, 0.60,
         "① ∂L/∂G_e",
         sub="= x_sᵀ · (p_e δ U_outᵀ)",
         color="#FEF9C3", lc=EXPERT_C)

    arrow(1.1, 3.41, 5.0, Y_MAIN,
          label="× p_e,  through U_out", color=FLOW_C)

    # ── Step 2: dL/dU_out ──────────────────────────────────────────
    node(3.0, Y_MAIN, 1.9, 0.60,
         "② ∂L/∂U_out",
         sub="= Σ_e (x_s G_e)ᵀ (p_e δ)",
         color="#DBEAFE", lc=SHARED_C)

    arrow(5.0, Y_MAIN, 3.75, Y_MAIN,
          label="Σ over top-k e", color=FLOW_C)

    # ── Step 3a: dL/dU_expert ──────────────────────────────────────
    node(5.7, Y_CORE, 2.1, 0.60,
         "③a ∂L/∂U_exp[e,a]",
         sub="= Σ_{b,c} ∂G_e[b,c] · G[a,b,c]",
         color="#FEF9C3", lc=EXPERT_C)

    # ── Step 3b: dL/dG (core) ──────────────────────────────────────
    node(5.7, Y_EXP, 2.1, 0.60,
         "③b ∂L/∂G[a,b,c]",
         sub="= Σ_{e∈E(x)} U_exp[e,a] · ∂G_e[b,c]",
         color="#DCFCE7", lc=GATE_C)

    fork(5.7, Y_MAIN - 0.30,
         [(5.7, Y_CORE + 0.30), (5.7, Y_EXP + 0.30)],
         color=EXPERT_C)

    # ── Step 4: dL/dx_s ────────────────────────────────────────────
    node(8.5, Y_MAIN, 2.1, 0.60,
         "④ ∂L/∂x_s",
         sub="= Σ_e (p_e δ U_outᵀ) G_eᵀ",
         color="#F0FDF4", lc=GATE_C)

    arrow(6.7, Y_MAIN, 7.45, Y_MAIN,
          label="Σ over top-k e", color=FLOW_C)

    # ── Step 5: dL/dU_in ───────────────────────────────────────────
    node(11.0, Y_MAIN, 1.9, 0.60,
         "⑤ ∂L/∂U_in",
         sub="= xᵀ · ∂L/∂(x U_in)",
         color="#DBEAFE", lc=SHARED_C)

    arrow(9.55, Y_MAIN, 10.05, Y_MAIN,
          label="through RMSNorm", color=FLOW_C)

    # ── Legend ─────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(facecolor="#DBEAFE", edgecolor=SHARED_C, label="Shared param (updated by all tokens)"),
        mpatches.Patch(facecolor="#FEF9C3", edgecolor=EXPERT_C, label="Expert-specific (sparse update, top-k only)"),
        mpatches.Patch(facecolor="#DCFCE7", edgecolor=GATE_C,   label="Shared core G (updated by selected experts)"),
        mpatches.Patch(facecolor="#FEE2E2", edgecolor=FLOW_C,   label="Loss / gradient signal"),
    ]
    ax.legend(handles=legend_items, loc="lower center",
              bbox_to_anchor=(0.5, -0.01), ncol=2,
              fontsize=8, framealpha=0.9, edgecolor="#CBD5E1")

    # ── Sparsity callout ───────────────────────────────────────────
    ax.text(5.7, 1.15,
            "  p_e = 0 for unselected experts → zero gradient for G_e, U_exp[e]  ",
            ha="center", va="center", fontsize=8.5, color="#7C3AED",
            bbox=dict(facecolor="#EDE9FE", edgecolor="#7C3AED",
                      boxstyle="round,pad=0.4", lw=1.2))

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out_path = OUT / "tucker_backward_flow.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[1] saved → {out_path}")


# ═══════════════════════════════════════════════════════════════════
# Figure 2 — Nine-square sparse matrix
# ═══════════════════════════════════════════════════════════════════
def make_sparse_matrix():
    T, E = 9, 8   # tokens × experts (3×3 visual with real E=8)
    rng = np.random.default_rng(42)

    # Simulate top-2 routing
    raw = rng.standard_normal((T, E))
    mask = np.zeros((T, E))
    for t in range(T):
        top2 = np.argsort(raw[t])[-2:]
        mask[t, top2] = 1.0

    # Soft weights (only where mask=1)
    softmax_w = np.exp(raw) / np.exp(raw).sum(axis=1, keepdims=True)
    P = mask * softmax_w

    # ── Figure layout: 1 row, 3 panels ─────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.2), facecolor=BG,
                              gridspec_kw={"wspace": 0.38})
    fig.suptitle("TuckerMoE — Token × Expert Sparse Gate Matrix  (top-2, T=9, E=8)",
                 fontsize=13, fontweight="bold", color="#1E293B", y=1.01)

    tok_labels = [f"t{i+1}" for i in range(T)]
    exp_labels = [f"e{j+1}" for j in range(E)]

    # ── Panel A: binary mask M ─────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor(BG)
    im = ax.imshow(mask, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(E)); ax.set_xticklabels(exp_labels, fontsize=8)
    ax.set_yticks(range(T)); ax.set_yticklabels(tok_labels, fontsize=8)
    ax.set_xlabel("Expert", fontsize=9)
    ax.set_ylabel("Token", fontsize=9)
    ax.set_title("(A)  Binary Mask  M", fontsize=10, fontweight="bold", pad=8)

    # cell annotations
    for i in range(T):
        for j in range(E):
            v = int(mask[i, j])
            color = "white" if v == 1 else "#94A3B8"
            ax.text(j, i, str(v), ha="center", va="center",
                    fontsize=8.5, fontweight="bold", color=color)

    # highlight 3×3 "九宮格" region
    rect = plt.Rectangle((-0.5, -0.5), 3, 3,
                          linewidth=2.2, edgecolor="#7C3AED",
                          facecolor="none", linestyle="--", zorder=5)
    ax.add_patch(rect)
    ax.text(1.0, -1.05, "3x3 nine-square example", ha="center", va="top",
            fontsize=7.5, color="#7C3AED")

    # ── Panel B: weighted mask P ───────────────────────────────────
    ax = axes[1]
    ax.set_facecolor(BG)
    vmax_p = P.max()
    im2 = ax.imshow(P, cmap="YlOrRd", vmin=0, vmax=vmax_p, aspect="auto")
    ax.set_xticks(range(E)); ax.set_xticklabels(exp_labels, fontsize=8)
    ax.set_yticks(range(T)); ax.set_yticklabels(tok_labels, fontsize=8)
    ax.set_xlabel("Expert", fontsize=9)
    ax.set_title("(B)  Weighted Mask  P = M ⊙ softmax", fontsize=10,
                 fontweight="bold", pad=8)
    plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

    for i in range(T):
        for j in range(E):
            v = P[i, j]
            if v > 0.001:
                color = "white" if v > vmax_p * 0.6 else "#1E293B"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7.5, color=color)
            else:
                ax.text(j, i, "0", ha="center", va="center",
                        fontsize=7.5, color="#CBD5E1")

    # ── Panel C: gradient sparsity per expert ──────────────────────
    ax = axes[2]
    ax.set_facecolor(BG)
    active_frac = mask.mean(axis=0)         # fraction of tokens that activate each expert
    grad_density = (P > 0).mean(axis=0)    # same here

    colors = [EXPERT_C if f > 0 else "#E2E8F0" for f in active_frac]
    bars = ax.bar(exp_labels, active_frac, color=colors,
                  edgecolor="#94A3B8", linewidth=0.9, width=0.6)

    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Expert", fontsize=9)
    ax.set_ylabel("Fraction of tokens routed (=grad density)", fontsize=8.5)
    ax.set_title("(C)  Per-Expert Gradient Density\n"
                 "p_e = 0 → zero gradient for unselected experts",
                 fontsize=10, fontweight="bold", pad=8)
    ax.axhline(y=0.5, color="#94A3B8", lw=0.8, ls=":")
    ax.text(7.5, 0.52, "50%", fontsize=7.5, color="#64748B")

    for bar, frac in zip(bars, active_frac):
        ax.text(bar.get_x() + bar.get_width()/2, frac + 0.02,
                f"{frac:.0%}", ha="center", va="bottom",
                fontsize=8, color="#1E293B")

    # annotation
    ax.text(3.5, 0.88,
            "Unselected experts:\n∂L/∂G_e = 0  (no gradient update)",
            ha="center", va="top", fontsize=8, color="#7C3AED",
            bbox=dict(facecolor="#EDE9FE", edgecolor="#7C3AED",
                      boxstyle="round,pad=0.4", lw=1.2))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out_path = OUT / "tucker_sparse_matrix.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[2] saved → {out_path}")


if __name__ == "__main__":
    make_backward_flow()
    make_sparse_matrix()
    print("Done.")
