#!/usr/bin/env python3
"""
Publication-quality MLX inference benchmark figure for §9.6.

Four panels:
  (A) Prefill & Decode throughput by device / config
  (B) Decode-state memory: Hybrid vs Pure Transformer @512 steps
  (C) Memory scaling with decode length (Mamba fixed vs KV growing)
  (D) Graph-level fusion: per-component decode latency breakdown

Data sources:
  - Table 4 in report.md (manually transcribed — real MLX benchmark runs)
  - analyze_kv_cache_sizes.py analytical model
  - mlx_profile_components.py / profile_mlx_infer.py component rollups
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent  # hybrid-mamba-15min/
OUT = ROOT / "assets" / "plots" / "mlx_inference_benchmark.png"

_RC = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "none",
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 9.5,
    "axes.edgecolor": "#2a2a2a",
    "axes.linewidth": 0.8,
    "axes.labelcolor": "#1a1a1a",
    "axes.titleweight": "600",
    "axes.titlesize": 11,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "xtick.color": "#333",
    "ytick.color": "#333",
    "grid.color": "#d8d8d8",
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.9,
    "legend.edgecolor": "#cccccc",
    "legend.fancybox": False,
    "figure.titlesize": 14,
}


def _clean(ax):
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── Data from Table 4 (real benchmark runs) ───────────────────────────
CONFIGS = [
    "M2 Pro\nbf16",
    "M2 Pro\n8-bit quant",
    "M1\nbf16",
]
PREFILL_TPS = [3800, 3650, 2100]
DECODE_TPS  = [42, 68, 27]
KV_MIB      = [22.3, 14.1, 22.3]

# Pure Transformer reference (30-layer, d=768, 12 heads, KV @512 decode, bf16)
PURE_XF_KV_MIB = 107.0

# ── Analytical cache model (from analyze_kv_cache_sizes.py) ───────────
DECODE_STEPS    = [1, 32, 64, 128, 256, 512, 1024, 2048]
MAMBA_STATE_MIB = 9.035
PREFILL_LEN     = 256
SLACK           = 8

def _xf_kv_mib(slot_len: int) -> float:
    n_layers = 6
    n_heads = 12
    head_dim = 64
    bpe = 2  # bf16
    per_layer = 2 * 1 * n_heads * slot_len * head_dim * bpe
    return n_layers * per_layer / (1024**2)

def _pure_xf_kv_mib(slot_len: int) -> float:
    n_layers = 30
    n_heads = 12
    head_dim = 64
    bpe = 2
    per_layer = 2 * 1 * n_heads * slot_len * head_dim * bpe
    return n_layers * per_layer / (1024**2)

HYBRID_TOTAL = []
PURE_XF_TOTAL = []
for d in DECODE_STEPS:
    slot = PREFILL_LEN + d + SLACK
    HYBRID_TOTAL.append(MAMBA_STATE_MIB + _xf_kv_mib(slot))
    PURE_XF_TOTAL.append(_pure_xf_kv_mib(slot))

# ── Decode component breakdown (representative M2 Pro eager decode) ───
# From profile_mlx_infer component rollup; keys → human labels.
MAMBA_COMPONENTS = {
    "In-proj + angles":  1.82,
    "B/C Norm + RoPE":   0.95,
    "x_up (Tucker MoE)": 3.41,
    "SSM core":          1.28,
    "Dense branch":      1.65,
    "out_proj (MoE)":    2.74,
}
XF_COMPONENTS = {
    "Attn Norm + QKV":    0.62,
    "KV + SDPA + O":      1.13,
    "FFN Norm":           0.18,
    "FFN gate (MoE)":     1.05,
    "FFN up (MoE)":       0.98,
    "FFN down (MoE)":     1.12,
    "Post-FFN LS":        0.14,
}


def main():
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.8))

        # ─── (A) Throughput ──────────────────────────────────────────
        ax = axes[0, 0]
        x = np.arange(len(CONFIGS))
        w = 0.32
        bars_p = ax.bar(x - w/2, PREFILL_TPS, w, label="Prefill",
                        color="#2166ac", edgecolor="#1a1a1a", linewidth=0.3)
        bars_d = ax.bar(x + w/2, DECODE_TPS, w, label="Decode",
                        color="#b2182b", edgecolor="#1a1a1a", linewidth=0.3)
        ax.set_ylabel("Throughput (tokens / sec)")
        ax.set_title("(A)  Prefill & Decode Throughput", pad=8)
        ax.set_xticks(x, CONFIGS)
        ax.set_ylim(0, max(PREFILL_TPS) * 1.22)
        _clean(ax)
        ax.yaxis.grid(True)

        for bar, val in zip(bars_p, PREFILL_TPS):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 60,
                    f"{val:,}", ha="center", va="bottom", fontsize=7.5, color="#2166ac")
        for bar, val in zip(bars_d, DECODE_TPS):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 60,
                    str(val), ha="center", va="bottom", fontsize=7.5, color="#b2182b")

        leg = ax.legend(frameon=True, fontsize=8, loc="upper right",
                        facecolor="white", framealpha=1.0)
        leg.get_frame().set_linewidth(0.5)

        # ─── (B) Memory comparison @512 decode ───────────────────────
        ax = axes[0, 1]
        categories = ["Hybrid\nMamba3-XR", "Pure\nTransformer"]
        mamba_part = [MAMBA_STATE_MIB, 0]
        kv_part    = [KV_MIB[0] - MAMBA_STATE_MIB, PURE_XF_KV_MIB]
        # stacked
        b1 = ax.bar(categories, mamba_part, 0.45, label="Mamba SSM state (fixed)",
                     color="#1b7837", edgecolor="#1a1a1a", linewidth=0.3)
        b2 = ax.bar(categories, kv_part, 0.45, bottom=mamba_part,
                     label="Transformer KV cache",
                     color="#762a83", edgecolor="#1a1a1a", linewidth=0.3)
        ax.set_ylabel("Decode-state memory (MiB)")
        ax.set_title("(B)  Memory @ 512 Decode Steps", pad=8)
        _clean(ax)
        ax.yaxis.grid(True)

        for i, (m, k) in enumerate(zip(mamba_part, kv_part)):
            total = m + k
            ax.text(i, total + 1.5, f"{total:.1f}", ha="center", fontsize=8.5, weight="600")

        savings_pct = (1 - KV_MIB[0] / PURE_XF_KV_MIB) * 100
        ax.annotate(
            f"{savings_pct:.0f}% smaller",
            xy=(0, KV_MIB[0]),
            xytext=(0.55, PURE_XF_KV_MIB * 0.75),
            fontsize=8.5, color="#b2182b", weight="600",
            arrowprops=dict(arrowstyle="->", color="#b2182b", lw=1.2),
        )
        leg = ax.legend(frameon=True, fontsize=8, loc="upper left",
                        facecolor="white", framealpha=1.0)
        leg.get_frame().set_linewidth(0.5)

        # ─── (C) Memory scaling ─────────────────────────────────────
        ax = axes[1, 0]
        ax.plot(DECODE_STEPS, HYBRID_TOTAL, "o-", color="#1b7837", lw=1.8,
                markersize=4, label="Hybrid Mamba3-XR", markeredgecolor="white",
                markeredgewidth=0.4)
        ax.plot(DECODE_STEPS, PURE_XF_TOTAL, "s--", color="#762a83", lw=1.8,
                markersize=4, label="Pure Transformer (30L)",
                markeredgecolor="white", markeredgewidth=0.4)
        ax.axhline(MAMBA_STATE_MIB, color="#aaaaaa", ls=":", lw=0.9, zorder=0)
        ax.text(DECODE_STEPS[-1]*0.65, MAMBA_STATE_MIB + 1.5,
                f"Mamba state = {MAMBA_STATE_MIB:.1f} MiB (fixed)",
                fontsize=7.5, color="#666666")
        ax.set_xlabel("Decode steps (new tokens)")
        ax.set_ylabel("Total decode-state memory (MiB)")
        ax.set_title("(C)  Memory Scaling with Decode Length", pad=8)
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v)}"))
        _clean(ax)
        ax.yaxis.grid(True)
        leg = ax.legend(frameon=True, fontsize=8, loc="upper left",
                        facecolor="white", framealpha=1.0)
        leg.get_frame().set_linewidth(0.5)

        # ─── (D) Decode component breakdown ──────────────────────────
        ax = axes[1, 1]
        all_labels = list(MAMBA_COMPONENTS.keys()) + list(XF_COMPONENTS.keys())
        all_values = list(MAMBA_COMPONENTS.values()) + list(XF_COMPONENTS.values())
        n_m = len(MAMBA_COMPONENTS)
        n_x = len(XF_COMPONENTS)
        colors_m = plt.cm.Blues(np.linspace(0.35, 0.85, n_m))
        colors_x = plt.cm.Reds(np.linspace(0.35, 0.85, n_x))
        colors = list(colors_m) + list(colors_x)

        y = np.arange(len(all_labels))
        bars = ax.barh(y, all_values, color=colors, edgecolor="#1a1a1a",
                       linewidth=0.25, height=0.7)
        ax.set_yticks(y, all_labels)
        ax.invert_yaxis()
        ax.set_xlabel("Wall time per decode step (ms)")
        ax.set_title("(D)  Decode Component Breakdown (eager, M2 Pro)", pad=8)
        _clean(ax)
        ax.xaxis.grid(True)

        for bar, val in zip(bars, all_values):
            ax.text(bar.get_width() + 0.06, bar.get_y() + bar.get_height()/2,
                    f"{val:.2f}", va="center", fontsize=7.5)

        ax.axhline(n_m - 0.5, color="#888888", ls="-", lw=0.6, zorder=0)
        mamba_total = sum(MAMBA_COMPONENTS.values())
        xf_total = sum(XF_COMPONENTS.values())
        ax.text(max(all_values)*0.92, n_m/2 - 0.5,
                f"Mamba\n{mamba_total:.1f} ms",
                ha="center", va="center", fontsize=8,
                color="#2166ac", weight="600",
                bbox=dict(fc="white", ec="#2166ac", lw=0.6, pad=2, boxstyle="round,pad=0.3"))
        ax.text(max(all_values)*0.92, n_m + n_x/2 - 0.5,
                f"XF\n{xf_total:.1f} ms",
                ha="center", va="center", fontsize=8,
                color="#b2182b", weight="600",
                bbox=dict(fc="white", ec="#b2182b", lw=0.6, pad=2, boxstyle="round,pad=0.3"))

        fig.suptitle(
            "Apple Silicon MLX Inference Benchmark  —  Hybrid Mamba3-XR",
            fontsize=13.5, weight="600", y=0.995,
        )
        fig.text(
            0.5, 0.008,
            "All measurements: batch=1, bf16, MLX backend, seq_len=256 prefill.  "
            "Data from benchmark_mlx.py / profile_mlx_infer.py / analyze_kv_cache_sizes.py.",
            ha="center", fontsize=8, color="#555555",
        )
        fig.subplots_adjust(
            left=0.08, right=0.97, top=0.91, bottom=0.07,
            wspace=0.30, hspace=0.40,
        )
        fig.savefig(
            str(OUT), dpi=240, bbox_inches="tight",
            facecolor="white", edgecolor="none", pad_inches=0.12,
        )
        plt.close(fig)
        print(f"[gen_mlx_inference_figure] Wrote: {OUT}")


if __name__ == "__main__":
    main()
