#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS = ROOT / "results_template.csv"
DEFAULT_OUTPUT = ROOT / "plots" / "tucker_validation_dashboard.png"


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def to_float(value: str) -> float | None:
    value = (value or "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def group_rows(rows: list[dict[str, str]], experiment_name: str) -> list[dict[str, str]]:
    return [row for row in rows if row.get("experiment_name") == experiment_name]


def plot_compression_frontier(ax, rows: list[dict[str, str]]) -> None:
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        compression = to_float(row.get("compression_ratio", ""))
        ppl = to_float(row.get("val_ppl", ""))
        method = row.get("method", "")
        if compression is None or ppl is None or not method:
            continue
        grouped[method].append((compression, ppl))

    for method, points in grouped.items():
        points.sort(key=lambda item: item[0])
        ax.plot([x for x, _ in points], [y for _, y in points], marker="o", label=method)

    ax.set_title("Compression Frontier")
    ax.set_xlabel("Compression Ratio")
    ax.set_ylabel("Validation PPL")
    ax.grid(True, linestyle="--", alpha=0.5)
    if grouped:
        ax.legend()


def plot_rank_sensitivity(ax, rows: list[dict[str, str]]) -> None:
    series = {"r1": [], "r2": [], "r3": []}
    for row in rows:
        ppl = to_float(row.get("val_ppl", ""))
        if ppl is None:
            continue
        for key in ("r1", "r2", "r3"):
            value = to_float(row.get(key, ""))
            if value is None:
                continue
            # Only include points where the other two ranks are at default settings.
            if key == "r1" and row.get("r2") == "1024" and row.get("r3") == "256":
                series["r1"].append((value, ppl))
            if key == "r2" and row.get("r1") == "4" and row.get("r3") == "256":
                series["r2"].append((value, ppl))
            if key == "r3" and row.get("r1") == "4" and row.get("r2") == "1024":
                series["r3"].append((value, ppl))

    for key, points in series.items():
        if not points:
            continue
        points.sort(key=lambda item: item[0])
        ax.plot([x for x, _ in points], [y for _, y in points], marker="o", label=key)

    ax.set_title("Rank Sensitivity")
    ax.set_xlabel("Rank Value")
    ax.set_ylabel("Validation PPL")
    ax.grid(True, linestyle="--", alpha=0.5)
    if any(series.values()):
        ax.legend()


def plot_shared_ablation(ax, rows: list[dict[str, str]]) -> None:
    labels = []
    values = []
    for row in rows:
        ppl = to_float(row.get("val_ppl", ""))
        method = row.get("method", "")
        if ppl is None or not method:
            continue
        labels.append(method)
        values.append(ppl)

    if labels:
        ax.bar(labels, values)
        ax.tick_params(axis="x", rotation=20)
    ax.set_title("Shared-Subspace Ablation")
    ax.set_ylabel("Validation PPL")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)


def plot_recovery(ax, rows: list[dict[str, str]]) -> None:
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        steps = to_float(row.get("finetune_steps", ""))
        ppl = to_float(row.get("val_ppl", ""))
        method = row.get("method", "")
        if steps is None or ppl is None or not method:
            continue
        grouped[method].append((steps, ppl))

    for method, points in grouped.items():
        points.sort(key=lambda item: item[0])
        ax.plot([x for x, _ in points], [y for _, y in points], marker="o", label=method)

    ax.set_title("Recovery Curve")
    ax.set_xlabel("Finetune Steps")
    ax.set_ylabel("Validation PPL")
    ax.grid(True, linestyle="--", alpha=0.5)
    if grouped:
        ax.legend()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Tucker validation experiment results.")
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = load_rows(args.results_csv)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Tucker Validation Dashboard", fontsize=16)

    plot_compression_frontier(axes[0, 0], group_rows(rows, "compression_frontier"))
    plot_rank_sensitivity(axes[0, 1], group_rows(rows, "rank_sensitivity"))
    plot_shared_ablation(axes[1, 0], group_rows(rows, "shared_subspace_ablation"))
    plot_recovery(axes[1, 1], group_rows(rows, "recovery_and_system_gain"))

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(args.output, dpi=180)
    print(f"Saved Tucker validation dashboard to: {args.output}")


if __name__ == "__main__":
    main()
