#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS = ROOT / "results_template.csv"


def to_float(value: str) -> float | None:
    value = (value or "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def summarize_by_experiment(rows: list[dict[str, str]]) -> str:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("experiment_name", "unknown")].append(row)

    parts: list[str] = []
    for experiment_name, exp_rows in grouped.items():
        parts.append(f"## {experiment_name}")
        parts.append("")
        parts.append("| method | runs with PPL | best PPL | best tok/s | min memory (MB) |")
        parts.append("|---|---:|---:|---:|---:|")

        method_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in exp_rows:
            method_groups[row.get("method", "unknown")].append(row)

        for method, method_rows in sorted(method_groups.items()):
            ppls = [to_float(row.get("val_ppl", "")) for row in method_rows]
            ppls = [value for value in ppls if value is not None]
            toks = [to_float(row.get("tok_per_s", "")) for row in method_rows]
            toks = [value for value in toks if value is not None]
            mems = [to_float(row.get("peak_memory_mb", "")) for row in method_rows]
            mems = [value for value in mems if value is not None]

            best_ppl = min(ppls) if ppls else None
            best_tok = max(toks) if toks else None
            min_mem = min(mems) if mems else None

            parts.append(
                "| {method} | {num_runs} | {best_ppl} | {best_tok} | {min_mem} |".format(
                    method=method,
                    num_runs=len(ppls),
                    best_ppl=f"{best_ppl:.3f}" if best_ppl is not None else "-",
                    best_tok=f"{best_tok:.2f}" if best_tok is not None else "-",
                    min_mem=f"{min_mem:.1f}" if min_mem is not None else "-",
                )
            )
        parts.append("")
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Tucker validation CSV results.")
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()

    rows = load_rows(args.results_csv)
    print(summarize_by_experiment(rows))


if __name__ == "__main__":
    main()
