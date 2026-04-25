#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[3]
TRAIN_PY = PROJECT_ROOT / "train.py"
DEFAULT_MANIFEST = ROOT / "experiment_manifest.json"
DEFAULT_RESULTS = ROOT / "results_template.csv"
DEFAULT_OUTPUT_ROOT = ROOT / "runs"
DEFAULT_STATUS_CSV = ROOT / "batch_status.csv"

SUPPORTED_METHODS = {"dense_moe", "tucker_moe", "tucker_shared"}
UNSUPPORTED_METHODS = {"svd_moe", "tucker_unshared"}


@dataclass
class RunSpec:
    run_id: str
    experiment_name: str
    method: str
    run_dir: str
    status: str
    reason: str
    command_preview: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch runner for Tucker validation experiments.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--status-csv", type=Path, default=DEFAULT_STATUS_CSV)
    parser.add_argument("--data-path", type=Path, help="Path to the tokenized .bin dataset.")
    parser.add_argument("--base-checkpoint", type=Path, help="Optional checkpoint used for recovery experiments.")
    parser.add_argument("--experiment", action="append", help="Only run selected experiment group(s).")
    parser.add_argument("--run-id", action="append", help="Only run selected run_id(s).")
    parser.add_argument("--max-runs", type=int, default=0, help="Limit how many planned runs to process. 0 = no limit.")
    parser.add_argument("--steps", type=int, default=2000, help="Training steps for non-recovery experiments.")
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--disable-compile", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Plan runs and write status CSV without executing training.")
    parser.add_argument("--allow-unsupported", action="store_true", help="Include unsupported rows in the status CSV.")
    return parser.parse_args()


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


def to_int(value: str) -> int | None:
    number = to_float(value)
    return int(number) if number is not None else None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def filter_rows(rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str]]:
    experiment_filter = set(args.experiment or [])
    run_filter = set(args.run_id or [])
    out: list[dict[str, str]] = []
    for row in rows:
        if experiment_filter and row.get("experiment_name") not in experiment_filter:
            continue
        if run_filter and row.get("run_id") not in run_filter:
            continue
        out.append(row)
    if args.max_runs > 0:
        return out[: args.max_runs]
    return out


def build_train_kwargs(row: dict[str, str], args: argparse.Namespace, run_dir: Path) -> tuple[dict[str, object] | None, str]:
    method = row.get("method", "")
    experiment = row.get("experiment_name", "")

    if method in UNSUPPORTED_METHODS:
        return None, f"backend for method '{method}' is not implemented in train.py"

    if method not in SUPPORTED_METHODS:
        return None, f"unknown method '{method}'"

    if args.data_path is None and not args.dry_run:
        return None, "missing --data-path"

    is_recovery = experiment == "recovery_and_system_gain"
    finetune_steps = to_int(row.get("finetune_steps", "")) or 0
    train_steps = finetune_steps if is_recovery and finetune_steps > 0 else args.steps

    checkpoint_path = run_dir / "checkpoint.pt"
    if is_recovery:
        if args.base_checkpoint is None and not args.dry_run:
            return None, "recovery experiment requires --base-checkpoint"
        if args.base_checkpoint is not None and not checkpoint_path.exists():
            shutil.copy2(args.base_checkpoint, checkpoint_path)

    kwargs: dict[str, object] = {
        "DATA_PATH": str(args.data_path) if args.data_path is not None else "__DRY_RUN_DATA_PATH__",
        "OUTPUT_DIR": str(run_dir),
        "LOG_FILE": str(run_dir / "train_log.csv"),
        "CHECKPOINT_SAVE_PATH": str(checkpoint_path),
        "SEQ_LEN": args.seq_len,
        "BATCH_SIZE": args.batch_size,
        "GRADIENT_ACCUMULATION_STEPS": args.grad_accum,
        "LR": args.lr,
        "WARMUP": args.warmup,
        "STEPS": train_steps,
        "CHECKPOINT_EVERY": args.checkpoint_every,
        "ENABLE_TORCH_COMPILE": not args.disable_compile,
        "USE_KMOE": method != "dense_moe",
    }

    r1 = to_int(row.get("r1", ""))
    r2 = to_int(row.get("r2", ""))
    r3 = to_int(row.get("r3", ""))
    if r1 is not None:
        kwargs["KMOE_R1"] = r1
    if r2 is not None:
        kwargs["KMOE_R2"] = r2
    if r3 is not None:
        kwargs["KMOE_R3"] = r3

    return kwargs, ""


def build_command_preview(kwargs: dict[str, object] | None) -> str:
    if kwargs is None:
        return ""
    visible_keys = [
        "USE_KMOE",
        "KMOE_R1",
        "KMOE_R2",
        "KMOE_R3",
        "STEPS",
        "LR",
        "SEQ_LEN",
        "BATCH_SIZE",
        "GRADIENT_ACCUMULATION_STEPS",
        "DATA_PATH",
        "CHECKPOINT_SAVE_PATH",
    ]
    parts = [f"{key}={kwargs[key]}" for key in visible_keys if key in kwargs]
    return "train.train(" + ", ".join(parts) + ")"


def run_train_subprocess(kwargs: dict[str, object]) -> subprocess.CompletedProcess[str]:
    payload = dict(kwargs)
    payload["_train_py"] = str(TRAIN_PY)
    code = (
        "import importlib.util, json, sys;"
        "payload=json.loads(sys.argv[1]);"
        "train_py=payload.pop('_train_py');"
        "spec=importlib.util.spec_from_file_location('mamba3_train', train_py);"
        "mod=importlib.util.module_from_spec(spec);"
        "spec.loader.exec_module(mod);"
        "mod.train(**payload)"
    )
    return subprocess.run(
        [sys.executable, "-c", code, json.dumps(payload)],
        cwd=str(PROJECT_ROOT),
        text=True,
        capture_output=True,
        check=False,
    )


def write_status_csv(status_csv: Path, specs: list[RunSpec]) -> None:
    ensure_dir(status_csv.parent)
    with status_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["run_id", "experiment_name", "method", "run_dir", "status", "reason", "command_preview"],
        )
        writer.writeheader()
        for spec in specs:
            writer.writerow(asdict(spec))


def main() -> None:
    args = parse_args()
    rows = filter_rows(load_rows(args.results_csv), args)
    ensure_dir(args.output_root)

    specs: list[RunSpec] = []
    for row in rows:
        run_id = row.get("run_id", "unknown_run")
        experiment = row.get("experiment_name", "unknown_experiment")
        method = row.get("method", "unknown_method")
        run_dir = args.output_root / experiment / run_id
        ensure_dir(run_dir)

        kwargs, reason = build_train_kwargs(row, args, run_dir)
        preview = build_command_preview(kwargs)

        if kwargs is None:
            status = "unsupported" if method in UNSUPPORTED_METHODS else "skipped"
            if status == "unsupported" and not args.allow_unsupported:
                specs.append(
                    RunSpec(
                        run_id=run_id,
                        experiment_name=experiment,
                        method=method,
                        run_dir=str(run_dir),
                        status=status,
                        reason=reason,
                        command_preview=preview,
                    )
                )
                continue
            specs.append(
                RunSpec(
                    run_id=run_id,
                    experiment_name=experiment,
                    method=method,
                    run_dir=str(run_dir),
                    status=status,
                    reason=reason,
                    command_preview=preview,
                )
            )
            continue

        if args.dry_run:
            specs.append(
                RunSpec(
                    run_id=run_id,
                    experiment_name=experiment,
                    method=method,
                    run_dir=str(run_dir),
                    status="planned",
                    reason="dry-run",
                    command_preview=preview,
                )
            )
            continue

        result = run_train_subprocess(kwargs)
        stdout_path = run_dir / "stdout.log"
        stderr_path = run_dir / "stderr.log"
        stdout_path.write_text(result.stdout or "", encoding="utf-8")
        stderr_path.write_text(result.stderr or "", encoding="utf-8")

        status = "completed" if result.returncode == 0 else "failed"
        reason = "ok" if result.returncode == 0 else f"exit_code={result.returncode}"
        specs.append(
            RunSpec(
                run_id=run_id,
                experiment_name=experiment,
                method=method,
                run_dir=str(run_dir),
                status=status,
                reason=reason,
                command_preview=preview,
            )
        )

    write_status_csv(args.status_csv, specs)
    summary = {
        "status_csv": str(args.status_csv),
        "num_rows": len(rows),
        "planned": sum(1 for spec in specs if spec.status == "planned"),
        "completed": sum(1 for spec in specs if spec.status == "completed"),
        "failed": sum(1 for spec in specs if spec.status == "failed"),
        "unsupported": sum(1 for spec in specs if spec.status == "unsupported"),
        "skipped": sum(1 for spec in specs if spec.status == "skipped"),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
