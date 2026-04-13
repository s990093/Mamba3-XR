from __future__ import annotations

import os
import re
import subprocess
import warnings
from functools import lru_cache

import psutil


def _parse_powermetrics_output(output: str) -> tuple[float | None, dict[str, float]]:
    """Ported from mac-gpu: parse active/idle residency and freq distribution."""
    freq_distribution: dict[str, float] = {}

    active_match = re.search(r"GPU\s+HW\s+active\s+residency:\s*([0-9]+(?:\.[0-9]+)?)%", output, re.IGNORECASE)
    if active_match:
        util = float(active_match.group(1))
    else:
        idle_match = re.search(r"GPU\s+idle\s+residency:\s*([0-9]+(?:\.[0-9]+)?)%", output, re.IGNORECASE)
        if not idle_match:
            return None, freq_distribution
        idle = float(idle_match.group(1))
        util = max(0.0, min(100.0, 100.0 - idle))

    # Example (from mac-gpu):
    # GPU HW active residency: 41.2% (390 MHz: 12.4% 648 MHz: 28.8%)
    freq_match = re.search(r"GPU\s+HW\s+active\s+residency:\s*[0-9]+(?:\.[0-9]+)?%\s*\(([^)]+)\)", output, re.IGNORECASE)
    if freq_match:
        inner = freq_match.group(1)
        for mhz, percent in re.findall(r"([0-9]+)\s*MHz:\s*([0-9]+(?:\.[0-9]+)?)%", inner):
            value = float(percent)
            if value > 0:
                freq_distribution[mhz] = value
    return util, freq_distribution


@lru_cache(maxsize=1)
def _apple_chip_info() -> tuple[str, int]:
    """Read Apple chip name and GPU core count once."""
    chip_name = ""
    gpu_cores = 0
    try:
        hw = subprocess.run(
            ["system_profiler", "SPHardwareDataType"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        hw_out = f"{hw.stdout}\n{hw.stderr}"
        chip_match = re.search(r"Chip:\s*(Apple[^\n]+)", hw_out, re.IGNORECASE)
        if chip_match:
            chip_name = chip_match.group(1).strip()
    except Exception:
        chip_name = ""

    try:
        disp = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True,
            text=True,
            timeout=2.5,
            check=False,
        )
        disp_out = f"{disp.stdout}\n{disp.stderr}"
        # Apple Silicon display section often has:
        # "Total Number of Cores: 19"
        cores_match = re.search(r"Total Number of Cores:\s*([0-9]+)", disp_out, re.IGNORECASE)
        if cores_match:
            gpu_cores = int(cores_match.group(1))
    except Exception:
        gpu_cores = 0

    return chip_name, gpu_cores


def _nvml_gpu_stats() -> tuple[float, float, float, str]:
    try:
        # Keep runtime compatible while suppressing upstream deprecation noise.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The pynvml package is deprecated.*",
                category=FutureWarning,
            )
            import pynvml  # type: ignore

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used_gb = float(mem.used) / (1024**3)
        total_gb = float(mem.total) / (1024**3)
        return util, used_gb, total_gb, "nvml"
    except Exception:
        return 0.0, 0.0, 0.0, ""


def _apple_gpu_stats() -> tuple[float, float, float, str, str, dict[str, float]]:
    # Apple Silicon real GPU usage from powermetrics.
    # This can require elevated privileges on some macOS versions.
    try:
        result = subprocess.run(
            ["powermetrics", "--samplers", "all", "-i", "1000", "-n", "1"],
            capture_output=True,
            text=True,
            timeout=2.5,
            check=False,
        )
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            hint = "powermetrics needs elevated privileges" if "superuser" in stderr.lower() else stderr
            return 0.0, 0.0, 0.0, "", hint[:200], {}

        output = f"{result.stdout}\n{result.stderr}"
        util, freq_distribution = _parse_powermetrics_output(output)
        if util is not None:
            return util, 0.0, 0.0, "powermetrics", "", freq_distribution
    except Exception:
        return 0.0, 0.0, 0.0, "", "", {}
    return 0.0, 0.0, 0.0, "", "", {}


def _safe_gpu_stats() -> tuple[float, float, float, str, str, dict[str, float]]:
    util, used, total, source = _nvml_gpu_stats()
    if source:
        return util, used, total, source, "", {}
    util, used, total, source, note, freq_distribution = _apple_gpu_stats()
    if source:
        return util, used, total, source, note, freq_distribution
    # Last-resort fallback so UI still updates on unsupported devices.
    return float(psutil.cpu_percent(interval=None)), 0.0, 0.0, "fallback", note or "", {}


def current_monitor_snapshot() -> dict[str, object]:
    gpu_util, vram_used_gb, vram_total_gb, gpu_source, apple_note, freq_distribution = _safe_gpu_stats()
    ram = psutil.virtual_memory()
    cpu_percent = float(psutil.cpu_percent(interval=None))
    chip_name, gpu_cores = _apple_chip_info()
    gpu_note = ""
    if gpu_source == "fallback" and os.uname().sysname.lower() == "darwin":
        gpu_note = apple_note or "powermetrics unavailable; using fallback estimate"

    return {
        "framework": "MLX/CUDA",
        "gpu_source": gpu_source,
        "chip_name": chip_name,
        "gpu_cores": gpu_cores,
        "gpu_note": gpu_note,
        "gpu_util": round(gpu_util, 2),
        "cpu_percent": round(cpu_percent, 2),
        "vram_used_gb": round(vram_used_gb, 3),
        "vram_total_gb": round(vram_total_gb, 3),
        "gpu_freq_mhz_distribution": freq_distribution,
        "ram_percent": round(float(ram.percent), 2),
        "ram_used_gb": round(float(ram.used) / (1024**3), 3),
        "ram_total_gb": round(float(ram.total) / (1024**3), 3),
    }


def full_monitor_payload() -> dict[str, object]:
    """Hardware snapshot plus inference lifecycle (for WS monitor and GET /api/status)."""
    from app.core.inference_status import inference_status

    snap = current_monitor_snapshot()
    snap["inference"] = inference_status.snapshot(os.getenv("INFERENCE_BACKEND", "inf"))
    return snap
