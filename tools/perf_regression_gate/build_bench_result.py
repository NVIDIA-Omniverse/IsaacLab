# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Post-benchmark script: normalizes benchmark output and writes perf_regression_gate_result.json.

Locates the timestamped benchmark output file written by benchmark_non_rl.py, renames it
to the canonical ``perf_regression_gate_info.json``, classifies the failure phase from the
captured log, and writes ``perf_regression_gate_result.json`` for the aggregate job.

Usage::

    python3 tools/perf_regression_gate/build_bench_result.py \\
        --task_id Isaac-Cartpole-Direct-v0 \\
        --artifact_dir artifacts/Isaac-Cartpole-Direct-v0 \\
        --exit_code 0 \\
        --wall_time_s 48.3 \\
        --timeout_s 600 \\
        --log_file artifacts/Isaac-Cartpole-Direct-v0/benchmark.log
"""

import argparse
import glob
import json
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

_MODULE_DIR = Path(__file__).parent
_TOOLS_DIR = _MODULE_DIR.parent
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from subprocess_runner import classify_failure_phase  # noqa: E402
from task_config import get_task  # noqa: E402


def _percentile(sorted_data: list[float], p: float) -> float:
    """Linear-interpolation percentile on a pre-sorted list"""
    n = len(sorted_data)
    if n == 1:
        return sorted_data[0]
    idx = p / 100.0 * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    return sorted_data[lo] + (sorted_data[hi] - sorted_data[lo]) * (idx - lo)


def _strip_phase_prefix(name: str, phase_name: str) -> str:
    """Strip the '{task_name} {phase_name} ' prefix added by JSONFileMetrics.finalize()"""
    marker = f" {phase_name} "
    idx = name.find(marker)
    return name[idx + len(marker) :] if idx >= 0 else name


def _gpu_driver_version() -> str | None:
    """Return the GPU driver version string from nvidia-smi, or None if unavailable"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            version = result.stdout.strip().splitlines()[0].strip()
            return version if version else None
    except Exception:
        pass
    return None


def _extract_info_provenance(info_path: Path) -> dict:
    """Parse ``perf_regression_gate_info.json`` and return provenance + FPS fields

    Extracts hardware metadata (GPU name/memory/CUDA, CPU, RAM), software versions,
    git provenance, FPS distribution statistics, GPU memory used at runtime, and
    startup time.  All fields are best-effort; missing data is omitted.

    Returns:
        Dict with a subset of the following keys:

        - ``raw_fps_{mean,std,min,max,median,p5,p95}``: float
        - ``startup_time_s``: float
        - ``gpu_diag``: dict with gpu_name, gpu_total_memory_gb, cuda_version,
          nvidia_driver_version, gpu_mem_used_mb
        - ``provenance``: dict with ``hardware``, ``software``, ``git`` sub-dicts
    """
    try:
        phases: list[dict] = json.loads(info_path.read_text())
    except Exception:
        return {}

    hardware: dict = {}
    software: dict = {}
    git: dict = {}
    fps_stats: dict = {}
    gpu_mem_used_gb: float | None = None
    startup_time_s: float | None = None

    for phase in phases:
        pname: str = phase.get("phase_name", "")
        metadata_map = {
            _strip_phase_prefix(m["name"], pname): m["data"]
            for m in phase.get("metadata", [])
            if "name" in m and "data" in m
        }

        if pname == "hardware_info":
            hardware["cpu_name"] = metadata_map.get("cpu_name")
            hardware["cpu_physical_cores"] = metadata_map.get("physical_cores")
            hardware["total_ram_gb"] = metadata_map.get("total_ram_gb")
            hardware["gpu_device_count"] = metadata_map.get("gpu_device_count")
            hardware["cuda_version"] = metadata_map.get("cuda_version")
            gpu_devices = metadata_map.get("gpu_devices")
            if isinstance(gpu_devices, dict):
                current = str(metadata_map.get("gpu_current_device", 0))
                dev = gpu_devices.get(current) or next(iter(gpu_devices.values()), {})
                hardware["gpu_name"] = dev.get("name")
                hardware["gpu_total_memory_gb"] = dev.get("total_memory_gb")
                hardware["gpu_compute_capability"] = dev.get("compute_capability")
                hardware["gpu_multi_processor_count"] = dev.get("multi_processor_count")

        elif pname == "version_info":
            dev_data: dict = metadata_map.pop("dev", {}) or {}
            for k, v in metadata_map.items():
                # strip trailing "_version" suffix added by VersionInfoRecorder
                key = k[: -len("_version")] if k.endswith("_version") else k
                if v is not None:
                    software[key] = v
            git = {
                k: dev_data[k]
                for k in ("commit_hash", "commit_hash_short", "branch", "commit_date", "dirty")
                if k in dev_data
            }

        elif pname == "runtime":
            for m in phase.get("measurements", []):
                name: str = m.get("name", "")
                value = m.get("value")
                # FPS series: DictMeasurement written by BenchmarkMonitor
                if name.endswith("Step Frametimes") and isinstance(value, dict):
                    fps_series: list[float] = value.get("Environment step effective FPS", [])
                    if fps_series:
                        sorted_fps = sorted(fps_series)
                        n = len(sorted_fps)
                        fps_stats = {
                            "raw_fps_mean": statistics.mean(fps_series),
                            "raw_fps_std": statistics.stdev(fps_series) if n > 1 else 0.0,
                            "raw_fps_min": sorted_fps[0],
                            "raw_fps_max": sorted_fps[-1],
                            "raw_fps_median": _percentile(sorted_fps, 50.0),
                            "raw_fps_p5": _percentile(sorted_fps, 5.0),
                            "raw_fps_p95": _percentile(sorted_fps, 95.0),
                        }
                # GPU memory used (mean over run, in GB):  SingleMeasurement from GPUInfoRecorder
                elif name.endswith("GPU Memory Used") and isinstance(value, (int, float)):
                    gpu_mem_used_gb = float(value)

        elif pname == "startup":
            for m in phase.get("measurements", []):
                if m.get("name", "").endswith("Total Start Time (Launch to Train)"):
                    val = m.get("value")
                    if isinstance(val, (int, float)):
                        startup_time_s = float(val)
                    break

    gpu_diag: dict = {k: v for k, v in {
        "gpu_name": hardware.get("gpu_name"),
        "gpu_total_memory_gb": hardware.get("gpu_total_memory_gb"),
        "cuda_version": hardware.get("cuda_version"),
        "nvidia_driver_version": _gpu_driver_version(),
        "gpu_mem_used_mb": round(gpu_mem_used_gb * 1024, 2) if gpu_mem_used_gb is not None else None,
    }.items() if v is not None}

    result: dict = {}
    result.update(fps_stats)
    if startup_time_s is not None:
        result["startup_time_s"] = startup_time_s
    if gpu_diag:
        result["gpu_diag"] = gpu_diag
    result["provenance"] = {
        "hardware": {k: v for k, v in hardware.items() if v is not None},
        "software": software,
        "git": git,
    }
    return result


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build perf_regression_gate_result.json from a benchmark run")
    p.add_argument("--task_id", required=True)
    p.add_argument("--physics_backend", required=True, help="Physics backend used (e.g. physx, newton)")
    p.add_argument("--render_backend", default="", help="Render backend used (e.g. rtx, warp, ovrtx); empty = none")
    p.add_argument("--artifact_dir", required=True, type=Path)
    p.add_argument("--exit_code", required=True, type=int)
    p.add_argument("--wall_time_s", required=True, type=float)
    p.add_argument("--timeout_s", required=True, type=float)
    p.add_argument("--log_file", type=Path, default=None)
    p.add_argument("--attempt", type=int, default=1, help="Attempt number (1 = first run, 2 = after one retry)")
    p.add_argument("--was_retried", action="store_true", help="Set when this result comes from a retry of a failed first attempt")
    return p.parse_args()


def _normalize_benchmark_output(artifact_dir: Path, task_id: str) -> bool:
    """Rename the timestamped benchmark JSON to ``perf_regression_gate_info.json``
    
    benchmark_non_rl.py writes ``benchmark_non_rl_{task_id}_{timestamp}.json``.
    The oracle reads ``perf_regression_gate_info.json``.  This function bridges the gap.

    Returns True if perf_regression_gate_info.json exists after call
    """
    perf_regression_gate_info = artifact_dir / "perf_regression_gate_info.json"
    if perf_regression_gate_info.exists():
        return True
    # Primary pattern: exact task_id match
    matches = sorted(glob.glob(str(artifact_dir / f"benchmark_non_rl_{task_id}_*.json")))
    if not matches:
        # Fallback: any benchmark_non_rl_*.json in the artifact dir
        matches = sorted(glob.glob(str(artifact_dir / "benchmark_non_rl_*.json")))
    if not matches:
        return False
    shutil.copy(matches[-1], perf_regression_gate_info)
    return True


def main() -> int:
    args = _parse_args()
    artifact_dir = args.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)

    physics_backend: str = args.physics_backend
    render_backend: str | None = args.render_backend if args.render_backend else None
    backend_key: str = f"{physics_backend}_{render_backend}" if render_backend else physics_backend

    # Load task config for metadata embedded in bench_result
    try:
        task = get_task(args.task_id, backend_key)
        num_envs = task.num_envs
        num_frames = task.num_frames
        excluded_frames_raw = task.excluded_frames_raw
        timeout_minutes = task.timeout_minutes
        preset = task.preset
        tags = task.tags
        seed = task.seed
    except KeyError:
        print(
            f"[build_bench_result] Warning: ({args.task_id!r}, {backend_key!r}) not found in tasks.json; using defaults"
        )
        num_envs = 0
        num_frames = 0
        excluded_frames_raw = []
        timeout_minutes = int(args.timeout_s / 60)
        preset = "default"
        tags = ["always"]
        seed = None

    # Read combined stdout/stderr log for failure classification
    log_text = ""
    if args.log_file and args.log_file.exists():
        log_text = args.log_file.read_text(errors="replace")

    perf_regression_gate_info_present = _normalize_benchmark_output(artifact_dir, args.task_id)

    failure_phase = classify_failure_phase(
        stdout=log_text,
        stderr="",
        exit_code=args.exit_code,
        wall_time_s=args.wall_time_s,
        timeout_s=args.timeout_s,
    )

    # Extract FPS stats, startup time, GPU diag, and full provenance from the info artifact
    info_provenance: dict = {}
    if perf_regression_gate_info_present:
        info_provenance = _extract_info_provenance(artifact_dir / "perf_regression_gate_info.json")

    bench_result = {
        "task_id": args.task_id,
        "backend": backend_key,
        "physics_backend": physics_backend,
        "render_backend": render_backend,
        "backend_key": backend_key,
        "preset": preset,
        "attempt": args.attempt,
        "was_retried": args.was_retried,
        "exit_code": args.exit_code,
        "failure_phase": failure_phase,
        "stdout_tail": log_text[-2000:] if len(log_text) > 2000 else log_text,
        "wall_time_s": args.wall_time_s,
        "startup_time_s": info_provenance.get("startup_time_s"),
        "perf_regression_gate_info_present": perf_regression_gate_info_present,
        "raw_fps_mean": info_provenance.get("raw_fps_mean"),
        "raw_fps_std": info_provenance.get("raw_fps_std"),
        "raw_fps_min": info_provenance.get("raw_fps_min"),
        "raw_fps_max": info_provenance.get("raw_fps_max"),
        "raw_fps_median": info_provenance.get("raw_fps_median"),
        "raw_fps_p5": info_provenance.get("raw_fps_p5"),
        "raw_fps_p95": info_provenance.get("raw_fps_p95"),
        "outlier_count": None,
        "gpu_diag": info_provenance.get("gpu_diag"),
        "provenance": info_provenance.get("provenance"),
        "task_config_snapshot": {
            "task_id": args.task_id,
            "backend": backend_key,
            "physics_backend": physics_backend,
            "render_backend": render_backend,
            "backend_key": backend_key,
            "preset": preset,
            "num_envs": num_envs,
            "num_frames": num_frames,
            "excluded_frames_raw": excluded_frames_raw,
            "timeout_minutes": timeout_minutes,
            "tags": tags,
            "seed": seed,
        },
    }

    out = artifact_dir / "perf_regression_gate_result.json"
    out.write_text(json.dumps(bench_result, indent=2))

    status = f"failure_phase={failure_phase!r}, perf_regression_gate_info_present={perf_regression_gate_info_present}, exit_code={args.exit_code}"
    print(f"[build_bench_result] {args.task_id}: {status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
