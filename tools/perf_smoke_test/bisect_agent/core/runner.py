"""Execution primitive for the bisect agent.

The ONLY module that triggers benchmark execution.  Both grounding and bisect
phases call :func:`run_commit` — nothing else may spawn containers or benchmark
scripts directly.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup: allow importing perf_smoke_test modules when needed by callers.
# (runner.py itself uses subprocess only, but callers may import from the tree)
# ---------------------------------------------------------------------------
# bisect_agent/core/ -> bisect_agent/ -> perf_smoke_test/
_PERF_SMOKE_TEST_DIR = Path(__file__).resolve().parent.parent.parent
if str(_PERF_SMOKE_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_PERF_SMOKE_TEST_DIR))

_STUB_BENCHMARK_PATH = _PERF_SMOKE_TEST_DIR / "dev" / "stub_benchmark.py"
_BUILD_BENCH_RESULT_PATH = _PERF_SMOKE_TEST_DIR / "build_bench_result.py"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _write_minimal_launch_config(output_dir: Path, task_id: str, backend: str) -> None:
    """Write a minimal launch_config.json that build_bench_result.py can consume.

    build_bench_result.py will fall back to tasks.json defaults for any field
    not present here.  The only fields that must be present for the backend
    identity resolution to work are ``task_id``, ``physics_backend``, and
    ``backend_key``.
    """
    # backend may be a compound key like "newton" or "physx_rtx".
    # We use the backend string as-is for physics_backend; build_bench_result
    # normalises it internally via normalize_physics_backend().
    launch_config = {
        "schema_version": 1,
        "task_id": task_id,
        "physics_backend": backend,
        "render_backend": None,
        "backend_key": backend,
        "backend": backend,
        "num_envs": 1,
        "num_frames": 200,
        "seed": 42,
        "excluded_frames_raw": [],
        "timeout_minutes": 10,
        "preset": "default",
        "tags": ["always"],
        "fps_mean_floor": 0.0,
        "baseline_epoch": 1,
        "benchmark_backend": "json",
        "camera_resolution": None,
        "hydra_args": [],
        "gpu_model": None,
        "gpu_model_raw": None,
    }
    config_path = output_dir / "launch_config.json"
    config_path.write_text(json.dumps(launch_config, indent=2))


def _extract_run_result(
    sha: str,
    task_id: str,
    backend: str,
    output_dir: Path,
    exit_code: int,
    wall_time: float,
    run_index: int = 0,
) -> dict:
    """Read perf_smoke_test_result.json and assemble the run_result dict.

    Fields map directly from perf_smoke_test_result.json.  Any field absent in
    the result (e.g. when the run failed) is set to None.
    """
    result_path = output_dir / "perf_smoke_test_result.json"

    bench: dict = {}
    if result_path.exists():
        try:
            bench = json.loads(result_path.read_text())
        except Exception:
            pass

    # gpu_mem_used_mb lives inside gpu_diag in perf_smoke_test_result.json
    gpu_diag: dict = bench.get("gpu_diag") or {}
    gpu_mem_used_mb: float | None = gpu_diag.get("gpu_mem_used_mb")

    run_result: dict = {
        "sha": sha,
        "task_id": task_id,
        "backend": backend,
        "run_index": run_index,
        "exit_code": exit_code,
        "wall_time_s": bench.get("wall_time_s", wall_time),
        "failure_phase": bench.get("failure_phase"),
        "raw_fps_mean": bench.get("raw_fps_mean"),
        "raw_fps_median": bench.get("raw_fps_median"),
        "raw_fps_p5": bench.get("raw_fps_p5"),
        "raw_fps_p95": bench.get("raw_fps_p95"),
        "gpu_mem_used_mb": gpu_mem_used_mb,
        "artifact_dir": str(output_dir),
    }
    return run_result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_commit(
    sha: str,
    task_id: str,
    backend: str,
    output_dir: Path,
    *,
    dev_mode: bool = False,
    dev_perf_map: dict | None = None,
    run_index: int = 0,
) -> dict:
    """Run a single benchmark for *sha* and return a run_result dict.

    Parameters
    ----------
    sha:
        Full commit SHA being tested.
    task_id:
        IsaacLab task identifier (e.g. ``Isaac-Cartpole-Direct``).
    backend:
        Backend key (e.g. ``newton``).
    output_dir:
        Directory where artifacts for this run are stored.  Created if absent.
    dev_mode:
        When True, calls stub_benchmark.py via subprocess instead of a real
        benchmark.  No Docker or GPU required.
    dev_perf_map:
        Mapping of ``{sha: fps_mean}`` used in dev mode.  If *sha* is not
        present, defaults to 200.0.
    run_index:
        0-based index of this run within its batch (grounding or bisect).

    Returns
    -------
    dict
        A run_result dict matching ``schemas/run_result.schema.json``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Cache: if the run_result already exists, return it immediately.
    # ------------------------------------------------------------------
    cache_path = output_dir / "run_result.json"
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
            # Patch run_index in case the cached file was written with a
            # different index (e.g. re-used output dir).
            cached.setdefault("run_index", run_index)
            return cached
        except Exception:
            pass  # corrupted cache — proceed to re-run

    if dev_mode:
        result = _run_dev(sha, task_id, backend, output_dir, dev_perf_map, run_index)
    else:
        result = _run_production(sha, task_id, backend, output_dir, run_index)

    # Persist for resume / cache
    cache_path.write_text(json.dumps(result, indent=2))
    return result


# ---------------------------------------------------------------------------
# Dev-mode execution
# ---------------------------------------------------------------------------

def _run_dev(
    sha: str,
    task_id: str,
    backend: str,
    output_dir: Path,
    dev_perf_map: dict | None,
    run_index: int,
) -> dict:
    """Run stub_benchmark.py then build_bench_result.py in dev mode."""
    fps_mean: float = 200.0
    if dev_perf_map is not None:
        fps_mean = float(dev_perf_map.get(sha, dev_perf_map.get(sha[:7], 200.0)))

    # ------------------------------------------------------------------
    # Phase 1: run stub_benchmark.py
    # ------------------------------------------------------------------
    log_path = output_dir / "benchmark.log"
    stub_env = dict(os.environ)
    stub_env["STUB_FPS_MEAN"] = str(fps_mean)

    stub_cmd = [
        sys.executable,
        str(_STUB_BENCHMARK_PATH),
        "--task_id", task_id,
        "--backend", backend,
        "--out_dir", str(output_dir),
        "--num_frames", "200",
        "--seed", "42",
    ]

    t0 = time.monotonic()
    stub_proc = subprocess.run(
        stub_cmd,
        env=stub_env,
        capture_output=True,
        text=True,
    )
    wall_time = time.monotonic() - t0

    # Write combined log so build_bench_result can classify failure phase
    combined_log = stub_proc.stdout + stub_proc.stderr
    log_path.write_text(combined_log)

    # ------------------------------------------------------------------
    # Phase 2: write launch_config.json then call build_bench_result.py
    # ------------------------------------------------------------------
    _write_minimal_launch_config(output_dir, task_id, backend)

    # Split backend into physics/render for the --physics_backend arg.
    # Simple heuristic: if backend contains "_" we split; otherwise treat
    # the whole string as physics_backend with no render backend.
    if "_" in backend:
        physics_backend, render_backend = backend.split("_", 1)
    else:
        physics_backend = backend
        render_backend = ""

    build_cmd = [
        sys.executable,
        str(_BUILD_BENCH_RESULT_PATH),
        "--task_id", task_id,
        "--physics_backend", physics_backend,
        "--render_backend", render_backend,
        "--artifact_dir", str(output_dir),
        "--exit_code", str(stub_proc.returncode),
        "--wall_time_s", str(round(wall_time, 3)),
        "--timeout_s", "600",
        "--log_file", str(log_path),
        "--launch_config", str(output_dir / "launch_config.json"),
    ]

    build_proc = subprocess.run(
        build_cmd,
        capture_output=True,
        text=True,
        cwd=str(_PERF_SMOKE_TEST_DIR),
    )

    if build_proc.returncode != 0:
        # Non-fatal: we may still get a partial result; log and continue.
        print(
            f"[runner] build_bench_result.py exited {build_proc.returncode} "
            f"for sha={sha[:7]}:\n{build_proc.stderr[-2000:]}",
            file=sys.stderr,
        )

    return _extract_run_result(
        sha=sha,
        task_id=task_id,
        backend=backend,
        output_dir=output_dir,
        exit_code=stub_proc.returncode,
        wall_time=wall_time,
        run_index=run_index,
    )


# ---------------------------------------------------------------------------
# Production-mode execution
# ---------------------------------------------------------------------------

def _run_production(
    sha: str,
    task_id: str,
    backend: str,
    output_dir: Path,
    run_index: int,
) -> dict:
    """Delegate to infra.container.run_in_container for production runs."""
    try:
        from infra.container import run_in_container  # type: ignore[import]
    except ImportError:
        from bisect_agent.infra.container import run_in_container  # type: ignore[import]

    isaaclab_repo_path = _PERF_SMOKE_TEST_DIR.parent.parent

    t0 = time.monotonic()
    container_result = run_in_container(
        sha=sha,
        task_id=task_id,
        backend=backend,
        output_dir=output_dir,
        isaaclab_repo_path=isaaclab_repo_path,
    )
    wall_time = time.monotonic() - t0

    exit_code: int = container_result.get("exit_code", -1)
    artifact_dir = Path(container_result.get("artifact_dir", str(output_dir)))

    return _extract_run_result(
        sha=sha,
        task_id=task_id,
        backend=backend,
        output_dir=artifact_dir,
        exit_code=exit_code,
        wall_time=wall_time,
        run_index=run_index,
    )
