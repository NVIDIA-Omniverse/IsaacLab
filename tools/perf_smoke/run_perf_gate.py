# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 1 perf-smoke gate runner.

Orchestrates the gate for one or more tasks:

1. For each task, read its run config (``num_envs``, ``num_frames``, ``seed``,
   optional ``benchmark_args``) from ``baseline.json`` so the gate and the
   baseline can never drift apart -- a measurement is always taken under the same
   config the baseline was calibrated with.
2. Launch ``scripts/benchmarks/benchmark_non_rl.py`` for that task as its own
   subprocess (Isaac Sim must own the process) with ``--benchmark_backend json``
   (the format the baselines were calibrated from), retrying once on a
   launch/crash failure.
3. Run the pure-logic comparator (``check_perf_regression.py``) against the
   produced result JSON and map its exit code to a per-task verdict.
4. Aggregate: the gate exits non-zero if *any* task is REGRESSION or
   HARD_FAILURE.

Optional warm cache (``--cache-dir``)
-------------------------------------
The dominant cold-start cost is Newton/Warp JIT compilation. Pointing
``WARP_CACHE_PATH`` / ``CUDA_CACHE_PATH`` at a persistent directory turns the
second run on a host into a warm run. ``--cache-dir DIR`` does exactly that; it
is the local stand-in for the S3 "sidecar" (see ``CACHING_SIDECAR.md``) -- in CI
the same directory would be restored from / saved to object storage around the
run. It is purely additive: omit it and the gate runs cold, which only shifts
FPS by ~0-3% and never changes the verdict.

This is deliberately a standalone script, not a pytest module: ``tools/conftest.py``
disables pytest collection under ``tools/`` and runs each Isaac Sim entry point as
its own process. The gate follows that same model.

Run it via the Isaac Lab launcher so the benchmark subprocess inherits the env::

    ./isaaclab.sh -p tools/perf_smoke/run_perf_gate.py --tasks Isaac-Cartpole-v0

The runner itself only needs the standard library, so ``--dry-run`` (which prints
the commands without launching anything) works under any Python.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Verdict exit codes mirror check_perf_regression.py so the gate's own exit code
# is meaningful when a single task is run.
EXIT_PASS = 0
EXIT_BLOCK = 1
EXIT_HARD_FAILURE = 2

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[1]
_COMPARATOR = _THIS_DIR / "check_perf_regression.py"
_BENCHMARK = _REPO_ROOT / "scripts" / "benchmarks" / "benchmark_non_rl.py"
_LAUNCHER = _REPO_ROOT / "isaaclab.sh"
_DEFAULT_HISTORY = _THIS_DIR / "perf_history"
_DEFAULT_OVERRIDES = _THIS_DIR / "baseline_overrides.json"

_VERDICT_NAME = {EXIT_PASS: "PASS", EXIT_BLOCK: "BLOCK", EXIT_HARD_FAILURE: "HARD_FAILURE"}


def _load_baseline(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _task_run_config(baseline: dict, task: str) -> dict:
    """Pull the launch config for ``task`` out of the baseline document.

    Returns a dict with ``num_envs``, ``num_frames``, ``seed`` and an optional
    ``benchmark_args`` list. Falls back to the calibration defaults
    (``num_frames=300``, ``seed=42``) when a field is absent.
    """
    entry = baseline.get(task)
    if not isinstance(entry, dict):
        raise KeyError(f"task {task!r} not present in baseline")
    # The gate key may be a variant like "Isaac-Cartpole-v0@newton"; ``task_id`` is the
    # real gym task the benchmark is launched with (the part before "@" by default).
    return {
        "task_id": entry.get("task_id", task.split("@", 1)[0]),
        "num_envs": entry.get("num_envs"),
        "num_frames": entry.get("num_frames", 300),
        "seed": entry.get("seed", 42),
        "benchmark_args": entry.get("benchmark_args", []),
    }


def _benchmark_cmd(task: str, cfg: dict, task_out: Path) -> list[str]:
    """Build the benchmark subprocess command for a task.

    The benchmark is always launched with the real gym id (``cfg['task_id']``), so a
    variant gate key like ``Isaac-Cartpole-v0@newton`` still runs the right task.
    """
    task_id = cfg.get("task_id", task)
    cmd = [str(_LAUNCHER), "-p", str(_BENCHMARK), "--task", task_id, "--headless"]
    if cfg.get("num_envs") is not None:
        cmd += ["--num_envs", str(cfg["num_envs"])]
    cmd += ["--num_frames", str(cfg["num_frames"]), "--seed", str(cfg["seed"])]
    # json backend: same shape the baselines were calibrated from; carries the
    # per-frame step-time array the comparator uses for advisory debug KPIs.
    cmd += ["--benchmark_backend", "json"]
    cmd += list(cfg.get("benchmark_args", []))
    cmd += ["--output_path", str(task_out)]
    return cmd


def _cache_env(cache_dir: str | None) -> dict[str, str] | None:
    """Build the JIT-cache env overlay for a warm run, or ``None`` to run cold.

    Persisting ``WARP_CACHE_PATH`` / ``CUDA_CACHE_PATH`` across runs is the whole
    of the sidecar mechanism: the first run populates the dir (cold), later runs
    reuse it (warm). In CI this dir is what gets restored from / saved to S3.
    """
    if not cache_dir:
        return None
    root = Path(cache_dir).resolve()
    warp_dir = root / "warp"
    cuda_dir = root / "nv"
    warp_dir.mkdir(parents=True, exist_ok=True)
    cuda_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["WARP_CACHE_PATH"] = str(warp_dir)
    env["CUDA_CACHE_PATH"] = str(cuda_dir)
    return env


def _run_benchmark(
    task: str, cfg: dict, task_out: Path, retries: int, dry_run: bool, cache_dir: str | None
) -> float | None:
    """Launch the benchmark for a task, retrying once on failure.

    Returns the wall-clock seconds of the successful run, or ``None`` if every
    attempt failed.
    """
    cmd = _benchmark_cmd(task, cfg, task_out)
    print(f"[gate] benchmark cmd: {' '.join(cmd)}", flush=True)
    if cache_dir:
        print(f"[gate] {task}: warm-cache dir = {Path(cache_dir).resolve()}", flush=True)
    if dry_run:
        return 0.0
    env = _cache_env(cache_dir)
    task_out.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 2):  # 1 initial + `retries` extra
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=str(_REPO_ROOT), env=env)
        dt = time.time() - t0
        if proc.returncode == 0:
            print(f"[gate] {task}: benchmark ok in {dt:.0f}s (attempt {attempt})", flush=True)
            return dt
        print(
            f"[gate] {task}: benchmark FAILED rc={proc.returncode} in {dt:.0f}s (attempt {attempt})",
            flush=True,
        )
    return None


def _run_comparator(
    task: str,
    results_dir: Path,
    baseline_path: Path,
    gpu_override: str | None,
    history_dir: str | None = None,
    overrides_path: Path | None = None,
    wall_s: float | None = None,
    fingerprint: str | None = None,
    task_id: str | None = None,
) -> tuple[int, str]:
    """Run the pure-logic comparator. Returns ``(exit_code, result_label)``.

    ``result_label`` is the ``RESULT=...`` token parsed from the comparator's
    output (PASS / WARN / BLOCK); it distinguishes an advisory WARN from a clean
    PASS, both of which exit 0. The rolling-window store (``history_dir``), the
    in-tree overrides, the measured wall-clock and the history fingerprint are
    forwarded so the comparator applies the doc's median+MAD test logic.
    """
    cmd = [
        sys.executable,
        str(_COMPARATOR),
        "--task",
        task,
        "--results-dir",
        str(results_dir),
        "--baseline",
        str(baseline_path),
    ]
    if task_id and task_id != task:
        cmd += ["--task-id", task_id]
    if gpu_override:
        cmd += ["--gpu-override", gpu_override]
    if history_dir:
        cmd += ["--history-dir", str(history_dir)]
    if overrides_path:
        cmd += ["--overrides", str(overrides_path)]
    if wall_s is not None:
        cmd += ["--measured-wall-s", str(wall_s)]
    if fingerprint:
        cmd += ["--fingerprint", fingerprint]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    print(out.rstrip(), flush=True)
    label = _VERDICT_NAME.get(proc.returncode, str(proc.returncode))
    for tok in out.split():
        if tok.startswith("RESULT="):
            label = tok.split("=", 1)[1]
            break
    return proc.returncode, label


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--tasks", nargs="+", required=True, help="Gate task name(s).")
    parser.add_argument("--baseline", default=str(_THIS_DIR / "baseline.json"), help="Path to baseline.json.")
    parser.add_argument("--output-dir", default=str(_REPO_ROOT / "perf-output"), help="Root output directory.")
    parser.add_argument("--retries", type=int, default=1, help="Extra benchmark retries on failure (default 1).")
    parser.add_argument("--gpu-override", default=None, help="Force the baseline GPU key (e.g. 'NVIDIA L40S').")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Persist Warp/CUDA JIT caches here (warm runs). Local stand-in for the S3 sidecar; "
        "omit to run cold. Purely additive -- never changes the verdict.",
    )
    parser.add_argument(
        "--history-dir",
        default=str(_DEFAULT_HISTORY),
        help="Rolling-window store (orphan-branch checkout / local stand-in). Empty disables it.",
    )
    parser.add_argument("--overrides", default=str(_DEFAULT_OVERRIDES), help="Path to baseline_overrides.json.")
    parser.add_argument("--fingerprint", default=None, help="History bucket key (git-subtree+deps hash).")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without launching anything.")
    args = parser.parse_args(argv)

    baseline_path = Path(args.baseline).resolve()
    out_root = Path(args.output_dir).resolve()
    baseline = _load_baseline(baseline_path)

    verdicts: dict[str, int] = {}
    labels: dict[str, str] = {}
    walls: dict[str, float | None] = {}
    for task in args.tasks:
        print(f"\n[gate] === {task} ===", flush=True)
        try:
            cfg = _task_run_config(baseline, task)
        except KeyError as e:
            print(f"[gate] {task}: {e} -> HARD_FAILURE", flush=True)
            verdicts[task] = EXIT_HARD_FAILURE
            labels[task] = "HARD_FAILURE"
            continue

        task_out = out_root / task
        wall = _run_benchmark(task, cfg, task_out, args.retries, args.dry_run, args.cache_dir)
        walls[task] = wall
        if args.dry_run:
            verdicts[task] = EXIT_PASS
            labels[task] = "DRY_RUN"
            continue
        if wall is None:
            # Benchmark could not produce a result after retries -> structural failure.
            print(f"[gate] {task}: benchmark unrunnable after retries -> HARD_FAILURE", flush=True)
            verdicts[task] = EXIT_HARD_FAILURE
            labels[task] = "HARD_FAILURE"
            continue

        code, label = _run_comparator(
            task,
            task_out,
            baseline_path,
            args.gpu_override,
            history_dir=args.history_dir or None,
            overrides_path=Path(args.overrides) if args.overrides else None,
            wall_s=wall,
            fingerprint=args.fingerprint,
            task_id=cfg.get("task_id"),
        )
        verdicts[task] = code
        labels[task] = label

    # Aggregate: worst verdict wins (HARD_FAILURE > BLOCK > PASS/WARN).
    print("\n[gate] === SUMMARY ===", flush=True)
    worst = EXIT_PASS
    for task, code in verdicts.items():
        wall = walls.get(task)
        wall_str = f"  wall={wall:.0f}s" if isinstance(wall, float) and wall > 0 else ""
        print(f"[gate] {task}: {labels.get(task, _VERDICT_NAME.get(code, code))}{wall_str}", flush=True)
        worst = max(worst, code)
    print(f"[gate] OVERALL: {_VERDICT_NAME.get(worst, worst)}", flush=True)
    return worst


if __name__ == "__main__":
    sys.exit(main())
