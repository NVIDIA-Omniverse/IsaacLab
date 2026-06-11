# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deterministically demonstrate the gate's PASS vs BLOCK discrimination.

Waiting for a real regression to land is a poor demo. Instead this takes a *real*
benchmark result for a task and runs the *real* comparator CLI twice:

1. the unmodified result -> expected **PASS**;
2. a copy whose FPS has been scaled down by ``--factor`` (a synthetic but
   realistically-shaped slowdown) -> expected **REGRESSION** once the drop
   exceeds the task's block threshold.

Only the result is synthetic; the comparator, baseline, thresholds, and exit
codes are exactly what CI uses, so this exercises the production decision path.

Usage (uses a result already produced by the gate; no GPU needed)::

    python3 tools/perf_smoke/demo_regression.py \\
        --task Isaac-Cartpole-v0 --results-dir perf-output/Isaac-Cartpole-v0
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

import check_perf_regression as cpr  # noqa: E402

_FPS_METRIC = f"benchmark_non_rl runtime {cpr.METRIC_NAME}"


def _scale_result(data: object, factor: float) -> object:
    """Return a copy of a result with the FPS metric (and per-frame arrays) scaled.

    Handles both result shapes. Scaling <1 makes the run look slower.
    """
    data = copy.deepcopy(data)
    if isinstance(data, dict):  # OmniPerf shape
        runtime = data.get("runtime")
        if isinstance(runtime, dict) and cpr.METRIC_NAME in runtime:
            runtime[cpr.METRIC_NAME] = float(runtime[cpr.METRIC_NAME]) * factor
        return data
    if isinstance(data, list):  # json-backend shape
        for phase in data:
            if not isinstance(phase, dict) or phase.get("phase_name") != "runtime":
                continue
            for m in phase.get("measurements", []) or []:
                if not isinstance(m, dict):
                    continue
                if m.get("name") == _FPS_METRIC and isinstance(m.get("value"), (int, float)):
                    m["value"] = float(m["value"]) * factor
                # Scale the per-frame arrays: effective FPS (the gating metric reads
                # this) down, and step times up, so the slowdown is self-consistent.
                if m.get("name", "").endswith(cpr.FRAMETIMES_NAME) and isinstance(m.get("value"), dict):
                    eff = m["value"].get(cpr.EFF_FPS_ARRAY)
                    if isinstance(eff, list):
                        m["value"][cpr.EFF_FPS_ARRAY] = [
                            (v * factor if isinstance(v, (int, float)) else v) for v in eff
                        ]
                    steps = m["value"].get(cpr.STEP_MS_ARRAY)
                    if isinstance(steps, list):
                        m["value"][cpr.STEP_MS_ARRAY] = [
                            (s / factor if isinstance(s, (int, float)) else s) for s in steps
                        ]
    return data


def _run_comparator(
    task: str,
    results_dir: Path,
    baseline: str,
    gpu_override: str | None,
    history_dir: str | None,
    overrides: str | None,
) -> int:
    cmd = [
        sys.executable,
        str(_THIS_DIR / "check_perf_regression.py"),
        "--task",
        task,
        "--results-dir",
        str(results_dir),
        "--baseline",
        baseline,
        "--allow-multiple",
    ]
    if gpu_override:
        cmd += ["--gpu-override", gpu_override]
    if history_dir:
        cmd += ["--history-dir", history_dir]
    if overrides:
        cmd += ["--overrides", overrides]
    return subprocess.run(cmd).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--task", required=True, help="Task name (must have a real result + baseline entry).")
    parser.add_argument("--results-dir", required=True, help="Directory with a real benchmark result for the task.")
    parser.add_argument("--baseline", default=str(_THIS_DIR / "baseline.json"), help="Path to baseline.json.")
    parser.add_argument("--history-dir", default=str(_THIS_DIR / "perf_history"), help="Rolling-window store.")
    parser.add_argument("--overrides", default=str(_THIS_DIR / "baseline_overrides.json"), help="Overrides file.")
    parser.add_argument("--factor", type=float, default=0.7, help="FPS scale for the slowed copy (0.7 = 30%% slower).")
    parser.add_argument("--gpu-override", default=None, help="Force the baseline GPU key.")
    args = parser.parse_args(argv)

    pattern = cpr.DEFAULT_GLOB_TEMPLATE.format(task=args.task)
    try:
        real_path = cpr._resolve_results(args.results_dir, pattern, allow_multiple=True)
    except cpr.CompareError as e:
        print(f"[demo] no real result found: {e}")
        print(
            f"[demo] produce one first, e.g.:\n  ./isaaclab.sh -p tools/perf_smoke/run_perf_gate.py --tasks {args.task}"
        )
        return 2
    raw = cpr._read_json(real_path)

    print("\n[demo] === 1) real result -> expect PASS ===")
    rc_pass = _run_comparator(
        args.task, Path(args.results_dir), args.baseline, args.gpu_override, args.history_dir, args.overrides
    )

    slowed_dir = Path(args.results_dir).parent / f"{args.task}__demo_slowed"
    slowed_dir.mkdir(parents=True, exist_ok=True)
    slowed_path = slowed_dir / real_path.name
    slowed_path.write_text(json.dumps(_scale_result(raw, args.factor)), encoding="utf-8")

    pct = (1.0 - args.factor) * 100.0
    print(f"\n[demo] === 2) same result, {pct:.0f}% slower -> expect BLOCK ===")
    rc_block = _run_comparator(
        args.task, slowed_dir, args.baseline, args.gpu_override, args.history_dir, args.overrides
    )

    print("\n[demo] === SUMMARY ===")
    print(f"[demo] real   -> exit {rc_pass} ({cpr.EXIT_PASS}=PASS)")
    print(f"[demo] slowed -> exit {rc_block} ({cpr.EXIT_BLOCK}=BLOCK)")
    ok = rc_pass == cpr.EXIT_PASS and rc_block == cpr.EXIT_BLOCK
    print(f"[demo] discrimination {'OK' if ok else 'UNEXPECTED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
