# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CI aggregate script: load per-task perf_regression_gate_result.json, run oracle, update baselines.

Scans the local (downloaded) artifacts directory for perf_regression_gate_result.json files, runs the
oracle for each, writes a GitHub Step Summary table, and optionally updates the baselines branch.

Usage::

    python3 tools/perf_regression_gate/aggregate.py \\
        --artifacts_dir artifacts/ \\
        --gpu_model L40S \\
        --gate_config tools/perf_regression_gate/gate_config.json \\
        --baseline_branch angehu/perf-baselines \\
        --allow_baseline_update true \\
        --summary_file "$GITHUB_STEP_SUMMARY"

For offline/test use, pass ``--baselines_dir`` to read/write flat files instead
of the git baselines branch::

    python3 tools/perf_regression_gate/aggregate.py \\
        --artifacts_dir artifacts/ \\
        --gpu_model L40S \\
        --baselines_dir local_baselines/
"""

import argparse
import json
import os
import sys
from pathlib import Path

_MODULE_DIR = Path(__file__).parent
_TOOLS_DIR = _MODULE_DIR.parent
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))
DEFAULT_BASELINE_BRANCH = "angehu/perf-baselines"  # TODO: replace w/ real branch name

from oracle import OracleVerdict, compare  # noqa: E402
from baseline_manager import load_baseline, load_baseline_git, update_baseline, update_baseline_git  # noqa: E402
from gate_config import load_gate_config  # noqa: E402
from task_config import get_task  # noqa: E402


def _parse_args():
    p = argparse.ArgumentParser(description="Aggregate bench results and run oracle.")
    p.add_argument(
        "--artifacts_dir",
        required=True,
        type=Path, 
        help="Root directory containing per-task artifact subdirectories",
    )
    p.add_argument("--gpu_model", default="L40S")
    p.add_argument("--gate_config", type=Path, default=_MODULE_DIR / "gate_config.json")
    p.add_argument(
        "--baseline_branch",
        default=f"{DEFAULT_BASELINE_BRANCH}",
        help=f"Git branch for baseline storage (default: {DEFAULT_BASELINE_BRANCH})",
    )
    p.add_argument(
        "--baselines_dir",
        type=Path,
        default=None,
        help="Flat-file baseline directory; bypasses git (use for offline testing)",
    )
    p.add_argument(
        "--allow_baseline_update",
        default="false",
        help="Update baselines for PASS/WARN results ('true'/'false', default: false)",
    )
    p.add_argument(
        "--summary_file",
        default=None,
        help="Append step-summary markdown to this path (set to $GITHUB_STEP_SUMMARY in CI)",
    )
    return p.parse_args()


def _find_bench_results(artifacts_dir: Path) -> list[tuple[Path, dict]]:
    """Return list of (artifact_dir: Path, perf_regression_gate_result: dict) sorted by task_id."""
    found = []
    for p in sorted(artifacts_dir.rglob("perf_regression_gate_result.json")):
        with p.open() as fh:
            bench_result = json.load(fh)
        found.append((p.parent, bench_result))
    return found


def _excluded_frames(bench_result: dict) -> frozenset:
    """Expand excluded_frames_raw from the task_config_snapshot into a frozenset."""
    raw = (bench_result.get("task_config_snapshot") or {}).get("excluded_frames_raw", [])
    indices: set[int] = set()
    for entry in raw:
        if isinstance(entry, list):
            indices.update(range(entry[0], entry[1] + 1))
        else:
            indices.add(int(entry))
    return frozenset(indices)


def _fmt(v, decimals: int = 1) -> str:
    return f"{v:.{decimals}f}" if v is not None else "N/A"


def _build_summary_table(rows: list) -> str:
    lines = [
        "| Task | Backend | Verdict | FPS (mean) | Baseline | Regression% | Wall (s) |",
        "|------|---------|---------|------------|----------|-------------|----------|",
    ]
    for r in rows:
        lines.append(
            f"| {r.task_id} | {r.backend} | {r.verdict.value}"
            f" | {_fmt(r.measured_fps)} | {_fmt(r.baseline_fps)}"
            f" | {_fmt(r.regression_pct, 2)} | {_fmt(r.wall_time_s, 1)} |"
        )
    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    use_flat = args.baselines_dir is not None
    allow_update = args.allow_baseline_update.strip().lower() in ("true", "1", "yes")

    gate_config = load_gate_config(args.gate_config)
    blocking = gate_config.get("blocking", False)

    items = _find_bench_results(args.artifacts_dir)
    if not items:
        print(f"[aggregate] No perf_regression_gate_result.json files found under {args.artifacts_dir}")
        return 1

    oracle_results = []
    has_block = False
    has_hard_failure = False
    baselines_updated = False

    for artifact_dir, bench_result in items:
        task_id = bench_result["task_id"]
        backend = bench_result.get("backend_key")

        # Load rolling baseline (None on seed passes/failed load)
        baseline = None
        try:
            if use_flat:
                baseline = load_baseline(args.baselines_dir, args.gpu_model, task_id, backend)
            else:
                baseline = load_baseline_git(args.baseline_branch, args.gpu_model, task_id, backend, None)
        except Exception as exc:
            print(f"[aggregate] Warning: baseline load failed for {task_id}/{backend}: {exc}")

        # Hard-floor FPS for this task (0.0 = disabled)
        try:
            task = get_task(task_id, backend)
            fps_mean_floor = task.fps_mean_floor.get(args.gpu_model, {}).get(backend, 0.0)
        except Exception:
            fps_mean_floor = 0.0

        oracle_result = compare(
            bench_result=bench_result,
            baseline=baseline,
            fps_mean_floor=fps_mean_floor,
            excluded_frames=_excluded_frames(bench_result),
            artifact_dir=artifact_dir,
        )
        oracle_results.append(oracle_result)

        print(
            f"[aggregate] {task_id}/{backend}: {oracle_result.verdict.value}"
            f"  fps={_fmt(oracle_result.measured_fps)}  baseline={_fmt(oracle_result.baseline_fps)}"
        )

        if oracle_result.verdict == OracleVerdict.BLOCK:
            has_block = True
        elif oracle_result.verdict == OracleVerdict.HARD_FAILURE:
            has_hard_failure = True

        # Update baseline only for measured PASS/WARN results (never for BLOCK/HARD_FAILURE) pushed
        # to designated branches or force update with flag for offline/testing use.
        if (
            allow_update
            and oracle_result.verdict in (OracleVerdict.PASS, OracleVerdict.WARN)
            and oracle_result.measured_fps is not None
        ):
            try:
                if use_flat:
                    update_baseline(args.baselines_dir, args.gpu_model, task_id, backend, oracle_result.measured_fps)
                else:
                    update_baseline_git(
                        args.baseline_branch,
                        args.gpu_model,
                        task_id,
                        backend,
                        oracle_result.measured_fps,
                        None,
                    )
                baselines_updated = True
                print(f"[aggregate]   -> baseline updated: {oracle_result.measured_fps:.1f} FPS")
            except Exception as exc:
                print(f"[aggregate] Warning: baseline update failed for {task_id}/{backend}: {exc}")

    table = _build_summary_table(oracle_results)
    print("\n## Performance Gate Results\n")
    print(table)
    print()

    if args.summary_file:
        with open(args.summary_file, "a") as fh:
            fh.write("\n## Performance Gate Results\n\n")
            fh.write(table)
            fh.write("\n")

    # Signal baseline push to the calling workflow step
    if baselines_updated:
        github_output = os.environ.get("GITHUB_OUTPUT", "")
        if github_output:
            with open(github_output, "a") as fh:
                fh.write("baselines_updated=true\n")
        print(f"[aggregate] Baselines updated; workflow will push {DEFAULT_BASELINE_BRANCH}") 

    if blocking:  # from gate_config.py, explicit PR to make gate blocking
        if has_block:
            return 1
        if has_hard_failure:
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
