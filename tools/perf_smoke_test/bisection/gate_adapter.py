# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Create bisection plans from existing perf-smoke artifacts."""

from __future__ import annotations

from pathlib import Path

from .models import BisectionPlan, RetryPolicy, RunnerSpec, TimeoutPolicy
from .oracle_adapter import evaluate_artifact, load_bench_result


def find_regressed_gate_cells(
    *,
    artifacts_dir: Path,
    gpu_model: str,
    baselines_dir: Path,
    gate_config: Path,
) -> list[dict]:
    """Return regressed cells from a phase-1 gate artifact directory."""
    rows: list[dict] = []
    for result_path in sorted(artifacts_dir.rglob("perf_smoke_test_result.json")):
        artifact_dir = result_path.parent
        bench_result = load_bench_result(artifact_dir)
        task_id = str(bench_result["task_id"])
        backend_key = str(bench_result.get("backend_key") or bench_result.get("backend"))
        try:
            _, oracle_result = evaluate_artifact(
                artifact_dir=artifact_dir,
                task_id=task_id,
                backend_key=backend_key,
                gpu_model=gpu_model,
                baselines_dir=baselines_dir,
                gate_config=gate_config,
            )
        except Exception as exc:
            rows.append(
                {
                    "task_id": task_id,
                    "backend_key": backend_key,
                    "artifact_dir": str(artifact_dir),
                    "oracle_verdict": "EVALUATION_ERROR",
                    "bisect_verdict": "SKIP",
                    "note": str(exc),
                }
            )
            continue
        if oracle_result.bisect_verdict != "BAD":
            continue
        rows.append(
            {
                "task_id": task_id,
                "backend_key": backend_key,
                "artifact_dir": str(artifact_dir),
                "oracle_verdict": oracle_result.verdict.value,
                "bisect_verdict": oracle_result.bisect_verdict,
                "measured_fps": oracle_result.measured_fps,
                "baseline_fps": oracle_result.baseline_fps,
                "regression_pct": oracle_result.regression_pct,
                "baseline_sample_count": oracle_result.baseline_sample_count,
                "threshold_source": oracle_result.threshold_source,
                "note": oracle_result.note,
            }
        )

    def sort_key(row: dict):
        regression_pct = row.get("regression_pct")
        return (0 if regression_pct is not None else 1, regression_pct if regression_pct is not None else 0)

    return sorted(rows, key=sort_key)


def make_plan_from_gate_cell(
    *,
    gate_cell: dict,
    good_ref: str,
    bad_ref: str,
    runner_command: str = "",
    runner: RunnerSpec | None = None,
    timeout: TimeoutPolicy | None = None,
    retry: RetryPolicy | None = None,
    gpu_model: str,
    baselines_dir: Path,
    gate_config: Path,
) -> BisectionPlan:
    """Build a bisection plan from one regressed gate cell."""
    return BisectionPlan(
        task_id=str(gate_cell["task_id"]),
        backend_key=str(gate_cell["backend_key"]),
        good_ref=good_ref,
        bad_ref=bad_ref,
        runner_command=runner_command,
        gpu_model=gpu_model,
        baselines_dir=str(baselines_dir),
        gate_config=str(gate_config),
        runner=runner,
        timeout=timeout or TimeoutPolicy(),
        retry=retry or RetryPolicy(),
        source_gate_artifact_dir=str(gate_cell["artifact_dir"]),
        source_gate_result=gate_cell,
    )
