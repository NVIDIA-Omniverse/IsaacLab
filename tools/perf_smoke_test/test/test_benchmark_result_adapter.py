# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for canonical RuntimeBundle resource projection."""

from __future__ import annotations

import sys
from pathlib import Path

_GATE_DIR = Path(__file__).resolve().parents[1]
if str(_GATE_DIR) not in sys.path:
    sys.path.insert(0, str(_GATE_DIR))

from benchmark_result_adapter import project_runtime  # noqa: E402
from bisection.models import MetricSpec  # noqa: E402
from bisection.paired_reference import metric_from_result  # noqa: E402
from contracts import BenchResult  # noqa: E402


def _runtime_bundle() -> dict:
    return {
        "schema_version": "1.0",
        "run": {"task": "Task-v0", "num_envs": 4, "seed": 1, "config": {}},
        "runtime": {
            "total_fps": {"mean": 100.0, "std": 2.0, "peak": 110.0},
            "iteration_time_s": {"peak": 0.05},
            "steps_per_iteration": 4,
            "startup_time_s": {},
        },
        "resources": {
            "cpu_util_pct": {"mean": 42.5, "std": 3.0, "peak": None},
            "ram_gb": {"mean": 12.0, "std": 0.5, "peak": 13.25},
        },
        "versions": {},
        "hardware": {},
    }


def test_runtime_bundle_projects_cpu_and_ram_diagnostics() -> None:
    sample = project_runtime(_runtime_bundle())

    assert sample is not None
    assert sample.runtime_resources["cpu_util_pct"] == 42.5
    assert sample.runtime_resources["system_ram_used_mb"] == 12288.0
    assert sample.runtime_resources["system_ram_peak_mb"] == 13568.0


def test_resource_diagnostics_round_trip_and_support_metric_selection() -> None:
    result = BenchResult(
        task_id="Task-v0",
        backend="physx",
        physics_backend="physx",
        render_backend=None,
        backend_key="physx",
        preset="default",
        runtime_resources={"system_ram_peak_mb": 13568.0},
    )
    restored = BenchResult.from_dict(result.to_dict())

    assert restored.runtime_resources == {"system_ram_peak_mb": 13568.0}
    metric = MetricSpec(
        name="peak_ram",
        result_path="runtime_resources.system_ram_peak_mb",
        regression_direction="increase",
        unit="MB",
    )
    assert metric_from_result(restored.to_dict(), metric) == 13568.0
