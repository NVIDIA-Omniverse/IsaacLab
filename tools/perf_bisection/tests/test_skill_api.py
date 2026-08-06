# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GPU-free tests for the performance Skill automation API."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from isaaclab_bisection.contracts import BenchResult
from isaaclab_bisection.skill_api import build_harness_command, execute, main

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _summary(label: str, median_value: float, spread_pct: float = 1.0) -> dict:
    return {
        "label": label,
        "metric_name": "raw_fps_mean",
        "unit": "fps",
        "values": [median_value],
        "median_value": median_value,
        "mean_value": median_value,
        "min_value": median_value,
        "max_value": median_value,
        "sample_count": 1,
        "spread_pct": spread_pct,
    }


def test_build_benchmark_command_maps_nested_contract() -> None:
    payload = {
        "operation": "benchmark_commit",
        "repo_root": "/target/IsaacLab",
        "work_dir": "/tmp/benchmark",
        "commit": "candidate",
        "tooling_ref": "tooling",
        "task_id": "Task-v0",
        "backend_key": "newton",
        "runner": {
            "mode": "docker-reconstruct",
            "image": "image:tag",
            "trust_target_code": True,
            "extra_args": ["--install_scope", "newton,isaacsim"],
        },
        "task": {"num_envs": 64, "camera_resolution": [64, 48], "hydra_args": ["presets=newton"]},
        "metric": {"result_path": "resource_diag.ram_used_gb_peak", "regression_direction": "increase"},
        "measurement": {"runs": 2, "max_runs": 4, "warmup_runs": 1},
    }

    command = build_harness_command(payload)

    assert command[3] == "benchmark-commit"
    assert command[command.index("--repo_root") + 1] == "/target/IsaacLab"
    assert command[command.index("--commit") + 1] == "candidate"
    assert command[command.index("--runner_mode") + 1] == "docker-reconstruct"
    assert "--trust_target_code" in command
    assert "--runner_extra_arg=--install_scope" in command
    assert "--runner_extra_arg=newton,isaacsim" in command
    assert command[command.index("--camera_resolution") + 1 : command.index("--camera_resolution") + 3] == ["64", "48"]
    assert "--hydra_arg=presets=newton" in command
    assert command[command.index("--runs") + 1] == "2"


def test_threshold_check_reproduces_paired_reference_signal() -> None:
    payload = {
        "schema_version": 1,
        "operation": "threshold_check",
        "mode": "paired_reference",
        "comparison": "check_reference",
        "metric": {"name": "raw_fps_mean", "result_path": "raw_fps_mean", "regression_direction": "decrease"},
        "measurement_policy": {"min_regression_pct": 5.0, "reference_noise_multiplier": 2.0},
        "good_summary": _summary("good_ref", 100.0),
        "bad_summary": _summary("bad_ref", 90.0),
    }

    exit_code, output = execute(payload)

    assert exit_code == 0
    assert output["status"] == "completed"
    assert output["result"]["reproduced"] is True
    assert output["result"]["regression_pct"] == pytest.approx(10.0)


def test_threshold_check_calls_ci_oracle(tmp_path: Path) -> None:
    bench_result = BenchResult(
        task_id="Task-v0",
        backend="physx",
        physics_backend="physx",
        render_backend=None,
        backend_key="physx",
        preset="default",
        perf_smoke_test_info_present=True,
        raw_fps_mean=80.0,
    )
    result_path = tmp_path / "perf_smoke_test_result.json"
    result_path.write_text(json.dumps(bench_result.to_dict()), encoding="utf-8")
    payload = {
        "schema_version": 1,
        "operation": "threshold_check",
        "mode": "ci_gate",
        "bench_result_path": str(result_path),
        "baseline": {"median_fps": 100.0, "mad_fps": 1.0, "sample_count": 10},
    }

    exit_code, output = execute(payload)

    assert exit_code == 0
    assert output["result"]["verdict"] == "BLOCK"
    assert output["result"]["bisect_verdict"] == "BAD"


def test_benchmark_response_embeds_canonical_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    summary = {"succeeded": True, "summary": _summary("candidate", 100.0), "attempts": []}
    (tmp_path / "measurement_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    monkeypatch.setattr(
        "isaaclab_bisection.skill_api.subprocess.run", lambda *args, **kwargs: SimpleNamespace(returncode=0)
    )
    payload = {
        "schema_version": 1,
        "operation": "benchmark_commit",
        "work_dir": str(tmp_path),
        "commit": "candidate",
        "tooling_ref": "tooling",
        "task_id": "Task-v0",
        "backend_key": "physx",
    }

    exit_code, output = execute(payload)

    assert exit_code == 0
    assert output["status"] == "completed"
    assert output["result"] == summary
    assert output["artifacts"]["primary"] == str(tmp_path / "measurement_summary.json")


def test_main_writes_error_response_for_unknown_schema(tmp_path: Path) -> None:
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    input_path.write_text(json.dumps({"schema_version": 99, "operation": "threshold_check"}), encoding="utf-8")

    exit_code = main(["--input", str(input_path), "--output", str(output_path)])
    output = json.loads(output_path.read_text(encoding="utf-8"))

    assert exit_code == 2
    assert output["status"] == "error"
    assert "schema_version must be 1" in output["error"]


def test_skill_schemas_are_valid_json() -> None:
    schema_paths = sorted((_REPO_ROOT / "skills" / "developer").glob("perf-*/**/*.schema.json"))
    assert len(schema_paths) >= 6
    for path in schema_paths:
        assert isinstance(json.loads(path.read_text(encoding="utf-8")), dict)
