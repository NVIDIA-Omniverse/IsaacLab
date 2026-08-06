# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""JSON automation API for the reusable performance Skills."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from .bisection.models import MeasurementPolicy, MetricSpec
from .bisection.paired_reference import (
    MeasurementSummary,
    check_reference_signal,
    compare_candidate,
)
from .contracts import BenchResult
from .gate_types import FpsMeanThreshold
from .oracle import Baseline, compare

SCHEMA_VERSION = 1
_ARTIFACTS = {
    "plan_resolved": "plan.resolved.json",
    "tooling_manifest": "tooling_manifest.json",
    "preflight": "preflight.json",
    "hardware_context": "hardware_context.json",
    "relaunch": "relaunch.json",
    "status": "status.json",
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _object(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{key} must be a JSON object")
    return value


def _required(payload: dict[str, Any], key: str) -> Any:
    value = payload.get(key)
    if value is None or value == "":
        raise ValueError(f"missing required field: {key}")
    return value


def _append_value(command: list[str], flag: str, value: Any) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def _append_repeated(command: list[str], flag: str, values: Any) -> None:
    if values is None:
        return
    if not isinstance(values, list):
        raise TypeError(f"{flag} values must be a list")
    for value in values:
        command.append(f"{flag}={value}")


def _common_harness_flags(payload: dict[str, Any]) -> list[str]:
    runner = _object(payload, "runner")
    task = _object(payload, "task")
    metric = _object(payload, "metric")
    command = ["--work_dir", str(_required(payload, "work_dir"))]
    _append_value(command, "--repo_root", payload.get("repo_root"))

    for flag, key in (
        ("--tooling_ref", "tooling_ref"),
        ("--task_id", "task_id"),
        ("--backend_key", "backend_key"),
        ("--plan", "plan"),
    ):
        _append_value(command, flag, payload.get(key))

    for flag, key in (
        ("--runner_mode", "mode"),
        ("--gpu_model", "gpu_model"),
        ("--image", "image"),
        ("--source_dir", "source_dir"),
        ("--jit_cache", "jit_cache"),
        ("--kit_cache", "kit_cache"),
        ("--local_env_dir", "local_env_dir"),
        ("--ld_preload", "ld_preload"),
    ):
        _append_value(command, flag, runner.get(key))
    _append_repeated(command, "--runner_extra_arg", runner.get("extra_args"))
    if runner.get("trust_target_code") is True:
        command.append("--trust_target_code")

    for flag, key in (
        ("--num_envs", "num_envs"),
        ("--num_frames", "num_frames"),
        ("--warmup_frames", "warmup_frames"),
        ("--seed", "seed"),
        ("--timeout_minutes", "timeout_minutes"),
    ):
        _append_value(command, flag, task.get(key))
    camera_resolution = task.get("camera_resolution")
    if camera_resolution is not None:
        if not isinstance(camera_resolution, list) or len(camera_resolution) != 2:
            raise ValueError("task.camera_resolution must contain [width, height]")
        command.extend(["--camera_resolution", str(camera_resolution[0]), str(camera_resolution[1])])
    _append_repeated(command, "--hydra_arg", task.get("hydra_args"))

    for flag, key in (
        ("--metric_name", "name"),
        ("--metric_path", "result_path"),
        ("--regression_direction", "regression_direction"),
        ("--metric_unit", "unit"),
    ):
        _append_value(command, flag, metric.get(key))
    return command


def build_harness_command(payload: dict[str, Any]) -> list[str]:
    """Build the existing harness command for a benchmark or bisection Skill request."""
    operation = payload.get("operation")
    if operation == "benchmark_commit":
        command = [sys.executable, "-m", "isaaclab_bisection.cli", "benchmark-commit"]
        command.extend(["--commit", str(_required(payload, "commit"))])
        command.extend(_common_harness_flags(payload))
        measurement = _object(payload, "measurement")
        for flag, key in (
            ("--runs", "runs"),
            ("--max_runs", "max_runs"),
            ("--warmup_runs", "warmup_runs"),
            ("--max_reference_spread_pct", "max_reference_spread_pct"),
            ("--candidate_timeout_s", "candidate_timeout_s"),
        ):
            _append_value(command, flag, measurement.get(key))
        return command

    if operation == "bisect_range":
        command = [sys.executable, "-m", "isaaclab_bisection.cli", "bisect-range"]
        command.extend(_common_harness_flags(payload))
        if payload.get("plan") is None:
            command.extend(["--good_ref", str(_required(payload, "good_ref"))])
            command.extend(["--bad_ref", str(_required(payload, "bad_ref"))])
        measurement = _object(payload, "measurement")
        for flag, key in (
            ("--reference_runs", "reference_runs"),
            ("--max_reference_runs", "max_reference_runs"),
            ("--candidate_runs", "candidate_runs"),
            ("--max_candidate_runs", "max_candidate_runs"),
            ("--warmup_runs", "warmup_runs"),
            ("--min_regression_pct", "min_regression_pct"),
            ("--gray_zone_pct", "gray_zone_pct"),
            ("--reference_noise_multiplier", "reference_noise_multiplier"),
            ("--max_reference_spread_pct", "max_reference_spread_pct"),
            ("--candidate_timeout_s", "candidate_timeout_s"),
            ("--max_tests", "max_tests"),
        ):
            _append_value(command, flag, measurement.get(key))
        return command

    raise ValueError(f"unsupported harness operation: {operation!r}")


def _artifact_paths(work_dir: Path, primary_name: str) -> dict[str, str]:
    names = {**_ARTIFACTS, "primary": primary_name}
    return {key: str(work_dir / name) for key, name in names.items() if (work_dir / name).exists()}


def _run_harness(payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    operation = str(payload["operation"])
    work_dir = Path(str(_required(payload, "work_dir"))).resolve()
    command = build_harness_command(payload)
    repo_root = Path(str(payload.get("repo_root", "."))).resolve()
    completed = subprocess.run(command, cwd=repo_root, check=False)
    primary_name = "measurement_summary.json" if operation == "benchmark_commit" else "summary.json"
    primary_path = work_dir / primary_name
    result = _read_json(primary_path) if primary_path.exists() else None
    if operation == "benchmark_commit":
        status = "completed" if result and result.get("succeeded") else "inconclusive"
    else:
        status = str(result.get("status", "inconclusive")) if result else "inconclusive"
    output = {
        "schema_version": SCHEMA_VERSION,
        "operation": operation,
        "status": status,
        "exit_code": completed.returncode,
        "result": result,
        "artifacts": _artifact_paths(work_dir, primary_name),
    }
    return completed.returncode, output


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _measurement_summary(raw: Any, field_name: str) -> MeasurementSummary:
    if not isinstance(raw, dict):
        raise TypeError(f"{field_name} must be a MeasurementSummary object")
    return MeasurementSummary(**raw)


def _run_threshold_check(payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    mode = str(_required(payload, "mode"))
    if mode == "ci_gate":
        bench_result_path = Path(str(_required(payload, "bench_result_path")))
        bench_result = BenchResult.from_dict(_read_json(bench_result_path))
        baseline_raw = payload.get("baseline")
        baseline = Baseline(**baseline_raw) if isinstance(baseline_raw, dict) else None
        threshold_raw = payload.get("fps_mean_thresholds", [])
        if not isinstance(threshold_raw, list):
            raise TypeError("fps_mean_thresholds must be a list")
        thresholds = [
            parsed
            for index, item in enumerate(threshold_raw)
            if (parsed := FpsMeanThreshold.from_dict(item, context=f"fps_mean_thresholds[{index}]")) is not None
        ]
        result = compare(
            bench_result,
            baseline,
            thresholds,
            min_block_regression_pct=float(payload.get("min_block_regression_pct", 5.0)),
            noise_floor_pct=float(payload.get("noise_floor_pct", 0.0)),
        )
    elif mode == "paired_reference":
        comparison = str(_required(payload, "comparison"))
        metric = MetricSpec.from_json(_object(payload, "metric"))
        policy = MeasurementPolicy.from_json(_object(payload, "measurement_policy"))
        good = _measurement_summary(payload.get("good_summary"), "good_summary")
        if comparison == "check_reference":
            bad = _measurement_summary(payload.get("bad_summary"), "bad_summary")
            result = check_reference_signal(good, bad, metric, policy)
        elif comparison == "compare_candidate":
            candidate = _measurement_summary(payload.get("candidate_summary"), "candidate_summary")
            result = compare_candidate(
                candidate,
                good,
                metric,
                policy,
                reference_noise_pct=float(payload.get("reference_noise_pct", 0.0)),
            )
        else:
            raise ValueError(f"unsupported paired-reference comparison: {comparison!r}")
    else:
        raise ValueError(f"unsupported threshold mode: {mode!r}")

    return 0, {
        "schema_version": SCHEMA_VERSION,
        "operation": "threshold_check",
        "status": "completed",
        "exit_code": 0,
        "mode": mode,
        "result": _jsonable(result),
    }


def execute(payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """Execute one versioned Skill request and return its process code and output envelope."""
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    operation = payload.get("operation")
    if operation in {"benchmark_commit", "bisect_range"}:
        return _run_harness(payload)
    if operation == "threshold_check":
        return _run_threshold_check(payload)
    raise ValueError(f"unsupported operation: {operation!r}")


def main(argv: list[str] | None = None) -> int:
    """Run one JSON Skill request."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Skill input JSON.")
    parser.add_argument("--output", required=True, type=Path, help="Skill output JSON.")
    args = parser.parse_args(argv)

    try:
        exit_code, output = execute(_read_json(args.input))
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        exit_code = 2
        output = {
            "schema_version": SCHEMA_VERSION,
            "operation": None,
            "status": "error",
            "exit_code": exit_code,
            "error": f"{type(error).__name__}: {error}",
        }
    _write_json(args.output, output)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
