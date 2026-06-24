# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Adapter from perf-smoke artifacts to bisection verdicts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_GATE_DIR = Path(__file__).resolve().parents[1]
if str(_GATE_DIR) not in sys.path:
    sys.path.insert(0, str(_GATE_DIR))

from aggregate import _bench_gpu_model, _excluded_frames, _hard_floor  # noqa: E402
from baseline_manager import load_baseline, match_context_from_bench_result  # noqa: E402
from gate_config import load_gate_config  # noqa: E402
from oracle import OracleResult, compare  # noqa: E402


def load_bench_result(artifact_dir: Path) -> dict:
    """Load ``perf_smoke_test_result.json`` from an artifact directory."""
    result_path = artifact_dir / "perf_smoke_test_result.json"
    with result_path.open(encoding="utf-8") as fh:
        return json.load(fh)


def evaluate_artifact(
    *,
    artifact_dir: Path,
    task_id: str,
    backend_key: str,
    gpu_model: str,
    baselines_dir: Path,
    gate_config: Path,
) -> tuple[dict, OracleResult]:
    """Evaluate a perf-smoke artifact directory with the existing oracle."""
    bench_result = load_bench_result(artifact_dir)
    config = load_gate_config(gate_config)
    bench_gpu_model = _bench_gpu_model(bench_result, gpu_model)
    match_context = match_context_from_bench_result(bench_result, gpu_model=bench_gpu_model)
    baseline = load_baseline(
        baselines_dir,
        bench_gpu_model,
        task_id,
        backend_key,
        match_context=match_context,
    )
    result = compare(
        bench_result=bench_result,
        baseline=baseline,
        fps_mean_floor=_hard_floor(bench_result, bench_gpu_model, backend_key),
        excluded_frames=_excluded_frames(bench_result),
        artifact_dir=artifact_dir,
        min_block_regression_pct=float(config.get("min_block_regression_pct", 3.0)),
    )
    return bench_result, result
