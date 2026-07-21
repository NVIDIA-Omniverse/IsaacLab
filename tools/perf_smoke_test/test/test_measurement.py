# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GPU-free tests for the reusable single-commit measurement API."""

from __future__ import annotations

import sys
from pathlib import Path

_GATE_DIR = Path(__file__).resolve().parents[1]
if str(_GATE_DIR) not in sys.path:
    sys.path.insert(0, str(_GATE_DIR))

from bisection.measurement import measure_commit  # noqa: E402
from bisection.models import (  # noqa: E402
    BisectionPlan,
    CandidateAttempt,
    MeasurementPolicy,
    MetricSpec,
    RunnerSpec,
)
from bisection.recovery import NoRecoveryPolicy  # noqa: E402


def _plan(*, warmup_runs: int = 0) -> BisectionPlan:
    return BisectionPlan(
        task_id="Isaac-Test-v0",
        backend_key="newton",
        good_ref="abc",
        bad_ref="abc",
        gpu_model="unknown-gpu",
        runner=RunnerSpec(mode="synthetic"),
        metric=MetricSpec(),
        measurement=MeasurementPolicy(
            reference_runs=2,
            max_reference_runs=3,
            max_reference_spread_pct=10.0,
            warmup_runs=warmup_runs,
        ),
    )


def test_measure_commit_summarizes_shared_measurement_primitive(tmp_path: Path) -> None:
    calls: list[str] = []

    def fake_measure(plan, output_dir, *, commit_sha, label, run_idx, policy, probe_policy):
        calls.append(label)
        return (
            CandidateAttempt(
                attempt=run_idx,
                artifact_dir=str(output_dir / label / str(run_idx)),
                command="fake",
                command_exit_code=0,
            ),
            100.0 + run_idx,
        )

    result = measure_commit(
        _plan(),
        tmp_path,
        commit_sha="abc",
        policy=NoRecoveryPolicy(),
        measure_with_recovery=fake_measure,
    )
    assert result.succeeded
    assert result.summary is not None
    assert result.summary.sample_count == 2
    assert calls == ["benchmark", "benchmark"]


def test_measure_commit_records_but_excludes_warmup(tmp_path: Path) -> None:
    values = iter([1.0, 101.0, 102.0])
    calls: list[str] = []

    def fake_measure(plan, output_dir, *, commit_sha, label, run_idx, policy, probe_policy):
        calls.append(label)
        value = next(values)
        return CandidateAttempt(attempt=run_idx, artifact_dir=str(tmp_path), command="fake"), value

    result = measure_commit(
        _plan(warmup_runs=1),
        tmp_path,
        commit_sha="abc",
        policy=NoRecoveryPolicy(),
        measure_with_recovery=fake_measure,
    )
    assert result.summary is not None
    assert result.summary.median_value == 101.5
    assert calls == ["benchmark_warmup", "benchmark", "benchmark"]
    assert "excluded_from_stats" in (tmp_path / "audit_log.jsonl").read_text(encoding="utf-8")


def test_measure_commit_returns_skip_note_for_missing_metric(tmp_path: Path) -> None:
    def fake_measure(plan, output_dir, *, commit_sha, label, run_idx, policy, probe_policy):
        return CandidateAttempt(attempt=run_idx, artifact_dir=str(tmp_path), command="fake", note="env_skip:x"), None

    result = measure_commit(
        _plan(),
        tmp_path,
        commit_sha="abc",
        policy=NoRecoveryPolicy(),
        measure_with_recovery=fake_measure,
    )
    assert not result.succeeded
    assert result.note == "env_skip:x"
