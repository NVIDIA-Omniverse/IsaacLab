# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Public single-commit measurement API shared by local tools and bisection."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .io import append_jsonl, read_json, write_json
from .models import BisectionPlan, CandidateAttempt
from .paired_reference import MeasurementSummary, summarize_measurements
from .preflight import PreflightReport, run_preflight
from .probe import ProbePolicy
from .recovery import RecoveryPolicy

MeasureWithRecovery = Callable[..., tuple[CandidateAttempt, float | None]]


@dataclass(frozen=True)
class CommitMeasurementResult:
    """Result of measuring one commit one or more times."""

    summary: MeasurementSummary | None
    attempts: list[CandidateAttempt]
    note: str | None = None

    @property
    def succeeded(self) -> bool:
        """Return whether a canonical metric summary was produced."""
        return self.summary is not None

    def to_json(self) -> dict:
        """Serialize the standalone measurement result."""
        return {
            "succeeded": self.succeeded,
            "summary": self.summary.to_json() if self.summary is not None else None,
            "attempts": [attempt.to_json() for attempt in self.attempts],
            "note": self.note,
        }


def write_measurement_preflight(output_dir: Path, plan: BisectionPlan) -> PreflightReport | None:
    """Record host readiness and advisory hardware trust for a measurement."""
    runner = plan.runner
    try:
        report = run_preflight(
            runner_mode=runner.mode if runner else "synthetic",
            image=runner.image if runner else None,
            expected_gpu_model=plan.gpu_model,
            work_dir=output_dir,
        )
    except Exception as exc:  # noqa: BLE001 - preflight must remain advisory
        write_json(output_dir / "preflight.json", {"error": f"preflight failed: {exc}"})
        return None
    write_json(output_dir / "preflight.json", report.to_json())
    if report.gpu_model_matches is True:
        hardware_trust = "target_match"
    elif report.gpu_model_matches is False:
        hardware_trust = "local_mismatch"
    else:
        hardware_trust = "unknown"
    write_json(
        output_dir / "hardware_context.json",
        {
            "hardware_trust": hardware_trust,
            "expected_gpu_model": plan.gpu_model,
            "detected_gpu": report.gpu.to_json(),
            "gpu_model_matches": report.gpu_model_matches,
            "advisory_only": True,
        },
    )
    for warning in report.warnings:
        append_jsonl(output_dir / "audit_log.jsonl", {"event": "preflight_warning", "warning": warning})
    return report


def _default_measure_with_recovery() -> MeasureWithRecovery:
    """Resolve the existing recovery substrate lazily to avoid an import cycle."""
    from .engine import _measure_with_recovery

    return _measure_with_recovery


def run_warmups(
    plan: BisectionPlan,
    output_dir: Path,
    *,
    commit_sha: str,
    label: str,
    policy: RecoveryPolicy,
    probe_policy: ProbePolicy | None = None,
    measure_with_recovery: MeasureWithRecovery | None = None,
) -> CandidateAttempt | None:
    """Run one process warmup per commit and return a failed warmup, if any."""
    if plan.measurement.warmup_runs <= 0:
        return None

    tooling_hash = plan.tooling.tooling_spec_hash if plan.tooling is not None else "unversioned"
    task_hash = plan.tooling.task_config_hash if plan.tooling is not None else f"{plan.task_id}:{plan.backend_key}"
    ledger_key = f"{commit_sha}:{tooling_hash}:{task_hash}"
    ledger_path = output_dir / "warmup_state.json"
    try:
        ledger = read_json(ledger_path) if ledger_path.exists() else {}
    except (OSError, TypeError, ValueError):
        ledger = {}
    completed = ledger.get("completed", {})
    if isinstance(completed, dict) and ledger_key in completed:
        append_jsonl(
            output_dir / "audit_log.jsonl",
            {
                "event": "warmup_reused",
                "commit_sha": commit_sha,
                "label": label,
                "excluded_from_stats": True,
                "warmup": completed[ledger_key],
            },
        )
        return None

    measure = measure_with_recovery or _default_measure_with_recovery()
    attempt, metric_value = measure(
        plan,
        output_dir,
        commit_sha=commit_sha,
        label=f"{label}_warmup",
        run_idx=1,
        policy=policy,
        probe_policy=probe_policy,
    )
    event = {
        "event": "warmup_measurement",
        "commit_sha": commit_sha,
        "label": label,
        "run_idx": 1,
        "note": attempt.note,
        "metric": metric_value,
        "excluded_from_stats": True,
        "artifact_dir": attempt.artifact_dir,
    }
    append_jsonl(output_dir / "audit_log.jsonl", event)
    if metric_value is None:
        return attempt

    stack_hash = None
    env_path = Path(attempt.artifact_dir) / "bisect_env.json"
    if env_path.exists():
        try:
            stack_hash = read_json(env_path).get("stack_hash")
        except (OSError, TypeError, ValueError):
            stack_hash = None
    if not isinstance(completed, dict):
        completed = {}
    completed[ledger_key] = {
        "commit_sha": commit_sha,
        "stack_hash": stack_hash,
        "tooling_spec_hash": tooling_hash,
        "task_config_hash": task_hash,
        "artifact_dir": attempt.artifact_dir,
    }
    write_json(ledger_path, {"schema_version": 1, "completed": completed})
    return None


def measure_commit(
    plan: BisectionPlan,
    output_dir: Path,
    *,
    commit_sha: str,
    label: str = "benchmark",
    policy: RecoveryPolicy,
    probe_policy: ProbePolicy | None = None,
    min_runs: int | None = None,
    max_runs: int | None = None,
    measure_with_recovery: MeasureWithRecovery | None = None,
) -> CommitMeasurementResult:
    """Measure one commit under a stable policy and summarize its metric.

    The same operation backs the standalone ``benchmark-commit`` workflow and
    good/bad reference measurement in full bisection.
    """
    measure = measure_with_recovery or _default_measure_with_recovery()
    warmup_failure = run_warmups(
        plan,
        output_dir,
        commit_sha=commit_sha,
        label=label,
        policy=policy,
        probe_policy=probe_policy,
        measure_with_recovery=measure,
    )
    if warmup_failure is not None:
        return CommitMeasurementResult(
            summary=None,
            attempts=[warmup_failure],
            note=warmup_failure.note or "warmup_failed",
        )
    required_runs = plan.measurement.reference_runs if min_runs is None else max(1, min_runs)
    capped_runs = plan.measurement.max_reference_runs if max_runs is None else max_runs
    capped_runs = max(required_runs, capped_runs)
    attempts: list[CandidateAttempt] = []
    metric_values: list[float] = []
    summary: MeasurementSummary | None = None

    for run_idx in range(1, capped_runs + 1):
        attempt, metric_value = measure(
            plan,
            output_dir,
            commit_sha=commit_sha,
            label=label,
            run_idx=run_idx,
            policy=policy,
            probe_policy=probe_policy,
        )
        attempts.append(attempt)
        if metric_value is None:
            return CommitMeasurementResult(
                summary=None,
                attempts=attempts,
                note=attempt.note or "missing_metric_measurement",
            )
        metric_values.append(metric_value)
        if run_idx < required_runs:
            continue
        summary = summarize_measurements(label, plan.metric, metric_values)
        if summary.spread_pct <= plan.measurement.max_reference_spread_pct:
            return CommitMeasurementResult(summary=summary, attempts=attempts)

    return CommitMeasurementResult(summary=summary, attempts=attempts)
