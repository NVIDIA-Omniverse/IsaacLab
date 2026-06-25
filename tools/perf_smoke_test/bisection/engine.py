# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bisection state machine for one IsaacLab perf regression."""

from __future__ import annotations

import shlex
import subprocess
import time
from pathlib import Path

from .git_utils import candidate_commits, resolve_ref
from .io import write_json
from .models import BisectionPlan, BisectionSummary, CandidateAttempt, CandidateEvaluation
from .oracle_adapter import evaluate_artifact
from .paired_reference import (
    MeasurementSummary,
    check_reference_signal,
    compare_candidate,
    fps_from_artifact,
    summarize_measurements,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _template_context(output_dir: Path, commit_sha: str, artifact_dir: Path, plan: BisectionPlan) -> dict[str, str]:
    """Return placeholder values supported by runner path/command templates."""
    return {
        "repo_root": str(_REPO_ROOT),
        "output_dir": str(output_dir),
        "commit_sha": commit_sha,
        "task_id": plan.task_id,
        "backend_key": plan.backend_key,
        "artifact_dir": str(artifact_dir),
    }


def _format_template(value: str | None, context: dict[str, str]) -> str | None:
    """Format an optional string template with the bisection context."""
    return value.format(**context) if value else None


def format_runner_command(
    plan: BisectionPlan, output_dir: Path, commit_sha: str, artifact_dir: Path
) -> list[str]:
    """Build the runner command for one candidate."""
    if plan.runner is None:
        raise ValueError("plan.runner is required to run a bisection candidate.")

    context = _template_context(output_dir, commit_sha, artifact_dir, plan)
    runner = plan.runner
    cmd = [
        str(_REPO_ROOT / "isaaclab.sh"),
        "-p",
        str(_REPO_ROOT / "tools" / "perf_smoke_test" / "bisect_single_commit_runner.py"),
        "--mode",
        runner.mode,
        "--commit",
        commit_sha,
        "--task_id",
        plan.task_id,
        "--backend_key",
        plan.backend_key,
        "--artifact_dir",
        str(artifact_dir),
        "--gpu_model",
        plan.gpu_model,
    ]
    optional_args = {
        "--image": _format_template(runner.image, context),
        "--source_dir": _format_template(runner.source_dir, context),
        "--jit_cache": _format_template(runner.jit_cache, context),
        "--kit_cache": _format_template(runner.kit_cache, context),
        "--local_env_dir": _format_template(runner.local_env_dir, context),
        "--ld_preload": _format_template(runner.ld_preload, context),
    }
    for flag, value in optional_args.items():
        if value:
            cmd.extend([flag, value])
    cmd.extend(_format_template(item, context) or "" for item in runner.extra_args)
    return cmd


def _command_display(command: list[str] | str) -> str:
    """Render a command for logs."""
    return command if isinstance(command, str) else shlex.join(command)


def _run_command(
    command: list[str] | str,
    *,
    command_log: Path,
    timeout_s: int | None,
) -> tuple[int | None, bool, float]:
    """Run a candidate command, streaming stdout/stderr into ``command_log``."""
    start = time.monotonic()
    shell = isinstance(command, str)
    with command_log.open("w", encoding="utf-8") as log_fh:
        log_fh.write(f"$ {_command_display(command)}\n\n")
        try:
            result = subprocess.run(
                command,
                shell=shell,
                cwd=_REPO_ROOT,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_s,
            )
            return result.returncode, False, time.monotonic() - start
        except subprocess.TimeoutExpired:
            log_fh.write(f"\n[bisection] candidate command timed out after {timeout_s}s\n")
            return None, True, time.monotonic() - start


def write_status(output_dir: Path, **values) -> None:
    """Write current harness status."""
    payload = {
        "phase": values.pop("phase", "running"),
        "status": values.pop("status", "running"),
        **values,
    }
    write_json(output_dir / "status.json", payload)


def build_candidates(plan: BisectionPlan) -> dict:
    """Resolve refs and build the candidate commit list."""
    good_sha = resolve_ref(_REPO_ROOT, plan.good_ref)
    bad_sha = resolve_ref(_REPO_ROOT, plan.bad_ref)
    commits = candidate_commits(_REPO_ROOT, good_sha, bad_sha)
    return {
        "good_sha": good_sha,
        "bad_sha": bad_sha,
        "candidate_count": len(commits),
        "candidates": commits,
    }


def run_candidate(plan: BisectionPlan, output_dir: Path, commit_sha: str) -> CandidateEvaluation:
    """Run and evaluate one candidate commit."""
    short = commit_sha[:12]
    candidate_dir = output_dir / "artifacts" / short / plan.task_id / plan.backend_key
    max_attempts = max(1, plan.retry.max_attempts)
    attempts: list[CandidateAttempt] = []
    last_evaluation: CandidateEvaluation | None = None

    for attempt_idx in range(1, max_attempts + 1):
        artifact_dir = candidate_dir / f"attempt_{attempt_idx}"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        command = format_runner_command(plan, output_dir, commit_sha, artifact_dir)
        command_display = _command_display(command)
        command_log = artifact_dir / "bisect_command.log"
        exit_code, timed_out, duration_s = _run_command(
            command,
            command_log=command_log,
            timeout_s=plan.timeout.candidate_timeout_s,
        )
        note = "candidate_timeout" if timed_out else None
        bench_result_path = artifact_dir / "perf_smoke_test_result.json"

        attempts.append(
            CandidateAttempt(
                attempt=attempt_idx,
                artifact_dir=str(artifact_dir),
                command=command_display,
                command_exit_code=exit_code,
                note=note,
                timed_out=timed_out,
                duration_s=duration_s,
            )
        )

        if timed_out:
            last_evaluation = CandidateEvaluation(
                commit_sha=commit_sha,
                bisect_verdict="SKIP",
                artifact_dir=str(artifact_dir),
                command_exit_code=exit_code,
                note="candidate_timeout",
                command=command_display,
                attempt_count=attempt_idx,
                attempts=[attempt.to_json() for attempt in attempts],
                timed_out=True,
                duration_s=duration_s,
                retry_reason="candidate_timeout" if attempt_idx < max_attempts else None,
                final_artifact_dir=str(artifact_dir),
            )
        elif exit_code != 0:
            last_evaluation = CandidateEvaluation(
                commit_sha=commit_sha,
                bisect_verdict="SKIP",
                artifact_dir=str(artifact_dir),
                command_exit_code=exit_code,
                note="runner_command_failed",
                command=command_display,
                attempt_count=attempt_idx,
                attempts=[attempt.to_json() for attempt in attempts],
                duration_s=duration_s,
                retry_reason="runner_command_failed" if attempt_idx < max_attempts else None,
                final_artifact_dir=str(artifact_dir),
            )
        elif not bench_result_path.exists():
            last_evaluation = CandidateEvaluation(
                commit_sha=commit_sha,
                bisect_verdict="SKIP",
                artifact_dir=str(artifact_dir),
                command_exit_code=exit_code,
                note="missing_perf_smoke_test_result",
                command=command_display,
                attempt_count=attempt_idx,
                attempts=[attempt.to_json() for attempt in attempts],
                duration_s=duration_s,
                retry_reason="missing_perf_smoke_test_result" if attempt_idx < max_attempts else None,
                final_artifact_dir=str(artifact_dir),
            )
        else:
            _, oracle_result = evaluate_artifact(
                artifact_dir=artifact_dir,
                task_id=plan.task_id,
                backend_key=plan.backend_key,
                gpu_model=plan.gpu_model,
                baselines_dir=Path(plan.baselines_dir),
                gate_config=Path(plan.gate_config),
            )
            last_evaluation = CandidateEvaluation(
                commit_sha=commit_sha,
                bisect_verdict=oracle_result.bisect_verdict,
                artifact_dir=str(artifact_dir),
                command_exit_code=exit_code,
                oracle_verdict=oracle_result.verdict.value,
                measured_fps=oracle_result.measured_fps,
                baseline_fps=oracle_result.baseline_fps,
                regression_pct=oracle_result.regression_pct,
                baseline_sample_count=oracle_result.baseline_sample_count,
                threshold_source=oracle_result.threshold_source,
                note=oracle_result.note,
                command=command_display,
                attempt_count=attempt_idx,
                attempts=[attempt.to_json() for attempt in attempts],
                duration_s=duration_s,
                final_artifact_dir=str(artifact_dir),
            )

        if last_evaluation.bisect_verdict in {"GOOD", "BAD"}:
            return last_evaluation
        if last_evaluation.note not in set(plan.retry.retryable_notes) or attempt_idx >= max_attempts:
            return last_evaluation
        if plan.retry.retry_delay_s:
            time.sleep(plan.retry.retry_delay_s)

    if last_evaluation is None:
        raise RuntimeError("candidate run did not produce an evaluation")
    return last_evaluation


def _measurement_artifact_dir(
    output_dir: Path, *, label: str, commit_sha: str, plan: BisectionPlan, run_idx: int
) -> Path:
    """Return the artifact directory for one local paired-reference measurement."""
    return output_dir / "measurements" / label / commit_sha[:12] / plan.task_id / plan.backend_key / f"run_{run_idx}"


def _run_single_measurement(
    plan: BisectionPlan, output_dir: Path, *, commit_sha: str, label: str, run_idx: int
) -> tuple[CandidateAttempt, float | None]:
    """Run one local measurement and return its attempt record and FPS value."""
    artifact_dir = _measurement_artifact_dir(output_dir, label=label, commit_sha=commit_sha, plan=plan, run_idx=run_idx)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    command = format_runner_command(plan, output_dir, commit_sha, artifact_dir)
    command_display = _command_display(command)
    command_log = artifact_dir / "bisect_command.log"
    exit_code, timed_out, duration_s = _run_command(
        command,
        command_log=command_log,
        timeout_s=plan.timeout.candidate_timeout_s,
    )
    bench_result_path = artifact_dir / "perf_smoke_test_result.json"
    note = None
    fps = None
    if timed_out:
        note = "candidate_timeout"
    elif exit_code != 0:
        note = "runner_command_failed"
    elif not bench_result_path.exists():
        note = "missing_perf_smoke_test_result"
    else:
        fps = fps_from_artifact(artifact_dir)

    return (
        CandidateAttempt(
            attempt=run_idx,
            artifact_dir=str(artifact_dir),
            command=command_display,
            command_exit_code=exit_code,
            note=note,
            timed_out=timed_out,
            duration_s=duration_s,
        ),
        fps,
    )


def _measure_reference_commit(
    plan: BisectionPlan, output_dir: Path, *, commit_sha: str, label: str
) -> tuple[MeasurementSummary | None, list[CandidateAttempt], str | None]:
    """Measure a reference commit until stable enough or capped."""
    min_runs = plan.measurement.reference_runs
    max_runs = max(min_runs, plan.measurement.max_reference_runs)
    attempts: list[CandidateAttempt] = []
    fps_values: list[float] = []
    summary: MeasurementSummary | None = None

    for run_idx in range(1, max_runs + 1):
        attempt, fps = _run_single_measurement(plan, output_dir, commit_sha=commit_sha, label=label, run_idx=run_idx)
        attempts.append(attempt)
        if fps is None:
            return None, attempts, attempt.note or "missing_fps_measurement"
        fps_values.append(fps)
        if run_idx < min_runs:
            continue
        summary = summarize_measurements(label, fps_values)
        if summary.spread_pct <= plan.measurement.max_reference_spread_pct:
            return summary, attempts, None

    return summary, attempts, None


def run_local_candidate(
    plan: BisectionPlan,
    output_dir: Path,
    commit_sha: str,
    good_summary: MeasurementSummary,
    *,
    reference_noise_pct: float,
) -> CandidateEvaluation:
    """Run and classify one candidate using local paired-reference measurements."""
    max_runs = max(plan.measurement.candidate_runs, plan.measurement.max_candidate_runs)
    attempts: list[CandidateAttempt] = []
    fps_values: list[float] = []
    last_comparison = None

    for run_idx in range(1, max_runs + 1):
        attempt, fps = _run_single_measurement(
            plan, output_dir, commit_sha=commit_sha, label="candidate", run_idx=run_idx
        )
        attempts.append(attempt)
        if fps is None:
            return CandidateEvaluation(
                commit_sha=commit_sha,
                bisect_verdict="SKIP",
                artifact_dir=attempt.artifact_dir,
                command_exit_code=attempt.command_exit_code,
                note=attempt.note,
                command=attempt.command,
                attempt_count=run_idx,
                attempts=[item.to_json() for item in attempts],
                timed_out=attempt.timed_out,
                duration_s=attempt.duration_s,
                retry_reason=attempt.note if run_idx < max_runs else None,
                final_artifact_dir=attempt.artifact_dir,
                comparison_mode="paired_reference",
            )
        fps_values.append(fps)
        if run_idx < plan.measurement.candidate_runs:
            continue

        candidate_summary = summarize_measurements("candidate", fps_values)
        last_comparison = compare_candidate(
            candidate_summary, good_summary, plan.measurement, reference_noise_pct=reference_noise_pct
        )
        if last_comparison.verdict != "UNCLEAR":
            break

    if last_comparison is None:
        candidate_summary = summarize_measurements("candidate", fps_values)
        last_comparison = compare_candidate(
            candidate_summary, good_summary, plan.measurement, reference_noise_pct=reference_noise_pct
        )

    final_attempt = attempts[-1]
    return CandidateEvaluation(
        commit_sha=commit_sha,
        bisect_verdict=last_comparison.verdict,
        artifact_dir=final_attempt.artifact_dir,
        command_exit_code=final_attempt.command_exit_code,
        measured_fps=last_comparison.measured_fps,
        baseline_fps=last_comparison.baseline_fps,
        regression_pct=last_comparison.regression_pct,
        baseline_sample_count=good_summary.sample_count,
        threshold_source=last_comparison.threshold_source,
        comparison_mode="paired_reference",
        note=last_comparison.note,
        command=final_attempt.command,
        attempt_count=len(attempts),
        attempts=[item.to_json() for item in attempts],
        timed_out=final_attempt.timed_out,
        duration_s=sum(float(item.duration_s or 0.0) for item in attempts),
        final_artifact_dir=final_attempt.artifact_dir,
    )


def run_local_bisection(plan: BisectionPlan, output_dir: Path, *, max_tests: int = 50) -> BisectionSummary:
    """Run local paired-reference bisection and write status/result artifacts."""
    candidate_payload = build_candidates(plan)
    candidates = list(candidate_payload["candidates"])
    write_json(output_dir / "candidates.json", candidate_payload)
    good_sha = candidate_payload["good_sha"]
    bad_sha = candidate_payload["bad_sha"]

    write_status(output_dir, phase="preflight", status="running", current_commit=good_sha)
    good_summary, good_attempts, good_note = _measure_reference_commit(
        plan,
        output_dir,
        commit_sha=good_sha,
        label="good_ref",
    )
    if good_summary is None:
        summary = BisectionSummary(
            status="inconclusive",
            reason=f"good_ref_measurement_failed:{good_note}",
            tested_commits=[],
            suspected_first_bad_commit=None,
            last_good_commit=None,
            good_ref=good_sha,
            bad_ref=bad_sha,
            comparison_mode="paired_reference",
            reference_stats={"good_attempts": [attempt.to_json() for attempt in good_attempts]},
        )
        write_json(output_dir / "summary.json", summary.to_json())
        return summary

    write_status(output_dir, phase="preflight", status="running", current_commit=bad_sha)
    bad_summary, bad_attempts, bad_note = _measure_reference_commit(
        plan,
        output_dir,
        commit_sha=bad_sha,
        label="bad_ref",
    )
    reference_stats = {
        "good": good_summary.to_json(),
        "bad": bad_summary.to_json() if bad_summary is not None else None,
        "good_attempts": [attempt.to_json() for attempt in good_attempts],
        "bad_attempts": [attempt.to_json() for attempt in bad_attempts],
    }
    if bad_summary is None:
        summary = BisectionSummary(
            status="inconclusive",
            reason=f"bad_ref_measurement_failed:{bad_note}",
            tested_commits=[],
            suspected_first_bad_commit=None,
            last_good_commit=None,
            good_ref=good_sha,
            bad_ref=bad_sha,
            comparison_mode="paired_reference",
            reference_stats=reference_stats,
        )
        write_json(output_dir / "summary.json", summary.to_json())
        return summary

    reference_check = check_reference_signal(good_summary, bad_summary, plan.measurement)
    reference_stats["check"] = reference_check.to_json()
    write_json(output_dir / "reference_measurements.json", reference_stats)
    if not reference_check.reproduced:
        summary = BisectionSummary(
            status="inconclusive",
            reason=reference_check.note or "local_regression_not_reproduced",
            tested_commits=[],
            suspected_first_bad_commit=None,
            last_good_commit=good_sha,
            good_ref=good_sha,
            bad_ref=bad_sha,
            comparison_mode="paired_reference",
            reference_stats=reference_stats,
        )
        write_json(output_dir / "summary.json", summary.to_json())
        return summary

    low = 0
    high = len(candidates) - 1
    tested: list[str] = []
    last_good: str | None = good_sha
    best_bad: str | None = None
    result_dir = output_dir / "results"

    while low <= high:
        if len(tested) >= max_tests:
            summary = BisectionSummary(
                status="inconclusive",
                reason="max_tests_reached",
                tested_commits=tested,
                suspected_first_bad_commit=best_bad,
                last_good_commit=last_good,
                good_ref=good_sha,
                bad_ref=bad_sha,
                comparison_mode="paired_reference",
                reference_stats=reference_stats,
            )
            write_json(output_dir / "summary.json", summary.to_json())
            return summary

        idx = (low + high) // 2
        commit_sha = candidates[idx]
        tested.append(commit_sha)
        write_status(
            output_dir,
            phase="running",
            status="running",
            total_candidates=len(candidates),
            completed_tests=len(tested) - 1,
            current_commit=commit_sha,
            last_good_commit=last_good,
            current_best_bad_commit=best_bad,
            search_low=low,
            search_high=high,
            comparison_mode="paired_reference",
        )

        evaluation = run_local_candidate(
            plan,
            output_dir,
            commit_sha,
            good_summary,
            reference_noise_pct=float(reference_check.reference_noise_pct or 0.0),
        )
        write_json(result_dir / f"{commit_sha[:12]}.json", evaluation.to_json())

        if evaluation.bisect_verdict == "GOOD":
            last_good = commit_sha
            low = idx + 1
        elif evaluation.bisect_verdict == "BAD":
            best_bad = commit_sha
            high = idx - 1
        else:
            summary = BisectionSummary(
                status="inconclusive",
                reason=f"candidate_returned_{evaluation.bisect_verdict}",
                tested_commits=tested,
                suspected_first_bad_commit=best_bad,
                last_good_commit=last_good,
                good_ref=good_sha,
                bad_ref=bad_sha,
                comparison_mode="paired_reference",
                reference_stats=reference_stats,
            )
            write_json(output_dir / "summary.json", summary.to_json())
            return summary

    summary = BisectionSummary(
        status="completed",
        reason="first_bad_found" if best_bad else "no_bad_commit_found",
        tested_commits=tested,
        suspected_first_bad_commit=best_bad,
        last_good_commit=last_good,
        good_ref=good_sha,
        bad_ref=bad_sha,
        comparison_mode="paired_reference",
        reference_stats=reference_stats,
    )
    write_json(output_dir / "summary.json", summary.to_json())
    return summary


def run_bisection(plan: BisectionPlan, output_dir: Path, *, max_tests: int = 50) -> BisectionSummary:
    """Run midpoint bisection and write status/result artifacts."""
    candidate_payload = build_candidates(plan)
    candidates = list(candidate_payload["candidates"])
    write_json(output_dir / "candidates.json", candidate_payload)

    low = 0
    high = len(candidates) - 1
    tested: list[str] = []
    last_good: str | None = candidate_payload["good_sha"]
    best_bad: str | None = None
    result_dir = output_dir / "results"

    while low <= high:
        if len(tested) >= max_tests:
            summary = BisectionSummary(
                status="inconclusive",
                reason="max_tests_reached",
                tested_commits=tested,
                suspected_first_bad_commit=best_bad,
                last_good_commit=last_good,
                good_ref=candidate_payload["good_sha"],
                bad_ref=candidate_payload["bad_sha"],
            )
            write_json(output_dir / "summary.json", summary.to_json())
            return summary

        idx = (low + high) // 2
        commit_sha = candidates[idx]
        tested.append(commit_sha)
        write_status(
            output_dir,
            phase="running",
            status="running",
            total_candidates=len(candidates),
            completed_tests=len(tested) - 1,
            current_commit=commit_sha,
            last_good_commit=last_good,
            current_best_bad_commit=best_bad,
            search_low=low,
            search_high=high,
        )

        evaluation = run_candidate(plan, output_dir, commit_sha)
        write_json(result_dir / f"{commit_sha[:12]}.json", evaluation.to_json())

        if evaluation.bisect_verdict == "GOOD":
            last_good = commit_sha
            low = idx + 1
        elif evaluation.bisect_verdict == "BAD":
            best_bad = commit_sha
            high = idx - 1
        else:
            summary = BisectionSummary(
                status="inconclusive",
                reason=f"candidate_returned_{evaluation.bisect_verdict}",
                tested_commits=tested,
                suspected_first_bad_commit=best_bad,
                last_good_commit=last_good,
                good_ref=candidate_payload["good_sha"],
                bad_ref=candidate_payload["bad_sha"],
            )
            write_json(output_dir / "summary.json", summary.to_json())
            return summary

    summary = BisectionSummary(
        status="completed",
        reason="first_bad_found" if best_bad else "no_bad_commit_found",
        tested_commits=tested,
        suspected_first_bad_commit=best_bad,
        last_good_commit=last_good,
        good_ref=candidate_payload["good_sha"],
        bad_ref=candidate_payload["bad_sha"],
    )
    write_json(output_dir / "summary.json", summary.to_json())
    return summary
