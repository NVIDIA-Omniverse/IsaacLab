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
