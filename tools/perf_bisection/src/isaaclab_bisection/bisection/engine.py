# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bisection state machine for one IsaacLab perf regression."""

from __future__ import annotations

import json
import os
import selectors
import shlex
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import replace
from functools import partial
from pathlib import Path

from ..artifact_security import scan_artifacts
from .artifacts import finalize_run_artifacts, write_attempt_summary
from .base_image_repair import docker_build_command, write_repair_dockerfile
from .env_setup import resolve_stack
from .git_utils import candidate_commits, resolve_ref
from .io import append_jsonl, read_json_or_empty, write_json
from .measurement import measure_commit, run_warmups, write_measurement_preflight
from .models import BisectionPlan, BisectionSummary, CandidateAttempt, CandidateEvaluation
from .paired_reference import (
    MeasurementSummary,
    check_reference_signal,
    compare_candidate,
    metric_from_artifact,
    summarize_measurements,
)
from .probe import (
    PROBE_ACTION_HARNESS_BLOCKED,
    PROBE_ACTION_PLAN_ISSUE,
    PROBE_ACTION_READY,
    PROBE_ACTION_REPAIR_BASE_IMAGE,
    PROBE_ACTION_RUN_DEBUG_COMMAND,
    ProbeContext,
    ProbeDecision,
    ProbePolicy,
)
from .progress import format_metric, get_progress_reporter
from .recovery import (
    ACTION_ACCEPT,
    DeterministicRecoveryPolicy,
    RecoveryContext,
    RecoveryEvent,
    RecoveryKnobs,
    RecoveryPolicy,
    knobs_for_action,
)
from .security import parse_probe_debug_command, resolve_path_within
from .tooling import verify_attempt_tooling

# Hard ceiling on recovery retries per measurement, guarding against a misbehaving
# policy that never accepts. Policies enforce their own (smaller) budgets.
_MAX_RECOVERY_RETRIES = 5

# Log substrings that strongly indicate a commit's environment could not run on this
# machine (missing module, ABI/symbol mismatch) rather than a genuine perf regression.
# These are deliberately narrow to avoid misclassifying real benchmark failures.
_RUNTIME_INCOMPAT_SIGNATURES = (
    "ModuleNotFoundError",
    "ImportError:",
    "undefined symbol",
    "GLIBC_",
    "cannot open shared object file",
)


def _read_bisect_env_status(artifact_dir: Path) -> tuple[str | None, str | None, str | None] | None:
    """Read ``(status, skip_category, skip_detail)`` from ``bisect_env.json`` if present."""
    path = artifact_dir / "bisect_env.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    return data.get("status"), data.get("skip_category"), data.get("skip_detail")


def _log_signals_runtime_incompat(artifact_dir: Path) -> bool:
    """Return True if the benchmark log shows an environment/ABI failure signature."""
    log_path = artifact_dir / "benchmark.log"
    if not log_path.exists():
        return False
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return any(signature in text for signature in _RUNTIME_INCOMPAT_SIGNATURES)


def _measurement_note(
    artifact_dir: Path, *, timed_out: bool, exit_code: int | None, bench_result_exists: bool
) -> str | None:
    """Classify a measurement outcome, or return None when a metric should be read.

    The ``bisect_env.json`` sidecar is authoritative for environment skips: when a
    candidate's environment could not be reconstructed the runner exits cleanly without a
    benchmark result, so the skip category is surfaced as an ``env_skip:<category>`` note.
    """
    if timed_out:
        return "candidate_timeout"
    env_status = _read_bisect_env_status(artifact_dir)
    if env_status and env_status[0] == "skip":
        return f"env_skip:{env_status[1] or 'unknown'}"
    if exit_code != 0:
        return "runner_command_failed"
    if not bench_result_exists:
        return "missing_perf_smoke_test_result"
    try:
        result = json.loads((artifact_dir / "perf_smoke_test_result.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        result = {}
    if result.get("failure_phase") == "tooling_incompatible":
        return "env_skip:perf_smoke_tooling_incompatible"
    return None


def _skip_category_from_note(note: str | None) -> str | None:
    """Extract the skip category from an ``env_skip:<category>`` note."""
    if note and note.startswith("env_skip:"):
        return note.split(":", 1)[1]
    return None


def _is_tooling_incompatible_note(note: str | None) -> bool:
    """Return whether a measurement stopped at the pinned tooling support boundary."""
    return _skip_category_from_note(note) == "perf_smoke_tooling_incompatible"


def _nearest_evaluable_index(idx: int, low: int, high: int, tested_idx: set[int]) -> int | None:
    """Return the closest untested index to ``idx`` within ``[low, high]``, or None.

    Used to probe outward to the nearest testable commit when a midpoint cannot be
    evaluated (a "hole"). Ties prefer the higher index for determinism.
    """
    span = max(idx - low, high - idx)
    for offset in range(1, span + 1):
        for candidate_idx in (idx + offset, idx - offset):
            if low <= candidate_idx <= high and candidate_idx not in tested_idx:
                return candidate_idx
    return None


def _template_context(
    repo_root: Path, output_dir: Path, commit_sha: str, artifact_dir: Path, plan: BisectionPlan
) -> dict[str, str]:
    """Return placeholder values supported by runner path/command templates."""
    return {
        "repo_root": str(repo_root),
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
    plan: BisectionPlan,
    output_dir: Path,
    commit_sha: str,
    artifact_dir: Path,
    *,
    repo_root: Path | None = None,
    knobs: RecoveryKnobs | None = None,
) -> list[str]:
    """Build the runner command for one candidate.

    ``knobs`` injects recovery overrides (cache clearing, forced reinstall) chosen
    by the recovery policy for a retry.
    """
    if plan.runner is None:
        raise ValueError("plan.runner is required to run a bisection candidate.")

    output_dir = output_dir.resolve()
    artifact_dir = resolve_path_within(output_dir, artifact_dir, "artifact_dir")
    repo_root = (repo_root or Path.cwd()).resolve()
    context = _template_context(repo_root, output_dir, commit_sha, artifact_dir, plan)
    runner = plan.runner
    cmd = [
        sys.executable,
        "-m",
        "isaaclab_bisection.runner",
        "--repo_root",
        str(repo_root),
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
    if plan.tooling is not None:
        cmd.extend(
            [
                "--tooling_root",
                str(resolve_path_within(output_dir, plan.tooling.snapshot_relpath, "tooling.snapshot_relpath")),
                "--tooling_spec_hash",
                plan.tooling.tooling_spec_hash,
                "--tooling_bundle_hash",
                plan.tooling.bundle_hash,
                "--tooling_contract_id",
                plan.tooling.contract_id,
                "--tooling_source_commit_sha",
                plan.tooling.source_commit_sha,
            ]
        )
        if plan.tooling.authoritative:
            cmd.append("--tooling_authoritative")
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
            if flag in {"--source_dir", "--jit_cache", "--kit_cache"}:
                value = str(resolve_path_within(output_dir, value, flag))
            cmd.extend([flag, value])
    # Forward the inline task definition (Option B) so the runner can resolve the task
    # without a tasks.json entry. Omitted fields fall back to the registry in the runner.
    task_spec = plan.task
    if task_spec.num_envs is not None:
        cmd.extend(["--num_envs", str(task_spec.num_envs)])
    if task_spec.num_frames is not None:
        cmd.extend(["--num_frames", str(task_spec.num_frames)])
    if task_spec.warmup_frames is not None:
        cmd.extend(["--warmup_frames", str(task_spec.warmup_frames)])
    if task_spec.seed is not None:
        cmd.extend(["--seed", str(task_spec.seed)])
    if task_spec.camera_resolution is not None:
        cmd.extend(["--camera_resolution", str(task_spec.camera_resolution[0]), str(task_spec.camera_resolution[1])])
    if task_spec.timeout_minutes is not None:
        cmd.extend(["--timeout_minutes", str(task_spec.timeout_minutes)])
    for hydra_arg in task_spec.hydra_args:
        cmd.extend(["--hydra_arg", hydra_arg])
    if runner.mode in ("local-reconstruct", "docker-reconstruct"):
        # Share one run-scoped env cache across all candidates so per-commit installs
        # amortize (uv hardlinks the heavy wheels from its global cache). For
        # docker-reconstruct this host path is bind-mounted into the container.
        cmd.extend(["--env_cache_dir", str(output_dir / "env-cache")])
    if runner.mode == "synthetic":
        if runner.synthetic_good_value is not None and runner.synthetic_bad_value is not None:
            synthetic_good_value, synthetic_bad_value = runner.synthetic_good_value, runner.synthetic_bad_value
        elif plan.metric.regression_direction == "increase":
            synthetic_good_value, synthetic_bad_value = 500.0, 1000.0
        else:
            synthetic_good_value, synthetic_bad_value = 1000.0, 500.0
        cmd.extend(
            [
                "--first_bad_ref",
                runner.synthetic_first_bad_ref or plan.bad_ref,
                "--synthetic_metric_path",
                plan.metric.result_path,
                "--synthetic_good_value",
                str(synthetic_good_value),
                "--synthetic_bad_value",
                str(synthetic_bad_value),
            ]
        )
    cmd.extend(_format_template(item, context) or "" for item in runner.extra_args)
    if knobs is not None:
        cmd.extend(knobs.runner_args())
    return cmd


def _command_display(command: list[str] | str) -> str:
    """Render a command for logs."""
    return command if isinstance(command, str) else shlex.join(command)


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    """Terminate a timed-out subprocess and every descendant in its session."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait(timeout=5)


def _run_command(
    command: list[str] | str,
    *,
    command_log: Path,
    timeout_s: int | None,
    cwd: Path | None = None,
) -> tuple[int | None, bool, float]:
    """Run a candidate command, streaming stdout/stderr into live artifacts.

    In addition to the human-readable ``bisect_command.log``, this writes
    ``live_output.jsonl`` next to the command log while the process is still
    running. The LLM orchestrator/probe can tail that JSONL stream to diagnose
    container/install/task friction before the process exits.
    """
    start = time.monotonic()
    progress = get_progress_reporter()
    shell = isinstance(command, str)
    live_log = command_log.parent / "live_output.jsonl"
    command_log.parent.mkdir(parents=True, exist_ok=True)
    with command_log.open("w", encoding="utf-8") as log_fh:
        log_fh.write(f"$ {_command_display(command)}\n\n")
        log_fh.flush()
        with live_log.open("w", encoding="utf-8") as live_fh:
            live_fh.write(
                json.dumps(
                    {
                        "event": "process_start",
                        "elapsed_s": 0.0,
                        "command": _command_display(command),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            live_fh.flush()
            process = subprocess.Popen(
                command,
                shell=shell,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            assert process.stdout is not None
            selector = selectors.DefaultSelector()
            selector.register(process.stdout, selectors.EVENT_READ)
            timed_out = False

            def emit(line: str) -> None:
                log_fh.write(line)
                log_fh.flush()
                live_fh.write(
                    json.dumps(
                        {
                            "event": "output",
                            "elapsed_s": round(time.monotonic() - start, 3),
                            "line": line.rstrip("\n"),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                live_fh.flush()
                progress.relay(line)

            while True:
                if timeout_s is not None and time.monotonic() - start > timeout_s:
                    timed_out = True
                    _terminate_process_tree(process)
                    break
                for key, _ in selector.select(timeout=1.0):
                    line = key.fileobj.readline()
                    if line:
                        emit(line)
                progress.heartbeat(f"subprocess active; details in {command_log}")
                if process.poll() is not None:
                    remainder = process.stdout.read()
                    if remainder:
                        for line in remainder.splitlines(keepends=True):
                            emit(line)
                    break

            selector.close()
            duration_s = time.monotonic() - start
            if timed_out:
                timeout_line = f"\n[bisection] candidate command timed out after {timeout_s}s\n"
                emit(timeout_line)
                live_fh.write(
                    json.dumps({"event": "process_timeout", "elapsed_s": round(duration_s, 3)}, sort_keys=True) + "\n"
                )
                return None, True, duration_s
            exit_code = process.returncode
            live_fh.write(
                json.dumps(
                    {
                        "event": "process_exit",
                        "elapsed_s": round(duration_s, 3),
                        "exit_code": exit_code,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            return exit_code, False, duration_s


def write_status(output_dir: Path, **values) -> None:
    """Write current harness status."""
    blocker_payload = read_json_or_empty(output_dir / "blockers.json")
    payload = {
        "phase": values.pop("phase", "running"),
        "status": values.pop("status", "running"),
        **values,
    }
    if (output_dir / "artifact_index.json").exists():
        payload["artifact_index"] = "artifact_index.json"
    if terminal_blocker := blocker_payload.get("terminal_blocker"):
        payload["terminal_blocker"] = terminal_blocker
    write_json(output_dir / "status.json", payload)


def _write_summary(output_dir: Path, plan: BisectionPlan, summary: BisectionSummary) -> None:
    """Write enriched summary + run-level handoff artifacts."""
    payload = finalize_run_artifacts(output_dir, plan.to_json(), summary.to_json())
    write_json(output_dir / "summary.json", payload)
    security_scan = scan_artifacts(output_dir)
    write_json(output_dir / "security_scan.json", security_scan)
    payload["artifact_security"] = {
        "status": security_scan["status"],
        "finding_count": security_scan["finding_count"],
        "report": "security_scan.json",
    }
    write_json(output_dir / "summary.json", payload)
    if security_scan["findings"]:
        get_progress_reporter().event(
            "SECURITY",
            f"artifact sharing blocked: {security_scan['finding_count']} potential credential(s); "
            "review security_scan.json",
        )


def build_candidates(plan: BisectionPlan, repo_root: Path | None = None) -> dict:
    """Resolve refs and build the candidate commit list."""
    repo_root = (repo_root or Path.cwd()).resolve()
    good_sha = resolve_ref(repo_root, plan.good_ref)
    bad_sha = resolve_ref(repo_root, plan.bad_ref)
    commits = candidate_commits(repo_root, good_sha, bad_sha)
    return {
        "good_sha": good_sha,
        "bad_sha": bad_sha,
        "candidate_count": len(commits),
        "candidates": commits,
    }


# StackSpec fields that name a pinned runtime component. ``ovphysx`` is PhysX and
# ``ovrtx`` is the Kit/renderer pin; both are surfaced because a perf regression can
# originate in any of these dependencies rather than in IsaacLab's own source.
_STACK_COMPONENT_FIELDS = ("isaacsim", "warp_lang", "newton", "ovrtx", "ovphysx", "python_version", "platform")


def _component_stack_diff(from_sha: str, to_sha: str, *, relation: str, repo_root: Path | None = None) -> dict | None:
    """Diff the pinned runtime stacks of two commits, or None if they cannot resolve.

    Resolving a stack is git-only and cheap. Reporting must never crash the run, so a
    resolution failure downgrades to ``None`` rather than propagating.
    """
    try:
        repo_root = (repo_root or Path.cwd()).resolve()
        from_stack = resolve_stack(repo_root, from_sha)
        to_stack = resolve_stack(repo_root, to_sha)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
        return None
    changed = {
        field_name: {"from": getattr(from_stack, field_name), "to": getattr(to_stack, field_name)}
        for field_name in _STACK_COMPONENT_FIELDS
        if getattr(from_stack, field_name) != getattr(to_stack, field_name)
    }
    return {
        "relation": relation,
        "isaaclab_commit": {"from": from_sha[:12], "to": to_sha[:12]},
        "changed_components": changed,
        "stack_hash": {
            "from": from_stack.stack_hash,
            "to": to_stack.stack_hash,
            "changed": from_stack.stack_hash != to_stack.stack_hash,
        },
    }


def _build_stack_diff(
    *,
    good_ref: str,
    bad_ref: str,
    last_good: str | None = None,
    first_bad: str | None = None,
    repo_root: Path | None = None,
) -> dict | None:
    """Build the component stack diff for a run's report.

    Always includes the ``range`` diff (good_ref -> bad_ref) so a non-reproduction still
    tells the reader what moved across the whole range (a component we do not build from
    source may be the lever). When a culprit is localized, adds the ``culprit`` diff
    (last_good -> first_bad) pinpointing exactly what that commit changed.
    """
    diff: dict[str, dict] = {}
    range_diff = _component_stack_diff(good_ref, bad_ref, relation="good_ref_to_bad_ref", repo_root=repo_root)
    if range_diff is not None:
        diff["range"] = range_diff
    if last_good and first_bad and last_good != first_bad:
        culprit = _component_stack_diff(last_good, first_bad, relation="last_good_to_first_bad", repo_root=repo_root)
        if culprit is not None:
            diff["culprit"] = culprit
    return diff or None


def _measurement_artifact_dir(
    output_dir: Path, *, label: str, commit_sha: str, plan: BisectionPlan, run_idx: int
) -> Path:
    """Return the artifact directory for one local paired-reference measurement."""
    return output_dir / "measurements" / label / commit_sha[:12] / plan.task_id / plan.backend_key / f"run_{run_idx}"


def _run_single_measurement(
    plan: BisectionPlan,
    output_dir: Path,
    *,
    commit_sha: str,
    label: str,
    run_idx: int,
    knobs: RecoveryKnobs | None = None,
    recovery_attempt: int = 0,
    probe_policy: ProbePolicy | None = None,
    repo_root: Path | None = None,
) -> tuple[CandidateAttempt, float | None]:
    """Run one local measurement and return its attempt record and selected metric value.

    ``recovery_attempt`` (>0) directs the measurement to a sibling artifact directory
    so a recovery retry does not clobber the failed attempt's evidence, and ``knobs``
    carries the recovery overrides applied to that retry.
    """
    progress = get_progress_reporter()
    retry_text = f", retry {recovery_attempt}" if recovery_attempt else ""
    runner_mode = plan.runner.mode if plan.runner is not None else "unknown"
    progress.event(
        "MEASURE",
        f"{label} {commit_sha[:12]} sample {run_idx}{retry_text} ({runner_mode})",
    )
    artifact_dir = _measurement_artifact_dir(output_dir, label=label, commit_sha=commit_sha, plan=plan, run_idx=run_idx)
    if recovery_attempt:
        artifact_dir = artifact_dir.parent / f"{artifact_dir.name}_r{recovery_attempt}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    probe_note, effective_plan = _run_probe_loop(
        plan,
        output_dir,
        commit_sha=commit_sha,
        artifact_dir=artifact_dir,
        policy=probe_policy,
    )
    repo_root = (repo_root or Path.cwd()).resolve()
    command = format_runner_command(
        effective_plan, output_dir, commit_sha, artifact_dir, repo_root=repo_root, knobs=knobs
    )
    command_display = _command_display(command)
    if probe_note is not None:
        attempt = CandidateAttempt(
            attempt=run_idx,
            artifact_dir=str(artifact_dir),
            command=command_display,
            command_exit_code=None,
            note=probe_note,
            timed_out=False,
            duration_s=0.0,
        )
        write_attempt_summary(
            output_dir,
            plan=effective_plan.to_json(),
            attempt=attempt.to_json(),
            commit_sha=commit_sha,
            label=label,
            run_idx=run_idx,
            recovery_attempt=recovery_attempt,
            metric_value=None,
        )
        progress.event("WARNING", f"{label} {commit_sha[:12]} stopped by probe: {probe_note}")
        return attempt, None
    command_log = artifact_dir / "bisect_command.log"
    timeout_s = plan.timeout.candidate_timeout_s
    if knobs is not None and knobs.extra_timeout_s:
        timeout_s = (timeout_s or 0) + knobs.extra_timeout_s
    exit_code, timed_out, duration_s = _run_command(
        command,
        command_log=command_log,
        timeout_s=timeout_s,
        cwd=repo_root,
    )
    bench_result_path = artifact_dir / "perf_smoke_test_result.json"
    metric_value = None
    note = _measurement_note(
        artifact_dir,
        timed_out=timed_out,
        exit_code=exit_code,
        bench_result_exists=bench_result_path.exists(),
    )
    if note is None:
        verification = verify_attempt_tooling(effective_plan, artifact_dir)
        write_json(artifact_dir / "tooling_verification.json", verification)
        if verification["status"] == "mismatch":
            note = "tooling_mismatch:" + ";".join(verification["mismatches"])
    if note is None:
        try:
            metric_value = metric_from_artifact(artifact_dir, effective_plan.metric)
        except (KeyError, TypeError, ValueError) as exc:
            # A built environment that still fails to import/load at runtime is an era/ABI
            # hole, not a measurable regression; classify it so the search can skip it.
            note = (
                "env_skip:runtime_incompatible"
                if _log_signals_runtime_incompat(artifact_dir)
                else f"metric_unavailable:{exc}"
            )

    attempt = CandidateAttempt(
        attempt=run_idx,
        artifact_dir=str(artifact_dir),
        command=command_display,
        command_exit_code=exit_code,
        note=note,
        timed_out=timed_out,
        duration_s=duration_s,
    )
    write_attempt_summary(
        output_dir,
        plan=effective_plan.to_json(),
        attempt=attempt.to_json(),
        commit_sha=commit_sha,
        label=label,
        run_idx=run_idx,
        recovery_attempt=recovery_attempt,
        metric_value=metric_value,
    )
    if metric_value is not None:
        progress.event(
            "RESULT",
            f"{label} {commit_sha[:12]} = {format_metric(metric_value, plan.metric.unit)} ({duration_s:.1f}s)",
        )
        env = read_json_or_empty(artifact_dir / "bisect_env.json")
        if env:
            if env.get("env_dir"):
                environment = f"environment={'reused' if env.get('env_reused') else 'fresh'}"
            else:
                environment = f"mode={env.get('mode', runner_mode)}"
            progress.event(
                "ENV",
                f"{commit_sha[:12]} stack={env.get('stack_hash', 'unknown')} {environment}",
                verbose_only=True,
            )
    else:
        progress.event("WARNING", f"{label} {commit_sha[:12]} produced no metric: {note or 'unknown reason'}")
    return attempt, metric_value


def _read_log_tail(artifact_dir: Path, *, max_chars: int = 4000) -> str:
    """Return the tail of the benchmark/command logs for recovery inspection."""
    parts: list[str] = []
    for name in ("benchmark.log", "bisect_command.log", "live_output.jsonl"):
        path = artifact_dir / name
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        parts.append(f"=== {name} (tail) ===\n{text[-max_chars:]}")
    return "\n".join(parts)


def _read_file_tail(path: Path, *, max_chars: int = 4000) -> str:
    """Return a file tail, tolerating absent artifacts."""
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-max_chars:]
    except OSError:
        return ""


def _active_repair_path(output_dir: Path) -> Path:
    """Return the run-scoped active Docker repair marker path."""
    return output_dir / "repairs" / "active_docker_image.json"


def _plan_with_active_repair(plan: BisectionPlan, output_dir: Path) -> BisectionPlan:
    """Use a previously built repair image for later docker-reconstruct attempts."""
    if plan.runner is None or plan.runner.mode != "docker-reconstruct":
        return plan
    payload = read_json_or_empty(_active_repair_path(output_dir))
    repaired_image = payload.get("repaired_image")
    if not repaired_image or plan.runner.image == repaired_image:
        return plan
    return replace(plan, runner=replace(plan.runner, image=str(repaired_image)))


def _apply_base_image_repair(
    plan: BisectionPlan,
    output_dir: Path,
    probe_dir: Path,
    decision: ProbeDecision,
) -> tuple[BisectionPlan | None, ProbeDecision | None]:
    """Build a generated Docker repair image requested by the probe."""
    if plan.runner is None or plan.runner.mode != "docker-reconstruct":
        return None, ProbeDecision(
            PROBE_ACTION_HARNESS_BLOCKED,
            "repair_base_image is only supported for docker-reconstruct runs",
            confidence="high",
        )
    if not plan.runner.image:
        return None, ProbeDecision(
            PROBE_ACTION_HARNESS_BLOCKED,
            "repair_base_image requires a runner image",
            confidence="high",
        )
    try:
        repair_dir = output_dir / "repairs" / "docker-images"
        repair = write_repair_dockerfile(plan.runner.image, list(decision.apt_packages), repair_dir)
    except ValueError as exc:
        return None, ProbeDecision(PROBE_ACTION_HARNESS_BLOCKED, str(exc), confidence="high")

    active_path = _active_repair_path(output_dir)
    active = read_json_or_empty(active_path)
    if active.get("repaired_image") == repair.repaired_image:
        return replace(plan, runner=replace(plan.runner, image=repair.repaired_image)), None

    command_log = probe_dir / "base_image_repair.log"
    exit_code, timed_out, duration_s = _run_command(
        docker_build_command(repair), command_log=command_log, timeout_s=1800
    )
    payload = {
        "base_image": repair.base_image,
        "repaired_image": repair.repaired_image,
        "apt_packages": repair.apt_packages,
        "dockerfile": str(repair.dockerfile),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "duration_s": duration_s,
    }
    write_json(probe_dir / "base_image_repair.json", payload)
    append_jsonl(probe_dir / "probe_events.jsonl", {"event": "base_image_repair", **payload})
    if exit_code != 0 or timed_out:
        return None, ProbeDecision(
            PROBE_ACTION_HARNESS_BLOCKED,
            f"base image repair failed for packages {repair.apt_packages}",
            confidence="high",
        )

    active_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(active_path, payload)
    return replace(plan, runner=replace(plan.runner, image=repair.repaired_image)), None


def _probe_context(
    plan: BisectionPlan,
    output_dir: Path,
    *,
    commit_sha: str,
    artifact_dir: Path,
    probe_dir: Path,
    attempt: int,
    max_attempts: int,
) -> ProbeContext:
    """Build the LLM probe context from live artifacts."""
    install_log = output_dir / "env-cache" / "logs" / f"install-{commit_sha[:12]}.log"
    return ProbeContext(
        commit_sha=commit_sha,
        task_id=plan.task_id,
        backend_key=plan.backend_key,
        artifact_dir=artifact_dir,
        plan=plan.to_json(),
        live_output_tail=_read_file_tail(probe_dir / "live_output.jsonl"),
        install_log_tail=_read_file_tail(install_log),
        benchmark_log_tail=_read_file_tail(artifact_dir / "benchmark.log"),
        sidecars={
            "bisect_env": read_json_or_empty(artifact_dir / "bisect_env.json"),
            "launch_config": read_json_or_empty(artifact_dir / "launch_config.json"),
            "probe_result": read_json_or_empty(probe_dir / "probe_result.json"),
            "base_image_repair": read_json_or_empty(probe_dir / "base_image_repair.json"),
            "active_docker_image": read_json_or_empty(_active_repair_path(output_dir)),
        },
        attempt=attempt,
        max_attempts=max_attempts,
    )


def _write_probe_result(probe_dir: Path, decision: ProbeDecision, *, status: str) -> None:
    """Persist the terminal probe decision."""
    payload = {"status": status, "decision": decision.to_json()}
    write_json(probe_dir / "probe_result.json", payload)
    append_jsonl(probe_dir / "probe_events.jsonl", {"event": "probe_result", **payload})


def _run_probe_loop(
    plan: BisectionPlan,
    output_dir: Path,
    *,
    commit_sha: str,
    artifact_dir: Path,
    policy: ProbePolicy | None,
    max_attempts: int = 3,
) -> tuple[str | None, BisectionPlan]:
    """Run the LLM-driven container validation/setup-doctor loop.

    Returns ``(None, plan)`` when the probe declares the attempt ready for
    deterministic benchmarking; otherwise returns a friction note such as
    ``probe_failed:plan_issue``. The returned plan may point at a generated,
    run-scoped Docker repair image. The probe can run debug shell commands and
    sees their output live via ``probe/live_output.jsonl``.
    """
    active_plan = _plan_with_active_repair(plan, output_dir)
    if policy is None:
        return None, active_plan
    probe_dir = artifact_dir / "probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    for attempt in range(max_attempts + 1):
        ctx = _probe_context(
            active_plan,
            output_dir,
            commit_sha=commit_sha,
            artifact_dir=artifact_dir,
            probe_dir=probe_dir,
            attempt=attempt,
            max_attempts=max_attempts,
        )
        decision = policy.decide(ctx)
        append_jsonl(
            probe_dir / "probe_events.jsonl",
            {
                "event": "probe_decision",
                "attempt": attempt,
                **decision.to_json(),
            },
        )
        if decision.action == PROBE_ACTION_READY:
            _write_probe_result(probe_dir, decision, status="ready")
            return None, active_plan
        if decision.action == PROBE_ACTION_PLAN_ISSUE:
            _write_probe_result(probe_dir, decision, status="plan_issue")
            return "probe_failed:plan_issue", active_plan
        if decision.action == PROBE_ACTION_HARNESS_BLOCKED:
            _write_probe_result(probe_dir, decision, status="harness_blocked")
            return "probe_failed:harness_blocked", active_plan
        if decision.action == PROBE_ACTION_REPAIR_BASE_IMAGE:
            repaired_plan, blocker = _apply_base_image_repair(active_plan, output_dir, probe_dir, decision)
            if blocker is not None:
                _write_probe_result(probe_dir, blocker, status="harness_blocked")
                return "probe_failed:harness_blocked", active_plan
            assert repaired_plan is not None
            active_plan = repaired_plan
            continue
        if decision.action == PROBE_ACTION_RUN_DEBUG_COMMAND:
            if not decision.command:
                _write_probe_result(
                    probe_dir,
                    ProbeDecision(PROBE_ACTION_HARNESS_BLOCKED, "probe requested debug command without command"),
                    status="harness_blocked",
                )
                return "probe_failed:harness_blocked", active_plan
            try:
                debug_command = parse_probe_debug_command(decision.command)
            except ValueError as exc:
                blocker = ProbeDecision(
                    PROBE_ACTION_HARNESS_BLOCKED,
                    f"probe requested unsafe diagnostic command: {exc}",
                    confidence="high",
                )
                _write_probe_result(probe_dir, blocker, status="harness_blocked")
                return "probe_failed:harness_blocked", active_plan
            command_log = probe_dir / f"debug_command_{attempt + 1}.log"
            exit_code, timed_out, duration_s = _run_command(debug_command, command_log=command_log, timeout_s=600)
            append_jsonl(
                probe_dir / "probe_events.jsonl",
                {
                    "event": "debug_command",
                    "attempt": attempt,
                    "command": decision.command,
                    "exit_code": exit_code,
                    "timed_out": timed_out,
                    "duration_s": duration_s,
                },
            )
            continue
        _write_probe_result(
            probe_dir,
            ProbeDecision(PROBE_ACTION_HARNESS_BLOCKED, f"unsupported probe action: {decision.action}"),
            status="harness_blocked",
        )
        return "probe_failed:harness_blocked", active_plan
    _write_probe_result(
        probe_dir,
        ProbeDecision(PROBE_ACTION_HARNESS_BLOCKED, "probe budget exhausted"),
        status="harness_blocked",
    )
    return "probe_failed:harness_blocked", active_plan


def _measure_with_recovery(
    plan: BisectionPlan,
    output_dir: Path,
    *,
    commit_sha: str,
    label: str,
    run_idx: int,
    policy: RecoveryPolicy,
    probe_policy: ProbePolicy | None = None,
    repo_root: Path | None = None,
) -> tuple[CandidateAttempt, float | None]:
    """Produce one measurement, letting the recovery policy retry through friction.

    Runs the measurement and, on any outcome that yields no metric, hands the failure
    to ``policy``. The policy either returns a bounded retry (with concrete knobs) or
    accepts the outcome as an honest skip. Every step is appended to
    ``audit_log.jsonl`` and to the returned attempt's ``recovery_events`` so the
    verdict layer never sees friction that recovery could have cleared. The recovery
    layer decides only how hard to try -- never GOOD/BAD/UNCLEAR.
    """
    audit_path = output_dir / "audit_log.jsonl"
    knobs = RecoveryKnobs()
    events: list[RecoveryEvent] = []
    attempt_idx = 0

    while True:
        measurement_kwargs = {
            "commit_sha": commit_sha,
            "label": label,
            "run_idx": run_idx,
            "knobs": knobs if attempt_idx else None,
            "recovery_attempt": attempt_idx,
            "probe_policy": probe_policy,
        }
        if repo_root is not None:
            measurement_kwargs["repo_root"] = repo_root
        attempt, metric_value = _run_single_measurement(plan, output_dir, **measurement_kwargs)
        append_jsonl(
            audit_path,
            {
                "event": "measurement",
                "commit_sha": commit_sha,
                "label": label,
                "run_idx": run_idx,
                "recovery_attempt": attempt_idx,
                "note": attempt.note,
                "metric": metric_value,
                "artifact_dir": attempt.artifact_dir,
            },
        )
        if metric_value is not None:
            final = replace(attempt, recovery_events=[event.to_json() for event in events])
            write_attempt_summary(
                output_dir,
                plan=plan.to_json(),
                attempt=final.to_json(),
                commit_sha=commit_sha,
                label=label,
                run_idx=run_idx,
                recovery_attempt=attempt_idx,
                metric_value=metric_value,
                recovery_events=final.recovery_events,
            )
            return final, metric_value

        ctx = RecoveryContext(
            commit_sha=commit_sha,
            label=label,
            run_idx=run_idx,
            attempt=attempt_idx,
            note=attempt.note,
            exit_code=attempt.command_exit_code,
            timed_out=attempt.timed_out,
            artifact_dir=Path(attempt.artifact_dir),
            log_tail=_read_log_tail(Path(attempt.artifact_dir)),
            env_status=_read_bisect_env_status(Path(attempt.artifact_dir)),
        )
        decision = policy.decide(ctx)
        get_progress_reporter().event(
            "RECOVERY",
            f"{label} {commit_sha[:12]}: {decision.action} ({decision.reason})",
        )
        events.append(
            RecoveryEvent(
                attempt=attempt_idx,
                note=attempt.note,
                decision=decision.action,
                reason=decision.reason,
                skip_category=decision.skip_category,
                knobs=knobs.__dict__.copy(),
            )
        )
        append_jsonl(
            audit_path,
            {
                "event": "recovery_decision",
                "commit_sha": commit_sha,
                "label": label,
                "run_idx": run_idx,
                "recovery_attempt": attempt_idx,
                **decision.to_json(),
            },
        )
        write_attempt_summary(
            output_dir,
            plan=plan.to_json(),
            attempt=attempt.to_json(),
            commit_sha=commit_sha,
            label=label,
            run_idx=run_idx,
            recovery_attempt=attempt_idx,
            metric_value=metric_value,
            recovery_events=[event.to_json() for event in events],
        )

        if decision.action == ACTION_ACCEPT or attempt_idx >= _MAX_RECOVERY_RETRIES:
            note = attempt.note
            if decision.skip_category:
                note = f"env_skip:{decision.skip_category}"
            final = replace(attempt, note=note, recovery_events=[event.to_json() for event in events])
            write_attempt_summary(
                output_dir,
                plan=plan.to_json(),
                attempt=final.to_json(),
                commit_sha=commit_sha,
                label=label,
                run_idx=run_idx,
                recovery_attempt=attempt_idx,
                metric_value=None,
                recovery_events=final.recovery_events,
            )
            return final, None

        knobs = knobs_for_action(decision.action, knobs)
        attempt_idx += 1


def _run_warmup_measurements(
    plan: BisectionPlan,
    output_dir: Path,
    *,
    commit_sha: str,
    label: str,
    policy: RecoveryPolicy,
    probe_policy: ProbePolicy | None = None,
    repo_root: Path | None = None,
) -> CandidateAttempt | None:
    """Run the shared one-per-commit warmup and return a failed attempt, if any."""
    return run_warmups(
        plan,
        output_dir,
        commit_sha=commit_sha,
        label=label,
        policy=policy,
        probe_policy=probe_policy,
        measure_with_recovery=partial(_measure_with_recovery, repo_root=repo_root),
    )


def _measure_reference_commit(
    plan: BisectionPlan,
    output_dir: Path,
    *,
    commit_sha: str,
    label: str,
    policy: RecoveryPolicy,
    probe_policy: ProbePolicy | None = None,
    repo_root: Path | None = None,
) -> tuple[MeasurementSummary | None, list[CandidateAttempt], str | None]:
    """Measure a reference commit until stable enough or capped."""
    warmup_failure = _run_warmup_measurements(
        plan,
        output_dir,
        commit_sha=commit_sha,
        label=label,
        policy=policy,
        probe_policy=probe_policy,
        repo_root=repo_root,
    )
    if warmup_failure is not None:
        return None, [warmup_failure], warmup_failure.note or "warmup_failed"
    min_runs = plan.measurement.reference_runs
    max_runs = max(min_runs, plan.measurement.max_reference_runs)
    attempts: list[CandidateAttempt] = []
    metric_values: list[float] = []
    summary: MeasurementSummary | None = None

    for run_idx in range(1, max_runs + 1):
        attempt, metric_value = _measure_with_recovery(
            plan,
            output_dir,
            commit_sha=commit_sha,
            label=label,
            run_idx=run_idx,
            policy=policy,
            probe_policy=probe_policy,
            repo_root=repo_root,
        )
        attempts.append(attempt)
        if metric_value is None:
            return None, attempts, attempt.note or "missing_metric_measurement"
        metric_values.append(metric_value)
        if run_idx < min_runs:
            continue
        summary = summarize_measurements(label, plan.metric, metric_values)
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
    policy: RecoveryPolicy,
    probe_policy: ProbePolicy | None = None,
    repo_root: Path | None = None,
) -> CandidateEvaluation:
    """Run and classify one candidate using local paired-reference measurements."""
    warmup_failure = _run_warmup_measurements(
        plan,
        output_dir,
        commit_sha=commit_sha,
        label="candidate",
        policy=policy,
        probe_policy=probe_policy,
        repo_root=repo_root,
    )
    if warmup_failure is not None:
        return CandidateEvaluation(
            commit_sha=commit_sha,
            bisect_verdict="SKIP",
            artifact_dir=warmup_failure.artifact_dir,
            command_exit_code=warmup_failure.command_exit_code,
            metric_name=plan.metric.name,
            metric_unit=plan.metric.unit,
            note=warmup_failure.note or "warmup_failed",
            command=warmup_failure.command,
            attempts=[warmup_failure.to_json()],
            timed_out=warmup_failure.timed_out,
            duration_s=warmup_failure.duration_s,
            final_artifact_dir=warmup_failure.artifact_dir,
            comparison_mode="paired_reference",
            skip_category=_skip_category_from_note(warmup_failure.note),
            recovery_events=warmup_failure.recovery_events,
        )
    max_runs = max(plan.measurement.candidate_runs, plan.measurement.max_candidate_runs)
    attempts: list[CandidateAttempt] = []
    metric_values: list[float] = []
    last_comparison = None

    for run_idx in range(1, max_runs + 1):
        attempt, metric_value = _measure_with_recovery(
            plan,
            output_dir,
            commit_sha=commit_sha,
            label="candidate",
            run_idx=run_idx,
            policy=policy,
            probe_policy=probe_policy,
            repo_root=repo_root,
        )
        attempts.append(attempt)
        if metric_value is None:
            return CandidateEvaluation(
                commit_sha=commit_sha,
                bisect_verdict="SKIP",
                artifact_dir=attempt.artifact_dir,
                command_exit_code=attempt.command_exit_code,
                metric_name=plan.metric.name,
                metric_unit=plan.metric.unit,
                note=attempt.note,
                command=attempt.command,
                attempt_count=run_idx,
                attempts=[item.to_json() for item in attempts],
                timed_out=attempt.timed_out,
                duration_s=attempt.duration_s,
                retry_reason=attempt.note if run_idx < max_runs else None,
                final_artifact_dir=attempt.artifact_dir,
                comparison_mode="paired_reference",
                skip_category=_skip_category_from_note(attempt.note),
                recovery_events=attempt.recovery_events,
            )
        metric_values.append(metric_value)
        if run_idx < plan.measurement.candidate_runs:
            continue

        candidate_summary = summarize_measurements("candidate", plan.metric, metric_values)
        last_comparison = compare_candidate(
            candidate_summary, good_summary, plan.metric, plan.measurement, reference_noise_pct=reference_noise_pct
        )
        if last_comparison.verdict != "UNCLEAR":
            break

    if last_comparison is None:
        candidate_summary = summarize_measurements("candidate", plan.metric, metric_values)
        last_comparison = compare_candidate(
            candidate_summary, good_summary, plan.metric, plan.measurement, reference_noise_pct=reference_noise_pct
        )

    final_attempt = attempts[-1]
    return CandidateEvaluation(
        commit_sha=commit_sha,
        bisect_verdict=last_comparison.verdict,
        artifact_dir=final_attempt.artifact_dir,
        command_exit_code=final_attempt.command_exit_code,
        metric_name=last_comparison.metric_name,
        metric_unit=last_comparison.metric_unit,
        measured_value=last_comparison.measured_value,
        baseline_value=last_comparison.baseline_value,
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
        recovery_events=[event for item in attempts for event in item.recovery_events],
    )


def _recommended_candidate_runs(reference_noise_pct: float, plan: BisectionPlan) -> int:
    """Recommend candidate run count from the measured reference noise (variance-driven).

    A quiet reference (noise within the gray zone) needs only the configured floor of
    runs; a noisier reference warrants more repeats so a candidate's median is trusted,
    bounded by ``max_candidate_runs``.
    """
    floor = plan.measurement.candidate_runs
    ceiling = max(floor, plan.measurement.max_candidate_runs)
    gray = max(plan.measurement.gray_zone_pct, 1e-9)
    if reference_noise_pct <= gray:
        return floor
    if reference_noise_pct <= 2.0 * gray:
        return min(ceiling, floor + 1)
    return ceiling


def _write_grounding_artifacts(
    output_dir: Path,
    plan: BisectionPlan,
    good_summary: MeasurementSummary,
    bad_summary: MeasurementSummary,
    reference_check,
    *,
    candidate_count: int,
) -> BisectionPlan:
    """Emit grounding + plan artifacts and return a plan with variance-driven runs.

    ``grounding.json`` records the reproduced good/bad separation the search is
    grounded on; ``bisect_plan.json`` records the resulting execution plan (recommended
    candidate runs and the search budget). The returned plan carries the recommended
    candidate-run floor so noisier references are measured more times.
    """
    reference_noise_pct = float(reference_check.reference_noise_pct or 0.0)
    recommended_runs = _recommended_candidate_runs(reference_noise_pct, plan)
    write_json(
        output_dir / "grounding.json",
        {
            "metric": plan.metric.to_json(),
            "reproduced": reference_check.reproduced,
            "observed_regression_pct": reference_check.regression_pct,
            "effective_threshold_pct": reference_check.effective_threshold_pct,
            "reference_noise_pct": reference_noise_pct,
            "good": good_summary.to_json(),
            "bad": bad_summary.to_json(),
        },
    )
    write_json(
        output_dir / "bisect_plan.json",
        {
            "candidate_count": candidate_count,
            "reference_noise_pct": reference_noise_pct,
            "configured_candidate_runs": plan.measurement.candidate_runs,
            "recommended_candidate_runs": recommended_runs,
            "max_candidate_runs": plan.measurement.max_candidate_runs,
            "expected_max_tests": min(candidate_count, (candidate_count.bit_length())),
        },
    )
    if recommended_runs == plan.measurement.candidate_runs:
        return plan
    return replace(plan, measurement=replace(plan.measurement, candidate_runs=recommended_runs))


def _bisect_search(
    candidates: list[str],
    good_sha: str,
    bad_sha: str,
    evaluate: Callable[[int], tuple[str, str | None]],
    *,
    max_tests: int = 50,
) -> dict:
    """Hole-tolerant binary search over ``candidates`` (good-exclusive, bad-inclusive).

    ``evaluate(idx)`` returns ``(verdict, note)`` where verdict is one of
    ``GOOD``/``BAD``/``UNCLEAR``/``SKIP``. ``SKIP`` marks an un-evaluable commit (a
    "hole"); the search probes outward to the nearest testable commit and keeps the hole
    inside the active window until it can be bracketed. The known-good (index -1) and
    known-bad (last index) commits bound the interval and are never re-tested here.
    """
    n = len(candidates)
    low, high = 0, n - 2  # candidates[n - 1] == bad_sha is already known-bad
    tested: list[str] = []
    tested_idx: set[int] = set()
    skipped: list[dict] = []
    last_good, last_good_idx = good_sha, -1
    best_bad, best_bad_idx = bad_sha, n - 1

    def _outcome(status: str, reason: str) -> dict:
        holes_between = sorted(
            (item for item in skipped if last_good_idx < item["index"] < best_bad_idx),
            key=lambda item: item["index"],
        )
        narrowed = None
        if status in {"completed_with_holes", "inconclusive_holes"} or holes_between:
            narrowed = {
                "last_good_commit": last_good,
                "first_bad_commit": best_bad,
                "unevaluable_between": [item["commit_sha"] for item in holes_between],
            }
        return {
            "status": status,
            "reason": reason,
            "suspected_first_bad_commit": None if status == "unsupported_tooling_contract" else best_bad,
            "last_good_commit": last_good,
            "tested_commits": tested,
            "skipped_commits": skipped,
            "narrowed_interval": narrowed,
        }

    while low <= high:
        if len(tested) >= max_tests:
            return _outcome("inconclusive", "max_tests_reached")
        idx = (low + high) // 2
        probe = idx if idx not in tested_idx else _nearest_evaluable_index(idx, low, high, tested_idx)
        verdict: str | None = None
        while probe is not None:
            verdict, note = evaluate(probe)
            tested.append(candidates[probe])
            tested_idx.add(probe)
            if verdict == "SKIP":
                if _is_tooling_incompatible_note(note):
                    skipped.append({"commit_sha": candidates[probe], "index": probe, "reason": note})
                    return _outcome("unsupported_tooling_contract", "perf_smoke_tooling_incompatible")
                skipped.append({"commit_sha": candidates[probe], "index": probe, "reason": note})
                probe = _nearest_evaluable_index(idx, low, high, tested_idx)
                verdict = None
                continue
            break
        if verdict is None:
            return _outcome("inconclusive_holes", "candidates_unevaluable")
        if verdict == "GOOD":
            last_good, last_good_idx = candidates[probe], probe
            low = probe + 1
        elif verdict == "BAD":
            best_bad, best_bad_idx = candidates[probe], probe
            high = probe - 1
        else:
            return _outcome("inconclusive", "candidate_unclear")

    if any(last_good_idx < item["index"] < best_bad_idx for item in skipped):
        return _outcome("completed_with_holes", "first_bad_in_narrowed_interval")
    return _outcome("completed", "first_bad_found")


def _write_preflight(output_dir: Path, plan: BisectionPlan) -> None:
    """Gather and record host readiness facts before measurement begins.

    Advisory only: writes ``preflight.json`` and mirrors any warnings into the audit log
    so a mismatched GPU, missing driver, or unreachable Docker daemon is visible up
    front. Never raises -- a preflight failure must not block the run.
    """
    write_measurement_preflight(output_dir, plan)


def _balanced_probe_commits(candidates: list[str], max_tests: int) -> list[str]:
    """Select commits in breadth-first midpoint order for an internal range probe."""
    if not candidates or max_tests <= 0:
        return []

    selected: list[str] = []
    intervals = [(0, len(candidates) - 1)]
    while intervals and len(selected) < max_tests:
        low, high = intervals.pop(0)
        if low > high:
            continue
        midpoint = (low + high) // 2
        selected.append(candidates[midpoint])
        intervals.extend(((low, midpoint - 1), (midpoint + 1, high)))
    return selected


def _cleanup_probe_environment(plan: BisectionPlan, output_dir: Path, commit_sha: str) -> bool | None:
    """Remove one reconstructed Docker environment while retaining download caches."""
    runner = plan.runner
    if runner is None or runner.mode != "docker-reconstruct" or not runner.image:
        return None

    env_cache_dir = output_dir / "env-cache"
    if not env_cache_dir.exists():
        return True

    command = [
        "docker",
        "run",
        "--rm",
        "--entrypoint",
        "rm",
        "-v",
        f"{env_cache_dir.resolve()}:/env-cache",
        runner.image,
        "-rf",
        f"/env-cache/envs/{commit_sha[:12]}",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _run_range_probe(
    plan: BisectionPlan,
    output_dir: Path,
    *,
    candidates: list[str],
    good_sha: str,
    bad_sha: str,
    max_tests: int,
    policy: RecoveryPolicy,
    probe_policy: ProbePolicy | None,
    cleanup_envs: bool = False,
    repo_root: Path | None = None,
) -> BisectionSummary:
    """Sweep commits through checkout, reconstruction, and benchmark execution."""
    progress = get_progress_reporter()
    # ``candidate_commits`` excludes the good endpoint and includes the bad endpoint.
    # Endpoints are always included. A bounded probe spreads interior coverage across
    # the ancestry path; when the limit covers every interior commit, this is a full
    # compatibility sweep of the range.
    interior_commits = candidates[:-1]
    selected_interior = _balanced_probe_commits(interior_commits, max_tests)
    probe_commits = [good_sha, *selected_interior, bad_sha]
    omitted_count = max(0, len(interior_commits) - len(selected_interior))
    result_dir = output_dir / "probe_results"
    measurements: list[dict] = []
    failures: list[dict] = []

    progress.event(
        "PROBE RANGE",
        f"compatibility sweep measuring {len(probe_commits)}/{len(candidates) + 1} range commit(s); "
        f"coverage={'complete' if omitted_count == 0 else 'bounded'}, first-bad detection disabled",
    )

    def _write_probe_artifact(status: str) -> None:
        write_json(
            output_dir / "probe_range.json",
            {
                "schema_version": 1,
                "status": status,
                "mode": "internal_range_probe",
                "authoritative": False,
                "first_bad_detection": False,
                "cleanup_environments": cleanup_envs,
                "traversal": "breadth_first_midpoint",
                "coverage_complete": omitted_count == 0,
                "good_ref": good_sha,
                "bad_ref": bad_sha,
                "reference_signal": None,
                "selected_commits": probe_commits,
                "omitted_interior_commit_count": omitted_count,
                "measurements": measurements,
                "failures": failures,
            },
        )

    _write_probe_artifact("running")
    for ordinal, commit_sha in enumerate(probe_commits, start=1):
        progress.event("PROBE", f"measurement {ordinal}/{len(probe_commits)}: {commit_sha[:12]}")
        write_status(
            output_dir,
            phase="probe_range",
            status="running",
            total_candidates=len(probe_commits),
            completed_tests=ordinal - 1,
            current_commit=commit_sha,
            metric=plan.metric.to_json(),
            comparison_mode="probe_range",
            authoritative=False,
        )
        result = measure_commit(
            plan,
            output_dir,
            commit_sha=commit_sha,
            label="probe_commit",
            policy=policy,
            probe_policy=probe_policy,
            min_runs=plan.measurement.candidate_runs,
            max_runs=plan.measurement.max_candidate_runs,
            measure_with_recovery=partial(_measure_with_recovery, repo_root=repo_root),
        )
        measurement = {
            "commit_sha": commit_sha,
            "succeeded": result.succeeded,
            "summary": result.summary.to_json() if result.summary is not None else None,
            "attempts": [attempt.to_json() for attempt in result.attempts],
            "note": result.note,
            "bisect_verdict": None,
            "authoritative": False,
        }
        if result.summary is None:
            failure = {"commit_sha": commit_sha, "note": result.note or "measurement_failed"}
            failures.append(failure)
            progress.event("PROBE", f"{commit_sha[:12]} measurement failed: {failure['note']}")
        else:
            progress.event(
                "PROBE",
                f"{commit_sha[:12]} benchmark succeeded: "
                f"{format_metric(result.summary.median_value, plan.metric.unit)} (not classified)",
            )
        if cleanup_envs:
            cleanup_succeeded = _cleanup_probe_environment(plan, output_dir, commit_sha)
            measurement["environment_cleanup_succeeded"] = cleanup_succeeded
            if cleanup_succeeded is False:
                progress.event("WARNING", f"could not clean reconstructed environment for {commit_sha[:12]}")
            elif cleanup_succeeded:
                progress.event("CLEANUP", f"released reconstructed environment for {commit_sha[:12]}")
        measurements.append(measurement)
        write_json(result_dir / f"{commit_sha[:12]}.json", measurement)
        _write_probe_artifact("running")

    status = "probe_completed" if not failures else "probe_completed_with_failures"
    reason = "internal_range_probe_completed" if not failures else "internal_range_probe_measurement_failures"
    _write_probe_artifact(status)
    summary = BisectionSummary(
        status=status,
        reason=reason,
        tested_commits=probe_commits,
        suspected_first_bad_commit=None,
        last_good_commit=None,
        good_ref=good_sha,
        bad_ref=bad_sha,
        metric=plan.metric.to_json(),
        comparison_mode="probe_range",
        skipped_commits=failures,
        stack_diff=_build_stack_diff(good_ref=good_sha, bad_ref=bad_sha, repo_root=repo_root),
    )
    _write_summary(output_dir, plan, summary)
    return summary


def run_local_bisection(
    plan: BisectionPlan,
    output_dir: Path,
    *,
    repo_root: Path | None = None,
    max_tests: int = 50,
    recovery_policy: RecoveryPolicy | None = None,
    probe_policy: ProbePolicy | None = None,
    probe_only: bool = False,
    cleanup_probe_envs: bool = False,
) -> BisectionSummary:
    """Run local paired-reference bisection and write status/result artifacts.

    ``recovery_policy`` supervises benchmark-execution friction (retry vs. skip),
    while ``probe_policy`` can validate setup before deterministic measurements.
    Neither policy influences GOOD/BAD/UNCLEAR verdicts or the search path.
    ``probe_only`` sweeps selected commits through the candidate execution pipeline,
    continuing after per-commit failures while disabling classification and first-bad
    reporting.
    Recovery defaults to the model-free
    :class:`~bisection.recovery.DeterministicRecoveryPolicy`.
    """
    repo_root = (repo_root or Path.cwd()).resolve()
    policy = recovery_policy or DeterministicRecoveryPolicy()
    progress = get_progress_reporter()
    candidate_payload = build_candidates(plan, repo_root)
    candidates = list(candidate_payload["candidates"])
    write_json(output_dir / "candidates.json", candidate_payload)
    good_sha = candidate_payload["good_sha"]
    bad_sha = candidate_payload["bad_sha"]
    progress.event(
        "RANGE",
        f"{good_sha[:12]} -> {bad_sha[:12]} ({len(candidates)} candidate commits, metric={plan.metric.name})",
    )

    _write_preflight(output_dir, plan)
    preflight = read_json_or_empty(output_dir / "preflight.json")
    gpu = preflight.get("gpu", {})
    gpu_name = gpu.get("name") if isinstance(gpu, dict) else None
    disk_free = preflight.get("disk_free_gib")
    facts = [gpu_name or "GPU unknown", f"runner={plan.runner.mode if plan.runner else 'unknown'}"]
    if disk_free is not None:
        facts.append(f"{disk_free:.1f} GiB free")
    progress.event("PREFLIGHT", ", ".join(facts))
    for warning in preflight.get("warnings", []):
        progress.event("WARNING", str(warning))

    if probe_only:
        return _run_range_probe(
            plan,
            output_dir,
            candidates=candidates,
            good_sha=good_sha,
            bad_sha=bad_sha,
            max_tests=max_tests,
            policy=policy,
            probe_policy=probe_policy,
            cleanup_envs=cleanup_probe_envs,
            repo_root=repo_root,
        )

    write_status(output_dir, phase="preflight", status="running", current_commit=good_sha, metric=plan.metric.to_json())
    progress.event("GOOD REF", f"qualifying {good_sha[:12]}")
    good_measurement = measure_commit(
        plan,
        output_dir,
        commit_sha=good_sha,
        label="good_ref",
        policy=policy,
        probe_policy=probe_policy,
        measure_with_recovery=partial(_measure_with_recovery, repo_root=repo_root),
    )
    good_summary, good_attempts, good_note = (
        good_measurement.summary,
        good_measurement.attempts,
        good_measurement.note,
    )
    if good_summary is None:
        tooling_incompatible = _is_tooling_incompatible_note(good_note)
        summary = BisectionSummary(
            status="unsupported_tooling_contract" if tooling_incompatible else "inconclusive",
            reason=(
                "perf_smoke_tooling_incompatible"
                if tooling_incompatible
                else f"good_ref_measurement_failed:{good_note}"
            ),
            tested_commits=[],
            suspected_first_bad_commit=None,
            last_good_commit=None,
            good_ref=good_sha,
            bad_ref=bad_sha,
            metric=plan.metric.to_json(),
            comparison_mode="paired_reference",
            reference_stats={"good_attempts": [attempt.to_json() for attempt in good_attempts]},
        )
        _write_summary(output_dir, plan, summary)
        return summary
    progress.event(
        "GOOD REF",
        f"median={format_metric(good_summary.median_value, plan.metric.unit)}, spread={good_summary.spread_pct:.2f}%",
    )

    write_status(output_dir, phase="preflight", status="running", current_commit=bad_sha, metric=plan.metric.to_json())
    progress.event("BAD REF", f"qualifying {bad_sha[:12]}")
    bad_measurement = measure_commit(
        plan,
        output_dir,
        commit_sha=bad_sha,
        label="bad_ref",
        policy=policy,
        probe_policy=probe_policy,
        measure_with_recovery=partial(_measure_with_recovery, repo_root=repo_root),
    )
    bad_summary, bad_attempts, bad_note = (
        bad_measurement.summary,
        bad_measurement.attempts,
        bad_measurement.note,
    )
    reference_stats = {
        "good": good_summary.to_json(),
        "bad": bad_summary.to_json() if bad_summary is not None else None,
        "good_attempts": [attempt.to_json() for attempt in good_attempts],
        "bad_attempts": [attempt.to_json() for attempt in bad_attempts],
    }
    if bad_summary is None:
        tooling_incompatible = _is_tooling_incompatible_note(bad_note)
        summary = BisectionSummary(
            status="unsupported_tooling_contract" if tooling_incompatible else "inconclusive",
            reason=(
                "perf_smoke_tooling_incompatible" if tooling_incompatible else f"bad_ref_measurement_failed:{bad_note}"
            ),
            tested_commits=[],
            suspected_first_bad_commit=None,
            last_good_commit=None,
            good_ref=good_sha,
            bad_ref=bad_sha,
            metric=plan.metric.to_json(),
            comparison_mode="paired_reference",
            reference_stats=reference_stats,
        )
        _write_summary(output_dir, plan, summary)
        return summary
    progress.event(
        "BAD REF",
        f"median={format_metric(bad_summary.median_value, plan.metric.unit)}, spread={bad_summary.spread_pct:.2f}%",
    )

    reference_check = check_reference_signal(good_summary, bad_summary, plan.metric, plan.measurement)
    reference_stats["check"] = reference_check.to_json()
    write_json(output_dir / "reference_measurements.json", reference_stats)
    regression_text = (
        f"{reference_check.regression_pct:.2f}%" if reference_check.regression_pct is not None else "unavailable"
    )
    threshold_text = (
        f"{reference_check.effective_threshold_pct:.2f}%"
        if reference_check.effective_threshold_pct is not None
        else "unavailable"
    )
    progress.event(
        "SIGNAL",
        f"regression={regression_text}, "
        f"threshold={threshold_text}, "
        f"reproduced={'yes' if reference_check.reproduced else 'no'}",
    )
    if not reference_check.reproduced:
        # Emit measured facts only. ``state`` is the deterministic outcome of the reference
        # check (e.g. "local_regression_not_reproduced", "good_ref_measurements_too_noisy").
        non_repro = {
            "state": reference_check.note or "below_detection_threshold",
            "observed_regression_pct": reference_check.regression_pct,
            "effective_threshold_pct": reference_check.effective_threshold_pct,
            "reference_noise_pct": reference_check.reference_noise_pct,
            "good_median": good_summary.median_value,
            "bad_median": bad_summary.median_value,
            "metric": plan.metric.name,
        }
        summary = BisectionSummary(
            status="inconclusive",
            reason=reference_check.note or "local_regression_not_reproduced",
            tested_commits=[],
            suspected_first_bad_commit=None,
            last_good_commit=good_sha,
            good_ref=good_sha,
            bad_ref=bad_sha,
            metric=plan.metric.to_json(),
            comparison_mode="paired_reference",
            reference_stats=reference_stats,
            non_repro=non_repro,
            stack_diff=_build_stack_diff(good_ref=good_sha, bad_ref=bad_sha, repo_root=repo_root),
        )
        _write_summary(output_dir, plan, summary)
        return summary

    plan = _write_grounding_artifacts(
        output_dir, plan, good_summary, bad_summary, reference_check, candidate_count=len(candidates)
    )

    result_dir = output_dir / "results"
    reference_noise_pct = float(reference_check.reference_noise_pct or 0.0)
    completed = 0

    def _evaluate(idx: int) -> tuple[str, str | None]:
        nonlocal completed
        commit_sha = candidates[idx]
        progress.event(
            "CANDIDATE",
            f"test {completed + 1}/{min(max_tests, max(1, len(candidates) - 1))}: {commit_sha[:12]}",
        )
        write_status(
            output_dir,
            phase="running",
            status="running",
            total_candidates=len(candidates),
            completed_tests=completed,
            current_commit=commit_sha,
            metric=plan.metric.to_json(),
            comparison_mode="paired_reference",
        )
        evaluation = run_local_candidate(
            plan,
            output_dir,
            commit_sha,
            good_summary,
            reference_noise_pct=reference_noise_pct,
            policy=policy,
            probe_policy=probe_policy,
            repo_root=repo_root,
        )
        write_json(result_dir / f"{commit_sha[:12]}.json", evaluation.to_json())
        result_detail = evaluation.bisect_verdict
        if evaluation.measured_value is not None:
            result_detail += f", {format_metric(evaluation.measured_value, evaluation.metric_unit)}"
        if evaluation.regression_pct is not None:
            result_detail += f", regression={evaluation.regression_pct:.2f}%"
        progress.event("CLASSIFY", f"{commit_sha[:12]} -> {result_detail}")
        completed += 1
        return evaluation.bisect_verdict, evaluation.note

    outcome = _bisect_search(candidates, good_sha, bad_sha, _evaluate, max_tests=max_tests)
    summary = BisectionSummary(
        status=outcome["status"],
        reason=outcome["reason"],
        tested_commits=outcome["tested_commits"],
        suspected_first_bad_commit=outcome["suspected_first_bad_commit"],
        last_good_commit=outcome["last_good_commit"],
        good_ref=good_sha,
        bad_ref=bad_sha,
        metric=plan.metric.to_json(),
        comparison_mode="paired_reference",
        reference_stats=reference_stats,
        narrowed_interval=outcome["narrowed_interval"],
        skipped_commits=outcome["skipped_commits"],
        stack_diff=_build_stack_diff(
            good_ref=good_sha,
            bad_ref=bad_sha,
            last_good=outcome["last_good_commit"],
            first_bad=outcome["suspected_first_bad_commit"],
            repo_root=repo_root,
        ),
    )
    _write_summary(output_dir, plan, summary)
    return summary
