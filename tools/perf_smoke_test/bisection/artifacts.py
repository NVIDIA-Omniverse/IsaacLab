# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Structured artifact builders for bisection runs.

This module intentionally does not decide recovery actions or bisection verdicts.
It normalizes already-produced evidence into stable handoff artifacts that humans
and outer agents can consume without scraping raw logs first.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from contracts import BenchResult

from .io import read_json_or_empty as _read_json
from .io import write_json
from .recovery import classify_host_blocker, looks_source_checkout_failure

# Human-readable reason + remediation for each host/operator blocker category that
# :func:`~bisection.recovery.classify_host_blocker` can return. Kept next to the
# classifier's vocabulary so the post-hoc blocker report matches the retry-time skip
# category a run recorded.
_HOST_BLOCKER_DETAILS: dict[str, dict[str, Any]] = {
    "host_resource": {
        "phase": "install",
        "reason": "Host storage filled while installing or extracting dependencies.",
        "next_steps": [
            "Free disk space or move the bisection work_dir/env-cache to a larger volume.",
            "Relaunch the same plan after the storage issue is resolved.",
        ],
    },
    "docker_unavailable": {
        "phase": "container_setup",
        "reason": "The Docker daemon is not reachable from the bisection host.",
        "next_steps": [
            "Start Docker (or fix the socket permissions) and confirm `docker run hello-world` works.",
            "Relaunch the same plan once Docker is available.",
        ],
    },
    "gpu_unavailable": {
        "phase": "container_setup",
        "reason": "The container could not access a GPU (driver or NVIDIA container toolkit missing).",
        "next_steps": [
            "Install/repair the NVIDIA driver and nvidia-container-toolkit, then confirm "
            "`docker run --gpus all nvidia/cuda nvidia-smi` works.",
            "Relaunch the same plan on a GPU-capable host.",
        ],
    },
    "base_image_missing": {
        "phase": "container_setup",
        "reason": "The bisection base image is not present locally and could not be pulled.",
        "next_steps": [
            "Build the base image (see the docker-reconstruct recipe) or pull/tag it as expected.",
            "Relaunch the same plan once the image is available.",
        ],
    },
}

ARTIFACT_SCHEMA_VERSION = 1
_TAIL_CHARS = 2500
_MISSING_MODULE_RE = re.compile(r"No module named ['\"]([^'\"]+)['\"]")


def relpath(path: Path, root: Path) -> str:
    """Return a portable path relative to ``root`` when possible."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _read_bench_result(path: Path) -> BenchResult | None:
    """Read ``perf_smoke_test_result.json`` as a typed :class:`~contracts.BenchResult`.

    Returns ``None`` when the artifact is absent or not a well-formed schema-v1
    result (e.g. a truncated write). Keeping the summary tolerant means a partial
    result still yields a handoff summary rather than crashing the artifact builder.
    """
    raw = _read_json(path)
    if not raw:
        return None
    try:
        return BenchResult.from_dict(raw)
    except (TypeError, ValueError):
        return None


def _read_tail(path: Path, *, max_chars: int = _TAIL_CHARS) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-max_chars:]
    except OSError:
        return ""


def _path_if_exists(path: Path, root: Path) -> str | None:
    return relpath(path, root) if path.exists() else None


def _attempt_files(artifact_dir: Path, output_dir: Path, commit_sha: str) -> dict[str, str]:
    """Return well-known artifact paths for one attempt."""
    files = {
        "attempt_summary": artifact_dir / "attempt_summary.json",
        "bisect_env": artifact_dir / "bisect_env.json",
        "bisect_runner": artifact_dir / "bisect_runner.json",
        "benchmark_log": artifact_dir / "benchmark.log",
        "bisect_command_log": artifact_dir / "bisect_command.log",
        "live_output": artifact_dir / "live_output.jsonl",
        "docker_command_log": artifact_dir / "docker_command.log",
        "docker_live_output": artifact_dir / "docker_live_output.jsonl",
        "perf_smoke_test_result": artifact_dir / "perf_smoke_test_result.json",
        "perf_smoke_test_info": artifact_dir / "perf_smoke_test_info.json",
        "tooling_capability": artifact_dir / "tooling_capability.json",
        "probe_result": artifact_dir / "probe" / "probe_result.json",
        "probe_events": artifact_dir / "probe" / "probe_events.jsonl",
        "probe_live_output": artifact_dir / "probe" / "live_output.jsonl",
        "install_log": output_dir / "env-cache" / "logs" / f"install-{commit_sha[:12]}.log",
    }
    indexed = {name: rel for name, path in files.items() if (rel := _path_if_exists(path, output_dir))}
    indexed["attempt_summary"] = relpath(files["attempt_summary"], output_dir)
    return indexed


def write_attempt_summary(
    output_dir: Path,
    *,
    plan: dict[str, Any],
    attempt: dict[str, Any],
    commit_sha: str,
    label: str,
    run_idx: int,
    recovery_attempt: int,
    metric_value: float | None,
    recovery_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Write ``attempt_summary.json`` beside one measurement attempt."""
    artifact_dir = Path(str(attempt["artifact_dir"]))
    files = _attempt_files(artifact_dir, output_dir, commit_sha)
    bisect_env = _read_json(artifact_dir / "bisect_env.json")
    probe_result = _read_json(artifact_dir / "probe" / "probe_result.json")
    bench_result = _read_bench_result(artifact_dir / "perf_smoke_test_result.json")
    install_log = output_dir / "env-cache" / "logs" / f"install-{commit_sha[:12]}.log"
    status = "measured" if metric_value is not None else "no_metric"

    summary = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "kind": "attempt_summary",
        # Wall-clock write time; used to order attempts deterministically in
        # load_attempt_summaries (filesystem mtime ties on fast/coarse filesystems).
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "label": label,
        "commit_sha": commit_sha,
        "run_idx": run_idx,
        "recovery_attempt": recovery_attempt,
        "artifact_dir": relpath(artifact_dir, output_dir),
        "task_id": plan.get("task_id"),
        "backend_key": plan.get("backend_key"),
        "metric": plan.get("metric", {}),
        "runner": plan.get("runner", {}),
        "command": attempt.get("command"),
        "command_exit_code": attempt.get("command_exit_code"),
        "timed_out": bool(attempt.get("timed_out", False)),
        "duration_s": attempt.get("duration_s"),
        "note": attempt.get("note"),
        "metric_value": metric_value,
        "env": {
            "status": bisect_env.get("status"),
            "skip_category": bisect_env.get("skip_category"),
            "skip_detail": bisect_env.get("skip_detail"),
            "env_reused": bisect_env.get("env_reused"),
            "env_dir": bisect_env.get("env_dir"),
            "stack_hash": bisect_env.get("stack_hash"),
            "isaacsim_version": bisect_env.get("isaacsim_version"),
            "python_version": bisect_env.get("python_version"),
            "python_requires": bisect_env.get("python_requires"),
        },
        "probe": {
            "status": probe_result.get("status"),
            "decision": probe_result.get("decision"),
        },
        "benchmark": {
            "result_present": bench_result is not None,
            "exit_code": bench_result.exit_code if bench_result else None,
            "failure_phase": bench_result.failure_phase if bench_result else None,
            "raw_fps_mean": bench_result.raw_fps_mean if bench_result else None,
            "stdout_tail": bench_result.stdout_tail if bench_result else None,
        },
        "recovery_events": list(recovery_events or attempt.get("recovery_events") or []),
        "paths": files,
        "evidence": {
            "install_log_tail": _read_tail(install_log),
            "benchmark_log_tail": _read_tail(artifact_dir / "benchmark.log"),
            "bisect_command_log_tail": _read_tail(artifact_dir / "bisect_command.log"),
            "live_output_tail": _read_tail(artifact_dir / "live_output.jsonl"),
            "docker_command_log_tail": _read_tail(artifact_dir / "docker_command.log"),
            "docker_live_output_tail": _read_tail(artifact_dir / "docker_live_output.jsonl"),
            "probe_live_output_tail": _read_tail(artifact_dir / "probe" / "live_output.jsonl"),
        },
    }
    write_json(artifact_dir / "attempt_summary.json", summary)
    return summary


def _combined_evidence(attempt: dict[str, Any]) -> str:
    probe_decision = attempt.get("probe", {}).get("decision") or {}
    parts: list[str] = [
        str(attempt.get("note") or ""),
        str(attempt.get("env", {}).get("skip_detail") or ""),
        str(attempt.get("benchmark", {}).get("stdout_tail") or ""),
        str(probe_decision.get("reason") or ""),
    ]
    parts.extend(str(value or "") for value in attempt.get("evidence", {}).values())
    return "\n".join(parts)


def _blocker(
    attempt: dict[str, Any],
    *,
    category: str,
    phase: str,
    reason: str,
    retryable: bool,
    owner: str,
    suggested_next_steps: list[str],
) -> dict[str, Any]:
    paths = attempt.get("paths", {})
    evidence_paths = [
        path
        for key in (
            "attempt_summary",
            "bisect_env",
            "perf_smoke_test_result",
            "benchmark_log",
            "bisect_command_log",
            "live_output",
            "probe_result",
            "probe_events",
            "install_log",
        )
        if (path := paths.get(key))
    ]
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "phase": phase,
        "category": category,
        "reason": reason,
        "retryable": retryable,
        "owner": owner,
        "actionable_by": [owner],
        "attempt": {
            "artifact_dir": attempt.get("artifact_dir"),
            "label": attempt.get("label"),
            "commit_sha": attempt.get("commit_sha"),
            "run_idx": attempt.get("run_idx"),
            "recovery_attempt": attempt.get("recovery_attempt"),
            "note": attempt.get("note"),
        },
        "evidence": {
            "paths": evidence_paths,
            "excerpt": _combined_evidence(attempt)[-_TAIL_CHARS:],
        },
        "suggested_next_steps": suggested_next_steps,
    }


def classify_blocker(attempt: dict[str, Any]) -> dict[str, Any] | None:
    """Classify the actionable blocker represented by an attempt summary."""
    if attempt.get("status") == "measured" or attempt.get("metric_value") is not None:
        return None

    text = _combined_evidence(attempt)
    lowered = text.lower()
    note = str(attempt.get("note") or "")
    env_category = str(attempt.get("env", {}).get("skip_category") or "")
    probe_status = str(attempt.get("probe", {}).get("status") or "")

    host_blocker = classify_host_blocker(text)
    if host_blocker is not None:
        details = _HOST_BLOCKER_DETAILS[host_blocker]
        return _blocker(
            attempt,
            category=host_blocker,
            phase=str(details["phase"]),
            reason=str(details["reason"]),
            retryable=False,
            owner="outer_agent",
            suggested_next_steps=list(details["next_steps"]),
        )

    if "unrecognized command-line option ‘-m64’" in lowered or "unrecognized command-line option '-m64'" in lowered:
        return _blocker(
            attempt,
            category="dependency_unavailable",
            phase="install",
            reason="A dependency source build emitted an x86-only -m64 flag on this platform.",
            retryable=False,
            owner="platform_unavailable",
            suggested_next_steps=[
                "Run this range on a compatible x86_64 GPU host or provide a platform-compatible dependency wheel.",
                "Treat this commit as unevaluable on the current platform.",
            ],
        )

    if "cannot find gmp" in lowered:
        return _blocker(
            attempt,
            category="install_failed",
            phase="install",
            reason="A source dependency requires GMP development headers/libraries.",
            retryable=True,
            owner="outer_agent",
            suggested_next_steps=[
                "Use a Docker base image or repair image that includes libgmp-dev.",
                "Relaunch the same plan after the base-image prerequisite is available.",
            ],
        )

    if (
        env_category == "perf_smoke_tooling_incompatible"
        or note.startswith("env_skip:perf_smoke_tooling_incompatible")
        or "PERF_SMOKE_TOOLING_INCOMPATIBLE" in text
    ):
        return _blocker(
            attempt,
            category="perf_smoke_tooling_incompatible",
            phase="benchmark_capability",
            reason="Candidate APIs do not satisfy the pinned perf-smoke tooling contract.",
            retryable=False,
            owner="tooling_support_window",
            suggested_next_steps=[
                "Narrow the bisection range to commits supported by this tooling revision.",
                "If an older maintained tooling contract exists, start a separate bisection "
                "with that exact tooling SHA.",
            ],
        )

    missing_module = _MISSING_MODULE_RE.search(text)
    if missing_module:
        module = missing_module.group(1)
        return _blocker(
            attempt,
            category="runtime_incompatible",
            phase="benchmark_import",
            reason=f"Benchmark import failed because Python module {module!r} is missing.",
            retryable=False,
            owner="outer_agent",
            suggested_next_steps=[
                f"Decide whether {module!r} is a benchmark support dependency or a commit dependency.",
                "If it is benchmark support, add it to the reconstructed benchmark-support install set and relaunch.",
            ],
        )

    if "failed to download" in lowered or "failed to extract archive" in lowered:
        return _blocker(
            attempt,
            category="install_failed",
            phase="install",
            reason="Dependency download or extraction failed during environment reconstruction.",
            retryable=True,
            owner="outer_agent",
            suggested_next_steps=[
                "Inspect the install log for network, cache, or filesystem errors.",
                "Relaunch after fixing the host/cache condition.",
            ],
        )

    if env_category == "dependency_unavailable":
        return _blocker(
            attempt,
            category="dependency_unavailable",
            phase="install",
            reason="A pinned dependency is unavailable or unsupported for this environment.",
            retryable=False,
            owner="platform_unavailable",
            suggested_next_steps=["Skip this commit on the current platform or use a compatible environment."],
        )

    if note == "probe_failed:plan_issue" or probe_status == "plan_issue":
        return _blocker(
            attempt,
            category="plan_issue",
            phase="probe",
            reason="The probe identified a benchmark plan issue.",
            retryable=False,
            owner="outer_agent",
            suggested_next_steps=["Review the probe result and relaunch with a corrected task/backend/metric plan."],
        )

    if note == "probe_failed:harness_blocked" or probe_status == "harness_blocked":
        return _blocker(
            attempt,
            category="probe_blocked",
            phase="probe",
            reason="The probe stopped before deterministic benchmarking.",
            retryable=False,
            owner="outer_agent",
            suggested_next_steps=["Review probe_events.jsonl and the referenced logs before relaunching."],
        )

    if env_category == "source_checkout_failed" or note.startswith("env_skip:source_checkout_failed"):
        return _blocker(
            attempt,
            category="source_checkout_failed",
            phase="source_checkout",
            reason="The per-commit source clone/checkout failed before the environment was built.",
            retryable=False,
            owner="outer_agent",
            suggested_next_steps=[
                "Inspect the attempt's live output for gitdir/safe-directory/commit reachability errors.",
                "Relaunch after fixing repository metadata mounts or ensuring the commit is reachable.",
            ],
        )

    if env_category or note.startswith("env_skip:"):
        category = env_category or note.split(":", 1)[1]
        return _blocker(
            attempt,
            category=category,
            phase="install" if category in {"install_failed", "dependency_unavailable"} else "runtime",
            reason=f"Measurement produced environment skip category {category!r}.",
            retryable=category == "install_failed",
            owner="outer_agent",
            suggested_next_steps=["Inspect the attempt summary and raw evidence paths before relaunching."],
        )

    if looks_source_checkout_failure(text):
        return _blocker(
            attempt,
            category="source_checkout_failed",
            phase="source_checkout",
            reason="The per-commit source clone/checkout failed before the environment was built.",
            retryable=True,
            owner="outer_agent",
            suggested_next_steps=[
                "Re-run: the harness materializes a fresh source dir per attempt, which clears "
                "stale locks and partial clones.",
                "If it persists, verify the commit SHA is reachable from the configured repo/remote.",
            ],
        )

    if note:
        return _blocker(
            attempt,
            category="install_failed" if "install" in note else "runtime_incompatible",
            phase="measurement",
            reason=f"Measurement produced no usable metric: {note}.",
            retryable=False,
            owner="outer_agent",
            suggested_next_steps=["Inspect the attempt summary and raw evidence paths before relaunching."],
        )
    return None


def _summary_order_key(entry: tuple[Path, dict[str, Any]]) -> tuple[str, str]:
    # Order by the summary's wall-clock write time, which reflects collection order.
    # Filesystem mtime is unreliable here (it ties on fast/coarse filesystems and then
    # falls back to the path, sorting bad_ref before good_ref); the path is only a final
    # tiebreaker for summaries written within the same clock tick.
    path, payload = entry
    return str(payload.get("generated_at") or ""), path.as_posix()


def load_attempt_summaries(output_dir: Path) -> list[dict[str, Any]]:
    """Load all attempt summaries for a run, ordered by write time."""
    loaded: list[tuple[Path, dict[str, Any]]] = []
    for path in output_dir.glob("measurements/**/attempt_summary.json"):
        payload = _read_json(path)
        if payload:
            loaded.append((path, payload))
    return [payload for _, payload in sorted(loaded, key=_summary_order_key)]


def _summary_attempt_dirs(output_dir: Path, summary: dict[str, Any]) -> set[str]:
    """Return exact attempt dirs referenced by the current run summary."""
    dirs: set[str] = set()
    stats = summary.get("reference_stats", {})
    for key in ("good_attempts", "bad_attempts"):
        for attempt in stats.get(key) or []:
            artifact_dir = attempt.get("artifact_dir")
            if artifact_dir:
                dirs.add(relpath(Path(artifact_dir), output_dir))

    final_attempt = summary.get("final_attempt") or {}
    if artifact_dir := final_attempt.get("artifact_dir"):
        dirs.add(relpath(Path(artifact_dir), output_dir))
    return dirs


def _summary_candidate_commits(summary: dict[str, Any]) -> set[str]:
    """Return candidate commit SHAs referenced by the current run summary."""
    commits = {str(commit) for commit in summary.get("tested_commits") or []}
    for skipped in summary.get("skipped_commits") or []:
        if commit_sha := skipped.get("commit_sha"):
            commits.add(str(commit_sha))
    return commits


def _filter_attempt_summaries(
    output_dir: Path, plan: dict[str, Any], summary: dict[str, Any], attempts: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Keep only attempts that belong to the current summary.

    Work dirs are often reused while iterating on a case. Filtering prevents
    stale attempts from older task/backend experiments from polluting the final
    blocker and artifact index for the active run.
    """
    exact_dirs = _summary_attempt_dirs(output_dir, summary)
    candidate_commits = _summary_candidate_commits(summary)
    task_id = plan.get("task_id")
    backend_key = plan.get("backend_key")

    filtered: list[dict[str, Any]] = []
    for attempt in attempts:
        artifact_dir = attempt.get("artifact_dir")
        if artifact_dir in exact_dirs:
            filtered.append(attempt)
            continue
        if (
            candidate_commits
            and attempt.get("commit_sha") in candidate_commits
            and attempt.get("task_id") == task_id
            and attempt.get("backend_key") == backend_key
        ):
            filtered.append(attempt)
    return filtered or attempts


def _final_artifact_dir_from_summary(summary: dict[str, Any]) -> str | None:
    stats = summary.get("reference_stats", {})
    reason = str(summary.get("reason") or "")
    if reason.startswith("good_ref_measurement_failed"):
        attempts = stats.get("good_attempts") or []
        return attempts[-1].get("artifact_dir") if attempts else None
    if reason.startswith("bad_ref_measurement_failed"):
        attempts = stats.get("bad_attempts") or []
        return attempts[-1].get("artifact_dir") if attempts else None
    return None


def _choose_terminal_blocker(
    output_dir: Path, summary: dict[str, Any], attempts: list[dict[str, Any]], blockers: list[dict[str, Any]]
) -> dict[str, Any] | None:
    if not blockers:
        return None
    final_dir = _final_artifact_dir_from_summary(summary)
    if final_dir:
        final_rel = relpath(Path(final_dir), output_dir)
        for blocker in reversed(blockers):
            if blocker.get("attempt", {}).get("artifact_dir") == final_rel:
                return blocker
    if attempts:
        final_attempt_dir = attempts[-1].get("artifact_dir")
        for blocker in reversed(blockers):
            if blocker.get("attempt", {}).get("artifact_dir") == final_attempt_dir:
                return blocker
    return blockers[-1]


def build_artifact_index(output_dir: Path, attempts: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a role-based index of important run artifacts."""
    top_level_names = [
        "plan.resolved.json",
        "tooling_manifest.json",
        "run_manifest.json",
        "candidates.json",
        "status.json",
        "summary.json",
        "artifact_index.json",
        "reference_measurements.json",
        "grounding.json",
        "bisect_plan.json",
        "audit_log.jsonl",
        "blockers.json",
        "report.md",
    ]
    expected = {"summary.json", "artifact_index.json", "blockers.json", "report.md", "run_manifest.json"}
    top_level = {name: name for name in top_level_names if name in expected or (output_dir / name).exists()}
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "top_level": top_level,
        "attempts": [
            {
                "label": attempt.get("label"),
                "commit_sha": attempt.get("commit_sha"),
                "run_idx": attempt.get("run_idx"),
                "recovery_attempt": attempt.get("recovery_attempt"),
                "artifact_dir": attempt.get("artifact_dir"),
                "status": attempt.get("status"),
                "note": attempt.get("note"),
                "paths": attempt.get("paths", {}),
            }
            for attempt in attempts
        ],
    }


def _one_stack_diff_block(title: str, diff: dict[str, Any]) -> list[str]:
    """Render one component stack diff (range or culprit) as report lines."""
    endpoints = diff.get("isaaclab_commit", {})
    lines = [f"### {title}", "", f"- IsaacLab commits: `{endpoints.get('from')}` -> `{endpoints.get('to')}`"]
    changed = diff.get("changed_components") or {}
    if changed:
        lines.append("- Changed components:")
        for name, delta in changed.items():
            lines.append(f"  - `{name}`: `{delta.get('from')}` -> `{delta.get('to')}`")
    else:
        lines.append("- Changed components: none (pinned stack identical across these endpoints)")
    lines.append("")
    return lines


def _stack_diff_lines(stack_diff: dict[str, Any] | None) -> list[str]:
    """Render the component stack diff section for the report, if present.

    A perf regression can originate in a pinned dependency (Isaac Sim, Kit/renderer,
    PhysX, Newton, Warp) rather than IsaacLab's own source. Surfacing what moved across
    the range (and, when localized, across the culprit commit) tells the reader which
    component to investigate next even though the bisection target is IsaacLab commits.
    """
    if not stack_diff:
        return []
    lines = ["## Component Stack Diff", ""]
    if "culprit" in stack_diff:
        lines.extend(_one_stack_diff_block("Across culprit commit (last good -> first bad)", stack_diff["culprit"]))
    if "range" in stack_diff:
        lines.extend(_one_stack_diff_block("Across full range (good ref -> bad ref)", stack_diff["range"]))
    return lines


def _report_lines(summary: dict[str, Any], terminal_blocker: dict[str, Any] | None) -> list[str]:
    lines = [
        "# Bisection Run Report",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Reason: `{summary.get('reason')}`",
        f"- Good ref: `{summary.get('good_ref')}`",
        f"- Bad ref: `{summary.get('bad_ref')}`",
        f"- Suspected first bad: `{summary.get('suspected_first_bad_commit')}`",
        f"- Last good: `{summary.get('last_good_commit')}`",
        "",
    ]
    lines.extend(_stack_diff_lines(summary.get("stack_diff")))
    if terminal_blocker:
        attempt = terminal_blocker.get("attempt", {})
        lines.extend(
            [
                "## Terminal Blocker",
                "",
                f"- Category: `{terminal_blocker.get('category')}`",
                f"- Phase: `{terminal_blocker.get('phase')}`",
                f"- Owner: `{terminal_blocker.get('owner')}`",
                f"- Retryable: `{terminal_blocker.get('retryable')}`",
                f"- Reason: {terminal_blocker.get('reason')}",
                f"- Attempt: `{attempt.get('artifact_dir')}`",
                "",
                "## Suggested Next Steps",
                "",
            ]
        )
        for step in terminal_blocker.get("suggested_next_steps", []):
            lines.append(f"- {step}")
        lines.extend(["", "## Evidence", ""])
        for path in terminal_blocker.get("evidence", {}).get("paths", []):
            lines.append(f"- `{path}`")
        lines.append("")
    else:
        lines.extend(["## Terminal Blocker", "", "No terminal blocker was identified.", ""])
    return lines


def finalize_run_artifacts(output_dir: Path, plan: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    """Write run-level handoff artifacts and return an enriched summary."""
    attempts = _filter_attempt_summaries(output_dir, plan, summary, load_attempt_summaries(output_dir))
    blockers = [blocker for attempt in attempts if (blocker := classify_blocker(attempt))]
    terminal_blocker = _choose_terminal_blocker(output_dir, summary, attempts, blockers)
    run_manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "plan": {
            "task_id": plan.get("task_id"),
            "backend_key": plan.get("backend_key"),
            "good_ref": plan.get("good_ref"),
            "bad_ref": plan.get("bad_ref"),
            "gpu_model": plan.get("gpu_model"),
            "metric": plan.get("metric", {}),
            "runner": plan.get("runner", {}),
            "measurement": plan.get("measurement", {}),
            "tooling": plan.get("tooling", {}),
        },
        "summary": {
            "status": summary.get("status"),
            "reason": summary.get("reason"),
            "good_ref": summary.get("good_ref"),
            "bad_ref": summary.get("bad_ref"),
            "suspected_first_bad_commit": summary.get("suspected_first_bad_commit"),
            "last_good_commit": summary.get("last_good_commit"),
        },
        "artifact_paths": {
            "summary": "summary.json",
            "artifact_index": "artifact_index.json",
            "blockers": "blockers.json",
            "report": "report.md",
        },
    }
    write_json(output_dir / "run_manifest.json", run_manifest)

    blockers_payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "terminal_blocker": terminal_blocker,
        "blockers": blockers,
    }
    write_json(output_dir / "blockers.json", blockers_payload)

    artifact_index = build_artifact_index(output_dir, attempts)
    write_json(output_dir / "artifact_index.json", artifact_index)

    enriched = {
        **summary,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_manifest": "run_manifest.json",
        "artifact_index": "artifact_index.json",
        "blockers": "blockers.json",
        "report": "report.md",
        "final_attempt": terminal_blocker.get("attempt") if terminal_blocker else None,
        "terminal_blocker": terminal_blocker,
    }
    (output_dir / "report.md").write_text("\n".join(_report_lines(enriched, terminal_blocker)), encoding="utf-8")
    write_json(output_dir / "artifact_index.json", build_artifact_index(output_dir, attempts))
    return enriched
