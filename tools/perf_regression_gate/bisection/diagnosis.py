# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Initial bad-commit diagnosis helpers for the bisection POC."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .git_utils import commit_summary, diff_name_status, diff_stat, git, resolve_ref
from .io import read_json, write_json

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _result_path(run_dir: Path, commit_sha: str) -> Path:
    return run_dir / "results" / f"{commit_sha[:12]}.json"


def _load_result(run_dir: Path, commit_sha: str | None) -> dict[str, Any] | None:
    if not commit_sha:
        return None
    path = _result_path(run_dir, commit_sha)
    return read_json(path) if path.exists() else None


def _parent_of(commit_sha: str) -> str | None:
    result = git(_REPO_ROOT, ["rev-parse", "--verify", f"{commit_sha}^"], check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def _classify_subsystems(changed_files: list[dict[str, str]]) -> list[str]:
    subsystems: set[str] = set()
    for row in changed_files:
        path = row["path"]
        if "/events.py" in path or "/mdp/events" in path:
            subsystems.add("event handling")
        if "/rewards.py" in path or "/mdp/rewards" in path:
            subsystems.add("reward computation")
        if "/observations.py" in path or "/mdp/observations" in path:
            subsystems.add("observation computation")
        if "sensors" in path or "camera" in path.lower() or "vision" in path.lower():
            subsystems.add("sensors/rendering")
        if "newton" in path.lower() or "physx" in path.lower() or "simulation" in path.lower():
            subsystems.add("physics/simulation")
        if "tasks" in path or "envs" in path:
            subsystems.add("task/environment config")
        if "tools/perf_regression_gate" in path:
            subsystems.add("perf-gate tooling")
        if "docker" in path or ".github" in path:
            subsystems.add("CI/runtime infrastructure")
    return sorted(subsystems) if subsystems else ["uncategorized"]


def _fmt(value: Any, decimals: int = 1) -> str:
    return f"{value:.{decimals}f}" if isinstance(value, (int, float)) else "N/A"


def _metric_delta(good: dict[str, Any] | None, bad: dict[str, Any] | None) -> dict[str, Any]:
    good_fps = (good or {}).get("measured_fps")
    bad_fps = (bad or {}).get("measured_fps")
    delta = None
    pct = None
    if isinstance(good_fps, (int, float)) and isinstance(bad_fps, (int, float)):
        delta = bad_fps - good_fps
        pct = (delta / good_fps) * 100.0 if good_fps else None
    return {
        "good_fps": good_fps,
        "bad_fps": bad_fps,
        "delta_fps": delta,
        "delta_pct": pct,
        "bad_regression_pct_vs_baseline": (bad or {}).get("regression_pct"),
        "bad_baseline_fps": (bad or {}).get("baseline_fps"),
    }


def build_diagnosis(run_dir: Path) -> dict[str, Any]:
    """Build a diagnosis payload from a completed bisection run directory."""
    summary = read_json(run_dir / "summary.json")
    plan = read_json(run_dir / "plan.resolved.json")
    first_bad = summary.get("suspected_first_bad_commit")
    if not first_bad:
        return {
            "schema_version": 1,
            "status": "no_first_bad_commit",
            "task_id": plan.get("task_id"),
            "backend_key": plan.get("backend_key"),
            "summary_status": summary.get("status"),
            "summary_reason": summary.get("reason"),
            "last_good_commit": summary.get("last_good_commit"),
            "good_ref": summary.get("good_ref"),
            "bad_ref": summary.get("bad_ref"),
            "tested_commits": summary.get("tested_commits", []),
            "note": (
                "Bisection completed without identifying a first bad commit. This usually means every "
                "tested candidate classified as GOOD or SKIP. See summary.json and results/*.json for "
                "per-candidate verdicts."
            ),
        }
    first_bad = resolve_ref(_REPO_ROOT, str(first_bad))
    parent = _parent_of(first_bad)
    if parent is None:
        raise RuntimeError(f"Could not resolve parent of first bad commit {first_bad}")
    last_good = summary.get("last_good_commit")
    last_good = resolve_ref(_REPO_ROOT, str(last_good)) if last_good else None

    bad_result = _load_result(run_dir, first_bad)
    good_result = _load_result(run_dir, last_good)
    changed_files = diff_name_status(_REPO_ROOT, parent, first_bad)
    subsystems = _classify_subsystems(changed_files)
    metrics = _metric_delta(good_result, bad_result)
    bad_commit = commit_summary(_REPO_ROOT, first_bad)
    parent_commit = commit_summary(_REPO_ROOT, parent)

    likely_cause = (
        f"The first bad commit changed {', '.join(subsystems)} and coincides with a "
        f"{_fmt(metrics.get('bad_regression_pct_vs_baseline'), 2)}% regression versus baseline."
    )
    if "event handling" in subsystems:
        likely_cause += " Event-handling changes are a strong candidate because they can add per-step runtime overhead."
    elif "sensors/rendering" in subsystems:
        likely_cause += " Sensor/rendering changes are a strong candidate because they can change per-frame GPU work."
    elif "CI/runtime infrastructure" in subsystems:
        likely_cause += (
            " Runtime/CI changes should be checked carefully because they may affect measurement environment."
        )

    return {
        "schema_version": 1,
        "task_id": plan.get("task_id"),
        "backend_key": plan.get("backend_key"),
        "first_bad_commit": bad_commit,
        "parent_commit": parent_commit,
        "last_good_commit": last_good,
        "changed_files": changed_files,
        "diff_stat": diff_stat(_REPO_ROOT, parent, first_bad),
        "subsystems": subsystems,
        "metrics": metrics,
        "likely_cause": likely_cause,
        "recommended_next_steps": [
            "Confirm the regression with a real single-cell run on the parent and first bad commit.",
            "Inspect the changed files for per-step work added to the regressed task path.",
            "If the cause is not obvious, capture Nsight Systems traces for parent vs first bad"
            " using the same task/backend.",
        ],
    }


def render_markdown(diagnosis: dict[str, Any]) -> str:
    """Render a diagnosis payload as Markdown."""
    if diagnosis.get("status") == "no_first_bad_commit":
        return "\n".join(
            [
                "# Bisection Diagnosis",
                "",
                "No first bad commit was identified.",
                "",
                f"- Task: `{diagnosis.get('task_id')}`",
                f"- Backend: `{diagnosis.get('backend_key')}`",
                f"- Summary status: {diagnosis.get('summary_status')}",
                f"- Summary reason: {diagnosis.get('summary_reason')}",
                f"- Last tested good commit: `{str(diagnosis.get('last_good_commit') or 'unknown')[:12]}`",
                f"- Good ref: `{str(diagnosis.get('good_ref') or 'unknown')[:12]}`",
                f"- Bad ref: `{str(diagnosis.get('bad_ref') or 'unknown')[:12]}`",
                f"- Tested commits: {len(diagnosis.get('tested_commits', []))}",
                "",
                diagnosis.get("note", ""),
                "",
            ]
        )
    first_bad = diagnosis["first_bad_commit"]
    parent = diagnosis["parent_commit"]
    metrics = diagnosis["metrics"]
    changed = diagnosis["changed_files"]
    lines = [
        "# Bisection Diagnosis",
        "",
        f"- Task: `{diagnosis['task_id']}`",
        f"- Backend: `{diagnosis['backend_key']}`",
        f"- First bad commit: `{first_bad['commit_sha'][:12]}` - {first_bad['subject']}",
        f"- Parent commit: `{parent['commit_sha'][:12]}` - {parent['subject']}",
        f"- Last tested good commit: `{str(diagnosis.get('last_good_commit') or 'unknown')[:12]}`",
        "",
        "## Performance Signal",
        "",
        f"- Last-good measured FPS: {_fmt(metrics.get('good_fps'))}",
        f"- First-bad measured FPS: {_fmt(metrics.get('bad_fps'))}",
        f"- Delta vs last-good: {_fmt(metrics.get('delta_pct'), 2)}%",
        f"- Delta vs baseline: {_fmt(metrics.get('bad_regression_pct_vs_baseline'), 2)}%",
        f"- Baseline FPS: {_fmt(metrics.get('bad_baseline_fps'))}",
        "",
        "## Likely Cause",
        "",
        diagnosis["likely_cause"],
        "",
        "## Changed Subsystems",
        "",
    ]
    lines.extend(f"- {item}" for item in diagnosis["subsystems"])
    lines.extend(["", "## Changed Files", ""])
    lines.extend(f"- `{row['status']}` `{row['path']}`" for row in changed[:20])
    if len(changed) > 20:
        lines.append(f"- ... {len(changed) - 20} more")
    lines.extend(["", "## Diff Stat", "", "```text", diagnosis["diff_stat"] or "(no diff stat)", "```"])
    lines.extend(["", "## Recommended Next Steps", ""])
    lines.extend(f"- {step}" for step in diagnosis["recommended_next_steps"])
    lines.append("")
    return "\n".join(lines)


def write_diagnosis(run_dir: Path) -> dict[str, Any]:
    """Build and write ``diagnosis.json`` and ``diagnosis.md``."""
    diagnosis = build_diagnosis(run_dir)
    write_json(run_dir / "diagnosis.json", diagnosis)
    (run_dir / "diagnosis.md").write_text(render_markdown(diagnosis), encoding="utf-8")
    return diagnosis
