#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Preliminary bisection agent for performance regressions.

This is intentionally a small orchestration layer. It chooses commits to test,
delegates the actual benchmark execution to a caller-provided command, then reads
the normal perf-smoke artifacts and asks the oracle whether the tested commit is
GOOD, BAD, or SKIP.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

_MODULE_DIR = Path(__file__).parent
_REPO_ROOT = _MODULE_DIR.parents[1]

if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))

from aggregate import _bench_gpu_model, _excluded_frames, _hard_floor  # noqa: E402
from baseline_manager import load_baseline, match_context_from_bench_result  # noqa: E402
from gate_config import load_gate_config  # noqa: E402
from gate_types import BisectVerdict  # noqa: E402
from oracle import compare  # noqa: E402


@dataclasses.dataclass(frozen=True)
class BisectPlan:
    """Input contract for one bisection run."""

    task_id: str
    backend_key: str
    good_ref: str
    bad_ref: str
    runner_command: str
    gpu_model: str = "unknown-gpu"
    baselines_dir: str = "tools/perf_smoke_test/local_baselines"
    gate_config: str = "tools/perf_smoke_test/gate_config.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a preliminary perf-regression bisection.")
    parser.add_argument("--plan", type=Path, help="Path to bisection plan JSON.")
    parser.add_argument("--task_id", default=None, help="Task id to bisect, e.g. Isaac-Cartpole-Direct.")
    parser.add_argument("--backend_key", default=None, help="Backend key to bisect, e.g. physx or newton.")
    parser.add_argument("--good_ref", default=None, help="Known-good git ref/SHA.")
    parser.add_argument("--bad_ref", default=None, help="Known-bad git ref/SHA.")
    parser.add_argument(
        "--runner_command",
        default=None,
        help=(
            "Command template that runs one commit and writes perf-smoke artifacts. "
            "Supported placeholders: {commit_sha}, {task_id}, {backend_key}, {artifact_dir}, {repo_root}."
        ),
    )
    parser.add_argument("--gpu_model", default=None, help="GPU bucket used for baseline lookup.")
    parser.add_argument("--baselines_dir", default=None, help="Flat-file baseline directory.")
    parser.add_argument("--gate_config", default=None, help="Gate config path.")
    parser.add_argument("--output_dir", required=True, type=Path, help="Directory for status/results/artifacts.")
    parser.add_argument("--dry_run", action="store_true", help="Only resolve and print candidate commits.")
    parser.add_argument(
        "--max_tests",
        type=int,
        default=50,
        help="Safety cap on benchmark executions before declaring the run inconclusive.",
    )
    return parser.parse_args()


def _load_plan(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open(encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict):
        raise TypeError(f"Plan at {path} must be a JSON object.")
    return raw


def _coalesce_plan(args: argparse.Namespace) -> BisectPlan:
    raw = _load_plan(args.plan)

    def value(key: str, default: str | None = None) -> str:
        cli_value = getattr(args, key, None)
        chosen = cli_value if cli_value is not None else raw.get(key, default)
        if chosen is None:
            raise ValueError(f"Missing required bisection input: {key}")
        return str(chosen)

    return BisectPlan(
        task_id=value("task_id"),
        backend_key=value("backend_key"),
        good_ref=value("good_ref"),
        bad_ref=value("bad_ref"),
        runner_command=value("runner_command", ""),
        gpu_model=value("gpu_model", "unknown-gpu"),
        baselines_dir=value("baselines_dir", "tools/perf_smoke_test/local_baselines"),
        gate_config=value("gate_config", "tools/perf_smoke_test/gate_config.json"),
    )


def _git(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(["git", *args], cwd=_REPO_ROOT, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"git {' '.join(args)} failed")
    return result


def _resolve_ref(ref: str) -> str:
    result = _git(["rev-parse", "--verify", f"{ref}^{{commit}}"])
    return result.stdout.strip()


def _candidate_commits(good_sha: str, bad_sha: str) -> list[str]:
    ancestry = _git(["merge-base", "--is-ancestor", good_sha, bad_sha], check=False)
    if ancestry.returncode != 0:
        raise RuntimeError(f"Known-good commit {good_sha[:12]} is not an ancestor of bad commit {bad_sha[:12]}.")
    result = _git(["rev-list", "--ancestry-path", "--reverse", f"{good_sha}..{bad_sha}"])
    commits = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not commits or commits[-1] != bad_sha:
        commits.append(bad_sha)
    return commits


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _status(output_dir: Path, **values: Any) -> None:
    payload = {
        "phase": values.pop("phase", "running"),
        "status": values.pop("status", "running"),
        **values,
    }
    _write_json(output_dir / "status.json", payload)


def _format_runner_command(plan: BisectPlan, commit_sha: str, artifact_dir: Path) -> str:
    if not plan.runner_command.strip():
        raise ValueError("--runner_command is required unless --dry_run is set.")
    return plan.runner_command.format(
        commit_sha=shlex.quote(commit_sha),
        task_id=shlex.quote(plan.task_id),
        backend_key=shlex.quote(plan.backend_key),
        artifact_dir=shlex.quote(str(artifact_dir)),
        repo_root=shlex.quote(str(_REPO_ROOT)),
    )


def _run_candidate(plan: BisectPlan, output_dir: Path, commit_sha: str) -> dict[str, Any]:
    short = commit_sha[:12]
    artifact_dir = output_dir / "artifacts" / short / plan.task_id / plan.backend_key
    artifact_dir.mkdir(parents=True, exist_ok=True)
    command = _format_runner_command(plan, commit_sha, artifact_dir)
    command_log = artifact_dir / "bisect_command.log"

    with command_log.open("w", encoding="utf-8") as log_fh:
        log_fh.write(f"$ {command}\n\n")
        result = subprocess.run(command, shell=True, cwd=_REPO_ROOT, stdout=log_fh, stderr=subprocess.STDOUT, text=True)

    bench_result_path = artifact_dir / "perf_smoke_test_result.json"
    record: dict[str, Any] = {
        "commit_sha": commit_sha,
        "artifact_dir": str(artifact_dir),
        "command": command,
        "command_exit_code": result.returncode,
    }
    if result.returncode != 0:
        record.update(
            {
                "bisect_verdict": BisectVerdict.SKIP.value,
                "note": "runner_command_failed",
            }
        )
        return record
    if not bench_result_path.exists():
        record.update(
            {
                "bisect_verdict": BisectVerdict.SKIP.value,
                "note": "missing_perf_smoke_test_result",
            }
        )
        return record

    with bench_result_path.open(encoding="utf-8") as fh:
        bench_result = json.load(fh)

    gate_config = load_gate_config(Path(plan.gate_config))
    gpu_model = _bench_gpu_model(bench_result, plan.gpu_model)
    match_context = match_context_from_bench_result(bench_result, gpu_model=gpu_model)
    baseline = load_baseline(
        Path(plan.baselines_dir),
        gpu_model,
        plan.task_id,
        plan.backend_key,
        match_context=match_context,
    )
    oracle_result = compare(
        bench_result=bench_result,
        baseline=baseline,
        fps_mean_floor=_hard_floor(bench_result, gpu_model, plan.backend_key),
        excluded_frames=_excluded_frames(bench_result),
        artifact_dir=artifact_dir,
        min_block_regression_pct=float(gate_config.get("min_block_regression_pct", 3.0)),
    )
    record.update(
        {
            "oracle_verdict": oracle_result.verdict.value,
            "bisect_verdict": oracle_result.bisect_verdict,
            "measured_fps": oracle_result.measured_fps,
            "baseline_fps": oracle_result.baseline_fps,
            "regression_pct": oracle_result.regression_pct,
            "baseline_sample_count": oracle_result.baseline_sample_count,
            "threshold_source": oracle_result.threshold_source,
            "note": oracle_result.note,
        }
    )
    return record


def _bisect(plan: BisectPlan, output_dir: Path, candidates: list[str], max_tests: int) -> dict[str, Any]:
    low = 0
    high = len(candidates) - 1
    tested: list[str] = []
    result_dir = output_dir / "results"
    best_bad: str | None = None

    while low <= high:
        if len(tested) >= max_tests:
            return {
                "status": "inconclusive",
                "reason": "max_tests_reached",
                "tested_commits": tested,
                "suspected_first_bad_commit": best_bad,
            }

        idx = (low + high) // 2
        commit_sha = candidates[idx]
        tested.append(commit_sha)
        _status(
            output_dir,
            phase="running",
            status="running",
            total_candidates=len(candidates),
            completed_tests=len(tested) - 1,
            current_commit=commit_sha,
            search_low=low,
            search_high=high,
        )
        record = _run_candidate(plan, output_dir, commit_sha)
        _write_json(result_dir / f"{commit_sha[:12]}.json", record)

        verdict = record.get("bisect_verdict")
        if verdict == BisectVerdict.GOOD.value:
            low = idx + 1
        elif verdict == BisectVerdict.BAD.value:
            best_bad = commit_sha
            high = idx - 1
        else:
            return {
                "status": "inconclusive",
                "reason": f"candidate_returned_{verdict or 'unknown'}",
                "tested_commits": tested,
                "suspected_first_bad_commit": best_bad,
                "last_record": record,
            }

    return {
        "status": "completed",
        "reason": "first_bad_found" if best_bad else "no_bad_commit_found",
        "tested_commits": tested,
        "suspected_first_bad_commit": best_bad,
    }


def main() -> int:
    args = _parse_args()
    plan = _coalesce_plan(args)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "plan.resolved.json", dataclasses.asdict(plan))

    good_sha = _resolve_ref(plan.good_ref)
    bad_sha = _resolve_ref(plan.bad_ref)
    candidates = _candidate_commits(good_sha, bad_sha)

    candidate_payload = {
        "good_sha": good_sha,
        "bad_sha": bad_sha,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    _write_json(output_dir / "candidates.json", candidate_payload)

    if args.dry_run:
        _status(
            output_dir,
            phase="dry_run",
            status="completed",
            total_candidates=len(candidates),
            good_sha=good_sha,
            bad_sha=bad_sha,
        )
        print(json.dumps(candidate_payload, indent=2))
        return 0

    summary = _bisect(plan, output_dir, candidates, args.max_tests)
    _write_json(output_dir / "summary.json", summary)
    _status(
        output_dir,
        phase="completed" if summary["status"] == "completed" else "failed",
        status=summary["status"],
        total_candidates=len(candidates),
        tested_count=len(summary.get("tested_commits", [])),
        suspected_first_bad_commit=summary.get("suspected_first_bad_commit"),
        reason=summary.get("reason"),
    )
    return 0 if summary["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
