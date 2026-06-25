#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CLI for the IsaacLab bisection harness POC."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_GATE_DIR = Path(__file__).parent
if str(_GATE_DIR) not in sys.path:
    sys.path.insert(0, str(_GATE_DIR))

from bisection.engine import build_candidates, run_bisection, run_local_bisection, write_status  # noqa: E402
from bisection.io import read_json, write_json  # noqa: E402
from bisection.models import BisectionPlan, MeasurementPolicy, RetryPolicy, RunnerSpec, TimeoutPolicy  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IsaacLab performance bisection harness.")
    sub = parser.add_subparsers(dest="command", required=True)

    dry = sub.add_parser("dry-run", help="Resolve a plan and list candidate commits.")
    dry.add_argument("--plan", required=True, type=Path)
    dry.add_argument("--output_dir", required=True, type=Path)

    run = sub.add_parser("run", help="Run the bisection loop.")
    run.add_argument("--plan", required=True, type=Path)
    run.add_argument("--output_dir", required=True, type=Path)
    run.add_argument("--max_tests", type=int, default=50)

    local = sub.add_parser("run-local", help="Run local paired-reference bisection.")
    local.add_argument("--good_ref", required=True)
    local.add_argument("--bad_ref", required=True)
    local.add_argument("--task_id", required=True)
    local.add_argument("--backend_key", required=True)
    local.add_argument("--work_dir", required=True, type=Path)
    local.add_argument("--runner_mode", choices=("synthetic", "local-source", "docker-source"), default="local-source")
    local.add_argument("--gpu_model", default="unknown-gpu")
    local.add_argument("--image", default=None)
    local.add_argument("--source_dir", default="{output_dir}/sources/{commit_sha}")
    local.add_argument("--jit_cache", default="{output_dir}/jit-cache")
    local.add_argument("--kit_cache", default="{output_dir}/kit-cache")
    local.add_argument("--local_env_dir", default=None)
    local.add_argument("--ld_preload", default=None)
    local.add_argument("--runner_extra_arg", action="append", default=[])
    local.add_argument("--reference_runs", type=int, default=3)
    local.add_argument("--max_reference_runs", type=int, default=7)
    local.add_argument("--candidate_runs", type=int, default=1)
    local.add_argument("--max_candidate_runs", type=int, default=3)
    local.add_argument("--min_regression_pct", type=float, default=5.0)
    local.add_argument("--gray_zone_pct", type=float, default=1.0)
    local.add_argument("--reference_noise_multiplier", type=float, default=2.0)
    local.add_argument("--max_reference_spread_pct", type=float, default=10.0)
    local.add_argument("--candidate_timeout_s", type=int, default=None)
    local.add_argument("--max_tests", type=int, default=50)
    local.add_argument("--baselines_dir", type=Path, default=_GATE_DIR / "local_baselines")
    local.add_argument("--gate_config", type=Path, default=_GATE_DIR / "gate_config.json")
    return parser.parse_args()


def _load_plan(path: Path) -> BisectionPlan:
    return BisectionPlan.from_json(read_json(path))


def _plan_from_local_args(args: argparse.Namespace) -> BisectionPlan:
    """Build a local paired-reference plan from CLI arguments."""
    return BisectionPlan(
        task_id=args.task_id,
        backend_key=args.backend_key,
        good_ref=args.good_ref,
        bad_ref=args.bad_ref,
        gpu_model=args.gpu_model,
        baselines_dir=str(args.baselines_dir),
        gate_config=str(args.gate_config),
        runner=RunnerSpec(
            mode=args.runner_mode,
            image=args.image,
            source_dir=args.source_dir,
            jit_cache=args.jit_cache,
            kit_cache=args.kit_cache,
            local_env_dir=args.local_env_dir,
            ld_preload=args.ld_preload,
            extra_args=list(args.runner_extra_arg),
        ),
        timeout=TimeoutPolicy(candidate_timeout_s=args.candidate_timeout_s),
        retry=RetryPolicy(),
        measurement=MeasurementPolicy(
            reference_runs=args.reference_runs,
            max_reference_runs=args.max_reference_runs,
            candidate_runs=args.candidate_runs,
            max_candidate_runs=args.max_candidate_runs,
            min_regression_pct=args.min_regression_pct,
            gray_zone_pct=args.gray_zone_pct,
            reference_noise_multiplier=args.reference_noise_multiplier,
            max_reference_spread_pct=args.max_reference_spread_pct,
        ),
    )


def main() -> int:
    args = _parse_args()
    output_dir = (args.work_dir if args.command == "run-local" else args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = _plan_from_local_args(args) if args.command == "run-local" else _load_plan(args.plan)
    write_json(output_dir / "plan.resolved.json", plan.to_json())

    if args.command == "dry-run":
        candidate_payload = build_candidates(plan)
        write_json(output_dir / "candidates.json", candidate_payload)
        write_status(
            output_dir,
            phase="dry_run",
            status="completed",
            total_candidates=candidate_payload["candidate_count"],
            good_sha=candidate_payload["good_sha"],
            bad_sha=candidate_payload["bad_sha"],
        )
        print(f"[bisection_harness] candidates={candidate_payload['candidate_count']} -> {output_dir}")
        return 0

    if args.command == "run-local":
        summary = run_local_bisection(plan, output_dir, max_tests=args.max_tests)
    else:
        summary = run_bisection(plan, output_dir, max_tests=args.max_tests)
    write_status(
        output_dir,
        phase="completed" if summary.status == "completed" else "failed",
        status=summary.status,
        reason=summary.reason,
        tested_count=len(summary.tested_commits),
        suspected_first_bad_commit=summary.suspected_first_bad_commit,
        last_good_commit=summary.last_good_commit,
    )
    print(
        "[bisection_harness] "
        f"status={summary.status} first_bad={summary.suspected_first_bad_commit} "
        f"last_good={summary.last_good_commit}"
    )
    return 0 if summary.status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
