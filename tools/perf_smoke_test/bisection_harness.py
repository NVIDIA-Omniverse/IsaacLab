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

from bisection.engine import build_candidates, run_bisection, write_status  # noqa: E402
from bisection.io import read_json, write_json  # noqa: E402
from bisection.models import BisectionPlan  # noqa: E402


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
    return parser.parse_args()


def _load_plan(path: Path) -> BisectionPlan:
    return BisectionPlan.from_json(read_json(path))


def main() -> int:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = _load_plan(args.plan)
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
