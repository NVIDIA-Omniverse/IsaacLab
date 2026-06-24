#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Create an IsaacLab bisection plan from perf-gate artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_GATE_DIR = Path(__file__).parent
if str(_GATE_DIR) not in sys.path:
    sys.path.insert(0, str(_GATE_DIR))

from bisection.gate_adapter import find_regressed_gate_cells, make_plan_from_gate_cell  # noqa: E402
from bisection.io import write_json  # noqa: E402
from bisection.models import RetryPolicy, RunnerSpec, TimeoutPolicy  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a bisection plan from a perf-gate artifact directory.")
    parser.add_argument("--artifacts_dir", required=True, type=Path)
    parser.add_argument("--good_ref", required=True)
    parser.add_argument("--bad_ref", required=True)
    parser.add_argument(
        "--runner_command",
        default="",
        help="Legacy command template. Prefer structured --runner_mode options for new plans.",
    )
    parser.add_argument(
        "--runner_mode",
        choices=("synthetic", "local-source", "docker-source"),
        default=None,
        help="Structured runner mode for the generated plan.",
    )
    parser.add_argument("--image", default=None, help="Docker image tag for --runner_mode docker-source.")
    parser.add_argument("--source_dir", default="{output_dir}/candidate-source")
    parser.add_argument("--jit_cache", default="{output_dir}/jit-cache")
    parser.add_argument("--kit_cache", default="{output_dir}/kit-cache")
    parser.add_argument("--local_env_dir", default=None)
    parser.add_argument("--ld_preload", default=None)
    parser.add_argument("--runner_extra_arg", action="append", default=[], help="Extra single-commit runner arg.")
    parser.add_argument("--candidate_timeout_s", type=int, default=None)
    parser.add_argument("--max_attempts", type=int, default=1)
    parser.add_argument("--retry_delay_s", type=int, default=0)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--gpu_model", default="L40S")
    parser.add_argument("--baselines_dir", type=Path, default=_GATE_DIR / "local_baselines")
    parser.add_argument("--gate_config", type=Path, default=_GATE_DIR / "gate_config.json")
    parser.add_argument("--task_id", default=None, help="Optional task filter.")
    parser.add_argument("--backend_key", default=None, help="Optional backend filter.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.runner_command and not args.runner_mode:
        raise ValueError("Either --runner_command or --runner_mode is required.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    regressions = find_regressed_gate_cells(
        artifacts_dir=args.artifacts_dir,
        gpu_model=args.gpu_model,
        baselines_dir=args.baselines_dir,
        gate_config=args.gate_config,
    )
    if args.task_id:
        regressions = [row for row in regressions if row.get("task_id") == args.task_id]
    if args.backend_key:
        regressions = [row for row in regressions if row.get("backend_key") == args.backend_key]
    write_json(args.output_dir / "gate_regressions.json", {"regressions": regressions})

    if not regressions:
        print("[bisection_plan_from_gate] no BAD gate cells found")
        return 2

    selected = regressions[0]
    runner = None
    if args.runner_mode:
        runner = RunnerSpec(
            mode=args.runner_mode,
            image=args.image,
            source_dir=args.source_dir,
            jit_cache=args.jit_cache,
            kit_cache=args.kit_cache,
            local_env_dir=args.local_env_dir,
            ld_preload=args.ld_preload,
            extra_args=list(args.runner_extra_arg),
        )
    plan = make_plan_from_gate_cell(
        gate_cell=selected,
        good_ref=args.good_ref,
        bad_ref=args.bad_ref,
        runner_command=args.runner_command,
        runner=runner,
        timeout=TimeoutPolicy(candidate_timeout_s=args.candidate_timeout_s),
        retry=RetryPolicy(max_attempts=max(1, args.max_attempts), retry_delay_s=max(0, args.retry_delay_s)),
        gpu_model=args.gpu_model,
        baselines_dir=args.baselines_dir,
        gate_config=args.gate_config,
    )
    plan_path = args.output_dir / "plan.json"
    write_json(plan_path, plan.to_json())
    print(
        "[bisection_plan_from_gate] "
        f"selected {selected['task_id']}/{selected['backend_key']} "
        f"regression={selected.get('regression_pct')} -> {plan_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
