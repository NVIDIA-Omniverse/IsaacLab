#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CLI for the IsaacLab bisection harness POC."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_GATE_DIR = Path(__file__).parent
_REPO_ROOT = _GATE_DIR.parents[1]
if str(_GATE_DIR) not in sys.path:
    sys.path.insert(0, str(_GATE_DIR))

from bisection.engine import build_candidates, run_local_bisection, write_status  # noqa: E402
from bisection.git_utils import resolve_ref  # noqa: E402
from bisection.io import read_json, write_json  # noqa: E402
from bisection.measurement import measure_commit, write_measurement_preflight  # noqa: E402
from bisection.models import (  # noqa: E402
    BisectionPlan,
    MeasurementPolicy,
    MetricSpec,
    RetryPolicy,
    RunnerSpec,
    TaskSpec,
    TimeoutPolicy,
)
from bisection.progress import PROGRESS_MODES, configure_progress, format_metric  # noqa: E402
from bisection.tooling import ToolingError, materialize_tooling_snapshot, resolve_tooling_plan  # noqa: E402


def _add_progress_argument(parser: argparse.ArgumentParser) -> None:
    """Add the common human-readable progress option to a workflow parser."""
    parser.add_argument(
        "--progress",
        choices=PROGRESS_MODES,
        default="compact",
        help="Terminal progress detail: compact (default), verbose setup milestones, or quiet.",
    )


def _sampling_warnings(plan: BisectionPlan) -> list[str]:
    """Return operator warnings for deliberately low-confidence sampling policies."""
    warnings: list[str] = []
    if plan.measurement.reference_runs < 2:
        warnings.append(
            "reference_runs=1 cannot estimate reference noise; use the default 3 or more for natural regressions"
        )
    if plan.measurement.warmup_runs < 1:
        warnings.append(
            "warmup_runs=0 includes cold-start effects; use the default process warmup for steady-state results"
        )
    return warnings


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IsaacLab performance bisection harness.")
    sub = parser.add_subparsers(dest="command", required=True)

    dry = sub.add_parser("dry-run", help="Resolve a plan and list candidate commits.")
    dry.add_argument("--plan", required=True, type=Path)
    dry.add_argument("--output_dir", required=True, type=Path)
    dry.add_argument(
        "--tooling_ref",
        default=None,
        help="Full tooling commit SHA for authoritative runs, or WORKTREE for non-authoritative development.",
    )
    _add_progress_argument(dry)

    local = sub.add_parser(
        "run-local",
        aliases=("bisect-range",),
        help="Run paired-reference bisection on the current host (local or dedicated).",
    )
    local.add_argument(
        "--plan",
        type=Path,
        default=None,
        help="Load a full BisectionPlan JSON instead of building one from flags (Option B, path 2). "
        "--work_dir still applies; the individual plan-building flags below are ignored when this is set.",
    )
    local.add_argument("--good_ref", default=None)
    local.add_argument("--bad_ref", default=None)
    local.add_argument("--task_id", default=None)
    local.add_argument("--backend_key", default=None)
    local.add_argument("--work_dir", required=True, type=Path)
    local.add_argument(
        "--tooling_ref",
        default=None,
        help="Full tooling commit SHA for authoritative runs, or WORKTREE for non-authoritative development.",
    )
    local.add_argument(
        "--runner_mode",
        choices=("synthetic", "local-source", "docker-source", "local-reconstruct", "docker-reconstruct"),
        default="local-source",
    )
    local.add_argument("--gpu_model", default="unknown-gpu")
    local.add_argument("--image", default=None)
    local.add_argument("--source_dir", default="{output_dir}/sources/{commit_sha}")
    local.add_argument("--jit_cache", default="{output_dir}/jit-cache")
    local.add_argument("--kit_cache", default="{output_dir}/kit-cache")
    local.add_argument("--local_env_dir", default=None)
    local.add_argument("--ld_preload", default=None)
    local.add_argument("--runner_extra_arg", action="append", default=[])
    local.add_argument(
        "--synthetic_first_bad_ref",
        default=None,
        help="For --runner_mode synthetic only: commit ref treated as the ground-truth first-bad "
        "commit (defaults to --bad_ref). Lets a synthetic run rehearse commit traversal and binary "
        "search over a real commit range/task/backend with a chosen regression point, at zero "
        "GPU/Docker cost, before committing to a real run.",
    )
    local.add_argument(
        "--synthetic_good_value",
        type=float,
        default=None,
        help="For --runner_mode synthetic only: metric value for commits before the ground-truth "
        "first-bad commit (defaults to a value derived from --regression_direction).",
    )
    local.add_argument(
        "--synthetic_bad_value",
        type=float,
        default=None,
        help="For --runner_mode synthetic only: metric value for the ground-truth first-bad commit "
        "and its descendants (defaults to a value derived from --regression_direction).",
    )
    local.add_argument("--metric_name", default="raw_fps_mean", help="Human-readable metric name.")
    local.add_argument(
        "--metric_path",
        default="raw_fps_mean",
        help="Dotted result path, e.g. raw_fps_mean, runtime_resources.gpu_mem_used_mb, "
        "runtime_resources.cpu_util_pct, or runtime_resources.system_ram_peak_mb.",
    )
    local.add_argument(
        "--regression_direction",
        choices=("decrease", "increase"),
        default=None,
        help="Whether a lower or higher value regresses. Defaults to increase for resource usage "
        "and decrease otherwise.",
    )
    local.add_argument("--metric_unit", default=None, help="Optional display unit for the selected metric.")
    # Inline task definition (Option B, path 1): describe the workload here instead of in
    # tasks.json. Omitted fields fall back to a tasks.json entry if one exists; --num_envs
    # is required to bisect a task that is not registered.
    local.add_argument("--num_envs", type=int, default=None, help="Inline task.num_envs.")
    local.add_argument("--num_frames", type=int, default=None, help="Inline task.num_frames (default 300 inline).")
    local.add_argument("--warmup_frames", type=int, default=None, help="Inline task.warmup_frames.")
    local.add_argument("--seed", type=int, default=None, help="Inline task.seed (default 42 inline).")
    local.add_argument(
        "--camera_resolution",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=None,
        help="Inline task.camera_resolution as two ints, e.g. --camera_resolution 64 64.",
    )
    local.add_argument(
        "--timeout_minutes", type=int, default=None, help="Inline task.timeout_minutes (default 30 inline)."
    )
    local.add_argument(
        "--hydra_arg",
        action="append",
        default=[],
        metavar="ARG",
        help="Extra Hydra CLI arg passed verbatim to the benchmark (repeatable). When any are given "
        "they replace the backend-derived presets, e.g. --hydra_arg presets=cube,newton,rgb64.",
    )
    local.add_argument("--reference_runs", type=int, default=3)
    local.add_argument("--max_reference_runs", type=int, default=7)
    local.add_argument("--candidate_runs", type=int, default=1)
    local.add_argument("--max_candidate_runs", type=int, default=3)
    local.add_argument(
        "--warmup_runs",
        type=int,
        choices=(0, 1),
        default=1,
        help="Run one process warmup per commit before measured runs (default 1); use 0 only "
        "for deliberate cold-start diagnostics. Warmups are recorded but excluded from statistics.",
    )
    local.add_argument("--min_regression_pct", type=float, default=5.0)
    local.add_argument("--gray_zone_pct", type=float, default=1.0)
    local.add_argument("--reference_noise_multiplier", type=float, default=2.0)
    local.add_argument("--max_reference_spread_pct", type=float, default=10.0)
    local.add_argument("--candidate_timeout_s", type=int, default=None)
    local.add_argument("--max_tests", type=int, default=50)
    local.add_argument(
        "--recovery",
        choices=("none", "deterministic", "llm"),
        default="deterministic",
        help="Recovery policy for benchmark-execution friction. 'deterministic' is the default "
        "model-free failure->inspect/retry->skip loop; 'llm' delegates the recovery decision to a "
        "model (needs --model); 'none' accepts the first outcome as-is. Recovery never affects verdicts.",
    )
    local.add_argument(
        "--recovery_max_retries",
        type=int,
        default=2,
        help="Max recovery retries per measurement before a friction outcome is accepted as a skip.",
    )
    local.add_argument(
        "--model",
        default=None,
        help="Model name for --recovery llm or --probe llm (e.g. an OpenAI-compatible chat model).",
    )
    local.add_argument(
        "--base_url",
        default=None,
        help="OpenAI-compatible base URL for --recovery llm (defaults to the provider default).",
    )
    local.add_argument(
        "--api_key_env",
        default="OPENAI_API_KEY",
        help="Environment variable holding the API key for --recovery llm or --probe llm.",
    )
    local.add_argument(
        "--probe",
        choices=("none", "llm"),
        default="none",
        help="Pre-benchmark container validation probe. 'llm' lets the orchestrator inspect live output "
        "and run bounded debug commands before deterministic benchmarking.",
    )
    local.add_argument(
        "--probe_max_attempts",
        type=int,
        default=3,
        help="Max LLM probe decisions before the probe reports a harness blocker.",
    )
    _add_progress_argument(local)

    commit = sub.add_parser(
        "benchmark-commit",
        help="Reconstruct and benchmark one commit without running binary search.",
    )
    commit.add_argument("--plan", type=Path, default=None, help="Optional BisectionPlan JSON supplying defaults.")
    commit.add_argument("--commit", required=True, help="Commit SHA/ref to benchmark.")
    commit.add_argument("--work_dir", required=True, type=Path)
    commit.add_argument(
        "--tooling_ref",
        default=None,
        help="Full tooling commit SHA for authoritative runs, or WORKTREE for non-authoritative development.",
    )
    commit.add_argument("--task_id", default=None)
    commit.add_argument("--backend_key", default=None)
    commit.add_argument(
        "--runner_mode",
        choices=("synthetic", "local-source", "docker-source", "local-reconstruct", "docker-reconstruct"),
        default="local-reconstruct",
    )
    commit.add_argument("--gpu_model", default="unknown-gpu")
    commit.add_argument("--image", default=None)
    commit.add_argument("--source_dir", default="{output_dir}/sources/{commit_sha}")
    commit.add_argument("--jit_cache", default="{output_dir}/jit-cache")
    commit.add_argument("--kit_cache", default="{output_dir}/kit-cache")
    commit.add_argument("--local_env_dir", default=None)
    commit.add_argument("--ld_preload", default=None)
    commit.add_argument("--runner_extra_arg", action="append", default=[])
    commit.add_argument("--num_envs", type=int, default=None)
    commit.add_argument("--num_frames", type=int, default=None)
    commit.add_argument("--warmup_frames", type=int, default=None)
    commit.add_argument("--seed", type=int, default=None)
    commit.add_argument("--camera_resolution", type=int, nargs=2, default=None)
    commit.add_argument("--timeout_minutes", type=int, default=None)
    commit.add_argument("--hydra_arg", action="append", default=[])
    commit.add_argument("--metric_name", default="raw_fps_mean")
    commit.add_argument(
        "--metric_path",
        default="raw_fps_mean",
        help="Dotted result path, including runtime_resources.cpu_util_pct or runtime_resources.system_ram_peak_mb.",
    )
    commit.add_argument("--regression_direction", choices=("decrease", "increase"), default=None)
    commit.add_argument("--metric_unit", default=None)
    commit.add_argument("--runs", type=int, default=3)
    commit.add_argument("--max_runs", type=int, default=7)
    commit.add_argument("--warmup_runs", type=int, choices=(0, 1), default=1)
    commit.add_argument("--max_reference_spread_pct", type=float, default=10.0)
    commit.add_argument("--candidate_timeout_s", type=int, default=None)
    commit.add_argument("--recovery", choices=("none", "deterministic", "llm"), default="deterministic")
    commit.add_argument("--recovery_max_retries", type=int, default=2)
    commit.add_argument("--model", default=None)
    commit.add_argument("--base_url", default=None)
    commit.add_argument("--api_key_env", default="OPENAI_API_KEY")
    commit.add_argument("--probe", choices=("none", "llm"), default="none")
    commit.add_argument("--probe_max_attempts", type=int, default=3)
    _add_progress_argument(commit)

    selftest = sub.add_parser(
        "recovery-selftest",
        help="Send one sample friction case through a recovery policy (verifies an LLM endpoint end-to-end).",
    )
    selftest.add_argument("--recovery", choices=("none", "deterministic", "llm"), default="llm")
    selftest.add_argument("--recovery_max_retries", type=int, default=2)
    selftest.add_argument("--model", default=None)
    selftest.add_argument("--base_url", default=None)
    selftest.add_argument("--api_key_env", default="OPENAI_API_KEY")
    selftest.add_argument(
        "--note",
        default="runner_command_failed",
        help="Friction note for the sample context (e.g. env_skip:runtime_incompatible, candidate_timeout).",
    )

    probe_selftest = sub.add_parser(
        "probe-selftest",
        help="Send one sample container-validation case through the LLM probe (verifies an endpoint end-to-end).",
    )
    probe_selftest.add_argument("--model", default=None, help="Model name for the probe (required).")
    probe_selftest.add_argument("--base_url", default=None)
    probe_selftest.add_argument("--api_key_env", default="OPENAI_API_KEY")
    probe_selftest.add_argument("--probe_max_attempts", type=int, default=3)
    probe_selftest.add_argument(
        "--scenario",
        choices=("backend_mismatch", "ready", "install_download_failed", "missing_gmp"),
        default="backend_mismatch",
        help="Which sample setup situation to show the probe.",
    )
    return parser.parse_args()


def _load_plan(path: Path) -> BisectionPlan:
    return BisectionPlan.from_json(read_json(path))


def _build_recovery_policy(args: argparse.Namespace):
    """Construct the recovery policy selected on the CLI.

    The ``llm`` policy is imported lazily so the model-free deterministic path (the
    default, and the open-source baseline) never requires an LLM client to be present.
    """
    from bisection.recovery import DeterministicRecoveryPolicy, NoRecoveryPolicy

    if args.recovery == "none":
        return NoRecoveryPolicy()
    if args.recovery == "deterministic":
        return DeterministicRecoveryPolicy(max_attempts=args.recovery_max_retries)
    if not args.model:
        raise SystemExit("--recovery llm requires --model")
    from bisection.recovery_llm import LLMRecoveryPolicy

    return LLMRecoveryPolicy(
        model=args.model,
        base_url=args.base_url,
        max_attempts=args.recovery_max_retries,
        api_key_env=args.api_key_env,
    )


def _build_probe_policy(args: argparse.Namespace):
    """Construct the optional pre-benchmark probe policy selected on the CLI."""
    if args.probe == "none":
        return None
    if not args.model:
        raise SystemExit("--probe llm requires --model")
    from bisection.probe import LLMProbePolicy

    return LLMProbePolicy(
        model=args.model,
        base_url=args.base_url,
        max_attempts=args.probe_max_attempts,
        api_key_env=args.api_key_env,
    )


def _run_recovery_selftest(args: argparse.Namespace) -> int:
    """Send one sample friction case through the selected policy and print the decision.

    For ``--recovery llm`` this actually calls the configured endpoint, so it is the
    quickest way to confirm a model/base_url/api_key is wired correctly before
    committing to a full bisection run.
    """
    from bisection.recovery import RecoveryContext

    # Use a representative log tail for the chosen note so the self-test exercises the
    # policy the way a real failure would (the log is a primary signal for the LLM).
    log_tails = {
        "candidate_timeout": "[selftest] app still logging progress at timeout; last line: 'compiling kernels 87%'",
        "env_skip:runtime_incompatible": "[selftest] ImportError: no module named isaacsim",
        "env_skip:install_failed": "[selftest] error: failed to download wheel (connection reset by peer)",
        "env_skip:dependency_unavailable": "[selftest] No solution found: isaacsim==6.0.0-dev2 is not on the index",
        "runner_command_failed": "[selftest] Segmentation fault (core dumped) during headless startup",
    }
    policy = _build_recovery_policy(args)
    ctx = RecoveryContext(
        commit_sha="0" * 40,
        label="selftest",
        run_idx=1,
        attempt=0,
        note=args.note,
        exit_code=1,
        timed_out=args.note == "candidate_timeout",
        artifact_dir=Path("."),
        log_tail=log_tails.get(args.note, f"[selftest] simulated failure with note={args.note}"),
    )
    decision = policy.decide(ctx)
    print(json.dumps({"policy": type(policy).__name__, "note": args.note, "decision": decision.to_json()}, indent=2))
    return 0


def _run_probe_selftest(args: argparse.Namespace) -> int:
    """Send one sample container-validation case through the LLM probe and print the decision.

    This makes a real call to the configured endpoint, so it is the quickest way to
    confirm a model/base_url/api_key is wired correctly for ``--probe llm`` before a
    full bisection run.
    """
    if not args.model:
        raise SystemExit("probe-selftest requires --model")
    from bisection.probe import LLMProbePolicy, ProbeContext

    scenarios = {
        "backend_mismatch": {
            "backend_key": "newton_newton_renderer",
            "plan": {
                "task_id": "Isaac-Dexsuite-Kuka-Allegro-Lift-v0",
                "backend_key": "newton_newton_renderer",
                "task": {"hydra_args": ["presets=cube,single_camera,newton,newton_renderer,rgb64"]},
            },
            "live_output_tail": (
                '{"event": "output", "line": "[INFO] resolved physics backend: physx"}\n'
                '{"event": "output", "line": "[INFO] renderer preset: physx_newton_renderer"}\n'
            ),
            "install_log_tail": "Successfully installed isaacsim newton warp\n",
        },
        "ready": {
            "backend_key": "newton",
            "plan": {"task_id": "Isaac-Cartpole-Direct", "backend_key": "newton"},
            "live_output_tail": (
                '{"event": "output", "line": "[INFO] Kit started, backend=newton"}\n'
                '{"event": "output", "line": "[INFO] benchmark warmup complete"}\n'
            ),
            "install_log_tail": "Successfully installed isaacsim newton warp\n",
        },
        "install_download_failed": {
            "backend_key": "newton",
            "plan": {"task_id": "Isaac-Cartpole-Direct", "backend_key": "newton"},
            "live_output_tail": (
                '{"event": "output", "line": "Downloading isaacsim wheel..."}\n'
                '{"event": "output", "line": "error: connection reset by peer"}\n'
            ),
            "install_log_tail": "error: failed to download wheel (connection reset by peer)\n",
        },
        "missing_gmp": {
            "backend_key": "newton_newton_renderer",
            "plan": {
                "task_id": "Isaac-Dexsuite-Kuka-Allegro-Lift-v0",
                "backend_key": "newton_newton_renderer",
                "runner": {"mode": "docker-reconstruct", "image": "isaaclab-bisect:base"},
            },
            "live_output_tail": '{"event": "output", "line": "ENV_SKIP=install_failed detail=Cannot find GMP"}\n',
            "install_log_tail": (
                "CMake Error at src/fTetWild/CMakeLists.txt:86 (message):\n"
                "  Cannot find GMP\n"
                "hint: `pytetwild` was included because `isaaclab` depends on `pytetwild`\n"
            ),
        },
    }
    scenario = scenarios[args.scenario]
    policy = LLMProbePolicy(
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        max_attempts=args.probe_max_attempts,
    )
    ctx = ProbeContext(
        commit_sha="0" * 40,
        task_id=str(scenario["plan"]["task_id"]),
        backend_key=str(scenario["backend_key"]),
        artifact_dir=Path("."),
        plan=scenario["plan"],
        live_output_tail=str(scenario["live_output_tail"]),
        install_log_tail=str(scenario["install_log_tail"]),
    )
    decision = policy.decide(ctx)
    print(
        json.dumps(
            {"policy": type(policy).__name__, "scenario": args.scenario, "decision": decision.to_json()}, indent=2
        )
    )
    return 0


def _require_local_plan_args(args: argparse.Namespace) -> argparse.Namespace:
    """Validate the flags needed to build a plan when ``--plan`` is not supplied."""
    missing = [name for name in ("good_ref", "bad_ref", "task_id", "backend_key") if getattr(args, name) is None]
    if missing:
        raise SystemExit("bisect-range requires --" + ", --".join(missing) + " (or pass a full plan with --plan).")
    return args


def _metric_spec_from_args(args: argparse.Namespace) -> MetricSpec:
    """Build a metric spec with safe defaults for canonical resource metrics."""
    path = args.metric_path
    resource_metric = path.startswith("runtime_resources.")
    direction = args.regression_direction or ("increase" if resource_metric else "decrease")
    unit = args.metric_unit
    if unit is None:
        if "_pct_" in path or path.endswith("_pct"):
            unit = "%"
        elif "_gb_" in path or path.endswith("_gb"):
            unit = "GB"
        elif "_mb_" in path or path.endswith("_mb"):
            unit = "MB"
        elif path == "raw_fps_mean":
            unit = "fps"
    return MetricSpec(name=args.metric_name, result_path=path, regression_direction=direction, unit=unit)


def _plan_from_local_args(args: argparse.Namespace) -> BisectionPlan:
    """Build a local paired-reference plan from CLI arguments."""
    return BisectionPlan(
        task_id=args.task_id,
        backend_key=args.backend_key,
        good_ref=args.good_ref,
        bad_ref=args.bad_ref,
        gpu_model=args.gpu_model,
        runner=RunnerSpec(
            mode=args.runner_mode,
            image=args.image,
            source_dir=args.source_dir,
            jit_cache=args.jit_cache,
            kit_cache=args.kit_cache,
            local_env_dir=args.local_env_dir,
            ld_preload=args.ld_preload,
            extra_args=list(args.runner_extra_arg),
            synthetic_first_bad_ref=args.synthetic_first_bad_ref,
            synthetic_good_value=args.synthetic_good_value,
            synthetic_bad_value=args.synthetic_bad_value,
        ),
        task=TaskSpec(
            num_envs=args.num_envs,
            num_frames=args.num_frames,
            warmup_frames=args.warmup_frames,
            seed=args.seed,
            camera_resolution=list(args.camera_resolution) if args.camera_resolution is not None else None,
            timeout_minutes=args.timeout_minutes,
            hydra_args=list(args.hydra_arg),
        ),
        metric=_metric_spec_from_args(args),
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
            warmup_runs=args.warmup_runs,
        ),
    )


def _plan_from_benchmark_args(args: argparse.Namespace) -> BisectionPlan:
    """Build a single-commit measurement plan from CLI arguments."""
    missing = [name for name in ("task_id", "backend_key") if getattr(args, name) is None]
    if missing:
        raise SystemExit("benchmark-commit requires --" + ", --".join(missing) + " (or pass a full plan with --plan).")
    return BisectionPlan(
        task_id=args.task_id,
        backend_key=args.backend_key,
        good_ref=args.commit,
        bad_ref=args.commit,
        gpu_model=args.gpu_model,
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
        task=TaskSpec(
            num_envs=args.num_envs,
            num_frames=args.num_frames,
            warmup_frames=args.warmup_frames,
            seed=args.seed,
            camera_resolution=list(args.camera_resolution) if args.camera_resolution is not None else None,
            timeout_minutes=args.timeout_minutes,
            hydra_args=list(args.hydra_arg),
        ),
        metric=_metric_spec_from_args(args),
        timeout=TimeoutPolicy(candidate_timeout_s=args.candidate_timeout_s),
        retry=RetryPolicy(),
        measurement=MeasurementPolicy(
            reference_runs=max(1, args.runs),
            max_reference_runs=max(1, args.max_runs),
            max_reference_spread_pct=max(0.0, args.max_reference_spread_pct),
            warmup_runs=max(0, args.warmup_runs),
        ),
    )


def _relaunch_tooling_fields(plan: BisectionPlan) -> dict:
    """Return SHA-only portability metadata for relaunch artifacts."""
    tooling = plan.tooling
    authoritative = bool(tooling and tooling.authoritative)
    if authoritative and tooling is not None:
        note = (
            "Copy plan.resolved.json to the second host, ensure its clone contains "
            f"tooling commit {tooling.source_commit_sha}, then run this command."
        )
    else:
        note = (
            "Development-only WORKTREE plan: commit the tooling and generate a new authoritative plan "
            "before relaunching on another host."
        )
    return {
        "authoritative": authoritative,
        "required_tooling_sha": tooling.source_commit_sha if tooling else None,
        "note": note,
    }


def main() -> int:
    args = _parse_args()
    if args.command == "recovery-selftest":
        return _run_recovery_selftest(args)
    if args.command == "probe-selftest":
        return _run_probe_selftest(args)
    range_commands = ("run-local", "bisect-range")
    progress = configure_progress(args.progress)
    os.environ["PERF_BISECT_PROGRESS"] = args.progress
    output_dir = (args.work_dir if args.command in (*range_commands, "benchmark-commit") else args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.command in range_commands:
        plan = _load_plan(args.plan) if args.plan else _plan_from_local_args(_require_local_plan_args(args))
    elif args.command == "benchmark-commit":
        plan = _load_plan(args.plan) if args.plan else _plan_from_benchmark_args(args)
    else:
        plan = _load_plan(args.plan)
    progress.event(
        "START",
        f"{args.command} task={plan.task_id}/{plan.backend_key} "
        f"runner={plan.runner.mode if plan.runner else 'unknown'}",
    )
    if args.command in range_commands:
        for warning in _sampling_warnings(plan):
            progress.event("WARNING", warning)
    progress.event("TOOLING", "resolving pinned benchmark tooling", verbose_only=True)
    try:
        if plan.tooling is None:
            plan = resolve_tooling_plan(plan, _REPO_ROOT, tooling_ref=args.tooling_ref)
        materialize_tooling_snapshot(plan, _REPO_ROOT, output_dir)
    except ToolingError as exc:
        write_json(
            output_dir / "tooling_blocker.json",
            {
                "status": "blocked",
                "category": exc.category,
                "detail": exc.detail,
                "retryable": False,
            },
        )
        write_status(output_dir, phase="tooling_setup", status="blocked", reason=exc.category)
        progress.event("BLOCKED", f"tooling {exc.category}: {exc.detail}")
        print(f"[bisection_harness] TOOLING_BLOCKED={exc.category}: {exc.detail}", file=sys.stderr)
        return 2
    tooling = plan.tooling
    progress.event(
        "TOOLING",
        f"pinned at {(tooling.source_commit_sha or 'WORKTREE')[:12]} "
        f"(spec={tooling.tooling_spec_hash or 'unversioned'})",
    )
    write_json(output_dir / "plan.resolved.json", plan.to_json())
    if args.command in range_commands:
        write_json(
            output_dir / "relaunch.json",
            {
                **_relaunch_tooling_fields(plan),
                "argv": [
                    "./isaaclab.sh",
                    "-p",
                    "tools/perf_smoke_test/bisection_harness.py",
                    "bisect-range",
                    "--plan",
                    str(output_dir / "plan.resolved.json"),
                    "--work_dir",
                    "<new-work-dir>",
                ],
            },
        )

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
            metric=plan.metric.to_json(),
        )
        progress.event("COMPLETE", f"resolved {candidate_payload['candidate_count']} candidates -> {output_dir}")
        print(f"[bisection_harness] candidates={candidate_payload['candidate_count']} -> {output_dir}")
        return 0

    recovery_policy = _build_recovery_policy(args)
    probe_policy = _build_probe_policy(args)
    if args.command == "benchmark-commit":
        commit_sha = resolve_ref(_REPO_ROOT, args.commit)
        preflight = write_measurement_preflight(output_dir, plan)
        if preflight is not None:
            progress.event(
                "PREFLIGHT",
                f"{preflight.gpu.name or 'GPU unknown'}, runner={preflight.runner_mode}, "
                f"{preflight.disk_free_gib:.1f} GiB free"
                if preflight.disk_free_gib is not None
                else f"{preflight.gpu.name or 'GPU unknown'}, runner={preflight.runner_mode}",
            )
            for warning in preflight.warnings:
                progress.event("WARNING", warning)
        result = measure_commit(
            plan,
            output_dir,
            commit_sha=commit_sha,
            label="benchmark",
            policy=recovery_policy,
            probe_policy=probe_policy,
            min_runs=max(1, args.runs),
            max_runs=max(1, args.max_runs),
        )
        write_json(output_dir / "measurement_summary.json", result.to_json())
        relaunch = [
            "./isaaclab.sh",
            "-p",
            "tools/perf_smoke_test/bisection_harness.py",
            "benchmark-commit",
            "--plan",
            str(output_dir / "plan.resolved.json"),
            "--commit",
            commit_sha,
            "--work_dir",
            "<new-work-dir>",
        ]
        write_json(
            output_dir / "relaunch.json",
            {**_relaunch_tooling_fields(plan), "argv": relaunch, "commit_sha": commit_sha},
        )
        write_status(
            output_dir,
            phase="completed" if result.succeeded else "failed",
            status="completed" if result.succeeded else "inconclusive",
            reason=result.note,
            tested_count=1,
            current_commit=commit_sha,
            metric=plan.metric.to_json(),
        )
        if result.summary is not None:
            progress.event(
                "COMPLETE",
                f"{commit_sha[:12]} median="
                f"{format_metric(result.summary.median_value, plan.metric.unit)} "
                f"spread={result.summary.spread_pct:.2f}% -> {output_dir}",
            )
        else:
            progress.event("INCONCLUSIVE", f"{commit_sha[:12]}: {result.note or 'measurement failed'}")
        print(
            f"[bisection_harness] benchmark-commit status="
            f"{'completed' if result.succeeded else 'inconclusive'} commit={commit_sha[:12]}"
        )
        return 0 if result.succeeded else 2

    summary = run_local_bisection(
        plan,
        output_dir,
        max_tests=args.max_tests,
        recovery_policy=recovery_policy,
        probe_policy=probe_policy,
    )
    write_status(
        output_dir,
        phase="completed" if summary.status == "completed" else "failed",
        status=summary.status,
        reason=summary.reason,
        tested_count=len(summary.tested_commits),
        suspected_first_bad_commit=summary.suspected_first_bad_commit,
        last_good_commit=summary.last_good_commit,
        metric=summary.metric,
    )
    if summary.status.startswith("completed"):
        progress.event(
            "COMPLETE",
            f"first_bad={(summary.suspected_first_bad_commit or 'unknown')[:12]} "
            f"last_good={(summary.last_good_commit or 'unknown')[:12]} -> {output_dir}",
        )
    else:
        progress.event("INCONCLUSIVE", f"status={summary.status} reason={summary.reason} -> {output_dir}")
    print(
        "[bisection_harness] "
        f"status={summary.status} first_bad={summary.suspected_first_bad_commit} "
        f"last_good={summary.last_good_commit}"
    )
    return 0 if summary.status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
