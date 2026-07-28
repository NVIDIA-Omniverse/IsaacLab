# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Post-rollout script: normalize golden output and write ``golden_result.json``.

The golden analogue of :mod:`build_bench_result`. Locates the timestamped
``PlayBundle`` written by ``golden_runtime.py``, copies it to the canonical
``golden_info.json``, classifies the failure phase from the captured log,
detects run-integrity drift (did the rollout run the requested task / num_envs /
seed / backend?), and writes ``golden_result.json`` for the golden aggregate.

Unlike ``build_bench_result``, the intended run shape is taken from CLI args (the
golden gate has no separate ``launch_config.json`` yet), and there is no baseline
or contract hashing -- the golden gate is standalone.

Usage::

    python3 tools/perf_smoke_test/build_golden_result.py \\
        --task_id Isaac-Cartpole-Direct --physics_backend physx \\
        --artifact_dir artifacts/Isaac-Cartpole-Direct/physx \\
        --exit_code 0 --wall_time_s 42.0 --timeout_s 600 \\
        --num_envs 64 --seed 42 --checkpoint_id cartpole-physx-v1 \\
        --log_file artifacts/Isaac-Cartpole-Direct/physx/golden.log
"""

import argparse
import glob
import json
import shutil
import sys
from pathlib import Path

_MODULE_DIR = Path(__file__).parent
_TOOLS_DIR = _MODULE_DIR.parent
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from backend_identity import (  # noqa: E402
    identity_from_parts,
    make_backend_key,
    normalize_physics_backend,
    normalize_render_backend,
)
from golden_config import get_golden_task  # noqa: E402
from golden_contracts import GoldenResult, GoldenSample  # noqa: E402
from golden_result_adapter import project_play  # noqa: E402
from gate_types import FailurePhase  # noqa: E402
from subprocess_runner import classify_failure_phase  # noqa: E402


def _config_drift(sample: GoldenSample, *, task_id: str, num_envs: int | None, seed: int | None, backend_key: str):
    """Return a compact run-integrity mismatch string, or ``None`` when the run matched intent."""
    mismatches: list[str] = []
    if sample.task and sample.task != task_id:
        mismatches.append(f"task(ran={sample.task},want={task_id})")
    if sample.num_envs is not None and num_envs is not None and sample.num_envs != num_envs:
        mismatches.append(f"num_envs(ran={sample.num_envs},want={num_envs})")
    if sample.seed is not None and seed is not None and sample.seed != seed:
        mismatches.append(f"seed(ran={sample.seed},want={seed})")
    ran_backend = identity_from_parts(sample.physics_backend, sample.render_backend)
    if ran_backend is not None and ran_backend.backend_key != backend_key:
        mismatches.append(f"backend(ran={ran_backend.backend_key},want={backend_key})")
    return " ".join(mismatches) if mismatches else None


def _normalize_golden_output(artifact_dir: Path, task_id: str) -> bool:
    """Copy the timestamped play bundle to ``golden_info.json``; return whether it now exists."""
    golden_info = artifact_dir / "golden_info.json"
    if golden_info.exists():
        return True
    matches = sorted(glob.glob(str(artifact_dir / f"benchmark_play_{task_id}_*.json")))
    if not matches:
        matches = sorted(glob.glob(str(artifact_dir / "benchmark_play_*.json")))
    if not matches:
        return False
    shutil.copy(matches[-1], golden_info)
    return True


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build golden_result.json from a golden rollout")
    p.add_argument("--task_id", required=True)
    p.add_argument("--physics_backend", required=True, help="Physics backend used (e.g. physx, newton)")
    p.add_argument("--render_backend", default="", help="Render backend used (e.g. newton_renderer); empty = none")
    p.add_argument("--preset", default="default")
    p.add_argument("--artifact_dir", required=True, type=Path)
    p.add_argument("--exit_code", required=True, type=int)
    p.add_argument("--wall_time_s", required=True, type=float)
    p.add_argument("--timeout_s", required=True, type=float)
    p.add_argument("--golden_tasks", type=Path, default=_MODULE_DIR / "golden_tasks.json")
    p.add_argument("--num_envs", type=int, default=None, help="Fallback eval env count if not in golden_tasks.json")
    p.add_argument("--seed", type=int, default=None, help="Fallback seed if not in golden_tasks.json")
    p.add_argument("--eval_steps", type=int, default=None, help="Fallback rollout steps if not in golden_tasks.json")
    p.add_argument("--checkpoint_id", default=None, help="Fallback logical golden checkpoint id")
    p.add_argument("--checkpoint_path", default=None, help="Local checkpoint path that was requested")
    p.add_argument("--log_file", type=Path, default=None)
    p.add_argument("--attempt", type=int, default=1)
    p.add_argument("--was_retried", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    artifact_dir = args.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)

    physics_backend = normalize_physics_backend(args.physics_backend)
    if physics_backend is None:
        raise ValueError("--physics_backend must name a concrete backend")
    render_backend = normalize_render_backend(args.render_backend)
    backend_key = make_backend_key(physics_backend, render_backend)

    # Single source of truth for run intent: prefer golden_tasks.json (the same
    # config that drives the rollout) over CLI args, so the driver and this builder
    # cannot disagree on the requested shape and spuriously flag drift.
    try:
        golden_task = get_golden_task(args.task_id, backend_key, args.golden_tasks)
    except Exception:
        golden_task = None
    want_num_envs = golden_task.num_envs if golden_task else args.num_envs
    want_seed = golden_task.seed if golden_task else args.seed
    want_eval_steps = golden_task.eval_steps if golden_task else args.eval_steps
    checkpoint_id = golden_task.checkpoint_id if golden_task else args.checkpoint_id

    log_text = ""
    if args.log_file and args.log_file.exists():
        log_text = args.log_file.read_text(errors="replace")

    golden_info_present = _normalize_golden_output(artifact_dir, args.task_id)

    failure_phase = classify_failure_phase(
        stdout=log_text,
        stderr="",
        exit_code=args.exit_code,
        wall_time_s=args.wall_time_s,
        timeout_s=args.timeout_s,
    )

    sample: GoldenSample | None = None
    config_mismatch: str | None = None
    if golden_info_present:
        info_path = artifact_dir / "golden_info.json"
        try:
            bundle = json.loads(info_path.read_text())
        except Exception:
            bundle = None
        sample = project_play(bundle) if isinstance(bundle, dict) else None
        if sample is None:
            # File exists but is not a valid schema-v1 play bundle (corrupt/truncated).
            golden_info_present = False
        else:
            config_mismatch = _config_drift(
                sample, task_id=args.task_id, num_envs=want_num_envs, seed=want_seed, backend_key=backend_key
            )
    if config_mismatch and failure_phase is None:
        failure_phase = FailurePhase.CONFIG_MISMATCH.value

    result = GoldenResult(
        task_id=args.task_id,
        backend=backend_key,
        physics_backend=physics_backend,
        render_backend=render_backend,
        backend_key=backend_key,
        preset=args.preset,
        checkpoint_id=checkpoint_id,
        checkpoint_path=(sample.checkpoint_path if sample and sample.checkpoint_path else args.checkpoint_path),
        attempt=args.attempt,
        was_retried=args.was_retried,
        exit_code=args.exit_code,
        failure_phase=failure_phase,
        stdout_tail=log_text[-2000:] if len(log_text) > 2000 else log_text,
        wall_time_s=args.wall_time_s,
        startup_time_s=sample.startup_time_s if sample else None,
        golden_info_present=golden_info_present,
        reward_mean=sample.reward_mean if sample else None,
        reward_std=sample.reward_std if sample else None,
        ep_length_mean=sample.ep_length_mean if sample else None,
        ep_length_std=sample.ep_length_std if sample else None,
        success_rate=sample.success_rate if sample else None,
        num_episodes=sample.num_episodes if sample else None,
        benchmark_info=sample.benchmark_info() if sample else {},
        config_mismatch=config_mismatch,
        runtime_resources=(sample.runtime_resources or None) if sample else None,
        provenance=sample.provenance if sample else None,
        launch_config={
            "task_id": args.task_id,
            "backend_key": backend_key,
            "physics_backend": physics_backend,
            "render_backend": render_backend,
            "num_envs": want_num_envs,
            "seed": want_seed,
            "eval_steps": want_eval_steps,
            "checkpoint_id": checkpoint_id,
        },
        task_config_snapshot={
            "task_id": args.task_id,
            "backend": backend_key,
            "physics_backend": physics_backend,
            "render_backend": render_backend,
            "backend_key": backend_key,
            "preset": args.preset,
            "num_envs": want_num_envs,
            "seed": want_seed,
            "eval_steps": want_eval_steps,
            "checkpoint_id": checkpoint_id,
        },
    )

    out = artifact_dir / "golden_result.json"
    out.write_text(json.dumps(result.to_dict(), indent=2))
    print(
        f"[build_golden_result] {args.task_id}/{backend_key}: "
        f"failure_phase={failure_phase!r}, golden_info_present={golden_info_present}, "
        f"exit_code={args.exit_code}, config_mismatch={config_mismatch!r}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
