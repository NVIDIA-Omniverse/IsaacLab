#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Local-testing stub that fakes a ``golden_runtime.py`` rollout without a GPU/sim.

Emits a schema-v1 :class:`~isaaclab.test.benchmark.schema.PlayBundle`
(``benchmark_play_{task}_{stamp}.json``) identical in shape to what the real
golden driver writes, so ``build_golden_result.py`` -> ``golden_result_adapter``
-> ``golden_oracle`` can be exercised end-to-end offline. Uses the real
``isaaclab.test.benchmark`` builders/serialize (pure-Python, no GPU) and the
shared stub fixtures from :mod:`stub_benchmark`, so it stays in lockstep with the
schema; run it with the Isaac Lab Python env.

The behavioural aggregates are fully configurable so a single stub can drive
every oracle path (healthy PASS, a reward-floor BLOCK, a pole-ceiling probe
BLOCK, a zero-episode WARN, and the import/init/runtime failure phases).
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

_DEV_DIR = Path(__file__).resolve().parent
_MODULE_DIR = _DEV_DIR.parent
for _p in (str(_MODULE_DIR), str(_DEV_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from backend_identity import split_backend_key  # noqa: E402
from stub_benchmark import _stub_hardware, _stub_resources, _stub_versions  # noqa: E402

from isaaclab.test.benchmark import builders, serialize  # noqa: E402
from isaaclab.test.benchmark.schema import MeanStd, StartupTime  # noqa: E402


def _mean_std(mean: float | None, std: float) -> MeanStd | None:
    """Build a :class:`MeanStd`, or ``None`` to simulate an aggregate over zero episodes."""
    return None if mean is None else MeanStd(mean=mean, std=std)


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit a stub golden PlayBundle for offline pipeline testing.")
    parser.add_argument("--task_id", required=True)
    parser.add_argument("--backend", required=True, help="Backend key, e.g. physx / newton / physx_newton_renderer")
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--reward_mean", type=float, default=480.0, help="Set negative to omit (simulate no episode).")
    parser.add_argument("--reward_std", type=float, default=5.0)
    parser.add_argument("--ep_length_mean", type=float, default=500.0, help="Set negative to omit.")
    parser.add_argument("--ep_length_std", type=float, default=0.0)
    parser.add_argument("--success_rate", type=float, default=None)
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=None,
        help="Completed-episode count to record; omit to match the real driver, which does not report one.",
    )
    parser.add_argument("--checkpoint_path", default="/opt/golden/stub/policy.pt")
    parser.add_argument("--failure_phase", default="none", choices=["none", "import", "init", "runtime"])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- simulated pre-output failures (no bundle written) ---
    if args.failure_phase == "import":
        print("Traceback (most recent call last):")
        print("ImportError: simulated import failure")
        sys.exit(1)
    print("AppLauncher initialization complete", flush=True)
    if args.failure_phase == "init":
        sys.exit(2)

    identity = split_backend_key(args.backend)
    if identity is None:
        raise RuntimeError(f"Cannot parse backend identity from {args.backend!r}")

    # A play bundle still carries a Runtime section; synthesize a trivial steady series.
    startup = StartupTime(app_launch=2.0, env_creation=1.0, first_step=0.3, python_imports=1.5)
    step_times = [args.num_envs / 200.0 for _ in range(max(1, args.eval_steps))]
    fps = [200.0 for _ in step_times]
    runtime = builders.build_runtime(
        startup_time_s=startup,
        iteration_times_s=step_times,
        collection_fps=fps,
        total_fps=fps,
        steps_per_iteration=args.num_envs,
    )
    cfg = builders.build_run_config(
        physics_backend=identity.physics_backend,
        rendering_backend=identity.render_backend or "none",
        presets=[],
    )
    start_utc = datetime.now(timezone.utc).isoformat()
    run = builders.build_run_identity(
        run_id=f"stub-golden-{args.task_id}-{identity.backend_key}",
        framework="rsl_rl",
        config=cfg,
        task=args.task_id,
        seed=args.seed,
        start_utc=start_utc,
        end_utc=start_utc,
        num_envs=args.num_envs,
    )
    extra: dict = {"stub": True}
    if args.num_episodes is not None:
        extra["num_episodes"] = args.num_episodes
    bundle = builders.build_play_bundle(
        run=run,
        versions=_stub_versions(),
        hardware=_stub_hardware(),
        runtime=runtime,
        resources=_stub_resources(),
        success_rate=args.success_rate,
        reward=_mean_std(None if args.reward_mean < 0 else args.reward_mean, args.reward_std),
        ep_length=_mean_std(None if args.ep_length_mean < 0 else args.ep_length_mean, args.ep_length_std),
        checkpoint_path=args.checkpoint_path,
        extra=extra,
    )

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    out_path = out_dir / f"benchmark_play_{args.task_id}_{stamp}.json"
    serialize.write_bundle_file(bundle, str(out_path))

    # Progress marker consumed by subprocess_runner.classify_failure_phase (matches the
    # perf driver's stdout contract) so a runtime crash is not misread as an init failure.
    print("Step Frametimes", flush=True)

    if args.failure_phase == "runtime":
        print("RuntimeError: simulated crash during rollout")
        sys.exit(3)
    sys.exit(0)


if __name__ == "__main__":
    main()
