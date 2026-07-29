# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Golden correctness rollout driver (checkpoint-driven, deterministic).

The golden analogue of :mod:`perf_runtime`. Rolls out a frozen policy on a task
for a fixed number of deterministic steps and emits a schema-v1
:class:`~isaaclab.test.benchmark.schema.PlayBundle` (behavioural aggregates:
reward, episode length, success rate), which ``build_golden_result.py`` then
normalizes into ``golden_result.json``.

Like :mod:`perf_runtime`, it imports only the *merged* Part-1 benchmark building
blocks (:mod:`~isaaclab.test.benchmark.stepping`/``builders``/``capture``) and
reuses :func:`~isaaclab.test.benchmark.stepping.run_play_loop`; it does **not**
depend on the still-unmerged ``scripts/benchmarks`` play scripts.

Two policy sources:

* ``--checkpoint <local path>``: load a trained RSL-RL policy from a **local**
  file (baked into the runner image at publish time). The upstream Nucleus
  fallback is intentionally *not* used, so the runner needs no network.
* ``--dummy_policy``: a zero-action policy that needs no checkpoint. This is a
  development aid to validate the rollout -> bundle -> result pipeline on a GPU
  without a blessed checkpoint; it is not a correctness reference.

Usage example::

    ./isaaclab.sh -p tools/perf_smoke_test/golden_runtime.py \\
        --task Isaac-Cartpole-Direct --num_envs 64 --eval_steps 200 \\
        --dummy_policy --output_path /tmp/golden_out \\
        presets=newton_mjwarp --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

from isaaclab_tasks.utils import setup_preset_cli

# --- argument parsing -------------------------------------------------------
parser = argparse.ArgumentParser(description="Golden correctness rollout (checkpoint-driven, deterministic).")
parser.add_argument("--task", type=str, required=True, help="Gym task id to roll out.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments.")
parser.add_argument("--eval_steps", type=int, default=200, help="Number of environment steps to roll out.")
parser.add_argument("--seed", type=int, default=None, help="Environment seed (determinism).")
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Local checkpoint path to roll out (never fetched over the network). Omit with --dummy_policy.",
)
parser.add_argument(
    "--dummy_policy",
    action="store_true",
    help="Dev only: use a deterministic dummy policy instead of a checkpoint (validates the pipeline, not correctness).",
)
parser.add_argument(
    "--dummy_mode",
    type=str,
    default="zero",
    choices=["zero", "stabilizer"],
    help="Dummy policy behaviour (with --dummy_policy): 'zero' (failure-like) or 'stabilizer' (recognizable signal).",
)
parser.add_argument("--dummy_kp", type=float, default=8.0, help="Stabilizer proportional gain on obs[:, 0].")
parser.add_argument("--dummy_kd", type=float, default=1.0, help="Stabilizer derivative gain on obs[:, 1].")
parser.add_argument("--rl_library", type=str, default="rsl_rl", help="RL library the checkpoint was trained with.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent config entry point.")
parser.add_argument("--output_path", type=str, default=".", help="Directory to write the output JSON.")
parser.add_argument(
    "--benchmark_formatter",
    type=str,
    default="schema",
    help="Output format(s): comma-separated 'schema' (default), 'omniperf', 'osmo', 'json', 'summary'.",
)

# append AppLauncher cli args and resolve Hydra preset tokens
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

if not args_cli.dummy_policy and not args_cli.checkpoint:
    parser.error("either --checkpoint <local path> or --dummy_policy is required")
if args_cli.dummy_policy and args_cli.checkpoint:
    print(
        "[golden_runtime] WARNING: --dummy_policy given; ignoring --checkpoint "
        "(zero-action policy is a pipeline aid, not a correctness reference).",
        flush=True,
    )

# --- heavy imports (after CLI parse, before app launch is measured) ---------
imports_time_begin = time.perf_counter_ns()

import contextlib

import gymnasium as gym

from isaaclab.app import launch_simulation
from isaaclab.test.benchmark import BaseIsaacLabBenchmark, BenchmarkMonitor, builders, capture, stepping
from isaaclab.test.benchmark.schema import StartupTime

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config

# PLACEHOLDER: Extension template (do not remove this comment)
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

imports_time_end = time.perf_counter_ns()


def _dummy_policy(env, mode: str, kp: float, kd: float):
    """Return a deterministic dummy policy callable for pipeline validation (no checkpoint).

    Two modes:

    * ``zero``: a task-agnostic zero-action baseline (Cartpole pole falls almost
      immediately -> very short episodes, ~0 reward). Deliberately *failure-like*;
      used to exercise the low-signal / forced-failure path.
    * ``stabilizer``: a fixed proportional-derivative feedback on the first two
      observation components (for Cartpole, pole angle + angular velocity),
      ``action = clamp(-(kp*obs[:,0] + kd*obs[:,1]), -1, 1)``. It *partially* balances
      the pole, giving a RECOGNIZABLE non-trivial signal (clearly-positive reward,
      mid-range episodes, non-zero success) that cannot be conflated with an error or
      a genuine no-reward result. Not a correctness reference -- a pipeline probe.

    Args:
        env: The (unwrapped-compatible) environment.
        mode: ``"zero"`` or ``"stabilizer"``.
        kp: Proportional gain on ``obs[:, 0]`` (stabilizer mode).
        kd: Derivative gain on ``obs[:, 1]`` (stabilizer mode).
    """
    import torch  # noqa: PLC0415

    u = env.unwrapped
    num_envs, device = u.num_envs, u.device
    action_dim = int(env.action_space.shape[-1])

    if mode == "zero":
        zeros = torch.zeros((num_envs, action_dim), device=device)
        return lambda obs: zeros

    def stabilizer(obs):
        x = obs[0] if isinstance(obs, tuple) else obs
        if isinstance(x, dict):
            x = x.get("policy", next(iter(x.values())))
        signal = -(kp * x[:, 0:1] + kd * x[:, 1:2])
        return torch.clamp(signal, -1.0, 1.0).expand(num_envs, action_dim)

    return stabilizer


def _load_rsl_rl_policy(env, agent_cfg, checkpoint_path: str):
    """Load a trained RSL-RL inference policy from a **local** checkpoint file.

    Mirrors ``scripts/benchmarks/rsl_rl/benchmark_rsl_rl_play.py`` but resolves the
    checkpoint as a plain local path -- the upstream Nucleus/``retrieve_file_path``
    fallback is deliberately avoided so the runner performs no network access.
    """
    import importlib.metadata as metadata  # noqa: PLC0415

    from rsl_rl.runners import DistillationRunner, OnPolicyRunner  # noqa: PLC0415

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: PLC0415

    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(f"golden checkpoint not found at local path: {checkpoint_path!r}")

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported RSL-RL runner class: {agent_cfg.class_name}")
    runner.load(checkpoint_path)
    return env, runner.get_inference_policy(device=env.unwrapped.device)


def main(env_cfg, agent_cfg, app_start_time_begin: int, app_start_time_end: int) -> None:
    """Run the golden rollout and write the selected formatter outputs.

    Args:
        env_cfg: Resolved environment configuration for :attr:`args_cli.task`.
        agent_cfg: Resolved RL agent configuration, or ``None`` in ``--dummy_policy`` mode.
        app_start_time_begin: ``perf_counter_ns`` sampled just before the app launch.
        app_start_time_end: ``perf_counter_ns`` sampled just after the app launch.
    """
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed

    cfg = capture.run_config_from_presets(hydra_args, env_cfg=env_cfg)
    start_utc = capture.now_utc_iso()

    benchmark = BaseIsaacLabBenchmark(
        benchmark_name="benchmark_play",
        formatter_type=args_cli.benchmark_formatter,
        output_path=args_cli.output_path,
        use_recorders=True,
        frametime_recorders=False,
        output_prefix=f"benchmark_play_{args_cli.task}",
        workflow_metadata={
            "metadata": [
                {"name": "task", "data": args_cli.task},
                {"name": "num_envs", "data": args_cli.num_envs},
                {"name": "eval_steps", "data": args_cli.eval_steps},
                {"name": "presets", "data": ",".join(cfg.presets)},
            ]
        },
    )

    env_t0 = time.perf_counter_ns()
    with contextlib.closing(gym.make(args_cli.task, cfg=env_cfg)) as base_env:
        env_t1 = time.perf_counter_ns()

        if args_cli.dummy_policy:
            env = base_env
            policy = _dummy_policy(env, args_cli.dummy_mode, args_cli.dummy_kp, args_cli.dummy_kd)
            checkpoint_path = None
        else:
            env, policy = _load_rsl_rl_policy(base_env, agent_cfg, args_cli.checkpoint)
            checkpoint_path = args_cli.checkpoint

        num_envs = env.unwrapped.num_envs

        with BenchmarkMonitor(benchmark, interval=1.0):
            step_times, reward, ep_length, success_rate = stepping.run_play_loop(env, policy, args_cli.eval_steps)

        # Progress marker consumed by subprocess_runner.classify_failure_phase (shared
        # stdout contract) so a runtime-phase crash is not misread as an init failure.
        print("Step Frametimes", flush=True)

        benchmark.update_manual_recorders()

        startup = StartupTime(
            app_launch=(app_start_time_end - app_start_time_begin) / 1e9,
            env_creation=(env_t1 - env_t0) / 1e9,
            first_step=(step_times[0] if step_times else 0.0),
            python_imports=(imports_time_end - imports_time_begin) / 1e9,
        )
        fps = [num_envs / t for t in step_times if t > 0]
        runtime = builders.build_runtime(
            startup_time_s=startup,
            iteration_times_s=step_times,
            collection_fps=fps,
            total_fps=fps,
            steps_per_iteration=num_envs,
        )

        versions = capture.capture_versions(benchmark)
        hardware = capture.capture_hardware(benchmark)
        resources = capture.capture_resources(benchmark)

        end_utc = capture.now_utc_iso()
        stamp = end_utc.translate(str.maketrans("", "", ":-"))[:15]
        seed = args_cli.seed if args_cli.seed is not None else 0
        run_id = capture.synth_run_id(args_cli.rl_library, cfg.physics_backend, args_cli.task, seed, stamp)

        run = builders.build_run_identity(
            run_id=run_id,
            framework=args_cli.rl_library,
            config=cfg,
            task=args_cli.task,
            seed=seed,
            start_utc=start_utc,
            end_utc=end_utc,
            num_envs=num_envs,
        )

        # ``run_play_loop`` returns only the aggregates, not a completed-episode
        # count, so ``num_episodes`` is intentionally not recorded here. A true
        # zero-episode run yields reward/ep_length/success_rate = None, which the
        # oracle already treats as unmeasured -- that None contract, not an episode
        # count, is what protects the golden verdict from an unfinished rollout.
        bundle = builders.build_play_bundle(
            run=run,
            versions=versions,
            hardware=hardware,
            runtime=runtime,
            resources=resources,
            success_rate=success_rate,
            reward=reward,
            ep_length=ep_length,
            checkpoint_path=checkpoint_path,
            extra={
                "eval_steps": args_cli.eval_steps,
                "dummy_policy": bool(args_cli.dummy_policy),
                "dummy_mode": (args_cli.dummy_mode if args_cli.dummy_policy else "checkpoint"),
            },
        )

        benchmark.attach_bundle(bundle)
        benchmark._finalize_impl()


if __name__ == "__main__":
    # Dummy mode needs no agent config; real mode resolves it for the checkpoint runner.
    agent_ref = None if args_cli.dummy_policy else args_cli.agent
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, agent_ref)

    app_start_time_begin = time.perf_counter_ns()
    with launch_simulation(env_cfg, args_cli):
        app_start_time_end = time.perf_counter_ns()
        main(env_cfg, agent_cfg, app_start_time_begin, app_start_time_end)
