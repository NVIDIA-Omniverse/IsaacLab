# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Self-contained effective-FPS probe for collecting hard-floor values from an
older IsaacLab release (e.g. IsaacLab 2.0).

This script is intentionally **standalone**: it imports nothing from the perf
gate tooling, so it can be mounted into an arbitrary IsaacLab container (a 2.0 /
Isaac Sim 4.5 image) and still run. It reproduces *exactly* the FPS metric the
gate's oracle compares on, so a number measured here is directly usable as an
``fps_mean_floor`` entry in ``tasks.json``:

* per-frame effective FPS = ``num_envs / env.step() wall time [s]``
* the reported figure is the arithmetic **mean over kept frames** (warmup frames
  dropped via ``--excluded_frames``), matching ``oracle.apply_excluded_frames``
  + ``statistics.mean``.

Only stable APIs are used (``isaaclab.app.AppLauncher``, ``gymnasium``,
``torch``, ``isaaclab_tasks``), all present since the 2.0 rename of
``omni.isaac.lab`` -> ``isaaclab``. Newton does not exist in 2.0; this probe is
physics-backend agnostic and simply benchmarks whatever the release's default
(PhysX) provides.

Example (inside a 2.0 container)::

    ./isaaclab.sh -p tools/perf_regression_gate/measure_legacy_fps.py \
        --task Isaac-Cartpole-Direct-v0 --num_envs 4096 --num_frames 300 \
        --excluded_frames 0-100 --seed 42 --output_path /tmp/floor_out
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def expand_excluded_frames(tokens: list[str]) -> frozenset[int]:
    """Expand ``--excluded_frames`` tokens into a set of 0-based frame indices.

    Accepts single indices (``"5"``) and inclusive ranges (``"0-100"``), mirroring
    the gate's ``excluded_frames_raw`` semantics (where ``[0, 100]`` drops frames
    0 through 100 inclusive).

    Args:
        tokens: Raw CLI tokens, each either ``"<int>"`` or ``"<lo>-<hi>"``.

    Returns:
        The set of frame indices to drop.
    """
    indices: set[int] = set()
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            lo_str, hi_str = token.split("-", 1)
            lo, hi = int(lo_str), int(hi_str)
            if lo > hi:
                raise ValueError(f"excluded_frames range start must be <= end, got {token!r}")
            indices.update(range(lo, hi + 1))
        else:
            indices.add(int(token))
    return frozenset(indices)


def effective_fps_series(step_times_s: list[float], num_envs: int) -> list[float]:
    """Convert per-frame step wall times [s] into per-frame effective FPS.

    Args:
        step_times_s: Per-frame ``env.step()`` wall times in seconds.
        num_envs: Number of parallel environments stepped each frame.

    Returns:
        Per-frame effective FPS (``num_envs / step_time``); frames with a
        non-positive step time are dropped (cannot yield a finite FPS).
    """
    return [num_envs / dt for dt in step_times_s if dt > 0.0]


def mean_kept(series: list[float], excluded: frozenset[int]) -> float | None:
    """Mean of ``series`` after dropping warmup frames, matching the oracle.

    Args:
        series: Per-frame effective FPS.
        excluded: 0-based frame indices to drop before averaging.

    Returns:
        Arithmetic mean over kept frames, or ``None`` when nothing remains.
    """
    kept = [fps for idx, fps in enumerate(series) if idx not in excluded]
    return statistics.mean(kept) if kept else None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure comparable effective FPS for a single task/backend.")
    parser.add_argument("--task", type=str, required=True, help="Gym task id as registered in this IsaacLab release")
    parser.add_argument("--num_envs", type=int, required=True, help="Number of parallel environments")
    parser.add_argument("--num_frames", type=int, default=300, help="Number of env.step() frames to time")
    parser.add_argument(
        "--excluded_frames",
        nargs="*",
        default=["0-100"],
        help="Warmup frame indices/ranges to drop before averaging (e.g. 0-100). Match tasks.json excluded_frames.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Environment seed")
    parser.add_argument("--output_path", type=str, default=".", help="Directory for the result JSON")
    return parser


def main() -> int:
    parser = _build_parser()
    # AppLauncher contributes --headless/--device/--enable_cameras and friends.
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    excluded = expand_excluded_frames(args.excluded_frames)

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Imports that require the running app come after launch (Isaac Sim convention).
    import time

    import gymnasium as gym
    import torch

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    device = getattr(args, "device", None) or "cuda:0"
    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
    if args.seed is not None:
        # Both Direct and Manager-based cfgs expose a top-level seed field in 2.x.
        env_cfg.seed = args.seed

    env = gym.make(args.task, cfg=env_cfg)
    env.reset()

    action_dim = env.unwrapped.single_action_space.shape[0]
    num_envs = env.unwrapped.num_envs

    step_times_s: list[float] = []
    frame = 0
    while frame < args.num_frames:
        actions = 2.0 * torch.rand(num_envs, action_dim, device=env.unwrapped.device) - 1.0
        begin = time.perf_counter_ns()
        env.step(actions)
        end = time.perf_counter_ns()
        step_times_s.append((end - begin) / 1e9)
        frame += 1

    env.close()

    series = effective_fps_series(step_times_s, num_envs)
    mean_fps = mean_kept(series, excluded)

    result = {
        "task_id": args.task,
        "num_envs": num_envs,
        "num_frames": args.num_frames,
        "excluded_frames": sorted(excluded),
        "kept_frame_count": len(series) - sum(1 for idx in range(len(series)) if idx in excluded),
        "effective_fps_series": series,
        "mean_kept_effective_fps": mean_fps,
    }
    out_dir = Path(args.output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_task = "".join(ch if ch.isalnum() else "-" for ch in args.task)
    out_path = out_dir / f"legacy_fps_{safe_task}.json"
    with out_path.open("w") as fh:
        json.dump(result, fh, indent=2)
        fh.write("\n")

    mean_str = f"{mean_fps:.1f}" if mean_fps is not None else "N/A"
    print(f"[legacy-fps] {args.task} num_envs={num_envs} mean_kept_effective_fps={mean_str}")
    print(f"[legacy-fps] wrote {out_path}")

    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
