# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Adapter: schema-v1 ``PlayBundle`` JSON -> golden-gate normalized fields.

``golden_runtime.py`` emits a :class:`~isaaclab.test.benchmark.schema.PlayBundle`
(a checkpoint-driven inference rollout) serialized by
:func:`~isaaclab.test.benchmark.serialize.write_bundle_file`. This module is the
single point that reads that JSON and projects it into the typed
:class:`~golden_contracts.GoldenSample` the golden :mod:`build_golden_result` and
:mod:`golden_oracle` consume -- the analogue of
:mod:`benchmark_result_adapter` on the performance side.

The generic bundle sections (provenance, resource utilisation, startup, render
backend) are shared between the runtime and play bundles, so this module reuses
those projections from :mod:`benchmark_result_adapter` and only adds the
play-specific behavioural aggregates (reward / episode length / success rate /
episode count / checkpoint path).
"""

from __future__ import annotations

from typing import Any

from benchmark_result_adapter import provenance, render_backend, runtime_resources, startup_seconds
from golden_contracts import GoldenSample


def _as_dict(value: Any) -> dict:
    """Return ``value`` if it is a dict, else an empty dict (graceful on malformed bundles)."""
    return value if isinstance(value, dict) else {}


def _num(value: Any) -> float | None:
    """Return ``value`` as a float, or ``None`` if not a real number (bools rejected)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _int(value: Any) -> int | None:
    """Return ``value`` as an int, or ``None`` if not coercible."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def is_play_bundle(data: Any) -> bool:
    """Return True when ``data`` looks like a schema-v1 play bundle.

    Distinguishes a play bundle from a runtime bundle by the presence of the
    inference-evaluation fields (``success_rate``/``reward``/``ep_length``), which
    a runtime bundle never serializes.
    """
    return (
        isinstance(data, dict)
        and isinstance(data.get("run"), dict)
        and "schema_version" in data
        and all(key in data for key in ("success_rate", "reward", "ep_length"))
    )


def project_play(bundle: dict) -> GoldenSample | None:
    """Project a play bundle into a typed :class:`~golden_contracts.GoldenSample`.

    Returns ``None`` when ``bundle`` is not a valid schema-v1 play bundle, so the
    caller can degrade to a HARD_FAILURE (missing/invalid golden output).
    """
    if not is_play_bundle(bundle):
        return None
    run = _as_dict(bundle.get("run"))
    config = _as_dict(run.get("config"))
    extra = _as_dict(bundle.get("extra"))
    reward = _as_dict(bundle.get("reward"))
    ep_length = _as_dict(bundle.get("ep_length"))
    return GoldenSample(
        reward_mean=_num(reward.get("mean")),
        reward_std=_num(reward.get("std")),
        ep_length_mean=_num(ep_length.get("mean")),
        ep_length_std=_num(ep_length.get("std")),
        success_rate=_num(bundle.get("success_rate")),
        num_episodes=_int(extra.get("num_episodes")),
        checkpoint_path=bundle.get("checkpoint_path"),
        startup_time_s=startup_seconds(bundle),
        task=run.get("task"),
        num_envs=_int(run.get("num_envs")),
        seed=_int(run.get("seed")),
        physics_backend=config.get("physics_backend"),
        render_backend=render_backend(bundle),
        presets=config.get("presets") or [],
        provenance=provenance(bundle),
        runtime_resources=runtime_resources(bundle),
    )
