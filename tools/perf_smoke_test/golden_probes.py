# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Optional per-task simulation-state probes for golden correctness checks.

A *probe* reads named scalar signals directly from the live environment's
simulation state -- e.g. a cartpole's pole angle -- so a golden check can assert
on *physical* correctness beyond the reward / episode-length / success-rate
signals that the rollout already produces on its own.

Probes are deliberately **optional**. A task with no registered probe is
evaluated on reward / ep_length / success_rate alone (which require no task
specific code, since the environment reports them). The probe registry is the
single, contained escape hatch for the open-ended "query arbitrary sim state"
case, so that adding one is a few lines of typed code rather than a new pipeline.

Probe contract::

    probe_sim_state(env) -> dict[str, float]

The callable receives the *unwrapped-compatible* Gym environment and returns a
mapping of signal name to a scalar reduced over the parallel environments (e.g.
the mean pole angle across all envs at the current step). Each returned name
must match a ``kpis`` key configured for the task in ``golden_tasks.json`` so the
oracle can threshold it. Reading simulation state forces a GPU->CPU sync, so a
probe should read only what the configured KPIs need and stay cheap.

v1 status: the seam is plumbed end-to-end -- the driver calls the probe when one
is registered, and its values flow through :attr:`GoldenResult.probe_kpis
<contracts>` into the oracle -- but the registry ships **empty**; no task
registers a probe yet.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# A probe maps a live environment to named scalar sim-state signals for one step.
ProbeFn = Callable[[Any], dict[str, float]]

# Intentionally empty in v1: the pipeline supports probes, but none are defined.
# Keyed by gym task id (namespace/version-insensitive callers should normalize
# before lookup if needed; v1 matches on the exact configured task id).
_PROBES: dict[str, ProbeFn] = {}


def register_probe(task_id: str, probe: ProbeFn) -> None:
    """Register a sim-state probe for a task id.

    Args:
        task_id: Gym task id the probe applies to.
        probe: Callable implementing the :data:`ProbeFn` contract.

    Raises:
        ValueError: If a probe is already registered for ``task_id`` (probes are
            unique per task; re-registration is a configuration error rather than
            a silent override).
    """
    if task_id in _PROBES:
        raise ValueError(f"a sim-state probe is already registered for {task_id!r}")
    _PROBES[task_id] = probe


def get_probe(task_id: str) -> ProbeFn | None:
    """Return the registered sim-state probe for ``task_id``, or ``None`` if unset."""
    return _PROBES.get(task_id)
