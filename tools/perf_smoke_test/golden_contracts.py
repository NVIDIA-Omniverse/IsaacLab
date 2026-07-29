# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Typed contract for the golden gate's own artifact (schema v1).

:class:`GoldenResult` is the golden analogue of :class:`~contracts.BenchResult`:
the typed shape of ``golden_result.json``, the per-(task, backend) artifact a
golden rollout produces and the golden oracle consumes. It follows the same
"typed gate-critical fields + open pass-through dicts" pattern -- the measured
behavioural KPIs and run identity are typed attributes, while provenance and
launch/config payloads stay as ``dict`` fields carried through untouched.

The behavioural KPIs come straight from the merged benchmark core's play
rollout (:func:`~isaaclab.test.benchmark.stepping.run_play_loop` -> a
:class:`~isaaclab.test.benchmark.schema.PlayBundle`): ``reward`` and
``ep_length`` as scalar mean/std aggregates over completed episodes, plus a
``success_rate``. :attr:`probe_kpis` carries any additional sim-state signals
produced by an optional per-task probe (see :mod:`golden_probes`); it is empty
unless a probe is registered.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any

CONTRACT_SCHEMA_VERSION = "1.0"

# The KPI names backed by typed fields on :class:`GoldenResult` rather than by an
# optional sim-state probe. All three are derived from *completed episodes*, so a
# run that completed no episode leaves them unmeasured (see
# :meth:`GoldenResult.has_completed_episodes`). Any other configured KPI name must
# be produced by a registered probe (see :mod:`golden_probes`).
UNIVERSAL_KPIS = frozenset({"reward", "ep_length", "success_rate"})


@dataclass(frozen=True)
class GoldenSample:
    """Adapter projection of a schema-v1 ``PlayBundle`` into the golden gate's fields.

    The behavioural aggregates and run identity are typed; ``provenance`` and
    ``runtime_resources`` stay dicts (open provenance payloads carried through).
    Mirrors :class:`~contracts.RuntimeSample` on the performance side.
    """

    reward_mean: float | None = None
    reward_std: float | None = None
    ep_length_mean: float | None = None
    ep_length_std: float | None = None
    success_rate: float | None = None
    num_episodes: int | None = None
    checkpoint_path: str | None = None
    startup_time_s: float | None = None
    task: str | None = None
    num_envs: int | None = None
    seed: int | None = None
    physics_backend: str | None = None
    render_backend: str | None = None
    presets: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    runtime_resources: dict[str, Any] = field(default_factory=dict)

    def benchmark_info(self) -> dict[str, Any]:
        """Return the run's self-reported identity as a dict (None-valued keys omitted)."""
        info = {
            "task": self.task,
            "num_envs": self.num_envs,
            "seed": self.seed,
            "physics_backend": self.physics_backend,
            "render_backend": self.render_backend,
            "presets": self.presets or None,
        }
        return {k: v for k, v in info.items() if v is not None}


@dataclass(frozen=True)
class GoldenResult:
    """Typed shape of ``golden_result.json`` (schema v1).

    The leading identity fields are required (a build that forgets one fails at
    construction); the rest default so a HARD_FAILURE result -- e.g. a missing
    checkpoint or a crashed rollout, where no KPI was measured -- still
    constructs cleanly.
    """

    # --- identity (required) ---
    task_id: str
    backend: str
    physics_backend: str
    render_backend: str | None
    backend_key: str
    preset: str
    # --- golden reference identity ---
    checkpoint_id: str | None = None
    checkpoint_path: str | None = None
    # --- run status ---
    attempt: int = 1
    was_retried: bool = False
    exit_code: int = 0
    failure_phase: str | None = None
    stdout_tail: str = ""
    wall_time_s: float | None = None
    startup_time_s: float | None = None
    golden_info_present: bool = False
    # --- behavioural KPIs (aggregates over completed episodes) ---
    reward_mean: float | None = None
    reward_std: float | None = None
    ep_length_mean: float | None = None
    ep_length_std: float | None = None
    success_rate: float | None = None
    num_episodes: int | None = None
    # --- optional sim-state probe signals (name -> scalar); empty unless a probe is registered ---
    probe_kpis: dict[str, float] = field(default_factory=dict)
    # --- open pass-through payloads ---
    benchmark_info: dict[str, Any] = field(default_factory=dict)
    config_mismatch: str | None = None
    runtime_resources: dict[str, Any] | None = None
    provenance: dict[str, Any] | None = None
    launch_config: dict[str, Any] = field(default_factory=dict)
    task_config_snapshot: dict[str, Any] = field(default_factory=dict)
    schema_version: str = CONTRACT_SCHEMA_VERSION

    def has_completed_episodes(self) -> bool:
        """Return whether the rollout completed at least one episode.

        Episode-derived KPIs (:data:`UNIVERSAL_KPIS`) are only meaningful when an
        episode finished. ``num_episodes`` is trusted when reported; an unset
        (``None``) count is treated as "unknown, don't force-unmeasure" so a run
        that reports a value but omits the count is not silently discarded. Only an
        explicit non-positive count marks the episode KPIs as unmeasured -- which
        closes the hole where a runner reports e.g. ``success_rate = 0.0`` over
        zero episodes.
        """
        return self.num_episodes is None or self.num_episodes > 0

    def kpi_value(self, kpi_name: str) -> float | None:
        """Return the measured value for a configured KPI name, or ``None`` if unmeasured.

        The three universal KPIs map to typed fields; any other name is looked up
        in :attr:`probe_kpis` (populated only when a task registers a sim-state
        probe). A ``None`` return means "configured but not measured this run"
        (e.g. ``reward`` when no episode completed) and is handled by the oracle.

        Args:
            kpi_name: KPI key as configured in ``golden_tasks.json``.

        Returns:
            The measured scalar, or ``None`` when the KPI was not produced.
        """
        universal = {
            "reward": self.reward_mean,
            "ep_length": self.ep_length_mean,
            "success_rate": self.success_rate,
        }
        if kpi_name in universal:
            return universal[kpi_name]
        return self.probe_kpis.get(kpi_name)

    def measured_kpi(self, kpi_name: str) -> float | None:
        """Return the value for a KPI treating unfinished-episode aggregates as unmeasured.

        Like :meth:`kpi_value`, but an episode-derived universal KPI is reported as
        ``None`` when no episode completed (see :meth:`has_completed_episodes`),
        regardless of any placeholder value the runner may have emitted.

        Args:
            kpi_name: KPI key as configured in ``golden_tasks.json``.

        Returns:
            The measured scalar, or ``None`` when the KPI was not produced.
        """
        if kpi_name in UNIVERSAL_KPIS and not self.has_completed_episodes():
            return None
        return self.kpi_value(kpi_name)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the on-disk JSON shape (plain dicts, wire-stable order)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GoldenResult:
        """Reconstruct from a parsed ``golden_result.json``.

        Unknown keys are dropped; missing required identity fields raise.
        """
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})
