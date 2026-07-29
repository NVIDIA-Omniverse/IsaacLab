# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Golden correctness task configuration.

Loads ``golden_tasks.json`` -- the sibling of ``tasks.json`` for the golden gate
-- into one :class:`GoldenTaskConfig` per (task, backend) combination. It is kept
as a separate file (rather than extra fields on ``tasks.json``) so the two gates
stay fully decoupled: the performance suite parses and runs with no awareness of
golden config, and a task can carry a performance entry, a golden entry, or both.

A golden entry names a frozen policy **checkpoint** (a logical id plus a path
relative to a baked checkpoint root -- never a URL, so the runner needs no
network) and the hard-set behavioural **KPI thresholds** to judge its rollout
against (see :class:`~golden_kpi.KpiThreshold`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

try:
    from .backend_identity import make_backend_key, normalize_physics_backend, normalize_render_backend
    from .golden_contracts import UNIVERSAL_KPIS
    from .golden_kpi import KpiThreshold
    from .golden_probes import get_probe
except ImportError:  # pragma: no cover - supports direct script imports
    from backend_identity import make_backend_key, normalize_physics_backend, normalize_render_backend
    from golden_contracts import UNIVERSAL_KPIS
    from golden_kpi import KpiThreshold
    from golden_probes import get_probe

_DEFAULT_GOLDEN_TASKS_JSON = Path(__file__).parent / "golden_tasks.json"


def parse_kpi_thresholds(raw) -> dict[str, list[KpiThreshold]]:
    """Parse the ``{kpi_name: [threshold entries]}`` map for one golden task.

    Validation (mandatory names, gating-verdict enum, direction enum, value-less
    skips) happens in :meth:`~golden_kpi.KpiThreshold.from_list`, so a malformed
    ``golden_tasks.json`` fails fast at load time.

    Args:
        raw: The raw ``kpis`` value from a task entry (may be empty/absent).

    Returns:
        Mapping of KPI name to its parsed threshold list.
    """
    if not raw:
        return {}
    if not isinstance(raw, dict):
        raise TypeError("kpis must be an object keyed by KPI name")
    return {kpi_name: KpiThreshold.from_list(entries, context=kpi_name) for kpi_name, entries in raw.items()}


@dataclass
class GoldenTaskConfig:
    """Golden correctness configuration for a single task and backend combination."""

    task_id: str
    physics_backend: str
    render_backend: str | None
    preset: str
    rl_library: str
    agent: str
    checkpoint_id: str
    checkpoint_relpath: str
    num_envs: int
    eval_steps: int
    seed: int | None
    deterministic: bool
    kpis: dict[str, list[KpiThreshold]]
    timeout_minutes: int
    tags: list[str] = field(default_factory=lambda: ["always"])
    task_type: str = "golden"
    runs_on: str = "gpu-l40s"

    @property
    def backend_key(self) -> str:
        """Composite key identifying the backend combination (see :func:`~backend_identity.make_backend_key`)."""
        return make_backend_key(self.physics_backend, self.render_backend)

    def resolve_checkpoint_path(self, golden_root: Path | str) -> Path:
        """Return the absolute local checkpoint path under a baked ``golden_root``.

        The checkpoint is resolved to a local path only; it is baked into the
        runner image at publish time (network-capable), so the runner itself
        never fetches it. :attr:`checkpoint_relpath` is already backend-expanded
        at load time.

        Args:
            golden_root: Directory the golden checkpoints were baked into.

        Returns:
            Absolute path to this task/backend's checkpoint file.
        """
        return Path(golden_root) / self.checkpoint_relpath


def _load_golden_tasks_json(path: Path) -> tuple[dict, list[dict]]:
    with open(path) as f:
        raw_data = json.load(f)

    if isinstance(raw_data, dict):
        defaults = raw_data.get("defaults", {})
        raw_list = raw_data.get("tasks", [])
        if not isinstance(raw_list, list):
            raise TypeError(f"'tasks' field in {path} must be a list")
    elif isinstance(raw_data, list):
        defaults = {}
        raw_list = raw_data
    else:
        raise TypeError(f"{path} must contain a JSON list or an object with a top-level 'tasks' list")

    if not isinstance(defaults, dict):
        raise TypeError(f"'defaults' field in {path} must be an object")

    return defaults, raw_list


def _require_positive(name: str, value: int, task_id: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"golden task {task_id!r}: {name} must be a positive integer, got {value!r}")
    return value


def _validate_kpi_names(task_id: str, kpis: dict[str, list[KpiThreshold]]) -> None:
    """Ensure every configured KPI name is measurable for this task.

    A KPI is measurable if it is one of the universal, episode-derived KPIs
    (:data:`~golden_contracts.UNIVERSAL_KPIS`) or the task has a registered
    sim-state probe (see :mod:`golden_probes`) that can produce it. Without this
    check a typo (e.g. ``"success"`` for ``"success_rate"``) or a probe KPI
    configured before its probe is registered would parse fine but silently never
    gate, since the oracle treats an unmeasured KPI as non-gating.

    Raises:
        ValueError: If a configured KPI name is neither universal nor backed by a
            registered probe for ``task_id``.
    """
    if not kpis:
        return
    has_probe = get_probe(task_id) is not None
    for kpi_name in kpis:
        if kpi_name not in UNIVERSAL_KPIS and not has_probe:
            raise ValueError(
                f"golden task {task_id!r} configures KPI {kpi_name!r}, which is neither a universal KPI "
                f"({sorted(UNIVERSAL_KPIS)}) nor produced by a registered sim-state probe"
            )


def _expand(template: object, fmt: dict, *, task_id: str, field_name: str) -> str:
    """Expand ``{task_id}``/``{backend_key}`` placeholders, with task context on error."""
    try:
        return str(template).format(**fmt)
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError(
            f"golden task {task_id!r}: bad placeholder in checkpoint {field_name} {template!r} ({exc})"
        ) from exc


def load_golden_tasks(golden_tasks_json_path: Path | str | None = None) -> list[GoldenTaskConfig]:
    """Load all golden tasks, producing one :class:`GoldenTaskConfig` per backend combination.

    Args:
        golden_tasks_json_path: Path to ``golden_tasks.json``. Defaults to the one
            next to this module.

    Returns:
        List of :class:`GoldenTaskConfig`, one per (task_id, backend) combination.
    """
    path = Path(golden_tasks_json_path) if golden_tasks_json_path is not None else _DEFAULT_GOLDEN_TASKS_JSON
    defaults, raw_list = _load_golden_tasks_json(path)

    tasks: list[GoldenTaskConfig] = []
    for raw in raw_list:
        if not isinstance(raw, dict):
            raise TypeError(f"golden task entry in {path} must be an object")
        merged = {**defaults, **raw}

        task_id = merged["task_id"]
        checkpoint = merged.get("checkpoint")
        if not isinstance(checkpoint, dict) or not checkpoint.get("id") or not checkpoint.get("path"):
            raise ValueError(
                f"golden task {task_id!r} must define a 'checkpoint' object with non-empty 'id' and 'path'"
            )

        kpis = parse_kpi_thresholds(merged.get("kpis", {}))
        _validate_kpi_names(task_id, kpis)
        num_envs = _require_positive("num_envs", int(merged["num_envs"]), task_id)
        eval_steps = _require_positive("eval_steps", int(merged["eval_steps"]), task_id)
        backends: list[dict] = merged.get("backends", [])
        if not backends:
            raise ValueError(f"golden task {task_id!r} must define at least one backend")

        for backend_entry in backends:
            physics = normalize_physics_backend(backend_entry["physics"])
            if physics is None:
                raise ValueError(f"golden backend entry in {path} must define a non-default physics backend")
            render = normalize_render_backend(backend_entry.get("render"))
            backend_key = make_backend_key(physics, render)
            # Expand path/id placeholders once per backend so downstream code sees a concrete reference.
            fmt = {"task_id": task_id, "backend_key": backend_key}
            tasks.append(
                GoldenTaskConfig(
                    task_id=task_id,
                    physics_backend=physics,
                    render_backend=render,
                    preset=merged["preset"],
                    rl_library=merged["rl_library"],
                    agent=merged["agent"],
                    checkpoint_id=_expand(checkpoint["id"], fmt, task_id=task_id, field_name="id"),
                    checkpoint_relpath=_expand(checkpoint["path"], fmt, task_id=task_id, field_name="path"),
                    num_envs=num_envs,
                    eval_steps=eval_steps,
                    seed=int(merged["seed"]) if merged.get("seed") is not None else None,
                    deterministic=bool(merged.get("deterministic", True)),
                    kpis=kpis,
                    timeout_minutes=int(merged["timeout_minutes"]),
                    tags=merged.get("tags", ["always"]),
                    task_type=merged.get("type", "golden"),
                    runs_on=merged.get("runs_on", "gpu-l40s"),
                )
            )
    return tasks


def get_golden_task(
    task_id: str,
    backend_key: str,
    golden_tasks_json_path: Path | str | None = None,
) -> GoldenTaskConfig:
    """Return the :class:`GoldenTaskConfig` for a (task_id, backend_key) pair.

    Args:
        task_id: Task identifier to look up.
        backend_key: Backend key (e.g. ``"physx"``, ``"newton"``).
        golden_tasks_json_path: Optional path to ``golden_tasks.json``.

    Returns:
        The matching :class:`GoldenTaskConfig`.

    Raises:
        KeyError: If no golden task with the given (task_id, backend_key) exists.
    """
    for task in load_golden_tasks(golden_tasks_json_path):
        if task.task_id == task_id and task.backend_key == backend_key:
            return task
    raise KeyError(f"Golden task not found: task_id={task_id!r} backend_key={backend_key!r}")
