# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Oracle layer for the golden correctness gate.

Unlike the performance oracle (:mod:`oracle`), the golden oracle is
**standalone**: it judges a rollout's behavioural KPIs against the hard-set,
blessed thresholds shipped in ``golden_tasks.json`` -- there is no rolling
baseline window, no cross-run pooling, and therefore no compatibility hashing.
A configured KPI that is measured is compared to its thresholds; the overall
verdict is the most severe verdict any gating threshold raises.

Verdict primitives and the bisect mapping are shared with the performance gate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

try:
    from .gate_types import (
        BisectVerdict,
        FailurePhase,
        OracleVerdict,
        worst_verdict,
    )
    from .golden_contracts import GoldenResult
    from .golden_kpi import KpiThreshold
except ImportError:  # pragma: no cover - supports direct script imports
    from gate_types import (
        BisectVerdict,
        FailurePhase,
        OracleVerdict,
        worst_verdict,
    )
    from golden_contracts import GoldenResult
    from golden_kpi import KpiThreshold


@dataclass
class GoldenOracleResult:
    """Full verdict record produced by :func:`evaluate`."""

    verdict: OracleVerdict
    bisect_verdict: str
    failure_phase: str | None
    task_id: str
    backend: str
    checkpoint_id: str | None
    reward_mean: float | None
    ep_length_mean: float | None
    success_rate: float | None
    num_episodes: int | None
    wall_time_s: float | None
    was_retried: bool
    kpi_results: list[dict] = field(default_factory=list)
    note: str | None = None


_BISECT_BAD_PHASES: frozenset[str] = frozenset({FailurePhase.INIT.value, FailurePhase.RUNTIME.value})


def _bisect_verdict(verdict: OracleVerdict, was_retried: bool, failure_phase: str | None) -> str:
    """Compute the bisect-friendly label for a golden verdict.

    A golden ``BLOCK`` is a clean behavioural good->bad signal, so it maps to
    ``BAD``. A HARD_FAILURE maps to ``BAD`` only when it looks like the code's
    fault (an init/runtime crash); setup failures such as a missing checkpoint
    map to ``SKIP`` so bisection does not blame an unrelated commit.
    """
    if verdict == OracleVerdict.PASS:
        return BisectVerdict.SKIP.value if was_retried else BisectVerdict.GOOD.value
    if verdict == OracleVerdict.WARN:
        return BisectVerdict.SKIP.value
    if verdict == OracleVerdict.BLOCK:
        return BisectVerdict.BAD.value
    if failure_phase in _BISECT_BAD_PHASES:
        return BisectVerdict.BAD.value
    return BisectVerdict.SKIP.value


def _hard_failure(result: GoldenResult, *, note: str | None, phase: str | None = None) -> GoldenOracleResult:
    verdict = OracleVerdict.HARD_FAILURE
    was_retried = bool(result.was_retried)
    # A config mismatch is normalized to CONFIG_MISMATCH so the bisect label is
    # SKIP (a setup problem), not BAD -- mirroring the perf oracle.
    failure_phase = phase if phase is not None else result.failure_phase
    return GoldenOracleResult(
        verdict=verdict,
        bisect_verdict=_bisect_verdict(verdict, was_retried, failure_phase),
        failure_phase=failure_phase,
        task_id=result.task_id,
        backend=result.backend_key or result.backend,
        checkpoint_id=result.checkpoint_id,
        reward_mean=result.reward_mean,
        ep_length_mean=result.ep_length_mean,
        success_rate=result.success_rate,
        num_episodes=result.num_episodes,
        wall_time_s=result.wall_time_s,
        was_retried=was_retried,
        note=note,
    )


def evaluate(result: GoldenResult, kpis: dict[str, list[KpiThreshold]]) -> GoldenOracleResult:
    """Judge a golden rollout's KPIs against their hard-set thresholds.

    Args:
        result: The rollout artifact for one (task, backend).
        kpis: The task's configured KPI thresholds, keyed by KPI name.

    Returns:
        A :class:`GoldenOracleResult` whose verdict is the most severe raised by
        any gating threshold. A run that produced no rollout output is a
        HARD_FAILURE; a configured KPI that was not measured this run is noted
        but does not gate on its own (a good stability policy may legitimately
        complete no episode, leaving ``reward`` unmeasured). If nothing at all was
        measured, the verdict is WARN.
    """
    was_retried = bool(result.was_retried)

    if result.config_mismatch or result.failure_phase == FailurePhase.CONFIG_MISMATCH.value:
        note = str(result.config_mismatch) if result.config_mismatch else "config_mismatch"
        return _hard_failure(result, note=note, phase=FailurePhase.CONFIG_MISMATCH.value)
    if not result.golden_info_present:
        return _hard_failure(result, note="no_golden_output")

    notes: list[str] = []
    kpi_results: list[dict] = []
    verdicts: list[OracleVerdict] = []
    measured_count = 0

    for kpi_name, thresholds in kpis.items():
        measured = result.measured_kpi(kpi_name)
        if measured is None:
            notes.append(f"unmeasured:{kpi_name}")
            kpi_results.append({"kpi": kpi_name, "measured": None, "verdict": None, "crossed_thresholds": []})
            continue
        # A non-finite KPI (NaN/inf) means a diverged/garbage rollout: it crosses no
        # floor or ceiling, so without this guard it would silently PASS. Fail the
        # whole run, mirroring the perf oracle's non-numeric/<=0 dead-run handling.
        if not math.isfinite(measured):
            return _hard_failure(result, note=f"non_finite:{kpi_name}={measured}")
        measured_count += 1

        crossed = [t for t in thresholds if t.crosses(measured)]
        kpi_verdict = OracleVerdict.PASS
        for t in crossed:
            if t.is_gating:
                kpi_verdict = worst_verdict(kpi_verdict, t.verdict)
        verdicts.append(kpi_verdict)
        kpi_results.append(
            {
                "kpi": kpi_name,
                "measured": measured,
                "verdict": kpi_verdict.value,
                "crossed_thresholds": [
                    {
                        "threshold_name": t.name,
                        "threshold": t.value,
                        "direction": t.direction.value,
                        "threshold_verdict": t.verdict.value if t.verdict is not None else None,
                        "gating": t.is_gating,
                    }
                    for t in crossed
                ],
            }
        )

    verdict = worst_verdict(*verdicts) if verdicts else OracleVerdict.PASS
    if measured_count == 0:
        verdict = worst_verdict(verdict, OracleVerdict.WARN)
        notes.append("no_kpi_measured")
    if verdict == OracleVerdict.PASS and was_retried:
        verdict = OracleVerdict.WARN
        notes.append("was_retried")

    return GoldenOracleResult(
        verdict=verdict,
        bisect_verdict=_bisect_verdict(verdict, was_retried, result.failure_phase),
        failure_phase=result.failure_phase,
        task_id=result.task_id,
        backend=result.backend_key or result.backend,
        checkpoint_id=result.checkpoint_id,
        reward_mean=result.reward_mean,
        ep_length_mean=result.ep_length_mean,
        success_rate=result.success_rate,
        num_episodes=result.num_episodes,
        wall_time_s=result.wall_time_s,
        was_retried=was_retried,
        kpi_results=kpi_results,
        note="; ".join(notes) if notes else None,
    )
