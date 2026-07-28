# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hard-set KPI thresholds for golden correctness checks.

The golden gate is *standalone*: unlike the performance gate (which compares a
measured FPS against a rolling baseline window), a golden KPI is judged against a
fixed, blessed target that ships in ``golden_tasks.json``. :class:`KpiThreshold`
is the golden analogue of :class:`~gate_types.FpsMeanThreshold`, with one added
degree of freedom: a :class:`KpiDirection`, because behavioural KPIs cross in
both directions (a *reward* below its floor is bad; a *pole angle* above its
ceiling is bad), whereas an FPS threshold is always a floor.

Verdict primitives (:class:`~gate_types.OracleVerdict`, the gating-verdict set)
are shared with the performance gate; only the threshold shape differs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

try:
    from .gate_types import THRESHOLD_VERDICTS, OracleVerdict
except ImportError:  # pragma: no cover - supports direct script imports
    from gate_types import THRESHOLD_VERDICTS, OracleVerdict


class KpiDirection(str, Enum):
    """Direction in which a KPI value crosses (violates) its threshold.

    ``FLOOR`` covers "more is better" KPIs (reward, success rate, survival
    length): a value *below* the threshold is a violation. ``CEILING`` covers
    "less is better" KPIs (e.g. a pole-angle bound, a drift metric): a value
    *above* the threshold is a violation.
    """

    FLOOR = "floor"
    CEILING = "ceiling"


@dataclass(frozen=True)
class KpiThreshold:
    """A single hard-set threshold for one behavioural KPI of a golden task.

    A threshold is *crossed* when the measured KPI violates it in its
    :attr:`direction`. A crossed threshold whose :attr:`verdict` is set
    contributes that verdict to the golden outcome; a threshold with
    :attr:`verdict` = ``None`` is *reporting-only* -- surfaced in outputs without
    ever changing the verdict (useful for shipping a placeholder target before it
    is calibrated and blessed).

    Args:
        name: Informative label for the threshold (e.g. ``"reward-floor"``).
        value: The threshold value, in the KPI's own unit.
        direction: Whether a violation is below (:attr:`KpiDirection.FLOOR`) or
            above (:attr:`KpiDirection.CEILING`) :attr:`value`.
        verdict: Gating verdict to raise when crossed, or ``None`` for
            reporting-only.
    """

    name: str
    value: float
    direction: KpiDirection
    verdict: OracleVerdict | None

    @property
    def is_gating(self) -> bool:
        """Whether crossing this threshold can change the golden verdict."""
        return self.verdict is not None

    def crosses(self, measured: float) -> bool:
        """Return whether ``measured`` violates this threshold in its direction."""
        if self.direction is KpiDirection.CEILING:
            return measured > self.value
        return measured < self.value

    def to_dict(self) -> dict:
        """Serialize to the ``golden_tasks.json`` entry shape."""
        return {
            "threshold_verdict": self.verdict.value if self.verdict is not None else None,
            "threshold_name": self.name,
            "threshold": self.value,
            "direction": self.direction.value,
        }

    @classmethod
    def from_dict(cls, raw: dict, *, context: str = "") -> KpiThreshold | None:
        """Parse and validate one raw threshold entry.

        Returns ``None`` when the entry has no ``threshold`` value (skipped), so a
        KPI can be declared with an empty/placeholder entry list without gating.

        Args:
            raw: One raw threshold object from ``golden_tasks.json``.
            context: Optional location hint (e.g. ``"cartpole/physx/reward"``)
                included in error messages.

        Raises:
            TypeError: If ``raw`` is not an object.
            ValueError: If ``threshold_name`` is missing/empty, ``threshold`` is
                non-numeric, ``direction`` is not a known :class:`KpiDirection`,
                or ``threshold_verdict`` is not a gating verdict.
        """
        where = f" ({context})" if context else ""
        if not isinstance(raw, dict):
            raise TypeError(f"kpi threshold entry{where} must be an object, got {type(raw).__name__}")

        name = raw.get("threshold_name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"kpi threshold entry{where} must define a non-empty 'threshold_name'")
        name = name.strip()

        value = raw.get("threshold")
        if value is None:
            return None
        try:
            threshold = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"kpi threshold entry {name!r} has non-numeric 'threshold': {value!r}")
        # Reject NaN/Infinity (which ``json.load`` and ``float`` both accept): a NaN
        # threshold never crosses and an infinite one is unreachable -- either would
        # silently disable the gate for this KPI.
        if not math.isfinite(threshold):
            raise ValueError(f"kpi threshold entry {name!r} has non-finite 'threshold': {value!r}")

        # Direction defaults to FLOOR (the common "more is better" case), so only
        # ceiling-style KPIs (e.g. a pole-angle bound) must declare it explicitly.
        direction_raw = raw.get("direction", KpiDirection.FLOOR.value)
        try:
            direction = KpiDirection(direction_raw)
        except ValueError:
            allowed_dirs = sorted(d.value for d in KpiDirection)
            raise ValueError(
                f"kpi threshold entry {name!r} has invalid 'direction' {direction_raw!r}; must be one of {allowed_dirs}"
            )

        verdict_raw = raw.get("threshold_verdict")
        if verdict_raw is None:
            verdict: OracleVerdict | None = None
        else:
            allowed = sorted(v.value for v in THRESHOLD_VERDICTS)
            try:
                verdict = OracleVerdict(verdict_raw)
            except ValueError:
                raise ValueError(
                    f"kpi threshold entry {name!r} has invalid 'threshold_verdict' {verdict_raw!r}; "
                    f"must be one of {allowed} (or omitted for reporting-only)"
                )
            if verdict not in THRESHOLD_VERDICTS:
                raise ValueError(
                    f"kpi threshold entry {name!r} has non-gating 'threshold_verdict' {verdict_raw!r}; "
                    f"must be one of {allowed} (or omitted for reporting-only)"
                )
        return cls(name=name, value=threshold, direction=direction, verdict=verdict)

    @classmethod
    def from_list(cls, raw_list, *, context: str = "") -> list[KpiThreshold]:
        """Parse a leaf list of raw threshold entries, dropping value-less entries."""
        if raw_list is None:
            return []
        if not isinstance(raw_list, list):
            raise TypeError(f"kpi threshold list ({context}) must be a list, got {type(raw_list).__name__}")
        parsed: list[KpiThreshold] = []
        for i, entry in enumerate(raw_list):
            threshold = cls.from_dict(entry, context=f"{context}[{i}]" if context else f"[{i}]")
            if threshold is not None:
                parsed.append(threshold)
        return parsed
