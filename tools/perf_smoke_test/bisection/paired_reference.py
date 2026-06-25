# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Local paired-reference comparison for bisection candidates."""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .models import MeasurementPolicy


@dataclass(frozen=True)
class MeasurementSummary:
    """Summary statistics for repeated local benchmark measurements."""

    label: str
    values: list[float]
    median_fps: float
    mean_fps: float
    min_fps: float
    max_fps: float
    sample_count: int
    spread_pct: float

    def to_json(self) -> dict[str, Any]:
        """Serialize the summary to JSON."""
        return asdict(self)


@dataclass(frozen=True)
class ReferenceCheck:
    """Result of checking whether local good/bad refs are separated enough."""

    reproduced: bool
    regression_pct: float | None
    note: str | None
    effective_threshold_pct: float | None = None
    reference_noise_pct: float | None = None

    def to_json(self) -> dict[str, Any]:
        """Serialize the reference check to JSON."""
        return asdict(self)


@dataclass(frozen=True)
class PairedComparison:
    """GOOD/BAD/UNCLEAR classification against local reference measurements."""

    verdict: str
    measured_fps: float
    baseline_fps: float
    regression_pct: float
    threshold_source: str
    effective_threshold_pct: float
    reference_noise_pct: float
    gray_zone_pct: float
    note: str | None = None

    def to_json(self) -> dict[str, Any]:
        """Serialize the comparison result to JSON."""
        return asdict(self)


def _regression_pct(measured_fps: float, baseline_fps: float) -> float:
    """Return the signed percent delta relative to ``baseline_fps``."""
    if baseline_fps <= 0.0:
        raise ValueError("baseline FPS must be positive for paired-reference comparison")
    return (measured_fps - baseline_fps) / baseline_fps * 100.0


def _reference_noise_pct(good: MeasurementSummary, bad: MeasurementSummary) -> float:
    """Return the reference noise floor as the wider good/bad spread."""
    return max(good.spread_pct, bad.spread_pct)


def _effective_threshold_pct(policy: MeasurementPolicy, reference_noise_pct: float) -> float:
    """Return the local regression threshold after accounting for observed noise."""
    return max(policy.min_regression_pct, policy.reference_noise_multiplier * reference_noise_pct)


def _effective_gray_zone_pct(policy: MeasurementPolicy, reference_noise_pct: float) -> float:
    """Return the uncertainty band around the effective threshold."""
    return max(policy.gray_zone_pct, reference_noise_pct)


def fps_from_result(bench_result: dict[str, Any]) -> float:
    """Extract the FPS value used for local bisection comparisons."""
    value = bench_result.get("raw_fps_mean")
    if value is None:
        raise KeyError("perf_smoke_test_result.json does not contain raw_fps_mean")
    return float(value)


def fps_from_artifact(artifact_dir: Path) -> float:
    """Extract FPS from a ``perf_smoke_test_result.json`` artifact directory."""
    with (artifact_dir / "perf_smoke_test_result.json").open(encoding="utf-8") as fh:
        return fps_from_result(json.load(fh))


def summarize_measurements(label: str, values: list[float]) -> MeasurementSummary:
    """Summarize repeated FPS measurements with robust, simple statistics."""
    if not values:
        raise ValueError(f"{label} needs at least one FPS measurement")
    median_fps = float(statistics.median(values))
    mean_fps = float(statistics.fmean(values))
    min_fps = min(values)
    max_fps = max(values)
    median_abs_deviation = float(statistics.median(abs(value - median_fps) for value in values))
    spread_pct = (1.4826 * median_abs_deviation / median_fps * 100.0) if median_fps > 0.0 else float("inf")
    return MeasurementSummary(
        label=label,
        values=[float(value) for value in values],
        median_fps=median_fps,
        mean_fps=mean_fps,
        min_fps=min_fps,
        max_fps=max_fps,
        sample_count=len(values),
        spread_pct=spread_pct,
    )


def check_reference_signal(
    good: MeasurementSummary, bad: MeasurementSummary, policy: MeasurementPolicy
) -> ReferenceCheck:
    """Check whether local good/bad measurements show a usable step regression."""
    if good.median_fps <= 0.0:
        return ReferenceCheck(False, None, "good_ref_median_fps_not_positive")
    if good.spread_pct > policy.max_reference_spread_pct:
        return ReferenceCheck(False, None, "good_ref_measurements_too_noisy")
    if bad.spread_pct > policy.max_reference_spread_pct:
        return ReferenceCheck(False, None, "bad_ref_measurements_too_noisy")

    reference_noise_pct = _reference_noise_pct(good, bad)
    effective_threshold_pct = _effective_threshold_pct(policy, reference_noise_pct)
    regression_pct = _regression_pct(bad.median_fps, good.median_fps)
    if regression_pct > -effective_threshold_pct:
        return ReferenceCheck(
            False,
            regression_pct,
            "local_regression_not_reproduced",
            effective_threshold_pct=effective_threshold_pct,
            reference_noise_pct=reference_noise_pct,
        )
    return ReferenceCheck(
        True,
        regression_pct,
        None,
        effective_threshold_pct=effective_threshold_pct,
        reference_noise_pct=reference_noise_pct,
    )


def compare_candidate(
    candidate: MeasurementSummary,
    good: MeasurementSummary,
    policy: MeasurementPolicy,
    *,
    reference_noise_pct: float = 0.0,
) -> PairedComparison:
    """Classify a candidate against the local good-ref baseline."""
    regression_pct = _regression_pct(candidate.median_fps, good.median_fps)
    effective_threshold_pct = _effective_threshold_pct(policy, reference_noise_pct)
    gray_zone_pct = _effective_gray_zone_pct(policy, reference_noise_pct)
    bad_cutoff = -(effective_threshold_pct + gray_zone_pct)
    good_cutoff = -(effective_threshold_pct - gray_zone_pct)
    if regression_pct <= bad_cutoff:
        verdict = "BAD"
        note = None
    elif regression_pct >= good_cutoff:
        verdict = "GOOD"
        note = None
    else:
        verdict = "UNCLEAR"
        note = "candidate_in_gray_zone"
    return PairedComparison(
        verdict=verdict,
        measured_fps=candidate.median_fps,
        baseline_fps=good.median_fps,
        regression_pct=regression_pct,
        threshold_source=f"paired_reference: max(min_regression_pct, reference_noise_multiplier * reference_noise_pct)"
        f" = max({policy.min_regression_pct:.3g}, {policy.reference_noise_multiplier:.3g} * "
        f"{reference_noise_pct:.3g}) = {effective_threshold_pct:.3g}",
        effective_threshold_pct=effective_threshold_pct,
        reference_noise_pct=reference_noise_pct,
        gray_zone_pct=gray_zone_pct,
        note=note,
    )
