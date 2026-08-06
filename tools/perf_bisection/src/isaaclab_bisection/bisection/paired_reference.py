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

from .models import MeasurementPolicy, MetricSpec


@dataclass(frozen=True)
class MeasurementSummary:
    """Summary statistics for repeated local benchmark measurements."""

    label: str
    metric_name: str
    unit: str | None
    values: list[float]
    median_value: float
    mean_value: float
    min_value: float
    max_value: float
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
    metric_name: str
    metric_unit: str | None
    measured_value: float
    baseline_value: float
    regression_pct: float
    threshold_source: str
    effective_threshold_pct: float
    reference_noise_pct: float
    gray_zone_pct: float
    note: str | None = None

    def to_json(self) -> dict[str, Any]:
        """Serialize the comparison result to JSON."""
        return asdict(self)


def _delta_pct(measured_value: float, baseline_value: float) -> float:
    """Return the signed percent delta relative to ``baseline_value``."""
    if baseline_value <= 0.0:
        raise ValueError("baseline value must be positive for paired-reference comparison")
    return (measured_value - baseline_value) / baseline_value * 100.0


def _regression_pct(measured_value: float, baseline_value: float, metric: MetricSpec) -> float:
    """Return a positive regression percentage for the selected metric direction."""
    delta_pct = _delta_pct(measured_value, baseline_value)
    if metric.regression_direction == "decrease":
        return -delta_pct
    if metric.regression_direction == "increase":
        return delta_pct
    raise ValueError("metric.regression_direction must be 'decrease' or 'increase'")


def _reference_noise_pct(good: MeasurementSummary, bad: MeasurementSummary) -> float:
    """Return the reference noise floor as the wider good/bad spread."""
    return max(good.spread_pct, bad.spread_pct)


def _effective_threshold_pct(policy: MeasurementPolicy, reference_noise_pct: float) -> float:
    """Return the local regression threshold after accounting for observed noise."""
    return max(policy.min_regression_pct, policy.reference_noise_multiplier * reference_noise_pct)


def _effective_gray_zone_pct(policy: MeasurementPolicy, reference_noise_pct: float) -> float:
    """Return the uncertainty band around the effective threshold."""
    return max(policy.gray_zone_pct, reference_noise_pct)


def _numeric_dotted_keys(data: Any, *, prefix: str = "", max_keys: int = 40) -> list[str]:
    """Collect dotted paths to numeric leaves in a JSON-like dict (bounded, sorted).

    Used to build actionable "did you mean" hints when a configured ``metric.result_path``
    is not present, so the caller can pick a valid metric key without reading the whole
    artifact by hand.
    """
    keys: list[str] = []

    def _walk(node: Any, path: str) -> None:
        if len(keys) >= max_keys or not isinstance(node, dict):
            return
        for key, value in node.items():
            dotted = f"{path}.{key}" if path else key
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                keys.append(dotted)
            elif isinstance(value, dict):
                _walk(value, dotted)
            if len(keys) >= max_keys:
                return

    _walk(data, prefix)
    return sorted(keys)


def _metric_key_error(data: dict[str, Any], dotted_path: str) -> KeyError:
    """Build a KeyError for a missing metric path, listing available numeric keys."""
    available = _numeric_dotted_keys(data)
    hint = ", ".join(available) if available else "(no numeric keys found in result)"
    return KeyError(
        f"metric path {dotted_path!r} not found in perf_smoke_test_result.json. Available numeric metric keys: {hint}"
    )


def _get_path(data: dict[str, Any], dotted_path: str, *, root: dict[str, Any] | None = None) -> Any:
    """Read a dotted path from a JSON-like dict."""
    value: Any = data
    for part in dotted_path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise _metric_key_error(root if root is not None else data, dotted_path)
        value = value[part]
    return value


def metric_from_result(bench_result: dict[str, Any], metric: MetricSpec) -> float:
    """Extract the selected numeric metric from a benchmark result."""
    value = _get_path(bench_result, metric.result_path, root=bench_result)
    if value is None:
        raise _metric_key_error(bench_result, metric.result_path)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"metric {metric.result_path!r} must be numeric, got {type(value).__name__}")
    return float(value)


def metric_from_artifact(artifact_dir: Path, metric: MetricSpec) -> float:
    """Extract the selected metric from a ``perf_smoke_test_result.json`` artifact directory."""
    with (artifact_dir / "perf_smoke_test_result.json").open(encoding="utf-8") as fh:
        return metric_from_result(json.load(fh), metric)


def summarize_measurements(label: str, metric: MetricSpec, values: list[float]) -> MeasurementSummary:
    """Summarize repeated metric measurements with robust, simple statistics."""
    if not values:
        raise ValueError(f"{label} needs at least one metric measurement")
    median_value = float(statistics.median(values))
    mean_value = float(statistics.fmean(values))
    min_value = min(values)
    max_value = max(values)
    median_abs_deviation = float(statistics.median(abs(value - median_value) for value in values))
    spread_pct = (1.4826 * median_abs_deviation / median_value * 100.0) if median_value > 0.0 else float("inf")
    return MeasurementSummary(
        label=label,
        metric_name=metric.name,
        unit=metric.unit,
        values=[float(value) for value in values],
        median_value=median_value,
        mean_value=mean_value,
        min_value=min_value,
        max_value=max_value,
        sample_count=len(values),
        spread_pct=spread_pct,
    )


def check_reference_signal(
    good: MeasurementSummary, bad: MeasurementSummary, metric: MetricSpec, policy: MeasurementPolicy
) -> ReferenceCheck:
    """Check whether local good/bad measurements show a usable step regression."""
    if good.median_value <= 0.0:
        return ReferenceCheck(False, None, "good_ref_median_value_not_positive")
    if good.spread_pct > policy.max_reference_spread_pct:
        return ReferenceCheck(False, None, "good_ref_measurements_too_noisy")
    if bad.spread_pct > policy.max_reference_spread_pct:
        return ReferenceCheck(False, None, "bad_ref_measurements_too_noisy")

    reference_noise_pct = _reference_noise_pct(good, bad)
    effective_threshold_pct = _effective_threshold_pct(policy, reference_noise_pct)
    regression_pct = _regression_pct(bad.median_value, good.median_value, metric)
    if regression_pct < effective_threshold_pct:
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
    metric: MetricSpec,
    policy: MeasurementPolicy,
    *,
    reference_noise_pct: float = 0.0,
) -> PairedComparison:
    """Classify a candidate against the local good-ref baseline."""
    regression_pct = _regression_pct(candidate.median_value, good.median_value, metric)
    effective_threshold_pct = _effective_threshold_pct(policy, reference_noise_pct)
    gray_zone_pct = _effective_gray_zone_pct(policy, reference_noise_pct)
    bad_cutoff = effective_threshold_pct + gray_zone_pct
    good_cutoff = max(0.0, effective_threshold_pct - gray_zone_pct)
    if regression_pct >= bad_cutoff:
        verdict = "BAD"
        note = None
    elif regression_pct <= good_cutoff:
        verdict = "GOOD"
        note = None
    else:
        verdict = "UNCLEAR"
        note = "candidate_in_gray_zone"
    return PairedComparison(
        verdict=verdict,
        metric_name=metric.name,
        metric_unit=metric.unit,
        measured_value=candidate.median_value,
        baseline_value=good.median_value,
        regression_pct=regression_pct,
        threshold_source=f"paired_reference: max(min_regression_pct, reference_noise_multiplier * reference_noise_pct)"
        f" = max({policy.min_regression_pct:.3g}, {policy.reference_noise_multiplier:.3g} * "
        f"{reference_noise_pct:.3g}) = {effective_threshold_pct:.3g}",
        effective_threshold_pct=effective_threshold_pct,
        reference_noise_pct=reference_noise_pct,
        gray_zone_pct=gray_zone_pct,
        note=note,
    )
