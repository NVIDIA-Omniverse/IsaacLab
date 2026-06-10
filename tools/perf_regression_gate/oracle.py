# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Oracle layer for the CI performance regression gate.

Provides verdict computation (PASS / WARN / BLOCK / HARD_FAILURE) by comparing
a measured FPS sample against a rolling baseline.\
"""

import json
import statistics
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class OracleVerdict(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    BLOCK = "BLOCK"
    HARD_FAILURE = "HARD_FAILURE"


@dataclass
class Baseline:
    """Rolling-window statistics used as the comparison reference for a single (task, backend) pair.

    Args:
        median_fps: Median FPS computed from the baseline window
        mad_fps: Median absolute deviation of FPS in the baseline window
        k_warn: Number of MADs below the median that triggers a WARN verdict
        k_block: Number of MADs below the median that triggers a BLOCK verdict
        sample_count: Number of samples in the window used to compute the stats
    """

    median_fps: float
    mad_fps: float
    k_warn: float = 2.5  # TODO: decide on real threshold values
    k_block: float = 4.0 # TODO: decide on real threshold values
    sample_count: int = 0


@dataclass
class OracleResult:
    """Full verdict record produced by :func:`compare`"""

    verdict: OracleVerdict        # High-level verdict: PASS / WARN / BLOCK / HARD_FAILURE
    bisect_verdict: str           # GOOD / BAD / SKIP for bisect compatibility
    failure_phase: str | None     # Phase classification from build_bench_result (e.g. "import", "init", "runtime", "oom", "hang", "driver", or None)
    measured_fps: float | None    # Mean FPS after excluded-frame filtering, or None on hard failure; blocking metric
    baseline_fps: float | None    # baseline.median_fps, or None
    regression_pct: float | None  # ((measured_fps - baseline_fps) / baseline_fps) * 100, or None
    fps_median: float | None      # Median FPS of the filtered series [informational]
    fps_p5: float | None          # 5th-percentile FPS of the filtered series [informational]
    fps_p95: float | None         # 95th-percentile FPS of the filtered series [informational]
    gpu_mem_used_mb: float | None # GPU memory used at benchmark time [MiB], or None
    startup_time_s: float | None  # Startup time reported by the benchmark process [s]
    wall_time_s: float | None     # Wall-clock time of the benchmark run [s]
    was_retried: bool             # True when the benchmark succeeded only after a retry, False otherwise
    task_id: str                  # Benchmark task identifier
    backend: str                  # Physics/Render backend name


# ---------------------------------------------------------------------------
# Bisect verdict mapping (spec section "Bisect verdict mapping")
# ---------------------------------------------------------------------------

# failure_phase values that map HARD_FAILURE -> "BAD"
_BISECT_BAD_PHASES: frozenset[str] = frozenset({"init", "runtime"})

# failure_phase values (and None) that map HARD_FAILURE -> "SKIP"
_BISECT_SKIP_PHASES: frozenset[str | None] = frozenset({"import", "driver", "oom", "hang", None})


def _bisect_verdict(verdict: OracleVerdict, was_retried: bool, failure_phase: str | None) -> str:
    """Compute the bisect-friendly label for a given verdict"""
    if verdict == OracleVerdict.PASS:
        return "SKIP" if was_retried else "GOOD"
    if verdict == OracleVerdict.WARN:
        return "SKIP"
    if verdict == OracleVerdict.BLOCK:
        return "BAD"
    # HARD_FAILURE
    if failure_phase in _BISECT_BAD_PHASES:
        return "BAD"
    return "SKIP"


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def _percentile(sorted_data: list[float], p: float) -> float:
    """Linear-interpolation percentile on a pre-sorted list

    Args:
        sorted_data: Ascending-sorted FPS values (must be non-empty)
        p: Percentile in [0, 100]

    Returns:
        Interpolated value at the requested percentile
    """
    n = len(sorted_data)
    if n == 1:
        return sorted_data[0]
    idx = p / 100.0 * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    return sorted_data[lo] + (sorted_data[hi] - sorted_data[lo]) * (idx - lo)


def apply_excluded_frames(fps_series: list[float], excluded_frames: frozenset[int]) -> list[float]:
    """Return fps_series with frames at 0-based indices listed in excluded_frames removed"""
    if not excluded_frames:
        return list(fps_series)
    return [fps for idx, fps in enumerate(fps_series) if idx not in excluded_frames]


def compare(
    bench_result: dict,
    baseline: "Baseline | None",
    fps_mean_floor: float,
    excluded_frames: "frozenset[int]",
    artifact_dir: "Path",
) -> OracleResult:
    """Compare a benchmark result against its baseline and return an :class:`OracleResult`

    Oracle logic (in order):

    1. If ``failure_phase`` is set **and** ``perf_regression_gate_info_present`` is False, return
       ``HARD_FAILURE`` immediately without reading any files
    2. Load ``artifact_dir/perf_regression_gate_info.json`` and extract the FPS series
    3. Filter the series with :func:`apply_excluded_frames`.
    4. Compute mean FPS with :mod:`statistics`.
    5. Apply hard-floor check, then baseline thresholds (or seed-run PASS)
    6. Downgrade PASS to WARN when ``was_retried`` is True
    7. Compute bisect verdict and regression percentage

    Args:
        bench_result: Dict matching the ``perf_regression_gate_result.json`` schema
        baseline: Rolling baseline statistics, or None for a seed run
        fps_mean_floor: Absolute minimum acceptable FPS (hard floor)
        excluded_frames: 0-based frame indices to drop before computing mean FPS
        artifact_dir: Directory that contains ``perf_regression_gate_info.json``

    Returns:
        Fully populated :class:`OracleResult`
    """
    task_id: str = bench_result["task_id"]
    backend: str = bench_result["backend"]
    failure_phase: str | None = bench_result["failure_phase"]
    was_retried: bool = bench_result["was_retried"]
    startup_time_s: float | None = bench_result.get("startup_time_s")
    wall_time_s: float | None = bench_result.get("wall_time_s")
    gpu_mem_used_mb: float | None = (bench_result.get("gpu_diag") or {}).get("gpu_mem_used_mb")

    # Treat missing perf data as HARD_FAILURE regardless of failure_phase
    if not bench_result["perf_regression_gate_info_present"]:
        bv = _bisect_verdict(OracleVerdict.HARD_FAILURE, was_retried, failure_phase)
        return OracleResult(
            verdict=OracleVerdict.HARD_FAILURE,
            bisect_verdict=bv,
            failure_phase=failure_phase,
            measured_fps=None,
            baseline_fps=None,
            regression_pct=None,
            fps_median=None,
            fps_p5=None,
            fps_p95=None,
            gpu_mem_used_mb=gpu_mem_used_mb,
            startup_time_s=startup_time_s,
            wall_time_s=wall_time_s,
            was_retried=was_retried,
            task_id=task_id,
            backend=backend,
        )

    # Load perf_regression_gate_info.json
    perf_regression_gate_info_path = Path(artifact_dir) / "perf_regression_gate_info.json"
    with perf_regression_gate_info_path.open() as fh:
        perf_regression_gate_info = json.load(fh)

    # Extract FPS series
    fps_series: list[float] = []
    for phase in perf_regression_gate_info:
        if phase.get("phase_name") == "runtime":
            for measurement in phase.get("measurements", []):
                if measurement.get("name", "").endswith("Step Frametimes"):
                    fps_series = measurement["value"]["Environment step effective FPS"]
                    break
            break

    # Compute filtered data, mean, and informational statistics
    filtered = apply_excluded_frames(fps_series, excluded_frames)
    if not filtered:
        bv = _bisect_verdict(OracleVerdict.HARD_FAILURE, was_retried, failure_phase)
        return OracleResult(
            verdict=OracleVerdict.HARD_FAILURE,
            bisect_verdict=bv,
            failure_phase=failure_phase,
            measured_fps=None,
            baseline_fps=None,
            regression_pct=None,
            fps_median=None,
            fps_p5=None,
            fps_p95=None,
            gpu_mem_used_mb=gpu_mem_used_mb,
            startup_time_s=startup_time_s,
            wall_time_s=wall_time_s,
            was_retried=was_retried,
            task_id=task_id,
            backend=backend,
        )
    mean_fps = statistics.mean(filtered)
    sorted_filtered = sorted(filtered)
    fps_median = _percentile(sorted_filtered, 50.0)
    fps_p5 = _percentile(sorted_filtered, 5.0)
    fps_p95 = _percentile(sorted_filtered, 95.0)

    # Verdict logic
    _MIN_SAMPLES_FOR_MAD = 2  # TODO: remove

    if mean_fps < fps_mean_floor:
        verdict = OracleVerdict.BLOCK
    elif baseline is None or baseline.sample_count < _MIN_SAMPLES_FOR_MAD:
        verdict = OracleVerdict.PASS
    else:
        block_thresh = baseline.median_fps - baseline.k_block * baseline.mad_fps
        warn_thresh = baseline.median_fps - baseline.k_warn * baseline.mad_fps
        if mean_fps < block_thresh:
            verdict = OracleVerdict.BLOCK
        elif mean_fps < warn_thresh:
            verdict = OracleVerdict.WARN
        else:
            verdict = OracleVerdict.PASS
    if verdict == OracleVerdict.PASS and was_retried:
        verdict = OracleVerdict.WARN

    baseline_fps: float | None = baseline.median_fps if baseline is not None else None
    regression_pct: float | None = None
    if baseline_fps is not None:
        regression_pct = ((mean_fps - baseline_fps) / baseline_fps) * 100.0

    bisection_verdict = _bisect_verdict(verdict, was_retried, failure_phase)

    return OracleResult(
        verdict=verdict,
        bisect_verdict=bisection_verdict,
        failure_phase=failure_phase,
        measured_fps=mean_fps,
        baseline_fps=baseline_fps,
        regression_pct=regression_pct,
        fps_median=fps_median,
        fps_p5=fps_p5,
        fps_p95=fps_p95,
        gpu_mem_used_mb=gpu_mem_used_mb,
        startup_time_s=startup_time_s,
        wall_time_s=wall_time_s,
        was_retried=was_retried,
        task_id=task_id,
        backend=backend,
    )
