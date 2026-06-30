"""Pure statistical analysis for bisect verdict decisions.

No side effects, no I/O, no subprocess calls.  All functions are deterministic
given their inputs.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Pre-load stdlib 'bisect' before 'statistics' can shadow it.
# When bisect_agent/ is on sys.path position 0, bisect.py (the CLI entry
# point) would shadow the stdlib bisect module; statistics→random→bisect
# would then fail.  Pre-populate sys.modules['bisect'] with the real stdlib
# version if it isn't already there.
# ---------------------------------------------------------------------------
import sys as _sys
if "bisect" not in _sys.modules or not hasattr(_sys.modules["bisect"], "bisect_left"):
    import importlib.util as _ilu
    import importlib.machinery as _ilm
    import os as _os
    # Search sys.path for the C extension or .py, skipping bisect.py in bisect_agent/
    _this_dir = _os.path.dirname(_os.path.abspath(__file__))  # bisect_agent/core/
    _agent_dir = _os.path.dirname(_this_dir)                  # bisect_agent/
    for _d in list(_sys.path):
        if _d in ("", _agent_dir):
            continue
        for _suffix in (
            ".cpython-312-x86_64-linux-gnu.so",
            ".cpython-311-x86_64-linux-gnu.so",
            ".cpython-310-x86_64-linux-gnu.so",
            ".py",
        ):
            _candidate = _os.path.join(_d, "bisect" + _suffix)
            if _os.path.exists(_candidate):
                _spec = _ilu.spec_from_file_location("bisect", _candidate)
                if _spec is not None:
                    _mod = _ilu.module_from_spec(_spec)
                    _spec.loader.exec_module(_mod)  # type: ignore[union-attr]
                    _sys.modules["bisect"] = _mod
                break
        if "bisect" in _sys.modules and hasattr(_sys.modules["bisect"], "bisect_left"):
            break
    del _ilu, _ilm, _os
del _sys

import statistics
from typing import Any

# ---------------------------------------------------------------------------
# KPI field mapping
# ---------------------------------------------------------------------------

# Maps canonical kpi_key (used in stats dicts) to the field name in run_result
_KPI_FIELDS: dict[str, str] = {
    "fps_mean": "raw_fps_mean",
    "fps_p5": "raw_fps_p5",
    "fps_median": "raw_fps_median",
    "wall_time_s": "wall_time_s",
    "gpu_mem_used_mb": "gpu_mem_used_mb",
}

# Regression direction: "lower" means lower observed value = regression (fps KPIs),
# "higher" means higher observed value = regression (gpu_mem_mb, wall_time).
_REGRESSION_DIRECTION: dict[str, str] = {
    "fps_mean": "lower",
    "fps_p5": "lower",
    "fps_median": "lower",
    "wall_time_s": "higher",
    "gpu_mem_used_mb": "higher",
}

# Primary KPIs used for separation / bisect verdict decisions (in priority order)
_PRIMARY_KPIS: tuple[str, ...] = ("fps_mean", "fps_p5", "fps_median")

# KPIs excluded from bisect verdict and kpis_regressing.
# wall_time_s has very high CV in production and is exactly 0 in dev mode —
# including it produces false regression signals and should not drive verdicts.
# It remains in _KPI_FIELDS so compute_kpi_stats reports it for informational use.
_VERDICT_EXCLUDE_KPIS: frozenset[str] = frozenset({"wall_time_s"})

# Grounding CV threshold — a primary KPI with CV above this triggers more runs.
_CV_THRESHOLD: float = 0.08

# Separation thresholds
_SEP_RATIO_THRESHOLD: float = 1.5
_REL_CHANGE_THRESHOLD_PCT: float = 5.0

# Bisect thresholds (in MAD units from good median)
_BISECT_BAD_MAD_FACTOR: float = 2.0

# failure_phase values that indicate the commit itself caused the failure (→ BAD).
# Mirrors oracle.py's _BISECT_BAD_PHASES plus "import" which oracle also considers bad.
_FAILURE_PHASE_BAD: frozenset[str] = frozenset({"import", "init", "runtime"})

# failure_phase values that indicate an infrastructure / environment problem (→ SKIP).
_FAILURE_PHASE_SKIP: frozenset[str] = frozenset({
    "oom", "hang", "driver", "config_mismatch",
    "runner_error", "missing_result",
})


# ---------------------------------------------------------------------------
# compute_kpi_stats
# ---------------------------------------------------------------------------

def compute_kpi_stats(run_results: list[dict]) -> dict:
    """Compute median, MAD, and CV for each KPI across multiple run_results.

    Parameters
    ----------
    run_results:
        List of run_result dicts (matching run_result.schema.json).

    Returns
    -------
    dict
        ``{kpi_key: {"median": float, "mad": float, "cv": float, "n": int}}``
        Only KPIs with at least one non-null value are included.
    """
    # Collect per-KPI samples (skipping None values)
    samples: dict[str, list[float]] = {kpi: [] for kpi in _KPI_FIELDS}

    for rr in run_results:
        for kpi, field in _KPI_FIELDS.items():
            val = rr.get(field)
            if val is not None:
                try:
                    samples[kpi].append(float(val))
                except (TypeError, ValueError):
                    pass

    result: dict = {}
    for kpi, vals in samples.items():
        if not vals:
            continue
        n = len(vals)
        med = statistics.median(vals)
        mad = statistics.median([abs(v - med) for v in vals])

        # Coefficient of variation: MAD / median (avoid div-by-zero)
        if med != 0.0:
            cv = mad / abs(med)
        else:
            cv = 0.0

        result[kpi] = {
            "median": med,
            "mad": mad,
            "cv": cv,
            "n": n,
        }

    return result


# ---------------------------------------------------------------------------
# check_separation
# ---------------------------------------------------------------------------

def check_separation(good_stats: dict, bad_stats: dict) -> dict:
    """Check whether good and bad run distributions are statistically separated.

    Separation ratio for a KPI is defined as::

        sep_ratio = |good_median - bad_median| / max(good_mad, bad_mad, epsilon)

    A KPI is considered *regressing* when **both** conditions hold:

    1. ``sep_ratio >= 1.5``
    2. Relative change exceeds 5 % in the regression direction
       (lower for fps KPIs; higher for gpu_mem / wall_time).

    Parameters
    ----------
    good_stats:
        Output of :func:`compute_kpi_stats` for good-SHA runs.
    bad_stats:
        Output of :func:`compute_kpi_stats` for bad-SHA runs.

    Returns
    -------
    dict
        Keys:

        - ``separated`` (bool): True if at least one KPI regresses.
        - ``kpis_regressing`` (list[str]): KPIs meeting both separation criteria.
        - ``kpi_deltas`` (dict[str, float]): Relative change in % per KPI.
        - ``separation_ratios`` (dict[str, float]): sep_ratio per KPI.
        - ``verdict`` (str): ``"PROCEED"`` or ``"WARN_NO_SEPARATION"``.
        - ``note`` (str | None): Human-readable note when not separated.
    """
    kpis_regressing: list[str] = []
    kpi_deltas: dict[str, float] = {}
    separation_ratios: dict[str, float] = {}

    # Only compare KPIs present in both stat sets
    common_kpis = set(good_stats.keys()) & set(bad_stats.keys())

    for kpi in common_kpis:
        g = good_stats[kpi]
        b = bad_stats[kpi]
        good_median: float = g["median"]
        bad_median: float = b["median"]
        good_mad: float = g["mad"]
        bad_mad: float = b["mad"]

        # Avoid division by zero with a small epsilon
        denominator = max(good_mad, bad_mad, 1e-9)
        sep_ratio = abs(good_median - bad_median) / denominator
        separation_ratios[kpi] = round(sep_ratio, 4)

        # Relative change (signed, in %)
        if good_median != 0.0:
            rel_change_pct = (bad_median - good_median) / abs(good_median) * 100.0
        else:
            rel_change_pct = 0.0
        kpi_deltas[kpi] = round(rel_change_pct, 2)

        # Direction check
        direction = _REGRESSION_DIRECTION.get(kpi, "lower")
        if direction == "lower":
            # Regression = bad_median < good_median → rel_change_pct < 0
            in_regression_direction = rel_change_pct < -_REL_CHANGE_THRESHOLD_PCT
        else:
            # Regression = bad_median > good_median → rel_change_pct > 0
            in_regression_direction = rel_change_pct > _REL_CHANGE_THRESHOLD_PCT

        if sep_ratio >= _SEP_RATIO_THRESHOLD and in_regression_direction and kpi not in _VERDICT_EXCLUDE_KPIS:
            kpis_regressing.append(kpi)

    separated = len(kpis_regressing) > 0

    if separated:
        verdict = "PROCEED"
        note: str | None = None
    else:
        verdict = "WARN_NO_SEPARATION"
        note = (
            "No KPI shows clear statistical separation between good and bad commits. "
            "Bisect may produce unreliable results."
        )

    return {
        "separated": separated,
        "kpis_regressing": kpis_regressing,
        "kpi_deltas": kpi_deltas,
        "separation_ratios": separation_ratios,
        "verdict": verdict,
        "note": note,
    }


# ---------------------------------------------------------------------------
# classify_bisect_verdict
# ---------------------------------------------------------------------------

def classify_bisect_verdict(
    run_result: dict,
    good_stats: dict,
    kpis_regressing: list[str],
) -> str:
    """Classify a single run as GOOD, BAD, or SKIP for binary search.

    Parameters
    ----------
    run_result:
        A single run_result dict.
    good_stats:
        Output of :func:`compute_kpi_stats` for the known-good runs (baseline).
    kpis_regressing:
        List of KPI keys identified as regressing (from :func:`check_separation`).

    Returns
    -------
    str
        ``"GOOD"``, ``"BAD"``, or ``"SKIP"``.

    Rules
    -----
    - **BAD** immediately if ``failure_phase`` is in ``_FAILURE_PHASE_BAD``
      (``import``, ``init``, ``runtime``): the commit itself caused the crash.
    - **SKIP** if ``failure_phase`` is in ``_FAILURE_PHASE_SKIP``
      (``oom``, ``hang``, ``driver``, ``config_mismatch``, infra errors):
      infrastructure/environment problem — not caused by the commit.
    - **SKIP** if ``raw_fps_mean`` is absent (benchmark produced no output).
    - **BAD** if, for the regressing KPIs, the run's value crosses the threshold
      defined as ``good_median ± _BISECT_BAD_MAD_FACTOR * good_mad`` in the
      regression direction.
    - **GOOD** otherwise.
    """
    failure_phase = run_result.get("failure_phase")

    # Commit-caused crash → BAD (mirrors oracle.py _BISECT_BAD_PHASES + import)
    if failure_phase in _FAILURE_PHASE_BAD:
        return "BAD"

    # Infrastructure failure → SKIP (retry policy handled by bisector)
    if failure_phase is not None:
        return "SKIP"

    # No FPS output → SKIP
    if run_result.get("raw_fps_mean") is None:
        return "SKIP"

    if not kpis_regressing:
        # No regressing KPIs identified — can't make a classification
        return "SKIP"

    bad_count = 0
    checked = 0

    for kpi in kpis_regressing:
        field = _KPI_FIELDS.get(kpi)
        if field is None:
            continue
        val = run_result.get(field)
        if val is None:
            continue

        kpi_stat = good_stats.get(kpi)
        if kpi_stat is None:
            continue

        checked += 1
        good_median: float = kpi_stat["median"]
        good_mad: float = kpi_stat["mad"]
        threshold_distance = _BISECT_BAD_MAD_FACTOR * good_mad

        direction = _REGRESSION_DIRECTION.get(kpi, "lower")
        if direction == "lower":
            # BAD if val < good_median - threshold
            if val < good_median - threshold_distance:
                bad_count += 1
        else:
            # BAD if val > good_median + threshold
            if val > good_median + threshold_distance:
                bad_count += 1

    if checked == 0:
        return "SKIP"

    # BAD if the majority of regressing KPIs are in the bad range
    if bad_count > checked / 2:
        return "BAD"

    return "GOOD"
