"""
core/grounding.py — Grounding phase for the bisect agent.

Runs repeated benchmark experiments on the known-good and known-bad SHAs to:
  1. Establish empirical performance baselines with sufficient statistical confidence.
  2. Adaptively add more runs when coefficient of variation (CV) exceeds the threshold.
  3. Assess whether the good/bad distributions are statistically separated.
  4. Write grounding/result.json and return the grounding result dict.

Hard constraints (from DESIGN.md §5):
  - runner.py is the ONLY thing that triggers benchmark execution; grounding only
    calls the runner_run_commit callable it receives.
  - Every stage checks for its output artifact before running (resumable).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

# ---------------------------------------------------------------------------
# Import verdict functions — try relative first, fall back to absolute.
# ---------------------------------------------------------------------------
try:
    from .verdict import compute_kpi_stats, check_separation, _PRIMARY_KPIS, _CV_THRESHOLD
except ImportError:
    from core.verdict import compute_kpi_stats, check_separation, _PRIMARY_KPIS, _CV_THRESHOLD  # type: ignore[no-redef]

_SHA_SHORT = 12  # chars used for artifact dir names and log messages

logger = logging.getLogger(__name__)


def compute_grounding_result(
    good_results: list[dict],
    bad_results: list[dict],
    good_sha: str,
    bad_sha: str,
    task_id: str,
    backend: str,
    *,
    cv_threshold: float = _CV_THRESHOLD,
) -> dict:
    """Compute the grounding result dict from run results (no I/O).

    Filters high-variance KPIs by _PRIMARY_KPIS so only benchmark-relevant
    KPIs influence the verdict.  Used by both run_grounding (which writes the
    result to disk) and orchestrator._assess_grounding (which reconstructs the
    result from pre-existing run files).
    """
    good_stats = compute_kpi_stats(good_results)
    bad_stats = compute_kpi_stats(bad_results)
    separation = check_separation(good_stats, bad_stats)

    high_variance_kpis: list[str] = []
    for stats_dict, label in [(good_stats, "good"), (bad_stats, "bad")]:
        for kpi, kpi_data in stats_dict.items():
            if kpi in _PRIMARY_KPIS:
                cv = kpi_data.get("cv", 0.0)
                if cv > cv_threshold:
                    tag = f"{kpi}({label})"
                    if tag not in high_variance_kpis:
                        high_variance_kpis.append(tag)

    if not separation["separated"]:
        verdict_str = "WARN_NO_SEPARATION"
    elif high_variance_kpis:
        verdict_str = "WARN_HIGH_VARIANCE"
    else:
        verdict_str = "PROCEED"

    note = separation.get("note")
    if high_variance_kpis:
        hv_note = f"High variance KPIs after max runs: {', '.join(high_variance_kpis)}."
        note = f"{note} {hv_note}".strip() if note else hv_note

    return {
        "good_sha": good_sha,
        "bad_sha": bad_sha,
        "task_id": task_id,
        "backend": backend,
        "n_good": len(good_results),
        "n_bad": len(bad_results),
        "good_stats": good_stats,
        "bad_stats": bad_stats,
        "separated": separation["separated"],
        "kpis_regressing": separation["kpis_regressing"],
        "kpi_deltas": separation["kpi_deltas"],
        "separation_ratios": separation["separation_ratios"],
        "high_variance_tasks": high_variance_kpis,
        "verdict": verdict_str,
        "note": note,
    }


def run_grounding(
    good_sha: str,
    bad_sha: str,
    task_id: str,
    backend: str,
    run_dir: Path,
    runner_run_commit: Callable,
    *,
    n_start: int = 3,
    n_max: int = 5,
    cv_threshold: float = _CV_THRESHOLD,
    dev_mode: bool = False,
    dev_perf_map: dict | None = None,
) -> dict:
    """Run grounding experiments for good_sha and bad_sha.

    Parameters
    ----------
    good_sha:
        Known-good commit SHA.
    bad_sha:
        Known-bad commit SHA.
    task_id:
        IsaacLab task identifier (e.g. "Isaac-Velocity-Flat-G1-Direct").
    backend:
        Backend key (e.g. "newton").
    run_dir:
        Root directory for all run artifacts.  Grounding artifacts go under
        ``run_dir/grounding/``.
    runner_run_commit:
        Reference to ``core/runner.py::run_commit``.  Signature::

            run_commit(sha, task_id, backend, output_dir, *, dev_mode, dev_perf_map) -> dict

    n_start:
        Number of initial runs per SHA.
    n_max:
        Maximum number of runs per SHA (adaptive ceiling).
    cv_threshold:
        Coefficient of variation threshold above which more runs are added.
    dev_mode:
        Whether to run in dev/stub mode (no Docker/GPU required).
    dev_perf_map:
        SHA -> fps_mean mapping used in dev mode.

    Returns
    -------
    dict
        Grounding result conforming to ``grounding_result.schema.json``.
    """
    run_dir = Path(run_dir)
    grounding_dir = run_dir / "grounding"
    result_path = grounding_dir / "result.json"

    # ------------------------------------------------------------------
    # Resume: if result already exists, return it immediately.
    # ------------------------------------------------------------------
    if result_path.exists():
        logger.info("Grounding result already exists at %s — skipping.", result_path)
        with result_path.open() as fh:
            return json.load(fh)

    grounding_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Helper: run a single experiment and return its run_result dict.
    # Runner exceptions are caught so one bad run doesn't abort grounding.
    # ------------------------------------------------------------------
    def _run_one(sha: str, run_index: int) -> dict:
        artifact_subdir = grounding_dir / f"{sha[:_SHA_SHORT]}_{run_index}"
        artifact_subdir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Grounding: running %s run_index=%d → %s", sha[:_SHA_SHORT], run_index, artifact_subdir
        )
        try:
            result = runner_run_commit(
                sha,
                task_id,
                backend,
                artifact_subdir,
                dev_mode=dev_mode,
                dev_perf_map=dev_perf_map,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Grounding: runner raised for %s run_index=%d: %s",
                sha[:_SHA_SHORT], run_index, exc,
            )
            result = {
                "sha": sha, "task_id": task_id, "backend": backend,
                "run_index": run_index, "exit_code": -1,
                "failure_phase": "runner_error", "raw_fps_mean": None,
                "raw_fps_median": None, "raw_fps_p5": None, "raw_fps_p95": None,
                "gpu_mem_used_mb": None, "wall_time_s": None,
            }
        # Ensure run_index is set correctly (runner may set it to 0 always).
        result["run_index"] = run_index
        # Store relative artifact_dir so the result is portable.
        result["artifact_dir"] = str(artifact_subdir.relative_to(run_dir))
        return result

    # ------------------------------------------------------------------
    # Helper: check whether any primary KPI in stats exceeds cv_threshold.
    # ------------------------------------------------------------------
    def _needs_more_runs(stats: dict) -> bool:
        for kpi_key, kpi_stat_key in [
            ("fps_mean", "fps_mean"),
            ("fps_p5", "fps_p5"),
            ("fps_median", "fps_median"),
        ]:
            kpi_data = stats.get(kpi_key)
            if kpi_data is not None:
                cv = kpi_data.get("cv", 0.0)
                if cv > cv_threshold:
                    logger.info(
                        "KPI %s CV=%.4f exceeds threshold %.4f — adding more runs.",
                        kpi_key,
                        cv,
                        cv_threshold,
                    )
                    return True
        return False

    # ------------------------------------------------------------------
    # Phase 1: collect n_start runs for each SHA.
    # ------------------------------------------------------------------
    good_results: list[dict] = []
    bad_results: list[dict] = []

    for i in range(n_start):
        good_results.append(_run_one(good_sha, i))
    for i in range(n_start):
        bad_results.append(_run_one(bad_sha, i))

    # ------------------------------------------------------------------
    # Phase 2: adaptive extension — add runs in batches of 3 until CV is
    # acceptable or we reach n_max.
    # ------------------------------------------------------------------
    batch_size = 3
    while True:
        good_stats = compute_kpi_stats(good_results)
        bad_stats = compute_kpi_stats(bad_results)

        good_needs = len(good_results) < n_max and _needs_more_runs(good_stats)
        bad_needs = len(bad_results) < n_max and _needs_more_runs(bad_stats)

        if not good_needs and not bad_needs:
            break

        if good_needs:
            new_count = min(batch_size, n_max - len(good_results))
            if new_count <= 0:
                break
            for _ in range(new_count):
                good_results.append(_run_one(good_sha, len(good_results)))

        if bad_needs:
            new_count = min(batch_size, n_max - len(bad_results))
            if new_count <= 0:
                break
            for _ in range(new_count):
                bad_results.append(_run_one(bad_sha, len(bad_results)))

    # ------------------------------------------------------------------
    # Phase 3-5: compute final grounding result and write to disk.
    # ------------------------------------------------------------------
    grounding_result = compute_grounding_result(
        good_results=good_results,
        bad_results=bad_results,
        good_sha=good_sha,
        bad_sha=bad_sha,
        task_id=task_id,
        backend=backend,
        cv_threshold=cv_threshold,
    )

    with result_path.open("w") as fh:
        json.dump(grounding_result, fh, indent=2)

    logger.info(
        "Grounding complete: verdict=%s separated=%s kpis_regressing=%s n_good=%d n_bad=%d",
        grounding_result["verdict"],
        grounding_result["separated"],
        grounding_result["kpis_regressing"],
        grounding_result["n_good"],
        grounding_result["n_bad"],
    )
    return grounding_result
