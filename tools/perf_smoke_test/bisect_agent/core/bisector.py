"""
core/bisector.py — Leftmost-BAD binary search for the bisect agent.

Given an ordered list of commits (oldest-first, exclusive of good_sha, inclusive
of bad_sha) and a grounding result, performs a standard leftmost-BAD binary search
to find the first commit that regressed the tracked KPIs.

Resume semantics:
  - bisect/state.json is written after every step so the search can be resumed
    if the process is interrupted mid-run.
  - bisect_result.json is written once convergence is reached.
  - If bisect_result.json already exists, it is returned immediately (idempotent).

Hard constraints (from DESIGN.md §5):
  - runner.py is the ONLY thing that triggers benchmark execution.
  - Every stage checks for its output artifact before running.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

# ---------------------------------------------------------------------------
# Import verdict helpers — try relative first, fall back to absolute.
# ---------------------------------------------------------------------------
try:
    from .verdict import (
        classify_bisect_verdict,
        _FAILURE_PHASE_BAD,
        _FAILURE_PHASE_SKIP,
        _KPI_FIELDS,
        _REGRESSION_DIRECTION,
    )
except ImportError:
    from core.verdict import (  # type: ignore[no-redef]
        classify_bisect_verdict,
        _FAILURE_PHASE_BAD,
        _FAILURE_PHASE_SKIP,
        _KPI_FIELDS,
        _REGRESSION_DIRECTION,
    )

logger = logging.getLogger(__name__)

_SHA_SHORT = 12  # chars used for artifact dir names and log messages

# Stop if this many consecutive skips are encountered without making progress.
_MAX_CONSECUTIVE_SKIPS = 5


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_state(state_path: Path) -> dict | None:
    """Load bisect state from disk, or return None if it does not exist."""
    if state_path.exists():
        with state_path.open() as fh:
            return json.load(fh)
    return None


def _save_state(state_path: Path, state: dict) -> None:
    """Persist bisect state to disk (atomic-ish via temp then rename)."""
    tmp = state_path.with_suffix(".tmp")
    with tmp.open("w") as fh:
        json.dump(state, fh, indent=2)
    tmp.replace(state_path)


def run_bisect(
    commits: list[dict],
    grounding_result: dict,
    task_id: str,
    backend: str,
    run_dir: Path,
    runner_run_commit: Callable,
    *,
    dev_mode: bool = False,
    dev_perf_map: dict | None = None,
) -> dict:
    """Perform leftmost-BAD binary search over *commits*.

    Parameters
    ----------
    commits:
        Ordered list of commit dicts (oldest-first).  Index 0 is the first
        commit after ``good_sha``; the last entry is ``bad_sha`` itself.
        Each dict must have at least: ``{"sha": str, "message": str}``.
    grounding_result:
        Dict returned by ``run_grounding`` — must contain ``good_stats`` and
        ``kpis_regressing``.
    task_id:
        IsaacLab task identifier.
    backend:
        Backend key.
    run_dir:
        Root directory for all run artifacts.  Bisect artifacts go under
        ``run_dir/bisect/``.
    runner_run_commit:
        Reference to ``core/runner.py::run_commit``.  Signature::

            run_commit(sha, task_id, backend, output_dir, *, dev_mode, dev_perf_map) -> dict

    dev_mode:
        Whether to run in dev/stub mode (no Docker/GPU required).
    dev_perf_map:
        SHA -> fps_mean mapping used in dev mode.

    Returns
    -------
    dict
        Bisect result conforming to ``bisect_result.schema.json``.
    """
    run_dir = Path(run_dir)
    bisect_dir = run_dir / "bisect"
    result_path = run_dir / "bisect_result.json"
    state_path = bisect_dir / "state.json"

    # ------------------------------------------------------------------
    # Resume: if final result already exists, return it immediately.
    # ------------------------------------------------------------------
    if result_path.exists():
        logger.info("Bisect result already exists at %s — skipping.", result_path)
        with result_path.open() as fh:
            return json.load(fh)

    bisect_dir.mkdir(parents=True, exist_ok=True)

    if not commits:
        raise ValueError("commits list is empty — nothing to bisect.")

    n = len(commits)
    good_stats: dict = grounding_result.get("good_stats", {})
    kpis_regressing: list[str] = grounding_result.get("kpis_regressing", [])

    # ------------------------------------------------------------------
    # Helper: run one commit and return (run_result, verdict_str).
    # ------------------------------------------------------------------
    audit_path = run_dir / "audit_log.jsonl"

    def _append_audit(entry: dict) -> None:
        with audit_path.open("a") as fh:
            fh.write(json.dumps(entry) + "\n")

    def _verdict_reason(
        verdict: str,
        failure_phase: str | None,
        run_result: dict,
        attempt: int,
    ) -> str:
        """One-line explanation of why this verdict was reached."""
        if failure_phase == "runtime":
            return f"runtime crash → BAD (commit-caused execution failure)"
        if failure_phase in _FAILURE_PHASE_BAD:
            sfx = "" if attempt == 1 else f" (confirmed on attempt {attempt})"
            return f"{failure_phase} crash → BAD{sfx}"
        if failure_phase in _FAILURE_PHASE_SKIP:
            return f"infra failure ({failure_phase}) after {attempt} attempt(s) → SKIP"
        if verdict == "SKIP":
            return "no KPI output (benchmark produced no fps data)"
        # GOOD or BAD from KPI comparison
        parts: list[str] = []
        for kpi in kpis_regressing:
            field = _KPI_FIELDS.get(kpi)
            if not field:
                continue
            val = run_result.get(field)
            gs = good_stats.get(kpi, {})
            if val is None or not gs:
                continue
            med: float = gs.get("median", 0.0)
            mad: float = gs.get("mad", 0.0)
            direction = _REGRESSION_DIRECTION.get(kpi, "lower")
            if direction == "lower":
                threshold = med - 2.0 * mad
                cmp = f"{val:.0f} {'<' if val < threshold else '>='} threshold={threshold:.0f}"
            else:
                threshold = med + 2.0 * mad
                cmp = f"{val:.0f} {'>' if val > threshold else '<='} threshold={threshold:.0f}"
            parts.append(f"{kpi}={cmp}")
        summary = "; ".join(parts) if parts else "KPIs evaluated"
        return f"{verdict}: {summary}"

    def _run_and_classify(sha: str, *, attempt: int = 1) -> tuple[dict, str]:
        """Run sha and return (run_result, verdict). Retries up to 3 total attempts.

        Retry policy (3 total attempts):
        - ``runtime`` → BAD immediately (clear execution crash, always commit-caused).
        - ``import``/``init`` on attempt 1 → retry once (might be agent env fluke);
          if still failing on attempt 2 → BAD (confirmed commit-caused).
        - Infra failures (``oom``, ``hang``, ``driver``, etc.) → retry up to 3 total;
          if still failing after all retries → SKIP.
        """
        suffix = "" if attempt == 1 else f"_retry{attempt}"
        artifact_dir = bisect_dir / f"{sha[:_SHA_SHORT]}{suffix}"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Bisect: running %s (attempt %d) → %s", sha[:_SHA_SHORT], attempt, artifact_dir)

        run_result = runner_run_commit(
            sha, task_id, backend, artifact_dir,
            dev_mode=dev_mode, dev_perf_map=dev_perf_map,
        )
        run_result["artifact_dir"] = str(artifact_dir.relative_to(run_dir))

        verdict = classify_bisect_verdict(run_result, good_stats, kpis_regressing)
        failure_phase = run_result.get("failure_phase")

        # ---- Retry decisions ------------------------------------------------
        if attempt < 3:
            # import/init: could be agent-side env setup fluke — retry once.
            # runtime: always BAD immediately (execution crash = commit-caused).
            if failure_phase in {"import", "init"} and attempt == 1:
                logger.info(
                    "Bisect: %s → %s (attempt 1); retrying to confirm commit-caused.",
                    sha[:_SHA_SHORT], failure_phase,
                )
                _append_audit({
                    "ts": _now_iso(), "step": "bisect_retry",
                    "sha": sha, "attempt": 1, "failure_phase": failure_phase,
                    "reason": (
                        f"{failure_phase} on attempt 1 — retrying to distinguish "
                        "commit-caused crash from agent env setup fluke"
                    ),
                })
                return _run_and_classify(sha, attempt=2)

            # Infra failures: retry up to 3 total.
            if verdict == "SKIP" and failure_phase in _FAILURE_PHASE_SKIP:
                logger.info(
                    "Bisect: %s → SKIP (infra: %s, attempt %d); retrying.",
                    sha[:_SHA_SHORT], failure_phase, attempt,
                )
                _append_audit({
                    "ts": _now_iso(), "step": "bisect_retry",
                    "sha": sha, "attempt": attempt, "failure_phase": failure_phase,
                    "reason": f"infra skip ({failure_phase}) on attempt {attempt}; retrying",
                })
                return _run_and_classify(sha, attempt=attempt + 1)

        # ---- Final verdict — write structured audit record ------------------
        reason = _verdict_reason(verdict, failure_phase, run_result, attempt)
        audit_entry: dict = {
            "ts": _now_iso(), "step": "bisect",
            "sha": sha, "attempt": attempt,
            "verdict": verdict,
            "failure_phase": failure_phase,
            "fps_mean": run_result.get("raw_fps_mean"),
            "gpu_mem_mb": run_result.get("gpu_mem_used_mb"),
            "reason": reason,
        }
        for kpi in kpis_regressing:
            gs = good_stats.get(kpi, {})
            if gs:
                audit_entry[f"good_median_{kpi}"] = gs.get("median")
                audit_entry[f"good_mad_{kpi}"] = gs.get("mad")
        _append_audit(audit_entry)

        logger.info("Bisect: %s (attempt %d) → %s  [%s]", sha[:_SHA_SHORT], attempt, verdict, reason)
        return run_result, verdict

    # ------------------------------------------------------------------
    # Load or initialise search state.
    # ------------------------------------------------------------------
    saved = _load_state(state_path)
    if saved is not None:
        lo: int = saved["lo"]
        hi: int = saved["hi"]
        tested: list[dict] = saved["tested"]
        skip_count: int = saved.get("skip_count", 0)
        logger.info(
            "Resuming bisect from state: lo=%d hi=%d tested=%d",
            lo,
            hi,
            len(tested),
        )
    else:
        lo = 0
        hi = n - 1
        tested = []
        skip_count = 0

    # Index of already-tested commits, keyed by commit index.
    tested_by_index: dict[int, str] = {t["index"]: t["verdict"] for t in tested}
    commits_tested_count = len(tested)
    consecutive_skips = 0

    # ------------------------------------------------------------------
    # Main binary search loop.
    # ------------------------------------------------------------------
    while lo < hi:
        mid = (lo + hi) // 2
        sha = commits[mid]["sha"]

        # Skip if already tested (resumability).
        if mid in tested_by_index:
            prev_verdict = tested_by_index[mid]
            logger.info(
                "Bisect: index %d (%s) already tested → %s (from state)",
                mid,
                sha[:_SHA_SHORT],
                prev_verdict,
            )
            verdict = prev_verdict
        else:
            _, verdict = _run_and_classify(sha)
            tested.append({"sha": sha, "index": mid, "verdict": verdict})
            tested_by_index[mid] = verdict
            commits_tested_count += 1

            if verdict == "SKIP":
                skip_count += 1
                consecutive_skips += 1
            else:
                consecutive_skips = 0

        # Persist state after every step.
        state = {
            "lo": lo,
            "hi": hi,
            "commits_total": n,
            "tested": tested,
            "skip_count": skip_count,
            "confirmed_first_bad_index": None,
        }
        _save_state(state_path, state)

        # Apply leftmost-BAD bisect transition.
        if verdict == "GOOD":
            lo = mid + 1
            consecutive_skips = 0
        elif verdict == "BAD":
            hi = mid
            consecutive_skips = 0
        else:
            # SKIP: try mid+1 then mid-1, else advance lo.
            resolved = False
            for fallback_idx in (mid + 1, mid - 1):
                if lo <= fallback_idx <= hi and fallback_idx not in tested_by_index:
                    fallback_sha = commits[fallback_idx]["sha"]
                    _, fb_verdict = _run_and_classify(fallback_sha)
                    tested.append(
                        {"sha": fallback_sha, "index": fallback_idx, "verdict": fb_verdict}
                    )
                    tested_by_index[fallback_idx] = fb_verdict
                    commits_tested_count += 1

                    if fb_verdict == "SKIP":
                        skip_count += 1
                        consecutive_skips += 1
                    else:
                        consecutive_skips = 0

                    _save_state(
                        state_path,
                        {
                            "lo": lo,
                            "hi": hi,
                            "commits_total": n,
                            "tested": tested,
                            "skip_count": skip_count,
                            "confirmed_first_bad_index": None,
                        },
                    )

                    if fb_verdict == "GOOD":
                        lo = fallback_idx + 1
                        resolved = True
                        break
                    elif fb_verdict == "BAD":
                        hi = fallback_idx
                        resolved = True
                        break
                    # else SKIP — keep trying

            if not resolved:
                # Both fallbacks were also SKIP or out of range; advance lo.
                lo = mid + 1

        # Abort if too many consecutive skips to avoid infinite looping.
        if consecutive_skips >= _MAX_CONSECUTIVE_SKIPS:
            logger.warning(
                "Bisect aborted: %d consecutive skips at lo=%d hi=%d.",
                consecutive_skips,
                lo,
                hi,
            )
            break

    # ------------------------------------------------------------------
    # Convergence: lo == hi points to the first bad commit.
    # ------------------------------------------------------------------
    first_bad_index = lo  # lo == hi after loop
    # Clamp to valid range in case all commits were good or skipped.
    first_bad_index = max(0, min(first_bad_index, n - 1))

    first_bad_sha = commits[first_bad_index]["sha"]
    first_bad_message = commits[first_bad_index].get("message", "")

    # Determine prev_good_sha: the commit immediately before first_bad_index.
    if first_bad_index > 0:
        prev_good_sha: str | None = commits[first_bad_index - 1]["sha"]
    else:
        # first_bad is the first commit after good_sha — fall back to grounding's good_sha
        prev_good_sha = grounding_result.get("good_sha")

    # If we didn't actually test the convergence point yet, run it once to confirm.
    # We always attempt this regardless of skip_count — an untested convergence point
    # means the result is based only on boundary shrinkage, not a measured verdict.
    if first_bad_index not in tested_by_index:
        logger.info("Bisect: confirming first_bad %s with one final run.", first_bad_sha[:_SHA_SHORT])
        _, confirm_verdict = _run_and_classify(first_bad_sha)
        tested.append(
            {"sha": first_bad_sha, "index": first_bad_index, "verdict": confirm_verdict}
        )
        tested_by_index[first_bad_index] = confirm_verdict
        commits_tested_count += 1
        if confirm_verdict == "SKIP":
            skip_count += 1

    # Persist final state.
    _save_state(
        state_path,
        {
            "lo": lo,
            "hi": hi,
            "commits_total": n,
            "tested": tested,
            "skip_count": skip_count,
            "confirmed_first_bad_index": first_bad_index,
        },
    )

    # ------------------------------------------------------------------
    # Determine confidence from skip_count.
    # ------------------------------------------------------------------
    if skip_count == 0:
        confidence = "high"
    elif skip_count < 3:
        confidence = "medium"
    else:
        confidence = "low"

    # ------------------------------------------------------------------
    # Assemble and write bisect_result.json.
    # ------------------------------------------------------------------
    bisect_result: dict = {
        "first_bad_sha": first_bad_sha,
        "prev_good_sha": prev_good_sha,
        "first_bad_message": first_bad_message,
        "commits_tested": commits_tested_count,
        "total_commits_in_range": n,
        "kpi_deltas": grounding_result.get("kpi_deltas", {}),
        "confidence": confidence,
        "skip_count": skip_count,
    }

    with result_path.open("w") as fh:
        json.dump(bisect_result, fh, indent=2)

    logger.info(
        "Bisect complete: first_bad=%s prev_good=%s confidence=%s commits_tested=%d skip_count=%d",
        first_bad_sha[:_SHA_SHORT],
        prev_good_sha[:_SHA_SHORT] if prev_good_sha else "None",
        confidence,
        commits_tested_count,
        skip_count,
    )
    return bisect_result
