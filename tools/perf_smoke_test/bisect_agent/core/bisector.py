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
# Import verdict function — try relative first, fall back to absolute.
# ---------------------------------------------------------------------------
try:
    from .verdict import classify_bisect_verdict
except ImportError:
    from core.verdict import classify_bisect_verdict  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

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
    def _run_and_classify(sha: str) -> tuple[dict, str]:
        artifact_dir = bisect_dir / sha[:12]
        artifact_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Bisect: running commit %s → %s", sha[:12], artifact_dir)
        run_result = runner_run_commit(
            sha,
            task_id,
            backend,
            artifact_dir,
            dev_mode=dev_mode,
            dev_perf_map=dev_perf_map,
        )
        run_result["artifact_dir"] = str(artifact_dir.relative_to(run_dir))
        verdict = classify_bisect_verdict(run_result, good_stats, kpis_regressing)
        logger.info("Bisect: %s → %s", sha[:12], verdict)
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
                sha[:12],
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

    # If we didn't actually test the convergence point yet, run it once to
    # confirm (unless we exhausted skips).
    if first_bad_index not in tested_by_index and skip_count < _MAX_CONSECUTIVE_SKIPS:
        logger.info("Bisect: confirming first_bad %s with one final run.", first_bad_sha[:12])
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
        first_bad_sha[:12],
        prev_good_sha[:12] if prev_good_sha else "None",
        confidence,
        commits_tested_count,
        skip_count,
    )
    return bisect_result
