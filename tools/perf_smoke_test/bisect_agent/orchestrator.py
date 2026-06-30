"""
orchestrator.py — Persistent LLM orchestrator for the bisect agent.

Coordinates all four phases (grounding, enumerate, bisect, diagnosis) by running
a persistent LLM session with tool-use.  Every tool is a closure over run_dir and
dev-mode state so the LLM never has to pass run_dir explicitly for internal tools.

Entry point:
    run_orchestrator(run_config, run_dir, llm_client, runner_run_commit,
                     commits_enumerate, commits_fetch_diff, grounding_run,
                     bisect_run, diagnosis_run, *, dev_mode=False,
                     dev_perf_map=None, repo_path=None) -> None
"""

from __future__ import annotations

import json
import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

# ---------------------------------------------------------------------------
# Import verdict functions — try package-relative first, fall back to absolute.
# ---------------------------------------------------------------------------
try:
    from core.verdict import compute_kpi_stats, check_separation, classify_bisect_verdict
except ImportError:
    try:
        from verdict import compute_kpi_stats, check_separation, classify_bisect_verdict  # type: ignore[no-redef]
    except ImportError:
        # Minimal stubs so the module loads in isolation (e.g. linting).
        compute_kpi_stats = None  # type: ignore[assignment]
        check_separation = None  # type: ignore[assignment]
        classify_bisect_verdict = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from infra.llm_client import LLMClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_system_prompt(bisect_agent_root: Path) -> str:
    """Load orchestrator system prompt from prompts/orchestrator.md."""
    prompt_path = bisect_agent_root / "prompts" / "orchestrator.md"
    if prompt_path.exists():
        try:
            return prompt_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            logger.warning("Could not read %s: %s", prompt_path, exc)
    # Minimal fallback prompt — should never be needed in practice.
    return (
        "You are the orchestrator for the IsaacLab bisect agent.\n"
        "Drive the four-stage protocol: grounding → enumerate → bisect → diagnosis.\n"
        "Use the tools provided. Always check for existing artifacts before running.\n"
        "Write status after every significant state change."
    )


# ---------------------------------------------------------------------------
# Tool schema builder (OpenAI function-calling format)
# ---------------------------------------------------------------------------

def _make_tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

_TOOLS: list[dict] = [
    _make_tool(
        name="run_experiments",
        description=(
            "Run the benchmark for one or more commit SHAs, n times each. "
            "Routes artifacts to grounding/ or bisect/ depending on pipeline phase. "
            "Returns a JSON list of run_result objects."
        ),
        properties={
            "shas": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of commit SHAs to run.",
            },
            "task": {
                "type": "string",
                "description": "IsaacLab task identifier.",
            },
            "backend": {
                "type": "string",
                "description": "Backend key (e.g. 'newton').",
            },
            "n": {
                "type": "integer",
                "description": "Number of runs per SHA.",
                "default": 1,
            },
        },
        required=["shas", "task", "backend"],
    ),
    _make_tool(
        name="assess_grounding",
        description=(
            "Compute statistics over all grounding runs and check whether the "
            "good and bad distributions are statistically separated. "
            "Returns cached grounding/result.json if it already exists; "
            "otherwise scans grounding/ for existing run_result.json files, "
            "computes stats, writes result.json, and returns it."
        ),
        properties={
            "run_dir": {
                "type": "string",
                "description": "Path to the run directory (ignored; bound by closure).",
            },
        },
        required=[],
    ),
    _make_tool(
        name="enumerate_commits",
        description=(
            "List all commits between good_sha (exclusive) and bad_sha (inclusive), "
            "ordered oldest-first. Idempotent: returns cached commits.json if present. "
            "Returns a summary string with the commit count."
        ),
        properties={
            "good_sha": {
                "type": "string",
                "description": "Known-good commit SHA.",
            },
            "bad_sha": {
                "type": "string",
                "description": "Known-bad commit SHA.",
            },
        },
        required=["good_sha", "bad_sha"],
    ),
    _make_tool(
        name="bisect_step",
        description=(
            "Execute one step of the leftmost-BAD binary search. "
            "Reads commits.json and grounding/result.json. "
            "Picks the midpoint of [lo, hi], runs it, classifies the result, "
            "updates state.json, and returns progress or a DONE result."
        ),
        properties={
            "run_dir": {
                "type": "string",
                "description": "Path to the run directory (ignored; bound by closure).",
            },
        },
        required=[],
    ),
    _make_tool(
        name="fetch_diff",
        description=(
            "Fetch the diff between two commits. "
            "Returns files changed, dep file diffs, commit message, and diff summary "
            "truncated to 4000 characters."
        ),
        properties={
            "sha_a": {
                "type": "string",
                "description": "The older/base commit SHA.",
            },
            "sha_b": {
                "type": "string",
                "description": "The newer/head commit SHA.",
            },
        },
        required=["sha_a", "sha_b"],
    ),
    _make_tool(
        name="run_diagnosis",
        description=(
            "Spawn the diagnostician LLM sub-session to perform root-cause analysis. "
            "Reads bisect_result.json and grounding/result.json. "
            "Writes report/diagnosis.json and report/report.md. "
            "Returns the completed diagnosis dict."
        ),
        properties={
            "run_dir": {
                "type": "string",
                "description": "Path to the run directory (ignored; bound by closure).",
            },
        },
        required=[],
    ),
    _make_tool(
        name="write_status",
        description=(
            "Write status.json to the run directory to report current phase and progress. "
            "Call after every significant state change."
        ),
        properties={
            "phase": {
                "type": "string",
                "enum": ["grounding", "enumerate", "bisect", "diagnosis", "done", "error"],
                "description": "Current pipeline phase.",
            },
            "status": {
                "type": "string",
                "enum": ["running", "complete", "warn", "error"],
                "description": "Status within the current phase.",
            },
            "progress": {
                "type": "string",
                "description": "Human-readable progress description.",
            },
            "bisect_lo": {
                "type": "integer",
                "description": "Current bisect lower bound (bisect phase only).",
            },
            "bisect_hi": {
                "type": "integer",
                "description": "Current bisect upper bound (bisect phase only).",
            },
        },
        required=["phase", "status", "progress"],
    ),
]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_orchestrator(
    run_config: dict,
    run_dir: Path,
    llm_client: "LLMClient",
    runner_run_commit: Callable,
    commits_enumerate: Callable,
    commits_fetch_diff: Callable,
    grounding_run: Callable,
    bisect_run: Callable,
    diagnosis_run: Callable,
    *,
    dev_mode: bool = False,
    dev_perf_map: dict | None = None,
    repo_path: Path | None = None,
) -> None:
    """Run the persistent LLM orchestrator for the full bisect pipeline.

    Parameters
    ----------
    run_config:
        The run configuration dict (from run_config.json).  Must contain at
        minimum: ``good_sha``, ``bad_sha``, ``task_id``, ``backend``.
    run_dir:
        Root directory for all run artifacts.
    llm_client:
        An :class:`~infra.llm_client.LLMClient` instance.
    runner_run_commit:
        ``core.runner.run_commit`` callable.
    commits_enumerate:
        ``infra.commits.enumerate_commits`` callable.
    commits_fetch_diff:
        ``infra.commits.fetch_diff`` callable.
    grounding_run:
        ``core.grounding.run_grounding`` callable.
    bisect_run:
        ``core.bisector.run_bisect`` callable.
    diagnosis_run:
        ``core.diagnosis.run_diagnosis`` callable.
    dev_mode:
        If True, all experiments use stub_benchmark (no Docker/GPU needed).
    dev_perf_map:
        SHA -> fps_mean mapping for dev-mode stub experiments.
    repo_path:
        Path to the local IsaacLab git repo (forwarded to commit helpers).
    """
    run_dir = Path(run_dir)

    good_sha: str = run_config["good_sha"]
    bad_sha: str = run_config["bad_sha"]
    task_id: str = run_config["task_id"]
    backend: str = run_config["backend"]

    # ------------------------------------------------------------------
    # Resolve bisect_agent root for loading prompts.
    # ------------------------------------------------------------------
    _this_file = Path(__file__).resolve()
    bisect_agent_root = _this_file.parent  # orchestrator.py lives at bisect_agent/

    # ------------------------------------------------------------------
    # Closure helpers
    # ------------------------------------------------------------------

    def _write_status(
        phase: str,
        message: str,
        *,
        status: str = "running",
        bisect_lo: int | None = None,
        bisect_hi: int | None = None,
    ) -> None:
        """Write run_dir/status.json and print a progress line."""
        payload: dict[str, Any] = {
            "phase": phase,
            "status": status,
            "progress": message,
            "last_update": _now_iso(),
        }
        if bisect_lo is not None:
            payload["bisect_lo"] = bisect_lo
        if bisect_hi is not None:
            payload["bisect_hi"] = bisect_hi

        status_path = run_dir / "status.json"
        try:
            status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not write status.json: %s", exc)

        print(f"[orchestrator] [{phase}] {message}", flush=True)

    def _run_experiments(
        shas: list[str],
        task: str,
        backend_key: str,
        n: int = 1,
    ) -> list[dict]:
        """Run runner_run_commit for each sha x n times.

        Routes artifacts to grounding/ or bisect/ depending on whether
        commits.json exists (grounding runs before enumerate).
        """
        commits_path = run_dir / "commits.json"
        phase_dir = run_dir / ("bisect" if commits_path.exists() else "grounding")

        results: list[dict] = []
        for sha in shas:
            for i in range(n):
                out_dir = phase_dir / f"{sha[:12]}_{i}"
                out_dir.mkdir(parents=True, exist_ok=True)
                logger.info(
                    "_run_experiments: sha=%s run=%d → %s", sha[:12], i, out_dir
                )
                try:
                    result = runner_run_commit(
                        sha,
                        task,
                        backend_key,
                        out_dir,
                        dev_mode=dev_mode,
                        dev_perf_map=dev_perf_map,
                        run_index=i,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "_run_experiments: runner failed for sha=%s run=%d: %s",
                        sha[:12], i, exc,
                    )
                    result = {
                        "sha": sha,
                        "task_id": task,
                        "backend": backend_key,
                        "run_index": i,
                        "exit_code": -1,
                        "wall_time_s": None,
                        "failure_phase": "runner_error",
                        "raw_fps_mean": None,
                        "raw_fps_median": None,
                        "raw_fps_p5": None,
                        "raw_fps_p95": None,
                        "gpu_mem_used_mb": None,
                        "artifact_dir": str(out_dir),
                    }
                results.append(result)
        return results

    def _assess_grounding() -> dict:
        """Check grounding/result.json or compute it from existing run files."""
        grounding_dir = run_dir / "grounding"
        result_path = grounding_dir / "result.json"

        # Return cached result if available.
        if result_path.exists():
            try:
                cached = json.loads(result_path.read_text(encoding="utf-8"))
                logger.info("_assess_grounding: returning cached result.")
                return cached
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(
                    "_assess_grounding: cached result unreadable (%s); recomputing.", exc
                )

        # Scan grounding/ for existing run_result.json files.
        good_results: list[dict] = []
        bad_results: list[dict] = []

        if grounding_dir.exists():
            for run_result_path in sorted(grounding_dir.rglob("run_result.json")):
                try:
                    rr = json.loads(run_result_path.read_text(encoding="utf-8"))
                    sha = rr.get("sha", "")
                    if sha == good_sha:
                        good_results.append(rr)
                    elif sha == bad_sha:
                        bad_results.append(rr)
                except (json.JSONDecodeError, OSError):
                    pass  # skip unreadable files

        if not good_results and not bad_results:
            return {
                "verdict": "NO_RUNS",
                "separated": False,
                "kpis_regressing": [],
                "kpi_deltas": {},
                "separation_ratios": {},
                "high_variance_tasks": [],
                "n_good": 0,
                "n_bad": 0,
                "note": "No grounding runs found.",
            }

        if compute_kpi_stats is None or check_separation is None:
            return {
                "verdict": "ERROR",
                "separated": False,
                "kpis_regressing": [],
                "kpi_deltas": {},
                "separation_ratios": {},
                "high_variance_tasks": [],
                "n_good": len(good_results),
                "n_bad": len(bad_results),
                "note": "verdict functions unavailable (ImportError).",
            }

        good_stats = compute_kpi_stats(good_results)
        bad_stats = compute_kpi_stats(bad_results)
        separation = check_separation(good_stats, bad_stats)

        _CV_THRESHOLD = 0.08
        _PRIMARY_KPIS = ("fps_mean", "fps_p5", "fps_median")

        high_variance_kpis: list[str] = []
        for stats_dict, label in [(good_stats, "good"), (bad_stats, "bad")]:
            for kpi, kpi_data in stats_dict.items():
                if kpi in _PRIMARY_KPIS:
                    cv = kpi_data.get("cv", 0.0)
                    if cv > _CV_THRESHOLD:
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
            hv_note = f"High variance KPIs: {', '.join(high_variance_kpis)}."
            note = f"{note} {hv_note}".strip() if note else hv_note

        grounding_result: dict = {
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

        grounding_dir.mkdir(parents=True, exist_ok=True)
        try:
            result_path.write_text(json.dumps(grounding_result, indent=2), encoding="utf-8")
            logger.info("_assess_grounding: wrote result.json.")
        except OSError as exc:
            logger.warning("_assess_grounding: could not write result.json: %s", exc)

        return grounding_result

    def _enumerate_commits(good: str, bad: str) -> str:
        """Enumerate commits between good and bad; return count summary string."""
        commits_path = run_dir / "commits.json"

        if commits_path.exists():
            try:
                commits = json.loads(commits_path.read_text(encoding="utf-8"))
                msg = f"{len(commits)} commits in range (cached)"
                logger.info("_enumerate_commits: %s", msg)
                return msg
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(
                    "_enumerate_commits: cached commits.json unreadable (%s); re-fetching.", exc
                )

        logger.info("_enumerate_commits: calling commits_enumerate(%s, %s)", good[:7], bad[:7])
        commits = commits_enumerate(good, bad, repo_path=repo_path)

        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            commits_path.write_text(json.dumps(commits, indent=2), encoding="utf-8")
            logger.info("_enumerate_commits: wrote commits.json with %d entries.", len(commits))
        except OSError as exc:
            logger.warning("_enumerate_commits: could not write commits.json: %s", exc)

        return f"{len(commits)} commits in range"

    def _bisect_step() -> dict:
        """Execute one step of the leftmost-BAD binary search.

        Reads commits.json and grounding/result.json.
        Returns a progress dict (status="IN_PROGRESS") or completion dict
        (status="DONE") — mirrors the schema described in prompts/orchestrator.md.
        """
        # ------------------------------------------------------------------
        # Check for completed bisect result.
        # ------------------------------------------------------------------
        result_path = run_dir / "bisect_result.json"
        if result_path.exists():
            try:
                bisect_result = json.loads(result_path.read_text(encoding="utf-8"))
                first_bad = bisect_result.get("first_bad_sha", "unknown")
                logger.info("_bisect_step: bisect_result.json exists → DONE (%s)", first_bad[:12])
                return {
                    "status": "DONE",
                    "first_bad_sha": bisect_result.get("first_bad_sha"),
                    "prev_good_sha": bisect_result.get("prev_good_sha"),
                    "commits_tested": bisect_result.get("commits_tested"),
                    "skip_count": bisect_result.get("skip_count", 0),
                    "confidence": bisect_result.get("confidence", "unknown"),
                }
            except (json.JSONDecodeError, OSError):
                pass  # re-run if file is corrupt

        # ------------------------------------------------------------------
        # Load required inputs.
        # ------------------------------------------------------------------
        commits_path = run_dir / "commits.json"
        if not commits_path.exists():
            return {"error": "commits.json not found — run enumerate_commits first."}

        grounding_path = run_dir / "grounding" / "result.json"
        if not grounding_path.exists():
            return {"error": "grounding/result.json not found — run grounding first."}

        try:
            commits: list[dict] = json.loads(commits_path.read_text(encoding="utf-8"))
            grounding_result: dict = json.loads(grounding_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            return {"error": f"Failed to read required inputs: {exc}"}

        n = len(commits)
        if n == 0:
            return {"error": "commits.json is empty — nothing to bisect."}

        good_stats: dict = grounding_result.get("good_stats", {})
        kpis_regressing: list[str] = grounding_result.get("kpis_regressing", [])

        # ------------------------------------------------------------------
        # Load or initialize bisect state.
        # ------------------------------------------------------------------
        bisect_dir = run_dir / "bisect"
        bisect_dir.mkdir(parents=True, exist_ok=True)
        state_path = bisect_dir / "state.json"

        saved: dict | None = None
        if state_path.exists():
            try:
                saved = json.loads(state_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass

        if saved is not None:
            lo: int = saved["lo"]
            hi: int = saved["hi"]
            tested: list[dict] = saved.get("tested", [])
            skip_count: int = saved.get("skip_count", 0)
        else:
            lo = 0
            hi = n - 1
            tested = []
            skip_count = 0

        tested_by_index: dict[int, str] = {t["index"]: t["verdict"] for t in tested}

        def _save_state() -> None:
            payload = {
                "lo": lo,
                "hi": hi,
                "commits_total": n,
                "tested": tested,
                "skip_count": skip_count,
                "confirmed_first_bad_index": None,
            }
            tmp = state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp.replace(state_path)

        # ------------------------------------------------------------------
        # Convergence check before running anything.
        # ------------------------------------------------------------------
        if lo >= hi:
            # Already converged — write bisect_result.json if not done.
            first_bad_index = max(0, min(lo, n - 1))
            first_bad_sha = commits[first_bad_index]["sha"]
            prev_good_sha: str | None = (
                commits[first_bad_index - 1]["sha"] if first_bad_index > 0 else None
            )
            confidence = "high" if skip_count == 0 else ("medium" if skip_count < 3 else "low")
            bisect_result_dict: dict = {
                "first_bad_sha": first_bad_sha,
                "prev_good_sha": prev_good_sha,
                "first_bad_message": commits[first_bad_index].get("message", ""),
                "commits_tested": len(tested),
                "total_commits_in_range": n,
                "kpi_deltas": grounding_result.get("kpi_deltas", {}),
                "confidence": confidence,
                "skip_count": skip_count,
            }
            try:
                result_path.write_text(
                    json.dumps(bisect_result_dict, indent=2), encoding="utf-8"
                )
            except OSError as exc:
                logger.warning("_bisect_step: could not write bisect_result.json: %s", exc)
            return {
                "status": "DONE",
                "first_bad_sha": first_bad_sha,
                "prev_good_sha": prev_good_sha,
                "commits_tested": len(tested),
                "skip_count": skip_count,
                "confidence": confidence,
            }

        # ------------------------------------------------------------------
        # Helper: run one commit and classify it.
        # ------------------------------------------------------------------
        def _run_and_classify(idx: int) -> tuple[str, str]:
            sha = commits[idx]["sha"]
            out_dir = bisect_dir / sha[:12]
            out_dir.mkdir(parents=True, exist_ok=True)
            logger.info("_bisect_step._run_and_classify: idx=%d sha=%s", idx, sha[:12])
            try:
                run_result = runner_run_commit(
                    sha,
                    task_id,
                    backend,
                    out_dir,
                    dev_mode=dev_mode,
                    dev_perf_map=dev_perf_map,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("_bisect_step: runner failed for sha=%s: %s", sha[:12], exc)
                return sha, "SKIP"

            if classify_bisect_verdict is None:
                return sha, "SKIP"

            verdict = classify_bisect_verdict(run_result, good_stats, kpis_regressing)
            logger.info("_bisect_step: sha=%s verdict=%s", sha[:12], verdict)
            return sha, verdict

        # ------------------------------------------------------------------
        # Execute ONE binary search step.
        # ------------------------------------------------------------------
        mid = (lo + hi) // 2

        # Nonlocal mutation — use a mutable container to allow inner-closure writes.
        _state: dict[str, Any] = {"lo": lo, "hi": hi, "skip_count": skip_count}

        if mid in tested_by_index:
            verdict = tested_by_index[mid]
            tested_sha = commits[mid]["sha"]
            logger.info(
                "_bisect_step: mid=%d already tested → %s (from state)", mid, verdict
            )
        else:
            tested_sha, verdict = _run_and_classify(mid)
            tested.append({"sha": tested_sha, "index": mid, "verdict": verdict})
            tested_by_index[mid] = verdict
            if verdict == "SKIP":
                _state["skip_count"] += 1

        # Apply leftmost-BAD bisect transition.
        if verdict == "GOOD":
            _state["lo"] = mid + 1
        elif verdict == "BAD":
            _state["hi"] = mid
        else:
            # SKIP: try mid+1 then mid-1.
            resolved = False
            for fallback_idx in (mid + 1, mid - 1):
                lo_cur = _state["lo"]
                hi_cur = _state["hi"]
                if lo_cur <= fallback_idx <= hi_cur and fallback_idx not in tested_by_index:
                    fb_sha, fb_verdict = _run_and_classify(fallback_idx)
                    tested.append(
                        {"sha": fb_sha, "index": fallback_idx, "verdict": fb_verdict}
                    )
                    tested_by_index[fallback_idx] = fb_verdict
                    if fb_verdict == "SKIP":
                        _state["skip_count"] += 1

                    if fb_verdict == "GOOD":
                        _state["lo"] = fallback_idx + 1
                        resolved = True
                        break
                    elif fb_verdict == "BAD":
                        _state["hi"] = fallback_idx
                        resolved = True
                        break

            if not resolved:
                _state["lo"] = mid + 1

        # Write back mutated state.
        lo = _state["lo"]
        hi = _state["hi"]
        skip_count = _state["skip_count"]

        _save_state()

        # ------------------------------------------------------------------
        # Check for convergence after this step.
        # ------------------------------------------------------------------
        if lo >= hi:
            first_bad_index = max(0, min(lo, n - 1))
            first_bad_sha = commits[first_bad_index]["sha"]
            prev_good_sha = (
                commits[first_bad_index - 1]["sha"] if first_bad_index > 0 else None
            )
            confidence = "high" if skip_count == 0 else ("medium" if skip_count < 3 else "low")
            bisect_result_dict = {
                "first_bad_sha": first_bad_sha,
                "prev_good_sha": prev_good_sha,
                "first_bad_message": commits[first_bad_index].get("message", ""),
                "commits_tested": len(tested),
                "total_commits_in_range": n,
                "kpi_deltas": grounding_result.get("kpi_deltas", {}),
                "confidence": confidence,
                "skip_count": skip_count,
            }
            try:
                result_path.write_text(
                    json.dumps(bisect_result_dict, indent=2), encoding="utf-8"
                )
                logger.info(
                    "_bisect_step: convergence → first_bad=%s (confidence=%s)",
                    first_bad_sha[:12],
                    confidence,
                )
            except OSError as exc:
                logger.warning("_bisect_step: could not write bisect_result.json: %s", exc)
            return {
                "status": "DONE",
                "first_bad_sha": first_bad_sha,
                "prev_good_sha": prev_good_sha,
                "commits_tested": len(tested),
                "skip_count": skip_count,
                "confidence": confidence,
            }

        return {
            "status": "IN_PROGRESS",
            "lo": lo,
            "hi": hi,
            "tested_sha": tested_sha,
            "verdict": verdict,
            "commits_remaining": hi - lo,
        }

    def _fetch_diff(sha_a: str, sha_b: str) -> str:
        """Fetch diff between two commits; returns JSON (truncated to 4000 chars)."""
        logger.info("_fetch_diff: %s..%s", sha_a[:7], sha_b[:7])
        try:
            result = commits_fetch_diff(sha_a, sha_b, repo_path=repo_path)
        except Exception as exc:  # noqa: BLE001
            logger.error("_fetch_diff: failed: %s", exc)
            return json.dumps({"error": str(exc)})

        # Truncate diff_summary to keep tool output within context budget.
        if isinstance(result, dict) and "diff_summary" in result:
            summary = result["diff_summary"] or ""
            if len(summary) > 4000:
                result = dict(result)
                result["diff_summary"] = summary[:4000] + "\n... [truncated]"

        return json.dumps(result, indent=2)

    def _run_diagnosis() -> dict:
        """Load bisect and grounding results, then call diagnosis_run."""
        result_path = run_dir / "bisect_result.json"
        grounding_path = run_dir / "grounding" / "result.json"

        if not result_path.exists():
            return {"error": "bisect_result.json not found — run bisect first."}
        if not grounding_path.exists():
            return {"error": "grounding/result.json not found."}

        try:
            bisect_result = json.loads(result_path.read_text(encoding="utf-8"))
            grounding_result = json.loads(grounding_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            return {"error": f"Failed to read required inputs: {exc}"}

        logger.info(
            "_run_diagnosis: first_bad=%s",
            bisect_result.get("first_bad_sha", "unknown")[:12],
        )
        try:
            diagnosis = diagnosis_run(
                bisect_result=bisect_result,
                grounding_result=grounding_result,
                run_dir=run_dir,
                llm_client=llm_client,
                commits_fetch_diff=commits_fetch_diff,
                runner_run_commit=runner_run_commit,
                repo_path=repo_path,
                dev_mode=dev_mode,
                dev_perf_map=dev_perf_map,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("_run_diagnosis: diagnosis_run raised: %s\n%s", exc, traceback.format_exc())
            return {"error": f"diagnosis_run raised: {exc}"}

        return diagnosis

    # ------------------------------------------------------------------
    # Tool dispatch wrappers — each translates LLM kwargs to closure calls.
    # ------------------------------------------------------------------

    def _tool_run_experiments(
        shas: list[str],
        task: str,
        backend: str,
        n: int = 1,
    ) -> str:
        results = _run_experiments(shas, task, backend, n)
        return json.dumps(results, default=str)

    def _tool_assess_grounding(**_kwargs: Any) -> str:
        result = _assess_grounding()
        return json.dumps(result, default=str)

    def _tool_enumerate_commits(good_sha: str, bad_sha: str) -> str:
        summary = _enumerate_commits(good_sha, bad_sha)
        return summary

    def _tool_bisect_step(**_kwargs: Any) -> str:
        result = _bisect_step()
        return json.dumps(result, default=str)

    def _tool_fetch_diff(sha_a: str, sha_b: str) -> str:
        return _fetch_diff(sha_a, sha_b)

    def _tool_run_diagnosis(**_kwargs: Any) -> str:
        result = _run_diagnosis()
        return json.dumps(result, default=str)

    def _tool_write_status(
        phase: str,
        status: str,
        progress: str,
        bisect_lo: int | None = None,
        bisect_hi: int | None = None,
    ) -> str:
        _write_status(
            phase,
            progress,
            status=status,
            bisect_lo=bisect_lo,
            bisect_hi=bisect_hi,
        )
        return json.dumps({"ok": True, "phase": phase, "status": status})

    tool_dispatch: dict[str, Callable] = {
        "run_experiments": _tool_run_experiments,
        "assess_grounding": _tool_assess_grounding,
        "enumerate_commits": _tool_enumerate_commits,
        "bisect_step": _tool_bisect_step,
        "fetch_diff": _tool_fetch_diff,
        "run_diagnosis": _tool_run_diagnosis,
        "write_status": _tool_write_status,
    }

    # ------------------------------------------------------------------
    # Build user prompt with run_config context.
    # ------------------------------------------------------------------
    user_prompt = (
        f"Run the four-stage bisection investigation.\n\n"
        f"## Run Configuration\n"
        f"- good_sha: {good_sha}\n"
        f"- bad_sha: {bad_sha}\n"
        f"- task_id: {task_id}\n"
        f"- backend: {backend}\n"
        f"- run_dir: {run_dir}\n"
        f"- dev_mode: {dev_mode}\n\n"
        f"Work through Stage 1 (Grounding) → Stage 2 (Enumerate) → "
        f"Stage 3 (Bisect) → Stage 4 (Diagnosis) in order.\n"
        f"Always call assess_grounding first to check whether grounding is already complete.\n"
        f"Always call write_status after every significant state change.\n"
        f"Do not skip any stage unless its output artifact already exists.\n"
    )

    # ------------------------------------------------------------------
    # Load system prompt.
    # ------------------------------------------------------------------
    system_prompt = _load_system_prompt(bisect_agent_root)

    # ------------------------------------------------------------------
    # Run LLM session, always writing "done" status in the finally block.
    # ------------------------------------------------------------------
    _write_status("grounding", "orchestrator starting", status="running")

    try:
        logger.info("run_orchestrator: starting LLM session (model=%s)", llm_client.model)
        final_response = llm_client.run_session(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            tools=_TOOLS,
            tool_dispatch=tool_dispatch,
        )
        logger.info("run_orchestrator: LLM session complete.")
        if final_response:
            print(f"[orchestrator] final response: {final_response[:500]}", flush=True)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "run_orchestrator: LLM session raised: %s\n%s", exc, traceback.format_exc()
        )
        _write_status("error", f"LLM session failed: {exc}", status="error")
        raise
    finally:
        # Always attempt to mark the run as done so polling agents don't hang.
        status_path = run_dir / "status.json"
        if status_path.exists():
            try:
                current = json.loads(status_path.read_text(encoding="utf-8"))
                if current.get("phase") not in ("done", "error"):
                    _write_status("done", "orchestrator finished", status="complete")
            except (json.JSONDecodeError, OSError):
                _write_status("done", "orchestrator finished", status="complete")
        else:
            _write_status("done", "orchestrator finished", status="complete")
