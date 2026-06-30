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
    from core.verdict import (
        classify_bisect_verdict,
        _CV_THRESHOLD, _FAILURE_PHASE_SKIP,
    )
except ImportError:
    try:
        from verdict import (  # type: ignore[no-redef]
            classify_bisect_verdict,
            _CV_THRESHOLD, _FAILURE_PHASE_SKIP,
        )
    except ImportError:
        # Minimal stubs so the module loads in isolation (e.g. linting).
        classify_bisect_verdict = None  # type: ignore[assignment]
        _CV_THRESHOLD = 0.08  # type: ignore[assignment]
        _FAILURE_PHASE_SKIP = frozenset({  # type: ignore[assignment]
            "oom", "hang", "driver", "config_mismatch", "runner_error", "missing_result",
        })

try:
    from core.grounding import compute_grounding_result
except ImportError:
    try:
        from grounding import compute_grounding_result  # type: ignore[no-redef]
    except ImportError:
        compute_grounding_result = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from infra.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SHA_SHORT = 12  # chars used for artifact dir names and log messages

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
            "Picks the midpoint of [lo, hi], runs it n_runs times for statistical "
            "confidence, classifies the aggregate, updates state.json, and returns "
            "progress (with raw run_results) or a DONE result. "
            "n_runs is for statistical re-runs (separate from infra retries). "
            "Each statistical run gets up to 3 infra retries before counting as lost."
        ),
        properties={
            "n_runs": {
                "type": "integer",
                "description": (
                    "Number of successful benchmark runs to aggregate before classifying. "
                    "Use the value from bisect_plan.json (n_runs_per_commit). "
                    "Defaults to 1. Infra retries (up to 3 per run) are separate from this count."
                ),
                "default": 1,
            },
        },
        required=[],
    ),
    _make_tool(
        name="write_plan",
        description=(
            "Write the bisect execution plan to bisect_plan.json. "
            "Call this after assess_grounding and before enumerate_commits. "
            "The plan documents how bisect steps will be executed based on task variance."
        ),
        properties={
            "n_runs_per_commit": {
                "type": "integer",
                "description": (
                    "Number of statistical runs per bisect step commit. "
                    "Derive from grounding CV: >=0.15 → 3, 0.08-0.15 → 2, <0.08 → 1."
                ),
            },
            "variance_class": {
                "type": "string",
                "enum": ["low", "medium", "high"],
                "description": "Task variance class derived from grounding CV.",
            },
            "rationale": {
                "type": "string",
                "description": "One-sentence explanation (cite CV values and task name).",
            },
            "kpis_regressing": {
                "type": "array",
                "items": {"type": "string"},
                "description": "KPIs identified as regressing in grounding.",
            },
            "grounding_cv": {
                "type": "object",
                "description": "CV per primary KPI from grounding (fps_mean, fps_p5, fps_median).",
            },
        },
        required=["n_runs_per_commit", "variance_class", "rationale", "kpis_regressing"],
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
        name="read_artifact",
        description=(
            "Read a file from the run directory (relative path). "
            "Use to inspect failure logs (e.g. 'bisect/<sha12>/benchmark.log') "
            "or any artifact under the run directory. "
            "Returns up to 4000 characters. "
            "Useful for diagnosing gray-area import/init failures before accepting a BAD verdict."
        ),
        properties={
            "relative_path": {
                "type": "string",
                "description": (
                    "Path to the file relative to the run directory "
                    "(e.g. 'bisect/abc123def456/benchmark.log')."
                ),
            },
        },
        required=["relative_path"],
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
                out_dir = phase_dir / f"{sha[:_SHA_SHORT]}_{i}"
                out_dir.mkdir(parents=True, exist_ok=True)
                logger.info(
                    "_run_experiments: sha=%s run=%d → %s", sha[:_SHA_SHORT], i, out_dir
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
                        sha[:_SHA_SHORT], i, exc,
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

        if compute_grounding_result is None:
            return {
                "verdict": "ERROR",
                "separated": False,
                "kpis_regressing": [],
                "kpi_deltas": {},
                "separation_ratios": {},
                "high_variance_tasks": [],
                "n_good": len(good_results),
                "n_bad": len(bad_results),
                "note": "compute_grounding_result unavailable (ImportError).",
            }

        grounding_result: dict = compute_grounding_result(
            good_results=good_results,
            bad_results=bad_results,
            good_sha=good_sha,
            bad_sha=bad_sha,
            task_id=task_id,
            backend=backend,
        )

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

    def _bisect_step(n_runs: int = 1) -> dict:
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
                logger.info("_bisect_step: bisect_result.json exists → DONE (%s)", first_bad[:_SHA_SHORT])
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
        # Helper: run once with infra retry (up to 3 attempts).
        #
        # This is the INFRA retry budget — it handles transient environment
        # failures (oom/hang/driver/runner_error).  It is separate from the
        # STATISTICAL re-run budget controlled by n_runs in _run_n_and_classify.
        #
        # Returns (sha, verdict, failure_phase, run_result).
        # ------------------------------------------------------------------

        def _run_one_with_infra_retry(
            idx: int, run_num: int = 0, *, attempt: int = 1
        ) -> tuple[str, str, str | None, dict]:
            sha = commits[idx]["sha"]
            stat_suffix = f"_s{run_num}" if run_num > 0 else ""
            infra_suffix = "" if attempt == 1 else f"_retry{attempt}"
            out_dir = bisect_dir / f"{sha[:_SHA_SHORT]}{stat_suffix}{infra_suffix}"
            out_dir.mkdir(parents=True, exist_ok=True)
            logger.info(
                "_bisect_step._run_one: idx=%d sha=%s run_num=%d attempt=%d",
                idx, sha[:_SHA_SHORT], run_num, attempt,
            )
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
                logger.error("_bisect_step: runner failed sha=%s: %s", sha[:_SHA_SHORT], exc)
                err_result: dict = {
                    "sha": sha, "failure_phase": "runner_error",
                    "raw_fps_mean": None, "exit_code": -1,
                }
                return sha, "SKIP", "runner_error", err_result

            if classify_bisect_verdict is None:
                return sha, "SKIP", None, run_result

            verdict = classify_bisect_verdict(run_result, good_stats, kpis_regressing)
            failure_phase: str | None = run_result.get("failure_phase")
            logger.info(
                "_bisect_step: sha=%s run=%d attempt=%d verdict=%s fp=%s",
                sha[:_SHA_SHORT], run_num, attempt, verdict, failure_phase,
            )

            if attempt < 3:
                # import/init: retry once — could be env setup fluke
                if failure_phase in {"import", "init"} and attempt == 1:
                    logger.info(
                        "_bisect_step: %s → %s run=%d attempt=1; retrying once.",
                        sha[:_SHA_SHORT], failure_phase, run_num,
                    )
                    return _run_one_with_infra_retry(idx, run_num, attempt=2)

                # Infra SKIP: retry up to 3 total
                if verdict == "SKIP" and failure_phase in _FAILURE_PHASE_SKIP:
                    logger.info(
                        "_bisect_step: %s → SKIP (infra: %s, run=%d, attempt=%d); retrying.",
                        sha[:_SHA_SHORT], failure_phase, run_num, attempt,
                    )
                    return _run_one_with_infra_retry(idx, run_num, attempt=attempt + 1)

            return sha, verdict, failure_phase, run_result

        # ------------------------------------------------------------------
        # Helper: run N times for statistics, aggregate, classify.
        #
        # This is the STATISTICAL re-run budget — separate from infra retries.
        #   - BAD-phase failure (runtime/import/init confirmed): return BAD
        #     immediately, no further statistical runs needed.
        #   - Infra SKIP after 3 retries: that statistical slot is lost; try
        #     another slot if the budget allows.
        #   - n_runs successful results collected → aggregate and classify.
        # ------------------------------------------------------------------
        def _run_n_and_classify(
            idx: int, n_runs: int
        ) -> tuple[str, str, str | None, list[dict]]:
            sha = commits[idx]["sha"]
            successful_results: list[dict] = []
            all_results: list[dict] = []
            # Allow extra attempts to absorb occasional infra losses
            max_stat_attempts = n_runs + 2

            for run_num in range(max_stat_attempts):
                if len(successful_results) >= n_runs:
                    break

                _, single_verdict, fp, run_result = _run_one_with_infra_retry(idx, run_num)
                all_results.append({
                    "run_num": run_num,
                    "verdict": single_verdict,
                    "failure_phase": fp,
                    "fps_mean": run_result.get("raw_fps_mean"),
                    "fps_p5": run_result.get("raw_fps_p5"),
                    "fps_median": run_result.get("raw_fps_median"),
                    "gpu_mem_mb": run_result.get("gpu_mem_used_mb"),
                    "exit_code": run_result.get("exit_code"),
                })

                if single_verdict == "BAD" and fp in {"import", "init", "runtime"}:
                    # Commit-caused crash — stop immediately, no more runs needed
                    logger.info(
                        "_bisect_step: %s → BAD (fp=%s run=%d); stopping statistical runs.",
                        sha[:_SHA_SHORT], fp, run_num,
                    )
                    return sha, "BAD", fp, all_results

                if single_verdict == "SKIP":
                    # Infra SKIP — slot lost, but try next run if budget remains
                    logger.info(
                        "_bisect_step: %s run=%d → SKIP (infra); slot lost.",
                        sha[:_SHA_SHORT], run_num,
                    )
                    continue

                # Successful run with KPI data
                successful_results.append(run_result)

            if not successful_results:
                logger.warning(
                    "_bisect_step: %s — all %d statistical runs SKIPped; returning SKIP.",
                    sha[:_SHA_SHORT], max_stat_attempts,
                )
                return sha, "SKIP", "missing_result", all_results

            # Aggregate N successful results — build a synthetic run_result from
            # per-KPI medians so classify_bisect_verdict can compare to good_stats.
            if compute_kpi_stats is not None and len(successful_results) > 1:
                agg_stats = compute_kpi_stats(successful_results)
                agg_result: dict = {
                    "sha": sha,
                    "failure_phase": None,
                    "raw_fps_mean": agg_stats.get("fps_mean", {}).get("median"),
                    "raw_fps_p5": agg_stats.get("fps_p5", {}).get("median"),
                    "raw_fps_median": agg_stats.get("fps_median", {}).get("median"),
                    "gpu_mem_used_mb": agg_stats.get("gpu_mem_used_mb", {}).get("median"),
                }
            else:
                # n_runs=1 or no stats module: use the single result directly
                agg_result = successful_results[0]

            agg_verdict = (
                classify_bisect_verdict(agg_result, good_stats, kpis_regressing)
                if classify_bisect_verdict is not None
                else "SKIP"
            )
            logger.info(
                "_bisect_step: %s → %s (aggregated %d/%d runs)",
                sha[:_SHA_SHORT], agg_verdict, len(successful_results), n_runs,
            )
            return sha, agg_verdict, None, all_results

        # ------------------------------------------------------------------
        # Execute ONE binary search step.
        # ------------------------------------------------------------------
        mid = (lo + hi) // 2

        _last_failure_phase: str | None = None
        _step_run_results: list[dict] = []

        if mid in tested_by_index:
            verdict = tested_by_index[mid]
            tested_sha = commits[mid]["sha"]
            logger.info(
                "_bisect_step: mid=%d already tested → %s (from state)", mid, verdict
            )
        else:
            tested_sha, verdict, _last_failure_phase, _step_run_results = (
                _run_n_and_classify(mid, n_runs)
            )
            tested.append({"sha": tested_sha, "index": mid, "verdict": verdict})
            tested_by_index[mid] = verdict
            if verdict == "SKIP":
                skip_count += 1

        # Apply leftmost-BAD bisect transition.
        if verdict == "GOOD":
            lo = mid + 1
        elif verdict == "BAD":
            hi = mid
        else:
            # SKIP: try mid+1 then mid-1.
            resolved = False
            for fallback_idx in (mid + 1, mid - 1):
                if lo <= fallback_idx <= hi and fallback_idx not in tested_by_index:
                    fb_sha, fb_verdict, fb_fp, _ = _run_n_and_classify(fallback_idx, n_runs)
                    tested.append(
                        {"sha": fb_sha, "index": fallback_idx, "verdict": fb_verdict}
                    )
                    tested_by_index[fallback_idx] = fb_verdict
                    if fb_verdict == "SKIP":
                        skip_count += 1

                    if fb_verdict == "GOOD":
                        lo = fallback_idx + 1
                        resolved = True
                        break
                    elif fb_verdict == "BAD":
                        hi = fallback_idx
                        resolved = True
                        break

            if not resolved:
                lo = mid + 1

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
                    first_bad_sha[:_SHA_SHORT],
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
            "failure_phase": _last_failure_phase,
            "commits_remaining": hi - lo,
            # Per-run structured results for LLM inspection.
            # Each entry: {run_num, verdict, failure_phase, fps_mean, fps_p5,
            #              fps_median, gpu_mem_mb, exit_code}
            "run_results": _step_run_results,
            "n_runs_requested": n_runs,
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

    def _write_plan(
        n_runs_per_commit: int,
        variance_class: str,
        rationale: str,
        kpis_regressing: list[str],
        grounding_cv: dict | None = None,
    ) -> dict:
        """Persist the bisect execution plan to bisect_plan.json."""
        plan: dict = {
            "n_runs_per_commit": n_runs_per_commit,
            "variance_class": variance_class,
            "rationale": rationale,
            "kpis_regressing": kpis_regressing,
            "grounding_cv": grounding_cv or {},
            "created_at": _now_iso(),
        }
        plan_path = run_dir / "bisect_plan.json"
        try:
            plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
            logger.info(
                "_write_plan: n_runs=%d variance=%s → %s",
                n_runs_per_commit, variance_class, plan_path,
            )
        except OSError as exc:
            logger.warning("_write_plan: could not write bisect_plan.json: %s", exc)
        return plan

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
            bisect_result.get("first_bad_sha", "unknown")[:_SHA_SHORT],
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
                max_turns=12,
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

    def _tool_bisect_step(n_runs: int = 1, **_kwargs: Any) -> str:
        result = _bisect_step(n_runs=max(1, n_runs))
        return json.dumps(result, default=str)

    def _tool_write_plan(
        n_runs_per_commit: int,
        variance_class: str,
        rationale: str,
        kpis_regressing: list[str],
        grounding_cv: dict | None = None,
        **_kwargs: Any,
    ) -> str:
        result = _write_plan(
            n_runs_per_commit=n_runs_per_commit,
            variance_class=variance_class,
            rationale=rationale,
            kpis_regressing=kpis_regressing,
            grounding_cv=grounding_cv,
        )
        return json.dumps(result, indent=2)

    def _tool_fetch_diff(sha_a: str, sha_b: str) -> str:
        return _fetch_diff(sha_a, sha_b)

    def _tool_run_diagnosis(**_kwargs: Any) -> str:
        result = _run_diagnosis()
        return json.dumps(result, default=str)

    def _tool_read_artifact(relative_path: str) -> str:
        """Read a file under run_dir; max 4000 characters."""
        try:
            target = (run_dir / relative_path).resolve()
            run_dir_resolved = run_dir.resolve()
            target.relative_to(run_dir_resolved)  # raises ValueError if outside
        except ValueError:
            return json.dumps({
                "error": f"Access denied: '{relative_path}' resolves outside run_dir."
            })
        except Exception as exc:  # noqa: BLE001
            return json.dumps({"error": f"Path resolution error: {exc}"})

        if not target.exists():
            return json.dumps({"error": f"File not found: {relative_path}"})
        if not target.is_file():
            return json.dumps({"error": f"Not a regular file: {relative_path}"})

        try:
            content = target.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return json.dumps({"error": f"Could not read file: {exc}"})

        truncated = len(content) > 4000
        result: dict[str, Any] = {"content": content[:4000]}
        if truncated:
            result["truncated"] = True
            result["total_chars"] = len(content)
        return json.dumps(result, indent=2)

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
        "write_plan": _tool_write_plan,
        "fetch_diff": _tool_fetch_diff,
        "read_artifact": _tool_read_artifact,
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
