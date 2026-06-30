"""
core/diagnosis.py — LLM-driven root-cause analysis for the bisect agent.

Implements run_diagnosis(), which:
1. Checks for a cached report/diagnosis.json.
2. Builds a user prompt summarising the bisect + grounding context.
3. Defines four tool functions for the LLM sub-session.
4. Runs an LLM tool-use loop via llm_client.run_session().
5. Falls back to an indeterminate diagnosis if the agent never calls write_diagnosis.
6. Generates report/report.md from the final diagnosis.
"""

from __future__ import annotations

import json
import logging
import os
import re
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

# Attempt to import LLMClient for type annotations only.
# The client is always passed as a parameter; this import is never load-critical.
try:
    from infra.llm_client import LLMClient
except ImportError:
    try:
        from llm_client import LLMClient  # type: ignore[no-redef]
    except ImportError:
        LLMClient = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Import report helpers
# ---------------------------------------------------------------------------
try:
    from .report import (
        build_user_prompt as _build_user_prompt,
        make_indeterminate_diagnosis as _indeterminate_diagnosis,
        write_report_md as _write_report_md,
    )
except ImportError:
    from core.report import (  # type: ignore[no-redef]
        build_user_prompt as _build_user_prompt,
        make_indeterminate_diagnosis as _indeterminate_diagnosis,
        write_report_md as _write_report_md,
    )

logger = logging.getLogger(__name__)

_SHA_SHORT = 12  # chars used for artifact dir names and log messages

# ---------------------------------------------------------------------------
# Default system prompt (fallback when prompts/diagnostician.md is absent)
# ---------------------------------------------------------------------------

_DEFAULT_SYSTEM_PROMPT = (
    "You are a forensic performance investigator for Isaac Lab benchmarks.\n"
    "Follow the triage protocol: fetch_diff -> analyse -> triage by case -> write_diagnosis.\n"
    "CASE A (dep-only): upstream_dep, no experiments. "
    "CASE B (dep+code): 1 optional experiment. "
    "CASE C (code-only): hot path analysis, 1 confirming experiment. "
    "CASE D (unclear): indeterminate.\n"
    "Max 3 experiments total. Only state conclusions backed by evidence."
)

# ---------------------------------------------------------------------------
# Tool schemas (OpenAI function-calling format)
# ---------------------------------------------------------------------------

_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "fetch_diff",
            "description": (
                "Fetch the diff between two commits. "
                "Returns files changed, dep file diffs, commit message, and a diff summary."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sha_a": {
                        "type": "string",
                        "description": "The base/older commit SHA.",
                    },
                    "sha_b": {
                        "type": "string",
                        "description": "The head/newer commit SHA.",
                    },
                },
                "required": ["sha_a", "sha_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_experiment",
            "description": (
                "Run the benchmark for a specific commit. "
                "Optionally override fps_mean for dev-mode stubs. "
                "Maximum 3 experiments may be run across the entire session."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sha": {
                        "type": "string",
                        "description": "The commit SHA to run.",
                    },
                    "fps_override": {
                        "type": "number",
                        "description": (
                            "Optional fps_mean override for dev-mode stub benchmarks."
                        ),
                    },
                },
                "required": ["sha"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_artifact",
            "description": (
                "Read a file from the run directory (relative path). "
                "Returns up to 4000 characters of the file content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "relative_path": {
                        "type": "string",
                        "description": (
                            "Path to the file, relative to the run directory "
                            "(e.g. 'grounding/result.json')."
                        ),
                    },
                },
                "required": ["relative_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_bisect_path",
            "description": (
                "Read KPI values for every commit tested during bisection (O(log N) points), "
                "sorted by commit index (oldest-first). "
                "Bisection does NOT test every commit — only ~log2(N) midpoints. "
                "Use this to assess the quality of the bisect evidence: "
                "a 'clean cliff' (all GOOD before first_bad, then BAD) = strong causal evidence; "
                "interspersed SKIPs or low fps variance = noisy/uncertain result. "
                "Returns fps_mean, verdict, and failure_phase per tested commit, "
                "plus good_baseline and total_commits_in_range for context."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_diagnosis",
            "description": (
                "Write the final diagnosis JSON. "
                "Call this exactly once when you have completed your analysis."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "diagnosis_json": {
                        "type": "object",
                        "description": (
                            "The completed diagnosis object matching the diagnosis.json schema."
                        ),
                    },
                },
                "required": ["diagnosis_json"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Helper: resolve system prompt
# ---------------------------------------------------------------------------


def _load_system_prompt(bisect_agent_root: Path | None) -> str:
    """Return diagnostician system prompt; fall back to default constant."""
    if bisect_agent_root is not None:
        prompt_path = bisect_agent_root / "prompts" / "diagnostician.md"
        if prompt_path.exists():
            try:
                return prompt_path.read_text(encoding="utf-8").strip()
            except OSError as exc:
                logger.warning("Could not read %s: %s", prompt_path, exc)
    return _DEFAULT_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Hot-path pre-filter
# ---------------------------------------------------------------------------

_HOT_PATH_DIRS: tuple[str, ...] = (
    "envs/", "physics/", "kernels/", "simulation/",
    "warp/", "cuda/", "assets/", "scene/",
)
_HOT_PATH_FILE_RE = re.compile(
    r"(step|reset|compute|simulate|update|forward|backward|rollout)",
    re.IGNORECASE,
)


def _annotate_hot_path(files_changed: list[dict]) -> list[dict]:
    """Return files that touch performance-critical paths, with reason annotation.

    A file is hot-path if its path contains a critical directory segment OR its
    filename matches a performance-critical verb (step, reset, compute, …).
    """
    hot: list[dict] = []
    for f in files_changed:
        path: str = f.get("path", "")
        if any(pat in path for pat in _HOT_PATH_DIRS):
            hot.append({**f, "hot_path_reason": "critical_dir"})
        elif _HOT_PATH_FILE_RE.search(path.rsplit("/", 1)[-1]):
            hot.append({**f, "hot_path_reason": "critical_filename"})
    return hot


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_diagnosis(
    bisect_result: dict,
    grounding_result: dict,
    run_dir: Path,
    llm_client: Any,
    commits_fetch_diff: Callable,
    runner_run_commit: Callable,
    repo_path: Path | None = None,
    dev_mode: bool = False,
    dev_perf_map: dict | None = None,
    max_turns: int = 12,
) -> dict:
    """
    LLM-driven root-cause analysis.

    Parameters
    ----------
    bisect_result:
        Contents of bisect_result.json (schema 4.4).
    grounding_result:
        Contents of grounding/result.json (schema 4.2).
    run_dir:
        Root directory for this bisect run (all artifacts live here).
    llm_client:
        An LLMClient instance (infra/llm_client.py).
    commits_fetch_diff:
        infra/commits.py::fetch_diff callable.
    runner_run_commit:
        core/runner.py::run_commit callable.
    repo_path:
        Optional path to the local git repo (forwarded to commits_fetch_diff).
    dev_mode:
        If True, experiments use stub_benchmark rather than Docker.
    dev_perf_map:
        Optional sha->fps_mean map used in dev mode experiments.

    Returns
    -------
    dict
        The diagnosis dict (schema 4.5), also written to report/diagnosis.json.
    """
    run_dir = Path(run_dir)
    report_dir = run_dir / "report"
    diagnosis_path = report_dir / "diagnosis.json"

    # ------------------------------------------------------------------
    # 1. Cache check
    # ------------------------------------------------------------------
    if diagnosis_path.exists():
        try:
            cached = json.loads(diagnosis_path.read_text(encoding="utf-8"))
            logger.info("Loaded cached diagnosis from %s", diagnosis_path)
            # Regenerate report.md in case it is missing
            report_md_path = report_dir / "report.md"
            if not report_md_path.exists():
                _write_report_md(cached, run_dir)
            return cached
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read cached diagnosis (%s); re-running.", exc)

    # ------------------------------------------------------------------
    # 2. Resolve system prompt
    # ------------------------------------------------------------------
    # Detect bisect_agent root from this file's location
    _this_file = Path(__file__).resolve()
    bisect_agent_root = _this_file.parent.parent  # bisect_agent/
    system_prompt = _load_system_prompt(bisect_agent_root)

    # ------------------------------------------------------------------
    # 3. Build user prompt
    # ------------------------------------------------------------------
    user_prompt = _build_user_prompt(bisect_result, grounding_result)

    # ------------------------------------------------------------------
    # 4. Prepare mutable state for tool closures
    # ------------------------------------------------------------------
    _state: dict[str, Any] = {
        "diagnosis": None,       # set by write_diagnosis tool
        "experiments_run": 0,
        "write_called": False,
    }

    first_bad_sha: str = bisect_result.get("first_bad_sha", "")
    task_id: str = grounding_result.get("task_id", "")
    backend: str = grounding_result.get("backend", "")

    # ------------------------------------------------------------------
    # 5. Define tool functions
    # ------------------------------------------------------------------

    def _tool_fetch_diff(sha_a: str, sha_b: str) -> str:
        """Fetch diff between two commits; returns JSON string."""
        logger.info("tool fetch_diff: %s..%s", sha_a, sha_b)
        try:
            result = commits_fetch_diff(
                sha_a,
                sha_b,
                repo_path=repo_path,
            )
            # Pre-annotate performance-critical files so the LLM focuses there first.
            hot = _annotate_hot_path(result.get("files_changed", []))
            if hot:
                result["hot_path_files"] = hot
            return json.dumps(result, indent=2)
        except Exception as exc:  # noqa: BLE001
            logger.error("fetch_diff failed: %s", exc)
            return json.dumps({"error": str(exc)})

    def _tool_run_experiment(sha: str, fps_override: float | None = None) -> str:
        """Run a benchmark experiment; max 3 total."""
        if _state["experiments_run"] >= 3:
            msg = "Experiment budget exhausted (max 3). No more experiments may be run."
            logger.warning(msg)
            return json.dumps({"error": msg})

        exp_num = _state["experiments_run"] + 1
        logger.info(
            "tool run_experiment: sha=%s fps_override=%s (experiment %d/3)",
            sha,
            fps_override,
            exp_num,
        )

        # Build a unique output directory inside run_dir
        exp_idx = _state["experiments_run"]
        exp_dir = run_dir / "diagnosis" / f"experiment_{exp_idx}_{sha[:_SHA_SHORT]}"
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Build per-experiment dev_perf_map override
        exp_dev_perf_map = dict(dev_perf_map) if dev_perf_map else {}
        if fps_override is not None and sha:
            exp_dev_perf_map[sha] = fps_override

        _state["experiments_run"] += 1

        # Audit: record experiment start
        audit_path = run_dir / "audit_log.jsonl"
        audit_start: dict = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "step": "diagnosis_experiment_start",
            "experiment_n": exp_num,
            "sha": sha,
            "fps_override": fps_override,
            "artifact_dir": str(exp_dir.relative_to(run_dir)),
        }
        try:
            with audit_path.open("a") as _fh:
                _fh.write(json.dumps(audit_start) + "\n")
        except OSError:
            pass

        try:
            result = runner_run_commit(
                sha=sha,
                task_id=task_id,
                backend=backend,
                output_dir=exp_dir,
                dev_mode=dev_mode,
                dev_perf_map=exp_dev_perf_map if dev_mode else None,
            )

            # Audit: record experiment result
            audit_result: dict = {
                "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "step": "diagnosis_experiment_result",
                "experiment_n": exp_num,
                "sha": sha,
                "fps_mean": result.get("raw_fps_mean"),
                "gpu_mem_mb": result.get("gpu_mem_used_mb"),
                "failure_phase": result.get("failure_phase"),
                "exit_code": result.get("exit_code"),
            }
            try:
                with audit_path.open("a") as _fh:
                    _fh.write(json.dumps(audit_result) + "\n")
            except OSError:
                pass

            return json.dumps(result, indent=2)
        except Exception as exc:  # noqa: BLE001
            logger.error("run_experiment failed: %s\n%s", exc, traceback.format_exc())
            return json.dumps({"error": str(exc)})

    def _tool_read_artifact(relative_path: str) -> str:
        """Read a file under run_dir; max 4000 characters."""
        # Safety: resolve and check containment
        try:
            target = (run_dir / relative_path).resolve()
            run_dir_resolved = run_dir.resolve()
            target.relative_to(run_dir_resolved)  # raises ValueError if outside
        except ValueError:
            msg = (
                f"Access denied: '{relative_path}' resolves outside run_dir. "
                "Only files within the run directory may be read."
            )
            logger.warning(msg)
            return json.dumps({"error": msg})
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
        excerpt = content[:4000]
        result: dict[str, Any] = {"content": excerpt}
        if truncated:
            result["truncated"] = True
            result["total_chars"] = len(content)
        return json.dumps(result, indent=2)

    def _tool_write_diagnosis(diagnosis_json: dict) -> str:
        """Write diagnosis.json; marks diagnosis as complete."""
        if _state["write_called"]:
            logger.warning("write_diagnosis called more than once; ignoring duplicate.")
            return json.dumps({"status": "ignored", "reason": "already called"})

        # Stamp experiment count
        diagnosis_json["experiments_run"] = _state["experiments_run"]

        # Persist
        report_dir.mkdir(parents=True, exist_ok=True)
        try:
            diagnosis_path.write_text(
                json.dumps(diagnosis_json, indent=2), encoding="utf-8"
            )
            logger.info("Wrote diagnosis.json to %s", diagnosis_path)
        except OSError as exc:
            logger.error("Failed to write diagnosis.json: %s", exc)
            return json.dumps({"status": "error", "reason": str(exc)})

        _state["diagnosis"] = diagnosis_json
        _state["write_called"] = True
        return json.dumps({"status": "ok", "path": str(diagnosis_path)})

    def _tool_read_bisect_path() -> str:
        """Return KPI values for all bisected commits, sorted by commit index."""
        state_path = run_dir / "bisect" / "state.json"
        if not state_path.exists():
            return json.dumps({"error": "bisect/state.json not found — bisect not yet complete."})

        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            return json.dumps({"error": f"Could not read bisect/state.json: {exc}"})

        tested: list[dict] = state.get("tested", [])
        bisect_dir = run_dir / "bisect"
        points: list[dict] = []

        for entry in sorted(tested, key=lambda x: x.get("index", 0)):
            sha: str = entry.get("sha", "")
            verdict: str = entry.get("verdict", "?")
            point: dict = {
                "commit_index": entry.get("index"),
                "sha": sha,
                "verdict": verdict,
                "failure_phase": None,
                "fps_mean": None,
                "fps_p5": None,
                "gpu_mem_mb": None,
            }
            # Glob all dirs for this SHA (handles _retry2, _retry3, _s1, _s1_retry2, etc.)
            # Sort alphabetically so the base dir (no suffix) comes first, then retries.
            sha_prefix = sha[:_SHA_SHORT]
            candidate_dirs = sorted(bisect_dir.glob(f"{sha_prefix}*/"))
            # Fall back to exact match dir if glob finds nothing (e.g. no trailing slash needed)
            if not candidate_dirs and (bisect_dir / sha_prefix).is_dir():
                candidate_dirs = [bisect_dir / sha_prefix]

            for candidate_dir in candidate_dirs:
                rr_path = candidate_dir / "run_result.json"
                if rr_path.exists():
                    try:
                        rr = json.loads(rr_path.read_text(encoding="utf-8"))
                        # Prefer the first dir with fps_mean data (successful run)
                        if rr.get("raw_fps_mean") is not None:
                            point["fps_mean"] = rr.get("raw_fps_mean")
                            point["fps_p5"] = rr.get("raw_fps_p5")
                            point["gpu_mem_mb"] = rr.get("gpu_mem_used_mb")
                            point["failure_phase"] = rr.get("failure_phase")
                            break
                        # Keep failure_phase from last run if no successful run found
                        point["failure_phase"] = rr.get("failure_phase")
                    except Exception:  # noqa: BLE001
                        pass
            points.append(point)

        # Good-SHA baseline from grounding for comparison
        good_stats = grounding_result.get("good_stats") or {}
        good_baseline = {
            kpi: stats.get("median")
            for kpi, stats in good_stats.items()
            if isinstance(stats, dict)
        }

        return json.dumps({
            "trend": points,
            "good_baseline": good_baseline,
            "total_commits_in_range": state.get("commits_total"),
            "skip_count": state.get("skip_count", 0),
        }, indent=2)

    # ------------------------------------------------------------------
    # 6. Build tool dispatch map
    # ------------------------------------------------------------------
    tool_dispatch: dict[str, Callable] = {
        "fetch_diff": _tool_fetch_diff,
        "run_experiment": _tool_run_experiment,
        "read_artifact": _tool_read_artifact,
        "read_bisect_path": _tool_read_bisect_path,
        "write_diagnosis": _tool_write_diagnosis,
    }

    # ------------------------------------------------------------------
    # 7. Run LLM sub-session
    # ------------------------------------------------------------------
    try:
        llm_client.run_session(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            tools=_TOOLS,
            tool_dispatch=tool_dispatch,
            max_tool_rounds=max_turns,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "LLM session raised an exception: %s\n%s", exc, traceback.format_exc()
        )

    # ------------------------------------------------------------------
    # 8. Fallback: agent never called write_diagnosis
    # ------------------------------------------------------------------
    if not _state["write_called"]:
        logger.warning("Agent never called write_diagnosis; writing indeterminate fallback.")
        fallback = _indeterminate_diagnosis(
            bisect_result,
            reason="Agent did not call write_diagnosis.",
        )
        fallback["experiments_run"] = _state["experiments_run"]
        report_dir.mkdir(parents=True, exist_ok=True)
        try:
            diagnosis_path.write_text(json.dumps(fallback, indent=2), encoding="utf-8")
        except OSError as exc:
            logger.error("Failed to write fallback diagnosis.json: %s", exc)
        _state["diagnosis"] = fallback

    diagnosis: dict = _state["diagnosis"]

    # ------------------------------------------------------------------
    # 9. Generate report.md
    # ------------------------------------------------------------------
    try:
        _write_report_md(diagnosis, run_dir)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Failed to generate report.md: %s\n%s", exc, traceback.format_exc()
        )

    return diagnosis
