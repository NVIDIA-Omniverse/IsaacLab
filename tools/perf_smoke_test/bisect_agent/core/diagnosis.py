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
import traceback
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

logger = logging.getLogger(__name__)

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
# Helper: build user prompt
# ---------------------------------------------------------------------------


def _build_user_prompt(bisect_result: dict, grounding_result: dict) -> str:
    first_bad = bisect_result.get("first_bad_sha", "unknown")
    prev_good = bisect_result.get("prev_good_sha", "unknown")
    first_bad_msg = bisect_result.get("first_bad_message", "")
    commits_tested = bisect_result.get("commits_tested", "?")
    confidence = bisect_result.get("confidence", "unknown")
    kpi_deltas: dict = bisect_result.get("kpi_deltas", {})
    kpis_regressing: list = grounding_result.get("kpis_regressing", [])
    separation_ratios: dict = grounding_result.get("separation_ratios", {})
    n_good = grounding_result.get("n_good", "?")
    n_bad = grounding_result.get("n_bad", "?")
    grounding_verdict = grounding_result.get("verdict", "?")

    # Format KPI delta table
    kpi_lines: list[str] = []
    for kpi, delta in kpi_deltas.items():
        sep = separation_ratios.get(kpi, "?")
        kpi_lines.append(f"  - {kpi}: {delta:+.1f}%  (separation_ratio={sep})")

    kpi_block = "\n".join(kpi_lines) if kpi_lines else "  (none recorded)"

    return (
        f"Bisection has identified the first bad commit. Perform root-cause analysis.\n\n"
        f"## Bisect Summary\n"
        f"- first_bad_sha: {first_bad}\n"
        f"- prev_good_sha: {prev_good}\n"
        f"- commit message: {first_bad_msg!r}\n"
        f"- commits_tested: {commits_tested}\n"
        f"- bisect confidence: {confidence}\n\n"
        f"## Grounding Statistics\n"
        f"- grounding verdict: {grounding_verdict}\n"
        f"- n_good runs: {n_good} | n_bad runs: {n_bad}\n"
        f"- KPIs regressing: {kpis_regressing}\n\n"
        f"## KPI Deltas (bad vs good)\n"
        f"{kpi_block}\n\n"
        f"## Your Task\n"
        f"1. Call fetch_diff(prev_good_sha='{prev_good}', first_bad_sha='{first_bad}') "
        f"to examine what changed.\n"
        f"2. Triage according to the four-case protocol.\n"
        f"3. Run experiments only when you have a specific, testable hypothesis "
        f"(max 3 total).\n"
        f"4. Call write_diagnosis with the completed diagnosis JSON when done.\n"
    )


# ---------------------------------------------------------------------------
# Helper: indeterminate fallback diagnosis
# ---------------------------------------------------------------------------


def _indeterminate_diagnosis(
    bisect_result: dict,
    reason: str = "Agent did not call write_diagnosis.",
) -> dict:
    first_bad = bisect_result.get("first_bad_sha", "unknown")
    kpi_deltas = bisect_result.get("kpi_deltas", {})
    kpi_impact: dict = {}
    for kpi, delta in kpi_deltas.items():
        kpi_impact[kpi] = {"delta_pct": delta, "good": None, "bad": None}

    return {
        "first_bad_sha": first_bad,
        "regression_class": "indeterminate",
        "kpi_impact": kpi_impact,
        "hypotheses": [],
        "root_cause": None,
        "recommended_actions": [
            "Manual investigation required.",
            "Use Tracy or Nsight profiler to identify the hot path regression.",
        ],
        "confidence": "low",
        "experiments_run": 0,
        "_fallback_reason": reason,
    }


# ---------------------------------------------------------------------------
# Helper: write report.md
# ---------------------------------------------------------------------------


def _write_report_md(diagnosis: dict, run_dir: Path) -> None:
    """Generate report/report.md from the diagnosis dict."""
    report_dir = run_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "report.md"

    first_bad = diagnosis.get("first_bad_sha", "unknown")
    regression_class = diagnosis.get("regression_class", "unknown")
    confidence = diagnosis.get("confidence", "unknown")
    commits_tested = diagnosis.get("commits_tested", "?")
    root_cause = diagnosis.get("root_cause") or "_No confirmed root cause._"
    experiments_run = diagnosis.get("experiments_run", 0)
    hypotheses: list[dict] = diagnosis.get("hypotheses", [])
    recommended_actions: list[str] = diagnosis.get("recommended_actions", [])
    kpi_impact: dict = diagnosis.get("kpi_impact", {})

    lines: list[str] = [
        "# Bisect Diagnosis Report",
        "",
        f"**First bad commit:** `{first_bad}`",
        f"**Regression class:** `{regression_class}`",
        f"**Confidence:** `{confidence}`",
        f"**Commits tested during bisect:** {commits_tested}",
        f"**Experiments run during diagnosis:** {experiments_run}",
        "",
        "---",
        "",
        "## KPI Impact",
        "",
    ]

    if kpi_impact:
        lines += [
            "| KPI | Delta (%) | Good baseline | Bad value |",
            "|-----|-----------|---------------|-----------|",
        ]
        for kpi, vals in kpi_impact.items():
            delta = vals.get("delta_pct")
            good = vals.get("good")
            bad = vals.get("bad")
            delta_str = f"{delta:+.1f}" if delta is not None else "?"
            good_str = f"{good:.1f}" if good is not None else "?"
            bad_str = f"{bad:.1f}" if bad is not None else "?"
            lines.append(f"| `{kpi}` | {delta_str}% | {good_str} | {bad_str} |")
    else:
        lines.append("_No KPI impact data available._")

    lines += [
        "",
        "---",
        "",
        "## Root Cause",
        "",
        root_cause,
        "",
        "---",
        "",
        "## Hypotheses",
        "",
    ]

    if hypotheses:
        for h in hypotheses:
            h_id = h.get("id", "?")
            desc = h.get("description", "")
            evidence = h.get("evidence", "")
            tested = h.get("tested", False)
            conclusion = h.get("conclusion", "")
            tested_str = "Yes" if tested else "No"
            lines += [
                f"### {h_id}: {desc}",
                "",
                f"- **Evidence:** {evidence}",
                f"- **Tested:** {tested_str}",
            ]
            if tested and conclusion:
                lines.append(f"- **Conclusion:** {conclusion}")
            lines.append("")
    else:
        lines.append("_No hypotheses recorded._")
        lines.append("")

    lines += [
        "---",
        "",
        "## Recommended Actions",
        "",
    ]

    if recommended_actions:
        for action in recommended_actions:
            lines.append(f"- {action}")
    else:
        lines.append("_None._")

    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote report.md to %s", report_path)


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

        logger.info(
            "tool run_experiment: sha=%s fps_override=%s (experiment %d/3)",
            sha,
            fps_override,
            _state["experiments_run"] + 1,
        )

        # Build a unique output directory inside run_dir
        exp_idx = _state["experiments_run"]
        exp_dir = run_dir / "diagnosis" / f"experiment_{exp_idx}_{sha[:7]}"
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Build per-experiment dev_perf_map override
        exp_dev_perf_map = dict(dev_perf_map) if dev_perf_map else {}
        if fps_override is not None and sha:
            exp_dev_perf_map[sha] = fps_override

        _state["experiments_run"] += 1

        try:
            result = runner_run_commit(
                sha=sha,
                task_id=task_id,
                backend=backend,
                output_dir=exp_dir,
                dev_mode=dev_mode,
                dev_perf_map=exp_dev_perf_map if dev_mode else None,
            )
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

    # ------------------------------------------------------------------
    # 6. Build tool dispatch map
    # ------------------------------------------------------------------
    tool_dispatch: dict[str, Callable] = {
        "fetch_diff": _tool_fetch_diff,
        "run_experiment": _tool_run_experiment,
        "read_artifact": _tool_read_artifact,
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
