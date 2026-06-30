"""
core/report.py — Report and prompt generation for the bisect agent.

Pure functions with no I/O side-effects except write_report_md (which writes
a single Markdown file). No LLM calls, no subprocess calls, no benchmark runs.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Diagnostician user prompt
# ---------------------------------------------------------------------------


def build_user_prompt(bisect_result: dict, grounding_result: dict) -> str:
    """Build the user prompt for the diagnostician LLM session."""
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
        f"**Budget: 12 tool calls total (including write_diagnosis). Spend them wisely.**\n\n"
        f"Typical call budgets by case:\n"
        f"- CASE A (dep-only): fetch_diff + read_bisect_path + write_diagnosis (~3 calls, 0 experiments)\n"
        f"- CASE B (dep+code): + 1 optional experiment (~5 calls)\n"
        f"- CASE C (code-only): + hot-path analysis + 1 confirming experiment (~6 calls)\n"
        f"- CASE D (unclear): fetch_diff + write_diagnosis(indeterminate) (~2 calls)\n\n"
        f"Steps:\n"
        f"1. Call fetch_diff(sha_a='{prev_good}', sha_b='{first_bad}') to examine "
        f"what changed. Check `hot_path_files` in the response — files touching "
        f"performance-critical paths (step/reset/compute, physics/kernels dirs). "
        f"Prioritize these when forming hypotheses.\n"
        f"2. Call read_bisect_path() to assess bisect evidence quality: clean cliff "
        f"(all GOOD then sudden BAD) = strong causal signal; interspersed SKIPs = noisy.\n"
        f"3. Triage by case. Run experiments only for specific testable hypotheses "
        f"(max 3 total, 0 for CASE A/D).\n"
        f"4. Call write_diagnosis when done. Do not delay — conclude with available evidence.\n"
    )


# ---------------------------------------------------------------------------
# Fallback diagnosis
# ---------------------------------------------------------------------------


def make_indeterminate_diagnosis(
    bisect_result: dict,
    reason: str = "Agent did not call write_diagnosis.",
) -> dict:
    """Return a minimal indeterminate diagnosis dict when the LLM fails to produce one."""
    first_bad = bisect_result.get("first_bad_sha", "unknown")
    kpi_deltas = bisect_result.get("kpi_deltas", {})
    kpi_impact: dict = {
        kpi: {"delta_pct": delta, "good": None, "bad": None}
        for kpi, delta in kpi_deltas.items()
    }
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
# Markdown report
# ---------------------------------------------------------------------------


def write_report_md(diagnosis: dict, run_dir: Path) -> None:
    """Write report/report.md from a completed diagnosis dict."""
    report_dir = Path(run_dir) / "report"
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

    lines += ["", "---", "", "## Root Cause", "", root_cause, "", "---", "", "## Hypotheses", ""]

    if hypotheses:
        for h in hypotheses:
            h_id = h.get("id", "?")
            desc = h.get("description", "")
            evidence = h.get("evidence", "")
            tested = h.get("tested", False)
            conclusion = h.get("conclusion", "")
            lines += [
                f"### {h_id}: {desc}",
                "",
                f"- **Evidence:** {evidence}",
                f"- **Tested:** {'Yes' if tested else 'No'}",
            ]
            if tested and conclusion:
                lines.append(f"- **Conclusion:** {conclusion}")
            lines.append("")
    else:
        lines += ["_No hypotheses recorded._", ""]

    lines += ["---", "", "## Recommended Actions", ""]
    if recommended_actions:
        lines += [f"- {a}" for a in recommended_actions]
    else:
        lines.append("_None._")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote report.md to %s", report_path)
