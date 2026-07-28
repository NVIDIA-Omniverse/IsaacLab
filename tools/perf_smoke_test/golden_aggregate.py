# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Aggregate golden results, run the golden oracle, and report.

The golden analogue of :mod:`aggregate`, but standalone: it scores each
``golden_result.json`` against the hard-set KPI thresholds in ``golden_tasks.json``
(no baselines, no git, no compatibility hashing), renders a *separate* golden
results table, and optionally emits a *separate* omni-github artifact
(``test_tool_id=golden-policy``). The performance aggregate is untouched.

The gate is advisory by default: golden verdicts affect the exit code only when
``gate_config.golden_blocking`` is explicitly enabled (independent of the perf
``blocking`` flag).
"""

import argparse
import json
import os
import sys
from pathlib import Path

_MODULE_DIR = Path(__file__).parent
_TOOLS_DIR = _MODULE_DIR.parent
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from gate_config import load_gate_config  # noqa: E402
from gate_types import OracleVerdict  # noqa: E402
from golden_config import get_golden_task  # noqa: E402
from golden_contracts import GoldenResult  # noqa: E402
from golden_omni_github import write_artifact as write_golden_omni_github  # noqa: E402
from golden_oracle import evaluate  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate golden results and run the golden oracle.")
    p.add_argument("--artifacts_dir", required=True, type=Path)
    p.add_argument("--golden_tasks", type=Path, default=_MODULE_DIR / "golden_tasks.json")
    p.add_argument("--gate_config", type=Path, default=_MODULE_DIR / "gate_config.json")
    p.add_argument("--summary_file", default=None)
    p.add_argument("--omni_github_dir", type=Path, default=None, help="Write the golden omni-github artifact here")
    p.add_argument("--omni_platform", default="linux-x86_64")
    p.add_argument("--omni_app_config", default="golden-policy")
    return p.parse_args()


def _find_golden_results(artifacts_dir: Path) -> list[tuple[Path, GoldenResult]]:
    found = []
    for path in sorted(artifacts_dir.rglob("golden_result.json")):
        with path.open() as fh:
            found.append((path.parent, GoldenResult.from_dict(json.load(fh))))
    return found


def _kpis_for(result: GoldenResult, golden_tasks_path: Path) -> dict:
    """Resolve the configured KPI thresholds for a result's (task, backend), or empty if none."""
    try:
        task = get_golden_task(result.task_id, result.backend_key or result.backend, golden_tasks_path)
    except KeyError:
        return {}
    return task.kpis


def _fmt(value, decimals: int = 2) -> str:
    return f"{value:.{decimals}f}" if isinstance(value, (int, float)) else "N/A"


def _crossed_tags(kpi_results: list[dict]) -> str:
    tags = []
    for kr in kpi_results:
        for ct in kr.get("crossed_thresholds", []):
            verdict = ct.get("threshold_verdict") or "report"
            tags.append(f"{kr['kpi']}:{ct.get('threshold_name')}({verdict})")
    return "; ".join(tags)


def _build_table(rows: list[tuple]) -> str:
    lines = [
        "| Task | Backend | Verdict | Checkpoint | Reward | EpLen | Success | Episodes | Crossed | Phase | Note |",
        "|---|---|---|---|---:|---:|---:|---:|---|---|---|",
    ]
    for result, _golden in rows:
        lines.append(
            f"| {result.task_id} | {result.backend} | {result.verdict.value}"
            f" | {result.checkpoint_id or ''} | {_fmt(result.reward_mean)} | {_fmt(result.ep_length_mean)}"
            f" | {_fmt(result.success_rate)} | {result.num_episodes if result.num_episodes is not None else 'N/A'}"
            f" | {_crossed_tags(result.kpi_results)} | {result.failure_phase or ''} | {result.note or ''} |"
        )
    return "\n".join(lines)


def _write_github_output(**values) -> None:
    github_output = os.environ.get("GITHUB_OUTPUT", "")
    if not github_output:
        return
    with open(github_output, "a") as fh:
        for key, value in values.items():
            if value is not None:
                fh.write(f"{key}={value}\n")


def main() -> int:
    args = _parse_args()
    gate_config = load_gate_config(args.gate_config)
    golden_blocking = bool(gate_config.get("golden_blocking", False))

    items = _find_golden_results(args.artifacts_dir)
    if not items:
        print(f"[golden_aggregate] No golden_result.json files found under {args.artifacts_dir}")
        # Advisory add-on: an empty/missing golden artifact set must not break the
        # build unless golden gating is explicitly enabled.
        return 1 if golden_blocking else 0

    rows = []
    has_block = False
    has_hard_failure = False
    for _artifact_dir, result in items:
        oracle_result = evaluate(result, _kpis_for(result, args.golden_tasks))
        rows.append((oracle_result, result))
        print(
            f"[golden_aggregate] {oracle_result.task_id}/{oracle_result.backend}: {oracle_result.verdict.value}"
            f"  reward={_fmt(oracle_result.reward_mean)}  ep_length={_fmt(oracle_result.ep_length_mean)}"
            f"  episodes={oracle_result.num_episodes}"
            + (f"  note={oracle_result.note}" if oracle_result.note else "")
        )
        if oracle_result.verdict == OracleVerdict.BLOCK:
            has_block = True
        elif oracle_result.verdict == OracleVerdict.HARD_FAILURE:
            has_hard_failure = True

    table = _build_table(rows)
    print("\n## Golden Correctness Results\n")
    print(table)
    print()

    if args.summary_file:
        with open(args.summary_file, "a") as fh:
            fh.write("\n## Golden Correctness Results\n\n")
            fh.write(table)
            fh.write("\n")

    if args.omni_github_dir:
        write_golden_omni_github(
            rows, args.omni_github_dir, platform=args.omni_platform, app_config=args.omni_app_config
        )

    _write_github_output(
        golden_has_block="true" if has_block else "false",
        golden_has_hard_failure="true" if has_hard_failure else "false",
        golden_blocking="true" if golden_blocking else "false",
    )

    if golden_blocking:
        if has_block:
            return 1
        if has_hard_failure:
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
