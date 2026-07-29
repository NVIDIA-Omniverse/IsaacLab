# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Emit the omni-github test-result artifact for the golden correctness gate.

The golden analogue of :mod:`omni_github`. Writes a manifest plus a result JSON
from the golden aggregate's scored rows, with a distinct ``test_tool_id`` and a
``custom.golden_policy.*`` namespace so omni-github ingests it as its own tool --
keeping the performance gate's artifact byte-for-byte unchanged (the golden gate
is an add-on, not a modification of the perf reporting path).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

try:
    from .gate_types import OracleVerdict
except ImportError:  # pragma: no cover - executed as a script, not a package
    from gate_types import OracleVerdict

MANIFEST_NAME = "omni-github-test-results-upload.json"
RESULT_REL_PATH = "_testoutput/golden_test_results.json"
RESULT_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
CUSTOM_NAMESPACE = "golden_policy"
TEST_TOOL_ID = "golden-policy"

_PASSING_VERDICTS = (OracleVerdict.PASS, OracleVerdict.WARN)


def _number(value: Any) -> float | int | None:
    """Return a finite number for storage, or ``None`` so the field is dropped (bools rejected)."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return None


def _drop_none(fields: dict[str, Any]) -> dict[str, Any]:
    """Drop keys whose value is ``None`` (omni-github stores omitted, not null)."""
    return {key: value for key, value in fields.items() if value is not None}


def _custom_fields(result: Any, golden_result: Any) -> dict[str, Any]:
    """Project the golden verdict and result into ``custom.golden_policy`` fields."""
    return _drop_none(
        {
            "task_id": result.task_id,
            "physics_backend": getattr(golden_result, "physics_backend", None),
            "render_backend": getattr(golden_result, "render_backend", None),
            "checkpoint_id": result.checkpoint_id,
            "verdict": result.verdict.value,
            "failure_phase": result.failure_phase,
            "wall_time_s": _number(getattr(golden_result, "wall_time_s", None)),
            "reward_mean": _number(result.reward_mean),
            "ep_length_mean": _number(result.ep_length_mean),
            "success_rate": _number(result.success_rate),
            "num_episodes": _number(result.num_episodes),
        }
    )


def _message(result: Any) -> str:
    parts = [result.verdict.value]
    if result.reward_mean is not None:
        parts.append(f"reward={result.reward_mean:.2f}")
    if result.failure_phase:
        parts.append(f"phase={result.failure_phase}")
    if result.note:
        parts.append(result.note)
    return " ".join(parts)


def _row(result: Any, golden_result: Any) -> dict[str, Any]:
    duration = max(float(getattr(golden_result, "wall_time_s", None) or 0.0), 0.0)
    passed = result.verdict in _PASSING_VERDICTS
    row: dict[str, Any] = {
        "test_id": f"golden.{result.task_id}::{result.backend}",
        "test_name": result.backend,
        "test_type": "correctness",
        "passed": passed,
        "duration": duration,
        "custom": {CUSTOM_NAMESPACE: _custom_fields(result, golden_result)},
    }
    if not passed:
        row["message"] = _message(result)
    return row


def build_result(rows, *, platform: str, app_config: str, test_tool_id: str = TEST_TOOL_ID) -> dict[str, Any]:
    """Build the omni-github result payload from ``(golden_oracle_result, golden_result)`` rows."""
    return {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "test_tool_id": test_tool_id,
        "app": {"platform": platform, "config": app_config},
        "tests": [_row(result, golden_result) for result, golden_result in rows],
    }


def write_artifact(rows, output_dir, *, platform: str, app_config: str, test_tool_id: str = TEST_TOOL_ID) -> Path:
    """Write the manifest and result JSON omni-github ingests; returns the artifact root.

    ``output_dir`` MUST be separate from the performance gate's artifact directory:
    the omni-github manifest filename is fixed by the ingestion contract, so a
    shared directory would clobber one gate's manifest and only one would ingest.
    Upload the golden artifact as its own GitHub Actions artifact.
    """
    output_dir = Path(output_dir)
    result_path = output_dir / RESULT_REL_PATH
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(build_result(rows, platform=platform, app_config=app_config, test_tool_id=test_tool_id)),
        encoding="utf-8",
    )
    manifest = {"schema_version": MANIFEST_SCHEMA_VERSION, "result_paths": [RESULT_REL_PATH]}
    (output_dir / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    return output_dir
