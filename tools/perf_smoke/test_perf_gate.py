# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pytest orchestrator for the perf-smoke gate (D1).

The doc asks for the gate to be *shelled out by pytest*: pytest is the test
framework, but it never launches Isaac Sim in-process (Isaac Sim is a process
singleton). This module honors that exactly -- each task is one parametrized
test that launches the benchmark as its own subprocess and then runs the
pure-logic comparator, asserting the verdict is not ``BLOCK``.

It runs as an ordinary pytest module because ``pytest.ini`` in this directory
pins the rootdir here, so the repo-wide ``tools/conftest.py`` harness (which
hijacks every pytest session under ``tools/``) is not loaded.

Launching Isaac Sim needs a GPU, so the gate tests self-skip unless
``GATE_RUN=1`` -- a plain ``pytest tools/perf_smoke`` still runs the comparator
unit tests but does not try to start the simulator. On the runner::

    GATE_RUN=1 GATE_TASKS="Isaac-Cartpole-v0 ..." \
        ./isaaclab.sh -p -m pytest tools/perf_smoke/test_perf_gate.py

Recognized environment:

* ``GATE_TASKS``     -- space-separated task list (default: all tasks in baseline.json).
* ``GATE_CACHE_DIR`` -- warm Warp/CUDA JIT cache dir (optional; additive).
* ``GATE_OUTPUT_DIR``-- where benchmark JSON is written (default: <repo>/perf-output).
* ``GATE_GPU``       -- baseline GPU key override (e.g. "NVIDIA L40S").
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import run_perf_gate as gate

_THIS = Path(__file__).resolve().parent
_BASELINE = _THIS / "baseline.json"
_HISTORY = _THIS / "perf_history"
_OVERRIDES = _THIS / "baseline_overrides.json"

pytestmark = pytest.mark.skipif(
    os.environ.get("GATE_RUN") != "1",
    reason="set GATE_RUN=1 to launch the perf gate (needs a GPU + Isaac Sim)",
)


def _gate_tasks() -> list[str]:
    env = os.environ.get("GATE_TASKS", "").split()
    if env:
        return env
    with open(_BASELINE, encoding="utf-8") as f:
        return [k for k in json.load(f) if not k.startswith("_")]


@pytest.mark.parametrize("task", _gate_tasks())
def test_perf_gate(task: str) -> None:
    """Run one task end-to-end (benchmark subprocess -> comparator) and assert no BLOCK."""
    baseline = gate._load_baseline(_BASELINE)
    cfg = gate._task_run_config(baseline, task)

    out_root = Path(os.environ.get("GATE_OUTPUT_DIR", gate._REPO_ROOT / "perf-output")).resolve()
    task_out = out_root / task
    cache_dir = os.environ.get("GATE_CACHE_DIR") or None
    gpu = os.environ.get("GATE_GPU") or None

    wall = gate._run_benchmark(task, cfg, task_out, retries=1, dry_run=False, cache_dir=cache_dir)
    assert wall is not None, f"{task}: benchmark did not produce a result after retries (BLOCK/hard_failure)"

    code, label = gate._run_comparator(
        task,
        task_out,
        _BASELINE,
        gpu,
        history_dir=str(_HISTORY),
        overrides_path=_OVERRIDES,
        wall_s=wall,
        task_id=cfg.get("task_id"),
    )
    # PASS and WARN both exit 0; BLOCK (regression) and hard_failure are non-zero.
    assert code == gate.EXIT_PASS, f"{task}: gate verdict {label} (exit {code})"
