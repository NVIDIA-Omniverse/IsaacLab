# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for human-readable bisection terminal progress."""

from __future__ import annotations

import io
import sys
from dataclasses import replace
from pathlib import Path

_GATE_DIR = Path(__file__).resolve().parents[1]
if str(_GATE_DIR) not in sys.path:
    sys.path.insert(0, str(_GATE_DIR))

from bisection.env_setup import _progress as emit_environment_progress  # noqa: E402
from bisection.models import BisectionPlan, RunnerSpec  # noqa: E402
from bisection.progress import ProgressReporter, configure_progress, format_metric  # noqa: E402
from bisection_harness import _sampling_warnings  # noqa: E402


def test_quiet_progress_emits_nothing() -> None:
    stream = io.StringIO()
    reporter = ProgressReporter(mode="quiet", stream=stream)

    reporter.event("START", "hidden")
    reporter.relay("[perf-bisect] hidden setup")
    reporter._last_heartbeat_at -= reporter.heartbeat_interval_s
    reporter.heartbeat("hidden heartbeat")

    assert stream.getvalue() == ""


def test_compact_progress_formats_phase_and_suppresses_verbose_details() -> None:
    stream = io.StringIO()
    reporter = ProgressReporter(mode="compact", stream=stream)

    reporter.event("good ref", "qualifying abc123")
    reporter.event("env", "verbose detail", verbose_only=True)
    reporter.relay("[perf-bisect] inner setup")

    output = stream.getvalue()
    assert "GOOD REF" in output
    assert "qualifying abc123" in output
    assert "verbose detail" not in output
    assert "inner setup" not in output


def test_verbose_progress_relays_structured_setup_only() -> None:
    stream = io.StringIO()
    reporter = configure_progress("verbose", stream=stream)

    reporter.relay("ordinary dependency installer noise")
    reporter.relay("[perf-bisect] installing pinned runtime stack")

    output = stream.getvalue()
    assert "ordinary dependency installer noise" not in output
    assert "SETUP" in output
    assert "installing pinned runtime stack" in output
    configure_progress("quiet")


def test_heartbeat_is_rate_limited() -> None:
    stream = io.StringIO()
    reporter = ProgressReporter(mode="compact", stream=stream, heartbeat_interval_s=60)

    reporter.heartbeat("still running")
    assert stream.getvalue() == ""

    reporter._last_heartbeat_at -= 60
    reporter.heartbeat("still running")
    assert "RUNNING" in stream.getvalue()


def test_metric_formatting_is_compact() -> None:
    assert format_metric(296741.104, "fps") == "296,741.1 fps"
    assert format_metric(7.422, "%") == "7.422 %"
    assert format_metric(12.3456, None) == "12.35"


def test_environment_setup_milestones_require_verbose_mode(monkeypatch, capsys) -> None:
    monkeypatch.setenv("PERF_BISECT_PROGRESS", "compact")
    emit_environment_progress("hidden")
    assert capsys.readouterr().out == ""

    monkeypatch.setenv("PERF_BISECT_PROGRESS", "verbose")
    emit_environment_progress("creating environment")
    assert capsys.readouterr().out == "[perf-bisect] creating environment\n"


def test_sampling_warnings_distinguish_defaults_from_smoke_overrides() -> None:
    plan = BisectionPlan(
        task_id="Isaac-Cartpole-Direct",
        backend_key="physx",
        good_ref="good",
        bad_ref="bad",
        gpu_model="NVIDIA L40S",
        runner=RunnerSpec(mode="synthetic"),
    )
    assert plan.measurement.reference_runs == 3
    assert plan.measurement.warmup_runs == 1
    assert _sampling_warnings(plan) == []

    low_confidence = replace(
        plan,
        measurement=replace(plan.measurement, reference_runs=1, max_reference_runs=1, warmup_runs=0),
    )
    warnings = _sampling_warnings(low_confidence)
    assert any("cannot estimate reference noise" in warning for warning in warnings)
    assert any("cold-start effects" in warning for warning in warnings)
