# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GPU-free tests for bisection artifact handoff contracts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_GATE_DIR = Path(__file__).resolve().parents[1]
if str(_GATE_DIR) not in sys.path:
    sys.path.insert(0, str(_GATE_DIR))

from isaaclab_bisection.bisection.artifacts import (  # noqa: E402
    classify_blocker,
    finalize_run_artifacts,
    write_attempt_summary,
)
from isaaclab_bisection.contracts import BenchResult  # noqa: E402


def _plan() -> dict:
    return {
        "task_id": "Isaac-Velocity-Flat-G1-v0",
        "backend_key": "newton",
        "metric": {"name": "raw_fps_mean", "result_path": "raw_fps_mean", "regression_direction": "decrease"},
        "runner": {"mode": "docker-reconstruct", "image": "isaaclab-bisect:base"},
    }


def _attempt(artifact_dir: Path, note: str) -> dict:
    return {
        "attempt": 1,
        "artifact_dir": str(artifact_dir),
        "command": "fake command",
        "command_exit_code": 0,
        "note": note,
        "timed_out": False,
        "duration_s": 1.0,
        "recovery_events": [],
    }


def test_attempt_summary_classifies_disk_full_install_failure(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "measurements" / "good_ref" / "abc123" / "task" / "newton" / "run_1"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "bisect_env.json").write_text(
        json.dumps(
            {
                "status": "skip",
                "skip_category": "install_failed",
                "skip_detail": "failed to write to file: No space left on device (os error 28)",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    install_log = tmp_path / "env-cache" / "logs" / "install-abc123def456.log"
    install_log.parent.mkdir(parents=True)
    install_log.write_text("Failed to extract archive: No space left on device\n", encoding="utf-8")

    summary = write_attempt_summary(
        tmp_path,
        plan=_plan(),
        attempt=_attempt(artifact_dir, "env_skip:install_failed"),
        commit_sha="abc123def4567890",
        label="good_ref",
        run_idx=1,
        recovery_attempt=0,
        metric_value=None,
    )
    blocker = classify_blocker(summary)

    assert blocker is not None
    assert blocker["category"] == "host_resource"
    assert blocker["phase"] == "install"
    assert blocker["owner"] == "outer_agent"
    assert "attempt_summary" in summary["paths"]


def test_attempt_summary_indexes_separate_docker_logs(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "measurements" / "benchmark" / "abc123" / "task" / "physx" / "run_1"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "docker_command.log").write_text("docker text\n", encoding="utf-8")
    (artifact_dir / "docker_live_output.jsonl").write_text('{"event": "process_exit"}\n', encoding="utf-8")

    summary = write_attempt_summary(
        tmp_path,
        plan=_plan(),
        attempt=_attempt(artifact_dir, ""),
        commit_sha="abc123def4567890",
        label="benchmark",
        run_idx=1,
        recovery_attempt=0,
        metric_value=123.0,
    )

    assert summary["paths"]["docker_command_log"].endswith("docker_command.log")
    assert summary["paths"]["docker_live_output"].endswith("docker_live_output.jsonl")
    assert summary["evidence"]["docker_command_log_tail"] == "docker text\n"
    assert summary["evidence"]["docker_live_output_tail"] == '{"event": "process_exit"}\n'


def test_attempt_summary_classifies_missing_module_runtime_failure(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "measurements" / "good_ref" / "abc123" / "task" / "newton" / "run_1"
    artifact_dir.mkdir(parents=True)
    stdout_tail = "ModuleNotFoundError: No module named 'h5py'"
    (artifact_dir / "perf_smoke_test_result.json").write_text(
        json.dumps({"exit_code": 1, "failure_phase": "import", "stdout_tail": stdout_tail}) + "\n",
        encoding="utf-8",
    )
    (artifact_dir / "benchmark.log").write_text(stdout_tail, encoding="utf-8")

    summary = write_attempt_summary(
        tmp_path,
        plan=_plan(),
        attempt=_attempt(artifact_dir, "env_skip:runtime_incompatible"),
        commit_sha="abc123def4567890",
        label="good_ref",
        run_idx=1,
        recovery_attempt=0,
        metric_value=None,
    )
    blocker = classify_blocker(summary)

    assert blocker is not None
    assert blocker["category"] == "runtime_incompatible"
    assert blocker["phase"] == "benchmark_import"
    assert "h5py" in blocker["reason"]


def test_attempt_summary_classifies_tooling_support_boundary(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "measurements" / "candidate" / "abc123" / "task" / "newton" / "run_1"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "bisect_env.json").write_text(
        json.dumps(
            {
                "status": "skip",
                "skip_category": "perf_smoke_tooling_incompatible",
                "skip_detail": "isaaclab.app.launch_simulation",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = write_attempt_summary(
        tmp_path,
        plan=_plan(),
        attempt=_attempt(artifact_dir, "env_skip:perf_smoke_tooling_incompatible"),
        commit_sha="abc123def4567890",
        label="candidate",
        run_idx=1,
        recovery_attempt=0,
        metric_value=None,
    )
    blocker = classify_blocker(summary)

    assert blocker is not None
    assert blocker["category"] == "perf_smoke_tooling_incompatible"
    assert blocker["retryable"] is False


def test_attempt_summary_reads_typed_bench_result(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "measurements" / "good_ref" / "abc123" / "task" / "newton" / "run_1"
    artifact_dir.mkdir(parents=True)
    bench_result = BenchResult(
        task_id="Isaac-Velocity-Flat-G1-v0",
        backend="newton",
        physics_backend="newton",
        render_backend=None,
        backend_key="newton",
        preset="default",
        exit_code=0,
        failure_phase=None,
        stdout_tail="Step Frametimes",
        raw_fps_mean=1234.5,
    )
    (artifact_dir / "perf_smoke_test_result.json").write_text(
        json.dumps(bench_result.to_dict()) + "\n", encoding="utf-8"
    )

    summary = write_attempt_summary(
        tmp_path,
        plan=_plan(),
        attempt=_attempt(artifact_dir, ""),
        commit_sha="abc123def4567890",
        label="good_ref",
        run_idx=1,
        recovery_attempt=0,
        metric_value=1234.5,
    )

    assert summary["benchmark"]["result_present"] is True
    assert summary["benchmark"]["raw_fps_mean"] == 1234.5
    assert summary["benchmark"]["exit_code"] == 0
    assert summary["benchmark"]["failure_phase"] is None


def test_attempt_summary_tolerates_truncated_bench_result(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "measurements" / "good_ref" / "abc123" / "task" / "newton" / "run_1"
    artifact_dir.mkdir(parents=True)
    # A truncated write missing required identity fields must degrade to "no result"
    # rather than crashing the artifact builder.
    (artifact_dir / "perf_smoke_test_result.json").write_text(
        json.dumps({"raw_fps_mean": 1.0}) + "\n", encoding="utf-8"
    )

    summary = write_attempt_summary(
        tmp_path,
        plan=_plan(),
        attempt=_attempt(artifact_dir, ""),
        commit_sha="abc123def4567890",
        label="good_ref",
        run_idx=1,
        recovery_attempt=0,
        metric_value=None,
    )

    assert summary["benchmark"]["result_present"] is False
    assert summary["benchmark"]["raw_fps_mean"] is None


def test_finalize_run_artifacts_writes_handoff_files(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "measurements" / "good_ref" / "abc123" / "task" / "newton" / "run_1"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "bisect_env.json").write_text(
        json.dumps(
            {
                "status": "skip",
                "skip_category": "install_failed",
                "skip_detail": "failed to write to file: No space left on device (os error 28)",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    write_attempt_summary(
        tmp_path,
        plan=_plan(),
        attempt=_attempt(artifact_dir, "env_skip:install_failed"),
        commit_sha="abc123def4567890",
        label="good_ref",
        run_idx=1,
        recovery_attempt=0,
        metric_value=None,
    )

    enriched = finalize_run_artifacts(
        tmp_path,
        _plan(),
        {
            "status": "inconclusive",
            "reason": "good_ref_measurement_failed:env_skip:install_failed",
            "good_ref": "abc123def4567890",
            "bad_ref": "bad",
            "suspected_first_bad_commit": None,
            "last_good_commit": None,
            "reference_stats": {"good_attempts": [{"artifact_dir": str(artifact_dir)}]},
        },
    )

    assert enriched["terminal_blocker"]["category"] == "host_resource"
    assert (tmp_path / "blockers.json").exists()
    assert (tmp_path / "artifact_index.json").exists()
    assert (tmp_path / "report.md").exists()
    assert "No space" not in (tmp_path / "report.md").read_text(encoding="utf-8")
    artifact_index = json.loads((tmp_path / "artifact_index.json").read_text(encoding="utf-8"))
    assert artifact_index["top_level"]["summary.json"] == "summary.json"


def test_finalize_run_artifacts_ignores_stale_attempts(tmp_path: Path) -> None:
    stale_dir = tmp_path / "measurements" / "good_ref" / "old" / "Old-Task" / "newton" / "run_1"
    stale_dir.mkdir(parents=True)
    (stale_dir / "perf_smoke_test_result.json").write_text(
        json.dumps({"exit_code": 1, "failure_phase": "import", "stdout_tail": "No module named 'old_dep'"}) + "\n",
        encoding="utf-8",
    )
    write_attempt_summary(
        tmp_path,
        plan={**_plan(), "task_id": "Old-Task"},
        attempt=_attempt(stale_dir, "env_skip:runtime_incompatible"),
        commit_sha="old123",
        label="good_ref",
        run_idx=1,
        recovery_attempt=0,
        metric_value=None,
    )

    good_dir = tmp_path / "measurements" / "good_ref" / "abc123" / "task" / "newton" / "run_1"
    bad_dir = tmp_path / "measurements" / "bad_ref" / "def456" / "task" / "newton" / "run_1"
    good_dir.mkdir(parents=True)
    bad_dir.mkdir(parents=True)
    write_attempt_summary(
        tmp_path,
        plan=_plan(),
        attempt=_attempt(good_dir, ""),
        commit_sha="abc123",
        label="good_ref",
        run_idx=1,
        recovery_attempt=0,
        metric_value=100.0,
    )
    write_attempt_summary(
        tmp_path,
        plan=_plan(),
        attempt=_attempt(bad_dir, ""),
        commit_sha="def456",
        label="bad_ref",
        run_idx=1,
        recovery_attempt=0,
        metric_value=90.0,
    )

    enriched = finalize_run_artifacts(
        tmp_path,
        _plan(),
        {
            "status": "completed",
            "reason": "first_bad_found",
            "good_ref": "abc123",
            "bad_ref": "def456",
            "suspected_first_bad_commit": "def456",
            "last_good_commit": "abc123",
            "reference_stats": {
                "good_attempts": [{"artifact_dir": str(good_dir)}],
                "bad_attempts": [{"artifact_dir": str(bad_dir)}],
            },
            "tested_commits": [],
            "skipped_commits": [],
        },
    )

    assert enriched["terminal_blocker"] is None
    blockers = json.loads((tmp_path / "blockers.json").read_text(encoding="utf-8"))
    assert blockers["blockers"] == []
    artifact_index = json.loads((tmp_path / "artifact_index.json").read_text(encoding="utf-8"))
    assert {attempt["artifact_dir"] for attempt in artifact_index["attempts"]} == {
        "measurements/good_ref/abc123/task/newton/run_1",
        "measurements/bad_ref/def456/task/newton/run_1",
    }


def test_report_renders_component_stack_diff(tmp_path: Path) -> None:
    """A summary carrying a ``stack_diff`` surfaces changed components in report.md.

    The regression may live in a pinned dependency bump rather than IsaacLab source, so
    the human-facing report must name which component moved (e.g. newton) across the
    culprit commit even though the bisection target is IsaacLab commits.
    """
    good_dir = tmp_path / "measurements" / "good_ref" / "abc123" / "task" / "newton" / "run_1"
    good_dir.mkdir(parents=True)
    write_attempt_summary(
        tmp_path,
        plan=_plan(),
        attempt=_attempt(good_dir, ""),
        commit_sha="abc123",
        label="good_ref",
        run_idx=1,
        recovery_attempt=0,
        metric_value=100.0,
    )
    finalize_run_artifacts(
        tmp_path,
        _plan(),
        {
            "status": "completed",
            "reason": "first_bad_found",
            "good_ref": "abc123",
            "bad_ref": "def456",
            "suspected_first_bad_commit": "def456",
            "last_good_commit": "abc123",
            "reference_stats": {"good_attempts": [{"artifact_dir": str(good_dir)}]},
            "tested_commits": [],
            "skipped_commits": [],
            "stack_diff": {
                "culprit": {
                    "relation": "last_good_to_first_bad",
                    "isaaclab_commit": {"from": "abc123", "to": "def456"},
                    "changed_components": {"newton": {"from": "==1.2.1", "to": "==1.3.0"}},
                    "stack_hash": {"from": "aaa", "to": "bbb", "changed": True},
                }
            },
        },
    )
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Component Stack Diff" in report
    assert "`newton`" in report
    assert "==1.2.1" in report and "==1.3.0" in report


def test_report_renders_decision_evidence_and_sampling_confidence(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "abc123def456.json").write_text(
        json.dumps(
            {
                "commit_sha": "abc123def4567890",
                "bisect_verdict": "GOOD",
                "measured_value": 1020.0,
                "baseline_value": 1000.0,
                "regression_pct": -2.0,
                "attempt_count": 1,
                "metric_unit": "fps",
                "threshold_source": "paired_reference",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (results_dir / "def456abc123.json").write_text(
        json.dumps(
            {
                "commit_sha": "def456abc1237890",
                "bisect_verdict": "BAD",
                "measured_value": 800.0,
                "baseline_value": 1000.0,
                "regression_pct": 20.0,
                "attempt_count": 2,
                "metric_unit": "fps",
                "threshold_source": "paired_reference",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    plan = {
        **_plan(),
        "measurement": {
            "reference_runs": 3,
            "max_reference_runs": 7,
            "candidate_runs": 1,
            "max_candidate_runs": 3,
            "warmup_runs": 1,
        },
    }
    enriched = finalize_run_artifacts(
        tmp_path,
        plan,
        {
            "status": "completed",
            "reason": "first_bad_found",
            "good_ref": "good",
            "bad_ref": "bad",
            "suspected_first_bad_commit": "def456abc1237890",
            "last_good_commit": "abc123def4567890",
            "metric": {"name": "raw_fps_mean", "unit": "fps", "regression_direction": "decrease"},
            "reference_stats": {
                "good": {
                    "median_value": 1000.0,
                    "sample_count": 3,
                    "spread_pct": 1.2,
                    "metric_name": "raw_fps_mean",
                    "unit": "fps",
                    "values": [995.0, 1000.0, 1005.0],
                },
                "bad": {
                    "median_value": 800.0,
                    "sample_count": 3,
                    "spread_pct": 1.5,
                    "metric_name": "raw_fps_mean",
                    "unit": "fps",
                    "values": [795.0, 800.0, 805.0],
                },
                "check": {
                    "reproduced": True,
                    "regression_pct": 20.0,
                    "effective_threshold_pct": 5.0,
                    "reference_noise_pct": 1.0,
                    "note": None,
                },
            },
            "tested_commits": ["abc123def4567890", "def456abc1237890"],
            "skipped_commits": [],
        },
    )

    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Decision Evidence" in report
    assert "Good reference: 1,000.0 fps (3 samples, 1.20% spread)" in report
    assert "Endpoint regression: 20.00% vs 5.00% threshold" in report
    assert "Last good `abc123def456`" in report and "`GOOD`" in report
    assert "First bad `def456abc123`" in report and "`BAD`" in report
    assert "Confidence: **nominal**" in report
    assert enriched["decision_evidence"]["sampling"]["confidence"] == "nominal"


def test_report_flags_single_sample_without_warmup_as_limited(tmp_path: Path) -> None:
    plan = {
        **_plan(),
        "measurement": {
            "reference_runs": 1,
            "max_reference_runs": 1,
            "candidate_runs": 1,
            "max_candidate_runs": 1,
            "warmup_runs": 0,
        },
    }
    enriched = finalize_run_artifacts(
        tmp_path,
        plan,
        {
            "status": "inconclusive",
            "reason": "regression_not_reproduced",
            "good_ref": "good",
            "bad_ref": "bad",
            "suspected_first_bad_commit": None,
            "last_good_commit": None,
            "metric": {"name": "raw_fps_mean", "unit": "fps", "regression_direction": "decrease"},
            "reference_stats": {
                "good": {"median_value": 1000.0, "sample_count": 1, "spread_pct": 0.0, "unit": "fps"},
                "bad": {"median_value": 980.0, "sample_count": 1, "spread_pct": 0.0, "unit": "fps"},
                "check": {
                    "reproduced": False,
                    "regression_pct": 2.0,
                    "effective_threshold_pct": 5.0,
                    "reference_noise_pct": 0.0,
                    "note": None,
                },
            },
            "tested_commits": [],
            "skipped_commits": [],
        },
    )

    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "1 sample, spread not estimable" in report
    assert "Confidence: **limited**" in report
    assert "reference noise is not estimable from a single sample" in report
    assert "process warmup was disabled" in report
    assert enriched["decision_evidence"]["sampling"]["confidence"] == "limited"


def test_probe_report_is_explicitly_non_authoritative(tmp_path: Path) -> None:
    finalize_run_artifacts(
        tmp_path,
        _plan(),
        {
            "status": "probe_completed",
            "reason": "internal_range_probe_completed",
            "good_ref": "good",
            "bad_ref": "bad",
            "suspected_first_bad_commit": None,
            "last_good_commit": None,
            "comparison_mode": "probe_range",
            "tested_commits": ["middle"],
            "skipped_commits": [],
        },
    )

    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Internal range probe" in report
    assert "non-authoritative" in report
    assert "do not identify a first-bad commit" in report
