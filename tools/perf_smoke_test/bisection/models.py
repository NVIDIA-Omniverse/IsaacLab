# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Data contracts for the IsaacLab bisection harness POC."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class RunnerSpec:
    """Structured configuration for running one bisection candidate."""

    mode: str = "synthetic"
    image: str | None = None
    source_dir: str | None = None
    jit_cache: str | None = None
    kit_cache: str | None = None
    local_env_dir: str | None = None
    ld_preload: str | None = None
    extra_args: list[str] = field(default_factory=list)

    @classmethod
    def from_json(cls, data: dict[str, Any] | None) -> RunnerSpec | None:
        """Build a runner spec from JSON, tolerating absent optional fields."""
        if not data:
            return None
        return cls(
            mode=str(data.get("mode", "synthetic")),
            image=data.get("image"),
            source_dir=data.get("source_dir"),
            jit_cache=data.get("jit_cache"),
            kit_cache=data.get("kit_cache"),
            local_env_dir=data.get("local_env_dir"),
            ld_preload=data.get("ld_preload"),
            extra_args=[str(item) for item in data.get("extra_args", [])],
        )


@dataclass(frozen=True)
class TimeoutPolicy:
    """Timeout configuration for candidate runs."""

    candidate_timeout_s: int | None = None

    @classmethod
    def from_json(cls, data: dict[str, Any] | None) -> TimeoutPolicy:
        """Build timeout policy from JSON."""
        if not data:
            return cls()
        value = data.get("candidate_timeout_s")
        return cls(candidate_timeout_s=int(value) if value is not None else None)


@dataclass(frozen=True)
class RetryPolicy:
    """Retry configuration for candidate runs."""

    max_attempts: int = 1
    retryable_notes: list[str] = field(
        default_factory=lambda: ["candidate_timeout", "runner_command_failed", "missing_perf_smoke_test_result"]
    )
    retry_delay_s: int = 0

    @classmethod
    def from_json(cls, data: dict[str, Any] | None) -> RetryPolicy:
        """Build retry policy from JSON."""
        if not data:
            return cls()
        max_attempts = max(1, int(data.get("max_attempts", 1)))
        retryable_notes = data.get("retryable_notes")
        if retryable_notes is None:
            retryable_notes = cls().retryable_notes
        return cls(
            max_attempts=max_attempts,
            retryable_notes=[str(item) for item in retryable_notes],
            retry_delay_s=max(0, int(data.get("retry_delay_s", 0))),
        )


@dataclass(frozen=True)
class BisectionPlan:
    """Plan for bisecting one regressed IsaacLab task/backend cell."""

    task_id: str
    backend_key: str
    good_ref: str
    bad_ref: str
    gpu_model: str
    baselines_dir: str
    gate_config: str
    runner: RunnerSpec | None = None
    timeout: TimeoutPolicy = field(default_factory=TimeoutPolicy)
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    source_gate_artifact_dir: str | None = None
    source_gate_result: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 2

    def to_json(self) -> dict[str, Any]:
        """Serialize the plan to JSON."""
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> BisectionPlan:
        """Build a plan from JSON, tolerating absent optional fields."""
        return cls(
            task_id=str(data["task_id"]),
            backend_key=str(data["backend_key"]),
            good_ref=str(data["good_ref"]),
            bad_ref=str(data["bad_ref"]),
            gpu_model=str(data.get("gpu_model", "unknown-gpu")),
            baselines_dir=str(data.get("baselines_dir", "tools/perf_smoke_test/local_baselines")),
            gate_config=str(data.get("gate_config", "tools/perf_smoke_test/gate_config.json")),
            runner=RunnerSpec.from_json(data.get("runner")),
            timeout=TimeoutPolicy.from_json(data.get("timeout")),
            retry=RetryPolicy.from_json(data.get("retry")),
            source_gate_artifact_dir=data.get("source_gate_artifact_dir"),
            source_gate_result=dict(data.get("source_gate_result") or {}),
            schema_version=int(data.get("schema_version", 2)),
        )


@dataclass(frozen=True)
class CandidateAttempt:
    """One execution attempt for a candidate commit."""

    attempt: int
    artifact_dir: str
    command: str
    command_exit_code: int | None = None
    note: str | None = None
    timed_out: bool = False
    duration_s: float | None = None

    def to_json(self) -> dict[str, Any]:
        """Serialize the candidate attempt to JSON."""
        return asdict(self)


@dataclass(frozen=True)
class CandidateEvaluation:
    """Result of evaluating one candidate commit."""

    commit_sha: str
    bisect_verdict: str
    artifact_dir: str
    command_exit_code: int | None = None
    oracle_verdict: str | None = None
    measured_fps: float | None = None
    baseline_fps: float | None = None
    regression_pct: float | None = None
    baseline_sample_count: int = 0
    threshold_source: str | None = None
    note: str | None = None
    command: str | None = None
    attempt_count: int = 1
    attempts: list[dict[str, Any]] = field(default_factory=list)
    timed_out: bool = False
    duration_s: float | None = None
    retry_reason: str | None = None
    final_artifact_dir: str | None = None

    def to_json(self) -> dict[str, Any]:
        """Serialize the candidate evaluation to JSON."""
        return asdict(self)


@dataclass(frozen=True)
class BisectionSummary:
    """Final bisection summary."""

    status: str
    reason: str
    tested_commits: list[str]
    suspected_first_bad_commit: str | None
    last_good_commit: str | None
    bad_ref: str
    good_ref: str

    def to_json(self) -> dict[str, Any]:
        """Serialize the summary to JSON."""
        return asdict(self)
