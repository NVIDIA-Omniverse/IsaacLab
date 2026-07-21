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
    # ``--mode synthetic`` only: lets a caller rehearse the search over a *real*
    # commit range/task/backend with a chosen ground-truth regression point and
    # magnitude, at zero GPU/Docker cost, before committing to an expensive real
    # run. Omitted fields keep the runner's existing defaults (first-bad = the
    # plan's bad_ref; good/bad values derived from the metric's direction).
    synthetic_first_bad_ref: str | None = None
    synthetic_good_value: float | None = None
    synthetic_bad_value: float | None = None

    @classmethod
    def from_json(cls, data: dict[str, Any] | None) -> RunnerSpec | None:
        """Build a runner spec from JSON, tolerating absent optional fields."""
        if not data:
            return None
        good_value = data.get("synthetic_good_value")
        bad_value = data.get("synthetic_bad_value")
        return cls(
            mode=str(data.get("mode", "synthetic")),
            image=data.get("image"),
            source_dir=data.get("source_dir"),
            jit_cache=data.get("jit_cache"),
            kit_cache=data.get("kit_cache"),
            local_env_dir=data.get("local_env_dir"),
            ld_preload=data.get("ld_preload"),
            extra_args=[str(item) for item in data.get("extra_args", [])],
            synthetic_first_bad_ref=data.get("synthetic_first_bad_ref"),
            synthetic_good_value=float(good_value) if good_value is not None else None,
            synthetic_bad_value=float(bad_value) if bad_value is not None else None,
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
class TaskSpec:
    """Inline task workload so the agent does not require a ``tasks.json`` entry.

    Every field is optional. When a field is omitted the runner falls back to the
    ``tasks.json`` registry entry for ``(task_id, backend_key)`` if one exists;
    inline values always win over the registry. ``hydra_args``, when non-empty,
    *replaces* the backend-derived ``presets=`` string so any task's Hydra config
    groups (e.g. ``cube,single_camera,rgb64``) are expressible without editing the
    shared registry. If neither inline fields nor a registry entry supply the
    launch essentials, the runner errors with guidance rather than guessing.
    """

    num_envs: int | None = None
    num_frames: int | None = None
    warmup_frames: int | None = None
    seed: int | None = None
    camera_resolution: list[int] | None = None
    timeout_minutes: int | None = None
    hydra_args: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        """Serialize the task spec to JSON."""
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any] | None) -> TaskSpec:
        """Build a task spec from JSON, tolerating absent optional fields."""
        if not data:
            return cls()
        camera = data.get("camera_resolution")
        return cls(
            num_envs=int(data["num_envs"]) if data.get("num_envs") is not None else None,
            num_frames=int(data["num_frames"]) if data.get("num_frames") is not None else None,
            warmup_frames=int(data["warmup_frames"]) if data.get("warmup_frames") is not None else None,
            seed=int(data["seed"]) if data.get("seed") is not None else None,
            camera_resolution=[int(v) for v in camera] if camera is not None else None,
            timeout_minutes=int(data["timeout_minutes"]) if data.get("timeout_minutes") is not None else None,
            hydra_args=[str(item) for item in data.get("hydra_args", [])],
        )


@dataclass(frozen=True)
class MetricSpec:
    """Metric selected for local paired-reference bisection."""

    name: str = "raw_fps_mean"
    result_path: str = "raw_fps_mean"
    regression_direction: str = "decrease"
    unit: str | None = None

    def __post_init__(self) -> None:
        """Validate metric direction."""
        if self.regression_direction not in {"decrease", "increase"}:
            raise ValueError("metric regression_direction must be 'decrease' or 'increase'")

    def to_json(self) -> dict[str, Any]:
        """Serialize the metric selection to JSON."""
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any] | None) -> MetricSpec:
        """Build metric selection from JSON."""
        if not data:
            return cls()
        return cls(
            name=str(data.get("name") or data.get("result_path") or "raw_fps_mean"),
            result_path=str(data.get("result_path") or data.get("name") or "raw_fps_mean"),
            regression_direction=str(data.get("regression_direction", "decrease")),
            unit=data.get("unit"),
        )


@dataclass(frozen=True)
class MeasurementPolicy:
    """Local paired-reference measurement settings."""

    reference_runs: int = 3
    max_reference_runs: int = 7
    candidate_runs: int = 1
    max_candidate_runs: int = 3
    min_regression_pct: float = 5.0
    gray_zone_pct: float = 1.0
    reference_noise_multiplier: float = 2.0
    max_reference_spread_pct: float = 10.0
    # Process-level warmup: full benchmark attempts run before the measured runs of each
    # commit, sharing the (commit-stable) JIT/Kit caches so first-run kernel compilation
    # and cache population land on the warmup, not on a measured sample. Warmup attempts
    # are recorded as artifacts (under a ``*_warmup`` label) but excluded from the
    # median/noise statistics. Values greater than zero mean exactly one warmup per
    # commit; the integer field remains for schema compatibility.
    warmup_runs: int = 1

    @classmethod
    def from_json(cls, data: dict[str, Any] | None) -> MeasurementPolicy:
        """Build measurement policy from JSON."""
        if not data:
            return cls()
        return cls(
            reference_runs=max(1, int(data.get("reference_runs", 3))),
            max_reference_runs=max(1, int(data.get("max_reference_runs", 7))),
            candidate_runs=max(1, int(data.get("candidate_runs", 1))),
            max_candidate_runs=max(1, int(data.get("max_candidate_runs", 3))),
            min_regression_pct=max(0.0, float(data.get("min_regression_pct", 5.0))),
            gray_zone_pct=max(0.0, float(data.get("gray_zone_pct", 1.0))),
            reference_noise_multiplier=max(0.0, float(data.get("reference_noise_multiplier", 2.0))),
            max_reference_spread_pct=max(0.0, float(data.get("max_reference_spread_pct", 10.0))),
            warmup_runs=1 if int(data.get("warmup_runs", 1)) > 0 else 0,
        )


@dataclass(frozen=True)
class ToolingSpec:
    """Pinned perf-smoke measurement tooling and workload contract."""

    source_ref: str
    source_commit_sha: str
    source_dirty: bool
    authoritative: bool
    bundle_hash: str
    bundle_file_count: int
    snapshot_relpath: str
    driver_relpath: str
    result_builder_relpath: str
    contract_id: str
    result_schema_version: str
    launch_config_hash: str
    benchmark_contract_hash: str
    task_config_hash: str
    metric_name: str
    metric_path: str
    regression_direction: str
    benchmark_warmup_frames: int
    process_warmup_runs: int
    benchmark_formatter: str = "schema"
    schema_version: int = 1
    tooling_spec_hash: str = ""

    def to_json(self) -> dict[str, Any]:
        """Serialize the tooling contract."""
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any] | None) -> ToolingSpec | None:
        """Build a tooling contract from JSON."""
        if not data:
            return None
        return cls(
            source_ref=str(data["source_ref"]),
            source_commit_sha=str(data["source_commit_sha"]),
            source_dirty=bool(data.get("source_dirty", False)),
            authoritative=bool(data.get("authoritative", data.get("source_ref") != "WORKTREE")),
            bundle_hash=str(data["bundle_hash"]),
            bundle_file_count=int(data["bundle_file_count"]),
            snapshot_relpath=str(data.get("snapshot_relpath", "tooling/perf_smoke_test")),
            driver_relpath=str(data.get("driver_relpath", "perf_runtime.py")),
            result_builder_relpath=str(data.get("result_builder_relpath", "build_bench_result.py")),
            contract_id=str(data["contract_id"]),
            result_schema_version=str(data["result_schema_version"]),
            launch_config_hash=str(data["launch_config_hash"]),
            benchmark_contract_hash=str(data["benchmark_contract_hash"]),
            task_config_hash=str(data["task_config_hash"]),
            metric_name=str(data["metric_name"]),
            metric_path=str(data["metric_path"]),
            regression_direction=str(data["regression_direction"]),
            benchmark_warmup_frames=int(data["benchmark_warmup_frames"]),
            process_warmup_runs=int(data["process_warmup_runs"]),
            benchmark_formatter=str(data.get("benchmark_formatter", "schema")),
            schema_version=int(data.get("schema_version", 1)),
            tooling_spec_hash=str(data.get("tooling_spec_hash", "")),
        )


@dataclass(frozen=True)
class BisectionPlan:
    """Plan for bisecting one regressed IsaacLab task/backend cell."""

    task_id: str
    backend_key: str
    good_ref: str
    bad_ref: str
    gpu_model: str
    runner: RunnerSpec | None = None
    task: TaskSpec = field(default_factory=TaskSpec)
    metric: MetricSpec = field(default_factory=MetricSpec)
    timeout: TimeoutPolicy = field(default_factory=TimeoutPolicy)
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    measurement: MeasurementPolicy = field(default_factory=MeasurementPolicy)
    tooling: ToolingSpec | None = None
    schema_version: int = 3

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
            runner=RunnerSpec.from_json(data.get("runner")),
            task=TaskSpec.from_json(data.get("task")),
            metric=MetricSpec.from_json(data.get("metric")),
            timeout=TimeoutPolicy.from_json(data.get("timeout")),
            retry=RetryPolicy.from_json(data.get("retry")),
            measurement=MeasurementPolicy.from_json(data.get("measurement")),
            tooling=ToolingSpec.from_json(data.get("tooling")),
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
    recovery_events: list[dict[str, Any]] = field(default_factory=list)

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
    metric_name: str | None = None
    metric_unit: str | None = None
    measured_value: float | None = None
    baseline_value: float | None = None
    regression_pct: float | None = None
    baseline_sample_count: int = 0
    threshold_source: str | None = None
    comparison_mode: str | None = None
    note: str | None = None
    command: str | None = None
    attempt_count: int = 1
    attempts: list[dict[str, Any]] = field(default_factory=list)
    timed_out: bool = False
    duration_s: float | None = None
    retry_reason: str | None = None
    final_artifact_dir: str | None = None
    skip_category: str | None = None
    recovery_events: list[dict[str, Any]] = field(default_factory=list)

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
    metric: dict[str, Any] = field(default_factory=dict)
    comparison_mode: str | None = None
    reference_stats: dict[str, Any] = field(default_factory=dict)
    narrowed_interval: dict[str, Any] | None = None
    skipped_commits: list[dict[str, Any]] = field(default_factory=list)
    non_repro: dict[str, Any] | None = None
    # Which pinned runtime components (isaacsim/kit/physx/newton/warp/python) moved
    # across the range and across the culprit commit. A regression can live in a
    # dependency bump rather than IsaacLab source, so this tells the reader which
    # component to investigate next. ``None`` when the stacks could not be resolved.
    stack_diff: dict[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        """Serialize the summary to JSON."""
        return asdict(self)
