# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GPU-free unit tests for the recovery layer and the engine's recovery loop.

Covers the deterministic ``failure -> inspect/retry -> accept`` policy, the knob
mapping, the LLM policy's guardrails/fallback (no network), and the engine wrapper
that retries a measurement through friction before the verdict layer sees it. No
GPU, network, or install is required.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

_GATE_DIR = Path(__file__).resolve().parents[1]
if str(_GATE_DIR) not in sys.path:
    sys.path.insert(0, str(_GATE_DIR))

from bisection import engine  # noqa: E402
from bisection.models import BisectionPlan, MeasurementPolicy, MetricSpec, RunnerSpec  # noqa: E402
from bisection.recovery import (  # noqa: E402
    ACTION_ACCEPT,
    ACTION_RETRY_CLEAR_CACHES,
    ACTION_RETRY_INCREASE_TIMEOUT,
    ACTION_RETRY_PLAIN,
    ACTION_RETRY_REINSTALL,
    DeterministicRecoveryPolicy,
    NoRecoveryPolicy,
    RecoveryContext,
    RecoveryKnobs,
    build_policy,
    knobs_for_action,
)


def _ctx(note: str | None, *, attempt: int = 0, log_tail: str = "", timed_out: bool = False) -> RecoveryContext:
    return RecoveryContext(
        commit_sha="abc123",
        label="candidate",
        run_idx=1,
        attempt=attempt,
        note=note,
        exit_code=None if note == "candidate_timeout" else 1,
        timed_out=timed_out,
        artifact_dir=Path("/tmp/does-not-exist"),
        log_tail=log_tail,
    )


def _ctx_with_env_detail(note: str | None, *, detail: str) -> RecoveryContext:
    ctx = _ctx(note)
    return replace(ctx, env_status=("skip", note.split(":", 1)[1] if note and ":" in note else None, detail))


class TestDeterministicRecoveryPolicy:
    """The model-free friction-recovery decisions."""

    def test_dependency_unavailable_is_accepted_immediately(self) -> None:
        policy = DeterministicRecoveryPolicy()
        decision = policy.decide(_ctx("env_skip:dependency_unavailable"))
        assert decision.action == ACTION_ACCEPT
        assert decision.skip_category == "dependency_unavailable"

    def test_tooling_incompatible_is_accepted_without_retry(self) -> None:
        decision = DeterministicRecoveryPolicy().decide(_ctx("env_skip:perf_smoke_tooling_incompatible"))

        assert decision.action == ACTION_ACCEPT
        assert decision.skip_category == "perf_smoke_tooling_incompatible"

    def test_install_failure_retries_a_reinstall_first(self) -> None:
        decision = DeterministicRecoveryPolicy().decide(_ctx("env_skip:install_failed"))
        assert decision.action == ACTION_RETRY_REINSTALL

    def test_runtime_incompat_clears_caches_then_accepts(self) -> None:
        policy = DeterministicRecoveryPolicy(max_attempts=2)
        first = policy.decide(_ctx("env_skip:runtime_incompatible", attempt=0))
        assert first.action == ACTION_RETRY_CLEAR_CACHES
        second = policy.decide(_ctx("env_skip:runtime_incompatible", attempt=1))
        assert second.action == ACTION_ACCEPT
        assert second.skip_category == "runtime_incompatible"

    def test_log_signature_routes_generic_failure_to_cache_clear(self) -> None:
        decision = DeterministicRecoveryPolicy().decide(
            _ctx("runner_command_failed", log_tail="ModuleNotFoundError: no module named x")
        )
        assert decision.action == ACTION_RETRY_CLEAR_CACHES

    def test_timeout_extends_then_accepts(self) -> None:
        policy = DeterministicRecoveryPolicy(max_attempts=2)
        first = policy.decide(_ctx("candidate_timeout", attempt=0, timed_out=True))
        assert first.action == ACTION_RETRY_INCREASE_TIMEOUT
        second = policy.decide(_ctx("candidate_timeout", attempt=1, timed_out=True))
        assert second.action == ACTION_ACCEPT
        assert second.skip_category == "runtime_incompatible"

    def test_generic_failure_plain_then_clear(self) -> None:
        policy = DeterministicRecoveryPolicy(max_attempts=3)
        assert policy.decide(_ctx("runner_command_failed", attempt=0)).action == ACTION_RETRY_PLAIN
        assert policy.decide(_ctx("runner_command_failed", attempt=1)).action == ACTION_RETRY_CLEAR_CACHES

    def test_budget_exhaustion_accepts_with_category(self) -> None:
        decision = DeterministicRecoveryPolicy(max_attempts=2).decide(_ctx("missing_perf_smoke_test_result", attempt=2))
        assert decision.action == ACTION_ACCEPT
        assert decision.skip_category == "infra"

    @pytest.mark.parametrize(
        ("log_tail", "expected_category"),
        [
            ("some output\nno space left on device\n", "host_resource"),
            ("docker: Cannot connect to the Docker daemon at unix:///var/run/docker.sock.", "docker_unavailable"),
            ("docker: Error response: could not select device driver with capabilities: [[gpu]].", "gpu_unavailable"),
            ("Unable to find image 'isaaclab-bisect:base' locally\ndocker: manifest unknown.", "base_image_missing"),
        ],
    )
    def test_host_blocker_is_accepted_without_retry(self, log_tail: str, expected_category: str) -> None:
        """Disk/Docker/GPU/image failures live on this machine, so retrying cannot help.

        The policy must accept on the first attempt (no wasted retries) and label the
        skip with the specific host category rather than a generic ``infra`` skip.
        """
        decision = DeterministicRecoveryPolicy().decide(_ctx("runner_command_failed", attempt=0, log_tail=log_tail))
        assert decision.action == ACTION_ACCEPT
        assert decision.skip_category == expected_category

    def test_host_blocker_in_env_skip_detail_is_accepted_without_retry(self) -> None:
        """Install skip details can carry the actionable disk-full signal.

        The runner writes disk/extraction failures into ``bisect_env.skip_detail``. The
        recovery policy must inspect that detail too; otherwise it treats disk-full as a
        retryable install failure and wastes reinstall attempts before accepting.
        """
        decision = DeterministicRecoveryPolicy().decide(
            _ctx_with_env_detail("env_skip:install_failed", detail="failed to write: No space left on device")
        )
        assert decision.action == ACTION_ACCEPT
        assert decision.skip_category == "host_resource"

    def test_source_checkout_failure_keeps_retrying_with_specific_label(self) -> None:
        """A failed clone/checkout is recoverable via a fresh source dir, so it stays retryable.

        Unlike host blockers it should not short-circuit to accept on the first attempt,
        but once the budget is exhausted it must be labeled ``source_checkout_failed`` (not ``infra``).
        """
        tail = "fatal: reference is not a tree: d1cb8e887"
        first = DeterministicRecoveryPolicy(max_attempts=2).decide(
            _ctx("runner_command_failed", attempt=0, log_tail=tail)
        )
        assert first.action == ACTION_RETRY_PLAIN
        exhausted = DeterministicRecoveryPolicy(max_attempts=2).decide(
            _ctx("runner_command_failed", attempt=2, log_tail=tail)
        )
        assert exhausted.action == ACTION_ACCEPT
        assert exhausted.skip_category == "source_checkout_failed"


class TestNoRecoveryPolicy:
    """--recovery none accepts the first outcome verbatim."""

    def test_accepts_first_outcome(self) -> None:
        decision = NoRecoveryPolicy().decide(_ctx("env_skip:install_failed"))
        assert decision.action == ACTION_ACCEPT
        assert decision.skip_category == "install_failed"


class TestKnobs:
    """Action-to-knob translation accumulates monotonically."""

    def test_clear_caches_sets_flag(self) -> None:
        knobs = knobs_for_action(ACTION_RETRY_CLEAR_CACHES, RecoveryKnobs())
        assert knobs.clear_caches is True
        assert knobs.runner_args() == ["--clear_caches"]

    def test_reinstall_implies_clear_caches(self) -> None:
        knobs = knobs_for_action(ACTION_RETRY_REINSTALL, RecoveryKnobs())
        assert knobs.force_reinstall is True
        assert knobs.clear_caches is True
        assert set(knobs.runner_args()) == {"--clear_caches", "--force_reinstall"}

    def test_increase_timeout_sets_extra_and_keeps_prior_flags(self) -> None:
        prior = RecoveryKnobs(clear_caches=True)
        knobs = knobs_for_action(ACTION_RETRY_INCREASE_TIMEOUT, prior)
        assert knobs.extra_timeout_s and knobs.extra_timeout_s > 0
        assert knobs.clear_caches is True

    def test_build_policy_factory(self) -> None:
        assert isinstance(build_policy("none"), NoRecoveryPolicy)
        assert isinstance(build_policy("deterministic"), DeterministicRecoveryPolicy)
        with pytest.raises(ValueError):
            build_policy("llm")


def _plan() -> BisectionPlan:
    return BisectionPlan(
        task_id="Isaac-Cartpole-Direct",
        backend_key="newton",
        good_ref="good",
        bad_ref="bad",
        gpu_model="L40S",
        runner=RunnerSpec(mode="synthetic"),
        metric=MetricSpec(),
        measurement=MeasurementPolicy(),
    )


class _FakeMeasurement:
    """Scripts ``_run_single_measurement`` results per recovery attempt."""

    def __init__(self, sequence: list[tuple[str | None, float | None]]) -> None:
        self.sequence = sequence
        self.calls: list[dict] = []

    def __call__(
        self, plan, output_dir, *, commit_sha, label, run_idx, knobs=None, recovery_attempt=0, probe_policy=None
    ):
        self.calls.append({"recovery_attempt": recovery_attempt, "knobs": knobs})
        note, metric = self.sequence[recovery_attempt]
        artifact_dir = output_dir / f"m_{recovery_attempt}"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        attempt = engine.CandidateAttempt(
            attempt=run_idx,
            artifact_dir=str(artifact_dir),
            command="fake",
            command_exit_code=0 if metric is not None else 1,
            note=note,
            timed_out=note == "candidate_timeout",
        )
        return attempt, metric


class TestEngineRecoveryLoop:
    """The engine retries a measurement through friction before verdicting."""

    def test_recovers_then_succeeds(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _FakeMeasurement([("runner_command_failed", None), (None, 123.0)])
        monkeypatch.setattr(engine, "_run_single_measurement", fake)
        attempt, metric = engine._measure_with_recovery(
            _plan(),
            tmp_path,
            commit_sha="abc123",
            label="candidate",
            run_idx=1,
            policy=DeterministicRecoveryPolicy(max_attempts=2),
        )
        assert metric == 123.0
        assert len(fake.calls) == 2
        assert fake.calls[0]["knobs"] is None
        assert len(attempt.recovery_events) == 1
        assert attempt.recovery_events[0]["decision"] == ACTION_RETRY_PLAIN
        audit = (tmp_path / "audit_log.jsonl").read_text(encoding="utf-8").strip().splitlines()
        events = [json.loads(line)["event"] for line in audit]
        assert "measurement" in events
        assert "recovery_decision" in events
        attempt_summary = json.loads((tmp_path / "m_1" / "attempt_summary.json").read_text(encoding="utf-8"))
        assert attempt_summary["metric_value"] == 123.0
        assert attempt_summary["recovery_events"][0]["decision"] == ACTION_RETRY_PLAIN

    def test_accepts_persistent_friction_as_skip(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _FakeMeasurement([("env_skip:runtime_incompatible", None)] * 5)
        monkeypatch.setattr(engine, "_run_single_measurement", fake)
        attempt, metric = engine._measure_with_recovery(
            _plan(),
            tmp_path,
            commit_sha="abc123",
            label="candidate",
            run_idx=1,
            policy=DeterministicRecoveryPolicy(max_attempts=1),
        )
        assert metric is None
        assert attempt.note == "env_skip:runtime_incompatible"
        assert len(fake.calls) == 2  # one try + one clear-cache retry, then accept
        attempt_summary = json.loads((tmp_path / "m_1" / "attempt_summary.json").read_text(encoding="utf-8"))
        assert attempt_summary["note"] == "env_skip:runtime_incompatible"
        assert attempt_summary["recovery_events"][-1]["skip_category"] == "runtime_incompatible"

    def test_no_recovery_accepts_immediately(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _FakeMeasurement([("runner_command_failed", None)])
        monkeypatch.setattr(engine, "_run_single_measurement", fake)
        _, metric = engine._measure_with_recovery(
            _plan(), tmp_path, commit_sha="abc123", label="candidate", run_idx=1, policy=NoRecoveryPolicy()
        )
        assert metric is None
        assert len(fake.calls) == 1


class _LabelRecordingMeasurement:
    """Records the label/run_idx of each measurement and returns a fixed metric."""

    def __init__(self, metric: float | None = 100.0) -> None:
        self.metric = metric
        self.calls: list[tuple[str, int]] = []

    def __call__(
        self, plan, output_dir, *, commit_sha, label, run_idx, knobs=None, recovery_attempt=0, probe_policy=None
    ):
        self.calls.append((label, run_idx))
        artifact_dir = output_dir / label / f"run_{run_idx}"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        attempt = engine.CandidateAttempt(
            attempt=run_idx,
            artifact_dir=str(artifact_dir),
            command="fake",
            command_exit_code=0 if self.metric is not None else 1,
            note=None if self.metric is not None else "runner_command_failed",
            timed_out=False,
        )
        return attempt, self.metric


class TestWarmupMeasurements:
    """Process-level warmup runs are recorded but excluded from measurement stats."""

    def _plan_with_warmup(self, warmup_runs: int) -> BisectionPlan:
        base = _plan()
        return replace(base, measurement=replace(base.measurement, warmup_runs=warmup_runs))

    def test_zero_warmup_runs_no_op(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """With warmup_runs=0 explicitly selected, no warmup or audit event is produced."""
        fake = _LabelRecordingMeasurement()
        monkeypatch.setattr(engine, "_run_single_measurement", fake)
        engine._run_warmup_measurements(
            self._plan_with_warmup(0),
            tmp_path,
            commit_sha="abc123",
            label="candidate",
            policy=DeterministicRecoveryPolicy(),
        )
        assert fake.calls == []
        assert not (tmp_path / "audit_log.jsonl").exists()

    def test_warmup_uses_warmup_label_and_audits_exclusion(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Warmup attempts run under a ``<label>_warmup`` label and are audited as excluded.

        The distinct label routes their artifacts to a separate tree so they never enter
        the measured attempts, and each warmup logs ``excluded_from_stats: true`` for audit.
        """
        fake = _LabelRecordingMeasurement()
        monkeypatch.setattr(engine, "_run_single_measurement", fake)
        engine._run_warmup_measurements(
            self._plan_with_warmup(2),
            tmp_path,
            commit_sha="abc123",
            label="candidate",
            policy=DeterministicRecoveryPolicy(),
        )
        assert fake.calls == [("candidate_warmup", 1)]
        audit = [json.loads(line) for line in (tmp_path / "audit_log.jsonl").read_text().strip().splitlines()]
        warmup_events = [e for e in audit if e["event"] == "warmup_measurement"]
        assert len(warmup_events) == 1
        assert all(e["excluded_from_stats"] is True and e["label"] == "candidate" for e in warmup_events)

        engine._run_warmup_measurements(
            self._plan_with_warmup(1),
            tmp_path,
            commit_sha="abc123",
            label="good_ref",
            policy=DeterministicRecoveryPolicy(),
        )
        assert fake.calls == [("candidate_warmup", 1)]
        audit = [json.loads(line) for line in (tmp_path / "audit_log.jsonl").read_text().strip().splitlines()]
        assert any(e["event"] == "warmup_reused" and e["label"] == "good_ref" for e in audit)

    def test_reference_measurement_runs_warmup_first_and_excludes_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reference measurement runs warmup first, then only measured runs feed the summary.

        The warmup label must precede the measured ``good_ref`` runs, and the summary's
        sample count must equal the measured runs only (warmup excluded).
        """
        fake = _LabelRecordingMeasurement(metric=100.0)
        monkeypatch.setattr(engine, "_run_single_measurement", fake)
        plan = self._plan_with_warmup(1)
        summary, attempts, note = engine._measure_reference_commit(
            plan,
            tmp_path,
            commit_sha="abc123",
            label="good_ref",
            policy=DeterministicRecoveryPolicy(),
        )
        assert note is None
        labels = [label for label, _ in fake.calls]
        assert labels[0] == "good_ref_warmup"
        assert labels[1:] == ["good_ref"] * plan.measurement.reference_runs
        # Warmup attempts are not part of the returned measured attempts nor the stats.
        assert all(a.artifact_dir.endswith(tuple(f"good_ref/run_{i}" for i in range(1, 8))) for a in attempts)
        assert summary is not None and summary.sample_count == plan.measurement.reference_runs


class TestRecommendedCandidateRuns:
    """Variance-driven candidate run recommendation from reference noise."""

    def test_quiet_reference_keeps_floor(self) -> None:
        plan = _plan()
        assert engine._recommended_candidate_runs(0.1, plan) == plan.measurement.candidate_runs

    def test_noisy_reference_scales_to_max(self) -> None:
        plan = _plan()
        big_noise = 100.0 * plan.measurement.gray_zone_pct
        assert engine._recommended_candidate_runs(big_noise, plan) == plan.measurement.max_candidate_runs


class TestLLMRecoveryPolicyGuardrails:
    """The LLM policy stays bounded and falls back deterministically (no network)."""

    def _policy(self):
        from bisection.recovery_llm import LLMRecoveryPolicy

        return LLMRecoveryPolicy(model="fake-model", max_attempts=2)

    def test_parses_valid_model_decision(self, monkeypatch: pytest.MonkeyPatch) -> None:
        policy = self._policy()
        monkeypatch.setattr(
            policy._client, "complete", lambda system, user: '{"action": "retry_clear_caches", "reason": "stale cache"}'
        )
        decision = policy.decide(_ctx("runner_command_failed"))
        assert decision.action == ACTION_RETRY_CLEAR_CACHES

    def test_falls_back_when_model_unreachable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from bisection.llm_client import LLMError

        policy = self._policy()

        def _boom(system, user):
            raise LLMError("no endpoint")

        monkeypatch.setattr(policy._client, "complete", _boom)
        decision = policy.decide(_ctx("env_skip:install_failed"))
        assert decision.action == ACTION_RETRY_REINSTALL  # deterministic fallback

    def test_invalid_action_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        policy = self._policy()
        monkeypatch.setattr(policy._client, "complete", lambda system, user: '{"action": "delete_repo"}')
        decision = policy.decide(_ctx("candidate_timeout", timed_out=True))
        assert decision.action == ACTION_RETRY_INCREASE_TIMEOUT  # deterministic fallback

    def test_budget_exhaustion_skips_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        policy = self._policy()

        def _should_not_call(system, user):
            raise AssertionError("model should not be consulted past the budget")

        monkeypatch.setattr(policy._client, "complete", _should_not_call)
        decision = policy.decide(_ctx("runner_command_failed", attempt=2))
        assert decision.action == ACTION_ACCEPT
