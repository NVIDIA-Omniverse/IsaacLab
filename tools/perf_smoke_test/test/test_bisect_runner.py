# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GPU-free unit tests for the local-reconstruct runner mode plumbing.

Covers the ``bisect_env.json`` sidecar contract, argument parsing for the new
mode, and the engine passing a shared ``--env_cache_dir`` for ``local-reconstruct``.
No GPU, network, or install is required (the actual reconstruction is exercised by
a separate opt-in smoke test).
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import pytest

_GATE_DIR = Path(__file__).resolve().parents[1]
if str(_GATE_DIR) not in sys.path:
    sys.path.insert(0, str(_GATE_DIR))

import bisect_single_commit_runner as runner  # noqa: E402
from bisection import engine  # noqa: E402
from bisection.base_image_repair import validate_apt_packages, write_repair_dockerfile  # noqa: E402
from bisection.engine import _run_command, _run_single_measurement, format_runner_command  # noqa: E402
from bisection.env_setup import DEFAULT_INSTALL_SCOPE, EnvSkip  # noqa: E402
from bisection.models import BisectionPlan, MetricSpec, RunnerSpec, TaskSpec  # noqa: E402
from bisection.probe import (  # noqa: E402
    PROBE_ACTION_PLAN_ISSUE,
    PROBE_ACTION_READY,
    PROBE_ACTION_REPAIR_BASE_IMAGE,
    PROBE_ACTION_RUN_DEBUG_COMMAND,
    LLMProbePolicy,
    ProbeContext,
    ProbeDecision,
)


@dataclass
class _FakeStack:
    commit_sha: str = "abc123def456789"
    stack_hash: str = "deadbeef"
    isaacsim: str | None = "6.0.0-dev2"
    python_version: str = "3.12"
    python_requires: str | None = ">=3.12"


@dataclass
class _FakeHandle:
    env_dir: str = "/tmp/env-cache/envs/abc123def456"
    reused: bool = False


class TestBisectEnvSidecar:
    """The bisect_env.json sidecar contract the engine consumes."""

    def _read(self, tmp_path: Path) -> dict:
        return json.loads((tmp_path / "bisect_env.json").read_text(encoding="utf-8"))

    def test_non_reconstruct_mode_records_ok_with_null_env_dir(self, tmp_path: Path) -> None:
        runner._write_bisect_env(tmp_path, stack=_FakeStack(), mode="synthetic", status="ok")
        payload = self._read(tmp_path)
        assert payload["status"] == "ok"
        assert payload["mode"] == "synthetic"
        assert payload["env_dir"] is None
        assert payload["stack_hash"] == "deadbeef"
        assert payload["isaacsim_version"] == "6.0.0-dev2"
        assert payload["skip_category"] is None

    def test_reconstruct_ok_records_env_dir(self, tmp_path: Path) -> None:
        runner._write_bisect_env(
            tmp_path, stack=_FakeStack(), mode="local-reconstruct", status="ok", env_handle=_FakeHandle()
        )
        payload = self._read(tmp_path)
        assert payload["status"] == "ok"
        assert payload["env_dir"] == "/tmp/env-cache/envs/abc123def456"
        assert payload["env_reused"] is False

    def test_skip_records_category_and_detail(self, tmp_path: Path) -> None:
        skip = EnvSkip("dependency_unavailable", "isaacsim==6.0.0-dev2 not found on index")
        runner._write_bisect_env(tmp_path, stack=_FakeStack(), mode="local-reconstruct", status="skip", skip=skip)
        payload = self._read(tmp_path)
        assert payload["status"] == "skip"
        assert payload["skip_category"] == "dependency_unavailable"
        assert "not found" in payload["skip_detail"]
        assert payload["env_dir"] is None


def test_clear_previous_attempt_outputs_prevents_stale_metric_reuse(tmp_path: Path) -> None:
    for name in (
        "benchmark.log",
        "bisect_env.json",
        "perf_smoke_test_info.json",
        "perf_smoke_test_result.json",
        "tooling.json",
        "tooling_verification.json",
        "tooling_capability.json",
        "benchmark_runtime_task_1.json",
    ):
        (tmp_path / name).write_text("stale", encoding="utf-8")
    (tmp_path / "keep.txt").write_text("keep", encoding="utf-8")

    runner._clear_previous_attempt_outputs(tmp_path)

    assert sorted(path.name for path in tmp_path.iterdir()) == ["keep.txt"]


class TestRunnerArgParsing:
    """The new mode and its flags parse as expected."""

    def _parse(self, argv: list[str], monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(sys, "argv", ["bisect_single_commit_runner.py", *argv])
        return runner._parse_args()

    def _base_argv(self) -> list[str]:
        return [
            "--commit",
            "HEAD",
            "--task_id",
            "Isaac-Cartpole-Direct",
            "--backend_key",
            "newton",
            "--artifact_dir",
            "/tmp/art",
        ]

    def test_local_reconstruct_mode_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        args = self._parse([*self._base_argv(), "--mode", "local-reconstruct"], monkeypatch)
        assert args.mode == "local-reconstruct"

    def test_install_scope_defaults_to_module_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        args = self._parse([*self._base_argv(), "--mode", "local-reconstruct"], monkeypatch)
        assert args.install_scope == DEFAULT_INSTALL_SCOPE
        assert args.env_cache_dir is None

    def test_env_cache_dir_parsed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        args = self._parse(
            [*self._base_argv(), "--mode", "local-reconstruct", "--env_cache_dir", "/tmp/shared-env"], monkeypatch
        )
        assert args.env_cache_dir == Path("/tmp/shared-env")

    def test_docker_reconstruct_mode_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        args = self._parse(
            [*self._base_argv(), "--mode", "docker-reconstruct", "--image", "isaaclab-bisect:base"], monkeypatch
        )
        assert args.mode == "docker-reconstruct"
        assert args.image == "isaaclab-bisect:base"
        assert args.harness_root is None

    def test_invalid_mode_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(SystemExit):
            self._parse([*self._base_argv(), "--mode", "bogus"], monkeypatch)


class TestEngineEnvCacheDir:
    """The engine shares one run-scoped env cache across candidates."""

    def _plan(self, mode: str) -> BisectionPlan:
        return BisectionPlan(
            task_id="Isaac-Cartpole-Direct",
            backend_key="newton",
            good_ref="good",
            bad_ref="bad",
            gpu_model="L40S",
            runner=RunnerSpec(mode=mode),
            metric=MetricSpec(),
        )

    def test_local_reconstruct_passes_shared_env_cache_dir(self, tmp_path: Path) -> None:
        cmd = format_runner_command(self._plan("local-reconstruct"), tmp_path, "abc123", tmp_path / "art")
        assert "--env_cache_dir" in cmd
        assert str(tmp_path / "env-cache") in cmd
        assert cmd[cmd.index("--env_cache_dir") + 1] == str(tmp_path / "env-cache")

    def test_synthetic_does_not_pass_env_cache_dir(self, tmp_path: Path) -> None:
        cmd = format_runner_command(self._plan("synthetic"), tmp_path, "abc123", tmp_path / "art")
        assert "--env_cache_dir" not in cmd

    def test_docker_reconstruct_passes_env_cache_dir(self, tmp_path: Path) -> None:
        cmd = format_runner_command(self._plan("docker-reconstruct"), tmp_path, "abc123", tmp_path / "art")
        assert "--env_cache_dir" in cmd
        assert cmd[cmd.index("--env_cache_dir") + 1] == str(tmp_path / "env-cache")

    def test_docker_reconstruct_forwards_image(self, tmp_path: Path) -> None:
        plan = BisectionPlan(
            task_id="Isaac-Cartpole-Direct",
            backend_key="newton",
            good_ref="good",
            bad_ref="bad",
            gpu_model="L40S",
            runner=RunnerSpec(mode="docker-reconstruct", image="isaaclab-bisect:base"),
            metric=MetricSpec(),
        )
        cmd = format_runner_command(plan, tmp_path, "abc123", tmp_path / "art")
        assert "--image" in cmd
        assert cmd[cmd.index("--image") + 1] == "isaaclab-bisect:base"


class TestSyntheticGroundTruthOverride:
    """Synthetic mode can rehearse the search over a real range with a chosen regression point."""

    def _plan(self, runner: RunnerSpec, *, regression_direction: str = "decrease") -> BisectionPlan:
        return BisectionPlan(
            task_id="Isaac-Dexsuite-Kuka-Allegro-Lift-v0",
            backend_key="newton_newton_renderer",
            good_ref="good",
            bad_ref="bad",
            gpu_model="L40S",
            runner=runner,
            metric=MetricSpec(result_path="raw_fps_mean", regression_direction=regression_direction),
        )

    def test_default_uses_bad_ref_and_direction_derived_values(self, tmp_path: Path) -> None:
        cmd = format_runner_command(self._plan(RunnerSpec(mode="synthetic")), tmp_path, "abc123", tmp_path / "art")
        assert cmd[cmd.index("--first_bad_ref") + 1] == "bad"
        assert cmd[cmd.index("--synthetic_good_value") + 1] == "1000.0"
        assert cmd[cmd.index("--synthetic_bad_value") + 1] == "500.0"

    def test_override_first_bad_ref_wins_over_plan_bad_ref(self, tmp_path: Path) -> None:
        runner_spec = RunnerSpec(mode="synthetic", synthetic_first_bad_ref="mid-commit-sha")
        cmd = format_runner_command(self._plan(runner_spec), tmp_path, "abc123", tmp_path / "art")
        assert cmd[cmd.index("--first_bad_ref") + 1] == "mid-commit-sha"

    def test_override_good_and_bad_values_win_over_direction_defaults(self, tmp_path: Path) -> None:
        runner_spec = RunnerSpec(mode="synthetic", synthetic_good_value=0.754, synthetic_bad_value=0.006)
        cmd = format_runner_command(self._plan(runner_spec), tmp_path, "abc123", tmp_path / "art")
        assert cmd[cmd.index("--synthetic_good_value") + 1] == "0.754"
        assert cmd[cmd.index("--synthetic_bad_value") + 1] == "0.006"

    def test_runner_spec_synthetic_overrides_round_trip_through_json(self) -> None:
        runner_spec = RunnerSpec(
            mode="synthetic",
            synthetic_first_bad_ref="mid-commit-sha",
            synthetic_good_value=0.754,
            synthetic_bad_value=0.006,
        )
        restored = RunnerSpec.from_json(json.loads(json.dumps(asdict(runner_spec))))
        assert restored == runner_spec


class TestDockerReconstructCommand:
    """The docker-reconstruct command builder mounts and configures the container."""

    def _cmd(self, tmp_path: Path, extra: list[str] | None = None) -> list[str]:
        return runner._docker_reconstruct_command(
            image="isaaclab-bisect:base",
            commit_sha="abc123def456",
            task_id="Isaac-Cartpole-Direct",
            backend_key="newton",
            harness_root=tmp_path / "harness",
            tooling_root=tmp_path / "tooling",
            artifact_dir=tmp_path / "art",
            source_dir=tmp_path / "candidate",
            env_cache_dir=tmp_path / "env-cache",
            jit_cache_root=tmp_path / "jit-cache",
            kit_cache_root=tmp_path / "kit-cache",
            extra_runner_args=extra or [],
            container_name="perf-bisect-recon-abc123def456-x",
        )

    def test_mounts_harness_read_only_and_writable_dirs(self, tmp_path: Path) -> None:
        cmd = self._cmd(tmp_path)
        joined = " ".join(cmd)
        assert f"{tmp_path / 'harness'}:/harness:ro" in cmd
        assert f"{tmp_path / 'tooling'}:/tooling:ro" in cmd
        assert f"{tmp_path / 'art'}:/artifacts" in cmd
        assert f"{tmp_path / 'candidate'}:/candidate" in cmd
        assert f"{tmp_path / 'env-cache'}:/env-cache" in cmd
        assert f"{tmp_path / 'jit-cache'}:/cache/jit-root" in cmd
        assert f"{tmp_path / 'kit-cache'}:/cache/kit-root" in cmd
        assert "--gpus" in cmd and "all" in cmd
        assert cmd[-1] == "isaaclab-bisect:base"  # image is the final positional arg
        assert "COMMIT_SHA=abc123def456" in joined
        assert "TASK_ID=Isaac-Cartpole-Direct" in joined
        assert "BACKEND=newton" in joined

    def test_extra_runner_args_are_passed_via_env(self, tmp_path: Path) -> None:
        cmd = self._cmd(tmp_path, extra=["--clear_caches", "--install_scope", "newton,isaacsim"])
        env_extra = next(part for part in cmd if part.startswith("EXTRA_RUNNER_ARGS="))
        assert "--clear_caches" in env_extra
        assert "newton,isaacsim" in env_extra

    def test_multiword_arg_is_quoted_and_round_trips(self, tmp_path: Path) -> None:
        """A value with a space (e.g. ``--gpu_model 'NVIDIA L40S'``) must survive as one token.

        The entrypoint re-parses ``EXTRA_RUNNER_ARGS`` with ``eval set --``, so the builder
        has to shell-quote each token. Without quoting the space would fracture the value and
        the inner runner would reject the trailing fragment (``unrecognized arguments: L40S``).
        """
        cmd = self._cmd(tmp_path, extra=["--gpu_model", "NVIDIA L40S"])
        env_extra = next(part for part in cmd if part.startswith("EXTRA_RUNNER_ARGS="))
        payload = env_extra[len("EXTRA_RUNNER_ARGS=") :]
        assert shlex.split(payload) == ["--gpu_model", "NVIDIA L40S"]

    def test_worktree_git_metadata_is_mounted_at_absolute_gitdir(self, tmp_path: Path) -> None:
        """Git worktrees need their parent repo ``.git`` mounted for in-container git.

        A worktree's ``.git`` is a file pointing to the parent repository's
        ``.git/worktrees/<name>`` path. Without mounting the parent ``.git`` at that
        exact absolute path, the container's `/harness` checkout is not a usable git
        repo and stack resolution fails before environment reconstruction starts.
        """
        parent_git = tmp_path / "repo" / ".git"
        worktree_git_dir = parent_git / "worktrees" / "bisection-post-migration"
        worktree_git_dir.mkdir(parents=True)
        harness = tmp_path / "harness"
        harness.mkdir()
        (harness / ".git").write_text(f"gitdir: {worktree_git_dir}\n", encoding="utf-8")

        cmd = runner._docker_reconstruct_command(
            image="isaaclab-bisect:base",
            commit_sha="abc123def456",
            task_id="Isaac-Cartpole-Direct",
            backend_key="newton",
            harness_root=harness,
            tooling_root=tmp_path / "tooling",
            artifact_dir=tmp_path / "art",
            source_dir=tmp_path / "candidate",
            env_cache_dir=tmp_path / "env-cache",
            jit_cache_root=tmp_path / "jit-cache",
            kit_cache_root=tmp_path / "kit-cache",
            extra_runner_args=[],
            container_name="perf-bisect-recon-abc123def456-x",
        )

        assert f"{parent_git}:{parent_git}:ro" in cmd


def test_cache_directories_are_scoped_by_component_stack(tmp_path: Path) -> None:
    args = argparse.Namespace(jit_cache=tmp_path / "jit", kit_cache=tmp_path / "kit")

    jit_cache, kit_cache = runner._stack_cache_dirs(args, tmp_path, _FakeStack(stack_hash="stack-123"))

    assert jit_cache == tmp_path / "jit" / "stack-123"
    assert kit_cache == tmp_path / "kit" / "stack-123"


def test_tooling_incompatible_commit_stops_search_instead_of_becoming_hole() -> None:
    outcome = engine._bisect_search(
        ["candidate", "bad"],
        "good",
        "bad",
        lambda idx: ("SKIP", "env_skip:perf_smoke_tooling_incompatible"),
    )

    assert outcome["status"] == "unsupported_tooling_contract"
    assert outcome["reason"] == "perf_smoke_tooling_incompatible"
    assert outcome["suspected_first_bad_commit"] is None
    assert outcome["skipped_commits"][0]["commit_sha"] == "candidate"


class TestClearStaleGitLocks:
    """Stale-lock cleanup keeps a reused source clone checkout-able after interruptions."""

    def test_removes_stale_index_lock(self, tmp_path: Path) -> None:
        """A leftover ``.git/index.lock`` (from a killed git process) must be deleted.

        docker-reconstruct reuses one clone across a commit's runs; an orphaned lock
        would otherwise fail every later checkout with exit 128 and be misread as an
        environment skip.
        """
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        lock = git_dir / "index.lock"
        lock.write_text("stale")
        runner._clear_stale_git_locks(tmp_path)
        assert not lock.exists()

    def test_noop_without_git_dir(self, tmp_path: Path) -> None:
        """A source dir without ``.git`` (fresh clone target) is left untouched."""
        runner._clear_stale_git_locks(tmp_path)  # must not raise
        assert list(tmp_path.iterdir()) == []


class TestPrepareSourceClone:
    """Self-healing source checkout lifecycle: reset+re-clone on failure, skip if terminal."""

    def test_reset_removes_dir_and_tolerates_missing(self, tmp_path: Path) -> None:
        """``_reset_source_dir`` clears a populated dir and is a no-op when it is absent."""
        src = tmp_path / "candidate-source"
        (src / ".git").mkdir(parents=True)
        (src / ".git" / "objects").mkdir()
        runner._reset_source_dir(src)
        assert not src.exists()
        runner._reset_source_dir(src)  # already gone: must not raise

    def test_recovers_by_resetting_and_recloning_once(self, tmp_path: Path, monkeypatch) -> None:
        """A first checkout failure triggers exactly one reset + re-clone, then succeeds.

        Simulates the real interruption case (a partial/corrupt clone from a killed
        prior run): the first materialize raises, the harness resets the dir, and the
        second materialize succeeds, so no skip is surfaced.
        """
        src = tmp_path / "candidate-source"
        src.mkdir()
        calls: list[str] = []

        def fake_materialize(source_dir: Path, commit_sha: str) -> None:
            calls.append(commit_sha)
            if len(calls) == 1:
                raise RuntimeError("command failed (exit 128): git checkout")

        reset_calls: list[Path] = []
        monkeypatch.setattr(runner, "_materialize_source_clone", fake_materialize)
        monkeypatch.setattr(runner, "_reset_source_dir", lambda p: reset_calls.append(p))

        runner._prepare_source_clone(src, "d1cb8e887")  # must not raise
        assert len(calls) == 2
        assert reset_calls == [src]

    def test_terminal_failure_raises_source_checkout_skip(self, tmp_path: Path, monkeypatch) -> None:
        """When even a fresh re-clone fails, the commit is unevaluable -> EnvSkip.

        An unreachable SHA fails both materialize attempts; the runner must raise a
        ``source_checkout_failed`` skip (not a bare RuntimeError) so the engine records
        an environment skip rather than a benchmark crash.
        """
        src = tmp_path / "candidate-source"
        src.mkdir()

        def always_fail(source_dir: Path, commit_sha: str) -> None:
            raise RuntimeError("fatal: reference is not a tree: deadbeef")

        monkeypatch.setattr(runner, "_materialize_source_clone", always_fail)
        monkeypatch.setattr(runner, "_reset_source_dir", lambda p: None)

        with pytest.raises(EnvSkip) as excinfo:
            runner._prepare_source_clone(src, "deadbeef")
        assert excinfo.value.category == "source_checkout_failed"

    def test_record_skip_writes_bisect_env_and_returns_false(self, tmp_path: Path, monkeypatch) -> None:
        """A terminal checkout skip is recorded as a clean ``bisect_env.json`` skip.

        The wrapper must write the skip sidecar (so the engine reads
        ``env_skip:source_checkout_failed``) and return False so the caller exits 0
        without producing a perf result.
        """

        def raise_skip(source_dir: Path, commit_sha: str) -> None:
            raise EnvSkip("source_checkout_failed", "unreachable commit")

        monkeypatch.setattr(runner, "_prepare_source_clone", raise_skip)
        stack = _FakeStack(commit_sha="deadbeefcafe0001")
        ok = runner._prepare_source_or_record_skip(
            tmp_path / "candidate-source",
            stack.commit_sha,
            artifact_dir=tmp_path,
            stack=stack,
            mode="docker-reconstruct",
        )
        assert ok is False
        payload = json.loads((tmp_path / "bisect_env.json").read_text())
        assert payload["status"] == "skip"
        assert payload["skip_category"] == "source_checkout_failed"


class TestDockerReconstructExtraArgs:
    """The inner-runner flags forwarded into the container."""

    def _args(self, **overrides):
        base = dict(
            install_scope=DEFAULT_INSTALL_SCOPE,
            gpu_model="L40S",
            tooling_spec_hash="spec",
            tooling_bundle_hash="bundle",
            tooling_contract_id="contract",
            clear_caches=False,
            force_reinstall=False,
            num_envs=None,
            num_frames=None,
            warmup_frames=None,
            seed=None,
            camera_resolution=None,
            timeout_minutes=None,
            hydra_arg=[],
        )
        base.update(overrides)
        return argparse.Namespace(**base)

    def test_defaults_forward_scope_and_gpu(self) -> None:
        extra = runner._docker_reconstruct_extra_args(self._args())
        assert "--install_scope" in extra
        assert DEFAULT_INSTALL_SCOPE in extra
        assert "--clear_caches" not in extra
        assert "--force_reinstall" not in extra

    def test_recovery_knobs_forwarded_when_set(self) -> None:
        extra = runner._docker_reconstruct_extra_args(self._args(clear_caches=True, force_reinstall=True))
        assert "--clear_caches" in extra
        assert "--force_reinstall" in extra

    def test_inline_task_fields_forwarded(self) -> None:
        extra = runner._docker_reconstruct_extra_args(
            self._args(
                num_envs=8,
                num_frames=150,
                seed=7,
                camera_resolution=[64, 48],
                timeout_minutes=25,
                hydra_arg=["presets=cube,newton,rgb64", "env.foo=1"],
            )
        )
        assert extra[extra.index("--num_envs") + 1] == "8"
        assert extra[extra.index("--num_frames") + 1] == "150"
        assert extra[extra.index("--seed") + 1] == "7"
        cam_idx = extra.index("--camera_resolution")
        assert extra[cam_idx + 1 : cam_idx + 3] == ["64", "48"]
        assert extra[extra.index("--timeout_minutes") + 1] == "25"
        assert extra.count("--hydra_arg") == 2
        assert "presets=cube,newton,rgb64" in extra
        assert "env.foo=1" in extra


class TestBuildTask:
    """Option B task resolution: inline-wins, registry-fallback, error-if-neither."""

    def _args(self, **overrides):
        base = dict(
            task_id="Isaac-Cartpole-Direct",
            backend_key="newton",
            tasks_json=None,
            num_envs=None,
            num_frames=None,
            warmup_frames=None,
            seed=None,
            camera_resolution=None,
            timeout_minutes=None,
        )
        base.update(overrides)
        return argparse.Namespace(**base)

    def test_registered_task_used_without_inline_fields(self) -> None:
        task = runner._build_task(self._args())
        assert task.task_id == "Isaac-Cartpole-Direct"
        assert task.physics_backend == "newton"
        assert task.num_envs == 4096  # from tasks.json

    def test_inline_fields_override_registered_task(self) -> None:
        task = runner._build_task(self._args(num_envs=16, num_frames=120, seed=9, timeout_minutes=8))
        assert task.num_envs == 16
        assert task.num_frames == 120
        assert task.seed == 9
        assert task.timeout_minutes == 8

    def test_unregistered_task_built_inline(self) -> None:
        task = runner._build_task(self._args(task_id="Not-A-Registered-Task-v0", backend_key="newton", num_envs=4))
        assert task.task_id == "Not-A-Registered-Task-v0"
        assert task.physics_backend == "newton"
        assert task.num_envs == 4
        assert task.num_frames == 300  # inline default
        assert task.seed == 42  # inline default
        assert task.timeout_minutes == 30  # inline default
        assert task.preset == "inline"

    def test_unregistered_task_without_num_envs_errors(self) -> None:
        with pytest.raises(SystemExit):
            runner._build_task(self._args(task_id="Not-A-Registered-Task-v0", backend_key="newton"))

    def test_unregistered_render_backend_parsed_inline(self) -> None:
        task = runner._build_task(self._args(task_id="Custom-Cam-v0", backend_key="newton_newton_renderer", num_envs=2))
        assert task.physics_backend == "newton"
        assert task.render_backend == "newton_renderer"

    def test_registered_task_inherits_warmup_from_registry(self) -> None:
        task = runner._build_task(self._args())
        # tasks.json owns warmup; it must be a valid steady-state exclusion for the run.
        assert isinstance(task.warmup_frames, int)
        assert 0 <= task.warmup_frames < task.num_frames

    def test_inline_task_defaults_warmup_frames(self) -> None:
        task = runner._build_task(self._args(task_id="Not-A-Registered-Task-v0", backend_key="newton", num_envs=4))
        assert task.warmup_frames == 100  # min(100, num_frames - 1) with the 300-frame inline default

    def test_inline_task_small_num_frames_clamps_warmup(self) -> None:
        task = runner._build_task(
            self._args(task_id="Not-A-Registered-Task-v0", backend_key="newton", num_envs=4, num_frames=30)
        )
        assert task.warmup_frames == 29  # min(100, num_frames - 1) keeps >= 1 measured frame

    def test_warmup_frames_override_applied_to_registry_task(self) -> None:
        task = runner._build_task(self._args(warmup_frames=7))
        assert task.warmup_frames == 7


class TestResolveHydraArgs:
    """Inline --hydra_arg wins verbatim; otherwise the backend-derived default is used."""

    def test_inline_hydra_args_win(self) -> None:
        task = runner._build_task(
            argparse.Namespace(
                task_id="Isaac-Cartpole-Direct",
                backend_key="newton",
                tasks_json=None,
                num_envs=None,
                num_frames=None,
                warmup_frames=None,
                seed=None,
                camera_resolution=None,
                timeout_minutes=None,
            )
        )
        args = argparse.Namespace(hydra_arg=["presets=cube,newton,rgb64"])
        assert runner._resolve_hydra_args(args, task) == ["presets=cube,newton,rgb64"]

    def test_default_hydra_args_used_when_none_given(self) -> None:
        task = runner._build_task(
            argparse.Namespace(
                task_id="Isaac-Cartpole-Direct",
                backend_key="newton",
                tasks_json=None,
                num_envs=None,
                num_frames=None,
                warmup_frames=None,
                seed=None,
                camera_resolution=None,
                timeout_minutes=None,
            )
        )
        args = argparse.Namespace(hydra_arg=[])
        resolved = runner._resolve_hydra_args(args, task)
        assert isinstance(resolved, list)
        assert any("newton" in item for item in resolved)


class TestEngineForwardsTaskSpec:
    """format_runner_command forwards the inline TaskSpec so tasks.json is optional."""

    def _plan(self, task: TaskSpec) -> BisectionPlan:
        return BisectionPlan(
            task_id="Custom-Task-v0",
            backend_key="newton",
            good_ref="good",
            bad_ref="bad",
            gpu_model="L40S",
            runner=RunnerSpec(mode="local-reconstruct"),
            task=task,
            metric=MetricSpec(),
        )

    def test_taskspec_fields_forwarded(self, tmp_path: Path) -> None:
        plan = self._plan(
            TaskSpec(
                num_envs=8,
                num_frames=150,
                seed=7,
                camera_resolution=[64, 48],
                timeout_minutes=25,
                hydra_args=["presets=cube,newton,rgb64"],
            )
        )
        cmd = format_runner_command(plan, tmp_path, "abc123", tmp_path / "art")
        assert cmd[cmd.index("--num_envs") + 1] == "8"
        assert cmd[cmd.index("--num_frames") + 1] == "150"
        assert cmd[cmd.index("--seed") + 1] == "7"
        cam_idx = cmd.index("--camera_resolution")
        assert cmd[cam_idx + 1 : cam_idx + 3] == ["64", "48"]
        assert cmd[cmd.index("--timeout_minutes") + 1] == "25"
        assert cmd[cmd.index("--hydra_arg") + 1] == "presets=cube,newton,rgb64"

    def test_empty_taskspec_forwards_nothing(self, tmp_path: Path) -> None:
        cmd = format_runner_command(self._plan(TaskSpec()), tmp_path, "abc123", tmp_path / "art")
        assert "--num_envs" not in cmd
        assert "--warmup_frames" not in cmd
        assert "--hydra_arg" not in cmd

    def test_taskspec_warmup_frames_forwarded(self, tmp_path: Path) -> None:
        cmd = format_runner_command(
            self._plan(TaskSpec(num_envs=8, num_frames=150, warmup_frames=40)), tmp_path, "abc123", tmp_path / "art"
        )
        assert cmd[cmd.index("--warmup_frames") + 1] == "40"


class TestTaskSpecPlanRoundTrip:
    """TaskSpec survives BisectionPlan JSON serialization (Option B, path 2)."""

    def test_plan_json_round_trip_preserves_task(self) -> None:
        plan = BisectionPlan(
            task_id="Custom-Task-v0",
            backend_key="newton",
            good_ref="good",
            bad_ref="bad",
            gpu_model="L40S",
            runner=RunnerSpec(mode="local-reconstruct"),
            task=TaskSpec(num_envs=8, num_frames=150, camera_resolution=[64, 48], hydra_args=["presets=cube"]),
            metric=MetricSpec(),
        )
        restored = BisectionPlan.from_json(plan.to_json())
        assert restored.task.num_envs == 8
        assert restored.task.num_frames == 150
        assert restored.task.camera_resolution == [64, 48]
        assert restored.task.hydra_args == ["presets=cube"]

    def test_plan_json_without_task_defaults_to_empty(self) -> None:
        plan = BisectionPlan.from_json(
            {
                "task_id": "t",
                "backend_key": "newton",
                "good_ref": "g",
                "bad_ref": "b",
                "runner": {"mode": "synthetic"},
            }
        )
        assert plan.task == TaskSpec()


class TestMetricKeyHint:
    """A missing metric path reports the available numeric keys."""

    def test_missing_key_lists_available_numeric_keys(self) -> None:
        from bisection.models import MetricSpec as _MetricSpec
        from bisection.paired_reference import metric_from_result

        bench = {
            "raw_fps_mean": 42.0,
            "runtime_resources": {"gpu_mem_used_mb": 1234.0},
            "name": "text-not-numeric",
        }
        with pytest.raises(KeyError) as excinfo:
            metric_from_result(bench, _MetricSpec(name="m", result_path="does.not.exist"))
        message = str(excinfo.value)
        assert "raw_fps_mean" in message
        assert "runtime_resources.gpu_mem_used_mb" in message
        assert "name" not in message  # non-numeric leaves are not suggested


class TestLiveCommandOutput:
    """The engine writes live output events while a command runs."""

    def test_run_command_writes_live_output_jsonl(self, tmp_path: Path) -> None:
        command_log = tmp_path / "bisect_command.log"
        exit_code, timed_out, duration_s = _run_command(
            [sys.executable, "-c", "print('probe line')"],
            command_log=command_log,
            timeout_s=10,
        )

        assert exit_code == 0
        assert timed_out is False
        assert duration_s >= 0.0
        assert "probe line" in command_log.read_text(encoding="utf-8")

        live_events = [
            json.loads(line) for line in (tmp_path / "live_output.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert live_events[0]["event"] == "process_start"
        assert any(event.get("line") == "probe line" for event in live_events)
        assert live_events[-1]["event"] == "process_exit"

    def test_runner_live_output_helper_writes_jsonl(self, tmp_path: Path) -> None:
        log_path = tmp_path / "bisect_command.log"
        exit_code = runner._run_with_live_output(
            [sys.executable, "-c", "print('container line')"],
            cwd=tmp_path,
            log_path=log_path,
        )

        assert exit_code == 0
        assert "container line" in log_path.read_text(encoding="utf-8")
        events = [
            json.loads(line) for line in (tmp_path / "live_output.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert events[0]["event"] == "process_start"
        assert any(event.get("line") == "container line" for event in events)
        assert events[-1]["event"] == "process_exit"


class TestProbeExecutionLoop:
    """The pre-benchmark probe can inspect live debug output and stop before benchmarking."""

    def test_probe_runs_debug_command_then_blocks_plan_issue(self, tmp_path: Path) -> None:
        class FakeProbePolicy:
            def __init__(self) -> None:
                self.contexts: list[ProbeContext] = []

            def decide(self, ctx: ProbeContext) -> ProbeDecision:
                self.contexts.append(ctx)
                if len(self.contexts) == 1:
                    return ProbeDecision(
                        PROBE_ACTION_RUN_DEBUG_COMMAND,
                        "inspect resolved backend before benchmark",
                        command=f"{sys.executable} -c \"print('resolved_backend=physx_newton_renderer')\"",
                    )
                assert "resolved_backend=physx_newton_renderer" in ctx.live_output_tail
                return ProbeDecision(
                    PROBE_ACTION_PLAN_ISSUE,
                    "requested newton backend resolved to physx_newton_renderer",
                    suggested_plan_change={"hydra_args": ["presets=cube,single_camera,newton,newton_renderer,rgb64"]},
                    confidence="high",
                )

        plan = BisectionPlan(
            task_id="Isaac-Dexsuite-Kuka-Allegro-Lift-v0",
            backend_key="newton_newton_renderer",
            good_ref="good",
            bad_ref="bad",
            gpu_model="L40S",
            runner=RunnerSpec(mode="synthetic"),
            task=TaskSpec(num_envs=64, hydra_args=["presets=cube,single_camera,newton,newton_renderer,rgb64"]),
            metric=MetricSpec(),
        )

        attempt, metric_value = _run_single_measurement(
            plan,
            tmp_path,
            commit_sha="abc123def456",
            label="candidate",
            run_idx=1,
            probe_policy=FakeProbePolicy(),
        )

        artifact_dir = Path(attempt.artifact_dir)
        probe_dir = artifact_dir / "probe"
        assert metric_value is None
        assert attempt.note == "probe_failed:plan_issue"
        assert not (artifact_dir / "bisect_command.log").exists()
        assert "resolved_backend=physx_newton_renderer" in (probe_dir / "live_output.jsonl").read_text(encoding="utf-8")
        probe_result = json.loads((probe_dir / "probe_result.json").read_text(encoding="utf-8"))
        assert probe_result["status"] == "plan_issue"
        assert probe_result["decision"]["confidence"] == "high"

    def test_probe_repairs_base_image_then_benchmarks_with_repaired_image(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FakeProbePolicy:
            def __init__(self) -> None:
                self.contexts: list[ProbeContext] = []

            def decide(self, ctx: ProbeContext) -> ProbeDecision:
                self.contexts.append(ctx)
                if len(self.contexts) == 1:
                    return ProbeDecision(
                        PROBE_ACTION_REPAIR_BASE_IMAGE,
                        "install log shows fTetWild cannot find GMP",
                        apt_packages=["libgmp-dev"],
                        confidence="high",
                    )
                assert ctx.plan["runner"]["image"].startswith("isaaclab-bisect:repair-")
                return ProbeDecision(PROBE_ACTION_READY, "repaired base image is ready", confidence="high")

        commands: list[list[str] | str] = []

        def fake_run_command(command, *, command_log: Path, timeout_s):
            commands.append(command)
            command_log.parent.mkdir(parents=True, exist_ok=True)
            command_log.write_text("fake command\n", encoding="utf-8")
            if command_log.name == "bisect_command.log":
                (command_log.parent / "perf_smoke_test_result.json").write_text(
                    json.dumps({"raw_fps_mean": 123.0}) + "\n", encoding="utf-8"
                )
            return 0, False, 0.1

        monkeypatch.setattr(engine, "_run_command", fake_run_command)
        plan = BisectionPlan(
            task_id="Isaac-Dexsuite-Kuka-Allegro-Lift-v0",
            backend_key="newton_newton_renderer",
            good_ref="good",
            bad_ref="bad",
            gpu_model="L40S",
            runner=RunnerSpec(mode="docker-reconstruct", image="isaaclab-bisect:base"),
            task=TaskSpec(num_envs=64, hydra_args=["presets=cube,single_camera,newton_mjwarp"]),
            metric=MetricSpec(),
        )

        attempt, metric_value = _run_single_measurement(
            plan,
            tmp_path,
            commit_sha="abc123def456",
            label="candidate",
            run_idx=1,
            probe_policy=FakeProbePolicy(),
        )

        assert metric_value == 123.0
        assert attempt.note is None
        assert "isaaclab-bisect:repair-" in attempt.command
        assert commands[0][:3] == ["docker", "build", "-t"]
        active = json.loads((tmp_path / "repairs" / "active_docker_image.json").read_text(encoding="utf-8"))
        assert active["apt_packages"] == ["libgmp-dev"]
        assert active["repaired_image"] in attempt.command

    def test_active_repair_image_is_reused_without_rebuilding(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repairs_dir = tmp_path / "repairs"
        repairs_dir.mkdir()
        (repairs_dir / "active_docker_image.json").write_text(
            json.dumps(
                {
                    "base_image": "isaaclab-bisect:base",
                    "repaired_image": "isaaclab-bisect:repair-existing",
                    "apt_packages": ["libgmp-dev"],
                }
            )
            + "\n",
            encoding="utf-8",
        )

        def fake_run_command(command, *, command_log: Path, timeout_s):
            assert command[0] != "docker" or command[1] != "build"
            (command_log.parent / "perf_smoke_test_result.json").write_text(
                json.dumps({"raw_fps_mean": 456.0}) + "\n", encoding="utf-8"
            )
            return 0, False, 0.1

        monkeypatch.setattr(engine, "_run_command", fake_run_command)
        plan = BisectionPlan(
            task_id="Isaac-Cartpole-Direct",
            backend_key="newton",
            good_ref="good",
            bad_ref="bad",
            gpu_model="L40S",
            runner=RunnerSpec(mode="docker-reconstruct", image="isaaclab-bisect:base"),
            metric=MetricSpec(),
        )

        attempt, metric_value = _run_single_measurement(
            plan,
            tmp_path,
            commit_sha="abc123def456",
            label="candidate",
            run_idx=1,
            probe_policy=None,
        )

        assert metric_value == 456.0
        assert "isaaclab-bisect:repair-existing" in attempt.command


class TestBaseImageRepair:
    """Generated Docker repair images are bounded and auditable."""

    def test_validate_accepts_allowlisted_package(self) -> None:
        assert validate_apt_packages(["libgmp-dev", "libgmp-dev"]) == ["libgmp-dev"]

    def test_validate_rejects_unallowlisted_package(self) -> None:
        with pytest.raises(ValueError, match="not allowlisted"):
            validate_apt_packages(["curl"])

    def test_write_repair_dockerfile(self, tmp_path: Path) -> None:
        repair = write_repair_dockerfile("isaaclab-bisect:base", ["libgmp-dev"], tmp_path)
        text = repair.dockerfile.read_text(encoding="utf-8")
        assert "FROM isaaclab-bisect:base" in text
        assert "libgmp-dev" in text
        assert repair.repaired_image.startswith("isaaclab-bisect:repair-")


class TestLLMProbePolicy:
    """The LLM-driven probe parser accepts structured setup-doctor decisions."""

    def test_parse_plan_issue_decision(self) -> None:
        policy = LLMProbePolicy(model="dummy-model")
        decision = policy._parse(
            """
            {
              "action": "plan_issue",
              "reason": "resolved backend does not match requested backend",
              "suggested_plan_change": {
                "hydra_args": ["presets=cube,single_camera,newton_mjwarp,newton_renderer,rgb64"]
              },
              "confidence": "high"
            }
            """
        )

        assert decision is not None
        assert decision.action == PROBE_ACTION_PLAN_ISSUE
        assert "newton_mjwarp" in decision.suggested_plan_change["hydra_args"][0]

    def test_parse_base_image_repair_decision(self) -> None:
        policy = LLMProbePolicy(model="dummy-model")
        decision = policy._parse(
            """
            {
              "action": "repair_base_image",
              "reason": "pytetwild build failed because fTetWild cannot find GMP",
              "apt_packages": ["libgmp-dev"],
              "confidence": "high"
            }
            """
        )

        assert decision is not None
        assert decision.action == PROBE_ACTION_REPAIR_BASE_IMAGE
        assert decision.apt_packages == ["libgmp-dev"]

    def test_parse_rejects_unknown_action(self) -> None:
        policy = LLMProbePolicy(model="dummy-model")
        assert policy._parse('{"action": "good", "reason": "not allowed"}') is None


@dataclass
class _FakeComponentStack:
    """A resolved pinned stack with the component fields the diff inspects."""

    commit_sha: str
    stack_hash: str = "hash"
    isaacsim: str | None = "6.0.0-dev2"
    warp_lang: str | None = "==1.5.0"
    newton: str | None = "==1.2.1"
    ovrtx: str | None = None
    ovphysx: str | None = None
    python_version: str = "3.12"
    platform: str = "linux-x86_64"


class TestComponentStackDiff:
    """The pinned-stack component diff surfaces which dependency moved across the range."""

    def _stacks(self, monkeypatch, mapping: dict[str, _FakeComponentStack]) -> None:
        monkeypatch.setattr(engine, "resolve_stack", lambda repo_root, sha: mapping[sha])

    def test_diff_reports_changed_component(self, monkeypatch) -> None:
        """A newton pin bump across the culprit commit is named in the diff."""
        self._stacks(
            monkeypatch,
            {
                "good": _FakeComponentStack("good", stack_hash="h_good", newton="==1.2.1"),
                "bad": _FakeComponentStack("bad", stack_hash="h_bad", newton="==1.3.0"),
            },
        )
        diff = engine._build_stack_diff(good_ref="good", bad_ref="bad", last_good="good", first_bad="bad")
        assert diff is not None
        changed = diff["culprit"]["changed_components"]
        assert changed["newton"] == {"from": "==1.2.1", "to": "==1.3.0"}
        assert diff["culprit"]["stack_hash"]["changed"] is True

    def test_diff_range_only_when_no_culprit(self, monkeypatch) -> None:
        """A non-repro (no culprit) still reports what moved across the full range."""
        self._stacks(
            monkeypatch,
            {
                "good": _FakeComponentStack("good", stack_hash="h_good", isaacsim="6.0.0-dev1"),
                "bad": _FakeComponentStack("bad", stack_hash="h_bad", isaacsim="6.0.0-dev2"),
            },
        )
        diff = engine._build_stack_diff(good_ref="good", bad_ref="bad")
        assert diff is not None
        assert "culprit" not in diff
        assert diff["range"]["changed_components"]["isaacsim"] == {"from": "6.0.0-dev1", "to": "6.0.0-dev2"}

    def test_diff_none_when_resolution_fails(self, monkeypatch) -> None:
        """A stack-resolution failure downgrades to None rather than crashing the run."""

        def boom(repo_root, sha):
            raise RuntimeError("git show failed")

        monkeypatch.setattr(engine, "resolve_stack", boom)
        assert engine._build_stack_diff(good_ref="good", bad_ref="bad") is None

    def test_identical_stack_reports_no_change(self, monkeypatch) -> None:
        """Identical pinned stacks yield an empty changed-components map (not a crash)."""
        self._stacks(
            monkeypatch,
            {"good": _FakeComponentStack("good"), "bad": _FakeComponentStack("bad")},
        )
        diff = engine._build_stack_diff(good_ref="good", bad_ref="bad")
        assert diff is not None
        assert diff["range"]["changed_components"] == {}
        assert diff["range"]["stack_hash"]["changed"] is False
