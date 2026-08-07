# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GPU-free tests for the pinned perf-smoke tooling contract."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

_GATE_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _GATE_DIR.parent.parent
if str(_GATE_DIR) not in sys.path:
    sys.path.insert(0, str(_GATE_DIR))

from isaaclab_bisection.bisection.git_utils import resolve_ref  # noqa: E402
from isaaclab_bisection.bisection.models import BisectionPlan, RunnerSpec, TaskSpec  # noqa: E402
from isaaclab_bisection.bisection.tooling import (  # noqa: E402
    ToolingError,
    materialize_tooling_snapshot,
    resolve_tooling_plan,
    tooling_bundle_hash,
    verify_attempt_tooling,
)
from isaaclab_bisection.cli import _relaunch_tooling_fields  # noqa: E402
from isaaclab_bisection.runner import (  # noqa: E402
    _build_tooling_benchmark_command,
    _build_tooling_capability_command,
    _verify_mounted_tooling,
)


def _load_tooling_capability_module() -> ModuleType:
    path = _REPO_ROOT / "tools" / "perf_smoke_test" / "tooling_capability.py"
    spec = importlib.util.spec_from_file_location("test_tooling_capability", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_module(name: str, attributes: tuple[str, ...] = ()) -> ModuleType:
    module = ModuleType(name)
    for attribute in attributes:
        setattr(module, attribute, object())
    return module


@pytest.fixture
def committed_tooling_repo(tmp_path: Path) -> tuple[Path, str]:
    """Create a repository whose committed tooling can be archived by SHA."""
    root = tmp_path / "repo"
    tooling = root / "tools" / "perf_smoke_test"
    shutil.copytree(_REPO_ROOT / "tools" / "perf_smoke_test", tooling)
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-m", "Add tooling"], cwd=root, check=True, capture_output=True)
    sha = resolve_ref(root, "HEAD")
    return root, sha


def _inline_plan() -> BisectionPlan:
    return BisectionPlan(
        task_id="Isaac-Tooling-Contract-Test-v0",
        backend_key="physx",
        good_ref="HEAD",
        bad_ref="HEAD",
        gpu_model="test-gpu",
        runner=RunnerSpec(mode="synthetic"),
        task=TaskSpec(
            num_envs=8,
            num_frames=20,
            warmup_frames=5,
            seed=7,
            hydra_args=["presets=physx"],
        ),
    )


def test_command_uses_tooling_driver_when_candidate_has_no_benchmark_code(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    tooling = tmp_path / "tooling"
    candidate.mkdir()
    tooling.mkdir()
    task = SimpleNamespace(task_id="Task-v0", num_envs=4, num_frames=10, warmup_frames=2, seed=3)

    command = _build_tooling_benchmark_command(
        tooling_root=tooling,
        isaaclab_sh=candidate / "isaaclab.sh",
        task=task,
        artifact_dir=tmp_path / "artifacts",
        hydra_args=["presets=physx"],
    )

    assert str(tooling / "perf_runtime.py") in command
    assert str(candidate / "tools/perf_smoke_test/perf_runtime.py") not in command
    assert command[-1] == "presets=physx"


def test_capability_command_uses_pinned_tooling(tmp_path: Path) -> None:
    command = _build_tooling_capability_command(
        tooling_root=tmp_path / "tooling",
        isaaclab_sh=tmp_path / "candidate" / "isaaclab.sh",
        artifact_dir=tmp_path / "artifacts",
    )

    assert command[2] == str(tmp_path / "tooling" / "tooling_capability.py")
    assert command[-1] == str(tmp_path / "artifacts" / "tooling_capability.json")


@pytest.mark.parametrize("benchmark_namespace", ["isaaclab.benchmark", "isaaclab.test.benchmark"])
def test_tooling_capability_accepts_current_and_legacy_benchmark_namespaces(
    monkeypatch: pytest.MonkeyPatch, benchmark_namespace: str
) -> None:
    benchmark_attributes = ("BaseIsaacLabBenchmark", "BenchmarkMonitor", "builders", "capture", "stepping")
    modules = {
        "isaaclab": _fake_module("isaaclab"),
        "isaaclab.app": _fake_module("isaaclab.app", ("AppLauncher", "launch_simulation")),
        benchmark_namespace: _fake_module(benchmark_namespace, benchmark_attributes),
        f"{benchmark_namespace}.schema": _fake_module(f"{benchmark_namespace}.schema", ("StartupTime",)),
        "isaaclab_tasks": _fake_module("isaaclab_tasks"),
        "isaaclab_tasks.utils": _fake_module("isaaclab_tasks.utils", ("setup_preset_cli", "resolve_task_config")),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    capability = _load_tooling_capability_module()

    assert capability.check_capabilities() == []


def test_resolved_plan_pins_task_metric_and_warmups() -> None:
    resolved = resolve_tooling_plan(_inline_plan(), _REPO_ROOT, tooling_ref="WORKTREE")

    assert resolved.schema_version == 3
    assert resolved.tooling is not None
    assert resolved.tooling.benchmark_warmup_frames == 5
    assert resolved.tooling.process_warmup_runs == 1
    assert resolved.tooling.metric_path == "raw_fps_mean"
    assert resolved.tooling.launch_config_hash
    assert resolved.tooling.benchmark_contract_hash
    assert resolved.tooling.tooling_spec_hash
    assert resolved.tooling.authoritative is False
    assert resolved.task.hydra_args == ["presets=physx"]


def test_materialized_snapshot_matches_pinned_bundle(tmp_path: Path) -> None:
    resolved = resolve_tooling_plan(_inline_plan(), _REPO_ROOT, tooling_ref="WORKTREE")
    tooling_root = materialize_tooling_snapshot(resolved, _REPO_ROOT, tmp_path)

    assert resolved.tooling is not None
    assert tooling_bundle_hash(tooling_root) == (
        resolved.tooling.bundle_hash,
        resolved.tooling.bundle_file_count,
    )
    assert (tooling_root / "build_bench_result.py").is_file()
    assert json.loads((tmp_path / "tooling_manifest.json").read_text())["tooling_spec_hash"]


def test_attempt_verification_rejects_contract_drift(tmp_path: Path) -> None:
    resolved = resolve_tooling_plan(_inline_plan(), _REPO_ROOT, tooling_ref="WORKTREE")
    assert resolved.tooling is not None
    spec = resolved.tooling
    (tmp_path / "tooling.json").write_text(
        json.dumps(
            {
                "tooling_spec_hash": spec.tooling_spec_hash,
                "bundle_hash": "wrong",
                "contract_id": spec.contract_id,
                "source_commit_sha": spec.source_commit_sha,
                "authoritative": spec.authoritative,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "launch_config.json").write_text(
        json.dumps(
            {
                "launch_config_hash": spec.launch_config_hash,
                "benchmark_contract_hash": spec.benchmark_contract_hash,
                "warmup_frames": spec.benchmark_warmup_frames,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "perf_smoke_test_result.json").write_text(
        json.dumps(
            {
                "schema_version": spec.result_schema_version,
                "launch_config_hash": spec.launch_config_hash,
                "benchmark_contract_hash": spec.benchmark_contract_hash,
            }
        ),
        encoding="utf-8",
    )

    verification = verify_attempt_tooling(resolved, tmp_path)

    assert verification["status"] == "mismatch"
    assert any(item.startswith("bundle_hash") for item in verification["mismatches"])


def test_runner_startup_rejects_mounted_tooling_hash_drift(tmp_path: Path) -> None:
    tooling_root = tmp_path / "tooling"
    tooling_root.mkdir()
    (tooling_root / "driver.py").write_text("original\n", encoding="utf-8")
    expected_hash, _ = tooling_bundle_hash(tooling_root)
    (tooling_root / "driver.py").write_text("changed\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="tooling_hash_mismatch"):
        _verify_mounted_tooling(tooling_root, expected_hash)


def test_authoritative_plan_requires_full_available_sha(
    committed_tooling_repo: tuple[Path, str],
) -> None:
    with pytest.raises(ToolingError, match="full 40-character commit SHA"):
        resolve_tooling_plan(_inline_plan(), _REPO_ROOT, tooling_ref="HEAD")

    root, sha = committed_tooling_repo
    resolved = resolve_tooling_plan(_inline_plan(), root, tooling_ref=sha)

    assert resolved.tooling is not None
    assert resolved.tooling.source_commit_sha == sha
    assert resolved.tooling.authoritative is True
    relaunch = _relaunch_tooling_fields(resolved)
    assert relaunch["authoritative"] is True
    assert relaunch["required_tooling_sha"] == sha
    assert sha in relaunch["note"]


def test_missing_tooling_sha_fails_cleanly() -> None:
    with pytest.raises(ToolingError) as exc_info:
        resolve_tooling_plan(_inline_plan(), _REPO_ROOT, tooling_ref="0" * 40)

    assert exc_info.value.category == "tooling_sha_unavailable"
