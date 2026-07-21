# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Resolve, materialize, and verify pinned perf-smoke measurement tooling."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tarfile
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

from backend_identity import split_backend_key
from contracts import CONTRACT_SCHEMA_VERSION
from hashing import stable_hash
from launch_config import hydra_args_for_task, task_to_launch_config
from task_config import TaskConfig, caches_for_backend, get_task

from .git_utils import git, resolve_ref
from .io import read_json_or_empty
from .models import BisectionPlan, TaskSpec, ToolingSpec

TOOLING_CONTRACT_ID = "perf_smoke_runtime_bundle_v1:raw_fps_mean:steady_state"
TOOLING_SNAPSHOT_RELPATH = "tooling/perf_smoke_test"
TOOLING_DRIVER_RELPATH = "perf_runtime.py"
TOOLING_RESULT_BUILDER_RELPATH = "build_bench_result.py"
_FULL_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")


class ToolingError(RuntimeError):
    """Structured tooling-resolution or integrity failure."""

    def __init__(self, category: str, detail: str):
        super().__init__(detail)
        self.category = category
        self.detail = detail


def _file_manifest(root: Path) -> dict[str, str]:
    """Return content hashes for all tooling files below ``root``."""
    manifest: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            payload = str(path.readlink()).encode("utf-8")
        else:
            payload = path.read_bytes()
        manifest[relative] = hashlib.sha256(payload).hexdigest()
    return manifest


def tooling_bundle_hash(root: Path) -> tuple[str, int]:
    """Return the stable content hash and file count for a tooling directory."""
    manifest = _file_manifest(root)
    if not manifest:
        raise RuntimeError(f"perf-smoke tooling directory is empty: {root}")
    return stable_hash(manifest), len(manifest)


def _resolve_task(plan: BisectionPlan) -> tuple[TaskConfig, list[str], TaskSpec]:
    """Resolve registry/default task values into a fully explicit task specification."""
    identity = split_backend_key(plan.backend_key)
    if identity is None:
        raise RuntimeError(f"Cannot parse backend key {plan.backend_key!r}")
    inline = plan.task
    try:
        registry_task = get_task(plan.task_id, identity.backend_key)
    except (KeyError, FileNotFoundError):
        registry_task = None

    if registry_task is not None:
        updates: dict[str, Any] = {}
        for name in ("num_envs", "num_frames", "warmup_frames", "seed", "timeout_minutes"):
            value = getattr(inline, name)
            if value is not None:
                updates[name] = value
        if inline.camera_resolution is not None:
            updates["camera_resolution"] = tuple(inline.camera_resolution)
        task = replace(registry_task, **updates) if updates else registry_task
    else:
        if inline.num_envs is None:
            raise ValueError(
                f"task {plan.task_id!r}/{identity.backend_key!r} is not registered; provide an inline num_envs value"
            )
        num_frames = inline.num_frames if inline.num_frames is not None else 300
        warmup_frames = inline.warmup_frames if inline.warmup_frames is not None else min(100, max(0, num_frames - 1))
        task = TaskConfig(
            task_id=plan.task_id,
            physics_backend=identity.physics_backend,
            render_backend=identity.render_backend,
            preset="inline",
            num_envs=inline.num_envs,
            num_frames=num_frames,
            warmup_frames=warmup_frames,
            camera_resolution=tuple(inline.camera_resolution) if inline.camera_resolution is not None else None,
            timeout_minutes=inline.timeout_minutes if inline.timeout_minutes is not None else 30,
            fps_mean_thresholds={},
            noise_floor_pct={},
            caches=caches_for_backend(identity.physics_backend),
            seed=inline.seed if inline.seed is not None else 42,
        )

    hydra_args = list(inline.hydra_args) if inline.hydra_args else hydra_args_for_task(task)
    explicit = TaskSpec(
        num_envs=task.num_envs,
        num_frames=task.num_frames,
        warmup_frames=task.warmup_frames,
        seed=task.seed,
        camera_resolution=list(task.camera_resolution) if task.camera_resolution is not None else None,
        timeout_minutes=task.timeout_minutes,
        hydra_args=hydra_args,
    )
    return task, hydra_args, explicit


def _tooling_spec_hash_payload(spec: ToolingSpec) -> dict[str, Any]:
    payload = spec.to_json()
    payload.pop("tooling_spec_hash", None)
    return payload


def resolve_tooling_plan(
    plan: BisectionPlan,
    repo_root: Path,
    *,
    tooling_ref: str | None = None,
) -> BisectionPlan:
    """Resolve a fully explicit plan with one pinned perf-smoke tooling contract."""
    if tooling_ref is None:
        raise ToolingError(
            "tooling_sha_required",
            "authoritative runs require --tooling_ref with a full 40-character commit SHA; "
            "use --tooling_ref WORKTREE only for non-authoritative development",
        )
    source_root = repo_root / "tools" / "perf_smoke_test"
    head_sha = resolve_ref(repo_root, "HEAD")
    if tooling_ref == "WORKTREE":
        commit_sha = head_sha
        bundle_hash, file_count = tooling_bundle_hash(source_root)
        source_dirty = bool(git(repo_root, ["status", "--porcelain", "--", "tools/perf_smoke_test"]).stdout.strip())
        authoritative = False
    else:
        if not _FULL_SHA_RE.fullmatch(tooling_ref):
            raise ToolingError(
                "tooling_sha_required",
                f"authoritative tooling_ref must be a full 40-character commit SHA, got {tooling_ref!r}",
            )
        try:
            commit_sha = resolve_ref(repo_root, tooling_ref)
        except RuntimeError as exc:
            raise ToolingError(
                "tooling_sha_unavailable",
                f"tooling commit {tooling_ref} is unavailable in this clone: {exc}",
            ) from exc
        if commit_sha.lower() != tooling_ref.lower():
            raise ToolingError(
                "tooling_sha_mismatch",
                f"tooling ref resolved to {commit_sha}, expected {tooling_ref}",
            )
        with tempfile.TemporaryDirectory(prefix="perf-smoke-tooling-") as temp_dir:
            extracted = _archive_tooling(repo_root, commit_sha, Path(temp_dir))
            bundle_hash, file_count = tooling_bundle_hash(extracted)
        source_dirty = False
        authoritative = True

    task, hydra_args, explicit_task = _resolve_task(plan)
    launch_config = task_to_launch_config(
        task,
        fps_mean_thresholds=task.thresholds_for(plan.gpu_model),
        gpu_model=plan.gpu_model,
        hydra_args=hydra_args,
        benchmark_formatter="schema",
    )
    task_payload = {
        "task_id": plan.task_id,
        "backend_key": plan.backend_key,
        **explicit_task.to_json(),
        "benchmark_formatter": "schema",
    }
    spec = ToolingSpec(
        source_ref=tooling_ref,
        source_commit_sha=commit_sha,
        source_dirty=source_dirty,
        authoritative=authoritative,
        bundle_hash=bundle_hash,
        bundle_file_count=file_count,
        snapshot_relpath=TOOLING_SNAPSHOT_RELPATH,
        driver_relpath=TOOLING_DRIVER_RELPATH,
        result_builder_relpath=TOOLING_RESULT_BUILDER_RELPATH,
        contract_id=TOOLING_CONTRACT_ID,
        result_schema_version=CONTRACT_SCHEMA_VERSION,
        launch_config_hash=str(launch_config["launch_config_hash"]),
        benchmark_contract_hash=str(launch_config["benchmark_contract_hash"]),
        task_config_hash=stable_hash(task_payload),
        metric_name=plan.metric.name,
        metric_path=plan.metric.result_path,
        regression_direction=plan.metric.regression_direction,
        benchmark_warmup_frames=task.warmup_frames,
        process_warmup_runs=1 if plan.measurement.warmup_runs > 0 else 0,
    )
    spec = replace(spec, tooling_spec_hash=stable_hash(_tooling_spec_hash_payload(spec)))
    return replace(plan, task=explicit_task, tooling=spec, schema_version=3)


def _archive_tooling(repo_root: Path, commit_sha: str, destination: Path) -> Path:
    """Extract the committed perf-smoke tooling tree into ``destination``."""
    archive = subprocess.run(
        ["git", "archive", "--format=tar", commit_sha, "tools/perf_smoke_test"],
        cwd=repo_root,
        capture_output=True,
        check=False,
    )
    if archive.returncode != 0:
        detail = archive.stderr.decode(errors="replace").strip() or "git archive failed"
        raise ToolingError(
            "tooling_sha_unavailable",
            f"could not materialize perf-smoke tooling from {commit_sha}: {detail}",
        )
    with tempfile.NamedTemporaryFile(suffix=".tar") as temp:
        temp.write(archive.stdout)
        temp.flush()
        with tarfile.open(temp.name) as tar:
            tar.extractall(destination, filter="data")
    return destination / "tools" / "perf_smoke_test"


def materialize_tooling_snapshot(plan: BisectionPlan, repo_root: Path, output_dir: Path) -> Path:
    """Materialize and verify the plan's immutable run-scoped tooling snapshot."""
    if plan.tooling is None:
        raise ValueError("plan.tooling is required")
    spec = plan.tooling
    destination = output_dir / spec.snapshot_relpath
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if spec.source_ref == "WORKTREE":
        source = repo_root / "tools" / "perf_smoke_test"
        current_hash, _ = tooling_bundle_hash(source)
        if current_hash != spec.bundle_hash:
            raise ToolingError(
                "tooling_hash_mismatch",
                f"perf-smoke worktree changed after plan resolution: {current_hash} != {spec.bundle_hash}",
            )
        shutil.copytree(
            source,
            destination,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )
    else:
        with tempfile.TemporaryDirectory(prefix="perf-smoke-tooling-") as temp_dir:
            extracted = _archive_tooling(repo_root, spec.source_commit_sha, Path(temp_dir))
            shutil.copytree(extracted, destination)

    actual_hash, actual_count = tooling_bundle_hash(destination)
    if actual_hash != spec.bundle_hash or actual_count != spec.bundle_file_count:
        raise ToolingError(
            "tooling_hash_mismatch",
            "materialized perf-smoke tooling does not match the pinned contract "
            f"(hash={actual_hash}, files={actual_count})",
        )
    (output_dir / "tooling_manifest.json").write_text(
        json.dumps(spec.to_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination


def verify_attempt_tooling(plan: BisectionPlan, artifact_dir: Path) -> dict[str, Any]:
    """Verify that one successful attempt used the pinned tooling and workload hashes."""
    if plan.tooling is None:
        return {"status": "skipped", "mismatches": ["legacy schema-v2 plan has no tooling pin"]}
    spec = plan.tooling
    mismatches: list[str] = []
    sidecar = read_json_or_empty(artifact_dir / "tooling.json")
    launch = read_json_or_empty(artifact_dir / "launch_config.json")
    result = read_json_or_empty(artifact_dir / "perf_smoke_test_result.json")

    checks = (
        ("tooling_spec_hash", sidecar.get("tooling_spec_hash"), spec.tooling_spec_hash),
        ("bundle_hash", sidecar.get("bundle_hash"), spec.bundle_hash),
        ("contract_id", sidecar.get("contract_id"), spec.contract_id),
        ("source_commit_sha", sidecar.get("source_commit_sha"), spec.source_commit_sha),
        ("authoritative", sidecar.get("authoritative"), spec.authoritative),
        ("launch_config_hash", launch.get("launch_config_hash"), spec.launch_config_hash),
        (
            "benchmark_contract_hash",
            launch.get("benchmark_contract_hash"),
            spec.benchmark_contract_hash,
        ),
        ("warmup_frames", launch.get("warmup_frames"), spec.benchmark_warmup_frames),
        ("result_schema_version", result.get("schema_version"), spec.result_schema_version),
        ("result_launch_config_hash", result.get("launch_config_hash"), spec.launch_config_hash),
        (
            "result_benchmark_contract_hash",
            result.get("benchmark_contract_hash"),
            spec.benchmark_contract_hash,
        ),
    )
    for name, actual, expected in checks:
        if actual != expected:
            mismatches.append(f"{name}(actual={actual!r},expected={expected!r})")
    return {
        "status": "ok" if not mismatches else "mismatch",
        "mismatches": mismatches,
        "tooling_spec_hash": spec.tooling_spec_hash,
    }
