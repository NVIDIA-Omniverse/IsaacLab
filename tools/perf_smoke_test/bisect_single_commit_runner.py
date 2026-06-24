#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Single-commit runner used by the preliminary bisection agent.

The runner has two modes:

* ``synthetic`` writes normal perf-smoke artifacts via the stub benchmark. This is
  fast and useful for demos without a GPU.
* ``docker-source`` checks out one candidate commit into an isolated clone,
  source-mounts it into a fixed IsaacLab CI image, runs one task/backend, and
  emits the same artifact contract.
* ``local-source`` checks out one candidate commit into an isolated clone and
  runs that clone with the host's existing IsaacLab Python environment.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

_MODULE_DIR = Path(__file__).parent
_REPO_ROOT = _MODULE_DIR.parents[1]

if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))

from backend_identity import split_backend_key  # noqa: E402
from baseline_manager import update_baseline  # noqa: E402
from gpu_identity import canonical_gpu_model  # noqa: E402
from launch_config import hydra_args_for_task, task_to_launch_config, write_launch_config  # noqa: E402
from task_config import get_task  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one commit/task/backend for the perf bisection POC.")
    parser.add_argument("--commit", required=True, help="Commit SHA/ref being tested.")
    parser.add_argument("--task_id", required=True)
    parser.add_argument("--backend_key", required=True)
    parser.add_argument("--artifact_dir", required=True, type=Path)
    parser.add_argument(
        "--mode",
        choices=("synthetic", "docker-source", "local-source"),
        default="synthetic",
        help="Runner mode. synthetic is GPU-free; docker-source/local-source run real IsaacLab.",
    )
    parser.add_argument(
        "--first_bad_ref",
        default=None,
        help="Synthetic demo knob: this ref and descendants are emitted as regressed.",
    )
    parser.add_argument("--image", default="", help="Docker image tag for --mode docker-source.")
    parser.add_argument(
        "--source_dir",
        type=Path,
        default=None,
        help="Reusable isolated clone for candidate source (default: sibling of artifact root).",
    )
    parser.add_argument(
        "--jit_cache",
        type=Path,
        default=None,
        help="Host JIT cache directory for Docker mode (default: artifact root / jit-cache).",
    )
    parser.add_argument(
        "--kit_cache",
        type=Path,
        default=None,
        help="Host Kit shader cache directory for real modes (default: artifact root / kit-cache).",
    )
    parser.add_argument(
        "--local_env_dir",
        type=Path,
        default=_REPO_ROOT / "env_isaaclab",
        help="Existing IsaacLab Python environment to symlink into the isolated clone for local-source mode.",
    )
    parser.add_argument(
        "--ld_preload",
        default="",
        help="Optional LD_PRELOAD value for local-source mode, useful on ARM hosts that require libgomp preload.",
    )
    parser.add_argument(
        "--override_num_envs",
        type=int,
        default=None,
        help="Smoke-test override for task.num_envs. Omit for normal gate-equivalent runs.",
    )
    parser.add_argument(
        "--override_num_frames",
        type=int,
        default=None,
        help="Smoke-test override for task.num_frames. Omit for normal gate-equivalent runs.",
    )
    parser.add_argument("--gpu_model", default="L40S")
    parser.add_argument("--good_fps", type=float, default=1000.0)
    parser.add_argument("--bad_fps", type=float, default=500.0)
    parser.add_argument("--baselines_dir", type=Path, default=_MODULE_DIR / "local_baselines")
    parser.add_argument("--ensure_baseline", action="store_true", help="Seed matching local baseline samples first.")
    parser.add_argument("--baseline_samples", type=int, default=8)
    parser.add_argument("--gate_config", type=Path, default=_MODULE_DIR / "gate_config.json")
    return parser.parse_args()


def _git(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(["git", *args], cwd=_REPO_ROOT, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"git {' '.join(args)} failed")
    return result


def _resolve_ref(ref: str) -> str:
    return _git(["rev-parse", "--verify", f"{ref}^{{commit}}"]).stdout.strip()


def _is_ancestor(ancestor: str, descendant: str) -> bool:
    return _git(["merge-base", "--is-ancestor", ancestor, descendant], check=False).returncode == 0


def _task_from_backend(task_id: str, backend_key: str):
    identity = split_backend_key(backend_key)
    if identity is None:
        raise RuntimeError(f"Cannot parse backend key {backend_key!r}")
    return get_task(task_id, identity.backend_key)


def _task_with_overrides(task, *, num_envs: int | None, num_frames: int | None):
    updates = {}
    if num_envs is not None:
        updates["num_envs"] = num_envs
    if num_frames is not None:
        updates["num_frames"] = num_frames
    return replace(task, **updates) if updates else task


def _safe_component(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in value)


def _artifact_root(artifact_dir: Path) -> Path:
    # Artifacts are normally <run>/artifacts/<sha>/<task>/<backend>. The runner
    # also supports arbitrary artifact dirs by falling back to the parent.
    try:
        parts = artifact_dir.parts
        idx = parts.index("artifacts")
        return Path(*parts[:idx]) if idx > 0 else artifact_dir.parent
    except ValueError:
        return artifact_dir.parent


def _run(cmd: list[str], *, cwd: Path | None = None, check: bool = True, env: dict[str, str] | None = None):
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True, env=env)
    if check and result.returncode != 0:
        raise RuntimeError(f"command failed (exit {result.returncode}): {' '.join(cmd)}")
    return result


def _prepare_source_clone(source_dir: Path, commit_sha: str) -> None:
    """Materialize ``commit_sha`` into a self-contained isolated clone."""
    git_env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
    if not (source_dir / ".git").exists():
        source_dir.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", "--no-checkout", str(_REPO_ROOT.resolve()), str(source_dir)], env=git_env)
    _run(["git", "fetch", "--no-tags", str(_REPO_ROOT.resolve()), commit_sha], cwd=source_dir, env=git_env)
    _run(["git", "checkout", "-f", "--detach", commit_sha], cwd=source_dir, env=git_env)
    subprocess.run(["chmod", "-R", "a+rwX", str(source_dir)], check=False)
    clean = _run(["git", "clean", "-fdx"], cwd=source_dir, check=False, env=git_env)
    if clean.returncode != 0:
        print(
            f"[bisect_single_commit_runner] warning: git clean left residue in {source_dir} "
            f"(exit {clean.returncode}); continuing",
            flush=True,
        )
    subprocess.run(["chmod", "-R", "a+rwX", str(source_dir)], check=False)


def _symlink_runtime_path(source_dir: Path, name: str, target: Path) -> None:
    link = source_dir / name
    if not target.exists():
        return
    if link.is_symlink() or link.exists():
        if link.resolve() == target.resolve():
            return
        if link.is_dir() and not link.is_symlink():
            raise RuntimeError(f"Cannot replace existing runtime directory: {link}")
        link.unlink()
    link.symlink_to(target.resolve(), target_is_directory=target.is_dir())


def _prepare_local_source_runtime(source_dir: Path, local_env_dir: Path) -> None:
    """Make a historical source clone runnable with the host IsaacLab install."""
    if not local_env_dir.exists():
        raise RuntimeError(f"Local IsaacLab environment not found: {local_env_dir}")
    exclude_path = source_dir / ".git" / "info" / "exclude"
    exclude_text = exclude_path.read_text(encoding="utf-8") if exclude_path.exists() else ""
    missing_excludes = [entry for entry in ("/env_isaaclab", "/_isaac_sim") if entry not in exclude_text.splitlines()]
    if missing_excludes:
        exclude_path.parent.mkdir(parents=True, exist_ok=True)
        with exclude_path.open("a", encoding="utf-8") as fh:
            for entry in missing_excludes:
                fh.write(f"{entry}\n")
    _symlink_runtime_path(source_dir, "env_isaaclab", local_env_dir)
    _symlink_runtime_path(source_dir, "_isaac_sim", _REPO_ROOT / "_isaac_sim")


def _run_stub_benchmark(task, artifact_dir: Path, fps_mean: float) -> tuple[int, float]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    log_file = artifact_dir / "benchmark.log"
    cmd = [
        sys.executable,
        str(_MODULE_DIR / "dev" / "stub_benchmark.py"),
        "--task_id",
        task.task_id,
        "--backend",
        task.backend_key,
        "--num_envs",
        str(task.num_envs),
        "--num_frames",
        str(task.num_frames),
        "--out_dir",
        str(artifact_dir),
        "--fps_mean",
        str(fps_mean),
    ]
    if task.seed is not None:
        cmd.extend(["--seed", str(task.seed)])

    start = time.monotonic()
    with log_file.open("w", encoding="utf-8") as log_fh:
        log_fh.write(f"$ {' '.join(cmd)}\n\n")
        result = subprocess.run(cmd, cwd=_REPO_ROOT, stdout=log_fh, stderr=subprocess.STDOUT, text=True)
    return result.returncode, time.monotonic() - start


def _run_docker_source_benchmark(
    *,
    image: str,
    task,
    artifact_dir: Path,
    source_dir: Path,
    jit_cache: Path,
    kit_cache: Path,
    commit_sha: str,
) -> tuple[int, float]:
    """Run one real IsaacLab benchmark in Docker with candidate source mounted."""
    if not image.strip():
        raise ValueError("--image is required for --mode docker-source")

    for path in (artifact_dir, jit_cache / "warp", jit_cache / "nv", kit_cache):
        path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["chmod", "-R", "0777", str(artifact_dir), str(jit_cache), str(kit_cache)], check=False)

    hydra_args = " ".join(shlex.quote(arg) for arg in hydra_args_for_task(task))
    seed_token = f"--seed {task.seed}" if task.seed is not None else ""
    inner = (
        "set -e\n"
        "cd /workspace/isaaclab\n"
        "rm -f _isaac_sim\n"
        "ln -s /isaac-sim _isaac_sim\n"
        "./isaaclab.sh -p scripts/benchmarks/benchmark_non_rl.py "
        f"--task {shlex.quote(task.task_id)} "
        f"--num_envs {task.num_envs} "
        f"--num_frames {task.num_frames} "
        "--benchmark_backend JSONFileMetrics "
        "--output_path /tmp/bench_out "
        f"{seed_token} {hydra_args}\n"
    )
    container_name = _safe_component(f"perf-bisect-{commit_sha[:12]}-{task.task_id}-{task.backend_key}")
    cmd = [
        "docker",
        "run",
        "--rm",
        "--name",
        container_name,
        "--init",
        "--stop-timeout",
        "10",
        "--entrypoint",
        "bash",
        "--gpus",
        "all",
        "--network=host",
        "--security-opt=no-new-privileges:true",
        "--ulimit",
        "nofile=65536:65536",
        "--ulimit",
        "nproc=4096:4096",
        "-e",
        "OMNI_KIT_ACCEPT_EULA=yes",
        "-e",
        "ACCEPT_EULA=Y",
        "-e",
        "OMNI_KIT_DISABLE_CUP=1",
        "-e",
        "ISAAC_SIM_HEADLESS=1",
        "-e",
        "PYTHONUNBUFFERED=1",
        "-e",
        "PYTHONDONTWRITEBYTECODE=1",
        "-e",
        "WARP_CACHE_PATH=/tmp/jit-cache/warp",
        "-e",
        "CUDA_CACHE_PATH=/tmp/jit-cache/nv",
        "-v",
        f"{artifact_dir}:/tmp/bench_out",
        "-v",
        f"{jit_cache}:/tmp/jit-cache",
        "-v",
        f"{kit_cache}:/isaac-sim/kit/cache",
        "-v",
        f"{source_dir}:/workspace/isaaclab",
        image,
        "-c",
        inner,
    ]

    subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, text=True)
    start = time.monotonic()
    with (artifact_dir / "benchmark.log").open("w", encoding="utf-8") as log_fh:
        log_fh.write(f"$ {' '.join(shlex.quote(part) for part in cmd)}\n\n")
        result = subprocess.run(cmd, cwd=_REPO_ROOT, stdout=log_fh, stderr=subprocess.STDOUT, text=True)
    return result.returncode, time.monotonic() - start


def _run_local_source_benchmark(
    *,
    task,
    artifact_dir: Path,
    source_dir: Path,
    jit_cache: Path,
    kit_cache: Path,
    local_env_dir: Path,
    ld_preload: str,
) -> tuple[int, float]:
    """Run one real IsaacLab benchmark from an isolated clone on the host."""
    _prepare_local_source_runtime(source_dir, local_env_dir)
    for path in (artifact_dir, jit_cache / "warp", jit_cache / "nv", kit_cache):
        path.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(source_dir / "isaaclab.sh"),
        "-p",
        "scripts/benchmarks/benchmark_non_rl.py",
        "--task",
        task.task_id,
        "--num_envs",
        str(task.num_envs),
        "--num_frames",
        str(task.num_frames),
        "--benchmark_backend",
        "JSONFileMetrics",
        "--output_path",
        str(artifact_dir),
    ]
    if task.seed is not None:
        cmd.extend(["--seed", str(task.seed)])
    cmd.extend(hydra_args_for_task(task))

    env = {
        **os.environ,
        "OMNI_KIT_ACCEPT_EULA": "yes",
        "ACCEPT_EULA": "Y",
        "OMNI_KIT_DISABLE_CUP": "1",
        "ISAAC_SIM_HEADLESS": "1",
        "PYTHONUNBUFFERED": "1",
        "WARP_CACHE_PATH": str(jit_cache / "warp"),
        "CUDA_CACHE_PATH": str(jit_cache / "nv"),
    }
    if ld_preload:
        env["LD_PRELOAD"] = ld_preload

    start = time.monotonic()
    with (artifact_dir / "benchmark.log").open("w", encoding="utf-8") as log_fh:
        log_fh.write(f"$ {' '.join(shlex.quote(part) for part in cmd)}\n\n")
        result = subprocess.run(cmd, cwd=source_dir, stdout=log_fh, stderr=subprocess.STDOUT, text=True, env=env)
    return result.returncode, time.monotonic() - start


def _run_build_bench_result(task, artifact_dir: Path, exit_code: int, wall_time_s: float, gate_config: Path) -> None:
    cmd = [
        sys.executable,
        str(_MODULE_DIR / "build_bench_result.py"),
        "--task_id",
        task.task_id,
        "--physics_backend",
        task.physics_backend,
        "--render_backend",
        task.render_backend or "",
        "--artifact_dir",
        str(artifact_dir),
        "--exit_code",
        str(exit_code),
        "--wall_time_s",
        f"{wall_time_s:.1f}",
        "--timeout_s",
        str(task.timeout_minutes * 60),
        "--log_file",
        str(artifact_dir / "benchmark.log"),
        "--launch_config",
        str(artifact_dir / "launch_config.json"),
        "--gate_config",
        str(gate_config),
    ]
    subprocess.run(cmd, cwd=_REPO_ROOT, check=True)


def _write_demo_baseline(
    *,
    baselines_dir: Path,
    task,
    gpu_model: str,
    good_fps: float,
    bench_result: dict,
    sample_count: int,
) -> None:
    launch_config = bench_result.get("launch_config") or {}
    gpu_bucket = canonical_gpu_model(gpu_model)
    for idx in range(sample_count):
        metadata = {
            "schema_version": 1,
            "fps": good_fps,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trusted_source": "bisect_demo",
            "gpu_model": gpu_bucket,
            "task_id": task.task_id,
            "backend_key": task.backend_key,
            "physics_backend": task.physics_backend,
            "render_backend": task.render_backend,
            "commit_sha": f"demo-baseline-{idx}",
            "launch_config_hash": bench_result.get("launch_config_hash") or launch_config.get("launch_config_hash"),
            "benchmark_contract_hash": bench_result.get("benchmark_contract_hash")
            or launch_config.get("benchmark_contract_hash"),
            "runtime_contract_hash": bench_result.get("runtime_contract_hash"),
            "baseline_epoch": bench_result.get("baseline_epoch") or launch_config.get("baseline_epoch", 1),
            "sample_id": f"bisect-demo-{task.task_id}-{task.backend_key}-{gpu_bucket}-{idx}",
        }
        update_baseline(baselines_dir, gpu_bucket, task.task_id, task.backend_key, good_fps, sample_metadata=metadata)


def main() -> int:
    args = _parse_args()
    commit_sha = _resolve_ref(args.commit)
    task = _task_with_overrides(
        _task_from_backend(args.task_id, args.backend_key),
        num_envs=args.override_num_envs,
        num_frames=args.override_num_frames,
    )
    artifact_dir = args.artifact_dir.resolve()
    artifact_root = _artifact_root(artifact_dir)

    launch_config = task_to_launch_config(
        task,
        fps_mean_floor=0.0,
        gpu_model=args.gpu_model,
        hydra_args=hydra_args_for_task(task),
    )
    write_launch_config(artifact_dir, launch_config)

    synthetic_state = None
    fps_mean = None
    if args.mode == "synthetic":
        if not args.first_bad_ref:
            raise ValueError("--first_bad_ref is required for --mode synthetic")
        first_bad_sha = _resolve_ref(args.first_bad_ref)
        is_bad = _is_ancestor(first_bad_sha, commit_sha)
        synthetic_state = "BAD" if is_bad else "GOOD"
        fps_mean = args.bad_fps if is_bad else args.good_fps
        exit_code, wall_time_s = _run_stub_benchmark(task, artifact_dir, fps_mean)
    elif args.mode == "docker-source":
        source_dir = (args.source_dir or (artifact_root / "candidate-source")).resolve()
        jit_cache = (args.jit_cache or (artifact_root / "jit-cache")).resolve()
        kit_cache = (args.kit_cache or (artifact_root / "kit-cache")).resolve()
        _prepare_source_clone(source_dir, commit_sha)
        exit_code, wall_time_s = _run_docker_source_benchmark(
            image=args.image,
            task=task,
            artifact_dir=artifact_dir,
            source_dir=source_dir,
            jit_cache=jit_cache,
            kit_cache=kit_cache,
            commit_sha=commit_sha,
        )
    else:
        source_dir = (args.source_dir or (artifact_root / "candidate-source")).resolve()
        jit_cache = (args.jit_cache or (artifact_root / "jit-cache")).resolve()
        kit_cache = (args.kit_cache or (artifact_root / "kit-cache")).resolve()
        _prepare_source_clone(source_dir, commit_sha)
        exit_code, wall_time_s = _run_local_source_benchmark(
            task=task,
            artifact_dir=artifact_dir,
            source_dir=source_dir,
            jit_cache=jit_cache,
            kit_cache=kit_cache,
            local_env_dir=args.local_env_dir.resolve(),
            ld_preload=args.ld_preload,
        )
    _run_build_bench_result(task, artifact_dir, exit_code, wall_time_s, args.gate_config)

    result_path = artifact_dir / "perf_smoke_test_result.json"
    bench_result = json.loads(result_path.read_text(encoding="utf-8"))
    bench_result["bisect_runner"] = {
        "commit_sha": commit_sha,
        "mode": args.mode,
        "first_bad_sha": _resolve_ref(args.first_bad_ref) if args.first_bad_ref else None,
        "synthetic_state": synthetic_state,
        "fps_mean": fps_mean,
    }
    result_path.write_text(json.dumps(bench_result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.ensure_baseline and args.mode == "synthetic":
        _write_demo_baseline(
            baselines_dir=args.baselines_dir,
            task=task,
            gpu_model=args.gpu_model,
            good_fps=args.good_fps,
            bench_result=bench_result,
            sample_count=args.baseline_samples,
        )

    print(
        f"[bisect_single_commit_runner] {commit_sha[:12]} {task.task_id}/{task.backend_key} "
        f"mode={args.mode} state={synthetic_state or 'measured'} "
        f"fps={fps_mean if fps_mean is not None else 'real'} artifacts={artifact_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
