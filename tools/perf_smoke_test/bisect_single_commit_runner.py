#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Single-commit runner used by the preliminary bisection agent.

The runner has five modes:

* ``synthetic`` writes normal perf-smoke artifacts via the stub benchmark. This is
  fast and useful for demos without a GPU.
* ``docker-source`` checks out one candidate commit into an isolated clone,
  source-mounts it into a fixed IsaacLab CI image, runs one task/backend, and
  emits the same artifact contract.
* ``local-source`` checks out one candidate commit into an isolated clone and
  runs that clone with the host's existing IsaacLab Python environment.
* ``local-reconstruct`` checks out one candidate commit into an isolated clone and
  rebuilds a fully isolated environment for it (its own Isaac Sim + pinned modular
  stack) via :mod:`bisection.env_setup`, then runs that clone with the
  reconstructed environment. This is the faithful, host-independent mode.
* ``docker-reconstruct`` runs that same per-commit ``local-reconstruct`` flow inside
  a hermetic container (``bisection/container/Dockerfile``) for stronger isolation:
  the image bakes no Isaac Sim, and the container's entrypoint invokes this runner
  in ``local-reconstruct`` mode against the mounted candidate checkout.

Every mode uses the same run-scoped, read-only perf-smoke tooling snapshot; candidate
checkouts provide IsaacLab source and runtime dependencies but never select a benchmark
driver or result parser. Every non-container mode writes a ``bisect_env.json`` sidecar into the artifact directory
recording the commit's resolved stack and the environment status (``ok``/``skip``),
so the engine can tell "could not build the environment" apart from "the benchmark
crashed". In ``local-reconstruct`` mode, an environment skip exits cleanly without
producing ``perf_smoke_test_result.json``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import selectors
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

_MODULE_DIR = Path(__file__).parent
_REPO_ROOT = _MODULE_DIR.parents[1]

if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))

from backend_identity import split_backend_key  # noqa: E402
from bisection.env_setup import (  # noqa: E402
    DEFAULT_INSTALL_SCOPE,
    EnvSkip,
    StackSpec,
    ensure_env,
    resolve_stack,
    with_arm_libgomp_preload,
)
from bisection.git_utils import is_ancestor, resolve_ref  # noqa: E402
from bisection.tooling import tooling_bundle_hash  # noqa: E402
from contracts import BenchResult  # noqa: E402
from launch_config import hydra_args_for_task, task_to_launch_config, write_launch_config  # noqa: E402
from task_config import TaskConfig, caches_for_backend, get_task  # noqa: E402
from tooling_capability import TOOLING_INCOMPATIBLE_EXIT_CODE  # noqa: E402

_PROGRESS_PREFIX = "[perf-bisect]"


def _progress(message: str) -> None:
    """Emit a structured inner-runner milestone in verbose mode."""
    if os.environ.get("PERF_BISECT_PROGRESS") == "verbose":
        print(f"{_PROGRESS_PREFIX} {message}", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one commit/task/backend for the perf bisection POC.")
    parser.add_argument("--commit", required=True, help="Commit SHA/ref being tested.")
    parser.add_argument("--task_id", required=True)
    parser.add_argument("--backend_key", required=True)
    parser.add_argument("--artifact_dir", required=True, type=Path)
    parser.add_argument(
        "--mode",
        choices=("synthetic", "docker-source", "local-source", "local-reconstruct", "docker-reconstruct"),
        default="synthetic",
        help=(
            "Runner mode. synthetic is GPU-free; the rest run real IsaacLab. local-reconstruct rebuilds "
            "a fully isolated env (incl. Isaac Sim) per commit on the host; docker-reconstruct runs that "
            "same per-commit reconstruction inside a hermetic container for stronger isolation."
        ),
    )
    parser.add_argument(
        "--first_bad_ref",
        default=None,
        help="Synthetic demo knob: this ref and descendants are emitted as regressed.",
    )
    parser.add_argument(
        "--image",
        default="",
        help="Docker image tag for --mode docker-source (baked IsaacLab CI image) or "
        "--mode docker-reconstruct (hermetic base image built from bisection/container/Dockerfile).",
    )
    parser.add_argument(
        "--harness_root",
        type=Path,
        default=None,
        help="Repo providing the harness tooling + git history mounted into the container for "
        "--mode docker-reconstruct (default: this repo).",
    )
    parser.add_argument(
        "--tooling_root",
        type=Path,
        default=_MODULE_DIR,
        help="Read-only run-scoped perf-smoke tooling snapshot (default: current tooling directory).",
    )
    parser.add_argument("--tooling_spec_hash", default="", help="Pinned tooling contract hash.")
    parser.add_argument("--tooling_bundle_hash", default="", help="Pinned tooling content hash.")
    parser.add_argument("--tooling_contract_id", default="", help="Pinned benchmark/result contract identifier.")
    parser.add_argument("--tooling_source_commit_sha", default="", help="Commit containing the pinned tooling.")
    parser.add_argument("--tooling_authoritative", action="store_true", help="Mark committed-SHA tooling runs.")
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
        "--env_cache_dir",
        type=Path,
        default=None,
        help="Root for reconstructed per-commit environments in local-reconstruct mode "
        "(default: artifact root / env-cache). Shared across candidates so installs amortize.",
    )
    parser.add_argument(
        "--install_scope",
        default=DEFAULT_INSTALL_SCOPE,
        help="./isaaclab.sh -i scope used to reconstruct the environment in local-reconstruct mode.",
    )
    parser.add_argument(
        "--clear_caches",
        action="store_true",
        help="Recovery knob: wipe this candidate's JIT and Kit shader caches before running "
        "(clears stale caches that can make a runnable commit appear broken).",
    )
    parser.add_argument(
        "--force_reinstall",
        action="store_true",
        help="Recovery knob: discard any cached reconstructed environment and rebuild it "
        "from scratch (local-reconstruct mode only).",
    )
    # Inline task definition (Option B): lets a caller bisect a task that is not in
    # tasks.json without editing the shared registry. When the task IS registered,
    # these act as overrides; when it is not, --num_envs is required to build the
    # task inline (num_frames defaults to 300). --hydra_arg (repeatable) replaces
    # the backend-derived presets so any task's Hydra config groups are expressible.
    parser.add_argument(
        "--num_envs",
        type=int,
        default=None,
        help="Inline task.num_envs. Required to bisect a task that is not in tasks.json.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=None,
        help="Inline task.num_frames (defaults to 300 when building an unregistered task inline).",
    )
    parser.add_argument(
        "--warmup_frames",
        type=int,
        default=None,
        help="Leading steps discarded at the source by perf_runtime.py before aggregation "
        "(steady-state warmup exclusion). Overrides the registry value; defaults inline to "
        "min(100, num_frames - 1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Inline task.seed (defaults to 42 when building an unregistered task inline).",
    )
    parser.add_argument(
        "--camera_resolution",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=None,
        help="Inline task.camera_resolution as two ints, e.g. --camera_resolution 64 64.",
    )
    parser.add_argument(
        "--timeout_minutes",
        type=int,
        default=None,
        help="Inline task.timeout_minutes (defaults to 30 when building an unregistered task inline).",
    )
    parser.add_argument(
        "--hydra_arg",
        action="append",
        default=[],
        metavar="ARG",
        help="Extra Hydra CLI arg passed verbatim to the benchmark (repeatable). When any are "
        "given they REPLACE the backend-derived presets, e.g. --hydra_arg presets=cube,newton,rgb64.",
    )
    parser.add_argument(
        "--tasks_json",
        type=Path,
        default=None,
        help="Optional tasks.json registry path used as a fallback when inline task fields are omitted.",
    )
    parser.add_argument("--gpu_model", default="L40S")
    parser.add_argument(
        "--synthetic_metric_path",
        default=None,
        help="Optional dotted result path to populate in synthetic mode for metric-agnostic bisection demos.",
    )
    parser.add_argument("--synthetic_good_value", type=float, default=1000.0)
    parser.add_argument("--synthetic_bad_value", type=float, default=500.0)
    parser.add_argument(
        "--gate_config",
        type=Path,
        default=None,
        help="Gate policy JSON (default: gate_config.json from the pinned tooling snapshot).",
    )
    return parser.parse_args()


def _build_task(args: argparse.Namespace) -> TaskConfig:
    """Resolve the task to run, preferring inline fields over the tasks.json registry.

    Resolution order (Option B — the caller need not edit the shared registry):

    1. If ``(task_id, backend_key)`` is in ``tasks.json``, start from that entry and
       apply any inline fields as overrides. This keeps registered tasks (e.g.
       Cartpole) working with zero extra flags.
    2. Otherwise build the task entirely from inline fields. ``--num_envs`` is then
       required; the rest fall back to sensible defaults (``num_frames=300``,
       ``seed=42``, ``timeout_minutes=30``).
    3. If neither a registry entry nor ``--num_envs`` is available, error with
       guidance rather than guessing.
    """
    identity = split_backend_key(args.backend_key)
    if identity is None:
        raise RuntimeError(f"Cannot parse backend key {args.backend_key!r}")

    num_envs = args.num_envs
    num_frames = args.num_frames
    camera = tuple(args.camera_resolution) if args.camera_resolution is not None else None

    registry_task: TaskConfig | None = None
    try:
        registry_task = get_task(args.task_id, identity.backend_key, args.tasks_json)
    except (KeyError, FileNotFoundError):
        registry_task = None

    if registry_task is not None:
        updates: dict = {}
        if num_envs is not None:
            updates["num_envs"] = num_envs
        if num_frames is not None:
            updates["num_frames"] = num_frames
        if args.warmup_frames is not None:
            updates["warmup_frames"] = args.warmup_frames
        if args.seed is not None:
            updates["seed"] = args.seed
        if camera is not None:
            updates["camera_resolution"] = camera
        if args.timeout_minutes is not None:
            updates["timeout_minutes"] = args.timeout_minutes
        return replace(registry_task, **updates) if updates else registry_task

    if num_envs is None:
        raise SystemExit(
            f"task {args.task_id!r}/{identity.backend_key!r} is not in tasks.json and no inline "
            "definition was supplied. Pass --num_envs (and optionally --num_frames, --seed, "
            "--camera_resolution, --timeout_minutes, --hydra_arg) to bisect it inline, or add it "
            "to a tasks.json referenced with --tasks_json."
        )
    resolved_num_frames = num_frames if num_frames is not None else 300
    resolved_warmup_frames = (
        args.warmup_frames if args.warmup_frames is not None else min(100, max(0, resolved_num_frames - 1))
    )
    return TaskConfig(
        task_id=args.task_id,
        physics_backend=identity.physics_backend,
        render_backend=identity.render_backend,
        preset="inline",
        num_envs=num_envs,
        num_frames=resolved_num_frames,
        warmup_frames=resolved_warmup_frames,
        camera_resolution=camera,
        timeout_minutes=args.timeout_minutes if args.timeout_minutes is not None else 30,
        fps_mean_thresholds={},
        noise_floor_pct={},
        caches=caches_for_backend(identity.physics_backend),
        seed=args.seed if args.seed is not None else 42,
    )


def _resolve_hydra_args(args: argparse.Namespace, task: TaskConfig) -> list[str]:
    """Return the Hydra args for the benchmark launch.

    Inline ``--hydra_arg`` values win verbatim (the caller fully controls presets);
    otherwise the backend-derived defaults from :func:`hydra_args_for_task` are used.
    """
    return list(args.hydra_arg) if args.hydra_arg else hydra_args_for_task(task)


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


def _clear_previous_attempt_outputs(artifact_dir: Path) -> None:
    """Remove stale outputs when an attempt directory is intentionally reused."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    names = (
        "benchmark.log",
        "bisect_env.json",
        "perf_smoke_test_info.json",
        "perf_smoke_test_result.json",
        "tooling.json",
        "tooling_verification.json",
        "tooling_capability.json",
    )
    for name in names:
        with contextlib.suppress(OSError):
            (artifact_dir / name).unlink()
    for pattern in ("benchmark_runtime_*.json",):
        for path in artifact_dir.glob(pattern):
            with contextlib.suppress(OSError):
                path.unlink()


def _clear_caches(*caches: Path) -> None:
    """Wipe the contents of the given cache directories (recovery knob).

    Removes each directory's contents rather than the directory itself so the run
    can recreate the expected subdirectories cleanly.
    """
    for cache in caches:
        if not cache.exists():
            continue
        for child in cache.iterdir():
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child, ignore_errors=True)
            else:
                with contextlib.suppress(OSError):
                    child.unlink()


def _stack_cache_dirs(args: argparse.Namespace, artifact_root: Path, stack: StackSpec) -> tuple[Path, Path]:
    """Return run-scoped JIT and Kit cache directories for one component stack."""
    jit_root = (args.jit_cache or (artifact_root / "jit-cache")).resolve()
    kit_root = (args.kit_cache or (artifact_root / "kit-cache")).resolve()
    return jit_root / stack.stack_hash, kit_root / stack.stack_hash


def _run(cmd: list[str], *, cwd: Path | None = None, check: bool = True, env: dict[str, str] | None = None):
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True, env=env)
    if check and result.returncode != 0:
        raise RuntimeError(f"command failed (exit {result.returncode}): {' '.join(cmd)}")
    return result


def _run_with_live_output(cmd: list[str], *, cwd: Path, log_path: Path) -> int:
    """Run a command, teeing output to ``log_path`` and ``live_output.jsonl``."""
    start = time.monotonic()
    live_path = log_path.parent / "live_output.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_fh, live_path.open("w", encoding="utf-8") as live_fh:
        log_fh.write(f"$ {' '.join(shlex.quote(part) for part in cmd)}\n\n")
        log_fh.flush()
        live_fh.write(
            json.dumps(
                {
                    "event": "process_start",
                    "elapsed_s": 0.0,
                    "command": " ".join(shlex.quote(part) for part in cmd),
                },
                sort_keys=True,
            )
            + "\n"
        )
        live_fh.flush()
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert process.stdout is not None
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)

        def emit(line: str) -> None:
            log_fh.write(line)
            log_fh.flush()
            live_fh.write(
                json.dumps(
                    {
                        "event": "output",
                        "elapsed_s": round(time.monotonic() - start, 3),
                        "line": line.rstrip("\n"),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            live_fh.flush()
            if line.startswith(_PROGRESS_PREFIX):
                print(line, end="" if line.endswith("\n") else "\n", flush=True)

        while True:
            for key, _ in selector.select(timeout=1.0):
                line = key.fileobj.readline()
                if line:
                    emit(line)
            if process.poll() is not None:
                remainder = process.stdout.read()
                if remainder:
                    for line in remainder.splitlines(keepends=True):
                        emit(line)
                break
        selector.close()
        live_fh.write(
            json.dumps(
                {
                    "event": "process_exit",
                    "elapsed_s": round(time.monotonic() - start, 3),
                    "exit_code": process.returncode,
                },
                sort_keys=True,
            )
            + "\n"
        )
        return int(process.returncode or 0)


def _clear_stale_git_locks(source_dir: Path) -> None:
    """Remove stale git lock files left behind by an interrupted git process.

    docker-reconstruct reuses one source clone across a commit's repeated runs and
    force-removes containers during recovery, which can kill a git process mid
    ``fetch``/``checkout`` and orphan ``.git/index.lock`` (or ``shallow.lock``).
    This runner is the only git user of ``source_dir`` and runs sequentially per
    commit, so any lock present on entry is stale and safe to delete. Leaving it
    turns every later ``git checkout`` into a spurious exit-128 (``Unable to create
    '.git/index.lock': File exists``) that the harness misreads as an environment
    skip, aborting an otherwise healthy reference/candidate measurement.
    """
    git_dir = source_dir / ".git"
    if not git_dir.is_dir():
        return
    for lock_name in ("index.lock", "shallow.lock", "HEAD.lock"):
        lock = git_dir / lock_name
        if lock.exists():
            print(f"[bisect_single_commit_runner] removing stale git lock: {lock}", flush=True)
            lock.unlink()


def _reset_source_dir(source_dir: Path) -> None:
    """Force-remove a source clone, defeating root-owned residue from prior docker runs.

    A ``docker-reconstruct`` container clones and installs as root into the
    bind-mounted source dir. If it is killed before this runner's post-clone
    ``chmod`` runs, the residue can be root-owned and unremovable by the (non-root)
    host user. Relaxing permissions best-effort (``chmod -R u+rwX``) before removal
    keeps the reset itself from failing on that residue. The removal ignores errors
    so a healthy re-clone can still proceed even if a few entries linger.
    """
    if not source_dir.exists():
        return
    subprocess.run(["chmod", "-R", "u+rwX", str(source_dir)], check=False)
    shutil.rmtree(source_dir, ignore_errors=True)


def _materialize_source_clone(source_dir: Path, commit_sha: str) -> None:
    """Clone (if absent), fetch, hard-checkout ``commit_sha``, and clean the tree."""
    git_env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
    if not (source_dir / ".git").exists():
        source_dir.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", "--no-checkout", str(_REPO_ROOT.resolve()), str(source_dir)], env=git_env)
    _clear_stale_git_locks(source_dir)
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


def _prepare_source_clone(source_dir: Path, commit_sha: str) -> None:
    """Materialize ``commit_sha`` into a self-contained isolated clone, self-healing once.

    The per-commit clone is reused across a commit's runs and recovery attempts: it is
    the path the cached editable venv is built against (``./isaaclab.sh -i`` installs
    ``source/*`` editable), so it must stay stable per commit rather than churn per
    attempt. Reuse means a prior run's interruption can leave the clone in a state a
    plain checkout cannot recover — a partial/corrupt object store, root-owned residue,
    or a wrong/dirty tree. On any git failure we reset the directory and re-clone once
    from scratch; if that still fails the commit is unevaluable and is surfaced as a
    ``source_checkout_failed`` skip (no ``perf_smoke_test_result.json``) rather than a
    benchmark crash.

    Raises:
        EnvSkip: With category ``source_checkout_failed`` when even a fresh re-clone
            cannot materialize the commit.
    """
    _progress(f"checking out candidate source {commit_sha[:12]}")
    try:
        _materialize_source_clone(source_dir, commit_sha)
        _progress(f"candidate source ready at {source_dir}")
        return
    except RuntimeError as first_error:
        print(
            f"[bisect_single_commit_runner] source checkout failed ({first_error}); "
            f"resetting {source_dir} and re-cloning once",
            flush=True,
        )
    _reset_source_dir(source_dir)
    try:
        _materialize_source_clone(source_dir, commit_sha)
        _progress(f"candidate source ready at {source_dir} after re-clone")
    except RuntimeError as retry_error:
        raise EnvSkip(
            "source_checkout_failed", f"could not materialize {commit_sha[:12]} after re-clone: {retry_error}"
        ) from retry_error


def _prepare_source_or_record_skip(
    source_dir: Path, commit_sha: str, *, artifact_dir: Path, stack: StackSpec, mode: str
) -> bool:
    """Prepare the source clone; on a terminal checkout failure, record a clean skip.

    Returns True when the clone is ready to use. Returns False after writing a
    ``source_checkout_failed`` ``bisect_env.json`` skip (and no perf result) so the
    engine classifies the attempt as an environment skip rather than a benchmark crash.
    """
    try:
        _prepare_source_clone(source_dir, commit_sha)
        return True
    except EnvSkip as skip:
        _write_bisect_env(artifact_dir, stack=stack, mode=mode, status="skip", skip=skip)
        print(
            f"[bisect_single_commit_runner] {commit_sha[:12]} mode={mode} "
            f"ENV_SKIP={skip.category} detail={skip.detail}",
            flush=True,
        )
        return False


def _build_tooling_benchmark_command(
    *,
    tooling_root: Path,
    isaaclab_sh: str | Path,
    task,
    artifact_dir: str | Path,
    hydra_args: list[str],
) -> list[str]:
    """Build the fixed harness-owned perf-smoke benchmark command."""
    cmd = [
        str(isaaclab_sh),
        "-p",
        str(tooling_root / "perf_runtime.py"),
        "--task",
        task.task_id,
        "--num_envs",
        str(task.num_envs),
        "--num_frames",
        str(task.num_frames),
        "--warmup_frames",
        str(task.warmup_frames),
        "--benchmark_formatter",
        "schema",
        "--output_path",
        str(artifact_dir),
    ]
    if task.seed is not None:
        cmd.extend(["--seed", str(task.seed)])
    cmd.extend(hydra_args)
    return cmd


def _build_tooling_capability_command(
    *, tooling_root: Path, isaaclab_sh: str | Path, artifact_dir: str | Path
) -> list[str]:
    """Build the candidate-API capability check command."""
    return [
        str(isaaclab_sh),
        "-p",
        str(tooling_root / "tooling_capability.py"),
        "--output",
        str(Path(artifact_dir) / "tooling_capability.json"),
    ]


def _with_tooling_pythonpath(env: dict[str, str], tooling_root: Path) -> dict[str, str]:
    """Prepend harness-owned modules needed by the external benchmark driver."""
    updated = dict(env)
    existing = updated.get("PYTHONPATH", "")
    updated["PYTHONPATH"] = str(tooling_root) + (f":{existing}" if existing else "")
    return updated


def _write_tooling_sidecar(args: argparse.Namespace, artifact_dir: Path) -> None:
    """Record the fixed measurement-tooling identity before benchmark launch."""
    payload = {
        "tooling_root": str(args.tooling_root.resolve()),
        "tooling_spec_hash": args.tooling_spec_hash,
        "bundle_hash": args.tooling_bundle_hash,
        "contract_id": args.tooling_contract_id,
        "source_commit_sha": args.tooling_source_commit_sha,
        "authoritative": args.tooling_authoritative,
        "driver": "perf_runtime.py",
        "result_builder": "build_bench_result.py",
        "candidate_native_fallback": False,
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "tooling.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _verify_mounted_tooling(tooling_root: Path, expected_hash: str) -> None:
    """Fail before candidate work when the mounted tooling content has drifted."""
    if not expected_hash:
        return
    actual_hash, _ = tooling_bundle_hash(tooling_root)
    if actual_hash != expected_hash:
        raise RuntimeError(
            "tooling_hash_mismatch: mounted perf-smoke tooling does not match the pinned bundle "
            f"({actual_hash} != {expected_hash})"
        )


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


def _set_dotted_path(data: dict, dotted_path: str, value: float) -> None:
    """Set a dotted path in a JSON-like dictionary."""
    target = data
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        next_value = target.get(part)
        if next_value is None:
            next_value = {}
            target[part] = next_value
        if not isinstance(next_value, dict):
            raise TypeError(f"Cannot set nested synthetic metric through non-object path: {dotted_path}")
        target = next_value
    target[parts[-1]] = value


def _run_stub_benchmark(task, artifact_dir: Path, fps_mean: float, *, tooling_root: Path) -> tuple[int, float]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    log_file = artifact_dir / "benchmark.log"
    cmd = [
        sys.executable,
        str(tooling_root / "dev" / "stub_benchmark.py"),
        "--task_id",
        task.task_id,
        "--backend",
        task.backend_key,
        "--num_envs",
        str(task.num_envs),
        "--num_frames",
        str(task.num_frames),
        "--warmup_frames",
        str(task.warmup_frames),
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
    tooling_root: Path,
    task,
    hydra_args: list[str],
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

    benchmark_command = _build_tooling_benchmark_command(
        tooling_root=Path("/tooling"),
        isaaclab_sh="./isaaclab.sh",
        task=task,
        artifact_dir="/tmp/bench_out",
        hydra_args=hydra_args,
    )
    capability_command = _build_tooling_capability_command(
        tooling_root=Path("/tooling"),
        isaaclab_sh="./isaaclab.sh",
        artifact_dir="/tmp/bench_out",
    )
    capability_command_str = " ".join(shlex.quote(arg) for arg in capability_command)
    benchmark_command_str = " ".join(shlex.quote(arg) for arg in benchmark_command)
    inner = (
        "set -e\n"
        "cd /workspace/isaaclab\n"
        "rm -f _isaac_sim\n"
        "ln -s /isaac-sim _isaac_sim\n"
        'export PYTHONPATH="/tooling${PYTHONPATH:+:$PYTHONPATH}"\n'
        f"{capability_command_str}\n"
        f"{benchmark_command_str}\n"
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
        "-v",
        f"{tooling_root}:/tooling:ro",
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
    tooling_root: Path,
    task,
    hydra_args: list[str],
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

    cmd = _build_tooling_benchmark_command(
        tooling_root=tooling_root,
        isaaclab_sh=source_dir / "isaaclab.sh",
        task=task,
        artifact_dir=artifact_dir,
        hydra_args=hydra_args,
    )
    capability_cmd = _build_tooling_capability_command(
        tooling_root=tooling_root,
        isaaclab_sh=source_dir / "isaaclab.sh",
        artifact_dir=artifact_dir,
    )

    env = _with_tooling_pythonpath(
        with_arm_libgomp_preload(
            {
                **os.environ,
                "OMNI_KIT_ACCEPT_EULA": "yes",
                "ACCEPT_EULA": "Y",
                "OMNI_KIT_DISABLE_CUP": "1",
                "ISAAC_SIM_HEADLESS": "1",
                "PYTHONUNBUFFERED": "1",
                "WARP_CACHE_PATH": str(jit_cache / "warp"),
                "CUDA_CACHE_PATH": str(jit_cache / "nv"),
            }
        ),
        tooling_root,
    )
    if ld_preload:
        env["LD_PRELOAD"] = ld_preload

    start = time.monotonic()
    with (artifact_dir / "benchmark.log").open("w", encoding="utf-8") as log_fh:
        log_fh.write(f"$ {' '.join(shlex.quote(part) for part in capability_cmd)}\n\n")
        capability = subprocess.run(
            capability_cmd, cwd=source_dir, stdout=log_fh, stderr=subprocess.STDOUT, text=True, env=env
        )
        if capability.returncode != 0:
            return capability.returncode, time.monotonic() - start
        log_fh.write(f"\n$ {' '.join(shlex.quote(part) for part in cmd)}\n\n")
        result = subprocess.run(cmd, cwd=source_dir, stdout=log_fh, stderr=subprocess.STDOUT, text=True, env=env)
    return result.returncode, time.monotonic() - start


def _run_reconstructed_benchmark(
    *,
    tooling_root: Path,
    task,
    hydra_args: list[str],
    artifact_dir: Path,
    source_dir: Path,
    jit_cache: Path,
    kit_cache: Path,
    env_dir: Path,
) -> tuple[int, float]:
    """Run one benchmark inside a fully reconstructed per-commit environment.

    The clone's own ``isaaclab.sh`` is used so the commit's launcher governs the run;
    we only inject the interpreter via ``VIRTUAL_ENV`` (the reconstructed venv, which
    pip-installed its own Isaac Sim, so no ``_isaac_sim`` symlink is needed).
    """
    for path in (artifact_dir, jit_cache / "warp", jit_cache / "nv", kit_cache):
        path.mkdir(parents=True, exist_ok=True)

    cmd = _build_tooling_benchmark_command(
        tooling_root=tooling_root,
        isaaclab_sh=source_dir / "isaaclab.sh",
        task=task,
        artifact_dir=artifact_dir,
        hydra_args=hydra_args,
    )
    capability_cmd = _build_tooling_capability_command(
        tooling_root=tooling_root,
        isaaclab_sh=source_dir / "isaaclab.sh",
        artifact_dir=artifact_dir,
    )

    env = _with_tooling_pythonpath(
        with_arm_libgomp_preload(
            {
                **os.environ,
                "VIRTUAL_ENV": str(env_dir),
                "PATH": f"{env_dir / 'bin'}:{os.environ.get('PATH', '')}",
                "OMNI_KIT_ACCEPT_EULA": "yes",
                "ACCEPT_EULA": "Y",
                "OMNI_KIT_DISABLE_CUP": "1",
                "ISAAC_SIM_HEADLESS": "1",
                "PYTHONUNBUFFERED": "1",
                "WARP_CACHE_PATH": str(jit_cache / "warp"),
                "CUDA_CACHE_PATH": str(jit_cache / "nv"),
            }
        ),
        tooling_root,
    )
    env.pop("CONDA_PREFIX", None)  # avoid shadowing the reconstructed venv

    start = time.monotonic()
    with (artifact_dir / "benchmark.log").open("w", encoding="utf-8") as log_fh:
        log_fh.write(f"$ {' '.join(shlex.quote(part) for part in capability_cmd)}\n\n")
        capability = subprocess.run(
            capability_cmd, cwd=source_dir, stdout=log_fh, stderr=subprocess.STDOUT, text=True, env=env
        )
        if capability.returncode != 0:
            return capability.returncode, time.monotonic() - start
        log_fh.write(f"\n$ {' '.join(shlex.quote(part) for part in cmd)}\n\n")
        result = subprocess.run(cmd, cwd=source_dir, stdout=log_fh, stderr=subprocess.STDOUT, text=True, env=env)
    return result.returncode, time.monotonic() - start


def _worktree_git_metadata_mounts(harness_root: Path) -> list[str]:
    """Return extra Docker ``-v`` args needed when ``harness_root`` is a Git worktree.

    A normal checkout has a real ``.git`` directory under ``harness_root`` and the
    read-only ``/harness`` bind mount is enough. A Git worktree instead has a small
    ``.git`` *file* whose ``gitdir:`` points at the parent repository's
    ``.git/worktrees/<name>`` path, often an absolute host path. Inside the container,
    Git follows that pointer and fails unless the parent repo's git metadata is mounted
    at the same absolute path. Mounting the common ``.git`` directory read-only keeps
    ``git rev-parse``/``git show`` working without exposing the worktree contents
    outside ``/harness``.
    """
    git_file = harness_root / ".git"
    if not git_file.is_file():
        return []
    try:
        text = git_file.read_text(encoding="utf-8").strip()
    except OSError:
        return []
    if not text.startswith("gitdir:"):
        return []
    git_dir = Path(text.split(":", 1)[1].strip())
    if not git_dir.is_absolute():
        git_dir = (harness_root / git_dir).resolve()
    common_git_dir = git_dir.parents[1] if len(git_dir.parents) >= 2 and git_dir.parent.name == "worktrees" else git_dir
    if not common_git_dir.exists():
        return []
    return ["-v", f"{common_git_dir}:{common_git_dir}:ro"]


def _docker_reconstruct_command(
    *,
    image: str,
    commit_sha: str,
    task_id: str,
    backend_key: str,
    harness_root: Path,
    tooling_root: Path,
    artifact_dir: Path,
    source_dir: Path,
    env_cache_dir: Path,
    jit_cache_root: Path,
    kit_cache_root: Path,
    extra_runner_args: list[str],
    container_name: str,
) -> list[str]:
    """Build the ``docker run`` command for the docker-reconstruct mode.

    The container is the isolation boundary; the entrypoint runs the very same
    ``local-reconstruct`` runner inside it (against the mounted harness + candidate
    checkout), so per-commit reconstruction is reused rather than re-implemented in
    bash. Pure/side-effect-free so it can be unit-tested without Docker.
    """
    # EXTRA_RUNNER_ARGS carries the inner-runner argv. Shell-quote each token so
    # values containing spaces survive as a single argument (e.g. a GPU model like
    # "NVIDIA L40S"): the entrypoint re-parses this with ``eval set --``, which
    # honours the quoting and reconstructs exact argv boundaries. Without quoting,
    # the entrypoint's word splitting would fracture any multi-word value.
    extra = " ".join(shlex.quote(arg) for arg in extra_runner_args)
    git_metadata_mounts = _worktree_git_metadata_mounts(harness_root)
    return [
        "docker",
        "run",
        "--rm",
        "--name",
        container_name,
        "--init",
        "--stop-timeout",
        "10",
        "--gpus",
        "all",
        "--network=host",
        "--security-opt=no-new-privileges:true",
        "--ulimit",
        "nofile=65536:65536",
        "-e",
        f"COMMIT_SHA={commit_sha}",
        "-e",
        f"TASK_ID={task_id}",
        "-e",
        f"BACKEND={backend_key}",
        "-e",
        f"EXTRA_RUNNER_ARGS={extra}",
        "-e",
        "OMNI_KIT_ACCEPT_EULA=yes",
        "-e",
        "ACCEPT_EULA=Y",
        "-e",
        "PYTHONUNBUFFERED=1",
        "-e",
        f"PERF_BISECT_PROGRESS={os.environ.get('PERF_BISECT_PROGRESS', 'quiet')}",
        "-v",
        f"{harness_root}:/harness:ro",
        *git_metadata_mounts,
        "-v",
        f"{tooling_root}:/tooling:ro",
        "-v",
        f"{artifact_dir}:/artifacts",
        "-v",
        f"{source_dir}:/candidate",
        "-v",
        f"{env_cache_dir}:/env-cache",
        "-v",
        f"{jit_cache_root}:/cache/jit-root",
        "-v",
        f"{kit_cache_root}:/cache/kit-root",
        image,
    ]


def _docker_reconstruct_extra_args(args: argparse.Namespace) -> list[str]:
    """Assemble the inner-runner flags forwarded into the container."""
    extra: list[str] = [
        "--install_scope",
        args.install_scope,
        "--gpu_model",
        args.gpu_model,
        "--tooling_root",
        "/tooling",
        "--tooling_spec_hash",
        args.tooling_spec_hash,
        "--tooling_bundle_hash",
        args.tooling_bundle_hash,
        "--tooling_contract_id",
        args.tooling_contract_id,
        "--tooling_source_commit_sha",
        getattr(args, "tooling_source_commit_sha", ""),
    ]
    if getattr(args, "tooling_authoritative", False):
        extra.append("--tooling_authoritative")
    if args.clear_caches:
        extra.append("--clear_caches")
    if args.force_reinstall:
        extra.append("--force_reinstall")
    # Forward the inline task definition so the container's inner runner resolves the
    # same task without depending on a tasks.json entry.
    if args.num_envs is not None:
        extra.extend(["--num_envs", str(args.num_envs)])
    if args.num_frames is not None:
        extra.extend(["--num_frames", str(args.num_frames)])
    if args.warmup_frames is not None:
        extra.extend(["--warmup_frames", str(args.warmup_frames)])
    if args.seed is not None:
        extra.extend(["--seed", str(args.seed)])
    if args.camera_resolution is not None:
        extra.extend(["--camera_resolution", str(args.camera_resolution[0]), str(args.camera_resolution[1])])
    if args.timeout_minutes is not None:
        extra.extend(["--timeout_minutes", str(args.timeout_minutes)])
    for hydra_arg in args.hydra_arg:
        extra.extend(["--hydra_arg", hydra_arg])
    return extra


def _run_docker_reconstruct(args: argparse.Namespace, *, commit_sha: str, task, artifact_dir: Path) -> int:
    """Run one candidate's per-commit reconstruction+benchmark inside a container.

    The inner ``local-reconstruct`` runner writes the full artifact contract
    (``perf_smoke_test_result.json`` + ``bisect_env.json``) directly into the mounted
    artifact dir, so this returns the container's exit code and does no host-side
    result building.
    """
    if not args.image.strip():
        raise ValueError("--image is required for --mode docker-reconstruct")
    artifact_root = _artifact_root(artifact_dir)
    harness_root = (args.harness_root or _REPO_ROOT).resolve()
    tooling_root = args.tooling_root.resolve()
    source_dir = (args.source_dir or (artifact_root / "candidate-source")).resolve()
    env_cache_dir = (args.env_cache_dir or (artifact_root / "env-cache")).resolve()
    jit_cache_root = (args.jit_cache or (artifact_root / "jit-cache")).resolve()
    kit_cache_root = (args.kit_cache or (artifact_root / "kit-cache")).resolve()
    for path in (artifact_dir, source_dir, env_cache_dir, jit_cache_root, kit_cache_root):
        path.mkdir(parents=True, exist_ok=True)
    # Only loosen the mount roots. The env cache may already contain root-owned
    # package files from a previous container run; recursively chmod'ing it from
    # the host can fail before Docker even starts. The container itself owns any
    # recursive cache maintenance it needs.
    subprocess.run(
        [
            "chmod",
            "a+rwX",
            str(artifact_dir),
            str(source_dir),
            str(env_cache_dir),
            str(jit_cache_root),
            str(kit_cache_root),
        ],
        check=False,
    )

    container_name = _safe_component(f"perf-bisect-recon-{commit_sha[:12]}-{task.task_id}-{task.backend_key}")
    cmd = _docker_reconstruct_command(
        image=args.image,
        commit_sha=commit_sha,
        task_id=args.task_id,
        backend_key=args.backend_key,
        harness_root=harness_root,
        tooling_root=tooling_root,
        artifact_dir=artifact_dir,
        source_dir=source_dir,
        env_cache_dir=env_cache_dir,
        jit_cache_root=jit_cache_root,
        kit_cache_root=kit_cache_root,
        extra_runner_args=_docker_reconstruct_extra_args(args),
        container_name=container_name,
    )
    subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, text=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return _run_with_live_output(cmd, cwd=_REPO_ROOT, log_path=artifact_dir / "bisect_command.log")


def _write_bisect_env(
    artifact_dir: Path,
    *,
    stack,
    mode: str,
    status: str,
    env_handle=None,
    skip: EnvSkip | None = None,
) -> None:
    """Write the ``bisect_env.json`` sidecar recording the commit's stack and env status.

    Written for every mode for a uniform contract: non-reconstruct modes record the
    resolved stack with ``status="ok"`` and a null ``env_dir``; ``local-reconstruct``
    records the reconstructed env or, on failure, the skip category/detail.
    """
    artifact_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "commit_sha": stack.commit_sha,
        "mode": mode,
        "status": status,
        "stack_hash": stack.stack_hash,
        "isaacsim_version": stack.isaacsim,
        "python_version": stack.python_version,
        "python_requires": stack.python_requires,
        "env_dir": env_handle.env_dir if env_handle is not None else None,
        "env_reused": env_handle.reused if env_handle is not None else None,
        "skip_category": skip.category if skip is not None else None,
        "skip_detail": skip.detail if skip is not None else None,
    }
    (artifact_dir / "bisect_env.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _run_build_bench_result(
    task,
    artifact_dir: Path,
    exit_code: int,
    wall_time_s: float,
    gate_config: Path,
    *,
    tooling_root: Path,
) -> None:
    cmd = [
        sys.executable,
        str(tooling_root / "build_bench_result.py"),
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
    subprocess.run(cmd, cwd=tooling_root, check=True)


def main() -> int:
    args = _parse_args()
    commit_sha = resolve_ref(_REPO_ROOT, args.commit)
    task = _build_task(args)
    hydra_args = _resolve_hydra_args(args, task)
    artifact_dir = args.artifact_dir.resolve()
    tooling_root = args.tooling_root.resolve()
    required_files = ["build_bench_result.py"]
    if args.mode == "synthetic":
        required_files.append("dev/stub_benchmark.py")
    else:
        required_files.extend(("perf_runtime.py", "tooling_capability.py"))
    for required in required_files:
        if not (tooling_root / required).is_file():
            raise ValueError(f"pinned perf-smoke tooling is missing {required}: {tooling_root}")
    _verify_mounted_tooling(tooling_root, args.tooling_bundle_hash)
    _clear_previous_attempt_outputs(artifact_dir)
    artifact_root = _artifact_root(artifact_dir)

    if args.mode == "docker-reconstruct":
        # The container's inner local-reconstruct runner owns the full artifact
        # contract (launch_config, bisect_env.json, result); just dispatch and relay.
        _progress(f"starting isolated reconstruction container for {commit_sha[:12]}")
        return _run_docker_reconstruct(args, commit_sha=commit_sha, task=task, artifact_dir=artifact_dir)

    launch_config = task_to_launch_config(
        task,
        fps_mean_thresholds=task.thresholds_for(args.gpu_model),
        gpu_model=args.gpu_model,
        hydra_args=hydra_args,
    )
    write_launch_config(artifact_dir, launch_config)
    _write_tooling_sidecar(args, artifact_dir)

    # Resolve the commit's pinned stack once for the bisect_env.json sidecar (cheap,
    # git-only; written for every mode for a uniform contract).
    stack = resolve_stack(_REPO_ROOT, commit_sha)

    synthetic_state = None
    fps_mean = None
    env_handle = None
    jit_cache = None
    kit_cache = None
    if args.mode == "synthetic":
        if not args.first_bad_ref:
            raise ValueError("--first_bad_ref is required for --mode synthetic")
        first_bad_sha = resolve_ref(_REPO_ROOT, args.first_bad_ref)
        is_bad = is_ancestor(_REPO_ROOT, first_bad_sha, commit_sha)
        synthetic_state = "BAD" if is_bad else "GOOD"
        fps_mean = args.synthetic_bad_value if is_bad else args.synthetic_good_value
        exit_code, wall_time_s = _run_stub_benchmark(task, artifact_dir, fps_mean, tooling_root=tooling_root)
    elif args.mode == "docker-source":
        source_dir = (args.source_dir or (artifact_root / "candidate-source")).resolve()
        jit_cache, kit_cache = _stack_cache_dirs(args, artifact_root, stack)
        if args.clear_caches:
            _clear_caches(jit_cache, kit_cache)
        if not _prepare_source_or_record_skip(
            source_dir, commit_sha, artifact_dir=artifact_dir, stack=stack, mode=args.mode
        ):
            return 0
        exit_code, wall_time_s = _run_docker_source_benchmark(
            image=args.image,
            tooling_root=tooling_root,
            task=task,
            hydra_args=hydra_args,
            artifact_dir=artifact_dir,
            source_dir=source_dir,
            jit_cache=jit_cache,
            kit_cache=kit_cache,
            commit_sha=commit_sha,
        )
    elif args.mode == "local-reconstruct":
        source_dir = (args.source_dir or (artifact_root / "candidate-source")).resolve()
        jit_cache, kit_cache = _stack_cache_dirs(args, artifact_root, stack)
        env_cache_dir = (args.env_cache_dir or (artifact_root / "env-cache")).resolve()
        if args.clear_caches:
            _clear_caches(jit_cache, kit_cache)
        if not _prepare_source_or_record_skip(
            source_dir, commit_sha, artifact_dir=artifact_dir, stack=stack, mode=args.mode
        ):
            return 0
        try:
            env_handle = ensure_env(
                stack, source_dir, env_cache_dir, install_scope=args.install_scope, force=args.force_reinstall
            )
        except EnvSkip as skip:
            # Could not build the environment: record the skip and exit cleanly
            # WITHOUT a perf_smoke_test_result.json so the engine classifies it as a
            # skip rather than a benchmark failure.
            _write_bisect_env(artifact_dir, stack=stack, mode=args.mode, status="skip", skip=skip)
            print(
                f"[bisect_single_commit_runner] {commit_sha[:12]} {task.task_id}/{task.backend_key} "
                f"mode={args.mode} ENV_SKIP={skip.category} detail={skip.detail}"
            )
            return 0
        _progress(f"running benchmark for {commit_sha[:12]}")
        exit_code, wall_time_s = _run_reconstructed_benchmark(
            tooling_root=tooling_root,
            task=task,
            hydra_args=hydra_args,
            artifact_dir=artifact_dir,
            source_dir=source_dir,
            jit_cache=jit_cache,
            kit_cache=kit_cache,
            env_dir=Path(env_handle.env_dir),
        )
    else:
        source_dir = (args.source_dir or (artifact_root / "candidate-source")).resolve()
        jit_cache, kit_cache = _stack_cache_dirs(args, artifact_root, stack)
        if args.clear_caches:
            _clear_caches(jit_cache, kit_cache)
        if not _prepare_source_or_record_skip(
            source_dir, commit_sha, artifact_dir=artifact_dir, stack=stack, mode=args.mode
        ):
            return 0
        exit_code, wall_time_s = _run_local_source_benchmark(
            tooling_root=tooling_root,
            task=task,
            hydra_args=hydra_args,
            artifact_dir=artifact_dir,
            source_dir=source_dir,
            jit_cache=jit_cache,
            kit_cache=kit_cache,
            local_env_dir=args.local_env_dir.resolve(),
            ld_preload=args.ld_preload,
        )

    if exit_code == TOOLING_INCOMPATIBLE_EXIT_CODE:
        capability_path = artifact_dir / "tooling_capability.json"
        try:
            capability = json.loads(capability_path.read_text(encoding="utf-8"))
            detail = "; ".join(str(item) for item in capability.get("missing", []))
        except (OSError, TypeError, ValueError):
            detail = "candidate APIs do not satisfy the pinned perf-smoke tooling contract"
        skip = EnvSkip("perf_smoke_tooling_incompatible", detail)
        _write_bisect_env(artifact_dir, stack=stack, mode=args.mode, status="skip", skip=skip)
        print(
            f"[bisect_single_commit_runner] {commit_sha[:12]} PERF_SMOKE_TOOLING_INCOMPATIBLE detail={detail}",
            flush=True,
        )
        return 0

    _write_bisect_env(artifact_dir, stack=stack, mode=args.mode, status="ok", env_handle=env_handle)
    _run_build_bench_result(
        task,
        artifact_dir,
        exit_code,
        wall_time_s,
        args.gate_config or tooling_root / "gate_config.json",
        tooling_root=tooling_root,
    )

    result_path = artifact_dir / "perf_smoke_test_result.json"
    raw_result = json.loads(result_path.read_text(encoding="utf-8"))
    # Validate the producer emitted a well-formed schema-v1 BenchResult so a broken
    # build_bench_result fails loudly here rather than surfacing as a "no metric" skip
    # downstream. from_dict requires the identity fields and drops unknown keys.
    BenchResult.from_dict(raw_result)

    # Bisection provenance goes to a sidecar (NOT into the result) so
    # perf_smoke_test_result.json stays a clean schema-v1 BenchResult artifact that the
    # gate's build_bench_result -> benchmark_result_adapter -> oracle path consumes.
    bisect_runner = {
        "commit_sha": commit_sha,
        "mode": args.mode,
        "first_bad_sha": resolve_ref(_REPO_ROOT, args.first_bad_ref) if args.first_bad_ref else None,
        "synthetic_state": synthetic_state,
        "fps_mean": fps_mean,
        "stack_hash": stack.stack_hash,
        "isaacsim_version": stack.isaacsim,
        "env_dir": env_handle.env_dir if env_handle is not None else None,
        "benchmark_driver": "harness_owned:perf_runtime.py",
        "tooling_spec_hash": args.tooling_spec_hash,
        "tooling_bundle_hash": args.tooling_bundle_hash,
        "tooling_contract_id": args.tooling_contract_id,
        "tooling_source_commit_sha": args.tooling_source_commit_sha,
        "tooling_authoritative": args.tooling_authoritative,
        "jit_cache": str(jit_cache) if jit_cache is not None else None,
        "kit_cache": str(kit_cache) if kit_cache is not None else None,
        "candidate_native_fallback": False,
    }
    if args.mode == "synthetic" and args.synthetic_metric_path:
        # Demo-only: populate an arbitrary numeric metric path so metric-agnostic
        # bisection can be rehearsed without a GPU. This key is intentionally outside
        # the typed BenchResult schema (from_dict drops it); the paired-reference metric
        # reader reads the raw result by dotted path, so it is written back to the raw
        # result and mirrored into the sidecar for provenance.
        if synthetic_state == "BAD":
            synthetic_value = args.synthetic_bad_value
        else:
            synthetic_value = args.synthetic_good_value
        _set_dotted_path(raw_result, args.synthetic_metric_path, float(synthetic_value))
        result_path.write_text(json.dumps(raw_result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        bisect_runner["synthetic_metric_path"] = args.synthetic_metric_path
        bisect_runner["synthetic_metric_value"] = float(synthetic_value)

    (artifact_dir / "bisect_runner.json").write_text(
        json.dumps(bisect_runner, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(
        f"[bisect_single_commit_runner] {commit_sha[:12]} {task.task_id}/{task.backend_key} "
        f"mode={args.mode} state={synthetic_state or 'measured'} "
        f"fps={fps_mean if fps_mean is not None else 'real'} artifacts={artifact_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
