# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import json
import shutil
import statistics
import subprocess
import tempfile
from pathlib import Path

from oracle import Baseline  # noqa: E402

DEFAULT_K_WARN = 2.5  # TODO: decide on real threshold values
DEFAULT_K_BLOCK = 4.0 # TODO: decide on real threshold values


def _stats_path(baselines_dir: Path, gpu_model: str, task_id: str, backend: str, fingerprint=None) -> Path:
    if fingerprint is None:
        return baselines_dir / gpu_model / task_id / backend / "stats.json"
    return baselines_dir / gpu_model / task_id / backend / fingerprint / "stats.json"


def _window_path(baselines_dir: Path, gpu_model: str, task_id: str, backend: str, fingerprint=None) -> Path:
    if fingerprint is None:
        return baselines_dir / gpu_model / task_id / backend / "window.ndjson"
    return baselines_dir / gpu_model / task_id / backend / fingerprint / "window.ndjson"


def load_baseline(baselines_dir: Path, gpu_model: str, task_id: str, backend: str, fingerprint=None) -> Baseline | None:
    """Load stats.json for a task/backend pair, or return None if it does not exist"""
    sp = _stats_path(baselines_dir, gpu_model, task_id, backend, fingerprint=fingerprint)
    if not sp.exists():
        return None
    with sp.open() as fh:
        d = json.load(fh)
    return Baseline(
        median_fps=d["median_fps"],
        mad_fps=d["mad_fps"],
        k_warn=d.get("k_warn", DEFAULT_K_WARN),
        k_block=d.get("k_block", DEFAULT_K_BLOCK),
        sample_count=d.get("sample_count", 0),
    )


def update_baseline(baselines_dir: Path, gpu_model: str, task_id: str, backend: str, fps: float, fingerprint=None) -> None:
    """Append fps to window.ndjson, recompute stats, and write stats.json"""
    wp = _window_path(baselines_dir, gpu_model, task_id, backend, fingerprint=fingerprint)
    sp = _stats_path(baselines_dir, gpu_model, task_id, backend, fingerprint=fingerprint)
    wp.parent.mkdir(parents=True, exist_ok=True)

    fps_window: list[float] = []
    if wp.exists():
        with wp.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    fps_window.append(float(line))

    fps_window.append(fps)

    with wp.open("a") as fh:
        fh.write(f"{fps}\n")

    median = statistics.median(fps_window)
    deviations = [abs(v - median) for v in fps_window]
    mad = statistics.median(deviations) if len(deviations) > 1 else 0.0

    stats = {
        "median_fps": median,
        "mad_fps": mad,
        "k_warn": DEFAULT_K_WARN,
        "k_block": DEFAULT_K_BLOCK,
        "sample_count": len(fps_window),
    }
    with sp.open("w") as fh:
        json.dump(stats, fh, indent=2)


def delete_baseline_files(baselines_dir: Path, gpu_model: str, task_id: str, backend: str, fingerprint=None) -> None:
    """Delete stats.json and window.ndjson for a task/backend pair"""
    for p in (
        _stats_path(baselines_dir, gpu_model, task_id, backend, fingerprint=fingerprint),
        _window_path(baselines_dir, gpu_model, task_id, backend, fingerprint=fingerprint),
    ):
        if p.exists():
            p.unlink()


def seed_baseline_with_spread(
    baselines_dir: Path,
    gpu_model: str,
    task_id: str,
    backend: str,
    center_fps: float,
    noise_fps: float = 5.0,
    n_samples: int = 10,
    seed: int = 0,
    fingerprint=None,
) -> None:
    """Populate the baseline window with n_samples of varied FPS data around center_fps and compute stats.json. 
    For testing tasks/backends with no existing baseline or when a deterministic baseline is needed."""
    import random as _random

    rng = _random.Random(seed)
    delete_baseline_files(baselines_dir, gpu_model, task_id, backend, fingerprint=fingerprint)
    for _ in range(n_samples):
        fps = max(1.0, rng.gauss(center_fps, noise_fps))
        update_baseline(baselines_dir, gpu_model, task_id, backend, fps, fingerprint=fingerprint)


def _git_show_file(branch: str, rel_path: str) -> str | None:
    try:
        r = subprocess.run(
            ["git", "show", f"{branch}:{rel_path}"],
            capture_output=True,
            text=True,
            check=True,
            cwd=str(Path(__file__).parent),
        )
        return r.stdout
    except subprocess.CalledProcessError:
        return None


@contextlib.contextmanager
def baseline_worktree(branch: str):
    tmpdir = tempfile.mkdtemp(prefix="perf-bl-wt-")
    committed = False
    try:
        subprocess.run(
            ["git", "worktree", "add", tmpdir, branch],
            check=True,
            capture_output=True,
            cwd=str(Path(__file__).parent),
        )
        yield Path(tmpdir)
        status = subprocess.run(
            ["git", "-C", tmpdir, "status", "--porcelain"],
            capture_output=True,
            text=True,
        )
        if status.stdout.strip():
            subprocess.run(["git", "-C", tmpdir, "add", "-A"], check=True, capture_output=True)
            subprocess.run(
                ["git", "-C", tmpdir, "commit", "-m", "[baseline_manager] Update baselines"],
                check=True,
                capture_output=True,
            )
            committed = True
    except subprocess.CalledProcessError as exc:
        if b"not found" in (exc.stderr or b"") or b"unknown" in (exc.stderr or b""):
            raise RuntimeError(
                f"Baseline branch {branch!r} not found. "
                "Run: git fetch origin angehu/perf-baselines   (or use --force-flat-baseline)" # TODO: replace w/ real branch name
            ) from exc
        raise
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", tmpdir],
            cwd=str(Path(__file__).parent),
            capture_output=True,
        )
        shutil.rmtree(tmpdir, ignore_errors=True)
    if committed:
        print(f"[baseline_manager]   -> committed baseline update to {branch!r}")


def load_baseline_git(branch: str, gpu_model: str, task_id: str, backend: str, fingerprint: str | None) -> Baseline | None:
    rel = str(_stats_path(Path(""), gpu_model, task_id, backend, fingerprint))
    content = _git_show_file(branch, rel)
    if content is None:
        return None
    d = json.loads(content)
    return Baseline(
        median_fps=d["median_fps"],
        mad_fps=d["mad_fps"],
        k_warn=d.get("k_warn", DEFAULT_K_WARN),
        k_block=d.get("k_block", DEFAULT_K_BLOCK),
        sample_count=d.get("sample_count", 0),
    )


def update_baseline_git(
    branch: str, gpu_model: str, task_id: str, backend: str, fps: float, fingerprint: str | None
) -> None:
    with baseline_worktree(branch) as wt:
        update_baseline(wt, gpu_model, task_id, backend, fps, fingerprint)


def seed_baseline_with_spread_git(
    branch: str,
    gpu_model: str,
    task_id: str,
    backend: str,
    center_fps: float,
    noise_fps: float,
    n_samples: int,
    seed: int,
    fingerprint: str | None,
) -> None:
    with baseline_worktree(branch) as wt:
        seed_baseline_with_spread(wt, gpu_model, task_id, backend, center_fps, noise_fps, n_samples, seed, fingerprint)


def delete_baseline_files_git(branch: str, gpu_model: str, task_id: str, backend: str, fingerprint: str | None) -> None:
    with baseline_worktree(branch) as wt:
        delete_baseline_files(wt, gpu_model, task_id, backend, fingerprint)
