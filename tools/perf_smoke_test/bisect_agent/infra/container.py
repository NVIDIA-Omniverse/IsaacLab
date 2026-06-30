"""Container dispatch for the bisect agent.

Each run launches a fresh Docker container from a shared base image
(``bisect-runner:latest`` by default).  The container's filesystem is ephemeral
— every ``docker run`` starts clean, so Commit A's installed packages have zero
effect on Commit B's run.  This gives stronger isolation than a venv (full
filesystem boundary per commit) without the redundant overhead of adding a
venv *inside* the container.

The entrypoint uses ``./isaaclab.sh -i``, the official IsaacLab installer, NOT
bare ``pip install``.  Isaac Sim ships a bundled Python environment with
pre-installed warp, physx, and torch at fixed versions.  Bare ``pip install``
risks silently downgrading or corrupting those bundled packages when a commit's
requirements specify an older version.  ``./isaaclab.sh -i`` targets the correct
Python and handles version conflicts safely.

Dev-mode stubbing is handled entirely by ``core/runner.py`` — this module only
contains the production Docker path.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_in_container(
    sha: str,
    task_id: str,
    backend: str,
    output_dir: Path,
    isaaclab_repo_path: Path,
    *,
    docker_image: str = "bisect-runner:latest",
) -> dict:
    """Run a benchmark for *sha* in an isolated Docker container.

    Parameters
    ----------
    sha:
        Full commit SHA being tested.
    task_id:
        IsaacLab task identifier (e.g. ``Isaac-Velocity-Flat-G1-Direct``).
    backend:
        Backend key (e.g. ``newton``).
    output_dir:
        Host-side directory where artifacts are stored.  Mounted at
        ``/artifacts`` inside the container.  Created if absent.
    isaaclab_repo_path:
        Path to the IsaacLab git repository on the host.  Mounted read-only
        at ``/isaaclab`` inside the container.
    docker_image:
        Docker image tag to run.

    Returns
    -------
    dict
        ``{"exit_code": int, "wall_time_s": float, "artifact_dir": str}``
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    return _run_production(
        sha=sha,
        task_id=task_id,
        backend=backend,
        output_dir=output_dir,
        isaaclab_repo_path=Path(isaaclab_repo_path),
        docker_image=docker_image,
    )


# ---------------------------------------------------------------------------
# Production path: docker run with GPU passthrough
# ---------------------------------------------------------------------------


def _run_production(
    sha: str,
    task_id: str,
    backend: str,
    output_dir: Path,
    isaaclab_repo_path: Path,
    docker_image: str,
) -> dict:
    """Run the commit inside a fresh Docker container.

    Container isolation notes
    -------------------------
    * ``--rm`` ensures the container's ephemeral filesystem is discarded after
      each run — no package pollution between commits.
    * ``--gpus all`` passes through all GPUs (required for Isaac Sim).
    * The IsaacLab repo is mounted **read-only** (``/isaaclab:ro``); the
      entrypoint checks out the target SHA inside the container's writable copy.
    * ``/artifacts`` is a writable bind-mount backed by *output_dir* on the host.

    Entrypoint responsibility
    -------------------------
    The container's ``entrypoint.sh`` must use ``./isaaclab.sh -i`` to install
    IsaacLab into the Isaac Sim bundled Python environment.  Bare ``pip install``
    would risk corrupting bundled warp/physx/torch versions.
    """
    log_path = output_dir / "benchmark.log"

    docker_cmd = [
        "docker", "run",
        "--rm",
        "--gpus", "all",
        # Writable artifacts volume — container writes results here.
        "-v", f"{output_dir}:/artifacts",
        # Read-only repo mount — entrypoint checks out the target SHA.
        "-v", f"{isaaclab_repo_path}:/isaaclab:ro",
        # Commit identity env vars consumed by entrypoint.sh.
        "-e", f"COMMIT_SHA={sha}",
        "-e", f"TASK_ID={task_id}",
        "-e", f"BACKEND={backend}",
        docker_image,
    ]

    t0 = time.monotonic()
    proc = subprocess.run(
        docker_cmd,
        capture_output=True,
        text=True,
    )
    wall_time_s = time.monotonic() - t0

    # Persist combined docker stdout+stderr as benchmark.log so callers
    # (and _extract_run_result) can inspect it for failure-phase detection.
    combined_log = proc.stdout + proc.stderr
    log_path.write_text(combined_log)

    if proc.returncode != 0:
        print(
            f"[container] docker run exited {proc.returncode} for sha={sha[:7]}; "
            f"log written to {log_path}",
            file=sys.stderr,
        )

    return {
        "exit_code": proc.returncode,
        "wall_time_s": round(wall_time_s, 3),
        "artifact_dir": str(output_dir),
    }
