# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Preflight environment checks for a bisection run.

A perf bisection is only as trustworthy as the host it runs on: a mismatched GPU, a
missing driver, too little VRAM, an unreachable Docker daemon, or an absent base image
all invalidate (or block) the run before a single commit is measured. This module
gathers those host facts up front, records them as a ``preflight.json`` artifact, and
emits actionable warnings so a user sees "your GPU is a 5090 but the plan targets an
L40S" or "the Docker daemon is not reachable" immediately rather than after a long,
wasted reconstruction.

Preflight is advisory: it records facts and warnings and never fails the run itself.
Genuine hard blockers still surface through the runner's skip categories (see
:mod:`bisection.recovery`) with the same vocabulary used here.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

# A command runner returns ``(returncode, combined_output)``. Injectable so the checks
# are unit-testable without a GPU, Docker, or any real subprocess.
CommandRunner = Callable[[list[str]], "tuple[int, str]"]

# Modes that build/run inside a container and therefore need Docker (and, for
# ``docker-reconstruct``, a base image) available on the host.
_DOCKER_MODES = frozenset({"docker-source", "docker-reconstruct"})


def default_command_runner(command: list[str]) -> tuple[int, str]:
    """Run ``command`` and return ``(returncode, combined_stdout_stderr)``.

    Never raises: a missing binary (``FileNotFoundError``) or a timeout is reported as a
    nonzero return code with the error text, so preflight degrades to a warning rather
    than crashing the run.
    """
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
    except FileNotFoundError as exc:
        return 127, str(exc)
    except subprocess.SubprocessError as exc:
        return 1, str(exc)
    return result.returncode, (result.stdout or "") + (result.stderr or "")


@dataclass(frozen=True)
class GpuInfo:
    """Identity of the first visible GPU, as reported by ``nvidia-smi``."""

    name: str | None = None
    driver_version: str | None = None
    memory_total_mib: int | None = None

    def to_json(self) -> dict:
        """Serialize the GPU info."""
        return asdict(self)


def parse_nvidia_smi_query(output: str) -> GpuInfo:
    """Parse ``nvidia-smi --query-gpu=name,driver_version,memory.total`` CSV output.

    Reads the first GPU row. Returns an empty :class:`GpuInfo` if no parsable row is
    present so a malformed/absent output degrades gracefully.
    """
    for line in output.splitlines():
        line = line.strip()
        if not line or line.lower().startswith("name"):
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 3:
            continue
        name = fields[0] or None
        driver = fields[1] or None
        mem_match = re.search(r"\d+", fields[2])
        memory = int(mem_match.group()) if mem_match else None
        return GpuInfo(name=name, driver_version=driver, memory_total_mib=memory)
    return GpuInfo()


@dataclass(frozen=True)
class PreflightReport:
    """Host readiness facts and warnings gathered before measurement begins."""

    runner_mode: str
    gpu_available: bool
    gpu: GpuInfo = field(default_factory=GpuInfo)
    expected_gpu_model: str | None = None
    gpu_model_matches: bool | None = None
    docker_required: bool = False
    docker_available: bool | None = None
    base_image: str | None = None
    base_image_present: bool | None = None
    work_dir: str | None = None
    disk_free_gib: float | None = None
    min_free_disk_gib: float | None = None
    disk_space_ok: bool | None = None
    warnings: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        """Serialize the preflight report."""
        payload = asdict(self)
        payload["gpu"] = self.gpu.to_json()
        return payload


def _gpu_model_matches(expected: str | None, actual: str | None) -> bool | None:
    """Return whether ``actual`` GPU name satisfies the ``expected`` model, or None.

    The comparison is a case-insensitive substring test in both directions because
    ``nvidia-smi`` names are verbose (``NVIDIA L40S``) while plans often carry a short
    key (``L40S``). ``None`` when either side is unknown.
    """
    if not expected or not actual:
        return None
    expected_norm = expected.strip().lower()
    actual_norm = actual.strip().lower()
    return expected_norm in actual_norm or actual_norm in expected_norm


def run_preflight(
    *,
    runner_mode: str,
    image: str | None = None,
    expected_gpu_model: str | None = None,
    work_dir: Path | str | None = None,
    min_free_disk_gib: float = 50.0,
    command_runner: CommandRunner | None = None,
) -> PreflightReport:
    """Gather host readiness facts for ``runner_mode`` into a :class:`PreflightReport`.

    Checks the visible GPU (name/driver/VRAM), free disk where the workdir/cache will
    live, and, for container modes, whether the Docker daemon is reachable and (for
    ``docker-reconstruct``) whether ``image`` is present locally. Populates ``warnings``
    with human-readable, actionable notes; the report is advisory and never raises.
    """
    run = command_runner or default_command_runner
    warnings: list[str] = []

    work_dir_text: str | None = None
    disk_free_gib: float | None = None
    disk_space_ok: bool | None = None
    if work_dir is not None:
        work_dir_path = Path(work_dir)
        work_dir_text = str(work_dir_path)
        disk_probe = work_dir_path if work_dir_path.exists() else work_dir_path.parent
        try:
            usage = shutil.disk_usage(disk_probe)
            disk_free_gib = usage.free / (1024**3)
            disk_space_ok = disk_free_gib >= min_free_disk_gib
        except OSError:
            disk_space_ok = None
        if disk_space_ok is False:
            warnings.append(
                f"Only {disk_free_gib:.1f} GiB free where the bisection work_dir/cache will live "
                f"({work_dir_text}); cold Isaac Sim reconstruction can require tens of GiB. "
                f"Use a filesystem with at least {min_free_disk_gib:.0f} GiB free."
            )

    gpu = GpuInfo()
    gpu_available = False
    if runner_mode != "synthetic":
        rc, out = run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"])
        if rc == 0:
            gpu = parse_nvidia_smi_query(out)
            gpu_available = gpu.name is not None
        if not gpu_available:
            warnings.append(
                "No GPU detected via nvidia-smi; benchmark measurements need a CUDA-capable GPU on this host."
            )

    gpu_model_matches = _gpu_model_matches(expected_gpu_model, gpu.name)
    if gpu_model_matches is False:
        warnings.append(
            f"Host GPU {gpu.name!r} does not match the plan's target GPU {expected_gpu_model!r}; "
            "baselines/regressions may not be comparable to the target hardware."
        )

    docker_required = runner_mode in _DOCKER_MODES
    docker_available: bool | None = None
    base_image_present: bool | None = None
    if docker_required:
        rc, _ = run(["docker", "info"])
        docker_available = rc == 0
        if not docker_available:
            warnings.append(
                "Docker daemon is not reachable, but the runner mode requires it; "
                "start Docker (or fix socket permissions) before running."
            )
        elif runner_mode == "docker-reconstruct" and image:
            rc, _ = run(["docker", "image", "inspect", image])
            base_image_present = rc == 0
            if not base_image_present:
                warnings.append(
                    f"Base image {image!r} is not present locally; build it (see the docker-reconstruct "
                    "recipe) or pull/tag it before running."
                )

    return PreflightReport(
        runner_mode=runner_mode,
        gpu_available=gpu_available,
        gpu=gpu,
        expected_gpu_model=expected_gpu_model,
        gpu_model_matches=gpu_model_matches,
        docker_required=docker_required,
        docker_available=docker_available,
        base_image=image if docker_required else None,
        base_image_present=base_image_present,
        work_dir=work_dir_text,
        disk_free_gib=round(disk_free_gib, 2) if disk_free_gib is not None else None,
        min_free_disk_gib=min_free_disk_gib,
        disk_space_ok=disk_space_ok,
        warnings=warnings,
    )
