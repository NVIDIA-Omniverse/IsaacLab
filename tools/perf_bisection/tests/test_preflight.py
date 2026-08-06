# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GPU-free unit tests for the bisection host preflight checks.

Command execution is injected so the checks are exercised without a GPU, Docker, or any
real subprocess. Covers nvidia-smi parsing, GPU model matching, Docker/base-image gating
per runner mode, and the advisory warnings surfaced for a mismatched or missing host.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

_GATE_DIR = Path(__file__).resolve().parents[1]
if str(_GATE_DIR) not in sys.path:
    sys.path.insert(0, str(_GATE_DIR))

from isaaclab_bisection.bisection.preflight import GpuInfo, parse_nvidia_smi_query, run_preflight  # noqa: E402


class _FakeCommands:
    """Maps a command's first two tokens to a scripted ``(returncode, output)``."""

    def __init__(self, responses: dict[tuple[str, ...], tuple[int, str]]) -> None:
        self.responses = responses
        self.seen: list[list[str]] = []

    def __call__(self, command: list[str]) -> tuple[int, str]:
        self.seen.append(command)
        for prefix, response in self.responses.items():
            if tuple(command[: len(prefix)]) == prefix:
                return response
        return 127, "command not found"


_SMI = ("nvidia-smi",)
_DOCKER_INFO = ("docker", "info")
_DOCKER_INSPECT = ("docker", "image", "inspect")


class TestParseNvidiaSmi:
    """The nvidia-smi CSV parser reads the first GPU row robustly."""

    def test_parses_name_driver_memory(self) -> None:
        info = parse_nvidia_smi_query("NVIDIA L40S, 550.90.07, 46068\n")
        assert info == GpuInfo(name="NVIDIA L40S", driver_version="550.90.07", memory_total_mib=46068)

    def test_empty_output_returns_empty_info(self) -> None:
        assert parse_nvidia_smi_query("") == GpuInfo()

    def test_takes_first_gpu_only(self) -> None:
        info = parse_nvidia_smi_query("NVIDIA L40S, 550.90.07, 46068\nNVIDIA A100, 550.90.07, 81920\n")
        assert info.name == "NVIDIA L40S"


class TestRunPreflight:
    """The preflight report gathers GPU/Docker facts and actionable warnings."""

    def test_synthetic_mode_skips_gpu_check(self) -> None:
        """Synthetic mode needs no GPU, so it must not probe nvidia-smi or warn about it."""
        cmds = _FakeCommands({})
        report = run_preflight(runner_mode="synthetic", command_runner=cmds)
        assert cmds.seen == []
        assert report.gpu_available is False
        assert report.warnings == []

    def test_local_reconstruct_reports_gpu_and_matches_model(self) -> None:
        """A GPU matching the plan's target yields no mismatch warning."""
        cmds = _FakeCommands({_SMI: (0, "NVIDIA L40S, 550.90.07, 46068\n")})
        report = run_preflight(runner_mode="local-reconstruct", expected_gpu_model="L40S", command_runner=cmds)
        assert report.gpu_available is True
        assert report.gpu.memory_total_mib == 46068
        assert report.gpu_model_matches is True
        assert report.warnings == []

    def test_gpu_model_mismatch_warns(self) -> None:
        """A workstation GPU that differs from the CI target GPU must warn about comparability."""
        cmds = _FakeCommands({_SMI: (0, "NVIDIA GeForce RTX 5090, 555.00, 32768\n")})
        report = run_preflight(runner_mode="local-reconstruct", expected_gpu_model="L40S", command_runner=cmds)
        assert report.gpu_model_matches is False
        assert any("does not match" in w for w in report.warnings)

    def test_missing_gpu_warns(self) -> None:
        """A host without a usable GPU must warn before any measurement is attempted."""
        cmds = _FakeCommands({_SMI: (127, "nvidia-smi: not found")})
        report = run_preflight(runner_mode="local-reconstruct", command_runner=cmds)
        assert report.gpu_available is False
        assert any("No GPU detected" in w for w in report.warnings)

    def test_docker_reconstruct_checks_daemon_and_image(self) -> None:
        """docker-reconstruct verifies the daemon is up and the base image is present."""
        cmds = _FakeCommands(
            {
                _SMI: (0, "NVIDIA L40S, 550.90.07, 46068\n"),
                _DOCKER_INFO: (0, "Server Version: 27.0"),
                _DOCKER_INSPECT: (0, "[]"),
            }
        )
        report = run_preflight(
            runner_mode="docker-reconstruct",
            image="isaaclab-bisect:base",
            expected_gpu_model="L40S",
            command_runner=cmds,
        )
        assert report.docker_required is True
        assert report.docker_available is True
        assert report.base_image_present is True
        assert report.warnings == []

    def test_docker_daemon_down_warns_and_skips_image_check(self) -> None:
        """When the daemon is down, warn and do not attempt the (impossible) image inspect."""
        cmds = _FakeCommands(
            {
                _SMI: (0, "NVIDIA L40S, 550.90.07, 46068\n"),
                _DOCKER_INFO: (1, "Cannot connect to the Docker daemon"),
            }
        )
        report = run_preflight(runner_mode="docker-reconstruct", image="isaaclab-bisect:base", command_runner=cmds)
        assert report.docker_available is False
        assert report.base_image_present is None
        assert any("Docker daemon is not reachable" in w for w in report.warnings)
        assert not any(cmd[:3] == list(_DOCKER_INSPECT) for cmd in cmds.seen)

    def test_missing_base_image_warns(self) -> None:
        """A reachable daemon but absent base image warns to build/pull it."""
        cmds = _FakeCommands(
            {
                _SMI: (0, "NVIDIA L40S, 550.90.07, 46068\n"),
                _DOCKER_INFO: (0, "Server Version: 27.0"),
                _DOCKER_INSPECT: (1, "Error: No such image"),
            }
        )
        report = run_preflight(runner_mode="docker-reconstruct", image="isaaclab-bisect:base", command_runner=cmds)
        assert report.base_image_present is False
        assert any("not present locally" in w for w in report.warnings)

    def test_local_mode_does_not_require_docker(self) -> None:
        """A non-container mode must not probe Docker at all."""
        cmds = _FakeCommands({_SMI: (0, "NVIDIA L40S, 550.90.07, 46068\n")})
        report = run_preflight(runner_mode="local-reconstruct", command_runner=cmds)
        assert report.docker_required is False
        assert report.docker_available is None
        assert not any(cmd[:2] == list(_DOCKER_INFO) for cmd in cmds.seen)

    def test_low_disk_space_warns_before_reconstruction(self, tmp_path: Path, monkeypatch) -> None:
        """Preflight warns when the workdir filesystem is too full for cold Isaac Sim installs."""

        def fake_disk_usage(path):
            return SimpleNamespace(total=100 * 1024**3, used=99 * 1024**3, free=1 * 1024**3)

        monkeypatch.setattr("isaaclab_bisection.bisection.preflight.shutil.disk_usage", fake_disk_usage)
        cmds = _FakeCommands({_SMI: (0, "NVIDIA L40S, 550.90.07, 46068\n")})
        report = run_preflight(
            runner_mode="local-reconstruct",
            work_dir=tmp_path / "run",
            min_free_disk_gib=50.0,
            command_runner=cmds,
        )
        assert report.disk_free_gib == 1.0
        assert report.disk_space_ok is False
        assert any("Only 1.0 GiB free" in w for w in report.warnings)
