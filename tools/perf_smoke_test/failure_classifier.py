# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Self-contained benchmark failure classification for pinned perf-smoke tooling."""

from __future__ import annotations

_PINNED_API_SYMBOLS = (
    "launch_simulation",
    "setup_preset_cli",
    "resolve_task_config",
    "BaseIsaacLabBenchmark",
    "BenchmarkMonitor",
    "run_runtime_loop",
)


def looks_tooling_incompatible(combined_output: str, exit_code: int) -> bool:
    """Conservatively recognize pinned-driver API failures before stepping."""
    if "PERF_SMOKE_TOOLING_INCOMPATIBLE" in combined_output:
        return True
    if exit_code == 0 or "Step Frametimes" in combined_output or "perf_runtime.py" not in combined_output:
        return False
    api_error = any(token in combined_output for token in ("ImportError:", "AttributeError:", "TypeError:"))
    return api_error and any(symbol in combined_output for symbol in _PINNED_API_SYMBOLS)


def classify_failure_phase(
    stdout: str,
    stderr: str,
    exit_code: int,
    wall_time_s: float,
    timeout_s: float,
) -> str | None:
    """Classify the phase in which a benchmark failed."""
    combined = stdout + stderr
    if looks_tooling_incompatible(combined, exit_code):
        return "tooling_incompatible"
    if exit_code == 137 or "oom-kill" in stderr:
        return "oom"
    if wall_time_s >= timeout_s * 0.95:
        return "hang"
    if "Traceback" in combined and "AppLauncher" not in stdout:
        return "import"
    if "CudaError" in combined or "CUDA_ERROR_" in combined:
        return "driver"
    if exit_code != 0 and "AppLauncher initialization complete" in stdout and "Step Frametimes" not in stdout:
        return "init"
    if exit_code != 0 and "Step Frametimes" in stdout:
        return "runtime"
    return None
