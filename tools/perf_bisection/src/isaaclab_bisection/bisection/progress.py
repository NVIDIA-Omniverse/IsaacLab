# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Human-readable terminal progress for long-running bisection workflows."""

from __future__ import annotations

import sys
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TextIO

PROGRESS_MODES = ("quiet", "compact", "verbose")
_CONTAINER_PROGRESS_PREFIX = "[perf-bisect]"


def _format_elapsed(seconds: float) -> str:
    """Format elapsed seconds as ``MM:SS`` or ``HH:MM:SS``."""
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def format_metric(value: float, unit: str | None) -> str:
    """Format a measured metric for concise terminal output."""
    magnitude = abs(value)
    if magnitude >= 1000:
        rendered = f"{value:,.1f}"
    elif magnitude >= 10:
        rendered = f"{value:.2f}"
    else:
        rendered = f"{value:.3f}"
    return f"{rendered} {unit}".rstrip() if unit else rendered


@dataclass
class ProgressReporter:
    """Render concise phase updates while preserving detailed artifact logs."""

    mode: str = "quiet"
    stream: TextIO = sys.stderr
    heartbeat_interval_s: float = 60.0
    started_at: float = field(default_factory=time.monotonic)
    _last_heartbeat_at: float = field(init=False)

    def __post_init__(self) -> None:
        if self.mode not in PROGRESS_MODES:
            raise ValueError(f"unsupported progress mode: {self.mode}")
        self._last_heartbeat_at = self.started_at

    @property
    def enabled(self) -> bool:
        """Return whether any human-readable progress should be emitted."""
        return self.mode != "quiet"

    def event(self, phase: str, message: str, *, verbose_only: bool = False) -> None:
        """Print one timestamped progress event."""
        if not self.enabled or (verbose_only and self.mode != "verbose"):
            return
        elapsed = _format_elapsed(time.monotonic() - self.started_at)
        print(f"[{elapsed}] {phase.upper():<11} {message}", file=self.stream, flush=True)

    def heartbeat(self, message: str) -> None:
        """Print a periodic heartbeat when a subprocess remains active."""
        if not self.enabled:
            return
        now = time.monotonic()
        if now - self._last_heartbeat_at < self.heartbeat_interval_s:
            return
        self._last_heartbeat_at = now
        self.event("RUNNING", message)

    def relay(self, line: str) -> None:
        """Relay a structured inner-runner setup event in verbose mode."""
        if self.mode != "verbose":
            return
        text = line.strip()
        if not text.startswith(_CONTAINER_PROGRESS_PREFIX):
            return
        self.event("SETUP", text.removeprefix(_CONTAINER_PROGRESS_PREFIX).strip())


_ACTIVE_REPORTER: ContextVar[ProgressReporter | None] = ContextVar("perf_bisect_progress", default=None)
_QUIET_REPORTER = ProgressReporter()


def configure_progress(mode: str, *, stream: TextIO | None = None) -> ProgressReporter:
    """Configure and return the reporter for the current execution context."""
    reporter = ProgressReporter(mode=mode, stream=stream or sys.stderr)
    _ACTIVE_REPORTER.set(reporter)
    return reporter


def get_progress_reporter() -> ProgressReporter:
    """Return the reporter configured for the current execution context."""
    return _ACTIVE_REPORTER.get() or _QUIET_REPORTER
