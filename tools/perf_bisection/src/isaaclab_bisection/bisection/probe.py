# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LLM-driven container validation probe for the bisection orchestrator.

The probe is the high-autonomy setup-doctor phase of the orchestrator. It reasons
over live container output, install logs, sidecars, and the intended benchmark
plan to decide whether the container is ready for authoritative benchmarking or
whether more setup/recovery work is needed.

It deliberately does **not** decide GOOD/BAD/UNCLEAR or diagnose the root cause
of a located regression. Those remain outside the setup probe.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .base_image_repair import ALLOWED_APT_PACKAGES
from .llm_client import ChatClient, LLMError
from .security import redact_sensitive_text

PROBE_ACTION_READY = "ready"
PROBE_ACTION_RUN_DEBUG_COMMAND = "run_debug_command"
PROBE_ACTION_REPAIR_BASE_IMAGE = "repair_base_image"
PROBE_ACTION_PLAN_ISSUE = "plan_issue"
PROBE_ACTION_HARNESS_BLOCKED = "harness_blocked"

PROBE_ACTIONS = frozenset(
    {
        PROBE_ACTION_READY,
        PROBE_ACTION_RUN_DEBUG_COMMAND,
        PROBE_ACTION_REPAIR_BASE_IMAGE,
        PROBE_ACTION_PLAN_ISSUE,
        PROBE_ACTION_HARNESS_BLOCKED,
    }
)

_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "container_probe.md"
_RESOLVED_BACKEND_PATTERNS = (
    re.compile(r"resolved physics backend:\s*([A-Za-z0-9_.-]+)", re.IGNORECASE),
    re.compile(r"\bKit started,\s*backend\s*=\s*([A-Za-z0-9_.-]+)", re.IGNORECASE),
)


@dataclass(frozen=True)
class ProbeContext:
    """Container validation context shown to the probe policy."""

    commit_sha: str
    task_id: str
    backend_key: str
    artifact_dir: Path
    plan: dict[str, Any]
    live_output_tail: str = ""
    install_log_tail: str = ""
    benchmark_log_tail: str = ""
    sidecars: dict[str, Any] = field(default_factory=dict)
    attempt: int = 0
    max_attempts: int = 3


@dataclass(frozen=True)
class ProbeDecision:
    """Structured decision emitted by the LLM-driven probe phase."""

    action: str
    reason: str
    command: str | None = None
    apt_packages: list[str] = field(default_factory=list)
    suggested_plan_change: dict[str, Any] = field(default_factory=dict)
    confidence: str = "medium"

    def to_json(self) -> dict[str, Any]:
        """Serialize the decision for probe artifacts/audit logs."""
        return {
            "action": self.action,
            "reason": self.reason,
            "command": self.command,
            "apt_packages": self.apt_packages,
            "suggested_plan_change": self.suggested_plan_change,
            "confidence": self.confidence,
        }


class ProbePolicy(Protocol):
    """Decides what the orchestrator should do during container validation."""

    def decide(self, ctx: ProbeContext) -> ProbeDecision:
        """Return the next probe decision for ``ctx``."""
        ...


def _load_system_prompt() -> str:
    try:
        return _PROMPT_PATH.read_text(encoding="utf-8")
    except OSError:
        return (
            "You are the bisection agent's container validation probe. "
            "Respond with JSON: {action, reason, command, suggested_plan_change, confidence}."
        )


def _context_prompt(ctx: ProbeContext) -> str:
    payload = {
        "commit_sha": ctx.commit_sha,
        "task_id": ctx.task_id,
        "backend_key": ctx.backend_key,
        "artifact_dir": str(ctx.artifact_dir),
        "attempt": ctx.attempt,
        "max_attempts": ctx.max_attempts,
        "plan": ctx.plan,
        "sidecars": ctx.sidecars,
        "live_output_tail": ctx.live_output_tail,
        "install_log_tail": ctx.install_log_tail,
        "benchmark_log_tail": ctx.benchmark_log_tail,
        "allowed_actions": sorted(PROBE_ACTIONS),
        "allowed_apt_packages": sorted(ALLOWED_APT_PACKAGES),
    }
    return redact_sensitive_text(json.dumps(payload, indent=2, sort_keys=True))


def _backend_family(value: str) -> str | None:
    """Return the supported physics-backend family named by ``value``."""
    normalized = value.strip().lower()
    if normalized.startswith("newton") or "newton_mjwarp" in normalized:
        return "newton"
    if normalized.startswith("physx") or "isaacsim_physx" in normalized:
        return "physx"
    return None


def _explicit_backend_mismatch(ctx: ProbeContext) -> tuple[str, str] | None:
    """Detect an explicit resolved physics backend that contradicts the plan."""
    expected = _backend_family(ctx.backend_key)
    if expected is None:
        return None
    evidence = "\n".join((ctx.live_output_tail, ctx.benchmark_log_tail))
    for pattern in _RESOLVED_BACKEND_PATTERNS:
        for match in pattern.finditer(evidence):
            resolved = _backend_family(match.group(1))
            if resolved is not None and resolved != expected:
                return expected, resolved
    return None


@dataclass
class NoProbePolicy:
    """Probe policy that immediately allows benchmarking."""

    def decide(self, ctx: ProbeContext) -> ProbeDecision:
        return ProbeDecision(PROBE_ACTION_READY, "container probe disabled; proceeding to benchmark")


@dataclass
class LLMProbePolicy:
    """LLM-backed probe policy with local validation and conservative fallback."""

    model: str
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    max_attempts: int = 3
    _client: ChatClient = field(init=False, repr=False)
    _system: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._client = ChatClient(model=self.model, base_url=self.base_url, api_key_env=self.api_key_env)
        self._system = _load_system_prompt()

    def decide(self, ctx: ProbeContext) -> ProbeDecision:
        """Ask the model for the next probe action, falling back to blocked."""
        if ctx.attempt >= self.max_attempts:
            return ProbeDecision(
                PROBE_ACTION_HARNESS_BLOCKED,
                f"probe budget exhausted after {ctx.attempt} attempts",
                confidence="high",
            )
        try:
            reply = self._client.complete(self._system, _context_prompt(ctx))
        except LLMError as exc:
            return ProbeDecision(
                PROBE_ACTION_HARNESS_BLOCKED,
                f"probe model unavailable: {exc}",
                confidence="high",
            )
        decision = self._parse(reply)
        if decision is None:
            return ProbeDecision(
                PROBE_ACTION_HARNESS_BLOCKED,
                "probe model returned an invalid decision",
                confidence="high",
            )
        if decision.action == PROBE_ACTION_READY and (mismatch := _explicit_backend_mismatch(ctx)):
            expected, resolved = mismatch
            return ProbeDecision(
                PROBE_ACTION_HARNESS_BLOCKED,
                f"probe model marked the container ready despite an explicit backend mismatch: "
                f"requested {expected}, resolved {resolved}",
                confidence="high",
            )
        return decision

    def _parse(self, reply: str) -> ProbeDecision | None:
        text = reply.strip()
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1 or end < start:
            return None
        try:
            data = json.loads(text[start : end + 1])
        except ValueError:
            return None
        action = data.get("action")
        if action not in PROBE_ACTIONS:
            return None
        return ProbeDecision(
            action=action,
            reason=str(data.get("reason") or "probe decision"),
            command=str(data["command"]) if data.get("command") else None,
            apt_packages=[str(item) for item in data.get("apt_packages", [])],
            suggested_plan_change=dict(data.get("suggested_plan_change") or {}),
            confidence=str(data.get("confidence") or "medium"),
        )
