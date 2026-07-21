# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LLM-backed recovery policy for the bisection agent.

This is the persistent supervisor's foothold: when a benchmark fails for a
practical reason (stale cache, transient download, headless quirk, first-run
compilation timeout), a model reads the failure's log tail and environment
sidecar and picks the *next recovery step*. Crucially, it chooses only among the
same bounded actions the deterministic policy uses -- it never emits a
GOOD/BAD/UNCLEAR verdict, never invents a first-bad commit, and never edits
source. The deterministic core still owns every verdict and the search path.

Guardrails keep the model honest and keep ``--no-llm`` behaviour recoverable:

* the model's action must be one of the known recovery actions; anything else
  falls back to the deterministic policy's decision for the same context;
* if the endpoint is unreachable or the reply cannot be parsed, we fall back;
* the retry budget is enforced here, not by the model, so a chatty model cannot
  loop forever.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .llm_client import ChatClient, LLMError
from .recovery import (
    ACTION_ACCEPT,
    RETRY_ACTIONS,
    DeterministicRecoveryPolicy,
    RecoveryContext,
    RecoveryDecision,
    _skip_category_for,
)

_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "recovery.md"

_VALID_ACTIONS = RETRY_ACTIONS | {ACTION_ACCEPT}


def _load_system_prompt() -> str:
    """Load the recovery system prompt, tolerating an absent file."""
    try:
        return _PROMPT_PATH.read_text(encoding="utf-8")
    except OSError:
        return (
            "You choose the next recovery step for a failed IsaacLab benchmark run. "
            "Respond with a JSON object {action, reason, skip_category}."
        )


def _context_prompt(ctx: RecoveryContext, *, budget_left: int) -> str:
    """Render the failure context the model reasons over."""
    env_status = None
    if ctx.env_status is not None:
        env_status = {"status": ctx.env_status[0], "skip_category": ctx.env_status[1], "detail": ctx.env_status[2]}
    payload = {
        "commit_sha": ctx.commit_sha,
        "label": ctx.label,
        "run_idx": ctx.run_idx,
        "recovery_attempt": ctx.attempt,
        "retries_left": budget_left,
        "note": ctx.note,
        "exit_code": ctx.exit_code,
        "timed_out": ctx.timed_out,
        "env_status": env_status,
        "log_tail": ctx.log_tail,
    }
    return json.dumps(payload, indent=2, sort_keys=True)


@dataclass
class LLMRecoveryPolicy:
    """Recovery policy that asks a model for the next step, with deterministic fallback.

    Args:
        model: Model name for the OpenAI-compatible endpoint.
        base_url: Endpoint base URL (defaults to the provider default).
        max_attempts: Retry budget; beyond it the outcome is accepted as a skip.
        api_key_env: Environment variable holding the API key.
    """

    model: str
    base_url: str | None = None
    max_attempts: int = 2
    api_key_env: str = "OPENAI_API_KEY"
    _client: ChatClient = field(init=False, repr=False)
    _fallback: DeterministicRecoveryPolicy = field(init=False, repr=False)
    _system: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._client = ChatClient(model=self.model, base_url=self.base_url, api_key_env=self.api_key_env)
        self._fallback = DeterministicRecoveryPolicy(max_attempts=self.max_attempts)
        self._system = _load_system_prompt()

    def decide(self, ctx: RecoveryContext) -> RecoveryDecision:
        """Ask the model for the next recovery step, falling back deterministically."""
        # Budget is enforced locally so the model cannot loop forever.
        if ctx.attempt >= self.max_attempts:
            return RecoveryDecision(
                ACTION_ACCEPT,
                f"recovery budget exhausted after {ctx.attempt} retries",
                _skip_category_for(ctx.note, ctx.log_tail),
            )
        # A definitively unavailable pin is never retried, model or not.
        if (ctx.note or "").startswith("env_skip:dependency_unavailable"):
            return self._fallback.decide(ctx)

        try:
            reply = self._client.complete(
                self._system, _context_prompt(ctx, budget_left=self.max_attempts - ctx.attempt)
            )
        except LLMError:
            return self._fallback.decide(ctx)

        decision = self._parse(reply, ctx)
        return decision if decision is not None else self._fallback.decide(ctx)

    def _parse(self, reply: str, ctx: RecoveryContext) -> RecoveryDecision | None:
        """Parse and validate a model reply into a bounded decision, or None to fall back."""
        text = reply.strip()
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1 or end < start:
            return None
        try:
            data = json.loads(text[start : end + 1])
        except ValueError:
            return None
        action = data.get("action")
        if action not in _VALID_ACTIONS:
            return None
        reason = str(data.get("reason") or "llm recovery decision")
        if action == ACTION_ACCEPT:
            skip_category = data.get("skip_category") or _skip_category_for(ctx.note, ctx.log_tail)
            return RecoveryDecision(ACTION_ACCEPT, reason, str(skip_category))
        return RecoveryDecision(action, reason)
