# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal, model-agnostic chat client for the bisection agent.

The agent's LLM usage is intentionally small: the supervisor only needs to ask a
model for a single structured recovery decision per benchmark failure, not to run
a long autonomous tool loop. This client therefore wraps one OpenAI-compatible
``/chat/completions`` call and returns the assistant's text.

It uses only the standard library (``urllib``) so the harness adds no new
dependency and stays open-source friendly: any provider exposing an
OpenAI-compatible endpoint (OpenAI, vLLM, Ollama, a gateway, etc.) works by
setting ``--base_url`` and the API-key environment variable.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"


class LLMError(RuntimeError):
    """Raised when the model endpoint cannot be reached or returns no usable text."""


@dataclass
class ChatClient:
    """A single-call OpenAI-compatible chat client.

    Args:
        model: Model name passed through to the endpoint.
        base_url: OpenAI-compatible base URL; defaults to :data:`DEFAULT_BASE_URL`.
        api_key_env: Environment variable holding the API key.
        timeout_s: Per-request timeout.
        temperature: Sampling temperature (kept low for stable, structured output).
    """

    model: str
    base_url: str | None = None
    api_key_env: str = DEFAULT_API_KEY_ENV
    timeout_s: int = 60
    temperature: float = 0.0

    def complete(self, system: str, user: str) -> str:
        """Return the assistant message for a system+user prompt pair.

        Raises:
            LLMError: If the endpoint is unreachable or the response has no content.
        """
        base = (self.base_url or DEFAULT_BASE_URL).rstrip("/")
        api_key = os.environ.get(self.api_key_env, "")
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        request = urllib.request.Request(
            f"{base}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            raise LLMError(f"chat completion request failed: {exc}") from exc
        try:
            return body["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"chat completion response missing content: {body}") from exc
