"""
infra/llm_client.py — LLM-agnostic tool-use loop using the openai library.

Works with: Claude (api.anthropic.com/v1), OpenAI, vLLM, NVIDIA NIM, Ollama.
Any provider that exposes an OpenAI-compatible /chat/completions endpoint is supported.

Configuration (env vars):
  BISECT_LLM_MODEL     — model name (default: claude-sonnet-4-6)
  BISECT_LLM_BASE_URL  — base URL  (default: https://api.anthropic.com/v1)
  BISECT_LLM_API_KEY   — API key   (falls back to ANTHROPIC_API_KEY)
"""

from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def make_tool_schema(name: str, description: str, parameters: dict) -> dict:
    """Return an OpenAI-format function-tool schema dict."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------

class LLMClient:
    """
    LLM-agnostic client that drives a tool-use loop over any OpenAI-compatible
    chat completions endpoint.

    Parameters (all optional — env vars are used as defaults):
      model     — model identifier string
      base_url  — base URL for the completions endpoint
      api_key   — API key; falls back to BISECT_LLM_API_KEY or ANTHROPIC_API_KEY
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.model: str = (
            model
            or os.environ.get("BISECT_LLM_MODEL", "claude-sonnet-4-6")
        )

        resolved_base_url: str = (
            base_url
            or os.environ.get("BISECT_LLM_BASE_URL", "https://api.anthropic.com/v1")
        )

        resolved_api_key: str | None = (
            api_key
            or os.environ.get("BISECT_LLM_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")
        )

        self.client = OpenAI(
            base_url=resolved_base_url,
            api_key=resolved_api_key,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_session(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: list[dict],
        tool_dispatch: dict[str, Any],
        *,
        max_tool_rounds: int = 40,
    ) -> str:
        """
        Run the tool-use loop and return the final assistant text.

        The loop:
          1. Send the current message list to the LLM.
          2. If the response contains tool_calls, dispatch each call via
             tool_dispatch[name](**args), append results, and repeat.
          3. If finish_reason is "stop" or "end_turn" (Claude), or there are
             no tool_calls, return message.content.

        On any exception raised during tool dispatch the loop terminates and
        returns a JSON-encoded error string so callers always get a string.

        Args:
          system_prompt:   Content for the initial system message.
          user_prompt:     Content for the initial user message.
          tools:           List of OpenAI-format tool schemas (see make_tool_schema).
          tool_dispatch:   Mapping of tool name -> callable(**kwargs) -> any.
          max_tool_rounds: Maximum number of tool-call rounds before aborting.

        Returns:
          The final assistant text response as a plain string.
        """
        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        for _round in range(max_tool_rounds + 1):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools if tools else None,
                tool_choice="auto" if tools else None,
            )

            choice = response.choices[0]
            message = choice.message
            finish_reason: str = choice.finish_reason or ""

            # Termination conditions: stop / end_turn / no tool calls present
            has_tool_calls = bool(
                message.tool_calls and len(message.tool_calls) > 0
            )

            if finish_reason in ("stop", "end_turn") or not has_tool_calls:
                # Return the assistant's text content (may be None for some models)
                return message.content or ""

            # Append the assistant turn (includes tool_calls metadata)
            messages.append(message.model_dump(exclude_unset=False))

            # Dispatch every requested tool call and collect results
            try:
                tool_result_messages = self._dispatch_tool_calls(
                    message.tool_calls, tool_dispatch
                )
            except Exception as exc:  # noqa: BLE001
                return json.dumps({"error": f"Tool dispatch failed: {exc}"})

            messages.extend(tool_result_messages)

        # max_tool_rounds exhausted — return whatever the last message said
        return message.content or json.dumps(
            {"error": f"max_tool_rounds ({max_tool_rounds}) exhausted without a final response"}
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _dispatch_tool_calls(
        self,
        tool_calls: list[Any],
        tool_dispatch: dict[str, Any],
    ) -> list[dict]:
        """
        Invoke each tool call in tool_calls and return a list of role="tool"
        messages ready to be appended to the conversation.
        """
        result_messages: list[dict] = []

        for tc in tool_calls:
            call_id: str = tc.id
            name: str = tc.function.name
            raw_args: str = tc.function.arguments or "{}"

            try:
                kwargs: dict = json.loads(raw_args)
            except json.JSONDecodeError as exc:
                result_content = json.dumps(
                    {"error": f"Failed to parse tool arguments as JSON: {exc}"}
                )
            else:
                if name not in tool_dispatch:
                    result_content = json.dumps(
                        {"error": f"Unknown tool: {name!r}"}
                    )
                else:
                    try:
                        raw_result = tool_dispatch[name](**kwargs)
                        # Serialise result to string if it is not already one
                        if isinstance(raw_result, str):
                            result_content = raw_result
                        else:
                            result_content = json.dumps(raw_result, default=str)
                    except Exception as exc:  # noqa: BLE001
                        result_content = json.dumps(
                            {"error": f"Tool {name!r} raised: {exc}"}
                        )

            result_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": result_content,
                }
            )

        return result_messages
