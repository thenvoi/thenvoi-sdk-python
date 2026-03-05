"""Gemini history converter."""

from __future__ import annotations

from typing import Any

try:
    from google.genai import types
except ImportError as e:
    raise ImportError(
        "Google GenAI dependencies not installed. "
        "Install with: uv add thenvoi-sdk[gemini]"
    ) from e

from thenvoi.core.protocols import HistoryConverter

from ._tool_parsing import parse_tool_call, parse_tool_result

GeminiMessages = list[types.Content]


def _flush_pending_tool_calls(
    messages: GeminiMessages,
    pending_tool_calls: list[types.Part],
) -> None:
    """Flush pending tool calls into one model content message."""
    if pending_tool_calls:
        messages.append(types.Content(role="model", parts=list(pending_tool_calls)))
        pending_tool_calls.clear()


def _flush_pending_tool_results(
    messages: GeminiMessages,
    pending_tool_results: list[types.Part],
) -> None:
    """Flush pending tool results into one user content message."""
    if pending_tool_results:
        messages.append(types.Content(role="user", parts=list(pending_tool_results)))
        pending_tool_results.clear()


class GeminiHistoryConverter(HistoryConverter[GeminiMessages]):
    """
    Convert platform history to Gemini content format.

    Output:
    - text messages -> Content(role="user", parts=[Part.from_text(...)])
    - tool_call events -> Content(role="model", parts=[Part.from_function_call(...)])
    - tool_result events -> Content(role="user", parts=[Part(function_response=...)])
    - this agent's assistant text messages -> Content(role="model", ...) to maintain turn alternation
    """

    def __init__(self, agent_name: str = ""):
        self._agent_name = agent_name

    def set_agent_name(self, name: str) -> None:
        self._agent_name = name

    def convert(self, raw: list[dict[str, Any]]) -> GeminiMessages:
        """Convert platform history to Gemini contents."""
        messages: GeminiMessages = []
        pending_tool_calls: list[types.Part] = []
        pending_tool_results: list[types.Part] = []

        for hist in raw:
            message_type = hist.get("message_type", "text")
            content = hist.get("content", "")

            if message_type == "tool_call":
                _flush_pending_tool_results(messages, pending_tool_results)

                parsed = parse_tool_call(content)
                if parsed:
                    pending_tool_calls.append(
                        types.Part(
                            function_call=types.FunctionCall(
                                id=parsed.tool_call_id,
                                name=parsed.name,
                                args=parsed.args,
                            )
                        )
                    )
                continue

            if message_type == "tool_result":
                _flush_pending_tool_calls(messages, pending_tool_calls)

                parsed = parse_tool_result(content)
                if parsed:
                    response_payload = (
                        {"error": parsed.output}
                        if parsed.is_error
                        else {"output": parsed.output}
                    )
                    pending_tool_results.append(
                        types.Part(
                            function_response=types.FunctionResponse(
                                id=parsed.tool_call_id,
                                name=parsed.name,
                                response=response_payload,
                            )
                        )
                    )
                continue

            if message_type != "text":
                continue

            _flush_pending_tool_calls(messages, pending_tool_calls)
            _flush_pending_tool_results(messages, pending_tool_results)

            role = hist.get("role", "user")
            sender_name = hist.get("sender_name", "")
            if role == "assistant" and sender_name == self._agent_name:
                # Keep own-agent text as model role to maintain turn alternation.
                # Gemini requires strict user/model alternation; skipping this
                # would create consecutive user messages after tool results.
                messages.append(
                    types.Content(
                        role="model", parts=[types.Part.from_text(text=content)]
                    )
                )
                continue

            formatted = f"[{sender_name}]: {content}" if sender_name else content
            messages.append(
                types.Content(role="user", parts=[types.Part.from_text(text=formatted)])
            )

        _flush_pending_tool_calls(messages, pending_tool_calls)
        _flush_pending_tool_results(messages, pending_tool_results)
        return messages
