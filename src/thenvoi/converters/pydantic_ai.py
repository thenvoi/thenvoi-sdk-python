"""Pydantic AI history converter."""

from __future__ import annotations

import json
import logging
from typing import Any, Union

try:
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        ToolCallPart,
        ToolReturnPart,
        UserPromptPart,
    )
except ImportError as e:
    raise ImportError(
        "Pydantic AI dependencies not installed. "
        "Install with: uv add thenvoi-sdk[pydantic_ai]"
    ) from e

from thenvoi.core.protocols import HistoryConverter

logger = logging.getLogger(__name__)

# Type alias for Pydantic AI messages (can be requests or responses)
PydanticAIMessages = list[Union[ModelRequest, ModelResponse]]


class PydanticAIHistoryConverter(HistoryConverter[PydanticAIMessages]):
    """
    Converts platform history to Pydantic AI message format.

    Output:
    - user messages → ModelRequest with UserPromptPart
    - other agents' messages → ModelRequest with UserPromptPart (with [name] prefix)
    - tool_call → ModelResponse with ToolCallPart
    - tool_result → ModelRequest with ToolReturnPart
    - this agent's text messages → skipped (redundant with tool results)

    Tool events are stored in platform as JSON:
    - tool_call: {"name": "...", "args": {...}, "tool_call_id": "..."}
    - tool_result: {"name": "...", "output": "...", "tool_call_id": "..."}
    """

    def __init__(self, agent_name: str = ""):
        """
        Initialize converter.

        Args:
            agent_name: Name of this agent. Messages from this agent are skipped
                       (they're redundant with tool results). Messages from other
                       agents are included as ModelRequest.
        """
        self._agent_name = agent_name

    def set_agent_name(self, name: str) -> None:
        """
        Set agent name so converter knows which messages to skip.

        Args:
            name: Name of this agent
        """
        self._agent_name = name

    def convert(self, raw: list[dict[str, Any]]) -> PydanticAIMessages:
        """Convert platform history to Pydantic AI format."""
        messages: PydanticAIMessages = []
        # Collect tool calls to batch them into a single ModelResponse
        pending_tool_calls: list[ToolCallPart] = []

        for hist in raw:
            message_type = hist.get("message_type", "text")
            content = hist.get("content", "")

            if message_type == "tool_call":
                # Parse tool call JSON and collect for batching
                try:
                    event = json.loads(content)
                    tool_call_part = ToolCallPart(
                        tool_name=event.get("name", "unknown"),
                        args=event.get("args", {}),
                        tool_call_id=event.get("tool_call_id", "unknown"),
                    )
                    pending_tool_calls.append(tool_call_part)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse tool_call: {content[:100]}")

            elif message_type == "tool_result":
                # Flush pending tool calls first
                if pending_tool_calls:
                    messages.append(ModelResponse(parts=pending_tool_calls))
                    pending_tool_calls = []

                # Parse tool result JSON
                try:
                    event = json.loads(content)
                    tool_return_part = ToolReturnPart(
                        tool_name=event.get("name", "unknown"),
                        content=event.get("output", ""),
                        tool_call_id=event.get("tool_call_id", "unknown"),
                    )
                    messages.append(ModelRequest(parts=[tool_return_part]))
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse tool_result: {content[:100]}")

            elif message_type == "text":
                # Flush pending tool calls first
                if pending_tool_calls:
                    messages.append(ModelResponse(parts=pending_tool_calls))
                    pending_tool_calls = []

                role = hist.get("role", "user")
                sender_name = hist.get("sender_name", "")

                if role == "assistant" and sender_name == self._agent_name:
                    # Skip THIS agent's text (redundant with tool results)
                    continue
                else:
                    # User messages AND other agents' messages
                    formatted_content = (
                        f"[{sender_name}]: {content}" if sender_name else content
                    )
                    messages.append(
                        ModelRequest(parts=[UserPromptPart(content=formatted_content)])
                    )

        # Flush any remaining pending tool calls
        if pending_tool_calls:
            messages.append(ModelResponse(parts=pending_tool_calls))

        return messages
