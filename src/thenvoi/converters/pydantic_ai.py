"""Pydantic AI history converter."""

from __future__ import annotations

from typing import Any

try:
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        UserPromptPart,
        TextPart,
    )
except ImportError as e:
    raise ImportError(
        "Pydantic AI dependencies not installed. "
        "Install with: uv add thenvoi-sdk[pydantic_ai]"
    ) from e

from thenvoi.core.protocols import HistoryConverter

# Type alias for Pydantic AI messages
PydanticAIMessages = list[ModelRequest | ModelResponse]


class PydanticAIHistoryConverter(HistoryConverter[PydanticAIMessages]):
    """
    Converts platform history to Pydantic AI message format.

    Output:
    - user messages → ModelRequest with UserPromptPart
    - assistant messages → ModelResponse with TextPart

    Note:
    - Only converts text messages (tool_call/tool_result events are skipped)
    - User messages are prefixed with sender name: "[Alice]: Hello"
    """

    def convert(self, raw: list[dict[str, Any]]) -> PydanticAIMessages:
        """Convert platform history to Pydantic AI format."""
        messages: PydanticAIMessages = []

        for hist in raw:
            message_type = hist.get("message_type", "text")

            # Only convert text messages
            if message_type != "text":
                continue

            role = hist.get("role", "user")
            content = hist.get("content", "")
            sender_name = hist.get("sender_name", "")

            if role == "assistant":
                # Agent's previous messages
                messages.append(ModelResponse(parts=[TextPart(content=content)]))
            else:
                # Messages from users or other agents
                formatted_content = (
                    f"[{sender_name}]: {content}" if sender_name else content
                )
                messages.append(
                    ModelRequest(parts=[UserPromptPart(content=formatted_content)])
                )

        return messages
