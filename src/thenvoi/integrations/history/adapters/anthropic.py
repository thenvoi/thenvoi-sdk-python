"""
Anthropic adapter - converts normalized messages to Anthropic MessageParam format.

Handles:
- NormalizedUserText → user message with formatted content
- NormalizedToolExchange → assistant (tool_use) + user (tool_result) pair
- Batching consecutive same-role messages (Anthropic requires alternating roles)
"""

from __future__ import annotations

from anthropic.types import (
    ContentBlock,
    MessageParam,
    TextBlockParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
)

from ..normalized import (
    NormalizedMessage,
    NormalizedToolExchange,
    NormalizedUserText,
)

# Type aliases
ContentBlockParam = TextBlockParam | ToolUseBlockParam | ToolResultBlockParam
ContentBlockTypes = ContentBlockParam | ContentBlock


def to_anthropic_messages(
    normalized: list[NormalizedMessage],
) -> list[MessageParam]:
    """
    Convert normalized messages to Anthropic MessageParam format.

    Args:
        normalized: List of normalized messages from parser

    Returns:
        List of Anthropic MessageParam, with consecutive same-role
        messages batched together.

    Example:
        >>> from thenvoi.integrations.history import parse_platform_history
        >>> normalized = parse_platform_history(raw_history)
        >>> messages = to_anthropic_messages(normalized)
        >>> # Ready for client.messages.create(messages=messages)
    """
    messages: list[MessageParam] = []

    for msg in normalized:
        if isinstance(msg, NormalizedUserText):
            formatted = f"[{msg.sender_name}]: {msg.content}"
            messages.append({"role": "user", "content": formatted})

        elif isinstance(msg, NormalizedToolExchange):
            # Assistant message with tool_use block
            tool_use_block: ToolUseBlockParam = {
                "type": "tool_use",
                "id": msg.tool_id,
                "name": msg.tool_name,
                "input": msg.input_args,
            }
            messages.append({"role": "assistant", "content": [tool_use_block]})

            # User message with tool_result block
            tool_result_block: ToolResultBlockParam = {
                "type": "tool_result",
                "tool_use_id": msg.tool_id,
                "content": msg.output,
                "is_error": msg.is_error,
            }
            messages.append({"role": "user", "content": [tool_result_block]})

        # Skip NormalizedSystemMessage - Anthropic uses separate system param

    return _batch_consecutive_messages(messages)


def _batch_consecutive_messages(
    messages: list[MessageParam],
) -> list[MessageParam]:
    """
    Batch consecutive same-role messages into single messages.

    Anthropic requires alternating user/assistant roles. This merges
    consecutive messages with the same role into a single message
    with multiple content blocks.
    """
    if not messages:
        return messages

    batched: list[MessageParam] = []
    current: MessageParam | None = None

    for msg in messages:
        if current and current["role"] == msg["role"]:
            # Same role - merge content blocks
            current_content = current["content"]
            msg_content = msg["content"]

            # Normalize to lists
            if isinstance(current_content, str):
                text_block: TextBlockParam = {"type": "text", "text": current_content}
                current = {"role": current["role"], "content": [text_block]}

            if isinstance(msg_content, str):
                msg_text_block: TextBlockParam = {"type": "text", "text": msg_content}
                msg_content = [msg_text_block]

            # Extend current content
            current_content_list = current["content"]
            if isinstance(current_content_list, list):
                current_content_list.extend(msg_content)
        else:
            # Different role - start new message
            if current:
                batched.append(current)
            current = {"role": msg["role"], "content": msg["content"]}

    if current:
        batched.append(current)

    return batched
