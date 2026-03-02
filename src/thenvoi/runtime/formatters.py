"""Pure functions for message formatting. No I/O, fully unit-testable."""

from __future__ import annotations

from typing import Any

from thenvoi.runtime.types import is_agent_sender_type


def replace_uuid_mentions(content: str, participants: list[dict]) -> str:
    """
    Replace UUID mentions in content with @handle format using participants list.

    Args:
        content: Message content potentially containing @[[uuid]] patterns
        participants: List of participants with {id, handle, name, type}

    Returns:
        Content with UUID mentions replaced by @handle
    """
    if not participants or not content:
        return content

    for p in participants:
        participant_id = p.get("id")
        handle = p.get("handle")
        if participant_id and handle:
            content = content.replace(f"@[[{participant_id}]]", f"@{handle}")

    return content


def format_message_for_llm(msg: dict, participants: list[dict] | None = None) -> dict:
    """
    Map platform message to LLM format.

    Args:
        msg: Platform message dict with type/sender_type, content, sender_name
        participants: Optional list of participants for UUID mention replacement

    Returns:
        Dict with role, content, sender_name, sender_type, message_type, metadata.
        Includes compatibility "type" when source message uses that field.
    """
    sender_type = msg.get("type") or msg.get("sender_type") or ""
    sender_name = msg.get("sender_name") or msg.get("name") or sender_type

    content = msg.get("content", "")
    if participants:
        content = replace_uuid_mentions(content, participants)

    metadata = msg.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}

    # Normalize to a single canonical history schema for converters.
    formatted: dict[str, Any] = {
        "role": "assistant" if is_agent_sender_type(sender_type) else "user",
        "content": content,
        "sender_name": sender_name,
        "message_type": msg.get("message_type", "text"),
    }
    formatted["sender_type"] = sender_type
    formatted["metadata"] = metadata

    # Backward compatibility alias for converters still reading "type".
    if "type" in msg:
        formatted["type"] = sender_type

    return formatted


def format_history_for_llm(
    messages: list[dict],
    exclude_id: str | None = None,
    participants: list[dict] | None = None,
) -> list[dict]:
    """
    Format platform message history for LLM injection.

    Args:
        messages: List of platform message dicts
        exclude_id: Message ID to exclude (usually current message)
        participants: Optional list of participants for UUID mention replacement

    Returns:
        List of formatted message dicts
    """
    return [
        format_message_for_llm(m, participants)
        for m in messages
        if m.get("id") != exclude_id
    ]


def build_participants_message(participants: list[dict]) -> str:
    """
    Build participant list message for LLM context.

    Includes instruction to use thenvoi_send_message with handles or names.

    Args:
        participants: List of participant dicts with id, name, type, handle

    Returns:
        Formatted string for LLM system message
    """
    if not participants:
        return "## Current Participants\nNo other participants in this room."

    lines = ["## Current Participants"]
    for p in participants:
        p_type = p.get("type", "Unknown")
        p_name = p.get("name", "Unknown")
        p_handle = p.get("handle", "Unknown")
        lines.append(f"- @{p_handle} — {p_name} ({p_type})")

    lines.append("")
    lines.append(
        "IMPORTANT: In thenvoi_send_message mentions, always use the exact "
        "handle shown above (e.g. '@john' for users, '@john/weather-agent' "
        "for agents), NOT the display name. Handles are lowercase with no spaces."
    )

    return "\n".join(lines)
