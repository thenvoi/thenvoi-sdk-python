"""Pure functions for message formatting. No I/O, fully unit-testable."""

from __future__ import annotations


def format_message_for_llm(msg: dict) -> dict:
    """
    Map platform message to LLM format.

    Args:
        msg: Platform message dict with sender_type, content, sender_name

    Returns:
        Dict with role, content, sender_name, sender_type
    """
    sender_type = msg.get("sender_type", "")
    sender_name = msg.get("sender_name") or msg.get("name") or sender_type

    return {
        "role": "assistant" if sender_type == "Agent" else "user",
        "content": msg.get("content", ""),
        "sender_name": sender_name,
        "sender_type": sender_type,
    }


def format_history_for_llm(
    messages: list[dict],
    exclude_id: str | None = None,
) -> list[dict]:
    """
    Format platform message history for LLM injection.

    Args:
        messages: List of platform message dicts
        exclude_id: Message ID to exclude (usually current message)

    Returns:
        List of formatted message dicts
    """
    return [format_message_for_llm(m) for m in messages if m.get("id") != exclude_id]


def build_participants_message(participants: list[dict]) -> str:
    """
    Build participant list message for LLM context.

    Args:
        participants: List of participant dicts with id, name, type

    Returns:
        Formatted string for LLM system message
    """
    if not participants:
        return "## Current Participants\nNo other participants in this room."

    lines = ["## Current Participants"]
    for p in participants:
        p_type = p.get("type", "Unknown")
        p_name = p.get("name", "Unknown")
        p_id = p.get("id", "")
        lines.append(f"- {p_name} (ID: {p_id}, Type: {p_type})")

    lines.append("")
    lines.append(
        "When using send_message, include mentions with ID and name from this list."
    )

    return "\n".join(lines)
