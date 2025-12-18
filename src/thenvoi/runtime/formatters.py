"""Pure functions for message formatting. No I/O, fully unit-testable."""

from __future__ import annotations


def format_message_for_llm(msg: dict) -> dict:
    """
    Map platform message to LLM format.

    Args:
        msg: Platform message dict from hydration (ChatMessage fields)

    Returns:
        Dict with role, content, sender_name, sender_type, message_type
    """
    sender_type = msg["sender_type"]
    sender_name = msg["sender_name"] or sender_type

    return {
        "role": "assistant" if sender_type == "Agent" else "user",
        "content": msg["content"],
        "sender_name": sender_name,
        "sender_type": sender_type,
        "message_type": msg["message_type"],
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
        lines.append(f"- {p_name} ({p_type})")

    lines.append("")
    lines.append(
        "To mention a participant in send_message, use their EXACT name (e.g., 'Weather Agent', not an ID)."
    )

    return "\n".join(lines)
