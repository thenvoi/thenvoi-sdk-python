"""
Base integration utilities shared across all frameworks.

This module provides common helpers that all framework integrations can use
for standardized behavior.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from thenvoi.core.session import AgentSession


def check_and_format_participants(session: "AgentSession") -> str | None:
    """
    Check if participants changed and return formatted message if so.

    This is a convenience function for framework integrations to ensure
    consistent participant notification across all implementations.

    Usage:
        participants_msg = check_and_format_participants(session)
        if participants_msg:
            # Inject into LLM context as system/user message
            ...

    Args:
        session: The AgentSession to check

    Returns:
        Formatted participant message if changed, None otherwise.
        Automatically calls mark_participants_sent() if changed.
    """
    if not session.participants_changed():
        return None

    msg = session.build_participants_message()
    session.mark_participants_sent()
    return msg
