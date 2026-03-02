"""Shared tool-session registry and wrapper payload helpers."""

from __future__ import annotations

import json
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class ToolSessionRegistry(Generic[T]):
    """Track per-session bound tools and send-message side effects."""

    def __init__(self) -> None:
        self._tools_by_session: dict[str, T] = {}
        self._message_sent_by_session: dict[str, bool] = {}

    def set_tools(self, session_id: str, tools: T | None) -> None:
        """Bind or clear tools for a session."""
        if tools is None:
            self._tools_by_session.pop(session_id, None)
            self._message_sent_by_session.pop(session_id, None)
            return
        self._tools_by_session[session_id] = tools
        self._message_sent_by_session[session_id] = False

    def get_tools(self, session_id: str) -> T | None:
        """Return tools for the session if present."""
        return self._tools_by_session.get(session_id)

    def mark_message_sent(self, session_id: str) -> None:
        """Record that a send-message tool invocation succeeded in session."""
        self._message_sent_by_session[session_id] = True

    def was_message_sent(self, session_id: str) -> bool:
        """Return whether session has emitted a send-message call."""
        return self._message_sent_by_session.get(session_id, False)

    def active_sessions(self) -> list[str]:
        """Return session IDs with bound tools."""
        return list(self._tools_by_session.keys())


def mcp_text_result(data: Any) -> dict[str, Any]:
    """Format success payload for Claude MCP SDK wrappers."""
    return {"content": [{"type": "text", "text": json.dumps(data, default=str)}]}


def mcp_text_error(message: str) -> dict[str, Any]:
    """Format error payload for Claude MCP SDK wrappers."""
    return {
        "content": [
            {
                "type": "text",
                "text": json.dumps({"status": "error", "message": message}),
            }
        ],
        "is_error": True,
    }

