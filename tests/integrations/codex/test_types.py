"""Tests for Codex integration state types."""

from __future__ import annotations

from datetime import datetime, timezone

from thenvoi.integrations.codex.types import CodexSessionState


def test_codex_session_state_has_thread_flag() -> None:
    state = CodexSessionState(thread_id="thread-1", room_id="room-1")

    assert state.has_thread() is True


def test_codex_session_state_without_thread_reports_false() -> None:
    state = CodexSessionState(
        thread_id=None,
        room_id="room-1",
        created_at=datetime.now(timezone.utc),
    )

    assert state.has_thread() is False
