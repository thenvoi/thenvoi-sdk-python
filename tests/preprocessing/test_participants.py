"""Tests for participant preprocessing helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from thenvoi.preprocessing.participants import check_and_format_participants


def test_check_and_format_participants_returns_none_when_unchanged() -> None:
    ctx = MagicMock()
    ctx.participants_changed.return_value = False

    result = check_and_format_participants(ctx)

    assert result is None
    ctx.mark_participants_sent.assert_not_called()


def test_check_and_format_participants_formats_and_marks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = MagicMock()
    ctx.participants_changed.return_value = True
    ctx.participants = [{"id": "user-1", "name": "Alice", "type": "User"}]
    build_message = MagicMock(return_value="Participants: Alice")
    monkeypatch.setattr(
        "thenvoi.runtime.formatters.build_participants_message",
        build_message,
    )

    result = check_and_format_participants(ctx)

    assert result == "Participants: Alice"
    build_message.assert_called_once_with(ctx.participants)
    ctx.mark_participants_sent.assert_called_once_with()
