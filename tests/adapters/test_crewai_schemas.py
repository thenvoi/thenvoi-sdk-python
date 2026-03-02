"""Tests for CrewAI schema override models."""

from __future__ import annotations

import pytest

from thenvoi.adapters.crewai_schemas import CrewAISendEventInput, CrewAISendMessageInput


def test_send_message_mentions_normalize_none_to_empty_list() -> None:
    payload = CrewAISendMessageInput(content="hello", mentions=None)
    assert payload.mentions == []


def test_send_message_mentions_reject_non_list_values() -> None:
    with pytest.raises(TypeError, match="mentions must be a list of handles"):
        CrewAISendMessageInput(content="hello", mentions="@john")


def test_send_event_defaults_to_thought_type() -> None:
    payload = CrewAISendEventInput(content="thinking")
    assert payload.message_type == "thought"
    assert payload.metadata is None

