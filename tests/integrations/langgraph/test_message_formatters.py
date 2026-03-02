"""Tests for LangGraph message formatter helpers."""

from __future__ import annotations

from thenvoi.client.streaming import MessageCreatedPayload
from thenvoi.integrations.langgraph.message_formatters import (
    default_messages_state_formatter,
)


def test_default_messages_state_formatter_includes_sender_metadata() -> None:
    payload = MessageCreatedPayload(
        id="m1",
        content="hello",
        message_type="text",
        sender_id="u1",
        sender_type="User",
        sender_name="Pat",
        inserted_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
    )
    formatted = default_messages_state_formatter(payload, sender_name="Pat")
    content = formatted["messages"][0]["content"]
    assert content.startswith("Message from Pat (User, ID: u1): ")
    assert content.endswith("hello")

