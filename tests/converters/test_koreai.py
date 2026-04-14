"""Tests for KoreAIHistoryConverter."""

from __future__ import annotations

import pytest

from thenvoi.converters.koreai import KoreAIHistoryConverter


@pytest.fixture
def converter() -> KoreAIHistoryConverter:
    return KoreAIHistoryConverter()


class TestKoreAIHistoryConverter:
    def test_empty_history_returns_empty_state(
        self, converter: KoreAIHistoryConverter
    ) -> None:
        result = converter.convert([])
        assert result.koreai_identity is None
        assert result.koreai_last_activity is None

    def test_extracts_session_state_from_task_event(
        self, converter: KoreAIHistoryConverter
    ) -> None:
        raw = [
            {
                "message_type": "text",
                "content": "Hello",
                "sender_name": "User",
            },
            {
                "message_type": "task",
                "content": "koreai session active",
                "metadata": {
                    "koreai_identity": "room-abc",
                    "koreai_last_activity": 1711843200.0,
                },
            },
        ]
        result = converter.convert(raw)
        assert result.koreai_identity == "room-abc"
        assert result.koreai_last_activity == 1711843200.0

    def test_returns_most_recent_task_event(
        self, converter: KoreAIHistoryConverter
    ) -> None:
        raw = [
            {
                "message_type": "task",
                "content": "koreai session active",
                "metadata": {
                    "koreai_identity": "room-abc",
                    "koreai_last_activity": 1000.0,
                },
            },
            {
                "message_type": "text",
                "content": "some message",
            },
            {
                "message_type": "task",
                "content": "koreai session active",
                "metadata": {
                    "koreai_identity": "room-abc",
                    "koreai_last_activity": 2000.0,
                },
            },
        ]
        result = converter.convert(raw)
        assert result.koreai_last_activity == 2000.0

    def test_ignores_non_koreai_task_events(
        self, converter: KoreAIHistoryConverter
    ) -> None:
        raw = [
            {
                "message_type": "task",
                "content": "a2a task completed",
                "metadata": {
                    "a2a_context_id": "ctx-123",
                    "a2a_task_id": "task-456",
                },
            },
        ]
        result = converter.convert(raw)
        assert result.koreai_identity is None
        assert result.koreai_last_activity is None

    def test_no_task_events_returns_empty(
        self, converter: KoreAIHistoryConverter
    ) -> None:
        raw = [
            {"message_type": "text", "content": "Hello", "sender_name": "User"},
            {"message_type": "text", "content": "World", "sender_name": "Bot"},
        ]
        result = converter.convert(raw)
        assert result.koreai_identity is None

    def test_malformed_metadata_handled_gracefully(
        self, converter: KoreAIHistoryConverter
    ) -> None:
        raw = [
            {
                "message_type": "task",
                "content": "koreai session active",
                "metadata": {
                    "koreai_identity": "room-abc",
                    "koreai_last_activity": "not-a-number",
                },
            },
        ]
        result = converter.convert(raw)
        assert result.koreai_identity == "room-abc"
        assert result.koreai_last_activity is None

    def test_missing_metadata_field(self, converter: KoreAIHistoryConverter) -> None:
        raw = [
            {
                "message_type": "task",
                "content": "some task",
                "metadata": {},
            },
        ]
        result = converter.convert(raw)
        assert result.koreai_identity is None

    def test_missing_metadata_key(self, converter: KoreAIHistoryConverter) -> None:
        raw = [
            {
                "message_type": "task",
                "content": "some task",
            },
        ]
        result = converter.convert(raw)
        assert result.koreai_identity is None

    def test_none_last_activity(self, converter: KoreAIHistoryConverter) -> None:
        raw = [
            {
                "message_type": "task",
                "content": "koreai session active",
                "metadata": {
                    "koreai_identity": "room-abc",
                    "koreai_last_activity": None,
                },
            },
        ]
        result = converter.convert(raw)
        assert result.koreai_identity == "room-abc"
        assert result.koreai_last_activity is None
