"""Tests for CodexHistoryConverter."""

from __future__ import annotations

from thenvoi.converters.codex import CodexHistoryConverter


class TestCodexHistoryConverter:
    """Converter should extract latest codex thread mapping from task events."""

    def test_convert_empty_history(self) -> None:
        converter = CodexHistoryConverter()
        state = converter.convert([])
        assert state.thread_id is None
        assert state.room_id is None
        assert state.created_at is None

    def test_convert_finds_latest_codex_mapping(self) -> None:
        converter = CodexHistoryConverter()
        raw_history = [
            {
                "message_type": "task",
                "metadata": {
                    "codex_thread_id": "thr_old",
                    "codex_room_id": "room-1",
                    "codex_created_at": "2026-02-17T10:00:00Z",
                },
            },
            {"message_type": "text", "content": "ignored"},
            {
                "message_type": "task",
                "metadata": {
                    "codex_thread_id": "thr_new",
                    "codex_room_id": "room-2",
                    "codex_created_at": "2026-02-18T12:34:56Z",
                },
            },
        ]

        state = converter.convert(raw_history)
        assert state.thread_id == "thr_new"
        assert state.room_id == "room-2"
        assert state.created_at is not None
        assert state.created_at.year == 2026

    def test_convert_ignores_non_codex_task_metadata(self) -> None:
        converter = CodexHistoryConverter()
        raw_history = [
            {
                "message_type": "task",
                "metadata": {"a2a_task_id": "task-1"},
            },
            {
                "message_type": "task",
                "metadata": {"other_key": "value"},
            },
        ]

        state = converter.convert(raw_history)
        assert state.thread_id is None
        assert state.room_id is None
