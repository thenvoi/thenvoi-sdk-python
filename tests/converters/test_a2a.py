"""Tests for A2A history converter."""

from __future__ import annotations

from thenvoi.converters.a2a import A2AHistoryConverter


def test_a2a_converter_extracts_latest_task_metadata() -> None:
    converter = A2AHistoryConverter()
    history = [
        {
            "message_type": "task",
            "metadata": {
                "a2a_context_id": "ctx-1",
                "a2a_task_id": "task-1",
                "a2a_task_state": "running",
            },
        },
        {
            "message_type": "task",
            "metadata": {
                "a2a_context_id": "ctx-2",
                "a2a_task_id": "task-2",
                "a2a_task_state": "completed",
            },
        },
    ]

    state = converter.convert(history)
    assert state.context_id == "ctx-2"
    assert state.task_id == "task-2"
    assert state.task_state == "completed"

