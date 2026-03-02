"""Tests for thenvoi.converters.normalized_events."""

from __future__ import annotations

import json

from thenvoi.converters.normalized_events import (
    TextHistoryEvent,
    ToolCallHistoryEvent,
    ToolResultHistoryEvent,
    _parse_tool_call_event,
    _parse_tool_result_event,
    normalize_history_events,
)


def test_normalize_history_events_maps_text_and_tool_events() -> None:
    raw = [
        {
            "message_type": "text",
            "role": "user",
            "content": "hello",
            "sender_name": "Alice",
            "type": "User",
        },
        {
            "message_type": "tool_call",
            "content": json.dumps(
                {
                    "name": "lookup_weather",
                    "args": {"city": "NYC"},
                    "tool_call_id": "call-1",
                }
            ),
        },
        {
            "message_type": "tool_result",
            "content": json.dumps(
                {
                    "name": "lookup_weather",
                    "output": "Sunny",
                    "tool_call_id": "call-1",
                }
            ),
        },
    ]

    events = normalize_history_events(raw)

    assert events == [
        TextHistoryEvent(
            kind="text",
            role="user",
            content="hello",
            sender_name="Alice",
            sender_type="User",
        ),
        ToolCallHistoryEvent(
            kind="tool_call",
            name="lookup_weather",
            args={"city": "NYC"},
            tool_call_id="call-1",
        ),
        ToolResultHistoryEvent(
            kind="tool_result",
            name="lookup_weather",
            output="Sunny",
            tool_call_id="call-1",
            is_error=False,
        ),
    ]


def test_parse_tool_call_event_supports_data_input_and_run_id_fallback() -> None:
    parsed = _parse_tool_call_event(
        json.dumps(
            {
                "name": "search_docs",
                "run_id": "run-1",
                "data": {"input": {"query": "retry policy"}},
            }
        )
    )

    assert parsed == ToolCallHistoryEvent(
        kind="tool_call",
        name="search_docs",
        args={"query": "retry policy"},
        tool_call_id="run-1",
    )


def test_parse_tool_call_event_returns_none_for_invalid_payload() -> None:
    assert _parse_tool_call_event("not-json") is None
    assert _parse_tool_call_event(json.dumps({"name": "", "tool_call_id": "x"})) is None
    assert _parse_tool_call_event(json.dumps({"name": "x"})) is None


def test_parse_tool_result_event_extracts_output_and_regex_call_id() -> None:
    parsed = _parse_tool_result_event(
        json.dumps(
            {
                "name": "calc",
                "output": "ToolOutput(tool_call_id='call-xyz', result=12)",
                "is_error": True,
            }
        )
    )

    assert parsed == ToolResultHistoryEvent(
        kind="tool_result",
        name="calc",
        output="ToolOutput(tool_call_id='call-xyz', result=12)",
        tool_call_id="call-xyz",
        is_error=True,
    )


def test_parse_tool_result_event_uses_data_output_and_run_id_fallback() -> None:
    parsed = _parse_tool_result_event(
        json.dumps(
            {
                "name": "sync",
                "run_id": "run-55",
                "data": {"output": {"status": "ok"}},
            }
        )
    )

    assert parsed == ToolResultHistoryEvent(
        kind="tool_result",
        name="sync",
        output="{'status': 'ok'}",
        tool_call_id="run-55",
        is_error=False,
    )


def test_parse_tool_result_event_returns_none_for_missing_fields() -> None:
    assert _parse_tool_result_event("not-json") is None
    assert _parse_tool_result_event(json.dumps({"name": ""})) is None
    assert _parse_tool_result_event(json.dumps({"name": "calc", "output": "done"})) is None


def test_normalize_history_events_skips_unparseable_tool_entries() -> None:
    raw = [
        {"message_type": "tool_call", "content": "not-json"},
        {"message_type": "tool_result", "content": "not-json"},
        {"message_type": "unknown", "content": "ignored"},
    ]

    assert normalize_history_events(raw) == []
