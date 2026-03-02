from __future__ import annotations

from dataclasses import dataclass

from thenvoi.adapters.codex.adapter_room import format_history_context


@dataclass
class _Config:
    max_history_messages: int = 2


@dataclass
class _Adapter:
    config: _Config


def test_format_history_context_truncates_to_configured_window() -> None:
    adapter = _Adapter(config=_Config(max_history_messages=2))
    raw = [
        {"message_type": "text", "sender_name": "A", "content": "first"},
        {"message_type": "event", "sender_name": "B", "content": "ignored"},
        {"message_type": "message", "sender_name": "C", "content": "second"},
        {"message_type": "text", "sender_name": "D", "content": "third"},
    ]
    rendered = format_history_context(adapter, raw)
    assert rendered is not None
    assert "[C]: second" in rendered
    assert "[D]: third" in rendered
    assert "[A]: first" not in rendered

