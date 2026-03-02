"""Tests for bridge platform message normalization helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from thenvoi.integrations.a2a_bridge.platform_message_factory import (
    build_platform_message,
    coerce_datetime,
    metadata_to_dict,
)


def test_coerce_datetime_parses_iso_string_with_z_suffix() -> None:
    parsed = coerce_datetime("2026-01-02T03:04:05Z")
    assert parsed == datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


def test_coerce_datetime_invalid_string_falls_back_to_now() -> None:
    before = datetime.now(timezone.utc)
    parsed = coerce_datetime("not-a-timestamp")
    after = datetime.now(timezone.utc)
    assert before <= parsed <= after


def test_metadata_to_dict_uses_model_dump_when_available() -> None:
    class _Metadata:
        def model_dump(self) -> dict[str, str]:
            return {"source": "model"}

    assert metadata_to_dict(_Metadata()) == {"source": "model"}


def test_build_platform_message_defaults_thread_id_to_room_id() -> None:
    payload = SimpleNamespace(
        id="msg-1",
        thread_id=None,
        content="hello",
        sender_id="user-1",
        sender_type="User",
        message_type=None,
        metadata={"mentions": []},
        inserted_at="2026-01-02T03:04:05Z",
    )

    message = build_platform_message(payload, "room-1", sender_name="Alice")

    assert message.thread_id == "room-1"
    assert message.sender_name == "Alice"
    assert message.created_at == datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
