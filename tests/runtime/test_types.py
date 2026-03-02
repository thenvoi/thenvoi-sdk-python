from __future__ import annotations

from datetime import datetime, timezone

import pytest

from thenvoi.core.types import PlatformMessage as CorePlatformMessage
from thenvoi.runtime.types import (
    PlatformMessage,
    is_agent_sender_type,
    normalize_handle,
)


def _timestamp() -> datetime:
    return datetime(2026, 1, 1, tzinfo=timezone.utc)


def _runtime_message(
    *,
    sender_name: str | None,
    sender_type: str,
    content: str = "hello",
) -> PlatformMessage:
    return PlatformMessage(
        id="msg-1",
        room_id="room-1",
        content=content,
        sender_id="sender-1",
        sender_type=sender_type,
        sender_name=sender_name,
        message_type="text",
        metadata={},
        created_at=_timestamp(),
    )


def _core_message(
    *,
    sender_name: str | None,
    sender_type: str,
    content: str = "hello",
) -> CorePlatformMessage:
    return CorePlatformMessage(
        id="msg-1",
        room_id="room-1",
        content=content,
        sender_id="sender-1",
        sender_type=sender_type,
        sender_name=sender_name,
        message_type="text",
        metadata={},
        created_at=_timestamp(),
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, None),
        ("", None),
        ("   ", None),
        ("alice", "@alice"),
        ("@alice", "@alice"),
        ("  alice  ", "@alice"),
        ("  @alice  ", "@alice"),
    ],
)
def test_normalize_handle_covers_empty_prefix_and_whitespace(
    raw: str | None,
    expected: str | None,
) -> None:
    assert normalize_handle(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, False),
        ("", False),
        (" ", False),
        ("Agent", True),
        ("agent", True),
        (" aGeNt ", True),
        ("User", False),
        ("system", False),
    ],
)
def test_is_agent_sender_type_normalization(raw: str | None, expected: bool) -> None:
    assert is_agent_sender_type(raw) is expected


def test_platform_message_format_for_llm_prefers_sender_name() -> None:
    message = _runtime_message(sender_name="Agent Tom", sender_type="Agent")
    assert message.format_for_llm() == "[Agent Tom]: hello"


def test_platform_message_format_for_llm_falls_back_to_sender_type() -> None:
    message = _runtime_message(sender_name=None, sender_type="System")
    assert message.format_for_llm() == "[System]: hello"


def test_platform_message_format_for_llm_falls_back_to_unknown() -> None:
    message = _runtime_message(sender_name=None, sender_type="")
    assert message.format_for_llm() == "[Unknown]: hello"


@pytest.mark.parametrize(
    ("sender_name", "sender_type"),
    [
        ("Agent Tom", "Agent"),
        (None, "User"),
        (None, ""),
    ],
)
def test_runtime_and_core_platform_message_formatting_parity(
    sender_name: str | None,
    sender_type: str,
) -> None:
    runtime_message = _runtime_message(
        sender_name=sender_name,
        sender_type=sender_type,
    )
    core_message = _core_message(
        sender_name=sender_name,
        sender_type=sender_type,
    )

    assert runtime_message.format_for_llm() == core_message.format_for_llm()
