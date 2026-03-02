"""Contract tests for shared event factory helpers."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from tests.support.events import (
    DEFAULT_EVENT_TIMESTAMP,
    make_contact_added_event,
    make_contact_request_received_event,
    make_message_event,
    make_room_added_event,
    make_room_removed_event,
)


def test_make_message_event_maps_explicit_optional_fields() -> None:
    event = make_message_event(
        sender_name="Alice",
        thread_id="thread-1",
        message_type="thought",
    )

    assert event.payload.sender_name == "Alice"
    assert event.payload.thread_id == "thread-1"
    assert event.payload.message_type == "thought"


def test_make_room_added_event_keeps_room_type_for_filter_contracts() -> None:
    event = make_room_added_event(type="task", status="active")

    assert event.payload.type == "task"
    assert event.payload.status == "active"


@pytest.mark.parametrize(
    ("factory", "kwargs"),
    [
        (make_message_event, {"unexpected_key": 1}),
        (make_contact_added_event, {"bad_key": True}),
    ],
)
def test_factories_reject_unknown_kwargs(
    factory: Callable[..., object],
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(TypeError, match="unexpected keyword arguments"):
        factory(**kwargs)


def test_event_factory_defaults_share_single_timestamp_baseline() -> None:
    message = make_message_event()
    room_added = make_room_added_event()
    room_removed = make_room_removed_event()
    contact_request = make_contact_request_received_event()
    contact_added = make_contact_added_event()

    assert message.payload.inserted_at == DEFAULT_EVENT_TIMESTAMP
    assert message.payload.updated_at == DEFAULT_EVENT_TIMESTAMP
    assert room_added.payload.inserted_at == DEFAULT_EVENT_TIMESTAMP
    assert room_added.payload.updated_at == DEFAULT_EVENT_TIMESTAMP
    assert room_removed.payload.removed_at == DEFAULT_EVENT_TIMESTAMP
    assert contact_request.payload.inserted_at == DEFAULT_EVENT_TIMESTAMP
    assert contact_added.payload.inserted_at == DEFAULT_EVENT_TIMESTAMP
