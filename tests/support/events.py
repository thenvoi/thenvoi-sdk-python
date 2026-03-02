"""SDK-native event factories for tests."""

from __future__ import annotations

from thenvoi.client.streaming import (
    ContactAddedPayload,
    ContactRemovedPayload,
    ContactRequestReceivedPayload,
    ContactRequestUpdatedPayload,
    MessageCreatedPayload,
    MessageMetadata,
    ParticipantAddedPayload,
    ParticipantRemovedPayload,
    RoomAddedPayload,
    RoomRemovedPayload,
)
from thenvoi.platform.event import (
    ContactAddedEvent,
    ContactRemovedEvent,
    ContactRequestReceivedEvent,
    ContactRequestUpdatedEvent,
    MessageEvent,
    ParticipantAddedEvent,
    ParticipantRemovedEvent,
    RoomAddedEvent,
    RoomRemovedEvent,
)

DEFAULT_EVENT_TIMESTAMP = "2024-01-01T00:00:00Z"


def _raise_on_unknown_kwargs(factory: str, kwargs: dict[str, object]) -> None:
    if not kwargs:
        return
    unknown = ", ".join(sorted(kwargs))
    raise TypeError(f"{factory}() got unexpected keyword arguments: {unknown}")


def _normalize_payload_overrides(
    factory: str,
    overrides: dict[str, object],
    *,
    aliases: dict[str, str],
    allowed_fields: set[str],
) -> dict[str, object]:
    normalized: dict[str, object] = {}
    unknown: list[str] = []
    for key, value in overrides.items():
        canonical_key = aliases.get(key, key)
        if canonical_key not in allowed_fields:
            unknown.append(key)
            continue
        normalized[canonical_key] = value
    if unknown:
        unknown_fields = ", ".join(sorted(unknown))
        raise TypeError(f"{factory}() got unexpected keyword arguments: {unknown_fields}")
    return normalized


def make_message_event(
    *,
    room_id: str = "room-123",
    metadata: MessageMetadata | None = None,
    **overrides: object,
) -> MessageEvent:
    """Create a MessageEvent using SDK-native payload types."""
    payload_values: dict[str, object] = {
        "id": "msg-123",
        "content": "Test message",
        "message_type": "text",
        "sender_id": "user-456",
        "sender_name": None,
        "sender_type": "User",
        "chat_room_id": room_id,
        "thread_id": None,
        "inserted_at": DEFAULT_EVENT_TIMESTAMP,
        "updated_at": DEFAULT_EVENT_TIMESTAMP,
        "metadata": metadata if metadata is not None else MessageMetadata(mentions=[]),
    }
    payload_values.update(
        _normalize_payload_overrides(
            "make_message_event",
            dict(overrides),
            aliases={"msg_id": "id", "room_id": "chat_room_id"},
            allowed_fields=set(payload_values),
        )
    )
    if payload_values["metadata"] is None:
        payload_values["metadata"] = MessageMetadata(mentions=[])
    payload = MessageCreatedPayload(**payload_values)
    return MessageEvent(room_id=room_id, payload=payload)


def make_room_added_event(
    *,
    room_id: str = "room-123",
    **overrides: object,
) -> RoomAddedEvent:
    """Create a RoomAddedEvent using SDK-native payload types."""
    payload_values: dict[str, object] = {
        "id": room_id,
        "title": "Test Room",
        "task_id": None,
        "type": "direct",
        "status": "active",
        "owner": None,
        "participant_role": None,
        "created_at": None,
        "inserted_at": DEFAULT_EVENT_TIMESTAMP,
        "updated_at": DEFAULT_EVENT_TIMESTAMP,
    }
    payload_values.update(
        _normalize_payload_overrides(
            "make_room_added_event",
            dict(overrides),
            aliases={"room_id": "id"},
            allowed_fields=set(payload_values),
        )
    )
    payload = RoomAddedPayload(**payload_values)
    return RoomAddedEvent(room_id=room_id, payload=payload)


def make_room_removed_event(
    room_id: str = "room-123",
    title: str = "Test Room",
    *,
    status: str = "removed",
    type: str = "direct",
    removed_at: str = DEFAULT_EVENT_TIMESTAMP,
    **kwargs: object,
) -> RoomRemovedEvent:
    """Create a RoomRemovedEvent using SDK-native payload types."""
    _raise_on_unknown_kwargs("make_room_removed_event", kwargs)
    payload = RoomRemovedPayload(
        id=room_id,
        status=status,
        type=type,
        title=title,
        removed_at=removed_at,
    )
    return RoomRemovedEvent(room_id=room_id, payload=payload)


def make_participant_added_event(
    room_id: str = "room-123",
    participant_id: str = "user-456",
    name: str = "Test User",
    type: str = "User",
) -> ParticipantAddedEvent:
    """Create a ParticipantAddedEvent."""
    payload = ParticipantAddedPayload(id=participant_id, name=name, type=type)
    return ParticipantAddedEvent(room_id=room_id, payload=payload)


def make_participant_removed_event(
    room_id: str = "room-123",
    participant_id: str = "user-456",
) -> ParticipantRemovedEvent:
    """Create a ParticipantRemovedEvent."""
    payload = ParticipantRemovedPayload(id=participant_id)
    return ParticipantRemovedEvent(room_id=room_id, payload=payload)


def make_contact_request_received_event(
    **overrides: object,
) -> ContactRequestReceivedEvent:
    """Create ContactRequestReceivedEvent for tests."""
    payload_values: dict[str, object] = {
        "id": "req-123",
        "from_handle": "john_doe",
        "from_name": "John Doe",
        "message": None,
        "status": "pending",
        "inserted_at": DEFAULT_EVENT_TIMESTAMP,
    }
    payload_values.update(
        _normalize_payload_overrides(
            "make_contact_request_received_event",
            dict(overrides),
            aliases={},
            allowed_fields=set(payload_values),
        )
    )
    payload = ContactRequestReceivedPayload(**payload_values)
    return ContactRequestReceivedEvent(payload=payload)


def make_contact_request_updated_event(
    id: str = "req-123",
    status: str = "approved",
) -> ContactRequestUpdatedEvent:
    """Create ContactRequestUpdatedEvent for tests."""
    payload = ContactRequestUpdatedPayload(
        id=id,
        status=status,
    )
    return ContactRequestUpdatedEvent(payload=payload)


def make_contact_added_event(
    contact_id: str = "contact-123",
    handle: str = "jane_smith",
    name: str = "Jane Smith",
    contact_type: str = "User",
    *,
    description: str | None = None,
    is_external: bool | None = None,
    inserted_at: str = DEFAULT_EVENT_TIMESTAMP,
    **kwargs: object,
) -> ContactAddedEvent:
    """Create ContactAddedEvent for tests."""
    _raise_on_unknown_kwargs("make_contact_added_event", kwargs)
    payload = ContactAddedPayload(
        id=contact_id,
        handle=handle,
        name=name,
        type=contact_type,
        description=description,
        is_external=is_external,
        inserted_at=inserted_at,
    )
    return ContactAddedEvent(payload=payload)


def make_contact_removed_event(
    contact_id: str = "contact-123",
) -> ContactRemovedEvent:
    """Create ContactRemovedEvent for tests."""
    payload = ContactRemovedPayload(id=contact_id)
    return ContactRemovedEvent(payload=payload)


__all__ = [
    "DEFAULT_EVENT_TIMESTAMP",
    "make_message_event",
    "make_room_added_event",
    "make_room_removed_event",
    "make_participant_added_event",
    "make_participant_removed_event",
    "make_contact_request_received_event",
    "make_contact_request_updated_event",
    "make_contact_added_event",
    "make_contact_removed_event",
]
