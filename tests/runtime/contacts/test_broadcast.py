"""Tests for contact broadcast helper."""

from __future__ import annotations

from unittest.mock import MagicMock

from thenvoi.client.streaming import ContactAddedPayload, ContactRequestUpdatedPayload
from thenvoi.platform.event import ContactAddedEvent, ContactRequestUpdatedEvent
from thenvoi.runtime.contacts.broadcast import ContactBroadcaster


def test_maybe_broadcast_emits_message_for_supported_event() -> None:
    sink = MagicMock()
    broadcaster = ContactBroadcaster(sink)
    event = ContactAddedEvent(
        payload=ContactAddedPayload(
            id="contact-1",
            handle="@bob",
            name="Bob",
            type="User",
            inserted_at="2024-01-01T00:00:00Z",
        )
    )

    broadcaster.maybe_broadcast(event)

    sink.broadcast.assert_called_once_with("@bob (Bob) is now a contact")


def test_maybe_broadcast_ignores_non_broadcastable_event() -> None:
    sink = MagicMock()
    broadcaster = ContactBroadcaster(sink)
    event = ContactRequestUpdatedEvent(
        payload=ContactRequestUpdatedPayload(id="req-1", status="approved")
    )

    broadcaster.maybe_broadcast(event)

    sink.broadcast.assert_not_called()
