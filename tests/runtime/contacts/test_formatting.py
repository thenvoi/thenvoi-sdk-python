"""Direct tests for contact event formatting helpers."""

from __future__ import annotations

from thenvoi.client.streaming import (
    ContactAddedPayload,
    ContactRemovedPayload,
    ContactRequestReceivedPayload,
    ContactRequestUpdatedPayload,
)
from thenvoi.platform.event import (
    ContactAddedEvent,
    ContactRemovedEvent,
    ContactRequestReceivedEvent,
    ContactRequestUpdatedEvent,
    MessageEvent,
)
from thenvoi.runtime.contacts.formatting import (
    format_broadcast_message,
    format_contact_event_for_room,
    get_contact_event_type,
)


async def _no_enrich(
    _event: ContactRequestUpdatedEvent,
) -> dict[str, str | None] | None:
    return None


def test_get_contact_event_type_covers_known_and_unknown() -> None:
    assert get_contact_event_type(ContactRequestReceivedEvent()) == "contact_request_received"
    assert get_contact_event_type(ContactRequestUpdatedEvent()) == "contact_request_updated"
    assert get_contact_event_type(ContactAddedEvent()) == "contact_added"
    assert get_contact_event_type(ContactRemovedEvent()) == "contact_removed"
    assert get_contact_event_type(MessageEvent()) == "unknown"


async def test_format_contact_request_received_includes_message_and_request_id() -> None:
    event = ContactRequestReceivedEvent(
        payload=ContactRequestReceivedPayload(
            id="req-123",
            from_handle="alice",
            from_name="Alice",
            message="Hello there",
            status="pending",
            inserted_at="2024-01-01T00:00:00Z",
        )
    )

    result = await format_contact_event_for_room(
        event,
        enrich_update_event=_no_enrich,
    )

    assert "[Contact Request] Alice (@alice) wants to connect." in result
    assert 'Message: "Hello there"' in result
    assert "Request ID: req-123" in result


async def test_format_contact_request_received_without_payload() -> None:
    result = await format_contact_event_for_room(
        ContactRequestReceivedEvent(payload=None),
        enrich_update_event=_no_enrich,
    )
    assert result == "[Contact Request] Unknown sender"


async def test_format_contact_request_update_prefers_enriched_from_identity() -> None:
    event = ContactRequestUpdatedEvent(
        payload=ContactRequestUpdatedPayload(id="req-123", status="approved")
    )

    async def enrich(_event: ContactRequestUpdatedEvent) -> dict[str, str | None]:
        return {
            "from_name": "Alice",
            "from_handle": "alice",
            "to_name": None,
            "to_handle": None,
        }

    result = await format_contact_event_for_room(event, enrich_update_event=enrich)
    assert "Request from Alice (@alice) status changed to: approved" in result
    assert "Request ID: req-123" in result


async def test_format_contact_request_update_uses_to_identity_when_from_missing() -> None:
    event = ContactRequestUpdatedEvent(
        payload=ContactRequestUpdatedPayload(id="req-456", status="rejected")
    )

    async def enrich(_event: ContactRequestUpdatedEvent) -> dict[str, str | None]:
        return {
            "from_name": None,
            "from_handle": None,
            "to_name": "Bob",
            "to_handle": "bob/agent",
        }

    result = await format_contact_event_for_room(event, enrich_update_event=enrich)
    assert "Request to Bob (@bob/agent) status changed to: rejected" in result


async def test_format_contact_request_update_falls_back_without_enrichment() -> None:
    event = ContactRequestUpdatedEvent(
        payload=ContactRequestUpdatedPayload(id="req-789", status="pending")
    )

    result = await format_contact_event_for_room(event, enrich_update_event=_no_enrich)
    assert result == "[Contact Request Update] Request req-789 status changed to: pending"


async def test_format_contact_request_update_without_payload() -> None:
    result = await format_contact_event_for_room(
        ContactRequestUpdatedEvent(payload=None),
        enrich_update_event=_no_enrich,
    )
    assert result == "[Contact Request Update] Unknown request"


async def test_format_contact_added_and_removed_events() -> None:
    added = ContactAddedEvent(
        payload=ContactAddedPayload(
            id="contact-1",
            handle="alice",
            name="Alice",
            type="User",
            inserted_at="2024-01-01T00:00:00Z",
        )
    )
    removed = ContactRemovedEvent(payload=ContactRemovedPayload(id="contact-1"))

    added_result = await format_contact_event_for_room(
        added,
        enrich_update_event=_no_enrich,
    )
    removed_result = await format_contact_event_for_room(
        removed,
        enrich_update_event=_no_enrich,
    )

    assert "[Contact Added] Alice (@alice) is now a contact." in added_result
    assert "Type: User, ID: contact-1" in added_result
    assert removed_result == "[Contact Removed] Contact contact-1 was removed."


async def test_format_contact_event_unknown_branch() -> None:
    result = await format_contact_event_for_room(
        MessageEvent(),
        enrich_update_event=_no_enrich,
    )
    assert result == "[Contact Event] Unknown event type: MessageEvent"


def test_format_broadcast_message_for_contact_events() -> None:
    added = ContactAddedEvent(
        payload=ContactAddedPayload(
            id="contact-1",
            handle="alice",
            name="Alice",
            type="User",
            inserted_at="2024-01-01T00:00:00Z",
        )
    )
    removed = ContactRemovedEvent(payload=ContactRemovedPayload(id="contact-1"))

    assert format_broadcast_message(added) == "@alice (Alice) is now a contact"
    assert format_broadcast_message(removed) == "Contact contact-1 was removed"
    assert format_broadcast_message(ContactRequestReceivedEvent()) is None
