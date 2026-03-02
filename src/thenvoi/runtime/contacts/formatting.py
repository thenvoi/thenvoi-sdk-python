"""Formatting helpers for contact events and hub-room messages."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from thenvoi.platform.event import (
    ContactAddedEvent,
    ContactEvent,
    ContactRemovedEvent,
    ContactRequestReceivedEvent,
    ContactRequestUpdatedEvent,
)
from thenvoi.prompt_policies.contacts import load_hub_room_system_prompt
from thenvoi.runtime.types import normalize_handle

# Hub room system prompt - injected when hub room is first used
HUB_ROOM_SYSTEM_PROMPT = load_hub_room_system_prompt()


RequestInfoLookup = Callable[
    [ContactRequestUpdatedEvent], Awaitable[dict[str, str | None] | None]
]


def get_contact_event_type(event: ContactEvent) -> str:
    """Return normalized contact event type labels for metadata."""
    match event:
        case ContactRequestReceivedEvent():
            return "contact_request_received"
        case ContactRequestUpdatedEvent():
            return "contact_request_updated"
        case ContactAddedEvent():
            return "contact_added"
        case ContactRemovedEvent():
            return "contact_removed"
        case _:
            return "unknown"


async def format_contact_event_for_room(
    event: ContactEvent,
    *,
    enrich_update_event: RequestInfoLookup,
) -> str:
    """Format contact events into a stable hub-room message contract."""
    match event:
        case ContactRequestReceivedEvent(payload=payload):
            if payload is None:
                return "[Contact Request] Unknown sender"
            from_handle = normalize_handle(payload.from_handle)
            msg_part = f'\nMessage: "{payload.message}"' if payload.message else ""
            return (
                f"[Contact Request] {payload.from_name} ({from_handle}) "
                f"wants to connect.{msg_part}\n"
                f"Request ID: {payload.id}"
            )

        case ContactRequestUpdatedEvent(payload=payload):
            if payload is None:
                return "[Contact Request Update] Unknown request"

            info = await enrich_update_event(event)
            if info:
                name = info.get("from_name") or info.get("to_name")
                raw_handle = info.get("from_handle") or info.get("to_handle") or ""
                handle = normalize_handle(raw_handle) or ""
                if name:
                    direction = "from" if info.get("from_name") else "to"
                    return (
                        f"[Contact Request Update] Request {direction} {name} "
                        f"({handle}) status changed to: {payload.status}\n"
                        f"Request ID: {payload.id}"
                    )

            return (
                f"[Contact Request Update] Request {payload.id} "
                f"status changed to: {payload.status}"
            )

        case ContactAddedEvent(payload=payload):
            if payload is None:
                return "[Contact Added] Unknown contact"
            handle = normalize_handle(payload.handle)
            return (
                f"[Contact Added] {payload.name} ({handle}) "
                f"is now a contact.\n"
                f"Type: {payload.type}, ID: {payload.id}"
            )

        case ContactRemovedEvent(payload=payload):
            if payload is None:
                return "[Contact Removed] Unknown contact"
            return f"[Contact Removed] Contact {payload.id} was removed."

        case _:
            return f"[Contact Event] Unknown event type: {type(event).__name__}"


def format_broadcast_message(event: ContactEvent) -> str | None:
    """Return broadcast text for contact_added/contact_removed events."""
    match event:
        case ContactAddedEvent(payload=payload):
            if payload is None:
                return None
            handle = normalize_handle(payload.handle)
            return f"{handle} ({payload.name}) is now a contact"

        case ContactRemovedEvent(payload=payload):
            if payload is None:
                return None
            return f"Contact {payload.id} was removed"

        case _:
            return None
