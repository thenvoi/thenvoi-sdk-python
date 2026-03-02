"""Deduplication cache for contact event handling."""

from __future__ import annotations

from collections import OrderedDict

from thenvoi.platform.event import ContactEvent, ContactRequestUpdatedEvent

from .formatting import get_contact_event_type


class ContactDedupCache:
    """Bounded deduplication cache keyed by contact event identity."""

    def __init__(self, *, max_size: int) -> None:
        self._max_size = max_size
        self._processed_events: OrderedDict[str, bool] = OrderedDict()

    @property
    def storage(self) -> OrderedDict[str, bool]:
        """Expose storage for compatibility with existing tests."""
        return self._processed_events

    def key_for(self, event: ContactEvent) -> str | None:
        """Return the dedup key for a contact event."""
        match event:
            case ContactRequestUpdatedEvent(payload=payload):
                return (
                    f"request_updated:{payload.id}:{payload.status}"
                    if payload
                    else None
                )
            case _:
                pass

        payload = getattr(event, "payload", None)
        if payload is None:
            return None

        event_type = get_contact_event_type(event)
        if event_type == "contact_request_received":
            return f"request_received:{payload.id}"
        if event_type == "contact_added":
            return f"contact_added:{payload.id}"
        if event_type == "contact_removed":
            return f"contact_removed:{payload.id}"
        return None

    def should_skip(self, event: ContactEvent) -> bool:
        """Check if an event has already been processed."""
        key = self.key_for(event)
        if key is None:
            return False
        return key in self._processed_events

    def mark_processed(self, event: ContactEvent) -> None:
        """Mark event as processed with bounded cache size."""
        key = self.key_for(event)
        if key is None:
            return

        self._processed_events[key] = True
        while len(self._processed_events) > self._max_size:
            self._processed_events.popitem(last=False)

    def clear(self, event: ContactEvent) -> None:
        """Clear an event from dedup cache to allow retries."""
        key = self.key_for(event)
        if key is not None:
            self._processed_events.pop(key, None)


__all__ = ["ContactDedupCache"]
