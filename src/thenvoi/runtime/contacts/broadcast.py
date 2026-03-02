"""Broadcast helpers for contact event notifications."""

from __future__ import annotations

import logging

from thenvoi.platform.event import ContactEvent

from .formatting import format_broadcast_message
from .sink import ContactEventSink

logger = logging.getLogger(__name__)


class ContactBroadcaster:
    """Format and enqueue contact broadcast messages."""

    def __init__(self, sink: ContactEventSink) -> None:
        self._sink = sink

    def maybe_broadcast(self, event: ContactEvent) -> None:
        """Broadcast contact-added/removed events."""
        message = format_broadcast_message(event)
        if message is not None:
            self._sink.broadcast(message)
            logger.debug("Queued broadcast: %s", message)


__all__ = ["ContactBroadcaster"]
