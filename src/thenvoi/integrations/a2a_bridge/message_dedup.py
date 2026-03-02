"""Message deduplication utilities for bridge consumers."""

from __future__ import annotations

import logging
from collections import OrderedDict

logger = logging.getLogger(__name__)


class MessageDeduplicator:
    """Bounded in-memory deduplicator for reconnect redelivery."""

    def __init__(self, processed_ids: OrderedDict[str, None], *, max_size: int) -> None:
        self._processed_ids = processed_ids
        self._max_size = max_size

    def seen(self, message_id: str) -> bool:
        """Record and check whether a message has been processed already."""
        if message_id in self._processed_ids:
            return True
        self._processed_ids[message_id] = None
        if len(self._processed_ids) > self._max_size:
            self._processed_ids.popitem(last=False)
            logger.debug("Message ID dedup cache at max size, evicted oldest entry")
        return False


__all__ = ["MessageDeduplicator"]
