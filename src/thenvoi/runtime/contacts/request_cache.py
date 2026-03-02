"""Request enrichment cache for contact event handling."""

from __future__ import annotations

import logging
from collections import OrderedDict

from thenvoi.platform.event import (
    ContactEvent,
    ContactRequestReceivedEvent,
    ContactRequestUpdatedEvent,
)

from .service import ContactService

logger = logging.getLogger(__name__)


class ContactRequestInfoStore:
    """Cache and fetch contact request sender/recipient metadata for updates."""

    def __init__(self, contact_service: ContactService, *, max_cache_size: int) -> None:
        self._contact_service = contact_service
        self._max_cache_size = max_cache_size
        self._request_cache: OrderedDict[str, dict[str, str | None]] = OrderedDict()

    def get_cached(self, request_id: str) -> dict[str, str | None] | None:
        """Return cached request info by ID if present."""
        return self._request_cache.get(request_id)

    def cache_from_event(self, event: ContactEvent) -> None:
        """Cache sender info from contact request received events."""
        match event:
            case ContactRequestReceivedEvent(payload=payload):
                if payload is None:
                    return
                self._request_cache[payload.id] = {
                    "from_handle": payload.from_handle,
                    "from_name": payload.from_name,
                    "message": payload.message,
                }
                self._trim_cache()
            case _:
                return

    async def enrich_update_event(
        self, event: ContactRequestUpdatedEvent
    ) -> dict[str, str | None] | None:
        """Resolve sender/recipient info for a request update event."""
        if event.payload is None:
            return None

        request_id = event.payload.id
        cached = self._request_cache.get(request_id)
        if cached:
            return cached

        logger.debug("Cache miss for request %s, fetching from API", request_id)
        return await self.fetch_request_details(request_id)

    async def fetch_request_details(
        self, request_id: str
    ) -> dict[str, str | None] | None:
        """Fetch details from API and populate cache on success."""
        return await self._fetch_request_details(request_id)

    async def _fetch_request_details(
        self, request_id: str
    ) -> dict[str, str | None] | None:
        """Fetch request details from API when cache misses."""
        try:
            response = await self._contact_service.list_contact_requests(
                page=1,
                page_size=100,
            )
        except Exception as error:
            logger.warning("Failed to fetch request details from API: %s", error)
            return None

        for request in response.get("received", []):
            if request.get("id") == request_id:
                info = {
                    "from_handle": request.get("from_handle"),
                    "from_name": request.get("from_name"),
                    "message": request.get("message"),
                }
                self._request_cache[request_id] = info
                self._trim_cache()
                return info

        for request in response.get("sent", []):
            if request.get("id") == request_id:
                info = {
                    "to_handle": request.get("to_handle"),
                    "to_name": request.get("to_name"),
                    "message": request.get("message"),
                }
                self._request_cache[request_id] = info
                self._trim_cache()
                return info

        logger.debug("Request not found in API: %s", request_id)
        return None

    def _trim_cache(self) -> None:
        while len(self._request_cache) > self._max_cache_size:
            self._request_cache.popitem(last=False)
