"""Hub room coordination for contact event routing."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from thenvoi.client.rest import DEFAULT_REQUEST_OPTIONS, ChatRoomRequest

logger = logging.getLogger(__name__)


class HubRoomCoordinator:
    """Coordinate hub room creation, readiness, and initialization state."""

    def __init__(
        self,
        *,
        create_room: Callable[..., Awaitable[Any]],
        task_id: str | None,
    ) -> None:
        self._create_room = create_room
        self._task_id = task_id
        self._room_id: str | None = None
        self._initialized = False
        self._ready = asyncio.Event()
        self._lock = asyncio.Lock()

    @property
    def room_id(self) -> str | None:
        return self._room_id

    async def get_or_create_room_id(self) -> str:
        """Create the hub room once and return its ID."""
        async with self._lock:
            if self._room_id is not None:
                return self._room_id

            logger.info(
                "Creating hub room for contact events at startup (task_id=%s)",
                self._task_id or "none",
            )
            response = await self._create_room(
                chat=ChatRoomRequest(task_id=self._task_id or None),
                request_options=DEFAULT_REQUEST_OPTIONS,
            )
            self._room_id = response.data.id
            logger.info("Hub room created at startup: %s", self._room_id)
            return self._room_id

    def mark_ready(self) -> None:
        """Mark hub room runtime as ready for event injection."""
        self._ready.set()
        logger.info("Hub room marked as ready: %s", self._room_id)

    async def wait_ready(self, timeout_seconds: float = 5.0) -> bool:
        """Wait for hub room readiness."""
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=timeout_seconds)
            return True
        except asyncio.TimeoutError:
            return False

    def should_initialize_prompt(self) -> bool:
        """Return True once, then False for subsequent checks."""
        if self._initialized:
            return False
        self._initialized = True
        return True


__all__ = ["HubRoomCoordinator"]
