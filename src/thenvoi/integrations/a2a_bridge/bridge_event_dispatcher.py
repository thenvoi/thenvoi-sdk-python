"""Event dispatch pipeline for bridge runtime events."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from thenvoi.platform.event import (
    MessageEvent,
    ParticipantAddedEvent,
    ParticipantRemovedEvent,
    RoomAddedEvent,
    RoomRemovedEvent,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from thenvoi.client.streaming import MessageCreatedPayload
    from thenvoi.platform.link import ThenvoiLink

    from .participant_directory import ParticipantDirectory
    from .session import SessionStore

logger = logging.getLogger(__name__)


class BridgeEventDispatcher:
    """Dispatch platform events to dedicated bridge collaborators."""

    def __init__(
        self,
        *,
        link: "ThenvoiLink",
        participant_directory: "ParticipantDirectory",
        session_store: "SessionStore",
        on_message: "Callable[[str, MessageCreatedPayload], Awaitable[None]]",
    ) -> None:
        self._link = link
        self._participant_directory = participant_directory
        self._session_store = session_store
        self._on_message = on_message

    def set_link(self, link: "ThenvoiLink") -> None:
        """Update link dependency after bridge test monkeypatching."""
        self._link = link

    async def dispatch(self, event: object) -> None:
        """Dispatch one platform event."""
        match event:
            case MessageEvent(room_id=room_id, payload=payload) if room_id and payload:
                await self._on_message(room_id, payload)

            case RoomAddedEvent(room_id=room_id) if room_id:
                logger.info("Room added: %s", room_id)
                try:
                    await self._link.subscribe_room(room_id)
                except Exception:
                    logger.warning(
                        "Failed to subscribe to room %s",
                        room_id,
                        exc_info=True,
                    )
                    return
                await self._participant_directory.preload_room(room_id)

            case RoomRemovedEvent(room_id=room_id) if room_id:
                logger.info("Room removed: %s", room_id)
                try:
                    await self._link.unsubscribe_room(room_id)
                except Exception:
                    logger.warning(
                        "Failed to unsubscribe from room %s",
                        room_id,
                        exc_info=True,
                    )
                self._participant_directory.remove_room(room_id)
                await self._session_store.remove(room_id)

            case ParticipantAddedEvent(room_id=room_id, payload=payload) if (
                room_id and payload
            ):
                self._participant_directory.on_participant_added(room_id, payload)

            case ParticipantRemovedEvent(room_id=room_id, payload=payload) if (
                room_id and payload
            ):
                self._participant_directory.on_participant_removed(room_id, payload)

            case _:
                logger.debug("Unhandled event: %s", type(event).__name__)


__all__ = ["BridgeEventDispatcher"]
