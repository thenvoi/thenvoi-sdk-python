"""
RoomPresence - Cross-room lifecycle management.

Extracted from ThenvoiAgent room lifecycle methods.
Handles agent's presence across rooms. Does NOT handle what happens inside rooms.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, Set

from thenvoi.client.rest import DEFAULT_REQUEST_OPTIONS
from thenvoi.platform.event import (
    RoomAddedEvent,
    RoomDeletedEvent,
    RoomRemovedEvent,
    DisconnectedEvent,
    PlatformEvent,
    ContactEvent,
    ContactRequestReceivedEvent,
    ContactRequestUpdatedEvent,
    ContactAddedEvent,
    ContactRemovedEvent,
)
from thenvoi.platform.link import ThenvoiLink

# Type alias for contact event callback (agent-level, no tools)
ContactEventHandler = Callable[[ContactEvent], Awaitable[None]]

logger = logging.getLogger(__name__)


class RoomPresence:
    """
    Manages agent's presence across rooms.

    Cross-room only. Does NOT handle what happens inside rooms.
    That's the job of Execution implementations.

    Extracted from ThenvoiAgent room lifecycle:
    - _on_room_added() -> on_room_joined callback
    - _on_room_removed() -> on_room_left callback
    - _subscribe_to_existing_rooms() -> start() auto-subscription

    Example:
        import logging
        logger = logging.getLogger(__name__)

        link = ThenvoiLink(agent_id, api_key, ...)
        presence = RoomPresence(link)

        async def on_joined(room_id: str, payload: dict):
            logger.info("Joined room %s", room_id)

        async def on_event(room_id: str, event: PlatformEvent):
            if isinstance(event, MessageEvent):
                logger.info("Message in %s: %s", room_id, event.payload.content)

        presence.on_room_joined = on_joined
        presence.on_room_event = on_event

        await presence.start()
        # Presence now consumes events via async iterator internally
        await link.run_forever()
    """

    def __init__(
        self,
        link: ThenvoiLink,
        room_filter: Callable[[dict], bool] | None = None,
        auto_subscribe_existing: bool = True,
    ):
        """
        Initialize RoomPresence.

        Args:
            link: ThenvoiLink for WebSocket events
            room_filter: Optional filter to decide which rooms to join
            auto_subscribe_existing: Subscribe to existing rooms on start
        """
        self.link = link
        self.room_filter = room_filter
        self.auto_subscribe_existing = auto_subscribe_existing

        # Track rooms we're present in
        self.rooms: Set[str] = set()

        # Callbacks (set by user or AgentRuntime)
        self.on_room_joined: Callable[[str, dict], Awaitable[None]] | None = None
        self.on_room_left: Callable[[str], Awaitable[None]] | None = None
        self.on_room_event: Callable[[str, PlatformEvent], Awaitable[None]] | None = (
            None
        )
        self.on_contact_event: ContactEventHandler | None = None
        self.on_disconnected: Callable[[str], Awaitable[None]] | None = None

        # Internal task for consuming events from link
        self._event_task: asyncio.Task | None = None

    async def start(self) -> None:
        """
        Start presence management.

        1. Connect link if not connected
        2. Subscribe to agent room events
        3. Subscribe to existing rooms (if configured)
        4. Spawn task to consume events from link
        """
        # Connect if needed
        if not self.link.is_connected:
            await self.link.connect()

        # Subscribe to room added/removed events
        await self.link.subscribe_agent_rooms(self.link.agent_id)

        # Subscribe to existing rooms
        if self.auto_subscribe_existing:
            await self._subscribe_to_existing_rooms()

        # Spawn task to consume events from link's async iterator
        self._event_task = asyncio.create_task(self._consume_events())

        logger.info("RoomPresence started for agent %s", self.link.agent_id)

    async def _consume_events(self) -> None:
        """Consume events from link's async iterator."""
        try:
            async for event in self.link:
                await self._on_platform_event(event)
        except asyncio.CancelledError:
            logger.debug("Event consumer task cancelled")
        except Exception as e:
            logger.error("Error in event consumer: %s", e, exc_info=True)

    async def stop(self) -> None:
        """
        Stop presence management.

        Cancels event consumer, unsubscribes from all rooms and clears state.
        Does NOT disconnect the link (caller may want to reuse it).
        """
        # Cancel event consumer task
        if self._event_task and not self._event_task.done():
            self._event_task.cancel()
            try:
                await self._event_task
            except asyncio.CancelledError:
                pass
            self._event_task = None

        # Notify left for all rooms
        for room_id in list(self.rooms):
            if self.on_room_left:
                try:
                    await self.on_room_left(room_id)
                except Exception as e:
                    logger.warning("on_room_left error for %s: %s", room_id, e)

        self.rooms.clear()
        logger.info("RoomPresence stopped")

    async def _on_platform_event(self, event: PlatformEvent) -> None:
        """
        Handle platform events from ThenvoiLink.

        Routes to appropriate handler based on event type.
        """
        match event:
            case RoomAddedEvent():
                await self._handle_room_added(event)
            case RoomRemovedEvent() | RoomDeletedEvent():
                await self._handle_room_left(event)
            case DisconnectedEvent():
                await self._handle_disconnected(event)
            case (
                ContactRequestReceivedEvent()
                | ContactRequestUpdatedEvent()
                | ContactAddedEvent()
                | ContactRemovedEvent()
            ):
                # Contact events have no room_id - forward to contact handler
                await self._handle_contact_event(event)
            case _ if event.room_id:
                # Room-specific event - forward to on_room_event
                await self._handle_room_event(event)

    async def _handle_room_added(self, event: RoomAddedEvent) -> None:
        """
        Handle room_added event.

        Extracted from ThenvoiAgent._on_room_added().
        """
        room_id = event.room_id
        if not room_id or not event.payload:
            logger.warning("room_added event without room_id or payload")
            return

        payload = event.payload.model_dump(exclude_none=True)

        # Apply filter if configured
        if self.room_filter and not self.room_filter(payload):
            logger.debug("Room %s filtered out", room_id)
            return

        # Track room
        self.rooms.add(room_id)

        # Subscribe to room channels
        await self.link.subscribe_room(room_id)

        # Notify callback
        if self.on_room_joined:
            try:
                await self.on_room_joined(room_id, payload)
            except Exception as e:
                logger.error(
                    "on_room_joined error for %s: %s", room_id, e, exc_info=True
                )

        logger.info("Agent joined room: %s", room_id)

    async def _handle_room_removed(self, event: RoomRemovedEvent) -> None:
        """Handle room_removed event."""
        await self._handle_room_left(event)

    async def _handle_room_left(
        self, event: RoomRemovedEvent | RoomDeletedEvent
    ) -> None:
        """
        Handle room_removed and room_deleted events.

        Both events mean the room should be torn down locally.
        """
        room_id = event.room_id
        if not room_id:
            logger.warning("%s event without room_id", event.type)
            return

        # Unsubscribe from room channels
        await self.link.unsubscribe_room(room_id)

        # Untrack room
        self.rooms.discard(room_id)

        # Notify callback
        if self.on_room_left:
            try:
                await self.on_room_left(room_id)
            except Exception as e:
                logger.error("on_room_left error for %s: %s", room_id, e, exc_info=True)

        logger.info("Agent left room via %s: %s", event.type, room_id)

    async def _handle_room_event(self, event: PlatformEvent) -> None:
        """
        Handle room-specific events (message, participant changes).

        Forwards to on_room_event callback.
        """
        room_id = event.room_id
        if not room_id:
            return

        # Only forward events for rooms we're tracking
        if room_id not in self.rooms:
            logger.debug("Event for untracked room %s, ignoring", room_id)
            return

        if self.on_room_event:
            try:
                await self.on_room_event(room_id, event)
            except Exception as e:
                logger.error(
                    "on_room_event error for %s: %s", room_id, e, exc_info=True
                )

    async def _handle_contact_event(self, event: ContactEvent) -> None:
        """
        Handle contact events (requests, added, removed).

        Contact events have no room context and are agent-level.
        Forwards to on_contact_event callback.
        """
        if self.on_contact_event:
            try:
                await self.on_contact_event(event)
            except Exception as e:
                logger.error(
                    "on_contact_event error for %s: %s",
                    type(event).__name__,
                    e,
                    exc_info=True,
                )

    async def _handle_disconnected(self, event: DisconnectedEvent) -> None:
        """Handle platform disconnect event.

        Logs the reason and forwards to ``on_disconnected`` callback so
        the runtime / adapter layer can react (e.g. stop processing).
        """
        logger.warning("Platform disconnected: %s", event.reason)
        if self.on_disconnected:
            try:
                await self.on_disconnected(event.reason)
            except Exception as e:
                logger.error("on_disconnected callback error: %s", e, exc_info=True)

    async def _subscribe_to_existing_rooms(self) -> None:
        """
        Subscribe to all rooms where agent is a participant.

        Extracted from ThenvoiAgent._subscribe_to_existing_rooms().
        """
        logger.debug("Subscribing to existing rooms")

        try:
            response = await self.link.rest.agent_api_chats.list_agent_chats(
                request_options=DEFAULT_REQUEST_OPTIONS,
            )
            if not response.data:
                return

            for room in response.data:
                room_id = room.id
                payload = room.model_dump(exclude_none=True)

                # Apply filter if configured
                if self.room_filter and not self.room_filter(payload):
                    continue

                # Track and subscribe
                self.rooms.add(room_id)
                await self.link.subscribe_room(room_id)

                # Notify callback
                if self.on_room_joined:
                    try:
                        await self.on_room_joined(room_id, payload)
                    except Exception as e:
                        logger.error(
                            "on_room_joined error for %s: %s",
                            room_id,
                            e,
                            exc_info=True,
                        )

            logger.info("Subscribed to %s existing rooms", len(self.rooms))

        except Exception as e:
            logger.warning("Failed to subscribe to existing rooms: %s", e)
