"""
RoomPresence - Cross-room lifecycle management.

Extracted from ThenvoiAgent room lifecycle methods.
Handles agent's presence across rooms. Does NOT handle what happens inside rooms.
"""

from __future__ import annotations

import logging
from typing import Awaitable, Callable, Set

from thenvoi.platform.event import (
    MessageEvent,
    RoomAddedEvent,
    RoomRemovedEvent,
    PlatformEvent,
)
from thenvoi.platform.link import ThenvoiLink

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
        link = ThenvoiLink(agent_id, api_key, ...)
        presence = RoomPresence(link)

        async def on_joined(room_id: str, payload: dict):
            print(f"Joined room {room_id}")
            await link.subscribe_room(room_id)

        async def on_event(room_id: str, event: PlatformEvent):
            if isinstance(event, MessageEvent):
                print(f"Message in {room_id}: {event.payload.content}")

        presence.on_room_joined = on_joined
        presence.on_room_event = on_event

        await presence.start()
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

    async def start(self) -> None:
        """
        Start presence management.

        1. Connect link if not connected
        2. Set up event handler
        3. Subscribe to agent room events
        4. Subscribe to existing rooms (if configured)
        """
        # Set up our event handler
        self.link.on_event = self._on_platform_event

        # Connect if needed
        if not self.link.is_connected:
            await self.link.connect()

        # Subscribe to room added/removed events
        await self.link.subscribe_agent_rooms(self.link.agent_id)

        # Subscribe to existing rooms
        if self.auto_subscribe_existing:
            await self._subscribe_to_existing_rooms()

        logger.info(f"RoomPresence started for agent {self.link.agent_id}")

    async def stop(self) -> None:
        """
        Stop presence management.

        Unsubscribes from all rooms and clears state.
        Does NOT disconnect the link (caller may want to reuse it).
        """
        # Notify left for all rooms
        for room_id in list(self.rooms):
            if self.on_room_left:
                try:
                    await self.on_room_left(room_id)
                except Exception as e:
                    logger.warning(f"on_room_left error for {room_id}: {e}")

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
            case RoomRemovedEvent():
                await self._handle_room_removed(event)
            case _ if event.room_id:
                # Room-specific event - forward to on_room_event
                await self._handle_room_event(event)

    async def _handle_room_added(self, event: RoomAddedEvent) -> None:
        """
        Handle room_added event.

        Extracted from ThenvoiAgent._on_room_added().
        """
        room_id = event.room_id
        if not room_id:
            logger.warning("room_added event without room_id")
            return

        payload = event.payload.model_dump()

        # Apply filter if configured
        if self.room_filter and not self.room_filter(payload):
            logger.debug(f"Room {room_id} filtered out")
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
                logger.error(f"on_room_joined error for {room_id}: {e}", exc_info=True)

        logger.info(f"Agent joined room: {room_id}")

    async def _handle_room_removed(self, event: RoomRemovedEvent) -> None:
        """
        Handle room_removed event.

        Extracted from ThenvoiAgent._on_room_removed().
        """
        room_id = event.room_id
        if not room_id:
            logger.warning("room_removed event without room_id")
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
                logger.error(f"on_room_left error for {room_id}: {e}", exc_info=True)

        logger.info(f"Agent left room: {room_id}")

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
            logger.debug(f"Event for untracked room {room_id}, ignoring")
            return

        if self.on_room_event:
            try:
                await self.on_room_event(room_id, event)
            except Exception as e:
                logger.error(f"on_room_event error for {room_id}: {e}", exc_info=True)

    async def _subscribe_to_existing_rooms(self) -> None:
        """
        Subscribe to all rooms where agent is a participant.

        Extracted from ThenvoiAgent._subscribe_to_existing_rooms().
        """
        logger.debug("Subscribing to existing rooms")

        try:
            response = await self.link.rest.agent_api.list_agent_chats()
            if not response.data:
                return

            for room in response.data:
                room_id = room.id
                payload = (
                    room.model_dump()
                    if hasattr(room, "model_dump")
                    else {"id": room_id}
                )

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
                            f"on_room_joined error for {room_id}: {e}", exc_info=True
                        )

            logger.info(f"Subscribed to {len(self.rooms)} existing rooms")

        except Exception as e:
            logger.warning(f"Failed to subscribe to existing rooms: {e}")
