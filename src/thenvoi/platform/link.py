"""
ThenvoiLink - Live link to Thenvoi platform.

Extracted from core/agent.py ThenvoiAgent - WebSocket management only.
REST client exposed directly for API calls.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Awaitable, Callable, Set

from thenvoi.client.rest import AsyncRestClient
from thenvoi.client.streaming import WebSocketClient

from .event import PlatformEvent

if TYPE_CHECKING:
    from thenvoi.client.streaming import (
        MessageCreatedPayload,
        RoomAddedPayload,
        RoomRemovedPayload,
    )
    from thenvoi.runtime.types import PlatformMessage

logger = logging.getLogger(__name__)


class ThenvoiLink:
    """
    Live link to Thenvoi platform.

    Extracted from ThenvoiAgent - handles WebSocket connection and event dispatch.
    REST client exposed directly via self.rest for API calls.

    Example:
        link = ThenvoiLink(agent_id="...", api_key="...")

        async def handle_event(event: PlatformEvent):
            if event.is_message:
                print(f"Message: {event.payload['content']}")
            elif event.is_room_added:
                await link.subscribe_room(event.room_id)

        link.on_event = handle_event
        await link.connect()
        await link.subscribe_agent_rooms(agent_id)
        await link.run_forever()
    """

    def __init__(
        self,
        agent_id: str,
        api_key: str,
        ws_url: str = "wss://api.thenvoi.com/ws",
        rest_url: str = "https://api.thenvoi.com",
    ):
        self.agent_id = agent_id
        self.api_key = api_key
        self.ws_url = ws_url
        self.rest_url = rest_url

        # REST client - exposed directly (from ThenvoiAgent._api_client)
        self.rest = AsyncRestClient(api_key=api_key, base_url=rest_url)

        # WebSocket client (from ThenvoiAgent._ws_client)
        self._ws: WebSocketClient | None = None
        self._is_connected = False

        # Subscription tracking (from ThenvoiAgent._subscribed_rooms)
        self._subscribed_rooms: Set[str] = set()

        # Single event callback (replaces ThenvoiAgent's multiple handlers)
        self.on_event: Callable[[PlatformEvent], Awaitable[None]] | None = None

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    # --- Connection lifecycle (from ThenvoiAgent.start/stop/run) ---

    async def connect(self) -> None:
        """
        Connect WebSocket.

        Extracted from ThenvoiAgent.start() lines 158-164.
        """
        if self._is_connected:
            logger.warning("Already connected")
            return

        self._ws = WebSocketClient(self.ws_url, self.api_key, self.agent_id)
        await self._ws.__aenter__()
        self._is_connected = True
        logger.info("Connected to platform")

    async def disconnect(self) -> None:
        """
        Disconnect WebSocket.

        Extracted from ThenvoiAgent.stop() lines 193-195.
        """
        if not self._is_connected or not self._ws:
            return

        await self._ws.__aexit__(None, None, None)
        self._ws = None
        self._is_connected = False
        self._subscribed_rooms.clear()
        logger.info("Disconnected from platform")

    async def run_forever(self) -> None:
        """
        Run until interrupted.

        From ThenvoiAgent.run() lines 208-209.
        """
        if not self._ws:
            raise RuntimeError("Not connected")
        await self._ws.run_forever()

    # --- Subscription management (from ThenvoiAgent) ---

    async def subscribe_agent_rooms(self, agent_id: str) -> None:
        """
        Subscribe to agent room events (room_added/removed).

        From ThenvoiAgent.start() lines 167-171.
        """
        if not self._ws:
            raise RuntimeError("Not connected")

        await self._ws.join_agent_rooms_channel(
            agent_id,
            on_room_added=self._on_room_added,
            on_room_removed=self._on_room_removed,
        )

    async def subscribe_room(self, room_id: str) -> None:
        """
        Subscribe to room messages and participants.

        Extracted from ThenvoiAgent._subscribe_to_room() lines 724-746.
        """
        if not self._ws:
            raise RuntimeError("Not connected")

        if room_id in self._subscribed_rooms:
            return

        # Subscribe to messages (from lines 733-736)
        await self._ws.join_chat_room_channel(
            room_id,
            on_message_created=lambda msg: self._on_message_created(room_id, msg),
        )

        # Subscribe to participant updates (from lines 739-743)
        await self._ws.join_room_participants_channel(
            room_id,
            on_participant_added=lambda p: self._on_participant_added(room_id, p),
            on_participant_removed=lambda p: self._on_participant_removed(room_id, p),
        )

        self._subscribed_rooms.add(room_id)
        logger.debug(f"Subscribed to room {room_id}")

    async def unsubscribe_room(self, room_id: str) -> None:
        """
        Unsubscribe from room.

        Extracted from ThenvoiAgent._unsubscribe_from_room() lines 748-769.
        """
        if not self._ws or room_id not in self._subscribed_rooms:
            return

        self._subscribed_rooms.discard(room_id)

        try:
            await self._ws.leave_chat_room_channel(room_id)
        except Exception as e:
            logger.warning(f"Error unsubscribing from chat_room:{room_id}: {e}")

        try:
            await self._ws.leave_room_participants_channel(room_id)
        except Exception as e:
            logger.warning(f"Error unsubscribing from room_participants:{room_id}: {e}")

        logger.debug(f"Unsubscribed from room {room_id}")

    # --- Event handlers (from ThenvoiAgent, unified into PlatformEvent) ---

    async def _dispatch(self, event: PlatformEvent) -> None:
        """Dispatch event via callback. Non-blocking via create_task."""
        if self.on_event:
            asyncio.create_task(self._safe_dispatch(event))

    async def _safe_dispatch(self, event: PlatformEvent) -> None:
        """Dispatch with exception handling."""
        try:
            if self.on_event:
                await self.on_event(event)
        except Exception as e:
            logger.error(f"Event handler error for {event.type}: {e}", exc_info=True)

    async def _on_room_added(self, payload: "RoomAddedPayload") -> None:
        """
        Handle room_added from WebSocket.

        From ThenvoiAgent._on_room_added() lines 619-630.
        Now creates PlatformEvent instead of calling _subscribe_to_room/_create_session.
        """
        event = PlatformEvent(
            type="room_added",
            room_id=payload.id,
            payload=payload.model_dump(),
        )
        await self._dispatch(event)

    async def _on_room_removed(self, payload: "RoomRemovedPayload") -> None:
        """
        Handle room_removed from WebSocket.

        From ThenvoiAgent._on_room_removed() lines 632-643.
        """
        event = PlatformEvent(
            type="room_removed",
            room_id=payload.id,
            payload=payload.model_dump(),
        )
        await self._dispatch(event)

    async def _on_message_created(
        self, room_id: str, payload: "MessageCreatedPayload"
    ) -> None:
        """
        Handle message_created from WebSocket.

        From ThenvoiAgent._on_message_created() lines 645-682.
        Now creates PlatformEvent instead of routing to session.
        """
        event = PlatformEvent(
            type="message_created",
            room_id=room_id,
            payload=payload.model_dump(),
        )
        await self._dispatch(event)

    async def _on_participant_added(self, room_id: str, payload: dict) -> None:
        """
        Handle participant_added from WebSocket.

        From ThenvoiAgent._on_participant_added() lines 771-786.
        """
        event = PlatformEvent(
            type="participant_added",
            room_id=room_id,
            payload=payload if isinstance(payload, dict) else dict(payload),
        )
        await self._dispatch(event)

    async def _on_participant_removed(self, room_id: str, payload: dict) -> None:
        """
        Handle participant_removed from WebSocket.

        From ThenvoiAgent._on_participant_removed() lines 788-805.
        """
        event = PlatformEvent(
            type="participant_removed",
            room_id=room_id,
            payload=payload if isinstance(payload, dict) else dict(payload),
        )
        await self._dispatch(event)

    # --- Message lifecycle (SDK internal operations) ---

    async def mark_processing(self, room_id: str, message_id: str) -> None:
        """
        Mark message as being processed on the server.

        Tells the server this message is being handled, so /next won't return it.
        """
        logger.debug(f"Marking message {message_id} as processing")
        try:
            await self.rest.agent_api.mark_agent_message_processing(
                chat_id=room_id,
                id=message_id,
            )
        except Exception as e:
            logger.warning(f"Failed to mark message {message_id} as processing: {e}")

    async def mark_processed(self, room_id: str, message_id: str) -> None:
        """
        Mark message as successfully processed on the server.

        Clears the message from unprocessed queue.
        """
        logger.debug(f"Marking message {message_id} as processed")
        try:
            await self.rest.agent_api.mark_agent_message_processed(
                chat_id=room_id,
                id=message_id,
            )
        except Exception as e:
            logger.warning(f"Failed to mark message {message_id} as processed: {e}")

    async def mark_failed(self, room_id: str, message_id: str, error: str) -> None:
        """
        Mark message as failed on the server.

        Records the error and may trigger retry logic on the server side.
        """
        logger.warning(f"Marking message {message_id} as failed: {error}")
        try:
            await self.rest.agent_api.mark_agent_message_failed(
                chat_id=room_id,
                id=message_id,
                error=error,
            )
        except Exception as e:
            logger.warning(f"Failed to mark message {message_id} as failed: {e}")

    async def get_next_message(self, room_id: str) -> "PlatformMessage | None":
        """
        Get next unprocessed message for a room from the server.

        Used during sync to process backlog messages missed while offline.

        Returns:
            PlatformMessage if there's an unprocessed message,
            None if no unprocessed messages (204 No Content) or on error.
        """
        from thenvoi_rest.core.api_error import ApiError
        from thenvoi.runtime.types import PlatformMessage

        logger.debug(f"Getting next message for room {room_id}")
        try:
            response = await self.rest.agent_api.get_agent_next_message(
                chat_id=room_id,
            )
            if response.data is None:
                return None

            item = response.data
            return PlatformMessage(
                id=item.id,
                room_id=item.chat_room_id or room_id,
                content=item.content,
                sender_id=item.sender_id,
                sender_type=item.sender_type,
                sender_name=item.sender_name or "",
                message_type=item.message_type,
                metadata=item.metadata or {},
                created_at=item.inserted_at or datetime.now(timezone.utc),
            )
        except ApiError as e:
            # 204 No Content means no unprocessed messages - expected
            if e.status_code == 204:
                logger.debug(f"No unprocessed messages for room {room_id}")
                return None
            logger.warning(f"Failed to get next message: {e}")
            return None
        except Exception as e:
            logger.warning(f"Failed to get next message: {e}")
            return None
