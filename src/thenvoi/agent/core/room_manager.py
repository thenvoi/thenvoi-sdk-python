"""
Room management for Thenvoi agents.

Handles the common pattern of:
- Subscribing to rooms
- Listening for room events
- Managing room add/remove

Each agent can plug in their own message handlers.
"""

import logging
from typing import Callable, Awaitable, List, Set

from thenvoi.client.rest import AsyncRestClient
from thenvoi.client.streaming import (
    WebSocketClient,
    MessageCreatedPayload,
    RoomAddedPayload,
    RoomRemovedPayload,
)

logger = logging.getLogger(__name__)


# Type aliases for handlers
MessageHandler = Callable[[MessageCreatedPayload], Awaitable[None]]
RoomEventHandler = Callable[[str], Awaitable[None]]


class RoomManager:
    """
    Manages room subscriptions and events for an agent.

    This handles the WHAT (subscribe to rooms, listen for events)
    while letting each framework decide the HOW (handle messages).
    """

    def __init__(
        self,
        agent_id: str,
        agent_name: str,
        api_client: AsyncRestClient,
        ws_client: WebSocketClient,
        message_handler: MessageHandler,
        on_room_added: RoomEventHandler | None = None,
        on_room_removed: RoomEventHandler | None = None,
    ):
        """
        Initialize room manager.

        Args:
            agent_id: Agent ID on platform
            agent_name: Agent name for @mention filtering
            api_client: API client for fetching room/participant data
            ws_client: WebSocket client for subscriptions
            message_handler: Async function to handle incoming messages
            on_room_added: Optional handler for room additions
            on_room_removed: Optional handler for room removals
        """
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.api_client = api_client
        self.ws_client = ws_client
        self.message_handler = message_handler
        self.on_room_added = on_room_added
        self.on_room_removed = on_room_removed
        self._subscribed_rooms: Set[str] = set()

    async def get_agent_rooms(self) -> List[str]:
        """Get list of room IDs where this agent is a participant."""
        response = await self.api_client.agent_api.list_agent_chats()
        if not response or not response.data:
            return []

        # list_agent_chats already returns only rooms where agent is a participant
        return [room.id for room in response.data]

    def _is_message_for_agent(self, message: MessageCreatedPayload) -> bool:
        """
        Check if message should be handled by this agent (platform rule).

        Platform rules:
        1. Ignore own messages
        2. Only respond to messages that @mention the agent (check metadata.mentions)
        """
        # Ignore own messages
        if message.sender_id == self.agent_id:
            logger.debug("Ignoring own message")
            return False

        # Check if agent is mentioned in metadata (reliable, structured data)
        is_mentioned = any(
            mention.id == self.agent_id for mention in message.metadata.mentions
        )

        if not is_mentioned:
            logger.debug("Agent not mentioned in message")

        return is_mentioned

    async def subscribe_to_room(self, room_id: str):
        """Subscribe to a single room."""
        if room_id in self._subscribed_rooms:
            logger.debug(f"Already subscribed to room: {room_id}")
            return

        async def room_message_wrapper(message: MessageCreatedPayload):
            """Apply platform filtering and pass to framework handler."""
            # Apply platform-level filtering
            if not self._is_message_for_agent(message):
                return

            # Pass to framework-specific handler
            await self.message_handler(message)

        await self.ws_client.join_chat_room_channel(room_id, room_message_wrapper)
        self._subscribed_rooms.add(room_id)
        logger.info(f"Subscribed to room: {room_id}")

    async def unsubscribe_from_room(self, room_id: str):
        """Unsubscribe from a single room."""
        if room_id not in self._subscribed_rooms:
            logger.debug(f"Not subscribed to room: {room_id}")
            return

        await self.ws_client.leave_chat_room_channel(room_id)
        self._subscribed_rooms.discard(room_id)
        logger.info(f"Unsubscribed from room: {room_id}")

    async def subscribe_to_all_rooms(self):
        """Subscribe to all rooms where agent is a participant."""
        rooms = await self.get_agent_rooms()
        logger.debug(f"Found {len(rooms)} rooms for agent")

        for room_id in rooms:
            await self.subscribe_to_room(room_id)

        return len(rooms)

    async def subscribe_to_room_events(self):
        """Subscribe to room add/remove events for this agent."""

        async def room_added_wrapper(room_data: RoomAddedPayload):
            room_id = room_data.id
            logger.info(f"Agent added to room: {room_id}")
            await self.subscribe_to_room(room_id)
            if self.on_room_added:
                await self.on_room_added(room_id)

        async def room_removed_wrapper(room_data: RoomRemovedPayload):
            room_id = room_data.id
            logger.info(f"Agent removed from room: {room_id}")
            await self.unsubscribe_from_room(room_id)
            if self.on_room_removed:
                await self.on_room_removed(room_id)

        await self.ws_client.join_agent_rooms_channel(
            self.agent_id, room_added_wrapper, room_removed_wrapper
        )
        logger.debug("Subscribed to room events")

    async def get_participant_name(
        self, participant_id: str, participant_type: str, room_id: str
    ) -> str:
        """Get participant name from room participants (helper utility)."""
        try:
            participants_response = (
                await self.api_client.agent_api.list_agent_chat_participants(
                    chat_id=room_id
                )
            )
            if not participants_response.data:
                return f"Unknown {participant_type}"

            for participant in participants_response.data:
                if (
                    participant.id == participant_id
                    and participant.type == participant_type
                ):
                    return participant.name or f"Unknown {participant_type}"
            return f"Unknown {participant_type}"
        except Exception as e:
            logger.error(f"Error getting participant name: {e}")
            return f"Unknown {participant_type}"
