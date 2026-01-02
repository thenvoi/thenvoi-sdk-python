from phoenix_channels_python_client.client import (
    PHXChannelsClient,
    PhoenixChannelsProtocolVersion,
)
from phoenix_channels_python_client.phx_messages import PHXMessage
from typing import Callable, Awaitable, Optional
from pydantic import BaseModel, ConfigDict
import logging

logger = logging.getLogger(__name__)


# WebSocket message payloads (based on actual backend messages)
# Using Pydantic for runtime validation


class Mention(BaseModel):
    """Mention object within message metadata."""

    id: str
    username: str


class MessageMetadata(BaseModel):
    """Metadata within message_created payload."""

    mentions: list[Mention]
    status: str


class MessageCreatedPayload(BaseModel):
    """Payload for message_created events (observed from real WebSocket)."""

    model_config = ConfigDict(
        extra="allow"
    )  # Allow extra fields backend might add later

    id: str
    content: str
    message_type: str
    metadata: MessageMetadata
    sender_id: str
    sender_type: str
    chat_room_id: str
    thread_id: Optional[str] = None
    inserted_at: str
    updated_at: str


class RoomOwner(BaseModel):
    """Owner object within room_added payload."""

    id: str
    name: str
    type: str


class RoomAddedPayload(BaseModel):
    """Payload for room_added events (observed from real WebSocket)."""

    model_config = ConfigDict(extra="allow")

    id: str
    owner: RoomOwner
    status: str
    type: str
    title: str
    created_at: str
    participant_role: str


class RoomRemovedPayload(BaseModel):
    """Payload for room_removed events (observed from real WebSocket)."""

    model_config = ConfigDict(extra="allow")

    id: str
    status: str
    type: str
    title: str
    removed_at: str


class ParticipantAddedPayload(BaseModel):
    """Payload for participant_added events."""

    model_config = ConfigDict(extra="allow")

    id: str
    name: str
    type: str


class ParticipantRemovedPayload(BaseModel):
    """Payload for participant_removed events."""

    model_config = ConfigDict(extra="allow")

    id: str


class WebSocketClient:
    def __init__(self, ws_url: str, api_key: str, agent_id: Optional[str] = None):
        self.ws_url = ws_url
        self.api_key = api_key
        self.agent_id = agent_id

    async def __aenter__(self):
        """Create and enter the PHXChannelsClient context"""
        self.client = PHXChannelsClient(
            self.ws_url,
            self.api_key,
            protocol_version=PhoenixChannelsProtocolVersion.V2,
        )
        if self.agent_id:
            self.client.channel_socket_url += f"&agent_id={self.agent_id}"
        await self.client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the PHXChannelsClient context"""
        if self.client:
            await self.client.__aexit__(exc_type, exc_val, exc_tb)

    async def _handle_events(self, message: PHXMessage, event_handlers: dict):
        """Generic async event handler that maps events to their corresponding async callbacks"""
        logger.debug(f"[WebSocket] Received event: {message.event}")

        # Check if we have a handler for this event
        if message.event not in event_handlers:
            logger.warning(
                f"[WebSocket] Received event '{message.event}' but no handler registered. "
                f"Available handlers: {list(event_handlers.keys())}"
            )
            return

        # Validate and parse payload into Pydantic models for known types
        if message.event == "message_created":
            validated = MessageCreatedPayload(**message.payload)
        elif message.event == "room_added":
            validated = RoomAddedPayload(**message.payload)
        elif message.event == "room_removed":
            validated = RoomRemovedPayload(**message.payload)
        else:
            # For other events (participant_added, participant_removed, etc.)
            # pass the raw payload dict
            validated = message.payload

        callback = event_handlers[message.event]
        if callback:
            await callback(validated)

    async def join_agent_rooms_channel(
        self,
        agent_id: str,
        on_room_added: Callable[[RoomAddedPayload], Awaitable[None]],
        on_room_removed: Callable[[RoomRemovedPayload], Awaitable[None]],
    ):
        """Subscribe to agent rooms topic with async callbacks"""
        topic = f"agent_rooms:{agent_id}"
        logger.info(f"[WebSocket] Subscribing to topic: {topic}")

        async def message_handler(message):
            await self._handle_events(
                message, {"room_added": on_room_added, "room_removed": on_room_removed}
            )

        result = await self.client.subscribe_to_topic(topic, message_handler)
        logger.info(f"[WebSocket] Subscribed to topic: {topic}")
        return result

    async def join_chat_room_channel(
        self,
        chat_room_id: str,
        on_message_created: Callable[[MessageCreatedPayload], Awaitable[None]],
    ):
        """Subscribe to chat room topic for message events with async callback"""
        topic = f"chat_room:{chat_room_id}"
        logger.info(f"[WebSocket] Subscribing to topic: {topic}")

        async def message_handler(message):
            await self._handle_events(message, {"message_created": on_message_created})

        return await self.client.subscribe_to_topic(topic, message_handler)

    async def join_user_rooms_channel(
        self,
        user_id: str,
        on_room_added: Callable[[RoomAddedPayload], Awaitable[None]],
        on_room_removed: Callable[[RoomRemovedPayload], Awaitable[None]],
    ):
        """Subscribe to user rooms topic with async callbacks"""
        topic = f"user_rooms:{user_id}"

        async def message_handler(message):
            await self._handle_events(
                message, {"room_added": on_room_added, "room_removed": on_room_removed}
            )

        return await self.client.subscribe_to_topic(topic, message_handler)

    async def join_room_participants_channel(
        self,
        chat_room_id: str,
        on_participant_added: Callable[[dict], Awaitable[None]],
        on_participant_removed: Callable[[dict], Awaitable[None]],
    ):
        """Subscribe to room participants topic with async callbacks"""
        topic = f"room_participants:{chat_room_id}"
        logger.info(f"[WebSocket] Subscribing to topic: {topic}")

        async def message_handler(message):
            await self._handle_events(
                message,
                {
                    "participant_added": on_participant_added,
                    "participant_removed": on_participant_removed,
                },
            )

        return await self.client.subscribe_to_topic(topic, message_handler)

    async def join_tasks_channel(
        self,
        user_id: str,
        on_task_created: Callable[[dict], Awaitable[None]],
        on_task_updated: Callable[[dict], Awaitable[None]],
    ):
        """Subscribe to tasks topic with async callbacks"""
        topic = f"tasks:{user_id}"

        async def message_handler(message):
            await self._handle_events(
                message,
                {"task_created": on_task_created, "task_updated": on_task_updated},
            )

        return await self.client.subscribe_to_topic(topic, message_handler)

    async def leave_agent_rooms_channel(self, agent_id: str):
        """Unsubscribe from agent rooms topic"""
        topic = f"agent_rooms:{agent_id}"
        logger.info(f"[WebSocket] Unsubscribing from topic: {topic}")
        return await self.client.unsubscribe_from_topic(topic)

    async def leave_chat_room_channel(self, chat_room_id: str):
        """Unsubscribe from chat room topic"""
        topic = f"chat_room:{chat_room_id}"
        logger.info(f"[WebSocket] Unsubscribing from topic: {topic}")
        return await self.client.unsubscribe_from_topic(topic)

    async def leave_user_rooms_channel(self, user_id: str):
        """Unsubscribe from user rooms topic"""
        topic = f"user_rooms:{user_id}"
        return await self.client.unsubscribe_from_topic(topic)

    async def leave_room_participants_channel(self, chat_room_id: str):
        """Unsubscribe from room participants topic"""
        topic = f"room_participants:{chat_room_id}"
        logger.info(f"[WebSocket] Unsubscribing from topic: {topic}")
        return await self.client.unsubscribe_from_topic(topic)

    async def leave_tasks_channel(self, user_id: str):
        """Unsubscribe from tasks topic"""
        topic = f"tasks:{user_id}"
        return await self.client.unsubscribe_from_topic(topic)

    async def run_forever(self):
        await self.client.run_forever()
