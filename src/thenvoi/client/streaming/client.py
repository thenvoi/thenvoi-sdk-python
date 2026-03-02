from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from phoenix_channels_python_client.client import (
    PHXChannelsClient,
    PhoenixChannelsProtocolVersion,
)
from phoenix_channels_python_client.phx_messages import PHXMessage
from pydantic import BaseModel, ConfigDict, ValidationError

logger = logging.getLogger(__name__)


# WebSocket message payloads (based on actual backend messages)
# Using Pydantic for runtime validation


class Mention(BaseModel):
    """Mention object within message metadata."""

    model_config = ConfigDict(extra="allow")

    id: str
    handle: str | None = None
    name: str | None = None
    username: str | None = None


class MessageMetadata(BaseModel):
    """Metadata within message_created payload."""

    model_config = ConfigDict(extra="allow")

    mentions: list[Mention] | None = None


class MessageCreatedPayload(BaseModel):
    """Payload for message_created events (observed from real WebSocket)."""

    model_config = ConfigDict(
        extra="allow"
    )  # Allow extra fields backend might add later

    id: str
    content: str
    message_type: str
    metadata: MessageMetadata | None = None
    sender_id: str
    sender_type: str
    sender_name: str | None = None
    chat_room_id: str | None = None
    thread_id: str | None = None
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
    title: str | None = None
    task_id: str | None = None
    inserted_at: str | None = None
    updated_at: str | None = None
    owner: RoomOwner | None = None
    status: str | None = None
    type: str | None = None
    created_at: str | None = None
    participant_role: str | None = None


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


# Contact event payloads


class ContactRequestReceivedPayload(BaseModel):
    """Payload for contact_request_received events."""

    model_config = ConfigDict(extra="allow")

    id: str
    from_handle: str
    from_name: str
    message: str | None = None
    status: str
    inserted_at: str


class ContactRequestUpdatedPayload(BaseModel):
    """Payload for contact_request_updated events."""

    model_config = ConfigDict(extra="allow")

    id: str
    status: str


class ContactAddedPayload(BaseModel):
    """Payload for contact_added events."""

    model_config = ConfigDict(extra="allow")

    id: str
    handle: str
    name: str
    type: str
    description: str | None = None
    is_external: bool | None = None
    inserted_at: str


class ContactRemovedPayload(BaseModel):
    """Payload for contact_removed events."""

    model_config = ConfigDict(extra="allow")

    id: str


_PAYLOAD_MODELS: dict[str, type[BaseModel]] = {
    "message_created": MessageCreatedPayload,
    "room_added": RoomAddedPayload,
    "room_removed": RoomRemovedPayload,
    "participant_added": ParticipantAddedPayload,
    "participant_removed": ParticipantRemovedPayload,
    "contact_request_received": ContactRequestReceivedPayload,
    "contact_request_updated": ContactRequestUpdatedPayload,
    "contact_added": ContactAddedPayload,
    "contact_removed": ContactRemovedPayload,
}

EventCallback = Callable[[Any], Awaitable[None]]


@dataclass(frozen=True)
class ChannelSpec:
    """Declarative WebSocket channel contract."""

    topic_prefix: str
    events: tuple[str, ...]


_CHANNEL_SPECS: dict[str, ChannelSpec] = {
    "agent_rooms": ChannelSpec(
        topic_prefix="agent_rooms",
        events=("room_added", "room_removed"),
    ),
    "chat_room": ChannelSpec(
        topic_prefix="chat_room",
        events=("message_created",),
    ),
    "user_rooms": ChannelSpec(
        topic_prefix="user_rooms",
        events=("room_added", "room_removed"),
    ),
    "room_participants": ChannelSpec(
        topic_prefix="room_participants",
        events=("participant_added", "participant_removed"),
    ),
    "tasks": ChannelSpec(
        topic_prefix="tasks",
        events=("task_created", "task_updated"),
    ),
    "agent_contacts": ChannelSpec(
        topic_prefix="agent_contacts",
        events=(
            "contact_request_received",
            "contact_request_updated",
            "contact_added",
            "contact_removed",
        ),
    ),
}


class WebSocketClient:
    def __init__(self, ws_url: str, api_key: str, agent_id: str | None = None):
        self.ws_url = ws_url
        self.api_key = api_key
        self.agent_id = agent_id
        self.client: PHXChannelsClient | None = None
        self._validation_error_count: int = 0

    @property
    def validation_error_count(self) -> int:
        """Number of events dropped due to payload validation errors."""
        return self._validation_error_count

    def reset_validation_error_count(self) -> int:
        """Reset the validation error counter and return the previous value.

        Useful for periodic metric flushes (non-atomic, safe for single event loop).
        """
        count = self._validation_error_count
        self._validation_error_count = 0
        return count

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
            self.client = None

    def _require_client(self) -> PHXChannelsClient:
        """Return connected PHX client or fail fast."""
        if self.client is None:
            raise RuntimeError("WebSocket client is not connected")
        return self.client

    def _build_topic(self, channel: str, identifier: str) -> str:
        """Build topic for a declarative channel key."""
        spec = _CHANNEL_SPECS[channel]
        return f"{spec.topic_prefix}:{identifier}"

    def _channel_event_handlers(
        self,
        channel: str,
        handlers: Mapping[str, EventCallback],
    ) -> dict[str, EventCallback]:
        """Return validated handlers in channel-declared event order."""
        spec = _CHANNEL_SPECS[channel]
        missing = [event for event in spec.events if event not in handlers]
        if missing:
            raise ValueError(
                f"Missing handlers for channel '{channel}': {', '.join(missing)}"
            )
        return {event: handlers[event] for event in spec.events}

    async def _subscribe_channel(
        self,
        channel: str,
        identifier: str,
        handlers: Mapping[str, EventCallback],
    ) -> Any:
        """Subscribe to a declared channel and route events through validator."""
        topic = self._build_topic(channel, identifier)
        logger.info("[WebSocket] Subscribing to topic: %s", topic)
        event_handlers = self._channel_event_handlers(channel, handlers)

        async def message_handler(message: PHXMessage) -> None:
            await self._handle_events(message, event_handlers)

        result = await self._require_client().subscribe_to_topic(topic, message_handler)
        logger.info("[WebSocket] Subscribed to topic: %s", topic)
        return result

    async def _unsubscribe_channel(self, channel: str, identifier: str) -> Any:
        """Unsubscribe from a declared channel topic."""
        topic = self._build_topic(channel, identifier)
        logger.info("[WebSocket] Unsubscribing from topic: %s", topic)
        return await self._require_client().unsubscribe_from_topic(topic)

    async def _handle_events(
        self,
        message: PHXMessage,
        event_handlers: Mapping[str, EventCallback],
    ) -> None:
        """Generic async event handler that maps events to their corresponding async callbacks"""
        logger.debug("[WebSocket] Received event: %s", message.event)

        # Check if we have a handler for this event
        if message.event not in event_handlers:
            logger.warning(
                "[WebSocket] Received event '%s' but no handler registered. "
                "Available handlers: %s",
                message.event,
                list(event_handlers.keys()),
            )
            return

        # Validate and parse payload into Pydantic models for known types
        model = _PAYLOAD_MODELS.get(message.event)
        if model is not None:
            try:
                validated = model(**message.payload)
            except ValidationError as e:
                errors = "; ".join(
                    f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}"
                    for err in e.errors()
                )
                logger.error(
                    "[WebSocket] Invalid %s payload: %s",
                    message.event,
                    errors,
                )
                logger.debug(
                    "[WebSocket] Raw payload for invalid %s: %s",
                    message.event,
                    message.payload,
                )
                self._validation_error_count += 1
                return
        else:
            # Unknown event types: pass the raw payload dict
            validated = message.payload

        callback = event_handlers[message.event]
        if callback:
            try:
                await callback(validated)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 – intentionally broad to protect event loop
                logger.exception(
                    "[WebSocket] Callback error for %s event", message.event
                )

    async def join_agent_rooms_channel(
        self,
        agent_id: str,
        on_room_added: Callable[[RoomAddedPayload], Awaitable[None]],
        on_room_removed: Callable[[RoomRemovedPayload], Awaitable[None]],
    ) -> Any:
        """Subscribe to agent rooms topic with async callbacks"""
        return await self._subscribe_channel(
            "agent_rooms",
            agent_id,
            {
                "room_added": on_room_added,
                "room_removed": on_room_removed,
            },
        )

    async def join_chat_room_channel(
        self,
        chat_room_id: str,
        on_message_created: Callable[[MessageCreatedPayload], Awaitable[None]],
    ) -> Any:
        """Subscribe to chat room topic for message events with async callback"""
        return await self._subscribe_channel(
            "chat_room",
            chat_room_id,
            {"message_created": on_message_created},
        )

    async def join_user_rooms_channel(
        self,
        user_id: str,
        on_room_added: Callable[[RoomAddedPayload], Awaitable[None]],
        on_room_removed: Callable[[RoomRemovedPayload], Awaitable[None]],
    ) -> Any:
        """Subscribe to user rooms topic with async callbacks"""
        return await self._subscribe_channel(
            "user_rooms",
            user_id,
            {
                "room_added": on_room_added,
                "room_removed": on_room_removed,
            },
        )

    async def join_room_participants_channel(
        self,
        chat_room_id: str,
        on_participant_added: Callable[[ParticipantAddedPayload], Awaitable[None]],
        on_participant_removed: Callable[[ParticipantRemovedPayload], Awaitable[None]],
    ) -> Any:
        """Subscribe to room participants topic with async callbacks"""
        return await self._subscribe_channel(
            "room_participants",
            chat_room_id,
            {
                "participant_added": on_participant_added,
                "participant_removed": on_participant_removed,
            },
        )

    async def join_tasks_channel(
        self,
        user_id: str,
        on_task_created: Callable[[dict[str, Any]], Awaitable[None]],
        on_task_updated: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> Any:
        """Subscribe to tasks topic with async callbacks"""
        return await self._subscribe_channel(
            "tasks",
            user_id,
            {
                "task_created": on_task_created,
                "task_updated": on_task_updated,
            },
        )

    async def leave_agent_rooms_channel(self, agent_id: str):
        """Unsubscribe from agent rooms topic"""
        return await self._unsubscribe_channel("agent_rooms", agent_id)

    async def leave_chat_room_channel(self, chat_room_id: str):
        """Unsubscribe from chat room topic"""
        return await self._unsubscribe_channel("chat_room", chat_room_id)

    async def leave_user_rooms_channel(self, user_id: str):
        """Unsubscribe from user rooms topic"""
        return await self._unsubscribe_channel("user_rooms", user_id)

    async def leave_room_participants_channel(self, chat_room_id: str):
        """Unsubscribe from room participants topic"""
        return await self._unsubscribe_channel("room_participants", chat_room_id)

    async def leave_tasks_channel(self, user_id: str):
        """Unsubscribe from tasks topic"""
        return await self._unsubscribe_channel("tasks", user_id)

    async def join_agent_contacts_channel(
        self,
        agent_id: str,
        on_contact_request_received: Callable[
            [ContactRequestReceivedPayload], Awaitable[None]
        ],
        on_contact_request_updated: Callable[
            [ContactRequestUpdatedPayload], Awaitable[None]
        ],
        on_contact_added: Callable[[ContactAddedPayload], Awaitable[None]],
        on_contact_removed: Callable[[ContactRemovedPayload], Awaitable[None]],
    ) -> Any:
        """Subscribe to agent contacts topic with async callbacks."""
        return await self._subscribe_channel(
            "agent_contacts",
            agent_id,
            {
                "contact_request_received": on_contact_request_received,
                "contact_request_updated": on_contact_request_updated,
                "contact_added": on_contact_added,
                "contact_removed": on_contact_removed,
            },
        )

    async def leave_agent_contacts_channel(self, agent_id: str):
        """Unsubscribe from agent contacts topic."""
        return await self._unsubscribe_channel("agent_contacts", agent_id)

    async def run_forever(self) -> Any:
        await self.client.run_forever()
