from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import logging

from phoenix_channels_python_client.client import (
    PHXChannelsClient,
    PhoenixChannelsProtocolVersion,
)
from phoenix_channels_python_client.phx_messages import PHXMessage
from pydantic import BaseModel, ConfigDict, ValidationError
from websockets.exceptions import ConnectionClosed

logger = logging.getLogger(__name__)

# Phoenix protocol events that indicate channel/connection errors
_PHX_CLOSE = "phx_close"
_PHX_ERROR = "phx_error"

# Callback type: (human_reason, raw_payload | None, topic | None) -> None
DisconnectCallback = Callable[[str, dict | None, str | None], Awaitable[None]]

# Map well-known backend reason strings to human-readable messages.
# Unknown reasons fall back to including the raw payload.
KNOWN_DISCONNECT_REASONS: dict[str, str] = {
    "replaced": (
        "Another instance connected with the same agent ID "
        "— only one connection per agent is allowed"
    ),
    "kicked": "Agent was removed by the platform",
    "token_expired": "Authentication token expired or was revoked",
    "invalid_token": "Authentication token is invalid",
    "normal": "Server closed the connection normally",
    "unauthorized": "Not authorized to join this channel",
}


# WebSocket message payloads (based on actual backend messages)
# Using Pydantic for runtime validation


class Mention(BaseModel):
    """Mention object within message metadata."""

    model_config = ConfigDict(extra="allow")

    id: str
    username: str | None = None
    handle: str | None = None
    name: str | None = None


class MessageMetadata(BaseModel):
    """Metadata within message_created payload."""

    model_config = ConfigDict(extra="allow")

    mentions: list[Mention] = []
    status: str | None = None


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


class RoomAddedPayload(BaseModel):
    """Payload for room_added events.

    Required/optional fields aligned with the Fern-generated ChatRoom model
    (thenvoi_rest.types.chat_room.ChatRoom). The WebSocket may include
    additional fields which are captured by ``extra="allow"``.
    """

    model_config = ConfigDict(extra="allow")

    id: str
    inserted_at: str
    updated_at: str
    title: str | None = None
    task_id: str | None = None


class RoomRemovedPayload(BaseModel):
    """Payload for room_removed events.

    WebSocket-only event with no Fern-generated model; all fields except
    ``id`` are kept optional as a defensive default.
    """

    model_config = ConfigDict(extra="allow")

    id: str
    status: str | None = None
    type: str | None = None
    title: str | None = None
    removed_at: str | None = None


class RoomDeletedPayload(BaseModel):
    """Payload for room_deleted events on room_participants channels."""

    model_config = ConfigDict(extra="allow")

    id: str


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
    "room_deleted": RoomDeletedPayload,
    "participant_added": ParticipantAddedPayload,
    "participant_removed": ParticipantRemovedPayload,
    "contact_request_received": ContactRequestReceivedPayload,
    "contact_request_updated": ContactRequestUpdatedPayload,
    "contact_added": ContactAddedPayload,
    "contact_removed": ContactRemovedPayload,
}


def extract_disconnect_reason(payload: dict) -> str:
    """Extract a reason string from a phx_close / phx_error payload.

    Tries common Phoenix payload shapes in order:
      1. ``{"reason": "..."}``
      2. ``{"response": {"reason": "..."}}``
      3. ``{"message": "..."}``
    Falls back to ``"unknown"`` if none match.
    """
    if "reason" in payload:
        return str(payload["reason"])
    response = payload.get("response")
    if isinstance(response, dict) and "reason" in response:
        return str(response["reason"])
    if "message" in payload:
        return str(payload["message"])
    return "unknown"


def humanize_disconnect_reason(reason: str, raw_payload: dict | None = None) -> str:
    """Return a human-readable disconnect message.

    If *reason* matches a well-known key the canned explanation is returned.
    Otherwise the raw reason (and optionally the full payload) is included so
    operators can diagnose unknown scenarios.
    """
    if reason in KNOWN_DISCONNECT_REASONS:
        return KNOWN_DISCONNECT_REASONS[reason]
    if raw_payload:
        return f"Disconnected: {reason} (raw: {raw_payload})"
    return f"Disconnected: {reason}"


def _extract_transport_reason(error: Exception | None) -> str:
    """Best-effort reason from a transport-level disconnect exception.

    The ``websockets`` library raises ``ConnectionClosed`` subclasses that
    carry ``.code`` and ``.reason`` attributes.  We surface those when
    available; otherwise we fall back to ``str(error)``.
    """
    if error is None:
        return "connection_closed"
    if isinstance(error, ConnectionClosed) and error.rcvd is not None:
        if error.rcvd.reason:
            return str(error.rcvd.reason)
        return f"close_code_{error.rcvd.code}"
    return str(error)


class WebSocketClient:
    def __init__(
        self,
        ws_url: str,
        api_key: str,
        agent_id: str | None = None,
        on_disconnect: DisconnectCallback | None = None,
    ):
        self.ws_url = ws_url
        self.api_key = api_key
        self.agent_id = agent_id
        self._on_disconnect = on_disconnect
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
            on_disconnect=self._on_transport_disconnect,
        )
        if self.agent_id:
            self.client.channel_socket_url += f"&agent_id={self.agent_id}"
        await self.client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the PHXChannelsClient context"""
        if self.client:
            await self.client.__aexit__(exc_type, exc_val, exc_tb)

    async def _on_transport_disconnect(self, error: Exception | None) -> None:
        """Handle transport-level disconnects from PHXChannelsClient."""
        reason = _extract_transport_reason(error)
        human = humanize_disconnect_reason(reason)
        logger.warning("[WebSocket] Transport disconnected: %s", human)
        if self._on_disconnect:
            raw = {"error": str(error)} if error else None
            try:
                await self._on_disconnect(human, raw, None)
            except Exception:  # noqa: BLE001
                logger.exception("[WebSocket] Error in disconnect callback")

    async def _handle_events(self, message: PHXMessage, event_handlers: dict):
        """Generic async event handler that maps events to their corresponding async callbacks"""
        logger.debug("[WebSocket] Received event: %s", message.event)

        # Intercept Phoenix close / error events before normal dispatch.
        # These indicate the server is closing the channel (e.g. duplicate
        # agent connection) and must be surfaced to the caller.
        if message.event in (_PHX_CLOSE, _PHX_ERROR):
            reason = extract_disconnect_reason(message.payload)
            human = humanize_disconnect_reason(reason, message.payload)
            logger.warning(
                "[WebSocket] Channel %s on topic %s: %s",
                message.event,
                message.topic,
                human,
            )
            if self._on_disconnect:
                try:
                    await self._on_disconnect(human, message.payload, message.topic)
                except Exception:  # noqa: BLE001
                    logger.exception("[WebSocket] Error in disconnect callback")
            return

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
    ):
        """Subscribe to agent rooms topic with async callbacks"""
        topic = f"agent_rooms:{agent_id}"
        logger.info("[WebSocket] Subscribing to topic: %s", topic)

        async def message_handler(message):
            await self._handle_events(
                message, {"room_added": on_room_added, "room_removed": on_room_removed}
            )

        result = await self.client.subscribe_to_topic(topic, message_handler)
        logger.info("[WebSocket] Subscribed to topic: %s", topic)
        return result

    async def join_chat_room_channel(
        self,
        chat_room_id: str,
        on_message_created: Callable[[MessageCreatedPayload], Awaitable[None]],
    ):
        """Subscribe to chat room topic for message events with async callback"""
        topic = f"chat_room:{chat_room_id}"
        logger.info("[WebSocket] Subscribing to topic: %s", topic)

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
        on_participant_added: Callable[[ParticipantAddedPayload], Awaitable[None]],
        on_participant_removed: Callable[[ParticipantRemovedPayload], Awaitable[None]],
        on_room_deleted: Callable[[RoomDeletedPayload], Awaitable[None]],
    ):
        """Subscribe to room participants topic with async callbacks"""
        topic = f"room_participants:{chat_room_id}"
        logger.info("[WebSocket] Subscribing to topic: %s", topic)

        async def message_handler(message):
            await self._handle_events(
                message,
                {
                    "participant_added": on_participant_added,
                    "participant_removed": on_participant_removed,
                    "room_deleted": on_room_deleted,
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
        logger.info("[WebSocket] Unsubscribing from topic: %s", topic)
        return await self.client.unsubscribe_from_topic(topic)

    async def leave_chat_room_channel(self, chat_room_id: str):
        """Unsubscribe from chat room topic"""
        topic = f"chat_room:{chat_room_id}"
        logger.info("[WebSocket] Unsubscribing from topic: %s", topic)
        return await self.client.unsubscribe_from_topic(topic)

    async def leave_user_rooms_channel(self, user_id: str):
        """Unsubscribe from user rooms topic"""
        topic = f"user_rooms:{user_id}"
        return await self.client.unsubscribe_from_topic(topic)

    async def leave_room_participants_channel(self, chat_room_id: str):
        """Unsubscribe from room participants topic"""
        topic = f"room_participants:{chat_room_id}"
        logger.info("[WebSocket] Unsubscribing from topic: %s", topic)
        return await self.client.unsubscribe_from_topic(topic)

    async def leave_tasks_channel(self, user_id: str):
        """Unsubscribe from tasks topic"""
        topic = f"tasks:{user_id}"
        return await self.client.unsubscribe_from_topic(topic)

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
    ):
        """Subscribe to agent contacts topic with async callbacks."""
        topic = f"agent_contacts:{agent_id}"
        logger.info("[WebSocket] Subscribing to topic: %s", topic)

        async def message_handler(message):
            await self._handle_events(
                message,
                {
                    "contact_request_received": on_contact_request_received,
                    "contact_request_updated": on_contact_request_updated,
                    "contact_added": on_contact_added,
                    "contact_removed": on_contact_removed,
                },
            )

        result = await self.client.subscribe_to_topic(topic, message_handler)
        logger.info("[WebSocket] Subscribed to topic: %s", topic)
        return result

    async def leave_agent_contacts_channel(self, agent_id: str):
        """Unsubscribe from agent contacts topic."""
        topic = f"agent_contacts:{agent_id}"
        logger.info("[WebSocket] Unsubscribing from topic: %s", topic)
        return await self.client.unsubscribe_from_topic(topic)

    async def run_forever(self):
        await self.client.run_forever()
