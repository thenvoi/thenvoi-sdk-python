"""Contact event orchestration for runtime contact strategies."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable

from thenvoi.client.rest import (
    DEFAULT_REQUEST_OPTIONS,
    ChatEventRequest,
)
from thenvoi.client.streaming import MessageCreatedPayload, MessageMetadata
from thenvoi.core.nonfatal import NonFatalErrorRecorder
from thenvoi.platform.event import (
    ContactEvent,
    ContactRequestUpdatedEvent,
    MessageEvent,
)
from thenvoi.runtime.types import (
    ContactEventConfig,
    ContactEventStrategy,
    SYNTHETIC_CONTACT_EVENTS_SENDER_ID,
    SYNTHETIC_CONTACT_EVENTS_SENDER_NAME,
    SYNTHETIC_SENDER_TYPE,
)

from .contact_tools import ContactTools
from .broadcast import ContactBroadcaster
from .dedup import ContactDedupCache
from .formatting import (
    HUB_ROOM_SYSTEM_PROMPT,
    format_contact_event_for_room,
    get_contact_event_type,
)
from .hub_room import HubRoomCoordinator
from .request_cache import ContactRequestInfoStore
from .service import ContactService
from .sink import (
    CallbackContactEventSink,
    ContactEventSink,
    HubEventCallback,
    HubInitCallback,
)

if TYPE_CHECKING:
    from thenvoi.platform.link import ThenvoiLink

logger = logging.getLogger(__name__)

MAX_DEDUP_CACHE_SIZE = 1000


@dataclass(frozen=True)
class ContactHandlerDiagnostics:
    """Observable state for contact-handler diagnostics and testing."""

    strategy: str
    dedup_cache_size: int
    hub_room_id: str | None
    broadcast_enabled: bool


class ContactEventHandler(NonFatalErrorRecorder):
    """Coordinate contact events based on configured runtime strategy."""

    def __init__(
        self,
        config: ContactEventConfig,
        link: "ThenvoiLink",
        on_broadcast: Callable[[str], None] | None = None,
        on_hub_event: HubEventCallback | None = None,
        on_hub_init: HubInitCallback | None = None,
        sink: ContactEventSink | None = None,
    ):
        """Create a contact-event handler.

        Prefer explicit ``sink`` injection for runtime integration.
        Callback arguments are retained for compatibility with existing tests/setup.
        """
        self._init_nonfatal_errors()
        self._config = config
        self._link = link
        if sink is not None and (
            on_broadcast is not None
            or on_hub_event is not None
            or on_hub_init is not None
        ):
            raise ValueError(
                "Provide either sink or callback arguments, not both"
            )
        self._sink: ContactEventSink = sink or CallbackContactEventSink(
            on_broadcast=on_broadcast,
            on_hub_event=on_hub_event,
            on_hub_init=on_hub_init,
        )

        self._broadcaster = ContactBroadcaster(self._sink)
        self._contact_service = ContactService(link.rest)
        self._request_info_store = ContactRequestInfoStore(
            self._contact_service,
            max_cache_size=MAX_DEDUP_CACHE_SIZE,
        )
        self._contact_tools: ContactTools | None = None

        self._dedup_cache = ContactDedupCache(max_size=MAX_DEDUP_CACHE_SIZE)
        self._hub_room = HubRoomCoordinator(
            create_room=self._link.rest.agent_api_chats.create_agent_chat,
            task_id=self._config.hub_task_id,
        )

    @property
    def contact_tools(self) -> ContactTools:
        """Get or create ContactTools instance."""
        if self._contact_tools is None:
            self._contact_tools = ContactTools(self._link.rest)
        return self._contact_tools

    @property
    def hub_room_id(self) -> str | None:
        """Get the hub room ID if created."""
        return self._hub_room.room_id

    def dedup_key_for_event(self, event: ContactEvent) -> str | None:
        """Return the deduplication key used for the event."""
        return self._dedup_cache.key_for(event)

    def is_event_processed(self, event: ContactEvent) -> bool:
        """Return whether the event currently exists in the dedup cache."""
        key = self.dedup_key_for_event(event)
        if key is None:
            return False
        return key in self._dedup_cache.storage

    def clear_processed_events(self) -> None:
        """Clear dedup state (for deterministic tests/debugging)."""
        self._dedup_cache.storage.clear()

    async def initialize_hub_room(self) -> str:
        """Create the hub room if needed and return its ID."""
        return await self._hub_room.get_or_create_room_id()

    def mark_hub_room_ready(self) -> None:
        """Mark hub room as ready (ExecutionContext exists)."""
        self._hub_room.mark_ready()

    async def handle(self, event: ContactEvent) -> None:
        """Handle one contact event using configured strategy."""
        logger.debug(
            "ContactEventHandler.handle: %s (strategy=%s)",
            type(event).__name__,
            self._config.strategy.value,
        )

        if self._should_skip_duplicate(event):
            logger.debug(
                "Skipping duplicate contact event: %s", self.dedup_key_for_event(event)
            )
            return

        self._request_info_store.cache_from_event(event)

        if self._config.broadcast_changes:
            await self._maybe_broadcast(event)

        success = True
        match self._config.strategy:
            case ContactEventStrategy.DISABLED:
                logger.debug("Contact event ignored (strategy=DISABLED)")
                return
            case ContactEventStrategy.CALLBACK:
                success = await self._handle_callback(event)
            case ContactEventStrategy.HUB_ROOM:
                success = await self._handle_hub_room(event)

        if success:
            self._dedup_cache.mark_processed(event)

    async def _handle_callback(self, event: ContactEvent) -> bool:
        if self._config.on_event is None:
            logger.warning("CALLBACK strategy but no on_event callback configured")
            return True

        try:
            logger.debug(
                "Calling contact event callback for %s",
                type(event).__name__,
            )
            await self._config.on_event(event, self.contact_tools)
            logger.debug("Contact event callback completed successfully")
            return True
        except Exception as error:
            logger.error("Contact event callback failed: %s", error, exc_info=True)
            return False

    async def _handle_hub_room(self, event: ContactEvent) -> bool:
        if self._hub_room.room_id is None:
            logger.info("Hub room not initialized, initializing now")
            await self.initialize_hub_room()

        if not self._sink.hub_enabled:
            logger.warning("HUB_ROOM strategy but sink does not support hub events")
            return False

        try:
            if not await self._hub_room.wait_ready(timeout_seconds=5.0):
                logger.warning("Hub room not ready after 5s, proceeding anyway (may fail)")

            hub_id = self._hub_room.room_id
            if hub_id is None:
                logger.error("Hub room initialization failed")
                return False

            if self._hub_room.should_initialize_prompt():
                logger.info("Initializing hub room with system prompt")
                await self._sink.initialize_hub_room(hub_id, HUB_ROOM_SYSTEM_PROMPT)

            content = await self._format_event_for_room(event)
            event_type = self._get_event_type(event)

            now = datetime.now(timezone.utc).isoformat()
            message_event = MessageEvent(
                type="message_created",
                room_id=hub_id,
                payload=MessageCreatedPayload(
                    id=str(uuid.uuid4()),
                    content=content,
                    message_type="text",
                    sender_type=SYNTHETIC_SENDER_TYPE,
                    sender_id=SYNTHETIC_CONTACT_EVENTS_SENDER_ID,
                    sender_name=SYNTHETIC_CONTACT_EVENTS_SENDER_NAME,
                    metadata=MessageMetadata(),
                    chat_room_id=hub_id,
                    inserted_at=now,
                    updated_at=now,
                ),
                raw={"contact_event_type": event_type},
            )

            logger.debug("Injecting contact event to hub room %s: %s", hub_id, event_type)
            await self._sink.inject_hub_event(hub_id, message_event)

            await self._post_task_event(hub_id, content, event_type)
            logger.debug("Contact event injected to hub room successfully")
            return True

        except Exception as error:
            logger.error(
                "Failed to inject contact event to hub room: %s",
                error,
                exc_info=True,
            )
            return False

    async def _post_task_event(
        self,
        room_id: str,
        content: str,
        event_type: str,
    ) -> None:
        """Post contact event as persisted task event to room history."""
        try:
            await self._link.rest.agent_api_events.create_agent_chat_event(
                chat_id=room_id,
                event=ChatEventRequest(
                    content=content,
                    message_type="task",
                    metadata={"contact_event_type": event_type},
                ),
                request_options=DEFAULT_REQUEST_OPTIONS,
            )
            logger.debug("Task event posted to hub room: %s", event_type)
        except Exception as error:
            self._record_nonfatal_error(
                "post_task_event",
                error,
                room_id=room_id,
                event_type=event_type,
            )

    async def _format_event_for_room(self, event: ContactEvent) -> str:
        """Format event content for hub-room processing."""
        return await format_contact_event_for_room(
            event,
            enrich_update_event=self._enrich_update_event,
        )

    def _get_event_type(self, event: ContactEvent) -> str:
        """Return metadata event type for a contact event."""
        return get_contact_event_type(event)

    async def _maybe_broadcast(self, event: ContactEvent) -> None:
        """Broadcast contact-added/removed events when enabled."""
        self._broadcaster.maybe_broadcast(event)

    def _should_skip_duplicate(self, event: ContactEvent) -> bool:
        return self._dedup_cache.should_skip(event)

    async def _enrich_update_event(
        self,
        event: ContactRequestUpdatedEvent,
    ) -> dict[str, str | None] | None:
        """Get enriched request metadata for update events."""
        return await self._request_info_store.enrich_update_event(event)

    def diagnostics(self) -> ContactHandlerDiagnostics:
        """Return explicit handler diagnostics for runtime visibility/tests."""
        return ContactHandlerDiagnostics(
            strategy=self._config.strategy.value,
            dedup_cache_size=len(self._dedup_cache.storage),
            hub_room_id=self._hub_room.room_id,
            broadcast_enabled=self._config.broadcast_changes,
        )

    def get_stats(self) -> dict[str, Any]:
        """Get handler statistics."""
        return {
            "strategy": self.diagnostics().strategy,
            "dedup_cache_size": self.diagnostics().dedup_cache_size,
            "hub_room_id": self.diagnostics().hub_room_id,
            "broadcast_enabled": self.diagnostics().broadcast_enabled,
        }
