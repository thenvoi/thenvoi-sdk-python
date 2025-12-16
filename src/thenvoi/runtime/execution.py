"""
Execution - Per-room execution interface and default implementation.

Extracted from AgentSession with simplified interface.

Crash Recovery:
    When an agent restarts, it may have missed messages while down.
    The sync mechanism handles this:
    1. First WebSocket message marks the sync point (_first_ws_msg_id)
    2. Before processing WS queue, _synchronize_with_next() polls REST API
    3. Process backlog messages until we reach the sync point
    4. Clear marker and continue with WebSocket queue
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from datetime import datetime, timezone
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Literal,
    Protocol,
    runtime_checkable,
)

from thenvoi.platform.event import PlatformEvent

from .types import ConversationContext, PlatformMessage, SessionConfig
from .retry_tracker import MessageRetryTracker

if TYPE_CHECKING:
    from thenvoi.platform.link import ThenvoiLink

logger = logging.getLogger(__name__)


@runtime_checkable
class Execution(Protocol):
    """
    Interface for per-room execution. Pluggable.

    Implementations handle what happens INSIDE a room.
    The default ExecutionContext uses context accumulation.
    Custom implementations (e.g., Letta) can use persistent agents.
    """

    room_id: str

    async def start(self) -> None:
        """Start the execution context."""
        ...

    async def stop(self) -> None:
        """Stop the execution context."""
        ...

    async def on_event(self, event: PlatformEvent) -> None:
        """Handle a platform event for this room."""
        ...


# Type for execution callback
ExecutionHandler = Callable[["ExecutionContext", PlatformEvent], Awaitable[None]]


class ExecutionContext:
    """
    Default execution: context accumulation model.

    Extracted from AgentSession.

    - Accumulates inputs (history, participants)
    - Queues messages
    - Feeds agent when instantiated
    - Agent disappears after execution

    Example:
        async def on_execute(ctx: ExecutionContext, event: PlatformEvent):
            if event.is_message:
                tools = AgentTools.from_context(ctx)
                history = ctx.get_history_for_llm()
                # Run LLM with context and tools...

        ctx = ExecutionContext(room_id, link, on_execute)
        await ctx.start()
    """

    def __init__(
        self,
        room_id: str,
        link: "ThenvoiLink",
        on_execute: ExecutionHandler,
        config: SessionConfig | None = None,
        agent_id: str | None = None,
    ):
        """
        Initialize execution context for a specific room.

        Args:
            room_id: The room this context manages
            link: ThenvoiLink for REST API calls
            on_execute: Callback for handling events
            config: Optional session configuration
            agent_id: Agent ID for filtering self-messages
        """
        self.room_id = room_id
        self.link = link
        self._on_execute = on_execute
        self.config = config or SessionConfig()
        self._agent_id = agent_id

        # Per-room state
        self.queue: asyncio.Queue[PlatformEvent] = asyncio.Queue()
        self.state: Literal["starting", "idle", "processing"] = "starting"
        self._is_running = False
        self._process_loop_task: asyncio.Task[None] | None = None
        self._context_cache: ConversationContext | None = None
        self._context_hydrated = False

        # Participant tracking (simplified from ParticipantTracker)
        self._participants: list[dict[str, Any]] = []
        self._participants_loaded = False
        self._last_participants_sent: list[dict[str, Any]] | None = None

        # LLM context tracking
        self._llm_initialized = False

        # Dedupe cache (LRU for detecting duplicates during sync)
        self._processed_ids: OrderedDict[str, bool] = OrderedDict()
        self._max_processed_ids: int = 5

        # Crash recovery: sync point marker and retry tracking
        self._first_ws_msg_id: str | None = None  # First WS message = sync point
        self._retry_tracker = MessageRetryTracker(
            max_retries=self.config.max_message_retries,
            room_id=room_id,
        )
        self._sync_complete = False  # True after sync with /next completes

    @property
    def thread_id(self) -> str:
        """LangGraph thread_id = room_id."""
        return self.room_id

    @property
    def is_processing(self) -> bool:
        """Check if context is currently processing an event."""
        return self.state == "processing"

    @property
    def is_running(self) -> bool:
        """Check if context is running (task exists and not done)."""
        return (
            self._process_loop_task is not None and not self._process_loop_task.done()
        )

    @property
    def participants(self) -> list[dict[str, Any]]:
        """Get current participants list (copy)."""
        return self._participants.copy()

    @property
    def is_llm_initialized(self) -> bool:
        """Check if LLM has been initialized with system prompt."""
        return self._llm_initialized

    def mark_llm_initialized(self) -> None:
        """Mark that system prompt has been sent to LLM."""
        self._llm_initialized = True
        logger.debug(f"ExecutionContext {self.room_id}: LLM initialized")

    # --- Execution protocol implementation ---

    async def start(self) -> None:
        """
        Start background processing for this room.

        Creates an asyncio task that processes events from the queue.
        """
        if self._is_running:
            logger.warning(f"ExecutionContext {self.room_id} already running")
            return

        logger.info(f"Starting ExecutionContext for room: {self.room_id}")
        self._is_running = True
        self._process_loop_task = asyncio.create_task(
            self._process_loop(),
            name=f"execution-{self.room_id}",
        )

    async def stop(self) -> None:
        """
        Stop processing instantly via asyncio cancellation.

        Uses task.cancel() which interrupts queue.get() immediately.
        """
        if self._process_loop_task is None:
            return

        logger.info(f"Stopping ExecutionContext for room: {self.room_id}")

        self._process_loop_task.cancel()
        try:
            await self._process_loop_task
        except asyncio.CancelledError:
            pass
        self._process_loop_task = None
        self._is_running = False

    async def on_event(self, event: PlatformEvent) -> None:
        """
        Handle a platform event - add to queue.

        Called by RoomPresence/AgentRuntime when an event arrives.
        Tracks first WebSocket message ID for crash recovery sync.
        """
        # Track first WebSocket message ID for sync point
        if event.is_message and self._first_ws_msg_id is None:
            msg_id = event.payload.get("id")
            if msg_id:
                self._first_ws_msg_id = msg_id
                logger.debug(f"Sync point marker set: {msg_id}")

        self.queue.put_nowait(event)
        logger.debug(f"Event {event.type} enqueued for room {self.room_id}")

    # --- Participant management ---

    def add_participant(self, participant: dict) -> bool:
        """
        Add participant (from WebSocket event).

        Returns:
            True if added, False if duplicate
        """
        if any(p.get("id") == participant.get("id") for p in self._participants):
            return False

        self._participants.append(
            {
                "id": participant.get("id"),
                "name": participant.get("name"),
                "type": participant.get("type"),
            }
        )
        logger.debug(
            f"ExecutionContext {self.room_id}: Added participant {participant.get('name')}"
        )
        return True

    def remove_participant(self, participant_id: str) -> bool:
        """
        Remove participant (from WebSocket event).

        Returns:
            True if removed, False if not found
        """
        before = len(self._participants)
        self._participants = [
            p for p in self._participants if p.get("id") != participant_id
        ]
        return len(self._participants) < before

    def participants_changed(self) -> bool:
        """Check if participants changed since last mark_participants_sent()."""
        if self._last_participants_sent is None:
            return True

        last_ids = {p.get("id") for p in self._last_participants_sent}
        current_ids = {p.get("id") for p in self._participants}
        return last_ids != current_ids

    def mark_participants_sent(self) -> None:
        """Mark current participants as sent to LLM."""
        self._last_participants_sent = self._participants.copy()

    async def load_participants(self) -> list[dict[str, Any]]:
        """Load participants from API."""
        if self._participants_loaded:
            return self._participants

        try:
            response = await self.link.rest.agent_api.list_agent_chat_participants(
                chat_id=self.room_id,
            )
            if response.data:
                self._participants = [
                    {"id": p.id, "name": p.name, "type": p.type} for p in response.data
                ]
            self._participants_loaded = True
        except Exception as e:
            logger.warning(f"Failed to load participants for room {self.room_id}: {e}")
            self._participants_loaded = True

        return self._participants

    # --- Context building ---

    async def hydrate(self) -> None:
        """
        Hydrate conversation context for this room.

        Called lazily on first event to load conversation history
        and participant list.

        If enable_context_hydration is False, skips API call and returns
        empty context (useful for agents that manage their own state like Letta).
        """
        if self._context_hydrated:
            return

        # Skip hydration if disabled
        if not self.config.enable_context_hydration:
            logger.debug(f"Context hydration disabled for room: {self.room_id}")
            self._context_cache = ConversationContext(
                room_id=self.room_id,
                messages=[],
                participants=[],
                hydrated_at=datetime.now(timezone.utc),
            )
            self._context_hydrated = True
            return

        logger.debug(f"Hydrating context for room: {self.room_id}")

        try:
            # Load participants first
            await self.load_participants()

            # Load context from API
            context_response = await self.link.rest.agent_api.get_agent_chat_context(
                chat_id=self.room_id,
            )

            messages = []
            if context_response.data:
                for item in context_response.data:
                    sender_name = getattr(item, "sender_name", None) or getattr(
                        item, "name", None
                    )
                    messages.append(
                        {
                            "id": item.id,
                            "content": getattr(item, "content", ""),
                            "sender_id": getattr(item, "sender_id", ""),
                            "sender_type": getattr(item, "sender_type", ""),
                            "sender_name": sender_name,
                            "message_type": getattr(item, "message_type", "text"),
                            "created_at": getattr(item, "inserted_at", None),
                        }
                    )

            self._context_cache = ConversationContext(
                room_id=self.room_id,
                messages=messages,
                participants=self._participants,
                hydrated_at=datetime.now(timezone.utc),
            )
            self._context_hydrated = True

            logger.debug(
                f"Context hydrated: {len(messages)} messages, "
                f"{len(self._participants)} participants"
            )

        except Exception as e:
            logger.warning(f"Context hydration failed: {e}")
            self._context_cache = ConversationContext(
                room_id=self.room_id,
                messages=[],
                participants=[],
                hydrated_at=datetime.now(timezone.utc),
            )
            self._context_hydrated = True

    def build_context(self) -> ConversationContext:
        """
        Build context dict for LLM.

        Returns cached context or empty context if not hydrated.
        """
        if self._context_cache:
            return self._context_cache

        return ConversationContext(
            room_id=self.room_id,
            messages=[],
            participants=self._participants,
            hydrated_at=datetime.now(timezone.utc),
        )

    async def get_context(self, force_refresh: bool = False) -> ConversationContext:
        """
        Get conversation context (lazy, cached).

        Args:
            force_refresh: Force refresh from API even if cached
        """
        if force_refresh or not self._context_hydrated:
            self._context_hydrated = False
            await self.hydrate()

        return self.build_context()

    def get_history_for_llm(
        self, exclude_message_id: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Get conversation history formatted for LLM injection.

        Returns list of dicts with:
        - role: "assistant" or "user"
        - content: message content
        - sender_name: original sender name

        Args:
            exclude_message_id: Message ID to exclude (usually current message)

        Returns:
            List of message dicts ready for LLM formatting.
        """
        if not self.config.enable_context_hydration:
            return []

        if not self._context_cache:
            return []

        # Import here to avoid circular dependency
        from thenvoi.runtime.formatters import format_history_for_llm

        return format_history_for_llm(
            self._context_cache.messages,
            exclude_id=exclude_message_id,
        )

    def build_participants_message(self) -> str:
        """Build a system message with current participant list for LLM."""
        from thenvoi.runtime.formatters import build_participants_message

        return build_participants_message(self._participants)

    # --- Internal processing ---

    async def _process_loop(self) -> None:
        """
        Main processing loop for this room.

        SYNCHRONIZATION FLOW:
        1. Call /next to get unprocessed messages from backend
        2. For each /next message, check if it matches WebSocket queue head
        3. If match → synchronized! Process once, then switch to WebSocket only
        4. If no match → process /next message, repeat
        5. After sync, process only from WebSocket queue

        Uses asyncio cancellation for shutdown.
        """
        try:
            # Phase 1: Sync via /next until we catch up with WebSocket
            await self._synchronize_with_next()
            self.state = "idle"
            logger.info(
                f"ExecutionContext {self.room_id}: Synchronized, switching to WebSocket"
            )

            # Phase 2: Process from WebSocket queue only
            while True:
                event = await self.queue.get()
                await self._process_event(event)

        except asyncio.CancelledError:
            logger.debug(f"ExecutionContext {self.room_id} cancelled")
        except Exception as e:
            logger.error(f"ExecutionContext {self.room_id} error: {e}", exc_info=True)

        logger.debug(f"ExecutionContext {self.room_id} loop exited")

    async def _synchronize_with_next(self) -> None:
        """
        Synchronize backlog via /next API until caught up with WebSocket.

        Uses _first_ws_msg_id marker:
        1. Call /next to get next unprocessed message
        2. If None → no backlog, we're synced
        3. Check if message ID matches _first_ws_msg_id (first WebSocket message)
        4. If match → synced! Process this message, pop duplicate from queue
        5. If no match → process /next message, repeat from step 1
        """
        logger.debug(f"ExecutionContext {self.room_id}: Starting /next synchronization")

        try:
            while True:  # Cancellation handles exit
                next_msg = await self._get_next_message()

                if next_msg is None:
                    logger.debug(
                        f"ExecutionContext {self.room_id}: /next returned None, synced"
                    )
                    break

                if self._retry_tracker.is_permanently_failed(next_msg.id):
                    logger.warning(
                        f"ExecutionContext {self.room_id}: Skipping permanently failed message {next_msg.id}"
                    )
                    break

                if next_msg.id == self._first_ws_msg_id:
                    logger.info(
                        f"ExecutionContext {self.room_id}: Sync point reached at message {next_msg.id}"
                    )
                    await self._process_backlog_message(next_msg)

                    # Pop duplicate from queue
                    try:
                        head = self.queue.get_nowait()
                        if head.payload.get("id") != next_msg.id:
                            # Put it back if it's not the duplicate
                            self.queue.put_nowait(head)
                    except asyncio.QueueEmpty:
                        pass

                    self._first_ws_msg_id = None  # Clear marker
                    self._processed_ids.clear()  # No longer needed after sync
                    break

                logger.debug(
                    f"ExecutionContext {self.room_id}: Processing backlog message {next_msg.id}"
                )
                await self._process_backlog_message(next_msg)

                if self._retry_tracker.is_permanently_failed(next_msg.id):
                    logger.warning(
                        f"ExecutionContext {self.room_id}: Message {next_msg.id} permanently failed"
                    )
                    break

        except Exception as e:
            logger.error(
                f"ExecutionContext {self.room_id}: Sync error: {e}", exc_info=True
            )

        logger.debug(f"ExecutionContext {self.room_id}: Synchronization complete")
        self._sync_complete = True

    async def _get_next_message(self) -> PlatformMessage | None:
        """
        Get next unprocessed message from REST API.

        Returns None if no more messages in backlog (204 No Content).
        Delegates to ThenvoiLink.get_next_message().
        """
        return await self.link.get_next_message(self.room_id)

    async def _process_backlog_message(self, msg: PlatformMessage) -> None:
        """
        Process a backlog message from /next during sync.

        Full lifecycle:
        1. Check if permanently failed or duplicate
        2. Record attempt with retry tracker
        3. Mark as processing on server
        4. Execute handler
        5. Mark as processed (success) or failed (exception)
        """
        msg_id = msg.id

        # Skip messages from self (agent's own messages) to avoid infinite loops
        if (
            self._agent_id
            and msg.sender_type == "Agent"
            and msg.sender_id == self._agent_id
        ):
            logger.debug(f"Skipping self-message {msg_id}")
            return

        # Skip permanently failed messages
        if self._retry_tracker.is_permanently_failed(msg_id):
            logger.debug(f"Skipping permanently failed message {msg_id}")
            return

        # Skip if already processed (dedupe)
        if msg_id in self._processed_ids:
            self._processed_ids.move_to_end(msg_id)
            logger.debug(f"Skipping duplicate backlog message: {msg_id}")
            return

        # Track attempts - check if exceeded BEFORE processing
        attempts, exceeded = self._retry_tracker.record_attempt(msg_id)
        if exceeded:
            logger.warning(
                f"Message {msg_id} exceeded max retries ({attempts} attempts)"
            )
            return

        self.state = "processing"
        logger.info(f"Processing backlog message {msg_id} in room {self.room_id}")

        try:
            # Mark as processing on server BEFORE we start
            await self.link.mark_processing(self.room_id, msg_id)

            # Hydrate context on first message if enabled
            if not self._context_hydrated and self.config.enable_context_hydration:
                await self.hydrate()

            # Format timestamps for MessageCreatedPayload validation
            created_at_str = (
                msg.created_at.isoformat()
                if msg.created_at
                else datetime.now(timezone.utc).isoformat()
            )

            # Normalize metadata.mentions to include username field
            metadata = msg.metadata.copy() if msg.metadata else {}
            if "mentions" in metadata:
                normalized_mentions = []
                for mention in metadata.get("mentions", []):
                    if isinstance(mention, dict):
                        normalized_mentions.append(
                            {
                                "id": mention.get("id", ""),
                                "username": mention.get("username")
                                or mention.get("name")
                                or mention.get("id", ""),
                            }
                        )
                metadata["mentions"] = normalized_mentions
            else:
                metadata["mentions"] = []

            if "status" not in metadata:
                metadata["status"] = "sent"

            # Create event from message for handler
            event = PlatformEvent(
                type="message_created",
                room_id=self.room_id,
                payload={
                    "id": msg.id,
                    "content": msg.content,
                    "sender_id": msg.sender_id,
                    "sender_type": msg.sender_type,
                    "sender_name": msg.sender_name,
                    "message_type": msg.message_type,
                    "metadata": metadata,
                    "chat_room_id": self.room_id,
                    "inserted_at": created_at_str,
                    "updated_at": created_at_str,
                },
            )

            # Call execution handler
            await self._on_execute(self, event)

            # SUCCESS: Mark as processed on server
            await self.link.mark_processed(self.room_id, msg_id)
            self._retry_tracker.mark_success(msg_id)

            # Track in dedupe cache
            self._processed_ids[msg_id] = True
            if len(self._processed_ids) > self._max_processed_ids:
                self._processed_ids.popitem(last=False)

            logger.debug(f"Message {msg_id} processed successfully")

        except Exception as e:
            # FAILURE: Mark as failed on server
            logger.error(
                f"Error processing backlog message {msg_id}: {e}", exc_info=True
            )
            await self.link.mark_failed(self.room_id, msg_id, str(e))

        finally:
            self.state = "idle"

    def _drain_duplicate_from_queue(self, msg_id: str) -> None:
        """
        Remove duplicate message from WebSocket queue after sync point reached.

        The message at sync point exists in both /next and WS queue.
        """
        # Drain queue and re-add non-duplicates
        items = []
        while not self.queue.empty():
            try:
                event = self.queue.get_nowait()
                if event.is_message and event.payload.get("id") == msg_id:
                    logger.debug(f"Removed duplicate from WS queue: {msg_id}")
                    continue
                items.append(event)
            except asyncio.QueueEmpty:
                break

        # Re-add non-duplicates
        for item in items:
            self.queue.put_nowait(item)

    async def _process_event(self, event: PlatformEvent) -> None:
        """
        Process single event through execution callback.

        For message events, handles full lifecycle:
        1. Check if permanently failed or duplicate
        2. Record attempt with retry tracker
        3. Mark as processing on server
        4. Execute handler
        5. Mark as processed (success) or failed (exception)
        """
        msg_id = event.payload.get("id") if event.is_message else None

        # For messages: check if we should skip
        if event.is_message and msg_id:
            # Skip messages from self (agent's own messages) to avoid infinite loops
            sender_id = event.payload.get("sender_id")
            sender_type = event.payload.get("sender_type")
            if (
                self._agent_id
                and sender_type == "Agent"
                and sender_id == self._agent_id
            ):
                logger.debug(f"Skipping self-message {msg_id}")
                return

            # Skip permanently failed messages
            if self._retry_tracker.is_permanently_failed(msg_id):
                logger.debug(f"Skipping permanently failed message {msg_id}")
                return

            # Skip duplicates
            if msg_id in self._processed_ids:
                self._processed_ids.move_to_end(msg_id)
                logger.debug(f"Skipping duplicate message {msg_id}")
                return

            # Track attempts
            attempts, exceeded = self._retry_tracker.record_attempt(msg_id)
            if exceeded:
                logger.warning(
                    f"Message {msg_id} exceeded max retries ({attempts} attempts)"
                )
                return

        self.state = "processing"
        logger.debug(f"Processing {event.type} in room {self.room_id}")

        try:
            # For messages: mark as processing on server
            if event.is_message and msg_id:
                await self.link.mark_processing(self.room_id, msg_id)

            # Hydrate context on first event if enabled
            if not self._context_hydrated and self.config.enable_context_hydration:
                await self.hydrate()

            # Handle participant events internally
            if event.is_participant_added:
                self.add_participant(event.payload)
            elif event.is_participant_removed:
                self.remove_participant(event.payload.get("id", ""))

            # Call execution handler
            await self._on_execute(self, event)

            # For messages: mark as processed on server
            if event.is_message and msg_id:
                await self.link.mark_processed(self.room_id, msg_id)
                self._retry_tracker.mark_success(msg_id)

                # Track in dedupe cache
                self._processed_ids[msg_id] = True
                if len(self._processed_ids) > self._max_processed_ids:
                    self._processed_ids.popitem(last=False)

            logger.debug(f"Event {event.type} processed successfully")

        except Exception as e:
            logger.error(f"Error processing {event.type}: {e}", exc_info=True)
            # For messages: mark as failed on server
            if event.is_message and msg_id:
                await self.link.mark_failed(self.room_id, msg_id, str(e))

        finally:
            self.state = "idle"
