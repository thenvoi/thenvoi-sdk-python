"""
Per-room session management for Thenvoi agents.

Each AgentSession handles message processing for a single room,
with its own state machine, message queue, and background task.

KEY DESIGN:
    - One AgentSession per room (room_id = thread_id for LangGraph)
    - Independent processing - rooms don't block each other
    - Session creates AgentTools for each message handler call
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal

from .types import (
    ConversationContext,
    MessageHandler,
    PlatformMessage,
    SessionConfig,
)
from .formatters import format_history_for_llm, build_participants_message
from .participant_tracker import ParticipantTracker
from .retry_tracker import MessageRetryTracker

if TYPE_CHECKING:
    from .agent import ThenvoiAgent
    from thenvoi.client.rest import AsyncRestClient

logger = logging.getLogger(__name__)


class AgentSession:
    """
    Per-room message processing with independent state machine.

    Each room gets its own session with:
    - Own message queue
    - Own state machine (starting → idle ↔ processing)
    - Own background processing task
    - Own context cache

    Example:
        session = AgentSession(
            room_id="room_123",
            api_client=api_client,
            on_message=my_handler,
            coordinator=thenvoi_agent,
        )
        await session.start()
        # ... session processes messages from its queue
        await session.stop()
    """

    def __init__(
        self,
        room_id: str,
        api_client: "AsyncRestClient",
        on_message: MessageHandler,
        coordinator: "ThenvoiAgent",
        config: SessionConfig | None = None,
    ):
        """
        Initialize session for a specific room.

        Args:
            room_id: The room this session manages
            api_client: REST API client for platform operations
            on_message: Async callback to handle messages
            coordinator: Parent ThenvoiAgent for internal operations
            config: Optional session configuration
        """
        self.room_id = room_id
        self._api_client = api_client
        self._on_message = on_message
        self._coordinator = coordinator
        self.config = config or SessionConfig()

        # Per-room state
        self.queue: asyncio.Queue[PlatformMessage] = asyncio.Queue()
        self.state: Literal["starting", "idle", "processing"] = "starting"
        self._is_running = False
        self._process_loop_task: asyncio.Task[None] | None = None
        self._context_cache: ConversationContext | None = None
        self._context_hydrated = False
        self._synchronized = False

        # Extracted trackers (sync, unit-testable)
        self._participant_tracker = ParticipantTracker(room_id=room_id)
        self._retry_tracker = MessageRetryTracker(
            max_retries=self.config.max_message_retries,
            room_id=room_id,
        )

        # LLM context tracking (for adapters)
        self._llm_initialized = False  # Has system prompt been sent?

        # Sync point tracking (replaces CPython queue internals peeking)
        self._first_ws_msg_id: str | None = None  # ID of first WebSocket message
        self._processed_ids: OrderedDict[str, bool] = OrderedDict()  # LRU dedupe cache
        self._max_processed_ids: int = 5  # Max entries in dedupe cache

    @property
    def thread_id(self) -> str:
        """
        LangGraph thread_id = room_id.

        This property makes it clear that room_id maps to LangGraph's
        thread concept for checkpointing.
        """
        return self.room_id

    @property
    def is_processing(self) -> bool:
        """Check if session is currently processing a message."""
        return self.state == "processing"

    @property
    def is_running(self) -> bool:
        """Check if session is running (task exists and not done)."""
        return (
            self._process_loop_task is not None and not self._process_loop_task.done()
        )

    @property
    def participants(self) -> list[dict[str, Any]]:
        """Get current participants list."""
        return self._participant_tracker.participants

    @property
    def is_llm_initialized(self) -> bool:
        """Check if LLM has been initialized with system prompt."""
        return self._llm_initialized

    def mark_llm_initialized(self) -> None:
        """Mark that system prompt has been sent to LLM."""
        self._llm_initialized = True
        logger.debug(f"Session {self.room_id}: LLM initialized")

    def participants_changed(self) -> bool:
        """Check if participants changed since last sent to LLM."""
        return self._participant_tracker.changed()

    def build_participants_message(self) -> str:
        """Build a system message with current participant list for LLM."""
        return build_participants_message(self._participant_tracker.participants)

    def mark_participants_sent(self) -> None:
        """Mark current participants as sent to LLM."""
        self._participant_tracker.mark_sent()

    async def get_history_for_llm(
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
            Returns empty list if enable_context_hydration is False.
        """
        # If hydration disabled, return empty (framework manages its own history)
        if not self.config.enable_context_hydration:
            logger.debug(
                f"Session {self.room_id}: Context hydration disabled, returning empty history"
            )
            return []

        # Ensure context is hydrated
        if not self._context_hydrated:
            await self._hydrate_context()
            self._context_hydrated = True

        if not self._context_cache:
            logger.debug(
                f"Session {self.room_id}: No context cache, returning empty history"
            )
            return []

        history = format_history_for_llm(
            self._context_cache.messages,
            exclude_id=exclude_message_id,
        )
        logger.info(
            f"Session {self.room_id}: Loaded {len(history)} historical messages"
        )
        return history

    def add_participant(self, participant: dict) -> None:
        """Add a participant (called from WebSocket event)."""
        self._participant_tracker.add(participant)

    def remove_participant(self, participant: dict) -> None:
        """Remove a participant (called from WebSocket event)."""
        self._participant_tracker.remove(participant.get("id", ""))

    async def load_participants(self) -> list[dict[str, Any]]:
        """Load participants from API (called during hydration)."""
        if self._participant_tracker.is_loaded:
            return self._participant_tracker.participants

        try:
            participants = await self._coordinator._get_participants_internal(
                self.room_id
            )
            self._participant_tracker.set_loaded(participants)
        except Exception as e:
            logger.warning(f"Failed to load participants for room {self.room_id}: {e}")
            self._participant_tracker.set_loaded([])

        return self._participant_tracker.participants

    async def start(self) -> None:
        """
        Start background processing for this room.

        Creates an asyncio task that:
        1. Synchronizes backlog via /next
        2. Processes messages from queue
        """
        if self._is_running:
            logger.warning(f"Session {self.room_id} already running")
            return

        logger.info(f"Starting session for room: {self.room_id}")
        self._is_running = True
        self._process_loop_task = asyncio.create_task(
            self._process_loop(),
            name=f"session-{self.room_id}",
        )

    async def stop(self) -> None:
        """
        Stop processing instantly via asyncio cancellation.

        Uses task.cancel() which interrupts queue.get() immediately,
        no timeout waiting required.
        """
        if self._process_loop_task is None:
            return

        logger.info(f"Stopping session for room: {self.room_id}")

        self._process_loop_task.cancel()  # Instant interrupt
        try:
            await self._process_loop_task
        except asyncio.CancelledError:
            pass
        self._process_loop_task = None
        self._is_running = False  # Keep for state queries

    async def _process_loop(self) -> None:
        """
        Main processing loop for this room.

        SYNCHRONIZATION FLOW:
        1. Call /next to get unprocessed messages from backend
        2. For each /next message, check if it matches WebSocket queue head
        3. If match → synchronized! Process once, then switch to WebSocket only
        4. If no match → process /next message, repeat
        5. After sync, process only from WebSocket queue

        Uses asyncio cancellation for shutdown (not threading-style flag polling).
        """
        try:
            # Phase 1: Sync via /next until we catch up with WebSocket
            await self._synchronize_with_next()
            self._synchronized = True
            self.state = "idle"
            logger.info(f"Session {self.room_id}: Synchronized, switching to WebSocket")

            # Phase 2: Process from WebSocket queue only
            # No timeout needed - task.cancel() interrupts queue.get() instantly
            while True:
                msg = await self.queue.get()
                await self._process_message(msg)

        except asyncio.CancelledError:
            logger.debug(f"Session {self.room_id} cancelled")
            # Clean exit - no re-raise needed
        except Exception as e:
            logger.error(f"Session {self.room_id} error: {e}", exc_info=True)

        logger.debug(f"Session {self.room_id} loop exited")

    async def _synchronize_with_next(self) -> None:
        """
        Synchronize backlog via /next API until caught up with WebSocket.

        Uses _first_ws_msg_id marker instead of peeking queue internals:
        1. Call /next to get next unprocessed message
        2. If None → no backlog, we're synced
        3. Check if message ID matches _first_ws_msg_id (first WebSocket message)
        4. If match → synced! Process this message, pop duplicate from queue
        5. If no match → process /next message, repeat from step 1
        """
        logger.debug(f"Session {self.room_id}: Starting /next synchronization")

        try:
            while True:  # Cancellation handles exit
                next_msg = await self._coordinator._get_next_message(self.room_id)

                if next_msg is None:
                    logger.debug(f"Session {self.room_id}: /next returned None, synced")
                    break

                if self._retry_tracker.is_permanently_failed(next_msg.id):
                    logger.warning(
                        f"Session {self.room_id}: Skipping permanently failed message {next_msg.id}"
                    )
                    break

                if next_msg.id == self._first_ws_msg_id:
                    logger.info(
                        f"Session {self.room_id}: Sync point reached at message {next_msg.id}"
                    )
                    await self._process_message(next_msg)

                    # Pop duplicate from queue (debug-only sanity check)
                    head = self.queue.get_nowait()
                    if __debug__:
                        assert head.id == next_msg.id, (
                            f"Queue head mismatch: {head.id} != {next_msg.id}"
                        )

                    self._first_ws_msg_id = None  # Clear marker
                    self._processed_ids.clear()  # No longer needed after sync
                    break

                logger.debug(
                    f"Session {self.room_id}: Processing backlog message {next_msg.id}"
                )
                await self._process_message(next_msg)

                if self._retry_tracker.is_permanently_failed(next_msg.id):
                    logger.warning(
                        f"Session {self.room_id}: Message {next_msg.id} permanently failed"
                    )
                    break

        except Exception as e:
            logger.error(f"Session {self.room_id}: Sync error: {e}", exc_info=True)

        logger.debug(f"Session {self.room_id}: Synchronization complete")

    async def _process_message(self, msg: PlatformMessage) -> None:
        """
        Process single message through full lifecycle.

        1. Check if message has exceeded max retries
        2. Mark message as processing
        3. Hydrate context (if first message)
        4. Create AgentTools for this message
        5. Call adapter callback with tools
        6. Mark message as processed (or failed)
        """
        # Skip permanently failed messages
        if self._retry_tracker.is_permanently_failed(msg.id):
            logger.debug(f"Skipping permanently failed message {msg.id}")
            return

        # Skip duplicates (LRU cache for late WebSocket duplicates)
        if msg.id in self._processed_ids:
            self._processed_ids.move_to_end(msg.id)
            logger.debug(f"Skipping duplicate message {msg.id}")
            return

        # Track attempts
        attempts, exceeded = self._retry_tracker.record_attempt(msg.id)
        if exceeded:
            return

        self.state = "processing"
        logger.info(f"Processing message {msg.id} in room {self.room_id}")

        try:
            # Mark as processing
            await self._coordinator._mark_processing(msg.id, self.room_id)

            # Hydrate context on first message (lazy loading)
            # Only hydrate if enabled - stateful frameworks may manage their own history
            if not self._context_hydrated and self.config.enable_context_hydration:
                await self._hydrate_context()
                self._context_hydrated = True

            # Create AgentTools for this message
            tools = self._coordinator._create_agent_tools(self.room_id)

            # Call the adapter/handler with message and tools
            await self._on_message(msg, tools)

            # Mark as processed - clear from attempts tracking
            await self._coordinator._mark_processed(msg.id, self.room_id)
            self._retry_tracker.mark_success(msg.id)

            # Add to LRU dedupe cache
            self._processed_ids[msg.id] = True
            if len(self._processed_ids) > self._max_processed_ids:
                self._processed_ids.popitem(last=False)

            logger.debug(f"Message {msg.id} processed successfully")

        except Exception as e:
            logger.error(f"Error processing message {msg.id}: {e}", exc_info=True)
            try:
                await self._coordinator._mark_failed(msg.id, self.room_id, str(e))
            except Exception as mark_error:
                logger.error(f"Failed to mark message as failed: {mark_error}")

        finally:
            self.state = "idle"

    async def _hydrate_context(self) -> None:
        """
        Hydrate conversation context for this room.

        Called lazily on first message to load conversation history
        and participant list.
        """
        logger.debug(f"Hydrating context for room: {self.room_id}")

        try:
            # Load participants first (needed for system prompt)
            await self.load_participants()

            # Load full context directly from API (not via get_context to avoid recursion)
            self._context_cache = await self._coordinator._fetch_context(self.room_id)
            logger.debug(
                f"Context hydrated: {len(self._context_cache.messages)} messages, "
                f"{len(self._participant_tracker.participants)} participants"
            )
        except Exception as e:
            logger.warning(f"Context hydration failed: {e}")
            # Create empty context as fallback
            self._context_cache = ConversationContext(
                room_id=self.room_id,
                messages=[],
                participants=[],
                hydrated_at=datetime.now(timezone.utc),
            )

    async def get_context(self, force_refresh: bool = False) -> ConversationContext:
        """
        Get conversation context (lazy, cached).

        Args:
            force_refresh: Force refresh from API even if cached

        Returns:
            ConversationContext with messages and participants
        """
        if force_refresh or self._context_cache is None:
            await self._hydrate_context()

        return self._context_cache or ConversationContext(
            room_id=self.room_id,
            messages=[],
            participants=[],
            hydrated_at=datetime.now(timezone.utc),
        )

    def enqueue_message(self, msg: PlatformMessage) -> None:
        """
        Add a message to this session's queue.

        Called by ThenvoiAgent when a WebSocket message arrives.

        Args:
            msg: Platform message to queue
        """
        # Track first WebSocket message ID for sync point detection
        if self._first_ws_msg_id is None:
            self._first_ws_msg_id = msg.id
        self.queue.put_nowait(msg)
        logger.debug(f"Message {msg.id} enqueued for room {self.room_id}")
