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
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal

from .types import (
    ConversationContext,
    MessageHandler,
    PlatformMessage,
    SessionConfig,
)

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

        # Participants list (updated via WebSocket events)
        self._participants: list[dict[str, Any]] = []
        self._participants_loaded = False

        # LLM context tracking (for adapters)
        self._llm_initialized = False  # Has system prompt been sent?
        self._last_participants_sent: list[dict[str, Any]] | None = (
            None  # Last sent to LLM
        )

        # Message retry tracking
        self._message_attempts: dict[str, int] = {}  # msg_id -> attempt count
        self._permanently_failed: set[str] = set()  # Messages that exceeded max retries

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
        """Check if session is running."""
        return self._is_running

    @property
    def participants(self) -> list[dict[str, Any]]:
        """Get current participants list."""
        return self._participants.copy()

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
        if self._last_participants_sent is None:
            return True  # First time, always send

        # Compare by IDs
        last_ids = {p.get("id") for p in self._last_participants_sent}
        current_ids = {p.get("id") for p in self._participants}
        return last_ids != current_ids

    def build_participants_message(self) -> str:
        """Build a system message with current participant list for LLM."""
        if not self._participants:
            return "## Current Participants\nNo other participants in this room."

        lines = ["## Current Participants"]
        for p in self._participants:
            p_type = p.get("type", "Unknown")
            p_name = p.get("name", "Unknown")
            p_id = p.get("id", "")
            lines.append(f"- {p_name} (ID: {p_id}, Type: {p_type})")

        lines.append("")
        lines.append(
            "When using send_message, include mentions with ID and name from this list."
        )

        return "\n".join(lines)

    def mark_participants_sent(self) -> None:
        """Mark current participants as sent to LLM."""
        self._last_participants_sent = self._participants.copy()
        logger.debug(f"Session {self.room_id}: Participants sent to LLM")

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
            List of message dicts ready for LLM formatting
        """
        # Ensure context is hydrated
        if not self._context_hydrated:
            await self._hydrate_context()
            self._context_hydrated = True

        if not self._context_cache:
            logger.debug(
                f"Session {self.room_id}: No context cache, returning empty history"
            )
            return []

        history = []
        for msg in self._context_cache.messages:
            msg_id = msg.get("id")
            if msg_id == exclude_message_id:
                continue

            sender_type = msg.get("sender_type", "")
            sender_name = msg.get("sender_name") or msg.get("name") or sender_type
            content = msg.get("content", "")

            # Map sender_type to LLM role
            if sender_type == "Agent":
                role = "assistant"
            else:
                role = "user"

            history.append(
                {
                    "role": role,
                    "content": content,
                    "sender_name": sender_name,
                    "sender_type": sender_type,
                }
            )

        logger.info(
            f"Session {self.room_id}: Loaded {len(history)} historical messages"
        )
        return history

    def add_participant(self, participant: dict) -> None:
        """Add a participant (called from WebSocket event)."""
        # Avoid duplicates
        if any(p.get("id") == participant.get("id") for p in self._participants):
            return
        self._participants.append(
            {
                "id": participant.get("id"),
                "name": participant.get("name"),
                "type": participant.get("type"),
            }
        )
        logger.debug(
            f"Session {self.room_id}: Added participant {participant.get('name')}"
        )
        logger.debug(
            f"Session {self.room_id}: Current participants: {self._participants}"
        )

    def remove_participant(self, participant: dict) -> None:
        """Remove a participant (called from WebSocket event)."""
        participant_id = participant.get("id")
        self._participants = [
            p for p in self._participants if p.get("id") != participant_id
        ]
        logger.debug(
            f"Session {self.room_id}: Removed participant {participant.get('name')}"
        )
        logger.debug(
            f"Session {self.room_id}: Current participants: {self._participants}"
        )

    async def load_participants(self) -> list[dict[str, Any]]:
        """Load participants from API (called during hydration)."""
        if self._participants_loaded:
            return self._participants

        try:
            self._participants = await self._coordinator._get_participants_internal(
                self.room_id
            )
            self._participants_loaded = True
            logger.debug(
                f"Session {self.room_id}: Loaded {len(self._participants)} participants: {self._participants}"
            )
        except Exception as e:
            logger.warning(f"Failed to load participants for room {self.room_id}: {e}")
            self._participants = []

        return self._participants

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
        Stop processing and wait for current message to complete.

        Gracefully stops the session, allowing any in-progress
        message processing to finish.
        """
        if not self._is_running:
            return

        logger.info(f"Stopping session for room: {self.room_id}")
        self._is_running = False

        if self._process_loop_task:
            # Wait for task to finish (it will exit on next queue timeout)
            try:
                await asyncio.wait_for(self._process_loop_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning(f"Session {self.room_id} stop timed out, cancelling")
                self._process_loop_task.cancel()
                try:
                    await self._process_loop_task
                except asyncio.CancelledError:
                    pass
            self._process_loop_task = None

    async def _process_loop(self) -> None:
        """
        Main processing loop for this room.

        SYNCHRONIZATION FLOW:
        1. Call /next to get unprocessed messages from backend
        2. For each /next message, check if it matches WebSocket queue head
        3. If match → synchronized! Process once, then switch to WebSocket only
        4. If no match → process /next message, repeat
        5. After sync, process only from WebSocket queue
        """
        try:
            # Phase 1: Sync via /next until we catch up with WebSocket
            await self._synchronize_with_next()
            self._synchronized = True
            self.state = "idle"
            logger.info(f"Session {self.room_id}: Synchronized, switching to WebSocket")

            # Phase 2: Process from WebSocket queue only
            while self._is_running:
                try:
                    # Wait for message with timeout (allows graceful shutdown)
                    msg = await asyncio.wait_for(self.queue.get(), timeout=60.0)
                    await self._process_message(msg)
                except asyncio.TimeoutError:
                    # No messages, stay idle (but check if still running)
                    pass
                except asyncio.CancelledError:
                    logger.debug(f"Session {self.room_id} cancelled")
                    break

        except Exception as e:
            logger.error(f"Session {self.room_id} error: {e}", exc_info=True)
            self._is_running = False

        logger.debug(f"Session {self.room_id} loop exited")

    async def _synchronize_with_next(self) -> None:
        """
        Synchronize backlog via /next API until caught up with WebSocket.

        ALGORITHM:
        1. Call /next to get next unprocessed message
        2. If None → no backlog, we're synced
        3. Check if message ID matches head of WebSocket queue
        4. If match → synced! Process this message once (it's in both)
        5. If no match → process /next message, repeat from step 1

        The sync point is when /next returns the same message that's
        at the head of the WebSocket queue. This means we've processed
        all messages that arrived while offline.
        """
        logger.debug(f"Session {self.room_id}: Starting /next synchronization")

        try:
            while self._is_running:
                # Get next unprocessed message from backend
                next_msg = await self._coordinator._get_next_message(self.room_id)

                if next_msg is None:
                    # No more messages in backlog → synced
                    logger.debug(f"Session {self.room_id}: /next returned None, synced")
                    break

                # Check if this message is also at the head of WebSocket queue
                ws_head_id = self._peek_queue_head_id()

                if ws_head_id is not None and next_msg.id == ws_head_id:
                    # SYNC POINT: Same message in both sources
                    # Check if permanently failed first
                    if next_msg.id in self._permanently_failed:
                        logger.warning(
                            f"Session {self.room_id}: Sync point message {next_msg.id} permanently failed, breaking sync"
                        )
                        self._remove_from_queue_head(next_msg.id)
                        break

                    # Process it once, remove from WebSocket queue, then we're done
                    logger.info(
                        f"Session {self.room_id}: Sync point reached at message {next_msg.id}"
                    )
                    await self._process_message(next_msg)
                    # Remove the duplicate from WebSocket queue
                    self._remove_from_queue_head(next_msg.id)
                    break
                else:
                    # Not synced yet - process this /next message
                    # But first check if it's permanently failed
                    if next_msg.id in self._permanently_failed:
                        logger.warning(
                            f"Session {self.room_id}: Skipping permanently failed message {next_msg.id}, breaking sync"
                        )
                        break

                    logger.debug(
                        f"Session {self.room_id}: Processing backlog message {next_msg.id}"
                    )
                    await self._process_message(next_msg)

                    # If message became permanently failed during processing, break sync
                    if next_msg.id in self._permanently_failed:
                        logger.warning(
                            f"Session {self.room_id}: Message {next_msg.id} permanently failed, breaking sync"
                        )
                        break

        except Exception as e:
            logger.error(f"Session {self.room_id}: Sync error: {e}", exc_info=True)
            # Continue anyway - fall through to WebSocket processing

        logger.debug(f"Session {self.room_id}: Synchronization complete")

    def _peek_queue_head_id(self) -> str | None:
        """
        Peek at the ID of the message at the head of WebSocket queue.

        Returns None if queue is empty.
        Does NOT remove the message from the queue.
        """
        if self.queue.empty():
            return None

        # Access internal deque to peek without removing
        # Note: This is safe because we're the only consumer
        try:
            head_msg = self.queue._queue[0]  # type: ignore[attr-defined]
            return head_msg.id
        except (IndexError, AttributeError):
            return None

    def _remove_from_queue_head(self, msg_id: str) -> bool:
        """
        Remove message from queue head if it matches the given ID.

        Returns True if removed, False otherwise.
        """
        if self.queue.empty():
            return False

        try:
            head_msg = self.queue._queue[0]  # type: ignore[attr-defined]
            if head_msg.id == msg_id:
                self.queue.get_nowait()  # Remove it
                logger.debug(
                    f"Session {self.room_id}: Removed duplicate {msg_id} from queue"
                )
                return True
        except (IndexError, AttributeError):
            pass

        return False

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
        if msg.id in self._permanently_failed:
            logger.debug(f"Skipping permanently failed message {msg.id}")
            return

        # Track attempts
        attempts = self._message_attempts.get(msg.id, 0) + 1
        self._message_attempts[msg.id] = attempts

        if attempts > self.config.max_message_retries:
            logger.error(
                f"Message {msg.id} exceeded max retries ({self.config.max_message_retries}), "
                "marking as permanently failed"
            )
            self._permanently_failed.add(msg.id)
            return

        self.state = "processing"
        logger.info(f"Processing message {msg.id} in room {self.room_id}")

        try:
            # Mark as processing
            await self._coordinator._mark_processing(msg.id, self.room_id)

            # Hydrate context on first message (lazy loading)
            if not self._context_hydrated:
                await self._hydrate_context()
                self._context_hydrated = True

            # Create AgentTools for this message
            tools = self._coordinator._create_agent_tools(self.room_id)

            # Call the adapter/handler with message and tools
            await self._on_message(msg, tools)

            # Mark as processed - clear from attempts tracking
            await self._coordinator._mark_processed(msg.id, self.room_id)
            self._message_attempts.pop(msg.id, None)
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
                f"{len(self._participants)} participants"
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

        # Check cache TTL
        if self._context_cache and self.config.enable_context_cache:
            age = (
                datetime.now(timezone.utc) - self._context_cache.hydrated_at
            ).total_seconds()
            if age > self.config.context_cache_ttl_seconds:
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
        self.queue.put_nowait(msg)
        logger.debug(f"Message {msg.id} enqueued for room {self.room_id}")
