"""
AgentRuntime - Convenience wrapper combining RoomPresence + Execution management.

For SDK-heavy users who want managed execution contexts.
Framework-light users can use RoomPresence or ThenvoiLink directly.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Awaitable, Callable

from thenvoi.platform.event import MessageEvent, PlatformEvent

from .execution import Execution, ExecutionContext, ExecutionHandler
from .presence import RoomPresence
from .types import SessionConfig

if TYPE_CHECKING:
    from thenvoi.platform.link import ThenvoiLink

logger = logging.getLogger(__name__)

# Factory type for custom execution implementations
ExecutionFactory = Callable[[str, "ThenvoiLink"], Execution]


class AgentRuntime:
    """
    Convenience wrapper: RoomPresence + Execution management.

    For SDK-heavy users who want managed execution contexts.
    Framework-light users can use RoomPresence directly.

    Manages:
    - Agent presence across rooms via RoomPresence
    - Per-room execution contexts via ExecutionContext (or custom)
    - Lifecycle coordination (start, stop, run)

    Example (default execution):
        link = ThenvoiLink(agent_id, api_key, ...)

        async def on_execute(ctx: ExecutionContext, event: PlatformEvent):
            if isinstance(event, MessageEvent):
                tools = AgentTools.from_context(ctx)
                # Process message with LLM...

        runtime = AgentRuntime(link, agent_id, on_execute=on_execute)
        await runtime.run()

    Example (custom execution factory):
        def letta_factory(room_id: str, link: ThenvoiLink) -> Execution:
            return LettaExecution(room_id, link)

        runtime = AgentRuntime(
            link,
            agent_id,
            on_execute=my_handler,
            execution_factory=letta_factory,
        )
        await runtime.run()
    """

    def __init__(
        self,
        link: "ThenvoiLink",
        agent_id: str,
        on_execute: ExecutionHandler,
        execution_factory: ExecutionFactory | None = None,
        room_filter: Callable[[dict], bool] | None = None,
        session_config: SessionConfig | None = None,
        on_session_cleanup: Callable[[str], Awaitable[None]] | None = None,
    ):
        """
        Initialize AgentRuntime.

        Args:
            link: ThenvoiLink for WebSocket and REST
            agent_id: Agent ID from Thenvoi platform
            on_execute: Callback for handling execution events
            execution_factory: Optional factory for custom Execution implementations
            room_filter: Optional filter to decide which rooms to join
            session_config: Configuration for ExecutionContext
            on_session_cleanup: Optional callback for session cleanup (receives room_id)
        """
        self.link = link
        self.agent_id = agent_id
        self._on_execute = on_execute
        self._execution_factory = execution_factory
        self._session_config = session_config or SessionConfig()
        self._on_session_cleanup = on_session_cleanup

        # RoomPresence for cross-room management
        self.presence = RoomPresence(link, room_filter)

        # Per-room executions
        self.executions: dict[str, Execution] = {}
        self._room_last_message_at: dict[str, float] = {}
        self._room_activity_gen: dict[str, int] = {}
        self._room_locks: dict[str, asyncio.Lock] = {}
        self._idle_monitor_task: asyncio.Task[None] | None = None

        # Set up presence callbacks
        self.presence.on_room_joined = self._on_room_joined
        self.presence.on_room_left = self._on_room_left
        self.presence.on_room_event = self._on_room_event

    @property
    def active_sessions(self) -> dict[str, Execution]:
        """Get active execution contexts by room_id."""
        return self.executions.copy()

    async def start(self) -> None:
        """
        Start the agent runtime.

        1. Starts RoomPresence (connects link, subscribes to rooms)
        2. Creates execution contexts for existing rooms
        """
        logger.info("Starting AgentRuntime for agent %s", self.agent_id)
        await self.presence.start()
        await self._start_idle_monitor()

    async def stop(self, timeout: float | None = None) -> bool:
        """
        Stop the agent runtime with optional graceful timeout.

        Args:
            timeout: Optional seconds to wait for current processing to complete
                     in each execution context. None means cancel immediately.

        Returns:
            True if all executions stopped gracefully, False if any had to be
            cancelled mid-processing.

        1. Stops all execution contexts (with timeout)
        2. Stops RoomPresence
        """
        logger.info("Stopping AgentRuntime for agent %s", self.agent_id)
        await self._stop_idle_monitor()

        # Stop all executions with timeout
        all_graceful = True
        for room_id in list(self.executions.keys()):
            graceful = await self._destroy_execution(room_id, timeout=timeout)
            all_graceful = all_graceful and graceful

        await self.presence.stop()
        self._clear_room_tracking_state()
        return all_graceful

    async def run(self) -> None:
        """
        Run the agent until stopped or interrupted.

        Starts the runtime and keeps the WebSocket connection alive.
        """
        await self.start()
        try:
            await self.link.run_forever()
        except Exception as e:
            logger.error("AgentRuntime error: %s", e)
            raise
        finally:
            await self.stop()

    # --- Presence callbacks ---

    async def _on_room_joined(self, room_id: str, payload: dict) -> None:
        """Handle room joined - create execution context."""
        await self._create_execution(room_id)

    async def _on_room_left(self, room_id: str) -> None:
        """Handle room left - destroy execution context."""
        await self._destroy_execution(room_id, retire_lock=True)

    async def _on_room_event(self, room_id: str, event: PlatformEvent) -> None:
        """Handle room event - forward to execution context."""
        if isinstance(event, MessageEvent):
            execution = await self._prepare_message_execution(room_id)
            if not execution:
                return
            await execution.on_event(event)
            return

        execution = self.executions.get(room_id)
        if execution:
            await execution.on_event(event)
            return

        if room_id in self.presence.rooms:
            logger.debug(
                "No execution for room %s and non-message event %s, event dropped",
                room_id,
                event.type,
            )
            return

        logger.warning("No execution for room %s, event dropped", room_id)

    # --- Execution management ---

    async def _create_execution(self, room_id: str) -> Execution:
        """Create and start execution context for a room."""
        lock = self._get_room_lock(room_id)
        async with lock:
            return await self._create_execution_locked(room_id)

    async def _create_execution_locked(self, room_id: str) -> Execution:
        """Create an execution while holding the per-room lock."""
        if room_id in self.executions:
            logger.debug("Execution already exists for room %s", room_id)
            return self.executions[room_id]

        # Use factory if provided, otherwise create ExecutionContext
        if self._execution_factory:
            execution = self._execution_factory(room_id, self.link)
        else:
            execution = ExecutionContext(
                room_id=room_id,
                link=self.link,
                on_execute=self._on_execute,
                config=self._session_config,
                agent_id=self.agent_id,
            )

        self.executions[room_id] = execution
        self._ensure_room_activity(room_id)
        try:
            await execution.start()
        except Exception:
            self.executions.pop(room_id, None)
            self._room_last_message_at.pop(room_id, None)
            self._room_activity_gen.pop(room_id, None)
            raise

        logger.debug("Created execution for room %s", room_id)
        return execution

    async def _destroy_execution(
        self,
        room_id: str,
        timeout: float | None = None,
        *,
        expected_generation: int | None = None,
        expected_last_message_at: float | None = None,
        idle_timeout_seconds: float | None = None,
        retire_lock: bool = False,
    ) -> bool:
        """
        Stop and cleanup execution context for a room.

        Args:
            room_id: Room ID to destroy execution for.
            timeout: Optional seconds to wait for graceful stop.

        Returns:
            True if stopped gracefully, False if cancelled mid-processing.
        """
        lock = self._get_room_lock(room_id)
        should_retire_lock = False
        async with lock:
            execution = self.executions.get(room_id)
            if execution is None:
                should_retire_lock = retire_lock and room_id not in self.presence.rooms
                if should_retire_lock:
                    self._retire_room_state(room_id, retire_lock=True)
                return True

            if not self._should_destroy_execution(
                room_id=room_id,
                execution=execution,
                expected_generation=expected_generation,
                expected_last_message_at=expected_last_message_at,
                idle_timeout_seconds=idle_timeout_seconds,
            ):
                return True

            execution = self.executions.pop(room_id)
            self._retire_room_state(room_id, retire_lock=False)
            if retire_lock and room_id not in self.presence.rooms:
                self._room_locks.pop(room_id, None)

        graceful = await execution.stop(timeout=timeout)

        # Call cleanup callback (for adapter to clean up checkpointer, etc.)
        if self._on_session_cleanup:
            try:
                await self._on_session_cleanup(room_id)
            except Exception as e:
                logger.warning("Session cleanup callback failed for %s: %s", room_id, e)

        logger.debug("Destroyed execution for room %s", room_id)
        return graceful

    async def _prepare_message_execution(self, room_id: str) -> Execution | None:
        """Touch room activity and ensure a message has an execution context."""
        lock = self._get_room_lock(room_id)
        async with lock:
            execution = self.executions.get(room_id)
            if execution is None:
                if room_id not in self.presence.rooms:
                    self._retire_room_state(room_id, retire_lock=True)
                    logger.warning("No execution for room %s, event dropped", room_id)
                    return None

                self._touch_room_activity(room_id)
                return await self._create_execution_locked(room_id)

            self._touch_room_activity(room_id)
            return execution

    async def _start_idle_monitor(self) -> None:
        """Start idle monitor if idle timeout is enabled."""
        if self._idle_monitor_task and not self._idle_monitor_task.done():
            return

        idle_timeout_seconds = self._idle_timeout_seconds()
        if idle_timeout_seconds is None:
            return

        interval = min(max(idle_timeout_seconds / 2.0, 1.0), 5.0)
        self._idle_monitor_task = asyncio.create_task(
            self._run_idle_monitor(idle_timeout_seconds, interval),
            name=f"idle-monitor-{self.agent_id}",
        )
        logger.debug(
            "Started idle monitor for agent %s (timeout=%ss interval=%ss)",
            self.agent_id,
            idle_timeout_seconds,
            interval,
        )

    async def _stop_idle_monitor(self) -> None:
        """Stop idle monitor task if running."""
        if not self._idle_monitor_task:
            return

        if not self._idle_monitor_task.done():
            self._idle_monitor_task.cancel()
            try:
                await self._idle_monitor_task
            except asyncio.CancelledError:
                pass
        self._idle_monitor_task = None

    async def _run_idle_monitor(
        self, idle_timeout_seconds: float, interval: float
    ) -> None:
        """Periodically evict idle execution contexts."""
        try:
            while True:
                await asyncio.sleep(interval)
                await self._evict_idle_executions(idle_timeout_seconds)
        except asyncio.CancelledError:
            logger.debug("Idle monitor cancelled for agent %s", self.agent_id)
            raise

    async def _evict_idle_executions(self, idle_timeout_seconds: float) -> None:
        """Evict rooms that exceed message idle timeout."""
        now = time.monotonic()
        candidates: list[tuple[str, int, float]] = []

        for room_id in list(self.executions.keys()):
            last_message_at = self._room_last_message_at.get(room_id)
            if last_message_at is None:
                continue
            if now - last_message_at <= idle_timeout_seconds:
                continue
            candidates.append(
                (
                    room_id,
                    self._room_activity_gen.get(room_id, 0),
                    last_message_at,
                )
            )

        for room_id, generation, last_message_at in candidates:
            await self._destroy_execution(
                room_id,
                timeout=0.0,
                expected_generation=generation,
                expected_last_message_at=last_message_at,
                idle_timeout_seconds=idle_timeout_seconds,
            )

    def _should_destroy_execution(
        self,
        *,
        room_id: str,
        execution: Execution,
        expected_generation: int | None,
        expected_last_message_at: float | None,
        idle_timeout_seconds: float | None,
    ) -> bool:
        """Validate optional idle-monitor snapshot before destroy commit."""
        if expected_generation is None:
            return True

        current_generation = self._room_activity_gen.get(room_id, 0)
        current_last_message_at = self._room_last_message_at.get(room_id)
        if current_generation != expected_generation:
            logger.debug("Skip idle destroy for room %s: generation changed", room_id)
            return False
        if current_last_message_at != expected_last_message_at:
            logger.debug("Skip idle destroy for room %s: timestamp changed", room_id)
            return False

        if idle_timeout_seconds is not None and current_last_message_at is not None:
            if time.monotonic() - current_last_message_at <= idle_timeout_seconds:
                logger.debug(
                    "Skip idle destroy for room %s: room no longer idle", room_id
                )
                return False

        busy_state = self._execution_busy_state(execution)
        if busy_state is None:
            logger.debug(
                "Skip idle destroy for room %s: execution has no busy-state signal",
                room_id,
            )
            return False
        if busy_state:
            logger.debug("Skip idle destroy for room %s: execution is busy", room_id)
            return False
        return True

    @staticmethod
    def _execution_busy_state(execution: Execution) -> bool | None:
        """Read execution busy state in a compatibility-safe way."""
        if isinstance(execution, ExecutionContext):
            return execution.is_processing

        busy_attr = getattr(execution, "is_processing", None)
        if busy_attr is None:
            return None

        try:
            busy_value = busy_attr() if callable(busy_attr) else busy_attr
        except Exception:
            logger.warning(
                "Failed to inspect is_processing for execution in room %s",
                execution.room_id,
                exc_info=True,
            )
            return None

        if isinstance(busy_value, bool):
            return busy_value

        logger.debug(
            "Execution in room %s returned non-bool is_processing value: %r",
            execution.room_id,
            busy_value,
        )
        return None

    def _idle_timeout_seconds(self) -> float | None:
        """Return normalized idle timeout (None when disabled)."""
        idle_timeout_seconds = self._session_config.idle_timeout_seconds
        if idle_timeout_seconds is None or idle_timeout_seconds == 0:
            return None
        return idle_timeout_seconds

    def _ensure_room_activity(self, room_id: str) -> None:
        """Initialize room activity tracking if missing."""
        now = time.monotonic()
        self._room_last_message_at.setdefault(room_id, now)
        self._room_activity_gen.setdefault(room_id, 0)

    def _touch_room_activity(self, room_id: str) -> None:
        """Record message activity for a room."""
        now = time.monotonic()
        self._room_last_message_at[room_id] = now
        self._room_activity_gen[room_id] = self._room_activity_gen.get(room_id, 0) + 1

    def _retire_room_state(self, room_id: str, *, retire_lock: bool) -> None:
        """Retire per-room tracking state for inactive rooms."""
        self._room_last_message_at.pop(room_id, None)
        self._room_activity_gen.pop(room_id, None)
        if retire_lock:
            self._room_locks.pop(room_id, None)

    def _get_room_lock(self, room_id: str) -> asyncio.Lock:
        """Get or create the per-room lock."""
        lock = self._room_locks.get(room_id)
        if lock is None:
            lock = asyncio.Lock()
            self._room_locks[room_id] = lock
        return lock

    def _clear_room_tracking_state(self) -> None:
        """Clear per-room tracking state during runtime shutdown."""
        self._room_last_message_at.clear()
        self._room_activity_gen.clear()
        self._room_locks.clear()
