"""
Session manager for Claude Agent SDK clients.

Maintains one ClaudeSDKClient instance per Thenvoi chat room to ensure
conversation continuity within each room.

Uses a dedicated background task for all session operations to ensure
connect() and disconnect() are always called from the same task context.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from thenvoi.core.nonfatal import NonFatalErrorRecorder
from thenvoi.integrations.lifecycle import AsyncIntegrationLifecycle
from thenvoi.integrations.claude_sdk.session_workflow import (
    do_cleanup_all,
    do_cleanup_session,
    do_create_session,
    fail_pending_commands,
    run_session_loop,
)

_CLAUDE_SDK_IMPORT_ERROR: ImportError | None = None

try:
    from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions
except ImportError as e:
    _CLAUDE_SDK_IMPORT_ERROR = e
    ClaudeSDKClient = Any
    ClaudeAgentOptions = Any

logger = logging.getLogger(__name__)


def _ensure_claude_sdk_available() -> None:
    """Raise a runtime error if optional Claude SDK dependency is missing."""
    if _CLAUDE_SDK_IMPORT_ERROR is not None:
        raise ImportError(
            "claude-agent-sdk is required for Claude SDK integrations.\n"
            "Install with: pip install claude-agent-sdk\n"
            "Or: uv add claude-agent-sdk"
        ) from _CLAUDE_SDK_IMPORT_ERROR


@dataclass
class _SessionCommand:
    """Command to be processed by the session manager task."""

    action: str  # "create", "cleanup", "cleanup_all", "stop"
    room_id: str | None = None
    resume_session_id: str | None = None
    result_future: asyncio.Future[Any] | None = None


class ClaudeSessionManager(NonFatalErrorRecorder):
    """
    Manages ClaudeSDKClient instances per chat room.

    Each room gets its own ClaudeSDKClient instance to maintain separate
    conversation histories and sessions.

    Key features:
    - Lazy initialization (clients created on first message)
    - Session reuse (same client for all messages in a room)
    - Graceful cleanup on room leave/disconnect
    - All session operations run in a single background task to avoid
      cross-task asyncio issues with connect/disconnect

    Example:
        import logging
        logger = logging.getLogger(__name__)

        manager = ClaudeSessionManager(base_options)
        await manager.start()  # Start background task

        # Get client for room (creates if doesn't exist)
        client = await manager.get_or_create_session("room-123")

        # Use client
        await client.query("Hello")
        async for msg in client.receive_response():
            logger.info("%s", msg)

        # Cleanup when done
        await manager.cleanup_session("room-123")
        await manager.stop()  # Stop background task
    """

    def __init__(self, base_options: ClaudeAgentOptions):
        """
        Initialize session manager.

        Args:
            base_options: Base ClaudeAgentOptions to use for all clients.
                          These options are shared across all room sessions.
        """
        _ensure_claude_sdk_available()
        self.base_options = base_options
        self._sessions: dict[str, ClaudeSDKClient] = {}
        self._command_queue: asyncio.Queue[_SessionCommand] = asyncio.Queue()
        self._task: asyncio.Task[Any] | None = None
        self._lifecycle = AsyncIntegrationLifecycle(
            owner="ClaudeSessionManager",
            logger=logger,
            on_task_error=self._on_lifecycle_task_error,
            fail_pending_operations=self._fail_pending_commands,
        )
        self._started = False
        self._init_nonfatal_errors()
        logger.info("ClaudeSessionManager initialized")

    async def start(self) -> None:
        """Start the background task that manages all sessions."""
        if self._started:
            return

        self._task = self._lifecycle.spawn_task(
            "session_loop",
            self._run_session_loop(),
        )
        self._started = True
        logger.info("ClaudeSessionManager background task started")

    async def stop(self) -> None:
        """Stop the background task and cleanup all sessions."""
        if not self._started:
            return

        # Send stop command
        stop_future: asyncio.Future[None] = asyncio.get_event_loop().create_future()
        await self._command_queue.put(
            _SessionCommand(action="stop", result_future=stop_future)
        )

        # Wait for cleanup to complete
        await stop_future

        await self._lifecycle.shutdown(
            fail_pending_reason="Claude session manager stopped"
        )
        self._task = None

        self._started = False
        logger.info("ClaudeSessionManager background task stopped")

    async def _run_session_loop(self) -> None:
        """Background task that processes all session commands."""
        await run_session_loop(self)

    def _on_lifecycle_task_error(self, task_name: str, error: Exception) -> None:
        """Record lifecycle task shutdown failures as nonfatal errors."""
        self._record_nonfatal_error(
            "session_manager_task",
            error,
            task=task_name,
        )

    def _fail_pending_commands(self, reason: str) -> None:
        """Fail queued command futures when lifecycle stops unexpectedly."""
        fail_pending_commands(self, reason)

    async def _do_create_session(
        self, room_id: str | None, resume_session_id: str | None
    ) -> ClaudeSDKClient:
        """Create or get session (runs in background task)."""
        return await do_create_session(
            room_id=room_id,
            resume_session_id=resume_session_id,
            base_options=self.base_options,
            sessions=self._sessions,
            options_type=ClaudeAgentOptions,
            client_type=ClaudeSDKClient,
        )

    async def _do_cleanup_session(self, room_id: str | None) -> None:
        """Cleanup single session (runs in background task)."""
        await do_cleanup_session(
            room_id=room_id,
            sessions=self._sessions,
            record_nonfatal_error=self._record_nonfatal_error,
        )

    async def _do_cleanup_all(self) -> None:
        """Cleanup all sessions (runs in background task)."""
        await do_cleanup_all(
            sessions=self._sessions,
            cleanup_session=self._do_cleanup_session,
        )

    async def get_or_create_session(
        self, room_id: str, resume_session_id: str | None = None
    ) -> ClaudeSDKClient:
        """
        Get existing ClaudeSDKClient for room or create new one.

        This method is idempotent - calling it multiple times for the same
        room_id returns the same client instance.

        Args:
            room_id: Thenvoi chat room ID (UUID)
            resume_session_id: Optional session ID to resume from a previous
                              session. Only used when creating a new session.

        Returns:
            ClaudeSDKClient instance for this room
        """
        if not self._started:
            await self.start()

        result_future: asyncio.Future[ClaudeSDKClient] = (
            asyncio.get_event_loop().create_future()
        )
        await self._command_queue.put(
            _SessionCommand(
                action="create",
                room_id=room_id,
                resume_session_id=resume_session_id,
                result_future=result_future,
            )
        )
        return await result_future

    async def cleanup_session(self, room_id: str) -> None:
        """
        Disconnect and remove session for a room.

        This should be called when:
        - Agent is removed from the room
        - Room is deleted
        - Adapter is shutting down

        Args:
            room_id: Thenvoi chat room ID
        """
        if not self._started:
            return

        result_future: asyncio.Future[None] = asyncio.get_event_loop().create_future()
        await self._command_queue.put(
            _SessionCommand(
                action="cleanup",
                room_id=room_id,
                result_future=result_future,
            )
        )
        await result_future

    async def cleanup_all(self) -> None:
        """
        Disconnect all sessions.

        This should be called when the adapter is shutting down to ensure
        all Claude SDK clients are properly disconnected.
        """
        if not self._started:
            return

        result_future: asyncio.Future[None] = asyncio.get_event_loop().create_future()
        await self._command_queue.put(
            _SessionCommand(
                action="cleanup_all",
                result_future=result_future,
            )
        )
        await result_future

    def has_session(self, room_id: str) -> bool:
        """Check if session exists for room."""
        return room_id in self._sessions

    def get_session_count(self) -> int:
        """Get number of active sessions."""
        return len(self._sessions)

    def get_active_rooms(self) -> list[str]:
        """Get list of room IDs with active sessions."""
        return list(self._sessions.keys())
