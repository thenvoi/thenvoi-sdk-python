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

try:
    from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions
except ImportError as e:
    raise ImportError(
        "claude-agent-sdk is required for Claude SDK examples.\n"
        "Install with: pip install claude-agent-sdk\n"
        "Or: uv add claude-agent-sdk"
    ) from e

logger = logging.getLogger(__name__)


@dataclass
class _SessionCommand:
    """Command to be processed by the session manager task."""

    action: str  # "create", "cleanup", "cleanup_all", "stop"
    room_id: str | None = None
    resume_session_id: str | None = None
    result_future: asyncio.Future[Any] | None = None


class ClaudeSessionManager:
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
        manager = ClaudeSessionManager(base_options)
        await manager.start()  # Start background task

        # Get client for room (creates if doesn't exist)
        client = await manager.get_or_create_session("room-123")

        # Use client
        await client.query("Hello")
        async for msg in client.receive_response():
            print(msg)

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
        self.base_options = base_options
        self._sessions: dict[str, ClaudeSDKClient] = {}
        self._command_queue: asyncio.Queue[_SessionCommand] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self._started = False
        logger.info("ClaudeSessionManager initialized")

    async def start(self) -> None:
        """Start the background task that manages all sessions."""
        if self._started:
            return

        self._task = asyncio.create_task(self._run_session_loop())
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

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        self._started = False
        logger.info("ClaudeSessionManager background task stopped")

    async def _run_session_loop(self) -> None:
        """Background task that processes all session commands."""
        logger.debug("Session loop started")

        while True:
            cmd: _SessionCommand | None = None
            try:
                cmd = await self._command_queue.get()

                if cmd.action == "create":
                    client = await self._do_create_session(
                        cmd.room_id, cmd.resume_session_id
                    )
                    if cmd.result_future:
                        cmd.result_future.set_result(client)

                elif cmd.action == "cleanup":
                    await self._do_cleanup_session(cmd.room_id)
                    if cmd.result_future:
                        cmd.result_future.set_result(None)

                elif cmd.action == "cleanup_all":
                    await self._do_cleanup_all()
                    if cmd.result_future:
                        cmd.result_future.set_result(None)

                elif cmd.action == "stop":
                    await self._do_cleanup_all()
                    if cmd.result_future:
                        cmd.result_future.set_result(None)
                    break

                self._command_queue.task_done()

            except asyncio.CancelledError:
                logger.debug("Session loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in session loop: {e}", exc_info=True)
                if cmd and cmd.result_future and not cmd.result_future.done():
                    cmd.result_future.set_exception(e)

        logger.debug("Session loop exited")

    async def _do_create_session(
        self, room_id: str | None, resume_session_id: str | None
    ) -> ClaudeSDKClient:
        """Create or get session (runs in background task)."""
        if not room_id:
            raise ValueError("room_id is required")

        if room_id not in self._sessions:
            # Build options, optionally with resume
            if resume_session_id:
                logger.info(f"Resuming session {resume_session_id} for room: {room_id}")
                # Create options with resume set
                options = ClaudeAgentOptions(
                    model=self.base_options.model,
                    system_prompt=self.base_options.system_prompt,
                    mcp_servers=self.base_options.mcp_servers,
                    allowed_tools=self.base_options.allowed_tools,
                    permission_mode=self.base_options.permission_mode,
                    resume=resume_session_id,
                )
                # Copy max_thinking_tokens if set
                if hasattr(self.base_options, "max_thinking_tokens"):
                    options.max_thinking_tokens = self.base_options.max_thinking_tokens
            else:
                logger.info(f"Creating new ClaudeSDKClient session for room: {room_id}")
                options = self.base_options

            # Create new client with options
            client = ClaudeSDKClient(options=options)

            # Connect the client (establishes session with Claude)
            await client.connect()

            # Store for reuse
            self._sessions[room_id] = client

            logger.info(
                f"Session created for room {room_id} "
                f"(total sessions: {len(self._sessions)})"
            )
        else:
            logger.debug(f"Reusing existing session for room: {room_id}")

        return self._sessions[room_id]

    async def _do_cleanup_session(self, room_id: str | None) -> None:
        """Cleanup single session (runs in background task)."""
        if not room_id or room_id not in self._sessions:
            logger.debug(f"No session to cleanup for room: {room_id}")
            return

        logger.info(f"Cleaning up session for room: {room_id}")

        try:
            await self._sessions[room_id].disconnect()
            logger.debug(f"Disconnected client for room {room_id}")
        except Exception as e:
            logger.warning(f"Error disconnecting session for room {room_id}: {e}")

        del self._sessions[room_id]

        logger.info(
            f"Session cleaned up for room {room_id} "
            f"(remaining sessions: {len(self._sessions)})"
        )

    async def _do_cleanup_all(self) -> None:
        """Cleanup all sessions (runs in background task)."""
        logger.info(f"Cleaning up all sessions (count: {len(self._sessions)})")

        room_ids = list(self._sessions.keys())
        for room_id in room_ids:
            await self._do_cleanup_session(room_id)

        logger.info("All sessions cleaned up")

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
