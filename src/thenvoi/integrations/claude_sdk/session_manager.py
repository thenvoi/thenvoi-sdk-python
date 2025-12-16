"""
Session manager for Claude Agent SDK clients.

Maintains one ClaudeSDKClient instance per Thenvoi chat room to ensure
conversation continuity within each room.
"""

from __future__ import annotations

import logging
from typing import Dict

try:
    from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions
except ImportError as e:
    raise ImportError(
        "claude-agent-sdk is required for Claude SDK examples.\n"
        "Install with: pip install claude-agent-sdk\n"
        "Or: uv add claude-agent-sdk"
    ) from e

logger = logging.getLogger(__name__)


class ClaudeSessionManager:
    """
    Manages ClaudeSDKClient instances per chat room.

    Each room gets its own ClaudeSDKClient instance to maintain separate
    conversation histories and sessions.

    Key features:
    - Lazy initialization (clients created on first message)
    - Session reuse (same client for all messages in a room)
    - Graceful cleanup on room leave/disconnect

    Example:
        manager = ClaudeSessionManager(base_options)

        # Get client for room (creates if doesn't exist)
        client = await manager.get_or_create_session("room-123")

        # Use client
        await client.query("Hello")
        async for msg in client.receive_response():
            print(msg)

        # Cleanup when done
        await manager.cleanup_session("room-123")
    """

    def __init__(self, base_options: ClaudeAgentOptions):
        """
        Initialize session manager.

        Args:
            base_options: Base ClaudeAgentOptions to use for all clients.
                          These options are shared across all room sessions.
        """
        self.base_options = base_options
        self._sessions: Dict[str, ClaudeSDKClient] = {}
        logger.info("ClaudeSessionManager initialized")

    async def get_or_create_session(self, room_id: str) -> ClaudeSDKClient:
        """
        Get existing ClaudeSDKClient for room or create new one.

        This method is idempotent - calling it multiple times for the same
        room_id returns the same client instance.

        Args:
            room_id: Thenvoi chat room ID (UUID)

        Returns:
            ClaudeSDKClient instance for this room
        """
        if room_id not in self._sessions:
            logger.info(f"Creating new ClaudeSDKClient session for room: {room_id}")

            # Create new client with base options
            client = ClaudeSDKClient(options=self.base_options)

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
        if room_id in self._sessions:
            logger.info(f"Cleaning up session for room: {room_id}")

            try:
                await self._sessions[room_id].disconnect()
                logger.debug(f"Disconnected client for room {room_id}")
            except Exception as e:
                logger.error(
                    f"Error disconnecting session for room {room_id}: {e}",
                    exc_info=True,
                )

            del self._sessions[room_id]

            logger.info(
                f"Session cleaned up for room {room_id} "
                f"(remaining sessions: {len(self._sessions)})"
            )
        else:
            logger.debug(f"No session to cleanup for room: {room_id}")

    async def cleanup_all(self) -> None:
        """
        Disconnect all sessions.

        This should be called when the adapter is shutting down to ensure
        all Claude SDK clients are properly disconnected.
        """
        logger.info(f"Cleaning up all sessions (count: {len(self._sessions)})")

        # Get list of room IDs to avoid modifying dict during iteration
        room_ids = list(self._sessions.keys())

        for room_id in room_ids:
            await self.cleanup_session(room_id)

        logger.info("All sessions cleaned up")

    def has_session(self, room_id: str) -> bool:
        """Check if session exists for room."""
        return room_id in self._sessions

    def get_session_count(self) -> int:
        """Get number of active sessions."""
        return len(self._sessions)

    def get_active_rooms(self) -> list[str]:
        """Get list of room IDs with active sessions."""
        return list(self._sessions.keys())
