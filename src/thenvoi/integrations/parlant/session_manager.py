"""
Session manager for Parlant SDK clients.

Maintains one Parlant session per Thenvoi chat room to ensure
conversation continuity within each room.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ParlantSession:
    """Represents an active Parlant session for a room."""

    session_id: str
    agent_id: str
    customer_id: str
    last_offset: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class ParlantSessionManager:
    """
    Manages Parlant sessions per chat room.

    Each room gets its own Parlant session to maintain separate
    conversation histories.

    Key features:
    - Lazy initialization (sessions created on first message)
    - Session reuse (same session for all messages in a room)
    - Graceful cleanup on room leave/disconnect

    Example:
        manager = ParlantSessionManager(client, agent_id)

        # Get session for room (creates if doesn't exist)
        session = await manager.get_or_create_session(
            room_id="room-123",
            customer_id="user-456"
        )

        # Use session_id with client
        await client.sessions.create_event(
            session_id=session.session_id,
            kind="message",
            source="customer",
            message="Hello!"
        )

        # Cleanup when done
        await manager.cleanup_session("room-123")
    """

    def __init__(
        self,
        client: Any,  # AsyncParlantClient - parlant is optional dependency
        agent_id: str,
    ):
        """
        Initialize session manager.

        Args:
            client: AsyncParlantClient instance for Parlant API calls
            agent_id: The Parlant agent ID to use for sessions
        """
        self.client: Any = client
        self.agent_id = agent_id
        self._sessions: dict[str, ParlantSession] = {}
        self._lock = asyncio.Lock()
        logger.info(f"ParlantSessionManager initialized for agent: {agent_id}")

    async def get_or_create_session(
        self,
        room_id: str,
        customer_id: str,
        customer_name: str | None = None,
    ) -> ParlantSession:
        """
        Get existing session for room or create a new one.

        This method is idempotent - calling it multiple times for the same
        room_id returns the same session.

        Args:
            room_id: Thenvoi chat room ID (UUID)
            customer_id: Customer/user ID for the session
            customer_name: Optional display name for the customer

        Returns:
            ParlantSession instance for this room
        """
        async with self._lock:
            if room_id in self._sessions:
                logger.debug(f"Reusing existing session for room: {room_id}")
                return self._sessions[room_id]

            # Create new session via Parlant API
            logger.info(f"Creating new Parlant session for room: {room_id}")

            try:
                # First, create or get customer in Parlant
                # Parlant uses its own customer IDs, not Thenvoi's UUIDs
                parlant_customer_id: str | None = None
                try:
                    # Parlant v3.x customers.create() only accepts 'name' parameter
                    customer_response = await self.client.customers.create(
                        name=customer_name or customer_id,
                    )
                    parlant_customer_id = customer_response.id
                    logger.debug(
                        f"Created Parlant customer {parlant_customer_id} "
                        f"for Thenvoi user {customer_id}"
                    )
                except Exception as e:
                    # Customer creation failed - try to proceed anyway
                    # Some Parlant setups may not require explicit customer creation
                    logger.warning(f"Could not create customer in Parlant: {e}")

                # Create session with Parlant's customer ID (or None if customer creation failed)
                session_response = await self.client.sessions.create(
                    agent_id=self.agent_id,
                    customer_id=parlant_customer_id,
                )

                session = ParlantSession(
                    session_id=session_response.id,
                    agent_id=self.agent_id,
                    customer_id=parlant_customer_id or customer_id,
                    last_offset=0,
                    metadata={"room_id": room_id, "thenvoi_customer_id": customer_id},
                )

                self._sessions[room_id] = session

                logger.info(
                    f"Session created for room {room_id}: {session.session_id} "
                    f"(total sessions: {len(self._sessions)})"
                )

                return session

            except Exception as e:
                logger.error(f"Failed to create Parlant session: {e}", exc_info=True)
                raise

    async def cleanup_session(self, room_id: str) -> None:
        """
        Remove session for a room.

        This should be called when:
        - Agent is removed from the room
        - Room is deleted
        - Adapter is shutting down

        Args:
            room_id: Thenvoi chat room ID
        """
        async with self._lock:
            if room_id not in self._sessions:
                logger.debug(f"No session to cleanup for room: {room_id}")
                return

            session = self._sessions.pop(room_id)
            logger.info(
                f"Session cleaned up for room {room_id}: {session.session_id} "
                f"(remaining sessions: {len(self._sessions)})"
            )

    async def cleanup_all(self) -> None:
        """
        Remove all sessions.

        This should be called when the adapter is shutting down.
        """
        async with self._lock:
            count = len(self._sessions)
            self._sessions.clear()
            logger.info(f"All sessions cleaned up (count: {count})")

    def has_session(self, room_id: str) -> bool:
        """Check if session exists for room."""
        return room_id in self._sessions

    def get_session_count(self) -> int:
        """Get number of active sessions."""
        return len(self._sessions)

    def get_active_rooms(self) -> list[str]:
        """Get list of room IDs with active sessions."""
        return list(self._sessions.keys())

    def update_offset(self, room_id: str, offset: int) -> None:
        """Update the last processed event offset for a session."""
        if room_id in self._sessions:
            self._sessions[room_id].last_offset = offset
