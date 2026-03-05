"""Protocol for bridge message handlers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from thenvoi.runtime.tools import AgentTools


@runtime_checkable
class Handler(Protocol):
    """Protocol for bridge message handlers.

    Implement this to create handlers for specific agents.
    Each handler receives parsed @mention messages and can
    respond using the provided AgentTools.
    """

    async def handle(
        self,
        content: str,
        room_id: str,
        thread_id: str,
        message_id: str,
        sender_id: str,
        sender_name: str | None,
        sender_type: str,
        mentioned_agent: str,
        tools: AgentTools,
    ) -> None:
        """Handle a routed @mention message.

        Args:
            content: The message content.
            room_id: The chat room ID where the message was sent.
            thread_id: The thread ID (defaults to room_id if not set).
            message_id: The platform message ID.
            sender_id: ID of the message sender.
            sender_name: Display name of the sender, or None if unresolvable.
            sender_type: Type of sender ("User", "Agent", "System").
            mentioned_agent: The agent name that was @mentioned.
            tools: AgentTools instance bound to the room for sending responses.
        """
        ...

    async def close(self) -> None:
        """Clean up handler resources (e.g. HTTP clients).

        Safe to call multiple times.  Handlers that don't own external
        resources can leave this as a no-op.
        """
        ...
