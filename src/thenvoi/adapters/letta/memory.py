"""
Memory block management for Letta agents.

Handles:
- Per-room context tracking (SHARED mode)
- Participant updates
- Memory consolidation on room exit
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from letta_client import Letta

logger = logging.getLogger(__name__)


class MemoryBlocks:
    """Standard memory block labels."""

    PERSONA = "persona"
    PARTICIPANTS = "participants"
    ROOM_CONTEXTS = "room_contexts"


class MemoryManager:
    """
    Manages Letta agent memory blocks.

    Used primarily in SHARED mode to track per-room context
    within a single agent's memory.
    """

    def __init__(self, client: Letta):
        self._client = client

    async def update_participants(
        self,
        agent_id: str,
        participants: list[dict[str, Any]],
    ) -> None:
        """Update the participants memory block."""
        formatted = self._format_participants(participants)

        try:
            # Letta client is sync, use thread
            await asyncio.to_thread(
                self._client.agents.blocks.update,
                agent_id=agent_id,
                block_label=MemoryBlocks.PARTICIPANTS,
                value=formatted,
            )
            logger.debug(f"Updated participants for agent {agent_id}")
        except Exception as e:
            logger.warning(f"Failed to update participants: {e}")

    async def get_room_contexts(self, agent_id: str) -> dict[str, str]:
        """
        Parse room_contexts memory block into dict.

        Returns:
            Dict mapping room_id to context string
        """
        try:
            block = await asyncio.to_thread(
                self._client.agents.blocks.retrieve,
                agent_id=agent_id,
                block_label=MemoryBlocks.ROOM_CONTEXTS,
            )

            if hasattr(block, "value") and block.value:
                return self._parse_room_contexts(block.value)

            return {}
        except Exception as e:
            logger.warning(f"Failed to get room contexts: {e}")
            return {}

    async def update_room_context(
        self,
        agent_id: str,
        room_id: str,
        topic: str,
        key_points: list[str] | None = None,
    ) -> None:
        """
        Update a specific room's context in room_contexts block.

        This is called after each message to keep context fresh.
        """
        try:
            # Get current contexts
            current_contexts = await self.get_room_contexts(agent_id)

            # Build this room's context
            room_context = f"Topic: {topic}"
            if key_points:
                room_context += f"\nKey points: {'; '.join(key_points)}"

            current_contexts[room_id] = room_context

            # Format and save
            formatted = self._format_room_contexts(current_contexts)

            await asyncio.to_thread(
                self._client.agents.blocks.update,
                agent_id=agent_id,
                block_label=MemoryBlocks.ROOM_CONTEXTS,
                value=formatted,
            )
            logger.debug(f"Updated room context for {room_id}")
        except Exception as e:
            logger.warning(f"Failed to update room context: {e}")

    async def consolidate_room_memory(
        self,
        agent_id: str,
        room_id: str,
        summary: str,
    ) -> None:
        """
        Consolidate room memory on exit (compress to essentials).

        Called during on_cleanup to store a compact summary of the room interaction.
        """
        try:
            current_contexts = await self.get_room_contexts(agent_id)

            # Store condensed summary
            current_contexts[room_id] = f"Summary: {summary[:200]}"

            formatted = self._format_room_contexts(current_contexts)

            await asyncio.to_thread(
                self._client.agents.blocks.update,
                agent_id=agent_id,
                block_label=MemoryBlocks.ROOM_CONTEXTS,
                value=formatted,
            )
            logger.debug(f"Consolidated memory for room {room_id}")
        except Exception as e:
            logger.warning(f"Failed to consolidate room memory: {e}")

    def _format_participants(self, participants: list[dict[str, Any]]) -> str:
        """Format participants list for memory block."""
        lines = ["## Current Room Participants\n"]
        for p in participants:
            p_type = p.get("type", "Unknown")
            p_name = p.get("name", "Unknown")
            lines.append(f"- {p_name} ({p_type})")

        lines.append("\nTo mention a participant, use their EXACT name.")
        return "\n".join(lines)

    def _parse_room_contexts(self, value: str) -> dict[str, str]:
        """Parse room_contexts block into dict."""
        if not value or value.startswith("No room contexts"):
            return {}

        contexts: dict[str, str] = {}
        current_room: str | None = None
        current_content: list[str] = []

        for line in value.split("\n"):
            if line.startswith("## Room:"):
                if current_room:
                    contexts[current_room] = "\n".join(current_content).strip()
                current_room = line.replace("## Room:", "").strip()
                current_content = []
            elif current_room:
                current_content.append(line)

        if current_room:
            contexts[current_room] = "\n".join(current_content).strip()

        return contexts

    def _format_room_contexts(self, contexts: dict[str, str]) -> str:
        """Format room contexts dict for memory block."""
        if not contexts:
            return (
                "No room contexts yet. Update this as you interact in different rooms."
            )

        lines: list[str] = []
        for room_id in sorted(contexts.keys()):
            context = contexts[room_id]
            lines.append(f"## Room: {room_id}")
            lines.append(context)
            lines.append("")

        return "\n".join(lines)
