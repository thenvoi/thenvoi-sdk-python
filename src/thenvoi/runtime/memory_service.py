"""Memory API operations extracted from AgentTools."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from thenvoi.client.rest import AsyncRestClient

logger = logging.getLogger(__name__)


class MemoryService:
    """Focused service for memory CRUD operations via AsyncRestClient."""

    def __init__(self, rest: "AsyncRestClient") -> None:
        self._rest = rest

    async def list_memories(
        self,
        subject_id: str | None = None,
        scope: str | None = None,
        system: str | None = None,
        type: str | None = None,
        segment: str | None = None,
        content_query: str | None = None,
        page_size: int = 50,
        status: str | None = None,
    ) -> dict[str, Any]:
        """List memories accessible to the agent."""
        logger.debug(
            "Listing memories: subject_id=%s, scope=%s, system=%s",
            subject_id,
            scope,
            system,
        )
        response = await self._rest.agent_api_memories.list_agent_memories(
            subject_id=subject_id,
            scope=scope,
            system=system,
            type=type,
            segment=segment,
            content_query=content_query,
            page_size=page_size,
            status=status,
        )

        memories = []
        if response.data:
            memories = [
                {
                    "id": m.id,
                    "content": m.content,
                    "system": m.system,
                    "type": m.type,
                    "segment": m.segment,
                    "scope": m.scope,
                    "status": m.status,
                    "thought": m.thought,
                    "subject_id": str(m.subject_id) if m.subject_id else None,
                    "source_agent_id": str(m.source_agent_id)
                    if m.source_agent_id
                    else None,
                    "inserted_at": str(m.inserted_at) if m.inserted_at else None,
                }
                for m in response.data
            ]

        metadata = {
            "page_size": response.meta.page_size if response.meta else page_size,
            "total_count": response.meta.total_count
            if response.meta
            else len(memories),
        }

        return {"memories": memories, "metadata": metadata}

    async def store_memory(
        self,
        content: str,
        system: str,
        type: str,
        segment: str,
        thought: str,
        scope: str = "subject",
        subject_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Store a new memory entry."""
        from thenvoi.client.rest import MemoryCreateRequest

        logger.debug(
            "Storing memory: system=%s, type=%s, segment=%s, scope=%s",
            system,
            type,
            segment,
            scope,
        )
        response = await self._rest.agent_api_memories.create_agent_memory(
            memory=MemoryCreateRequest(
                content=content,
                system=system,
                type=type,
                segment=segment,
                thought=thought,
                scope=scope,
                subject_id=subject_id,
                metadata=metadata,
            )
        )
        if not response.data:
            raise RuntimeError("Failed to store memory - no response data")
        return {
            "id": response.data.id,
            "content": response.data.content,
            "system": response.data.system,
            "type": response.data.type,
            "segment": response.data.segment,
            "scope": response.data.scope,
            "status": response.data.status,
            "thought": response.data.thought,
            "inserted_at": str(response.data.inserted_at)
            if response.data.inserted_at
            else None,
        }

    async def get_memory(self, memory_id: str) -> dict[str, Any]:
        """Retrieve a specific memory by ID."""
        logger.debug("Getting memory: id=%s", memory_id)
        response = await self._rest.agent_api_memories.get_agent_memory(id=memory_id)
        if not response.data:
            raise RuntimeError("Failed to get memory - no response data")
        return {
            "id": response.data.id,
            "content": response.data.content,
            "system": response.data.system,
            "type": response.data.type,
            "segment": response.data.segment,
            "scope": response.data.scope,
            "status": response.data.status,
            "thought": response.data.thought,
            "subject_id": str(response.data.subject_id)
            if response.data.subject_id
            else None,
            "source_agent_id": str(response.data.source_agent_id)
            if response.data.source_agent_id
            else None,
            "inserted_at": str(response.data.inserted_at)
            if response.data.inserted_at
            else None,
        }

    async def supersede_memory(self, memory_id: str) -> dict[str, Any]:
        """Mark a memory as superseded (soft delete)."""
        logger.debug("Superseding memory: id=%s", memory_id)
        response = await self._rest.agent_api_memories.supersede_agent_memory(
            id=memory_id
        )
        if not response.data:
            raise RuntimeError("Failed to supersede memory - no response data")
        return {
            "id": response.data.id,
            "status": response.data.status,
        }

    async def archive_memory(self, memory_id: str) -> dict[str, Any]:
        """Archive a memory (hide but preserve)."""
        logger.debug("Archiving memory: id=%s", memory_id)
        response = await self._rest.agent_api_memories.archive_agent_memory(id=memory_id)
        if not response.data:
            raise RuntimeError("Failed to archive memory - no response data")
        return {
            "id": response.data.id,
            "status": response.data.status,
        }
