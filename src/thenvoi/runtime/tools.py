"""AgentTools - tools for LLM platform interaction."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from thenvoi.core.protocols import AgentToolsProtocol
from thenvoi.runtime.contacts.operations import ContactOperationsMixin
from thenvoi.runtime.contacts.service import ContactService
from thenvoi.runtime.memory_service import MemoryService
from thenvoi.runtime.tool_definitions import (
    ALL_TOOL_NAMES,
    BASE_TOOL_NAMES,
    CHAT_TOOL_NAMES,
    CONTACT_TOOL_NAMES,
    MCP_TOOL_PREFIX,
    MEMORY_TOOL_NAMES,
    TOOL_MODELS,
    AddContactInput,
    AddParticipantInput,
    ArchiveMemoryInput,
    CreateChatroomInput,
    GetMemoryInput,
    GetParticipantsInput,
    ListContactRequestsInput,
    ListContactsInput,
    ListMemoriesInput,
    LookupPeersInput,
    RemoveContactInput,
    RemoveParticipantInput,
    RespondContactRequestInput,
    SendEventInput,
    SendMessageInput,
    StoreMemoryInput,
    SupersedeMemoryInput,
    get_tool_description,
    mcp_tool_names,
)
from thenvoi.runtime.tool_dispatcher import ToolDispatcher
from thenvoi.runtime.tool_room_operations import RoomToolOperationsMixin
from thenvoi.runtime.tool_schema_provider import ToolSchemaProvider

if TYPE_CHECKING:
    from anthropic.types import ToolParam

    from thenvoi.client.rest import AsyncRestClient
    from thenvoi.runtime.execution import ExecutionContext

logger = logging.getLogger(__name__)


class AgentTools(
    RoomToolOperationsMixin,
    ContactOperationsMixin,
    AgentToolsProtocol,
):
    """Room-bound tools for LLM platform interaction."""

    def __init__(
        self,
        room_id: str,
        rest: "AsyncRestClient",
        participants: list[dict[str, Any]] | None = None,
    ):
        self.room_id = room_id
        self.rest = rest
        self._participants = participants or []
        self._contact_service = ContactService(rest)
        self._memory_service = MemoryService(rest)
        self._schema_provider = ToolSchemaProvider(
            tool_models=TOOL_MODELS,
            memory_tool_names=MEMORY_TOOL_NAMES,
        )
        self._tool_dispatcher = ToolDispatcher(tool_models=TOOL_MODELS)

    @classmethod
    def from_context(cls, ctx: "ExecutionContext") -> "AgentTools":
        """Create AgentTools from an ExecutionContext."""
        return cls(ctx.room_id, ctx.link.rest, ctx.participants)

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
        return await self._memory_service.list_memories(
            subject_id=subject_id,
            scope=scope,
            system=system,
            type=type,
            segment=segment,
            content_query=content_query,
            page_size=page_size,
            status=status,
        )

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
        return await self._memory_service.store_memory(
            content=content,
            system=system,
            type=type,
            segment=segment,
            thought=thought,
            scope=scope,
            subject_id=subject_id,
            metadata=metadata,
        )

    async def get_memory(self, memory_id: str) -> dict[str, Any]:
        """Retrieve a specific memory by ID."""
        return await self._memory_service.get_memory(memory_id)

    async def supersede_memory(self, memory_id: str) -> dict[str, Any]:
        """Mark a memory as superseded (soft delete)."""
        return await self._memory_service.supersede_memory(memory_id)

    async def archive_memory(self, memory_id: str) -> dict[str, Any]:
        """Archive a memory (hide but preserve)."""
        return await self._memory_service.archive_memory(memory_id)

    @property
    def tool_models(self) -> dict[str, type[BaseModel]]:
        """Get Pydantic models for all tools."""
        return self._schema_provider.tool_models

    def get_tool_schemas(
        self, format: str, *, include_memory: bool = False
    ) -> list[dict[str, Any]] | list["ToolParam"]:
        """Get tool schemas in provider-specific format."""
        return self._schema_provider.get_tool_schemas(
            format,
            include_memory=include_memory,
        )

    def get_anthropic_tool_schemas(
        self, *, include_memory: bool = False
    ) -> list["ToolParam"]:
        """Get tool schemas in Anthropic format (strongly typed)."""
        return self._schema_provider.get_anthropic_tool_schemas(
            include_memory=include_memory,
        )

    def get_openai_tool_schemas(
        self, *, include_memory: bool = False
    ) -> list[dict[str, Any]]:
        """Get tool schemas in OpenAI format (strongly typed)."""
        return self._schema_provider.get_openai_tool_schemas(
            include_memory=include_memory,
        )

    async def execute_tool_call_or_raise(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Any:
        """Execute tool call and raise ToolExecutionError on failure."""
        return await self._tool_dispatcher.execute_or_raise(self, tool_name, arguments)

    async def execute_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Execute a tool call via shared dispatcher with validation."""
        return await self._tool_dispatcher.execute(self, tool_name, arguments)


__all__ = [
    "AgentTools",
    "TOOL_MODELS",
    "MEMORY_TOOL_NAMES",
    "CONTACT_TOOL_NAMES",
    "ALL_TOOL_NAMES",
    "BASE_TOOL_NAMES",
    "CHAT_TOOL_NAMES",
    "MCP_TOOL_PREFIX",
    "mcp_tool_names",
    "get_tool_description",
    "SendMessageInput",
    "SendEventInput",
    "AddParticipantInput",
    "RemoveParticipantInput",
    "LookupPeersInput",
    "GetParticipantsInput",
    "CreateChatroomInput",
    "ListContactsInput",
    "AddContactInput",
    "RemoveContactInput",
    "ListContactRequestsInput",
    "RespondContactRequestInput",
    "ListMemoriesInput",
    "StoreMemoryInput",
    "GetMemoryInput",
    "SupersedeMemoryInput",
    "ArchiveMemoryInput",
]
