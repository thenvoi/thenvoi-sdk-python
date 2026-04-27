"""Shared CrewAI BaseTool wrappers for Thenvoi platform tools.

Both CrewAIAdapter and CrewAIFlowAdapter consume the same tool builder so that
the platform tool surface stays consistent across adapters and Flow authors who
spawn sub-Crews inside @listen methods get platform tools without copying code.

The builder takes three injectables:
- get_context: callable returning the current room context (room_id + tools).
  Each adapter owns its own ContextVar and supplies its own getter.
- reporter: CrewAIToolReporter implementation. Two ship in this module:
  EmitExecutionReporter (gates by Emit.EXECUTION) and NoopReporter.
- capabilities: frozenset[Capability] — controls which tool subset is exposed.

Extracted from src/thenvoi/adapters/crewai.py during Phase 0 of the
CrewAIFlowAdapter spec.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Protocol, Type, runtime_checkable

from pydantic import BaseModel, Field, field_validator

try:
    from crewai.tools import BaseTool
except ImportError as e:  # pragma: no cover - same import guard as the adapter
    raise ImportError(
        "crewai is required for CrewAI adapter.\n"
        "Install with: pip install 'thenvoi-sdk[crewai]'\n"
        "Or: uv add crewai nest-asyncio"
    ) from e

from thenvoi.core.protocols import AgentToolsProtocol
from thenvoi.core.types import AdapterFeatures, Capability, Emit
from thenvoi.integrations.crewai.runtime import run_async
from thenvoi.runtime.custom_tools import CustomToolDef, get_custom_tool_name
from thenvoi.runtime.tools import get_tool_description

logger = logging.getLogger(__name__)


# --- Shared context + reporter contracts ---


@dataclass(frozen=True)
class CrewAIToolContext:
    """Snapshot of the current room context passed to tool wrappers.

    Each adapter owns its own ContextVar and supplies its own getter that
    returns this dataclass. Tools never reach back into the adapter directly.
    """

    room_id: str
    tools: AgentToolsProtocol


@runtime_checkable
class CrewAIToolReporter(Protocol):
    """Hook for tool execution event emission.

    Implementations decide whether to send tool_call / tool_result events to
    the platform. The default EmitExecutionReporter gates emission on
    Emit.EXECUTION. NoopReporter never emits.

    Both methods are best-effort: implementations must not raise on transport
    failure. Wrappers depend on this contract.
    """

    async def report_call(
        self,
        tools: AgentToolsProtocol,
        tool_name: str,
        input_data: dict[str, Any],
    ) -> None: ...

    async def report_result(
        self,
        tools: AgentToolsProtocol,
        tool_name: str,
        result: Any,
        is_error: bool = False,
    ) -> None: ...


class EmitExecutionReporter:
    """Reporter gated by Emit.EXECUTION — matches legacy CrewAIAdapter behavior."""

    def __init__(self, features: AdapterFeatures) -> None:
        self._features = features

    async def report_call(
        self,
        tools: AgentToolsProtocol,
        tool_name: str,
        input_data: dict[str, Any],
    ) -> None:
        if Emit.EXECUTION not in self._features.emit:
            return
        try:
            await tools.send_event(
                content=json.dumps({"tool": tool_name, "input": input_data}),
                message_type="tool_call",
            )
        except Exception as e:
            logger.warning("Failed to send tool_call event: %s", e)

    async def report_result(
        self,
        tools: AgentToolsProtocol,
        tool_name: str,
        result: Any,
        is_error: bool = False,
    ) -> None:
        if Emit.EXECUTION not in self._features.emit:
            return
        try:
            key = "error" if is_error else "result"
            await tools.send_event(
                content=json.dumps({"tool": tool_name, key: result}),
                message_type="tool_result",
            )
        except Exception as e:
            logger.warning("Failed to send tool_result event: %s", e)


class NoopReporter:
    """Reporter that emits nothing — useful for adapters that report elsewhere."""

    async def report_call(
        self,
        tools: AgentToolsProtocol,
        tool_name: str,
        input_data: dict[str, Any],
    ) -> None:
        return None

    async def report_result(
        self,
        tools: AgentToolsProtocol,
        tool_name: str,
        result: Any,
        is_error: bool = False,
    ) -> None:
        return None


# --- Helpers ---


def serialize_success_result(result: Any) -> str:
    """Serialize a successful tool result without losing domain status fields.

    Pydantic models are converted via model_dump at the serialization boundary.
    Dicts that already carry a "status" key (e.g. domain status from REST
    responses) get that field renamed to "result_status" so the wrapper's
    own "status": "success" envelope stays unambiguous.
    """
    if hasattr(result, "model_dump"):
        result = result.model_dump()
    if isinstance(result, dict):
        payload = dict(result)
        result_status = payload.pop("status", None)
        response: dict[str, Any] = {"status": "success", **payload}
        if result_status is not None:
            response["result_status"] = result_status
        return json.dumps(response, default=str)
    return json.dumps({"status": "success", "result": result}, default=str)


def _execute_tool(
    *,
    tool_name: str,
    coro_factory: Callable[[AgentToolsProtocol], Any],
    get_context: Callable[[], CrewAIToolContext | None],
    reporter: CrewAIToolReporter,
    fallback_loop: asyncio.AbstractEventLoop | None,
) -> str:
    """Execute a tool with common error handling and reporting.

    Returns a JSON string with status and result/error.
    """
    context = get_context()
    if context is None:
        return json.dumps(
            {
                "status": "error",
                "message": "No room context available - tool called outside message handling",
            }
        )

    room_id = context.room_id
    tools = context.tools

    async def _execute() -> str:
        try:
            return await coro_factory(tools)
        except Exception as e:
            error_msg = str(e)
            logger.error("%s failed in room %s: %s", tool_name, room_id, error_msg)
            await reporter.report_result(tools, tool_name, error_msg, is_error=True)
            return json.dumps({"status": "error", "message": error_msg})

    return run_async(_execute(), fallback_loop=fallback_loop)


# --- Input models ---


class _SendMessageInput(BaseModel):
    content: str = Field(..., description="The message content to send")
    mentions: str = Field(
        default="[]",
        description='JSON array of participant handles to @mention (e.g., \'["@john", "@john/weather-agent"]\')',
    )

    @field_validator("mentions", mode="before")
    @classmethod
    def normalize_mentions(cls, v: Any) -> str:
        if v is None:
            return "[]"
        if isinstance(v, list):
            return json.dumps(v)
        return v


class _SendEventInput(BaseModel):
    content: str = Field(..., description="Human-readable event content")
    message_type: str = Field(
        default="thought",
        description="Type of event: 'thought', 'error', or 'task'",
    )


class _AddParticipantInput(BaseModel):
    identifier: str = Field(
        ...,
        description=(
            "Identifier of participant to add — can be a handle, name, "
            "or ID (from thenvoi_lookup_peers). Prefer the exact ID "
            "returned by thenvoi_lookup_peers; handles are mainly for mentions."
        ),
    )
    role: str = Field(
        default="member", description="Role: 'owner', 'admin', or 'member'"
    )


class _RemoveParticipantInput(BaseModel):
    identifier: str = Field(
        ...,
        description=(
            "Identifier of the participant to remove — can be a handle, name, or ID"
        ),
    )


class _GetParticipantsInput(BaseModel):
    pass


class _LookupPeersInput(BaseModel):
    pass


class _CreateChatroomInput(BaseModel):
    task_id: str | None = Field(
        default=None, description="Associated task ID (optional)"
    )


class _ListContactsInput(BaseModel):
    page: int = Field(default=1, description="Page number")
    page_size: int = Field(default=50, description="Items per page (max 100)")


class _AddContactInput(BaseModel):
    handle: str = Field(
        ...,
        description="Handle of user/agent to add (e.g., '@john' or '@john/agent-name')",
    )
    message: str | None = Field(
        default=None, description="Optional message with the request"
    )


class _RemoveContactInput(BaseModel):
    handle: str | None = Field(default=None, description="Contact's handle")
    contact_id: str | None = Field(
        default=None, description="Or contact record ID (UUID)"
    )


class _ListContactRequestsInput(BaseModel):
    page: int = Field(default=1, description="Page number")
    page_size: int = Field(default=50, description="Items per page (max 100)")
    sent_status: str = Field(
        default="pending", description="Filter sent requests by status"
    )


class _RespondContactRequestInput(BaseModel):
    action: str = Field(
        ..., description="Action to take ('approve', 'reject', 'cancel')"
    )
    handle: str | None = Field(default=None, description="Other party's handle")
    request_id: str | None = Field(default=None, description="Or request ID (UUID)")


class _ListMemoriesInput(BaseModel):
    subject_id: str | None = Field(default=None, description="Filter by subject UUID")
    scope: str | None = Field(
        default=None, description="Filter by scope (subject, organization, all)"
    )
    system: str | None = Field(
        default=None,
        description="Filter by memory system (sensory, working, long_term)",
    )
    memory_type: str | None = Field(default=None, description="Filter by memory type")
    segment: str | None = Field(
        default=None, description="Filter by segment (user, agent, tool, guideline)"
    )
    content_query: str | None = Field(
        default=None, description="Full-text search query"
    )
    page_size: int = Field(default=50, description="Number of results per page")
    status: str | None = Field(
        default=None,
        description="Filter by status (active, superseded, archived, all)",
    )


class _StoreMemoryInput(BaseModel):
    content: str = Field(..., description="The memory content")
    system: str = Field(..., description="Memory system tier")
    memory_type: str = Field(..., description="Memory type")
    segment: str = Field(..., description="Logical segment")
    thought: str = Field(..., description="Agent's reasoning for storing this memory")
    scope: str = Field(default="subject", description="Visibility scope")
    subject_id: str | None = Field(
        default=None, description="UUID of the subject (required for subject scope)"
    )


class _GetMemoryInput(BaseModel):
    memory_id: str = Field(..., description="Memory ID (UUID)")


class _SupersedeMemoryInput(BaseModel):
    memory_id: str = Field(..., description="Memory ID (UUID)")


class _ArchiveMemoryInput(BaseModel):
    memory_id: str = Field(..., description="Memory ID (UUID)")


# --- Tool factory ---

_no_cache: Any = staticmethod(lambda *_a, **_kw: False)


def _make_platform_tools(
    *,
    get_context: Callable[[], CrewAIToolContext | None],
    reporter: CrewAIToolReporter,
    fallback_loop: asyncio.AbstractEventLoop | None,
) -> tuple[list[BaseTool], list[BaseTool], list[BaseTool]]:
    """Build the 7 base + 5 contact + 5 memory platform tools.

    Returns a (base, contacts, memory) triple. ``build_thenvoi_crewai_tools``
    is responsible for stitching them together based on the requested
    capabilities.
    """

    def _exec(tool_name: str, factory: Callable[[AgentToolsProtocol], Any]) -> str:
        return _execute_tool(
            tool_name=tool_name,
            coro_factory=factory,
            get_context=get_context,
            reporter=reporter,
            fallback_loop=fallback_loop,
        )

    class SendMessageTool(BaseTool):
        name: str = "thenvoi_send_message"
        description: str = get_tool_description("thenvoi_send_message")
        args_schema: Type[BaseModel] = _SendMessageInput
        cache_function: Any = _no_cache

        def _run(self, *_args: Any, **kwargs: Any) -> Any:
            content: str = kwargs.get("content", "")
            mentions: str = kwargs.get("mentions", "[]")
            try:
                mention_list = json.loads(mentions) if mentions else []
            except json.JSONDecodeError:
                mention_list = []

            async def execute(tools: AgentToolsProtocol) -> str:
                await reporter.report_call(
                    tools,
                    "thenvoi_send_message",
                    {"content": content, "mentions": mention_list},
                )
                await tools.send_message(content, mention_list)
                await reporter.report_result(tools, "thenvoi_send_message", "success")
                return json.dumps({"status": "success", "message": "Message sent"})

            return _exec("thenvoi_send_message", execute)

    class SendEventTool(BaseTool):
        name: str = "thenvoi_send_event"
        description: str = get_tool_description("thenvoi_send_event")
        args_schema: Type[BaseModel] = _SendEventInput
        cache_function: Any = _no_cache

        def _run(self, *_args: Any, **kwargs: Any) -> Any:
            content: str = kwargs.get("content", "")
            message_type: str = kwargs.get("message_type", "thought")

            async def execute(tools: AgentToolsProtocol) -> str:
                # No execution reporting for send_event to avoid meta-events.
                await tools.send_event(content, message_type)
                return json.dumps({"status": "success", "message": "Event sent"})

            return _exec("thenvoi_send_event", execute)

    class AddParticipantTool(BaseTool):
        name: str = "thenvoi_add_participant"
        description: str = get_tool_description("thenvoi_add_participant")
        args_schema: Type[BaseModel] = _AddParticipantInput
        cache_function: Any = _no_cache

        def _run(self, *_args: Any, **kwargs: Any) -> Any:
            identifier: str = kwargs.get("identifier", "")
            role: str = kwargs.get("role", "member")

            async def execute(tools: AgentToolsProtocol) -> str:
                await reporter.report_call(
                    tools,
                    "thenvoi_add_participant",
                    {"identifier": identifier, "role": role},
                )
                result = await tools.add_participant(identifier, role)
                await reporter.report_result(tools, "thenvoi_add_participant", result)
                return serialize_success_result(result)

            return _exec("thenvoi_add_participant", execute)

    class RemoveParticipantTool(BaseTool):
        name: str = "thenvoi_remove_participant"
        description: str = get_tool_description("thenvoi_remove_participant")
        args_schema: Type[BaseModel] = _RemoveParticipantInput
        cache_function: Any = _no_cache

        def _run(self, *_args: Any, **kwargs: Any) -> Any:
            identifier: str = kwargs.get("identifier", "")

            async def execute(tools: AgentToolsProtocol) -> str:
                await reporter.report_call(
                    tools, "thenvoi_remove_participant", {"identifier": identifier}
                )
                result = await tools.remove_participant(identifier)
                await reporter.report_result(
                    tools, "thenvoi_remove_participant", result
                )
                return serialize_success_result(result)

            return _exec("thenvoi_remove_participant", execute)

    class GetParticipantsTool(BaseTool):
        name: str = "thenvoi_get_participants"
        description: str = get_tool_description("thenvoi_get_participants")
        args_schema: Type[BaseModel] = _GetParticipantsInput
        cache_function: Any = _no_cache

        def _run(self, *_args: Any, **_kwargs: Any) -> Any:
            async def execute(tools: AgentToolsProtocol) -> str:
                await reporter.report_call(tools, "thenvoi_get_participants", {})
                participants = await tools.get_participants()
                serialized = (
                    [
                        p.model_dump() if hasattr(p, "model_dump") else p
                        for p in participants
                    ]
                    if isinstance(participants, list)
                    else participants
                )
                result = {
                    "status": "success",
                    "participants": serialized,
                    "count": len(participants) if isinstance(participants, list) else 0,
                }
                await reporter.report_result(tools, "thenvoi_get_participants", result)
                return json.dumps(result, default=str)

            return _exec("thenvoi_get_participants", execute)

    class LookupPeersTool(BaseTool):
        name: str = "thenvoi_lookup_peers"
        description: str = get_tool_description("thenvoi_lookup_peers")
        args_schema: Type[BaseModel] = _LookupPeersInput
        cache_function: Any = _no_cache

        def _run(self, *_args: Any, **_kwargs: Any) -> Any:
            page, page_size = 1, 50

            async def execute(tools: AgentToolsProtocol) -> str:
                await reporter.report_call(
                    tools,
                    "thenvoi_lookup_peers",
                    {"page": page, "page_size": page_size},
                )
                result = await tools.lookup_peers(page, page_size)
                await reporter.report_result(tools, "thenvoi_lookup_peers", result)
                return serialize_success_result(result)

            return _exec("thenvoi_lookup_peers", execute)

    class CreateChatroomTool(BaseTool):
        name: str = "thenvoi_create_chatroom"
        description: str = get_tool_description("thenvoi_create_chatroom")
        args_schema: Type[BaseModel] = _CreateChatroomInput
        cache_function: Any = _no_cache

        def _run(self, *_args: Any, **kwargs: Any) -> Any:
            task_id: str | None = kwargs.get("task_id")

            async def execute(tools: AgentToolsProtocol) -> str:
                await reporter.report_call(
                    tools, "thenvoi_create_chatroom", {"task_id": task_id}
                )
                new_room_id = await tools.create_chatroom(task_id)
                result = {
                    "status": "success",
                    "message": "Chat room created",
                    "room_id": new_room_id,
                }
                await reporter.report_result(tools, "thenvoi_create_chatroom", result)
                return json.dumps(result)

            return _exec("thenvoi_create_chatroom", execute)

    class ListContactsTool(BaseTool):
        name: str = "thenvoi_list_contacts"
        description: str = get_tool_description("thenvoi_list_contacts")
        args_schema: Type[BaseModel] = _ListContactsInput
        cache_function: Any = _no_cache

        def _run(self, *_args: Any, **kwargs: Any) -> Any:
            page: int = kwargs.get("page", 1)
            page_size: int = kwargs.get("page_size", 50)

            async def execute(tools: AgentToolsProtocol) -> str:
                await reporter.report_call(
                    tools,
                    "thenvoi_list_contacts",
                    {"page": page, "page_size": page_size},
                )
                result = await tools.list_contacts(page, page_size)
                await reporter.report_result(tools, "thenvoi_list_contacts", result)
                return serialize_success_result(result)

            return _exec("thenvoi_list_contacts", execute)

    class AddContactTool(BaseTool):
        name: str = "thenvoi_add_contact"
        description: str = get_tool_description("thenvoi_add_contact")
        args_schema: Type[BaseModel] = _AddContactInput
        cache_function: Any = _no_cache

        def _run(self, *_args: Any, **kwargs: Any) -> Any:
            handle: str = kwargs.get("handle", "")
            message: str | None = kwargs.get("message")

            async def execute(tools: AgentToolsProtocol) -> str:
                await reporter.report_call(
                    tools,
                    "thenvoi_add_contact",
                    {"handle": handle, "message": message},
                )
                result = await tools.add_contact(handle, message)
                await reporter.report_result(tools, "thenvoi_add_contact", result)
                return serialize_success_result(result)

            return _exec("thenvoi_add_contact", execute)

    class RemoveContactTool(BaseTool):
        name: str = "thenvoi_remove_contact"
        description: str = get_tool_description("thenvoi_remove_contact")
        args_schema: Type[BaseModel] = _RemoveContactInput
        cache_function: Any = _no_cache

        def _run(self, *_args: Any, **kwargs: Any) -> Any:
            handle: str | None = kwargs.get("handle")
            contact_id: str | None = kwargs.get("contact_id")

            async def execute(tools: AgentToolsProtocol) -> str:
                await reporter.report_call(
                    tools,
                    "thenvoi_remove_contact",
                    {"handle": handle, "contact_id": contact_id},
                )
                result = await tools.remove_contact(handle, contact_id)
                await reporter.report_result(tools, "thenvoi_remove_contact", result)
                return serialize_success_result(result)

            return _exec("thenvoi_remove_contact", execute)

    class ListContactRequestsTool(BaseTool):
        name: str = "thenvoi_list_contact_requests"
        description: str = get_tool_description("thenvoi_list_contact_requests")
        args_schema: Type[BaseModel] = _ListContactRequestsInput
        cache_function: Any = _no_cache

        def _run(self, *_args: Any, **kwargs: Any) -> Any:
            page: int = kwargs.get("page", 1)
            page_size: int = kwargs.get("page_size", 50)
            sent_status: str = kwargs.get("sent_status", "pending")

            async def execute(tools: AgentToolsProtocol) -> str:
                await reporter.report_call(
                    tools,
                    "thenvoi_list_contact_requests",
                    {
                        "page": page,
                        "page_size": page_size,
                        "sent_status": sent_status,
                    },
                )
                result = await tools.list_contact_requests(page, page_size, sent_status)
                await reporter.report_result(
                    tools, "thenvoi_list_contact_requests", result
                )
                return serialize_success_result(result)

            return _exec("thenvoi_list_contact_requests", execute)

    class RespondContactRequestTool(BaseTool):
        name: str = "thenvoi_respond_contact_request"
        description: str = get_tool_description("thenvoi_respond_contact_request")
        args_schema: Type[BaseModel] = _RespondContactRequestInput
        cache_function: Any = _no_cache

        def _run(self, *_args: Any, **kwargs: Any) -> Any:
            action: str = kwargs.get("action", "")
            handle: str | None = kwargs.get("handle")
            request_id: str | None = kwargs.get("request_id")

            async def execute(tools: AgentToolsProtocol) -> str:
                await reporter.report_call(
                    tools,
                    "thenvoi_respond_contact_request",
                    {"action": action, "handle": handle, "request_id": request_id},
                )
                result = await tools.respond_contact_request(action, handle, request_id)
                await reporter.report_result(
                    tools, "thenvoi_respond_contact_request", result
                )
                return serialize_success_result(result)

            return _exec("thenvoi_respond_contact_request", execute)

    class ListMemoriesTool(BaseTool):
        name: str = "thenvoi_list_memories"
        description: str = get_tool_description("thenvoi_list_memories")
        args_schema: Type[BaseModel] = _ListMemoriesInput
        cache_function: Any = _no_cache

        def _run(self, *_args: Any, **kwargs: Any) -> Any:
            subject_id = kwargs.get("subject_id")
            scope = kwargs.get("scope")
            system = kwargs.get("system")
            memory_type = kwargs.get("memory_type")
            segment = kwargs.get("segment")
            content_query = kwargs.get("content_query")
            page_size = kwargs.get("page_size", 50)
            status = kwargs.get("status")

            async def execute(tools: AgentToolsProtocol) -> str:
                await reporter.report_call(
                    tools,
                    "thenvoi_list_memories",
                    {
                        "subject_id": subject_id,
                        "scope": scope,
                        "system": system,
                        "type": memory_type,
                        "segment": segment,
                        "content_query": content_query,
                        "page_size": page_size,
                        "status": status,
                    },
                )
                result = await tools.list_memories(
                    subject_id=subject_id,
                    scope=scope,
                    system=system,
                    type=memory_type,
                    segment=segment,
                    content_query=content_query,
                    page_size=page_size,
                    status=status,
                )
                await reporter.report_result(tools, "thenvoi_list_memories", result)
                return serialize_success_result(result)

            return _exec("thenvoi_list_memories", execute)

    class StoreMemoryTool(BaseTool):
        name: str = "thenvoi_store_memory"
        description: str = get_tool_description("thenvoi_store_memory")
        args_schema: Type[BaseModel] = _StoreMemoryInput
        cache_function: Any = _no_cache

        def _run(self, *_args: Any, **kwargs: Any) -> Any:
            content = kwargs.get("content", "")
            system = kwargs.get("system", "")
            memory_type = kwargs.get("memory_type", "")
            segment = kwargs.get("segment", "")
            thought = kwargs.get("thought", "")
            scope = kwargs.get("scope", "subject")
            subject_id = kwargs.get("subject_id")

            async def execute(tools: AgentToolsProtocol) -> str:
                await reporter.report_call(
                    tools,
                    "thenvoi_store_memory",
                    {
                        "content": content,
                        "system": system,
                        "type": memory_type,
                        "segment": segment,
                        "thought": thought,
                        "scope": scope,
                        "subject_id": subject_id,
                    },
                )
                result = await tools.store_memory(
                    content=content,
                    system=system,
                    type=memory_type,
                    segment=segment,
                    thought=thought,
                    scope=scope,
                    subject_id=subject_id,
                )
                await reporter.report_result(tools, "thenvoi_store_memory", result)
                return serialize_success_result(result)

            return _exec("thenvoi_store_memory", execute)

    class GetMemoryTool(BaseTool):
        name: str = "thenvoi_get_memory"
        description: str = get_tool_description("thenvoi_get_memory")
        args_schema: Type[BaseModel] = _GetMemoryInput
        cache_function: Any = _no_cache

        def _run(self, *_args: Any, **kwargs: Any) -> Any:
            memory_id = kwargs.get("memory_id", "")

            async def execute(tools: AgentToolsProtocol) -> str:
                await reporter.report_call(
                    tools, "thenvoi_get_memory", {"memory_id": memory_id}
                )
                result = await tools.get_memory(memory_id)
                await reporter.report_result(tools, "thenvoi_get_memory", result)
                return serialize_success_result(result)

            return _exec("thenvoi_get_memory", execute)

    class SupersedeMemoryTool(BaseTool):
        name: str = "thenvoi_supersede_memory"
        description: str = get_tool_description("thenvoi_supersede_memory")
        args_schema: Type[BaseModel] = _SupersedeMemoryInput
        cache_function: Any = _no_cache

        def _run(self, *_args: Any, **kwargs: Any) -> Any:
            memory_id = kwargs.get("memory_id", "")

            async def execute(tools: AgentToolsProtocol) -> str:
                await reporter.report_call(
                    tools, "thenvoi_supersede_memory", {"memory_id": memory_id}
                )
                result = await tools.supersede_memory(memory_id)
                await reporter.report_result(tools, "thenvoi_supersede_memory", result)
                return serialize_success_result(result)

            return _exec("thenvoi_supersede_memory", execute)

    class ArchiveMemoryTool(BaseTool):
        name: str = "thenvoi_archive_memory"
        description: str = get_tool_description("thenvoi_archive_memory")
        args_schema: Type[BaseModel] = _ArchiveMemoryInput
        cache_function: Any = _no_cache

        def _run(self, *_args: Any, **kwargs: Any) -> Any:
            memory_id = kwargs.get("memory_id", "")

            async def execute(tools: AgentToolsProtocol) -> str:
                await reporter.report_call(
                    tools, "thenvoi_archive_memory", {"memory_id": memory_id}
                )
                result = await tools.archive_memory(memory_id)
                await reporter.report_result(tools, "thenvoi_archive_memory", result)
                return serialize_success_result(result)

            return _exec("thenvoi_archive_memory", execute)

    base_tools: list[BaseTool] = [
        SendMessageTool(),
        SendEventTool(),
        AddParticipantTool(),
        RemoveParticipantTool(),
        GetParticipantsTool(),
        LookupPeersTool(),
        CreateChatroomTool(),
    ]
    contact_tools: list[BaseTool] = [
        ListContactsTool(),
        AddContactTool(),
        RemoveContactTool(),
        ListContactRequestsTool(),
        RespondContactRequestTool(),
    ]
    memory_tools: list[BaseTool] = [
        ListMemoriesTool(),
        StoreMemoryTool(),
        GetMemoryTool(),
        SupersedeMemoryTool(),
        ArchiveMemoryTool(),
    ]

    return base_tools, contact_tools, memory_tools


def _make_custom_tools(
    *,
    custom_tools: list[CustomToolDef],
    get_context: Callable[[], CrewAIToolContext | None],
    reporter: CrewAIToolReporter,
    fallback_loop: asyncio.AbstractEventLoop | None,
) -> list[BaseTool]:
    """Convert CustomToolDef tuples to CrewAI BaseTool instances."""
    crewai_tools: list[BaseTool] = []

    def _exec(tool_name: str, factory: Callable[[AgentToolsProtocol], Any]) -> str:
        return _execute_tool(
            tool_name=tool_name,
            coro_factory=factory,
            get_context=get_context,
            reporter=reporter,
            fallback_loop=fallback_loop,
        )

    for input_model, func in custom_tools:
        tool_name = get_custom_tool_name(input_model)
        tool_description = input_model.__doc__ or f"Execute {tool_name}"

        def make_tool(
            tool_name_param: str,
            tool_desc_param: str,
            model: type[BaseModel],
            handler: Any,
        ) -> BaseTool:
            _tool_name = tool_name_param
            _tool_desc = tool_desc_param

            class CustomCrewAITool(BaseTool):
                name: str = _tool_name  # type: ignore[misc]
                description: str = _tool_desc  # type: ignore[misc]
                args_schema: Type[BaseModel] = model
                cache_function: Any = staticmethod(lambda *_a, **_kw: False)

                def _run(self, *_args: Any, **kwargs: Any) -> Any:
                    async def execute(_tools: AgentToolsProtocol) -> str:
                        try:
                            validated = model.model_validate(kwargs)
                            await reporter.report_call(_tools, _tool_name, kwargs)

                            if asyncio.iscoroutinefunction(handler):
                                result = await handler(validated)
                            else:
                                result = handler(validated)

                            await reporter.report_result(_tools, _tool_name, result)
                            if isinstance(result, str):
                                return json.dumps(
                                    {"status": "success", "result": result}
                                )
                            return json.dumps(
                                {"status": "success", "result": result}, default=str
                            )
                        except Exception as e:
                            error_msg = str(e)
                            logger.error(
                                "Custom tool %s failed: %s", _tool_name, error_msg
                            )
                            await reporter.report_result(
                                _tools, _tool_name, error_msg, is_error=True
                            )
                            return json.dumps({"status": "error", "message": error_msg})

                    return _exec(_tool_name, execute)

            return CustomCrewAITool()

        crewai_tools.append(make_tool(tool_name, tool_description, input_model, func))

    return crewai_tools


def build_thenvoi_crewai_tools(
    *,
    get_context: Callable[[], CrewAIToolContext | None],
    reporter: CrewAIToolReporter,
    capabilities: frozenset[Capability] = frozenset(),
    custom_tools: list[CustomToolDef] | None = None,
    fallback_loop: asyncio.AbstractEventLoop | None = None,
) -> list[BaseTool]:
    """Build the list of CrewAI BaseTool instances for the platform tool surface.

    Selection:
      - 7 base tools always.
      - +5 contact tools when Capability.CONTACTS is in `capabilities`.
      - +5 memory tools when Capability.MEMORY is in `capabilities`.
      - +N custom tools after platform tools.

    The returned tools close over `get_context`, `reporter`, and `fallback_loop`.
    Each adapter passes its own getter/reporter so the wrappers stay
    framework-agnostic.
    """
    base, contacts, memories = _make_platform_tools(
        get_context=get_context,
        reporter=reporter,
        fallback_loop=fallback_loop,
    )

    selected: list[BaseTool] = list(base)
    if Capability.CONTACTS in capabilities:
        selected.extend(contacts)
    if Capability.MEMORY in capabilities:
        selected.extend(memories)

    if custom_tools:
        selected.extend(
            _make_custom_tools(
                custom_tools=custom_tools,
                get_context=get_context,
                reporter=reporter,
                fallback_loop=fallback_loop,
            )
        )

    return selected


__all__ = [
    "CrewAIToolContext",
    "CrewAIToolReporter",
    "EmitExecutionReporter",
    "NoopReporter",
    "build_thenvoi_crewai_tools",
    "serialize_success_result",
]
