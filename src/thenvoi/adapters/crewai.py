"""CrewAI adapter using SimpleAdapter pattern with official CrewAI SDK.

Important: This module uses nest_asyncio to enable nested event loops, which is
required because CrewAI tools are synchronous but need to call async platform
methods. The nest_asyncio.apply() call is IRREVERSIBLE and affects the entire
Python process - all event loops will allow nesting after this is applied.
The patch is applied lazily on first tool execution, not at import time.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import warnings
from contextvars import ContextVar
from typing import ClassVar, Any, Coroutine, Literal, Type, TypeVar

from pydantic import BaseModel, Field, field_validator

try:
    from crewai import Agent as CrewAIAgent
    from crewai import LLM
    from crewai.tools import BaseTool
    import nest_asyncio
except ImportError as e:
    raise ImportError(
        "crewai is required for CrewAI adapter.\n"
        "Install with: pip install 'thenvoi-sdk[crewai]'\n"
        "Or: uv add crewai nest-asyncio"
    ) from e

from thenvoi.core.exceptions import ThenvoiConfigError
from thenvoi.core.protocols import AgentToolsProtocol
from thenvoi.core.simple_adapter import SimpleAdapter
from thenvoi.core.types import AdapterFeatures, Capability, Emit, PlatformMessage
from thenvoi.converters.crewai import CrewAIHistoryConverter, CrewAIMessages
from thenvoi.runtime.custom_tools import CustomToolDef, get_custom_tool_name
from thenvoi.runtime.tools import get_tool_description

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Module-level state for nest_asyncio patch.
# See module docstring for important notes about global process impact.
_nest_asyncio_applied = False
_nest_asyncio_lock = threading.Lock()

# Context variable for thread-safe room context access.
# Set automatically when processing messages, accessed by tools.
_current_room_context: ContextVar[tuple[str, AgentToolsProtocol] | None] = ContextVar(
    "_current_room_context", default=None
)

MessageType = Literal["thought", "error", "task"]

PLATFORM_INSTRUCTIONS = """## Environment

Multi-participant chat on Thenvoi platform. Messages show sender: [Name]: content.
Use the `thenvoi_send_message` tool to respond. Plain text output is not delivered.

## CRITICAL: Delegate When You Cannot Help Directly

You have NO internet access and NO real-time data. When asked about weather, news, stock prices,
or any current information you cannot answer directly:

1. Call `thenvoi_lookup_peers` to find available specialized agents
2. If a relevant agent exists, call `thenvoi_add_participant` to add them
3. Ask that agent using `thenvoi_send_message` with their handle in mentions
4. Wait for their response and relay it back to the user

NEVER say "I can't do that" without first checking if another agent can help via `thenvoi_lookup_peers`.

## CRITICAL: Do NOT Remove Agents Automatically

After adding an agent to help with a task:
1. Ask your question and wait for their response
2. Relay their response back to the original requester
3. **Do NOT remove the agent** - they stay silent unless mentioned and may be useful for follow-ups

Only remove agents if the user explicitly requests it.

## CRITICAL: Always Relay Information Back to the Requester

When someone asks you to get information from another agent:
1. Ask the other agent for the information
2. When you receive the response, IMMEDIATELY relay it back to the ORIGINAL REQUESTER
3. Do NOT just thank the helper agent - the requester is waiting for their answer!

## IMPORTANT: Always Share Your Thinking

Call `thenvoi_send_event` with message_type="thought" BEFORE every action to share your reasoning."""


def _ensure_nest_asyncio() -> None:
    """Apply nest_asyncio patch lazily on first use.

    This function is thread-safe via a lock to prevent race conditions
    when multiple threads attempt to apply the patch simultaneously.

    See module docstring for important notes about global process impact.
    """
    global _nest_asyncio_applied
    if _nest_asyncio_applied:
        return

    with _nest_asyncio_lock:
        # Double-check after acquiring lock (double-checked locking pattern)
        if not _nest_asyncio_applied:
            nest_asyncio.apply()
            _nest_asyncio_applied = True
            logger.debug("Applied nest_asyncio patch for nested event loops")


def _run_async(
    coro: Coroutine[Any, Any, T],
    fallback_loop: asyncio.AbstractEventLoop | None = None,
) -> T:
    """Run an async coroutine from sync context.

    CrewAI tools are synchronous but need to call async platform methods.
    With nest_asyncio applied, we can safely run coroutines even when
    an event loop is already running.

    This function handles two scenarios:
    1. An event loop is running - uses run_until_complete with nest_asyncio
    2. No event loop is running - uses asyncio.run to create one
    """
    _ensure_nest_asyncio()

    try:
        loop = asyncio.get_running_loop()
        logger.debug("Running coroutine in existing event loop via nest_asyncio")
    except RuntimeError:
        # No running event loop - prefer the adapter's main loop when available.
        # CrewAI may execute tools in worker threads, and platform clients are
        # bound to the runtime loop created during agent startup.
        if fallback_loop is not None and fallback_loop.is_running():
            logger.debug(
                "Running coroutine on fallback event loop via thread-safe submit"
            )
            future = asyncio.run_coroutine_threadsafe(coro, fallback_loop)
            return future.result(timeout=60)

        # No running event loop and no active fallback loop - use asyncio.run
        logger.debug("Running coroutine in new event loop via asyncio.run")
        return asyncio.run(coro)

    # Event loop is running - use run_until_complete (safe with nest_asyncio)
    return loop.run_until_complete(coro)


class CrewAIAdapter(SimpleAdapter[CrewAIMessages]):
    """CrewAI adapter using the official CrewAI SDK.

    Integrates the CrewAI framework (https://docs.crewai.com/) with Thenvoi
    platform for building collaborative multi-agent systems.

    Example:
        adapter = CrewAIAdapter(
            model="gpt-4o",
            role="Research Assistant",
            goal="Help users find and analyze information",
            backstory="Expert researcher with deep knowledge across domains",
        )
        agent = Agent.create(adapter=adapter, agent_id="...", api_key="...")
        await agent.run()

    Note:
        API keys are configured through environment variables as expected by
        the CrewAI LLM class (e.g., OPENAI_API_KEY, ANTHROPIC_API_KEY).
    """

    SUPPORTED_EMIT: ClassVar[frozenset[Emit]] = frozenset({Emit.EXECUTION})
    SUPPORTED_CAPABILITIES: ClassVar[frozenset[Capability]] = frozenset(
        {Capability.MEMORY, Capability.CONTACTS}
    )

    def __init__(
        self,
        model: str = "gpt-4o",
        role: str | None = None,
        goal: str | None = None,
        backstory: str | None = None,
        custom_section: str | None = None,
        enable_execution_reporting: bool = False,
        enable_memory_tools: bool = False,
        verbose: bool = False,
        max_iter: int = 20,
        max_rpm: int | None = None,
        allow_delegation: bool = False,
        history_converter: CrewAIHistoryConverter | None = None,
        additional_tools: list[CustomToolDef] | None = None,
        system_prompt: str | None = None,  # Deprecated
        features: AdapterFeatures | None = None,
    ):
        """Initialize the CrewAI adapter.

        Args:
            model: Model name (e.g., "gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet").
                   API keys are read from environment variables by CrewAI's LLM class.
            role: Agent's role in the crew (e.g., "Research Assistant")
            goal: Agent's primary goal or objective
            backstory: Agent's background and expertise description
            custom_section: Custom instructions added to the agent's backstory
            enable_execution_reporting: If True, sends tool_call/tool_result events
            verbose: If True, enables detailed logging from CrewAI
            max_iter: Maximum iterations for the agent (default: 20)
            max_rpm: Maximum requests per minute (rate limiting)
            allow_delegation: Whether to allow task delegation to other agents
            history_converter: Custom history converter (optional)
            additional_tools: List of custom tools as (InputModel, callable) tuples.
                Each InputModel is a Pydantic model defining the tool's input schema,
                and the callable is the function to execute (sync or async).
            system_prompt: Deprecated. Use 'backstory' instead for prompt customization.
        """
        if system_prompt is not None:
            warnings.warn(
                "The 'system_prompt' parameter is deprecated and will be removed in a "
                "future version. Use 'backstory' parameter instead for prompt "
                "customization. The CrewAI SDK uses role/goal/backstory pattern.",
                DeprecationWarning,
                stacklevel=2,
            )
            # If backstory not provided, use system_prompt as backstory for compatibility
            if backstory is None:
                backstory = system_prompt

        # --- Deprecation shim: boolean → features migration ---
        _has_legacy_booleans = enable_execution_reporting or enable_memory_tools
        if _has_legacy_booleans and features is not None:
            raise ThenvoiConfigError(
                "Cannot pass both legacy boolean flags "
                "(enable_execution_reporting / enable_memory_tools) and 'features'. "
                "Use features=AdapterFeatures(...) instead."
            )

        if _has_legacy_booleans:
            warnings.warn(
                "enable_execution_reporting and enable_memory_tools are deprecated. "
                "Use features=AdapterFeatures(emit={Emit.EXECUTION}, "
                "capabilities={Capability.MEMORY}) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            # NOTE: unlike ClaudeSDK, CrewAI's legacy enable_execution_reporting
            # maps to {Emit.EXECUTION} only (no THOUGHTS). CrewAI had no native
            # thought emission under this flag, so migrating to THOUGHTS would
            # turn on a new behavior, not preserve existing behavior.
            features = AdapterFeatures(
                emit=frozenset({Emit.EXECUTION})
                if enable_execution_reporting
                else frozenset(),
                capabilities=frozenset({Capability.MEMORY})
                if enable_memory_tools
                else frozenset(),
            )

        super().__init__(
            history_converter=history_converter or CrewAIHistoryConverter(),
            features=features,
        )

        self.model = model
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.custom_section = custom_section
        self.verbose = verbose
        self.max_iter = max_iter
        self.max_rpm = max_rpm
        self.allow_delegation = allow_delegation

        self._crewai_agent: CrewAIAgent | None = None
        self._message_history: dict[str, list[dict[str, Any]]] = {}
        self._custom_tools: list[CustomToolDef] = additional_tools or []
        self._tool_loop: asyncio.AbstractEventLoop | None = None

    async def on_started(self, agent_name: str, agent_description: str) -> None:
        """Initialize CrewAI agent after metadata is fetched."""
        await super().on_started(agent_name, agent_description)
        self._tool_loop = asyncio.get_running_loop()

        role = self.role or agent_name
        goal = self.goal or agent_description or "Help users accomplish their tasks"

        backstory_parts = []
        if self.backstory:
            backstory_parts.append(self.backstory)
        else:
            backstory_parts.append(
                f"You are {agent_name}, a collaborative AI agent on the Thenvoi platform."
            )

        if self.custom_section:
            backstory_parts.append(self.custom_section)

        backstory_parts.append(PLATFORM_INSTRUCTIONS)
        backstory = "\n\n".join(backstory_parts)

        tools = self._create_crewai_tools()

        self._crewai_agent = CrewAIAgent(
            role=role,
            goal=goal,
            backstory=backstory,
            llm=LLM(model=self.model),
            tools=tools,
            verbose=self.verbose,
            max_iter=self.max_iter,
            max_rpm=self.max_rpm,
            allow_delegation=self.allow_delegation,
        )

        logger.info(
            "CrewAI adapter started for agent: %s (model=%s, role=%s)",
            agent_name,
            self.model,
            role,
        )

    def _get_current_room_context(self) -> tuple[str, AgentToolsProtocol] | None:
        """Get current room context from context variable.

        Returns:
            Tuple of (room_id, tools) if context is set, None otherwise.
        """
        return _current_room_context.get()

    def _execute_tool(
        self,
        tool_name: str,
        coro_factory: Any,
    ) -> str:
        """Execute a tool with common error handling and reporting.

        Args:
            tool_name: Name of the tool for error messages
            coro_factory: Callable that takes tools and returns a coroutine

        Returns:
            JSON string with status and result/error
        """
        context = self._get_current_room_context()
        if not context:
            return json.dumps(
                {
                    "status": "error",
                    "message": "No room context available - tool called outside message handling",
                }
            )

        room_id, tools = context
        adapter = self

        async def _execute() -> str:
            try:
                return await coro_factory(tools)
            except Exception as e:
                error_msg = str(e)
                logger.error("%s failed in room %s: %s", tool_name, room_id, error_msg)
                # Report error if execution reporting is enabled
                await adapter._report_tool_result(
                    tools, tool_name, error_msg, is_error=True
                )
                return json.dumps({"status": "error", "message": error_msg})

        return _run_async(_execute(), fallback_loop=self._tool_loop)

    async def _report_tool_call(
        self,
        tools: AgentToolsProtocol,
        tool_name: str,
        input_data: dict[str, Any],
    ) -> None:
        """Report tool call event if execution reporting is enabled.

        Best-effort: event reporting must never crash tool execution.
        """
        if Emit.EXECUTION in self.features.emit:
            try:
                await tools.send_event(
                    content=json.dumps({"tool": tool_name, "input": input_data}),
                    message_type="tool_call",
                )
            except Exception as e:
                logger.warning(
                    "Failed to send tool_call event: %s",
                    e,
                )

    async def _report_tool_result(
        self,
        tools: AgentToolsProtocol,
        tool_name: str,
        result: Any,
        is_error: bool = False,
    ) -> None:
        """Report tool result event if execution reporting is enabled.

        Best-effort: event reporting must never crash tool execution.
        """
        if Emit.EXECUTION in self.features.emit:
            try:
                key = "error" if is_error else "result"
                await tools.send_event(
                    content=json.dumps({"tool": tool_name, key: result}),
                    message_type="tool_result",
                )
            except Exception as e:
                logger.warning(
                    "Failed to send tool_result event: %s",
                    e,
                )

    @staticmethod
    def _serialize_success_result(result: Any) -> str:
        """Serialize a successful tool result without losing domain status fields."""
        # Convert Pydantic models to dicts at serialization boundary
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

    def _convert_custom_tools_to_crewai(self) -> list[BaseTool]:
        """Convert CustomToolDef tuples to CrewAI BaseTool instances.

        Each custom tool is wrapped in a dynamically created BaseTool subclass
        that handles sync/async bridging and error handling.

        Returns:
            List of CrewAI BaseTool instances.
        """
        adapter = self
        crewai_tools: list[BaseTool] = []

        for input_model, func in self._custom_tools:
            tool_name = get_custom_tool_name(input_model)
            tool_description = input_model.__doc__ or f"Execute {tool_name}"

            # Create a closure to capture the current input_model and func
            def make_tool(
                tool_name_param: str,
                tool_desc_param: str,
                model: type[BaseModel],
                handler: Any,
            ) -> BaseTool:
                # Capture values in local variables for the closure
                _tool_name = tool_name_param
                _tool_desc = tool_desc_param

                class CustomCrewAITool(BaseTool):
                    name: str = _tool_name  # type: ignore[misc]
                    description: str = _tool_desc  # type: ignore[misc]
                    args_schema: Type[BaseModel] = model

                    def _run(self, *_args: Any, **kwargs: Any) -> Any:
                        async def execute(_tools: AgentToolsProtocol) -> str:
                            try:
                                # Validate input using Pydantic model
                                validated = model.model_validate(kwargs)

                                # Report tool call if enabled
                                await adapter._report_tool_call(
                                    _tools, _tool_name, kwargs
                                )

                                # Execute the handler (sync or async)
                                if asyncio.iscoroutinefunction(handler):
                                    result = await handler(validated)
                                else:
                                    result = handler(validated)

                                # Report tool result if enabled
                                await adapter._report_tool_result(
                                    _tools, _tool_name, result
                                )

                                # Return JSON-serialized result
                                if isinstance(result, str):
                                    return json.dumps(
                                        {"status": "success", "result": result}
                                    )
                                return json.dumps(
                                    {"status": "success", "result": result},
                                    default=str,
                                )
                            except Exception as e:
                                error_msg = str(e)
                                logger.error(
                                    "Custom tool %s failed: %s",
                                    _tool_name,
                                    error_msg,
                                )
                                await adapter._report_tool_result(
                                    _tools, _tool_name, error_msg, is_error=True
                                )
                                return json.dumps(
                                    {"status": "error", "message": error_msg}
                                )

                        return adapter._execute_tool(_tool_name, execute)

                return CustomCrewAITool()

            crewai_tools.append(
                make_tool(tool_name, tool_description, input_model, func)
            )

        return crewai_tools

    def _create_crewai_tools(self) -> list[BaseTool]:
        """Create CrewAI-compatible tools for Thenvoi platform.

        Tools access the current room context via a context variable that is set
        automatically when processing messages. This removes the need for the LLM
        to pass room_id explicitly, making tool calls more reliable.
        """
        adapter = self

        class SendMessageInput(BaseModel):
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

        class SendEventInput(BaseModel):
            content: str = Field(..., description="Human-readable event content")
            message_type: MessageType = Field(
                default="thought",
                description="Type of event: 'thought', 'error', or 'task'",
            )

        class AddParticipantInput(BaseModel):
            participant_name: str = Field(
                ...,
                description="Name of participant to add (must match from thenvoi_lookup_peers)",
            )
            role: str = Field(
                default="member", description="Role: 'owner', 'admin', or 'member'"
            )

        class RemoveParticipantInput(BaseModel):
            participant_name: str = Field(
                ..., description="Name of the participant to remove"
            )

        class GetParticipantsInput(BaseModel):
            pass  # No parameters needed - room context from context variable

        class LookupPeersInput(BaseModel):
            # No user-facing parameters - pagination is handled internally with defaults.
            # This matches the Parlant adapter's approach for simplicity.
            pass

        class CreateChatroomInput(BaseModel):
            task_id: str | None = Field(
                default=None, description="Associated task ID (optional)"
            )

        # Contact management input models
        class ListContactsInput(BaseModel):
            page: int = Field(default=1, description="Page number")
            page_size: int = Field(default=50, description="Items per page (max 100)")

        class AddContactInput(BaseModel):
            handle: str = Field(
                ...,
                description="Handle of user/agent to add (e.g., '@john' or '@john/agent-name')",
            )
            message: str | None = Field(
                default=None, description="Optional message with the request"
            )

        class RemoveContactInput(BaseModel):
            handle: str | None = Field(default=None, description="Contact's handle")
            contact_id: str | None = Field(
                default=None, description="Or contact record ID (UUID)"
            )

        class ListContactRequestsInput(BaseModel):
            page: int = Field(default=1, description="Page number")
            page_size: int = Field(default=50, description="Items per page (max 100)")
            sent_status: str = Field(
                default="pending", description="Filter sent requests by status"
            )

        class RespondContactRequestInput(BaseModel):
            action: str = Field(
                ..., description="Action to take ('approve', 'reject', 'cancel')"
            )
            handle: str | None = Field(default=None, description="Other party's handle")
            request_id: str | None = Field(
                default=None, description="Or request ID (UUID)"
            )

        # Memory management input models
        class ListMemoriesInput(BaseModel):
            subject_id: str | None = Field(
                default=None, description="Filter by subject UUID"
            )
            scope: str | None = Field(
                default=None, description="Filter by scope (subject, organization, all)"
            )
            system: str | None = Field(
                default=None,
                description="Filter by memory system (sensory, working, long_term)",
            )
            memory_type: str | None = Field(
                default=None, description="Filter by memory type"
            )
            segment: str | None = Field(
                default=None,
                description="Filter by segment (user, agent, tool, guideline)",
            )
            content_query: str | None = Field(
                default=None, description="Full-text search query"
            )
            page_size: int = Field(default=50, description="Number of results per page")
            status: str | None = Field(
                default=None,
                description="Filter by status (active, superseded, archived, all)",
            )

        class StoreMemoryInput(BaseModel):
            content: str = Field(..., description="The memory content")
            system: str = Field(..., description="Memory system tier")
            memory_type: str = Field(..., description="Memory type")
            segment: str = Field(..., description="Logical segment")
            thought: str = Field(
                ..., description="Agent's reasoning for storing this memory"
            )
            scope: str = Field(default="subject", description="Visibility scope")
            subject_id: str | None = Field(
                default=None,
                description="UUID of the subject (required for subject scope)",
            )

        class GetMemoryInput(BaseModel):
            memory_id: str = Field(..., description="Memory ID (UUID)")

        class SupersedeMemoryInput(BaseModel):
            memory_id: str = Field(..., description="Memory ID (UUID)")

        class ArchiveMemoryInput(BaseModel):
            memory_id: str = Field(..., description="Memory ID (UUID)")

        class SendMessageTool(BaseTool):
            name: str = "thenvoi_send_message"
            description: str = get_tool_description("thenvoi_send_message")
            args_schema: Type[BaseModel] = SendMessageInput

            # *_args is required by BaseTool's _run signature even though we don't use it
            def _run(self, *_args: Any, **kwargs: Any) -> Any:
                content: str = kwargs.get("content", "")
                mentions: str = kwargs.get("mentions", "[]")

                try:
                    mention_list = json.loads(mentions) if mentions else []
                except json.JSONDecodeError:
                    mention_list = []

                async def execute(tools: AgentToolsProtocol) -> str:
                    await adapter._report_tool_call(
                        tools,
                        "thenvoi_send_message",
                        {"content": content, "mentions": mention_list},
                    )
                    await tools.send_message(content, mention_list)
                    await adapter._report_tool_result(
                        tools, "thenvoi_send_message", "success"
                    )
                    return json.dumps({"status": "success", "message": "Message sent"})

                return adapter._execute_tool("thenvoi_send_message", execute)

        class SendEventTool(BaseTool):
            name: str = "thenvoi_send_event"
            description: str = get_tool_description("thenvoi_send_event")
            args_schema: Type[BaseModel] = SendEventInput

            def _run(self, *_args: Any, **kwargs: Any) -> Any:
                content: str = kwargs.get("content", "")
                message_type: str = kwargs.get("message_type", "thought")

                async def execute(tools: AgentToolsProtocol) -> str:
                    # Note: No execution reporting for send_event to avoid
                    # meta-events (reporting an event about sending an event)
                    await tools.send_event(content, message_type)
                    return json.dumps({"status": "success", "message": "Event sent"})

                return adapter._execute_tool("thenvoi_send_event", execute)

        class AddParticipantTool(BaseTool):
            name: str = "thenvoi_add_participant"
            description: str = get_tool_description("thenvoi_add_participant")
            args_schema: Type[BaseModel] = AddParticipantInput

            def _run(self, *_args: Any, **kwargs: Any) -> Any:
                participant_name: str = kwargs.get("participant_name", "")
                role: str = kwargs.get("role", "member")

                async def execute(tools: AgentToolsProtocol) -> str:
                    await adapter._report_tool_call(
                        tools,
                        "thenvoi_add_participant",
                        {"name": participant_name, "role": role},
                    )
                    result = await tools.add_participant(participant_name, role)
                    await adapter._report_tool_result(
                        tools, "thenvoi_add_participant", result
                    )
                    return adapter._serialize_success_result(result)

                return adapter._execute_tool("thenvoi_add_participant", execute)

        class RemoveParticipantTool(BaseTool):
            name: str = "thenvoi_remove_participant"
            description: str = get_tool_description("thenvoi_remove_participant")
            args_schema: Type[BaseModel] = RemoveParticipantInput

            def _run(self, *_args: Any, **kwargs: Any) -> Any:
                participant_name: str = kwargs.get("participant_name", "")

                async def execute(tools: AgentToolsProtocol) -> str:
                    await adapter._report_tool_call(
                        tools,
                        "thenvoi_remove_participant",
                        {"name": participant_name},
                    )
                    result = await tools.remove_participant(participant_name)
                    await adapter._report_tool_result(
                        tools, "thenvoi_remove_participant", result
                    )
                    return adapter._serialize_success_result(result)

                return adapter._execute_tool("thenvoi_remove_participant", execute)

        class GetParticipantsTool(BaseTool):
            name: str = "thenvoi_get_participants"
            description: str = get_tool_description("thenvoi_get_participants")
            args_schema: Type[BaseModel] = GetParticipantsInput

            def _run(self, *_args: Any, **_kwargs: Any) -> Any:
                async def execute(tools: AgentToolsProtocol) -> str:
                    await adapter._report_tool_call(
                        tools, "thenvoi_get_participants", {}
                    )
                    participants = await tools.get_participants()
                    # Convert Fern models to dicts for JSON serialization
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
                        "count": len(participants),
                    }
                    await adapter._report_tool_result(
                        tools, "thenvoi_get_participants", result
                    )
                    return json.dumps(result, default=str)

                return adapter._execute_tool("thenvoi_get_participants", execute)

        class LookupPeersTool(BaseTool):
            name: str = "thenvoi_lookup_peers"
            description: str = get_tool_description("thenvoi_lookup_peers")
            args_schema: Type[BaseModel] = LookupPeersInput

            # *_args is required by BaseTool's _run signature even though we don't use it
            def _run(self, *_args: Any, **_kwargs: Any) -> Any:
                # Use hardcoded pagination defaults for simplicity
                page, page_size = 1, 50

                async def execute(tools: AgentToolsProtocol) -> str:
                    await adapter._report_tool_call(
                        tools,
                        "thenvoi_lookup_peers",
                        {"page": page, "page_size": page_size},
                    )
                    result = await tools.lookup_peers(page, page_size)
                    await adapter._report_tool_result(
                        tools, "thenvoi_lookup_peers", result
                    )
                    return adapter._serialize_success_result(result)

                return adapter._execute_tool("thenvoi_lookup_peers", execute)

        class CreateChatroomTool(BaseTool):
            name: str = "thenvoi_create_chatroom"
            description: str = get_tool_description("thenvoi_create_chatroom")
            args_schema: Type[BaseModel] = CreateChatroomInput

            def _run(self, *_args: Any, **kwargs: Any) -> Any:
                task_id: str | None = kwargs.get("task_id")

                async def execute(tools: AgentToolsProtocol) -> str:
                    await adapter._report_tool_call(
                        tools, "thenvoi_create_chatroom", {"task_id": task_id}
                    )
                    new_room_id = await tools.create_chatroom(task_id)
                    result = {
                        "status": "success",
                        "message": "Chat room created",
                        "room_id": new_room_id,
                    }
                    await adapter._report_tool_result(
                        tools, "thenvoi_create_chatroom", result
                    )
                    return json.dumps(result)

                return adapter._execute_tool("thenvoi_create_chatroom", execute)

        # Contact management tools
        class ListContactsTool(BaseTool):
            name: str = "thenvoi_list_contacts"
            description: str = get_tool_description("thenvoi_list_contacts")
            args_schema: Type[BaseModel] = ListContactsInput

            def _run(self, *_args: Any, **kwargs: Any) -> Any:
                page: int = kwargs.get("page", 1)
                page_size: int = kwargs.get("page_size", 50)

                async def execute(tools: AgentToolsProtocol) -> str:
                    await adapter._report_tool_call(
                        tools,
                        "thenvoi_list_contacts",
                        {"page": page, "page_size": page_size},
                    )
                    result = await tools.list_contacts(page, page_size)
                    await adapter._report_tool_result(
                        tools, "thenvoi_list_contacts", result
                    )
                    return adapter._serialize_success_result(result)

                return adapter._execute_tool("thenvoi_list_contacts", execute)

        class AddContactTool(BaseTool):
            name: str = "thenvoi_add_contact"
            description: str = get_tool_description("thenvoi_add_contact")
            args_schema: Type[BaseModel] = AddContactInput

            def _run(self, *_args: Any, **kwargs: Any) -> Any:
                handle: str = kwargs.get("handle", "")
                message: str | None = kwargs.get("message")

                async def execute(tools: AgentToolsProtocol) -> str:
                    await adapter._report_tool_call(
                        tools,
                        "thenvoi_add_contact",
                        {"handle": handle, "message": message},
                    )
                    result = await tools.add_contact(handle, message)
                    await adapter._report_tool_result(
                        tools, "thenvoi_add_contact", result
                    )
                    return adapter._serialize_success_result(result)

                return adapter._execute_tool("thenvoi_add_contact", execute)

        class RemoveContactTool(BaseTool):
            name: str = "thenvoi_remove_contact"
            description: str = get_tool_description("thenvoi_remove_contact")
            args_schema: Type[BaseModel] = RemoveContactInput

            def _run(self, *_args: Any, **kwargs: Any) -> Any:
                handle: str | None = kwargs.get("handle")
                contact_id: str | None = kwargs.get("contact_id")

                async def execute(tools: AgentToolsProtocol) -> str:
                    await adapter._report_tool_call(
                        tools,
                        "thenvoi_remove_contact",
                        {"handle": handle, "contact_id": contact_id},
                    )
                    result = await tools.remove_contact(handle, contact_id)
                    await adapter._report_tool_result(
                        tools, "thenvoi_remove_contact", result
                    )
                    return adapter._serialize_success_result(result)

                return adapter._execute_tool("thenvoi_remove_contact", execute)

        class ListContactRequestsTool(BaseTool):
            name: str = "thenvoi_list_contact_requests"
            description: str = get_tool_description("thenvoi_list_contact_requests")
            args_schema: Type[BaseModel] = ListContactRequestsInput

            def _run(self, *_args: Any, **kwargs: Any) -> Any:
                page: int = kwargs.get("page", 1)
                page_size: int = kwargs.get("page_size", 50)
                sent_status: str = kwargs.get("sent_status", "pending")

                async def execute(tools: AgentToolsProtocol) -> str:
                    await adapter._report_tool_call(
                        tools,
                        "thenvoi_list_contact_requests",
                        {
                            "page": page,
                            "page_size": page_size,
                            "sent_status": sent_status,
                        },
                    )
                    result = await tools.list_contact_requests(
                        page, page_size, sent_status
                    )
                    await adapter._report_tool_result(
                        tools, "thenvoi_list_contact_requests", result
                    )
                    return adapter._serialize_success_result(result)

                return adapter._execute_tool("thenvoi_list_contact_requests", execute)

        class RespondContactRequestTool(BaseTool):
            name: str = "thenvoi_respond_contact_request"
            description: str = get_tool_description("thenvoi_respond_contact_request")
            args_schema: Type[BaseModel] = RespondContactRequestInput

            def _run(self, *_args: Any, **kwargs: Any) -> Any:
                action: str = kwargs.get("action", "")
                handle: str | None = kwargs.get("handle")
                request_id: str | None = kwargs.get("request_id")

                async def execute(tools: AgentToolsProtocol) -> str:
                    await adapter._report_tool_call(
                        tools,
                        "thenvoi_respond_contact_request",
                        {"action": action, "handle": handle, "request_id": request_id},
                    )
                    result = await tools.respond_contact_request(
                        action, handle, request_id
                    )
                    await adapter._report_tool_result(
                        tools, "thenvoi_respond_contact_request", result
                    )
                    return adapter._serialize_success_result(result)

                return adapter._execute_tool("thenvoi_respond_contact_request", execute)

        # Memory management tools
        class ListMemoriesTool(BaseTool):
            name: str = "thenvoi_list_memories"
            description: str = get_tool_description("thenvoi_list_memories")
            args_schema: Type[BaseModel] = ListMemoriesInput

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
                    await adapter._report_tool_call(
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
                    await adapter._report_tool_result(
                        tools, "thenvoi_list_memories", result
                    )
                    return adapter._serialize_success_result(result)

                return adapter._execute_tool("thenvoi_list_memories", execute)

        class StoreMemoryTool(BaseTool):
            name: str = "thenvoi_store_memory"
            description: str = get_tool_description("thenvoi_store_memory")
            args_schema: Type[BaseModel] = StoreMemoryInput

            def _run(self, *_args: Any, **kwargs: Any) -> Any:
                content = kwargs.get("content", "")
                system = kwargs.get("system", "")
                memory_type = kwargs.get("memory_type", "")
                segment = kwargs.get("segment", "")
                thought = kwargs.get("thought", "")
                scope = kwargs.get("scope", "subject")
                subject_id = kwargs.get("subject_id")

                async def execute(tools: AgentToolsProtocol) -> str:
                    await adapter._report_tool_call(
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
                    await adapter._report_tool_result(
                        tools, "thenvoi_store_memory", result
                    )
                    return adapter._serialize_success_result(result)

                return adapter._execute_tool("thenvoi_store_memory", execute)

        class GetMemoryTool(BaseTool):
            name: str = "thenvoi_get_memory"
            description: str = get_tool_description("thenvoi_get_memory")
            args_schema: Type[BaseModel] = GetMemoryInput

            def _run(self, *_args: Any, **kwargs: Any) -> Any:
                memory_id = kwargs.get("memory_id", "")

                async def execute(tools: AgentToolsProtocol) -> str:
                    await adapter._report_tool_call(
                        tools, "thenvoi_get_memory", {"memory_id": memory_id}
                    )
                    result = await tools.get_memory(memory_id)
                    await adapter._report_tool_result(
                        tools, "thenvoi_get_memory", result
                    )
                    return adapter._serialize_success_result(result)

                return adapter._execute_tool("thenvoi_get_memory", execute)

        class SupersedeMemoryTool(BaseTool):
            name: str = "thenvoi_supersede_memory"
            description: str = get_tool_description("thenvoi_supersede_memory")
            args_schema: Type[BaseModel] = SupersedeMemoryInput

            def _run(self, *_args: Any, **kwargs: Any) -> Any:
                memory_id = kwargs.get("memory_id", "")

                async def execute(tools: AgentToolsProtocol) -> str:
                    await adapter._report_tool_call(
                        tools, "thenvoi_supersede_memory", {"memory_id": memory_id}
                    )
                    result = await tools.supersede_memory(memory_id)
                    await adapter._report_tool_result(
                        tools, "thenvoi_supersede_memory", result
                    )
                    return adapter._serialize_success_result(result)

                return adapter._execute_tool("thenvoi_supersede_memory", execute)

        class ArchiveMemoryTool(BaseTool):
            name: str = "thenvoi_archive_memory"
            description: str = get_tool_description("thenvoi_archive_memory")
            args_schema: Type[BaseModel] = ArchiveMemoryInput

            def _run(self, *_args: Any, **kwargs: Any) -> Any:
                memory_id = kwargs.get("memory_id", "")

                async def execute(tools: AgentToolsProtocol) -> str:
                    await adapter._report_tool_call(
                        tools, "thenvoi_archive_memory", {"memory_id": memory_id}
                    )
                    result = await tools.archive_memory(memory_id)
                    await adapter._report_tool_result(
                        tools, "thenvoi_archive_memory", result
                    )
                    return adapter._serialize_success_result(result)

                return adapter._execute_tool("thenvoi_archive_memory", execute)

        platform_tools: list[BaseTool] = [
            SendMessageTool(),
            SendEventTool(),
            AddParticipantTool(),
            RemoveParticipantTool(),
            GetParticipantsTool(),
            LookupPeersTool(),
            CreateChatroomTool(),
            # Contact management tools
            ListContactsTool(),
            AddContactTool(),
            RemoveContactTool(),
            ListContactRequestsTool(),
            RespondContactRequestTool(),
        ]

        # Memory management tools (enterprise only - opt-in)
        if Capability.MEMORY in self.features.capabilities:
            platform_tools.extend(
                [
                    ListMemoriesTool(),
                    StoreMemoryTool(),
                    GetMemoryTool(),
                    SupersedeMemoryTool(),
                    ArchiveMemoryTool(),
                ]
            )

        # Add custom tools converted to CrewAI format
        custom_tools = self._convert_custom_tools_to_crewai()
        if custom_tools:
            logger.debug(
                "Added %s custom tools: %s",
                len(custom_tools),
                [t.name for t in custom_tools],
            )

        return platform_tools + custom_tools

    async def on_message(
        self,
        msg: PlatformMessage,
        tools: AgentToolsProtocol,
        history: CrewAIMessages,
        participants_msg: str | None,
        contacts_msg: str | None,
        *,
        is_session_bootstrap: bool,
        room_id: str,
    ) -> None:
        """Handle incoming message using CrewAI agent."""
        logger.debug("Handling message %s in room %s", msg.id, room_id)

        if not self._crewai_agent:
            raise RuntimeError(
                "CrewAI agent not initialized - ensure on_started() was called"
            )

        # Set context variable for tool access (thread-safe room context).
        # Wrap in try/finally immediately to ensure cleanup even if code
        # before the main try block raises an exception.
        _current_room_context.set((room_id, tools))
        try:
            await self._process_message(
                msg=msg,
                tools=tools,
                history=history,
                participants_msg=participants_msg,
                contacts_msg=contacts_msg,
                is_session_bootstrap=is_session_bootstrap,
                room_id=room_id,
            )
        finally:
            # Clear context after processing to prevent stale context in async
            # environments with task reuse
            _current_room_context.set(None)

    async def _process_message(
        self,
        msg: PlatformMessage,
        tools: AgentToolsProtocol,
        history: CrewAIMessages,
        participants_msg: str | None,
        contacts_msg: str | None,
        *,
        is_session_bootstrap: bool,
        room_id: str,
    ) -> None:
        """Internal message processing logic."""
        if is_session_bootstrap:
            if history:
                self._message_history[room_id] = [
                    {"role": h["role"], "content": h["content"]} for h in history
                ]
                logger.info(
                    "Room %s: Loaded %s historical messages",
                    room_id,
                    len(history),
                )
            else:
                self._message_history[room_id] = []
                logger.info("Room %s: No historical messages found", room_id)
        elif room_id not in self._message_history:
            self._message_history[room_id] = []

        messages = []

        if is_session_bootstrap and self._message_history.get(room_id):
            history_text = "\n".join(
                f"{m['role']}: {m['content']}" for m in self._message_history[room_id]
            )
            messages.append(
                {
                    "role": "user",
                    "content": f"[Previous conversation:]\n{history_text}",
                }
            )

        if participants_msg:
            messages.append(
                {
                    "role": "user",
                    "content": f"[System]: {participants_msg}",
                }
            )
            logger.info("Room %s: Participants updated", room_id)

        if contacts_msg:
            messages.append(
                {
                    "role": "user",
                    "content": f"[System]: {contacts_msg}",
                }
            )
            logger.info("Room %s: Contacts broadcast received", room_id)

        user_message = msg.format_for_llm()
        messages.append({"role": "user", "content": user_message})

        self._message_history[room_id].append(
            {
                "role": "user",
                "content": user_message,
            }
        )

        total_messages = len(self._message_history[room_id])
        logger.info(
            "Room %s: Processing with %s messages (first_msg=%s)",
            room_id,
            total_messages,
            is_session_bootstrap,
        )

        try:
            # Type ignore explanation: CrewAI's kickoff_async is typed to accept
            # only a string prompt, but the implementation also accepts a list of
            # message dicts (similar to OpenAI's messages format) for multi-turn
            # context. This is documented behavior but the type stubs haven't been
            # updated. See: https://docs.crewai.com/concepts/agents
            result = await self._crewai_agent.kickoff_async(messages)  # type: ignore[arg-type]

            if result and result.raw:
                self._message_history[room_id].append(
                    {
                        "role": "assistant",
                        "content": result.raw,
                    }
                )

            logger.info(
                "Room %s: CrewAI agent completed (output_length=%s)",
                room_id,
                len(result.raw) if result and result.raw else 0,
            )

        except Exception as e:
            logger.error("Error processing message: %s", e, exc_info=True)
            await self._report_error(tools, str(e))
            raise

        logger.debug(
            "Message %s processed successfully (history now has %s messages)",
            msg.id,
            len(self._message_history[room_id]),
        )

    async def on_cleanup(self, room_id: str) -> None:
        """Clean up message history when agent leaves a room."""
        if room_id in self._message_history:
            del self._message_history[room_id]
            logger.debug("Room %s: Cleaned up CrewAI session", room_id)

    async def _report_error(self, tools: AgentToolsProtocol, error: str) -> None:
        """Send error event (best effort)."""
        try:
            await tools.send_event(content=f"Error: {error}", message_type="error")
        except Exception as e:
            logger.warning("Failed to send error event: %s", e)
