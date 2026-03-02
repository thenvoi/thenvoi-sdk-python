"""CrewAI adapter using SimpleAdapter pattern with official CrewAI SDK.

Important: This module uses nest_asyncio to enable nested event loops, which is
required because CrewAI tools are synchronous but need to call async platform
methods. The nest_asyncio.apply() call is IRREVERSIBLE and affects the entire
Python process - all event loops will allow nesting after this is applied.
The patch is applied lazily on first tool execution, not at import time.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import warnings
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Coroutine, TypeVar

from pydantic import BaseModel
from thenvoi.adapters.optional_dependencies import ensure_optional_dependency

try:
    from crewai import Agent as CrewAIAgent
    from crewai import LLM
    from crewai.tools import BaseTool
    import nest_asyncio
except ImportError as e:
    _CREWAI_IMPORT_ERROR = e
    CrewAIAgent = Any
    LLM = Any
    BaseTool = Any
    nest_asyncio = Any
else:
    _CREWAI_IMPORT_ERROR = None

from thenvoi.core.protocols import MessagingDispatchToolsProtocol
from thenvoi.adapters.crewai_processing import build_backstory, process_message
from thenvoi.adapters.crewai_prompts import PLATFORM_INSTRUCTIONS
from thenvoi.adapters.crewai_schemas import (
    CREWAI_SCHEMA_OVERRIDES,
    CrewAISendEventInput,
    CrewAISendMessageInput,
)
from thenvoi.adapters.crewai_tooling import CrewAIToolRuntime
from thenvoi.core.simple_adapter import SimpleAdapter, legacy_chat_turn_compat
from thenvoi.core.types import ChatMessageTurnContext, PlatformMessage
from thenvoi.core.nonfatal import NonFatalErrorRecorder
from thenvoi.converters.crewai import CrewAIHistoryConverter, CrewAIMessages
from thenvoi.runtime.tooling.custom_tools import CustomToolDef

logger = logging.getLogger(__name__)

T = TypeVar("T")
_CREWAI_INSTALL_COMMANDS = (
    "pip install 'thenvoi-sdk[crewai]'",
    "uv add crewai nest-asyncio",
)

__all__ = [
    "CREWAI_SCHEMA_OVERRIDES",
    "CrewAIAdapter",
    "CrewAIAdapterConfig",
    "CrewAISendEventInput",
    "CrewAISendMessageInput",
    "PLATFORM_INSTRUCTIONS",
]

# Module-level state for nest_asyncio patch.
# See module docstring for important notes about global process impact.
_nest_asyncio_applied = False
_nest_asyncio_lock = threading.Lock()

# Context variable for thread-safe room context access.
# Set automatically when processing messages, accessed by tools.
_current_room_context: ContextVar[
    tuple[str, MessagingDispatchToolsProtocol] | None
] = ContextVar(
    "_current_room_context", default=None
)


@dataclass(frozen=True)
class CrewAIAdapterConfig:
    """Typed configuration surface for CrewAIAdapter."""

    model: str = "gpt-4o"
    role: str | None = None
    goal: str | None = None
    backstory: str | None = None
    custom_section: str | None = None
    enable_execution_reporting: bool = False
    enable_memory_tools: bool = False
    verbose: bool = False
    max_iter: int = 20
    max_rpm: int | None = None
    allow_delegation: bool = False
    history_converter: CrewAIHistoryConverter | None = None
    additional_tools: list[CustomToolDef] | None = None
    system_prompt: str | None = None



def _ensure_crewai_available() -> None:
    """Raise a consistent runtime error when CrewAI extras are missing."""
    ensure_optional_dependency(
        _CREWAI_IMPORT_ERROR,
        package="crewai",
        integration="CrewAI",
        install_commands=_CREWAI_INSTALL_COMMANDS,
    )


def _ensure_nest_asyncio() -> None:
    """Apply nest_asyncio patch lazily on first use.

    This function is thread-safe via a lock to prevent race conditions
    when multiple threads attempt to apply the patch simultaneously.

    See module docstring for important notes about global process impact.
    """
    _ensure_crewai_available()

    global _nest_asyncio_applied
    if _nest_asyncio_applied:
        return

    with _nest_asyncio_lock:
        # Double-check after acquiring lock (double-checked locking pattern)
        if not _nest_asyncio_applied:
            nest_asyncio.apply()
            _nest_asyncio_applied = True
            logger.debug("Applied nest_asyncio patch for nested event loops")


def _run_async(coro: Coroutine[Any, Any, T]) -> T:
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
        # No running event loop - use asyncio.run to create one
        logger.debug("Running coroutine in new event loop via asyncio.run")
        return asyncio.run(coro)

    # Event loop is running - use run_until_complete (safe with nest_asyncio)
    return loop.run_until_complete(coro)


class CrewAIAdapter(
    NonFatalErrorRecorder,
    SimpleAdapter[CrewAIMessages, MessagingDispatchToolsProtocol],
):
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
        _ensure_crewai_available()

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

        super().__init__(
            history_converter=history_converter or CrewAIHistoryConverter()
        )

        self.model = model
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.custom_section = custom_section
        self.enable_execution_reporting = enable_execution_reporting
        self.enable_memory_tools = enable_memory_tools
        self.verbose = verbose
        self.max_iter = max_iter
        self.max_rpm = max_rpm
        self.allow_delegation = allow_delegation

        self._crewai_agent: CrewAIAgent | None = None
        self._message_history: dict[str, list[dict[str, Any]]] = {}
        self._custom_tools: list[CustomToolDef] = additional_tools or []
        self._tool_runtime = CrewAIToolRuntime(
            base_tool_type=BaseTool,
            run_async=_run_async,
            get_room_context=self._get_current_room_context,
            enable_execution_reporting=self.enable_execution_reporting,
            enable_memory_tools=self.enable_memory_tools,
            custom_tools=self._custom_tools,
            schema_overrides=CREWAI_SCHEMA_OVERRIDES,
        )
        self._init_nonfatal_errors()

    async def on_started(self, agent_name: str, agent_description: str) -> None:
        """Initialize CrewAI agent after metadata is fetched."""
        await super().on_started(agent_name, agent_description)

        role = self.role or agent_name
        goal = self.goal or agent_description or "Help users accomplish their tasks"
        backstory = build_backstory(
            agent_name=agent_name,
            backstory=self.backstory,
            custom_section=self.custom_section,
            platform_instructions=PLATFORM_INSTRUCTIONS,
        )

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

    def _get_current_room_context(
        self,
    ) -> tuple[str, MessagingDispatchToolsProtocol] | None:
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
        """Execute a tool call using the composed CrewAI tool runtime."""
        return self._tool_runtime.execute_tool(tool_name, coro_factory)

    async def _report_tool_call(
        self,
        tools: MessagingDispatchToolsProtocol,
        tool_name: str,
        input_data: dict[str, Any],
    ) -> None:
        """Report a tool-call event through composed runtime bindings."""
        await self._tool_runtime.report_tool_call(tools, tool_name, input_data)

    async def _report_tool_result(
        self,
        tools: MessagingDispatchToolsProtocol,
        tool_name: str,
        result: Any,
        is_error: bool = False,
    ) -> None:
        """Report a tool-result event through composed runtime bindings."""
        await self._tool_runtime.report_tool_result(
            tools,
            tool_name,
            result,
            is_error=is_error,
        )

    def _convert_custom_tools_to_crewai(self) -> list[BaseTool]:
        """Convert custom tool defs into CrewAI tool instances."""
        return self._tool_runtime.build_custom_tools()

    @staticmethod
    def _result_mapping(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        return {"result": value}

    @classmethod
    def _format_success_payload(cls, tool_name: str, result: Any) -> dict[str, Any]:
        del cls
        return CrewAIToolRuntime.format_success_payload(tool_name, result)

    @staticmethod
    def _report_result_payload(
        tool_name: str,
        success_payload: dict[str, Any],
        result: Any,
    ) -> Any:
        return CrewAIToolRuntime.report_result_payload(
            tool_name,
            success_payload,
            result,
        )

    def _build_tool_class(
        self,
        tool_name: str,
        args_schema: type[BaseModel],
        *,
        report_execution: bool = True,
    ) -> type[BaseTool]:
        return self._tool_runtime.build_platform_tool_class(
            tool_name,
            args_schema,
            report_execution=report_execution,
        )

    def _create_crewai_tools(self) -> list[BaseTool]:
        """Create CrewAI-compatible platform/custom tools."""
        built_tools = self._tool_runtime.build_tools()
        return [tool for tool in built_tools if isinstance(tool, BaseTool)]

    @legacy_chat_turn_compat
    async def on_message(
        self,
        turn: ChatMessageTurnContext[
            CrewAIMessages,
            MessagingDispatchToolsProtocol,
        ],
    ) -> None:
        """Handle incoming message using CrewAI agent."""
        msg = turn.msg
        tools = turn.tools
        history = turn.history
        participants_msg = turn.participants_msg
        contacts_msg = turn.contacts_msg
        is_session_bootstrap = turn.is_session_bootstrap
        room_id = turn.room_id

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
        tools: MessagingDispatchToolsProtocol,
        history: CrewAIMessages,
        participants_msg: str | None,
        contacts_msg: str | None,
        *,
        is_session_bootstrap: bool,
        room_id: str,
    ) -> None:
        """Internal message processing logic."""
        await process_message(
            self,
            msg=msg,
            tools=tools,
            history=history,
            participants_msg=participants_msg,
            contacts_msg=contacts_msg,
            is_session_bootstrap=is_session_bootstrap,
            room_id=room_id,
        )

    async def on_cleanup(self, room_id: str) -> None:
        """Clean up message history when agent leaves a room."""
        if self.cleanup_room_state(self._message_history, room_id=room_id):
            logger.debug("Room %s: Cleaned up CrewAI session", room_id)
