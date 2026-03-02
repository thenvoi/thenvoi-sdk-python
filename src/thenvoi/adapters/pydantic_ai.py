"""
Pydantic AI adapter using SimpleAdapter pattern.

Extracted from thenvoi.integrations.pydantic_ai.agent.ThenvoiPydanticAgent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from pydantic_ai import (
    Agent,
    AgentRunResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    RunContext,
)
from pydantic_ai.messages import (
    ModelRequest,
    UserPromptPart,
)

from thenvoi.adapters.platform_tool_bindings import (
    build_pydantic_tool_function,
    platform_tool_names,
)
from thenvoi.core.nonfatal import NonFatalErrorRecorder
from thenvoi.core.protocols import (
    MessagingDispatchToolsProtocol,
)
from thenvoi.core.room_state import RoomStateStore
from thenvoi.core.simple_adapter import SimpleAdapter, legacy_chat_turn_compat
from thenvoi.core.types import ChatMessageTurnContext
from thenvoi.converters.pydantic_ai import (
    PydanticAIHistoryConverter,
    PydanticAIMessages,
)
from thenvoi.runtime.tool_bridge import format_tool_error, invoke_platform_tool
from thenvoi.runtime.prompts import render_system_prompt

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PydanticAIAdapterConfig:
    """Typed configuration surface for PydanticAIAdapter."""

    model: str
    system_prompt: str | None = None
    custom_section: str | None = None
    enable_execution_reporting: bool = False
    enable_memory_tools: bool = False
    history_converter: PydanticAIHistoryConverter | None = None
    additional_tools: list[Callable[..., Any]] | None = None


class PydanticAIAdapter(
    NonFatalErrorRecorder,
    SimpleAdapter[PydanticAIMessages, MessagingDispatchToolsProtocol],
):
    """
    Pydantic AI adapter using SimpleAdapter pattern.

    Uses Pydantic AI's Agent for LLM interactions,
    with platform tools registered via @agent.tool decorators.

    Example:
        adapter = PydanticAIAdapter(
            model="openai:gpt-4o",
            custom_section="You are a helpful assistant.",
        )
        agent = Agent.create(adapter=adapter, agent_id="...", api_key="...")
        await agent.run()
    """

    def __init__(
        self,
        model: str,
        system_prompt: str | None = None,
        custom_section: str | None = None,
        enable_execution_reporting: bool = False,
        enable_memory_tools: bool = False,
        history_converter: PydanticAIHistoryConverter | None = None,
        additional_tools: list[Callable[..., Any]] | None = None,
    ):
        """
        Initialize the Pydantic AI adapter.

        Args:
            model: Pydantic AI model string (e.g., "openai:gpt-4o", "anthropic:claude-3-5-sonnet-latest")
            system_prompt: Optional custom system prompt (overrides default)
            custom_section: Optional custom section added to default system prompt
            enable_execution_reporting: If True, emit tool_call and tool_result events
                to the platform for real-time visibility into agent activity.
                Defaults to False for backwards compatibility.
            enable_memory_tools: If True, includes memory management tools (enterprise only).
                Defaults to False.
            history_converter: Optional custom history converter
            additional_tools: Optional list of PydanticAI-compatible tool functions.
                Each function should follow PydanticAI's tool signature:
                `def my_tool(ctx: RunContext[MessagingDispatchToolsProtocol], arg1: str, ...) -> T`
                These are registered via agent.tool() alongside platform tools.
        """
        super().__init__(
            history_converter=history_converter or PydanticAIHistoryConverter()
        )

        self.model = model
        self.system_prompt = system_prompt
        self.custom_section = custom_section
        self.enable_execution_reporting = enable_execution_reporting
        self.enable_memory_tools = enable_memory_tools

        self._agent: Agent[MessagingDispatchToolsProtocol, None] | None = None
        # Conversation history per room (Pydantic AI is stateless, we maintain state)
        self._message_history = RoomStateStore[PydanticAIMessages]()
        # Custom tools (PydanticAI-compatible functions)
        self._custom_tools: list[Callable[..., Any]] = additional_tools or []
        self._init_nonfatal_errors()

    # --- Adapted from ThenvoiPydanticAgent._on_started ---
    async def on_started(self, agent_name: str, agent_description: str) -> None:
        """Create the Pydantic AI agent after metadata is fetched."""
        await super().on_started(agent_name, agent_description)
        self._agent = self._create_agent()
        logger.info("Pydantic AI adapter started for agent: %s", agent_name)

    # --- Copied from ThenvoiPydanticAgent._create_agent ---
    def _create_agent(self) -> Agent[MessagingDispatchToolsProtocol, None]:
        """Create Pydantic AI Agent with platform tools."""
        system = self.system_prompt or render_system_prompt(
            agent_name=self.agent_name,
            agent_description=self.agent_description or "An AI assistant",
            custom_section=self.custom_section or "",
        )

        # output_type=None disables output validation - we respond via tools only
        agent: Agent[MessagingDispatchToolsProtocol, None] = Agent(  # type: ignore[call-overload]
            self.model,
            system_prompt=system,
            deps_type=MessagingDispatchToolsProtocol,
            output_type=None,
        )

        # Register platform tools dynamically from centralized definitions
        # All wrappers use shared tool-bridge execution and error mapping.

        async def _invoke_tool(
            ctx: RunContext[MessagingDispatchToolsProtocol],
            tool_name: str,
            arguments: dict[str, Any],
        ) -> Any:
            try:
                return await invoke_platform_tool(ctx.deps, tool_name, arguments)
            except Exception as error:
                return format_tool_error(tool_name, arguments, error)

        for tool_name in platform_tool_names(
            include_memory_tools=self.enable_memory_tools
        ):
            agent.tool(
                build_pydantic_tool_function(
                    tool_name,
                    context_annotation=RunContext[MessagingDispatchToolsProtocol],
                    invoker=_invoke_tool,
                )
            )

        # Register custom tools (user-provided PydanticAI-compatible functions)
        for custom_tool in self._custom_tools:
            agent.tool(custom_tool)
            logger.debug("Registered custom tool: %s", custom_tool.__name__)

        return agent

    # --- Adapted from ThenvoiPydanticAgent._handle_message ---
    @legacy_chat_turn_compat
    async def on_message(
        self,
        turn: ChatMessageTurnContext[PydanticAIMessages, MessagingDispatchToolsProtocol],
    ) -> None:
        """Handle incoming platform message."""
        msg = turn.msg
        tools = turn.tools
        history = turn.history
        participants_msg = turn.participants_msg
        contacts_msg = turn.contacts_msg
        is_session_bootstrap = turn.is_session_bootstrap
        room_id = turn.room_id

        if self._agent is None:
            # Safety: create agent if not yet created (should be done in on_started)
            self._agent = self._create_agent()

        # Note: history is already converted by SimpleAdapter via history_converter
        room_history, system_update_count = self.stage_room_history_with_updates(
            self._message_history,
            room_id=room_id,
            is_session_bootstrap=is_session_bootstrap,
            hydrated_history=list(history) if history else None,
            participants_msg=participants_msg,
            contacts_msg=contacts_msg,
            make_update_entry=lambda update: ModelRequest(
                parts=[UserPromptPart(content=update)]
            ),
        )

        if is_session_bootstrap and history:
            logger.debug(
                "Room %s: Loaded %s Pydantic AI messages", room_id, len(history)
            )

        if system_update_count:
            logger.debug(
                "Room %s: Injected %d system updates into history",
                room_id,
                system_update_count,
            )

        # Build user message with sender prefix
        user_message = msg.format_for_llm()

        logger.debug(
            "Room %s: Running Pydantic AI agent (history: %s msgs, prompt: %s...)",
            room_id,
            len(room_history),
            user_message[:80],
        )

        # Run agent with streaming to capture tool events
        async for event in self._agent.run_stream_events(
            user_message,
            deps=tools,
            message_history=room_history,
        ):
            if isinstance(event, FunctionToolCallEvent):
                if self.enable_execution_reporting:
                    await self.send_tool_call_event(
                        tools,
                        payload={
                            "name": event.part.tool_name,
                            "args": event.part.args,
                            "tool_call_id": event.part.tool_call_id,
                        },
                        room_id=room_id,
                        tool_name=event.part.tool_name,
                        tool_call_id=event.part.tool_call_id,
                    )
            elif isinstance(event, FunctionToolResultEvent):
                if self.enable_execution_reporting:
                    await self.send_tool_result_event(
                        tools,
                        payload={
                            "name": event.result.tool_name,
                            "output": str(event.result.content),
                            "tool_call_id": event.tool_call_id,
                        },
                        room_id=room_id,
                        tool_name=event.result.tool_name,
                        tool_call_id=event.tool_call_id,
                    )
            elif isinstance(event, AgentRunResultEvent):
                # Update stored history with all messages from this run
                room_history = list(event.result.all_messages())
                self._message_history[room_id] = room_history

        logger.debug(
            "Room %s: Pydantic AI agent completed (history now has %s messages)",
            room_id,
            len(room_history),
        )

    # --- Copied from ThenvoiPydanticAgent._cleanup_session ---
    async def on_cleanup(self, room_id: str) -> None:
        """Clean up message history when agent leaves a room."""
        if self.cleanup_room_state(self._message_history, room_id=room_id):
            logger.debug("Room %s: Cleaned up message history", room_id)
