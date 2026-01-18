"""
Pydantic AI adapter using SimpleAdapter pattern.

Extracted from thenvoi.integrations.pydantic_ai.agent.ThenvoiPydanticAgent.
"""

from __future__ import annotations

import json
import logging
from typing import Any

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

from thenvoi.core.protocols import AgentToolsProtocol
from thenvoi.core.simple_adapter import SimpleAdapter
from thenvoi.core.types import PlatformMessage
from thenvoi.converters.pydantic_ai import (
    PydanticAIHistoryConverter,
    PydanticAIMessages,
)
from thenvoi.runtime.prompts import render_system_prompt
from thenvoi.runtime.tools import get_tool_description

logger = logging.getLogger(__name__)


class PydanticAIAdapter(SimpleAdapter[PydanticAIMessages]):
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
        history_converter: PydanticAIHistoryConverter | None = None,
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
            history_converter: Optional custom history converter
        """
        super().__init__(
            history_converter=history_converter or PydanticAIHistoryConverter()
        )

        self.model = model
        self.system_prompt = system_prompt
        self.custom_section = custom_section
        self.enable_execution_reporting = enable_execution_reporting

        self._agent: Agent[AgentToolsProtocol, None] | None = None
        # Conversation history per room (Pydantic AI is stateless, we maintain state)
        self._message_history: dict[str, list] = {}

    # --- Adapted from ThenvoiPydanticAgent._on_started ---
    async def on_started(self, agent_name: str, agent_description: str) -> None:
        """Create the Pydantic AI agent after metadata is fetched."""
        await super().on_started(agent_name, agent_description)
        self._agent = self._create_agent()
        logger.info(f"Pydantic AI adapter started for agent: {agent_name}")

    # --- Copied from ThenvoiPydanticAgent._create_agent ---
    def _create_agent(self) -> Agent[AgentToolsProtocol, None]:
        """Create Pydantic AI Agent with platform tools."""
        system = self.system_prompt or render_system_prompt(
            agent_name=self.agent_name,
            agent_description=self.agent_description or "An AI assistant",
            custom_section=self.custom_section or "",
        )

        # output_type=None disables output validation - we respond via tools only
        agent: Agent[AgentToolsProtocol, None] = Agent(  # type: ignore[call-overload]
            self.model,
            system_prompt=system,
            deps_type=AgentToolsProtocol,
            output_type=None,
        )

        # Register platform tools dynamically from centralized definitions
        # All tools catch exceptions and return error strings so LLM can see failures

        async def send_message(
            ctx: RunContext[AgentToolsProtocol],
            content: str,
            mentions: list[str],
        ) -> dict[str, Any] | str:
            try:
                return await ctx.deps.send_message(content, mentions)
            except Exception as e:
                return f"Error sending message: {e}"

        send_message.__doc__ = get_tool_description("send_message")
        agent.tool(send_message)

        async def send_event(
            ctx: RunContext[AgentToolsProtocol],
            content: str,
            message_type: str,
            metadata: dict[str, Any] | None = None,
        ) -> dict[str, Any] | str:
            try:
                return await ctx.deps.send_event(content, message_type, metadata)
            except Exception as e:
                return f"Error sending event: {e}"

        send_event.__doc__ = get_tool_description("send_event")
        agent.tool(send_event)

        async def add_participant(
            ctx: RunContext[AgentToolsProtocol],
            name: str,
            role: str = "member",
        ) -> dict[str, Any] | str:
            try:
                return await ctx.deps.add_participant(name, role)
            except Exception as e:
                return f"Error adding participant '{name}': {e}"

        add_participant.__doc__ = get_tool_description("add_participant")
        agent.tool(add_participant)

        async def remove_participant(
            ctx: RunContext[AgentToolsProtocol],
            name: str,
        ) -> dict[str, Any] | str:
            try:
                return await ctx.deps.remove_participant(name)
            except Exception as e:
                return f"Error removing participant '{name}': {e}"

        remove_participant.__doc__ = get_tool_description("remove_participant")
        agent.tool(remove_participant)

        async def lookup_peers(
            ctx: RunContext[AgentToolsProtocol],
            page: int = 1,
            page_size: int = 50,
        ) -> dict[str, Any] | str:
            try:
                return await ctx.deps.lookup_peers(page, page_size)
            except Exception as e:
                return f"Error looking up peers: {e}"

        lookup_peers.__doc__ = get_tool_description("lookup_peers")
        agent.tool(lookup_peers)

        async def get_participants(
            ctx: RunContext[AgentToolsProtocol],
        ) -> list[dict[str, Any]] | str:
            try:
                return await ctx.deps.get_participants()
            except Exception as e:
                return f"Error getting participants: {e}"

        get_participants.__doc__ = get_tool_description("get_participants")
        agent.tool(get_participants)

        async def create_chatroom(
            ctx: RunContext[AgentToolsProtocol],
            task_id: str | None = None,
        ) -> str:
            try:
                return await ctx.deps.create_chatroom(task_id)
            except Exception as e:
                return f"Error creating chatroom (task_id={task_id}): {e}"

        create_chatroom.__doc__ = get_tool_description("create_chatroom")
        agent.tool(create_chatroom)

        return agent

    # --- Adapted from ThenvoiPydanticAgent._handle_message ---
    async def on_message(
        self,
        msg: PlatformMessage,
        tools: AgentToolsProtocol,
        history: PydanticAIMessages,  # Already converted by SimpleAdapter
        participants_msg: str | None,
        *,
        is_session_bootstrap: bool,
        room_id: str,
    ) -> None:
        """Handle incoming platform message."""
        if self._agent is None:
            # Safety: create agent if not yet created (should be done in on_started)
            self._agent = self._create_agent()

        # Initialize message history for this room on first message
        # Note: history is already converted by SimpleAdapter via history_converter
        if is_session_bootstrap:
            if history:
                self._message_history[room_id] = list(history)
                logger.debug(
                    f"Room {room_id}: Loaded {len(history)} Pydantic AI messages"
                )
            else:
                self._message_history[room_id] = []
        elif room_id not in self._message_history:
            # Safety: ensure history exists even if not first message
            self._message_history[room_id] = []

        # Inject participants message if changed
        if participants_msg:
            self._message_history[room_id].append(
                ModelRequest(
                    parts=[UserPromptPart(content=f"[System]: {participants_msg}")]
                )
            )
            logger.debug(f"Room {room_id}: Injected participant update into history")

        # Build user message with sender prefix
        user_message = msg.format_for_llm()

        logger.debug(
            f"Room {room_id}: Running Pydantic AI agent "
            f"(history: {len(self._message_history[room_id])} msgs, "
            f"prompt: {user_message[:80]}...)"
        )

        # Run agent with streaming to capture tool events
        async for event in self._agent.run_stream_events(
            user_message,
            deps=tools,
            message_history=self._message_history[room_id],
        ):
            if isinstance(event, FunctionToolCallEvent):
                if self.enable_execution_reporting:
                    try:
                        await tools.send_event(
                            content=json.dumps(
                                {
                                    "name": event.part.tool_name,
                                    "args": event.part.args,
                                    "tool_call_id": event.part.tool_call_id,
                                }
                            ),
                            message_type="tool_call",
                        )
                    except Exception as e:
                        logger.warning(f"Failed to send tool_call event: {e}")
            elif isinstance(event, FunctionToolResultEvent):
                if self.enable_execution_reporting:
                    try:
                        await tools.send_event(
                            content=json.dumps(
                                {
                                    "name": event.result.tool_name,
                                    "output": str(event.result.content),
                                    "tool_call_id": event.tool_call_id,
                                }
                            ),
                            message_type="tool_result",
                        )
                    except Exception as e:
                        logger.warning(f"Failed to send tool_result event: {e}")
            elif isinstance(event, AgentRunResultEvent):
                # Update stored history with all messages from this run
                self._message_history[room_id] = list(event.result.all_messages())

        logger.debug(
            f"Room {room_id}: Pydantic AI agent completed "
            f"(history now has {len(self._message_history[room_id])} messages)"
        )

    # --- Copied from ThenvoiPydanticAgent._cleanup_session ---
    async def on_cleanup(self, room_id: str) -> None:
        """Clean up message history when agent leaves a room."""
        if room_id in self._message_history:
            del self._message_history[room_id]
            logger.debug(f"Room {room_id}: Cleaned up message history")
