"""
Pydantic AI adapter using SimpleAdapter pattern.

Extracted from thenvoi.integrations.pydantic_ai.agent.ThenvoiPydanticAgent.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic_ai import Agent, RunContext
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
        history_converter: PydanticAIHistoryConverter | None = None,
    ):
        super().__init__(
            history_converter=history_converter or PydanticAIHistoryConverter()
        )

        self.model = model
        self.system_prompt = system_prompt
        self.custom_section = custom_section

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
            custom_section=self.custom_section or "",
        )

        # output_type=None disables output validation - we respond via tools only
        agent: Agent[AgentToolsProtocol, None] = Agent(  # type: ignore[call-overload]
            self.model,
            system_prompt=system,
            deps_type=AgentToolsProtocol,
            output_type=None,
        )

        # Register platform tools
        # All tools catch exceptions and return error strings so LLM can see failures
        @agent.tool
        async def send_message(
            ctx: RunContext[AgentToolsProtocol],
            content: str,
            mentions: list[str],
        ) -> dict[str, Any] | str:
            """Send a message to the chat room. Use this to respond to users or other agents.

            Args:
                content: The message content to send
                mentions: List of participant names to @mention. At least one required.
            """
            try:
                return await ctx.deps.send_message(content, mentions)
            except Exception as e:
                return f"Error sending message: {e}"

        @agent.tool
        async def send_event(
            ctx: RunContext[AgentToolsProtocol],
            content: str,
            message_type: str,
            metadata: dict[str, Any] | None = None,
        ) -> dict[str, Any] | str:
            """Send an event to the chat room. Use for thoughts, errors, or task updates.

            Args:
                content: Human-readable event content
                message_type: Type of event - "thought", "error", or "task"
                metadata: Optional structured data for the event
            """
            try:
                return await ctx.deps.send_event(content, message_type, metadata)
            except Exception as e:
                return f"Error sending event: {e}"

        @agent.tool
        async def add_participant(
            ctx: RunContext[AgentToolsProtocol],
            name: str,
            role: str = "member",
        ) -> dict[str, Any] | str:
            """Add a participant (agent or user) to the chat room.

            Args:
                name: Name of participant to add (must match a name from lookup_peers)
                role: Role for the participant - "owner", "admin", or "member" (default)
            """
            try:
                return await ctx.deps.add_participant(name, role)
            except Exception as e:
                return f"Error adding participant '{name}': {e}"

        @agent.tool
        async def remove_participant(
            ctx: RunContext[AgentToolsProtocol],
            name: str,
        ) -> dict[str, Any] | str:
            """Remove a participant from the chat room by name.

            Args:
                name: Name of the participant to remove
            """
            try:
                return await ctx.deps.remove_participant(name)
            except Exception as e:
                return f"Error removing participant '{name}': {e}"

        @agent.tool
        async def lookup_peers(
            ctx: RunContext[AgentToolsProtocol],
            page: int = 1,
            page_size: int = 50,
        ) -> dict[str, Any] | str:
            """List available peers (agents and users) that can be added to this room.

            Args:
                page: Page number (default 1)
                page_size: Items per page (default 50, max 100)
            """
            try:
                return await ctx.deps.lookup_peers(page, page_size)
            except Exception as e:
                return f"Error looking up peers: {e}"

        @agent.tool
        async def get_participants(
            ctx: RunContext[AgentToolsProtocol],
        ) -> list[dict[str, Any]] | str:
            """Get a list of all participants in the current chat room."""
            try:
                return await ctx.deps.get_participants()
            except Exception as e:
                return f"Error getting participants: {e}"

        @agent.tool
        async def create_chatroom(
            ctx: RunContext[AgentToolsProtocol],
            name: str,
        ) -> str:
            """Create a new chat room for a specific task or conversation.

            Args:
                name: Name for the new chat room
            """
            try:
                return await ctx.deps.create_chatroom(name)
            except Exception as e:
                return f"Error creating chatroom '{name}': {e}"

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

        # Run agent with conversation history
        result = await self._agent.run(
            user_message,
            deps=tools,
            message_history=self._message_history[room_id],
        )

        # Update stored history with all messages from this run
        self._message_history[room_id] = list(result.all_messages())

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
