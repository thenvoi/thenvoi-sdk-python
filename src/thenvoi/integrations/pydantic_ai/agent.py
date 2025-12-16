"""
ThenvoiPydanticAgent - Pydantic AI agent connected to Thenvoi platform.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    UserPromptPart,
    TextPart,
)

from thenvoi.agents import BaseFrameworkAgent
from thenvoi.core import (
    PlatformMessage,
    AgentTools,
    AgentConfig,
    SessionConfig,
    render_system_prompt,
)
from thenvoi.core.session import AgentSession

logger = logging.getLogger(__name__)


class ThenvoiPydanticAgent(BaseFrameworkAgent):
    """
    Pydantic AI adapter for Thenvoi platform.

    This adapter uses Pydantic AI's Agent for LLM interactions,
    with platform tools registered via @agent.tool decorators.

    Args:
        model: Pydantic AI model string (e.g., "openai:gpt-4o", "anthropic:claude-3-5-sonnet-latest")
        agent_id: Thenvoi agent ID
        api_key: Thenvoi API key
        system_prompt: Optional custom system prompt (overrides default)
        custom_section: Optional custom section to add to default prompt
        ws_url: WebSocket URL for real-time events
        rest_url: REST API URL
        config: Agent configuration
        session_config: Session configuration

    Usage:
        adapter = ThenvoiPydanticAgent(
            model="openai:gpt-4o",
            agent_id="your-agent-id",
            api_key="your-api-key",
            custom_section="You are a helpful assistant.",
        )
        await adapter.run()
    """

    def __init__(
        self,
        model: str,
        agent_id: str,
        api_key: str,
        system_prompt: str | None = None,
        custom_section: str | None = None,
        ws_url: str = "wss://api.thenvoi.com/ws",
        rest_url: str = "https://api.thenvoi.com",
        config: AgentConfig | None = None,
        session_config: SessionConfig | None = None,
    ):
        super().__init__(
            agent_id=agent_id,
            api_key=api_key,
            ws_url=ws_url,
            rest_url=rest_url,
            config=config,
            session_config=session_config,
        )

        self.model = model
        self.system_prompt = system_prompt
        self.custom_section = custom_section

        self._agent: Agent[AgentTools, None] | None = None
        # Conversation history per room (Pydantic AI is stateless, we maintain state)
        self._message_history: dict[str, list] = {}

    async def _on_started(self) -> None:
        """Create the Pydantic AI agent after metadata is fetched."""
        self._agent = self._create_agent()
        logger.info(f"Pydantic AI adapter started for agent: {self.agent_name}")

    def _create_agent(self) -> Agent[AgentTools, None]:
        """Create Pydantic AI Agent with platform tools."""
        system = self.system_prompt or render_system_prompt(
            agent_name=self.agent_name,
            custom_section=self.custom_section or "",
        )

        # output_type=None disables output validation - we respond via tools only
        agent: Agent[AgentTools, None] = Agent(  # type: ignore[call-overload]
            self.model,
            system_prompt=system,
            deps_type=AgentTools,
            output_type=None,
        )

        # Register platform tools
        # All tools catch exceptions and return error strings so LLM can see failures
        @agent.tool
        async def send_message(
            ctx: RunContext[AgentTools],
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
            ctx: RunContext[AgentTools],
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
            ctx: RunContext[AgentTools],
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
            ctx: RunContext[AgentTools],
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
            ctx: RunContext[AgentTools],
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
            ctx: RunContext[AgentTools],
        ) -> list[dict[str, Any]] | str:
            """Get a list of all participants in the current chat room."""
            try:
                return await ctx.deps.get_participants()
            except Exception as e:
                return f"Error getting participants: {e}"

        @agent.tool
        async def create_chatroom(
            ctx: RunContext[AgentTools],
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

    async def _handle_message(
        self,
        msg: PlatformMessage,
        tools: AgentTools,
        session: AgentSession,
        history: list[dict[str, Any]] | None,
        participants_msg: str | None,
    ) -> None:
        """Handle incoming platform message."""
        if self._agent is None:
            # Safety: create agent if not yet created (should be done in _on_started)
            self._agent = self._create_agent()

        room_id = msg.room_id
        is_first_message = history is not None

        # Initialize message history for this room on first message
        if is_first_message:
            if history:
                self._message_history[room_id] = self._convert_platform_history(history)
                logger.debug(
                    f"Room {room_id}: Converted {len(history)} platform messages "
                    f"to {len(self._message_history[room_id])} Pydantic AI messages"
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

    def _convert_platform_history(
        self, platform_history: list[dict[str, Any]]
    ) -> list[ModelRequest | ModelResponse]:
        """
        Convert platform history to Pydantic AI message format.

        Platform history format:
            {"role": "user"|"assistant", "content": str, "sender_name": str}

        Pydantic AI format:
            - user messages → ModelRequest with UserPromptPart
            - assistant messages → ModelResponse with TextPart
        """
        messages: list[ModelRequest | ModelResponse] = []

        for h in platform_history:
            role = h.get("role", "user")
            content = h.get("content", "")
            sender_name = h.get("sender_name", "Unknown")

            if role == "assistant":
                # Our agent's previous messages
                messages.append(ModelResponse(parts=[TextPart(content=content)]))
            else:
                # Messages from users or other agents
                formatted_content = f"[{sender_name}]: {content}"
                messages.append(
                    ModelRequest(parts=[UserPromptPart(content=formatted_content)])
                )

        return messages

    async def _cleanup_session(self, room_id: str) -> None:
        """Clean up message history when agent leaves a room."""
        if room_id in self._message_history:
            del self._message_history[room_id]
            logger.debug(f"Room {room_id}: Cleaned up message history")


async def create_pydantic_agent(
    model: str,
    agent_id: str,
    api_key: str,
    **kwargs,
) -> ThenvoiPydanticAgent:
    """
    Create and start a ThenvoiPydanticAgent.

    Convenience function for quick setup.

    Args:
        model: Pydantic AI model string (e.g., "openai:gpt-4o")
        agent_id: Thenvoi agent ID
        api_key: Thenvoi API key
        **kwargs: Additional arguments for ThenvoiPydanticAgent

    Returns:
        Started ThenvoiPydanticAgent instance
    """
    agent = ThenvoiPydanticAgent(
        model=model,
        agent_id=agent_id,
        api_key=api_key,
        **kwargs,
    )
    await agent.start()
    return agent
