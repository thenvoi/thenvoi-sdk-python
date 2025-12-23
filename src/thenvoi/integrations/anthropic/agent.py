"""
ThenvoiAnthropicAgent - Anthropic SDK agent connected to Thenvoi platform.

This agent uses the Anthropic Python SDK directly for LLM interactions,
with full control over conversation history and tool loop management.

KEY DESIGN:
    SDK does NOT send messages directly.
    Anthropic agent uses tools.send_message() to respond.
"""

from __future__ import annotations

import logging
from typing import Any

from anthropic import AsyncAnthropic
from anthropic.types import Message, MessageParam, ToolParam

from thenvoi.agents import BaseFrameworkAgent
from thenvoi.runtime import (
    AgentConfig,
    AgentTools,
    ExecutionContext,
    PlatformMessage,
    SessionConfig,
    render_system_prompt,
)

from .history import AnthropicHistoryManager
from .message_utils import extract_text_content, filter_content_blocks
from .tool_executor import AnthropicToolExecutor

logger = logging.getLogger(__name__)


class ThenvoiAnthropicAgent(BaseFrameworkAgent):
    """
    Anthropic SDK adapter for Thenvoi platform.

    This adapter uses the Anthropic Python SDK directly for Claude interactions,
    with manual conversation history and tool loop management.

    Features:
    - Per-room conversation history management
    - Platform history hydration on first message
    - Participant tracking with automatic updates
    - Tool calling with Anthropic format
    - Event reporting (tool calls, results, errors)

    Args:
        model: Claude model ID (e.g., "claude-sonnet-4-5-20250929")
        agent_id: Thenvoi agent ID
        api_key: Thenvoi API key
        anthropic_api_key: Optional Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
        system_prompt: Optional custom system prompt (overrides default)
        custom_section: Optional custom section to add to default prompt
        max_tokens: Maximum tokens in response (default: 4096)
        ws_url: WebSocket URL for real-time events
        rest_url: REST API URL
        config: Agent configuration
        session_config: Session configuration

    Usage:
        agent = ThenvoiAnthropicAgent(
            model="claude-sonnet-4-20250514",
            agent_id="your-agent-id",
            api_key="your-thenvoi-api-key",
            custom_section="You are a helpful assistant.",
        )
        await agent.run()
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-5-20250929",
        agent_id: str = "",
        api_key: str = "",
        anthropic_api_key: str | None = None,
        system_prompt: str | None = None,
        custom_section: str | None = None,
        max_tokens: int = 4096,
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
        self.max_tokens = max_tokens

        # Anthropic client (uses ANTHROPIC_API_KEY env var if not provided)
        self.client = AsyncAnthropic(api_key=anthropic_api_key)

        # Components
        self._history_manager = AnthropicHistoryManager()
        self._tool_executor = AnthropicToolExecutor()

        # Rendered system prompt (set after start)
        self._system_prompt: str = ""
        # Max tool iterations to prevent infinite loops
        self._max_tool_iterations = 10

    async def _on_started(self) -> None:
        """Render system prompt after agent metadata is fetched."""
        self._system_prompt = self.system_prompt or render_system_prompt(
            agent_name=self.agent_name,
            agent_description=self.agent_description,
            custom_section=self.custom_section or "",
        )
        logger.info(f"Anthropic adapter started for agent: {self.agent_name}")

    async def _handle_message(
        self,
        msg: PlatformMessage,
        tools: AgentTools,
        ctx: ExecutionContext,
        history: list[dict[str, Any]] | None,
        participants_msg: str | None,
    ) -> None:
        """
        Handle incoming message.

        KEY DESIGN:
        - System prompt sent ONLY on first API call (part of messages param)
        - Historical messages injected on first message to prime conversation
        - Participant list injected ONLY when it changes
        - Tool loop runs until no more tool_use blocks
        """
        room_id = msg.room_id
        is_first_message = history is not None

        logger.debug(f"Handling message {msg.id} in room {room_id}")

        # Initialize history for this room on first message
        if is_first_message:
            self._history_manager.initialize_room(room_id, history)
        else:
            # Safety: ensure history exists even if not first message
            self._history_manager.ensure_room_exists(room_id)

        # Inject participants message if changed
        if participants_msg:
            self._history_manager.add_message(
                room_id, "user", f"[System]: {participants_msg}"
            )
            logger.info(
                f"Room {room_id}: Participants updated: "
                f"{[p.get('name') for p in ctx.participants]}"
            )

        # Add current message
        user_message = msg.format_for_llm()
        self._history_manager.add_message(room_id, "user", user_message)

        # Log message count
        room_history = self._history_manager.get_history(room_id)
        logger.info(
            f"Room {room_id}: Calling Anthropic with {len(room_history)} messages "
            f"(first_msg={is_first_message})"
        )

        # Get tool schemas in Anthropic format
        tool_schemas = tools.get_tool_schemas("anthropic")

        # Run tool loop
        await self._run_tool_loop(room_id, tool_schemas, tools)

        logger.debug(
            f"Message {msg.id} processed successfully "
            f"(history now has {len(self._history_manager.get_history(room_id))} messages)"
        )

    async def _run_tool_loop(
        self,
        room_id: str,
        tool_schemas: list[ToolParam],
        tools: AgentTools,
    ) -> None:
        """
        Run the tool loop until completion or max iterations.

        Args:
            room_id: Room identifier
            tool_schemas: Tool schemas in Anthropic format
            tools: AgentTools instance for execution
        """
        iteration = 0
        while iteration < self._max_tool_iterations:
            iteration += 1

            try:
                response = await self._call_anthropic(
                    messages=self._history_manager.get_history(room_id),
                    tools=tool_schemas,
                )
            except Exception as e:
                logger.error(f"Error calling Anthropic: {e}", exc_info=True)
                await self._report_error(tools, str(e))
                raise  # Re-raise so message is marked as failed

            # Check for tool use
            if response.stop_reason != "tool_use":
                # No more tool calls - extract text content if any
                text_content = extract_text_content(response.content)
                if text_content:
                    self._history_manager.add_message(
                        room_id, "assistant", text_content
                    )
                logger.debug(
                    f"Room {room_id}: Completed with stop_reason={response.stop_reason}"
                )
                break

            # Add assistant response with tool_use blocks to history
            content_blocks = filter_content_blocks(response.content)
            self._history_manager.add_message(room_id, "assistant", content_blocks)

            # Process tool calls
            tool_results = await self._tool_executor.process_tool_calls(response, tools)

            # Add tool results to history
            self._history_manager.add_message(room_id, "user", tool_results)

        if iteration >= self._max_tool_iterations:
            logger.warning(
                f"Room {room_id}: Hit max tool iterations ({self._max_tool_iterations})"
            )

    async def _call_anthropic(
        self,
        messages: list[MessageParam],
        tools: list[ToolParam],
    ) -> Message:
        """
        Call Anthropic API with messages and tools.

        Args:
            messages: Conversation history
            tools: Tool schemas in Anthropic format

        Returns:
            Anthropic Message response
        """
        return await self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self._system_prompt,
            messages=messages,
            tools=tools,
        )

    async def _cleanup_session(self, room_id: str) -> None:
        """Clean up message history when agent leaves a room."""
        self._history_manager.clear_room(room_id)


async def create_anthropic_agent(
    model: str,
    agent_id: str,
    api_key: str,
    **kwargs,
) -> ThenvoiAnthropicAgent:
    """
    Create and start a ThenvoiAnthropicAgent.

    Convenience function for quick setup.

    Args:
        model: Claude model ID (e.g., "claude-sonnet-4-5-20250929")
        agent_id: Thenvoi agent ID
        api_key: Thenvoi API key
        **kwargs: Additional arguments for ThenvoiAnthropicAgent

    Returns:
        Started ThenvoiAnthropicAgent instance
    """
    agent = ThenvoiAnthropicAgent(
        model=model,
        agent_id=agent_id,
        api_key=api_key,
        **kwargs,
    )
    await agent.start()
    return agent
