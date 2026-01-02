"""
Anthropic adapter using SimpleAdapter pattern.

Extracted from thenvoi.integrations.anthropic.agent.ThenvoiAnthropicAgent.
"""

from __future__ import annotations

import json
import logging
from typing import Any, cast

from anthropic import AsyncAnthropic
from anthropic.types import Message, MessageParam, ToolParam, ToolUseBlock

from thenvoi.core.protocols import AgentToolsProtocol
from thenvoi.core.simple_adapter import SimpleAdapter
from thenvoi.core.types import PlatformMessage
from thenvoi.converters.anthropic import AnthropicHistoryConverter, AnthropicMessages
from thenvoi.runtime.prompts import render_system_prompt

logger = logging.getLogger(__name__)


class AnthropicAdapter(SimpleAdapter[AnthropicMessages]):
    """
    Anthropic SDK adapter using SimpleAdapter pattern.

    Uses the Anthropic Python SDK directly for Claude interactions,
    with manual conversation history and tool loop management.

    Example:
        adapter = AnthropicAdapter(
            model="claude-sonnet-4-5-20250929",
            custom_section="You are a helpful assistant.",
        )
        agent = Agent.create(adapter=adapter, agent_id="...", api_key="...")
        await agent.run()
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-5-20250929",
        anthropic_api_key: str | None = None,
        system_prompt: str | None = None,
        custom_section: str | None = None,
        max_tokens: int = 4096,
        enable_execution_reporting: bool = False,
        history_converter: AnthropicHistoryConverter | None = None,
    ):
        super().__init__(
            history_converter=history_converter or AnthropicHistoryConverter()
        )

        self.model = model
        self.system_prompt = system_prompt
        self.custom_section = custom_section
        self.max_tokens = max_tokens
        self.enable_execution_reporting = enable_execution_reporting

        # Anthropic client (uses ANTHROPIC_API_KEY env var if not provided)
        self.client = AsyncAnthropic(api_key=anthropic_api_key)

        # Per-room conversation history (Anthropic SDK is stateless)
        self._message_history: dict[str, list[dict[str, Any]]] = {}
        # Rendered system prompt (set after start)
        self._system_prompt: str = ""
        # Max tool iterations to prevent infinite loops
        self._max_tool_iterations = 10

    # --- Copied from ThenvoiAnthropicAgent._on_started ---
    async def on_started(self, agent_name: str, agent_description: str) -> None:
        """Render system prompt after agent metadata is fetched."""
        await super().on_started(agent_name, agent_description)
        self._system_prompt = self.system_prompt or render_system_prompt(
            agent_name=agent_name,
            agent_description=agent_description,
            custom_section=self.custom_section or "",
        )
        logger.info(f"Anthropic adapter started for agent: {agent_name}")

    # --- Adapted from ThenvoiAnthropicAgent._handle_message ---
    async def on_message(
        self,
        msg: PlatformMessage,
        tools: AgentToolsProtocol,
        history: AnthropicMessages,  # Already converted by SimpleAdapter
        participants_msg: str | None,
        *,
        is_session_bootstrap: bool,
        room_id: str,
    ) -> None:
        """
        Handle incoming message.

        KEY DESIGN:
        - System prompt sent ONLY on first API call (part of messages param)
        - Historical messages injected on first message to prime conversation
        - Participant list injected ONLY when it changes
        - Tool loop runs until no more tool_use blocks
        """
        logger.debug(f"Handling message {msg.id} in room {room_id}")

        # Initialize history for this room on first message
        # Note: history is already converted by SimpleAdapter via history_converter
        if is_session_bootstrap:
            if history:
                self._message_history[room_id] = list(history)
                logger.info(
                    f"Room {room_id}: Loaded {len(history)} historical messages"
                )
            else:
                self._message_history[room_id] = []
                logger.info(f"Room {room_id}: No historical messages found")
        elif room_id not in self._message_history:
            # Safety: ensure history exists even if not first message
            self._message_history[room_id] = []

        # Inject participants message if changed
        if participants_msg:
            self._message_history[room_id].append(
                {
                    "role": "user",
                    "content": f"[System]: {participants_msg}",
                }
            )
            logger.info(f"Room {room_id}: Participants updated")

        # Add current message
        user_message = msg.format_for_llm()
        self._message_history[room_id].append(
            {
                "role": "user",
                "content": user_message,
            }
        )

        # Log message count
        total_messages = len(self._message_history[room_id])
        logger.info(
            f"Room {room_id}: Calling Anthropic with {total_messages} messages "
            f"(first_msg={is_session_bootstrap})"
        )

        # Get tool schemas in Anthropic format (typed helper)
        tool_schemas = tools.get_anthropic_tool_schemas()

        # Tool loop
        iteration = 0
        while iteration < self._max_tool_iterations:
            iteration += 1

            try:
                response = await self._call_anthropic(
                    messages=self._message_history[room_id],
                    tools=tool_schemas,
                )
            except Exception as e:
                logger.error(f"Error calling Anthropic: {e}", exc_info=True)
                await self._report_error(tools, str(e))
                raise  # Re-raise so message is marked as failed

            # Check for tool use
            if response.stop_reason != "tool_use":
                # No more tool calls - extract text content if any
                text_content = self._extract_text_content(response.content)
                if text_content:
                    self._message_history[room_id].append(
                        {
                            "role": "assistant",
                            "content": text_content,
                        }
                    )
                logger.debug(
                    f"Room {room_id}: Completed with stop_reason={response.stop_reason}"
                )
                break

            # Add assistant response with tool_use blocks to history
            serialized_content = self._serialize_content_blocks(response.content)
            self._message_history[room_id].append(
                {
                    "role": "assistant",
                    "content": serialized_content,
                }
            )

            # Process tool calls
            tool_results = await self._process_tool_calls(response, tools)

            # Add tool results to history
            self._message_history[room_id].append(
                {
                    "role": "user",
                    "content": tool_results,
                }
            )

        if iteration >= self._max_tool_iterations:
            logger.warning(
                f"Room {room_id}: Hit max tool iterations ({self._max_tool_iterations})"
            )

        logger.debug(
            f"Message {msg.id} processed successfully "
            f"(history now has {len(self._message_history[room_id])} messages)"
        )

    # --- Copied from ThenvoiAnthropicAgent._cleanup_session ---
    async def on_cleanup(self, room_id: str) -> None:
        """Clean up message history when agent leaves a room."""
        if room_id in self._message_history:
            del self._message_history[room_id]
            logger.debug(f"Room {room_id}: Cleaned up message history")

    # --- Copied from ThenvoiAnthropicAgent._call_anthropic ---
    async def _call_anthropic(
        self,
        messages: list[dict[str, Any]],
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
            messages=cast(list[MessageParam], messages),
            tools=tools,
        )

    # --- Copied from ThenvoiAnthropicAgent._extract_text_content ---
    def _extract_text_content(self, content: list) -> str:
        """Extract text content from response content blocks."""
        from anthropic.types import TextBlock

        texts = []
        for block in content:
            if isinstance(block, TextBlock) and block.text:
                texts.append(block.text)
        return " ".join(texts) if texts else ""

    # --- Copied from ThenvoiAnthropicAgent._serialize_content_blocks ---
    def _serialize_content_blocks(self, content: list) -> list[dict[str, Any]]:
        """Serialize content blocks to dict format for message history."""
        from anthropic.types import TextBlock

        serialized = []
        for block in content:
            if isinstance(block, ToolUseBlock):
                serialized.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )
            elif isinstance(block, TextBlock):
                if block.text:  # Only include non-empty text
                    serialized.append(
                        {
                            "type": "text",
                            "text": block.text,
                        }
                    )
        return serialized

    # --- Copied from ThenvoiAnthropicAgent._process_tool_calls ---
    async def _process_tool_calls(
        self, response: Message, tools: AgentToolsProtocol
    ) -> list[dict[str, Any]]:
        """
        Process tool_use blocks from response and execute tools.

        Args:
            response: Anthropic Message with tool_use blocks
            tools: AgentToolsProtocol instance for execution

        Returns:
            List of tool_result content blocks for next API call
        """
        tool_results = []

        for block in response.content:
            if not isinstance(block, ToolUseBlock):
                continue

            tool_name = block.name
            tool_input = block.input
            tool_use_id = block.id

            logger.debug(f"Executing tool: {tool_name} with input: {tool_input}")

            # Report tool call if enabled (JSON format with tool_call_id for linking)
            if self.enable_execution_reporting:
                await tools.send_event(
                    content=json.dumps(
                        {
                            "name": tool_name,
                            "args": tool_input,
                            "tool_call_id": tool_use_id,
                        }
                    ),
                    message_type="tool_call",
                )

            # Execute tool
            try:
                result = await tools.execute_tool_call(tool_name, tool_input)
                result_str = (
                    json.dumps(result, default=str)
                    if not isinstance(result, str)
                    else result
                )
                is_error = False
            except Exception as e:
                result_str = f"Error: {e}"
                is_error = True
                logger.error(f"Tool {tool_name} failed: {e}")

            # Report tool result if enabled (JSON format with tool_call_id for linking)
            if self.enable_execution_reporting:
                await tools.send_event(
                    content=json.dumps(
                        {
                            "name": tool_name,
                            "output": result_str,
                            "tool_call_id": tool_use_id,
                        }
                    ),
                    message_type="tool_result",
                )

            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": result_str,
                    "is_error": is_error,
                }
            )

        return tool_results

    # --- Copied from BaseFrameworkAgent._report_error ---
    async def _report_error(self, tools: AgentToolsProtocol, error: str) -> None:
        """Send error event (best effort)."""
        try:
            await tools.send_event(content=f"Error: {error}", message_type="error")
        except Exception:
            pass
