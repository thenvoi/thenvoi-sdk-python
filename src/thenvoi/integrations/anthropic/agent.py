"""
ThenvoiAnthropicAgent - Anthropic SDK agent connected to Thenvoi platform.

This agent uses the Anthropic Python SDK directly for LLM interactions,
with full control over conversation history and tool loop management.

KEY DESIGN:
    SDK does NOT send messages directly.
    Anthropic agent uses tools.send_message() to respond.
"""

from __future__ import annotations

import json
import logging
from typing import Any, cast

from anthropic import AsyncAnthropic
from anthropic.types import Message, MessageParam, ToolParam, ToolUseBlock

from thenvoi.agents import BaseFrameworkAgent
from thenvoi.runtime import (
    AgentConfig,
    AgentTools,
    ExecutionContext,
    PlatformMessage,
    SessionConfig,
    render_system_prompt,
)

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

        # Per-room conversation history (Anthropic SDK is stateless)
        self._message_history: dict[str, list[dict[str, Any]]] = {}
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
            if history:
                self._message_history[room_id] = self._convert_platform_history(history)
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
            logger.info(
                f"Room {room_id}: Participants updated: "
                f"{[p.get('name') for p in ctx.participants]}"
            )

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
            f"(first_msg={is_first_message})"
        )

        # Get tool schemas in Anthropic format
        tool_schemas = tools.get_tool_schemas("anthropic")

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

    def _extract_text_content(self, content: list) -> str:
        """Extract text content from response content blocks."""
        from anthropic.types import TextBlock

        texts = []
        for block in content:
            if isinstance(block, TextBlock) and block.text:
                texts.append(block.text)
        return " ".join(texts) if texts else ""

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

    async def _process_tool_calls(
        self, response: Message, tools: AgentTools
    ) -> list[dict[str, Any]]:
        """
        Process tool_use blocks from response and execute tools.

        Args:
            response: Anthropic Message with tool_use blocks
            tools: AgentTools instance for execution

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

            # Report tool call (required for history reconstruction)
            await tools.send_event(
                content=json.dumps(
                    {
                        "run_id": tool_use_id,
                        "name": tool_name,
                        "data": {"input": tool_input},
                    },
                    default=str,
                ),
                message_type="tool_call",
                metadata=None,
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

            # Report tool result (required for history reconstruction)
            await tools.send_event(
                content=json.dumps(
                    {
                        "run_id": tool_use_id,
                        "name": tool_name,
                        "data": {"output": result_str, "is_error": is_error},
                    },
                    default=str,
                ),
                message_type="tool_result",
                metadata=None,
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

    def _convert_platform_history(
        self, platform_history: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Convert platform history to Anthropic message format.

        Handles message_type: "text", "tool_call", "tool_result"
        Pairs tool_calls with tool_results using run_id (primary) or name (LIFO fallback).

        Platform history format:
            {"message_type": "text"|"tool_call"|"tool_result", "content": str, ...}

        Anthropic format:
            {"role": "user"|"assistant", "content": str | list[ContentBlock]}
        """
        messages: list[dict[str, Any]] = []

        # Pending tool calls for pairing (matching LangGraph strategy)
        pending_by_run_id: dict[str, dict[str, Any]] = {}
        pending_by_name: dict[str, list[dict[str, Any]]] = {}

        for hist in platform_history:
            message_type = hist.get("message_type", "text")
            content = hist.get("content", "")
            role = hist.get("role")
            sender_name = hist.get("sender_name", "Unknown")

            if message_type == "tool_call":
                # Parse and store pending tool call
                try:
                    event = json.loads(content)
                    run_id = event.get("run_id")
                    tool_name = event.get("name", "unknown")

                    if run_id:
                        pending_by_run_id[run_id] = event
                    else:
                        # Fallback: store by name (LIFO stack)
                        if tool_name not in pending_by_name:
                            pending_by_name[tool_name] = []
                        pending_by_name[tool_name].append(event)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse tool_call: {content[:100]}")

            elif message_type == "tool_result":
                # Match to pending tool_call, emit assistant + user messages
                try:
                    event = json.loads(content)
                    tool_name = event.get("name", "unknown")
                    run_id = event.get("run_id")
                    output = event.get("data", {}).get("output", "")
                    is_error = event.get("data", {}).get("is_error", False)

                    # Use run_id as tool_use_id (matches how we store it)
                    tool_use_id = run_id or f"tool_{tool_name}"

                    # Find matching pending tool_call
                    matching_call = None

                    # Primary: match by run_id
                    if run_id and run_id in pending_by_run_id:
                        matching_call = pending_by_run_id.pop(run_id)

                    # Fallback: match by name (LIFO)
                    if not matching_call and tool_name in pending_by_name:
                        if pending_by_name[tool_name]:
                            matching_call = pending_by_name[tool_name].pop()

                    if matching_call:
                        tool_input = matching_call.get("data", {}).get("input", {})

                        # Emit assistant message with tool_use block
                        messages.append(
                            {
                                "role": "assistant",
                                "content": [
                                    {
                                        "type": "tool_use",
                                        "id": tool_use_id,
                                        "name": tool_name,
                                        "input": tool_input,
                                    }
                                ],
                            }
                        )
                    else:
                        logger.warning(
                            f"tool_result without matching tool_call: "
                            f"name={tool_name}, run_id={run_id}"
                        )

                    # Emit user message with tool_result block
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tool_use_id,
                                    "content": str(output),
                                    "is_error": is_error,
                                }
                            ],
                        }
                    )

                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse tool_result: {content[:100]}")

            elif message_type == "text":
                if role == "assistant":
                    # Skip assistant text - redundant with tool_call/tool_result
                    logger.debug(f"Skipping redundant assistant text: {content[:50]}")
                else:
                    # User text message
                    formatted = f"[{sender_name}]: {content}"
                    messages.append({"role": "user", "content": formatted})

            # Skip other message types (thought, error, task, etc.)

        # Warn about unmatched tool calls
        unmatched = len(pending_by_run_id) + sum(
            len(v) for v in pending_by_name.values()
        )
        if unmatched:
            logger.warning(
                f"Found {unmatched} tool_calls without matching tool_results"
            )

        return self._batch_consecutive_messages(messages)

    def _batch_consecutive_messages(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Batch consecutive same-role messages into single messages.

        Anthropic requires alternating user/assistant roles. This merges
        consecutive messages with the same role into a single message
        with multiple content blocks.
        """
        if not messages:
            return messages

        batched: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None

        for msg in messages:
            if current and current["role"] == msg["role"]:
                # Same role - merge content blocks
                current_content = current["content"]
                msg_content = msg["content"]

                # Normalize to lists
                if isinstance(current_content, str):
                    current["content"] = [{"type": "text", "text": current_content}]
                if isinstance(msg_content, str):
                    msg_content = [{"type": "text", "text": msg_content}]

                current["content"].extend(msg_content)
            else:
                # Different role - start new message
                if current:
                    batched.append(current)
                current = {
                    "role": msg["role"],
                    "content": msg["content"]
                    if isinstance(msg["content"], list)
                    else msg["content"],
                }

        if current:
            batched.append(current)

        return batched

    async def _cleanup_session(self, room_id: str) -> None:
        """Clean up message history when agent leaves a room."""
        if room_id in self._message_history:
            del self._message_history[room_id]
            logger.debug(f"Room {room_id}: Cleaned up message history")


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
