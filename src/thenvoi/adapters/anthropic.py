"""
Anthropic adapter using SimpleAdapter pattern.

Extracted from thenvoi.integrations.anthropic.agent.ThenvoiAnthropicAgent.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, cast

from anthropic import AsyncAnthropic
from anthropic.types import Message, MessageParam, ToolParam, ToolUseBlock

from thenvoi.core.nonfatal import NonFatalErrorRecorder
from thenvoi.core.protocols import AnthropicSchemaToolsProtocol
from thenvoi.core.room_state import RoomStateStore
from thenvoi.core.simple_adapter import SimpleAdapter, legacy_chat_turn_compat
from thenvoi.core.types import ChatMessageTurnContext, PlatformMessage
from thenvoi.converters.anthropic import AnthropicHistoryConverter, AnthropicMessages
from thenvoi.runtime.tooling.custom_tools import (
    CustomToolDef,
    custom_tools_to_schemas,
    execute_custom_tool,
    find_custom_tool,
)
from thenvoi.runtime.prompts import render_system_prompt
from thenvoi.runtime.tool_bridge import format_tool_error, invoke_platform_tool

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnthropicAdapterConfig:
    """Typed configuration surface for AnthropicAdapter."""

    model: str = "claude-sonnet-4-5-20250929"
    anthropic_api_key: str | None = None
    system_prompt: str | None = None
    custom_section: str | None = None
    max_tokens: int = 4096
    enable_execution_reporting: bool = False
    enable_memory_tools: bool = False
    history_converter: AnthropicHistoryConverter | None = None
    additional_tools: list[CustomToolDef] | None = None


class AnthropicAdapter(
    NonFatalErrorRecorder,
    SimpleAdapter[AnthropicMessages, AnthropicSchemaToolsProtocol],
):
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
        enable_memory_tools: bool = False,
        history_converter: AnthropicHistoryConverter | None = None,
        additional_tools: list[CustomToolDef] | None = None,
    ):
        super().__init__(
            history_converter=history_converter or AnthropicHistoryConverter()
        )

        self.model = model
        self.system_prompt = system_prompt
        self.custom_section = custom_section
        self.max_tokens = max_tokens
        self.enable_execution_reporting = enable_execution_reporting
        self.enable_memory_tools = enable_memory_tools

        # Anthropic client (uses ANTHROPIC_API_KEY env var if not provided)
        self.client = AsyncAnthropic(api_key=anthropic_api_key)

        # Per-room conversation history (Anthropic SDK is stateless)
        self._message_history = RoomStateStore[list[dict[str, Any]]]()
        # Rendered system prompt (set after start)
        self._system_prompt: str = ""
        # Custom tools (user-provided)
        self._custom_tools: list[CustomToolDef] = additional_tools or []
        self._init_nonfatal_errors()

    # --- Copied from ThenvoiAnthropicAgent._on_started ---
    async def on_started(self, agent_name: str, agent_description: str) -> None:
        """Render system prompt after agent metadata is fetched."""
        await super().on_started(agent_name, agent_description)
        self._system_prompt = self.system_prompt or render_system_prompt(
            agent_name=agent_name,
            agent_description=agent_description,
            custom_section=self.custom_section or "",
        )
        logger.info("Anthropic adapter started for agent: %s", agent_name)

    # --- Adapted from ThenvoiAnthropicAgent._handle_message ---
    @legacy_chat_turn_compat
    async def on_message(
        self,
        turn: ChatMessageTurnContext[AnthropicMessages, AnthropicSchemaToolsProtocol],
    ) -> None:
        """
        Handle incoming message.

        KEY DESIGN:
        - System prompt sent ONLY on first API call (part of messages param)
        - Historical messages injected on first message to prime conversation
        - Participant list injected ONLY when it changes
        - Contact changes injected when broadcast
        - Tool loop runs until no more tool_use blocks
        """
        msg = turn.msg
        tools = turn.tools
        history = turn.history
        participants_msg = turn.participants_msg
        contacts_msg = turn.contacts_msg
        is_session_bootstrap = turn.is_session_bootstrap
        room_id = turn.room_id

        logger.debug("Handling message %s in room %s", msg.id, room_id)
        room_history = self._prepare_room_history(
            room_id=room_id,
            history=history,
            is_session_bootstrap=is_session_bootstrap,
            participants_msg=participants_msg,
            contacts_msg=contacts_msg,
        )
        self._append_user_message(room_history, msg)
        self._log_message_count(
            room_id=room_id,
            total_messages=len(room_history),
            is_session_bootstrap=is_session_bootstrap,
        )

        tool_schemas = self._build_tool_schemas(tools)
        await self._run_tool_loop(
            room_id=room_id,
            room_history=room_history,
            tool_schemas=tool_schemas,
            tools=tools,
        )

        logger.debug(
            "Message %s processed successfully (history now has %s messages)",
            msg.id,
            len(room_history),
        )

    def _prepare_room_history(
        self,
        *,
        room_id: str,
        history: AnthropicMessages,
        is_session_bootstrap: bool,
        participants_msg: str | None,
        contacts_msg: str | None,
    ) -> list[dict[str, Any]]:
        """Stage per-room history and inject system updates."""
        room_history, system_update_count = self.stage_room_history_with_updates(
            self._message_history,
            room_id=room_id,
            is_session_bootstrap=is_session_bootstrap,
            hydrated_history=list(history) if history else None,
            participants_msg=participants_msg,
            contacts_msg=contacts_msg,
            make_update_entry=lambda update: {
                "role": "user",
                "content": update,
            },
        )

        if is_session_bootstrap:
            if history:
                logger.info(
                    "Room %s: Loaded %s historical messages",
                    room_id,
                    len(history),
                )
            else:
                logger.info("Room %s: No historical messages found", room_id)

        if system_update_count:
            logger.info(
                "Room %s: Injected %d system updates",
                room_id,
                system_update_count,
            )
        return room_history

    @staticmethod
    def _append_user_message(
        room_history: list[dict[str, Any]],
        msg: PlatformMessage,
    ) -> None:
        """Append current user message to staged history."""
        room_history.append(
            {
                "role": "user",
                "content": msg.format_for_llm(),
            }
        )

    @staticmethod
    def _log_message_count(
        *,
        room_id: str,
        total_messages: int,
        is_session_bootstrap: bool,
    ) -> None:
        logger.info(
            "Room %s: Calling Anthropic with %s messages (first_msg=%s)",
            room_id,
            total_messages,
            is_session_bootstrap,
        )

    def _build_tool_schemas(
        self,
        tools: AnthropicSchemaToolsProtocol,
    ) -> list[ToolParam]:
        """Merge platform tool schemas with optional custom Anthropic schemas."""
        tool_schemas = tools.get_anthropic_tool_schemas(
            include_memory=self.enable_memory_tools
        )
        if self._custom_tools:
            tool_schemas = list(tool_schemas)
            custom_schemas = custom_tools_to_schemas(self._custom_tools, "anthropic")
            tool_schemas.extend(cast(list[ToolParam], custom_schemas))
        return tool_schemas

    async def _run_tool_loop(
        self,
        *,
        room_id: str,
        room_history: list[dict[str, Any]],
        tool_schemas: list[ToolParam],
        tools: AnthropicSchemaToolsProtocol,
    ) -> None:
        """Run Anthropic tool loop until stop_reason is not tool_use."""
        while True:
            try:
                response = await self._call_anthropic(
                    messages=room_history,
                    tools=tool_schemas,
                )
            except Exception as error:
                logger.error("Error calling Anthropic: %s", error, exc_info=True)
                await self.report_adapter_error(
                    tools,
                    error=error,
                    operation="report_error_event",
                )
                raise

            if response.stop_reason != "tool_use":
                text_content = self._extract_text_content(response.content)
                if text_content:
                    room_history.append(
                        {
                            "role": "assistant",
                            "content": text_content,
                        }
                    )
                logger.debug(
                    "Room %s: Completed with stop_reason=%s",
                    room_id,
                    response.stop_reason,
                )
                return

            serialized_content = self._serialize_content_blocks(response.content)
            room_history.append(
                {
                    "role": "assistant",
                    "content": serialized_content,
                }
            )
            tool_results = await self._process_tool_calls(response, tools)
            room_history.append(
                {
                    "role": "user",
                    "content": tool_results,
                }
            )

    # --- Copied from ThenvoiAnthropicAgent._cleanup_session ---
    async def on_cleanup(self, room_id: str) -> None:
        """Clean up message history when agent leaves a room."""
        if self.cleanup_room_state(self._message_history, room_id=room_id):
            logger.debug("Room %s: Cleaned up message history", room_id)

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
        self,
        response: Message,
        tools: AnthropicSchemaToolsProtocol,
    ) -> list[dict[str, Any]]:
        """
        Process tool_use blocks from response and execute tools.

        Args:
            response: Anthropic Message with tool_use blocks
            tools: AnthropicSchemaToolsProtocol instance for execution

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

            logger.debug("Executing tool: %s with input: %s", tool_name, tool_input)

            # Report tool call if enabled (JSON format with tool_call_id for linking)
            # Best-effort: event reporting must never crash tool execution
            if self.enable_execution_reporting:
                await self.send_tool_call_event(
                    tools,
                    payload={
                        "name": tool_name,
                        "args": tool_input,
                        "tool_call_id": tool_use_id,
                    },
                    tool_name=tool_name,
                )

            # Execute tool (check custom tools first, then platform tools)
            try:
                custom_tool = find_custom_tool(self._custom_tools, tool_name)
                if custom_tool:
                    result = await execute_custom_tool(custom_tool, tool_input)
                else:
                    result = await invoke_platform_tool(tools, tool_name, tool_input)
                result_str = (
                    json.dumps(result, default=str)
                    if not isinstance(result, str)
                    else result
                )
                is_error = False
            except Exception as e:
                result_str = format_tool_error(tool_name, tool_input, e)
                is_error = True
                logger.error("Tool %s failed: %s", tool_name, e)

            # Report tool result if enabled (JSON format with tool_call_id for linking)
            # Best-effort: event reporting must never crash tool execution
            if self.enable_execution_reporting:
                await self.send_tool_result_event(
                    tools,
                    payload={
                        "name": tool_name,
                        "output": result_str,
                        "tool_call_id": tool_use_id,
                    },
                    tool_name=tool_name,
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
