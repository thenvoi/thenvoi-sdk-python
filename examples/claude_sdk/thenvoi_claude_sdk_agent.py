"""
ThenvoiClaudeSDKAgent - Claude Agent SDK agent connected to Thenvoi platform.

This agent uses the Claude Agent SDK for LLM interactions, which provides:
- Automatic conversation history management
- Streaming responses via async iterator
- Extended thinking support
- MCP-based tool integration

Architecture:
    - MCP tools execute real API calls (not stubs)
    - room_id is passed in each message, Claude passes it to tools
    - Tools call coordinator methods directly
    - No interception needed - MCP handles everything
"""

from __future__ import annotations

import logging
from typing import Any

try:
    from claude_agent_sdk import (
        ClaudeSDKClient,
        ClaudeAgentOptions,
        AssistantMessage,
        TextBlock,
        ThinkingBlock,
        ToolUseBlock,
        ToolResultBlock,
        ResultMessage,
    )
except ImportError as e:
    raise ImportError(
        "claude-agent-sdk is required for Claude SDK examples.\n"
        "Install with: pip install claude-agent-sdk\n"
        "Or: uv add claude-agent-sdk"
    ) from e

from thenvoi.core import (
    ThenvoiAgent,
    AgentTools,
    AgentConfig,
    PlatformMessage,
    SessionConfig,
)

from session_manager import ClaudeSessionManager
from prompts import generate_claude_sdk_agent_prompt
from tools import create_thenvoi_mcp_server, THENVOI_TOOLS

logger = logging.getLogger(__name__)


class ThenvoiClaudeSDKAgent:
    """
    Claude Agent SDK adapter for Thenvoi platform.

    This adapter uses the Claude Agent SDK for LLM interactions, providing
    automatic conversation history management, streaming responses, and
    extended thinking support.

    Architecture:
    - MCP tools execute real API calls via coordinator
    - room_id is included in each message to Claude
    - Claude passes room_id to tool calls
    - No tool interception needed - MCP handles execution

    Args:
        model: Claude model ID (e.g., "claude-sonnet-4-5-20250929")
        agent_id: Thenvoi agent ID
        api_key: Thenvoi API key
        custom_section: Optional custom instructions to append to system prompt
        max_thinking_tokens: Optional max tokens for extended thinking
        permission_mode: SDK permission mode ("acceptEdits", "bypassPermissions", etc.)
        ws_url: WebSocket URL for real-time events
        rest_url: REST API URL
        config: Agent configuration
        session_config: Session configuration

    Usage:
        agent = ThenvoiClaudeSDKAgent(
            model="claude-sonnet-4-5-20250929",
            agent_id="your-agent-id",
            api_key="your-thenvoi-api-key",
            custom_section="You are a helpful assistant.",
        )
        await agent.run()

    Note:
        Requires Node.js 20+ and @anthropic-ai/claude-code CLI installed:
        npm install -g @anthropic-ai/claude-code
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-5-20250929",
        agent_id: str = "",
        api_key: str = "",
        custom_section: str | None = None,
        max_thinking_tokens: int | None = None,
        permission_mode: str = "acceptEdits",
        ws_url: str = "wss://api.thenvoi.com/ws",
        rest_url: str = "https://api.thenvoi.com",
        config: AgentConfig | None = None,
        session_config: SessionConfig | None = None,
    ):
        self.model = model
        self.custom_section = custom_section
        self.max_thinking_tokens = max_thinking_tokens
        self.permission_mode = permission_mode

        # Thenvoi coordinator
        self.thenvoi = ThenvoiAgent(
            agent_id=agent_id,
            api_key=api_key,
            ws_url=ws_url,
            rest_url=rest_url,
            config=config,
            session_config=session_config,
        )

        # Session manager and MCP server (created after start)
        self._session_manager: ClaudeSessionManager | None = None
        self._mcp_server = None

    @property
    def agent_name(self) -> str:
        """Get agent name from Thenvoi coordinator."""
        return self.thenvoi.agent_name

    async def start(self) -> None:
        """Start the adapter and begin processing messages."""
        # Register cleanup callback
        self.thenvoi._on_session_cleanup = self._cleanup_session
        await self.thenvoi.start(on_message=self._handle_message)

        # Create MCP server with coordinator (tools call coordinator methods)
        self._mcp_server = create_thenvoi_mcp_server(self.thenvoi)

        # Generate system prompt with agent info
        system_prompt = generate_claude_sdk_agent_prompt(
            agent_name=self.thenvoi.agent_name,
            agent_description=self.thenvoi.agent_description,
            custom_section=self.custom_section,
        )

        # Build SDK options
        sdk_options = ClaudeAgentOptions(
            model=self.model,
            system_prompt=system_prompt,
            mcp_servers={"thenvoi": self._mcp_server},
            allowed_tools=THENVOI_TOOLS,
            permission_mode=self.permission_mode,
        )

        # Add extended thinking if configured
        if self.max_thinking_tokens:
            sdk_options.max_thinking_tokens = self.max_thinking_tokens

        # Create session manager
        self._session_manager = ClaudeSessionManager(sdk_options)

        logger.info(
            f"Claude SDK adapter started for agent: {self.thenvoi.agent_name} "
            f"(model={self.model}, thinking={self.max_thinking_tokens})"
        )

    async def stop(self) -> None:
        """Stop the adapter."""
        if self._session_manager:
            await self._session_manager.cleanup_all()
        await self.thenvoi.stop()

    async def run(self) -> None:
        """Start and run until interrupted."""
        await self.start()
        try:
            await self.thenvoi.run()
        finally:
            await self.stop()

    async def _handle_message(self, msg: PlatformMessage, tools: AgentTools) -> None:
        """
        Handle incoming message.

        - Get or create ClaudeSDKClient for this room
        - Include room_id in the message so Claude can pass it to tools
        - Stream response and log events (tools execute via MCP)
        """
        room_id = msg.room_id

        logger.debug(f"Handling message {msg.id} in room {room_id}")

        if not self._session_manager:
            logger.error("Session manager not initialized")
            return

        # Get session for this room
        session = self.thenvoi.active_sessions.get(room_id)
        if not session:
            logger.error(f"No session for room {room_id}")
            return

        # Get or create Claude SDK client for this room
        client = await self._session_manager.get_or_create_session(room_id)

        # Check if this is first message for hydration
        is_first_message = not session.is_llm_initialized
        participants_changed = session.participants_changed()

        logger.debug(
            f"Room {room_id}: is_first_message={is_first_message}, "
            f"participants_changed={participants_changed}"
        )

        # Build message with room_id context
        messages_to_send = []

        # Add room_id context (Claude needs this for tool calls)
        room_context = f"[room_id: {room_id}]"

        # Hydrate history on first message
        if is_first_message:
            try:
                logger.info(
                    f"Room {room_id}: Loading platform history (first message)..."
                )
                platform_history = await session.get_history_for_llm(
                    exclude_message_id=msg.id
                )
                logger.info(
                    f"Room {room_id}: Platform returned "
                    f"{len(platform_history) if platform_history else 0} messages"
                )

                if platform_history:
                    history_text = self._format_history_for_context(
                        platform_history, room_id
                    )
                    if history_text:
                        messages_to_send.append(
                            f"Previous conversation history:\n{history_text}"
                        )
                        logger.info(
                            f"Room {room_id}: Injected {len(platform_history)} "
                            "historical messages as context"
                        )
            except Exception as e:
                logger.warning(
                    f"Room {room_id}: Failed to load history: {e}", exc_info=True
                )

            session.mark_llm_initialized()

        # Inject participants message if changed
        if participants_changed:
            participants_msg = session.build_participants_message()
            messages_to_send.append(f"{room_context}[System]: {participants_msg}")
            logger.info(
                f"Room {room_id}: Participants updated: "
                f"{[p.get('name') for p in session.participants]}"
            )
            session.mark_participants_sent()

        # Add current message with room_id context
        user_message = f"{room_context}{msg.format_for_llm()}"
        messages_to_send.append(user_message)

        # Send combined message to Claude
        full_message = "\n\n".join(messages_to_send)

        logger.info(
            f"Room {room_id}: Sending query to Claude SDK "
            f"(first_msg={is_first_message}, parts={len(messages_to_send)})"
        )

        try:
            # Send query to Claude
            await client.query(full_message)

            # Process streaming response (MCP tools handle execution)
            await self._process_response(client, room_id)

        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            # Try to report error to chat
            try:
                await tools.send_event(
                    content=f"Error: {e}",
                    message_type="error",
                )
            except Exception:
                pass
            raise

        logger.debug(f"Message {msg.id} processed successfully")

    async def _process_response(self, client: ClaudeSDKClient, room_id: str) -> None:
        """
        Process streaming response from Claude SDK.

        MCP tools handle actual execution - we just log events here.
        """
        async for sdk_message in client.receive_response():
            if isinstance(sdk_message, AssistantMessage):
                for block in sdk_message.content:
                    if isinstance(block, TextBlock):
                        if block.text:
                            logger.debug(f"Room {room_id}: Text: {block.text[:100]}...")

                    elif isinstance(block, ThinkingBlock):
                        if block.thinking:
                            logger.debug(
                                f"Room {room_id}: Thinking: {block.thinking[:100]}..."
                            )

                    elif isinstance(block, ToolUseBlock):
                        logger.info(
                            f"Room {room_id}: Tool call: {block.name} "
                            f"with {str(block.input)[:100]}..."
                        )

                    elif isinstance(block, ToolResultBlock):
                        logger.debug(
                            f"Room {room_id}: Tool result: "
                            f"{block.tool_use_id[:20]}... error={block.is_error}"
                        )

            elif isinstance(sdk_message, ResultMessage):
                logger.info(
                    f"Room {room_id}: Complete - "
                    f"{sdk_message.duration_ms}ms, "
                    f"${sdk_message.total_cost_usd or 0:.4f}"
                )
                break

    def _format_history_for_context(
        self, platform_history: list[dict[str, Any]], room_id: str
    ) -> str:
        """Format platform history with room_id context."""
        lines = []
        room_context = f"[room_id: {room_id}]"
        for h in platform_history:
            sender_name = h.get("sender_name", "Unknown")
            content = h.get("content", "")
            if content:
                lines.append(f"{room_context}[{sender_name}]: {content}")

        return "\n".join(lines)

    async def _cleanup_session(self, room_id: str) -> None:
        """Clean up Claude SDK session when agent leaves a room."""
        if self._session_manager:
            await self._session_manager.cleanup_session(room_id)
            logger.debug(f"Room {room_id}: Cleaned up Claude SDK session")


async def create_claude_sdk_agent(
    model: str,
    agent_id: str,
    api_key: str,
    **kwargs,
) -> ThenvoiClaudeSDKAgent:
    """
    Create and start a ThenvoiClaudeSDKAgent.

    Convenience function for quick setup.

    Args:
        model: Claude model ID (e.g., "claude-sonnet-4-5-20250929")
        agent_id: Thenvoi agent ID
        api_key: Thenvoi API key
        **kwargs: Additional arguments for ThenvoiClaudeSDKAgent

    Returns:
        Started ThenvoiClaudeSDKAgent instance
    """
    agent = ThenvoiClaudeSDKAgent(
        model=model,
        agent_id=agent_id,
        api_key=api_key,
        **kwargs,
    )
    await agent.start()
    return agent
