"""
Claude SDK adapter using SimpleAdapter pattern.

Extracted from thenvoi.integrations.claude_sdk.agent.ThenvoiClaudeSDKAgent.

Note: This adapter is more complex than Anthropic/PydanticAI because Claude SDK
uses MCP servers which need access to tools by room_id. We store tools per-room
when on_message is called so the MCP server can access them.

Custom Tools Support:
    You can pass custom MCP tools to the adapter using the `custom_tools` parameter.
    Tools must be created using the `@tool` decorator from `claude_agent_sdk`.

    Example:
        from claude_agent_sdk import tool

        @tool("calculator", "Perform calculations", {"expression": str})
        async def calculator(args: dict) -> dict:
            result = eval(args["expression"])
            return {"content": [{"type": "text", "text": str(result)}]}

        adapter = ClaudeSDKAdapter(
            model="claude-sonnet-4-5-20250929",
            custom_tools=[calculator],
        )
"""

from __future__ import annotations

import json
import logging
from typing import Any, Literal

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
        tool,
        create_sdk_mcp_server,
    )
except ImportError as e:
    raise ImportError(
        "claude-agent-sdk is required for Claude SDK examples.\n"
        "Install with: pip install claude-agent-sdk\n"
        "Or: uv add claude-agent-sdk"
    ) from e

from thenvoi.core.protocols import AgentToolsProtocol
from thenvoi.core.simple_adapter import SimpleAdapter
from thenvoi.core.types import PlatformMessage
from thenvoi.converters.claude_sdk import ClaudeSDKHistoryConverter
from thenvoi.integrations.claude_sdk.session_manager import ClaudeSessionManager
from thenvoi.integrations.claude_sdk.prompts import generate_claude_sdk_agent_prompt
from thenvoi.runtime.tools import get_tool_description

logger = logging.getLogger(__name__)

# Type alias for custom tool functions (decorated with @tool from claude_agent_sdk)
# Using Any since @tool decorator transforms the function into SdkMcpTool
CustomTool = Any


# Tool names as constants (MCP naming convention: mcp__{server}__{tool})
THENVOI_TOOLS = [
    "mcp__thenvoi__send_message",
    "mcp__thenvoi__send_event",
    "mcp__thenvoi__add_participant",
    "mcp__thenvoi__remove_participant",
    "mcp__thenvoi__get_participants",
    "mcp__thenvoi__lookup_peers",
]


class ClaudeSDKAdapter(SimpleAdapter[str]):
    """
    Claude Agent SDK adapter using SimpleAdapter pattern.

    Uses the Claude Agent SDK for LLM interactions with MCP-based tool integration.

    Note: This adapter stores tools per-room so the MCP server can access them.
    The history is converted to a text string for context injection.

    Args:
        model: Claude model to use (default: claude-sonnet-4-5-20250929)
        custom_section: Custom instructions to add to system prompt
        max_thinking_tokens: Enable extended thinking with token budget
        permission_mode: Tool permission mode
        enable_execution_reporting: Report tool calls as events to chat
        custom_tools: List of custom MCP tools created with @tool decorator
        history_converter: Custom history converter

    Example - Basic:
        adapter = ClaudeSDKAdapter(
            model="claude-sonnet-4-5-20250929",
            custom_section="You are a helpful assistant.",
        )
        agent = Agent.create(adapter=adapter, agent_id="...", api_key="...")
        await agent.run()

    Example - With Custom Tools:
        from claude_agent_sdk import tool

        @tool("calculator", "Perform math calculations", {"expression": str})
        async def calculator(args: dict) -> dict:
            result = eval(args["expression"])
            return {"content": [{"type": "text", "text": str(result)}]}

        @tool("weather", "Get weather for a city", {"city": str})
        async def get_weather(args: dict) -> dict:
            return {"content": [{"type": "text", "text": f"Weather in {args['city']}: Sunny"}]}

        adapter = ClaudeSDKAdapter(
            model="claude-sonnet-4-5-20250929",
            custom_tools=[calculator, get_weather],
        )
    """

    PermissionMode = Literal["default", "acceptEdits", "plan", "bypassPermissions"]

    def __init__(
        self,
        model: str = "claude-sonnet-4-5-20250929",
        custom_section: str | None = None,
        max_thinking_tokens: int | None = None,
        permission_mode: PermissionMode = "acceptEdits",
        enable_execution_reporting: bool = False,
        custom_tools: list[CustomTool] | None = None,
        history_converter: ClaudeSDKHistoryConverter | None = None,
    ):
        super().__init__(
            history_converter=history_converter or ClaudeSDKHistoryConverter()
        )

        self.model = model
        self.custom_section = custom_section
        self.max_thinking_tokens = max_thinking_tokens
        self.permission_mode: ClaudeSDKAdapter.PermissionMode = permission_mode
        self.enable_execution_reporting = enable_execution_reporting
        self.custom_tools = custom_tools or []

        # Session manager and MCP server (created after start)
        self._session_manager: ClaudeSessionManager | None = None
        self._mcp_server = None

        # Per-room tools storage for MCP server access
        self._room_tools: dict[str, AgentToolsProtocol] = {}

        # Per-room session context (text history for Claude SDK)
        self._session_context: dict[str, str] = {}

        # Per-room session IDs (for SDK session resume)
        self._session_ids: dict[str, str] = {}

    # --- Adapted from ThenvoiClaudeSDKAgent._on_started ---
    async def on_started(self, agent_name: str, agent_description: str) -> None:
        """Create MCP server and session manager after agent metadata is fetched."""
        await super().on_started(agent_name, agent_description)

        # Create MCP server with self (provides tool access via _room_tools)
        self._mcp_server = self._create_mcp_server()

        # Generate system prompt with agent info
        system_prompt = generate_claude_sdk_agent_prompt(
            agent_name=agent_name,
            agent_description=agent_description,
            custom_section=self.custom_section,
        )

        # Build allowed tools list (thenvoi tools + custom tools)
        allowed_tools = list(THENVOI_TOOLS)
        for custom_tool in self.custom_tools:
            # Custom tools are prefixed with mcp__thenvoi__ when registered
            tool_name = self._get_tool_name(custom_tool)
            if tool_name:
                allowed_tools.append(f"mcp__thenvoi__{tool_name}")

        # Build SDK options
        sdk_options = ClaudeAgentOptions(
            model=self.model,
            system_prompt=system_prompt,
            mcp_servers={"thenvoi": self._mcp_server},
            allowed_tools=allowed_tools,
            permission_mode=self.permission_mode,
        )

        # Add extended thinking if configured
        if self.max_thinking_tokens:
            sdk_options.max_thinking_tokens = self.max_thinking_tokens

        # Create session manager
        self._session_manager = ClaudeSessionManager(sdk_options)

        logger.info(
            f"Claude SDK adapter started for agent: {agent_name} "
            f"(model={self.model}, thinking={self.max_thinking_tokens}, "
            f"custom_tools={len(self.custom_tools)})"
        )

    def _get_tool_name(self, tool_func: CustomTool) -> str | None:
        """Extract tool name from a @tool decorated function."""
        # The @tool decorator adds metadata to the function
        if hasattr(tool_func, "_tool_name"):
            return tool_func._tool_name
        if hasattr(tool_func, "name"):
            return tool_func.name
        # Fallback: try to get from __name__
        return getattr(tool_func, "__name__", None)

    def _create_mcp_server(self):
        """Create MCP SDK server that uses stored room tools."""
        adapter = self  # Capture reference for tool closures

        def _make_result(data: Any) -> dict[str, Any]:
            """Format tool result for MCP response."""
            return {
                "content": [{"type": "text", "text": json.dumps(data, default=str)}]
            }

        def _make_error(error: str) -> dict[str, Any]:
            """Format error result for MCP response."""
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({"status": "error", "message": error}),
                    }
                ],
                "is_error": True,
            }

        def _get_tools(room_id: str) -> AgentToolsProtocol | None:
            """Get stored tools for a room."""
            return adapter._room_tools.get(room_id)

        @tool(
            "send_message",
            get_tool_description("send_message"),
            {
                "room_id": str,
                "content": str,
                "mentions": str,
            },
        )
        async def send_message(args: dict[str, Any]) -> dict[str, Any]:
            """Send message to chat room via API."""
            try:
                room_id = args.get("room_id", "")
                content = args.get("content", "")
                mentions_str = args.get("mentions", "[]")

                mention_names: list[str] = []
                if mentions_str:
                    try:
                        mention_names = (
                            json.loads(mentions_str)
                            if isinstance(mentions_str, str)
                            else mentions_str
                        )
                    except json.JSONDecodeError:
                        pass

                logger.info(f"[{room_id}] send_message: {content[:100]}...")

                tools = _get_tools(room_id)
                if not tools:
                    return _make_error(f"No tools available for room {room_id}")

                await tools.send_message(content, mention_names)

                return _make_result({"status": "success", "message": "Message sent"})

            except Exception as e:
                logger.error(f"send_message failed: {e}", exc_info=True)
                return _make_error(str(e))

        @tool(
            "send_event",
            get_tool_description("send_event"),
            {"room_id": str, "content": str, "message_type": str},
        )
        async def send_event(args: dict[str, Any]) -> dict[str, Any]:
            """Send event to chat room via API."""
            try:
                room_id = args.get("room_id", "")
                content = args.get("content", "")
                message_type = args.get("message_type", "thought")

                tools = _get_tools(room_id)
                if not tools:
                    return _make_error(f"No tools available for room {room_id}")

                await tools.send_event(content, message_type)

                return _make_result({"status": "success", "message": "Event sent"})

            except Exception as e:
                logger.error(f"send_event failed: {e}", exc_info=True)
                return _make_error(str(e))

        @tool(
            "add_participant",
            get_tool_description("add_participant"),
            {"room_id": str, "name": str, "role": str},
        )
        async def add_participant(args: dict[str, Any]) -> dict[str, Any]:
            """Add participant to chat room via API."""
            try:
                room_id = args.get("room_id", "")
                name = args.get("name", "")
                role = args.get("role", "member")

                tools = _get_tools(room_id)
                if not tools:
                    return _make_error(f"No tools available for room {room_id}")

                result = await tools.add_participant(name, role)

                return _make_result(
                    {
                        "status": "success",
                        "message": f"Participant '{name}' added as {role}",
                        **result,
                    }
                )

            except Exception as e:
                logger.error(f"add_participant failed: {e}", exc_info=True)
                return _make_error(str(e))

        @tool(
            "remove_participant",
            get_tool_description("remove_participant"),
            {"room_id": str, "name": str},
        )
        async def remove_participant(args: dict[str, Any]) -> dict[str, Any]:
            """Remove participant from chat room via API."""
            try:
                room_id = args.get("room_id", "")
                name = args.get("name", "")

                tools = _get_tools(room_id)
                if not tools:
                    return _make_error(f"No tools available for room {room_id}")

                result = await tools.remove_participant(name)

                return _make_result(
                    {
                        "status": "success",
                        "message": f"Participant '{name}' removed",
                        **result,
                    }
                )

            except Exception as e:
                logger.error(f"remove_participant failed: {e}", exc_info=True)
                return _make_error(str(e))

        @tool(
            "get_participants",
            get_tool_description("get_participants"),
            {"room_id": str},
        )
        async def get_participants(args: dict[str, Any]) -> dict[str, Any]:
            """Get participants in chat room via API."""
            try:
                room_id = args.get("room_id", "")

                tools = _get_tools(room_id)
                if not tools:
                    return _make_error(f"No tools available for room {room_id}")

                participants = await tools.get_participants()

                return _make_result(
                    {
                        "status": "success",
                        "participants": participants,
                        "count": len(participants),
                    }
                )

            except Exception as e:
                logger.error(f"get_participants failed: {e}", exc_info=True)
                return _make_error(str(e))

        @tool(
            "lookup_peers",
            get_tool_description("lookup_peers"),
            {"room_id": str, "page": int, "page_size": int},
        )
        async def lookup_peers(args: dict[str, Any]) -> dict[str, Any]:
            """Look up available peers via API."""
            try:
                room_id = args.get("room_id", "")
                page = args.get("page", 1)
                page_size = args.get("page_size", 50)

                tools = _get_tools(room_id)
                if not tools:
                    return _make_error(f"No tools available for room {room_id}")

                result = await tools.lookup_peers(page, page_size)

                return _make_result({"status": "success", **result})

            except Exception as e:
                logger.error(f"lookup_peers failed: {e}", exc_info=True)
                return _make_error(str(e))

        # Combine built-in thenvoi tools with custom tools
        all_tools = [
            send_message,
            send_event,
            add_participant,
            remove_participant,
            get_participants,
            lookup_peers,
        ]

        # Add custom tools
        all_tools.extend(self.custom_tools)

        server = create_sdk_mcp_server(
            name="thenvoi",
            version="1.0.0",
            tools=all_tools,
        )

        logger.info(
            f"Thenvoi MCP SDK server created with {len(all_tools)} tools "
            f"(6 built-in + {len(self.custom_tools)} custom)"
        )

        return server

    # --- Adapted from ThenvoiClaudeSDKAgent._handle_message ---
    async def on_message(
        self,
        msg: PlatformMessage,
        tools: AgentToolsProtocol,
        history: str,  # We ignore this, handle history ourselves
        participants_msg: str | None,
        *,
        is_session_bootstrap: bool,
        room_id: str,
    ) -> None:
        """
        Handle incoming message.

        - Store tools for MCP server access
        - Get or create ClaudeSDKClient for this room
        - Include room_id in the message so Claude can pass it to tools
        - Stream response and log events (tools execute via MCP)
        """
        logger.debug(f"Handling message {msg.id} in room {room_id}")

        if not self._session_manager:
            logger.error("Session manager not initialized")
            return

        # Store tools for MCP server access
        self._room_tools[room_id] = tools

        # Get stored session_id for potential resume (only on bootstrap/reconnect)
        stored_session_id = (
            self._session_ids.get(room_id) if is_session_bootstrap else None
        )

        # Get or create Claude SDK client for this room (optionally resuming)
        client = await self._session_manager.get_or_create_session(
            room_id, resume_session_id=stored_session_id
        )

        # Add room_id context (Claude needs this for tool calls)
        room_context = f"[room_id: {room_id}]"

        # Initialize history for this room on first message
        if is_session_bootstrap:
            if history:  # Already converted to text by SimpleAdapter
                self._session_context[room_id] = history
                logger.info(
                    f"Room {room_id}: Loaded historical context ({len(history)} chars)"
                )
            else:
                self._session_context[room_id] = ""
        elif room_id not in self._session_context:
            # Safety: ensure context exists even if not first message
            self._session_context[room_id] = ""

        # Build message with context
        messages_to_send = []

        # Include historical context on first message
        if is_session_bootstrap and self._session_context.get(room_id):
            messages_to_send.append(
                f"[Previous conversation context:]\n{self._session_context[room_id]}"
            )

        # Inject participants message if changed
        if participants_msg:
            messages_to_send.append(f"{room_context}[System]: {participants_msg}")
            logger.info(f"Room {room_id}: Participants updated")

        # Add current message with room_id context
        user_message = f"{room_context}{msg.format_for_llm()}"
        messages_to_send.append(user_message)

        # Send combined message to Claude
        full_message = "\n\n".join(messages_to_send)

        logger.info(
            f"Room {room_id}: Sending query to Claude SDK "
            f"(first_msg={is_session_bootstrap}, parts={len(messages_to_send)})"
        )

        try:
            # Send query to Claude
            await client.query(full_message)

            # Process streaming response (MCP tools handle execution)
            await self._process_response(client, room_id, tools)

        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            await self._report_error(tools, str(e))
            raise

        logger.debug(f"Message {msg.id} processed successfully")

    # --- Copied from ThenvoiClaudeSDKAgent._process_response ---
    async def _process_response(
        self, client: ClaudeSDKClient, room_id: str, tools: AgentToolsProtocol
    ) -> None:
        """
        Process streaming response from Claude SDK.

        MCP tools handle actual execution - we log and optionally report events here.
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
                            # Report thinking as event if enabled
                            if self.enable_execution_reporting:
                                try:
                                    await tools.send_event(
                                        content=block.thinking,
                                        message_type="thought",
                                    )
                                except Exception as e:
                                    logger.warning(
                                        f"Failed to send thinking event: {e}"
                                    )

                    elif isinstance(block, ToolUseBlock):
                        logger.info(
                            f"Room {room_id}: Tool call: {block.name} "
                            f"with {str(block.input)[:100]}..."
                        )
                        if self.enable_execution_reporting:
                            try:
                                await tools.send_event(
                                    content=json.dumps(
                                        {
                                            "name": block.name,
                                            "args": block.input,
                                            "tool_call_id": block.id,
                                        }
                                    ),
                                    message_type="tool_call",
                                )
                            except Exception as e:
                                logger.warning(f"Failed to send tool_call event: {e}")

                    elif isinstance(block, ToolResultBlock):
                        logger.debug(
                            f"Room {room_id}: Tool result: "
                            f"{block.tool_use_id[:20]}... error={block.is_error}"
                        )
                        if self.enable_execution_reporting:
                            try:
                                await tools.send_event(
                                    content=json.dumps(
                                        {
                                            "output": block.content,
                                            "tool_call_id": block.tool_use_id,
                                        }
                                    ),
                                    message_type="tool_result",
                                )
                            except Exception as e:
                                logger.warning(f"Failed to send tool_result event: {e}")

            elif isinstance(sdk_message, ResultMessage):
                logger.info(
                    f"Room {room_id}: Complete - "
                    f"{sdk_message.duration_ms}ms, "
                    f"${sdk_message.total_cost_usd or 0:.4f}"
                )
                # Capture session_id for potential resume
                if sdk_message.session_id:
                    self._session_ids[room_id] = sdk_message.session_id
                    logger.debug(
                        f"Room {room_id}: Captured session_id {sdk_message.session_id}"
                    )
                break

    # --- Copied from ThenvoiClaudeSDKAgent._cleanup_session ---
    async def on_cleanup(self, room_id: str) -> None:
        """Clean up Claude SDK session and stored tools when agent leaves a room."""
        if self._session_manager:
            await self._session_manager.cleanup_session(room_id)
        if room_id in self._room_tools:
            del self._room_tools[room_id]
        if room_id in self._session_context:
            del self._session_context[room_id]
        if room_id in self._session_ids:
            del self._session_ids[room_id]
        logger.debug(f"Room {room_id}: Cleaned up Claude SDK session")

    # --- Copied from BaseFrameworkAgent._report_error ---
    async def _report_error(self, tools: AgentToolsProtocol, error: str) -> None:
        """Send error event (best effort)."""
        try:
            await tools.send_event(content=f"Error: {error}", message_type="error")
        except Exception:
            pass

    async def cleanup_all(self) -> None:
        """Cleanup all sessions (call on stop)."""
        if self._session_manager:
            await self._session_manager.stop()
        self._room_tools.clear()
        self._session_context.clear()
        self._session_ids.clear()
