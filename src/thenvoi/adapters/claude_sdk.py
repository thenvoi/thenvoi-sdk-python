"""
Claude SDK adapter using SimpleAdapter pattern.

Extracted from thenvoi.integrations.claude_sdk.agent.ThenvoiClaudeSDKAgent.

Note: This adapter is more complex than Anthropic/PydanticAI because Claude SDK
uses MCP servers which need access to tools by room_id. We store tools per-room
when on_message is called so the MCP server can access them.
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
from thenvoi.runtime.custom_tools import (
    CustomToolDef,
    execute_custom_tool,
    get_custom_tool_name,
)
from thenvoi.runtime.tools import (
    BASE_TOOL_NAMES,
    MEMORY_TOOL_NAMES,
    get_tool_description,
    mcp_tool_names,
)

logger = logging.getLogger(__name__)


# Tool names as constants (MCP naming convention: mcp__{server}__{tool})
# Derived from TOOL_MODELS — single source of truth
THENVOI_BASE_TOOLS = mcp_tool_names(BASE_TOOL_NAMES)
THENVOI_MEMORY_TOOLS = mcp_tool_names(MEMORY_TOOL_NAMES)
THENVOI_TOOLS = THENVOI_BASE_TOOLS + THENVOI_MEMORY_TOOLS


class ClaudeSDKAdapter(SimpleAdapter[str]):
    """
    Claude Agent SDK adapter using SimpleAdapter pattern.

    Uses the Claude Agent SDK for LLM interactions with MCP-based tool integration.

    Note: This adapter stores tools per-room so the MCP server can access them.
    The history is converted to a text string for context injection.

    Example:
        adapter = ClaudeSDKAdapter(
            model="claude-sonnet-4-5-20250929",
            custom_section="You are a helpful assistant.",
        )
        agent = Agent.create(adapter=adapter, agent_id="...", api_key="...")
        await agent.run()
    """

    PermissionMode = Literal["default", "acceptEdits", "plan", "bypassPermissions"]

    def __init__(
        self,
        model: str = "claude-sonnet-4-5-20250929",
        custom_section: str | None = None,
        max_thinking_tokens: int | None = None,
        permission_mode: PermissionMode = "acceptEdits",
        enable_execution_reporting: bool = False,
        enable_memory_tools: bool = False,
        history_converter: ClaudeSDKHistoryConverter | None = None,
        additional_tools: list[CustomToolDef] | None = None,
    ):
        """
        Initialize the Claude SDK adapter.

        Args:
            model: Claude model to use
            custom_section: Custom instructions added to system prompt
            max_thinking_tokens: Max tokens for extended thinking (optional)
            permission_mode: SDK permission mode
            enable_execution_reporting: If True, emit tool_call/tool_result events
            enable_memory_tools: If True, includes memory management tools (enterprise only).
                Defaults to False.
            history_converter: Optional custom history converter
            additional_tools: Optional list of custom tools as (PydanticModel, callable)
                tuples. These are converted to MCP tools internally.
        """
        super().__init__(
            history_converter=history_converter or ClaudeSDKHistoryConverter()
        )

        self.model = model
        self.custom_section = custom_section
        self.max_thinking_tokens = max_thinking_tokens
        self.permission_mode: ClaudeSDKAdapter.PermissionMode = permission_mode
        self.enable_execution_reporting = enable_execution_reporting
        self.enable_memory_tools = enable_memory_tools

        # Session manager and MCP server (created after start)
        self._session_manager: ClaudeSessionManager | None = None
        self._mcp_server = None

        # Per-room tools storage for MCP server access
        self._room_tools: dict[str, AgentToolsProtocol] = {}

        # Per-room session context (text history for Claude SDK)
        self._session_context: dict[str, str] = {}

        # Per-room session IDs (for SDK session resume)
        self._session_ids: dict[str, str] = {}

        # Custom tools (user-provided)
        self._custom_tools: list[CustomToolDef] = additional_tools or []

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

        # Build allowed tools list (platform + custom)
        allowed_tools = list(THENVOI_BASE_TOOLS)
        if self.enable_memory_tools:
            allowed_tools.extend(THENVOI_MEMORY_TOOLS)
        for custom_tool_def in self._custom_tools:
            input_model, _ = custom_tool_def
            tool_name = get_custom_tool_name(input_model)
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
            "Claude SDK adapter started for agent: %s (model=%s, thinking=%s)",
            agent_name,
            self.model,
            self.max_thinking_tokens,
        )

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
            "thenvoi_send_message",
            get_tool_description("thenvoi_send_message"),
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

                mention_handles: list[str] = []
                if mentions_str:
                    try:
                        mention_handles = (
                            json.loads(mentions_str)
                            if isinstance(mentions_str, str)
                            else mentions_str
                        )
                    except json.JSONDecodeError:
                        pass

                logger.info("[%s] send_message: %s...", room_id, content[:100])

                tools = _get_tools(room_id)
                if not tools:
                    return _make_error(f"No tools available for room {room_id}")

                await tools.send_message(content, mention_handles)

                return _make_result({"status": "success", "message": "Message sent"})

            except Exception as e:
                logger.error("send_message failed: %s", e, exc_info=True)
                return _make_error(str(e))

        @tool(
            "thenvoi_send_event",
            get_tool_description("thenvoi_send_event"),
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
                logger.error("send_event failed: %s", e, exc_info=True)
                return _make_error(str(e))

        @tool(
            "thenvoi_add_participant",
            get_tool_description("thenvoi_add_participant"),
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
                logger.error("add_participant failed: %s", e, exc_info=True)
                return _make_error(str(e))

        @tool(
            "thenvoi_remove_participant",
            get_tool_description("thenvoi_remove_participant"),
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
                logger.error("remove_participant failed: %s", e, exc_info=True)
                return _make_error(str(e))

        @tool(
            "thenvoi_get_participants",
            get_tool_description("thenvoi_get_participants"),
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
                logger.error("get_participants failed: %s", e, exc_info=True)
                return _make_error(str(e))

        @tool(
            "thenvoi_lookup_peers",
            get_tool_description("thenvoi_lookup_peers"),
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
                logger.error("lookup_peers failed: %s", e, exc_info=True)
                return _make_error(str(e))

        @tool(
            "thenvoi_create_chatroom",
            get_tool_description("thenvoi_create_chatroom"),
            {"room_id": str, "task_id": str},
        )
        async def create_chatroom(args: dict[str, Any]) -> dict[str, Any]:
            """Create a new chat room via API."""
            task_id = args.get("task_id") or None
            try:
                room_id = args.get("room_id", "")

                tools = _get_tools(room_id)
                if not tools:
                    return _make_error(f"No tools available for room {room_id}")

                new_room_id = await tools.create_chatroom(task_id)

                return _make_result(
                    {
                        "status": "success",
                        "message": "Chat room created",
                        "room_id": new_room_id,
                    }
                )

            except Exception as e:
                logger.error(
                    "create_chatroom failed (task_id=%s): %s",
                    task_id,
                    e,
                    exc_info=True,
                )
                return _make_error(str(e))

        # Contact management tools
        @tool(
            "thenvoi_list_contacts",
            get_tool_description("thenvoi_list_contacts"),
            {"room_id": str, "page": int, "page_size": int},
        )
        async def list_contacts(args: dict[str, Any]) -> dict[str, Any]:
            """List agent's contacts."""
            try:
                room_id = args.get("room_id", "")
                page = args.get("page", 1)
                page_size = args.get("page_size", 50)

                tools = _get_tools(room_id)
                if not tools:
                    return _make_error(f"No tools available for room {room_id}")

                result = await tools.list_contacts(page, page_size)
                return _make_result({"status": "success", **result})

            except Exception as e:
                logger.error("list_contacts failed: %s", e, exc_info=True)
                return _make_error(str(e))

        @tool(
            "thenvoi_add_contact",
            get_tool_description("thenvoi_add_contact"),
            {"room_id": str, "handle": str, "message": str},
        )
        async def add_contact(args: dict[str, Any]) -> dict[str, Any]:
            """Send a contact request."""
            try:
                room_id = args.get("room_id", "")
                handle = args.get("handle", "")
                message = args.get("message") or None

                tools = _get_tools(room_id)
                if not tools:
                    return _make_error(f"No tools available for room {room_id}")

                result = await tools.add_contact(handle, message)
                return _make_result({"status": "success", **result})

            except Exception as e:
                logger.error("add_contact failed: %s", e, exc_info=True)
                return _make_error(str(e))

        @tool(
            "thenvoi_remove_contact",
            get_tool_description("thenvoi_remove_contact"),
            {"room_id": str, "handle": str, "contact_id": str},
        )
        async def remove_contact(args: dict[str, Any]) -> dict[str, Any]:
            """Remove a contact."""
            try:
                room_id = args.get("room_id", "")
                handle = args.get("handle") or None
                contact_id = args.get("contact_id") or None

                tools = _get_tools(room_id)
                if not tools:
                    return _make_error(f"No tools available for room {room_id}")

                result = await tools.remove_contact(handle, contact_id)
                return _make_result({"status": "success", **result})

            except Exception as e:
                logger.error("remove_contact failed: %s", e, exc_info=True)
                return _make_error(str(e))

        @tool(
            "thenvoi_list_contact_requests",
            get_tool_description("thenvoi_list_contact_requests"),
            {"room_id": str, "page": int, "page_size": int, "sent_status": str},
        )
        async def list_contact_requests(args: dict[str, Any]) -> dict[str, Any]:
            """List contact requests."""
            try:
                room_id = args.get("room_id", "")
                page = args.get("page", 1)
                page_size = args.get("page_size", 50)
                sent_status = args.get("sent_status", "pending")

                tools = _get_tools(room_id)
                if not tools:
                    return _make_error(f"No tools available for room {room_id}")

                result = await tools.list_contact_requests(page, page_size, sent_status)
                return _make_result({"status": "success", **result})

            except Exception as e:
                logger.error("list_contact_requests failed: %s", e, exc_info=True)
                return _make_error(str(e))

        @tool(
            "thenvoi_respond_contact_request",
            get_tool_description("thenvoi_respond_contact_request"),
            {"room_id": str, "action": str, "handle": str, "request_id": str},
        )
        async def respond_contact_request(args: dict[str, Any]) -> dict[str, Any]:
            """Respond to a contact request."""
            try:
                room_id = args.get("room_id", "")
                action = args.get("action", "")
                handle = args.get("handle") or None
                request_id = args.get("request_id") or None

                tools = _get_tools(room_id)
                if not tools:
                    return _make_error(f"No tools available for room {room_id}")

                result = await tools.respond_contact_request(action, handle, request_id)
                return _make_result({"status": "success", **result})

            except Exception as e:
                logger.error("respond_contact_request failed: %s", e, exc_info=True)
                return _make_error(str(e))

        # Memory management tools
        @tool(
            "thenvoi_list_memories",
            get_tool_description("thenvoi_list_memories"),
            {
                "room_id": str,
                "subject_id": str,
                "scope": str,
                "system": str,
                "type": str,
                "segment": str,
                "content_query": str,
                "page_size": int,
                "status": str,
            },
        )
        async def list_memories(args: dict[str, Any]) -> dict[str, Any]:
            """List memories."""
            try:
                room_id = args.get("room_id", "")
                subject_id = args.get("subject_id") or None
                scope = args.get("scope") or None
                system = args.get("system") or None
                memory_type = args.get("type") or None
                segment = args.get("segment") or None
                content_query = args.get("content_query") or None
                page_size = args.get("page_size", 50)
                status = args.get("status") or None

                tools = _get_tools(room_id)
                if not tools:
                    return _make_error(f"No tools available for room {room_id}")

                result = await tools.list_memories(
                    subject_id=subject_id,
                    scope=scope,
                    system=system,
                    type=memory_type,
                    segment=segment,
                    content_query=content_query,
                    page_size=page_size,
                    status=status,
                )
                return _make_result({"status": "success", **result})

            except Exception as e:
                logger.error("list_memories failed: %s", e, exc_info=True)
                return _make_error(str(e))

        @tool(
            "thenvoi_store_memory",
            get_tool_description("thenvoi_store_memory"),
            {
                "room_id": str,
                "content": str,
                "system": str,
                "type": str,
                "segment": str,
                "thought": str,
                "scope": str,
                "subject_id": str,
            },
        )
        async def store_memory(args: dict[str, Any]) -> dict[str, Any]:
            """Store a new memory."""
            try:
                room_id = args.get("room_id", "")
                content = args.get("content", "")
                system = args.get("system", "")
                memory_type = args.get("type", "")
                segment = args.get("segment", "")
                thought = args.get("thought", "")
                scope = args.get("scope", "subject")
                subject_id = args.get("subject_id") or None

                tools = _get_tools(room_id)
                if not tools:
                    return _make_error(f"No tools available for room {room_id}")

                result = await tools.store_memory(
                    content=content,
                    system=system,
                    type=memory_type,
                    segment=segment,
                    thought=thought,
                    scope=scope,
                    subject_id=subject_id,
                )
                return _make_result({"status": "success", **result})

            except Exception as e:
                logger.error("store_memory failed: %s", e, exc_info=True)
                return _make_error(str(e))

        @tool(
            "thenvoi_get_memory",
            get_tool_description("thenvoi_get_memory"),
            {"room_id": str, "memory_id": str},
        )
        async def get_memory(args: dict[str, Any]) -> dict[str, Any]:
            """Get a specific memory."""
            try:
                room_id = args.get("room_id", "")
                memory_id = args.get("memory_id", "")

                tools = _get_tools(room_id)
                if not tools:
                    return _make_error(f"No tools available for room {room_id}")

                result = await tools.get_memory(memory_id)
                return _make_result({"status": "success", **result})

            except Exception as e:
                logger.error("get_memory failed: %s", e, exc_info=True)
                return _make_error(str(e))

        @tool(
            "thenvoi_supersede_memory",
            get_tool_description("thenvoi_supersede_memory"),
            {"room_id": str, "memory_id": str},
        )
        async def supersede_memory(args: dict[str, Any]) -> dict[str, Any]:
            """Supersede a memory."""
            try:
                room_id = args.get("room_id", "")
                memory_id = args.get("memory_id", "")

                tools = _get_tools(room_id)
                if not tools:
                    return _make_error(f"No tools available for room {room_id}")

                result = await tools.supersede_memory(memory_id)
                return _make_result({"status": "success", **result})

            except Exception as e:
                logger.error("supersede_memory failed: %s", e, exc_info=True)
                return _make_error(str(e))

        @tool(
            "thenvoi_archive_memory",
            get_tool_description("thenvoi_archive_memory"),
            {"room_id": str, "memory_id": str},
        )
        async def archive_memory(args: dict[str, Any]) -> dict[str, Any]:
            """Archive a memory."""
            try:
                room_id = args.get("room_id", "")
                memory_id = args.get("memory_id", "")

                tools = _get_tools(room_id)
                if not tools:
                    return _make_error(f"No tools available for room {room_id}")

                result = await tools.archive_memory(memory_id)
                return _make_result({"status": "success", **result})

            except Exception as e:
                logger.error("archive_memory failed: %s", e, exc_info=True)
                return _make_error(str(e))

        # Start with platform tools
        all_tools = [
            send_message,
            send_event,
            add_participant,
            remove_participant,
            get_participants,
            lookup_peers,
            create_chatroom,
            # Contact management tools
            list_contacts,
            add_contact,
            remove_contact,
            list_contact_requests,
            respond_contact_request,
        ]

        # Memory management tools (enterprise only - opt-in)
        if adapter.enable_memory_tools:
            all_tools.extend(
                [
                    list_memories,
                    store_memory,
                    get_memory,
                    supersede_memory,
                    archive_memory,
                ]
            )

        # Add custom tools wrapped as MCP tools
        for custom_tool_def in adapter._custom_tools:
            input_model, _ = custom_tool_def  # func used via execute_custom_tool
            tool_name = get_custom_tool_name(input_model)
            tool_description = input_model.__doc__ or f"Custom tool: {tool_name}"

            # Build schema from Pydantic model
            schema = input_model.model_json_schema()
            properties = schema.get("properties", {})
            # Convert to simple type dict for MCP (name: type)
            mcp_schema: dict[str, type] = {"room_id": str}  # Always include room_id
            for prop_name, prop_def in properties.items():
                prop_type = prop_def.get("type", "string")
                if prop_type == "string":
                    mcp_schema[prop_name] = str
                elif prop_type == "number":
                    mcp_schema[prop_name] = float
                elif prop_type == "integer":
                    mcp_schema[prop_name] = int
                elif prop_type == "boolean":
                    mcp_schema[prop_name] = bool
                else:
                    mcp_schema[prop_name] = str  # Default to string

            # Create MCP wrapper function
            # Need to capture variables in closure
            def make_mcp_wrapper(tool_def: CustomToolDef, name: str):
                async def mcp_wrapper(args: dict[str, Any]) -> dict[str, Any]:
                    try:
                        # Remove room_id from args (not part of custom tool input)
                        tool_args = {k: v for k, v in args.items() if k != "room_id"}

                        # Execute custom tool
                        result = await execute_custom_tool(tool_def, tool_args)

                        # Format result for MCP
                        return _make_result(result)

                    except Exception as e:
                        logger.error(
                            "Custom tool %s failed: %s", name, e, exc_info=True
                        )
                        return _make_error(str(e))

                return mcp_wrapper

            wrapper = make_mcp_wrapper(custom_tool_def, tool_name)
            wrapper.__name__ = tool_name  # Set function name for tool decorator

            # Apply @tool decorator
            decorated = tool(tool_name, tool_description, mcp_schema)(wrapper)
            all_tools.append(decorated)

            logger.debug("Registered custom MCP tool: %s", tool_name)

        server = create_sdk_mcp_server(
            name="thenvoi",
            version="1.0.0",
            tools=all_tools,
        )

        logger.info(
            "Thenvoi MCP SDK server created with %s tools (%s custom)",
            len(all_tools),
            len(adapter._custom_tools),
        )

        return server

    # --- Adapted from ThenvoiClaudeSDKAgent._handle_message ---
    async def on_message(
        self,
        msg: PlatformMessage,
        tools: AgentToolsProtocol,
        history: str,  # We ignore this, handle history ourselves
        participants_msg: str | None,
        contacts_msg: str | None,
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
        logger.debug("Handling message %s in room %s", msg.id, room_id)

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
                    "Room %s: Loaded historical context (%s chars)",
                    room_id,
                    len(history),
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
            logger.info("Room %s: Participants updated", room_id)

        # Inject contacts message if present
        if contacts_msg:
            messages_to_send.append(f"{room_context}[System]: {contacts_msg}")
            logger.info("Room %s: Contacts broadcast received", room_id)

        # Add current message with room_id context
        user_message = f"{room_context}{msg.format_for_llm()}"
        messages_to_send.append(user_message)

        # Send combined message to Claude
        full_message = "\n\n".join(messages_to_send)

        logger.info(
            "Room %s: Sending query to Claude SDK (first_msg=%s, parts=%s)",
            room_id,
            is_session_bootstrap,
            len(messages_to_send),
        )

        try:
            # Send query to Claude
            await client.query(full_message)

            # Process streaming response (MCP tools handle execution)
            await self._process_response(client, room_id, tools)

        except Exception as e:
            logger.error("Error processing message: %s", e, exc_info=True)
            await self._report_error(tools, str(e))
            raise

        logger.debug("Message %s processed successfully", msg.id)

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
                            logger.debug(
                                "Room %s: Text: %s...", room_id, block.text[:100]
                            )

                    elif isinstance(block, ThinkingBlock):
                        if block.thinking:
                            logger.debug(
                                "Room %s: Thinking: %s...",
                                room_id,
                                block.thinking[:100],
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
                                        "Failed to send thinking event: %s", e
                                    )

                    elif isinstance(block, ToolUseBlock):
                        logger.info(
                            "Room %s: Tool call: %s with %s...",
                            room_id,
                            block.name,
                            str(block.input)[:100],
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
                                logger.warning("Failed to send tool_call event: %s", e)

                    elif isinstance(block, ToolResultBlock):
                        logger.debug(
                            "Room %s: Tool result: %s... error=%s",
                            room_id,
                            block.tool_use_id[:20],
                            block.is_error,
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
                                logger.warning(
                                    "Failed to send tool_result event: %s", e
                                )

            elif isinstance(sdk_message, ResultMessage):
                logger.info(
                    "Room %s: Complete - %sms, $%.4f",
                    room_id,
                    sdk_message.duration_ms,
                    sdk_message.total_cost_usd or 0,
                )
                # Capture session_id for potential resume
                if sdk_message.session_id:
                    self._session_ids[room_id] = sdk_message.session_id
                    logger.debug(
                        "Room %s: Captured session_id %s",
                        room_id,
                        sdk_message.session_id,
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
        logger.debug("Room %s: Cleaned up Claude SDK session", room_id)

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
