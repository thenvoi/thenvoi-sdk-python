"""
Claude SDK adapter using SimpleAdapter pattern.

Extracted from thenvoi.integrations.claude_sdk.agent.ThenvoiClaudeSDKAgent.

Note: This adapter is more complex than Anthropic/PydanticAI because Claude SDK
uses MCP servers which need access to tools by room_id. We store tools per-room
when on_message is called so the MCP server can access them.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from thenvoi.core.nonfatal import NonFatalErrorRecorder
from thenvoi.core.protocols import MessagingDispatchToolsProtocol
from thenvoi.core.room_state import RoomStateStore
from thenvoi.core.simple_adapter import SimpleAdapter, legacy_chat_turn_compat
from thenvoi.core.types import ChatMessageTurnContext
from thenvoi.converters.claude_sdk import (
    ClaudeSDKHistoryConverter,
    ClaudeSDKSessionState,
)
from thenvoi.adapters.optional_dependencies import ensure_optional_dependency
from thenvoi.integrations.claude_sdk.session_manager import ClaudeSessionManager
from thenvoi.integrations.claude_sdk.mcp_server import ClaudeMcpServerFactory
from thenvoi.integrations.claude_sdk.prompts import generate_claude_sdk_agent_prompt
from thenvoi.runtime.tooling.custom_tools import (
    CustomToolDef,
    get_custom_tool_name,
)
from thenvoi.runtime.tools import (
    ALL_TOOL_NAMES,
    BASE_TOOL_NAMES,
    MEMORY_TOOL_NAMES,
    mcp_tool_names,
)

_CLAUDE_SDK_IMPORT_ERROR: ImportError | None = None
_CLAUDE_SDK_INSTALL_COMMANDS = (
    "pip install claude-agent-sdk",
    "uv add claude-agent-sdk",
)

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
    _CLAUDE_SDK_IMPORT_ERROR = e
    ClaudeSDKClient = Any
    ClaudeAgentOptions = Any
    AssistantMessage = Any
    TextBlock = Any
    ThinkingBlock = Any
    ToolUseBlock = Any
    ToolResultBlock = Any
    ResultMessage = Any

    def tool(*_args: Any, **_kwargs: Any):
        def _decorator(func: Any) -> Any:
            return func

        return _decorator

    def create_sdk_mcp_server(*_args: Any, **_kwargs: Any) -> Any:
        ensure_optional_dependency(
            _CLAUDE_SDK_IMPORT_ERROR,
            package="claude-agent-sdk",
            integration="Claude SDK",
            install_commands=_CLAUDE_SDK_INSTALL_COMMANDS,
        )
        raise AssertionError("unreachable")


logger = logging.getLogger(__name__)


def _ensure_claude_sdk_available() -> None:
    """Raise a runtime error if optional Claude SDK dependency is missing."""
    ensure_optional_dependency(
        _CLAUDE_SDK_IMPORT_ERROR,
        package="claude-agent-sdk",
        integration="Claude SDK",
        install_commands=_CLAUDE_SDK_INSTALL_COMMANDS,
    )


# Tool names as constants (MCP naming convention: mcp__{server}__{tool})
# Derived from TOOL_MODELS — single source of truth
THENVOI_BASE_TOOLS: list[str] = mcp_tool_names(BASE_TOOL_NAMES)
THENVOI_MEMORY_TOOLS: list[str] = mcp_tool_names(MEMORY_TOOL_NAMES)
# All tools: chat + contacts + memory (17 total). For chat-only tools (7),
# see thenvoi.integrations.claude_sdk.tools.THENVOI_CHAT_TOOLS.
THENVOI_ALL_TOOLS: list[str] = mcp_tool_names(ALL_TOOL_NAMES)

_THENVOI_TOOLS: list[str] = THENVOI_ALL_TOOLS


@dataclass(frozen=True)
class ClaudeSDKAdapterConfig:
    """Typed configuration surface for ClaudeSDKAdapter."""

    model: str = "claude-sonnet-4-5-20250929"
    custom_section: str | None = None
    max_thinking_tokens: int | None = None
    permission_mode: Literal["default", "acceptEdits", "plan", "bypassPermissions"] = (
        "acceptEdits"
    )
    enable_execution_reporting: bool = False
    enable_memory_tools: bool = False
    history_converter: ClaudeSDKHistoryConverter | None = None
    additional_tools: list[CustomToolDef] | None = None
    cwd: str | None = None


def _as_mapping(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return result
    return {"result": result}


def _format_mcp_success_payload(
    tool_name: str,
    args: dict[str, Any],
    result: Any,
) -> dict[str, Any]:
    if tool_name == "thenvoi_send_message":
        return {"status": "success", "message": "Message sent"}
    if tool_name == "thenvoi_send_event":
        return {"status": "success", "message": "Event sent"}
    if tool_name == "thenvoi_add_participant":
        name = args.get("name", "")
        role = args.get("role", "member")
        return {
            "status": "success",
            "message": f"Participant '{name}' added as {role}",
            **_as_mapping(result),
        }
    if tool_name == "thenvoi_remove_participant":
        name = args.get("name", "")
        return {
            "status": "success",
            "message": f"Participant '{name}' removed",
            **_as_mapping(result),
        }
    if tool_name == "thenvoi_get_participants":
        participants = result if isinstance(result, list) else []
        return {
            "status": "success",
            "participants": participants,
            "count": len(participants),
        }
    if tool_name == "thenvoi_create_chatroom":
        return {
            "status": "success",
            "message": "Chat room created",
            "room_id": result,
        }
    return {"status": "success", **_as_mapping(result)}


def __getattr__(name: str) -> Any:
    if name == "THENVOI_TOOLS":
        warnings.warn(
            "THENVOI_TOOLS is deprecated, use THENVOI_ALL_TOOLS instead. "
            f"Note: this contains all {len(_THENVOI_TOOLS)} tools (chat + contacts + memory). "
            "For chat-only tools, use "
            "thenvoi.integrations.claude_sdk.tools.THENVOI_CHAT_TOOLS.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _THENVOI_TOOLS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class ClaudeSDKAdapter(
    NonFatalErrorRecorder,
    SimpleAdapter[ClaudeSDKSessionState, MessagingDispatchToolsProtocol],
):
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
        cwd: str | None = None,
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
            cwd: Working directory for Claude Code sessions. If set, Claude Code
                will operate in this directory (e.g., a mounted git repo).
        """
        _ensure_claude_sdk_available()
        super().__init__(
            history_converter=history_converter or ClaudeSDKHistoryConverter()
        )

        self.model = model
        self.custom_section = custom_section
        self.max_thinking_tokens = max_thinking_tokens
        self.permission_mode: ClaudeSDKAdapter.PermissionMode = permission_mode
        self.enable_execution_reporting = enable_execution_reporting
        self.enable_memory_tools = enable_memory_tools
        if cwd and not Path(cwd).is_dir():
            raise ValueError(f"cwd does not exist or is not a directory: {cwd}")
        self.cwd = cwd

        # Session manager and MCP server (created after start)
        self._session_manager: ClaudeSessionManager | None = None
        self._mcp_server = None

        # Per-room tools storage for MCP server access
        self._room_tools = RoomStateStore[MessagingDispatchToolsProtocol]()

        # Per-room session context (text history for Claude SDK)
        self._session_context = RoomStateStore[str]()

        # Per-room session IDs (for SDK session resume)
        self._session_ids = RoomStateStore[str]()
        # Custom tools (user-provided)
        self._custom_tools: list[CustomToolDef] = additional_tools or []
        self._init_nonfatal_errors()

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

        # Set working directory if configured
        if self.cwd:
            sdk_options.cwd = self.cwd

        # Create session manager
        self._session_manager = ClaudeSessionManager(sdk_options)

        logger.info(
            "Claude SDK adapter started for agent: %s (model=%s, thinking=%s)",
            agent_name,
            self.model,
            self.max_thinking_tokens,
        )

    def _create_mcp_server(self):
        """Create MCP SDK server using shared Claude MCP tool bindings."""
        factory = ClaudeMcpServerFactory(
            tool_decorator=tool,
            create_server=create_sdk_mcp_server,
            get_tools=self._room_tools.get,
            include_memory_tools=self.enable_memory_tools,
            custom_tools=self._custom_tools,
            format_success_payload=_format_mcp_success_payload,
        )
        return factory.create()

    # --- Adapted from ThenvoiClaudeSDKAgent._handle_message ---
    @legacy_chat_turn_compat
    async def on_message(
        self,
        turn: ChatMessageTurnContext[
            ClaudeSDKSessionState,
            MessagingDispatchToolsProtocol,
        ],
    ) -> None:
        """
        Handle incoming message.

        - Store tools for MCP server access
        - Get or create ClaudeSDKClient for this room
        - Include room_id in the message so Claude can pass it to tools
        - Stream response and log events (tools execute via MCP)
        """
        msg = turn.msg
        tools = turn.tools
        history = turn.history
        participants_msg = turn.participants_msg
        contacts_msg = turn.contacts_msg
        is_session_bootstrap = turn.is_session_bootstrap
        room_id = turn.room_id

        logger.debug("Handling message %s in room %s", msg.id, room_id)

        if not self._session_manager:
            raise RuntimeError(
                "ClaudeSDKAdapter session manager not initialized — was on_started() called?"
            )

        # Store tools for MCP server access
        self._room_tools[room_id] = tools

        # Determine session_id for resume: prefer history (persisted) then
        # in-memory cache.  Only used on bootstrap/reconnect.
        stored_session_id: str | None = None
        if is_session_bootstrap:
            stored_session_id = history.session_id or self._session_ids.get(room_id)

        # Get or create Claude SDK client for this room (optionally resuming)
        try:
            client = await self._session_manager.get_or_create_session(
                room_id, resume_session_id=stored_session_id
            )
        except Exception as resume_exc:
            if stored_session_id:
                logger.warning(
                    "Room %s: Session resume failed (session_id=%s): %s. "
                    "Creating new session",
                    room_id,
                    stored_session_id,
                    resume_exc,
                )
                client = await self._session_manager.get_or_create_session(
                    room_id, resume_session_id=None
                )
            else:
                raise

        # Add room_id context (Claude needs this for tool calls)
        room_context = f"[room_id: {room_id}]"

        # Initialize history for this room on first message
        if is_session_bootstrap:
            if history.text:  # Already converted to text by SimpleAdapter
                self._session_context[room_id] = history.text
                logger.info(
                    "Room %s: Loaded historical context (%s chars)",
                    room_id,
                    len(history.text),
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

        system_updates = self.build_metadata_updates(
            participants_msg=participants_msg,
            contacts_msg=contacts_msg,
        )
        messages_to_send.extend(
            [f"{room_context}{update}" for update in system_updates]
        )
        if system_updates:
            logger.info(
                "Room %s: Injected %d system updates", room_id, len(system_updates)
            )

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
            logger.exception("Error processing message: %s", e)
            await self.report_adapter_error(
                tools,
                error=e,
                operation="report_error_event",
            )
            raise

        logger.debug("Message %s processed successfully", msg.id)

    # --- Copied from ThenvoiClaudeSDKAgent._process_response ---
    async def _process_response(
        self,
        client: ClaudeSDKClient,
        room_id: str,
        tools: MessagingDispatchToolsProtocol,
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
                                await self.send_lifecycle_event(
                                    tools,
                                    content=block.thinking,
                                    message_type="thought",
                                    operation="thinking_event",
                                    room_id=room_id,
                                )

                    elif isinstance(block, ToolUseBlock):
                        logger.info(
                            "Room %s: Tool call: %s with %s...",
                            room_id,
                            block.name,
                            str(block.input)[:100],
                        )
                        if self.enable_execution_reporting:
                            await self.send_tool_call_event(
                                tools,
                                payload={
                                    "name": block.name,
                                    "args": block.input,
                                    "tool_call_id": block.id,
                                },
                                room_id=room_id,
                                tool_name=block.name,
                            )

                    elif isinstance(block, ToolResultBlock):
                        logger.debug(
                            "Room %s: Tool result: %s... error=%s",
                            room_id,
                            block.tool_use_id[:20],
                            block.is_error,
                        )
                        if self.enable_execution_reporting:
                            await self.send_tool_result_event(
                                tools,
                                payload={
                                    "output": block.content,
                                    "tool_call_id": block.tool_use_id,
                                },
                                room_id=room_id,
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
                    prev_session_id = self._session_ids.get(room_id)
                    self._session_ids[room_id] = sdk_message.session_id
                    logger.debug(
                        "Room %s: Captured session_id %s",
                        room_id,
                        sdk_message.session_id,
                    )
                    # Persist session_id as task event (best-effort, only on change)
                    if sdk_message.session_id != prev_session_id:
                        await self.send_lifecycle_event(
                            tools,
                            content="Claude SDK session",
                            message_type="task",
                            operation="persist_session_id_event",
                            metadata={
                                "claude_sdk_session_id": sdk_message.session_id,
                            },
                            room_id=room_id,
                        )
                break

    # --- Copied from ThenvoiClaudeSDKAgent._cleanup_session ---
    async def on_cleanup(self, room_id: str) -> None:
        """Clean up Claude SDK session and stored tools when agent leaves a room."""
        if self._session_manager:
            await self._session_manager.cleanup_session(room_id)
        self._room_tools.pop(room_id, None)
        self._session_context.pop(room_id, None)
        self._session_ids.pop(room_id, None)
        logger.debug("Room %s: Cleaned up Claude SDK session", room_id)

    async def cleanup_all(self) -> None:
        """Cleanup all sessions (call on stop)."""
        if self._session_manager:
            await self._session_manager.stop()
        self._room_tools.clear()
        self._session_context.clear()
        self._session_ids.clear()
