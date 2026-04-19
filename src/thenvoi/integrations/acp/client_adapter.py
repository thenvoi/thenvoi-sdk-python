"""ACP Client Adapter that forwards Thenvoi messages to external ACP agents."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from typing import ClassVar, Any, Literal, Protocol, cast

from acp import spawn_agent_process, text_block
from acp.schema import HttpMcpServer, SseMcpServer

from thenvoi.converters.acp_client import ACPClientHistoryConverter
from thenvoi.core.protocols import AgentToolsProtocol
from thenvoi.core.simple_adapter import SimpleAdapter
from thenvoi.core.types import AdapterFeatures, Capability, Emit, PlatformMessage
from thenvoi.integrations.acp.client_types import (
    ACPClientSessionState,
    ThenvoiACPClient,
)
from thenvoi.integrations.mcp.backends import (
    ThenvoiMCPBackend,
    create_thenvoi_mcp_backend,
)
from thenvoi.runtime.custom_tools import CustomToolDef
from thenvoi.runtime.mcp_server import LocalMCPServer
from thenvoi.runtime.tools import iter_tool_definitions

logger = logging.getLogger(__name__)

ACP_STDIO_LIMIT_BYTES = 16 * 1024 * 1024


class ACPConnectionProtocol(Protocol):
    """Protocol for the ACP agent connection returned by spawn_agent_process."""

    async def initialize(self, *, protocol_version: int) -> object: ...

    async def authenticate(self, *, method_id: str) -> object: ...

    async def new_session(self, *, cwd: str, mcp_servers: list[object]) -> object: ...

    async def prompt(self, *, session_id: str, prompt: list[object]) -> object: ...


class ACPSpawnContextProtocol(Protocol):
    """Protocol for the spawn_agent_process async context manager."""

    async def __aenter__(self) -> tuple[ACPConnectionProtocol, object]: ...

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> object: ...


PermissionHandler = Callable[..., Awaitable[dict[str, object]]]
LocalMcpServerConfig = HttpMcpServer | SseMcpServer
MCPTransportKind = Literal["http", "sse"]


class ACPNewSessionProtocol(Protocol):
    """Protocol for ACP session creation responses."""

    session_id: str


class ACPClientAdapter(SimpleAdapter[ACPClientSessionState]):
    """Adapter that forwards Thenvoi messages to an external ACP agent.

    Spawns a local ACP agent process (e.g., Codex CLI, Gemini CLI, Claude
    Code, Goose) and communicates via ACP protocol over stdio. Responses
    are posted back to the Thenvoi room.

    Uses ACP SDK's spawn_agent_process for subprocess management.

    Lifecycle:
        1. ``on_started()`` spawns the subprocess and initializes the ACP connection.
        2. ``on_message()`` forwards messages; respawns if the process died.
        3. ``on_cleanup(room_id)`` removes per-room state.
        4. ``stop()`` terminates the subprocess. Called internally on error
           and should be called externally when the agent is shutting down.
           After ``stop()``, the next ``on_message()`` will auto-respawn.

    Example:
        from thenvoi import Agent
        from thenvoi.integrations.acp import ACPClientAdapter

        adapter = ACPClientAdapter(
            command="codex",
            cwd="/workspace",
        )
        agent = Agent.create(
            adapter=adapter,
            agent_id="codex-bridge",
            api_key="...",
        )
        await agent.run()
    """

    SUPPORTED_EMIT: ClassVar[frozenset[Emit]] = frozenset()
    SUPPORTED_CAPABILITIES: ClassVar[frozenset[Capability]] = frozenset()

    def __init__(
        self,
        command: str | list[str],
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        mcp_servers: list[dict[str, Any]] | None = None,
        additional_tools: list[CustomToolDef] | None = None,
        rest_url: str | None = None,
        inject_thenvoi_tools: bool = True,
        auth_method: str | None = None,
        features: AdapterFeatures | None = None,
    ) -> None:
        """Initialize ACP client adapter.

        Args:
            command: Command to spawn the external ACP agent.
                     Can be a string ("codex") or list (["gemini", "cli"]).
            env: Optional environment variables for the subprocess.
            cwd: Working directory for ACP sessions. Resolved to an absolute
                 path. Defaults to the current working directory.
            mcp_servers: Optional list of MCP server configs to pass to agent.
            additional_tools: Optional custom tools to expose through the local
                              Thenvoi MCP server.
            rest_url: Thenvoi REST API base URL (default: https://app.band.ai).
            inject_thenvoi_tools: Whether to auto-inject Thenvoi MCP tools
                                  into each session via a local MCP server.
            auth_method: ACP authentication method to call after initialize.
                         Required for agents that need auth (e.g., "cursor_login"
                         for Cursor). Set to None to skip authentication.
        """
        super().__init__(
            history_converter=ACPClientHistoryConverter(),
            features=features,
        )
        self._command = command if isinstance(command, list) else [command]
        self._env = env
        self._cwd = os.path.abspath(cwd or ".")
        self._mcp_servers = list(mcp_servers or [])
        self._custom_tools: list[CustomToolDef] = list(additional_tools or [])
        self._rest_url = rest_url or "https://app.band.ai"
        self._validate_rest_url(self._rest_url)
        self._inject_thenvoi_tools = inject_thenvoi_tools
        self._auth_method = auth_method
        self._agent_mcp_transport: MCPTransportKind = "http"

        # ACP connection state
        self._conn: ACPConnectionProtocol | None = None
        self._client: ThenvoiACPClient | None = None
        self._ctx: (
            AbstractAsyncContextManager[tuple[ACPConnectionProtocol, object]] | None
        ) = None
        self._stop_lock = asyncio.Lock()  # Guards stop() to prevent TOCTOU race

        # Room -> session mapping and prompt serialization (guarded by _session_lock)
        self._room_to_session: dict[str, str] = {}
        self._room_tools: dict[str, AgentToolsProtocol] = {}
        self._thenvoi_mcp_backend: ThenvoiMCPBackend | None = None
        self._thenvoi_mcp_server: LocalMCPServer | None = None
        self._bootstrapped_sessions: set[str] = (
            set()
        )  # Sessions that received system prompt
        self._session_lock = asyncio.Lock()

    async def on_started(self, agent_name: str, agent_description: str) -> None:
        """Spawn external ACP agent process and initialize connection.

        Args:
            agent_name: Name of this agent.
            agent_description: Description of this agent.
        """
        await super().on_started(agent_name, agent_description)
        await self._spawn_process()

    async def _spawn_process(self) -> None:
        """Spawn or respawn the ACP agent subprocess.

        Safe to call when the process is already stopped — creates a
        fresh ThenvoiACPClient and enters the context manager.
        """
        self._client = ThenvoiACPClient()  # type: ignore[abstract]  # ACP Client optional methods not all implemented

        # Use ACP SDK to spawn and connect
        # Note: spawn_agent_process is an async context manager -
        # we need to keep it alive, so we enter it manually
        ctx = cast(
            AbstractAsyncContextManager[tuple[ACPConnectionProtocol, object]],
            spawn_agent_process(
                self._client,
                self._command[0],
                *self._command[1:],
                env=self._env,
                transport_kwargs={"limit": ACP_STDIO_LIMIT_BYTES},
            ),
        )
        self._ctx = ctx
        try:
            self._conn, _ = await ctx.__aenter__()
            init_response = await self._conn.initialize(protocol_version=1)
            self._agent_mcp_transport = self._select_mcp_transport(init_response)
            if self._auth_method:
                await self._conn.authenticate(method_id=self._auth_method)
                logger.info("Authenticated with method: %s", self._auth_method)
        except (asyncio.CancelledError, KeyboardInterrupt):
            try:
                await ctx.__aexit__(None, None, None)
            except Exception:
                logger.exception("Error cleaning up ACP subprocess after init cancel")
            self._ctx = None
            self._conn = None
            raise
        except Exception:
            # Ensure subprocess is cleaned up if init fails
            try:
                await ctx.__aexit__(None, None, None)
            except Exception:
                logger.exception("Error cleaning up ACP subprocess after init failure")
            self._ctx = None
            self._conn = None
            raise
        logger.info("Connected to ACP agent: %s", " ".join(self._command))

    async def on_message(
        self,
        msg: PlatformMessage,
        tools: AgentToolsProtocol,
        history: ACPClientSessionState,
        participants_msg: str | None,
        contacts_msg: str | None,
        *,
        is_session_bootstrap: bool,
        room_id: str,
    ) -> None:
        """Forward message to external ACP agent, post response to Thenvoi.

        Args:
            msg: Platform message to forward.
            tools: Agent tools for sending responses back to the platform.
            history: Converted history as ACPClientSessionState.
            participants_msg: Participants update message, or None.
            contacts_msg: Contact changes broadcast message, or None.
            is_session_bootstrap: True if this is first message from room.
            room_id: The room identifier.
        """
        conn = await self._ensure_connection()

        if is_session_bootstrap and history:
            async with self._session_lock:
                self._rehydrate(room_id, history)

        if self._inject_thenvoi_tools:
            async with self._session_lock:
                self._room_tools[room_id] = tools

        # Get or create ACP session for this room
        session_id = await self._get_or_create_session(room_id, conn)

        # Per-session buffer: reset before prompt, collect after.
        # No global lock needed — each session has its own buffer in ThenvoiACPClient.
        if self._client:
            self._client.reset_session(session_id)
            self._client.set_permission_handler(
                session_id, self._make_permission_handler(tools, room_id)
            )

        # Build prompt with system context on first message per session
        prompt_text = msg.content
        async with self._session_lock:
            needs_bootstrap = session_id not in self._bootstrapped_sessions
            if needs_bootstrap:
                self._bootstrapped_sessions.add(session_id)
        if needs_bootstrap:
            system_context = self._build_system_context(room_id, msg)
            prompt_text = f"{system_context}\n\n{msg.content}"

        # Send prompt to external ACP agent
        try:
            await conn.prompt(
                session_id=session_id,
                prompt=[text_block(prompt_text)],
            )

            # Collect rich response chunks from per-session buffer
            if self._client:
                chunks = self._client.get_collected_chunks(session_id)
                sender_name = msg.sender_name or msg.sender_id or "Unknown"
                mentions = [{"id": msg.sender_id, "name": sender_name}]
                for chunk in chunks:
                    match chunk.chunk_type:
                        case "text":
                            if chunk.content:
                                await tools.send_message(
                                    content=chunk.content,
                                    mentions=mentions,
                                )
                        case "thought":
                            await tools.send_event(
                                content=chunk.content,
                                message_type="thought",
                                metadata=chunk.metadata,
                            )
                        case "tool_call" | "tool_result":
                            await tools.send_event(
                                content=chunk.content,
                                message_type=chunk.chunk_type,
                                metadata=chunk.metadata,
                            )
                        case "plan":
                            await tools.send_event(
                                content=chunk.content,
                                message_type="task",
                                metadata=chunk.metadata,
                            )

        except Exception as e:
            logger.exception("ACP agent error: %s", e)
            # Clean up dead connection so next on_message triggers respawn
            await self.stop()
            await tools.send_event(
                content=f"ACP agent error: {e}",
                message_type="error",
                metadata={"acp_error": str(e)},
            )
            return

        # Emit task event for session rehydration
        await tools.send_event(
            content="ACP client session",
            message_type="task",
            metadata={
                "acp_client_session_id": session_id,
                "acp_client_room_id": room_id,
            },
        )

    def _make_permission_handler(
        self, tools: AgentToolsProtocol, room_id: str
    ) -> PermissionHandler:
        """Create a permission handler that posts requests to the platform.

        The handler posts permission request details as an event to the
        Thenvoi room, making them visible to other participants. It then
        auto-allows the operation since true blocking permission flow
        requires platform-level bidirectional support.

        Args:
            tools: Agent tools for posting events to the platform.
            room_id: The room to post permission events to.

        Returns:
            An async callable that handles permission requests.
        """

        async def handler(
            options: object,
            session_id: str,
            tool_call: object,
            **kwargs: object,
        ) -> dict[str, object]:
            tool_name = getattr(tool_call, "title", None) or getattr(
                tool_call, "name", "unknown"
            )
            tool_call_id = getattr(tool_call, "tool_call_id", "")

            logger.info(
                "Permission request: tool=%s, session=%s, room=%s",
                tool_name,
                session_id,
                room_id,
            )

            await tools.send_event(
                content=f"Permission requested: {tool_name}",
                message_type="tool_call",
                metadata={
                    "permission_request": True,
                    "tool_name": tool_name,
                    "tool_call_id": tool_call_id,
                    "acp_session_id": session_id,
                    "auto_allowed": True,
                },
            )

            return {"outcome": {"outcome": "allowed"}}

        return handler

    @staticmethod
    def _validate_rest_url(rest_url: str) -> None:
        """Validate the configured Thenvoi base URL."""
        if not rest_url.startswith(("http://", "https://")):
            raise ValueError("rest_url must be a valid HTTP(S) URL")

    def _build_system_context(self, room_id: str, msg: PlatformMessage) -> str:
        """Build system context to prepend to the first prompt in a session.

        Provides the external ACP agent with identity, room context, and
        tool usage instructions so it can interact with the Thenvoi platform.

        Args:
            room_id: The Thenvoi room identifier.
            msg: The first platform message (used for sender context).

        Returns:
            System context string to prepend to the prompt.
        """
        from thenvoi.runtime.prompts import render_system_prompt

        agent_name = self.agent_name or "Agent"
        agent_desc = self.agent_description or "An AI assistant"
        requester_name = msg.sender_name or msg.sender_id or "Unknown"
        requester_id = msg.sender_id or "unknown"

        system_prompt = render_system_prompt(
            agent_name=agent_name,
            agent_description=agent_desc,
            include_base_instructions=False,
            features=self.features,
        )

        room_context = (
            f"\n## Room Context\n"
            f"You are connected to Thenvoi using the Thenvoi tools.\n"
            f"Use the Thenvoi tools for any visible room action. Plain text "
            f"output is not posted back to the room.\n"
            f"\n"
            f"Current room_id: {room_id}\n"
            f"Current requester name: {requester_name}\n"
            f"Current requester id: {requester_id}\n"
            f"\n"
            f"All Thenvoi tool calls must include room_id.\n"
        )

        return f"[System Context]\n{system_prompt}\n{room_context}"

    def _select_mcp_transport(self, init_response: object) -> MCPTransportKind:
        """Choose the MCP transport supported by the connected ACP agent."""
        capabilities = getattr(init_response, "agent_capabilities", None)
        mcp_capabilities = getattr(capabilities, "mcp_capabilities", None)

        if getattr(mcp_capabilities, "http", False):
            return "http"
        if getattr(mcp_capabilities, "sse", False):
            return "sse"

        return "http"

    def _build_local_mcp_server_config(
        self,
        local_server: LocalMCPServer,
    ) -> LocalMcpServerConfig:
        """Build the MCP server config supported by the connected ACP agent."""
        if self._agent_mcp_transport == "sse":
            return SseMcpServer(
                type="sse",
                name="thenvoi",
                url=local_server.sse_url,
                headers=[],
            )

        return HttpMcpServer(
            type="http",
            name="thenvoi",
            url=local_server.http_url,
            headers=[],
        )

    async def _get_or_start_thenvoi_mcp_server(
        self,
    ) -> LocalMcpServerConfig:
        """Start or reuse the shared local Thenvoi MCP server."""
        backend = self._thenvoi_mcp_backend
        if backend is None:
            backend = await create_thenvoi_mcp_backend(
                kind=self._agent_mcp_transport,
                tool_definitions=list(iter_tool_definitions(include_memory=False)),
                get_tools=self._room_tools.get,
                additional_tools=self._custom_tools,
            )
            self._thenvoi_mcp_backend = backend
            self._thenvoi_mcp_server = backend.local_server

        local_server = backend.local_server
        if local_server is None:
            raise RuntimeError("ACP MCP backend did not create a local server")

        return self._build_local_mcp_server_config(local_server)

    async def _get_or_create_session(
        self,
        room_id: str,
        conn: ACPConnectionProtocol,
    ) -> str:
        """Get existing session for room or create new one.

        Uses a lock to prevent duplicate session creation when
        concurrent messages arrive for the same room.

        Args:
            room_id: The Thenvoi room identifier.

        Returns:
            The ACP session_id for this room.
        """
        # Fast path outside lock
        if room_id in self._room_to_session:
            return self._room_to_session[room_id]

        async with self._session_lock:
            # Re-check after acquiring lock
            if room_id in self._room_to_session:
                return self._room_to_session[room_id]

            mcp_servers: list[object] = list(self._mcp_servers)
            if self._inject_thenvoi_tools:
                mcp_servers.append(await self._get_or_start_thenvoi_mcp_server())

            session = cast(
                ACPNewSessionProtocol,
                await conn.new_session(
                    cwd=self._cwd,
                    mcp_servers=mcp_servers,
                ),
            )
            self._room_to_session[room_id] = session.session_id
            logger.info(
                "Created ACP session %s for room %s (mcp_servers=%d)",
                session.session_id,
                room_id,
                len(mcp_servers),
            )
            return session.session_id

    async def on_cleanup(self, room_id: str) -> None:
        """Remove room -> session mapping and bootstrap state.

        Args:
            room_id: The room identifier.
        """
        async with self._session_lock:
            session_id = self._room_to_session.pop(room_id, None)
            self._room_tools.pop(room_id, None)
            if session_id:
                self._bootstrapped_sessions.discard(session_id)

        logger.debug("Cleaned up ACP client resources for room %s", room_id)

    async def stop(self) -> None:
        """Clean shutdown of ACP agent process. Idempotent.

        After stop(), the subprocess can be respawned by calling
        ``_spawn_process()`` again (triggered by the next ``on_message``
        when ``_conn is None``).
        """
        ctx: AbstractAsyncContextManager[tuple[ACPConnectionProtocol, object]] | None
        async with self._stop_lock:
            ctx = self._ctx
            self._ctx = None
            self._conn = None
            self._client = None
        local_mcp_server: LocalMCPServer | None
        async with self._session_lock:
            self._room_to_session.clear()
            self._room_tools.clear()
            self._bootstrapped_sessions.clear()
            backend = self._thenvoi_mcp_backend
            local_mcp_server = self._thenvoi_mcp_server
            self._thenvoi_mcp_backend = None
            self._thenvoi_mcp_server = None
        if backend is not None:
            await backend.stop()
        elif local_mcp_server is not None:
            await local_mcp_server.stop()
        if ctx is None:
            return
        try:
            await ctx.__aexit__(None, None, None)
        except Exception:
            logger.exception("Error during ACP agent shutdown")
        logger.info("ACP client adapter stopped")

    def _rehydrate(self, room_id: str, history: ACPClientSessionState) -> None:
        """Restore room -> session mappings from history.

        Args:
            room_id: The current room ID.
            history: Session state extracted from platform history.
        """
        for rid, sid in history.room_to_session.items():
            if rid not in self._room_to_session:
                self._room_to_session[rid] = sid
                logger.debug(
                    "Restored ACP client session mapping: %s -> %s",
                    rid,
                    sid,
                )

        logger.info(
            "Rehydrated ACP client state: %d room-session mappings",
            len(self._room_to_session),
        )

    async def _ensure_connection(self) -> ACPConnectionProtocol:
        """Return a stable connection snapshot, respawning if needed."""
        async with self._stop_lock:
            if self._conn is None:
                if self._ctx is None and self.agent_name:
                    logger.info("Respawning ACP agent subprocess for new room")
                    await self._spawn_process()
                else:
                    raise RuntimeError(
                        "ACP client not initialized. Call on_started first."
                    )

            conn = self._conn

        if conn is None:
            raise RuntimeError("ACP client connection dropped before prompt")
        return conn
