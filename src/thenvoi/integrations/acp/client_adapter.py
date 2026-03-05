"""ACP Client Adapter that forwards Thenvoi messages to external ACP agents."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from acp import spawn_agent_process, text_block

from thenvoi.converters.acp_client import ACPClientHistoryConverter
from thenvoi.core.protocols import AgentToolsProtocol
from thenvoi.core.simple_adapter import SimpleAdapter
from thenvoi.core.types import PlatformMessage
from thenvoi.integrations.acp.client_types import (
    ACPClientSessionState,
    ThenvoiACPClient,
)

logger = logging.getLogger(__name__)


class ACPClientAdapter(SimpleAdapter[ACPClientSessionState]):
    """Adapter that forwards Thenvoi messages to an external ACP agent.

    Spawns a local ACP agent process (e.g., Codex CLI, Gemini CLI, Claude
    Code, Goose) and communicates via ACP protocol over stdio. Responses
    are posted back to the Thenvoi room.

    Uses ACP SDK's spawn_agent_process for subprocess management.

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

    def __init__(
        self,
        command: str | list[str],
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        mcp_servers: list[dict[str, Any]] | None = None,
    ) -> None:
        """Initialize ACP client adapter.

        Args:
            command: Command to spawn the external ACP agent.
                     Can be a string ("codex") or list (["gemini", "cli"]).
            env: Optional environment variables for the subprocess.
            cwd: Working directory for ACP sessions (default: ".").
            mcp_servers: Optional list of MCP server configs to pass to agent.
        """
        super().__init__(history_converter=ACPClientHistoryConverter())
        self._command = command if isinstance(command, list) else [command]
        self._env = env
        self._cwd = cwd or "."
        self._mcp_servers = mcp_servers or []

        # ACP connection state
        self._conn: Any = None  # ACP agent connection
        self._client: ThenvoiACPClient | None = None
        self._ctx: Any = None  # spawn_agent_process context manager
        self._stopping = False  # Prevents double stop()

        # Room -> session mapping and prompt serialization (guarded by _session_lock)
        self._room_to_session: dict[str, str] = {}
        self._session_lock = asyncio.Lock()

    async def on_started(self, agent_name: str, agent_description: str) -> None:
        """Spawn external ACP agent process and initialize connection.

        Args:
            agent_name: Name of this agent.
            agent_description: Description of this agent.
        """
        await super().on_started(agent_name, agent_description)
        self._client = ThenvoiACPClient()  # type: ignore[abstract]  # ACP Client optional methods not needed for Phase 1

        # Use ACP SDK to spawn and connect
        # Note: spawn_agent_process is an async context manager -
        # we need to keep it alive, so we enter it manually
        self._ctx = spawn_agent_process(
            self._client,
            self._command[0],
            *self._command[1:],
            env=self._env,
        )
        self._conn, _ = await self._ctx.__aenter__()
        await self._conn.initialize(protocol_version=1)
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
        if self._conn is None:
            raise RuntimeError("ACP client not initialized. Call on_started first.")

        if is_session_bootstrap and history:
            async with self._session_lock:
                self._rehydrate(room_id, history)

        # Get or create ACP session for this room
        session_id = await self._get_or_create_session(room_id)

        # Serialize prompt + response collection to prevent buffer races
        # across concurrent rooms sharing the same _client instance.
        async with self._session_lock:
            # Reset client response buffer and wire up permission handler
            if self._client:
                self._client.reset()
                self._client.set_permission_handler(
                    self._make_permission_handler(tools, room_id)
                )

            # Send prompt to external ACP agent
            try:
                await self._conn.prompt(
                    session_id=session_id,
                    prompt=[text_block(msg.content)],
                )

                # Collect rich response chunks from ThenvoiACPClient's buffer
                if self._client:
                    chunks = self._client.get_collected_chunks()
                    mentions = [{"id": msg.sender_id, "name": msg.sender_name or ""}]
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
                await tools.send_event(
                    content=f"ACP agent error: {e}",
                    message_type="error",
                    metadata={"acp_error": str(e)},
                )
                return

        # Emit task event for session rehydration (outside lock — no shared state)
        await tools.send_event(
            content="ACP client session",
            message_type="task",
            metadata={
                "acp_client_session_id": session_id,
                "acp_client_room_id": room_id,
            },
        )

    def _make_permission_handler(self, tools: AgentToolsProtocol, room_id: str) -> Any:
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

    async def _get_or_create_session(self, room_id: str) -> str:
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

            session = await self._conn.new_session(
                cwd=self._cwd,
                mcp_servers=self._mcp_servers,
            )
            self._room_to_session[room_id] = session.session_id
            logger.info(
                "Created ACP session %s for room %s",
                session.session_id,
                room_id,
            )
            return session.session_id

    async def on_cleanup(self, room_id: str) -> None:
        """Remove room -> session mapping. Stops subprocess on last room.

        Args:
            room_id: The room identifier.
        """
        should_stop = False
        async with self._session_lock:
            self._room_to_session.pop(room_id, None)
            if not self._room_to_session and self._ctx:
                should_stop = True

        logger.debug("Cleaned up ACP client resources for room %s", room_id)

        if should_stop:
            await self.stop()

    async def stop(self) -> None:
        """Clean shutdown of ACP agent process. Idempotent."""
        if self._stopping or not self._ctx:
            return
        self._stopping = True
        try:
            await self._ctx.__aexit__(None, None, None)
        except Exception:
            logger.exception("Error during ACP agent shutdown")
        self._ctx = None
        self._conn = None
        self._stopping = False
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
