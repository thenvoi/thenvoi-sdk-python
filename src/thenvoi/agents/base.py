"""Base class for framework-specific Thenvoi agents."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from thenvoi.platform.link import ThenvoiLink
from thenvoi.platform.event import MessageEvent, PlatformEvent
from thenvoi.runtime.runtime import AgentRuntime
from thenvoi.runtime.execution import ExecutionContext
from thenvoi.runtime.tools import AgentTools
from thenvoi.runtime.types import AgentConfig, PlatformMessage, SessionConfig
from thenvoi.runtime.formatters import format_history_for_llm
from thenvoi.integrations.base import check_and_format_participants

logger = logging.getLogger(__name__)


class BaseFrameworkAgent(ABC):
    """
    Base class for framework agents.

    Uses the new runtime layer:
    - ThenvoiLink: WebSocket + REST transport
    - AgentRuntime: Room presence + execution management
    - ExecutionContext: Per-room context and event handling
    - AgentTools: Tool interface for LLM

    Handles:
    - AgentRuntime lifecycle (start/stop/run)
    - First message detection + history loading
    - Participant change detection

    Subclasses implement:
    - _handle_message() - Framework-specific processing
    - _cleanup_session() - Framework-specific cleanup (optional)
    - _on_started() - Post-start setup (optional)
    """

    def __init__(
        self,
        agent_id: str,
        api_key: str,
        ws_url: str = "wss://api.thenvoi.com/ws",
        rest_url: str = "https://api.thenvoi.com",
        config: AgentConfig | None = None,
        session_config: SessionConfig | None = None,
    ):
        self._agent_id = agent_id
        self._api_key = api_key
        self._ws_url = ws_url
        self._rest_url = rest_url
        self._config = config or AgentConfig()
        self._session_config = session_config or SessionConfig()

        # Will be set during start()
        self._link: ThenvoiLink | None = None
        self._runtime: AgentRuntime | None = None
        self._agent_name: str = ""
        self._agent_description: str = ""

    # --- Properties ---

    @property
    def agent_name(self) -> str:
        """Get agent name from platform."""
        return self._agent_name

    @property
    def agent_description(self) -> str:
        """Get agent description from platform."""
        return self._agent_description

    @property
    def link(self) -> ThenvoiLink:
        """Get the ThenvoiLink instance."""
        if not self._link:
            raise RuntimeError("Agent not started")
        return self._link

    @property
    def runtime(self) -> AgentRuntime:
        """Get the AgentRuntime instance."""
        if not self._runtime:
            raise RuntimeError("Agent not started")
        return self._runtime

    # --- Lifecycle ---

    async def start(self) -> None:
        """Start the agent."""
        # Create link
        self._link = ThenvoiLink(
            agent_id=self._agent_id,
            api_key=self._api_key,
            ws_url=self._ws_url,
            rest_url=self._rest_url,
        )

        # Fetch agent metadata
        await self._fetch_agent_metadata()

        # Create runtime with our execution handler
        self._runtime = AgentRuntime(
            link=self._link,
            agent_id=self._agent_id,
            on_execute=self._dispatch_message,
            session_config=self._session_config,
            on_session_cleanup=self._cleanup_session,
        )

        await self._runtime.start()
        await self._on_started()

    async def stop(self) -> None:
        """Stop the agent."""
        if self._runtime:
            await self._runtime.stop()
        if self._link:
            await self._link.disconnect()

    async def run(self) -> None:
        """Start and run until interrupted."""
        await self.start()
        try:
            if self._link:
                await self._link.run_forever()
        finally:
            await self.stop()

    # --- Internal methods ---

    async def _fetch_agent_metadata(self) -> None:
        """Fetch agent metadata from platform."""
        if not self._link:
            raise RuntimeError("Link not initialized")

        response = await self._link.rest.agent_api.get_agent_me()
        if not response.data:
            raise RuntimeError("Failed to fetch agent metadata")

        agent = response.data
        if not agent.description:
            raise ValueError(f"Agent {self._agent_id} has no description")

        self._agent_name = agent.name
        self._agent_description = agent.description

        logger.debug(f"Fetched metadata for agent: {self._agent_name}")

    # --- Common pre-processing ---

    async def _dispatch_message(
        self, ctx: ExecutionContext, event: PlatformEvent
    ) -> None:
        """Pre-process event and delegate to framework handler."""
        # Only handle message events
        if not isinstance(event, MessageEvent):
            return

        room_id = ctx.room_id
        msg_data = event.payload
        if not msg_data:
            return

        # Skip messages from self to avoid infinite loops
        if msg_data.sender_type == "Agent" and msg_data.sender_id == self._agent_id:
            logger.debug(f"Room {room_id}: Skipping own message {msg_data.id}")
            return

        # Convert to PlatformMessage
        from datetime import datetime

        msg = PlatformMessage(
            id=msg_data.id,
            room_id=msg_data.chat_room_id,
            content=msg_data.content,
            sender_id=msg_data.sender_id,
            sender_type=msg_data.sender_type,
            sender_name=None,  # Will be hydrated if needed
            message_type=msg_data.message_type,
            metadata={
                "mentions": [
                    {"id": m.id, "username": m.username}
                    for m in msg_data.metadata.mentions
                ],
                "status": msg_data.metadata.status,
            },
            created_at=datetime.fromisoformat(
                msg_data.inserted_at.replace("Z", "+00:00")
            ),
        )

        is_first = not ctx.is_llm_initialized

        logger.debug(
            f"Room {room_id}: is_first={is_first}, "
            f"participants_changed={ctx.participants_changed()}"
        )

        # Load history on first message
        history: list[dict[str, Any]] | None = None
        if is_first:
            history = await self._load_history(ctx, msg)
            ctx.mark_llm_initialized()

        # Check participants
        participants_msg = check_and_format_participants(ctx)
        if participants_msg:
            logger.info(f"Room {room_id}: Participants updated")

        # Create tools for this context
        tools = AgentTools.from_context(ctx)

        await self._handle_message(
            msg=msg,
            tools=tools,
            ctx=ctx,
            history=history,
            participants_msg=participants_msg,
        )

    async def _load_history(
        self, ctx: ExecutionContext, msg: PlatformMessage
    ) -> list[dict[str, Any]]:
        """Load platform history for first message."""
        try:
            logger.info(f"Room {ctx.room_id}: Loading history...")
            context = await ctx.get_context()
            history = format_history_for_llm(context.messages, exclude_id=msg.id)
            logger.info(
                f"Room {ctx.room_id}: Got {len(history) if history else 0} messages"
            )
            return history or []
        except Exception as e:
            logger.warning(f"Room {ctx.room_id}: Failed to load history: {e}")
            return []

    async def _report_error(self, tools: AgentTools, error: str) -> None:
        """Send error event (best effort)."""
        try:
            await tools.send_event(content=f"Error: {error}", message_type="error")
        except Exception:
            pass

    # --- Hooks for subclasses ---

    @abstractmethod
    async def _handle_message(
        self,
        msg: PlatformMessage,
        tools: AgentTools,
        ctx: ExecutionContext,
        history: list[dict[str, Any]] | None,
        participants_msg: str | None,
    ) -> None:
        """Framework-specific message handling."""
        ...

    async def _cleanup_session(self, room_id: str) -> None:
        """Override for framework-specific cleanup."""
        pass

    async def _on_started(self) -> None:
        """Override for post-start setup."""
        pass
