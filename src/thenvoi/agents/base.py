"""Base class for framework-specific Thenvoi agents."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from thenvoi.core.agent import ThenvoiAgent
from thenvoi.core.session import AgentSession
from thenvoi.core.types import AgentConfig, AgentTools, PlatformMessage, SessionConfig
from thenvoi.integrations.base import check_and_format_participants

logger = logging.getLogger(__name__)


class BaseFrameworkAgent(ABC):
    """
    Base class for framework agents.

    Handles:
    - ThenvoiAgent lifecycle (start/stop/run)
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
        self.thenvoi = ThenvoiAgent(
            agent_id=agent_id,
            api_key=api_key,
            ws_url=ws_url,
            rest_url=rest_url,
            config=config,
            session_config=session_config,
        )

    # --- Properties ---

    @property
    def agent_name(self) -> str:
        """Get agent name from Thenvoi coordinator."""
        return self.thenvoi.agent_name

    @property
    def agent_description(self) -> str:
        """Get agent description from Thenvoi coordinator."""
        return self.thenvoi.agent_description

    # --- Lifecycle ---

    async def start(self) -> None:
        """Start the agent."""
        self.thenvoi._on_session_cleanup = self._cleanup_session
        await self.thenvoi.start(on_message=self._dispatch_message)
        await self._on_started()

    async def stop(self) -> None:
        """Stop the agent."""
        await self.thenvoi.stop()

    async def run(self) -> None:
        """Start and run until interrupted."""
        await self.start()
        try:
            await self.thenvoi.run()
        finally:
            await self.stop()

    # --- Common pre-processing ---

    async def _dispatch_message(self, msg: PlatformMessage, tools: AgentTools) -> None:
        """Pre-process message and delegate to framework handler."""
        room_id = msg.room_id
        session = self.thenvoi.active_sessions.get(room_id)

        if not session:
            logger.warning(f"Room {room_id}: No session found")
            return

        is_first = not session.is_llm_initialized

        logger.debug(
            f"Room {room_id}: is_first={is_first}, "
            f"participants_changed={session.participants_changed()}"
        )

        # Load history on first message
        history: list[dict[str, Any]] | None = None
        if is_first:
            history = await self._load_history(session, msg)
            session.mark_llm_initialized()

        # Check participants
        participants_msg = check_and_format_participants(session)
        if participants_msg:
            logger.info(f"Room {room_id}: Participants updated")

        await self._handle_message(
            msg=msg,
            tools=tools,
            session=session,
            history=history,
            participants_msg=participants_msg,
        )

    async def _load_history(
        self, session: AgentSession, msg: PlatformMessage
    ) -> list[dict[str, Any]]:
        """Load platform history for first message."""
        try:
            logger.info(f"Room {session.room_id}: Loading history...")
            history = await session.get_history_for_llm(exclude_message_id=msg.id)
            logger.info(
                f"Room {session.room_id}: Got {len(history) if history else 0} messages"
            )
            return history or []
        except Exception as e:
            logger.warning(f"Room {session.room_id}: Failed to load history: {e}")
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
        session: AgentSession,
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
