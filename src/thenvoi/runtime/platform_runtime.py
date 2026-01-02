"""Platform runtime - framework-agnostic connectivity."""

from __future__ import annotations

import logging
from typing import Callable, Awaitable

from thenvoi.platform.link import ThenvoiLink
from thenvoi.platform.event import PlatformEvent
from thenvoi.runtime.runtime import AgentRuntime
from thenvoi.runtime.execution import ExecutionContext
from thenvoi.runtime.types import AgentConfig, SessionConfig

logger = logging.getLogger(__name__)


class PlatformRuntime:
    """
    Manages platform connectivity.

    Handles:
    - ThenvoiLink (WebSocket + REST)
    - AgentRuntime (room presence, execution)
    - Agent metadata fetching

    Does NOT handle:
    - Message preprocessing
    - History conversion
    - LLM framework logic
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

        self._link: ThenvoiLink | None = None
        self._runtime: AgentRuntime | None = None
        self._agent_name: str = ""
        self._agent_description: str = ""

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def agent_name(self) -> str:
        return self._agent_name

    @property
    def agent_description(self) -> str:
        return self._agent_description

    @property
    def link(self) -> ThenvoiLink:
        if not self._link:
            raise RuntimeError("Runtime not started")
        return self._link

    @property
    def runtime(self) -> AgentRuntime:
        if not self._runtime:
            raise RuntimeError("Runtime not started")
        return self._runtime

    async def initialize(self) -> None:
        """
        Initialize link and fetch agent metadata without starting message processing.

        Call this before start() to access agent name/description before
        the runtime begins processing messages. This enables adapters to
        initialize their system prompts before any messages arrive.

        This method is idempotent - safe to call multiple times.

        Note: This only creates the REST client and fetches metadata.
        WebSocket connection happens later in start() when RoomPresence starts.
        """
        if self._link:
            return  # Already initialized

        self._link = ThenvoiLink(
            agent_id=self._agent_id,
            api_key=self._api_key,
            ws_url=self._ws_url,
            rest_url=self._rest_url,
        )

        await self._fetch_agent_metadata()
        logger.debug(f"Platform runtime initialized for agent: {self._agent_name}")

    async def start(
        self,
        on_execute: Callable[[ExecutionContext, PlatformEvent], Awaitable[None]],
        on_cleanup: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        """
        Start platform runtime (begin processing messages).

        Call initialize() first if you need to access agent metadata
        before starting. Otherwise, this method will initialize automatically.

        Args:
            on_execute: Callback for message execution
            on_cleanup: Callback for session cleanup
        """
        # Auto-initialize if not already done
        await self.initialize()

        # Type narrowing: initialize() guarantees _link is set
        assert self._link is not None

        self._runtime = AgentRuntime(
            link=self._link,
            agent_id=self._agent_id,
            on_execute=on_execute,
            session_config=self._session_config,
            on_session_cleanup=on_cleanup or self._noop_cleanup,
        )

        await self._runtime.start()
        logger.info(f"Platform runtime started for agent: {self._agent_name}")

    async def stop(self) -> None:
        """Stop platform runtime."""
        if self._runtime:
            await self._runtime.stop()
        if self._link:
            await self._link.disconnect()
        logger.info("Platform runtime stopped")

    async def run_forever(self) -> None:
        """Run until interrupted."""
        if self._link:
            await self._link.run_forever()

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

    @staticmethod
    async def _noop_cleanup(room_id: str) -> None:
        pass
