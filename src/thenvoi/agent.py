"""Agent - composes runtime, preprocessor, and adapter."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from thenvoi.core.protocols import FrameworkAdapter, Preprocessor
from thenvoi.core.simple_adapter import SimpleAdapter
from thenvoi.runtime.platform_runtime import PlatformRuntime
from thenvoi.runtime.types import AgentConfig, SessionConfig
from thenvoi.preprocessing.default import DefaultPreprocessor

if TYPE_CHECKING:
    from thenvoi.platform.event import PlatformEvent
    from thenvoi.runtime.execution import ExecutionContext

logger = logging.getLogger(__name__)


class Agent:
    """
    Composes platform runtime, preprocessor, and adapter.

    Two ways to create:

    1. Full composition (power users):
        agent = Agent(
            runtime=PlatformRuntime(...),
            preprocessor=CustomPreprocessor(),
            adapter=MyAdapter(),
        )

    2. Simple factory (most users):
        agent = Agent.create(
            adapter=MyAdapter(),
            agent_id="...",
            api_key="...",
        )
    """

    def __init__(
        self,
        runtime: PlatformRuntime,
        adapter: FrameworkAdapter | SimpleAdapter,
        preprocessor: Preprocessor | None = None,
    ):
        self._runtime = runtime
        self._adapter = adapter
        self._preprocessor = preprocessor or DefaultPreprocessor()

    @classmethod
    def create(
        cls,
        adapter: FrameworkAdapter | SimpleAdapter,
        agent_id: str,
        api_key: str,
        ws_url: str = "wss://api.thenvoi.com/ws",
        rest_url: str = "https://api.thenvoi.com",
        config: AgentConfig | None = None,
        session_config: SessionConfig | None = None,
        preprocessor: Preprocessor | None = None,
    ) -> "Agent":
        """
        Create agent with default runtime.

        Convenience factory for most users.
        """
        runtime = PlatformRuntime(
            agent_id=agent_id,
            api_key=api_key,
            ws_url=ws_url,
            rest_url=rest_url,
            config=config,
            session_config=session_config,
        )
        return cls(
            runtime=runtime,
            adapter=adapter,
            preprocessor=preprocessor,
        )

    @property
    def runtime(self) -> PlatformRuntime:
        return self._runtime

    @property
    def agent_name(self) -> str:
        return self._runtime.agent_name

    @property
    def agent_description(self) -> str:
        return self._runtime.agent_description

    async def start(self) -> None:
        """Start agent."""
        # 1. Initialize runtime (fetch metadata via REST, no WebSocket yet)
        await self._runtime.initialize()

        # 2. Initialize adapter with agent metadata BEFORE message processing
        await self._adapter.on_started(
            self._runtime.agent_name,
            self._runtime.agent_description,
        )

        # 3. NOW start message processing (connects WebSocket)
        await self._runtime.start(
            on_execute=self._on_execute,
            on_cleanup=self._adapter.on_cleanup,
        )

    async def stop(self) -> None:
        """Stop agent."""
        await self._runtime.stop()

    async def run(self) -> None:
        """Run until interrupted."""
        await self.start()
        try:
            await self._runtime.run_forever()
        finally:
            await self.stop()

    async def _on_execute(
        self,
        ctx: "ExecutionContext",
        event: "PlatformEvent",
    ) -> None:
        """Handle platform event."""
        # Preprocessor is the single source of truth for event filtering.
        # It returns None for non-MessageEvent types.
        inp = await self._preprocessor.process(
            ctx=ctx,
            event=event,
            agent_id=self._runtime.agent_id,
        )

        if inp is None:
            return

        await self._adapter.on_event(inp)
