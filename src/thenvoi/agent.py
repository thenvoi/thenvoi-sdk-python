"""Agent - composes runtime, preprocessor, and adapter."""

from __future__ import annotations

import logging
from types import TracebackType
from typing import TYPE_CHECKING, cast

from thenvoi.core.protocols import FrameworkAdapter, Preprocessor
from thenvoi.core.simple_adapter import SimpleAdapter
from thenvoi.runtime.platform_runtime import PlatformRuntime
from thenvoi.runtime.types import AgentConfig, SessionConfig
from thenvoi.preprocessing.default import DefaultPreprocessor

if TYPE_CHECKING:
    from thenvoi.platform.event import PlatformEvent
    from thenvoi.runtime.execution import ExecutionContext

logger = logging.getLogger(__name__)

# Default graceful shutdown timeout in seconds
DEFAULT_SHUTDOWN_TIMEOUT: float = 30.0

# Sentinel to distinguish "not set" from "explicitly set to None"
_TIMEOUT_NOT_SET: object = object()

# Type alias for shutdown timeout (float, None, or sentinel)
_ShutdownTimeout = float | None | object


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
        self._started = False
        # Tracks shutdown_timeout from run() for use in __aexit__
        # Uses sentinel to distinguish "not set" from "explicitly set to None"
        self._shutdown_timeout: _ShutdownTimeout = _TIMEOUT_NOT_SET

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

    @property
    def is_running(self) -> bool:
        """Check if agent is currently running."""
        return self._started

    async def start(self) -> None:
        """Start agent."""
        if self._started:
            logger.warning("Agent already started")
            return

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

        self._started = True
        logger.info("Agent started: %s", self._runtime.agent_name)

    async def stop(self, timeout: float | None = None) -> bool:
        """
        Stop agent with optional graceful timeout.

        If timeout is provided, waits up to that many seconds for any ongoing
        message processing to complete before stopping. If timeout is None,
        stops immediately by cancelling any in-progress processing.

        Args:
            timeout: Optional seconds to wait for graceful shutdown.
                     None means stop immediately.

        Returns:
            True if stopped gracefully (processing completed or was idle),
            False if had to cancel mid-processing after timeout.
        """
        if not self._started:
            return True

        graceful = await self._runtime.stop(timeout=timeout)
        self._started = False
        logger.info(
            "Agent stopped: %s (graceful=%s)", self._runtime.agent_name, graceful
        )
        return graceful

    async def run(
        self, shutdown_timeout: float | None = DEFAULT_SHUTDOWN_TIMEOUT
    ) -> None:
        """
        Run until interrupted.

        Args:
            shutdown_timeout: Seconds to wait for graceful shutdown on interrupt.
                              Set to None for immediate cancellation.
                              Default is 30 seconds.
        """
        self._shutdown_timeout = shutdown_timeout
        await self.start()
        try:
            await self._runtime.run_forever()
        finally:
            await self.stop(timeout=shutdown_timeout)

    # --- Async context manager ---

    async def __aenter__(self) -> "Agent":
        """
        Enter async context - start the agent.

        Example:
            async with Agent.create(...) as agent:
                await agent.run_forever()  # or just wait
        """
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """
        Exit async context - stop the agent gracefully.

        Uses the shutdown_timeout configured in run(), or DEFAULT_SHUTDOWN_TIMEOUT
        if run() was never called. If run() was called with shutdown_timeout=None,
        stops immediately without waiting.
        """
        # Use default only if run() was never called (sentinel value)
        # If run() was called with None, respect that (immediate cancellation)
        if self._shutdown_timeout is _TIMEOUT_NOT_SET:
            timeout: float | None = DEFAULT_SHUTDOWN_TIMEOUT
        else:
            # Cast is safe: at this point it's either float or None (not sentinel)
            timeout = cast(float | None, self._shutdown_timeout)
        await self.stop(timeout=timeout)

    async def run_forever(self) -> None:
        """
        Keep the agent running forever.

        Use this inside an async context manager:
            async with agent:
                await agent.run_forever()

        Or after manually calling start():
            await agent.start()
            try:
                await agent.run_forever()
            finally:
                await agent.stop()
        """
        await self._runtime.run_forever()

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
