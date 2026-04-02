"""Agent - composes runtime, preprocessor, and adapter."""

from __future__ import annotations

import logging
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _get_version
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, cast

from thenvoi.config import build_adapter_from_config, load_agent_config
from thenvoi.core.exceptions import ThenvoiConfigError
from thenvoi.core.protocols import FrameworkAdapter, Preprocessor
from thenvoi.core.simple_adapter import SimpleAdapter
from thenvoi.preprocessing.default import DefaultPreprocessor
from thenvoi.runtime.platform_runtime import PlatformRuntime
from thenvoi.runtime.types import ContactEventConfig, SessionConfig

if TYPE_CHECKING:
    from thenvoi.platform.event import PlatformEvent
    from thenvoi.runtime.execution import ExecutionContext

logger = logging.getLogger(__name__)

try:
    _SDK_VERSION = _get_version("thenvoi-sdk")
except PackageNotFoundError:
    _SDK_VERSION = "unknown"

# Default graceful shutdown timeout in seconds
DEFAULT_SHUTDOWN_TIMEOUT: float = 30.0


class _TimeoutNotSet:
    """Sentinel class to distinguish 'not set' from 'explicitly set to None'."""

    _instance: "_TimeoutNotSet | None" = None

    def __new__(cls) -> "_TimeoutNotSet":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "<TIMEOUT_NOT_SET>"


# Singleton sentinel instance
_TIMEOUT_NOT_SET: _TimeoutNotSet = _TimeoutNotSet()

# Type alias for shutdown timeout (float, None, or sentinel)
_ShutdownTimeout = float | None | _TimeoutNotSet


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
        ws_url: str = "wss://app.thenvoi.com/api/v1/socket/websocket",
        rest_url: str = "https://app.thenvoi.com",
        config: Any | None = None,
        session_config: SessionConfig | None = None,
        contact_config: ContactEventConfig | None = None,
        preprocessor: Preprocessor | None = None,
        tools: list[Any] | None = None,
        capabilities: list[str] | None = None,
        include_categories: list[str] | None = None,
        include_tools: list[str] | None = None,
        exclude_tools: list[str] | None = None,
        emit: list[str] | None = None,
        prompt: str | None = None,
        prompt_path: str | Path | None = None,
    ) -> "Agent":
        """
        Create agent with default runtime.

        Convenience factory for most users.

        Args:
            adapter: Framework adapter (e.g., PydanticAIAdapter)
            agent_id: UUID of the agent
            api_key: API key for authentication
            ws_url: WebSocket URL (default: wss://api.thenvoi.com/ws)
            rest_url: REST API URL (default: https://api.thenvoi.com)
            config: Agent configuration options
            session_config: Session lifecycle configuration
            contact_config: Contact event handling configuration.
                            Controls how contact requests and updates are processed.
                            See ContactEventConfig for strategies (DISABLED, CALLBACK, HUB_ROOM).
            preprocessor: Custom event preprocessor (default: DefaultPreprocessor)
        """
        runtime = PlatformRuntime(
            agent_id=agent_id,
            api_key=api_key,
            ws_url=ws_url,
            rest_url=rest_url,
            config=config,
            session_config=session_config,
            contact_config=contact_config,
        )
        cls._apply_composition_options(
            adapter=adapter,
            tools=tools,
            capabilities=capabilities,
            include_categories=include_categories,
            include_tools=include_tools,
            exclude_tools=exclude_tools,
            emit=emit,
            prompt=prompt,
            prompt_path=prompt_path,
        )
        return cls(
            runtime=runtime,
            adapter=adapter,
            preprocessor=preprocessor,
        )

    @classmethod
    def from_config(
        cls,
        name: str,
        *,
        adapter: FrameworkAdapter | SimpleAdapter | None = None,
        config_path: str | Path | None = None,
        ws_url: str = "wss://app.thenvoi.com/api/v1/socket/websocket",
        rest_url: str = "https://app.thenvoi.com",
        config: Any | None = None,
        session_config: SessionConfig | None = None,
        contact_config: ContactEventConfig | None = None,
        preprocessor: Preprocessor | None = None,
        tools: list[Any] | None = None,
        capabilities: list[str] | None = None,
        include_categories: list[str] | None = None,
        include_tools: list[str] | None = None,
        exclude_tools: list[str] | None = None,
        emit: list[str] | None = None,
        prompt: str | None = None,
        prompt_path: str | Path | None = None,
    ) -> "Agent":
        loaded = load_agent_config(name, config_path=config_path)
        loaded_adapter = adapter
        if loaded_adapter is None and loaded.adapter is not None:
            loaded_adapter = build_adapter_from_config(loaded.adapter)

        if loaded_adapter is None:
            raise ThenvoiConfigError(
                f"No adapter configured for agent '{name}'. Pass adapter= to Agent.from_config() or add an adapter: section to agent_config.yaml."
            )

        merged_capabilities = cls._merge_lists(loaded.capabilities, capabilities)
        merged_emit = cls._merge_lists(loaded.emit, emit)
        merged_include_categories = cls._merge_lists(
            loaded.include_categories, include_categories
        )
        merged_include_tools = cls._merge_lists(loaded.include_tools, include_tools)
        merged_exclude_tools = cls._merge_lists(loaded.exclude_tools, exclude_tools)

        resolved_prompt_path = prompt_path or loaded.prompt_path
        resolved_prompt = loaded.prompt if prompt is None else prompt
        resolved_prompt = cls._compose_prompt(
            resolved_prompt_path,
            loaded.prompt,
            prompt,
        )

        return cls.create(
            adapter=loaded_adapter,
            agent_id=str(loaded.agent_id),
            api_key=loaded.api_key,
            ws_url=ws_url,
            rest_url=rest_url,
            config=config,
            session_config=session_config,
            contact_config=contact_config,
            preprocessor=preprocessor,
            tools=tools,
            capabilities=merged_capabilities,
            include_categories=merged_include_categories,
            include_tools=merged_include_tools,
            exclude_tools=merged_exclude_tools,
            emit=merged_emit,
            prompt=resolved_prompt,
            prompt_path=resolved_prompt_path,
        )

    @staticmethod
    def _merge_lists(
        base: list[str] | None,
        override: list[str] | None,
    ) -> list[str] | None:
        if base is None and override is None:
            return None
        merged: list[str] = []
        for value in (base or []) + (override or []):
            if value not in merged:
                merged.append(value)
        return merged

    @staticmethod
    def _compose_prompt(
        loaded_prompt_path: str | Path | None,
        loaded_prompt: str | None,
        override_prompt: str | None,
    ) -> str | None:
        parts: list[str] = []
        if loaded_prompt_path:
            path = Path(loaded_prompt_path)
            if not path.exists():
                raise ThenvoiConfigError(
                    f"Prompt file not found at {path}. Update prompt_path to a readable file."
                )
            parts.append(path.read_text(encoding="utf-8").strip())
        if loaded_prompt:
            parts.append(loaded_prompt.strip())
        if override_prompt:
            parts.append(override_prompt.strip())
        if not parts:
            return None
        return "\n\n# Developer Instructions\n\n" + "\n\n".join(
            part for part in parts if part
        )

    @staticmethod
    def _apply_composition_options(
        *,
        adapter: FrameworkAdapter | SimpleAdapter,
        tools: list[Any] | None,
        capabilities: list[str] | None,
        include_categories: list[str] | None,
        include_tools: list[str] | None,
        exclude_tools: list[str] | None,
        emit: list[str] | None,
        prompt: str | None,
        prompt_path: str | Path | None,
    ) -> None:
        setattr(adapter, "custom_tools", tools or [])
        setattr(adapter, "capabilities", capabilities or [])
        setattr(adapter, "include_categories", include_categories)
        setattr(adapter, "include_tools", include_tools)
        setattr(adapter, "exclude_tools", exclude_tools)
        setattr(adapter, "emit", emit or [])
        if prompt is not None:
            if hasattr(adapter, "custom_section"):
                setattr(adapter, "custom_section", prompt)
            elif hasattr(adapter, "config") and hasattr(
                adapter.config, "custom_section"
            ):
                adapter.config.custom_section = prompt
        if prompt_path is not None:
            setattr(adapter, "prompt_path", Path(prompt_path))

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

    @property
    def contact_config(self) -> ContactEventConfig:
        """Get the contact event configuration."""
        return self._runtime.contact_config

    @property
    def is_contacts_subscribed(self) -> bool:
        """Check if agent is subscribed to contact events."""
        return self._runtime.is_contacts_subscribed

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
        logger.info(
            "Agent started: %s (thenvoi-sdk %s)", self._runtime.agent_name, _SDK_VERSION
        )

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
