"""Main bridge orchestrator — connects to platform, routes @mentions to handlers."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, field_validator

from thenvoi.client.rest import DEFAULT_REQUEST_OPTIONS
from thenvoi.core.nonfatal import NonFatalErrorRecorder
from thenvoi.platform.link import ThenvoiLink
from thenvoi.runtime.tools import AgentTools

from .bridge_event_dispatcher import BridgeEventDispatcher
from .event_pump import ShutdownAwareEventPump
from .health import HealthServer
from .message_dedup import MessageDeduplicator
from .participant_directory import ParticipantDirectory, ParticipantRecord
from .reconnect_supervisor import ReconnectSupervisor
from .router import MentionRouter
from .session import InMemorySessionStore

if TYPE_CHECKING:
    from thenvoi.client.streaming import MessageCreatedPayload

    from .handler import BaseHandler

logger = logging.getLogger(__name__)


class BridgeConfig(BaseModel):
    """Bridge configuration loaded from environment variables."""

    agent_id: str
    api_key: str = Field(repr=False)
    ws_url: str = "wss://app.thenvoi.com/api/v1/socket/websocket"
    rest_url: str = "https://app.thenvoi.com"
    agent_mapping: str
    health_port: int = 8080
    health_host: str = "0.0.0.0"
    session_ttl: float = 86400.0  # 24 hours; 0 disables eviction
    handler_timeout: float = 300.0  # 5 minutes; 0 disables timeout

    @field_validator("agent_id")
    @classmethod
    def validate_agent_id(cls, v: str) -> str:
        """Validate agent_id is non-empty."""
        if not v.strip():
            raise ValueError("THENVOI_AGENT_ID is required and cannot be empty")
        return v

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Validate api_key is non-empty."""
        if not v.strip():
            raise ValueError("THENVOI_API_KEY is required and cannot be empty")
        return v

    @field_validator("agent_mapping")
    @classmethod
    def validate_agent_mapping(cls, v: str) -> str:
        """Validate agent_mapping is non-empty."""
        if not v.strip():
            raise ValueError("AGENT_MAPPING is required and cannot be empty")
        return v

    @field_validator("health_port")
    @classmethod
    def validate_health_port(cls, v: int) -> int:
        """Validate health_port is in valid TCP port range."""
        if not (1 <= v <= 65535):
            raise ValueError(f"HEALTH_PORT must be between 1 and 65535, got: {v}")
        return v

    @field_validator("session_ttl")
    @classmethod
    def validate_session_ttl(cls, v: float) -> float:
        """Validate session_ttl is non-negative (0 disables eviction)."""
        if v < 0:
            raise ValueError(f"SESSION_TTL must be non-negative, got: {v}")
        return v

    @field_validator("handler_timeout")
    @classmethod
    def validate_handler_timeout(cls, v: float) -> float:
        """Validate handler_timeout is non-negative (0 disables timeout)."""
        if v < 0:
            raise ValueError(f"HANDLER_TIMEOUT must be non-negative, got: {v}")
        return v

    @classmethod
    def from_env(cls) -> BridgeConfig:
        """Load configuration from environment variables.

        Only env vars that are explicitly set are passed to the constructor;
        unset vars fall through to model field defaults.

        Returns:
            BridgeConfig instance.

        Raises:
            ValueError: If required environment variables are missing or invalid.
        """
        required_vars = {
            "agent_id": "THENVOI_AGENT_ID",
            "api_key": "THENVOI_API_KEY",
            "agent_mapping": "AGENT_MAPPING",
        }
        missing = [
            env_var for env_var in required_vars.values() if env_var not in os.environ
        ]
        if missing:
            raise ValueError(
                f"Required environment variable(s) not set: {', '.join(missing)}"
            )

        kwargs: dict[str, Any] = {
            field: os.environ[env_var] for field, env_var in required_vars.items()
        }

        if "THENVOI_WS_URL" in os.environ:
            kwargs["ws_url"] = os.environ["THENVOI_WS_URL"]
        if "THENVOI_REST_URL" in os.environ:
            kwargs["rest_url"] = os.environ["THENVOI_REST_URL"]
        if "HEALTH_HOST" in os.environ:
            kwargs["health_host"] = os.environ["HEALTH_HOST"]

        if "HEALTH_PORT" in os.environ:
            health_port_str = os.environ["HEALTH_PORT"]
            try:
                kwargs["health_port"] = int(health_port_str)
            except ValueError:
                raise ValueError(
                    f"HEALTH_PORT must be a valid integer, got: '{health_port_str}'"
                ) from None

        if "SESSION_TTL" in os.environ:
            session_ttl_str = os.environ["SESSION_TTL"]
            try:
                kwargs["session_ttl"] = float(session_ttl_str)
            except ValueError:
                raise ValueError(
                    f"SESSION_TTL must be a valid number, got: '{session_ttl_str}'"
                ) from None

        if "HANDLER_TIMEOUT" in os.environ:
            handler_timeout_str = os.environ["HANDLER_TIMEOUT"]
            try:
                kwargs["handler_timeout"] = float(handler_timeout_str)
            except ValueError:
                raise ValueError(
                    f"HANDLER_TIMEOUT must be a valid number, got: '{handler_timeout_str}'"
                ) from None

        return cls(**kwargs)


class ReconnectConfig(BaseModel):
    """Reconnection backoff configuration."""

    initial_delay: float = 1.0
    max_delay: float = 60.0
    multiplier: float = 2.0
    jitter: float = 0.5
    max_retries: int = 0  # 0 = unlimited

    @field_validator("initial_delay")
    @classmethod
    def validate_initial_delay(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"initial_delay must be positive, got: {v}")
        return v

    @field_validator("max_delay")
    @classmethod
    def validate_max_delay(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"max_delay must be positive, got: {v}")
        return v

    @field_validator("multiplier")
    @classmethod
    def validate_multiplier(cls, v: float) -> float:
        if v < 1:
            raise ValueError(f"multiplier must be >= 1, got: {v}")
        return v

    @field_validator("jitter")
    @classmethod
    def validate_jitter(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"jitter must be non-negative, got: {v}")
        return v

    @field_validator("max_retries")
    @classmethod
    def validate_max_retries(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"max_retries must be non-negative, got: {v}")
        return v


class ThenvoiBridge(NonFatalErrorRecorder):
    """Main bridge orchestrator.

    Connects to the Thenvoi platform via WebSocket, listens for @mention
    messages, and routes them to the appropriate handler.
    """

    _DEDUP_MAX_SIZE = 10_000

    def set_link(self, link: ThenvoiLink) -> None:
        """Explicitly rebind bridge link and collaborator link dependencies."""
        self._link = link

        router = self.__dict__.get("_router")
        if router is not None:
            router.set_link(link)

        participant_directory = self.__dict__.get("_participant_directory")
        if participant_directory is not None:
            participant_directory.set_link(link)

        event_dispatcher = self.__dict__.get("_event_dispatcher")
        if event_dispatcher is not None:
            event_dispatcher.set_link(link)

    def __init__(
        self,
        config: BridgeConfig,
        handlers: dict[str, BaseHandler],
        reconnect_config: ReconnectConfig | None = None,
    ) -> None:
        """Initialize the bridge.

        Args:
            config: Bridge configuration.
            handlers: Map of handler name -> handler instance.
            reconnect_config: Optional reconnection backoff config.

        Raises:
            ValueError: If agent_mapping references handler names not in handlers dict.
        """
        self._config = config
        self._handlers = handlers
        self._reconnect = reconnect_config or ReconnectConfig()
        self._shutdown_event = asyncio.Event()
        self._connected_event = asyncio.Event()
        self._init_nonfatal_errors()
        self._participant_cache: dict[str, list[ParticipantRecord]] = {}
        self._processed_message_ids: OrderedDict[str, None] = OrderedDict()
        self._deduplicator = MessageDeduplicator(
            self._processed_message_ids,
            max_size=self._DEDUP_MAX_SIZE,
        )

        # Parse and validate agent mapping
        self._agent_mapping = MentionRouter.parse_agent_mapping(config.agent_mapping)
        self._validate_handlers()

        # Create SDK link
        link = ThenvoiLink(
            agent_id=config.agent_id,
            api_key=config.api_key,
            ws_url=config.ws_url,
            rest_url=config.rest_url,
        )
        self._link = link

        # Session store — default 24h TTL prevents leaks if room-removed events
        # are missed during network interruptions. TTL of 0 disables eviction.
        effective_ttl = config.session_ttl if config.session_ttl > 0 else None
        self._session_store = InMemorySessionStore(session_ttl=effective_ttl)
        self._participant_directory = ParticipantDirectory(
            link,
            self._participant_cache,
        )

        # Router
        effective_timeout = (
            config.handler_timeout if config.handler_timeout > 0 else None
        )
        self._router = MentionRouter(
            agent_mapping=self._agent_mapping,
            handlers=self._handlers,
            session_store=self._session_store,
            agent_id=config.agent_id,
            link=link,
            handler_timeout=effective_timeout,
        )
        self._event_dispatcher = BridgeEventDispatcher(
            link=link,
            participant_directory=self._participant_directory,
            session_store=self._session_store,
            on_message=self._on_message,
        )
        self._event_pump = ShutdownAwareEventPump(self._shutdown_event)
        self._reconnect_supervisor = ReconnectSupervisor(self._reconnect)

        # Health server
        self._health = HealthServer(
            self._link,
            port=config.health_port,
            host=config.health_host,
            session_store=self._session_store,
            handler_count=len(self._handlers),
        )

        # Keep dependency rebinding explicit and testable.
        self.set_link(link)

    @property
    def _shutting_down(self) -> bool:
        return self._shutdown_event.is_set()

    def _validate_handlers(self) -> None:
        """Validate that all mapped handler names have registered handlers."""
        for agent_name, handler_name in self._agent_mapping.items():
            if handler_name not in self._handlers:
                raise ValueError(
                    f"Agent '{agent_name}' maps to handler '{handler_name}', "
                    f"but no handler with that name is registered. "
                    f"Available handlers: {list(self._handlers.keys())}"
                )

    async def run(self) -> None:
        """Start the bridge and run until shutdown.

        Starts the health server and event loop concurrently.
        Handles SIGINT/SIGTERM for graceful shutdown.
        """
        loop = asyncio.get_running_loop()

        # add_signal_handler is only available on Unix (Linux/macOS).
        # On Windows this raises NotImplementedError; fall back to no-op.
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._request_shutdown)
            except NotImplementedError:
                logger.debug(
                    "Signal handler for %s not supported on this platform", sig.name
                )

        if not self._handlers:
            logger.warning("Bridge starting with no handlers registered")

        logger.info(
            "Starting bridge with agent_id=%s, mapping=%s",
            self._config.agent_id,
            self._agent_mapping,
        )

        try:
            await self._health.start()
            await self._run_with_reconnect()
        finally:
            await self._shutdown()

    def _request_shutdown(self) -> None:
        """Signal handler for graceful shutdown."""
        if not self._shutdown_event.is_set():
            logger.info("Shutdown requested")
            self._shutdown_event.set()

    async def _shutdown(self) -> None:
        """Perform graceful shutdown."""
        logger.info("Shutting down bridge...")
        try:
            await self._link.disconnect()
        except Exception as error:
            self._record_nonfatal_error("shutdown_disconnect", error)
        await self._health.stop()
        logger.info("Bridge shutdown complete")

    async def _run_with_reconnect(self) -> None:
        """Run the event loop with exponential backoff reconnection."""
        await self._reconnect_supervisor.run(
            connect_once=self._connect_and_consume,
            disconnect=self._link.disconnect,
            connected_event=self._connected_event,
            shutdown_event=self._shutdown_event,
        )

    async def _connect_and_consume(self) -> None:
        """Connect to platform and consume events."""
        # Clear stale participant data from the previous connection.
        # The cache is re-populated below via _cache_room_participants
        # after re-subscribing to existing rooms.
        self._participant_directory.clear()
        await self._link.connect()
        self._connected_event.set()

        # Subscribe to agent room events (room added/removed)
        await self._link.subscribe_agent_rooms(self._config.agent_id)

        # Subscribe to existing rooms in parallel to reduce startup latency
        existing_rooms = await self._fetch_existing_rooms()
        if existing_rooms:
            await asyncio.gather(
                *[self._link.subscribe_room(rid) for rid in existing_rooms]
            )
            # Pre-populate participant cache to avoid per-message REST calls
            await asyncio.gather(
                *[self._participant_directory.preload_room(rid) for rid in existing_rooms]
            )
            logger.info("Subscribed to %d existing rooms", len(existing_rooms))

        logger.info("Bridge connected and listening for events")
        await self._event_pump.run(
            self._link,
            handle_event=self._handle_event,
            next_event=anext,
        )

    async def _fetch_existing_rooms(self) -> list[str]:
        """Fetch the list of rooms the agent is already in.

        Returns:
            List of room IDs.
        """
        try:
            response = await self._link.rest.agent_api_chats.list_agent_chats(
                request_options=DEFAULT_REQUEST_OPTIONS,
            )
            if response.data:
                return [room.id for room in response.data]
        except Exception as error:
            self._record_nonfatal_error(
                "fetch_existing_rooms",
                error,
                agent_id=self._config.agent_id,
            )
        return []

    async def _handle_event(self, event: object) -> None:
        """Dispatch a platform event to the appropriate handler.

        Args:
            event: A PlatformEvent from the link.
        """
        await self._event_dispatcher.dispatch(event)

    async def _on_message(self, room_id: str, payload: MessageCreatedPayload) -> None:
        """Handle an incoming message event.

        Args:
            room_id: The room the message was received in.
            payload: MessageCreatedPayload from the platform.
        """
        # Deduplicate messages (reconnect may redeliver the same event)
        if self._is_duplicate(payload.id):
            logger.debug("Skipping duplicate message %s", payload.id)
            return

        # Quick pre-checks to avoid unnecessary AgentTools creation when
        # no handlers will be dispatched.  The router repeats these checks
        # authoritatively; these are just an optimisation.
        if payload.sender_id == self._config.agent_id:
            return
        if not payload.metadata or not payload.metadata.mentions:
            return

        participants = await self._participant_directory.get_for_room(room_id)
        sender_name = self._participant_directory.resolve_sender_name(
            participants,
            payload.sender_id,
        )

        tools = AgentTools(
            room_id=room_id,
            rest=self._link.rest,
            participants=participants,
        )

        await self._router.route(payload, room_id, tools, sender_name=sender_name)

    def _is_duplicate(self, message_id: str) -> bool:
        """Check if a message has already been processed (reconnect dedup).

        Uses a bounded OrderedDict so memory stays capped even under
        sustained high throughput.
        """
        return self._deduplicator.seen(message_id)

    async def _cache_room_participants(self, room_id: str) -> None:
        """Fetch and cache participants for a room.

        Errors are logged but do not propagate — the cache miss path in
        ``_on_message`` will fall back to a REST call per message.
        """
        await self._participant_directory.preload_room(room_id)

    async def _get_room_participants(self, room_id: str) -> list[ParticipantRecord]:
        """Fetch participants for a room.

        Args:
            room_id: The room ID.

        Returns:
            List of ParticipantRecord dicts with id, name, type.
        """
        return await self._participant_directory.fetch_room(room_id)


async def main(handlers: dict[str, BaseHandler]) -> None:
    """Bridge entry point.

    Users should call this from their own script with registered handlers::

        import asyncio
        from bridge_core.bridge import main
        from my_handlers import MyHandler

        asyncio.run(main(handlers={"my_handler": MyHandler()}))

    Args:
        handlers: Map of handler name -> handler instance.
    """
    from dotenv import load_dotenv

    load_dotenv()

    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    logger = logging.getLogger(__name__)
    try:
        config = BridgeConfig.from_env()
    except ValueError:
        logger.exception("Bridge configuration error")
        raise

    bridge = ThenvoiBridge(config=config, handlers=handlers)
    await bridge.run()
