"""A2A Gateway Adapter that exposes Thenvoi peers as A2A endpoints."""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterator

from a2a.types import (
    Message as A2AMessage,
    TaskStatusUpdateEvent,
)
from a2a.utils import get_message_text

from thenvoi.client.rest import (
    AsyncRestClient,
    ChatMessageRequest,
    ChatMessageRequestMentionsItem,
    DEFAULT_REQUEST_OPTIONS,
)
from thenvoi.config.defaults import DEFAULT_REST_URL
from thenvoi.converters.a2a_gateway import GatewayHistoryConverter
from thenvoi.core.control_plane_adapter import (
    ControlPlaneAdapter,
    legacy_control_turn_compat,
)
from thenvoi.core.types import ControlMessageTurnContext
from thenvoi.integrations.a2a.gateway.peer_directory import PeerDirectory, PeerRef
from thenvoi.integrations.a2a.gateway.session_manager import GatewaySessionManager
from thenvoi.integrations.a2a.gateway.server import GatewayServer
from thenvoi.integrations.a2a.gateway.task_correlator import GatewayTaskCorrelator
from thenvoi.integrations.a2a.gateway.types import GatewaySessionState
from thenvoi_rest import Peer

logger = logging.getLogger(__name__)


def slugify(name: str) -> str:
    """Convert name to URL-safe slug.

    Args:
        name: The name to slugify.

    Returns:
        URL-safe slug (lowercase, alphanumeric with dashes).
    """
    slug = name.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)  # Replace non-alphanumeric with -
    return slug.strip("-")  # Remove leading/trailing dashes


class A2AGatewayAdapter(ControlPlaneAdapter[GatewaySessionState]):
    """Gateway adapter exposing Thenvoi peers as A2A endpoints.

    This adapter enables external A2A agents to interact with Thenvoi platform
    peers through standard A2A HTTP endpoints. It acts as a bridge:
    - Receives A2A messages via HTTP server
    - Creates/reuses Thenvoi chat rooms for context management
    - Sends messages to peers via REST API
    - Streams responses back via SSE

    Uses a control-plane adapter contract because gateway requests are initiated
    through HTTP and resolved through REST-bound orchestration, not room-bound
    `AgentToolsProtocol` calls.

    Example:
        from thenvoi import Agent
        from thenvoi.integrations.a2a.gateway import A2AGatewayAdapter

        adapter = A2AGatewayAdapter(
            rest_url="https://app.thenvoi.com",
            api_key="your-api-key",
            gateway_url="http://localhost:10000",
            port=10000,
        )
        agent = Agent.create(
            adapter=adapter,
            agent_id="sap-gateway",
            api_key="your-api-key",
        )
        await agent.run()
    """

    def __init__(
        self,
        rest_url: str = DEFAULT_REST_URL,
        api_key: str = "",
        gateway_url: str = "http://localhost:10000",
        port: int = 10000,
    ) -> None:
        """Initialize gateway adapter.

        Args:
            rest_url: Base URL for Thenvoi REST API.
            api_key: API key for authentication (same as Agent.create()).
            gateway_url: Base URL for A2A endpoints exposed by this gateway.
            port: Port for HTTP server to listen on.
        """
        super().__init__(history_converter=GatewayHistoryConverter())
        self.gateway_url = gateway_url
        self.port = port

        # Direct REST client for room/message operations
        self._rest = AsyncRestClient(base_url=rest_url, api_key=api_key)

        # Canonical peer state (slug + UUID aliases resolved by one directory).
        self._peer_directory = PeerDirectory()
        self._server: GatewayServer | None = None

        # Dedicated gateway collaborators (adapter acts as composition root).
        self._session_manager = GatewaySessionManager(self._rest)
        self._task_correlator = GatewayTaskCorrelator()

    @property
    def session_manager(self) -> GatewaySessionManager:
        """Gateway session/context manager (public for diagnostics/testing)."""
        return self._session_manager

    @property
    def task_correlator(self) -> GatewayTaskCorrelator:
        """Gateway task correlator (public for diagnostics/testing)."""
        return self._task_correlator

    @property
    def peer_directory(self) -> PeerDirectory:
        """Canonical peer directory used for slug/UUID peer resolution."""
        return self._peer_directory

    async def on_started(self, agent_name: str, agent_description: str) -> None:
        """Fetch peers via REST and start HTTP server.

        Args:
            agent_name: Name of this agent.
            agent_description: Description of this agent.
        """
        await super().on_started(agent_name, agent_description)

        # Fetch ALL peers at startup using REST client (with pagination)
        all_peers: list[Peer] = []
        page = 1
        page_size = 100

        while True:
            response = await self._rest.agent_api_peers.list_agent_peers(
                page=page,
                page_size=page_size,
                request_options=DEFAULT_REQUEST_OPTIONS,
            )
            all_peers.extend(response.data)

            # Check if more pages exist
            if len(response.data) < page_size:
                break
            page += 1

        # Build canonical peer directory.
        self._peer_directory.replace_from_peers(all_peers, slugify=slugify)

        logger.info(
            "Discovered %d peers for gateway",
            len(self._peer_directory.peers),
        )

        # Create and start HTTP server with peer routes
        self._server = GatewayServer(
            peer_directory=self._peer_directory,
            gateway_url=self.gateway_url,
            port=self.port,
            on_request=self._handle_a2a_request,
        )
        await self._server.start()

        logger.info("Gateway HTTP server started on port %d", self.port)

    @legacy_control_turn_compat
    async def on_message(
        self,
        turn: ControlMessageTurnContext[GatewaySessionState],
    ) -> None:
        """Receive Thenvoi response, correlate with pending A2A task.

        This is called when a peer responds in a room. We correlate the
        response with the pending A2A task and stream it back via SSE.

        Args:
            turn: Canonical control-plane turn context for this room message.
        """
        msg = turn.msg
        history = turn.history
        is_session_bootstrap = turn.is_session_bootstrap
        room_id = turn.room_id

        # Rehydrate on bootstrap
        if is_session_bootstrap and history:
            self._session_manager.rehydrate(history)

        await self._task_correlator.ingest_platform_message(
            room_id=room_id,
            message=msg,
        )

    async def on_cleanup(self, room_id: str) -> None:
        """Clean up resources for a room.

        Args:
            room_id: The room identifier.
        """
        self._task_correlator.pop_room(room_id)
        logger.debug("Cleaned up gateway resources for room %s", room_id)

    async def stop(self) -> None:
        """Stop the HTTP server and clean up resources."""
        if self._server:
            await self._server.stop()
            self._server = None
        logger.info("Gateway adapter stopped")

    async def _handle_a2a_request(
        self, peer_ref: PeerRef, message: A2AMessage
    ) -> AsyncIterator[TaskStatusUpdateEvent]:
        """Handle incoming A2A request from external agent.

        Args:
            peer_ref: Resolved target peer reference.
            message: A2A message from external agent.

        Yields:
            TaskStatusUpdateEvent for SSE streaming.
        """
        peer = peer_ref.peer

        # Use the peer's actual UUID for Thenvoi API calls
        peer_uuid = peer.id

        # Get or create room for context
        room_id, context_id = await self._session_manager.get_or_create_room(
            context_id=message.context_id,
            target_peer_id=peer_uuid,
        )

        pending = self._task_correlator.register_pending(
            room_id=room_id,
            context_id=context_id,
            peer_id=peer_uuid,
        )

        # Emit task event to track context mapping in history
        await self._session_manager.emit_context_event(room_id, context_id)

        # Send message to Thenvoi via REST client
        content = get_message_text(message) or ""

        # Use peer name for mention
        peer_name = peer.name

        await self._rest.agent_api_messages.create_agent_chat_message(
            chat_id=room_id,
            message=ChatMessageRequest(
                content=f"@{peer_name} {content}",
                mentions=[ChatMessageRequestMentionsItem(id=peer_uuid, name=peer_name)],
            ),
            request_options=DEFAULT_REQUEST_OPTIONS,
        )

        logger.debug(
            "Sent message to peer %s (%s, slug=%s) in room %s (context=%s)",
            peer_name,
            peer_uuid,
            peer_ref.slug,
            room_id,
            context_id,
        )

        # Stream events from queue (populated by on_message())
        while True:
            event = await pending.sse_queue.get()
            yield event
            if event.final:
                break
