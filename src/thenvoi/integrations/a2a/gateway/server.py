"""HTTP server for A2A Gateway adapter."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    Message as A2AMessage,
    Task,
    TaskStatusUpdateEvent,
)
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from thenvoi.integrations.a2a.gateway.peer_directory import PeerDirectory, PeerRef
from thenvoi_rest import Peer

logger = logging.getLogger(__name__)

# Type alias for the on_request callback
OnRequestCallback = Callable[[PeerRef, A2AMessage], AsyncIterator[TaskStatusUpdateEvent]]


class GatewayServer:
    """Starlette HTTP server exposing A2A endpoints for each peer.

    Creates per-peer routes for AgentCard discovery and message streaming.
    Routes are created at startup based on the discovered peers.

    Peers are addressed by slug (e.g., "weather-agent") with UUID fallback.

    Attributes:
        peers: Projection mapping slug to Peer objects (primary lookup).
        peers_by_uuid: Projection mapping UUID to Peer objects (fallback lookup).
        gateway_url: Base URL for the gateway (used in AgentCard URLs).
        port: Port to listen on.
        on_request: Callback invoked when an A2A message is received.
    """

    def __init__(
        self,
        gateway_url: str,
        port: int,
        on_request: OnRequestCallback,
        peers: dict[str, Peer] | None = None,
        peers_by_uuid: dict[str, Peer] | None = None,
        peer_directory: PeerDirectory | None = None,
    ) -> None:
        """Initialize gateway server.

        Args:
            gateway_url: Base URL for AgentCard URLs (e.g., "http://localhost:10000").
            port: Port to listen on.
            on_request: Async callback invoked with (peer_ref, message) → events.
            peers: Optional slug → peer map (legacy constructor compatibility).
            peers_by_uuid: Optional UUID → peer map (legacy constructor compatibility).
            peer_directory: Optional prebuilt peer directory.
        """
        if peer_directory is not None and (peers or peers_by_uuid):
            raise ValueError(
                "Provide either peer_directory or peers/peers_by_uuid, not both."
            )

        self.peer_directory = peer_directory or PeerDirectory(
            peers=peers or {},
            peers_by_uuid=peers_by_uuid or {},
        )
        self.gateway_url = gateway_url
        self.port = port
        self.on_request = on_request
        self._app: Starlette | None = None
        self._server_task: asyncio.Task[Any] | None = None

    @property
    def peers(self) -> dict[str, Peer]:
        """Projection of canonical peer directory keyed by slug."""
        return self.peer_directory.peers

    @property
    def peers_by_uuid(self) -> dict[str, Peer]:
        """Projection of canonical peer directory keyed by UUID aliases."""
        return self.peer_directory.peers_by_uuid

    def _resolve_peer(self, peer_id: str) -> PeerRef | None:
        """Resolve peer by slug or UUID.

        Args:
            peer_id: Peer slug or UUID from URL path.

        Returns:
            Resolved peer reference if found, None otherwise.
        """
        return self.peer_directory.resolve(peer_id)

    def _build_app(self) -> Starlette:
        """Build Starlette application with routes."""
        routes = [
            # List all available peers
            Route(
                "/peers",
                self._handle_list_peers,
                methods=["GET"],
            ),
            # Per-peer agent card discovery (support both naming conventions)
            Route(
                "/agents/{peer_id}/.well-known/agent.json",
                self._handle_agent_card,
                methods=["GET"],
            ),
            Route(
                "/agents/{peer_id}/.well-known/agent-card.json",
                self._handle_agent_card,
                methods=["GET"],
            ),
            # Per-peer message streaming (legacy REST endpoint)
            Route(
                "/agents/{peer_id}/v1/message:stream",
                self._handle_message_stream,
                methods=["POST"],
            ),
            # JSON-RPC endpoint (A2A SDK posts here with method field)
            Route(
                "/agents/{peer_id}",
                self._handle_jsonrpc,
                methods=["POST"],
            ),
        ]
        return Starlette(routes=routes)

    async def _handle_list_peers(self, request: Request) -> JSONResponse:
        """Return list of all available peers.

        Args:
            request: Starlette request.

        Returns:
            JSONResponse with list of peers (slug, id, name, description).
        """
        peers_list = [
            {
                "slug": slug,  # Primary identifier for URLs
                "id": peer.id,  # UUID fallback
                "name": peer.name,  # Display name
                "description": peer.description or "",
            }
            for slug, peer in self.peer_directory.items()
        ]
        return JSONResponse({"peers": peers_list, "count": len(peers_list)})

    async def _handle_agent_card(self, request: Request) -> JSONResponse:
        """Return AgentCard for the specified peer.

        Args:
            request: Starlette request with peer_id path parameter (slug or UUID).

        Returns:
            JSONResponse with AgentCard or 404 if peer not found.
        """
        peer_id = request.path_params["peer_id"]
        resolved = self._resolve_peer(peer_id)
        if not resolved:
            return JSONResponse({"error": "Not found"}, status_code=404)

        slug = resolved.slug
        peer = resolved.peer

        card = AgentCard(
            name=peer.name,
            description=peer.description or "",
            url=f"{self.gateway_url}/agents/{slug}",  # Use slug in URL
            version="1.0.0",
            capabilities=AgentCapabilities(streaming=True),
            skills=[
                AgentSkill(
                    id="default",
                    name=peer.name,
                    description=peer.description or "",
                    tags=["thenvoi", "gateway"],
                )
            ],
            default_input_modes=["text/plain"],
            default_output_modes=["text/plain"],
        )
        return JSONResponse(card.model_dump(mode="json", by_alias=True))

    async def _handle_message_stream(self, request: Request) -> StreamingResponse:
        """Handle incoming A2A message and stream response events.

        Args:
            request: Starlette request with peer_id path parameter (slug or UUID) and JSON body.

        Returns:
            StreamingResponse with SSE events.
        """
        peer_id = request.path_params["peer_id"]

        # Resolve peer by slug or UUID
        resolved = self._resolve_peer(peer_id)
        if not resolved:
            return JSONResponse({"error": "Not found"}, status_code=404)  # type: ignore[return-value]

        slug = resolved.slug
        peer = resolved.peer

        body = await request.json()
        message = A2AMessage(**body)

        logger.debug(
            "Received A2A message for peer %s (%s): %s",
            peer.name,
            slug,
            message.message_id,
        )

        async def event_stream() -> AsyncIterator[str]:
            """Generate SSE events from on_request callback."""
            try:
                async for event in self.on_request(resolved, message):
                    yield f"data: {event.model_dump_json()}\n\n"
            except Exception as e:
                logger.exception("Error in A2A request handler: %s", e)
                # Send error event
                yield f'data: {{"error": "{e!s}"}}\n\n'

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    async def _handle_jsonrpc(
        self, request: Request
    ) -> JSONResponse | StreamingResponse:
        """Handle JSON-RPC requests from A2A SDK.

        The A2A SDK posts JSON-RPC requests to the base agent URL with
        a method field to indicate the operation (message/send, message/stream, etc.).

        Args:
            request: Starlette request with peer_id path parameter and JSON-RPC body.

        Returns:
            JSONResponse for sync methods, StreamingResponse for streaming methods.
        """
        peer_id = request.path_params["peer_id"]

        # Resolve peer by slug or UUID
        resolved = self._resolve_peer(peer_id)
        if not resolved:
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "error": {"code": -32001, "message": "Peer not found"},
                    "id": None,
                },
                status_code=404,
            )

        slug = resolved.slug
        peer = resolved.peer
        body = await request.json()

        # Extract JSON-RPC fields
        method = body.get("method", "")
        request_id = body.get("id")
        params = body.get("params", {})

        logger.debug(
            "Received JSON-RPC %s for peer %s (%s), request_id=%s",
            method,
            peer.name,
            slug,
            request_id,
        )

        # Route based on method
        if method == "message/send":
            return await self._handle_jsonrpc_send(resolved, params, request_id)
        elif method == "message/stream":
            return await self._handle_jsonrpc_stream(resolved, params, request_id)
        else:
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                    "id": request_id,
                },
                status_code=400,
            )

    async def _handle_jsonrpc_send(
        self, peer_ref: PeerRef, params: dict[str, Any], request_id: str | None
    ) -> JSONResponse:
        """Handle synchronous message/send JSON-RPC request.

        Args:
            peer_ref: Resolved peer reference.
            params: JSON-RPC params containing message.
            request_id: JSON-RPC request ID.

        Returns:
            JSONResponse with JSON-RPC result (Task).
        """
        # Extract message from params
        message_data = params.get("message", {})
        message = A2AMessage(**message_data)

        # Collect all events until final
        final_event: TaskStatusUpdateEvent | None = None
        try:
            async for event in self.on_request(peer_ref, message):
                final_event = event
                if event.final:
                    break
        except Exception as e:
            logger.exception("Error in JSON-RPC send handler: %s", e)
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "error": {"code": -32000, "message": str(e)},
                    "id": request_id,
                },
                status_code=500,
            )

        # Build Task result from final event
        if final_event:
            task = Task(
                id=final_event.task_id,
                context_id=final_event.context_id,
                status=final_event.status,
            )
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "result": task.model_dump(mode="json", by_alias=True),
                    "id": request_id,
                }
            )
        else:
            # No events received - create empty task
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "error": {"code": -32000, "message": "No response from peer"},
                    "id": request_id,
                },
                status_code=500,
            )

    async def _handle_jsonrpc_stream(
        self, peer_ref: PeerRef, params: dict[str, Any], request_id: str | None
    ) -> StreamingResponse:
        """Handle streaming message/stream JSON-RPC request.

        Args:
            peer_ref: Resolved peer reference.
            params: JSON-RPC params containing message.
            request_id: JSON-RPC request ID.

        Returns:
            StreamingResponse with SSE events in JSON-RPC format.
        """
        # Extract message from params
        message_data = params.get("message", {})
        message = A2AMessage(**message_data)

        async def event_stream() -> AsyncIterator[str]:
            """Generate SSE events in JSON-RPC format."""
            try:
                async for event in self.on_request(peer_ref, message):
                    # Wrap event in JSON-RPC response
                    jsonrpc_response = {
                        "jsonrpc": "2.0",
                        "result": event.model_dump(mode="json", by_alias=True),
                        "id": request_id,
                    }
                    yield f"data: {json.dumps(jsonrpc_response)}\n\n"
            except Exception as e:
                logger.exception("Error in JSON-RPC stream handler: %s", e)
                error_response = {
                    "jsonrpc": "2.0",
                    "error": {"code": -32000, "message": str(e)},
                    "id": request_id,
                }
                yield f"data: {json.dumps(error_response)}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    async def start(self) -> None:
        """Start the HTTP server.

        Creates the Starlette app and starts serving on the configured port.
        """
        import uvicorn

        self._app = self._build_app()

        config = uvicorn.Config(
            self._app,
            host="0.0.0.0",
            port=self.port,
            log_level="warning",
        )
        server = uvicorn.Server(config)

        logger.info(
            "Starting A2A Gateway server on port %d with %d peers",
            self.port,
            len(self.peer_directory.peers),
        )

        # Run server in background task
        self._server_task = asyncio.create_task(server.serve())

    async def stop(self) -> None:
        """Stop the HTTP server."""
        if self._server_task:
            self._server_task.cancel()
            result = (await asyncio.gather(self._server_task, return_exceptions=True))[0]
            if isinstance(result, asyncio.CancelledError):
                logger.debug("A2A Gateway server task cancelled")
            elif isinstance(result, Exception):
                logger.warning("A2A Gateway server task exited with error: %s", result)
            self._server_task = None
            logger.info("A2A Gateway server stopped")
