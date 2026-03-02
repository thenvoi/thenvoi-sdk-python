"""Gateway client for calling A2A Gateway peers.

This module provides a client wrapper for calling Thenvoi platform peers
via the A2A Gateway using the standard A2A protocol.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, TypedDict
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    MessageSendParams,
    Part,
    SendMessageRequest,
    SendMessageSuccessResponse,
    TextPart,
)

logger = logging.getLogger(__name__)


class PeerData(TypedDict, total=False):
    """Gateway peer metadata."""

    id: str
    slug: str
    name: str
    description: str


class GatewayClientError(RuntimeError):
    """Base error type for gateway client failures."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        peer_id: str | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.peer_id = peer_id
        self.retryable = retryable


class GatewayDiscoveryError(GatewayClientError):
    """Gateway discovery failed due to transport/protocol errors."""


class PeerUnavailableError(GatewayClientError):
    """Requested peer is not exposed by the gateway."""


class GatewayRequestError(GatewayClientError):
    """Calling a peer failed due to transport/runtime issues."""


class GatewayResponseError(GatewayClientError):
    """Peer call succeeded but response payload could not be parsed."""


@dataclass(frozen=True)
class PeerDiscoveryResult:
    """Structured result for peer discovery checks."""

    available: bool
    reason: str | None = None


def format_gateway_error_for_user(error: GatewayClientError) -> str:
    """Format gateway failures into a stable user-facing response."""
    if isinstance(error, PeerUnavailableError):
        peer_hint = f" '{error.peer_id}'" if error.peer_id else ""
        return f"Peer{peer_hint} is not available via the gateway right now."
    if isinstance(error, GatewayResponseError):
        return "The peer returned an unexpected response. Please try again."
    if error.retryable:
        return "Gateway communication failed temporarily. Please retry."
    return "Gateway request failed. Please try again later."


ResolverFactory = Callable[[httpx.AsyncClient, str], Any]
ClientFactory = Callable[[httpx.AsyncClient, Any, str], Any]


def extract_text_from_parts(parts: list[Part]) -> str:
    """Extract text content from A2A message parts."""
    texts = []
    for part in parts:
        if isinstance(part.root, TextPart):
            texts.append(part.root.text)
    return " ".join(texts)


class GatewayClient:
    """Client for calling A2A Gateway peers.

    This client wraps the A2A SDK's client to provide a simple interface
    for calling peers exposed by the A2A Gateway.

    Example:
        client = GatewayClient("http://localhost:10000")
        response = await client.call_peer("weather", "What's the weather in NYC?")
    """

    def __init__(
        self,
        gateway_url: str,
        timeout: float = 60.0,
        *,
        resolver_factory: ResolverFactory | None = None,
        client_factory: ClientFactory | None = None,
    ):
        """Initialize the gateway client.

        Args:
            gateway_url: Base URL of the A2A Gateway (e.g., "http://localhost:10000")
            timeout: Request timeout in seconds
            resolver_factory: Optional factory for A2A card resolver instances
            client_factory: Optional factory for A2A client instances
        """
        self.gateway_url = gateway_url.rstrip("/")
        self.timeout = timeout
        self._http_client: httpx.AsyncClient | None = None
        self._resolver_factory = resolver_factory or self._default_resolver_factory
        self._client_factory = client_factory or self._default_client_factory

    @staticmethod
    def _default_resolver_factory(http_client: httpx.AsyncClient, peer_url: str) -> Any:
        return A2ACardResolver(http_client, peer_url)

    @staticmethod
    def _default_client_factory(
        http_client: httpx.AsyncClient, card: Any, peer_url: str
    ) -> Any:
        return A2AClient(http_client, card, url=peer_url)

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        if isinstance(exc, httpx.TimeoutException):
            return True
        if isinstance(exc, httpx.RequestError):
            return True
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code >= 500
        return False

    @staticmethod
    def _is_http_status(exc: Exception, status_code: int) -> bool:
        return (
            isinstance(exc, httpx.HTTPStatusError)
            and exc.response.status_code == status_code
        )

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=self.timeout)
        return self._http_client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
            self._http_client = None

    async def list_peers(self) -> list[PeerData]:
        """Fetch list of available peers from gateway.

        Returns:
            List of peer dicts with id, name, description
        """
        http_client = await self._get_http_client()
        try:
            response = await http_client.get(f"{self.gateway_url}/peers")
            response.raise_for_status()
            data = response.json()
            peers = data.get("peers", [])
            if not isinstance(peers, list):
                raise ValueError("Expected 'peers' to be a list")
            return peers
        except Exception as e:
            raise GatewayDiscoveryError(
                "Failed to list gateway peers",
                code="list_peers_failed",
                retryable=self._is_retryable(e),
            ) from e

    async def discover_peer(self, peer_id: str) -> PeerDiscoveryResult:
        """Check if a peer is available via the gateway.

        Returns:
            Structured availability result. Transport/protocol failures raise.
        """
        try:
            await self._resolve_peer_card(peer_id)
            return PeerDiscoveryResult(available=True)
        except PeerUnavailableError:
            return PeerDiscoveryResult(available=False, reason="peer_unavailable")

    async def call_peer(
        self,
        peer_id: str,
        message: str,
        context_id: str | None = None,
    ) -> str:
        """Call a peer via the A2A Gateway.

        Args:
            peer_id: The ID of the peer to call (e.g., "weather", "servicenow")
            message: The message to send to the peer
            context_id: Optional context ID for conversation continuity

        Returns:
            The peer's response text

        Raises:
            GatewayClientError: If the peer is unavailable or call fails
        """
        peer_url = f"{self.gateway_url}/agents/{peer_id}"
        http_client = await self._get_http_client()

        logger.info("Calling peer '%s' via gateway: %s", peer_id, peer_url)

        card = await self._resolve_peer_card(peer_id)
        logger.debug("Resolved agent card for peer '%s': %s", peer_id, card.name)
        client = self._client_factory(http_client, card, peer_url)

        message_id = str(uuid4())
        ctx_id = context_id or str(uuid4())
        request = SendMessageRequest(
            id=str(uuid4()),
            params=MessageSendParams(
                message={
                    "role": "user",
                    "parts": [{"kind": "text", "text": message}],
                    "messageId": message_id,
                    "contextId": ctx_id,
                }
            ),
        )

        try:
            response = await client.send_message(request)
        except Exception as exc:
            raise GatewayRequestError(
                f"Failed to call peer '{peer_id}'",
                code="send_message_failed",
                peer_id=peer_id,
                retryable=self._is_retryable(exc),
            ) from exc
        return self._extract_response(response, peer_id=peer_id)

    async def _resolve_peer_card(self, peer_id: str) -> Any:
        """Resolve and return an A2A agent card for a peer."""
        peer_url = f"{self.gateway_url}/agents/{peer_id}"
        http_client = await self._get_http_client()
        resolver = self._resolver_factory(http_client, peer_url)
        try:
            return await resolver.get_agent_card()
        except Exception as exc:
            if self._is_http_status(exc, 404):
                raise PeerUnavailableError(
                    f"Peer '{peer_id}' not found",
                    code="peer_not_found",
                    peer_id=peer_id,
                    retryable=False,
                ) from exc
            raise GatewayDiscoveryError(
                f"Failed to resolve peer '{peer_id}'",
                code="peer_discovery_failed",
                peer_id=peer_id,
                retryable=self._is_retryable(exc),
            ) from exc

    def _extract_response(
        self, response: SendMessageSuccessResponse, *, peer_id: str
    ) -> str:
        """Extract text response from A2A response.

        Args:
            response: The A2A send message response

        Returns:
            The extracted text response
        """
        # The response contains a Task with status and/or artifacts
        task = response.root.result

        # Try artifacts first (completed tasks)
        if task.artifacts:
            for artifact in task.artifacts:
                if artifact.parts:
                    text = extract_text_from_parts(artifact.parts)
                    if text:
                        return text

        # Fall back to status message
        if task.status and task.status.message:
            parts = task.status.message.parts
            if parts:
                return extract_text_from_parts(parts)

        raise GatewayResponseError(
            f"Peer '{peer_id}' returned no text response",
            code="empty_peer_response",
            peer_id=peer_id,
            retryable=False,
        )

    async def __aenter__(self) -> GatewayClient:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
