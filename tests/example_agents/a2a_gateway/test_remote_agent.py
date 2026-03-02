"""Unit tests for demo orchestrator gateway client error contracts."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from a2a.types import Part, TextPart

from thenvoi.integrations.a2a_gateway.orchestrator.remote_agent import (
    GatewayClient,
    GatewayDiscoveryError,
    GatewayRequestError,
    GatewayResponseError,
    PeerUnavailableError,
)


class _ResolverNotFound:
    async def get_agent_card(self) -> Any:
        request = httpx.Request("GET", "http://gateway/agents/weather")
        response = httpx.Response(404, request=request)
        raise httpx.HTTPStatusError("not found", request=request, response=response)


class _ResolverTimeout:
    async def get_agent_card(self) -> Any:
        request = httpx.Request("GET", "http://gateway/agents/weather")
        raise httpx.ConnectTimeout("timeout", request=request)


class _ResolverOk:
    async def get_agent_card(self) -> Any:
        return SimpleNamespace(name="weather")


class _ClientSendError:
    async def send_message(self, request: Any) -> Any:
        raise httpx.ReadTimeout("timed out")


class _ClientNoResponse:
    async def send_message(self, request: Any) -> Any:
        empty_task = SimpleNamespace(artifacts=[], status=None)
        return SimpleNamespace(root=SimpleNamespace(result=empty_task))


class _ClientSuccess:
    async def send_message(self, request: Any) -> Any:
        status = SimpleNamespace(message=SimpleNamespace(parts=[Part(root=TextPart(text="ok"))]))
        task = SimpleNamespace(artifacts=[], status=status)
        return SimpleNamespace(root=SimpleNamespace(result=task))


def _resolver_factory(resolver: Any):
    def _factory(http_client: httpx.AsyncClient, peer_url: str) -> Any:  # noqa: ARG001
        return resolver

    return _factory


def _client_factory(client: Any):
    def _factory(http_client: httpx.AsyncClient, card: Any, peer_url: str) -> Any:  # noqa: ARG001
        return client

    return _factory


@pytest.mark.asyncio
async def test_discover_peer_returns_unavailable_for_404() -> None:
    client = GatewayClient(
        "http://gateway",
        resolver_factory=_resolver_factory(_ResolverNotFound()),
    )
    try:
        result = await client.discover_peer("weather")
    finally:
        await client.close()

    assert not result.available
    assert result.reason == "peer_unavailable"


@pytest.mark.asyncio
async def test_discover_peer_raises_on_transport_error() -> None:
    client = GatewayClient(
        "http://gateway",
        resolver_factory=_resolver_factory(_ResolverTimeout()),
    )
    try:
        with pytest.raises(GatewayDiscoveryError) as exc_info:
            await client.discover_peer("weather")
    finally:
        await client.close()

    assert exc_info.value.code == "peer_discovery_failed"
    assert exc_info.value.retryable is True


@pytest.mark.asyncio
async def test_call_peer_raises_peer_unavailable_error_for_404() -> None:
    client = GatewayClient(
        "http://gateway",
        resolver_factory=_resolver_factory(_ResolverNotFound()),
    )
    try:
        with pytest.raises(PeerUnavailableError) as exc_info:
            await client.call_peer("weather", "hello")
    finally:
        await client.close()

    assert exc_info.value.code == "peer_not_found"
    assert exc_info.value.peer_id == "weather"


@pytest.mark.asyncio
async def test_call_peer_raises_request_error_for_send_failures() -> None:
    client = GatewayClient(
        "http://gateway",
        resolver_factory=_resolver_factory(_ResolverOk()),
        client_factory=_client_factory(_ClientSendError()),
    )
    try:
        with pytest.raises(GatewayRequestError) as exc_info:
            await client.call_peer("weather", "hello")
    finally:
        await client.close()

    assert exc_info.value.code == "send_message_failed"
    assert exc_info.value.retryable is True


@pytest.mark.asyncio
async def test_call_peer_raises_response_error_for_empty_payload() -> None:
    client = GatewayClient(
        "http://gateway",
        resolver_factory=_resolver_factory(_ResolverOk()),
        client_factory=_client_factory(_ClientNoResponse()),
    )
    try:
        with pytest.raises(GatewayResponseError) as exc_info:
            await client.call_peer("weather", "hello")
    finally:
        await client.close()

    assert exc_info.value.code == "empty_peer_response"


@pytest.mark.asyncio
async def test_call_peer_returns_text_response() -> None:
    client = GatewayClient(
        "http://gateway",
        resolver_factory=_resolver_factory(_ResolverOk()),
        client_factory=_client_factory(_ClientSuccess()),
    )
    try:
        response = await client.call_peer("weather", "hello")
    finally:
        await client.close()

    assert response == "ok"
