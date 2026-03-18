"""Tests for bridge health endpoint."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest
from aiohttp.test_utils import TestClient, TestServer

from bridge_core.health import HealthServer
from bridge_core.session import InMemorySessionStore


@pytest.fixture
def mock_link() -> MagicMock:
    link = MagicMock()
    type(link).is_connected = PropertyMock(return_value=True)
    return link


@pytest.fixture
def session_store() -> InMemorySessionStore:
    return InMemorySessionStore()


@pytest.fixture
def health_server(
    mock_link: MagicMock, session_store: InMemorySessionStore
) -> HealthServer:
    return HealthServer(
        link=mock_link,
        port=0,
        session_store=session_store,
        handler_count=2,
    )


async def test_health_returns_healthy_when_connected(
    mock_link: MagicMock, health_server: HealthServer
) -> None:
    type(mock_link).is_connected = PropertyMock(return_value=True)

    async with TestClient(TestServer(health_server._app)) as client:
        resp = await client.get("/health")
        assert resp.status == 200
        data = await resp.json()
        assert data["status"] == "healthy"
        assert data["websocket_connected"] is True


async def test_health_returns_unhealthy_when_disconnected(
    mock_link: MagicMock, health_server: HealthServer
) -> None:
    type(mock_link).is_connected = PropertyMock(return_value=False)

    async with TestClient(TestServer(health_server._app)) as client:
        resp = await client.get("/health")
        assert resp.status == 503
        data = await resp.json()
        assert data["status"] == "unhealthy"
        assert data["websocket_connected"] is False


async def test_health_includes_handler_count(
    mock_link: MagicMock, health_server: HealthServer
) -> None:
    async with TestClient(TestServer(health_server._app)) as client:
        resp = await client.get("/health")
        data = await resp.json()
        assert data["handlers_registered"] == 2


async def test_health_includes_active_sessions(
    mock_link: MagicMock,
    health_server: HealthServer,
    session_store: InMemorySessionStore,
) -> None:
    await session_store.get_or_create("room-1")
    await session_store.get_or_create("room-2")

    async with TestClient(TestServer(health_server._app)) as client:
        resp = await client.get("/health")
        data = await resp.json()
        assert data["active_sessions"] == 2


async def test_health_without_session_store(mock_link: MagicMock) -> None:
    server = HealthServer(link=mock_link, port=0, handler_count=1)

    async with TestClient(TestServer(server._app)) as client:
        resp = await client.get("/health")
        data = await resp.json()
        assert "active_sessions" not in data
        assert data["handlers_registered"] == 1


async def test_health_warns_when_no_handlers(mock_link: MagicMock) -> None:
    """Health response should include a warning when no handlers are registered."""
    server = HealthServer(link=mock_link, port=0, handler_count=0)

    async with TestClient(TestServer(server._app)) as client:
        resp = await client.get("/health")
        data = await resp.json()
        assert data["handlers_registered"] == 0
        assert data["warning"] == "no handlers registered"


async def test_health_no_warning_with_handlers(
    mock_link: MagicMock, health_server: HealthServer
) -> None:
    """Health response should NOT include a warning when handlers are registered."""
    async with TestClient(TestServer(health_server._app)) as client:
        resp = await client.get("/health")
        data = await resp.json()
        assert "warning" not in data


async def test_stop_handles_cleanup_error(mock_link: MagicMock) -> None:
    """stop() should not raise even if runner.cleanup() fails."""
    server = HealthServer(link=mock_link, port=0)
    server._runner = MagicMock()
    server._runner.cleanup = AsyncMock(side_effect=OSError("cleanup boom"))

    await server.stop()  # Should not raise

    assert server._runner is None
