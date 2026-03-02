"""Shared pytest fixture factories for ThenvoiLink/WebSocket doubles."""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.support.fakes_link import make_thenvoi_link_mock, make_websocket_client_mock


@pytest.fixture
def link_mock_factory() -> Callable[..., MagicMock]:
    """Provide canonical ThenvoiLink fake builder for test-local customization."""
    return make_thenvoi_link_mock


@pytest.fixture
def ws_client_mock_factory() -> Callable[..., AsyncMock]:
    """Provide canonical WebSocketClient fake builder for test-local customization."""
    return make_websocket_client_mock


__all__ = [
    "link_mock_factory",
    "ws_client_mock_factory",
]
