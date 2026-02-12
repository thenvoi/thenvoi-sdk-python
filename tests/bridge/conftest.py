"""Conftest for bridge tests — adds thenvoi-bridge to sys.path and shared fixtures."""

from __future__ import annotations

import os
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

# Add thenvoi-bridge directory to sys.path so we can import core.* and handlers.*
_bridge_dir = os.path.join(os.path.dirname(__file__), "..", "..", "thenvoi-bridge")
if _bridge_dir not in sys.path:
    sys.path.insert(0, os.path.abspath(_bridge_dir))

from core.bridge import BridgeConfig, ThenvoiBridge  # noqa: E402


@pytest.fixture
def bridge_config() -> BridgeConfig:
    """Standard bridge config for tests."""
    return BridgeConfig(
        agent_id="agent-1",
        api_key="key-1",
        agent_mapping="alice:handler_a",
    )


@pytest.fixture
def bridge_with_mock_link(bridge_config: BridgeConfig) -> ThenvoiBridge:
    """Bridge instance with a mocked link and handler."""
    handler = AsyncMock()
    b = ThenvoiBridge(config=bridge_config, handlers={"handler_a": handler})
    mock_link = MagicMock()
    mock_link.subscribe_room = AsyncMock()
    mock_link.unsubscribe_room = AsyncMock()
    mock_link.rest = MagicMock()
    b._link = mock_link
    b._router._link = mock_link
    return b
