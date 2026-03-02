"""Conftest for bridge tests with canonical a2a_bridge imports."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from thenvoi.integrations.a2a_bridge.bridge import BridgeConfig, ThenvoiBridge
from thenvoi.integrations.a2a_bridge.handler import HandlerResult

pytestmark = pytest.mark.contract_gate


@pytest.fixture
def bridge_config() -> BridgeConfig:
    """Standard bridge config for tests."""
    return BridgeConfig(
        agent_id="agent-1",
        api_key="key-1",
        agent_mapping="alice:handler_a",
    )


@pytest.fixture
def bridge_with_mock_link(
    bridge_config: BridgeConfig, link_mock_factory
) -> ThenvoiBridge:
    """Bridge instance with a mocked link and handler."""
    handler = AsyncMock()
    handler.handle.return_value = HandlerResult.handled()
    b = ThenvoiBridge(config=bridge_config, handlers={"handler_a": handler})
    mock_link = link_mock_factory()
    b.set_link(mock_link)
    return b


@pytest.fixture
def bridge_with_full_mock(
    bridge_config: BridgeConfig, link_mock_factory
) -> ThenvoiBridge:
    """Bridge with all link methods mocked (superset of bridge_with_mock_link).

    Covers connect, disconnect, subscribe/unsubscribe, lifecycle marks,
    and REST endpoints needed by dedup, participant cache, and
    connect-and-consume tests.
    """
    handler = AsyncMock()
    handler.handle.return_value = HandlerResult.handled()
    b = ThenvoiBridge(config=bridge_config, handlers={"handler_a": handler})
    mock_link = link_mock_factory()
    mock_link.rest.agent_api_chats.list_agent_chats = AsyncMock(
        return_value=MagicMock(data=None)
    )
    b.set_link(mock_link)
    return b
