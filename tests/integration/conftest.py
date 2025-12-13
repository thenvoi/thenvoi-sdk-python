"""Pytest configuration for integration tests.

Re-exports fixtures from the parent module.
Credentials are loaded from .env.test automatically.
"""

from tests.conftest_integration import (
    api_client,
    api_client_2,
    get_api_key,
    get_api_key_2,
    get_base_url,
    get_test_agent_id,
    get_test_agent_id_2,
    get_ws_url,
    integration_settings,
    requires_api,
    requires_multi_agent,
    test_chat,
    test_peer_id,
    test_settings,
)

__all__ = [
    "api_client",
    "api_client_2",
    "get_api_key",
    "get_api_key_2",
    "get_base_url",
    "get_test_agent_id",
    "get_test_agent_id_2",
    "get_ws_url",
    "integration_settings",
    "requires_api",
    "requires_multi_agent",
    "test_chat",
    "test_peer_id",
    "test_settings",
]
