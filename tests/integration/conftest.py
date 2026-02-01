"""Pytest configuration for integration tests.

Re-exports fixtures from the parent module.
Credentials are loaded from .env.test automatically.

See tests/conftest_integration.py for cleanup behavior documentation.
"""

from tests.conftest_integration import (
    # Pytest hooks (must be re-exported for pytest to find them)
    pytest_addoption,
    # Settings and helpers
    api_client,
    api_client_2,
    get_api_key,
    get_api_key_2,
    get_base_url,
    get_test_agent_id,
    get_test_agent_id_2,
    get_user_api_key,
    get_ws_url,
    integration_settings,
    # Skip markers
    requires_api,
    requires_multi_agent,
    requires_user_api,
    # Test fixtures
    test_chat,
    test_peer_id,
    test_settings,
    user_api_client,
    # Cleanup helper
    is_no_clean_mode,
)

__all__ = [
    # Pytest hooks
    "pytest_addoption",
    # Settings and helpers
    "api_client",
    "api_client_2",
    "get_api_key",
    "get_api_key_2",
    "get_base_url",
    "get_test_agent_id",
    "get_test_agent_id_2",
    "get_user_api_key",
    "get_ws_url",
    "integration_settings",
    # Skip markers
    "requires_api",
    "requires_multi_agent",
    "requires_user_api",
    # Test fixtures
    "test_chat",
    "test_peer_id",
    "test_settings",
    "user_api_client",
    # Cleanup helper
    "is_no_clean_mode",
]
