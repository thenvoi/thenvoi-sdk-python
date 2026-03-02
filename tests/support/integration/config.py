"""Deprecated compatibility facade for integration support contracts.

Prefer importing from `tests.support.integration.contracts.*` directly.
Planned removal target: 2026-06-30.
"""

from __future__ import annotations

from tests.support.integration.contracts.cleanup import (
    CleanupStepFailure,
    ContactCleanupError,
    _error_detail,
    _is_contact_state_settled,
    _iter_contact_handles,
    _iter_pending_counterparty_requests,
    _wait_for_contact_state_settled,
    cleanup_contact_state,
    is_no_clean_mode,
)
from tests.support.integration.contracts.markers import (
    _enforce_live_fixture_environment,
    _enforce_live_fixture_policy,
    _is_integration_mode,
    _is_truthy_env,
)
from tests.support.integration.contracts.selection import (
    _peer_sort_key,
    _select_preferred_peer,
)
from tests.support.integration.contracts.settings import TestSettings, get_test_settings
from tests.support.integration.fixtures import (
    api_client,
    api_client_2,
    integration_run_id,
    integration_settings,
    test_chat,
    test_peer_id,
    user_api_client,
)

__all__ = [
    "CleanupStepFailure",
    "ContactCleanupError",
    "TestSettings",
    "_enforce_live_fixture_environment",
    "_enforce_live_fixture_policy",
    "_error_detail",
    "_is_contact_state_settled",
    "_is_integration_mode",
    "_is_truthy_env",
    "_iter_contact_handles",
    "_iter_pending_counterparty_requests",
    "_peer_sort_key",
    "_select_preferred_peer",
    "_wait_for_contact_state_settled",
    "api_client",
    "api_client_2",
    "cleanup_contact_state",
    "get_test_settings",
    "integration_run_id",
    "integration_settings",
    "is_no_clean_mode",
    "test_chat",
    "test_peer_id",
    "user_api_client",
]
