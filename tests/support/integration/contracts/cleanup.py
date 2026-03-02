"""Canonical cleanup contract for integration tests."""

from __future__ import annotations

import logging

from thenvoi_rest import AsyncRestClient

from tests.support.integration import cleanup as cleanup_module

CleanupStepFailure = cleanup_module.CleanupStepFailure
ContactCleanupError = cleanup_module.ContactCleanupError
_error_detail = cleanup_module._error_detail
_iter_contact_handles = cleanup_module._iter_contact_handles
_iter_pending_counterparty_requests = cleanup_module._iter_pending_counterparty_requests
is_no_clean_mode = cleanup_module.is_no_clean_mode
_is_contact_state_settled = cleanup_module._is_contact_state_settled


async def _wait_for_contact_state_settled(
    api_client: AsyncRestClient,
    api_client_2: AsyncRestClient,
    *,
    agent1_handle: str,
    agent2_handle: str,
    poll_interval_s: float,
    timeout_s: float,
    log: logging.Logger,
) -> bool:
    """Compatibility wrapper preserving monkeypatch points in this contract module."""
    return await cleanup_module._wait_for_contact_state_settled(
        api_client,
        api_client_2,
        agent1_handle=agent1_handle,
        agent2_handle=agent2_handle,
        poll_interval_s=poll_interval_s,
        timeout_s=timeout_s,
        log=log,
        probe=_is_contact_state_settled,
    )


async def cleanup_contact_state(
    api_client: AsyncRestClient | None,
    api_client_2: AsyncRestClient | None,
    *,
    log: logging.Logger | None = None,
    settle_delay_s: float = 0.3,
    settle_timeout_s: float = 5.0,
    best_effort: bool | None = None,
) -> None:
    """Compatibility wrapper preserving monkeypatch points in this contract module."""
    await cleanup_module.cleanup_contact_state(
        api_client,
        api_client_2,
        log=log,
        settle_delay_s=settle_delay_s,
        settle_timeout_s=settle_timeout_s,
        best_effort=best_effort,
        wait_for_settle=_wait_for_contact_state_settled,
    )

__all__ = [
    "CleanupStepFailure",
    "ContactCleanupError",
    "_error_detail",
    "_is_contact_state_settled",
    "_iter_contact_handles",
    "_iter_pending_counterparty_requests",
    "_wait_for_contact_state_settled",
    "cleanup_contact_state",
    "is_no_clean_mode",
]
