"""Contact cleanup and no-clean policy helpers for integration tests."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import logging
import os

import pytest
from thenvoi_rest import AsyncRestClient

from tests.support.integration.policy import is_truthy_env

logger = logging.getLogger(__name__)

_CLEANUP_TRANSIENT_ERRORS: tuple[type[Exception], ...] = (
    ConnectionError,
    OSError,
    RuntimeError,
    TimeoutError,
    ValueError,
)


@dataclass(frozen=True)
class CleanupStepFailure:
    """One recoverable cleanup failure captured for post-run reporting."""

    step: str
    detail: str


class ContactCleanupError(RuntimeError):
    """Raised when cleanup fails in strict mode."""


def _cleanup_best_effort_enabled() -> bool:
    return is_truthy_env("THENVOI_TEST_BEST_EFFORT_CLEANUP")


def is_no_clean_mode(request: pytest.FixtureRequest | None = None) -> bool:
    """Check if no-clean mode is enabled via env var or pytest option."""
    if os.environ.get("THENVOI_TEST_NO_CLEAN", "").lower() in ("1", "true", "yes"):
        return True

    if request is not None:
        try:
            return bool(request.config.getoption("--no-clean", default=False))
        except ValueError:
            pass

    return False


def _iter_contact_handles(data: object | None) -> set[str]:
    contacts = data
    if contacts is None:
        return set()

    if hasattr(contacts, "contacts"):
        contacts = getattr(contacts, "contacts", None)

    if contacts is None:
        return set()

    handles: set[str] = set()
    for contact in contacts:
        handle = getattr(contact, "handle", None)
        if handle:
            handles.add(str(handle))
    return handles


def _request_matches_counterparty(request: object, counterparty_handle: str) -> bool:
    if getattr(request, "status", None) != "pending":
        return False

    for attr_name in ("from_handle", "to_handle", "handle"):
        if getattr(request, attr_name, None) == counterparty_handle:
            return True
    return False


def _iter_pending_counterparty_requests(
    requests_data: object | None,
    counterparty_handle: str,
) -> list[object]:
    if requests_data is None:
        return []

    pending: list[object] = []
    received = getattr(requests_data, "received", []) or []
    sent = getattr(requests_data, "sent", []) or []
    for request in [*received, *sent]:
        if _request_matches_counterparty(request, counterparty_handle):
            pending.append(request)
    return pending


async def _is_contact_state_settled(
    api_client: AsyncRestClient,
    api_client_2: AsyncRestClient,
    *,
    agent1_handle: str,
    agent2_handle: str,
    log: logging.Logger,
) -> bool:
    try:
        contacts_1 = await api_client.agent_api_contacts.list_agent_contacts()
        contacts_2 = await api_client_2.agent_api_contacts.list_agent_contacts()
        requests_1 = await api_client.agent_api_contacts.list_agent_contact_requests()
        requests_2 = await api_client_2.agent_api_contacts.list_agent_contact_requests()
    except Exception as exc:
        log.debug("Contact-settle probe failed, retrying: %s", exc)
        return False

    handles_1 = _iter_contact_handles(getattr(contacts_1, "data", None))
    handles_2 = _iter_contact_handles(getattr(contacts_2, "data", None))
    pending_1 = _iter_pending_counterparty_requests(
        getattr(requests_1, "data", None),
        agent2_handle,
    )
    pending_2 = _iter_pending_counterparty_requests(
        getattr(requests_2, "data", None),
        agent1_handle,
    )

    return (
        agent2_handle not in handles_1
        and agent1_handle not in handles_2
        and not pending_1
        and not pending_2
    )


async def _wait_for_contact_state_settled(
    api_client: AsyncRestClient,
    api_client_2: AsyncRestClient,
    *,
    agent1_handle: str,
    agent2_handle: str,
    poll_interval_s: float,
    timeout_s: float,
    log: logging.Logger,
    probe: Callable[..., Awaitable[bool]] | None = None,
) -> bool:
    state_probe = probe or _is_contact_state_settled
    deadline = asyncio.get_running_loop().time() + timeout_s
    while True:
        if await state_probe(
            api_client,
            api_client_2,
            agent1_handle=agent1_handle,
            agent2_handle=agent2_handle,
            log=log,
        ):
            return True

        if asyncio.get_running_loop().time() >= deadline:
            return False

        await asyncio.sleep(poll_interval_s)


async def cleanup_contact_state(
    api_client: AsyncRestClient | None,
    api_client_2: AsyncRestClient | None,
    *,
    log: logging.Logger | None = None,
    settle_delay_s: float = 0.3,
    settle_timeout_s: float = 5.0,
    best_effort: bool | None = None,
    wait_for_settle: Callable[..., Awaitable[bool]] | None = None,
) -> None:
    """Reset two agents to a clean contact state for integration tests."""
    if api_client is None or api_client_2 is None:
        pytest.skip("THENVOI_API_KEY and THENVOI_API_KEY_2 must both be set")

    active_logger = log or logger
    allow_best_effort = (
        _cleanup_best_effort_enabled() if best_effort is None else best_effort
    )
    failures: list[CleanupStepFailure] = []

    async def _run_cleanup_step(
        step: str,
        operation: Callable[[], Awaitable[None]],
    ) -> None:
        try:
            await operation()
        except _CLEANUP_TRANSIENT_ERRORS as exc:
            failures.append(CleanupStepFailure(step=step, detail=_error_detail(exc)))
            active_logger.debug("%s skipped: %s", step, exc)

    async def _reject_pending_requests(
        *,
        source_client: AsyncRestClient,
        counterparty_handle: str,
        step: str,
    ) -> None:
        try:
            response = (
                await source_client.agent_api_contacts.list_agent_contact_requests()
            )
            received = getattr(response.data, "received", []) or []
            for request in received:
                from_handle = getattr(request, "from_handle", None)
                status = getattr(request, "status", None)
                if from_handle == counterparty_handle and status == "pending":
                    await source_client.agent_api_contacts.respond_to_agent_contact_request(
                        action="reject",
                        request_id=request.id,
                    )
                    active_logger.debug("%s: rejected pending inbound request", step)
        except _CLEANUP_TRANSIENT_ERRORS as exc:
            failures.append(CleanupStepFailure(step=step, detail=_error_detail(exc)))
            active_logger.debug("%s skipped: %s", step, exc)

    response1 = await api_client.agent_api_identity.get_agent_me()
    agent1_handle = response1.data.handle

    response2 = await api_client_2.agent_api_identity.get_agent_me()
    agent2_handle = response2.data.handle

    active_logger.info(
        "Cleaning up contact state between %s and %s", agent1_handle, agent2_handle
    )

    await _run_cleanup_step(
        "agent1.remove_contact",
        lambda: api_client.agent_api_contacts.remove_agent_contact(
            handle=agent2_handle
        ),
    )
    await _run_cleanup_step(
        "agent2.remove_contact",
        lambda: api_client_2.agent_api_contacts.remove_agent_contact(
            handle=agent1_handle
        ),
    )
    await _run_cleanup_step(
        "agent1.cancel_request",
        lambda: api_client.agent_api_contacts.respond_to_agent_contact_request(
            action="cancel",
            handle=agent2_handle,
        ),
    )
    await _run_cleanup_step(
        "agent2.cancel_request",
        lambda: api_client_2.agent_api_contacts.respond_to_agent_contact_request(
            action="cancel",
            handle=agent1_handle,
        ),
    )
    await _reject_pending_requests(
        source_client=api_client,
        counterparty_handle=agent2_handle,
        step="agent1.reject_inbound",
    )
    await _reject_pending_requests(
        source_client=api_client_2,
        counterparty_handle=agent1_handle,
        step="agent2.reject_inbound",
    )

    settle_waiter = wait_for_settle or _wait_for_contact_state_settled
    settled = await settle_waiter(
        api_client,
        api_client_2,
        agent1_handle=agent1_handle,
        agent2_handle=agent2_handle,
        poll_interval_s=settle_delay_s,
        timeout_s=settle_timeout_s,
        log=active_logger,
    )
    if not settled:
        failures.append(
            CleanupStepFailure(
                step="settle_timeout",
                detail=(
                    f"timed out after {settle_timeout_s:.1f}s "
                    f"for {agent1_handle} <-> {agent2_handle}"
                ),
            )
        )
        active_logger.warning(
            "Contact cleanup timed out after %.1fs for %s <-> %s",
            settle_timeout_s,
            agent1_handle,
            agent2_handle,
        )
    else:
        active_logger.info("Contact cleanup complete")

    if failures:
        summary = "; ".join(f"{failure.step}: {failure.detail}" for failure in failures)
        if allow_best_effort:
            active_logger.warning(
                "Contact cleanup completed with recoverable failures: %s", summary
            )
            return
        raise ContactCleanupError(f"Contact cleanup failed: {summary}")


def _error_detail(exc: Exception) -> str:
    message = str(exc).strip()
    return message or type(exc).__name__


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
