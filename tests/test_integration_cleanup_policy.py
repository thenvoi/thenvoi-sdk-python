"""Unit tests for integration cleanup isolation helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import tests.support.integration.contracts.cleanup as integration_config


def _response(data: object) -> SimpleNamespace:
    return SimpleNamespace(data=data)


def test_iter_contact_handles_supports_both_data_shapes() -> None:
    direct_data = [SimpleNamespace(handle="@alice"), SimpleNamespace(handle="@bob")]
    wrapped_data = SimpleNamespace(contacts=direct_data)

    direct_handles = integration_config._iter_contact_handles(direct_data)
    wrapped_handles = integration_config._iter_contact_handles(wrapped_data)

    assert direct_handles == {"@alice", "@bob"}
    assert wrapped_handles == {"@alice", "@bob"}


def test_iter_pending_counterparty_requests_checks_received_and_sent() -> None:
    requests_data = SimpleNamespace(
        received=[
            SimpleNamespace(status="pending", from_handle="@bob"),
            SimpleNamespace(status="approved", from_handle="@bob"),
        ],
        sent=[
            SimpleNamespace(status="pending", to_handle="@bob"),
            SimpleNamespace(status="pending", to_handle="@carol"),
        ],
    )

    pending = integration_config._iter_pending_counterparty_requests(
        requests_data,
        "@bob",
    )

    assert len(pending) == 2


@pytest.mark.asyncio
async def test_wait_for_contact_state_settled_polls_until_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = AsyncMock(side_effect=[False, False, True])
    monkeypatch.setattr(integration_config, "_is_contact_state_settled", probe)

    result = await integration_config._wait_for_contact_state_settled(
        AsyncMock(),
        AsyncMock(),
        agent1_handle="@alice",
        agent2_handle="@bob",
        poll_interval_s=0.001,
        timeout_s=1.0,
        log=AsyncMock(),
    )

    assert result is True
    assert probe.await_count == 3


@pytest.mark.asyncio
async def test_is_contact_state_settled_detects_pending_or_existing_contacts() -> None:
    api_client = AsyncMock()
    api_client_2 = AsyncMock()

    api_client.agent_api_contacts.list_agent_contacts = AsyncMock(
        return_value=_response([SimpleNamespace(handle="@bob")])
    )
    api_client_2.agent_api_contacts.list_agent_contacts = AsyncMock(
        return_value=_response([])
    )
    api_client.agent_api_contacts.list_agent_contact_requests = AsyncMock(
        return_value=_response(SimpleNamespace(received=[], sent=[]))
    )
    api_client_2.agent_api_contacts.list_agent_contact_requests = AsyncMock(
        return_value=_response(
            SimpleNamespace(
                received=[SimpleNamespace(status="pending", from_handle="@alice")],
                sent=[],
            )
        )
    )

    settled = await integration_config._is_contact_state_settled(
        api_client,
        api_client_2,
        agent1_handle="@alice",
        agent2_handle="@bob",
        log=AsyncMock(),
    )

    assert settled is False


@pytest.mark.asyncio
async def test_is_contact_state_settled_returns_true_when_clean() -> None:
    api_client = AsyncMock()
    api_client_2 = AsyncMock()

    api_client.agent_api_contacts.list_agent_contacts = AsyncMock(
        return_value=_response([])
    )
    api_client_2.agent_api_contacts.list_agent_contacts = AsyncMock(
        return_value=_response([])
    )
    api_client.agent_api_contacts.list_agent_contact_requests = AsyncMock(
        return_value=_response(SimpleNamespace(received=[], sent=[]))
    )
    api_client_2.agent_api_contacts.list_agent_contact_requests = AsyncMock(
        return_value=_response(SimpleNamespace(received=[], sent=[]))
    )

    settled = await integration_config._is_contact_state_settled(
        api_client,
        api_client_2,
        agent1_handle="@alice",
        agent2_handle="@bob",
        log=AsyncMock(),
    )

    assert settled is True


def _client_with_identity(handle: str) -> AsyncMock:
    client = AsyncMock()
    client.agent_api_identity.get_agent_me = AsyncMock(
        return_value=_response(SimpleNamespace(handle=handle))
    )
    client.agent_api_contacts.remove_agent_contact = AsyncMock()
    client.agent_api_contacts.respond_to_agent_contact_request = AsyncMock()
    client.agent_api_contacts.list_agent_contact_requests = AsyncMock(
        return_value=_response(SimpleNamespace(received=[], sent=[]))
    )
    return client


@pytest.mark.asyncio
async def test_cleanup_contact_state_raises_when_strict_and_cleanup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client1 = _client_with_identity("@alice")
    client2 = _client_with_identity("@bob")
    client1.agent_api_contacts.remove_agent_contact = AsyncMock(
        side_effect=RuntimeError("remove failed")
    )
    monkeypatch.setattr(
        integration_config,
        "_wait_for_contact_state_settled",
        AsyncMock(return_value=True),
    )

    with pytest.raises(integration_config.ContactCleanupError, match="remove_contact"):
        await integration_config.cleanup_contact_state(
            client1,
            client2,
            best_effort=False,
        )


@pytest.mark.asyncio
async def test_cleanup_contact_state_best_effort_does_not_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client1 = _client_with_identity("@alice")
    client2 = _client_with_identity("@bob")
    client2.agent_api_contacts.respond_to_agent_contact_request = AsyncMock(
        side_effect=RuntimeError("cancel failed")
    )
    monkeypatch.setattr(
        integration_config,
        "_wait_for_contact_state_settled",
        AsyncMock(return_value=True),
    )

    await integration_config.cleanup_contact_state(
        client1,
        client2,
        best_effort=True,
    )
