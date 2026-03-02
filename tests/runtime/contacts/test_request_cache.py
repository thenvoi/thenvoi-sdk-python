"""Tests for contact request metadata cache behavior."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from thenvoi.client.streaming import (
    ContactAddedPayload,
    ContactRequestReceivedPayload,
    ContactRequestUpdatedPayload,
)
from thenvoi.platform.event import (
    ContactAddedEvent,
    ContactRequestReceivedEvent,
    ContactRequestUpdatedEvent,
)
from thenvoi.runtime.contacts.request_cache import ContactRequestInfoStore


def _received_event(
    request_id: str, *, from_name: str = "Alice"
) -> ContactRequestReceivedEvent:
    return ContactRequestReceivedEvent(
        payload=ContactRequestReceivedPayload(
            id=request_id,
            from_handle="@alice",
            from_name=from_name,
            message="hello",
            status="pending",
            inserted_at="2024-01-01T00:00:00Z",
        )
    )


def _updated_event(request_id: str) -> ContactRequestUpdatedEvent:
    return ContactRequestUpdatedEvent(
        payload=ContactRequestUpdatedPayload(id=request_id, status="approved")
    )


def test_cache_from_received_event_populates_store() -> None:
    store = ContactRequestInfoStore(MagicMock(), max_cache_size=2)
    event = _received_event("req-1")

    store.cache_from_event(event)

    assert store.get_cached("req-1") == {
        "from_handle": "@alice",
        "from_name": "Alice",
        "message": "hello",
    }


def test_cache_from_event_ignores_non_request_received_events() -> None:
    store = ContactRequestInfoStore(MagicMock(), max_cache_size=2)
    event = ContactAddedEvent(
        payload=ContactAddedPayload(
            id="contact-1",
            handle="@bob",
            name="Bob",
            type="User",
            inserted_at="2024-01-01T00:00:00Z",
        )
    )

    store.cache_from_event(event)

    assert store.get_cached("contact-1") is None


@pytest.mark.asyncio
async def test_enrich_update_event_prefers_cached_value() -> None:
    store = ContactRequestInfoStore(MagicMock(), max_cache_size=2)
    store.cache_from_event(_received_event("req-1", from_name="Cached"))
    store.fetch_request_details = AsyncMock(return_value={"from_name": "Fetched"})  # type: ignore[method-assign]

    info = await store.enrich_update_event(_updated_event("req-1"))

    assert info == {
        "from_handle": "@alice",
        "from_name": "Cached",
        "message": "hello",
    }
    store.fetch_request_details.assert_not_called()


@pytest.mark.asyncio
async def test_enrich_update_event_handles_missing_payload() -> None:
    store = ContactRequestInfoStore(MagicMock(), max_cache_size=2)
    event = ContactRequestUpdatedEvent(payload=None)

    info = await store.enrich_update_event(event)

    assert info is None


@pytest.mark.asyncio
async def test_fetch_request_details_uses_received_results_and_trims_cache() -> None:
    contact_service = MagicMock()
    contact_service.list_contact_requests = AsyncMock(
        return_value={
            "received": [
                {
                    "id": "req-1",
                    "from_handle": "@alice",
                    "from_name": "Alice",
                    "message": "hello",
                }
            ],
            "sent": [],
        }
    )
    store = ContactRequestInfoStore(contact_service, max_cache_size=1)
    store.cache_from_event(_received_event("old"))

    info = await store.fetch_request_details("req-1")

    assert info == {
        "from_handle": "@alice",
        "from_name": "Alice",
        "message": "hello",
    }
    assert store.get_cached("old") is None
    contact_service.list_contact_requests.assert_awaited_once_with(
        page=1, page_size=100
    )


@pytest.mark.asyncio
async def test_fetch_request_details_uses_sent_results() -> None:
    contact_service = MagicMock()
    contact_service.list_contact_requests = AsyncMock(
        return_value={
            "received": [],
            "sent": [
                {
                    "id": "req-2",
                    "to_handle": "@bob",
                    "to_name": "Bob",
                    "message": "hi",
                }
            ],
        }
    )
    store = ContactRequestInfoStore(contact_service, max_cache_size=2)

    info = await store.fetch_request_details("req-2")

    assert info == {
        "to_handle": "@bob",
        "to_name": "Bob",
        "message": "hi",
    }


@pytest.mark.asyncio
async def test_fetch_request_details_handles_service_errors() -> None:
    contact_service = MagicMock()
    contact_service.list_contact_requests = AsyncMock(
        side_effect=RuntimeError("service unavailable")
    )
    store = ContactRequestInfoStore(contact_service, max_cache_size=2)

    info = await store.fetch_request_details("req-1")

    assert info is None
