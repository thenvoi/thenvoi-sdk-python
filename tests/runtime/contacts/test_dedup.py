"""Tests for contact event dedup cache."""

from __future__ import annotations

from thenvoi.client.streaming import (
    ContactAddedPayload,
    ContactRemovedPayload,
    ContactRequestReceivedPayload,
    ContactRequestUpdatedPayload,
)
from thenvoi.platform.event import (
    ContactAddedEvent,
    ContactRemovedEvent,
    ContactRequestReceivedEvent,
    ContactRequestUpdatedEvent,
)
from thenvoi.runtime.contacts.dedup import ContactDedupCache


def test_key_generation_for_supported_events() -> None:
    cache = ContactDedupCache(max_size=4)

    received_key = cache.key_for(
        ContactRequestReceivedEvent(
            payload=ContactRequestReceivedPayload(
                id="req-1",
                from_handle="@alice",
                from_name="Alice",
                message="hello",
                status="pending",
                inserted_at="2024-01-01T00:00:00Z",
            )
        )
    )
    updated_key = cache.key_for(
        ContactRequestUpdatedEvent(
            payload=ContactRequestUpdatedPayload(id="req-1", status="approved")
        )
    )
    added_key = cache.key_for(
        ContactAddedEvent(
            payload=ContactAddedPayload(
                id="contact-1",
                handle="@bob",
                name="Bob",
                type="User",
                inserted_at="2024-01-01T00:00:00Z",
            )
        )
    )
    removed_key = cache.key_for(
        ContactRemovedEvent(payload=ContactRemovedPayload(id="contact-2"))
    )

    assert received_key == "request_received:req-1"
    assert updated_key == "request_updated:req-1:approved"
    assert added_key == "contact_added:contact-1"
    assert removed_key == "contact_removed:contact-2"


def test_mark_processed_and_should_skip() -> None:
    cache = ContactDedupCache(max_size=4)
    event = ContactRemovedEvent(payload=ContactRemovedPayload(id="contact-2"))

    assert cache.should_skip(event) is False

    cache.mark_processed(event)

    assert cache.should_skip(event) is True


def test_clear_removes_processed_key() -> None:
    cache = ContactDedupCache(max_size=4)
    event = ContactRemovedEvent(payload=ContactRemovedPayload(id="contact-2"))
    cache.mark_processed(event)

    cache.clear(event)

    assert cache.should_skip(event) is False


def test_cache_eviction_respects_max_size() -> None:
    cache = ContactDedupCache(max_size=1)
    first = ContactRemovedEvent(payload=ContactRemovedPayload(id="contact-1"))
    second = ContactRemovedEvent(payload=ContactRemovedPayload(id="contact-2"))

    cache.mark_processed(first)
    cache.mark_processed(second)

    assert cache.should_skip(first) is False
    assert cache.should_skip(second) is True
    assert len(cache.storage) == 1
