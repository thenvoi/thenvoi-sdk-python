"""Unit tests for ThenvoiLink contact subscription."""

from __future__ import annotations

from unittest.mock import patch

import pytest

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
from thenvoi.platform.link import ThenvoiLink


@pytest.fixture
def mock_ws_client(ws_client_mock_factory):
    """Mock WebSocketClient for testing."""
    return ws_client_mock_factory()


@pytest.fixture
async def connected_link(
    mock_ws_client,
):
    """Connected ThenvoiLink using shared websocket fake boundary."""
    with patch("thenvoi.platform.link.WebSocketClient", return_value=mock_ws_client):
        link = ThenvoiLink(agent_id="agent-123", api_key="test-key")
        await link.connect()
        yield link


class TestContactSubscription:
    """Tests for contact channel subscription."""

    async def test_subscribe_agent_contacts_joins_channel(
        self, connected_link: ThenvoiLink, mock_ws_client
    ) -> None:
        """subscribe_agent_contacts() should join agent contacts channel."""
        await connected_link.subscribe_agent_contacts("agent-123")

        mock_ws_client.join_agent_contacts_channel.assert_called_once()
        call_args = mock_ws_client.join_agent_contacts_channel.call_args
        assert call_args[0][0] == "agent-123"  # agent_id

    async def test_subscribe_agent_contacts_passes_all_handlers(
        self, connected_link: ThenvoiLink, mock_ws_client
    ) -> None:
        """subscribe_agent_contacts() should pass all 4 event handlers."""
        await connected_link.subscribe_agent_contacts("agent-123")

        call_kwargs = mock_ws_client.join_agent_contacts_channel.call_args[1]
        assert "on_contact_request_received" in call_kwargs
        assert "on_contact_request_updated" in call_kwargs
        assert "on_contact_added" in call_kwargs
        assert "on_contact_removed" in call_kwargs

    async def test_subscribe_agent_contacts_requires_connection(
        self, mock_ws_client
    ) -> None:
        """subscribe_agent_contacts() should raise when not connected."""
        with patch("thenvoi.platform.link.WebSocketClient", return_value=mock_ws_client):
            link = ThenvoiLink(agent_id="agent-123", api_key="test-key")
            with pytest.raises(RuntimeError, match="Not connected"):
                await link.subscribe_agent_contacts("agent-123")

    async def test_unsubscribe_agent_contacts_leaves_channel(
        self, connected_link: ThenvoiLink, mock_ws_client
    ) -> None:
        """unsubscribe_agent_contacts() should leave agent contacts channel."""
        await connected_link.unsubscribe_agent_contacts()

        mock_ws_client.leave_agent_contacts_channel.assert_called_once_with("agent-123")

    async def test_unsubscribe_agent_contacts_handles_errors(
        self, connected_link: ThenvoiLink, mock_ws_client
    ) -> None:
        """unsubscribe_agent_contacts() should handle errors gracefully."""
        mock_ws_client.leave_agent_contacts_channel.side_effect = Exception(
            "Leave failed"
        )

        # Should not raise
        await connected_link.unsubscribe_agent_contacts()

    async def test_unsubscribe_agent_contacts_noop_when_not_connected(self) -> None:
        """unsubscribe_agent_contacts() should be no-op when not connected."""
        link = ThenvoiLink(agent_id="agent-123", api_key="test-key")
        # Should not raise
        await link.unsubscribe_agent_contacts()


class TestContactEventHandlers:
    """Tests for contact event handlers."""

    async def test_on_contact_request_received_queues_event(
        self, connected_link: ThenvoiLink
    ) -> None:
        """_on_contact_request_received() should queue ContactRequestReceivedEvent."""
        payload = ContactRequestReceivedPayload(
            id="req-123",
            from_handle="john_doe",
            from_name="John Doe",
            status="pending",
            inserted_at="2026-02-09T10:30:00Z",
        )
        await connected_link._on_contact_request_received(payload)

        event = await connected_link._event_queue.get()
        assert isinstance(event, ContactRequestReceivedEvent)
        assert event.payload.id == "req-123"
        assert event.room_id is None

    async def test_on_contact_request_updated_queues_event(
        self, connected_link: ThenvoiLink
    ) -> None:
        """_on_contact_request_updated() should queue ContactRequestUpdatedEvent."""
        payload = ContactRequestUpdatedPayload(
            id="req-123",
            status="approved",
        )
        await connected_link._on_contact_request_updated(payload)

        event = await connected_link._event_queue.get()
        assert isinstance(event, ContactRequestUpdatedEvent)
        assert event.payload.status == "approved"
        assert event.room_id is None

    async def test_on_contact_added_queues_event(
        self, connected_link: ThenvoiLink
    ) -> None:
        """_on_contact_added() should queue ContactAddedEvent."""
        payload = ContactAddedPayload(
            id="contact-123",
            handle="jane_smith",
            name="Jane Smith",
            type="User",
            inserted_at="2026-02-09T10:35:00Z",
        )
        await connected_link._on_contact_added(payload)

        event = await connected_link._event_queue.get()
        assert isinstance(event, ContactAddedEvent)
        assert event.payload.name == "Jane Smith"
        assert event.room_id is None

    async def test_on_contact_removed_queues_event(
        self, connected_link: ThenvoiLink
    ) -> None:
        """_on_contact_removed() should queue ContactRemovedEvent."""
        payload = ContactRemovedPayload(id="contact-123")
        await connected_link._on_contact_removed(payload)

        event = await connected_link._event_queue.get()
        assert isinstance(event, ContactRemovedEvent)
        assert event.payload.id == "contact-123"
        assert event.room_id is None


class TestPublicQueueMethod:
    """Tests for public queue_event() method."""

    async def test_queue_event_public_method(self, connected_link: ThenvoiLink) -> None:
        """queue_event() should add event to queue (public API)."""
        event = ContactAddedEvent(
            payload=ContactAddedPayload(
                id="contact-123",
                handle="test",
                name="Test",
                type="User",
                inserted_at="2026-01-01T00:00:00Z",
            )
        )
        connected_link.queue_event(event)

        queued = await connected_link._event_queue.get()
        assert queued is event

    def test_queue_event_works_without_connection(self) -> None:
        """queue_event() should work even when not connected."""
        link = ThenvoiLink(agent_id="agent-123", api_key="test-key")

        event = ContactRemovedEvent(payload=ContactRemovedPayload(id="contact-123"))
        link.queue_event(event)

        assert link._event_queue.qsize() == 1
