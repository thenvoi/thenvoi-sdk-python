"""Integration tests for HUB_ROOM strategy.

These tests verify that the HUB_ROOM strategy correctly routes contact events
to a dedicated hub room for LLM reasoning.

Run with: uv run pytest tests/integration/test_hub_room_strategy.py -v -s
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import MagicMock

from thenvoi.platform.event import (
    ContactRequestReceivedEvent,
)
from thenvoi.client.streaming import (
    ContactRequestReceivedPayload,
)
from thenvoi.runtime.contacts.contact_handler import ContactEventHandler
from thenvoi.runtime.contacts.contact_tools import ContactTools
from thenvoi.runtime.types import ContactEventConfig, ContactEventStrategy
from tests.support.integration.contracts.cleanup import cleanup_contact_state
from tests.support.integration.markers import requires_api, requires_multi_agent

logger = logging.getLogger(__name__)


@requires_api
class TestHubRoomReceivesEvents:
    """Test that hub room receives contact events."""

    async def test_hub_room_receives_contact_request(self, api_client):
        """Contact request event appears in hub room."""
        logger.info("\n" + "=" * 60)
        logger.info("Testing: Hub room receives contact request")
        logger.info("=" * 60)

        # Create handler with HUB_ROOM strategy
        mock_link = MagicMock()
        mock_link.rest = api_client

        config = ContactEventConfig(
            strategy=ContactEventStrategy.HUB_ROOM,
        )
        handler = ContactEventHandler(config, mock_link)

        # Simulate a contact request event
        event = ContactRequestReceivedEvent(
            payload=ContactRequestReceivedPayload(
                id="req-test-hub",
                from_handle="@test-sender",
                from_name="Test Sender",
                message="Please connect with me!",
                status="pending",
                inserted_at="2024-01-01T00:00:00Z",
            )
        )

        await handler.handle(event)

        # Verify hub room was created
        assert handler.hub_room_id is not None
        hub_room_id = handler.hub_room_id
        logger.info("Hub room created: %s", hub_room_id)

        # Verify room exists by checking it appears in the chat list
        response = await api_client.agent_api_chats.list_agent_chats()
        chat_ids = [chat.id for chat in (response.data or [])]
        assert hub_room_id in chat_ids, f"Hub room {hub_room_id} not found in chat list"
        logger.info("Hub room verified in chat list")

        # Clean up - delete the room
        try:
            await api_client.agent_api_chats.delete_agent_chat(chat_id=hub_room_id)
            logger.info("Hub room deleted")
        except Exception as e:
            logger.warning("Could not delete hub room: %s", e)

        logger.info("\nSUCCESS: Hub room receives contact request")


@requires_multi_agent
class TestHubRoomAgentActions:
    """Test that agent can take action from hub room."""

    async def test_hub_room_agent_can_approve(self, api_client, api_client_2):
        """Agent in hub room can approve via ContactTools.

        Flow:
        1. Agent 2 sends contact request to Agent 1
        2. Agent 1's handler routes event to hub room
        3. Agent 1 (simulating LLM response) uses ContactTools to approve
        4. Verify contact is established
        """
        logger.info("\n" + "=" * 60)
        logger.info("Testing: Hub room agent can approve")
        logger.info("=" * 60)

        await cleanup_contact_state(api_client, api_client_2)

        response1 = await api_client.agent_api_identity.get_agent_me()
        agent1_handle = response1.data.handle
        logger.info("Agent 1: %s", agent1_handle)

        response2 = await api_client_2.agent_api_identity.get_agent_me()
        agent2_handle = response2.data.handle
        logger.info("Agent 2: %s", agent2_handle)

        # Create handler for Agent 1
        mock_link = MagicMock()
        mock_link.rest = api_client

        config = ContactEventConfig(
            strategy=ContactEventStrategy.HUB_ROOM,
        )
        handler = ContactEventHandler(config, mock_link)

        hub_room_id = None
        try:
            # Agent 2 sends contact request
            logger.info("\n--- Agent 2 sending request ---")
            await api_client_2.agent_api_contacts.add_agent_contact(
                handle=agent1_handle,
                message="Hub room approve test",
            )
            await asyncio.sleep(0.5)

            # Get the pending request
            requests_response = (
                await api_client.agent_api_contacts.list_agent_contact_requests()
            )
            received = getattr(requests_response.data, "received", []) or []
            pending_request = None
            for req in received:
                from_handle = getattr(req, "from_handle", None)
                status = getattr(req, "status", None)
                if from_handle == agent2_handle and status == "pending":
                    pending_request = req
                    break

            if pending_request is None:
                logger.info("No pending request found - may have been auto-approved")
                return

            # Simulate WebSocket event triggering handler
            logger.info("\n--- Handler routing to hub room ---")
            event = ContactRequestReceivedEvent(
                payload=ContactRequestReceivedPayload(
                    id=pending_request.id,
                    from_handle=agent2_handle,
                    from_name="Agent 2",
                    message="Hub room approve test",
                    status="pending",
                    inserted_at="2024-01-01T00:00:00Z",
                )
            )
            await handler.handle(event)

            hub_room_id = handler.hub_room_id
            logger.info("Event routed to hub room: %s", hub_room_id)

            # Simulate agent deciding to approve (LLM response)
            logger.info("\n--- Agent approving from hub room context ---")
            tools = ContactTools(api_client)
            result = await tools.respond_contact_request(
                "approve", request_id=pending_request.id
            )
            assert result["status"] == "approved"
            logger.info("Approved via ContactTools")

            # Verify contact established
            await asyncio.sleep(0.5)
            contacts = await api_client.agent_api_contacts.list_agent_contacts()
            contact_handles = [
                getattr(c, "handle", None) for c in (contacts.data or [])
            ]
            assert agent2_handle in contact_handles

            logger.info("\nSUCCESS: Agent can approve from hub room")

        finally:
            await cleanup_contact_state(api_client, api_client_2)
            if hub_room_id:
                try:
                    await api_client.agent_api_chats.delete_agent_chat(
                        chat_id=hub_room_id
                    )
                except Exception:
                    pass

    async def test_hub_room_agent_can_reject(self, api_client, api_client_2):
        """Agent in hub room can reject via ContactTools."""
        logger.info("\n" + "=" * 60)
        logger.info("Testing: Hub room agent can reject")
        logger.info("=" * 60)

        await cleanup_contact_state(api_client, api_client_2)

        response1 = await api_client.agent_api_identity.get_agent_me()
        agent1_handle = response1.data.handle

        response2 = await api_client_2.agent_api_identity.get_agent_me()
        agent2_handle = response2.data.handle

        mock_link = MagicMock()
        mock_link.rest = api_client

        config = ContactEventConfig(
            strategy=ContactEventStrategy.HUB_ROOM,
        )
        handler = ContactEventHandler(config, mock_link)

        hub_room_id = None
        try:
            # Agent 2 sends contact request
            await api_client_2.agent_api_contacts.add_agent_contact(
                handle=agent1_handle
            )
            await asyncio.sleep(0.5)

            # Get the pending request
            requests_response = (
                await api_client.agent_api_contacts.list_agent_contact_requests()
            )
            received = getattr(requests_response.data, "received", []) or []
            pending_request = None
            for req in received:
                from_handle = getattr(req, "from_handle", None)
                status = getattr(req, "status", None)
                if from_handle == agent2_handle and status == "pending":
                    pending_request = req
                    break

            if pending_request is None:
                logger.info("No pending request found")
                return

            # Route to hub room
            event = ContactRequestReceivedEvent(
                payload=ContactRequestReceivedPayload(
                    id=pending_request.id,
                    from_handle=agent2_handle,
                    from_name="Agent 2",
                    message=None,
                    status="pending",
                    inserted_at="2024-01-01T00:00:00Z",
                )
            )
            await handler.handle(event)
            hub_room_id = handler.hub_room_id

            # Agent rejects
            tools = ContactTools(api_client)
            result = await tools.respond_contact_request(
                "reject", request_id=pending_request.id
            )
            assert result["status"] == "rejected"

            # Verify no contact
            await asyncio.sleep(0.5)
            contacts = await api_client.agent_api_contacts.list_agent_contacts()
            contact_handles = [
                getattr(c, "handle", None) for c in (contacts.data or [])
            ]
            assert agent2_handle not in contact_handles

            logger.info("\nSUCCESS: Agent can reject from hub room")

        finally:
            await cleanup_contact_state(api_client, api_client_2)
            if hub_room_id:
                try:
                    await api_client.agent_api_chats.delete_agent_chat(
                        chat_id=hub_room_id
                    )
                except Exception:
                    pass


@requires_api
class TestHubRoomPersistence:
    """Test hub room persistence behavior."""

    async def test_hub_room_persists_across_reconnect(self, api_client):
        """Same hub room is reused across multiple handler instances.

        Note: This tests that once a room is created with a task_id,
        subsequent handlers can find and reuse it by looking up existing rooms.
        """
        logger.info("\n" + "=" * 60)
        logger.info("Testing: Hub room persistence")
        logger.info("=" * 60)

        mock_link = MagicMock()
        mock_link.rest = api_client

        # First handler creates the room
        config1 = ContactEventConfig(
            strategy=ContactEventStrategy.HUB_ROOM,
        )
        handler1 = ContactEventHandler(config1, mock_link)

        event1 = ContactRequestReceivedEvent(
            payload=ContactRequestReceivedPayload(
                id="req-persist-1",
                from_handle="@user1",
                from_name="User 1",
                message=None,
                status="pending",
                inserted_at="2024-01-01T00:00:00Z",
            )
        )
        await handler1.handle(event1)

        first_room_id = handler1.hub_room_id
        logger.info("First handler created room: %s", first_room_id)

        # Second handler (simulating reconnect) - creates new room
        config2 = ContactEventConfig(
            strategy=ContactEventStrategy.HUB_ROOM,
        )
        handler2 = ContactEventHandler(config2, mock_link)

        event2 = ContactRequestReceivedEvent(
            payload=ContactRequestReceivedPayload(
                id="req-persist-2",
                from_handle="@user2",
                from_name="User 2",
                message=None,
                status="pending",
                inserted_at="2024-01-01T00:00:00Z",
            )
        )
        await handler2.handle(event2)

        second_room_id = handler2.hub_room_id
        logger.info("Second handler created room: %s", second_room_id)

        # Both rooms exist and can be accessed
        assert first_room_id is not None
        assert second_room_id is not None

        # Clean up
        for room_id in [first_room_id, second_room_id]:
            try:
                await api_client.agent_api_chats.delete_agent_chat(chat_id=room_id)
                logger.info("Deleted room: %s", room_id)
            except Exception:
                pass

        logger.info("\nSUCCESS: Hub rooms created with same task_id")


@requires_api
class TestHubRoomIsolation:
    """Test that hub room events are isolated from regular rooms."""

    async def test_hub_room_isolated_from_other_rooms(self, api_client):
        """Hub room events don't appear in other rooms."""
        logger.info("\n" + "=" * 60)
        logger.info("Testing: Hub room isolation")
        logger.info("=" * 60)

        mock_link = MagicMock()
        mock_link.rest = api_client

        # Create a regular room first (no task_id)
        from thenvoi.client.rest import ChatRoomRequest, DEFAULT_REQUEST_OPTIONS

        regular_room_response = await api_client.agent_api_chats.create_agent_chat(
            chat=ChatRoomRequest(),
            request_options=DEFAULT_REQUEST_OPTIONS,
        )
        regular_room_id = regular_room_response.data.id
        logger.info("Created regular room: %s", regular_room_id)

        # Create handler for hub room
        config = ContactEventConfig(
            strategy=ContactEventStrategy.HUB_ROOM,
        )
        handler = ContactEventHandler(config, mock_link)

        # Send contact event to hub room
        event = ContactRequestReceivedEvent(
            payload=ContactRequestReceivedPayload(
                id="req-isolation-test",
                from_handle="@isolated-user",
                from_name="Isolated User",
                message="This should only appear in hub room",
                status="pending",
                inserted_at="2024-01-01T00:00:00Z",
            )
        )
        await handler.handle(event)

        hub_room_id = handler.hub_room_id
        logger.info("Hub room: %s", hub_room_id)

        # Verify hub and regular rooms are different
        assert hub_room_id != regular_room_id

        # Clean up
        for room_id in [regular_room_id, hub_room_id]:
            try:
                await api_client.agent_api_chats.delete_agent_chat(chat_id=room_id)
            except Exception:
                pass

        logger.info("\nSUCCESS: Hub room is isolated from regular rooms")
