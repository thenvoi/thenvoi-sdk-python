"""Full workflow integration tests for SDK against real API.

Tests the complete agent workflow from identity to message lifecycle.
Run with: uv run pytest tests/integration/test_full_workflow.py -v -s
"""

import logging

from thenvoi_rest import ChatRoomRequest, ChatMessageRequest, ChatEventRequest
from thenvoi_rest.types import (
    ChatMessageRequestMentionsItem as Mention,
    ParticipantRequest,
)

from tests.integration.conftest import get_test_agent_id, requires_api

logger = logging.getLogger(__name__)


@requires_api
class TestFullWorkflow:
    """End-to-end integration test covering all SDK operations in a realistic workflow."""

    async def test_complete_agent_workflow(self, api_client, integration_settings):
        """Test complete workflow: identity -> chat -> participants -> messages -> events -> lifecycle."""

        # ============================================================
        # STEP 1: Identity - Get agent profile
        # ============================================================
        logger.info("\n" + "=" * 60)
        logger.info("STEP 1: Get Agent Identity")
        logger.info("=" * 60)

        response = await api_client.agent_api.get_agent_me()
        assert response.data is not None, "Agent profile should not be None"

        agent = response.data
        agent_id = agent.id
        agent_name = agent.name
        logger.info(f"Agent: {agent_name} (ID: {agent_id})")

        # Verify against expected test agent if configured
        expected_agent_id = get_test_agent_id()
        if expected_agent_id:
            assert agent_id == expected_agent_id, f"Expected agent {expected_agent_id}"

        # ============================================================
        # STEP 2: Identity - List available peers
        # ============================================================
        logger.info("\n" + "=" * 60)
        logger.info("STEP 2: List Available Peers")
        logger.info("=" * 60)

        response = await api_client.agent_api.list_agent_peers()
        peers = response.data
        assert peers is not None and len(peers) > 0, "Need at least one peer"
        logger.info(f"Found {len(peers)} available peers")

        # Find a User peer (human) - this is the key test: agent communicating with human
        user_peer = next((p for p in peers if p.type == "User"), None)
        assert user_peer is not None, (
            "Need at least one User peer to test agent-human communication"
        )

        peer = user_peer
        peer_id = peer.id
        peer_name = peer.name
        peer_type = peer.type
        logger.info(f"Will use User peer: {peer_name} ({peer_type}, ID: {peer_id})")

        # ============================================================
        # STEP 3: Chats - Create a new chat
        # ============================================================
        logger.info("\n" + "=" * 60)
        logger.info("STEP 3: Create New Chat")
        logger.info("=" * 60)

        response = await api_client.agent_api.create_agent_chat(chat=ChatRoomRequest())
        assert response.data is not None, "Created chat should not be None"

        chat = response.data
        chat_id = chat.id
        logger.info(f"Created chat (ID: {chat_id}, title: {chat.title})")

        # ============================================================
        # STEP 4: Chats - Get the created chat
        # ============================================================
        logger.info("\n" + "=" * 60)
        logger.info("STEP 4: Get Chat Details")
        logger.info("=" * 60)

        response = await api_client.agent_api.get_agent_chat(id=chat_id)
        assert response.data is not None, "Chat should exist"
        assert response.data.id == chat_id, "Chat ID should match"
        logger.info(f"Retrieved chat: {response.data.title}")

        # ============================================================
        # STEP 5: Chats - Verify chat appears in list
        # ============================================================
        logger.info("\n" + "=" * 60)
        logger.info("STEP 5: List Chats (verify new chat appears)")
        logger.info("=" * 60)

        response = await api_client.agent_api.list_agent_chats()
        chat_ids = [c.id for c in response.data] if response.data else []
        assert chat_id in chat_ids, "New chat should appear in chat list"
        logger.info(f"Chat list contains {len(chat_ids)} chats, including our test chat")

        # ============================================================
        # STEP 6: Participants - List initial participants
        # ============================================================
        logger.info("\n" + "=" * 60)
        logger.info("STEP 6: List Initial Participants")
        logger.info("=" * 60)

        response = await api_client.agent_api.list_agent_chat_participants(chat_id)
        initial_participants = response.data or []
        logger.info(f"Initial participants: {len(initial_participants)}")
        for p in initial_participants:
            logger.info(f"  - {p.name} ({p.type}, role: {p.role})")

        # ============================================================
        # STEP 7: Participants - Add peer to chat
        # ============================================================
        logger.info("\n" + "=" * 60)
        logger.info("STEP 7: Add Participant to Chat")
        logger.info("=" * 60)

        await api_client.agent_api.add_agent_chat_participant(
            chat_id,
            participant=ParticipantRequest(participant_id=peer_id, role="member"),
        )
        logger.info(f"Added participant: {peer_name}")

        # ============================================================
        # STEP 8: Participants - Verify participant was added
        # ============================================================
        logger.info("\n" + "=" * 60)
        logger.info("STEP 8: Verify Participant Added")
        logger.info("=" * 60)

        response = await api_client.agent_api.list_agent_chat_participants(chat_id)
        participants = response.data or []
        participant_ids = [p.id for p in participants]
        assert peer_id in participant_ids, "Peer should now be a participant"
        logger.info(f"Participants after adding: {len(participants)}")
        for p in participants:
            logger.info(f"  - {p.name} ({p.type}, role: {p.role})")

        # ============================================================
        # STEP 9: Messages - Send a message with mention
        # ============================================================
        logger.info("\n" + "=" * 60)
        logger.info("STEP 9: Send Message with Mention")
        logger.info("=" * 60)

        message_content = (
            f"Hello @{peer_name}, this is an SDK integration test message!"
        )
        mentions = [Mention(id=peer_id, name=peer_name)]

        response = await api_client.agent_api.create_agent_chat_message(
            chat_id,
            message=ChatMessageRequest(
                content=message_content,
                mentions=mentions,
            ),
        )
        assert response.data is not None, "Message should be created"

        message = response.data
        message_id = message.id
        logger.info(f"Sent message: '{message_content[:50]}...' (ID: {message_id})")

        # ============================================================
        # STEP 10: Messages - Get chat context (verify message)
        # ============================================================
        logger.info("\n" + "=" * 60)
        logger.info("STEP 10: Get Chat Context")
        logger.info("=" * 60)

        response = await api_client.agent_api.get_agent_chat_context(chat_id)
        context = response.data or []
        message_ids = [m.id for m in context if hasattr(m, "id")]
        assert message_id in message_ids, "Our message should appear in context"
        logger.info(f"Chat context contains {len(context)} items")

        # ============================================================
        # STEP 11: Events - Create a thought event
        # ============================================================
        logger.info("\n" + "=" * 60)
        logger.info("STEP 11: Create Thought Event")
        logger.info("=" * 60)

        event_content = "Processing the user's request about integration testing..."
        response = await api_client.agent_api.create_agent_chat_event(
            chat_id,
            event=ChatEventRequest(
                content=event_content,
                message_type="thought",
            ),
        )
        assert response.data is not None, "Event should be created"

        event = response.data
        event_id = event.id
        logger.info(f"Created thought event (ID: {event_id})")

        # ============================================================
        # STEP 12: Events - Create a tool_call event
        # ============================================================
        logger.info("\n" + "=" * 60)
        logger.info("STEP 12: Create Tool Call Event")
        logger.info("=" * 60)

        tool_metadata = {
            "function": {
                "name": "search_database",
                "arguments": {"query": "integration test"},
            }
        }
        response = await api_client.agent_api.create_agent_chat_event(
            chat_id,
            event=ChatEventRequest(
                content="Calling search_database",
                message_type="tool_call",
                metadata=tool_metadata,
            ),
        )
        assert response.data is not None, "Tool call event should be created"
        logger.info(f"Created tool_call event (ID: {response.data.id})")

        # ============================================================
        # STEP 13: Events - Create a tool_result event
        # ============================================================
        logger.info("\n" + "=" * 60)
        logger.info("STEP 13: Create Tool Result Event")
        logger.info("=" * 60)

        result_metadata = {"result": {"found": 5, "items": ["item1", "item2"]}}
        response = await api_client.agent_api.create_agent_chat_event(
            chat_id,
            event=ChatEventRequest(
                content="Search completed successfully",
                message_type="tool_result",
                metadata=result_metadata,
            ),
        )
        assert response.data is not None, "Tool result event should be created"
        logger.info(f"Created tool_result event (ID: {response.data.id})")

        # ============================================================
        # STEP 14: Lifecycle - Mark message as processing
        # ============================================================
        logger.info("\n" + "=" * 60)
        logger.info("STEP 14: Mark Message Processing")
        logger.info("=" * 60)

        await api_client.agent_api.mark_agent_message_processing(chat_id, message_id)
        logger.info(f"Marked message {message_id} as processing")

        # ============================================================
        # STEP 15: Lifecycle - Mark message as processed
        # ============================================================
        logger.info("\n" + "=" * 60)
        logger.info("STEP 15: Mark Message Processed")
        logger.info("=" * 60)

        await api_client.agent_api.mark_agent_message_processed(chat_id, message_id)
        logger.info(f"Marked message {message_id} as processed")

        # ============================================================
        # STEP 16: Verify User still in chat after all operations
        # ============================================================
        logger.info("\n" + "=" * 60)
        logger.info("STEP 16: Verify User Still in Chat")
        logger.info("=" * 60)

        response = await api_client.agent_api.list_agent_chat_participants(chat_id)
        participants = response.data or []
        participant_ids = [p.id for p in participants]
        assert peer_id in participant_ids, "User should still be a participant"
        logger.info(f"Verified: User '{peer_name}' is still in chat")
        logger.info(f"Total participants: {len(participants)}")
        for p in participants:
            logger.info(f"  - {p.name} ({p.type}, role: {p.role})")

        # ============================================================
        # COMPLETE
        # ============================================================
        logger.info("\n" + "=" * 60)
        logger.info("WORKFLOW COMPLETE - All 16 steps passed!")
        logger.info("=" * 60)
        logger.info(f"Test chat ID: {chat_id}")
        logger.info(f"User '{peer_name}' remains in chat as expected")


@requires_api
class TestMessageFailureLifecycle:
    """Test the message failure lifecycle separately."""

    async def test_mark_message_failed(self, api_client):
        """Test marking a message as failed with error message."""
        logger.info("\n" + "=" * 60)
        logger.info("Testing Message Failure Lifecycle")
        logger.info("=" * 60)

        # Create a chat for this test
        response = await api_client.agent_api.create_agent_chat(chat=ChatRoomRequest())
        chat_id = response.data.id
        logger.info(f"Created test chat: {chat_id}")

        # Get peers and find a User peer to add to the chat
        response = await api_client.agent_api.list_agent_peers()
        peers = response.data or []
        assert len(peers) > 0, "Need at least one peer"

        # Find a User peer (human) for this test
        user_peer = next((p for p in peers if p.type == "User"), None)
        assert user_peer is not None, "Need at least one User peer"

        peer_id = user_peer.id
        peer_name = user_peer.name

        await api_client.agent_api.add_agent_chat_participant(
            chat_id,
            participant=ParticipantRequest(participant_id=peer_id, role="member"),
        )
        logger.info(f"Added User peer: {peer_name}")

        # Add descriptive message (triggers auto-title)
        await api_client.agent_api.create_agent_chat_message(
            chat_id,
            message=ChatMessageRequest(
                content=f"Message failure lifecycle test: @{peer_name} testing mark_message_failed with error message",
                mentions=[Mention(id=peer_id, name=peer_name)],
            ),
        )

        # Send a message
        response = await api_client.agent_api.create_agent_chat_message(
            chat_id,
            message=ChatMessageRequest(
                content=f"Test message for @{peer_name}",
                mentions=[Mention(id=peer_id, name=peer_name)],
            ),
        )
        message_id = response.data.id
        logger.info(f"Created message: {message_id}")

        # Mark as processing
        await api_client.agent_api.mark_agent_message_processing(chat_id, message_id)
        logger.info("Marked as processing")

        # Mark as failed
        error_message = "SDK integration test simulated failure"
        await api_client.agent_api.mark_agent_message_failed(
            chat_id, message_id, error=error_message
        )
        logger.info(f"Marked as failed with error: {error_message}")

        # Verify User is still in the chat
        response = await api_client.agent_api.list_agent_chat_participants(chat_id)
        participants = response.data or []
        participant_ids = [p.id for p in participants]
        assert peer_id in participant_ids, "User should still be a participant"
        logger.info(f"Verified: User '{peer_name}' is still in chat")

        logger.info("\nFailure lifecycle test complete!")


@requires_api
class TestParticipantOperations:
    """Test participant add/remove operations."""

    async def test_add_and_remove_participant(
        self, api_client, test_chat, test_peer_id
    ):
        """Test adding and removing a participant from a chat."""
        logger.info("\n" + "=" * 60)
        logger.info("Testing: Add -> Verify -> Remove -> Verify Cycle")
        logger.info("=" * 60)

        if not test_peer_id:
            import pytest

            pytest.skip("No peer available for testing")

        # Step 1: Add participant
        logger.info("\nStep 1: Adding participant...")
        await api_client.agent_api.add_agent_chat_participant(
            test_chat,
            participant=ParticipantRequest(participant_id=test_peer_id, role="member"),
        )
        logger.info("Added participant")

        # Step 2: Verify participant is present
        logger.info("\nStep 2: Verifying participant is present...")
        response = await api_client.agent_api.list_agent_chat_participants(test_chat)
        participants = response.data or []
        participant_ids = [p.id for p in participants]
        assert test_peer_id in participant_ids, "Peer should be in participant list"
        logger.info(f"Participant found in list (total: {len(participants)})")

        # Step 3: Remove participant
        logger.info("\nStep 3: Removing participant...")
        await api_client.agent_api.remove_agent_chat_participant(
            test_chat, test_peer_id
        )
        logger.info("Removed participant")

        # Step 4: Verify participant is removed
        logger.info("\nStep 4: Verifying participant is removed...")
        response = await api_client.agent_api.list_agent_chat_participants(test_chat)
        participants = response.data or []
        participant_ids = [p.id for p in participants]
        assert test_peer_id not in participant_ids, (
            "Peer should not be in participant list"
        )
        logger.info(f"Participant removed (remaining: {len(participants)})")

        logger.info("\nFull add/remove cycle completed successfully!")
