"""Integration coverage for multi-agent message/event visibility boundaries."""

from __future__ import annotations

import logging

from thenvoi_rest import ChatRoomRequest, ChatMessageRequest
from thenvoi_rest.types import (
    ChatMessageRequestMentionsItem as Mention,
    ParticipantRequest,
)

from tests.support.integration.markers import requires_multi_agent

logger = logging.getLogger(__name__)


@requires_multi_agent
class TestMultiAgentChatRoom:
    async def test_member_agent_can_list_peers_and_add_participants(
        self, api_client, api_client_2, integration_settings
    ):
        """Test that a member agent can list peers and add participants to a chat.

        This tests the SDK's add_participant flow where:
        1. Agent 1 creates a chat (becomes owner)
        2. Agent 1 adds Agent 2 as a member
        3. Agent 2 (as member) lists peers
        4. Agent 2 (as member) tries to add a User to the chat
        5. Agent 2 (as member) tries to add another Agent to the chat

        This validates whether members have permission to add participants.
        """
        logger.debug("\n" + "=" * 60)
        logger.debug("Testing: Member agent adding participants")
        logger.debug("=" * 60)

        # Get Agent 1 info (will be owner)
        response = await api_client.agent_api_identity.get_agent_me()
        agent1_id = response.data.id
        agent1_name = response.data.name
        logger.debug("Agent 1 (Owner): %s (ID: %s)", agent1_name, agent1_id)

        # Get Agent 2 info (will be member)
        response = await api_client_2.agent_api_identity.get_agent_me()
        agent2_id = response.data.id
        agent2_name = response.data.name
        logger.debug("Agent 2 (Member): %s (ID: %s)", agent2_name, agent2_id)

        # Find a User peer (the owner of agents)
        response = await api_client.agent_api_peers.list_agent_peers()
        user_peer = next((p for p in response.data if p.type == "User"), None)
        assert user_peer is not None, "Need a User peer for this test"
        user_id = user_peer.id
        user_name = user_peer.name
        logger.debug("User peer: %s (ID: %s)", user_name, user_id)

        # Find another Agent peer (not Agent 1 or Agent 2)
        other_agent = next(
            (
                p
                for p in response.data
                if p.type == "Agent" and p.id not in [agent1_id, agent2_id]
            ),
            None,
        )
        if other_agent:
            other_agent_id = other_agent.id
            other_agent_name = other_agent.name
            logger.debug(
                "Other Agent peer: %s (ID: %s)", other_agent_name, other_agent_id
            )
        else:
            other_agent_id = None
            other_agent_name = None
            logger.debug("No other Agent peer available for testing")

        # Step 1: Agent 1 creates a chat (becomes OWNER)
        logger.debug("\n--- Step 1: Agent 1 creates chat ---")
        response = await api_client.agent_api_chats.create_agent_chat(
            chat=ChatRoomRequest()
        )
        chat_id = response.data.id
        logger.debug("Agent 1 created chat: %s", chat_id)

        # Step 2: Agent 1 adds Agent 2 as MEMBER
        logger.debug("\n--- Step 2: Agent 1 adds Agent 2 as member ---")
        await api_client.agent_api_participants.add_agent_chat_participant(
            chat_id,
            participant=ParticipantRequest(participant_id=agent2_id, role="member"),
        )
        logger.debug("Agent 1 added Agent 2 as member")

        # Add descriptive message (triggers auto-title)
        await api_client.agent_api_messages.create_agent_chat_message(
            chat_id,
            message=ChatMessageRequest(
                content=f"Multi-agent permission test: @{agent2_name} testing member agent list peers and add participants",
                mentions=[Mention(id=agent2_id, name=agent2_name)],
            ),
        )

        # Verify participants
        response = await api_client.agent_api_participants.list_agent_chat_participants(
            chat_id
        )
        participants = response.data or []
        logger.debug("Current participants (%s):", len(participants))
        for p in participants:
            logger.debug("  - %s (%s, role: %s)", p.name, p.type, p.role)

        # Step 3: Agent 2 lists peers (should work)
        logger.debug("\n--- Step 3: Agent 2 lists peers ---")
        response = await api_client_2.agent_api_peers.list_agent_peers(
            not_in_chat=chat_id
        )
        peers = response.data or []
        logger.debug("Agent 2 sees %s peers not in chat:", len(peers))
        for p in peers[:5]:  # Show first 5
            logger.debug("  - %s (%s)", p.name, p.type)
        if len(peers) > 5:
            logger.debug("  ... and %s more", len(peers) - 5)

        # Step 4: Agent 2 (member) tries to add User
        logger.debug("\n--- Step 4: Agent 2 (member) tries to add User ---")
        add_user_success = False
        try:
            await api_client_2.agent_api_participants.add_agent_chat_participant(
                chat_id,
                participant=ParticipantRequest(participant_id=user_id, role="member"),
            )
            add_user_success = True
            logger.debug("SUCCESS: Agent 2 added User '%s' to chat", user_name)
        except Exception as e:
            logger.debug("FAILED: Agent 2 could not add User: %s", type(e).__name__)
            if "403" in str(e) or "forbidden" in str(e).lower():
                logger.debug("  -> 403 Forbidden: Members cannot add participants")

        # Step 5: Agent 2 (member) tries to add another Agent
        if other_agent_id:
            logger.debug("\n--- Step 5: Agent 2 (member) tries to add another Agent ---")
            add_agent_success = False
            try:
                await api_client_2.agent_api_participants.add_agent_chat_participant(
                    chat_id,
                    participant=ParticipantRequest(
                        participant_id=other_agent_id, role="member"
                    ),
                )
                add_agent_success = True
                logger.debug(
                    "SUCCESS: Agent 2 added Agent '%s' to chat", other_agent_name
                )
            except Exception as e:
                logger.debug("FAILED: Agent 2 could not add Agent: %s", type(e).__name__)
                if "403" in str(e) or "forbidden" in str(e).lower():
                    logger.debug("  -> 403 Forbidden: Members cannot add participants")
        else:
            add_agent_success = None

        # Final participants check
        logger.debug("\n--- Final participants ---")
        response = await api_client.agent_api_participants.list_agent_chat_participants(
            chat_id
        )
        final_participants = response.data or []
        logger.debug("Final participants (%s):", len(final_participants))
        for p in final_participants:
            logger.debug("  - %s (%s, role: %s)", p.name, p.type, p.role)

        # Summary
        logger.debug("\n" + "=" * 60)
        logger.debug("SUMMARY:")
        logger.debug(
            "  - Agent 2 (member) add User: %s",
            "SUCCESS" if add_user_success else "FAILED (403)",
        )
        if other_agent_id:
            logger.debug(
                "  - Agent 2 (member) add Agent: %s",
                "SUCCESS" if add_agent_success else "FAILED (403)",
            )
        logger.debug("=" * 60)

        # Note: We're not asserting success/failure here because we want to
        # document the actual behavior. If members SHOULD be able to add
        # participants, the platform needs to be updated.
        #
        # Current expected behavior: Members CANNOT add participants (403)
        # If this test shows SUCCESS, then the platform allows members to add.

    async def test_member_agent_promoted_to_admin_can_add_participants(
        self, api_client, api_client_2, integration_settings
    ):
        """Test that a member promoted to admin CAN add participants.

        This validates the fix: if we want agents to add participants,
        they need to be admins, not just members.

        1. Agent 1 creates chat (owner)
        2. Agent 1 adds Agent 2 as ADMIN (not member)
        3. Agent 2 (as admin) should be able to add other participants
        """
        logger.debug("\n" + "=" * 60)
        logger.debug("Testing: Admin agent adding participants")
        logger.debug("=" * 60)

        # Get Agent 1 info
        response = await api_client.agent_api_identity.get_agent_me()
        agent1_id = response.data.id
        agent1_name = response.data.name
        logger.debug("Agent 1 (Owner): %s (ID: %s)", agent1_name, agent1_id)

        # Get Agent 2 info
        response = await api_client_2.agent_api_identity.get_agent_me()
        agent2_id = response.data.id
        agent2_name = response.data.name
        logger.debug("Agent 2 (will be Admin): %s (ID: %s)", agent2_name, agent2_id)

        # Find a User peer
        response = await api_client.agent_api_peers.list_agent_peers()
        user_peer = next((p for p in response.data if p.type == "User"), None)
        assert user_peer is not None, "Need a User peer for this test"
        user_id = user_peer.id
        user_name = user_peer.name
        logger.debug("User peer: %s (ID: %s)", user_name, user_id)

        # Step 1: Agent 1 creates chat
        logger.debug("\n--- Step 1: Agent 1 creates chat ---")
        response = await api_client.agent_api_chats.create_agent_chat(
            chat=ChatRoomRequest()
        )
        chat_id = response.data.id
        logger.debug("Agent 1 created chat: %s", chat_id)

        # Step 2: Agent 1 adds Agent 2 as ADMIN
        logger.debug("\n--- Step 2: Agent 1 adds Agent 2 as ADMIN ---")
        await api_client.agent_api_participants.add_agent_chat_participant(
            chat_id,
            participant=ParticipantRequest(participant_id=agent2_id, role="admin"),
        )
        logger.debug("Agent 1 added Agent 2 as ADMIN")

        # Add descriptive message (triggers auto-title)
        await api_client.agent_api_messages.create_agent_chat_message(
            chat_id,
            message=ChatMessageRequest(
                content=f"Multi-agent permission test: @{agent2_name} testing admin agent adding participants",
                mentions=[Mention(id=agent2_id, name=agent2_name)],
            ),
        )

        # Verify Agent 2's role
        response = await api_client.agent_api_participants.list_agent_chat_participants(
            chat_id
        )
        participants = response.data or []
        agent2_participant = next((p for p in participants if p.id == agent2_id), None)
        assert agent2_participant is not None, "Agent 2 should be in chat"
        logger.debug("Agent 2's role: %s", agent2_participant.role)
        assert agent2_participant.role == "admin", "Agent 2 should be admin"

        # Step 3: Agent 2 (admin) adds User
        logger.debug("\n--- Step 3: Agent 2 (admin) adds User ---")
        add_success = False
        try:
            await api_client_2.agent_api_participants.add_agent_chat_participant(
                chat_id,
                participant=ParticipantRequest(participant_id=user_id, role="member"),
            )
            add_success = True
            logger.debug("SUCCESS: Agent 2 (admin) added User '%s' to chat", user_name)
        except Exception as e:
            logger.debug("FAILED: Agent 2 (admin) could not add User: %s", e)

        # Verify User was added
        response = await api_client.agent_api_participants.list_agent_chat_participants(
            chat_id
        )
        final_participants = response.data or []
        user_in_chat = any(p.id == user_id for p in final_participants)

        logger.debug("\nFinal participants (%s):", len(final_participants))
        for p in final_participants:
            logger.debug("  - %s (%s, role: %s)", p.name, p.type, p.role)

        logger.debug("\n" + "=" * 60)
        if add_success and user_in_chat:
            logger.debug("SUCCESS: Admin agent CAN add participants")
        else:
            logger.debug("UNEXPECTED: Admin agent could not add participants")
        logger.debug("=" * 60)

        # Assert that admins CAN add participants
        assert add_success, "Admin should be able to add participants"
        assert user_in_chat, "User should now be in the chat"
