"""Multi-agent integration tests.

Tests the interaction between multiple agents in the same chat room.
Validates that agents only see messages where they are @mentioned.
Validates that agents do NOT see other agents' events (thoughts, tool_calls, tool_results).

Setup:
- Agent 1: Primary test agent
- Agent 2: Secondary test agent
- User: Owner of both agents

Test scenarios:
1. Agent 1 sends message to User (no @mention of Agent 2) -> Agent 2 should NOT see it
2. Agent 1 creates events (thought, tool_call, tool_result) -> Agent 2 should NOT see them
3. WebSocket should NOT broadcast messages to agents that are not @mentioned

Run with: uv run pytest tests/integration/test_multi_agent.py -v -s
"""

import asyncio

from thenvoi_rest import ChatRoomRequest, ChatMessageRequest, ChatEventRequest
from thenvoi_rest.types import (
    ChatMessageRequestMentionsItem as Mention,
    ParticipantRequest,
)

from thenvoi.client.streaming import WebSocketClient, MessageCreatedPayload
from tests.integration.conftest import requires_multi_agent


@requires_multi_agent
class TestMultiAgentChatRoom:
    """Test multi-agent chat room interactions."""

    async def test_agent_does_not_see_messages_without_mention_via_context(
        self, api_client, api_client_2, integration_settings
    ):
        """Test that Agent 2 does NOT see messages in /context where it's not @mentioned.

        Scenario:
        1. Agent 1 creates chat and adds Agent 2 + User
        2. Agent 1 sends message mentioning only the User (not Agent 2)
        3. Agent 2 queries /context - should NOT see Agent 1's message
        """
        print("\n" + "=" * 60)
        print("Testing: Agent isolation via /context API")
        print("=" * 60)

        # Get Agent 1 info
        response = await api_client.agent_api.get_agent_me()
        agent1_id = response.data.id
        agent1_name = response.data.name
        print(f"Agent 1: {agent1_name} (ID: {agent1_id})")

        # Get Agent 2 info
        response = await api_client_2.agent_api.get_agent_me()
        agent2_id = response.data.id
        agent2_name = response.data.name
        print(f"Agent 2: {agent2_name} (ID: {agent2_id})")

        # Find a User peer (the owner)
        response = await api_client.agent_api.list_agent_peers()
        user_peer = next((p for p in response.data if p.type == "User"), None)
        assert user_peer is not None, "Need a User peer for this test"
        user_id = user_peer.id
        user_name = user_peer.name
        print(f"User (owner): {user_name} (ID: {user_id})")

        # Agent 1 creates a chat
        response = await api_client.agent_api.create_agent_chat(
            chat=ChatRoomRequest(title="Multi-Agent Test Chat")
        )
        chat_id = response.data.id
        print(f"\nAgent 1 created chat: {chat_id}")

        # Agent 1 adds Agent 2 to the chat
        await api_client.agent_api.add_agent_chat_participant(
            chat_id,
            participant=ParticipantRequest(participant_id=agent2_id, role="member"),
        )
        print("Agent 1 added Agent 2 to chat")

        # Agent 1 adds User to the chat
        await api_client.agent_api.add_agent_chat_participant(
            chat_id,
            participant=ParticipantRequest(participant_id=user_id, role="member"),
        )
        print("Agent 1 added User to chat")

        # Verify all participants are in the chat
        response = await api_client.agent_api.list_agent_chat_participants(chat_id)
        participants = response.data or []
        print(f"\nChat participants ({len(participants)}):")
        for p in participants:
            print(f"  - {p.name} ({p.type}, role: {p.role})")

        assert any(p.id == agent1_id for p in participants), "Agent 1 should be in chat"
        assert any(p.id == agent2_id for p in participants), "Agent 2 should be in chat"
        assert any(p.id == user_id for p in participants), "User should be in chat"

        # Agent 1 sends a message mentioning ONLY the User (not Agent 2)
        message_content = f"Hello @{user_name}, this message is just for you!"
        response = await api_client.agent_api.create_agent_chat_message(
            chat_id,
            message=ChatMessageRequest(
                content=message_content,
                mentions=[Mention(id=user_id, name=user_name)],
            ),
        )
        agent1_message_id = response.data.id
        print(f"\nAgent 1 sent message (ID: {agent1_message_id})")
        print(f"  Content: '{message_content}'")
        print("  Mentions: User only (NOT Agent 2)")

        # Give the platform time to process
        await asyncio.sleep(0.5)

        # Agent 2 queries /context for the chat
        print(f"\nAgent 2 querying /context for chat {chat_id}...")
        response = await api_client_2.agent_api.get_agent_chat_context(chat_id)
        context = response.data or []
        print(f"Agent 2 received {len(context)} items in context")

        # Check if Agent 1's message is in Agent 2's context
        agent1_message_in_context = any(
            hasattr(item, "id") and item.id == agent1_message_id for item in context
        )

        print(f"\nAgent 1's message visible to Agent 2: {agent1_message_in_context}")

        # ASSERTION: Agent 2 should NOT see Agent 1's message (no @mention of Agent 2)
        assert not agent1_message_in_context, (
            f"Agent 2 should NOT see Agent 1's message (ID: {agent1_message_id}) "
            f"because Agent 2 was not @mentioned"
        )

        print("\n" + "=" * 60)
        print("SUCCESS: Agent 2 correctly does NOT see Agent 1's message")
        print("=" * 60)

    async def test_agent_does_not_receive_websocket_notification_without_mention(
        self, api_client, api_client_2, integration_settings
    ):
        """Test that Agent 2's WebSocket does NOT receive messages where it's not @mentioned.

        This tests BACKEND filtering - the platform should NOT broadcast messages
        to agents that are not @mentioned. This is NOT SDK-level filtering.

        Scenario:
        1. Agent 2 connects to WebSocket and subscribes to chat room
        2. Agent 1 sends message mentioning only the User (not Agent 2)
        3. Agent 2's raw WebSocket should NOT receive the message_created event
        """
        print("\n" + "=" * 60)
        print("Testing: Backend WebSocket filtering (agent isolation)")
        print("=" * 60)

        # Get Agent 1 info
        response = await api_client.agent_api.get_agent_me()
        agent1_id = response.data.id
        agent1_name = response.data.name
        print(f"Agent 1: {agent1_name} (ID: {agent1_id})")

        # Get Agent 2 info
        response = await api_client_2.agent_api.get_agent_me()
        agent2_id = response.data.id
        agent2_name = response.data.name
        print(f"Agent 2: {agent2_name} (ID: {agent2_id})")

        # Find a User peer
        response = await api_client.agent_api.list_agent_peers()
        user_peer = next((p for p in response.data if p.type == "User"), None)
        assert user_peer is not None, "Need a User peer for this test"
        user_id = user_peer.id
        user_name = user_peer.name
        print(f"User: {user_name} (ID: {user_id})")

        # Agent 1 creates a chat
        response = await api_client.agent_api.create_agent_chat(
            chat=ChatRoomRequest(title="WebSocket Multi-Agent Test")
        )
        chat_id = response.data.id
        print(f"\nAgent 1 created chat: {chat_id}")

        # Add Agent 2 and User to the chat
        await api_client.agent_api.add_agent_chat_participant(
            chat_id,
            participant=ParticipantRequest(participant_id=agent2_id, role="member"),
        )
        await api_client.agent_api.add_agent_chat_participant(
            chat_id,
            participant=ParticipantRequest(participant_id=user_id, role="member"),
        )
        print("Added Agent 2 and User to chat")

        # Track messages received by Agent 2's raw WebSocket
        agent2_received_messages: list[MessageCreatedPayload] = []

        async def agent2_message_handler(payload: MessageCreatedPayload):
            print(f"  [Agent 2 WS] Received: {payload.id} from {payload.sender_id}")
            agent2_received_messages.append(payload)

        # Agent 2 connects to WebSocket and subscribes to the chat room
        agent2_ws = WebSocketClient(
            ws_url=integration_settings.thenvoi_ws_url,
            api_key=integration_settings.thenvoi_api_key_2,
            agent_id=agent2_id,
        )

        async with agent2_ws:
            await agent2_ws.join_chat_room_channel(chat_id, agent2_message_handler)
            print(f"Agent 2 subscribed to chat_room:{chat_id}")

            await asyncio.sleep(0.3)

            # Agent 1 sends message mentioning ONLY the User (not Agent 2)
            message_content = f"@{user_name}, this is a private message for you!"
            response = await api_client.agent_api.create_agent_chat_message(
                chat_id,
                message=ChatMessageRequest(
                    content=message_content,
                    mentions=[Mention(id=user_id, name=user_name)],
                ),
            )
            agent1_message_id = response.data.id
            print(f"\nAgent 1 sent message (ID: {agent1_message_id})")
            print("  Mentions: User only (NOT Agent 2)")

            # Wait to see if Agent 2 receives the message
            await asyncio.sleep(1.5)

        print(
            f"\nMessages received by Agent 2's raw WebSocket: {len(agent2_received_messages)}"
        )

        # Check if Agent 1's specific message was received
        agent1_message_received = any(
            m.id == agent1_message_id for m in agent2_received_messages
        )

        # ASSERTION: Agent 2's raw WebSocket should NOT receive Agent 1's message
        # Backend should filter WebSocket broadcasts by @mention
        assert not agent1_message_received, (
            f"BACKEND BUG: Agent 2's WebSocket received Agent 1's message (ID: {agent1_message_id}) "
            f"but Agent 2 was NOT @mentioned. Platform should filter WebSocket broadcasts."
        )

        print("\n" + "=" * 60)
        print("SUCCESS: Backend correctly filtered WebSocket broadcast")
        print("=" * 60)

    async def test_agent_sees_messages_with_own_mention(
        self, api_client, api_client_2, integration_settings
    ):
        """Test that Agent 2 DOES see messages where it IS @mentioned.

        Scenario:
        1. Agent 1 sends message mentioning Agent 2
        2. Agent 2 should see the message via /context
        """
        print("\n" + "=" * 60)
        print("Testing: Agent visibility when @mentioned")
        print("=" * 60)

        # Get both agents' info
        response = await api_client.agent_api.get_agent_me()
        agent1_id = response.data.id
        agent1_name = response.data.name
        print(f"Agent 1: {agent1_name} (ID: {agent1_id})")

        response = await api_client_2.agent_api.get_agent_me()
        agent2_id = response.data.id
        agent2_name = response.data.name
        print(f"Agent 2: {agent2_name} (ID: {agent2_id})")

        # Agent 1 creates chat and adds Agent 2
        response = await api_client.agent_api.create_agent_chat(
            chat=ChatRoomRequest(title="Mention Test Chat")
        )
        chat_id = response.data.id
        print(f"\nAgent 1 created chat: {chat_id}")

        await api_client.agent_api.add_agent_chat_participant(
            chat_id,
            participant=ParticipantRequest(participant_id=agent2_id, role="member"),
        )
        print("Added Agent 2 to chat")

        # Agent 1 sends message mentioning Agent 2
        message_content = f"Hey @{agent2_name}, can you help with this?"
        response = await api_client.agent_api.create_agent_chat_message(
            chat_id,
            message=ChatMessageRequest(
                content=message_content,
                mentions=[Mention(id=agent2_id, name=agent2_name)],
            ),
        )
        agent1_message_id = response.data.id
        print(f"\nAgent 1 sent message mentioning Agent 2 (ID: {agent1_message_id})")

        await asyncio.sleep(0.5)

        # Agent 2 queries /context
        response = await api_client_2.agent_api.get_agent_chat_context(chat_id)
        context = response.data or []
        print(f"Agent 2 received {len(context)} items in context")

        # Check if Agent 1's message is visible
        agent1_message_visible = any(
            hasattr(item, "id") and item.id == agent1_message_id for item in context
        )

        print(f"Agent 1's message visible to Agent 2: {agent1_message_visible}")

        # ASSERTION: Agent 2 SHOULD see the message (it was @mentioned)
        assert agent1_message_visible, (
            f"Agent 2 SHOULD see Agent 1's message (ID: {agent1_message_id}) "
            f"because Agent 2 WAS @mentioned"
        )

        print("\n" + "=" * 60)
        print("SUCCESS: Agent 2 correctly sees message where it's @mentioned")
        print("=" * 60)

    async def test_agent_does_not_see_other_agent_events_via_context(
        self, api_client, api_client_2, integration_settings
    ):
        """Test that Agent 2 does NOT see Agent 1's events (thoughts, tool_calls, tool_results).

        Events are private to the agent that created them. Other agents in the same
        chat should NOT see another agent's reasoning/tool execution.

        Scenario:
        1. Agent 1 creates a chat with Agent 2
        2. Agent 1 creates thought, tool_call, and tool_result events
        3. Agent 1 queries /context - should see its own events
        4. Agent 2 queries /context - should NOT see Agent 1's events
        """
        print("\n" + "=" * 60)
        print("Testing: Event isolation between agents")
        print("=" * 60)

        # Get both agents' info
        response = await api_client.agent_api.get_agent_me()
        agent1_id = response.data.id
        agent1_name = response.data.name
        print(f"Agent 1: {agent1_name} (ID: {agent1_id})")

        response = await api_client_2.agent_api.get_agent_me()
        agent2_id = response.data.id
        agent2_name = response.data.name
        print(f"Agent 2: {agent2_name} (ID: {agent2_id})")

        # Agent 1 creates a chat
        response = await api_client.agent_api.create_agent_chat(
            chat=ChatRoomRequest(title="Event Isolation Test")
        )
        chat_id = response.data.id
        print(f"\nAgent 1 created chat: {chat_id}")

        # Add Agent 2 to the chat
        await api_client.agent_api.add_agent_chat_participant(
            chat_id,
            participant=ParticipantRequest(participant_id=agent2_id, role="member"),
        )
        print("Added Agent 2 to chat")

        # Agent 1 creates events: thought, tool_call, tool_result
        print("\n--- Agent 1 creating events ---")

        # Thought event
        response = await api_client.agent_api.create_agent_chat_event(
            chat_id,
            event=ChatEventRequest(
                content="Analyzing the user's request...",
                message_type="thought",
            ),
        )
        thought_event_id = response.data.id
        print(f"Agent 1 created thought event (ID: {thought_event_id})")

        # Tool call event
        tool_metadata = {
            "function": {
                "name": "search_database",
                "arguments": {"query": "test query"},
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
        tool_call_event_id = response.data.id
        print(f"Agent 1 created tool_call event (ID: {tool_call_event_id})")

        # Tool result event
        result_metadata = {"result": {"found": 3, "items": ["a", "b", "c"]}}
        response = await api_client.agent_api.create_agent_chat_event(
            chat_id,
            event=ChatEventRequest(
                content="Search returned 3 results",
                message_type="tool_result",
                metadata=result_metadata,
            ),
        )
        tool_result_event_id = response.data.id
        print(f"Agent 1 created tool_result event (ID: {tool_result_event_id})")

        await asyncio.sleep(0.5)

        # Agent 1 queries /context - should see its own events
        print("\n--- Agent 1 querying /context ---")
        response = await api_client.agent_api.get_agent_chat_context(chat_id)
        agent1_context = response.data or []
        agent1_context_ids = {item.id for item in agent1_context if hasattr(item, "id")}
        print(f"Agent 1 received {len(agent1_context)} items in context")

        # Check which events Agent 1 sees
        agent1_sees_thought = thought_event_id in agent1_context_ids
        agent1_sees_tool_call = tool_call_event_id in agent1_context_ids
        agent1_sees_tool_result = tool_result_event_id in agent1_context_ids
        print(f"  Agent 1 sees thought: {agent1_sees_thought}")
        print(f"  Agent 1 sees tool_call: {agent1_sees_tool_call}")
        print(f"  Agent 1 sees tool_result: {agent1_sees_tool_result}")

        # Agent 1 SHOULD see its own events
        assert agent1_sees_thought, "Agent 1 should see its own thought event"
        assert agent1_sees_tool_call, "Agent 1 should see its own tool_call event"
        assert agent1_sees_tool_result, "Agent 1 should see its own tool_result event"

        # Agent 2 queries /context - should NOT see Agent 1's events
        print("\n--- Agent 2 querying /context ---")
        response = await api_client_2.agent_api.get_agent_chat_context(chat_id)
        agent2_context = response.data or []
        agent2_context_ids = {item.id for item in agent2_context if hasattr(item, "id")}
        print(f"Agent 2 received {len(agent2_context)} items in context")

        # Check which of Agent 1's events Agent 2 sees
        agent2_sees_thought = thought_event_id in agent2_context_ids
        agent2_sees_tool_call = tool_call_event_id in agent2_context_ids
        agent2_sees_tool_result = tool_result_event_id in agent2_context_ids
        print(f"  Agent 2 sees Agent 1's thought: {agent2_sees_thought}")
        print(f"  Agent 2 sees Agent 1's tool_call: {agent2_sees_tool_call}")
        print(f"  Agent 2 sees Agent 1's tool_result: {agent2_sees_tool_result}")

        # Agent 2 should NOT see ANY of Agent 1's events
        assert not agent2_sees_thought, (
            f"Agent 2 should NOT see Agent 1's thought event (ID: {thought_event_id})"
        )
        assert not agent2_sees_tool_call, (
            f"Agent 2 should NOT see Agent 1's tool_call event (ID: {tool_call_event_id})"
        )
        assert not agent2_sees_tool_result, (
            f"Agent 2 should NOT see Agent 1's tool_result event (ID: {tool_result_event_id})"
        )

        print("\n" + "=" * 60)
        print("SUCCESS: Agent 2 correctly does NOT see Agent 1's events")
        print("=" * 60)

    async def test_agent_sees_own_events_but_not_others(
        self, api_client, api_client_2, integration_settings
    ):
        """Test that each agent sees only their own events, not other agents' events.

        Scenario:
        1. Both agents are in the same chat
        2. Agent 1 creates events
        3. Agent 2 creates events
        4. Each agent queries /context
        5. Agent 1 should see only Agent 1's events
        6. Agent 2 should see only Agent 2's events
        """
        print("\n" + "=" * 60)
        print("Testing: Each agent sees only their own events")
        print("=" * 60)

        # Get both agents' info
        response = await api_client.agent_api.get_agent_me()
        agent1_id = response.data.id
        agent1_name = response.data.name
        print(f"Agent 1: {agent1_name} (ID: {agent1_id})")

        response = await api_client_2.agent_api.get_agent_me()
        agent2_id = response.data.id
        agent2_name = response.data.name
        print(f"Agent 2: {agent2_name} (ID: {agent2_id})")

        # Agent 1 creates a chat
        response = await api_client.agent_api.create_agent_chat(
            chat=ChatRoomRequest(title="Dual Event Test")
        )
        chat_id = response.data.id
        print(f"\nCreated chat: {chat_id}")

        # Add Agent 2 to the chat
        await api_client.agent_api.add_agent_chat_participant(
            chat_id,
            participant=ParticipantRequest(participant_id=agent2_id, role="member"),
        )
        print("Both agents are in the chat")

        # Agent 1 creates a thought event
        response = await api_client.agent_api.create_agent_chat_event(
            chat_id,
            event=ChatEventRequest(
                content="Agent 1 is thinking...",
                message_type="thought",
            ),
        )
        agent1_thought_id = response.data.id
        print(f"\nAgent 1 created thought (ID: {agent1_thought_id})")

        # Agent 2 creates a thought event
        response = await api_client_2.agent_api.create_agent_chat_event(
            chat_id,
            event=ChatEventRequest(
                content="Agent 2 is thinking...",
                message_type="thought",
            ),
        )
        agent2_thought_id = response.data.id
        print(f"Agent 2 created thought (ID: {agent2_thought_id})")

        await asyncio.sleep(0.5)

        # Agent 1 queries /context
        print("\n--- Agent 1 querying /context ---")
        response = await api_client.agent_api.get_agent_chat_context(chat_id)
        agent1_context = response.data or []
        agent1_context_ids = {item.id for item in agent1_context if hasattr(item, "id")}

        agent1_sees_own_thought = agent1_thought_id in agent1_context_ids
        agent1_sees_agent2_thought = agent2_thought_id in agent1_context_ids
        print(f"Agent 1 sees own thought: {agent1_sees_own_thought}")
        print(f"Agent 1 sees Agent 2's thought: {agent1_sees_agent2_thought}")

        # Agent 2 queries /context
        print("\n--- Agent 2 querying /context ---")
        response = await api_client_2.agent_api.get_agent_chat_context(chat_id)
        agent2_context = response.data or []
        agent2_context_ids = {item.id for item in agent2_context if hasattr(item, "id")}

        agent2_sees_own_thought = agent2_thought_id in agent2_context_ids
        agent2_sees_agent1_thought = agent1_thought_id in agent2_context_ids
        print(f"Agent 2 sees own thought: {agent2_sees_own_thought}")
        print(f"Agent 2 sees Agent 1's thought: {agent2_sees_agent1_thought}")

        # Assertions
        assert agent1_sees_own_thought, "Agent 1 should see its own thought"
        assert not agent1_sees_agent2_thought, (
            "Agent 1 should NOT see Agent 2's thought"
        )
        assert agent2_sees_own_thought, "Agent 2 should see its own thought"
        assert not agent2_sees_agent1_thought, (
            "Agent 2 should NOT see Agent 1's thought"
        )

        print("\n" + "=" * 60)
        print("SUCCESS: Each agent sees only their own events")
        print("=" * 60)

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
        print("\n" + "=" * 60)
        print("Testing: Member agent adding participants")
        print("=" * 60)

        # Get Agent 1 info (will be owner)
        response = await api_client.agent_api.get_agent_me()
        agent1_id = response.data.id
        agent1_name = response.data.name
        print(f"Agent 1 (Owner): {agent1_name} (ID: {agent1_id})")

        # Get Agent 2 info (will be member)
        response = await api_client_2.agent_api.get_agent_me()
        agent2_id = response.data.id
        agent2_name = response.data.name
        print(f"Agent 2 (Member): {agent2_name} (ID: {agent2_id})")

        # Find a User peer (the owner of agents)
        response = await api_client.agent_api.list_agent_peers()
        user_peer = next((p for p in response.data if p.type == "User"), None)
        assert user_peer is not None, "Need a User peer for this test"
        user_id = user_peer.id
        user_name = user_peer.name
        print(f"User peer: {user_name} (ID: {user_id})")

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
            print(f"Other Agent peer: {other_agent_name} (ID: {other_agent_id})")
        else:
            other_agent_id = None
            other_agent_name = None
            print("No other Agent peer available for testing")

        # Step 1: Agent 1 creates a chat (becomes OWNER)
        print("\n--- Step 1: Agent 1 creates chat ---")
        response = await api_client.agent_api.create_agent_chat(
            chat=ChatRoomRequest(title="Member Add Participant Test")
        )
        chat_id = response.data.id
        print(f"Agent 1 created chat: {chat_id}")

        # Step 2: Agent 1 adds Agent 2 as MEMBER
        print("\n--- Step 2: Agent 1 adds Agent 2 as member ---")
        await api_client.agent_api.add_agent_chat_participant(
            chat_id,
            participant=ParticipantRequest(participant_id=agent2_id, role="member"),
        )
        print("Agent 1 added Agent 2 as member")

        # Verify participants
        response = await api_client.agent_api.list_agent_chat_participants(chat_id)
        participants = response.data or []
        print(f"Current participants ({len(participants)}):")
        for p in participants:
            print(f"  - {p.name} ({p.type}, role: {p.role})")

        # Step 3: Agent 2 lists peers (should work)
        print("\n--- Step 3: Agent 2 lists peers ---")
        response = await api_client_2.agent_api.list_agent_peers(not_in_chat=chat_id)
        peers = response.data or []
        print(f"Agent 2 sees {len(peers)} peers not in chat:")
        for p in peers[:5]:  # Show first 5
            print(f"  - {p.name} ({p.type})")
        if len(peers) > 5:
            print(f"  ... and {len(peers) - 5} more")

        # Step 4: Agent 2 (member) tries to add User
        print("\n--- Step 4: Agent 2 (member) tries to add User ---")
        add_user_success = False
        try:
            await api_client_2.agent_api.add_agent_chat_participant(
                chat_id,
                participant=ParticipantRequest(participant_id=user_id, role="member"),
            )
            add_user_success = True
            print(f"SUCCESS: Agent 2 added User '{user_name}' to chat")
        except Exception as e:
            print(f"FAILED: Agent 2 could not add User: {type(e).__name__}")
            if "403" in str(e) or "forbidden" in str(e).lower():
                print("  -> 403 Forbidden: Members cannot add participants")

        # Step 5: Agent 2 (member) tries to add another Agent
        if other_agent_id:
            print("\n--- Step 5: Agent 2 (member) tries to add another Agent ---")
            add_agent_success = False
            try:
                await api_client_2.agent_api.add_agent_chat_participant(
                    chat_id,
                    participant=ParticipantRequest(
                        participant_id=other_agent_id, role="member"
                    ),
                )
                add_agent_success = True
                print(f"SUCCESS: Agent 2 added Agent '{other_agent_name}' to chat")
            except Exception as e:
                print(f"FAILED: Agent 2 could not add Agent: {type(e).__name__}")
                if "403" in str(e) or "forbidden" in str(e).lower():
                    print("  -> 403 Forbidden: Members cannot add participants")
        else:
            add_agent_success = None

        # Final participants check
        print("\n--- Final participants ---")
        response = await api_client.agent_api.list_agent_chat_participants(chat_id)
        final_participants = response.data or []
        print(f"Final participants ({len(final_participants)}):")
        for p in final_participants:
            print(f"  - {p.name} ({p.type}, role: {p.role})")

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY:")
        print(
            f"  - Agent 2 (member) add User: {'SUCCESS' if add_user_success else 'FAILED (403)'}"
        )
        if other_agent_id:
            print(
                f"  - Agent 2 (member) add Agent: {'SUCCESS' if add_agent_success else 'FAILED (403)'}"
            )
        print("=" * 60)

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
        print("\n" + "=" * 60)
        print("Testing: Admin agent adding participants")
        print("=" * 60)

        # Get Agent 1 info
        response = await api_client.agent_api.get_agent_me()
        agent1_id = response.data.id
        agent1_name = response.data.name
        print(f"Agent 1 (Owner): {agent1_name} (ID: {agent1_id})")

        # Get Agent 2 info
        response = await api_client_2.agent_api.get_agent_me()
        agent2_id = response.data.id
        agent2_name = response.data.name
        print(f"Agent 2 (will be Admin): {agent2_name} (ID: {agent2_id})")

        # Find a User peer
        response = await api_client.agent_api.list_agent_peers()
        user_peer = next((p for p in response.data if p.type == "User"), None)
        assert user_peer is not None, "Need a User peer for this test"
        user_id = user_peer.id
        user_name = user_peer.name
        print(f"User peer: {user_name} (ID: {user_id})")

        # Step 1: Agent 1 creates chat
        print("\n--- Step 1: Agent 1 creates chat ---")
        response = await api_client.agent_api.create_agent_chat(
            chat=ChatRoomRequest(title="Admin Add Participant Test")
        )
        chat_id = response.data.id
        print(f"Agent 1 created chat: {chat_id}")

        # Step 2: Agent 1 adds Agent 2 as ADMIN
        print("\n--- Step 2: Agent 1 adds Agent 2 as ADMIN ---")
        await api_client.agent_api.add_agent_chat_participant(
            chat_id,
            participant=ParticipantRequest(participant_id=agent2_id, role="admin"),
        )
        print("Agent 1 added Agent 2 as ADMIN")

        # Verify Agent 2's role
        response = await api_client.agent_api.list_agent_chat_participants(chat_id)
        participants = response.data or []
        agent2_participant = next((p for p in participants if p.id == agent2_id), None)
        assert agent2_participant is not None, "Agent 2 should be in chat"
        print(f"Agent 2's role: {agent2_participant.role}")
        assert agent2_participant.role == "admin", "Agent 2 should be admin"

        # Step 3: Agent 2 (admin) adds User
        print("\n--- Step 3: Agent 2 (admin) adds User ---")
        add_success = False
        try:
            await api_client_2.agent_api.add_agent_chat_participant(
                chat_id,
                participant=ParticipantRequest(participant_id=user_id, role="member"),
            )
            add_success = True
            print(f"SUCCESS: Agent 2 (admin) added User '{user_name}' to chat")
        except Exception as e:
            print(f"FAILED: Agent 2 (admin) could not add User: {e}")

        # Verify User was added
        response = await api_client.agent_api.list_agent_chat_participants(chat_id)
        final_participants = response.data or []
        user_in_chat = any(p.id == user_id for p in final_participants)

        print(f"\nFinal participants ({len(final_participants)}):")
        for p in final_participants:
            print(f"  - {p.name} ({p.type}, role: {p.role})")

        print("\n" + "=" * 60)
        if add_success and user_in_chat:
            print("SUCCESS: Admin agent CAN add participants")
        else:
            print("UNEXPECTED: Admin agent could not add participants")
        print("=" * 60)

        # Assert that admins CAN add participants
        assert add_success, "Admin should be able to add participants"
        assert user_in_chat, "User should now be in the chat"
