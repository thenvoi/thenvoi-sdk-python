"""Multi-agent integration tests.

Tests the interaction between multiple agents in the same chat room.
Validates that agents only see messages where they are @mentioned.
Validates that agents do NOT see other agents' events (thoughts, tool_calls, tool_results).

Setup:
- Agent 1 (AR-2 Darter): Primary test agent
- Agent 2 (MQ 12): Secondary test agent
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
        print(f"Agent 1 added Agent 2 to chat")

        # Agent 1 adds User to the chat
        await api_client.agent_api.add_agent_chat_participant(
            chat_id,
            participant=ParticipantRequest(participant_id=user_id, role="member"),
        )
        print(f"Agent 1 added User to chat")

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
        print(f"  Mentions: User only (NOT Agent 2)")

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
            print(f"  Mentions: User only (NOT Agent 2)")

            # Wait to see if Agent 2 receives the message
            await asyncio.sleep(1.5)

        print(f"\nMessages received by Agent 2's raw WebSocket: {len(agent2_received_messages)}")

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
        print(f"Added Agent 2 to chat")

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
        print(f"Added Agent 2 to chat")

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

        agent1_event_ids = {thought_event_id, tool_call_event_id, tool_result_event_id}

        await asyncio.sleep(0.5)

        # Agent 1 queries /context - should see its own events
        print("\n--- Agent 1 querying /context ---")
        response = await api_client.agent_api.get_agent_chat_context(chat_id)
        agent1_context = response.data or []
        agent1_context_ids = {
            item.id for item in agent1_context if hasattr(item, "id")
        }
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
        agent2_context_ids = {
            item.id for item in agent2_context if hasattr(item, "id")
        }
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
        print(f"Both agents are in the chat")

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
        agent1_context_ids = {
            item.id for item in agent1_context if hasattr(item, "id")
        }

        agent1_sees_own_thought = agent1_thought_id in agent1_context_ids
        agent1_sees_agent2_thought = agent2_thought_id in agent1_context_ids
        print(f"Agent 1 sees own thought: {agent1_sees_own_thought}")
        print(f"Agent 1 sees Agent 2's thought: {agent1_sees_agent2_thought}")

        # Agent 2 queries /context
        print("\n--- Agent 2 querying /context ---")
        response = await api_client_2.agent_api.get_agent_chat_context(chat_id)
        agent2_context = response.data or []
        agent2_context_ids = {
            item.id for item in agent2_context if hasattr(item, "id")
        }

        agent2_sees_own_thought = agent2_thought_id in agent2_context_ids
        agent2_sees_agent1_thought = agent1_thought_id in agent2_context_ids
        print(f"Agent 2 sees own thought: {agent2_sees_own_thought}")
        print(f"Agent 2 sees Agent 1's thought: {agent2_sees_agent1_thought}")

        # Assertions
        assert agent1_sees_own_thought, "Agent 1 should see its own thought"
        assert not agent1_sees_agent2_thought, "Agent 1 should NOT see Agent 2's thought"
        assert agent2_sees_own_thought, "Agent 2 should see its own thought"
        assert not agent2_sees_agent1_thought, "Agent 2 should NOT see Agent 1's thought"

        print("\n" + "=" * 60)
        print("SUCCESS: Each agent sees only their own events")
        print("=" * 60)
