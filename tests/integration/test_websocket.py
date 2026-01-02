"""WebSocket integration tests.

Tests real-time WebSocket notifications against the local platform.

Run with: uv run pytest tests/integration/test_websocket.py -v -s
"""

import asyncio

from thenvoi_rest import ChatRoomRequest, ChatMessageRequest
from thenvoi_rest.types import (
    ChatMessageRequestMentionsItem as Mention,
    ParticipantRequest,
)

from thenvoi.client.streaming import (
    WebSocketClient,
    MessageCreatedPayload,
    RoomAddedPayload,
)
from tests.integration.conftest import requires_api, requires_multi_agent


class TestWebSocketNotifications:
    """Test WebSocket notifications for real-time events."""

    @requires_multi_agent
    async def test_receives_message_created_event(
        self, api_client, api_client_2, integration_settings
    ):
        """Test that WebSocket receives message_created when @mentioned by another agent.

        Multi-agent flow:
        1. Agent 1 creates chat and subscribes to WebSocket
        2. Agent 2 is added to the chat
        3. Agent 2 sends message @mentioning Agent 1
        4. Agent 1 should receive the message via WebSocket
        """
        print("\n" + "=" * 60)
        print("Testing: WebSocket message_created notification (multi-agent)")
        print("=" * 60)

        # Get Agent 1 info (receiver, will subscribe to WebSocket)
        response = await api_client.agent_api.get_agent_me()
        agent1_id = response.data.id
        agent1_name = response.data.name
        print(f"Agent 1 (receiver): {agent1_name} (ID: {agent1_id})")

        # Get Agent 2 info (sender)
        response = await api_client_2.agent_api.get_agent_me()
        agent2_id = response.data.id
        agent2_name = response.data.name
        print(f"Agent 2 (sender): {agent2_name} (ID: {agent2_id})")

        # Agent 1 creates a chat
        response = await api_client.agent_api.create_agent_chat(
            chat=ChatRoomRequest(title="WebSocket Multi-Agent Test")
        )
        chat_id = response.data.id
        print(f"Created chat: {chat_id}")

        # Add Agent 2 to the chat
        await api_client.agent_api.add_agent_chat_participant(
            chat_id,
            participant=ParticipantRequest(participant_id=agent2_id, role="member"),
        )
        print(f"Added Agent 2 to chat: {agent2_name}")

        # Track received messages
        received_messages: list[MessageCreatedPayload] = []
        message_received = asyncio.Event()

        async def on_message_created(payload: MessageCreatedPayload):
            print(f"  [WS] Agent 1 received message_created: {payload.id}")
            received_messages.append(payload)
            message_received.set()

        # Agent 1 connects WebSocket and subscribes to chat room
        ws = WebSocketClient(
            ws_url=integration_settings.thenvoi_ws_url,
            api_key=integration_settings.thenvoi_api_key,
            agent_id=agent1_id,
        )

        async with ws:
            await ws.join_chat_room_channel(chat_id, on_message_created)
            print(f"Agent 1 subscribed to chat_room:{chat_id}")

            # Small delay to ensure subscription is active
            await asyncio.sleep(0.2)

            # Agent 2 sends a message @mentioning Agent 1
            message_content = f"Hello @{agent1_name}, WebSocket multi-agent test!"
            response = await api_client_2.agent_api.create_agent_chat_message(
                chat_id,
                message=ChatMessageRequest(
                    content=message_content,
                    mentions=[Mention(id=agent1_id, name=agent1_name)],
                ),
            )
            sent_message_id = response.data.id
            print(f"Agent 2 sent message via REST: {sent_message_id}")

            # Wait for Agent 1's WebSocket to receive the message
            try:
                await asyncio.wait_for(message_received.wait(), timeout=5.0)
                print("Agent 1's WebSocket received the message!")
            except asyncio.TimeoutError:
                print("Timeout waiting for WebSocket message")

        # Verify Agent 1 received the message
        assert len(received_messages) > 0, (
            "Agent 1 should have received at least one message"
        )

        # Find the message Agent 2 sent
        our_message = next(
            (m for m in received_messages if m.id == sent_message_id), None
        )
        assert our_message is not None, (
            f"Agent 1 should have received message {sent_message_id}"
        )
        assert our_message.content == message_content
        assert our_message.chat_room_id == chat_id
        print(f"Verified message content: '{our_message.content[:50]}...'")

        print("\nWebSocket message_created test passed!")

    @requires_api
    async def test_receives_room_added_event(self, api_client, integration_settings):
        """Test that WebSocket receives room_added when agent is added to a new chat."""
        print("\n" + "=" * 60)
        print("Testing: WebSocket room_added notification")
        print("=" * 60)

        # Get agent info
        response = await api_client.agent_api.get_agent_me()
        agent_id = response.data.id
        agent_name = response.data.name
        print(f"Agent: {agent_name} (ID: {agent_id})")

        # Track received room events
        received_rooms: list[RoomAddedPayload] = []
        room_added = asyncio.Event()

        async def on_room_added(payload: RoomAddedPayload):
            print(f"  [WS] Received room_added: {payload.id} - {payload.title}")
            received_rooms.append(payload)
            room_added.set()

        async def on_room_removed(payload):
            print(f"  [WS] Received room_removed: {payload.id}")

        # Connect WebSocket and subscribe to agent rooms channel
        ws = WebSocketClient(
            ws_url=integration_settings.thenvoi_ws_url,
            api_key=integration_settings.thenvoi_api_key,
            agent_id=agent_id,
        )

        async with ws:
            await ws.join_agent_rooms_channel(agent_id, on_room_added, on_room_removed)
            print(f"Subscribed to agent_rooms:{agent_id}")

            # Small delay to ensure subscription is active
            await asyncio.sleep(0.2)

            # Create a new chat via REST API (agent is automatically owner)
            response = await api_client.agent_api.create_agent_chat(
                chat=ChatRoomRequest(title="WebSocket Room Test")
            )
            created_chat_id = response.data.id
            print(f"Created chat via REST: {created_chat_id}")

            # Wait for WebSocket to receive the room_added event
            try:
                await asyncio.wait_for(room_added.wait(), timeout=5.0)
                print("WebSocket received the room_added event!")
            except asyncio.TimeoutError:
                print("Timeout waiting for room_added event")

        # Verify we received the room event
        assert len(received_rooms) > 0, (
            "Should have received at least one room_added event"
        )

        # Find our room in received
        our_room = next((r for r in received_rooms if r.id == created_chat_id), None)
        assert our_room is not None, (
            f"Should have received room_added for {created_chat_id}"
        )
        assert our_room.title == "WebSocket Room Test"
        print(f"Verified room title: '{our_room.title}'")

        print("\nWebSocket room_added test passed!")
