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

    @requires_api
    async def test_message_filtering_by_mention(self, api_client, integration_settings):
        """Test that RoomManager correctly filters messages.

        RoomManager filtering rules:
        1. Own messages (sender_id == agent_id) are filtered out
        2. Messages without agent @mention are filtered out

        Since agents can't mention themselves via API, we test:
        - Send message mentioning another user (not agent) -> should be filtered (no @agent mention)
        """
        print("\n" + "=" * 60)
        print("Testing: Message filtering by RoomManager")
        print("=" * 60)

        from thenvoi.agent.core.room_manager import RoomManager

        # Get agent info
        response = await api_client.agent_api.get_agent_me()
        agent_id = response.data.id
        agent_name = response.data.name
        print(f"Agent: {agent_name} (ID: {agent_id})")

        # Create a chat
        response = await api_client.agent_api.create_agent_chat(
            chat=ChatRoomRequest(title="Filter Test Chat")
        )
        chat_id = response.data.id
        print(f"Created chat: {chat_id}")

        # Get a peer
        response = await api_client.agent_api.list_agent_peers()
        user_peer = next((p for p in response.data if p.type == "User"), None)
        assert user_peer is not None

        await api_client.agent_api.add_agent_chat_participant(
            chat_id,
            participant=ParticipantRequest(participant_id=user_peer.id, role="member"),
        )
        print(f"Added participant: {user_peer.name}")

        # Track messages that pass the filter
        filtered_messages: list[MessageCreatedPayload] = []

        async def filtered_handler(payload: MessageCreatedPayload):
            """Only called for messages that pass RoomManager filter."""
            print(
                f"  [Filtered] Message passed: {payload.id} from {payload.sender_type}"
            )
            filtered_messages.append(payload)

        ws = WebSocketClient(
            ws_url=integration_settings.thenvoi_ws_url,
            api_key=integration_settings.thenvoi_api_key,
            agent_id=agent_id,
        )

        async with ws:
            # Create RoomManager with our handler
            room_manager = RoomManager(
                agent_id=agent_id,
                agent_name=agent_name,
                api_client=api_client,
                ws_client=ws,
                message_handler=filtered_handler,
            )

            # Subscribe room manager to the chat (uses internal filtering)
            await room_manager.subscribe_to_room(chat_id)
            print(f"RoomManager subscribed to chat: {chat_id}")
            await asyncio.sleep(0.2)

            # Send message mentioning the USER (not the agent)
            # This should be filtered out because:
            # 1. It's sent by the agent (sender_id == agent_id) - own message filter
            # 2. It doesn't mention the agent - @mention filter
            print(f"\nSending message @{user_peer.name} (no agent mention)...")
            response = await api_client.agent_api.create_agent_chat_message(
                chat_id,
                message=ChatMessageRequest(
                    content=f"Hello @{user_peer.name}, how are you?",
                    mentions=[Mention(id=user_peer.id, name=user_peer.name)],
                ),
            )
            msg_id = response.data.id
            print(f"Sent message ID: {msg_id}")

            # Give time for message to arrive through WebSocket
            await asyncio.sleep(1.0)

        # Verify filtering works:
        print(f"\nMessages that passed filter: {len(filtered_messages)}")

        # The message should NOT pass the filter because:
        # 1. RoomManager filters out messages where sender_id == agent_id (own messages)
        # 2. AND it doesn't have agent @mention
        message_passed = any(m.id == msg_id for m in filtered_messages)
        assert not message_passed, (
            "Message without agent @mention should be filtered out"
        )
        print("Message without agent @mention was correctly filtered out")

        print("\nMessage filtering test passed!")


@requires_api
class TestWebSocketRoomManager:
    """Test RoomManager WebSocket integration."""

    async def test_room_manager_subscribes_to_existing_rooms(
        self, api_client, integration_settings
    ):
        """Test that RoomManager can subscribe to all agent's existing rooms."""
        print("\n" + "=" * 60)
        print("Testing: RoomManager subscribes to existing rooms")
        print("=" * 60)

        from thenvoi.agent.core.room_manager import RoomManager

        # Get agent info
        response = await api_client.agent_api.get_agent_me()
        agent_id = response.data.id
        agent_name = response.data.name
        print(f"Agent: {agent_name} (ID: {agent_id})")

        # Track events
        messages_received: list[MessageCreatedPayload] = []

        async def message_handler(payload: MessageCreatedPayload):
            messages_received.append(payload)
            print(f"  [Handler] Message in room {payload.chat_room_id}")

        ws = WebSocketClient(
            ws_url=integration_settings.thenvoi_ws_url,
            api_key=integration_settings.thenvoi_api_key,
            agent_id=agent_id,
        )

        async with ws:
            room_manager = RoomManager(
                agent_id=agent_id,
                agent_name=agent_name,
                api_client=api_client,
                ws_client=ws,
                message_handler=message_handler,
            )

            # Subscribe to all existing rooms
            num_rooms = await room_manager.subscribe_to_all_rooms()
            print(f"Subscribed to {num_rooms} existing rooms")

            # Also subscribe to room events for dynamic updates
            await room_manager.subscribe_to_room_events()
            print("Subscribed to room add/remove events")

            # Give time for subscriptions to be established
            await asyncio.sleep(0.5)

        print(f"\nRoomManager subscribed to {num_rooms} rooms successfully!")
        assert num_rooms >= 0, "Should return number of rooms (may be 0)"

        print("\nRoomManager subscription test passed!")

    async def test_room_manager_handles_dynamic_room_addition(
        self, api_client, integration_settings
    ):
        """Test that RoomManager automatically subscribes when agent is added to new room."""
        print("\n" + "=" * 60)
        print("Testing: RoomManager handles dynamic room addition")
        print("=" * 60)

        from thenvoi.agent.core.room_manager import RoomManager

        # Get agent info
        response = await api_client.agent_api.get_agent_me()
        agent_id = response.data.id
        agent_name = response.data.name
        print(f"Agent: {agent_name} (ID: {agent_id})")

        # Track events
        rooms_added: list[str] = []
        room_added_event = asyncio.Event()

        async def message_handler(payload: MessageCreatedPayload):
            pass  # Not testing messages here

        async def on_room_added(room_id: str):
            rooms_added.append(room_id)
            room_added_event.set()
            print(f"  [RoomManager] Room added callback: {room_id}")

        ws = WebSocketClient(
            ws_url=integration_settings.thenvoi_ws_url,
            api_key=integration_settings.thenvoi_api_key,
            agent_id=agent_id,
        )

        async with ws:
            room_manager = RoomManager(
                agent_id=agent_id,
                agent_name=agent_name,
                api_client=api_client,
                ws_client=ws,
                message_handler=message_handler,
                on_room_added=on_room_added,
            )

            # Subscribe to room events
            await room_manager.subscribe_to_room_events()
            print("RoomManager subscribed to room events")

            await asyncio.sleep(0.2)

            # Create a new room (agent becomes owner automatically)
            response = await api_client.agent_api.create_agent_chat(
                chat=ChatRoomRequest(title="Dynamic Room Test")
            )
            new_room_id = response.data.id
            print(f"Created new room: {new_room_id}")

            # Wait for room_added callback
            try:
                await asyncio.wait_for(room_added_event.wait(), timeout=5.0)
                print("RoomManager received room_added and auto-subscribed!")
            except asyncio.TimeoutError:
                print("Timeout waiting for room_added")

        # Verify room was added
        assert new_room_id in rooms_added, (
            f"Should have received room_added for {new_room_id}"
        )
        print(f"\nVerified: Room {new_room_id} was dynamically added")

        print("\nDynamic room addition test passed!")
