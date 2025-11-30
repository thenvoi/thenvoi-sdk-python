"""
Test that RoomManager correctly calls user-provided callbacks.

These are BEHAVIOR tests - users depend on these callbacks being called.
"""

from thenvoi.agent.core.room_manager import RoomManager
from thenvoi.client.streaming import (
    MessageCreatedPayload,
    MessageMetadata,
    Mention,
    RoomAddedPayload,
    RoomRemovedPayload,
    RoomOwner,
)


async def test_calls_message_handler_for_valid_messages(
    mock_api_client, mock_websocket
):
    """RoomManager should call message_handler for messages that pass filtering."""
    # Track what messages we received
    received_messages = []

    async def my_handler(msg: MessageCreatedPayload):
        received_messages.append(msg)

    manager = RoomManager(
        agent_id="agent-123",
        agent_name="TestBot",
        api_client=mock_api_client,
        ws_client=mock_websocket,
        message_handler=my_handler,
    )

    # Subscribe to a room
    await manager.subscribe_to_room("room-123")

    # Get the WebSocket callback that was registered
    call_args = mock_websocket.join_chat_room_channel.call_args
    room_id, ws_callback = call_args[0]

    # Simulate WebSocket sending us a message with @mention
    payload = MessageCreatedPayload(
        id="msg-test",
        content="irrelevant",
        message_type="text",
        metadata=MessageMetadata(
            mentions=[Mention(id="agent-123", username="TestBot")], status="sent"
        ),
        sender_id="user-456",
        sender_type="User",
        chat_room_id="room-123",
        inserted_at="not-relevant",
        updated_at="not-relevant",
    )
    await ws_callback(payload)

    # Verify our handler was called
    assert len(received_messages) == 1
    assert received_messages[0].sender_id == "user-456"
    assert received_messages[0].chat_room_id == "room-123"


async def test_does_not_call_handler_for_filtered_messages(
    mock_api_client, mock_websocket
):
    """RoomManager should NOT call message_handler for filtered messages."""
    received_messages = []

    async def my_handler(msg: MessageCreatedPayload):
        received_messages.append(msg)

    manager = RoomManager(
        agent_id="agent-123",
        agent_name="TestBot",
        api_client=mock_api_client,
        ws_client=mock_websocket,
        message_handler=my_handler,
    )

    await manager.subscribe_to_room("room-123")
    call_args = mock_websocket.join_chat_room_channel.call_args
    _, ws_callback = call_args[0]

    # Message without @mention - should be filtered
    payload = MessageCreatedPayload(
        id="msg-filtered",
        content="irrelevant",
        message_type="text",
        metadata=MessageMetadata(mentions=[], status="sent"),
        sender_id="user-456",
        sender_type="User",
        chat_room_id="room-123",
        inserted_at="not-relevant",
        updated_at="not-relevant",
    )
    await ws_callback(payload)

    # Handler should NOT have been called
    assert len(received_messages) == 0


async def test_calls_on_room_added_callback(
    mock_api_client, mock_websocket, dummy_message_handler
):
    """RoomManager should call on_room_added when WebSocket reports room addition."""
    added_rooms = []

    async def my_room_added_handler(room_id: str):
        added_rooms.append(room_id)

    manager = RoomManager(
        agent_id="agent-123",
        agent_name="TestBot",
        api_client=mock_api_client,
        ws_client=mock_websocket,
        message_handler=dummy_message_handler,
        on_room_added=my_room_added_handler,
    )

    await manager.subscribe_to_room_events()

    # Get the WebSocket callback that was registered
    call_args = mock_websocket.join_agent_rooms_channel.call_args
    _, room_added_cb, _ = call_args[0]

    # Simulate WebSocket telling us about room addition
    room_payload = RoomAddedPayload(
        id="room-999",
        owner=RoomOwner(id="user-123", name="Test User", type="User"),
        status="active",
        type="direct",
        title="Test Room",
        created_at="not-relevant",
        participant_role="member",
    )
    await room_added_cb(room_payload)

    # Verify our handler was called
    assert len(added_rooms) == 1
    assert added_rooms[0] == "room-999"


async def test_calls_on_room_removed_callback(
    mock_api_client, mock_websocket, dummy_message_handler
):
    """RoomManager should call on_room_removed when WebSocket reports room removal."""
    removed_rooms = []

    async def my_room_removed_handler(room_id: str):
        removed_rooms.append(room_id)

    manager = RoomManager(
        agent_id="agent-123",
        agent_name="TestBot",
        api_client=mock_api_client,
        ws_client=mock_websocket,
        message_handler=dummy_message_handler,
        on_room_removed=my_room_removed_handler,
    )

    # First subscribe to the room
    await manager.subscribe_to_room("room-999")

    # Subscribe to room events
    await manager.subscribe_to_room_events()

    # Get the WebSocket callback
    call_args = mock_websocket.join_agent_rooms_channel.call_args
    _, _, room_removed_cb = call_args[0]

    # Simulate WebSocket telling us about room removal
    room_payload = RoomRemovedPayload(
        id="room-999",
        status="removed",
        type="direct",
        title="Test Room",
        removed_at="not-relevant",
    )
    await room_removed_cb(room_payload)

    # Verify our handler was called
    assert len(removed_rooms) == 1
    assert removed_rooms[0] == "room-999"
