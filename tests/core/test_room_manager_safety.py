"""
Core safety tests for RoomManager.

These tests prevent catastrophic bugs:
- Infinite loops (agent responding to itself)
- Spam (responding to every message)
"""

from thenvoi.agent.core.room_manager import RoomManager
from thenvoi.client.streaming import MessageCreatedPayload, MessageMetadata, Mention


def test_filters_own_messages(mock_api_client, mock_websocket, dummy_message_handler):
    """CRITICAL: Agent must never respond to its own messages (prevents infinite loops)."""
    manager = RoomManager(
        agent_id="agent-123",
        agent_name="TestBot",
        api_client=mock_api_client,
        ws_client=mock_websocket,
        message_handler=dummy_message_handler,
    )

    own_message = MessageCreatedPayload(
        id="msg-own",
        content="@TestBot hello",
        message_type="text",
        metadata=MessageMetadata(
            mentions=[Mention(id="agent-123", username="TestBot")], status="sent"
        ),
        sender_id="agent-123",  # Same as agent_id
        sender_type="Agent",
        chat_room_id="room-123",
        inserted_at="not-relevant",
        updated_at="not-relevant",
    )

    # Should be filtered out to prevent infinite loops
    assert not manager._is_message_for_agent(own_message)


def test_only_responds_to_at_mentions(
    mock_api_client, mock_websocket, dummy_message_handler
):
    """Agent should ONLY respond to @mentions, not every message with its name."""
    manager = RoomManager(
        agent_id="agent-123",
        agent_name="TestBot",
        api_client=mock_api_client,
        ws_client=mock_websocket,
        message_handler=dummy_message_handler,
    )

    # No mention at all - should ignore
    no_mention = MessageCreatedPayload(
        id="msg-no-mention",
        content="irrelevant",
        message_type="text",
        metadata=MessageMetadata(mentions=[], status="sent"),
        sender_id="user-456",
        sender_type="User",
        chat_room_id="room-123",
        inserted_at="not-relevant",
        updated_at="not-relevant",
    )
    assert not manager._is_message_for_agent(no_mention)

    # With @mention - should respond
    with_mention = MessageCreatedPayload(
        id="msg-with-mention",
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
    assert manager._is_message_for_agent(with_mention)

    # Just name without @ - should ignore (no mention in metadata)
    just_name = MessageCreatedPayload(
        id="msg-just-name",
        content="irrelevant",
        message_type="text",
        metadata=MessageMetadata(mentions=[], status="sent"),
        sender_id="user-456",
        sender_type="User",
        chat_room_id="room-123",
        inserted_at="not-relevant",
        updated_at="not-relevant",
    )
    assert not manager._is_message_for_agent(just_name)


def test_ignores_other_agent_mentions(
    mock_api_client, mock_websocket, dummy_message_handler
):
    """Should ignore messages mentioning OTHER agents."""
    manager = RoomManager(
        agent_id="agent-123",
        agent_name="TestBot",
        api_client=mock_api_client,
        ws_client=mock_websocket,
        message_handler=dummy_message_handler,
    )

    other_agent = MessageCreatedPayload(
        id="msg-other",
        content="irrelevant",
        message_type="text",
        metadata=MessageMetadata(
            mentions=[Mention(id="agent-999", username="OtherBot")], status="sent"
        ),
        sender_id="user-456",
        sender_type="User",
        chat_room_id="room-123",
        inserted_at="not-relevant",
        updated_at="not-relevant",
    )
    assert not manager._is_message_for_agent(other_agent)


def test_handles_empty_content_safely(
    mock_api_client, mock_websocket, dummy_message_handler
):
    """Should handle edge cases without crashing."""
    manager = RoomManager(
        agent_id="agent-123",
        agent_name="TestBot",
        api_client=mock_api_client,
        ws_client=mock_websocket,
        message_handler=dummy_message_handler,
    )

    empty_msg = MessageCreatedPayload(
        id="msg-empty",
        content="",
        message_type="text",
        metadata=MessageMetadata(mentions=[], status="sent"),
        sender_id="user-456",
        sender_type="User",
        chat_room_id="room-123",
        inserted_at="not-relevant",
        updated_at="not-relevant",
    )
    assert not manager._is_message_for_agent(empty_msg)
