"""
Unit tests for AgentSession - per-room session management.

Tests critical logic:
1. Participant management (add/remove/deduplication)
2. participants_changed() detection
3. get_history_for_llm() role mapping
4. State machine transitions
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone

from thenvoi.agent.core.session import AgentSession
from thenvoi.agent.core.types import PlatformMessage, ConversationContext


@pytest.fixture
def mock_coordinator():
    """Mock ThenvoiAgent coordinator for session tests."""
    coordinator = AsyncMock()
    coordinator.agent_id = "agent-123"
    coordinator._get_participants_internal = AsyncMock(return_value=[])
    coordinator._fetch_context = AsyncMock(
        return_value=ConversationContext(
            room_id="room-123",
            messages=[],
            participants=[],
            hydrated_at=datetime.now(timezone.utc),
        )
    )
    coordinator._mark_processing = AsyncMock()
    coordinator._mark_processed = AsyncMock()
    coordinator._mark_failed = AsyncMock()
    coordinator._get_next_message = AsyncMock(return_value=None)
    coordinator._create_agent_tools = MagicMock()
    return coordinator


@pytest.fixture
def session(mock_api_client, mock_coordinator):
    """Create a session for testing."""

    async def dummy_handler(msg, tools):
        pass

    return AgentSession(
        room_id="room-123",
        api_client=mock_api_client,
        on_message=dummy_handler,
        coordinator=mock_coordinator,
    )


class TestParticipantManagement:
    """Tests for participant add/remove/deduplication."""

    def test_add_participant(self, session):
        """Should add participant to list."""
        session.add_participant({"id": "user-456", "name": "Test User", "type": "User"})

        assert len(session.participants) == 1
        assert session.participants[0]["id"] == "user-456"
        assert session.participants[0]["name"] == "Test User"

    def test_add_participant_deduplicates_by_id(self, session):
        """Should not add duplicate participant with same ID."""
        session.add_participant({"id": "user-456", "name": "Test User", "type": "User"})
        session.add_participant(
            {"id": "user-456", "name": "Test User Updated", "type": "User"}
        )

        assert len(session.participants) == 1
        # Name should be original (first add wins)
        assert session.participants[0]["name"] == "Test User"

    def test_add_multiple_participants(self, session):
        """Should add multiple different participants."""
        session.add_participant({"id": "user-1", "name": "User One", "type": "User"})
        session.add_participant({"id": "agent-2", "name": "Agent Two", "type": "Agent"})

        assert len(session.participants) == 2

    def test_remove_participant(self, session):
        """Should remove participant by ID."""
        session.add_participant({"id": "user-456", "name": "Test User", "type": "User"})
        session.add_participant({"id": "agent-789", "name": "Helper", "type": "Agent"})

        session.remove_participant({"id": "user-456"})

        assert len(session.participants) == 1
        assert session.participants[0]["id"] == "agent-789"

    def test_remove_nonexistent_participant(self, session):
        """Should handle removing participant that doesn't exist."""
        session.add_participant({"id": "user-456", "name": "Test User", "type": "User"})

        # Should not raise
        session.remove_participant({"id": "nonexistent"})

        assert len(session.participants) == 1

    def test_participants_property_returns_copy(self, session):
        """participants property should return a copy, not the internal list."""
        session.add_participant({"id": "user-456", "name": "Test User", "type": "User"})

        participants = session.participants
        participants.append({"id": "fake", "name": "Fake"})

        # Internal list should be unaffected
        assert len(session.participants) == 1


class TestParticipantsChanged:
    """Tests for participants_changed() detection."""

    def test_returns_true_when_never_sent(self, session):
        """Should return True when participants never sent to LLM."""
        session.add_participant({"id": "user-456", "name": "Test User", "type": "User"})

        assert session.participants_changed() is True

    def test_returns_false_after_marked_sent(self, session):
        """Should return False after participants marked as sent."""
        session.add_participant({"id": "user-456", "name": "Test User", "type": "User"})
        session.mark_participants_sent()

        assert session.participants_changed() is False

    def test_returns_true_when_participant_added(self, session):
        """Should return True when new participant added after last sent."""
        session.add_participant({"id": "user-456", "name": "Test User", "type": "User"})
        session.mark_participants_sent()

        session.add_participant({"id": "agent-789", "name": "Helper", "type": "Agent"})

        assert session.participants_changed() is True

    def test_returns_true_when_participant_removed(self, session):
        """Should return True when participant removed after last sent."""
        session.add_participant({"id": "user-456", "name": "Test User", "type": "User"})
        session.add_participant({"id": "agent-789", "name": "Helper", "type": "Agent"})
        session.mark_participants_sent()

        session.remove_participant({"id": "agent-789"})

        assert session.participants_changed() is True

    def test_compares_by_id_not_name(self, session):
        """Should compare participants by ID, not name."""
        session.add_participant(
            {"id": "user-456", "name": "Original Name", "type": "User"}
        )
        session.mark_participants_sent()

        # Same ID, different name - should NOT count as changed
        session._participants[0]["name"] = "Updated Name"

        assert session.participants_changed() is False


class TestGetHistoryForLLM:
    """Tests for get_history_for_llm() role mapping."""

    async def test_maps_agent_to_assistant_role(self, session, mock_coordinator):
        """Agent messages should have role 'assistant'."""
        mock_coordinator._fetch_context = AsyncMock(
            return_value=ConversationContext(
                room_id="room-123",
                messages=[
                    {
                        "id": "msg-1",
                        "content": "Hello",
                        "sender_type": "Agent",
                        "sender_name": "TestBot",
                    },
                ],
                participants=[],
                hydrated_at=datetime.now(timezone.utc),
            )
        )

        history = await session.get_history_for_llm()

        assert len(history) == 1
        assert history[0]["role"] == "assistant"

    async def test_maps_user_to_user_role(self, session, mock_coordinator):
        """User messages should have role 'user'."""
        mock_coordinator._fetch_context = AsyncMock(
            return_value=ConversationContext(
                room_id="room-123",
                messages=[
                    {
                        "id": "msg-1",
                        "content": "Hello",
                        "sender_type": "User",
                        "sender_name": "Test User",
                    },
                ],
                participants=[],
                hydrated_at=datetime.now(timezone.utc),
            )
        )

        history = await session.get_history_for_llm()

        assert len(history) == 1
        assert history[0]["role"] == "user"

    async def test_maps_system_to_user_role(self, session, mock_coordinator):
        """System messages should have role 'user' (non-Agent = user)."""
        mock_coordinator._fetch_context = AsyncMock(
            return_value=ConversationContext(
                room_id="room-123",
                messages=[
                    {
                        "id": "msg-1",
                        "content": "System message",
                        "sender_type": "System",
                        "sender_name": "System",
                    },
                ],
                participants=[],
                hydrated_at=datetime.now(timezone.utc),
            )
        )

        history = await session.get_history_for_llm()

        assert len(history) == 1
        assert history[0]["role"] == "user"

    async def test_excludes_specified_message_id(self, session, mock_coordinator):
        """Should exclude message with specified ID (usually current message)."""
        mock_coordinator._fetch_context = AsyncMock(
            return_value=ConversationContext(
                room_id="room-123",
                messages=[
                    {
                        "id": "msg-1",
                        "content": "First",
                        "sender_type": "User",
                        "sender_name": "User",
                    },
                    {
                        "id": "msg-2",
                        "content": "Second",
                        "sender_type": "Agent",
                        "sender_name": "Bot",
                    },
                    {
                        "id": "msg-3",
                        "content": "Third",
                        "sender_type": "User",
                        "sender_name": "User",
                    },
                ],
                participants=[],
                hydrated_at=datetime.now(timezone.utc),
            )
        )

        history = await session.get_history_for_llm(exclude_message_id="msg-2")

        assert len(history) == 2
        assert all(h["content"] != "Second" for h in history)

    async def test_preserves_sender_name(self, session, mock_coordinator):
        """Should include sender_name in history."""
        mock_coordinator._fetch_context = AsyncMock(
            return_value=ConversationContext(
                room_id="room-123",
                messages=[
                    {
                        "id": "msg-1",
                        "content": "Hello",
                        "sender_type": "User",
                        "sender_name": "John Doe",
                    },
                ],
                participants=[],
                hydrated_at=datetime.now(timezone.utc),
            )
        )

        history = await session.get_history_for_llm()

        assert history[0]["sender_name"] == "John Doe"


class TestBuildParticipantsMessage:
    """Tests for build_participants_message() formatting."""

    def test_empty_participants(self, session):
        """Should handle empty participants list."""
        message = session.build_participants_message()

        assert "No other participants" in message

    def test_includes_participant_details(self, session):
        """Should include name, ID, and type for each participant."""
        session.add_participant({"id": "user-456", "name": "Test User", "type": "User"})
        session.add_participant(
            {"id": "agent-789", "name": "Helper Bot", "type": "Agent"}
        )

        message = session.build_participants_message()

        assert "Test User" in message
        assert "user-456" in message
        assert "User" in message
        assert "Helper Bot" in message
        assert "agent-789" in message
        assert "Agent" in message


class TestStateMachine:
    """Tests for session state transitions."""

    def test_initial_state_is_starting(self, session):
        """Session should start in 'starting' state."""
        assert session.state == "starting"

    def test_is_processing_property(self, session):
        """is_processing should reflect state."""
        assert session.is_processing is False

        session.state = "processing"
        assert session.is_processing is True

        session.state = "idle"
        assert session.is_processing is False


class TestLLMInitialization:
    """Tests for LLM initialization tracking."""

    def test_initial_state_not_initialized(self, session):
        """Session should start with LLM not initialized."""
        assert session.is_llm_initialized is False

    def test_mark_llm_initialized(self, session):
        """mark_llm_initialized should set flag."""
        session.mark_llm_initialized()

        assert session.is_llm_initialized is True


class TestEnqueueMessage:
    """Tests for message enqueuing."""

    def test_enqueue_adds_to_queue(self, session):
        """enqueue_message should add message to queue."""
        msg = PlatformMessage(
            id="msg-123",
            room_id="room-123",
            content="Hello",
            sender_id="user-456",
            sender_type="User",
            sender_name="Test User",
            message_type="text",
            metadata={},
            created_at=datetime.now(timezone.utc),
        )

        session.enqueue_message(msg)

        assert session.queue.qsize() == 1

    def test_enqueue_multiple_messages(self, session):
        """Should enqueue multiple messages in order."""
        for i in range(3):
            msg = PlatformMessage(
                id=f"msg-{i}",
                room_id="room-123",
                content=f"Message {i}",
                sender_id="user-456",
                sender_type="User",
                sender_name="Test User",
                message_type="text",
                metadata={},
                created_at=datetime.now(timezone.utc),
            )
            session.enqueue_message(msg)

        assert session.queue.qsize() == 3


class TestSessionLifecycle:
    """Tests for session start() and stop() lifecycle."""

    async def test_start_sets_is_running(self, session):
        """start() should set _is_running = True."""
        assert session._is_running is False

        await session.start()

        assert session._is_running is True

        # Cleanup
        await session.stop()

    async def test_start_creates_process_loop_task(self, session):
        """start() should create _process_loop_task."""
        assert session._process_loop_task is None

        await session.start()

        assert session._process_loop_task is not None
        assert session._process_loop_task.get_name() == "session-room-123"

        # Cleanup
        await session.stop()

    async def test_start_idempotent(self, session):
        """Calling start() twice should not create duplicate tasks."""
        await session.start()
        first_task = session._process_loop_task

        await session.start()  # Second call
        second_task = session._process_loop_task

        assert first_task is second_task

        # Cleanup
        await session.stop()

    async def test_stop_sets_is_running_false(self, session):
        """stop() should set _is_running = False."""
        await session.start()
        assert session._is_running is True

        await session.stop()

        assert session._is_running is False

    async def test_stop_clears_task(self, session):
        """stop() should clear _process_loop_task."""
        await session.start()
        assert session._process_loop_task is not None

        await session.stop()

        assert session._process_loop_task is None

    async def test_stop_idempotent(self, session):
        """Calling stop() when not running should be safe."""
        assert session._is_running is False

        # Should not raise
        await session.stop()

        assert session._is_running is False


class TestSynchronizeWithNext:
    """
    Tests for _synchronize_with_next() - ID-based synchronization.

    ALGORITHM:
    1. Call /next to get next unprocessed message
    2. If None → no backlog, synced
    3. Check if message ID matches head of WebSocket queue
    4. If match → synced! Process once, remove duplicate from queue
    5. If no match → process /next message, repeat
    """

    async def test_empty_backlog_syncs_immediately(self, session, mock_coordinator):
        """When /next returns None, sync completes immediately."""
        mock_coordinator._get_next_message = AsyncMock(return_value=None)
        session._is_running = True  # Required for sync loop

        await session._synchronize_with_next()

        mock_coordinator._get_next_message.assert_called_once_with("room-123")
        # Queue should remain empty
        assert session.queue.qsize() == 0

    async def test_processes_backlog_messages_directly(self, session, mock_coordinator):
        """Backlog messages should be processed directly, not queued."""
        processed_ids = []

        async def tracking_handler(msg, tools):
            processed_ids.append(msg.id)

        session._on_message = tracking_handler
        session._is_running = True  # Required for sync loop

        msg1 = PlatformMessage(
            id="backlog-1",
            room_id="room-123",
            content="Backlog 1",
            sender_id="user-456",
            sender_type="User",
            sender_name="User",
            message_type="text",
            metadata={},
            created_at=datetime.now(timezone.utc),
        )
        msg2 = PlatformMessage(
            id="backlog-2",
            room_id="room-123",
            content="Backlog 2",
            sender_id="user-456",
            sender_type="User",
            sender_name="User",
            message_type="text",
            metadata={},
            created_at=datetime.now(timezone.utc),
        )
        mock_coordinator._get_next_message = AsyncMock(side_effect=[msg1, msg2, None])

        await session._synchronize_with_next()

        # Messages should be processed directly (not in queue)
        assert processed_ids == ["backlog-1", "backlog-2"]
        assert session.queue.qsize() == 0

    async def test_sync_point_when_next_matches_websocket_head(
        self, session, mock_coordinator
    ):
        """
        When /next returns a message that matches WebSocket queue head,
        we've reached the sync point.
        """
        processed_ids = []

        async def tracking_handler(msg, tools):
            processed_ids.append(msg.id)

        session._on_message = tracking_handler
        session._is_running = True  # Required for sync loop

        # Backlog has 3 messages: backlog-1, backlog-2, sync-point
        backlog_msg1 = PlatformMessage(
            id="backlog-1",
            room_id="room-123",
            content="Backlog 1",
            sender_id="user-456",
            sender_type="User",
            sender_name="User",
            message_type="text",
            metadata={},
            created_at=datetime.now(timezone.utc),
        )
        backlog_msg2 = PlatformMessage(
            id="backlog-2",
            room_id="room-123",
            content="Backlog 2",
            sender_id="user-456",
            sender_type="User",
            sender_name="User",
            message_type="text",
            metadata={},
            created_at=datetime.now(timezone.utc),
        )
        sync_point_msg = PlatformMessage(
            id="sync-point",
            room_id="room-123",
            content="Sync point",
            sender_id="user-456",
            sender_type="User",
            sender_name="User",
            message_type="text",
            metadata={},
            created_at=datetime.now(timezone.utc),
        )

        # WebSocket already has sync-point and a newer message
        ws_sync_point = PlatformMessage(
            id="sync-point",
            room_id="room-123",
            content="Sync point",
            sender_id="user-456",
            sender_type="User",
            sender_name="User",
            message_type="text",
            metadata={},
            created_at=datetime.now(timezone.utc),
        )
        ws_newer = PlatformMessage(
            id="websocket-new",
            room_id="room-123",
            content="New WS",
            sender_id="user-456",
            sender_type="User",
            sender_name="User",
            message_type="text",
            metadata={},
            created_at=datetime.now(timezone.utc),
        )
        session.enqueue_message(ws_sync_point)
        session.enqueue_message(ws_newer)

        # /next returns backlog-1, backlog-2, sync-point
        mock_coordinator._get_next_message = AsyncMock(
            side_effect=[backlog_msg1, backlog_msg2, sync_point_msg]
        )

        await session._synchronize_with_next()

        # Should have processed: backlog-1, backlog-2, sync-point (once)
        assert processed_ids == ["backlog-1", "backlog-2", "sync-point"]

        # Duplicate was removed from queue, only websocket-new remains
        assert session.queue.qsize() == 1
        remaining = await session.queue.get()
        assert remaining.id == "websocket-new"

    async def test_sync_point_message_processed_only_once(
        self, session, mock_coordinator
    ):
        """The sync point message should be processed only once (not twice)."""
        processed_ids = []

        async def tracking_handler(msg, tools):
            processed_ids.append(msg.id)

        session._on_message = tracking_handler
        session._is_running = True  # Required for sync loop

        # Both /next and WebSocket have the same message
        msg = PlatformMessage(
            id="same-msg",
            room_id="room-123",
            content="Same",
            sender_id="user-456",
            sender_type="User",
            sender_name="User",
            message_type="text",
            metadata={},
            created_at=datetime.now(timezone.utc),
        )
        ws_msg = PlatformMessage(
            id="same-msg",
            room_id="room-123",
            content="Same",
            sender_id="user-456",
            sender_type="User",
            sender_name="User",
            message_type="text",
            metadata={},
            created_at=datetime.now(timezone.utc),
        )

        session.enqueue_message(ws_msg)
        mock_coordinator._get_next_message = AsyncMock(return_value=msg)

        await session._synchronize_with_next()

        # Should be processed exactly once
        assert processed_ids == ["same-msg"]
        # Queue should be empty (duplicate removed)
        assert session.queue.qsize() == 0

    async def test_handles_sync_error_gracefully(self, session, mock_coordinator):
        """Should continue to WebSocket on error."""
        session._is_running = True  # Required for sync loop
        mock_coordinator._get_next_message = AsyncMock(
            side_effect=Exception("API Error")
        )

        # Should not raise
        await session._synchronize_with_next()

        # Queue unchanged
        assert session.queue.qsize() == 0


class TestProcessingLoop:
    """Tests for _process_loop() - full message processing flow."""

    async def test_transitions_to_idle_after_sync(self, session, mock_coordinator):
        """After sync, state should be 'idle' and _synchronized True."""
        mock_coordinator._get_next_message = AsyncMock(return_value=None)

        await session.start()
        await asyncio.sleep(0.1)

        assert session.state == "idle"
        assert session._synchronized is True

        await session.stop()

    async def test_full_flow_backlog_then_websocket(self, session, mock_coordinator):
        """
        FULL FLOW TEST:

        Scenario:
        - Agent was offline, 2 messages accumulated (backlog-1, backlog-2)
        - Agent starts, WebSocket receives backlog-2 and websocket-1
        - Agent should:
          1. Process backlog-1 from /next
          2. Process backlog-2 from /next (matches WebSocket head → sync point)
          3. Remove backlog-2 duplicate from WebSocket queue
          4. Process websocket-1 from WebSocket queue
        """
        processed_ids = []

        async def tracking_handler(msg, tools):
            processed_ids.append(msg.id)

        session._on_message = tracking_handler

        # Backlog messages
        backlog_msg1 = PlatformMessage(
            id="backlog-1",
            room_id="room-123",
            content="Backlog 1",
            sender_id="user-456",
            sender_type="User",
            sender_name="User",
            message_type="text",
            metadata={},
            created_at=datetime.now(timezone.utc),
        )
        backlog_msg2 = PlatformMessage(
            id="backlog-2",
            room_id="room-123",
            content="Backlog 2",
            sender_id="user-456",
            sender_type="User",
            sender_name="User",
            message_type="text",
            metadata={},
            created_at=datetime.now(timezone.utc),
        )

        # WebSocket already has backlog-2 (overlap) and a newer message
        ws_backlog2 = PlatformMessage(
            id="backlog-2",
            room_id="room-123",
            content="Backlog 2",
            sender_id="user-456",
            sender_type="User",
            sender_name="User",
            message_type="text",
            metadata={},
            created_at=datetime.now(timezone.utc),
        )
        ws_new = PlatformMessage(
            id="websocket-1",
            room_id="room-123",
            content="New WebSocket",
            sender_id="user-456",
            sender_type="User",
            sender_name="User",
            message_type="text",
            metadata={},
            created_at=datetime.now(timezone.utc),
        )

        # Setup: /next returns backlog-1, backlog-2, then None (but we sync on backlog-2)
        mock_coordinator._get_next_message = AsyncMock(
            side_effect=[backlog_msg1, backlog_msg2]
        )

        # Pre-populate WebSocket queue
        session.enqueue_message(ws_backlog2)
        session.enqueue_message(ws_new)

        await session.start()
        await asyncio.sleep(0.3)

        # Verify correct order: backlog-1, backlog-2 (once), websocket-1
        assert processed_ids == ["backlog-1", "backlog-2", "websocket-1"], (
            f"Expected correct order, got: {processed_ids}"
        )

        # /next should have been called twice (backlog-1, backlog-2 → sync point)
        assert mock_coordinator._get_next_message.call_count == 2

        await session.stop()

    async def test_websocket_messages_during_backlog_processing(
        self, session, mock_coordinator
    ):
        """
        WebSocket messages arriving during backlog processing should be queued
        and processed after sync completes.
        """
        processed_ids = []

        async def tracking_handler(msg, tools):
            processed_ids.append(msg.id)
            # Simulate processing time
            await asyncio.sleep(0.05)

        session._on_message = tracking_handler

        backlog_msg = PlatformMessage(
            id="backlog-1",
            room_id="room-123",
            content="Backlog 1",
            sender_id="user-456",
            sender_type="User",
            sender_name="User",
            message_type="text",
            metadata={},
            created_at=datetime.now(timezone.utc),
        )

        call_count = [0]

        async def get_next_with_side_effect(room_id):
            call_count[0] += 1
            if call_count[0] == 1:
                # While processing, a new WebSocket message arrives
                ws_new = PlatformMessage(
                    id="ws-during-sync",
                    room_id="room-123",
                    content="During sync",
                    sender_id="user-456",
                    sender_type="User",
                    sender_name="User",
                    message_type="text",
                    metadata={},
                    created_at=datetime.now(timezone.utc),
                )
                session.enqueue_message(ws_new)
                return backlog_msg
            return None

        mock_coordinator._get_next_message = get_next_with_side_effect

        await session.start()
        await asyncio.sleep(0.3)

        # Should process backlog first, then WebSocket
        assert "backlog-1" in processed_ids
        assert "ws-during-sync" in processed_ids
        # Backlog should come before the new WebSocket message
        assert processed_ids.index("backlog-1") < processed_ids.index("ws-during-sync")

        await session.stop()

    async def test_processes_messages_from_queue(self, session, mock_coordinator):
        """Should process messages from queue."""
        mock_coordinator._get_next_message = AsyncMock(return_value=None)
        handler_called = []

        async def tracking_handler(msg, tools):
            handler_called.append(msg.id)

        session._on_message = tracking_handler

        # Start session
        await session.start()
        await asyncio.sleep(0.1)  # Let backlog sync complete

        # Enqueue a message
        msg = PlatformMessage(
            id="test-msg",
            room_id="room-123",
            content="Hello",
            sender_id="user-456",
            sender_type="User",
            sender_name="User",
            message_type="text",
            metadata={},
            created_at=datetime.now(timezone.utc),
        )
        session.enqueue_message(msg)
        await asyncio.sleep(0.1)  # Let processing happen

        assert "test-msg" in handler_called

        await session.stop()

    async def test_exits_on_cancelled_error(self, session, mock_coordinator):
        """Should exit cleanly on CancelledError."""
        mock_coordinator._get_next_message = AsyncMock(return_value=None)

        await session.start()
        await asyncio.sleep(0.1)

        # Cancel the task
        session._process_loop_task.cancel()

        # Wait for cleanup
        try:
            await session._process_loop_task
        except asyncio.CancelledError:
            pass

        # Task should have exited
        assert session.state in ["idle", "starting"]


class TestMessageLifecycle:
    """Tests for _process_message() - full message lifecycle."""

    @pytest.fixture
    def sample_msg(self):
        return PlatformMessage(
            id="msg-123",
            room_id="room-123",
            content="Hello",
            sender_id="user-456",
            sender_type="User",
            sender_name="User",
            message_type="text",
            metadata={},
            created_at=datetime.now(timezone.utc),
        )

    async def test_transitions_state_to_processing(self, session, sample_msg):
        """Should set state to 'processing' during message handling."""
        states_during_processing = []

        async def capture_state(msg, tools):
            states_during_processing.append(session.state)

        session._on_message = capture_state

        await session._process_message(sample_msg)

        assert "processing" in states_during_processing

    async def test_transitions_state_back_to_idle(self, session, sample_msg):
        """Should return to 'idle' after processing."""

        async def dummy_handler(msg, tools):
            pass

        session._on_message = dummy_handler

        await session._process_message(sample_msg)

        assert session.state == "idle"

    async def test_calls_mark_processing(self, session, mock_coordinator, sample_msg):
        """Should call _mark_processing with message ID."""

        async def dummy_handler(msg, tools):
            pass

        session._on_message = dummy_handler

        await session._process_message(sample_msg)

        mock_coordinator._mark_processing.assert_called_once_with("msg-123", "room-123")

    async def test_calls_mark_processed_on_success(
        self, session, mock_coordinator, sample_msg
    ):
        """Should call _mark_processed on successful processing."""

        async def dummy_handler(msg, tools):
            pass

        session._on_message = dummy_handler

        await session._process_message(sample_msg)

        mock_coordinator._mark_processed.assert_called_once_with("msg-123", "room-123")

    async def test_calls_mark_failed_on_error(
        self, session, mock_coordinator, sample_msg
    ):
        """Should call _mark_failed when handler raises."""

        async def failing_handler(msg, tools):
            raise ValueError("Handler error")

        session._on_message = failing_handler

        await session._process_message(sample_msg)

        mock_coordinator._mark_failed.assert_called_once()
        call_args = mock_coordinator._mark_failed.call_args
        assert call_args[0][0] == "msg-123"
        assert call_args[0][1] == "room-123"
        assert "Handler error" in call_args[0][2]

    async def test_returns_to_idle_on_error(self, session, sample_msg):
        """Should return to 'idle' even when handler fails."""

        async def failing_handler(msg, tools):
            raise ValueError("Handler error")

        session._on_message = failing_handler

        await session._process_message(sample_msg)

        assert session.state == "idle"

    async def test_hydrates_context_on_first_message(
        self, session, mock_coordinator, sample_msg
    ):
        """Should hydrate context on first message only."""

        async def dummy_handler(msg, tools):
            pass

        session._on_message = dummy_handler
        assert session._context_hydrated is False

        await session._process_message(sample_msg)

        assert session._context_hydrated is True
        mock_coordinator._fetch_context.assert_called_once()

    async def test_skips_hydration_on_subsequent_messages(
        self, session, mock_coordinator, sample_msg
    ):
        """Should not re-hydrate context on subsequent messages."""

        async def dummy_handler(msg, tools):
            pass

        session._on_message = dummy_handler
        session._context_hydrated = True  # Simulate already hydrated

        await session._process_message(sample_msg)

        # Should not call _fetch_context again
        mock_coordinator._fetch_context.assert_not_called()

    async def test_creates_agent_tools(self, session, mock_coordinator, sample_msg):
        """Should create AgentTools for the message."""

        async def dummy_handler(msg, tools):
            pass

        session._on_message = dummy_handler

        await session._process_message(sample_msg)

        mock_coordinator._create_agent_tools.assert_called_once_with("room-123")

    async def test_calls_handler_with_message_and_tools(
        self, session, mock_coordinator, sample_msg
    ):
        """Should call handler with message and tools."""
        received_args = []

        async def tracking_handler(msg, tools):
            received_args.append((msg, tools))

        session._on_message = tracking_handler
        mock_tools = MagicMock()
        mock_coordinator._create_agent_tools = MagicMock(return_value=mock_tools)

        await session._process_message(sample_msg)

        assert len(received_args) == 1
        assert received_args[0][0] == sample_msg
        assert received_args[0][1] == mock_tools


class TestContextHydration:
    """Tests for _hydrate_context() and get_context() - lazy context loading."""

    async def test_hydrate_calls_load_participants_first(
        self, session, mock_coordinator
    ):
        """_hydrate_context should call load_participants() before fetching context."""
        call_order = []

        async def track_participants(room_id):
            call_order.append("load_participants")
            return [{"id": "user-1", "name": "User", "type": "User"}]

        async def track_fetch_context(room_id):
            call_order.append("fetch_context")
            return ConversationContext(
                room_id=room_id,
                messages=[],
                participants=[],
                hydrated_at=datetime.now(timezone.utc),
            )

        mock_coordinator._get_participants_internal = track_participants
        mock_coordinator._fetch_context = track_fetch_context

        await session._hydrate_context()

        assert call_order == ["load_participants", "fetch_context"]

    async def test_hydrate_populates_context_cache(self, session, mock_coordinator):
        """_hydrate_context should populate _context_cache."""
        context = ConversationContext(
            room_id="room-123",
            messages=[{"id": "msg-1", "content": "Hello"}],
            participants=[{"id": "user-1", "name": "User"}],
            hydrated_at=datetime.now(timezone.utc),
        )
        mock_coordinator._fetch_context = AsyncMock(return_value=context)

        assert session._context_cache is None

        await session._hydrate_context()

        assert session._context_cache is not None
        assert len(session._context_cache.messages) == 1
        assert session._context_cache.messages[0]["content"] == "Hello"

    async def test_hydrate_creates_fallback_on_error(self, session, mock_coordinator):
        """_hydrate_context should create fallback empty context on error."""
        mock_coordinator._fetch_context = AsyncMock(side_effect=Exception("API Error"))

        await session._hydrate_context()

        # Should have created fallback context
        assert session._context_cache is not None
        assert session._context_cache.room_id == "room-123"
        assert session._context_cache.messages == []
        assert session._context_cache.participants == []

    async def test_get_context_hydrates_on_first_call(self, session, mock_coordinator):
        """get_context() should hydrate on first call when cache is None."""
        context = ConversationContext(
            room_id="room-123",
            messages=[],
            participants=[],
            hydrated_at=datetime.now(timezone.utc),
        )
        mock_coordinator._fetch_context = AsyncMock(return_value=context)

        assert session._context_cache is None

        result = await session.get_context()

        assert result is not None
        mock_coordinator._fetch_context.assert_called_once()

    async def test_get_context_uses_cache_on_subsequent_calls(
        self, session, mock_coordinator
    ):
        """get_context() should use cache and not re-fetch."""
        context = ConversationContext(
            room_id="room-123",
            messages=[{"id": "cached"}],
            participants=[],
            hydrated_at=datetime.now(timezone.utc),
        )
        session._context_cache = context

        result = await session.get_context()

        assert result.messages[0]["id"] == "cached"
        mock_coordinator._fetch_context.assert_not_called()

    async def test_get_context_force_refresh(self, session, mock_coordinator):
        """get_context(force_refresh=True) should re-fetch even with cache."""
        old_context = ConversationContext(
            room_id="room-123",
            messages=[{"id": "old"}],
            participants=[],
            hydrated_at=datetime.now(timezone.utc),
        )
        new_context = ConversationContext(
            room_id="room-123",
            messages=[{"id": "new"}],
            participants=[],
            hydrated_at=datetime.now(timezone.utc),
        )
        session._context_cache = old_context
        mock_coordinator._fetch_context = AsyncMock(return_value=new_context)

        result = await session.get_context(force_refresh=True)

        assert result.messages[0]["id"] == "new"
        mock_coordinator._fetch_context.assert_called_once()

    async def test_get_context_refreshes_on_stale_cache(
        self, session, mock_coordinator
    ):
        """get_context() should refresh when cache TTL expires."""
        from datetime import timedelta

        # Create stale context (older than TTL)
        stale_time = datetime.now(timezone.utc) - timedelta(
            seconds=session.config.context_cache_ttl_seconds + 10
        )
        stale_context = ConversationContext(
            room_id="room-123",
            messages=[{"id": "stale"}],
            participants=[],
            hydrated_at=stale_time,
        )
        fresh_context = ConversationContext(
            room_id="room-123",
            messages=[{"id": "fresh"}],
            participants=[],
            hydrated_at=datetime.now(timezone.utc),
        )
        session._context_cache = stale_context
        mock_coordinator._fetch_context = AsyncMock(return_value=fresh_context)

        result = await session.get_context()

        # Should have refreshed
        assert result.messages[0]["id"] == "fresh"
        mock_coordinator._fetch_context.assert_called_once()
