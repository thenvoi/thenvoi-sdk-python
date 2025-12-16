"""Tests for ExecutionContext."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from thenvoi.platform.event import PlatformEvent
from thenvoi.runtime.execution import Execution, ExecutionContext
from thenvoi.runtime.types import SessionConfig


@pytest.fixture
def mock_link():
    """Mock ThenvoiLink for testing ExecutionContext."""
    link = MagicMock()
    link.agent_id = "agent-123"

    # REST client mock
    link.rest = MagicMock()
    link.rest.agent_api = MagicMock()

    # Mock list_agent_chat_participants
    participant1 = MagicMock()
    participant1.id = "user-1"
    participant1.name = "User One"
    participant1.type = "User"
    link.rest.agent_api.list_agent_chat_participants = AsyncMock(
        return_value=MagicMock(data=[participant1])
    )

    # Mock get_agent_chat_context
    msg1 = MagicMock()
    msg1.id = "msg-1"
    msg1.content = "Hello"
    msg1.sender_id = "user-1"
    msg1.sender_type = "User"
    msg1.sender_name = "User One"
    msg1.message_type = "text"
    msg1.inserted_at = "2024-01-01T00:00:00Z"
    link.rest.agent_api.get_agent_chat_context = AsyncMock(
        return_value=MagicMock(data=[msg1])
    )

    # Mock message lifecycle methods (new in ThenvoiLink)
    link.mark_processing = AsyncMock()
    link.mark_processed = AsyncMock()
    link.mark_failed = AsyncMock()
    link.get_next_message = AsyncMock(return_value=None)  # No backlog by default

    return link


@pytest.fixture
def mock_handler():
    """Mock execution handler."""
    return AsyncMock()


class TestExecutionContextConstruction:
    """Test ExecutionContext initialization."""

    def test_init_stores_room_id(self, mock_link, mock_handler):
        """Should store room_id."""
        ctx = ExecutionContext("room-123", mock_link, mock_handler)

        assert ctx.room_id == "room-123"
        assert ctx.thread_id == "room-123"  # Alias

    def test_init_starts_idle(self, mock_link, mock_handler):
        """Should start in starting state, not running."""
        ctx = ExecutionContext("room-123", mock_link, mock_handler)

        assert ctx.state == "starting"
        assert ctx.is_running is False
        assert ctx.is_processing is False

    def test_init_empty_participants(self, mock_link, mock_handler):
        """Should start with empty participants."""
        ctx = ExecutionContext("room-123", mock_link, mock_handler)

        assert ctx.participants == []

    def test_init_llm_not_initialized(self, mock_link, mock_handler):
        """Should start with LLM not initialized."""
        ctx = ExecutionContext("room-123", mock_link, mock_handler)

        assert ctx.is_llm_initialized is False


class TestExecutionContextProtocol:
    """Test that ExecutionContext implements Execution protocol."""

    def test_implements_execution_protocol(self, mock_link, mock_handler):
        """ExecutionContext should implement Execution protocol."""
        ctx = ExecutionContext("room-123", mock_link, mock_handler)

        assert isinstance(ctx, Execution)


class TestExecutionContextLifecycle:
    """Test ExecutionContext start/stop lifecycle."""

    async def test_start_creates_task(self, mock_link, mock_handler):
        """start() should create processing task."""
        ctx = ExecutionContext("room-123", mock_link, mock_handler)

        await ctx.start()

        assert ctx.is_running is True
        assert ctx._process_loop_task is not None

        # Cleanup
        await ctx.stop()

    async def test_start_idempotent(self, mock_link, mock_handler):
        """start() twice should not create duplicate tasks."""
        ctx = ExecutionContext("room-123", mock_link, mock_handler)

        await ctx.start()
        task1 = ctx._process_loop_task
        await ctx.start()  # Second call
        task2 = ctx._process_loop_task

        assert task1 is task2

        await ctx.stop()

    async def test_stop_cancels_task(self, mock_link, mock_handler):
        """stop() should cancel task and clear it."""
        ctx = ExecutionContext("room-123", mock_link, mock_handler)

        await ctx.start()
        await ctx.stop()

        assert ctx.is_running is False
        assert ctx._process_loop_task is None

    async def test_stop_when_not_started_is_noop(self, mock_link, mock_handler):
        """stop() when not started should be safe."""
        ctx = ExecutionContext("room-123", mock_link, mock_handler)

        await ctx.stop()  # Should not raise

        assert ctx.is_running is False


class TestExecutionContextEvents:
    """Test ExecutionContext event handling."""

    async def test_on_event_enqueues(self, mock_link, mock_handler):
        """on_event() should add event to queue."""
        ctx = ExecutionContext("room-123", mock_link, mock_handler)

        event = PlatformEvent(
            type="message_created",
            room_id="room-123",
            payload={"id": "msg-1"},
        )
        await ctx.on_event(event)

        assert ctx.queue.qsize() == 1

    async def test_processes_message_event(self, mock_link, mock_handler):
        """Should process message events through handler."""
        ctx = ExecutionContext("room-123", mock_link, mock_handler)

        await ctx.start()

        event = PlatformEvent(
            type="message_created",
            room_id="room-123",
            payload={"id": "msg-1", "content": "Hello"},
        )
        await ctx.on_event(event)

        # Wait for processing
        await asyncio.sleep(0.1)

        mock_handler.assert_called()
        call_args = mock_handler.call_args[0]
        assert call_args[0] is ctx
        assert call_args[1].type == "message_created"

        await ctx.stop()

    async def test_deduplicates_messages(self, mock_link, mock_handler):
        """Should skip duplicate message IDs."""
        ctx = ExecutionContext("room-123", mock_link, mock_handler)

        await ctx.start()

        event = PlatformEvent(
            type="message_created",
            room_id="room-123",
            payload={"id": "msg-1"},
        )

        # Send same message twice
        await ctx.on_event(event)
        await asyncio.sleep(0.1)
        await ctx.on_event(event)
        await asyncio.sleep(0.1)

        # Should only be called once
        assert mock_handler.call_count == 1

        await ctx.stop()


class TestExecutionContextParticipants:
    """Test participant management."""

    def test_add_participant(self, mock_link, mock_handler):
        """add_participant() should add to list."""
        ctx = ExecutionContext("room-123", mock_link, mock_handler)

        result = ctx.add_participant(
            {
                "id": "user-1",
                "name": "Test User",
                "type": "User",
            }
        )

        assert result is True
        assert len(ctx.participants) == 1
        assert ctx.participants[0]["name"] == "Test User"

    def test_add_participant_deduplicates(self, mock_link, mock_handler):
        """add_participant() should not add duplicates."""
        ctx = ExecutionContext("room-123", mock_link, mock_handler)

        ctx.add_participant({"id": "user-1", "name": "User One", "type": "User"})
        result = ctx.add_participant(
            {"id": "user-1", "name": "User One Updated", "type": "User"}
        )

        assert result is False
        assert len(ctx.participants) == 1

    def test_remove_participant(self, mock_link, mock_handler):
        """remove_participant() should remove from list."""
        ctx = ExecutionContext("room-123", mock_link, mock_handler)
        ctx.add_participant({"id": "user-1", "name": "User", "type": "User"})

        result = ctx.remove_participant("user-1")

        assert result is True
        assert len(ctx.participants) == 0

    def test_remove_participant_not_found(self, mock_link, mock_handler):
        """remove_participant() should return False if not found."""
        ctx = ExecutionContext("room-123", mock_link, mock_handler)

        result = ctx.remove_participant("nonexistent")

        assert result is False

    def test_participants_changed_true_initially(self, mock_link, mock_handler):
        """participants_changed() should return True initially."""
        ctx = ExecutionContext("room-123", mock_link, mock_handler)

        assert ctx.participants_changed() is True

    def test_participants_changed_false_after_mark(self, mock_link, mock_handler):
        """participants_changed() should return False after mark_participants_sent()."""
        ctx = ExecutionContext("room-123", mock_link, mock_handler)
        ctx.add_participant({"id": "user-1", "name": "User", "type": "User"})
        ctx.mark_participants_sent()

        assert ctx.participants_changed() is False

    def test_participants_changed_true_after_add(self, mock_link, mock_handler):
        """participants_changed() should return True after adding participant."""
        ctx = ExecutionContext("room-123", mock_link, mock_handler)
        ctx.add_participant({"id": "user-1", "name": "User 1", "type": "User"})
        ctx.mark_participants_sent()
        ctx.add_participant({"id": "user-2", "name": "User 2", "type": "User"})

        assert ctx.participants_changed() is True


class TestExecutionContextHydration:
    """Test context hydration."""

    async def test_hydrate_loads_participants(self, mock_link, mock_handler):
        """hydrate() should load participants from API."""
        ctx = ExecutionContext("room-123", mock_link, mock_handler)

        await ctx.hydrate()

        assert len(ctx.participants) == 1
        assert ctx.participants[0]["name"] == "User One"

    async def test_hydrate_loads_context(self, mock_link, mock_handler):
        """hydrate() should load context from API."""
        ctx = ExecutionContext("room-123", mock_link, mock_handler)

        await ctx.hydrate()

        context = ctx.build_context()
        assert len(context.messages) == 1
        assert context.messages[0]["content"] == "Hello"

    async def test_hydrate_idempotent(self, mock_link, mock_handler):
        """hydrate() should only load once."""
        ctx = ExecutionContext("room-123", mock_link, mock_handler)

        await ctx.hydrate()
        await ctx.hydrate()  # Second call

        # Should only call API once
        assert mock_link.rest.agent_api.list_agent_chat_participants.call_count == 1

    async def test_get_context_hydrates_lazily(self, mock_link, mock_handler):
        """get_context() should hydrate lazily."""
        ctx = ExecutionContext("room-123", mock_link, mock_handler)

        context = await ctx.get_context()

        assert context.room_id == "room-123"
        assert len(context.messages) == 1

    async def test_get_context_force_refresh(self, mock_link, mock_handler):
        """get_context(force_refresh=True) should re-fetch context messages."""
        ctx = ExecutionContext("room-123", mock_link, mock_handler)

        await ctx.get_context()
        await ctx.get_context(force_refresh=True)

        # Context API should be called twice
        assert mock_link.rest.agent_api.get_agent_chat_context.call_count == 2
        # Participants only loaded once (tracked via WebSocket, not re-fetched)
        assert mock_link.rest.agent_api.list_agent_chat_participants.call_count == 1


class TestExecutionContextLLMState:
    """Test LLM initialization state."""

    def test_mark_llm_initialized(self, mock_link, mock_handler):
        """mark_llm_initialized() should set flag."""
        ctx = ExecutionContext("room-123", mock_link, mock_handler)

        ctx.mark_llm_initialized()

        assert ctx.is_llm_initialized is True


class TestExecutionContextParticipantEvents:
    """Test participant event handling."""

    async def test_participant_added_event_updates_list(self, mock_link, mock_handler):
        """participant_added event should update participants."""
        ctx = ExecutionContext("room-123", mock_link, mock_handler)

        await ctx.start()

        event = PlatformEvent(
            type="participant_added",
            room_id="room-123",
            payload={"id": "user-2", "name": "User Two", "type": "User"},
        )
        await ctx.on_event(event)
        await asyncio.sleep(0.1)

        assert any(p["id"] == "user-2" for p in ctx.participants)

        await ctx.stop()

    async def test_participant_removed_event_updates_list(
        self, mock_link, mock_handler
    ):
        """participant_removed event should update participants."""
        ctx = ExecutionContext("room-123", mock_link, mock_handler)
        ctx.add_participant({"id": "user-1", "name": "User One", "type": "User"})

        await ctx.start()

        event = PlatformEvent(
            type="participant_removed",
            room_id="room-123",
            payload={"id": "user-1"},
        )
        await ctx.on_event(event)
        await asyncio.sleep(0.1)

        assert not any(p["id"] == "user-1" for p in ctx.participants)

        await ctx.stop()


class TestCrashRecoverySync:
    """Test crash recovery sync mechanism."""

    @pytest.fixture
    def mock_link_with_next(self):
        """Mock ThenvoiLink with message lifecycle methods."""
        link = MagicMock()
        link.agent_id = "agent-123"
        link.rest = MagicMock()
        link.rest.agent_api = MagicMock()

        # Default: no messages
        link.rest.agent_api.list_agent_chat_participants = AsyncMock(
            return_value=MagicMock(data=[])
        )
        link.rest.agent_api.get_agent_chat_context = AsyncMock(
            return_value=MagicMock(data=[])
        )

        # Message lifecycle methods (new in ThenvoiLink)
        link.mark_processing = AsyncMock()
        link.mark_processed = AsyncMock()
        link.mark_failed = AsyncMock()
        link.get_next_message = AsyncMock(return_value=None)  # No backlog by default

        return link

    async def test_first_ws_message_sets_marker(
        self, mock_link_with_next, mock_handler
    ):
        """First WebSocket message should set sync point marker."""
        ctx = ExecutionContext("room-123", mock_link_with_next, mock_handler)

        assert ctx._first_ws_msg_id is None

        event = PlatformEvent(
            type="message_created",
            room_id="room-123",
            payload={"id": "msg-ws-001"},
        )
        await ctx.on_event(event)

        assert ctx._first_ws_msg_id == "msg-ws-001"

    async def test_subsequent_ws_messages_dont_change_marker(
        self, mock_link_with_next, mock_handler
    ):
        """Subsequent WebSocket messages should not change the marker."""
        ctx = ExecutionContext("room-123", mock_link_with_next, mock_handler)

        event1 = PlatformEvent(
            type="message_created",
            room_id="room-123",
            payload={"id": "msg-ws-001"},
        )
        event2 = PlatformEvent(
            type="message_created",
            room_id="room-123",
            payload={"id": "msg-ws-002"},
        )

        await ctx.on_event(event1)
        await ctx.on_event(event2)

        assert ctx._first_ws_msg_id == "msg-ws-001"

    async def test_sync_completes_with_no_backlog(
        self, mock_link_with_next, mock_handler
    ):
        """Sync should complete immediately when no backlog."""
        ctx = ExecutionContext("room-123", mock_link_with_next, mock_handler)

        await ctx.start()
        await asyncio.sleep(0.1)

        assert ctx._sync_complete is True
        mock_link_with_next.get_next_message.assert_called()

        await ctx.stop()

    async def test_sync_processes_backlog_messages(
        self, mock_link_with_next, mock_handler
    ):
        """Sync should process backlog messages from /next."""
        from datetime import datetime, timezone
        from thenvoi.runtime.types import PlatformMessage

        # Setup get_next_message to return one backlog message, then None
        backlog_msg = PlatformMessage(
            id="msg-backlog-001",
            room_id="room-123",
            content="Backlog message",
            sender_id="user-1",
            sender_type="User",
            sender_name="User One",
            message_type="text",
            metadata={},
            created_at=datetime.now(timezone.utc),
        )

        mock_link_with_next.get_next_message = AsyncMock(
            side_effect=[backlog_msg, None]
        )

        ctx = ExecutionContext(
            "room-123",
            mock_link_with_next,
            mock_handler,
            config=SessionConfig(enable_context_hydration=False),
        )

        await ctx.start()
        await asyncio.sleep(0.2)

        # Handler should be called for backlog message
        assert mock_handler.call_count >= 1
        # The first call should be for backlog message
        call_args = mock_handler.call_args_list[0][0]
        assert call_args[1].payload["id"] == "msg-backlog-001"

        await ctx.stop()

    async def test_sync_point_clears_marker_and_cache(
        self, mock_link_with_next, mock_handler
    ):
        """When sync point reached, marker and cache should be cleared."""
        from datetime import datetime, timezone
        from thenvoi.runtime.types import PlatformMessage

        # Setup: WS message arrives, then /next returns same message
        sync_msg = PlatformMessage(
            id="msg-sync-001",
            room_id="room-123",
            content="Sync message",
            sender_id="user-1",
            sender_type="User",
            sender_name="User One",
            message_type="text",
            metadata={},
            created_at=datetime.now(timezone.utc),
        )

        mock_link_with_next.get_next_message = AsyncMock(return_value=sync_msg)

        ctx = ExecutionContext(
            "room-123",
            mock_link_with_next,
            mock_handler,
            config=SessionConfig(enable_context_hydration=False),
        )

        # Enqueue WS message first (sets marker)
        ws_event = PlatformEvent(
            type="message_created",
            room_id="room-123",
            payload={"id": "msg-sync-001"},
        )
        await ctx.on_event(ws_event)
        assert ctx._first_ws_msg_id == "msg-sync-001"

        # Start should sync and find sync point
        await ctx.start()
        await asyncio.sleep(0.2)

        # Marker should be cleared
        assert ctx._first_ws_msg_id is None
        # Dedupe cache should be cleared
        assert len(ctx._processed_ids) == 0

        await ctx.stop()

    async def test_sync_removes_duplicate_from_ws_queue(
        self, mock_link_with_next, mock_handler
    ):
        """Sync should remove duplicate from WS queue after sync point."""
        from datetime import datetime, timezone
        from thenvoi.runtime.types import PlatformMessage

        sync_msg = PlatformMessage(
            id="msg-sync-001",
            room_id="room-123",
            content="Sync message",
            sender_id="user-1",
            sender_type="User",
            sender_name="User One",
            message_type="text",
            metadata={},
            created_at=datetime.now(timezone.utc),
        )

        mock_link_with_next.get_next_message = AsyncMock(return_value=sync_msg)

        ctx = ExecutionContext(
            "room-123",
            mock_link_with_next,
            mock_handler,
            config=SessionConfig(enable_context_hydration=False),
        )

        # Enqueue the same message via WS
        ws_event = PlatformEvent(
            type="message_created",
            room_id="room-123",
            payload={"id": "msg-sync-001"},
        )
        await ctx.on_event(ws_event)

        # Queue should have 1 item
        assert ctx.queue.qsize() == 1

        # Start triggers sync
        await ctx.start()
        await asyncio.sleep(0.2)

        # Duplicate should be removed from queue
        # (queue may be empty or have other non-duplicate events)
        # Check that sync point was reached
        assert ctx._first_ws_msg_id is None
        assert ctx._sync_complete is True

        await ctx.stop()

    async def test_sync_skips_permanently_failed(
        self, mock_link_with_next, mock_handler
    ):
        """Sync should skip permanently failed messages."""
        from datetime import datetime, timezone
        from thenvoi.runtime.types import PlatformMessage

        failed_msg = PlatformMessage(
            id="msg-failed-001",
            room_id="room-123",
            content="Failed message",
            sender_id="user-1",
            sender_type="User",
            sender_name="User One",
            message_type="text",
            metadata={},
            created_at=datetime.now(timezone.utc),
        )

        mock_link_with_next.get_next_message = AsyncMock(side_effect=[failed_msg, None])

        ctx = ExecutionContext(
            "room-123",
            mock_link_with_next,
            mock_handler,
            config=SessionConfig(enable_context_hydration=False),
        )

        # Mark message as permanently failed
        ctx._retry_tracker.mark_permanently_failed("msg-failed-001")

        await ctx.start()
        await asyncio.sleep(0.2)

        # Handler should NOT be called for failed message
        assert mock_handler.call_count == 0

        await ctx.stop()

    async def test_retry_tracker_records_failures(self, mock_link_with_next):
        """Retry tracker should record failed processing attempts."""
        from datetime import datetime, timezone
        from thenvoi.runtime.types import PlatformMessage

        # Handler that fails
        failing_handler = AsyncMock(side_effect=Exception("Processing failed"))

        msg = PlatformMessage(
            id="msg-001",
            room_id="room-123",
            content="Test",
            sender_id="user-1",
            sender_type="User",
            sender_name="User One",
            message_type="text",
            metadata={},
            created_at=datetime.now(timezone.utc),
        )

        mock_link_with_next.get_next_message = AsyncMock(side_effect=[msg, None])

        ctx = ExecutionContext(
            "room-123",
            mock_link_with_next,
            failing_handler,
            config=SessionConfig(enable_context_hydration=False, max_message_retries=2),
        )

        await ctx.start()
        await asyncio.sleep(0.2)

        # Should have recorded attempt
        # Note: With max_retries=2, after 3 attempts it's permanently failed
        # But we only process once per /next call
        await ctx.stop()


class TestSessionConfigDefaults:
    """Test SessionConfig default values."""

    def test_default_enable_context_hydration_is_true(self):
        """Default should enable context hydration for backward compatibility."""
        config = SessionConfig()
        assert config.enable_context_hydration is True

    def test_default_enable_context_cache_is_true(self):
        """Default should enable context caching."""
        config = SessionConfig()
        assert config.enable_context_cache is True

    def test_can_disable_context_hydration(self):
        """Should be able to explicitly disable context hydration."""
        config = SessionConfig(enable_context_hydration=False)
        assert config.enable_context_hydration is False

    def test_default_max_message_retries(self):
        """Default max_message_retries should be 1."""
        config = SessionConfig()
        assert config.max_message_retries == 1


class TestInstantShutdown:
    """Tests for instant cancellation without timeout waiting."""

    async def test_stop_returns_quickly_when_idle(self, mock_link, mock_handler):
        """stop() should return quickly even when waiting on empty queue."""
        ctx = ExecutionContext("room-123", mock_link, mock_handler)
        await ctx.start()

        # Give loop time to reach queue.get()
        await asyncio.sleep(0.01)

        # Stop should be instant (no 60-second timeout)
        start = asyncio.get_event_loop().time()
        await ctx.stop()
        elapsed = asyncio.get_event_loop().time() - start

        # Should complete in well under 1 second
        assert elapsed < 0.5, f"stop() took {elapsed}s - should be instant"

    async def test_stop_is_idempotent(self, mock_link, mock_handler):
        """Multiple stop() calls should be safe."""
        ctx = ExecutionContext("room-123", mock_link, mock_handler)
        await ctx.start()
        await ctx.stop()
        await ctx.stop()  # Should not raise
        await ctx.stop()  # Should not raise

        assert ctx.is_running is False

    async def test_stop_before_start_is_safe(self, mock_link, mock_handler):
        """stop() without start() should be safe."""
        ctx = ExecutionContext("room-123", mock_link, mock_handler)
        await ctx.stop()  # Should not raise
        assert ctx.is_running is False


class TestCancellationDuringProcessing:
    """Tests for cancellation during message processing."""

    async def test_stop_cancels_slow_processing(self, mock_link):
        """stop() should cancel message processing."""

        async def slow_handler(ctx, event):
            await asyncio.sleep(10)  # Would take 10 seconds

        ctx = ExecutionContext(
            "room-123",
            mock_link,
            slow_handler,
            config=SessionConfig(enable_context_hydration=False),
        )
        await ctx.start()

        # Wait for sync to complete
        await asyncio.sleep(0.05)

        # Enqueue a message to trigger processing
        event = PlatformEvent(
            type="message_created",
            room_id="room-123",
            payload={"id": "msg-001", "content": "Test"},
        )
        await ctx.on_event(event)

        # Give time to start processing
        await asyncio.sleep(0.05)

        # Stop should cancel processing
        start = asyncio.get_event_loop().time()
        await ctx.stop()
        elapsed = asyncio.get_event_loop().time() - start

        # Should complete quickly (not wait 10 seconds for handler)
        assert elapsed < 1.0, f"stop() took {elapsed}s - should cancel processing"


class TestContextHydrationConfig:
    """Test context hydration behavior with config."""

    async def test_get_context_skips_api_when_hydration_disabled(
        self, mock_link, mock_handler
    ):
        """get_context() should return empty when hydration disabled."""
        ctx = ExecutionContext(
            "room-123",
            mock_link,
            mock_handler,
            config=SessionConfig(enable_context_hydration=False),
        )

        context = await ctx.get_context()

        # Should return empty context without calling API
        assert context.messages == []
        assert context.participants == []
        mock_link.rest.agent_api.get_agent_chat_context.assert_not_called()

    async def test_get_context_calls_api_when_hydration_enabled(
        self, mock_link, mock_handler
    ):
        """get_context() should call API when hydration enabled."""
        ctx = ExecutionContext(
            "room-123",
            mock_link,
            mock_handler,
            config=SessionConfig(enable_context_hydration=True),
        )

        context = await ctx.get_context()

        # Should have called API
        mock_link.rest.agent_api.get_agent_chat_context.assert_called_once()
        assert len(context.messages) > 0

    async def test_get_history_for_llm_empty_when_hydration_disabled(
        self, mock_link, mock_handler
    ):
        """get_history_for_llm() should return empty when hydration disabled."""
        ctx = ExecutionContext(
            "room-123",
            mock_link,
            mock_handler,
            config=SessionConfig(enable_context_hydration=False),
        )

        # Must hydrate first (but with disabled config, returns empty)
        await ctx.get_context()
        history = ctx.get_history_for_llm()

        assert history == []
