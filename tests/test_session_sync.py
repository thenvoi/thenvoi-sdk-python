"""
Tests for AgentSession sync point tracking and LRU deduplication.

Tests cover:
- _first_ws_msg_id marker tracking in enqueue_message()
- LRU dedupe cache in _process_message()
- _synchronize_with_next() using marker instead of queue peeking
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone

from thenvoi.core.session import AgentSession
from thenvoi.core.types import PlatformMessage, SessionConfig


def make_message(msg_id: str, room_id: str = "room-123") -> PlatformMessage:
    """Helper to create test messages."""
    return PlatformMessage(
        id=msg_id,
        room_id=room_id,
        content=f"Content for {msg_id}",
        sender_id="user-456",
        sender_type="User",
        sender_name="Test User",
        message_type="text",
        metadata={},
        created_at=datetime.now(timezone.utc),
    )


class TestFirstWsMessageIdMarker:
    """Tests for _first_ws_msg_id tracking in enqueue_message()."""

    @pytest.fixture
    def session(self):
        """Create session with mocked dependencies."""
        mock_api = AsyncMock()
        mock_handler = AsyncMock()
        mock_coordinator = MagicMock()
        return AgentSession(
            room_id="room-123",
            api_client=mock_api,
            on_message=mock_handler,
            coordinator=mock_coordinator,
        )

    def test_first_enqueue_sets_marker(self, session):
        """First enqueued message should set _first_ws_msg_id."""
        assert session._first_ws_msg_id is None

        msg = make_message("msg-001")
        session.enqueue_message(msg)

        assert session._first_ws_msg_id == "msg-001"

    def test_subsequent_enqueue_does_not_change_marker(self, session):
        """Subsequent enqueues should not change the marker."""
        msg1 = make_message("msg-001")
        msg2 = make_message("msg-002")
        msg3 = make_message("msg-003")

        session.enqueue_message(msg1)
        session.enqueue_message(msg2)
        session.enqueue_message(msg3)

        # Should still be first message
        assert session._first_ws_msg_id == "msg-001"

    def test_marker_preserved_after_queue_empty(self, session):
        """Marker should be preserved even if queue becomes empty."""
        msg = make_message("msg-001")
        session.enqueue_message(msg)

        # Drain the queue
        session.queue.get_nowait()
        assert session.queue.empty()

        # Marker should still be set
        assert session._first_ws_msg_id == "msg-001"


class TestLruDedupeCache:
    """Tests for LRU dedupe cache in _process_message()."""

    @pytest.fixture
    def session(self):
        """Create session with mocked dependencies."""
        mock_api = AsyncMock()
        mock_handler = AsyncMock()
        mock_coordinator = MagicMock()
        mock_coordinator._mark_processing = AsyncMock()
        mock_coordinator._mark_processed = AsyncMock()
        mock_coordinator._mark_failed = AsyncMock()
        mock_coordinator._create_agent_tools = MagicMock()
        return AgentSession(
            room_id="room-123",
            api_client=mock_api,
            on_message=mock_handler,
            coordinator=mock_coordinator,
            config=SessionConfig(enable_context_hydration=False),
        )

    async def test_processed_message_added_to_cache(self, session):
        """Processed messages should be added to LRU cache."""
        msg = make_message("msg-001")

        await session._process_message(msg)

        assert "msg-001" in session._processed_ids

    async def test_duplicate_message_skipped(self, session):
        """Duplicate messages should be skipped."""
        msg = make_message("msg-001")

        # Process first time
        await session._process_message(msg)
        assert session._on_message.call_count == 1

        # Process same message again - should be skipped
        await session._process_message(msg)
        assert session._on_message.call_count == 1  # Still 1

    async def test_lru_cache_evicts_oldest(self, session):
        """LRU cache should evict oldest entries when full."""
        # _max_processed_ids is 5

        # Process 5 messages
        for i in range(5):
            msg = make_message(f"msg-{i:03d}")
            await session._process_message(msg)

        assert len(session._processed_ids) == 5
        assert "msg-000" in session._processed_ids

        # Process 6th message - should evict msg-000
        msg6 = make_message("msg-005")
        await session._process_message(msg6)

        assert len(session._processed_ids) == 5
        assert "msg-000" not in session._processed_ids
        assert "msg-005" in session._processed_ids

    async def test_duplicate_refreshes_lru_position(self, session):
        """Accessing duplicate should refresh its LRU position."""
        # Process 3 messages
        for i in range(3):
            msg = make_message(f"msg-{i:03d}")
            await session._process_message(msg)

        # Access msg-000 again (duplicate) - should move to end
        msg0_dup = make_message("msg-000")
        await session._process_message(msg0_dup)

        # Process 3 more to fill and overflow
        for i in range(3, 6):
            msg = make_message(f"msg-{i:03d}")
            await session._process_message(msg)

        # msg-000 should still be there (was refreshed)
        # msg-001 should be evicted (oldest after refresh)
        assert "msg-000" in session._processed_ids
        assert "msg-001" not in session._processed_ids


class TestSynchronizeWithNext:
    """Tests for _synchronize_with_next() using marker."""

    @pytest.fixture
    def session(self):
        """Create session with mocked dependencies."""
        mock_api = AsyncMock()
        mock_handler = AsyncMock()
        mock_coordinator = MagicMock()
        mock_coordinator._get_next_message = AsyncMock()
        mock_coordinator._mark_processing = AsyncMock()
        mock_coordinator._mark_processed = AsyncMock()
        mock_coordinator._create_agent_tools = MagicMock()
        session = AgentSession(
            room_id="room-123",
            api_client=mock_api,
            on_message=mock_handler,
            coordinator=mock_coordinator,
            config=SessionConfig(enable_context_hydration=False),
        )
        session._is_running = True
        return session

    async def test_sync_completes_when_next_returns_none(self, session):
        """Sync should complete when /next returns None (no backlog)."""
        session._coordinator._get_next_message.return_value = None

        await session._synchronize_with_next()

        session._coordinator._get_next_message.assert_called_once()

    async def test_sync_processes_backlog_messages(self, session):
        """Sync should process messages from /next until sync point."""
        backlog_msg = make_message("backlog-001")

        # Return backlog message, then None
        session._coordinator._get_next_message.side_effect = [backlog_msg, None]

        await session._synchronize_with_next()

        # Handler should be called for backlog message
        assert session._on_message.call_count == 1

    async def test_sync_point_reached_clears_marker_and_cache(self, session):
        """When sync point reached, marker and LRU cache should be cleared."""
        # First process a backlog message (populates LRU cache)
        backlog_msg = make_message("backlog-001")
        sync_msg = make_message("sync-001")

        # Enqueue sync message via WebSocket
        session.enqueue_message(sync_msg)
        assert session._first_ws_msg_id == "sync-001"

        # /next returns backlog first, then sync message
        session._coordinator._get_next_message.side_effect = [backlog_msg, sync_msg]

        await session._synchronize_with_next()

        # Marker should be cleared
        assert session._first_ws_msg_id is None
        # Queue should be empty (duplicate removed)
        assert session.queue.empty()
        # LRU cache should be cleared (no longer needed after sync)
        assert len(session._processed_ids) == 0

    async def test_sync_skips_permanently_failed(self, session):
        """Sync should skip permanently failed messages."""
        failed_msg = make_message("failed-001")
        session._retry_tracker.mark_permanently_failed("failed-001")

        session._coordinator._get_next_message.return_value = failed_msg

        await session._synchronize_with_next()

        # Handler should NOT be called
        session._on_message.assert_not_called()
