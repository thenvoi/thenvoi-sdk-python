"""
Tests for AgentSession asyncio shutdown pattern.

Tests cover:
- Instant cancellation via task.cancel() (no timeout waiting)
- is_running property reflects task state
- stop() returns immediately even when queue.get() is blocked
"""

import asyncio
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


class TestIsRunningProperty:
    """Tests for is_running property reflecting task state."""

    @pytest.fixture
    def session(self):
        """Create session with mocked dependencies."""
        mock_api = AsyncMock()
        mock_handler = AsyncMock()
        mock_coordinator = MagicMock()
        mock_coordinator._get_next_message = AsyncMock(return_value=None)
        return AgentSession(
            room_id="room-123",
            api_client=mock_api,
            on_message=mock_handler,
            coordinator=mock_coordinator,
        )

    def test_is_running_false_before_start(self, session):
        """is_running should be False before start()."""
        assert session.is_running is False

    async def test_is_running_true_after_start(self, session):
        """is_running should be True after start()."""
        await session.start()
        try:
            assert session.is_running is True
        finally:
            await session.stop()

    async def test_is_running_false_after_stop(self, session):
        """is_running should be False after stop()."""
        await session.start()
        await session.stop()
        assert session.is_running is False


class TestInstantShutdown:
    """Tests for instant cancellation without timeout waiting."""

    @pytest.fixture
    def session(self):
        """Create session with mocked dependencies."""
        mock_api = AsyncMock()
        mock_handler = AsyncMock()
        mock_coordinator = MagicMock()
        mock_coordinator._get_next_message = AsyncMock(return_value=None)
        return AgentSession(
            room_id="room-123",
            api_client=mock_api,
            on_message=mock_handler,
            coordinator=mock_coordinator,
        )

    async def test_stop_returns_quickly_when_idle(self, session):
        """stop() should return quickly even when waiting on empty queue."""
        await session.start()

        # Give loop time to reach queue.get()
        await asyncio.sleep(0.01)

        # Stop should be instant (no 60-second timeout)
        start = asyncio.get_event_loop().time()
        await session.stop()
        elapsed = asyncio.get_event_loop().time() - start

        # Should complete in well under 1 second
        assert elapsed < 0.5, f"stop() took {elapsed}s - should be instant"

    async def test_stop_is_idempotent(self, session):
        """Multiple stop() calls should be safe."""
        await session.start()
        await session.stop()
        await session.stop()  # Should not raise
        await session.stop()  # Should not raise

        assert session.is_running is False

    async def test_stop_before_start_is_safe(self, session):
        """stop() without start() should be safe."""
        await session.stop()  # Should not raise
        assert session.is_running is False


class TestCancellationDuringSync:
    """Tests for cancellation during synchronization phase."""

    @pytest.fixture
    def slow_sync_session(self):
        """Create session with slow /next that can be cancelled."""
        mock_api = AsyncMock()
        mock_handler = AsyncMock()
        mock_coordinator = MagicMock()

        # Simulate slow /next API that can be cancelled
        async def slow_get_next(room_id):
            await asyncio.sleep(10)  # Would take 10 seconds if not cancelled
            return None

        mock_coordinator._get_next_message = slow_get_next

        return AgentSession(
            room_id="room-123",
            api_client=mock_api,
            on_message=mock_handler,
            coordinator=mock_coordinator,
        )

    async def test_stop_cancels_sync_immediately(self, slow_sync_session):
        """stop() should cancel sync phase immediately."""
        await slow_sync_session.start()

        # Give time to enter sync
        await asyncio.sleep(0.01)

        # Stop should be instant despite slow sync
        start = asyncio.get_event_loop().time()
        await slow_sync_session.stop()
        elapsed = asyncio.get_event_loop().time() - start

        # Should complete in well under 1 second (not 10 seconds)
        assert elapsed < 0.5, f"stop() took {elapsed}s - sync should be cancelled"


class TestCancellationDuringProcessing:
    """Tests for cancellation during message processing."""

    @pytest.fixture
    def session_with_slow_handler(self):
        """Create session with slow message handler."""
        mock_api = AsyncMock()
        mock_coordinator = MagicMock()
        mock_coordinator._get_next_message = AsyncMock(return_value=None)
        mock_coordinator._mark_processing = AsyncMock()
        mock_coordinator._mark_processed = AsyncMock()
        mock_coordinator._create_agent_tools = MagicMock()

        async def slow_handler(msg, tools):
            await asyncio.sleep(10)  # Would take 10 seconds

        return AgentSession(
            room_id="room-123",
            api_client=mock_api,
            on_message=slow_handler,
            coordinator=mock_coordinator,
            config=SessionConfig(enable_context_hydration=False),
        )

    async def test_stop_cancels_processing(self, session_with_slow_handler):
        """stop() should cancel message processing."""
        session = session_with_slow_handler
        await session.start()

        # Wait for sync to complete
        await asyncio.sleep(0.05)

        # Enqueue a message to trigger processing
        msg = make_message("msg-001")
        session.enqueue_message(msg)

        # Give time to start processing
        await asyncio.sleep(0.05)

        # Stop should cancel processing
        start = asyncio.get_event_loop().time()
        await session.stop()
        elapsed = asyncio.get_event_loop().time() - start

        # Should complete quickly (not wait 10 seconds for handler)
        assert elapsed < 1.0, f"stop() took {elapsed}s - should cancel processing"
