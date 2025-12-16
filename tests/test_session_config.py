"""
Session configuration tests - verify enable_context_hydration behavior.

Tests cover:
- SessionConfig defaults
- get_history_for_llm() returns empty when hydration disabled
- _process_message() skips hydration when disabled
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from thenvoi.core.types import SessionConfig, PlatformMessage
from thenvoi.core.session import AgentSession


class TestSessionConfigDefaults:
    """Verify SessionConfig default values."""

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


class TestGetHistoryForLLMHydrationDisabled:
    """Test get_history_for_llm() when hydration is disabled."""

    @pytest.fixture
    def mock_coordinator(self):
        """Create a mock ThenvoiAgent coordinator."""
        coordinator = AsyncMock()
        coordinator.agent_id = "agent-123"
        coordinator._get_participants_internal = AsyncMock(return_value=[])
        coordinator._fetch_context = AsyncMock()
        return coordinator

    @pytest.fixture
    def mock_api_client(self):
        """Create a mock API client."""
        return AsyncMock()

    @pytest.fixture
    def dummy_handler(self):
        """Create a dummy message handler."""

        async def handler(msg, tools):
            pass

        return handler

    @pytest.mark.asyncio
    async def test_returns_empty_when_hydration_disabled(
        self, mock_api_client, mock_coordinator, dummy_handler
    ):
        """Should return empty list when enable_context_hydration is False."""
        config = SessionConfig(enable_context_hydration=False)
        session = AgentSession(
            room_id="room-123",
            api_client=mock_api_client,
            on_message=dummy_handler,
            coordinator=mock_coordinator,
            config=config,
        )

        history = await session.get_history_for_llm()

        assert history == []
        # Should NOT have called _hydrate_context or fetch_context
        mock_coordinator._fetch_context.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_empty_with_exclude_id_when_hydration_disabled(
        self, mock_api_client, mock_coordinator, dummy_handler
    ):
        """Should return empty list even with exclude_message_id parameter."""
        config = SessionConfig(enable_context_hydration=False)
        session = AgentSession(
            room_id="room-123",
            api_client=mock_api_client,
            on_message=dummy_handler,
            coordinator=mock_coordinator,
            config=config,
        )

        history = await session.get_history_for_llm(exclude_message_id="msg-456")

        assert history == []


class TestGetHistoryForLLMHydrationEnabled:
    """Test get_history_for_llm() when hydration is enabled (default)."""

    @pytest.fixture
    def mock_coordinator(self):
        """Create a mock ThenvoiAgent coordinator."""
        from thenvoi.core.types import ConversationContext

        coordinator = AsyncMock()
        coordinator.agent_id = "agent-123"
        coordinator._get_participants_internal = AsyncMock(return_value=[])
        coordinator._fetch_context = AsyncMock(
            return_value=ConversationContext(
                room_id="room-123",
                messages=[
                    {
                        "id": "msg-1",
                        "content": "Hello",
                        "sender_type": "User",
                        "sender_name": "Alice",
                    },
                    {
                        "id": "msg-2",
                        "content": "Hi there!",
                        "sender_type": "Agent",
                        "sender_name": "TestBot",
                    },
                ],
                participants=[],
                hydrated_at=datetime.now(timezone.utc),
            )
        )
        return coordinator

    @pytest.fixture
    def mock_api_client(self):
        """Create a mock API client."""
        return AsyncMock()

    @pytest.fixture
    def dummy_handler(self):
        """Create a dummy message handler."""

        async def handler(msg, tools):
            pass

        return handler

    @pytest.mark.asyncio
    async def test_fetches_and_returns_history_when_enabled(
        self, mock_api_client, mock_coordinator, dummy_handler
    ):
        """Should fetch and return history when enable_context_hydration is True."""
        config = SessionConfig(enable_context_hydration=True)
        session = AgentSession(
            room_id="room-123",
            api_client=mock_api_client,
            on_message=dummy_handler,
            coordinator=mock_coordinator,
            config=config,
        )

        history = await session.get_history_for_llm()

        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello"
        assert history[1]["role"] == "assistant"
        assert history[1]["content"] == "Hi there!"
        # Should have called _fetch_context
        mock_coordinator._fetch_context.assert_called_once()


class TestProcessMessageHydration:
    """Test _process_message() hydration behavior."""

    @pytest.fixture
    def mock_coordinator(self):
        """Create a mock ThenvoiAgent coordinator."""
        from thenvoi.core.types import ConversationContext, AgentTools

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

        # Mock _create_agent_tools to return a mock AgentTools
        mock_tools = MagicMock(spec=AgentTools)
        coordinator._create_agent_tools = MagicMock(return_value=mock_tools)

        return coordinator

    @pytest.fixture
    def mock_api_client(self):
        """Create a mock API client."""
        return AsyncMock()

    @pytest.fixture
    def sample_message(self):
        """Create a sample platform message."""
        return PlatformMessage(
            id="msg-123",
            room_id="room-123",
            content="Hello",
            sender_id="user-456",
            sender_type="User",
            sender_name="Alice",
            message_type="text",
            metadata={},
            created_at=datetime.now(timezone.utc),
        )

    @pytest.mark.asyncio
    async def test_skips_hydration_when_disabled(
        self, mock_api_client, mock_coordinator, sample_message
    ):
        """Should NOT call _hydrate_context when enable_context_hydration is False."""
        handler_called = False

        async def track_handler(msg, tools):
            nonlocal handler_called
            handler_called = True

        config = SessionConfig(enable_context_hydration=False)
        session = AgentSession(
            room_id="room-123",
            api_client=mock_api_client,
            on_message=track_handler,
            coordinator=mock_coordinator,
            config=config,
        )

        await session._process_message(sample_message)

        assert handler_called
        # Should NOT have fetched context
        mock_coordinator._fetch_context.assert_not_called()

    @pytest.mark.asyncio
    async def test_hydrates_context_when_enabled(
        self, mock_api_client, mock_coordinator, sample_message
    ):
        """Should call _hydrate_context when enable_context_hydration is True."""
        handler_called = False

        async def track_handler(msg, tools):
            nonlocal handler_called
            handler_called = True

        config = SessionConfig(enable_context_hydration=True)
        session = AgentSession(
            room_id="room-123",
            api_client=mock_api_client,
            on_message=track_handler,
            coordinator=mock_coordinator,
            config=config,
        )

        await session._process_message(sample_message)

        assert handler_called
        # Should have fetched context
        mock_coordinator._fetch_context.assert_called_once()
