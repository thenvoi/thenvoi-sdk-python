"""
Unit tests for BaseFrameworkAgent.

Tests the common behavior shared by all framework agents:
1. First message detection + history loading
2. Participant change detection
3. Session cleanup callback wiring
4. Lifecycle methods (start, stop, run)
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone
from typing import Any

from thenvoi.agents import BaseFrameworkAgent
from thenvoi.core.types import PlatformMessage, AgentTools
from thenvoi.core.session import AgentSession


class ConcreteAgent(BaseFrameworkAgent):
    """Concrete implementation for testing."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.handle_message_calls = []
        self.cleanup_calls = []
        self.on_started_called = False

    async def _handle_message(
        self,
        msg: PlatformMessage,
        tools: AgentTools,
        session: AgentSession,
        history: list[dict[str, Any]] | None,
        participants_msg: str | None,
    ) -> None:
        """Capture calls for test verification."""
        self.handle_message_calls.append(
            {
                "msg": msg,
                "tools": tools,
                "session": session,
                "history": history,
                "participants_msg": participants_msg,
            }
        )

    async def _cleanup_session(self, room_id: str) -> None:
        """Capture cleanup calls."""
        self.cleanup_calls.append(room_id)

    async def _on_started(self) -> None:
        """Track if called."""
        self.on_started_called = True


class TestDispatchMessage:
    """Tests for _dispatch_message pre-processing."""

    @pytest.fixture
    def agent(self):
        """Create agent with mocked ThenvoiAgent."""
        with patch("thenvoi.agents.base.ThenvoiAgent") as mock_cls:
            mock_thenvoi = AsyncMock()
            mock_thenvoi.active_sessions = {}
            mock_cls.return_value = mock_thenvoi

            return ConcreteAgent(agent_id="test", api_key="key")

    @pytest.fixture
    def mock_session(self):
        """Create mock session."""
        session = MagicMock(spec=AgentSession)
        session.room_id = "room-123"
        session.is_llm_initialized = False
        session.participants = [{"id": "user-1", "name": "User", "type": "User"}]
        session.participants_changed = MagicMock(return_value=False)
        session.mark_llm_initialized = MagicMock()
        session.mark_participants_sent = MagicMock()
        session.build_participants_message = MagicMock(
            return_value="## Participants\n- User"
        )
        session.get_history_for_llm = AsyncMock(return_value=[])
        return session

    @pytest.fixture
    def sample_message(self):
        """Sample platform message."""
        return PlatformMessage(
            id="msg-1",
            room_id="room-123",
            content="Hello",
            sender_id="user-1",
            sender_type="User",
            sender_name="User",
            message_type="text",
            metadata={},
            created_at=datetime.now(timezone.utc),
        )

    @pytest.fixture
    def mock_tools(self):
        """Mock AgentTools."""
        return MagicMock(spec=AgentTools)

    async def test_first_message_loads_history(
        self, agent, mock_session, sample_message, mock_tools
    ):
        """First message should load history and mark initialized."""
        mock_session.is_llm_initialized = False
        mock_session.get_history_for_llm = AsyncMock(
            return_value=[
                {"role": "user", "content": "Previous", "sender_name": "User"},
            ]
        )
        agent.thenvoi.active_sessions = {"room-123": mock_session}

        await agent._dispatch_message(sample_message, mock_tools)

        # Should have called get_history_for_llm
        mock_session.get_history_for_llm.assert_called_once_with(
            exclude_message_id="msg-1"
        )

        # Should have marked as initialized
        mock_session.mark_llm_initialized.assert_called_once()

        # Should have passed history to handler
        assert len(agent.handle_message_calls) == 1
        call = agent.handle_message_calls[0]
        assert call["history"] == [
            {"role": "user", "content": "Previous", "sender_name": "User"}
        ]

    async def test_subsequent_message_skips_history(
        self, agent, mock_session, sample_message, mock_tools
    ):
        """Subsequent messages should not reload history."""
        mock_session.is_llm_initialized = True  # Already initialized
        agent.thenvoi.active_sessions = {"room-123": mock_session}

        await agent._dispatch_message(sample_message, mock_tools)

        # Should NOT call get_history_for_llm
        mock_session.get_history_for_llm.assert_not_called()

        # Should NOT mark as initialized again
        mock_session.mark_llm_initialized.assert_not_called()

        # History should be None
        assert len(agent.handle_message_calls) == 1
        call = agent.handle_message_calls[0]
        assert call["history"] is None

    async def test_participants_changed_sends_message(
        self, agent, mock_session, sample_message, mock_tools
    ):
        """Should pass participants message when participants changed."""
        mock_session.is_llm_initialized = True
        mock_session.participants_changed = MagicMock(return_value=True)
        agent.thenvoi.active_sessions = {"room-123": mock_session}

        await agent._dispatch_message(sample_message, mock_tools)

        # Should have marked participants as sent
        mock_session.mark_participants_sent.assert_called_once()

        # Should have passed participants message
        call = agent.handle_message_calls[0]
        assert call["participants_msg"] is not None
        assert "Participants" in call["participants_msg"]

    async def test_no_participant_message_when_unchanged(
        self, agent, mock_session, sample_message, mock_tools
    ):
        """Should not pass participants message when unchanged."""
        mock_session.is_llm_initialized = True
        mock_session.participants_changed = MagicMock(return_value=False)
        agent.thenvoi.active_sessions = {"room-123": mock_session}

        await agent._dispatch_message(sample_message, mock_tools)

        # Should NOT mark participants as sent
        mock_session.mark_participants_sent.assert_not_called()

        # participants_msg should be None
        call = agent.handle_message_calls[0]
        assert call["participants_msg"] is None

    async def test_no_session_returns_early(self, agent, sample_message, mock_tools):
        """Should return early if no session found."""
        agent.thenvoi.active_sessions = {}  # No session

        await agent._dispatch_message(sample_message, mock_tools)

        # Should not have called handler
        assert len(agent.handle_message_calls) == 0

    async def test_history_load_failure_returns_empty(
        self, agent, mock_session, sample_message, mock_tools
    ):
        """History load failure should return empty list, not fail."""
        mock_session.is_llm_initialized = False
        mock_session.get_history_for_llm = AsyncMock(side_effect=Exception("API error"))
        agent.thenvoi.active_sessions = {"room-123": mock_session}

        await agent._dispatch_message(sample_message, mock_tools)

        # Should still call handler with empty history
        call = agent.handle_message_calls[0]
        assert call["history"] == []


class TestLifecycle:
    """Tests for start/stop/run lifecycle."""

    @pytest.fixture
    def agent(self):
        """Create agent with mocked ThenvoiAgent."""
        with patch("thenvoi.agents.base.ThenvoiAgent") as mock_cls:
            mock_thenvoi = AsyncMock()
            mock_thenvoi.active_sessions = {}
            mock_thenvoi.start = AsyncMock()
            mock_thenvoi.stop = AsyncMock()
            mock_thenvoi.run = AsyncMock()
            mock_cls.return_value = mock_thenvoi

            return ConcreteAgent(agent_id="test", api_key="key")

    async def test_start_wires_cleanup_callback(self, agent):
        """start() should wire _cleanup_session as callback."""
        await agent.start()

        # Cleanup callback should be wired
        assert agent.thenvoi._on_session_cleanup == agent._cleanup_session

        # Should have called thenvoi.start
        agent.thenvoi.start.assert_called_once()

    async def test_start_calls_on_started(self, agent):
        """start() should call _on_started hook."""
        await agent.start()

        assert agent.on_started_called is True

    async def test_stop_calls_thenvoi_stop(self, agent):
        """stop() should call thenvoi.stop()."""
        await agent.stop()

        agent.thenvoi.stop.assert_called_once()

    async def test_cleanup_session_called_with_room_id(self, agent):
        """_cleanup_session should receive room_id."""
        await agent._cleanup_session("room-456")

        assert agent.cleanup_calls == ["room-456"]


class TestProperties:
    """Tests for agent properties."""

    def test_agent_name_delegates_to_thenvoi(self):
        """agent_name should delegate to thenvoi."""
        with patch("thenvoi.agents.base.ThenvoiAgent") as mock_cls:
            mock_thenvoi = MagicMock()
            mock_thenvoi.agent_name = "TestBot"
            mock_cls.return_value = mock_thenvoi

            agent = ConcreteAgent(agent_id="test", api_key="key")

            assert agent.agent_name == "TestBot"

    def test_agent_description_delegates_to_thenvoi(self):
        """agent_description should delegate to thenvoi."""
        with patch("thenvoi.agents.base.ThenvoiAgent") as mock_cls:
            mock_thenvoi = MagicMock()
            mock_thenvoi.agent_description = "A test bot"
            mock_cls.return_value = mock_thenvoi

            agent = ConcreteAgent(agent_id="test", api_key="key")

            assert agent.agent_description == "A test bot"


class TestReportError:
    """Tests for _report_error helper."""

    async def test_sends_error_event(self):
        """Should send error event via tools."""
        with patch("thenvoi.agents.base.ThenvoiAgent"):
            agent = ConcreteAgent(agent_id="test", api_key="key")

        mock_tools = MagicMock()
        mock_tools.send_event = AsyncMock()

        await agent._report_error(mock_tools, "Something failed")

        mock_tools.send_event.assert_called_once_with(
            content="Error: Something failed",
            message_type="error",
        )

    async def test_swallows_exception_on_send_failure(self):
        """Should not raise if send_event fails."""
        with patch("thenvoi.agents.base.ThenvoiAgent"):
            agent = ConcreteAgent(agent_id="test", api_key="key")

        mock_tools = MagicMock()
        mock_tools.send_event = AsyncMock(side_effect=Exception("Send failed"))

        # Should not raise
        await agent._report_error(mock_tools, "Something failed")
