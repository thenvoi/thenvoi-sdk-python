"""
Unit tests for BaseFrameworkAgent.

Tests the common behavior shared by all framework agents using
the new runtime layer (ThenvoiLink, AgentRuntime, ExecutionContext).

Tests:
1. Lifecycle methods (start, stop, run)
2. First message detection + history loading
3. Participant change detection
4. Session cleanup callback wiring
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any

from thenvoi.agents import BaseFrameworkAgent
from thenvoi.runtime.execution import ExecutionContext
from thenvoi.runtime.tools import AgentTools
from thenvoi.runtime.types import PlatformMessage

# Import test helpers from conftest
from tests.conftest import make_message_event, make_participant_added_event


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
        ctx: ExecutionContext,
        history: list[dict[str, Any]] | None,
        participants_msg: str | None,
    ) -> None:
        """Capture calls for test verification."""
        self.handle_message_calls.append(
            {
                "msg": msg,
                "tools": tools,
                "ctx": ctx,
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


@pytest.fixture
def mock_link():
    """Create mock ThenvoiLink."""
    link = MagicMock()
    link.agent_id = "agent-123"
    link.disconnect = AsyncMock()
    link.run_forever = AsyncMock()

    # REST client mock
    link.rest = MagicMock()
    link.rest.agent_api = MagicMock()

    # Mock get_agent_me
    agent_me = MagicMock()
    agent_me.name = "TestBot"
    agent_me.description = "A test bot"
    link.rest.agent_api.get_agent_me = AsyncMock(return_value=MagicMock(data=agent_me))

    return link


@pytest.fixture
def mock_runtime():
    """Create mock AgentRuntime."""
    runtime = MagicMock()
    runtime.start = AsyncMock()
    runtime.stop = AsyncMock()
    return runtime


@pytest.fixture
def mock_ctx():
    """Create mock ExecutionContext."""
    ctx = MagicMock(spec=ExecutionContext)
    ctx.room_id = "room-123"
    ctx.is_llm_initialized = False
    ctx.participants = [{"id": "user-1", "name": "User", "type": "User"}]
    ctx.participants_changed = MagicMock(return_value=False)
    ctx.mark_llm_initialized = MagicMock()
    ctx.mark_participants_sent = MagicMock()
    ctx.get_context = AsyncMock(return_value=MagicMock(messages=[], participants=[]))
    # Mock link.rest for AgentTools.from_context()
    ctx.link = MagicMock()
    ctx.link.rest = MagicMock()
    return ctx


class TestLifecycle:
    """Tests for start/stop/run lifecycle."""

    @patch("thenvoi.agents.base.ThenvoiLink")
    @patch("thenvoi.agents.base.AgentRuntime")
    async def test_start_creates_link_and_runtime(
        self, mock_runtime_cls, mock_link_cls, mock_link, mock_runtime
    ):
        """start() should create ThenvoiLink and AgentRuntime."""
        mock_link_cls.return_value = mock_link
        mock_runtime_cls.return_value = mock_runtime

        agent = ConcreteAgent(agent_id="agent-123", api_key="key")
        await agent.start()

        mock_link_cls.assert_called_once()
        mock_runtime_cls.assert_called_once()
        mock_runtime.start.assert_called_once()

    @patch("thenvoi.agents.base.ThenvoiLink")
    @patch("thenvoi.agents.base.AgentRuntime")
    async def test_start_calls_on_started(
        self, mock_runtime_cls, mock_link_cls, mock_link, mock_runtime
    ):
        """start() should call _on_started hook."""
        mock_link_cls.return_value = mock_link
        mock_runtime_cls.return_value = mock_runtime

        agent = ConcreteAgent(agent_id="agent-123", api_key="key")
        await agent.start()

        assert agent.on_started_called is True

    @patch("thenvoi.agents.base.ThenvoiLink")
    @patch("thenvoi.agents.base.AgentRuntime")
    async def test_stop_stops_runtime_and_disconnects(
        self, mock_runtime_cls, mock_link_cls, mock_link, mock_runtime
    ):
        """stop() should stop runtime and disconnect link."""
        mock_link_cls.return_value = mock_link
        mock_runtime_cls.return_value = mock_runtime

        agent = ConcreteAgent(agent_id="agent-123", api_key="key")
        await agent.start()
        await agent.stop()

        mock_runtime.stop.assert_called_once()
        mock_link.disconnect.assert_called_once()


class TestAgentMetadata:
    """Tests for agent metadata fetching."""

    @patch("thenvoi.agents.base.ThenvoiLink")
    @patch("thenvoi.agents.base.AgentRuntime")
    async def test_fetches_agent_name(
        self, mock_runtime_cls, mock_link_cls, mock_link, mock_runtime
    ):
        """start() should fetch and store agent name."""
        mock_link_cls.return_value = mock_link
        mock_runtime_cls.return_value = mock_runtime

        agent = ConcreteAgent(agent_id="agent-123", api_key="key")
        await agent.start()

        assert agent.agent_name == "TestBot"
        assert agent.agent_description == "A test bot"


class TestDispatchMessage:
    """Tests for _dispatch_message pre-processing."""

    @patch("thenvoi.agents.base.ThenvoiLink")
    @patch("thenvoi.agents.base.AgentRuntime")
    async def test_first_message_loads_history(
        self, mock_runtime_cls, mock_link_cls, mock_link, mock_runtime, mock_ctx
    ):
        """First message should load history and mark initialized."""
        mock_link_cls.return_value = mock_link
        mock_runtime_cls.return_value = mock_runtime

        # Setup context with history
        mock_ctx.is_llm_initialized = False
        mock_ctx.get_context = AsyncMock(
            return_value=MagicMock(
                messages=[
                    {
                        "id": "msg-1",
                        "content": "Previous",
                        "sender_type": "User",
                        "sender_name": "User",
                    }
                ],
                participants=[],
            )
        )

        agent = ConcreteAgent(agent_id="agent-123", api_key="key")
        await agent.start()

        # Create message event
        event = make_message_event(
            room_id="room-123",
            msg_id="msg-2",
            content="Hello",
            sender_id="user-1",
            sender_type="User",
        )

        await agent._dispatch_message(mock_ctx, event)

        # Should have loaded history
        mock_ctx.get_context.assert_called_once()
        mock_ctx.mark_llm_initialized.assert_called_once()

        # Should have called handler with history
        assert len(agent.handle_message_calls) == 1
        call = agent.handle_message_calls[0]
        assert call["history"] is not None

    @patch("thenvoi.agents.base.ThenvoiLink")
    @patch("thenvoi.agents.base.AgentRuntime")
    async def test_subsequent_message_skips_history(
        self, mock_runtime_cls, mock_link_cls, mock_link, mock_runtime, mock_ctx
    ):
        """Subsequent messages should not reload history."""
        mock_link_cls.return_value = mock_link
        mock_runtime_cls.return_value = mock_runtime

        mock_ctx.is_llm_initialized = True  # Already initialized

        agent = ConcreteAgent(agent_id="agent-123", api_key="key")
        await agent.start()

        event = make_message_event(
            room_id="room-123",
            msg_id="msg-2",
            content="Hello",
            sender_id="user-1",
            sender_type="User",
        )

        await agent._dispatch_message(mock_ctx, event)

        # Should NOT have loaded history
        mock_ctx.get_context.assert_not_called()
        mock_ctx.mark_llm_initialized.assert_not_called()

        # History should be None
        call = agent.handle_message_calls[0]
        assert call["history"] is None

    @patch("thenvoi.agents.base.ThenvoiLink")
    @patch("thenvoi.agents.base.AgentRuntime")
    async def test_ignores_non_message_events(
        self, mock_runtime_cls, mock_link_cls, mock_link, mock_runtime, mock_ctx
    ):
        """Non-message events should be ignored."""
        mock_link_cls.return_value = mock_link
        mock_runtime_cls.return_value = mock_runtime

        agent = ConcreteAgent(agent_id="agent-123", api_key="key")
        await agent.start()

        event = make_participant_added_event(
            room_id="room-123",
            participant_id="user-2",
            name="User Two",
        )

        await agent._dispatch_message(mock_ctx, event)

        # Should not have called handler
        assert len(agent.handle_message_calls) == 0


class TestParticipantChanges:
    """Tests for participant change detection."""

    @patch("thenvoi.agents.base.ThenvoiLink")
    @patch("thenvoi.agents.base.AgentRuntime")
    async def test_sends_participant_message_when_changed(
        self, mock_runtime_cls, mock_link_cls, mock_link, mock_runtime, mock_ctx
    ):
        """Should pass participants message when participants changed."""
        mock_link_cls.return_value = mock_link
        mock_runtime_cls.return_value = mock_runtime

        mock_ctx.is_llm_initialized = True
        mock_ctx.participants_changed = MagicMock(return_value=True)

        agent = ConcreteAgent(agent_id="agent-123", api_key="key")
        await agent.start()

        event = make_message_event(
            room_id="room-123",
            msg_id="msg-1",
            content="Hello",
            sender_id="user-1",
            sender_type="User",
        )

        await agent._dispatch_message(mock_ctx, event)

        # Should have marked participants sent
        mock_ctx.mark_participants_sent.assert_called_once()

        # Should have passed participants message
        call = agent.handle_message_calls[0]
        assert call["participants_msg"] is not None


class TestReportError:
    """Tests for _report_error helper."""

    @patch("thenvoi.agents.base.ThenvoiLink")
    @patch("thenvoi.agents.base.AgentRuntime")
    async def test_sends_error_event(
        self, mock_runtime_cls, mock_link_cls, mock_link, mock_runtime
    ):
        """Should send error event via tools."""
        mock_link_cls.return_value = mock_link
        mock_runtime_cls.return_value = mock_runtime

        agent = ConcreteAgent(agent_id="agent-123", api_key="key")

        mock_tools = MagicMock()
        mock_tools.send_event = AsyncMock()

        await agent._report_error(mock_tools, "Something failed")

        mock_tools.send_event.assert_called_once_with(
            content="Error: Something failed",
            message_type="error",
        )

    @patch("thenvoi.agents.base.ThenvoiLink")
    @patch("thenvoi.agents.base.AgentRuntime")
    async def test_swallows_exception_on_send_failure(
        self, mock_runtime_cls, mock_link_cls, mock_link, mock_runtime
    ):
        """Should not raise if send_event fails."""
        mock_link_cls.return_value = mock_link
        mock_runtime_cls.return_value = mock_runtime

        agent = ConcreteAgent(agent_id="agent-123", api_key="key")

        mock_tools = MagicMock()
        mock_tools.send_event = AsyncMock(side_effect=Exception("Send failed"))

        # Should not raise
        await agent._report_error(mock_tools, "Something failed")
