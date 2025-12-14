"""
Unit tests for LangGraphAdapter.

Tests:
1. Constructor validation (requires graph or graph_factory)
2. Message building in _handle_message (system prompt, history, participants)
3. Stream event handling
4. Session cleanup callback
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from thenvoi.agent.langgraph.adapter import LangGraphAdapter
from thenvoi.agent.core.types import PlatformMessage
from datetime import datetime, timezone


class TestConstructor:
    """Tests for LangGraphAdapter initialization."""

    def test_requires_graph_or_graph_factory(self):
        """Should raise ValueError if neither graph nor graph_factory provided."""
        with pytest.raises(ValueError) as exc_info:
            LangGraphAdapter(
                agent_id="agent-123",
                api_key="test-key",
                # Neither graph nor graph_factory
            )

        assert "Must provide either graph_factory or graph" in str(exc_info.value)

    def test_accepts_graph_factory(self):
        """Should accept graph_factory parameter."""
        with patch("thenvoi.agent.langgraph.adapter.ThenvoiAgent"):
            adapter = LangGraphAdapter(
                graph_factory=lambda tools: MagicMock(),
                agent_id="agent-123",
                api_key="test-key",
            )

        assert adapter.graph_factory is not None

    def test_accepts_static_graph(self):
        """Should accept pre-built graph."""
        with patch("thenvoi.agent.langgraph.adapter.ThenvoiAgent"):
            adapter = LangGraphAdapter(
                graph=MagicMock(),
                agent_id="agent-123",
                api_key="test-key",
            )

        assert adapter._static_graph is not None


class TestHandleMessage:
    """Tests for _handle_message message building logic."""

    @pytest.fixture
    def mock_graph(self):
        """Mock LangGraph that captures inputs."""
        graph = MagicMock()
        graph.astream_events = AsyncMock(return_value=AsyncIterator([]))
        return graph

    @pytest.fixture
    def adapter(self, mock_graph):
        """Create adapter with mocked dependencies."""
        with patch("thenvoi.agent.langgraph.adapter.ThenvoiAgent") as mock_thenvoi_cls:
            mock_thenvoi = AsyncMock()
            mock_thenvoi.agent_name = "TestBot"
            mock_thenvoi.agent_description = "A test agent"
            mock_thenvoi.active_sessions = {}
            mock_thenvoi_cls.return_value = mock_thenvoi

            adapter = LangGraphAdapter(
                graph_factory=lambda tools: mock_graph,
                agent_id="agent-123",
                api_key="test-key",
            )
            adapter._system_prompt = "You are TestBot, A test agent."
            adapter._mock_graph = mock_graph
            return adapter

    @pytest.fixture
    def mock_session(self):
        """Create a mock session."""
        session = MagicMock()
        session.is_llm_initialized = False
        session.participants = [{"id": "user-456", "name": "Test User", "type": "User"}]
        session.participants_changed = MagicMock(return_value=True)
        session.mark_llm_initialized = MagicMock()
        session.mark_participants_sent = MagicMock()
        session.build_participants_message = MagicMock(
            return_value="## Participants\n- Test User"
        )
        session.get_history_for_llm = AsyncMock(return_value=[])
        return session

    @pytest.fixture
    def sample_message(self):
        """Sample incoming message."""
        return PlatformMessage(
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

    @pytest.fixture
    def mock_tools(self):
        """Mock AgentTools."""
        tools = MagicMock()
        tools.to_langchain_tools = MagicMock(return_value=[])
        tools.send_event = AsyncMock()
        return tools

    async def test_first_message_includes_system_prompt(
        self, adapter, mock_session, sample_message, mock_tools
    ):
        """First message should include system prompt."""
        mock_session.is_llm_initialized = False
        adapter.thenvoi.active_sessions = {"room-123": mock_session}

        captured_input = {}

        async def capture_input(input_data, **kwargs):
            captured_input.update(input_data)
            return
            yield  # Make it an async generator

        adapter._mock_graph.astream_events = capture_input

        await adapter._handle_message(sample_message, mock_tools)

        # Should have marked as initialized
        mock_session.mark_llm_initialized.assert_called_once()

        # First message in list should be system prompt
        messages = captured_input.get("messages", [])
        assert len(messages) > 0
        assert messages[0][0] == "system"
        assert "TestBot" in messages[0][1] or adapter._system_prompt in messages[0][1]

    async def test_subsequent_message_skips_system_prompt(
        self, adapter, mock_session, sample_message, mock_tools
    ):
        """Subsequent messages should NOT include system prompt."""
        mock_session.is_llm_initialized = True  # Already initialized
        mock_session.participants_changed = MagicMock(return_value=False)
        adapter.thenvoi.active_sessions = {"room-123": mock_session}

        captured_input = {}

        async def capture_input(input_data, **kwargs):
            captured_input.update(input_data)
            return
            yield

        adapter._mock_graph.astream_events = capture_input

        await adapter._handle_message(sample_message, mock_tools)

        # Should NOT mark as initialized again
        mock_session.mark_llm_initialized.assert_not_called()

        # Messages should only contain the user message
        messages = captured_input.get("messages", [])
        assert len(messages) == 1
        assert messages[0][0] == "user"

    async def test_injects_participants_when_changed(
        self, adapter, mock_session, sample_message, mock_tools
    ):
        """Should inject participants message when participants changed."""
        mock_session.is_llm_initialized = True
        mock_session.participants_changed = MagicMock(return_value=True)
        adapter.thenvoi.active_sessions = {"room-123": mock_session}

        captured_input = {}

        async def capture_input(input_data, **kwargs):
            captured_input.update(input_data)
            return
            yield

        adapter._mock_graph.astream_events = capture_input

        await adapter._handle_message(sample_message, mock_tools)

        mock_session.mark_participants_sent.assert_called_once()

        # Should have participants message
        messages = captured_input.get("messages", [])
        system_messages = [m for m in messages if m[0] == "system"]
        assert len(system_messages) == 1
        assert "Participants" in system_messages[0][1]

    async def test_loads_history_on_first_message(
        self, adapter, mock_session, sample_message, mock_tools
    ):
        """First message should load and inject history."""
        mock_session.is_llm_initialized = False
        mock_session.get_history_for_llm = AsyncMock(
            return_value=[
                {"role": "user", "content": "Previous message", "sender_name": "User"},
                {
                    "role": "assistant",
                    "content": "Previous response",
                    "sender_name": "Bot",
                },
            ]
        )
        adapter.thenvoi.active_sessions = {"room-123": mock_session}

        captured_input = {}

        async def capture_input(input_data, **kwargs):
            captured_input.update(input_data)
            return
            yield

        adapter._mock_graph.astream_events = capture_input

        await adapter._handle_message(sample_message, mock_tools)

        # Should have called get_history_for_llm
        mock_session.get_history_for_llm.assert_called_once_with(
            exclude_message_id="msg-123"
        )

        # Messages should include history
        messages = captured_input.get("messages", [])
        # Should have: system prompt, history (2), participants, current message
        assert len(messages) >= 3


class TestHandleStreamEvent:
    """Tests for _handle_stream_event."""

    @pytest.fixture
    def adapter(self):
        """Create adapter with mocked dependencies."""
        with patch("thenvoi.agent.langgraph.adapter.ThenvoiAgent"):
            return LangGraphAdapter(
                graph_factory=lambda tools: MagicMock(),
                agent_id="agent-123",
                api_key="test-key",
            )

    @pytest.fixture
    def mock_tools(self):
        """Mock AgentTools."""
        tools = MagicMock()
        tools.send_event = AsyncMock()
        return tools

    async def test_sends_tool_call_event_on_tool_start(self, adapter, mock_tools):
        """on_tool_start should send tool_call event."""
        event = {
            "event": "on_tool_start",
            "name": "send_message",
            "data": {"input": {"content": "Hello"}},
        }

        await adapter._handle_stream_event(event, "room-123", mock_tools)

        mock_tools.send_event.assert_called_once()
        call_args = mock_tools.send_event.call_args
        assert call_args.kwargs["message_type"] == "tool_call"

    async def test_sends_tool_result_event_on_tool_end(self, adapter, mock_tools):
        """on_tool_end should send tool_result event."""
        event = {
            "event": "on_tool_end",
            "name": "send_message",
            "data": {"output": "Success"},
        }

        await adapter._handle_stream_event(event, "room-123", mock_tools)

        mock_tools.send_event.assert_called_once()
        call_args = mock_tools.send_event.call_args
        assert call_args.kwargs["message_type"] == "tool_result"

    async def test_ignores_other_events(self, adapter, mock_tools):
        """Other events should be ignored."""
        event = {
            "event": "on_chat_model_stream",
            "data": {"chunk": "streaming..."},
        }

        await adapter._handle_stream_event(event, "room-123", mock_tools)

        mock_tools.send_event.assert_not_called()


class TestCleanupSession:
    """Tests for _cleanup_session checkpointer clearing."""

    async def test_calls_checkpointer_adelete_thread(self):
        """Should call checkpointer.adelete_thread with room_id."""
        mock_checkpointer = AsyncMock()
        mock_checkpointer.adelete_thread = AsyncMock()

        def graph_factory(tools):
            graph = MagicMock()
            return graph

        graph_factory.checkpointer = mock_checkpointer

        with patch("thenvoi.agent.langgraph.adapter.ThenvoiAgent"):
            adapter = LangGraphAdapter(
                graph_factory=graph_factory,
                agent_id="agent-123",
                api_key="test-key",
            )

        await adapter._cleanup_session("room-123")

        mock_checkpointer.adelete_thread.assert_called_once_with("room-123")

    async def test_handles_missing_checkpointer(self):
        """Should handle graph_factory without checkpointer."""
        with patch("thenvoi.agent.langgraph.adapter.ThenvoiAgent"):
            adapter = LangGraphAdapter(
                graph_factory=lambda tools: MagicMock(),  # No checkpointer attribute
                agent_id="agent-123",
                api_key="test-key",
            )

        # Should not raise
        await adapter._cleanup_session("room-123")


# Helper for async iteration
class AsyncIterator:
    """Helper to create async iterator from list."""

    def __init__(self, items):
        self.items = items
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.items):
            raise StopAsyncIteration
        item = self.items[self.index]
        self.index += 1
        return item
