"""
Unit tests for ThenvoiLangGraphAgent (example agent).

Tests:
1. Constructor validation (requires graph or graph_factory)
2. Message building in _handle_message (system prompt, history, participants)
3. Stream event handling
4. Session cleanup callback
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from thenvoi.integrations.langgraph import ThenvoiLangGraphAgent
from thenvoi.runtime.types import PlatformMessage
from thenvoi.runtime.execution import ExecutionContext


class TestConstructor:
    """Tests for ThenvoiLangGraphAgent initialization."""

    def test_requires_graph_or_graph_factory(self):
        """Should raise ValueError if neither graph nor graph_factory provided."""
        with pytest.raises(ValueError) as exc_info:
            ThenvoiLangGraphAgent(
                agent_id="agent-123",
                api_key="test-key",
                # Neither graph nor graph_factory
            )

        assert "Must provide either graph_factory or graph" in str(exc_info.value)

    @patch("thenvoi.agents.base.ThenvoiLink")
    @patch("thenvoi.agents.base.AgentRuntime")
    def test_accepts_graph_factory(self, mock_runtime_cls, mock_link_cls):
        """Should accept graph_factory parameter."""
        adapter = ThenvoiLangGraphAgent(
            graph_factory=lambda tools: MagicMock(),
            agent_id="agent-123",
            api_key="test-key",
        )

        assert adapter.graph_factory is not None

    @patch("thenvoi.agents.base.ThenvoiLink")
    @patch("thenvoi.agents.base.AgentRuntime")
    def test_accepts_static_graph(self, mock_runtime_cls, mock_link_cls):
        """Should accept pre-built graph."""
        adapter = ThenvoiLangGraphAgent(
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
    def mock_ctx(self):
        """Create mock ExecutionContext."""
        ctx = MagicMock(spec=ExecutionContext)
        ctx.room_id = "room-123"
        ctx.is_llm_initialized = False
        ctx.participants = [{"id": "user-456", "name": "Test User", "type": "User"}]
        ctx.participants_changed = MagicMock(return_value=True)
        ctx.mark_llm_initialized = MagicMock()
        ctx.mark_participants_sent = MagicMock()
        ctx.get_context = AsyncMock(
            return_value=MagicMock(messages=[], participants=[])
        )
        ctx.link = MagicMock()
        ctx.link.rest = MagicMock()
        return ctx

    @pytest.fixture
    def adapter(self, mock_graph):
        """Create adapter with mocked dependencies."""
        with patch("thenvoi.agents.base.ThenvoiLink") as mock_link_cls:
            with patch("thenvoi.agents.base.AgentRuntime") as mock_runtime_cls:
                mock_link = MagicMock()
                mock_link.agent_id = "agent-123"
                mock_link.rest = MagicMock()
                mock_link.rest.agent_api = MagicMock()
                agent_me = MagicMock()
                agent_me.name = "TestBot"
                agent_me.description = "A test agent"
                mock_link.rest.agent_api.get_agent_me = AsyncMock(
                    return_value=MagicMock(data=agent_me)
                )
                mock_link_cls.return_value = mock_link

                mock_runtime = MagicMock()
                mock_runtime.start = AsyncMock()
                mock_runtime_cls.return_value = mock_runtime

                adapter = ThenvoiLangGraphAgent(
                    graph_factory=lambda tools: mock_graph,
                    agent_id="agent-123",
                    api_key="test-key",
                )
                adapter._system_prompt = "You are TestBot, A test agent."
                adapter._mock_graph = mock_graph
                return adapter

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
        self, adapter, mock_ctx, sample_message, mock_tools
    ):
        """First message should include system prompt."""
        mock_ctx.is_llm_initialized = False

        captured_input = {}

        async def capture_input(input_data, **kwargs):
            captured_input.update(input_data)
            return
            yield  # Make it an async generator

        adapter._mock_graph.astream_events = capture_input

        # Convert history format
        history = [{"role": "user", "content": "Previous", "sender_name": "User"}]
        await adapter._handle_message(
            sample_message, mock_tools, mock_ctx, history, None
        )

        # First message in list should be system prompt
        messages = captured_input.get("messages", [])
        assert len(messages) > 0
        assert messages[0][0] == "system"

    async def test_subsequent_message_skips_system_prompt(
        self, adapter, mock_ctx, sample_message, mock_tools
    ):
        """Subsequent messages should NOT include system prompt."""
        mock_ctx.is_llm_initialized = True  # Already initialized
        mock_ctx.participants_changed = MagicMock(return_value=False)

        captured_input = {}

        async def capture_input(input_data, **kwargs):
            captured_input.update(input_data)
            return
            yield

        adapter._mock_graph.astream_events = capture_input

        await adapter._handle_message(
            sample_message,
            mock_tools,
            mock_ctx,
            None,
            None,  # No history
        )

        # Messages should only contain the user message
        messages = captured_input.get("messages", [])
        assert len(messages) == 1
        assert messages[0][0] == "user"

    async def test_injects_participants_when_provided(
        self, adapter, mock_ctx, sample_message, mock_tools
    ):
        """Should inject participants message when participants_msg provided."""
        mock_ctx.is_llm_initialized = True

        captured_input = {}

        async def capture_input(input_data, **kwargs):
            captured_input.update(input_data)
            return
            yield

        adapter._mock_graph.astream_events = capture_input

        participants_msg = "## Participants\n- Test User"
        await adapter._handle_message(
            sample_message, mock_tools, mock_ctx, None, participants_msg
        )

        # Should have participants message
        messages = captured_input.get("messages", [])
        system_messages = [m for m in messages if m[0] == "system"]
        assert len(system_messages) == 1
        assert "Participants" in system_messages[0][1]

    async def test_loads_history_on_first_message(
        self, adapter, mock_ctx, sample_message, mock_tools
    ):
        """First message should include history if provided."""
        mock_ctx.is_llm_initialized = False

        captured_input = {}

        async def capture_input(input_data, **kwargs):
            captured_input.update(input_data)
            return
            yield

        adapter._mock_graph.astream_events = capture_input

        history = [
            {"role": "user", "content": "Previous message", "sender_name": "User"},
            {"role": "assistant", "content": "Previous response", "sender_name": "Bot"},
        ]
        await adapter._handle_message(
            sample_message, mock_tools, mock_ctx, history, None
        )

        # Messages should include history
        messages = captured_input.get("messages", [])
        # Should have: system prompt, history (2), current message
        assert len(messages) >= 3


class TestStaticGraphNoSystemPrompt:
    """Tests for static graph (pre-compiled) NOT getting system prompt injected."""

    @pytest.fixture
    def mock_graph(self):
        """Mock LangGraph that captures inputs."""
        graph = MagicMock()
        graph.astream_events = AsyncMock(return_value=AsyncIterator([]))
        return graph

    @pytest.fixture
    def mock_ctx(self):
        """Create mock ExecutionContext."""
        ctx = MagicMock(spec=ExecutionContext)
        ctx.room_id = "room-123"
        ctx.is_llm_initialized = False
        ctx.participants = [{"id": "user-456", "name": "Test User", "type": "User"}]
        ctx.participants_changed = MagicMock(return_value=False)
        ctx.mark_llm_initialized = MagicMock()
        ctx.mark_participants_sent = MagicMock()
        ctx.get_context = AsyncMock(
            return_value=MagicMock(messages=[], participants=[])
        )
        ctx.link = MagicMock()
        ctx.link.rest = MagicMock()
        return ctx

    @pytest.fixture
    def adapter_with_static_graph(self, mock_graph):
        """Create adapter with pre-compiled static graph (no graph_factory)."""
        with patch("thenvoi.agents.base.ThenvoiLink") as mock_link_cls:
            with patch("thenvoi.agents.base.AgentRuntime") as mock_runtime_cls:
                mock_link = MagicMock()
                mock_link.agent_id = "agent-123"
                mock_link.rest = MagicMock()
                mock_link.rest.agent_api = MagicMock()
                agent_me = MagicMock()
                agent_me.name = "TestBot"
                agent_me.description = "A test agent"
                mock_link.rest.agent_api.get_agent_me = AsyncMock(
                    return_value=MagicMock(data=agent_me)
                )
                mock_link_cls.return_value = mock_link

                mock_runtime = MagicMock()
                mock_runtime.start = AsyncMock()
                mock_runtime_cls.return_value = mock_runtime

                adapter = ThenvoiLangGraphAgent(
                    graph=mock_graph,  # Static graph, NOT graph_factory
                    agent_id="agent-123",
                    api_key="test-key",
                )
                adapter._system_prompt = "You are TestBot, A test agent."
                adapter._mock_graph = mock_graph
                return adapter

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

    async def test_static_graph_first_message_skips_system_prompt(
        self, adapter_with_static_graph, mock_ctx, sample_message, mock_tools
    ):
        """Static graph should NOT receive system prompt on first message."""
        mock_ctx.is_llm_initialized = False

        captured_input = {}

        async def capture_input(input_data, **kwargs):
            captured_input.update(input_data)
            return
            yield  # Make it an async generator

        adapter_with_static_graph._mock_graph.astream_events = capture_input

        await adapter_with_static_graph._handle_message(
            sample_message, mock_tools, mock_ctx, None, None
        )

        # Messages should NOT include system prompt (only user message)
        messages = captured_input.get("messages", [])
        system_messages = [m for m in messages if m[0] == "system"]
        assert len(system_messages) == 0, (
            "Static graph should NOT receive system prompt"
        )

        # Should still have the user message
        user_messages = [m for m in messages if m[0] == "user"]
        assert len(user_messages) == 1

    async def test_graph_factory_first_message_includes_system_prompt(
        self, mock_ctx, sample_message, mock_tools
    ):
        """Verify graph_factory DOES receive system prompt (contrast test)."""
        mock_graph = MagicMock()
        mock_graph.astream_events = AsyncMock(return_value=AsyncIterator([]))

        with patch("thenvoi.agents.base.ThenvoiLink") as mock_link_cls:
            with patch("thenvoi.agents.base.AgentRuntime") as mock_runtime_cls:
                mock_link = MagicMock()
                mock_link.agent_id = "agent-123"
                mock_link.rest = MagicMock()
                mock_link.rest.agent_api = MagicMock()
                agent_me = MagicMock()
                agent_me.name = "TestBot"
                agent_me.description = "A test agent"
                mock_link.rest.agent_api.get_agent_me = AsyncMock(
                    return_value=MagicMock(data=agent_me)
                )
                mock_link_cls.return_value = mock_link

                mock_runtime = MagicMock()
                mock_runtime.start = AsyncMock()
                mock_runtime_cls.return_value = mock_runtime

                adapter = ThenvoiLangGraphAgent(
                    graph_factory=lambda tools: mock_graph,  # Factory, NOT static
                    agent_id="agent-123",
                    api_key="test-key",
                )
                adapter._system_prompt = "You are TestBot, A test agent."

        mock_ctx.is_llm_initialized = False
        mock_ctx.participants_changed = MagicMock(return_value=False)

        captured_input = {}

        async def capture_input(input_data, **kwargs):
            captured_input.update(input_data)
            return
            yield

        mock_graph.astream_events = capture_input

        history = []  # No history
        await adapter._handle_message(
            sample_message, mock_tools, mock_ctx, history, None
        )

        # Messages SHOULD include system prompt
        messages = captured_input.get("messages", [])
        system_messages = [m for m in messages if m[0] == "system"]
        assert len(system_messages) == 1, "Graph factory SHOULD receive system prompt"


class TestHandleStreamEvent:
    """Tests for _handle_stream_event."""

    @pytest.fixture
    def adapter(self):
        """Create adapter with mocked dependencies."""
        with patch("thenvoi.agents.base.ThenvoiLink"):
            with patch("thenvoi.agents.base.AgentRuntime"):
                return ThenvoiLangGraphAgent(
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

        with patch("thenvoi.agents.base.ThenvoiLink"):
            with patch("thenvoi.agents.base.AgentRuntime"):
                adapter = ThenvoiLangGraphAgent(
                    graph_factory=graph_factory,
                    agent_id="agent-123",
                    api_key="test-key",
                )

        await adapter._cleanup_session("room-123")

        mock_checkpointer.adelete_thread.assert_called_once_with("room-123")

    async def test_handles_missing_checkpointer(self):
        """Should handle graph_factory without checkpointer."""
        with patch("thenvoi.agents.base.ThenvoiLink"):
            with patch("thenvoi.agents.base.AgentRuntime"):
                adapter = ThenvoiLangGraphAgent(
                    graph_factory=lambda tools: MagicMock(),  # No checkpointer attribute
                    agent_id="agent-123",
                    api_key="test-key",
                )

        # Should not raise
        await adapter._cleanup_session("room-123")


class TestReconstructMessages:
    """Tests for _reconstruct_messages tool call/result pairing."""

    @pytest.fixture
    def adapter(self):
        """Create adapter for testing _reconstruct_messages."""
        with patch("thenvoi.agents.base.ThenvoiLink"):
            with patch("thenvoi.agents.base.AgentRuntime"):
                return ThenvoiLangGraphAgent(
                    graph_factory=lambda tools: MagicMock(),
                    agent_id="agent-123",
                    api_key="test-key",
                )

    def test_matches_tool_call_and_result_by_run_id(self, adapter):
        """Should match tool_call and tool_result by run_id."""
        import json

        history = [
            {
                "message_type": "tool_call",
                "content": json.dumps(
                    {
                        "event": "on_tool_start",
                        "name": "send_message",
                        "run_id": "run-123",
                        "data": {"input": {"content": "Hello"}},
                    }
                ),
            },
            {
                "message_type": "tool_result",
                "content": json.dumps(
                    {
                        "event": "on_tool_end",
                        "name": "send_message",
                        "run_id": "run-123",
                        "data": {
                            "input": {"content": "Hello"},
                            "output": "content='success' tool_call_id='call_abc123'",
                        },
                    }
                ),
            },
        ]

        messages = adapter._reconstruct_messages(history)

        # Should have AIMessage with tool_calls + ToolMessage
        assert len(messages) == 2
        assert messages[0].tool_calls[0]["name"] == "send_message"
        assert messages[0].tool_calls[0]["args"] == {"content": "Hello"}
        assert messages[1].tool_call_id == "call_abc123"

    def test_matches_back_to_back_same_tool_by_run_id(self, adapter):
        """Back-to-back calls to same tool should match correctly by run_id."""
        import json

        history = [
            # First send_message call
            {
                "message_type": "tool_call",
                "content": json.dumps(
                    {
                        "event": "on_tool_start",
                        "name": "send_message",
                        "run_id": "run-111",
                        "data": {"input": {"content": "First message"}},
                    }
                ),
            },
            # Second send_message call (same tool name!)
            {
                "message_type": "tool_call",
                "content": json.dumps(
                    {
                        "event": "on_tool_start",
                        "name": "send_message",
                        "run_id": "run-222",
                        "data": {"input": {"content": "Second message"}},
                    }
                ),
            },
            # Result for SECOND call arrives first (out of order)
            {
                "message_type": "tool_result",
                "content": json.dumps(
                    {
                        "event": "on_tool_end",
                        "name": "send_message",
                        "run_id": "run-222",
                        "data": {
                            "input": {"content": "Second message"},
                            "output": "content='ok' tool_call_id='call_222'",
                        },
                    }
                ),
            },
            # Result for FIRST call arrives second
            {
                "message_type": "tool_result",
                "content": json.dumps(
                    {
                        "event": "on_tool_end",
                        "name": "send_message",
                        "run_id": "run-111",
                        "data": {
                            "input": {"content": "First message"},
                            "output": "content='ok' tool_call_id='call_111'",
                        },
                    }
                ),
            },
        ]

        messages = adapter._reconstruct_messages(history)

        # Should have 4 messages: 2 AIMessage + 2 ToolMessage
        assert len(messages) == 4

        # First pair should be run-222 (result arrived first)
        assert messages[0].tool_calls[0]["args"]["content"] == "Second message"
        assert messages[1].tool_call_id == "call_222"

        # Second pair should be run-111
        assert messages[2].tool_calls[0]["args"]["content"] == "First message"
        assert messages[3].tool_call_id == "call_111"

    def test_fallback_to_name_matching_without_run_id(self, adapter):
        """Should fall back to name-based LIFO matching when run_id missing."""
        import json

        history = [
            {
                "message_type": "tool_call",
                "content": json.dumps(
                    {
                        "event": "on_tool_start",
                        "name": "lookup_peers",
                        # No run_id
                        "data": {"input": {}},
                    }
                ),
            },
            {
                "message_type": "tool_result",
                "content": json.dumps(
                    {
                        "event": "on_tool_end",
                        "name": "lookup_peers",
                        # No run_id
                        "data": {
                            "input": {},
                            "output": "content='peers' tool_call_id='call_peers'",
                        },
                    }
                ),
            },
        ]

        messages = adapter._reconstruct_messages(history)

        # Should still match by name
        assert len(messages) == 2
        assert messages[0].tool_calls[0]["name"] == "lookup_peers"
        assert messages[1].tool_call_id == "call_peers"

    def test_unmatched_result_emits_tool_message_only(self, adapter):
        """tool_result without matching tool_call should emit ToolMessage only."""
        import json

        history = [
            # Only tool_result, no matching tool_call
            {
                "message_type": "tool_result",
                "content": json.dumps(
                    {
                        "event": "on_tool_end",
                        "name": "send_message",
                        "run_id": "orphan-run",
                        "data": {
                            "input": {"content": "Hello"},
                            "output": "content='ok' tool_call_id='call_orphan'",
                        },
                    }
                ),
            },
        ]

        messages = adapter._reconstruct_messages(history)

        # Should only have ToolMessage, no fabricated AIMessage
        assert len(messages) == 1
        assert messages[0].tool_call_id == "call_orphan"
        assert messages[0].content == "content='ok' tool_call_id='call_orphan'"

    def test_mixed_tools_match_correctly(self, adapter):
        """Multiple different tools should match correctly."""
        import json

        history = [
            {
                "message_type": "tool_call",
                "content": json.dumps(
                    {
                        "name": "send_event",
                        "run_id": "run-event",
                        "data": {
                            "input": {"content": "thinking", "message_type": "thought"}
                        },
                    }
                ),
            },
            {
                "message_type": "tool_call",
                "content": json.dumps(
                    {
                        "name": "lookup_peers",
                        "run_id": "run-peers",
                        "data": {"input": {}},
                    }
                ),
            },
            {
                "message_type": "tool_result",
                "content": json.dumps(
                    {
                        "name": "send_event",
                        "run_id": "run-event",
                        "data": {"output": "tool_call_id='call_event'"},
                    }
                ),
            },
            {
                "message_type": "tool_result",
                "content": json.dumps(
                    {
                        "name": "lookup_peers",
                        "run_id": "run-peers",
                        "data": {"output": "tool_call_id='call_peers'"},
                    }
                ),
            },
        ]

        messages = adapter._reconstruct_messages(history)

        # Should have 4 messages, correctly paired
        assert len(messages) == 4
        assert messages[0].tool_calls[0]["name"] == "send_event"
        assert messages[1].tool_call_id == "call_event"
        assert messages[2].tool_calls[0]["name"] == "lookup_peers"
        assert messages[3].tool_call_id == "call_peers"


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
