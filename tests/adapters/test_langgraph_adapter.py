"""Tests for LangGraphAdapter."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from thenvoi.adapters.langgraph import LangGraphAdapter
from thenvoi.core.types import PlatformMessage


@pytest.fixture
def sample_message():
    """Create a sample platform message."""
    return PlatformMessage(
        id="msg-123",
        room_id="room-123",
        content="Hello, agent!",
        sender_id="user-456",
        sender_type="User",
        sender_name="Alice",
        message_type="text",
        metadata={},
        created_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def mock_tools():
    """Create mock AgentToolsProtocol."""
    tools = AsyncMock()
    tools.send_message = AsyncMock(return_value={"status": "sent"})
    tools.send_event = AsyncMock(return_value={"status": "sent"})
    tools.add_participant = AsyncMock(return_value={"id": "user-1"})
    tools.remove_participant = AsyncMock(return_value={"status": "removed"})
    tools.lookup_peers = AsyncMock(return_value={"peers": []})
    tools.get_participants = AsyncMock(return_value=[])
    return tools


@pytest.fixture
def mock_llm():
    """Create mock LangChain LLM."""
    return MagicMock()


@pytest.fixture
def mock_checkpointer():
    """Create mock checkpointer."""
    return MagicMock()


class TestInitialization:
    """Tests for adapter initialization."""

    def test_simple_pattern_with_llm(self, mock_llm, mock_checkpointer):
        """Should accept simple pattern with llm and checkpointer."""
        adapter = LangGraphAdapter(
            llm=mock_llm,
            checkpointer=mock_checkpointer,
        )

        assert adapter.graph_factory is not None
        assert adapter._static_graph is None

    def test_simple_pattern_with_custom_section(self, mock_llm, mock_checkpointer):
        """Should accept custom_section in simple pattern."""
        adapter = LangGraphAdapter(
            llm=mock_llm,
            checkpointer=mock_checkpointer,
            custom_section="Be helpful.",
        )

        assert adapter.custom_section == "Be helpful."

    def test_simple_pattern_with_additional_tools(self, mock_llm, mock_checkpointer):
        """Should integrate additional_tools in simple pattern."""
        mock_tool = MagicMock()
        adapter = LangGraphAdapter(
            llm=mock_llm,
            checkpointer=mock_checkpointer,
            additional_tools=[mock_tool],
        )

        assert adapter.graph_factory is not None
        # additional_tools cleared after baking into factory
        assert adapter.additional_tools == []

    def test_advanced_pattern_with_graph_factory(self):
        """Should accept graph_factory for advanced pattern."""
        mock_factory = MagicMock()
        adapter = LangGraphAdapter(graph_factory=mock_factory)

        assert adapter.graph_factory is mock_factory

    def test_advanced_pattern_with_static_graph(self):
        """Should accept static graph for advanced pattern."""
        mock_graph = MagicMock()
        adapter = LangGraphAdapter(graph=mock_graph)

        assert adapter._static_graph is mock_graph

    def test_raises_without_llm_or_graph(self):
        """Should raise if neither llm nor graph_factory/graph provided."""
        with pytest.raises(ValueError, match="Must provide either llm"):
            LangGraphAdapter()

    def test_default_values(self, mock_llm, mock_checkpointer):
        """Should use default values for optional params."""
        adapter = LangGraphAdapter(
            llm=mock_llm,
            checkpointer=mock_checkpointer,
        )

        assert adapter.prompt_template == "default"
        assert adapter.custom_section == ""
        assert adapter.history_converter is not None


class TestOnStarted:
    """Tests for on_started() method."""

    @pytest.mark.asyncio
    async def test_renders_system_prompt(self, mock_llm, mock_checkpointer):
        """Should render system prompt from agent metadata."""
        adapter = LangGraphAdapter(
            llm=mock_llm,
            checkpointer=mock_checkpointer,
        )

        await adapter.on_started(agent_name="TestBot", agent_description="A test bot")

        assert adapter.agent_name == "TestBot"
        assert adapter.agent_description == "A test bot"
        assert adapter._system_prompt != ""
        assert "TestBot" in adapter._system_prompt

    @pytest.mark.asyncio
    async def test_includes_custom_section(self, mock_llm, mock_checkpointer):
        """Should include custom_section in system prompt."""
        adapter = LangGraphAdapter(
            llm=mock_llm,
            checkpointer=mock_checkpointer,
            custom_section="Always be concise.",
        )

        await adapter.on_started(agent_name="TestBot", agent_description="A test bot")

        assert "Always be concise." in adapter._system_prompt


class TestOnMessage:
    """Tests for on_message() method."""

    @pytest.mark.asyncio
    async def test_calls_graph_with_messages(
        self, sample_message, mock_tools, mock_llm, mock_checkpointer
    ):
        """Should call graph with formatted messages."""
        adapter = LangGraphAdapter(
            llm=mock_llm,
            checkpointer=mock_checkpointer,
        )
        await adapter.on_started("TestBot", "Test bot")

        captured_input = {}

        async def capture_astream_events(graph_input, **kwargs):
            captured_input.update(graph_input)
            return
            yield  # async generator

        mock_graph = MagicMock()
        mock_graph.astream_events = capture_astream_events
        adapter.graph_factory = MagicMock(return_value=mock_graph)

        # Patch at the module where it's imported
        with patch(
            "thenvoi.integrations.langgraph.langchain_tools.agent_tools_to_langchain"
        ) as mock_convert:
            mock_convert.return_value = []

            await adapter.on_message(
                msg=sample_message,
                tools=mock_tools,
                history=[],
                participants_msg=None,
                is_session_bootstrap=True,
                room_id="room-123",
            )

            # Verify graph was created with tools
            adapter.graph_factory.assert_called_once()
            assert "messages" in captured_input

    @pytest.mark.asyncio
    async def test_includes_system_prompt_on_bootstrap(
        self, sample_message, mock_tools, mock_llm, mock_checkpointer
    ):
        """Should include system prompt on session bootstrap."""
        adapter = LangGraphAdapter(
            llm=mock_llm,
            checkpointer=mock_checkpointer,
        )
        await adapter.on_started("TestBot", "Test bot")

        captured_input = {}

        async def capture_astream_events(graph_input, **kwargs):
            captured_input.update(graph_input)
            return
            yield

        mock_graph = MagicMock()
        mock_graph.astream_events = capture_astream_events
        adapter.graph_factory = MagicMock(return_value=mock_graph)

        with patch(
            "thenvoi.integrations.langgraph.langchain_tools.agent_tools_to_langchain"
        ) as mock_convert:
            mock_convert.return_value = []

            await adapter.on_message(
                msg=sample_message,
                tools=mock_tools,
                history=[],
                participants_msg=None,
                is_session_bootstrap=True,
                room_id="room-123",
            )

            # System prompt should be in messages
            messages = captured_input.get("messages", [])
            system_msg = [m for m in messages if m[0] == "system"]
            assert len(system_msg) >= 1

    @pytest.mark.asyncio
    async def test_injects_history_on_bootstrap(
        self, sample_message, mock_tools, mock_llm, mock_checkpointer
    ):
        """Should inject converted history on bootstrap."""
        adapter = LangGraphAdapter(
            llm=mock_llm,
            checkpointer=mock_checkpointer,
        )
        await adapter.on_started("TestBot", "Test bot")

        history = [
            HumanMessage(content="Previous question"),
            AIMessage(content="Previous answer"),
        ]

        captured_input = {}

        async def capture_astream_events(graph_input, **kwargs):
            captured_input.update(graph_input)
            return
            yield

        mock_graph = MagicMock()
        mock_graph.astream_events = capture_astream_events
        adapter.graph_factory = MagicMock(return_value=mock_graph)

        with patch(
            "thenvoi.integrations.langgraph.langchain_tools.agent_tools_to_langchain"
        ) as mock_convert:
            mock_convert.return_value = []

            await adapter.on_message(
                msg=sample_message,
                tools=mock_tools,
                history=history,
                participants_msg=None,
                is_session_bootstrap=True,
                room_id="room-123",
            )

            messages = captured_input.get("messages", [])
            # Should include history messages (system + 2 history + user)
            assert len(messages) >= 3

    @pytest.mark.asyncio
    async def test_injects_participants_message(
        self, sample_message, mock_tools, mock_llm, mock_checkpointer
    ):
        """Should inject participants update when provided."""
        adapter = LangGraphAdapter(
            llm=mock_llm,
            checkpointer=mock_checkpointer,
        )
        await adapter.on_started("TestBot", "Test bot")

        captured_input = {}

        async def capture_astream_events(graph_input, **kwargs):
            captured_input.update(graph_input)
            return
            yield

        mock_graph = MagicMock()
        mock_graph.astream_events = capture_astream_events
        adapter.graph_factory = MagicMock(return_value=mock_graph)

        with patch(
            "thenvoi.integrations.langgraph.langchain_tools.agent_tools_to_langchain"
        ) as mock_convert:
            mock_convert.return_value = []

            await adapter.on_message(
                msg=sample_message,
                tools=mock_tools,
                history=[],
                participants_msg="Alice joined the room",
                is_session_bootstrap=True,
                room_id="room-123",
            )

            messages = captured_input.get("messages", [])
            # Find system message with participant info
            system_msgs = [m for m in messages if m[0] == "system"]
            participant_msg = any("Alice joined" in str(m[1]) for m in system_msgs)
            assert participant_msg


class TestStreamEventHandling:
    """Tests for _handle_stream_event() method."""

    @pytest.mark.asyncio
    async def test_handles_on_tool_start(self, mock_tools, mock_llm, mock_checkpointer):
        """Should send tool_call event on on_tool_start."""
        adapter = LangGraphAdapter(
            llm=mock_llm,
            checkpointer=mock_checkpointer,
        )

        event = {
            "event": "on_tool_start",
            "name": "send_message",
            "run_id": "run-123",
            "data": {"input": {"content": "Hello"}},
        }

        await adapter._handle_stream_event(event, "room-123", mock_tools)

        mock_tools.send_event.assert_awaited_once()
        call_kwargs = mock_tools.send_event.call_args.kwargs
        assert call_kwargs["message_type"] == "tool_call"

    @pytest.mark.asyncio
    async def test_handles_on_tool_end(self, mock_tools, mock_llm, mock_checkpointer):
        """Should send tool_result event on on_tool_end."""
        adapter = LangGraphAdapter(
            llm=mock_llm,
            checkpointer=mock_checkpointer,
        )

        event = {
            "event": "on_tool_end",
            "name": "send_message",
            "run_id": "run-123",
            "data": {"output": "success"},
        }

        await adapter._handle_stream_event(event, "room-123", mock_tools)

        mock_tools.send_event.assert_awaited_once()
        call_kwargs = mock_tools.send_event.call_args.kwargs
        assert call_kwargs["message_type"] == "tool_result"

    @pytest.mark.asyncio
    async def test_ignores_other_events(self, mock_tools, mock_llm, mock_checkpointer):
        """Should ignore events other than tool_start/end."""
        adapter = LangGraphAdapter(
            llm=mock_llm,
            checkpointer=mock_checkpointer,
        )

        event = {
            "event": "on_chat_model_start",
            "name": "ChatOpenAI",
        }

        await adapter._handle_stream_event(event, "room-123", mock_tools)

        mock_tools.send_event.assert_not_awaited()


class TestOnCleanup:
    """Tests for on_cleanup() method."""

    @pytest.mark.asyncio
    async def test_cleanup_with_graph_factory(self, mock_llm, mock_checkpointer):
        """Should handle cleanup when using graph_factory."""
        adapter = LangGraphAdapter(
            llm=mock_llm,
            checkpointer=mock_checkpointer,
        )

        # Should not raise
        await adapter.on_cleanup("room-123")

    @pytest.mark.asyncio
    async def test_cleanup_without_graph_factory(self):
        """Should handle cleanup when using static graph."""
        mock_graph = MagicMock()
        adapter = LangGraphAdapter(graph=mock_graph)

        # Should not raise (early return)
        await adapter.on_cleanup("room-123")


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_reports_error_on_graph_failure(
        self, sample_message, mock_tools, mock_llm, mock_checkpointer
    ):
        """Should report error when graph execution fails."""
        adapter = LangGraphAdapter(
            llm=mock_llm,
            checkpointer=mock_checkpointer,
        )
        await adapter.on_started("TestBot", "Test bot")

        async def failing_stream(*args, **kwargs):
            raise Exception("Graph error!")
            yield  # Make it async generator

        mock_graph = MagicMock()
        mock_graph.astream_events = failing_stream
        adapter.graph_factory = MagicMock(return_value=mock_graph)

        with patch(
            "thenvoi.integrations.langgraph.langchain_tools.agent_tools_to_langchain"
        ) as mock_convert:
            mock_convert.return_value = []

            with pytest.raises(Exception, match="Graph error!"):
                await adapter.on_message(
                    msg=sample_message,
                    tools=mock_tools,
                    history=[],
                    participants_msg=None,
                    is_session_bootstrap=True,
                    room_id="room-123",
                )

            # Should have tried to report error
            mock_tools.send_event.assert_awaited()


class TestStaticGraph:
    """Tests for static graph pattern."""

    @pytest.mark.asyncio
    async def test_uses_static_graph_when_provided(self, sample_message, mock_tools):
        """Should use static graph instead of factory when provided."""
        captured_input = {}

        async def capture_astream_events(graph_input, **kwargs):
            captured_input.update(graph_input)
            return
            yield

        mock_graph = MagicMock()
        mock_graph.astream_events = capture_astream_events

        adapter = LangGraphAdapter(graph=mock_graph)
        await adapter.on_started("TestBot", "Test bot")

        with patch(
            "thenvoi.integrations.langgraph.langchain_tools.agent_tools_to_langchain"
        ) as mock_convert:
            mock_convert.return_value = []

            await adapter.on_message(
                msg=sample_message,
                tools=mock_tools,
                history=[],
                participants_msg=None,
                is_session_bootstrap=True,
                room_id="room-123",
            )

            # Graph should have been called
            assert "messages" in captured_input
