"""
Integration tests for the new architecture (LangGraphAdapter with mocked ThenvoiAgent).

These tests verify that LangGraphAdapter works correctly with ThenvoiAgent.
Lower-level ThenvoiAgent + AgentSession tests are in tests/core/test_agent.py.
For real backend integration tests, see test_full_workflow.py which uses @requires_api.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from thenvoi.agent.core.types import PlatformMessage
from thenvoi.agent.langgraph.adapter import LangGraphAdapter


class TestLangGraphAdapterIntegration:
    """Tests for LangGraphAdapter with ThenvoiAgent."""

    @pytest.fixture
    def mock_graph(self):
        """Mock LangGraph that echoes input."""
        graph = MagicMock()

        async def mock_stream_events(input_data, **kwargs):
            # Yield some events
            yield {"event": "on_chat_model_start", "data": {}}
            yield {"event": "on_chat_model_stream", "data": {"chunk": "Hello"}}
            yield {"event": "on_chat_model_end", "data": {}}

        graph.astream_events = mock_stream_events
        return graph

    @pytest.fixture
    def mock_rest_client(self):
        """Mock REST API client."""
        client = AsyncMock()
        client.get_me = AsyncMock(
            return_value={
                "id": "agent-123",
                "name": "TestBot",
                "description": "A test agent",
            }
        )
        client.get_chat_rooms = AsyncMock(
            return_value=[{"id": "room-123", "title": "Test Room"}]
        )
        client.get_chat_room_participants = AsyncMock(
            return_value=[{"id": "user-456", "name": "Test User", "type": "User"}]
        )
        client.get_chat_context = AsyncMock(
            return_value={
                "messages": [],
                "participants": [
                    {"id": "user-456", "name": "Test User", "type": "User"}
                ],
            }
        )
        client.send_message = AsyncMock(
            return_value={"id": "msg-sent", "status": "sent"}
        )
        client.send_event = AsyncMock(return_value={"id": "event-sent"})
        return client

    @pytest.fixture
    def mock_ws_client(self):
        """Mock WebSocket client."""
        ws = AsyncMock()
        ws.__aenter__ = AsyncMock(return_value=ws)
        ws.__aexit__ = AsyncMock(return_value=None)
        ws.join_agent_rooms_channel = AsyncMock()
        ws.join_chat_room_channel = AsyncMock()
        ws.leave_topic = AsyncMock()
        return ws

    async def test_adapter_processes_message_through_graph(
        self, mock_graph, mock_rest_client, mock_ws_client
    ):
        """Adapter should process messages through LangGraph."""
        graph_invoked = []

        async def tracking_stream(input_data, **kwargs):
            graph_invoked.append(input_data)
            yield {"event": "on_chat_model_end", "data": {}}

        mock_graph.astream_events = tracking_stream

        with patch("thenvoi.agent.langgraph.adapter.ThenvoiAgent") as mock_agent_cls:
            mock_agent = AsyncMock()
            mock_agent.agent_id = "agent-123"
            mock_agent.agent_name = "TestBot"
            mock_agent.agent_description = "A test agent"
            mock_agent.active_sessions = {}
            mock_agent.start = AsyncMock()
            mock_agent.stop = AsyncMock()
            mock_agent_cls.return_value = mock_agent

            adapter = LangGraphAdapter(
                graph=mock_graph,
                agent_id="agent-123",
                api_key="test-key",
            )

            await adapter.start()

            # Get the message handler that was registered
            on_message = mock_agent.start.call_args.kwargs["on_message"]

            # Create mock session
            mock_session = MagicMock()
            mock_session.is_llm_initialized = False
            mock_session.participants = [
                {"id": "user-456", "name": "Test User", "type": "User"}
            ]
            mock_session.participants_changed = MagicMock(return_value=True)
            mock_session.mark_llm_initialized = MagicMock()
            mock_session.mark_participants_sent = MagicMock()
            mock_session.build_participants_message = MagicMock(
                return_value="## Participants\n- Test User"
            )
            mock_session.get_history_for_llm = AsyncMock(return_value=[])
            mock_agent.active_sessions = {"room-123": mock_session}

            # Create mock tools
            mock_tools = MagicMock()
            mock_tools.to_langchain_tools = MagicMock(return_value=[])
            mock_tools.send_event = AsyncMock()

            # Process a message
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

            await on_message(msg, mock_tools)

            # Graph should have been invoked
            assert len(graph_invoked) == 1
            assert "messages" in graph_invoked[0]

            await adapter.stop()

    async def test_adapter_includes_system_prompt_on_first_message(
        self, mock_graph, mock_rest_client, mock_ws_client
    ):
        """First message should include system prompt."""
        captured_input = {}

        async def capture_stream(input_data, **kwargs):
            captured_input.update(input_data)
            yield {"event": "on_chat_model_end", "data": {}}

        mock_graph.astream_events = capture_stream

        with patch("thenvoi.agent.langgraph.adapter.ThenvoiAgent") as mock_agent_cls:
            mock_agent = AsyncMock()
            mock_agent.agent_id = "agent-123"
            mock_agent.agent_name = "TestBot"
            mock_agent.agent_description = "A test agent"
            mock_agent.active_sessions = {}
            mock_agent.start = AsyncMock()
            mock_agent.stop = AsyncMock()
            mock_agent_cls.return_value = mock_agent

            adapter = LangGraphAdapter(
                graph=mock_graph,
                agent_id="agent-123",
                api_key="test-key",
                custom_section="You are a helpful assistant.",
            )

            await adapter.start()

            on_message = mock_agent.start.call_args.kwargs["on_message"]

            # Setup mock session (first message)
            mock_session = MagicMock()
            mock_session.is_llm_initialized = False
            mock_session.participants = []
            mock_session.participants_changed = MagicMock(return_value=False)
            mock_session.mark_llm_initialized = MagicMock()
            mock_session.mark_participants_sent = MagicMock()
            mock_session.get_history_for_llm = AsyncMock(return_value=[])
            mock_agent.active_sessions = {"room-123": mock_session}

            mock_tools = MagicMock()
            mock_tools.to_langchain_tools = MagicMock(return_value=[])
            mock_tools.send_event = AsyncMock()

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

            await on_message(msg, mock_tools)

            # First message should be system prompt
            messages = captured_input.get("messages", [])
            assert len(messages) >= 1
            assert messages[0][0] == "system"
            assert "TestBot" in messages[0][1]

            # Should mark as initialized
            mock_session.mark_llm_initialized.assert_called_once()

            await adapter.stop()

    async def test_cleanup_callback_invoked_on_room_removed(
        self, mock_graph, mock_rest_client, mock_ws_client
    ):
        """Cleanup callback should clear checkpointer state."""
        mock_checkpointer = AsyncMock()
        mock_checkpointer.adelete_thread = AsyncMock()

        def graph_factory(tools):
            return mock_graph

        graph_factory.checkpointer = mock_checkpointer

        with patch("thenvoi.agent.langgraph.adapter.ThenvoiAgent") as mock_agent_cls:
            mock_agent = AsyncMock()
            mock_agent.agent_id = "agent-123"
            mock_agent.agent_name = "TestBot"
            mock_agent.agent_description = "A test agent"
            mock_agent.active_sessions = {}
            mock_agent.start = AsyncMock()
            mock_agent.stop = AsyncMock()
            mock_agent_cls.return_value = mock_agent

            adapter = LangGraphAdapter(
                graph_factory=graph_factory,
                agent_id="agent-123",
                api_key="test-key",
            )

            await adapter.start()

            # Simulate cleanup callback
            await adapter._cleanup_session("room-123")

            # Checkpointer should be called
            mock_checkpointer.adelete_thread.assert_called_once_with("room-123")

            await adapter.stop()
