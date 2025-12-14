"""
Unit tests for ThenvoiPydanticAgent (example agent).

Tests:
1. Constructor validation
2. Agent creation with tools
3. Message handling flow
4. Session history handling
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from thenvoi_pydantic_agent import ThenvoiPydanticAgent, create_pydantic_agent
from thenvoi.core.types import PlatformMessage


class TestConstructor:
    """Tests for ThenvoiPydanticAgent initialization."""

    def test_initializes_with_required_params(self):
        """Should initialize with required parameters."""
        with patch("thenvoi_pydantic_agent.ThenvoiAgent"):
            adapter = ThenvoiPydanticAgent(
                model="openai:gpt-4o",
                agent_id="agent-123",
                api_key="test-key",
            )

        assert adapter.model == "openai:gpt-4o"
        assert adapter._agent is None  # Lazy created

    def test_accepts_custom_section(self):
        """Should accept custom_section parameter."""
        with patch("thenvoi_pydantic_agent.ThenvoiAgent"):
            adapter = ThenvoiPydanticAgent(
                model="openai:gpt-4o",
                agent_id="agent-123",
                api_key="test-key",
                custom_section="You are helpful.",
            )

        assert adapter.custom_section == "You are helpful."

    def test_accepts_system_prompt_override(self):
        """Should accept system_prompt parameter."""
        with patch("thenvoi_pydantic_agent.ThenvoiAgent"):
            adapter = ThenvoiPydanticAgent(
                model="openai:gpt-4o",
                agent_id="agent-123",
                api_key="test-key",
                system_prompt="Custom system prompt",
            )

        assert adapter.system_prompt == "Custom system prompt"


class TestAgentCreation:
    """Tests for _create_agent method."""

    @pytest.fixture
    def adapter(self):
        """Create adapter with mocked dependencies."""
        with patch("thenvoi_pydantic_agent.ThenvoiAgent") as mock_thenvoi_cls:
            mock_thenvoi = AsyncMock()
            mock_thenvoi._agent_name = "TestBot"
            mock_thenvoi.active_sessions = {}
            mock_thenvoi_cls.return_value = mock_thenvoi

            adapter = ThenvoiPydanticAgent(
                model="openai:gpt-4o",
                agent_id="agent-123",
                api_key="test-key",
            )
            adapter._agent_name = "TestBot"
            return adapter

    def test_creates_agent_with_tools(self, adapter):
        """Should create Pydantic AI Agent with all platform tools registered."""
        with patch("thenvoi_pydantic_agent.Agent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.tool = MagicMock(return_value=lambda x: x)
            mock_agent_cls.return_value = mock_agent

            adapter._create_agent()

            # Should have created agent with model
            mock_agent_cls.assert_called_once()
            call_kwargs = mock_agent_cls.call_args.kwargs
            assert "system_prompt" in call_kwargs
            assert "deps_type" in call_kwargs

    def test_uses_custom_section_in_prompt(self, adapter):
        """Should include custom_section in system prompt."""
        adapter.custom_section = "Be concise."

        with patch("thenvoi_pydantic_agent.Agent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.tool = MagicMock(return_value=lambda x: x)
            mock_agent_cls.return_value = mock_agent

            with patch("thenvoi_pydantic_agent.render_system_prompt") as mock_render:
                mock_render.return_value = "rendered prompt"
                adapter._create_agent()

                mock_render.assert_called_once()
                call_kwargs = mock_render.call_args.kwargs
                assert call_kwargs.get("custom_section") == "Be concise."


class TestHandleMessage:
    """Tests for _handle_message method."""

    @pytest.fixture
    def adapter(self):
        """Create adapter with mocked dependencies."""
        with patch("thenvoi_pydantic_agent.ThenvoiAgent") as mock_thenvoi_cls:
            mock_thenvoi = AsyncMock()
            mock_thenvoi._agent_name = "TestBot"
            mock_thenvoi.active_sessions = {}
            mock_thenvoi_cls.return_value = mock_thenvoi

            adapter = ThenvoiPydanticAgent(
                model="openai:gpt-4o",
                agent_id="agent-123",
                api_key="test-key",
            )
            adapter._agent_name = "TestBot"
            return adapter

    @pytest.fixture
    def mock_session(self):
        """Create a mock session."""
        session = MagicMock()
        session.is_llm_initialized = False
        session.participants = [{"id": "user-456", "name": "Test User", "type": "User"}]
        session.mark_llm_initialized = MagicMock()
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
        tools.send_message = AsyncMock()
        tools.send_event = AsyncMock()
        return tools

    async def test_lazy_creates_agent_on_first_message(
        self, adapter, mock_session, sample_message, mock_tools
    ):
        """Agent should be lazily created on first message."""
        adapter.thenvoi.active_sessions = {"room-123": mock_session}

        with patch("thenvoi_pydantic_agent.Agent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.tool = MagicMock(return_value=lambda x: x)
            mock_agent.run = AsyncMock(return_value=MagicMock(output="response"))
            mock_agent_cls.return_value = mock_agent

            assert adapter._agent is None

            await adapter._handle_message(sample_message, mock_tools)

            assert adapter._agent is not None

    async def test_loads_history_on_first_message(
        self, adapter, mock_session, sample_message, mock_tools
    ):
        """First message should load platform history and pass via message_history."""
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

        captured_kwargs = {}

        with patch("thenvoi_pydantic_agent.Agent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.tool = MagicMock(return_value=lambda x: x)

            # Mock all_messages to return something for history update
            mock_result = MagicMock()
            mock_result.all_messages = MagicMock(return_value=[])

            async def capture_run(prompt, **kwargs):
                captured_kwargs.update(kwargs)
                return mock_result

            mock_agent.run = capture_run
            mock_agent_cls.return_value = mock_agent

            await adapter._handle_message(sample_message, mock_tools)

        # Should have called get_history_for_llm
        mock_session.get_history_for_llm.assert_called_once()

        # Should have marked as initialized
        mock_session.mark_llm_initialized.assert_called_once()

        # History should be passed via message_history parameter
        assert "message_history" in captured_kwargs
        assert len(captured_kwargs["message_history"]) == 2

    async def test_skips_history_load_on_subsequent_messages(
        self, adapter, mock_session, sample_message, mock_tools
    ):
        """Subsequent messages should use cached history, not reload from platform."""
        adapter.thenvoi.active_sessions = {"room-123": mock_session}
        # Pre-populate message history to simulate subsequent message
        adapter._message_history["room-123"] = []

        with patch("thenvoi_pydantic_agent.Agent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.tool = MagicMock(return_value=lambda x: x)

            mock_result = MagicMock()
            mock_result.all_messages = MagicMock(return_value=[])

            async def capture_run(prompt, **kwargs):
                return mock_result

            mock_agent.run = capture_run
            mock_agent_cls.return_value = mock_agent

            await adapter._handle_message(sample_message, mock_tools)

        # Should NOT call get_history_for_llm on subsequent messages
        mock_session.get_history_for_llm.assert_not_called()

        # Should NOT mark as initialized again
        mock_session.mark_llm_initialized.assert_not_called()


class TestCreatePydanticAgent:
    """Tests for create_pydantic_agent convenience function."""

    async def test_creates_and_starts_agent(self):
        """Should create and start the agent."""
        with patch("thenvoi_pydantic_agent.ThenvoiAgent") as mock_thenvoi_cls:
            mock_thenvoi = AsyncMock()
            mock_thenvoi._agent_name = "TestBot"
            mock_thenvoi.start = AsyncMock()
            mock_thenvoi_cls.return_value = mock_thenvoi

            agent = await create_pydantic_agent(
                model="openai:gpt-4o",
                agent_id="agent-123",
                api_key="test-key",
            )

            assert agent is not None
            mock_thenvoi.start.assert_called_once()
