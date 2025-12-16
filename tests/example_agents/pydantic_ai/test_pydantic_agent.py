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

from thenvoi.integrations.pydantic_ai import ThenvoiPydanticAgent, create_pydantic_agent
from thenvoi.runtime.types import PlatformMessage
from thenvoi.runtime.execution import ExecutionContext


class TestConstructor:
    """Tests for ThenvoiPydanticAgent initialization."""

    @patch("thenvoi.agents.base.ThenvoiLink")
    @patch("thenvoi.agents.base.AgentRuntime")
    def test_initializes_with_required_params(self, mock_runtime_cls, mock_link_cls):
        """Should initialize with required parameters."""
        adapter = ThenvoiPydanticAgent(
            model="openai:gpt-4o",
            agent_id="agent-123",
            api_key="test-key",
        )

        assert adapter.model == "openai:gpt-4o"
        assert adapter._agent is None  # Lazy created

    @patch("thenvoi.agents.base.ThenvoiLink")
    @patch("thenvoi.agents.base.AgentRuntime")
    def test_accepts_custom_section(self, mock_runtime_cls, mock_link_cls):
        """Should accept custom_section parameter."""
        adapter = ThenvoiPydanticAgent(
            model="openai:gpt-4o",
            agent_id="agent-123",
            api_key="test-key",
            custom_section="You are helpful.",
        )

        assert adapter.custom_section == "You are helpful."

    @patch("thenvoi.agents.base.ThenvoiLink")
    @patch("thenvoi.agents.base.AgentRuntime")
    def test_accepts_system_prompt_override(self, mock_runtime_cls, mock_link_cls):
        """Should accept system_prompt parameter."""
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
    def adapter(self):
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

                adapter = ThenvoiPydanticAgent(
                    model="openai:gpt-4o",
                    agent_id="agent-123",
                    api_key="test-key",
                )
                adapter._agent_name = "TestBot"
                return adapter

    def test_creates_agent_with_tools(self, adapter):
        """Should create Pydantic AI Agent with all platform tools registered."""
        with patch("thenvoi.integrations.pydantic_ai.agent.Agent") as mock_agent_cls:
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

        with patch("thenvoi.integrations.pydantic_ai.agent.Agent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.tool = MagicMock(return_value=lambda x: x)
            mock_agent_cls.return_value = mock_agent

            with patch(
                "thenvoi.integrations.pydantic_ai.agent.render_system_prompt"
            ) as mock_render:
                mock_render.return_value = "rendered prompt"
                adapter._create_agent()

                mock_render.assert_called_once()
                call_kwargs = mock_render.call_args.kwargs
                assert call_kwargs.get("custom_section") == "Be concise."


class TestHandleMessage:
    """Tests for _handle_message method."""

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
    def adapter(self):
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

                adapter = ThenvoiPydanticAgent(
                    model="openai:gpt-4o",
                    agent_id="agent-123",
                    api_key="test-key",
                )
                adapter._agent_name = "TestBot"
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
        tools.send_message = AsyncMock()
        tools.send_event = AsyncMock()
        return tools

    async def test_lazy_creates_agent_on_first_message(
        self, adapter, mock_ctx, sample_message, mock_tools
    ):
        """Agent should be lazily created on first message."""
        with patch("thenvoi.integrations.pydantic_ai.agent.Agent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.tool = MagicMock(return_value=lambda x: x)
            mock_agent.run = AsyncMock(return_value=MagicMock(output="response"))
            mock_agent_cls.return_value = mock_agent

            assert adapter._agent is None

            await adapter._handle_message(
                sample_message, mock_tools, mock_ctx, None, None
            )

            assert adapter._agent is not None

    async def test_converts_history_to_pydantic_format(
        self, adapter, mock_ctx, sample_message, mock_tools
    ):
        """History should be converted to Pydantic AI ModelRequest/ModelResponse format."""
        captured_kwargs = {}

        with patch("thenvoi.integrations.pydantic_ai.agent.Agent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.tool = MagicMock(return_value=lambda x: x)

            mock_result = MagicMock()
            mock_result.all_messages = MagicMock(return_value=[])

            async def capture_run(prompt, **kwargs):
                captured_kwargs.update(kwargs)
                return mock_result

            mock_agent.run = capture_run
            mock_agent_cls.return_value = mock_agent

            history = [
                {"role": "user", "content": "Previous message", "sender_name": "User"},
                {
                    "role": "assistant",
                    "content": "Previous response",
                    "sender_name": "Bot",
                },
            ]
            await adapter._handle_message(
                sample_message, mock_tools, mock_ctx, history, None
            )

        # History should be passed via message_history parameter (Pydantic AI specific)
        assert "message_history" in captured_kwargs
        history = captured_kwargs["message_history"]
        assert len(history) == 2

        # Verify Pydantic AI message format (ModelRequest for user, ModelResponse for assistant)
        from pydantic_ai.messages import ModelRequest, ModelResponse

        assert isinstance(history[0], ModelRequest)  # User message
        assert isinstance(history[1], ModelResponse)  # Assistant message


class TestCreatePydanticAgent:
    """Tests for create_pydantic_agent convenience function."""

    async def test_creates_and_starts_agent(self):
        """Should create and start the agent."""
        with patch("thenvoi.agents.base.ThenvoiLink") as mock_link_cls:
            with patch("thenvoi.agents.base.AgentRuntime") as mock_runtime_cls:
                mock_link = MagicMock()
                mock_link.agent_id = "agent-123"
                mock_link.disconnect = AsyncMock()
                mock_link.run_forever = AsyncMock()
                mock_link.rest = MagicMock()
                mock_link.rest.agent_api = MagicMock()
                agent_me = MagicMock()
                agent_me.name = "TestBot"
                agent_me.description = "A test bot"
                mock_link.rest.agent_api.get_agent_me = AsyncMock(
                    return_value=MagicMock(data=agent_me)
                )
                mock_link_cls.return_value = mock_link

                mock_runtime = MagicMock()
                mock_runtime.start = AsyncMock()
                mock_runtime.stop = AsyncMock()
                mock_runtime_cls.return_value = mock_runtime

                # Also mock the Pydantic AI Agent to avoid OpenAI API call
                with patch(
                    "thenvoi.integrations.pydantic_ai.agent.Agent"
                ) as mock_agent_cls:
                    mock_agent = MagicMock()
                    mock_agent.tool = MagicMock(return_value=lambda x: x)
                    mock_agent_cls.return_value = mock_agent

                    agent = await create_pydantic_agent(
                        model="openai:gpt-4o",
                        agent_id="agent-123",
                        api_key="test-key",
                    )

                    assert agent is not None
                    mock_runtime.start.assert_called_once()
