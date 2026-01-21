"""Tests for ParlantAdapter with official Parlant SDK."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
import sys

import pytest

from thenvoi.adapters.parlant import ParlantAdapter
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
    tools.get_tool_schemas = MagicMock(return_value=[])
    tools.get_openai_tool_schemas = MagicMock(return_value=[])
    tools.send_message = AsyncMock(return_value={"status": "sent"})
    tools.send_event = AsyncMock(return_value={"status": "sent"})
    tools.execute_tool_call = AsyncMock(return_value={"status": "success"})
    return tools


@pytest.fixture
def mock_parlant_client():
    """Create mock Parlant client."""
    client = AsyncMock()

    # Mock agents API
    client.agents = AsyncMock()
    client.agents.create = AsyncMock(return_value=MagicMock(id="agent-123"))
    client.agents.create_guideline = AsyncMock()

    # Mock customers API
    client.customers = AsyncMock()
    client.customers.create = AsyncMock()

    # Mock sessions API
    client.sessions = AsyncMock()
    client.sessions.create = AsyncMock(return_value=MagicMock(id="session-123"))
    client.sessions.create_event = AsyncMock(return_value=MagicMock(offset=1))
    client.sessions.list_events = AsyncMock(return_value=[])
    client.sessions.submit_tool_result = AsyncMock()

    return client


@pytest.fixture
def mock_parlant_module(mock_parlant_client):
    """Create mock parlant.client module."""
    mock_module = MagicMock()
    mock_module.AsyncParlantClient = MagicMock(return_value=mock_parlant_client)
    return mock_module


class TestInitialization:
    """Tests for adapter initialization."""

    def test_default_initialization(self):
        """Should initialize with default values."""
        adapter = ParlantAdapter()

        assert adapter.parlant_url == "http://localhost:8000"
        assert adapter.agent_id is None
        assert adapter.guidelines == []
        assert adapter.enable_execution_reporting is False
        assert adapter.wait_timeout == 60
        assert adapter.history_converter is not None

    def test_custom_initialization(self):
        """Should accept custom parameters."""
        guidelines = [
            {"condition": "User asks about refunds", "action": "Check order status"}
        ]
        adapter = ParlantAdapter(
            parlant_url="http://custom:9000",
            agent_id="custom-agent",
            custom_section="Be helpful.",
            guidelines=guidelines,
            enable_execution_reporting=True,
            wait_timeout=30,
        )

        assert adapter.parlant_url == "http://custom:9000"
        assert adapter.agent_id == "custom-agent"
        assert adapter.custom_section == "Be helpful."
        assert adapter.guidelines == guidelines
        assert adapter.enable_execution_reporting is True
        assert adapter.wait_timeout == 30

    def test_env_var_fallback(self):
        """Should use environment variables as fallback."""
        with patch.dict(
            "os.environ",
            {"PARLANT_URL": "http://env:8080", "PARLANT_AGENT_ID": "env-agent"},
        ):
            adapter = ParlantAdapter()

            assert adapter.parlant_url == "http://env:8080"
            assert adapter.agent_id == "env-agent"

    def test_system_prompt_override(self):
        """Should use custom system_prompt if provided."""
        adapter = ParlantAdapter(
            system_prompt="You are a custom assistant.",
        )

        assert adapter.system_prompt == "You are a custom assistant."


class TestOnStarted:
    """Tests for on_started() method."""

    @pytest.mark.asyncio
    async def test_creates_agent_dynamically_when_no_agent_id(
        self, mock_parlant_client, mock_parlant_module
    ):
        """Should create agent dynamically when agent_id is not provided."""
        adapter = ParlantAdapter()

        with patch.dict(sys.modules, {"parlant.client": mock_parlant_module}):
            # Mock the session manager
            with patch(
                "thenvoi.integrations.parlant.session_manager.ParlantSessionManager"
            ):
                await adapter.on_started(
                    agent_name="TestBot", agent_description="A test bot"
                )

        mock_parlant_client.agents.create.assert_called_once_with(
            name="TestBot",
            description="A test bot",
        )
        assert adapter.agent_id == "agent-123"

    @pytest.mark.asyncio
    async def test_uses_existing_agent_when_agent_id_provided(
        self, mock_parlant_client, mock_parlant_module
    ):
        """Should use existing agent when agent_id is provided."""
        adapter = ParlantAdapter(agent_id="existing-agent")

        with patch.dict(sys.modules, {"parlant.client": mock_parlant_module}):
            with patch(
                "thenvoi.integrations.parlant.session_manager.ParlantSessionManager"
            ):
                await adapter.on_started(
                    agent_name="TestBot", agent_description="A test bot"
                )

        mock_parlant_client.agents.create.assert_not_called()
        assert adapter.agent_id == "existing-agent"

    @pytest.mark.asyncio
    async def test_registers_guidelines_on_dynamic_creation(
        self, mock_parlant_client, mock_parlant_module
    ):
        """Should register guidelines when creating agent dynamically."""
        guidelines = [
            {"condition": "Condition 1", "action": "Action 1"},
            {"condition": "Condition 2", "action": "Action 2"},
        ]
        adapter = ParlantAdapter(guidelines=guidelines)

        with patch.dict(sys.modules, {"parlant.client": mock_parlant_module}):
            with patch(
                "thenvoi.integrations.parlant.session_manager.ParlantSessionManager"
            ):
                await adapter.on_started(
                    agent_name="TestBot", agent_description="A test bot"
                )

        assert mock_parlant_client.agents.create_guideline.call_count == 2

    @pytest.mark.asyncio
    async def test_renders_system_prompt(
        self, mock_parlant_client, mock_parlant_module
    ):
        """Should render system prompt from agent metadata."""
        adapter = ParlantAdapter(agent_id="test-agent")

        with patch.dict(sys.modules, {"parlant.client": mock_parlant_module}):
            with patch(
                "thenvoi.integrations.parlant.session_manager.ParlantSessionManager"
            ):
                await adapter.on_started(
                    agent_name="TestBot", agent_description="A test bot"
                )

        assert adapter.agent_name == "TestBot"
        assert adapter.agent_description == "A test bot"
        assert adapter._system_prompt != ""
        assert "TestBot" in adapter._system_prompt


class TestOnMessage:
    """Tests for on_message() method."""

    @pytest.mark.asyncio
    async def test_creates_session_for_room(
        self, sample_message, mock_tools, mock_parlant_client
    ):
        """Should create or get session for room."""
        mock_session_manager = AsyncMock()
        mock_session_manager.get_or_create_session = AsyncMock(
            return_value=MagicMock(session_id="session-123")
        )
        mock_session_manager.update_offset = MagicMock()

        adapter = ParlantAdapter(agent_id="test-agent")
        adapter._client = mock_parlant_client
        adapter._session_manager = mock_session_manager
        adapter._system_prompt = "Test prompt"

        await adapter.on_message(
            msg=sample_message,
            tools=mock_tools,
            history=[],
            participants_msg=None,
            is_session_bootstrap=True,
            room_id="room-123",
        )

        mock_session_manager.get_or_create_session.assert_called_once_with(
            room_id="room-123",
            customer_id="user-456",
            customer_name="Alice",
        )

    @pytest.mark.asyncio
    async def test_sends_customer_event_to_parlant(
        self, sample_message, mock_tools, mock_parlant_client
    ):
        """Should send customer message as event to Parlant."""
        mock_session_manager = AsyncMock()
        mock_session_manager.get_or_create_session = AsyncMock(
            return_value=MagicMock(session_id="session-123")
        )
        mock_session_manager.update_offset = MagicMock()

        adapter = ParlantAdapter(agent_id="test-agent")
        adapter._client = mock_parlant_client
        adapter._session_manager = mock_session_manager
        adapter._system_prompt = "Test prompt"

        await adapter.on_message(
            msg=sample_message,
            tools=mock_tools,
            history=[],
            participants_msg=None,
            is_session_bootstrap=True,
            room_id="room-123",
        )

        mock_parlant_client.sessions.create_event.assert_called_once()
        call_kwargs = mock_parlant_client.sessions.create_event.call_args[1]
        assert call_kwargs["session_id"] == "session-123"
        assert call_kwargs["kind"] == "message"
        assert call_kwargs["source"] == "customer"

    @pytest.mark.asyncio
    async def test_stores_tools_for_room(
        self, sample_message, mock_tools, mock_parlant_client
    ):
        """Should store tools for room for tool execution."""
        mock_session_manager = AsyncMock()
        mock_session_manager.get_or_create_session = AsyncMock(
            return_value=MagicMock(session_id="session-123")
        )
        mock_session_manager.update_offset = MagicMock()

        adapter = ParlantAdapter(agent_id="test-agent")
        adapter._client = mock_parlant_client
        adapter._session_manager = mock_session_manager
        adapter._system_prompt = "Test prompt"

        await adapter.on_message(
            msg=sample_message,
            tools=mock_tools,
            history=[],
            participants_msg=None,
            is_session_bootstrap=True,
            room_id="room-123",
        )

        assert "room-123" in adapter._room_tools
        assert adapter._room_tools["room-123"] is mock_tools


class TestOnCleanup:
    """Tests for on_cleanup() method."""

    @pytest.mark.asyncio
    async def test_cleans_up_session(self):
        """Should clean up Parlant session."""
        mock_session_manager = AsyncMock()

        adapter = ParlantAdapter()
        adapter._session_manager = mock_session_manager
        adapter._room_tools["room-123"] = MagicMock()

        await adapter.on_cleanup("room-123")

        mock_session_manager.cleanup_session.assert_called_once_with("room-123")
        assert "room-123" not in adapter._room_tools

    @pytest.mark.asyncio
    async def test_cleanup_nonexistent_room_is_safe(self):
        """Should handle cleanup of non-existent room."""
        mock_session_manager = AsyncMock()

        adapter = ParlantAdapter()
        adapter._session_manager = mock_session_manager

        # Should not raise
        await adapter.on_cleanup("nonexistent-room")


class TestGuidelines:
    """Tests for behavioral guidelines."""

    @pytest.mark.asyncio
    async def test_guidelines_registered_with_parlant(
        self, mock_parlant_client, mock_parlant_module
    ):
        """Should register guidelines with Parlant server."""
        guidelines = [
            {"condition": "Condition 1", "action": "Action 1"},
            {"condition": "Condition 2", "action": "Action 2"},
        ]
        adapter = ParlantAdapter(guidelines=guidelines)

        with patch.dict(sys.modules, {"parlant.client": mock_parlant_module}):
            with patch(
                "thenvoi.integrations.parlant.session_manager.ParlantSessionManager"
            ):
                await adapter.on_started(
                    agent_name="TestBot", agent_description="A test bot"
                )

        # Verify guidelines were registered
        calls = mock_parlant_client.agents.create_guideline.call_args_list
        assert len(calls) == 2

        # Check first guideline
        first_call = calls[0][1]
        assert first_call["condition"] == "Condition 1"
        assert first_call["action"] == "Action 1"

        # Check second guideline
        second_call = calls[1][1]
        assert second_call["condition"] == "Condition 2"
        assert second_call["action"] == "Action 2"

    @pytest.mark.asyncio
    async def test_skips_invalid_guidelines(
        self, mock_parlant_client, mock_parlant_module
    ):
        """Should skip guidelines with missing condition or action."""
        guidelines = [
            {"condition": "Valid", "action": "Valid action"},
            {"condition": "Missing action"},  # Invalid
            {"action": "Missing condition"},  # Invalid
            {},  # Invalid
        ]
        adapter = ParlantAdapter(guidelines=guidelines)

        with patch.dict(sys.modules, {"parlant.client": mock_parlant_module}):
            with patch(
                "thenvoi.integrations.parlant.session_manager.ParlantSessionManager"
            ):
                await adapter.on_started(
                    agent_name="TestBot", agent_description="A test bot"
                )

        # Only valid guideline should be registered
        assert mock_parlant_client.agents.create_guideline.call_count == 1

    def test_adapter_without_guidelines(self):
        """Should work without guidelines (basic mode)."""
        adapter = ParlantAdapter()

        assert adapter.guidelines == []


class TestCleanupAll:
    """Tests for cleanup_all() method."""

    @pytest.mark.asyncio
    async def test_cleans_up_all_sessions(self):
        """Should cleanup all sessions and close client."""
        mock_session_manager = AsyncMock()
        mock_client = AsyncMock()
        mock_client.close = AsyncMock()

        adapter = ParlantAdapter()
        adapter._session_manager = mock_session_manager
        adapter._client = mock_client
        adapter._room_tools["room-1"] = MagicMock()
        adapter._room_tools["room-2"] = MagicMock()

        await adapter.cleanup_all()

        mock_session_manager.cleanup_all.assert_called_once()
        mock_client.close.assert_called_once()
        assert len(adapter._room_tools) == 0


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_reports_error_on_failure(
        self, sample_message, mock_tools, mock_parlant_client
    ):
        """Should report error when processing fails during message send."""
        mock_session_manager = AsyncMock()
        mock_session_manager.get_or_create_session = AsyncMock(
            return_value=MagicMock(session_id="session-123")
        )
        mock_session_manager.update_offset = MagicMock()

        # Make the Parlant client fail when sending the event
        mock_parlant_client.sessions.create_event = AsyncMock(
            side_effect=Exception("API error")
        )

        adapter = ParlantAdapter(agent_id="test-agent")
        adapter._client = mock_parlant_client
        adapter._session_manager = mock_session_manager
        adapter._system_prompt = "Test prompt"

        with pytest.raises(Exception, match="API error"):
            await adapter.on_message(
                msg=sample_message,
                tools=mock_tools,
                history=[],
                participants_msg=None,
                is_session_bootstrap=True,
                room_id="room-123",
            )

        # Should have tried to report error
        mock_tools.send_event.assert_called()

    @pytest.mark.asyncio
    async def test_raises_import_error_without_parlant(self):
        """Should raise ImportError if parlant is not installed."""
        adapter = ParlantAdapter()

        # Simulate parlant not being installed
        with patch.dict(sys.modules, {"parlant.client": None}):
            with pytest.raises(ImportError, match="parlant"):
                await adapter.on_started(
                    agent_name="TestBot", agent_description="A test bot"
                )
