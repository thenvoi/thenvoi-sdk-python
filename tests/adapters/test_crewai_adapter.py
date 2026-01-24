"""Tests for CrewAIAdapter."""

from datetime import datetime, timezone
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from thenvoi.core.types import PlatformMessage


# Create mock CrewAI module and classes
mock_crewai_module = MagicMock()
mock_crewai_tools_module = MagicMock()


class MockBaseTool:
    """Mock BaseTool class for testing."""

    name: str = ""
    description: str = ""

    def __init__(self):
        pass


mock_crewai_module.Agent = MagicMock()
mock_crewai_module.LLM = MagicMock()
mock_crewai_tools_module.BaseTool = MockBaseTool

# Patch before any imports from the adapter
sys.modules["crewai"] = mock_crewai_module
sys.modules["crewai.tools"] = mock_crewai_tools_module

# Now import the adapter
from thenvoi.adapters.crewai import CrewAIAdapter


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
    tools.add_participant = AsyncMock(
        return_value={"id": "123", "name": "Test", "status": "added"}
    )
    tools.remove_participant = AsyncMock(
        return_value={"id": "123", "name": "Test", "status": "removed"}
    )
    tools.get_participants = AsyncMock(
        return_value=[{"id": "123", "name": "Alice", "type": "User"}]
    )
    tools.lookup_peers = AsyncMock(
        return_value={
            "peers": [],
            "metadata": {
                "page": 1,
                "page_size": 50,
                "total_count": 0,
                "total_pages": 1,
            },
        }
    )
    tools.create_chatroom = AsyncMock(return_value="new-room-123")
    return tools


@pytest.fixture
def mock_crewai_agent():
    """Create a mock CrewAI agent."""
    mock_result = MagicMock()
    mock_result.raw = "Hello! I'm here to help."

    mock_agent = MagicMock()
    mock_agent.kickoff_async = AsyncMock(return_value=mock_result)
    return mock_agent


class TestInitialization:
    """Tests for adapter initialization."""

    def test_default_initialization(self):
        """Should initialize with default values."""
        adapter = CrewAIAdapter()

        assert adapter.model == "gpt-4o"
        assert adapter.role is None
        assert adapter.goal is None
        assert adapter.backstory is None
        assert adapter.enable_execution_reporting is False
        assert adapter.verbose is False
        assert adapter.max_iter == 20
        assert adapter.allow_delegation is False
        assert adapter.history_converter is not None

    def test_custom_initialization(self):
        """Should accept custom parameters."""
        adapter = CrewAIAdapter(
            model="gpt-4o-mini",
            role="Research Analyst",
            goal="Find and analyze information",
            backstory="Expert researcher with years of experience",
            custom_section="Be thorough.",
            enable_execution_reporting=True,
            verbose=True,
            max_iter=30,
            max_rpm=10,
            allow_delegation=True,
        )

        assert adapter.model == "gpt-4o-mini"
        assert adapter.role == "Research Analyst"
        assert adapter.goal == "Find and analyze information"
        assert adapter.backstory == "Expert researcher with years of experience"
        assert adapter.custom_section == "Be thorough."
        assert adapter.enable_execution_reporting is True
        assert adapter.verbose is True
        assert adapter.max_iter == 30
        assert adapter.max_rpm == 10
        assert adapter.allow_delegation is True


class TestOnStarted:
    """Tests for on_started() method."""

    @pytest.mark.asyncio
    async def test_creates_crewai_agent(self):
        """Should create CrewAI agent after start."""
        # Reset the mock
        mock_crewai_module.Agent.reset_mock()

        adapter = CrewAIAdapter()
        await adapter.on_started(agent_name="TestBot", agent_description="A test bot")

        assert adapter.agent_name == "TestBot"
        assert adapter.agent_description == "A test bot"
        mock_crewai_module.Agent.assert_called_once()

    @pytest.mark.asyncio
    async def test_uses_custom_role_goal_backstory(self):
        """Should use custom role, goal, and backstory in agent creation."""
        mock_crewai_module.Agent.reset_mock()

        adapter = CrewAIAdapter(
            role="Research Analyst",
            goal="Find information",
            backstory="Expert researcher",
        )

        await adapter.on_started(agent_name="TestBot", agent_description="")

        call_kwargs = mock_crewai_module.Agent.call_args[1]
        assert call_kwargs["role"] == "Research Analyst"
        assert call_kwargs["goal"] == "Find information"
        assert "Expert researcher" in call_kwargs["backstory"]

    @pytest.mark.asyncio
    async def test_uses_agent_name_as_default_role(self):
        """Should use agent name as role if role not provided."""
        mock_crewai_module.Agent.reset_mock()

        adapter = CrewAIAdapter()
        await adapter.on_started(agent_name="TestBot", agent_description="A test bot")

        call_kwargs = mock_crewai_module.Agent.call_args[1]
        assert call_kwargs["role"] == "TestBot"

    @pytest.mark.asyncio
    async def test_creates_platform_tools(self):
        """Should create 7 platform tools for CrewAI."""
        mock_crewai_module.Agent.reset_mock()

        adapter = CrewAIAdapter()
        await adapter.on_started(agent_name="TestBot", agent_description="A test bot")

        call_kwargs = mock_crewai_module.Agent.call_args[1]
        tools = call_kwargs["tools"]
        assert len(tools) == 7

        tool_names = [t.name for t in tools]
        assert "send_message" in tool_names
        assert "send_event" in tool_names
        assert "add_participant" in tool_names
        assert "remove_participant" in tool_names
        assert "get_participants" in tool_names
        assert "lookup_peers" in tool_names
        assert "create_chatroom" in tool_names


class TestOnMessage:
    """Tests for on_message() method."""

    @pytest.mark.asyncio
    async def test_initializes_history_on_bootstrap(
        self, sample_message, mock_tools, mock_crewai_agent
    ):
        """Should initialize room history on first message."""
        adapter = CrewAIAdapter()
        await adapter.on_started("TestBot", "Test bot")
        adapter._crewai_agent = mock_crewai_agent

        await adapter.on_message(
            msg=sample_message,
            tools=mock_tools,
            history=[],
            participants_msg=None,
            is_session_bootstrap=True,
            room_id="room-123",
        )

        assert "room-123" in adapter._message_history
        assert "room-123" in adapter._room_tools

    @pytest.mark.asyncio
    async def test_loads_existing_history(
        self, sample_message, mock_tools, mock_crewai_agent
    ):
        """Should load historical messages on bootstrap."""
        adapter = CrewAIAdapter()
        await adapter.on_started("TestBot", "Test bot")
        adapter._crewai_agent = mock_crewai_agent

        existing_history = [
            {"role": "user", "content": "[Bob]: Previous message"},
            {"role": "assistant", "content": "Previous response"},
        ]

        await adapter.on_message(
            msg=sample_message,
            tools=mock_tools,
            history=existing_history,
            participants_msg=None,
            is_session_bootstrap=True,
            room_id="room-123",
        )

        # Should have existing 2 + current message + response
        assert len(adapter._message_history["room-123"]) >= 3

    @pytest.mark.asyncio
    async def test_calls_kickoff_async(
        self, sample_message, mock_tools, mock_crewai_agent
    ):
        """Should call CrewAI agent's kickoff_async method."""
        adapter = CrewAIAdapter()
        await adapter.on_started("TestBot", "Test bot")
        adapter._crewai_agent = mock_crewai_agent

        await adapter.on_message(
            msg=sample_message,
            tools=mock_tools,
            history=[],
            participants_msg=None,
            is_session_bootstrap=True,
            room_id="room-123",
        )

        mock_crewai_agent.kickoff_async.assert_called_once()


class TestOnCleanup:
    """Tests for on_cleanup() method."""

    @pytest.mark.asyncio
    async def test_cleans_up_room_history_and_tools(self, mock_crewai_agent):
        """Should remove room history and tools on cleanup."""
        adapter = CrewAIAdapter()
        await adapter.on_started("TestBot", "Test bot")

        # Add some data
        adapter._message_history["room-123"] = [{"role": "user", "content": "test"}]
        adapter._room_tools["room-123"] = MagicMock()
        assert "room-123" in adapter._message_history
        assert "room-123" in adapter._room_tools

        await adapter.on_cleanup("room-123")

        assert "room-123" not in adapter._message_history
        assert "room-123" not in adapter._room_tools

    @pytest.mark.asyncio
    async def test_cleanup_nonexistent_room_is_safe(self):
        """Should handle cleanup of non-existent room."""
        adapter = CrewAIAdapter()

        # Should not raise
        await adapter.on_cleanup("nonexistent-room")


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_reports_error_on_kickoff_failure(
        self, sample_message, mock_tools, mock_crewai_agent
    ):
        """Should report error when CrewAI agent fails."""
        mock_crewai_agent.kickoff_async.side_effect = Exception("Agent Error")

        adapter = CrewAIAdapter()
        await adapter.on_started("TestBot", "Test bot")
        adapter._crewai_agent = mock_crewai_agent

        with pytest.raises(Exception, match="Agent Error"):
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


class TestVerboseMode:
    """Tests for verbose mode."""

    @pytest.mark.asyncio
    async def test_verbose_mode_passed_to_agent(self):
        """Verbose mode should be passed to CrewAI agent."""
        mock_crewai_module.Agent.reset_mock()

        adapter = CrewAIAdapter(verbose=True)
        await adapter.on_started("TestBot", "Test bot")

        call_kwargs = mock_crewai_module.Agent.call_args[1]
        assert call_kwargs["verbose"] is True


class TestRoomTools:
    """Tests for room-specific tools storage."""

    @pytest.mark.asyncio
    async def test_stores_tools_per_room(
        self, sample_message, mock_tools, mock_crewai_agent
    ):
        """Should store tools per room for CrewAI tool access."""
        adapter = CrewAIAdapter()
        await adapter.on_started("TestBot", "Test bot")
        adapter._crewai_agent = mock_crewai_agent

        await adapter.on_message(
            msg=sample_message,
            tools=mock_tools,
            history=[],
            participants_msg=None,
            is_session_bootstrap=True,
            room_id="room-123",
        )

        assert adapter._room_tools.get("room-123") is mock_tools


class TestParticipantsUpdate:
    """Tests for participants update handling."""

    @pytest.mark.asyncio
    async def test_includes_participants_update_in_message(
        self, sample_message, mock_tools, mock_crewai_agent
    ):
        """Should include participants update in the message to CrewAI."""
        adapter = CrewAIAdapter()
        await adapter.on_started("TestBot", "Test bot")
        adapter._crewai_agent = mock_crewai_agent

        await adapter.on_message(
            msg=sample_message,
            tools=mock_tools,
            history=[],
            participants_msg="Alice joined the room",
            is_session_bootstrap=True,
            room_id="room-123",
        )

        # Check that kickoff was called with messages including participants update
        call_args = mock_crewai_agent.kickoff_async.call_args
        messages = call_args[0][0]

        # Find the participants message
        found = any("Alice joined" in str(m.get("content", "")) for m in messages)
        assert found
