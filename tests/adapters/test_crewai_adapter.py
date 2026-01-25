"""Tests for CrewAIAdapter."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel, Field

from thenvoi.core.types import PlatformMessage

if TYPE_CHECKING:
    from thenvoi.adapters.crewai import CrewAIAdapter as CrewAIAdapterType


class MockBaseTool:
    """Mock BaseTool class for testing."""

    name: str = ""
    description: str = ""

    def __init__(self):
        pass


@pytest.fixture
def crewai_mocks():
    """Set up CrewAI mocks with proper cleanup."""
    import sys

    # Store original modules
    original_modules = {}
    modules_to_mock = ["crewai", "crewai.tools", "nest_asyncio"]

    for mod in modules_to_mock:
        original_modules[mod] = sys.modules.get(mod)

    # Create mocks
    mock_crewai_module = MagicMock()
    mock_crewai_tools_module = MagicMock()
    mock_nest_asyncio = MagicMock()

    mock_crewai_module.Agent = MagicMock()
    mock_crewai_module.LLM = MagicMock()
    mock_crewai_tools_module.BaseTool = MockBaseTool

    sys.modules["crewai"] = mock_crewai_module
    sys.modules["crewai.tools"] = mock_crewai_tools_module
    sys.modules["nest_asyncio"] = mock_nest_asyncio

    yield mock_crewai_module

    # Cleanup: restore original modules or remove mocks
    for mod in modules_to_mock:
        if original_modules[mod] is not None:
            sys.modules[mod] = original_modules[mod]
        else:
            sys.modules.pop(mod, None)

    # Also remove any cached adapter module to force re-import
    sys.modules.pop("thenvoi.adapters.crewai", None)


@pytest.fixture
def CrewAIAdapter(crewai_mocks) -> type["CrewAIAdapterType"]:
    """Import and return CrewAIAdapter with mocks applied."""
    import importlib

    module = importlib.import_module("thenvoi.adapters.crewai")
    return module.CrewAIAdapter


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

    def test_default_initialization(self, CrewAIAdapter):
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

    def test_custom_initialization(self, CrewAIAdapter):
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
    async def test_creates_crewai_agent(self, CrewAIAdapter, crewai_mocks):
        """Should create CrewAI agent after start."""
        crewai_mocks.Agent.reset_mock()

        adapter = CrewAIAdapter()
        await adapter.on_started(agent_name="TestBot", agent_description="A test bot")

        assert adapter.agent_name == "TestBot"
        assert adapter.agent_description == "A test bot"
        crewai_mocks.Agent.assert_called_once()

    @pytest.mark.asyncio
    async def test_uses_custom_role_goal_backstory(self, CrewAIAdapter, crewai_mocks):
        """Should use custom role, goal, and backstory in agent creation."""
        crewai_mocks.Agent.reset_mock()

        adapter = CrewAIAdapter(
            role="Research Analyst",
            goal="Find information",
            backstory="Expert researcher",
        )

        await adapter.on_started(agent_name="TestBot", agent_description="")

        call_kwargs = crewai_mocks.Agent.call_args[1]
        assert call_kwargs["role"] == "Research Analyst"
        assert call_kwargs["goal"] == "Find information"
        assert "Expert researcher" in call_kwargs["backstory"]

    @pytest.mark.asyncio
    async def test_uses_agent_name_as_default_role(self, CrewAIAdapter, crewai_mocks):
        """Should use agent name as role if role not provided."""
        crewai_mocks.Agent.reset_mock()

        adapter = CrewAIAdapter()
        await adapter.on_started(agent_name="TestBot", agent_description="A test bot")

        call_kwargs = crewai_mocks.Agent.call_args[1]
        assert call_kwargs["role"] == "TestBot"

    @pytest.mark.asyncio
    async def test_creates_platform_tools(self, CrewAIAdapter, crewai_mocks):
        """Should create 7 platform tools for CrewAI."""
        crewai_mocks.Agent.reset_mock()

        adapter = CrewAIAdapter()
        await adapter.on_started(agent_name="TestBot", agent_description="A test bot")

        call_kwargs = crewai_mocks.Agent.call_args[1]
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

    @pytest.mark.asyncio
    async def test_includes_platform_instructions_in_backstory(
        self, CrewAIAdapter, crewai_mocks
    ):
        """Should include platform instructions in the backstory."""
        crewai_mocks.Agent.reset_mock()

        adapter = CrewAIAdapter()
        await adapter.on_started(agent_name="TestBot", agent_description="A test bot")

        call_kwargs = crewai_mocks.Agent.call_args[1]
        backstory = call_kwargs["backstory"]

        # Check that platform instructions are included
        assert "Multi-participant chat on Thenvoi platform" in backstory
        assert "send_message" in backstory
        assert "lookup_peers" in backstory


class TestOnMessage:
    """Tests for on_message() method."""

    @pytest.mark.asyncio
    async def test_initializes_history_on_bootstrap(
        self, CrewAIAdapter, sample_message, mock_tools, mock_crewai_agent
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
        self, CrewAIAdapter, sample_message, mock_tools, mock_crewai_agent
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
        self, CrewAIAdapter, sample_message, mock_tools, mock_crewai_agent
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
    async def test_cleans_up_room_history_and_tools(
        self, CrewAIAdapter, mock_crewai_agent
    ):
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
    async def test_cleanup_nonexistent_room_is_safe(self, CrewAIAdapter):
        """Should handle cleanup of non-existent room."""
        adapter = CrewAIAdapter()

        # Should not raise
        await adapter.on_cleanup("nonexistent-room")


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_reports_error_on_kickoff_failure(
        self, CrewAIAdapter, sample_message, mock_tools, mock_crewai_agent
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
    async def test_verbose_mode_passed_to_agent(self, CrewAIAdapter, crewai_mocks):
        """Verbose mode should be passed to CrewAI agent."""
        crewai_mocks.Agent.reset_mock()

        adapter = CrewAIAdapter(verbose=True)
        await adapter.on_started("TestBot", "Test bot")

        call_kwargs = crewai_mocks.Agent.call_args[1]
        assert call_kwargs["verbose"] is True


class TestRoomTools:
    """Tests for room-specific tools storage."""

    @pytest.mark.asyncio
    async def test_stores_tools_per_room(
        self, CrewAIAdapter, sample_message, mock_tools, mock_crewai_agent
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
        self, CrewAIAdapter, sample_message, mock_tools, mock_crewai_agent
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


class TestToolExecution:
    """Tests for individual tool execution.

    Note: These tests verify tool schemas and error handling for unknown rooms.
    Full tool execution tests require proper nest_asyncio setup which is mocked
    in the test environment. The actual async tool execution is tested via
    integration tests with real dependencies.
    """

    def test_tool_returns_error_for_unknown_room(self, CrewAIAdapter, crewai_mocks):
        """Should return error when room_id is not found."""
        import asyncio

        crewai_mocks.Agent.reset_mock()

        adapter = CrewAIAdapter()
        # Run on_started synchronously
        asyncio.run(adapter.on_started("TestBot", "Test bot"))

        call_kwargs = crewai_mocks.Agent.call_args[1]
        tools = call_kwargs["tools"]
        send_message_tool = next(t for t in tools if t.name == "send_message")

        # Don't add tools for the room - this returns early with error
        result = send_message_tool._run(
            room_id="unknown-room", content="Hello!", mentions="[]"
        )

        result_data = json.loads(result)
        assert result_data["status"] == "error"
        assert "No tools for room" in result_data["message"]

    @pytest.mark.asyncio
    async def test_all_tools_have_correct_schemas(self, CrewAIAdapter, crewai_mocks):
        """Should create tools with correct input schemas."""
        crewai_mocks.Agent.reset_mock()

        adapter = CrewAIAdapter()
        await adapter.on_started("TestBot", "Test bot")

        call_kwargs = crewai_mocks.Agent.call_args[1]
        tools = call_kwargs["tools"]

        # Verify send_message schema
        send_message = next(t for t in tools if t.name == "send_message")
        assert send_message.args_schema is not None
        schema_fields = send_message.args_schema.model_fields
        assert "room_id" in schema_fields
        assert "content" in schema_fields
        assert "mentions" in schema_fields

        # Verify add_participant schema
        add_participant = next(t for t in tools if t.name == "add_participant")
        schema_fields = add_participant.args_schema.model_fields
        assert "room_id" in schema_fields
        assert "participant_name" in schema_fields
        assert "role" in schema_fields

        # Verify lookup_peers schema
        lookup_peers = next(t for t in tools if t.name == "lookup_peers")
        schema_fields = lookup_peers.args_schema.model_fields
        assert "room_id" in schema_fields
        assert "page" in schema_fields
        assert "page_size" in schema_fields

    @pytest.mark.asyncio
    async def test_send_event_message_type_validation(
        self, CrewAIAdapter, crewai_mocks
    ):
        """Should have Literal type for message_type in send_event."""
        crewai_mocks.Agent.reset_mock()

        adapter = CrewAIAdapter()
        await adapter.on_started("TestBot", "Test bot")

        call_kwargs = crewai_mocks.Agent.call_args[1]
        tools = call_kwargs["tools"]

        send_event = next(t for t in tools if t.name == "send_event")
        schema_fields = send_event.args_schema.model_fields

        # Check that message_type field exists and has annotation
        assert "message_type" in schema_fields
        message_type_field = schema_fields["message_type"]
        assert message_type_field.default == "thought"


class TestExecutionReporting:
    """Tests for execution reporting configuration."""

    @pytest.mark.asyncio
    async def test_execution_reporting_flag_stored(self, CrewAIAdapter, crewai_mocks):
        """Execution reporting flag should be stored in adapter."""
        adapter_enabled = CrewAIAdapter(enable_execution_reporting=True)
        adapter_disabled = CrewAIAdapter(enable_execution_reporting=False)

        assert adapter_enabled.enable_execution_reporting is True
        assert adapter_disabled.enable_execution_reporting is False


class TestLazyNestAsyncio:
    """Tests for lazy nest_asyncio initialization."""

    def test_nest_asyncio_not_applied_on_import(self, crewai_mocks):
        """nest_asyncio should not be applied at import time."""
        import importlib
        import sys

        # Force reimport
        sys.modules.pop("thenvoi.adapters.crewai", None)

        # Reset the mock to track calls
        crewai_mocks_nest = sys.modules["nest_asyncio"]
        crewai_mocks_nest.reset_mock()

        # Import the module
        importlib.import_module("thenvoi.adapters.crewai")

        # nest_asyncio.apply() should NOT have been called at import
        crewai_mocks_nest.apply.assert_not_called()

    def test_ensure_nest_asyncio_applies_once(self, CrewAIAdapter, crewai_mocks):
        """_ensure_nest_asyncio should only apply patch once."""
        import importlib
        import sys

        module = importlib.import_module("thenvoi.adapters.crewai")

        # Reset the flag and mock
        module._nest_asyncio_applied = False
        nest_mock = sys.modules["nest_asyncio"]
        nest_mock.reset_mock()

        # Call ensure twice
        module._ensure_nest_asyncio()
        module._ensure_nest_asyncio()

        # Should only call apply() once
        assert nest_mock.apply.call_count == 1


class TestPlatformInstructionsConstant:
    """Tests for platform instructions constant."""

    def test_platform_instructions_is_constant(self, CrewAIAdapter):
        """PLATFORM_INSTRUCTIONS should be a module-level constant."""
        import importlib

        module = importlib.import_module("thenvoi.adapters.crewai")

        assert hasattr(module, "PLATFORM_INSTRUCTIONS")
        assert isinstance(module.PLATFORM_INSTRUCTIONS, str)
        assert len(module.PLATFORM_INSTRUCTIONS) > 100  # Non-trivial content

    def test_platform_instructions_contains_key_info(self, CrewAIAdapter):
        """PLATFORM_INSTRUCTIONS should contain key platform information."""
        import importlib

        module = importlib.import_module("thenvoi.adapters.crewai")

        instructions = module.PLATFORM_INSTRUCTIONS

        # Check for key sections
        assert "Environment" in instructions
        assert "send_message" in instructions
        assert "lookup_peers" in instructions
        assert "add_participant" in instructions
