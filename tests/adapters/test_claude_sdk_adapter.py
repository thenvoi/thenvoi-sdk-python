"""Tests for ClaudeSDKAdapter."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from thenvoi.adapters.claude_sdk import (
    ClaudeSDKAdapter,
    THENVOI_TOOLS,
    THENVOI_BASE_TOOLS,
    THENVOI_MEMORY_TOOLS,
)
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


class TestInitialization:
    """Tests for adapter initialization."""

    def test_default_initialization(self):
        """Should initialize with default values."""
        adapter = ClaudeSDKAdapter()

        assert adapter.model == "claude-sonnet-4-5-20250929"
        assert adapter.custom_section is None
        assert adapter.max_thinking_tokens is None
        assert adapter.permission_mode == "acceptEdits"
        assert adapter.enable_execution_reporting is False
        assert adapter.enable_memory_tools is False

    def test_custom_initialization(self):
        """Should accept custom parameters."""
        adapter = ClaudeSDKAdapter(
            model="claude-opus-4-20250514",
            custom_section="Be helpful.",
            max_thinking_tokens=10000,
            permission_mode="bypassPermissions",
            enable_execution_reporting=True,
            enable_memory_tools=True,
        )

        assert adapter.model == "claude-opus-4-20250514"
        assert adapter.custom_section == "Be helpful."
        assert adapter.max_thinking_tokens == 10000
        assert adapter.permission_mode == "bypassPermissions"
        assert adapter.enable_execution_reporting is True
        assert adapter.enable_memory_tools is True


class TestOnStarted:
    """Tests for on_started() method."""

    @pytest.mark.asyncio
    async def test_creates_mcp_server_and_session_manager(self):
        """Should create MCP server and session manager on start."""
        adapter = ClaudeSDKAdapter()

        # Mock the session manager
        with patch(
            "thenvoi.adapters.claude_sdk.ClaudeSessionManager"
        ) as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager_class.return_value = mock_manager

            await adapter.on_started(
                agent_name="TestBot", agent_description="A test bot"
            )

            assert adapter.agent_name == "TestBot"
            assert adapter._session_manager is not None
            assert adapter._mcp_server is not None


class TestRoomToolsStorage:
    """Tests for room tools storage."""

    def test_stores_tools_per_room(self):
        """Should store tools per room for MCP server access."""
        adapter = ClaudeSDKAdapter()

        mock_tools_1 = MagicMock()
        mock_tools_2 = MagicMock()

        adapter._room_tools["room-1"] = mock_tools_1
        adapter._room_tools["room-2"] = mock_tools_2

        assert adapter._room_tools["room-1"] is mock_tools_1
        assert adapter._room_tools["room-2"] is mock_tools_2


class TestOnCleanup:
    """Tests for on_cleanup() method."""

    @pytest.mark.asyncio
    async def test_cleans_up_session_and_tools(self):
        """Should cleanup session and remove room tools."""
        adapter = ClaudeSDKAdapter()

        # Set up mock session manager
        mock_session_manager = AsyncMock()
        adapter._session_manager = mock_session_manager
        adapter._room_tools["room-123"] = MagicMock()

        await adapter.on_cleanup("room-123")

        mock_session_manager.cleanup_session.assert_awaited_once_with("room-123")
        assert "room-123" not in adapter._room_tools

    @pytest.mark.asyncio
    async def test_cleanup_without_session_manager_is_safe(self):
        """Should handle cleanup when session manager not initialized."""
        adapter = ClaudeSDKAdapter()
        adapter._room_tools["room-123"] = MagicMock()

        # Should not raise
        await adapter.on_cleanup("room-123")

        assert "room-123" not in adapter._room_tools


class TestCleanupAll:
    """Tests for cleanup_all() method."""

    @pytest.mark.asyncio
    async def test_cleans_up_all_sessions(self):
        """Should stop session manager and clear room tools."""
        adapter = ClaudeSDKAdapter()

        mock_session_manager = AsyncMock()
        adapter._session_manager = mock_session_manager
        adapter._room_tools["room-1"] = MagicMock()
        adapter._room_tools["room-2"] = MagicMock()

        await adapter.cleanup_all()

        mock_session_manager.stop.assert_awaited_once()
        assert len(adapter._room_tools) == 0


class TestThenvoiTools:
    """Tests for Thenvoi tool names constants."""

    def test_thenvoi_base_tools_list(self):
        """Should define base platform tools (always included)."""
        expected = [
            "mcp__thenvoi__send_message",
            "mcp__thenvoi__send_event",
            "mcp__thenvoi__add_participant",
            "mcp__thenvoi__remove_participant",
            "mcp__thenvoi__get_participants",
            "mcp__thenvoi__lookup_peers",
            "mcp__thenvoi__create_chatroom",
            # Contact management tools
            "mcp__thenvoi__list_contacts",
            "mcp__thenvoi__add_contact",
            "mcp__thenvoi__remove_contact",
            "mcp__thenvoi__list_contact_requests",
            "mcp__thenvoi__respond_contact_request",
        ]

        assert THENVOI_BASE_TOOLS == expected

    def test_thenvoi_memory_tools_list(self):
        """Should define memory tools (enterprise only - opt-in)."""
        expected = [
            "mcp__thenvoi__list_memories",
            "mcp__thenvoi__store_memory",
            "mcp__thenvoi__get_memory",
            "mcp__thenvoi__supersede_memory",
            "mcp__thenvoi__archive_memory",
        ]

        assert THENVOI_MEMORY_TOOLS == expected

    def test_thenvoi_tools_combines_base_and_memory(self):
        """THENVOI_TOOLS should combine base and memory tools."""
        assert THENVOI_TOOLS == THENVOI_BASE_TOOLS + THENVOI_MEMORY_TOOLS
        assert len(THENVOI_TOOLS) == 17  # 12 base + 5 memory


class TestCustomTools:
    """Tests for custom tool support (CustomToolDef → MCP)."""

    def test_accepts_additional_tools_parameter(self):
        """Adapter accepts list of CustomToolDef tuples."""
        from pydantic import BaseModel, Field

        class EchoInput(BaseModel):
            """Echo the message."""

            message: str = Field(description="Message to echo")

        async def echo(args: EchoInput) -> str:
            return f"Echo: {args.message}"

        adapter = ClaudeSDKAdapter(
            additional_tools=[(EchoInput, echo)],
        )

        assert len(adapter._custom_tools) == 1
        assert adapter._custom_tools[0][0] is EchoInput

    def test_empty_additional_tools_by_default(self):
        """Should have empty custom tools list by default."""
        adapter = ClaudeSDKAdapter()

        assert adapter._custom_tools == []

    def test_multiple_custom_tools(self):
        """Should accept multiple custom tools."""
        from pydantic import BaseModel

        class Tool1Input(BaseModel):
            """Tool 1."""

            x: int

        class Tool2Input(BaseModel):
            """Tool 2."""

            y: str

        def tool1(args: Tool1Input) -> int:
            return args.x + 1

        def tool2(args: Tool2Input) -> str:
            return args.y.upper()

        adapter = ClaudeSDKAdapter(
            additional_tools=[(Tool1Input, tool1), (Tool2Input, tool2)],
        )

        assert len(adapter._custom_tools) == 2

    @pytest.mark.asyncio
    async def test_custom_tools_added_to_allowed_tools(self):
        """Custom tools should be added to allowed_tools list."""
        from pydantic import BaseModel

        class CalculatorInput(BaseModel):
            """Perform calculations."""

            a: float
            b: float

        def calc(args: CalculatorInput) -> float:
            return args.a + args.b

        adapter = ClaudeSDKAdapter(
            additional_tools=[(CalculatorInput, calc)],
        )

        # Mock the session manager creation
        with patch(
            "thenvoi.adapters.claude_sdk.ClaudeSessionManager"
        ) as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager_class.return_value = mock_manager

            # Capture the ClaudeAgentOptions passed to session manager
            await adapter.on_started(
                agent_name="TestBot", agent_description="A test bot"
            )

            # Get the options passed to ClaudeSessionManager
            call_args = mock_manager_class.call_args
            sdk_options = call_args[0][0]

            # Verify custom tool is in allowed_tools
            assert "mcp__thenvoi__calculator" in sdk_options.allowed_tools
            # Platform tools should still be there
            assert "mcp__thenvoi__send_message" in sdk_options.allowed_tools

    @pytest.mark.asyncio
    async def test_custom_tools_registered_in_mcp_server(self):
        """Custom tools should be registered in MCP server (memory tools disabled)."""
        from pydantic import BaseModel

        class EchoInput(BaseModel):
            """Echo tool."""

            message: str

        async def echo(args: EchoInput) -> str:
            return f"Echo: {args.message}"

        adapter = ClaudeSDKAdapter(
            additional_tools=[(EchoInput, echo)],
        )

        # Create the MCP server
        with patch(
            "thenvoi.adapters.claude_sdk.create_sdk_mcp_server"
        ) as mock_create_server:
            mock_create_server.return_value = MagicMock()

            adapter._create_mcp_server()

            # Verify create_sdk_mcp_server was called with extra tools
            call_args = mock_create_server.call_args
            tools_list = call_args[1]["tools"]

            # Should have 12 base platform tools + 1 custom tool = 13
            # (7 basic + 5 contact + 1 custom = 13, memory tools disabled by default)
            assert len(tools_list) == 13

    @pytest.mark.asyncio
    async def test_custom_tools_registered_with_memory_tools_enabled(self):
        """Custom tools should be registered in MCP server (memory tools enabled)."""
        from pydantic import BaseModel

        class EchoInput(BaseModel):
            """Echo tool."""

            message: str

        async def echo(args: EchoInput) -> str:
            return f"Echo: {args.message}"

        adapter = ClaudeSDKAdapter(
            additional_tools=[(EchoInput, echo)],
            enable_memory_tools=True,
        )

        # Create the MCP server
        with patch(
            "thenvoi.adapters.claude_sdk.create_sdk_mcp_server"
        ) as mock_create_server:
            mock_create_server.return_value = MagicMock()

            adapter._create_mcp_server()

            # Verify create_sdk_mcp_server was called with extra tools
            call_args = mock_create_server.call_args
            tools_list = call_args[1]["tools"]

            # Should have 17 platform tools + 1 custom tool = 18
            # (7 basic + 5 contact + 5 memory + 1 custom = 18)
            assert len(tools_list) == 18

    def test_tool_name_derived_from_input_model(self):
        """Tool name should be derived from Pydantic model class name."""
        from thenvoi.runtime.custom_tools import get_custom_tool_name
        from pydantic import BaseModel

        class MyCustomToolInput(BaseModel):
            """A custom tool."""

            value: str

        name = get_custom_tool_name(MyCustomToolInput)
        assert name == "mycustomtool"

        class CalculatorInput(BaseModel):
            """Calculator."""

            x: int

        name = get_custom_tool_name(CalculatorInput)
        assert name == "calculator"
