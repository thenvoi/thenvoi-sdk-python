"""Claude SDK adapter-specific tests.

Tests for Claude SDK adapter-specific behavior that isn't covered by conformance tests:
- Room tools storage (MCP server access pattern)
- THENVOI_TOOLS constant validation
- MCP server and session manager lifecycle
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from thenvoi.adapters.claude_sdk import ClaudeSDKAdapter, THENVOI_TOOLS


class TestRoomToolsStorage:
    """Tests for room tools storage (MCP server access pattern)."""

    def test_stores_tools_per_room(self):
        """Should store tools per room for MCP server access."""
        adapter = ClaudeSDKAdapter()

        mock_tools_1 = MagicMock()
        mock_tools_2 = MagicMock()

        adapter._room_tools["room-1"] = mock_tools_1
        adapter._room_tools["room-2"] = mock_tools_2

        assert adapter._room_tools["room-1"] is mock_tools_1
        assert adapter._room_tools["room-2"] is mock_tools_2

    def test_room_tools_isolated_per_room(self):
        """Each room should have isolated tools."""
        adapter = ClaudeSDKAdapter()

        mock_tools_1 = MagicMock(name="tools_for_room_1")
        mock_tools_2 = MagicMock(name="tools_for_room_2")

        adapter._room_tools["room-1"] = mock_tools_1
        adapter._room_tools["room-2"] = mock_tools_2

        # Verify they are different objects
        assert adapter._room_tools["room-1"] is not adapter._room_tools["room-2"]
        assert adapter._room_tools["room-1"]._mock_name == "tools_for_room_1"
        assert adapter._room_tools["room-2"]._mock_name == "tools_for_room_2"

    def test_room_tools_starts_empty(self):
        """Room tools dict should start empty."""
        adapter = ClaudeSDKAdapter()
        assert adapter._room_tools == {}


class TestThenvoiTools:
    """Tests for THENVOI_TOOLS constant (MCP tool names)."""

    def test_thenvoi_tools_list_contains_all_platform_tools(self):
        """Should define all expected MCP tool names."""
        expected = [
            "mcp__thenvoi__send_message",
            "mcp__thenvoi__send_event",
            "mcp__thenvoi__add_participant",
            "mcp__thenvoi__remove_participant",
            "mcp__thenvoi__get_participants",
            "mcp__thenvoi__lookup_peers",
            "mcp__thenvoi__create_chatroom",
        ]

        assert THENVOI_TOOLS == expected

    def test_thenvoi_tools_follows_mcp_naming_convention(self):
        """All tools should follow mcp__<server>__<tool> naming convention."""
        for tool_name in THENVOI_TOOLS:
            assert tool_name.startswith("mcp__thenvoi__")
            # Should have tool name after prefix
            tool_part = tool_name.replace("mcp__thenvoi__", "")
            assert len(tool_part) > 0
            # Should be snake_case
            assert tool_part == tool_part.lower()

    def test_thenvoi_tools_has_correct_count(self):
        """Should have exactly 7 platform tools."""
        assert len(THENVOI_TOOLS) == 7


class TestMCPServerLifecycle:
    """Tests for MCP server and session manager lifecycle."""

    @pytest.mark.asyncio
    async def test_creates_mcp_server_on_started(self):
        """Should create MCP server on on_started()."""
        adapter = ClaudeSDKAdapter()

        with patch(
            "thenvoi.adapters.claude_sdk.ClaudeSessionManager"
        ) as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager_class.return_value = mock_manager

            await adapter.on_started(
                agent_name="TestBot", agent_description="A test bot"
            )

            assert adapter._mcp_server is not None

    @pytest.mark.asyncio
    async def test_creates_session_manager_on_started(self):
        """Should create session manager on on_started()."""
        adapter = ClaudeSDKAdapter()

        with patch(
            "thenvoi.adapters.claude_sdk.ClaudeSessionManager"
        ) as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager_class.return_value = mock_manager

            await adapter.on_started(
                agent_name="TestBot", agent_description="A test bot"
            )

            assert adapter._session_manager is not None


class TestOnCleanup:
    """Tests for on_cleanup() method."""

    @pytest.mark.asyncio
    async def test_cleans_up_session_and_tools(self):
        """Should cleanup session and remove room tools."""
        adapter = ClaudeSDKAdapter()

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

    @pytest.mark.asyncio
    async def test_cleanup_nonexistent_room_is_safe(self):
        """Should handle cleanup of non-existent room gracefully."""
        adapter = ClaudeSDKAdapter()
        mock_session_manager = AsyncMock()
        adapter._session_manager = mock_session_manager

        # Should not raise
        await adapter.on_cleanup("nonexistent-room")


class TestCleanupAll:
    """Tests for cleanup_all() method."""

    @pytest.mark.asyncio
    async def test_cleans_up_all_sessions(self):
        """Should stop session manager and clear all room tools."""
        adapter = ClaudeSDKAdapter()

        mock_session_manager = AsyncMock()
        adapter._session_manager = mock_session_manager
        adapter._room_tools["room-1"] = MagicMock()
        adapter._room_tools["room-2"] = MagicMock()

        await adapter.cleanup_all()

        mock_session_manager.stop.assert_awaited_once()
        assert len(adapter._room_tools) == 0

    @pytest.mark.asyncio
    async def test_cleanup_all_without_session_manager_is_safe(self):
        """Should handle cleanup_all when session manager not initialized."""
        adapter = ClaudeSDKAdapter()
        adapter._room_tools["room-1"] = MagicMock()

        # Should not raise
        await adapter.cleanup_all()

        assert len(adapter._room_tools) == 0


class TestCustomTools:
    """Tests for custom tool support (CustomToolDef -> MCP)."""

    def test_accepts_additional_tools_parameter(self):
        """Adapter should accept list of CustomToolDef tuples."""
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
            a: int

        class Tool2Input(BaseModel):
            b: str

        async def tool1(args: Tool1Input) -> int:
            return args.a + 1

        async def tool2(args: Tool2Input) -> str:
            return args.b.upper()

        adapter = ClaudeSDKAdapter(
            additional_tools=[(Tool1Input, tool1), (Tool2Input, tool2)],
        )

        assert len(adapter._custom_tools) == 2

    @pytest.mark.asyncio
    async def test_custom_tools_added_to_allowed_tools(self):
        """allowed_tools should contain mcp__thenvoi__<tool_name> for each custom tool."""
        from pydantic import BaseModel, Field

        class EchoInput(BaseModel):
            """Echo the message."""

            message: str = Field(description="Message to echo")

        async def echo(args: EchoInput) -> str:
            return f"Echo: {args.message}"

        adapter = ClaudeSDKAdapter(
            additional_tools=[(EchoInput, echo)],
        )

        with patch(
            "thenvoi.adapters.claude_sdk.ClaudeSessionManager"
        ) as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager_class.return_value = mock_manager

            await adapter.on_started(
                agent_name="TestBot", agent_description="A test bot"
            )

            # Check the ClaudeAgentOptions passed to ClaudeSessionManager
            call_args = mock_manager_class.call_args
            sdk_options = call_args[0][0]
            assert "mcp__thenvoi__echo" in sdk_options.allowed_tools

    @pytest.mark.asyncio
    async def test_custom_tools_registered_in_mcp_server(self):
        """_create_mcp_server should be called and MCP server should not be None."""
        from pydantic import BaseModel, Field

        class EchoInput(BaseModel):
            """Echo the message."""

            message: str = Field(description="Message to echo")

        async def echo(args: EchoInput) -> str:
            return f"Echo: {args.message}"

        adapter = ClaudeSDKAdapter(
            additional_tools=[(EchoInput, echo)],
        )

        with patch(
            "thenvoi.adapters.claude_sdk.ClaudeSessionManager"
        ) as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager_class.return_value = mock_manager

            await adapter.on_started(
                agent_name="TestBot", agent_description="A test bot"
            )

        assert adapter._mcp_server is not None

    def test_tool_name_derived_from_input_model(self):
        """get_custom_tool_name(EchoInput) → 'echo', get_custom_tool_name(CalculatorInput) → 'calculator'."""
        from pydantic import BaseModel

        from thenvoi.runtime.custom_tools import get_custom_tool_name

        class EchoInput(BaseModel):
            message: str

        class CalculatorInput(BaseModel):
            a: int
            b: int

        assert get_custom_tool_name(EchoInput) == "echo"
        assert get_custom_tool_name(CalculatorInput) == "calculator"
