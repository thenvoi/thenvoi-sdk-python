"""Tests for ClaudeSDKAdapter."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from thenvoi.adapters.claude_sdk import ClaudeSDKAdapter, THENVOI_TOOLS
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

    def test_custom_initialization(self):
        """Should accept custom parameters."""
        adapter = ClaudeSDKAdapter(
            model="claude-opus-4-20250514",
            custom_section="Be helpful.",
            max_thinking_tokens=10000,
            permission_mode="bypassPermissions",
            enable_execution_reporting=True,
        )

        assert adapter.model == "claude-opus-4-20250514"
        assert adapter.custom_section == "Be helpful."
        assert adapter.max_thinking_tokens == 10000
        assert adapter.permission_mode == "bypassPermissions"
        assert adapter.enable_execution_reporting is True


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
    """Tests for Thenvoi tool names constant."""

    def test_thenvoi_tools_list(self):
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
