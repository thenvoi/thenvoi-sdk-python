"""Tests for ClaudeSessionManager."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def mock_options() -> MagicMock:
    """Create mock ClaudeAgentOptions."""
    opts = MagicMock()
    opts.model = "claude-sonnet-4-5-20250929"
    opts.system_prompt = "You are a test bot."
    opts.mcp_servers = {}
    opts.allowed_tools = []
    opts.permission_mode = "acceptEdits"
    return opts


class TestInvalidateSession:
    """Tests for invalidate_session() — evicts dead clients without disconnect."""

    @pytest.mark.asyncio
    async def test_invalidate_removes_session_without_disconnect(
        self, mock_options: MagicMock
    ) -> None:
        """invalidate_session should remove the client without calling disconnect()."""
        from thenvoi.integrations.claude_sdk.session_manager import (
            ClaudeSessionManager,
        )

        manager = ClaudeSessionManager(mock_options)
        await manager.start()

        # Inject a mock dead client directly into _sessions
        dead_client = MagicMock()
        dead_client.disconnect = AsyncMock()
        manager._sessions["room-dead"] = dead_client

        await manager.invalidate_session("room-dead")

        assert "room-dead" not in manager._sessions
        dead_client.disconnect.assert_not_awaited()

        await manager.stop()

    @pytest.mark.asyncio
    async def test_invalidate_nonexistent_room_is_noop(
        self, mock_options: MagicMock
    ) -> None:
        """invalidate_session on a room that doesn't exist should be a safe no-op."""
        from thenvoi.integrations.claude_sdk.session_manager import (
            ClaudeSessionManager,
        )

        manager = ClaudeSessionManager(mock_options)
        await manager.start()

        # Should not raise
        await manager.invalidate_session("room-nonexistent")

        assert manager.get_session_count() == 0

        await manager.stop()

    @pytest.mark.asyncio
    async def test_get_or_create_after_invalidate_creates_fresh_client(
        self, mock_options: MagicMock
    ) -> None:
        """After invalidation, get_or_create_session should create a new client."""
        from thenvoi.integrations.claude_sdk.session_manager import (
            ClaudeSessionManager,
        )

        manager = ClaudeSessionManager(mock_options)
        await manager.start()

        # Inject a mock dead client
        dead_client = MagicMock()
        dead_client.disconnect = AsyncMock()
        manager._sessions["room-1"] = dead_client

        await manager.invalidate_session("room-1")
        assert "room-1" not in manager._sessions

        # Now create a fresh session
        fresh_client = MagicMock()
        fresh_client.connect = AsyncMock()

        with patch(
            "thenvoi.integrations.claude_sdk.session_manager.ClaudeSDKClient",
            return_value=fresh_client,
        ):
            client = await manager.get_or_create_session("room-1")

        assert client is fresh_client
        fresh_client.connect.assert_awaited_once()

        await manager.stop()

    @pytest.mark.asyncio
    async def test_invalidate_when_not_started_is_noop(
        self, mock_options: MagicMock
    ) -> None:
        """invalidate_session before start() should return immediately."""
        from thenvoi.integrations.claude_sdk.session_manager import (
            ClaudeSessionManager,
        )

        manager = ClaudeSessionManager(mock_options)

        # Should not raise or hang
        await manager.invalidate_session("room-1")

    @pytest.mark.asyncio
    async def test_invalidate_does_not_affect_other_rooms(
        self, mock_options: MagicMock
    ) -> None:
        """Invalidating one room should leave other rooms' sessions intact."""
        from thenvoi.integrations.claude_sdk.session_manager import (
            ClaudeSessionManager,
        )

        manager = ClaudeSessionManager(mock_options)
        await manager.start()

        client_a = MagicMock()
        client_b = MagicMock()
        manager._sessions["room-a"] = client_a
        manager._sessions["room-b"] = client_b

        await manager.invalidate_session("room-a")

        assert "room-a" not in manager._sessions
        assert manager._sessions["room-b"] is client_b

        await manager.stop()
