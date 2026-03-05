"""Tests for ACPServer."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from thenvoi.integrations.acp.server import ACPServer
from thenvoi.integrations.acp.server_adapter import ThenvoiACPServerAdapter


class TestACPServerInitialize:
    """Tests for ACPServer.initialize()."""

    @pytest.mark.asyncio
    async def test_initialize_returns_protocol_version(self) -> None:
        """Should return the same protocol version as requested."""
        adapter = ThenvoiACPServerAdapter()
        server = ACPServer(adapter)

        response = await server.initialize(protocol_version=1)

        assert response.protocol_version == 1

    @pytest.mark.asyncio
    async def test_initialize_returns_agent_info(self) -> None:
        """Should return agent info with name, title, and version."""
        adapter = ThenvoiACPServerAdapter()
        await adapter.on_started("My Agent", "A test agent")
        server = ACPServer(adapter)

        response = await server.initialize(protocol_version=1)

        assert response.agent_info.name == "thenvoi-agent"
        assert response.agent_info.title == "My Agent"
        assert response.agent_info.version is not None

    @pytest.mark.asyncio
    async def test_initialize_with_client_info(self) -> None:
        """Should accept optional client_capabilities and client_info."""
        adapter = ThenvoiACPServerAdapter()
        server = ACPServer(adapter)

        response = await server.initialize(
            protocol_version=1,
            client_capabilities={"streaming": True},
            client_info={"name": "Zed", "version": "1.0"},
        )

        assert response.protocol_version == 1


class TestACPServerNewSession:
    """Tests for ACPServer.new_session()."""

    @pytest.mark.asyncio
    async def test_new_session_delegates_to_adapter(self) -> None:
        """Should delegate session creation to adapter."""
        adapter = ThenvoiACPServerAdapter()
        adapter.create_session = AsyncMock(return_value="session-abc123")
        server = ACPServer(adapter)

        response = await server.new_session(cwd="/workspace")

        assert response.session_id == "session-abc123"
        adapter.create_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_new_session_with_mcp_servers(self) -> None:
        """Should accept optional mcp_servers parameter."""
        adapter = ThenvoiACPServerAdapter()
        adapter.create_session = AsyncMock(return_value="session-xyz")
        server = ACPServer(adapter)

        response = await server.new_session(
            cwd="/workspace",
            mcp_servers=[{"type": "stdio", "command": "some-mcp-server"}],
        )

        assert response.session_id == "session-xyz"


class TestACPServerPrompt:
    """Tests for ACPServer.prompt()."""

    @pytest.mark.asyncio
    async def test_prompt_extracts_text_and_delegates(self) -> None:
        """Should extract text from content blocks and delegate to adapter."""
        adapter = ThenvoiACPServerAdapter()
        adapter.handle_prompt = AsyncMock()
        server = ACPServer(adapter)

        prompt_blocks = [{"text": "Hello world"}]
        response = await server.prompt(prompt=prompt_blocks, session_id="session-1")

        assert response.stop_reason == "end_turn"
        adapter.handle_prompt.assert_called_once_with("session-1", "Hello world")

    @pytest.mark.asyncio
    async def test_prompt_multiple_text_blocks(self) -> None:
        """Should concatenate text from multiple blocks."""
        adapter = ThenvoiACPServerAdapter()
        adapter.handle_prompt = AsyncMock()
        server = ACPServer(adapter)

        prompt_blocks = [
            {"text": "First line"},
            {"text": "Second line"},
        ]
        await server.prompt(prompt=prompt_blocks, session_id="session-1")

        adapter.handle_prompt.assert_called_once_with(
            "session-1", "First line\nSecond line"
        )

    @pytest.mark.asyncio
    async def test_prompt_skips_non_text_blocks(self) -> None:
        """Should skip content blocks without text."""
        adapter = ThenvoiACPServerAdapter()
        adapter.handle_prompt = AsyncMock()
        server = ACPServer(adapter)

        prompt_blocks = [
            {"text": "Hello"},
            {"type": "image", "data": "base64..."},  # No text field
            {"text": "World"},
        ]
        await server.prompt(prompt=prompt_blocks, session_id="session-1")

        adapter.handle_prompt.assert_called_once_with("session-1", "Hello\nWorld")

    @pytest.mark.asyncio
    async def test_prompt_with_object_blocks(self) -> None:
        """Should handle content blocks as objects (not just dicts)."""
        adapter = ThenvoiACPServerAdapter()
        adapter.handle_prompt = AsyncMock()
        server = ACPServer(adapter)

        # Simulate object-style content blocks
        block = MagicMock()
        block.text = "Object block text"

        await server.prompt(prompt=[block], session_id="session-1")

        adapter.handle_prompt.assert_called_once_with("session-1", "Object block text")


class TestACPServerCancel:
    """Tests for ACPServer.cancel()."""

    @pytest.mark.asyncio
    async def test_cancel_delegates_to_adapter(self) -> None:
        """Should delegate cancellation to adapter."""
        adapter = ThenvoiACPServerAdapter()
        adapter.cancel_prompt = AsyncMock()
        server = ACPServer(adapter)

        await server.cancel(session_id="session-1")

        adapter.cancel_prompt.assert_called_once_with("session-1")


class TestACPServerOnConnect:
    """Tests for ACPServer.on_connect()."""

    def test_on_connect_stores_client(self) -> None:
        """Should store client reference and pass to adapter."""
        adapter = ThenvoiACPServerAdapter()
        adapter.set_acp_client = MagicMock()
        server = ACPServer(adapter)

        mock_client = MagicMock()
        server.on_connect(mock_client)

        assert server._conn is mock_client
        adapter.set_acp_client.assert_called_once_with(mock_client)


class TestACPServerExtractText:
    """Tests for ACPServer._extract_text()."""

    def test_extract_text_empty(self) -> None:
        """Should return empty string for empty list."""
        assert ACPServer._extract_text([]) == ""

    def test_extract_text_single_dict(self) -> None:
        """Should extract text from single dict block."""
        assert ACPServer._extract_text([{"text": "Hello"}]) == "Hello"

    def test_extract_text_single_object(self) -> None:
        """Should extract text from single object block."""
        block = MagicMock()
        block.text = "Hello"
        assert ACPServer._extract_text([block]) == "Hello"

    def test_extract_text_mixed(self) -> None:
        """Should handle mix of dict and object blocks."""
        obj_block = MagicMock()
        obj_block.text = "Object"
        blocks = [{"text": "Dict"}, obj_block]
        assert ACPServer._extract_text(blocks) == "Dict\nObject"

    def test_extract_text_skips_empty(self) -> None:
        """Should skip blocks with no text."""
        blocks = [{"text": ""}, {"text": "Hello"}, {"other": "data"}]
        assert ACPServer._extract_text(blocks) == "Hello"


class TestACPServerLoadSession:
    """Tests for ACPServer.load_session()."""

    @pytest.mark.asyncio
    async def test_load_session_existing(self) -> None:
        """Should return LoadSessionResponse for existing session."""
        adapter = ThenvoiACPServerAdapter()
        adapter._session_to_room["session-1"] = "room-1"
        server = ACPServer(adapter)

        response = await server.load_session(cwd="/workspace", session_id="session-1")

        assert response is not None

    @pytest.mark.asyncio
    async def test_load_session_not_found(self) -> None:
        """Should return None for non-existent session."""
        adapter = ThenvoiACPServerAdapter()
        server = ACPServer(adapter)

        response = await server.load_session(cwd="/workspace", session_id="unknown")

        assert response is None


class TestACPServerListSessions:
    """Tests for ACPServer.list_sessions()."""

    @pytest.mark.asyncio
    async def test_list_sessions_returns_active(self) -> None:
        """Should return all active sessions."""
        adapter = ThenvoiACPServerAdapter()
        adapter._session_to_room["session-1"] = "room-1"
        adapter._session_to_room["session-2"] = "room-2"
        server = ACPServer(adapter)

        response = await server.list_sessions()

        assert len(response.sessions) == 2
        session_ids = {s.session_id for s in response.sessions}
        assert session_ids == {"session-1", "session-2"}

    @pytest.mark.asyncio
    async def test_list_sessions_empty(self) -> None:
        """Should return empty list when no sessions."""
        adapter = ThenvoiACPServerAdapter()
        server = ACPServer(adapter)

        response = await server.list_sessions()

        assert response.sessions == []


class TestACPServerSetSessionMode:
    """Tests for ACPServer.set_session_mode()."""

    @pytest.mark.asyncio
    async def test_set_session_mode(self) -> None:
        """Should store mode in adapter state."""
        adapter = ThenvoiACPServerAdapter()
        server = ACPServer(adapter)

        response = await server.set_session_mode(mode_id="code", session_id="session-1")

        assert response is not None
        assert adapter._session_modes["session-1"] == "code"


class TestACPServerSetSessionModel:
    """Tests for ACPServer.set_session_model()."""

    @pytest.mark.asyncio
    async def test_set_session_model(self) -> None:
        """Should store model in adapter state."""
        adapter = ThenvoiACPServerAdapter()
        server = ACPServer(adapter)

        response = await server.set_session_model(
            model_id="gpt-4o", session_id="session-1"
        )

        assert response is not None
        assert adapter._session_models["session-1"] == "gpt-4o"


class TestACPServerAuthenticate:
    """Tests for ACPServer.authenticate()."""

    @pytest.mark.asyncio
    async def test_authenticate_valid_key(self) -> None:
        """Should return AuthenticateResponse for valid key."""
        adapter = ThenvoiACPServerAdapter()
        mock_rest = MagicMock()
        mock_rest.agent_api_identity.get_agent_identity = AsyncMock()
        adapter._rest = mock_rest
        server = ACPServer(adapter)

        response = await server.authenticate(method_id="api_key")

        assert response is not None

    @pytest.mark.asyncio
    async def test_authenticate_invalid_key(self) -> None:
        """Should return None for invalid key."""
        adapter = ThenvoiACPServerAdapter()
        mock_rest = MagicMock()
        mock_rest.agent_api_identity.get_agent_identity = AsyncMock(
            side_effect=RuntimeError("Unauthorized")
        )
        adapter._rest = mock_rest
        server = ACPServer(adapter)

        response = await server.authenticate(method_id="api_key")

        assert response is None

    @pytest.mark.asyncio
    async def test_authenticate_unknown_method(self) -> None:
        """Should return None for unsupported auth method."""
        adapter = ThenvoiACPServerAdapter()
        server = ACPServer(adapter)

        response = await server.authenticate(method_id="oauth")

        assert response is None


class TestACPServerExtMethod:
    """Tests for ACPServer.ext_method()."""

    @pytest.mark.asyncio
    async def test_ext_method_returns_error_for_unknown(self) -> None:
        """Should return error for unknown extension method."""
        adapter = ThenvoiACPServerAdapter()
        server = ACPServer(adapter)

        result = await server.ext_method("thenvoi/unknown", {"key": "val"})

        assert "error" in result


class TestACPServerExtNotification:
    """Tests for ACPServer.ext_notification()."""

    @pytest.mark.asyncio
    async def test_ext_notification_does_not_crash(self) -> None:
        """Should handle extension notifications gracefully."""
        adapter = ThenvoiACPServerAdapter()
        server = ACPServer(adapter)

        # Should not raise
        await server.ext_notification("thenvoi/status", {"status": "ok"})
