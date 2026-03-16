"""Tests for ACP MCP server tool implementations."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from thenvoi.integrations.acp.mcp_server import (
    _get_config,
    _make_error,
    _make_result,
    _parse_json,
)


class TestGetConfig:
    """Tests for _get_config()."""

    def test_returns_config_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should return api_key, rest_url, room_id from environment."""
        monkeypatch.setenv("THENVOI_API_KEY", "test-key")
        monkeypatch.setenv("THENVOI_REST_URL", "https://test.example.com")
        monkeypatch.setenv("THENVOI_ROOM_ID", "room-123")

        api_key, rest_url, room_id = _get_config()

        assert api_key == "test-key"
        assert rest_url == "https://test.example.com"
        assert room_id == "room-123"

    def test_uses_default_rest_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should use default REST URL when not set."""
        monkeypatch.setenv("THENVOI_API_KEY", "test-key")
        monkeypatch.setenv("THENVOI_ROOM_ID", "room-123")
        monkeypatch.delenv("THENVOI_REST_URL", raising=False)

        _, rest_url, _ = _get_config()

        assert rest_url == "https://app.thenvoi.com"

    def test_raises_on_missing_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should raise ValueError when API key is missing."""
        monkeypatch.delenv("THENVOI_API_KEY", raising=False)
        monkeypatch.setenv("THENVOI_ROOM_ID", "room-123")

        with pytest.raises(ValueError, match="THENVOI_API_KEY"):
            _get_config()

    def test_raises_on_missing_room_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should raise ValueError when room ID is missing."""
        monkeypatch.setenv("THENVOI_API_KEY", "test-key")
        monkeypatch.delenv("THENVOI_ROOM_ID", raising=False)

        with pytest.raises(ValueError, match="THENVOI_ROOM_ID"):
            _get_config()

    def test_raises_on_invalid_rest_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should validate THENVOI_REST_URL before client creation."""
        monkeypatch.setenv("THENVOI_API_KEY", "test-key")
        monkeypatch.setenv("THENVOI_ROOM_ID", "room-123")
        monkeypatch.setenv("THENVOI_REST_URL", "ftp://invalid")

        with pytest.raises(ValueError, match="THENVOI_REST_URL"):
            _get_config()


class TestParseJson:
    """Tests for _parse_json()."""

    def test_parses_valid_json(self) -> None:
        assert _parse_json('["a", "b"]') == ["a", "b"]

    def test_parses_json_object(self) -> None:
        assert _parse_json('{"key": "val"}') == {"key": "val"}

    def test_returns_fallback_on_empty_string(self) -> None:
        assert _parse_json("", fallback="default") == "default"

    def test_returns_fallback_on_empty_object(self) -> None:
        assert _parse_json("{}", fallback="default") == "default"

    def test_returns_fallback_on_empty_array(self) -> None:
        assert _parse_json("[]", fallback="default") == "default"

    def test_returns_fallback_on_invalid_json(self) -> None:
        assert _parse_json("not json", fallback=[]) == []

    def test_returns_non_string_as_is(self) -> None:
        assert _parse_json({"already": "parsed"}) == {"already": "parsed"}


class TestMakeResult:
    """Tests for _make_result()."""

    def test_formats_dict_as_json(self) -> None:
        result = _make_result({"status": "success", "count": 5})
        parsed = json.loads(result)
        assert parsed["status"] == "success"
        assert parsed["count"] == 5

    def test_handles_non_serializable_with_default_str(self) -> None:
        """Should use str() for non-serializable values."""
        from datetime import datetime

        result = _make_result({"time": datetime(2024, 1, 1)})
        parsed = json.loads(result)
        assert "2024" in parsed["time"]


class TestMakeError:
    """Tests for _make_error()."""

    def test_formats_error_as_json(self) -> None:
        result = _make_error("Something went wrong")
        parsed = json.loads(result)
        assert parsed["status"] == "error"
        assert parsed["message"] == "Something went wrong"


class TestThenvoiSendMessage:
    """Tests for thenvoi_send_message tool."""

    @pytest.mark.asyncio
    async def test_send_message_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should call tools.send_message and return success."""
        monkeypatch.setenv("THENVOI_API_KEY", "test-key")
        monkeypatch.setenv("THENVOI_ROOM_ID", "room-123")

        mock_tools = AsyncMock()
        mock_tools.send_message = AsyncMock()

        with patch(
            "thenvoi.integrations.acp.mcp_server._get_tools", return_value=mock_tools
        ):
            from thenvoi.integrations.acp.mcp_server import thenvoi_send_message

            result = await thenvoi_send_message("Hello!", '["@alice"]')

        parsed = json.loads(result)
        assert parsed["status"] == "success"
        mock_tools.send_message.assert_called_once_with("Hello!", ["@alice"])

    @pytest.mark.asyncio
    async def test_send_message_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should return error JSON on exception."""
        monkeypatch.setenv("THENVOI_API_KEY", "test-key")
        monkeypatch.setenv("THENVOI_ROOM_ID", "room-123")

        mock_tools = AsyncMock()
        mock_tools.send_message = AsyncMock(side_effect=RuntimeError("Network error"))

        with patch(
            "thenvoi.integrations.acp.mcp_server._get_tools", return_value=mock_tools
        ):
            from thenvoi.integrations.acp.mcp_server import thenvoi_send_message

            result = await thenvoi_send_message("Hello!", "[]")

        parsed = json.loads(result)
        assert parsed["status"] == "error"
        assert "Network error" in parsed["message"]


class TestThenvoiRespondContactRequest:
    """Tests for thenvoi_respond_contact_request action validation."""

    @pytest.mark.asyncio
    async def test_rejects_invalid_action(self) -> None:
        """Should return error for invalid action without calling API."""
        from thenvoi.integrations.acp.mcp_server import thenvoi_respond_contact_request

        result = await thenvoi_respond_contact_request(action="delete", handle="@alice")

        parsed = json.loads(result)
        assert parsed["status"] == "error"
        assert "Invalid action" in parsed["message"]
        assert "approve" in parsed["message"]

    @pytest.mark.asyncio
    async def test_accepts_valid_actions(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should accept approve, reject, cancel."""
        monkeypatch.setenv("THENVOI_API_KEY", "test-key")
        monkeypatch.setenv("THENVOI_ROOM_ID", "room-123")

        mock_tools = AsyncMock()
        mock_tools.respond_contact_request = AsyncMock(
            return_value={"request_id": "req-1"}
        )

        with patch(
            "thenvoi.integrations.acp.mcp_server._get_tools", return_value=mock_tools
        ):
            from thenvoi.integrations.acp.mcp_server import (
                thenvoi_respond_contact_request,
            )

            for action in ("approve", "reject", "cancel"):
                result = await thenvoi_respond_contact_request(
                    action=action, handle="@alice"
                )
                parsed = json.loads(result)
                assert parsed["status"] == "success"


class TestThenvoiSendEvent:
    """Tests for thenvoi_send_event tool."""

    @pytest.mark.asyncio
    async def test_send_event_with_metadata(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should parse metadata JSON and forward to tools."""
        monkeypatch.setenv("THENVOI_API_KEY", "test-key")
        monkeypatch.setenv("THENVOI_ROOM_ID", "room-123")

        mock_tools = AsyncMock()
        mock_tools.send_event = AsyncMock()

        with patch(
            "thenvoi.integrations.acp.mcp_server._get_tools", return_value=mock_tools
        ):
            from thenvoi.integrations.acp.mcp_server import thenvoi_send_event

            result = await thenvoi_send_event("Processing...", "thought", '{"step": 1}')

        parsed = json.loads(result)
        assert parsed["status"] == "success"
        mock_tools.send_event.assert_called_once_with(
            "Processing...", "thought", {"step": 1}
        )


class TestThenvoiStoreMemory:
    """Tests for thenvoi_store_memory tool."""

    @pytest.mark.asyncio
    async def test_store_memory_with_optional_fields(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should build kwargs correctly with optional subject_id and metadata."""
        monkeypatch.setenv("THENVOI_API_KEY", "test-key")
        monkeypatch.setenv("THENVOI_ROOM_ID", "room-123")

        mock_tools = AsyncMock()
        mock_tools.store_memory = AsyncMock(return_value={"id": "mem-1"})

        with patch(
            "thenvoi.integrations.acp.mcp_server._get_tools", return_value=mock_tools
        ):
            from thenvoi.integrations.acp.mcp_server import thenvoi_store_memory

            result = await thenvoi_store_memory(
                content="User prefers dark mode",
                system="long_term",
                type="semantic",
                segment="user",
                thought="Storing preference",
                subject_id="user-123",
                metadata='{"tags": ["preference"]}',
            )

        parsed = json.loads(result)
        assert parsed["status"] == "success"
        call_kwargs = mock_tools.store_memory.call_args[1]
        assert call_kwargs["subject_id"] == "user-123"
        assert call_kwargs["metadata"] == {"tags": ["preference"]}
