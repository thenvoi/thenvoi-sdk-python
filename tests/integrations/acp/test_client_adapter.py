"""Tests for ACPClientAdapter."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from thenvoi.integrations.acp.client_adapter import ACPClientAdapter
from thenvoi.integrations.acp.client_types import (
    ACPClientSessionState,
    ThenvoiACPClient,
)
from thenvoi.integrations.acp.types import CollectedChunk
from thenvoi.testing import FakeAgentTools

from .conftest import make_platform_message


class TestACPClientAdapterInit:
    """Tests for ACPClientAdapter initialization."""

    def test_init_string_command(self) -> None:
        """Should accept string command."""
        adapter = ACPClientAdapter(command="codex")
        assert adapter._command == ["codex"]

    def test_init_list_command(self) -> None:
        """Should accept list command."""
        adapter = ACPClientAdapter(command=["gemini", "cli"])
        assert adapter._command == ["gemini", "cli"]

    def test_init_default_values(self) -> None:
        """Should initialize with default values."""
        adapter = ACPClientAdapter(command="codex")
        assert adapter._cwd == "."
        assert adapter._env is None
        assert adapter._mcp_servers == []
        assert adapter._conn is None
        assert adapter._client is None
        assert adapter._room_to_session == {}

    def test_init_with_custom_values(self) -> None:
        """Should accept custom configuration."""
        adapter = ACPClientAdapter(
            command="codex",
            env={"API_KEY": "test"},
            cwd="/workspace",
            mcp_servers=[{"type": "stdio", "command": "server"}],
        )
        assert adapter._cwd == "/workspace"
        assert adapter._env == {"API_KEY": "test"}
        assert len(adapter._mcp_servers) == 1

    def test_init_sets_history_converter(self) -> None:
        """Should set ACPClientHistoryConverter."""
        adapter = ACPClientAdapter(command="codex")
        assert adapter.history_converter is not None


class TestACPClientAdapterOnStarted:
    """Tests for ACPClientAdapter.on_started()."""

    @pytest.mark.asyncio
    async def test_on_started_spawns_process(self) -> None:
        """Should spawn ACP process and initialize connection."""
        adapter = ACPClientAdapter(command="codex")

        mock_conn = AsyncMock()
        mock_conn.initialize = AsyncMock()
        mock_proc = MagicMock()

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=(mock_conn, mock_proc))

        with patch(
            "thenvoi.integrations.acp.client_adapter.spawn_agent_process",
            return_value=mock_ctx,
        ):
            await adapter.on_started("Codex Bridge", "Bridge to Codex")

        assert adapter._conn is mock_conn
        mock_conn.initialize.assert_called_once_with(protocol_version=1)

    @pytest.mark.asyncio
    async def test_on_started_stores_agent_info(self) -> None:
        """Should store agent name and description."""
        adapter = ACPClientAdapter(command="codex")

        mock_conn = AsyncMock()
        mock_conn.initialize = AsyncMock()
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=(mock_conn, MagicMock()))

        with patch(
            "thenvoi.integrations.acp.client_adapter.spawn_agent_process",
            return_value=mock_ctx,
        ):
            await adapter.on_started("Test Agent", "A test agent")

        assert adapter.agent_name == "Test Agent"
        assert adapter.agent_description == "A test agent"


class TestACPClientAdapterOnMessage:
    """Tests for ACPClientAdapter.on_message()."""

    @pytest.fixture
    def adapter_with_mocks(self) -> ACPClientAdapter:
        """Create adapter with mocked ACP connection."""
        adapter = ACPClientAdapter(command="codex")

        # Mock ACP connection
        adapter._conn = AsyncMock()
        mock_session = MagicMock()
        mock_session.session_id = "acp-session-123"
        adapter._conn.new_session = AsyncMock(return_value=mock_session)
        adapter._conn.prompt = AsyncMock()

        # Mock client with response text
        adapter._client = ThenvoiACPClient()

        return adapter

    @pytest.mark.asyncio
    async def test_on_message_creates_session(
        self, adapter_with_mocks: ACPClientAdapter
    ) -> None:
        """Should create ACP session for new room."""
        tools = FakeAgentTools()
        msg = make_platform_message("Hello", room_id="room-123")

        await adapter_with_mocks.on_message(
            msg,
            tools,
            ACPClientSessionState(),
            None,
            None,
            is_session_bootstrap=False,
            room_id="room-123",
        )

        adapter_with_mocks._conn.new_session.assert_called_once()
        assert adapter_with_mocks._room_to_session["room-123"] == "acp-session-123"

    @pytest.mark.asyncio
    async def test_on_message_reuses_session(
        self, adapter_with_mocks: ACPClientAdapter
    ) -> None:
        """Should reuse existing session for same room."""
        adapter_with_mocks._room_to_session["room-123"] = "existing-session"
        tools = FakeAgentTools()
        msg = make_platform_message("Hello", room_id="room-123")

        await adapter_with_mocks.on_message(
            msg,
            tools,
            ACPClientSessionState(),
            None,
            None,
            is_session_bootstrap=False,
            room_id="room-123",
        )

        adapter_with_mocks._conn.new_session.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_message_sends_prompt(
        self, adapter_with_mocks: ACPClientAdapter
    ) -> None:
        """Should send prompt to external ACP agent."""
        tools = FakeAgentTools()
        msg = make_platform_message("What is the weather?", room_id="room-123")

        await adapter_with_mocks.on_message(
            msg,
            tools,
            ACPClientSessionState(),
            None,
            None,
            is_session_bootstrap=False,
            room_id="room-123",
        )

        adapter_with_mocks._conn.prompt.assert_called_once()
        call_kwargs = adapter_with_mocks._conn.prompt.call_args.kwargs
        assert call_kwargs["session_id"] == "acp-session-123"

    @pytest.mark.asyncio
    async def test_on_message_posts_response(
        self, adapter_with_mocks: ACPClientAdapter
    ) -> None:
        """Should post collected response back to Thenvoi room."""

        # Make prompt() populate the per-session buffer (simulating session_update)
        async def mock_prompt(**kwargs):
            session_id = kwargs.get("session_id", "acp-session-123")
            adapter_with_mocks._client._session_chunks[session_id] = [
                CollectedChunk(chunk_type="text", content="The weather is sunny.")
            ]

        adapter_with_mocks._conn.prompt = AsyncMock(side_effect=mock_prompt)

        tools = FakeAgentTools()
        msg = make_platform_message("Weather?", room_id="room-123")

        await adapter_with_mocks.on_message(
            msg,
            tools,
            ACPClientSessionState(),
            None,
            None,
            is_session_bootstrap=False,
            room_id="room-123",
        )

        # Should have sent message back
        assert len(tools.messages_sent) > 0
        assert tools.messages_sent[0]["content"] == "The weather is sunny."

    @pytest.mark.asyncio
    async def test_on_message_posts_thought_event(
        self, adapter_with_mocks: ACPClientAdapter
    ) -> None:
        """Should post thought chunks as thought events."""

        async def mock_prompt(**kwargs):
            session_id = kwargs.get("session_id", "acp-session-123")
            adapter_with_mocks._client._session_chunks[session_id] = [
                CollectedChunk(chunk_type="thought", content="Let me think...")
            ]

        adapter_with_mocks._conn.prompt = AsyncMock(side_effect=mock_prompt)

        tools = FakeAgentTools()
        msg = make_platform_message("Question?", room_id="room-123")

        await adapter_with_mocks.on_message(
            msg,
            tools,
            ACPClientSessionState(),
            None,
            None,
            is_session_bootstrap=False,
            room_id="room-123",
        )

        thought_events = [
            e for e in tools.events_sent if e.get("message_type") == "thought"
        ]
        assert len(thought_events) == 1
        assert thought_events[0]["content"] == "Let me think..."

    @pytest.mark.asyncio
    async def test_on_message_posts_tool_call_event(
        self, adapter_with_mocks: ACPClientAdapter
    ) -> None:
        """Should post tool_call chunks as tool_call events."""

        async def mock_prompt(**kwargs):
            session_id = kwargs.get("session_id", "acp-session-123")
            adapter_with_mocks._client._session_chunks[session_id] = [
                CollectedChunk(
                    chunk_type="tool_call",
                    content="search",
                    metadata={"tool_call_id": "tc-1"},
                )
            ]

        adapter_with_mocks._conn.prompt = AsyncMock(side_effect=mock_prompt)

        tools = FakeAgentTools()
        msg = make_platform_message("Find info", room_id="room-123")

        await adapter_with_mocks.on_message(
            msg,
            tools,
            ACPClientSessionState(),
            None,
            None,
            is_session_bootstrap=False,
            room_id="room-123",
        )

        tool_events = [
            e for e in tools.events_sent if e.get("message_type") == "tool_call"
        ]
        assert len(tool_events) == 1
        assert tool_events[0]["metadata"]["tool_call_id"] == "tc-1"

    @pytest.mark.asyncio
    async def test_on_message_posts_plan_as_task_event(
        self, adapter_with_mocks: ACPClientAdapter
    ) -> None:
        """Should post plan chunks as task events."""

        async def mock_prompt(**kwargs):
            session_id = kwargs.get("session_id", "acp-session-123")
            adapter_with_mocks._client._session_chunks[session_id] = [
                CollectedChunk(chunk_type="plan", content="Step 1: Do stuff")
            ]

        adapter_with_mocks._conn.prompt = AsyncMock(side_effect=mock_prompt)

        tools = FakeAgentTools()
        msg = make_platform_message("Plan it", room_id="room-123")

        await adapter_with_mocks.on_message(
            msg,
            tools,
            ACPClientSessionState(),
            None,
            None,
            is_session_bootstrap=False,
            room_id="room-123",
        )

        task_events = [
            e
            for e in tools.events_sent
            if e.get("message_type") == "task"
            and "acp_client_session_id" not in e.get("metadata", {})
        ]
        assert len(task_events) == 1
        assert task_events[0]["content"] == "Step 1: Do stuff"

    @pytest.mark.asyncio
    async def test_on_message_emits_task_event(
        self, adapter_with_mocks: ACPClientAdapter
    ) -> None:
        """Should emit task event for session rehydration."""
        tools = FakeAgentTools()
        msg = make_platform_message("Hello", room_id="room-123")

        await adapter_with_mocks.on_message(
            msg,
            tools,
            ACPClientSessionState(),
            None,
            None,
            is_session_bootstrap=False,
            room_id="room-123",
        )

        # Should have sent task event
        task_events = [e for e in tools.events_sent if e.get("message_type") == "task"]
        assert len(task_events) == 1
        assert task_events[0]["metadata"]["acp_client_session_id"] == "acp-session-123"

    @pytest.mark.asyncio
    async def test_on_message_bootstrap_rehydrates(
        self, adapter_with_mocks: ACPClientAdapter
    ) -> None:
        """Should rehydrate room -> session mappings on bootstrap."""
        tools = FakeAgentTools()
        msg = make_platform_message("Hello", room_id="room-123")

        history = ACPClientSessionState(room_to_session={"room-abc": "session-abc"})

        await adapter_with_mocks.on_message(
            msg,
            tools,
            history,
            None,
            None,
            is_session_bootstrap=True,
            room_id="room-123",
        )

        assert adapter_with_mocks._room_to_session["room-abc"] == "session-abc"

    @pytest.mark.asyncio
    async def test_on_message_error_sends_error_event(
        self, adapter_with_mocks: ACPClientAdapter
    ) -> None:
        """Should send error event when ACP agent fails."""
        adapter_with_mocks._conn.prompt = AsyncMock(
            side_effect=RuntimeError("Agent crashed")
        )

        tools = FakeAgentTools()
        msg = make_platform_message("Hello", room_id="room-123")

        await adapter_with_mocks.on_message(
            msg,
            tools,
            ACPClientSessionState(),
            None,
            None,
            is_session_bootstrap=False,
            room_id="room-123",
        )

        error_events = [
            e for e in tools.events_sent if e.get("message_type") == "error"
        ]
        assert len(error_events) == 1
        assert "Agent crashed" in error_events[0]["content"]

    @pytest.mark.asyncio
    async def test_on_message_not_initialized_raises(self) -> None:
        """Should raise RuntimeError if not initialized."""
        adapter = ACPClientAdapter(command="codex")
        tools = FakeAgentTools()
        msg = make_platform_message("Hello", room_id="room-123")

        with pytest.raises(RuntimeError, match="ACP client not initialized"):
            await adapter.on_message(
                msg,
                tools,
                ACPClientSessionState(),
                None,
                None,
                is_session_bootstrap=False,
                room_id="room-123",
            )


class TestACPClientAdapterPermissionHandler:
    """Tests for bidirectional permission proxying."""

    @pytest.fixture
    def adapter_with_mocks(self) -> ACPClientAdapter:
        """Create adapter with mocked ACP connection."""
        adapter = ACPClientAdapter(command="codex")

        # Mock ACP connection
        adapter._conn = AsyncMock()
        mock_session = MagicMock()
        mock_session.session_id = "acp-session-123"
        adapter._conn.new_session = AsyncMock(return_value=mock_session)
        adapter._conn.prompt = AsyncMock()

        # Mock client with response text
        adapter._client = ThenvoiACPClient()

        return adapter

    @pytest.mark.asyncio
    async def test_permission_handler_wired_on_message(
        self, adapter_with_mocks: ACPClientAdapter
    ) -> None:
        """Should set permission handler on client before sending prompt."""
        tools = FakeAgentTools()
        msg = make_platform_message("Hello", room_id="room-123")

        await adapter_with_mocks.on_message(
            msg,
            tools,
            ACPClientSessionState(),
            None,
            None,
            is_session_bootstrap=False,
            room_id="room-123",
        )

        # Permission handler should have been set
        assert adapter_with_mocks._client._permission_handler is not None

    @pytest.mark.asyncio
    async def test_permission_handler_posts_event(
        self, adapter_with_mocks: ACPClientAdapter
    ) -> None:
        """Should post permission request event to platform."""
        tools = FakeAgentTools()
        msg = make_platform_message("Hello", room_id="room-123")

        # Simulate permission request during prompt
        async def mock_prompt(**kwargs):
            # Trigger the permission handler directly
            tool_call = MagicMock()
            tool_call.title = "write_file"
            tool_call.tool_call_id = "tc-perm-1"

            result = await adapter_with_mocks._client.request_permission(
                options={},
                session_id="acp-session-123",
                tool_call=tool_call,
            )
            assert result == {"outcome": {"outcome": "allowed"}}

        adapter_with_mocks._conn.prompt = AsyncMock(side_effect=mock_prompt)

        await adapter_with_mocks.on_message(
            msg,
            tools,
            ACPClientSessionState(),
            None,
            None,
            is_session_bootstrap=False,
            room_id="room-123",
        )

        # Should have posted a permission event
        perm_events = [
            e
            for e in tools.events_sent
            if e.get("metadata", {}).get("permission_request")
        ]
        assert len(perm_events) == 1
        assert perm_events[0]["metadata"]["tool_name"] == "write_file"
        assert perm_events[0]["metadata"]["auto_allowed"] is True
        assert perm_events[0]["message_type"] == "tool_call"

    @pytest.mark.asyncio
    async def test_permission_handler_returns_allowed(
        self, adapter_with_mocks: ACPClientAdapter
    ) -> None:
        """Should auto-allow permission requests."""
        tools = FakeAgentTools()
        msg = make_platform_message("Hello", room_id="room-123")

        captured_result = {}

        async def mock_prompt(**kwargs):
            tool_call = MagicMock()
            tool_call.title = "read_file"
            tool_call.tool_call_id = "tc-read"

            result = await adapter_with_mocks._client.request_permission(
                options={},
                session_id="acp-session-123",
                tool_call=tool_call,
            )
            captured_result.update(result)

        adapter_with_mocks._conn.prompt = AsyncMock(side_effect=mock_prompt)

        await adapter_with_mocks.on_message(
            msg,
            tools,
            ACPClientSessionState(),
            None,
            None,
            is_session_bootstrap=False,
            room_id="room-123",
        )

        assert captured_result == {"outcome": {"outcome": "allowed"}}

    @pytest.mark.asyncio
    async def test_permission_handler_uses_name_fallback(
        self, adapter_with_mocks: ACPClientAdapter
    ) -> None:
        """Should fall back to 'name' attr if 'title' is not available."""
        tools = FakeAgentTools()
        msg = make_platform_message("Hello", room_id="room-123")

        async def mock_prompt(**kwargs):
            tool_call = MagicMock(spec=[])  # No attributes by default
            tool_call.name = "bash"
            tool_call.tool_call_id = "tc-bash"

            await adapter_with_mocks._client.request_permission(
                options={},
                session_id="acp-session-123",
                tool_call=tool_call,
            )

        adapter_with_mocks._conn.prompt = AsyncMock(side_effect=mock_prompt)

        await adapter_with_mocks.on_message(
            msg,
            tools,
            ACPClientSessionState(),
            None,
            None,
            is_session_bootstrap=False,
            room_id="room-123",
        )

        perm_events = [
            e
            for e in tools.events_sent
            if e.get("metadata", {}).get("permission_request")
        ]
        assert len(perm_events) == 1
        assert perm_events[0]["metadata"]["tool_name"] == "bash"


class TestACPClientAdapterCleanup:
    """Tests for ACPClientAdapter cleanup."""

    @pytest.mark.asyncio
    async def test_on_cleanup_removes_mapping(self) -> None:
        """Should remove room -> session mapping."""
        adapter = ACPClientAdapter(command="codex")
        adapter._room_to_session["room-123"] = "session-123"

        await adapter.on_cleanup("room-123")

        assert "room-123" not in adapter._room_to_session

    @pytest.mark.asyncio
    async def test_on_cleanup_idempotent(self) -> None:
        """Should handle cleanup of non-existent room."""
        adapter = ACPClientAdapter(command="codex")

        await adapter.on_cleanup("nonexistent-room")

    @pytest.mark.asyncio
    async def test_on_cleanup_twice(self) -> None:
        """Should handle cleanup called twice."""
        adapter = ACPClientAdapter(command="codex")
        adapter._room_to_session["room-123"] = "session-123"

        await adapter.on_cleanup("room-123")
        await adapter.on_cleanup("room-123")

        assert "room-123" not in adapter._room_to_session


class TestACPClientAdapterStop:
    """Tests for ACPClientAdapter.stop()."""

    @pytest.mark.asyncio
    async def test_stop_closes_connection(self) -> None:
        """Should close ACP connection gracefully."""
        adapter = ACPClientAdapter(command="codex")
        mock_ctx = MagicMock()
        mock_ctx.__aexit__ = AsyncMock(return_value=None)
        adapter._ctx = mock_ctx
        adapter._conn = AsyncMock()

        await adapter.stop()

        mock_ctx.__aexit__.assert_called_once()
        assert adapter._ctx is None
        assert adapter._conn is None

    @pytest.mark.asyncio
    async def test_stop_no_connection(self) -> None:
        """Should handle stop when not connected."""
        adapter = ACPClientAdapter(command="codex")

        # Should not raise
        await adapter.stop()

    @pytest.mark.asyncio
    async def test_stop_handles_exit_error(self) -> None:
        """Should handle errors during shutdown."""
        adapter = ACPClientAdapter(command="codex")
        adapter._ctx = AsyncMock()
        adapter._ctx.__aexit__ = AsyncMock(side_effect=RuntimeError("Cleanup error"))

        # Should not raise
        await adapter.stop()
        assert adapter._ctx is None


class TestThenvoiACPClientCursorExtensions:
    """Tests for Cursor-specific extension handling in ThenvoiACPClient."""

    @pytest.mark.asyncio
    async def test_ext_method_cursor_ask_question(self) -> None:
        """Should auto-select first option for cursor/ask_question."""
        client = ThenvoiACPClient()

        result = await client.ext_method(
            "cursor/ask_question",
            {
                "options": [
                    {"optionId": "a", "name": "Option A"},
                    {"optionId": "b", "name": "Option B"},
                ],
            },
        )

        assert result["outcome"]["type"] == "selected"
        assert result["outcome"]["optionId"] == "a"

    @pytest.mark.asyncio
    async def test_ext_method_cursor_ask_question_empty_options(self) -> None:
        """Should cancel when no options provided."""
        client = ThenvoiACPClient()

        result = await client.ext_method("cursor/ask_question", {"options": []})

        assert result["outcome"]["type"] == "cancelled"

    @pytest.mark.asyncio
    async def test_ext_method_cursor_create_plan(self) -> None:
        """Should auto-approve cursor/create_plan."""
        client = ThenvoiACPClient()

        result = await client.ext_method("cursor/create_plan", {"plan": "stuff"})

        assert result["outcome"]["type"] == "approved"

    @pytest.mark.asyncio
    async def test_ext_method_unknown_returns_empty(self) -> None:
        """Should return empty dict for unknown extension methods."""
        client = ThenvoiACPClient()

        result = await client.ext_method("unknown/method", {})

        assert result == {}

    @pytest.mark.asyncio
    async def test_ext_notification_cursor_update_todos(self) -> None:
        """Should collect todo updates as plan chunks."""
        client = ThenvoiACPClient()

        await client.ext_notification(
            "cursor/update_todos",
            {
                "sessionId": "sess-1",
                "todos": [
                    {"content": "Read code", "completed": True},
                    {"content": "Write tests", "completed": False},
                ],
            },
        )

        chunks = client.get_collected_chunks("sess-1")
        assert len(chunks) == 1
        assert chunks[0].chunk_type == "plan"
        assert "[x] Read code" in chunks[0].content
        assert "[ ] Write tests" in chunks[0].content

    @pytest.mark.asyncio
    async def test_ext_notification_cursor_task(self) -> None:
        """Should collect task results as text chunks."""
        client = ThenvoiACPClient()

        await client.ext_notification(
            "cursor/task",
            {"sessionId": "sess-1", "result": "Refactored the module"},
        )

        chunks = client.get_collected_chunks("sess-1")
        assert len(chunks) == 1
        assert chunks[0].chunk_type == "text"
        assert "Refactored the module" in chunks[0].content

    @pytest.mark.asyncio
    async def test_ext_notification_no_session_id_is_noop(self) -> None:
        """Should do nothing when no session_id is present."""
        client = ThenvoiACPClient()

        await client.ext_notification(
            "cursor/update_todos",
            {"todos": [{"content": "Test", "completed": False}]},
        )

        # No session_id → no chunks collected
        assert client.get_collected_chunks() == []


class TestACPClientAdapterDeadConnectionRecovery:
    """Tests for dead connection recovery after subprocess crash."""

    @pytest.mark.asyncio
    async def test_prompt_error_clears_connection(self) -> None:
        """Should stop connection on prompt error so next message respawns."""
        adapter = ACPClientAdapter(command="codex")
        adapter._conn = AsyncMock()
        adapter._conn.prompt = AsyncMock(
            side_effect=RuntimeError("Process died")
        )
        mock_session = MagicMock()
        mock_session.session_id = "sess-1"
        adapter._conn.new_session = AsyncMock(return_value=mock_session)
        adapter._client = ThenvoiACPClient()

        mock_ctx = MagicMock()
        mock_ctx.__aexit__ = AsyncMock(return_value=None)
        adapter._ctx = mock_ctx

        tools = FakeAgentTools()
        msg = make_platform_message("Hello", room_id="room-1")

        await adapter.on_message(
            msg, tools, ACPClientSessionState(), None, None,
            is_session_bootstrap=False, room_id="room-1",
        )

        # Connection should be cleared after error
        assert adapter._conn is None
        assert adapter._ctx is None

        # Error event should be sent
        error_events = [
            e for e in tools.events_sent if e.get("message_type") == "error"
        ]
        assert len(error_events) == 1


class TestACPClientAdapterInjectToolsWarning:
    """Tests for inject_thenvoi_tools warning when no api_key."""

    def test_inject_tools_without_api_key_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Should warn when inject_thenvoi_tools=True but no api_key."""
        import logging

        with caplog.at_level(logging.WARNING):
            adapter = ACPClientAdapter(
                command="codex",
                inject_thenvoi_tools=True,
                api_key="",
            )

        assert not adapter._inject_thenvoi_tools
        assert "inject_thenvoi_tools=True but no api_key" in caplog.text

    def test_inject_tools_with_api_key_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Should not warn when api_key is provided."""
        import logging

        with caplog.at_level(logging.WARNING):
            adapter = ACPClientAdapter(
                command="codex",
                inject_thenvoi_tools=True,
                api_key="test-key",
            )

        assert adapter._inject_thenvoi_tools
        assert "inject_thenvoi_tools" not in caplog.text
