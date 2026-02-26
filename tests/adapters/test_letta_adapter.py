"""Tests for LettaAdapter."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from pydantic import ValidationError

from thenvoi.adapters.letta import (
    LettaAdapter,
    LettaAdapterConfig,
    _LETTA_TOOL_ENFORCEMENT,
    _RoomContext,
)
from thenvoi.converters.letta import LettaSessionState
from thenvoi.core.types import PlatformMessage
from thenvoi.testing import FakeAgentTools


def make_platform_message(
    room_id: str = "room-1", content: str = "hello"
) -> PlatformMessage:
    return PlatformMessage(
        id=str(uuid4()),
        room_id=room_id,
        content=content,
        sender_id="user-1",
        sender_type="User",
        sender_name="Alice",
        message_type="text",
        metadata={},
        created_at=datetime.now(),
    )


class ToolSchemaFakeTools(FakeAgentTools):
    """FakeAgentTools with OpenAI tool schema support."""

    def get_openai_tool_schemas(
        self, *, include_memory: bool = False
    ) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "thenvoi_send_message",
                    "description": "Send a message",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "mentions": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["content", "mentions"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "thenvoi_send_event",
                    "description": "Send an event",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "message_type": {"type": "string"},
                        },
                        "required": ["content", "message_type"],
                    },
                },
            },
        ]


def _make_letta_message(msg_type: str, **kwargs: Any) -> MagicMock:
    """Create a fake Letta response message."""
    msg = MagicMock()
    msg.message_type = msg_type
    for key, value in kwargs.items():
        setattr(msg, key, value)
    return msg


def _make_assistant_message(content: str = "Hello!") -> MagicMock:
    return _make_letta_message("assistant_message", content=content)


def _make_approval_request(
    tool_name: str = "thenvoi_send_message",
    tool_call_id: str = "tc-1",
    arguments: str = '{"content": "Hi", "mentions": ["@alice"]}',
) -> MagicMock:
    tool_call = MagicMock()
    tool_call.name = tool_name
    tool_call.tool_call_id = tool_call_id
    tool_call.arguments = arguments
    return _make_letta_message("approval_request_message", tool_call=tool_call)


def _make_letta_response(*messages: MagicMock) -> MagicMock:
    """Create a fake Letta API response."""
    response = MagicMock()
    response.messages = list(messages)
    return response


class TestLettaAdapterInit:
    """Test adapter initialization."""

    def test_default_config(self) -> None:
        adapter = LettaAdapter()
        assert adapter.config == LettaAdapterConfig()
        assert adapter._client is None
        assert adapter._rooms == {}

    def test_custom_config(self) -> None:
        config = LettaAdapterConfig(
            model="openai/gpt-4o",
            api_key="sk-let-test",
            enable_execution_reporting=True,
        )
        adapter = LettaAdapter(config=config)
        assert adapter.config.model == "openai/gpt-4o"
        assert adapter.config.api_key == "sk-let-test"
        assert adapter.config.enable_execution_reporting is True

    def test_has_history_converter(self) -> None:
        adapter = LettaAdapter()
        assert adapter.history_converter is not None


def _mock_letta_client_module() -> MagicMock:
    """Create a mock letta_client module with AsyncLetta."""
    mock_module = MagicMock()
    mock_module.AsyncLetta = MagicMock()
    return mock_module


class TestLettaAdapterOnStarted:
    """Test on_started lifecycle."""

    @pytest.mark.asyncio
    async def test_on_started_builds_system_prompt(self) -> None:
        adapter = LettaAdapter(config=LettaAdapterConfig(api_key="sk-let-test"))
        mock_module = _mock_letta_client_module()
        with patch.dict("sys.modules", {"letta_client": mock_module}):
            await adapter.on_started("TestBot", "A helpful assistant")

        assert "TestBot" in adapter._system_prompt
        assert "A helpful assistant" in adapter._system_prompt
        assert adapter._client is not None

    @pytest.mark.asyncio
    async def test_on_started_creates_client(self) -> None:
        adapter = LettaAdapter(
            config=LettaAdapterConfig(
                api_key="sk-let-test",
                base_url="https://custom.letta.com",
            )
        )
        mock_module = _mock_letta_client_module()
        with patch.dict("sys.modules", {"letta_client": mock_module}):
            await adapter.on_started("TestBot", "A helper")
            mock_module.AsyncLetta.assert_called_once_with(
                api_key="sk-let-test",
                base_url="https://custom.letta.com",
            )

    @pytest.mark.asyncio
    async def test_on_started_sets_agent_name(self) -> None:
        adapter = LettaAdapter(config=LettaAdapterConfig(api_key="sk-let-test"))
        mock_module = _mock_letta_client_module()
        with patch.dict("sys.modules", {"letta_client": mock_module}):
            await adapter.on_started("TestBot", "A helper")

        assert adapter.agent_name == "TestBot"
        assert adapter.agent_description == "A helper"

    @pytest.mark.asyncio
    async def test_on_started_import_error(self) -> None:
        adapter = LettaAdapter(config=LettaAdapterConfig(api_key="sk-let-test"))
        with (
            patch.dict("sys.modules", {"letta_client": None}),
            pytest.raises(ImportError, match="letta-client is required"),
        ):
            await adapter.on_started("TestBot", "A helper")


class TestLettaAdapterOnMessage:
    """Test on_message message handling."""

    @pytest.fixture
    def adapter(self) -> LettaAdapter:
        a = LettaAdapter(config=LettaAdapterConfig(api_key="sk-let-test"))
        a._client = AsyncMock()
        a._system_prompt = "You are TestBot."
        a.agent_name = "TestBot"
        return a

    @pytest.fixture
    def tools(self) -> ToolSchemaFakeTools:
        t = ToolSchemaFakeTools()
        t.execute_tool_call = AsyncMock(return_value={"ok": True})
        return t

    @pytest.mark.asyncio
    async def test_simple_response(
        self, adapter: LettaAdapter, tools: ToolSchemaFakeTools
    ) -> None:
        """Test a simple assistant response without tool calls."""
        # Mock agent creation
        mock_agent = MagicMock()
        mock_agent.id = "letta-agent-1"
        adapter._client.agents.create = AsyncMock(return_value=mock_agent)

        # Mock message response
        adapter._client.agents.messages.create = AsyncMock(
            return_value=_make_letta_response(_make_assistant_message("Hello Alice!"))
        )

        msg = make_platform_message()
        history = LettaSessionState()

        await adapter.on_message(
            msg,
            tools,
            history,
            None,
            None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        # Fallback auto-relay: agent didn't call thenvoi_send_message
        assert len(tools.messages_sent) == 1
        assert tools.messages_sent[0]["content"] == "Hello Alice!"
        assert tools.messages_sent[0]["mentions"] == ["user-1"]

        # Agent should be tracked
        assert adapter._rooms["room-1"].agent_id == "letta-agent-1"

    @pytest.mark.asyncio
    async def test_tool_call_loop(
        self, adapter: LettaAdapter, tools: ToolSchemaFakeTools
    ) -> None:
        """Test the approval_request_message -> tool execution -> result loop."""
        mock_agent = MagicMock()
        mock_agent.id = "letta-agent-1"
        adapter._client.agents.create = AsyncMock(return_value=mock_agent)

        # First response: tool call, second response: final text
        adapter._client.agents.messages.create = AsyncMock(
            side_effect=[
                _make_letta_response(
                    _make_approval_request(
                        tool_name="thenvoi_lookup_peers",
                        arguments="{}",
                    )
                ),
                _make_letta_response(_make_assistant_message("Found some peers!")),
            ]
        )

        msg = make_platform_message(content="Who else is here?")
        history = LettaSessionState()

        await adapter.on_message(
            msg,
            tools,
            history,
            None,
            None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        # Tool should have been executed
        tools.execute_tool_call.assert_called_once_with("thenvoi_lookup_peers", {})

        # Fallback auto-relay: thenvoi_lookup_peers != thenvoi_send_message
        assert len(tools.messages_sent) == 1
        assert tools.messages_sent[0]["content"] == "Found some peers!"

        # Second call should include approval result
        second_call = adapter._client.agents.messages.create.call_args_list[1]
        messages_arg = second_call.kwargs.get("messages") or second_call[1].get(
            "messages"
        )
        assert messages_arg[0]["type"] == "approval"
        assert messages_arg[0]["approvals"][0]["status"] == "success"

    @pytest.mark.asyncio
    async def test_send_message_tool_skips_auto_relay(
        self, adapter: LettaAdapter, tools: ToolSchemaFakeTools
    ) -> None:
        """When agent calls thenvoi_send_message, auto-relay should be skipped."""
        mock_agent = MagicMock()
        mock_agent.id = "letta-agent-1"
        adapter._client.agents.create = AsyncMock(return_value=mock_agent)

        adapter._client.agents.messages.create = AsyncMock(
            side_effect=[
                _make_letta_response(
                    _make_approval_request(
                        tool_name="thenvoi_send_message",
                        arguments='{"content": "Direct!", "mentions": ["user-1"]}',
                    )
                ),
                _make_letta_response(_make_assistant_message("Also said something")),
            ]
        )

        msg = make_platform_message()
        history = LettaSessionState()

        await adapter.on_message(
            msg,
            tools,
            history,
            None,
            None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        # Tool was executed (thenvoi_send_message)
        tools.execute_tool_call.assert_called_once()
        # Auto-relay should NOT have fired — agent used send_message directly
        assert len(tools.messages_sent) == 0

    @pytest.mark.asyncio
    async def test_tool_execution_error(
        self, adapter: LettaAdapter, tools: ToolSchemaFakeTools
    ) -> None:
        """Test tool execution error is handled and sent back to Letta."""
        mock_agent = MagicMock()
        mock_agent.id = "letta-agent-1"
        adapter._client.agents.create = AsyncMock(return_value=mock_agent)

        tools.execute_tool_call = AsyncMock(
            side_effect=RuntimeError("Connection failed")
        )

        adapter._client.agents.messages.create = AsyncMock(
            side_effect=[
                _make_letta_response(
                    _make_approval_request(
                        tool_name="thenvoi_lookup_peers",
                        arguments="{}",
                    )
                ),
                _make_letta_response(
                    _make_assistant_message("Sorry, I encountered an error.")
                ),
            ]
        )

        msg = make_platform_message()
        history = LettaSessionState()

        await adapter.on_message(
            msg,
            tools,
            history,
            None,
            None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        # Approval result should have error status
        second_call = adapter._client.agents.messages.create.call_args_list[1]
        messages_arg = second_call.kwargs.get("messages") or second_call[1].get(
            "messages"
        )
        assert messages_arg[0]["approvals"][0]["status"] == "error"
        assert "Connection failed" in messages_arg[0]["approvals"][0]["tool_return"]

    @pytest.mark.asyncio
    async def test_tool_validation_error(
        self, adapter: LettaAdapter, tools: ToolSchemaFakeTools
    ) -> None:
        """Test pydantic ValidationError is caught separately."""
        mock_agent = MagicMock()
        mock_agent.id = "letta-agent-1"
        adapter._client.agents.create = AsyncMock(return_value=mock_agent)

        # Create a real ValidationError
        from pydantic import BaseModel

        class StrictModel(BaseModel):
            name: str

        try:
            StrictModel(name=123)  # type: ignore[arg-type]
        except ValidationError as real_error:
            tools.execute_tool_call = AsyncMock(side_effect=real_error)

        adapter._client.agents.messages.create = AsyncMock(
            side_effect=[
                _make_letta_response(
                    _make_approval_request(
                        tool_name="thenvoi_lookup_peers",
                        arguments="{}",
                    )
                ),
                _make_letta_response(_make_assistant_message("Handled error.")),
            ]
        )

        msg = make_platform_message()
        history = LettaSessionState()

        await adapter.on_message(
            msg,
            tools,
            history,
            None,
            None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        # Approval result should have error status with validation message
        second_call = adapter._client.agents.messages.create.call_args_list[1]
        messages_arg = second_call.kwargs.get("messages") or second_call[1].get(
            "messages"
        )
        assert messages_arg[0]["approvals"][0]["status"] == "error"
        assert (
            "Invalid arguments for thenvoi_lookup_peers"
            in (messages_arg[0]["approvals"][0]["tool_return"])
        )

    @pytest.mark.asyncio
    async def test_timeout_reports_error(
        self, adapter: LettaAdapter, tools: ToolSchemaFakeTools
    ) -> None:
        """Test that a timeout during Letta turn reports an error event."""
        adapter.config.turn_timeout_s = 0.01  # Very short timeout

        mock_agent = MagicMock()
        mock_agent.id = "letta-agent-1"
        adapter._client.agents.create = AsyncMock(return_value=mock_agent)

        # Simulate a slow response
        async def slow_response(**kwargs: Any) -> MagicMock:
            import asyncio

            await asyncio.sleep(1.0)
            return _make_letta_response(_make_assistant_message("Too late"))

        adapter._client.agents.messages.create = AsyncMock(side_effect=slow_response)

        msg = make_platform_message()
        history = LettaSessionState()

        await adapter.on_message(
            msg,
            tools,
            history,
            None,
            None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        # Should have sent an error event about timeout
        error_events = [e for e in tools.events_sent if e["message_type"] == "error"]
        assert len(error_events) == 1
        assert "timed out" in error_events[0]["content"]

    @pytest.mark.asyncio
    async def test_agent_resumption(
        self, adapter: LettaAdapter, tools: ToolSchemaFakeTools
    ) -> None:
        """Test resuming an existing Letta agent from history."""
        adapter._client.agents.retrieve = AsyncMock(return_value=MagicMock())
        adapter._client.agents.messages.create = AsyncMock(
            return_value=_make_letta_response(_make_assistant_message("Resumed!"))
        )

        msg = make_platform_message()
        history = LettaSessionState(
            agent_id="existing-agent-id",
            conversation_id="existing-conv-id",
            room_id="room-1",
        )

        await adapter.on_message(
            msg,
            tools,
            history,
            None,
            None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        # Should have tried to retrieve the existing agent
        adapter._client.agents.retrieve.assert_called_once_with("existing-agent-id")
        # Should NOT have created a new agent
        adapter._client.agents.create.assert_not_called()
        # Agent should be tracked
        assert adapter._rooms["room-1"].agent_id == "existing-agent-id"
        assert adapter._rooms["room-1"].conversation_id == "existing-conv-id"

    @pytest.mark.asyncio
    async def test_agent_resume_failure_creates_new(
        self, adapter: LettaAdapter, tools: ToolSchemaFakeTools
    ) -> None:
        """If resuming fails, create a new agent."""
        adapter._client.agents.retrieve = AsyncMock(side_effect=Exception("Not found"))

        mock_agent = MagicMock()
        mock_agent.id = "new-agent-id"
        adapter._client.agents.create = AsyncMock(return_value=mock_agent)
        adapter._client.agents.messages.create = AsyncMock(
            return_value=_make_letta_response(_make_assistant_message("New agent!"))
        )

        msg = make_platform_message()
        history = LettaSessionState(agent_id="dead-agent-id")

        await adapter.on_message(
            msg,
            tools,
            history,
            None,
            None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        adapter._client.agents.create.assert_called_once()
        assert adapter._rooms["room-1"].agent_id == "new-agent-id"

    @pytest.mark.asyncio
    async def test_config_agent_id_reuses_existing(
        self, tools: ToolSchemaFakeTools
    ) -> None:
        """Test that config.agent_id reuses a pre-existing Letta agent."""
        adapter = LettaAdapter(
            config=LettaAdapterConfig(
                api_key="sk-let-test",
                agent_id="preconfigured-agent-id",
            )
        )
        adapter._client = AsyncMock()
        adapter._system_prompt = "You are TestBot."
        adapter.agent_name = "TestBot"

        adapter._client.agents.retrieve = AsyncMock(return_value=MagicMock())
        adapter._client.agents.messages.create = AsyncMock(
            return_value=_make_letta_response(_make_assistant_message("Reused!"))
        )

        msg = make_platform_message()
        history = LettaSessionState()  # No agent_id in history

        await adapter.on_message(
            msg,
            tools,
            history,
            None,
            None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        # Should have retrieved the config agent, not created a new one
        adapter._client.agents.retrieve.assert_called_once_with(
            "preconfigured-agent-id"
        )
        adapter._client.agents.create.assert_not_called()
        assert adapter._rooms["room-1"].agent_id == "preconfigured-agent-id"

    @pytest.mark.asyncio
    async def test_history_agent_id_takes_precedence_over_config(
        self, tools: ToolSchemaFakeTools
    ) -> None:
        """History agent_id should be preferred over config.agent_id."""
        adapter = LettaAdapter(
            config=LettaAdapterConfig(
                api_key="sk-let-test",
                agent_id="config-agent-id",
            )
        )
        adapter._client = AsyncMock()
        adapter._system_prompt = "You are TestBot."
        adapter.agent_name = "TestBot"

        adapter._client.agents.retrieve = AsyncMock(return_value=MagicMock())
        adapter._client.agents.messages.create = AsyncMock(
            return_value=_make_letta_response(_make_assistant_message("From history!"))
        )

        msg = make_platform_message()
        history = LettaSessionState(agent_id="history-agent-id")

        await adapter.on_message(
            msg,
            tools,
            history,
            None,
            None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        # History agent_id should win
        adapter._client.agents.retrieve.assert_called_once_with("history-agent-id")
        assert adapter._rooms["room-1"].agent_id == "history-agent-id"

    @pytest.mark.asyncio
    async def test_participants_and_contacts_injected(
        self, adapter: LettaAdapter, tools: ToolSchemaFakeTools
    ) -> None:
        """Participants and contacts messages are included in the user message."""
        mock_agent = MagicMock()
        mock_agent.id = "letta-agent-1"
        adapter._client.agents.create = AsyncMock(return_value=mock_agent)
        adapter._client.agents.messages.create = AsyncMock(
            return_value=_make_letta_response(_make_assistant_message("Got it!"))
        )

        msg = make_platform_message(content="What's up?")
        history = LettaSessionState()

        await adapter.on_message(
            msg,
            tools,
            history,
            "Bob joined the room",
            "Charlie is now a contact",
            is_session_bootstrap=True,
            room_id="room-1",
        )

        # Check the message sent to Letta includes participants and contacts
        call_kwargs = adapter._client.agents.messages.create.call_args
        messages_arg = call_kwargs.kwargs.get("messages") or call_kwargs[1].get(
            "messages"
        )
        content = messages_arg[0]["content"]
        assert "[System]: Bob joined the room" in content
        assert "[System]: Charlie is now a contact" in content

    @pytest.mark.asyncio
    async def test_no_client_reports_error(self, tools: ToolSchemaFakeTools) -> None:
        """If client is not initialized, report error."""
        adapter = LettaAdapter()
        adapter._client = None

        msg = make_platform_message()
        history = LettaSessionState()

        await adapter.on_message(
            msg,
            tools,
            history,
            None,
            None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        # Should have sent an error event
        error_events = [e for e in tools.events_sent if e["message_type"] == "error"]
        assert len(error_events) == 1
        assert "not initialized" in error_events[0]["content"]

    @pytest.mark.asyncio
    async def test_empty_response_no_message_sent(
        self, adapter: LettaAdapter, tools: ToolSchemaFakeTools
    ) -> None:
        """If Letta returns no text, don't send an empty message."""
        mock_agent = MagicMock()
        mock_agent.id = "letta-agent-1"
        adapter._client.agents.create = AsyncMock(return_value=mock_agent)
        adapter._client.agents.messages.create = AsyncMock(
            return_value=_make_letta_response(
                _make_letta_message("internal_monologue", content="Thinking...")
            )
        )

        msg = make_platform_message()
        history = LettaSessionState()

        await adapter.on_message(
            msg,
            tools,
            history,
            None,
            None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        assert len(tools.messages_sent) == 0


class TestLettaAdapterOnCleanup:
    """Test on_cleanup lifecycle."""

    @pytest.mark.asyncio
    async def test_cleanup_removes_room_state(self) -> None:
        adapter = LettaAdapter()
        adapter._rooms["room-1"] = _RoomContext(
            agent_id="agent-1", conversation_id="conv-1"
        )

        await adapter.on_cleanup("room-1")

        assert "room-1" not in adapter._rooms

    @pytest.mark.asyncio
    async def test_cleanup_idempotent(self) -> None:
        """Cleaning up a non-existent room should not raise."""
        adapter = LettaAdapter()
        await adapter.on_cleanup("room-nonexistent")
        await adapter.on_cleanup("room-nonexistent")  # Second call also safe

    @pytest.mark.asyncio
    async def test_cleanup_before_started(self) -> None:
        """Cleaning up before adapter is started should not raise."""
        adapter = LettaAdapter()
        assert adapter._client is None
        await adapter.on_cleanup("room-1")  # No error

    @pytest.mark.asyncio
    async def test_cleanup_multiple_rooms(self) -> None:
        """Cleaning up one room should not affect others."""
        adapter = LettaAdapter()
        adapter._rooms["room-1"] = _RoomContext(agent_id="agent-1")
        adapter._rooms["room-2"] = _RoomContext(agent_id="agent-2")

        await adapter.on_cleanup("room-1")

        assert "room-1" not in adapter._rooms
        assert adapter._rooms["room-2"].agent_id == "agent-2"


class TestLettaAdapterBuildClientTools:
    """Test _build_client_tools method."""

    def test_builds_from_openai_schemas(self) -> None:
        adapter = LettaAdapter()
        tools = ToolSchemaFakeTools()

        client_tools = adapter._build_client_tools(tools)

        assert len(client_tools) == 2
        assert client_tools[0]["name"] == "thenvoi_send_message"
        assert client_tools[0]["description"] == "Send a message"
        assert "properties" in client_tools[0]["parameters"]
        assert client_tools[1]["name"] == "thenvoi_send_event"


class TestLettaAdapterExecutionReporting:
    """Test execution reporting when enabled."""

    @pytest.mark.asyncio
    async def test_reports_tool_calls_when_enabled(self) -> None:
        adapter = LettaAdapter(
            config=LettaAdapterConfig(
                api_key="sk-let-test",
                enable_execution_reporting=True,
            )
        )
        adapter._client = AsyncMock()
        adapter._system_prompt = "You are TestBot."
        adapter.agent_name = "TestBot"

        mock_agent = MagicMock()
        mock_agent.id = "letta-agent-1"
        adapter._client.agents.create = AsyncMock(return_value=mock_agent)

        tools = ToolSchemaFakeTools()
        tools.execute_tool_call = AsyncMock(return_value={"peers": []})

        adapter._client.agents.messages.create = AsyncMock(
            side_effect=[
                _make_letta_response(
                    _make_approval_request(
                        tool_name="thenvoi_lookup_peers",
                        arguments="{}",
                    )
                ),
                _make_letta_response(_make_assistant_message("No peers found.")),
            ]
        )

        msg = make_platform_message()
        history = LettaSessionState()

        await adapter.on_message(
            msg,
            tools,
            history,
            None,
            None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        # Should have reported tool_call and tool_result events
        tool_call_events = [
            e for e in tools.events_sent if e["message_type"] == "tool_call"
        ]
        tool_result_events = [
            e for e in tools.events_sent if e["message_type"] == "tool_result"
        ]
        assert len(tool_call_events) == 1
        assert len(tool_result_events) == 1
        assert "thenvoi_lookup_peers" in tool_call_events[0]["content"]

    @pytest.mark.asyncio
    async def test_silent_tools_not_reported(self) -> None:
        """thenvoi_send_message and thenvoi_send_event should not be reported."""
        adapter = LettaAdapter(
            config=LettaAdapterConfig(
                api_key="sk-let-test",
                enable_execution_reporting=True,
            )
        )
        adapter._client = AsyncMock()
        adapter._system_prompt = "You are TestBot."
        adapter.agent_name = "TestBot"

        mock_agent = MagicMock()
        mock_agent.id = "letta-agent-1"
        adapter._client.agents.create = AsyncMock(return_value=mock_agent)

        tools = ToolSchemaFakeTools()
        tools.execute_tool_call = AsyncMock(return_value={"id": "msg-1"})

        adapter._client.agents.messages.create = AsyncMock(
            side_effect=[
                _make_letta_response(
                    _make_approval_request(
                        tool_name="thenvoi_send_message",
                        arguments='{"content": "Hi", "mentions": ["@alice"]}',
                    )
                ),
                _make_letta_response(_make_assistant_message("Done.")),
            ]
        )

        msg = make_platform_message()
        history = LettaSessionState()

        await adapter.on_message(
            msg,
            tools,
            history,
            None,
            None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        # Should NOT have reported tool_call/tool_result for silent tools
        tool_events = [
            e
            for e in tools.events_sent
            if e["message_type"] in ("tool_call", "tool_result")
        ]
        assert len(tool_events) == 0


class TestLettaAdapterInstructionBlockUpdate:
    """Test _update_instruction_block fallback chain."""

    @pytest.mark.asyncio
    async def test_updates_persona_block(self) -> None:
        """Should update 'persona' block when it exists."""
        adapter = LettaAdapter(config=LettaAdapterConfig(api_key="sk-let-test"))
        adapter._client = AsyncMock()
        adapter._system_prompt = "You are TestBot."

        # First label ("persona") succeeds
        adapter._client.agents.blocks.update = AsyncMock(return_value=None)

        await adapter._update_instruction_block("agent-1", "room-1")

        adapter._client.agents.blocks.update.assert_called_once_with(
            "persona",
            agent_id="agent-1",
            value=_LETTA_TOOL_ENFORCEMENT + "You are TestBot.",
        )

    @pytest.mark.asyncio
    async def test_falls_back_to_custom_instructions(self) -> None:
        """Should try 'custom_instructions' when 'persona' fails."""
        adapter = LettaAdapter(config=LettaAdapterConfig(api_key="sk-let-test"))
        adapter._client = AsyncMock()
        adapter._system_prompt = "You are TestBot."

        # "persona" fails, "custom_instructions" succeeds
        adapter._client.agents.blocks.update = AsyncMock(
            side_effect=[Exception("Not found"), None]
        )

        await adapter._update_instruction_block("agent-1", "room-1")

        assert adapter._client.agents.blocks.update.call_count == 2
        second_call = adapter._client.agents.blocks.update.call_args_list[1]
        assert second_call.args[0] == "custom_instructions"

    @pytest.mark.asyncio
    async def test_creates_persona_when_all_labels_fail(self) -> None:
        """Should create a new 'persona' block when no known labels exist."""
        adapter = LettaAdapter(config=LettaAdapterConfig(api_key="sk-let-test"))
        adapter._client = AsyncMock()
        adapter._system_prompt = "You are TestBot."

        # All label updates fail
        adapter._client.agents.blocks.update = AsyncMock(
            side_effect=Exception("Not found")
        )
        adapter._client.agents.blocks.create = AsyncMock(return_value=None)

        await adapter._update_instruction_block("agent-1", "room-1")

        # Should have tried all 3 labels, then created
        assert adapter._client.agents.blocks.update.call_count == 3
        adapter._client.agents.blocks.create.assert_called_once()
        create_kwargs = adapter._client.agents.blocks.create.call_args.kwargs
        assert create_kwargs["label"] == "persona"


class TestFormatTimeAgo:
    """Test _format_time_ago static method."""

    def test_seconds(self) -> None:
        dt = datetime.now(timezone.utc) - timedelta(seconds=30)
        result = LettaAdapter._format_time_ago(dt)
        assert result == "30s"

    def test_minutes(self) -> None:
        dt = datetime.now(timezone.utc) - timedelta(minutes=15)
        result = LettaAdapter._format_time_ago(dt)
        assert result == "15m"

    def test_one_hour(self) -> None:
        dt = datetime.now(timezone.utc) - timedelta(hours=1)
        result = LettaAdapter._format_time_ago(dt)
        assert result == "1 hour"

    def test_multiple_hours(self) -> None:
        dt = datetime.now(timezone.utc) - timedelta(hours=5)
        result = LettaAdapter._format_time_ago(dt)
        assert result == "5h"

    def test_one_day(self) -> None:
        dt = datetime.now(timezone.utc) - timedelta(days=1)
        result = LettaAdapter._format_time_ago(dt)
        assert result == "1 day"

    def test_multiple_days(self) -> None:
        dt = datetime.now(timezone.utc) - timedelta(days=3)
        result = LettaAdapter._format_time_ago(dt)
        assert result == "3d"

    def test_naive_datetime_treated_as_utc(self) -> None:
        dt = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=10)
        result = LettaAdapter._format_time_ago(dt)
        assert result == "10m"


class TestExtractSummary:
    """Test _extract_summary static method."""

    def test_first_sentence(self) -> None:
        result = LettaAdapter._extract_summary(
            ["We discussed the API design. Then we moved on."]
        )
        assert result == "We discussed the API design."

    def test_first_sentence_exclamation(self) -> None:
        result = LettaAdapter._extract_summary(["Great news! More details follow."])
        assert result == "Great news!"

    def test_first_sentence_question(self) -> None:
        result = LettaAdapter._extract_summary(["How are you? Fine thanks."])
        assert result == "How are you?"

    def test_truncation_when_no_delimiter(self) -> None:
        long_text = "a " * 100  # 200 chars, no sentence delimiter
        result = LettaAdapter._extract_summary([long_text], max_length=50)
        assert len(result) <= 53  # 50 + "..."
        assert result.endswith("...")

    def test_short_text_no_delimiter(self) -> None:
        result = LettaAdapter._extract_summary(["Short text"])
        assert result == "Short text"

    def test_empty_parts(self) -> None:
        result = LettaAdapter._extract_summary([])
        assert result == ""

    def test_empty_string_parts(self) -> None:
        result = LettaAdapter._extract_summary(["", ""])
        assert result == ""

    def test_multiple_parts_joined(self) -> None:
        result = LettaAdapter._extract_summary(["Hello world.", "More text."])
        assert result == "Hello world."

    def test_custom_max_length(self) -> None:
        result = LettaAdapter._extract_summary(["Short."], max_length=10)
        assert result == "Short."


class TestRejoinContextInjection:
    """Test rejoin context injection in _handle_message."""

    @pytest.fixture
    def adapter(self) -> LettaAdapter:
        a = LettaAdapter(config=LettaAdapterConfig(api_key="sk-let-test"))
        a._client = AsyncMock()
        a._system_prompt = "You are TestBot."
        a.agent_name = "TestBot"
        return a

    @pytest.fixture
    def tools(self) -> ToolSchemaFakeTools:
        t = ToolSchemaFakeTools()
        t.execute_tool_call = AsyncMock(return_value={"ok": True})
        return t

    @pytest.mark.asyncio
    async def test_rejoin_context_injected_on_bootstrap(
        self, adapter: LettaAdapter, tools: ToolSchemaFakeTools
    ) -> None:
        """When resuming a room with last_interaction, inject rejoin context."""
        two_hours_ago = datetime.now(timezone.utc) - timedelta(hours=2)
        adapter._rooms["room-1"] = _RoomContext(
            agent_id="agent-1",
            last_interaction=two_hours_ago,
            summary="User asked about API design",
        )

        adapter._client.agents.messages.create = AsyncMock(
            return_value=_make_letta_response(_make_assistant_message("Welcome back!"))
        )

        msg = make_platform_message()
        history = LettaSessionState()

        await adapter.on_message(
            msg,
            tools,
            history,
            None,
            None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        # Check that the message includes rejoin context
        call_kwargs = adapter._client.agents.messages.create.call_args
        messages_arg = call_kwargs.kwargs.get("messages") or call_kwargs[1].get(
            "messages"
        )
        content = messages_arg[0]["content"]
        assert "[System: You have rejoined this room after 2h." in content
        assert "Previous topic: User asked about API design" in content

    @pytest.mark.asyncio
    async def test_no_rejoin_without_last_interaction(
        self, adapter: LettaAdapter, tools: ToolSchemaFakeTools
    ) -> None:
        """No rejoin context when room has no last_interaction."""
        adapter._rooms["room-1"] = _RoomContext(agent_id="agent-1")

        adapter._client.agents.messages.create = AsyncMock(
            return_value=_make_letta_response(_make_assistant_message("Hello!"))
        )

        msg = make_platform_message()
        history = LettaSessionState()

        await adapter.on_message(
            msg,
            tools,
            history,
            None,
            None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        call_kwargs = adapter._client.agents.messages.create.call_args
        messages_arg = call_kwargs.kwargs.get("messages") or call_kwargs[1].get(
            "messages"
        )
        content = messages_arg[0]["content"]
        assert "[System: You have rejoined" not in content

    @pytest.mark.asyncio
    async def test_no_rejoin_when_not_bootstrap(
        self, adapter: LettaAdapter, tools: ToolSchemaFakeTools
    ) -> None:
        """No rejoin context when is_session_bootstrap is False."""
        two_hours_ago = datetime.now(timezone.utc) - timedelta(hours=2)
        adapter._rooms["room-1"] = _RoomContext(
            agent_id="agent-1",
            last_interaction=two_hours_ago,
            summary="Previous topic",
        )

        adapter._client.agents.messages.create = AsyncMock(
            return_value=_make_letta_response(_make_assistant_message("Hello!"))
        )

        msg = make_platform_message()
        history = LettaSessionState()

        await adapter.on_message(
            msg,
            tools,
            history,
            None,
            None,
            is_session_bootstrap=False,
            room_id="room-1",
        )

        call_kwargs = adapter._client.agents.messages.create.call_args
        messages_arg = call_kwargs.kwargs.get("messages") or call_kwargs[1].get(
            "messages"
        )
        content = messages_arg[0]["content"]
        assert "[System: You have rejoined" not in content


class TestSummaryExtraction:
    """Test that summary is stored in _RoomContext after successful turn."""

    @pytest.mark.asyncio
    async def test_summary_stored_after_turn(self) -> None:
        adapter = LettaAdapter(config=LettaAdapterConfig(api_key="sk-let-test"))
        adapter._client = AsyncMock()
        adapter._system_prompt = "You are TestBot."
        adapter.agent_name = "TestBot"

        mock_agent = MagicMock()
        mock_agent.id = "letta-agent-1"
        adapter._client.agents.create = AsyncMock(return_value=mock_agent)
        adapter._client.agents.messages.create = AsyncMock(
            return_value=_make_letta_response(
                _make_assistant_message("The API uses REST endpoints. More details.")
            )
        )

        tools = ToolSchemaFakeTools()
        tools.execute_tool_call = AsyncMock(return_value={"ok": True})

        msg = make_platform_message()
        history = LettaSessionState()

        await adapter.on_message(
            msg,
            tools,
            history,
            None,
            None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        room_ctx = adapter._rooms["room-1"]
        assert room_ctx.summary == "The API uses REST endpoints."
        assert room_ctx.last_interaction is not None

    @pytest.mark.asyncio
    async def test_no_summary_when_no_text(self) -> None:
        adapter = LettaAdapter(config=LettaAdapterConfig(api_key="sk-let-test"))
        adapter._client = AsyncMock()
        adapter._system_prompt = "You are TestBot."
        adapter.agent_name = "TestBot"

        mock_agent = MagicMock()
        mock_agent.id = "letta-agent-1"
        adapter._client.agents.create = AsyncMock(return_value=mock_agent)
        adapter._client.agents.messages.create = AsyncMock(
            return_value=_make_letta_response(
                _make_letta_message("internal_monologue", content="Thinking...")
            )
        )

        tools = ToolSchemaFakeTools()
        tools.execute_tool_call = AsyncMock(return_value={"ok": True})

        msg = make_platform_message()
        history = LettaSessionState()

        await adapter.on_message(
            msg,
            tools,
            history,
            None,
            None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        room_ctx = adapter._rooms["room-1"]
        # No assistant text, so summary should remain None
        assert room_ctx.summary is None
        # last_interaction should still be set
        assert room_ctx.last_interaction is not None


class TestMemoryConsolidation:
    """Test memory consolidation on cleanup."""

    @pytest.mark.asyncio
    async def test_consolidation_sent_on_cleanup(self) -> None:
        adapter = LettaAdapter()
        adapter._client = AsyncMock()
        adapter._rooms["room-1"] = _RoomContext(agent_id="agent-1")

        await adapter.on_cleanup("room-1")

        # Should have sent consolidation prompt
        adapter._client.agents.messages.create.assert_called_once()
        call_kwargs = adapter._client.agents.messages.create.call_args
        messages_arg = call_kwargs.kwargs.get("messages") or call_kwargs[1].get(
            "messages"
        )
        assert "leaving this room" in messages_arg[0]["content"]
        assert "Consolidate key decisions" in messages_arg[0]["content"]

        # Room should be removed
        assert "room-1" not in adapter._rooms

    @pytest.mark.asyncio
    async def test_consolidation_failure_does_not_block_cleanup(self) -> None:
        adapter = LettaAdapter()
        adapter._client = AsyncMock()
        adapter._client.agents.messages.create = AsyncMock(
            side_effect=RuntimeError("Network error")
        )
        adapter._rooms["room-1"] = _RoomContext(agent_id="agent-1")

        await adapter.on_cleanup("room-1")

        # Room should still be cleaned up despite consolidation failure
        assert "room-1" not in adapter._rooms

    @pytest.mark.asyncio
    async def test_no_consolidation_without_client(self) -> None:
        adapter = LettaAdapter()
        adapter._client = None
        adapter._rooms["room-1"] = _RoomContext(agent_id="agent-1")

        await adapter.on_cleanup("room-1")

        assert "room-1" not in adapter._rooms

    @pytest.mark.asyncio
    async def test_no_consolidation_for_unknown_room(self) -> None:
        adapter = LettaAdapter()
        adapter._client = AsyncMock()

        await adapter.on_cleanup("room-nonexistent")

        # No messages should have been sent
        adapter._client.agents.messages.create.assert_not_called()
