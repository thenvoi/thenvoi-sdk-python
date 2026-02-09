"""Parlant adapter-specific tests.

Tests for Parlant adapter-specific behavior that isn't covered by conformance tests:
- History injection (complete exchange injection into Parlant sessions)
- Session and customer management
- Parlant SDK integration patterns
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from thenvoi.adapters.parlant import ParlantAdapter
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
    tools.get_tool_schemas = MagicMock(return_value=[])
    tools.get_openai_tool_schemas = MagicMock(return_value=[])
    tools.send_message = AsyncMock(return_value={"status": "sent"})
    tools.send_event = AsyncMock(return_value={"status": "sent"})
    tools.execute_tool_call = AsyncMock(return_value={"status": "success"})
    return tools


@pytest.fixture
def mock_parlant_server():
    """Create mock Parlant SDK Server."""
    server = MagicMock()

    mock_app = MagicMock()
    mock_app.sessions = AsyncMock()
    mock_app.sessions.create = AsyncMock(return_value=MagicMock(id="session-123"))
    mock_app.sessions.create_customer_message = AsyncMock(
        return_value=MagicMock(offset=1)
    )
    mock_app.sessions.create_event = AsyncMock()
    mock_app.sessions.wait_for_update = AsyncMock(return_value=True)
    mock_app.sessions.find_events = AsyncMock(return_value=[])

    server.container = {MagicMock: mock_app}
    server.create_customer = AsyncMock(return_value=MagicMock(id="customer-123"))

    return server


@pytest.fixture
def mock_parlant_agent():
    """Create mock Parlant Agent."""
    agent = MagicMock()
    agent.id = "parlant-agent-123"
    agent.name = "TestBot"
    return agent


class TestHistoryInjection:
    """Tests for history injection into Parlant sessions."""

    @pytest.fixture
    def adapter_with_app(self, mock_parlant_server, mock_parlant_agent):
        """Create adapter with mocked application."""
        adapter = ParlantAdapter(
            server=mock_parlant_server,
            parlant_agent=mock_parlant_agent,
        )
        adapter.agent_name = "TestBot"

        mock_app = MagicMock()
        mock_app.sessions = AsyncMock()
        mock_app.sessions.create_customer_message = AsyncMock(
            return_value=MagicMock(offset=1)
        )
        mock_app.sessions.create_event = AsyncMock()

        adapter._app = mock_app
        return adapter

    @pytest.mark.asyncio
    async def test_injects_complete_exchanges_only(self, adapter_with_app):
        """Should only inject complete user-assistant exchanges."""
        history = [
            {"role": "user", "content": "Hello", "sender": "Alice"},
            {"role": "assistant", "content": "Hi there!", "sender": "TestBot"},
            {
                "role": "user",
                "content": "Pending question",
            },  # No response - should skip
        ]

        mock_moderation = MagicMock()
        mock_moderation.NONE = "none"

        mock_event_kind = MagicMock()
        mock_event_kind.MESSAGE = "message"

        mock_event_source = MagicMock()
        mock_event_source.CUSTOMER = "customer"
        mock_event_source.AI_AGENT = "ai_agent"

        with patch.dict(
            sys.modules,
            {
                "parlant.core.app_modules.sessions": MagicMock(
                    Moderation=mock_moderation
                ),
                "parlant.core.sessions": MagicMock(
                    EventKind=mock_event_kind,
                    EventSource=mock_event_source,
                ),
            },
        ):
            count = await adapter_with_app._inject_history("session-123", history)

        # Should inject 2 messages (complete exchange), skip the pending question
        assert count == 2

    @pytest.mark.asyncio
    async def test_handles_empty_history(self, adapter_with_app):
        """Should handle empty history gracefully."""
        count = await adapter_with_app._inject_history("session-123", [])
        assert count == 0

    @pytest.mark.asyncio
    async def test_handles_history_with_only_user_messages(self, adapter_with_app):
        """Should handle history with no complete exchanges."""
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "Anyone there?"},
        ]

        count = await adapter_with_app._inject_history("session-123", history)
        assert count == 0

    @pytest.mark.asyncio
    async def test_handles_multiple_exchanges(self, adapter_with_app):
        """Should inject multiple complete exchanges."""
        history = [
            {"role": "user", "content": "Q1", "sender": "Alice"},
            {"role": "assistant", "content": "A1", "sender": "TestBot"},
            {"role": "user", "content": "Q2", "sender": "Alice"},
            {"role": "assistant", "content": "A2", "sender": "TestBot"},
        ]

        mock_moderation = MagicMock()
        mock_moderation.NONE = "none"

        mock_event_kind = MagicMock()
        mock_event_kind.MESSAGE = "message"

        mock_event_source = MagicMock()
        mock_event_source.CUSTOMER = "customer"
        mock_event_source.AI_AGENT = "ai_agent"

        with patch.dict(
            sys.modules,
            {
                "parlant.core.app_modules.sessions": MagicMock(
                    Moderation=mock_moderation
                ),
                "parlant.core.sessions": MagicMock(
                    EventKind=mock_event_kind,
                    EventSource=mock_event_source,
                ),
            },
        ):
            count = await adapter_with_app._inject_history("session-123", history)

        # Should inject all 4 messages (2 complete exchanges)
        assert count == 4


class TestSessionManagement:
    """Tests for Parlant session and customer management."""

    def test_internal_state_initialized(self, mock_parlant_server, mock_parlant_agent):
        """Should initialize internal state correctly."""
        adapter = ParlantAdapter(
            server=mock_parlant_server,
            parlant_agent=mock_parlant_agent,
        )

        assert adapter._app is None
        assert adapter._room_sessions == {}
        assert adapter._room_customers == {}
        assert adapter._system_prompt == ""

    @pytest.mark.asyncio
    async def test_cleans_up_session(self, mock_parlant_server, mock_parlant_agent):
        """Should clean up Parlant session on cleanup."""
        adapter = ParlantAdapter(
            server=mock_parlant_server,
            parlant_agent=mock_parlant_agent,
        )
        adapter._room_sessions["room-123"] = "session-123"
        adapter._room_customers["room-123"] = "customer-123"

        await adapter.on_cleanup("room-123")

        assert "room-123" not in adapter._room_sessions
        assert "room-123" not in adapter._room_customers

    @pytest.mark.asyncio
    async def test_cleanup_nonexistent_room_is_safe(
        self, mock_parlant_server, mock_parlant_agent
    ):
        """Should handle cleanup of non-existent room."""
        adapter = ParlantAdapter(
            server=mock_parlant_server,
            parlant_agent=mock_parlant_agent,
        )

        # Should not raise
        await adapter.on_cleanup("nonexistent-room")

    @pytest.mark.asyncio
    async def test_cleans_up_all_sessions(
        self, mock_parlant_server, mock_parlant_agent
    ):
        """Should cleanup all sessions on cleanup_all."""
        adapter = ParlantAdapter(
            server=mock_parlant_server,
            parlant_agent=mock_parlant_agent,
        )
        adapter._room_sessions["room-1"] = "session-1"
        adapter._room_sessions["room-2"] = "session-2"
        adapter._room_customers["room-1"] = "customer-1"
        adapter._room_customers["room-2"] = "customer-2"

        await adapter.cleanup_all()

        assert len(adapter._room_sessions) == 0
        assert len(adapter._room_customers) == 0


class TestSystemPromptHandling:
    """Tests for system prompt rendering."""

    @pytest.fixture
    def mock_application_class(self):
        """Create a mock Application class for testing."""
        return MagicMock(name="Application")

    @pytest.mark.asyncio
    async def test_renders_system_prompt_from_agent_metadata(
        self, mock_parlant_server, mock_parlant_agent, mock_application_class
    ):
        """Should render system prompt from agent metadata."""
        adapter = ParlantAdapter(
            server=mock_parlant_server,
            parlant_agent=mock_parlant_agent,
        )

        mock_app = MagicMock()
        mock_module = MagicMock()
        mock_module.Application = mock_application_class
        mock_parlant_server.container = {mock_application_class: mock_app}

        with patch.dict(
            sys.modules,
            {"parlant.core.application": mock_module},
        ):
            await adapter.on_started(
                agent_name="TestBot", agent_description="A test bot"
            )

        assert adapter.agent_name == "TestBot"
        assert adapter.agent_description == "A test bot"
        assert adapter._system_prompt != ""
        assert "TestBot" in adapter._system_prompt

    @pytest.mark.asyncio
    async def test_uses_custom_system_prompt_if_provided(
        self, mock_parlant_server, mock_parlant_agent, mock_application_class
    ):
        """Should use custom system_prompt if provided."""
        adapter = ParlantAdapter(
            server=mock_parlant_server,
            parlant_agent=mock_parlant_agent,
            system_prompt="You are a custom assistant.",
        )

        mock_app = MagicMock()
        mock_module = MagicMock()
        mock_module.Application = mock_application_class
        mock_parlant_server.container = {mock_application_class: mock_app}

        with patch.dict(
            sys.modules,
            {"parlant.core.application": mock_module},
        ):
            await adapter.on_started(
                agent_name="TestBot", agent_description="A test bot"
            )

        assert adapter._system_prompt == "You are a custom assistant."


class TestOnMessage:
    """Tests for on_message() session management and message sending."""

    @pytest.fixture
    def initialized_adapter(self, mock_parlant_server, mock_parlant_agent):
        """Create adapter with _app pre-set."""
        adapter = ParlantAdapter(
            server=mock_parlant_server,
            parlant_agent=mock_parlant_agent,
        )
        adapter.agent_name = "TestBot"
        adapter.agent_description = "A test bot"

        mock_app = MagicMock()
        mock_app.sessions = AsyncMock()
        mock_app.sessions.create = AsyncMock(return_value=MagicMock(id="session-123"))
        mock_app.sessions.create_customer_message = AsyncMock(
            return_value=MagicMock(offset=1)
        )
        mock_app.sessions.wait_for_update = AsyncMock(return_value=True)
        mock_app.sessions.find_events = AsyncMock(return_value=[])

        adapter._app = mock_app
        return adapter

    @pytest.mark.asyncio
    async def test_creates_session_for_room(
        self, initialized_adapter, sample_message, mock_tools
    ):
        """New room → create_customer + sessions.create called, session cached."""
        adapter = initialized_adapter

        mock_moderation = MagicMock()
        mock_moderation.NONE = "none"
        mock_event_source = MagicMock()
        mock_event_source.CUSTOMER = "customer"
        mock_event_source.AI_AGENT = "ai_agent"
        mock_event_kind = MagicMock()
        mock_event_kind.MESSAGE = "message"

        with patch.dict(
            sys.modules,
            {
                "parlant.core.app_modules.sessions": MagicMock(
                    Moderation=mock_moderation
                ),
                "parlant.core.sessions": MagicMock(
                    EventSource=mock_event_source, EventKind=mock_event_kind
                ),
                "parlant.core.async_utils": MagicMock(Timeout=lambda x: x),
            },
        ):
            await adapter.on_message(
                msg=sample_message,
                tools=mock_tools,
                history=[],
                participants_msg=None,
                is_session_bootstrap=True,
                room_id="room-123",
            )

        # Verify session was created and cached
        adapter._app.sessions.create.assert_awaited_once()
        assert "room-123" in adapter._room_sessions

    @pytest.mark.asyncio
    async def test_sends_customer_message_to_parlant(
        self, initialized_adapter, sample_message, mock_tools
    ):
        """create_customer_message should be called with correct params."""
        adapter = initialized_adapter

        mock_moderation = MagicMock()
        mock_moderation.NONE = "none"
        mock_event_source = MagicMock()
        mock_event_source.CUSTOMER = "customer"
        mock_event_source.AI_AGENT = "ai_agent"
        mock_event_kind = MagicMock()
        mock_event_kind.MESSAGE = "message"

        with patch.dict(
            sys.modules,
            {
                "parlant.core.app_modules.sessions": MagicMock(
                    Moderation=mock_moderation
                ),
                "parlant.core.sessions": MagicMock(
                    EventSource=mock_event_source, EventKind=mock_event_kind
                ),
                "parlant.core.async_utils": MagicMock(Timeout=lambda x: x),
            },
        ):
            await adapter.on_message(
                msg=sample_message,
                tools=mock_tools,
                history=[],
                participants_msg=None,
                is_session_bootstrap=True,
                room_id="room-123",
            )

        adapter._app.sessions.create_customer_message.assert_awaited_once()
        call_kwargs = adapter._app.sessions.create_customer_message.call_args.kwargs
        assert call_kwargs["trigger_processing"] is True

    @pytest.mark.asyncio
    async def test_sets_session_tools_for_tool_execution(
        self, initialized_adapter, sample_message, mock_tools
    ):
        """set_session_tools called with tools, then cleared with None in finally."""
        adapter = initialized_adapter
        set_calls = []

        mock_moderation = MagicMock()
        mock_moderation.NONE = "none"
        mock_event_source = MagicMock()
        mock_event_source.CUSTOMER = "customer"
        mock_event_source.AI_AGENT = "ai_agent"
        mock_event_kind = MagicMock()
        mock_event_kind.MESSAGE = "message"

        with (
            patch.dict(
                sys.modules,
                {
                    "parlant.core.app_modules.sessions": MagicMock(
                        Moderation=mock_moderation
                    ),
                    "parlant.core.sessions": MagicMock(
                        EventSource=mock_event_source, EventKind=mock_event_kind
                    ),
                    "parlant.core.async_utils": MagicMock(Timeout=lambda x: x),
                },
            ),
            patch(
                "thenvoi.adapters.parlant.set_session_tools",
                side_effect=lambda sid, t: set_calls.append((sid, t is not None)),
            ),
        ):
            await adapter.on_message(
                msg=sample_message,
                tools=mock_tools,
                history=[],
                participants_msg=None,
                is_session_bootstrap=True,
                room_id="room-123",
            )

        # First call sets tools (True), last call clears (False)
        assert len(set_calls) >= 2
        assert set_calls[0][1] is True  # tools set
        assert set_calls[-1][1] is False  # tools cleared

    @pytest.mark.asyncio
    async def test_reuses_existing_session(
        self, initialized_adapter, sample_message, mock_tools
    ):
        """Pre-populated _room_sessions → no session create calls."""
        adapter = initialized_adapter
        adapter._room_sessions["room-123"] = "existing-session-id"
        adapter._room_customers["room-123"] = "existing-customer-id"

        mock_moderation = MagicMock()
        mock_moderation.NONE = "none"
        mock_event_source = MagicMock()
        mock_event_source.CUSTOMER = "customer"
        mock_event_source.AI_AGENT = "ai_agent"
        mock_event_kind = MagicMock()
        mock_event_kind.MESSAGE = "message"

        with patch.dict(
            sys.modules,
            {
                "parlant.core.app_modules.sessions": MagicMock(
                    Moderation=mock_moderation
                ),
                "parlant.core.sessions": MagicMock(
                    EventSource=mock_event_source, EventKind=mock_event_kind
                ),
                "parlant.core.async_utils": MagicMock(Timeout=lambda x: x),
            },
        ):
            await adapter.on_message(
                msg=sample_message,
                tools=mock_tools,
                history=[],
                participants_msg=None,
                is_session_bootstrap=False,
                room_id="room-123",
            )

        # Session create should NOT have been called
        adapter._app.sessions.create.assert_not_awaited()


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_handles_uninitialized_app(
        self, mock_parlant_server, mock_parlant_agent, sample_message, mock_tools
    ):
        """Should handle case when app is not initialized."""
        adapter = ParlantAdapter(
            server=mock_parlant_server,
            parlant_agent=mock_parlant_agent,
        )
        # Don't set _app

        # Should return early without error
        await adapter.on_message(
            msg=sample_message,
            tools=mock_tools,
            history=[],
            participants_msg=None,
            is_session_bootstrap=True,
            room_id="room-123",
        )

        # No calls should be made
        mock_tools.send_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_clears_tools_on_error(
        self, mock_parlant_server, mock_parlant_agent, sample_message, mock_tools
    ):
        """When create_customer_message raises, set_session_tools(id, None) still called."""
        adapter = ParlantAdapter(
            server=mock_parlant_server,
            parlant_agent=mock_parlant_agent,
        )
        adapter.agent_name = "TestBot"

        mock_app = MagicMock()
        mock_app.sessions = AsyncMock()
        mock_app.sessions.create = AsyncMock(return_value=MagicMock(id="session-123"))
        mock_app.sessions.create_customer_message = AsyncMock(
            side_effect=Exception("Parlant Error")
        )
        adapter._app = mock_app

        set_calls = []

        mock_moderation = MagicMock()
        mock_moderation.NONE = "none"
        mock_event_source = MagicMock()
        mock_event_source.CUSTOMER = "customer"

        with (
            patch.dict(
                sys.modules,
                {
                    "parlant.core.app_modules.sessions": MagicMock(
                        Moderation=mock_moderation
                    ),
                    "parlant.core.sessions": MagicMock(
                        EventSource=mock_event_source,
                    ),
                },
            ),
            patch(
                "thenvoi.adapters.parlant.set_session_tools",
                side_effect=lambda sid, t: set_calls.append((sid, t)),
            ),
            pytest.raises(Exception, match="Parlant Error"),
        ):
            await adapter.on_message(
                msg=sample_message,
                tools=mock_tools,
                history=[],
                participants_msg=None,
                is_session_bootstrap=True,
                room_id="room-123",
            )

        # Even after error, finally block should clear tools (last call has None)
        assert any(t is None for _, t in set_calls)

    @pytest.mark.asyncio
    async def test_reports_error_on_failure(
        self, mock_parlant_server, mock_parlant_agent, sample_message, mock_tools
    ):
        """When processing fails, tools.send_event called with error type, exception re-raised."""
        adapter = ParlantAdapter(
            server=mock_parlant_server,
            parlant_agent=mock_parlant_agent,
        )
        adapter.agent_name = "TestBot"

        mock_app = MagicMock()
        mock_app.sessions = AsyncMock()
        mock_app.sessions.create = AsyncMock(return_value=MagicMock(id="session-123"))
        mock_app.sessions.create_customer_message = AsyncMock(
            side_effect=Exception("Processing failed")
        )
        adapter._app = mock_app

        mock_moderation = MagicMock()
        mock_moderation.NONE = "none"
        mock_event_source = MagicMock()
        mock_event_source.CUSTOMER = "customer"

        with (
            patch.dict(
                sys.modules,
                {
                    "parlant.core.app_modules.sessions": MagicMock(
                        Moderation=mock_moderation
                    ),
                    "parlant.core.sessions": MagicMock(
                        EventSource=mock_event_source,
                    ),
                },
            ),
            patch("thenvoi.adapters.parlant.set_session_tools"),
            pytest.raises(Exception, match="Processing failed"),
        ):
            await adapter.on_message(
                msg=sample_message,
                tools=mock_tools,
                history=[],
                participants_msg=None,
                is_session_bootstrap=True,
                room_id="room-123",
            )

        # Error should have been reported via send_event
        mock_tools.send_event.assert_awaited()
        call_kwargs = mock_tools.send_event.call_args.kwargs
        assert call_kwargs["message_type"] == "error"
        assert "Processing failed" in call_kwargs["content"]
