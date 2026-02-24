"""Tests for ClaudeSDKAdapter.

Tests for shared adapter behavior (initialization defaults, custom kwargs,
history_converter, on_message callable, cleanup safety) live in
tests/framework_conformance/test_adapter_conformance.py.
This file contains ClaudeSDK-specific behavior: MCP server/session manager
creation, room tools storage, SDK query invocation, custom tools, and
session persistence.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from thenvoi.adapters.claude_sdk import (
    ClaudeSDKAdapter,
    THENVOI_TOOLS,
    THENVOI_BASE_TOOLS,
    THENVOI_MEMORY_TOOLS,
)
from thenvoi.converters.claude_sdk import ClaudeSDKSessionState
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
    """Create mock AgentToolsProtocol (MagicMock base, AsyncMock methods)."""
    tools = MagicMock()
    tools.send_message = AsyncMock(return_value={"status": "sent"})
    tools.send_event = AsyncMock(return_value={"status": "sent"})
    tools.add_participant = AsyncMock(return_value={"id": "user-1"})
    tools.remove_participant = AsyncMock(return_value={"status": "removed"})
    tools.lookup_peers = AsyncMock(return_value={"peers": []})
    tools.get_participants = AsyncMock(return_value=[])
    return tools


class TestInitialization:
    """Tests for adapter initialization (memory tools specific)."""

    def test_default_initialization(self):
        """Should initialize with enable_memory_tools=False by default."""
        adapter = ClaudeSDKAdapter()

        assert adapter.enable_memory_tools is False

    def test_enable_memory_tools(self):
        """Should accept enable_memory_tools parameter."""
        adapter = ClaudeSDKAdapter(enable_memory_tools=True)

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
            assert adapter.agent_description == "A test bot"
            assert adapter._session_manager is not None
            assert adapter._mcp_server is not None


class TestOnMessage:
    """Tests for on_message() method (bootstrap, history, invoke and response)."""

    @pytest.mark.asyncio
    async def test_initializes_history_on_bootstrap(self, sample_message, mock_tools):
        """First message in a room initializes session context and triggers invoke."""
        adapter = ClaudeSDKAdapter()
        mock_client = MagicMock()
        mock_client.query = AsyncMock()
        mock_manager = AsyncMock()
        mock_manager.get_or_create_session = AsyncMock(return_value=mock_client)

        with (
            patch(
                "thenvoi.adapters.claude_sdk.ClaudeSessionManager",
                return_value=mock_manager,
            ),
            patch.object(
                adapter, "_process_response", new_callable=AsyncMock
            ) as mock_process,
        ):
            await adapter.on_started(
                agent_name="TestBot", agent_description="A test bot"
            )
            await adapter.on_message(
                msg=sample_message,
                tools=mock_tools,
                history=ClaudeSDKSessionState(text=""),
                participants_msg=None,
                contacts_msg=None,
                is_session_bootstrap=True,
                room_id="room-123",
            )

            assert adapter._room_tools["room-123"] is mock_tools
            assert adapter._session_context["room-123"] == ""
            mock_manager.get_or_create_session.assert_awaited_once_with(
                "room-123", resume_session_id=None
            )
            mock_client.query.assert_awaited_once()
            mock_process.assert_awaited_once_with(mock_client, "room-123", mock_tools)

    @pytest.mark.asyncio
    async def test_loads_existing_history_on_bootstrap(
        self, sample_message, mock_tools
    ):
        """When history is provided on bootstrap, it is loaded and used for the next invoke."""
        adapter = ClaudeSDKAdapter()
        mock_client = MagicMock()
        mock_client.query = AsyncMock()
        mock_manager = AsyncMock()
        mock_manager.get_or_create_session = AsyncMock(return_value=mock_client)
        prior_context = "[Alice]: Hello\n[Bot]: Hi there."

        with (
            patch(
                "thenvoi.adapters.claude_sdk.ClaudeSessionManager",
                return_value=mock_manager,
            ),
            patch.object(adapter, "_process_response", new_callable=AsyncMock),
        ):
            await adapter.on_started(
                agent_name="TestBot", agent_description="A test bot"
            )
            await adapter.on_message(
                msg=sample_message,
                tools=mock_tools,
                history=ClaudeSDKSessionState(text=prior_context),
                participants_msg=None,
                contacts_msg=None,
                is_session_bootstrap=True,
                room_id="room-123",
            )

            assert adapter._session_context["room-123"] == prior_context
            call_args = mock_client.query.call_args[0][0]
            assert "[Previous conversation context:]" in call_args
            assert prior_context in call_args

    @pytest.mark.asyncio
    async def test_invoke_and_response(self, sample_message, mock_tools):
        """Adapter invokes the SDK client and processes response."""
        adapter = ClaudeSDKAdapter()
        mock_client = MagicMock()
        mock_client.query = AsyncMock()
        mock_manager = AsyncMock()
        mock_manager.get_or_create_session = AsyncMock(return_value=mock_client)

        with (
            patch(
                "thenvoi.adapters.claude_sdk.ClaudeSessionManager",
                return_value=mock_manager,
            ),
            patch.object(
                adapter, "_process_response", new_callable=AsyncMock
            ) as mock_process,
        ):
            await adapter.on_started(
                agent_name="TestBot", agent_description="A test bot"
            )
            await adapter.on_message(
                msg=sample_message,
                tools=mock_tools,
                history=ClaudeSDKSessionState(text=""),
                participants_msg=None,
                contacts_msg=None,
                is_session_bootstrap=True,
                room_id="room-123",
            )

            mock_client.query.assert_awaited_once()
            full_message = mock_client.query.call_args[0][0]
            assert "room-123" in full_message
            assert "Hello, agent!" in full_message
            mock_process.assert_awaited_once()


class TestErrorHandling:
    """Tests for error handling when SDK or tools raise."""

    @pytest.mark.asyncio
    async def test_reports_error_on_query_failure(self, sample_message, mock_tools):
        """When client.query raises, adapter reports error via send_event and re-raises."""
        adapter = ClaudeSDKAdapter()
        mock_client = MagicMock()
        mock_client.query = AsyncMock(side_effect=Exception("API Error"))
        mock_manager = AsyncMock()
        mock_manager.get_or_create_session = AsyncMock(return_value=mock_client)

        with patch(
            "thenvoi.adapters.claude_sdk.ClaudeSessionManager",
            return_value=mock_manager,
        ):
            await adapter.on_started(
                agent_name="TestBot", agent_description="A test bot"
            )

            with pytest.raises(Exception, match="API Error"):
                await adapter.on_message(
                    msg=sample_message,
                    tools=mock_tools,
                    history=ClaudeSDKSessionState(text=""),
                    participants_msg=None,
                    contacts_msg=None,
                    is_session_bootstrap=True,
                    room_id="room-123",
                )

            mock_tools.send_event.assert_called()
            call_kwargs = mock_tools.send_event.call_args[1]
            assert call_kwargs.get("message_type") == "error"
            assert "API Error" in call_kwargs.get("content", "")


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
            "mcp__thenvoi__thenvoi_send_message",
            "mcp__thenvoi__thenvoi_send_event",
            "mcp__thenvoi__thenvoi_add_participant",
            "mcp__thenvoi__thenvoi_remove_participant",
            "mcp__thenvoi__thenvoi_get_participants",
            "mcp__thenvoi__thenvoi_lookup_peers",
            "mcp__thenvoi__thenvoi_create_chatroom",
            # Contact management tools
            "mcp__thenvoi__thenvoi_list_contacts",
            "mcp__thenvoi__thenvoi_add_contact",
            "mcp__thenvoi__thenvoi_remove_contact",
            "mcp__thenvoi__thenvoi_list_contact_requests",
            "mcp__thenvoi__thenvoi_respond_contact_request",
        ]

        assert THENVOI_BASE_TOOLS == expected

    def test_thenvoi_memory_tools_list(self):
        """Should define memory tools (enterprise only - opt-in)."""
        expected = [
            "mcp__thenvoi__thenvoi_list_memories",
            "mcp__thenvoi__thenvoi_store_memory",
            "mcp__thenvoi__thenvoi_get_memory",
            "mcp__thenvoi__thenvoi_supersede_memory",
            "mcp__thenvoi__thenvoi_archive_memory",
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
            assert "mcp__thenvoi__thenvoi_send_message" in sdk_options.allowed_tools

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


class TestSessionPersistence:
    """Tests for session persistence via task events."""

    @pytest.mark.asyncio
    async def test_emits_task_event_after_session_id_capture(self, mock_tools):
        """Should emit task event with session_id after ResultMessage."""
        adapter = ClaudeSDKAdapter()

        # Create a mock ResultMessage with session_id
        mock_result_msg = MagicMock()
        mock_result_msg.session_id = "sess-xyz-789"
        mock_result_msg.duration_ms = 1500
        mock_result_msg.total_cost_usd = 0.01

        # Create mock client that yields the ResultMessage
        mock_client = MagicMock()

        async def mock_receive():
            yield mock_result_msg

        mock_client.receive_response = mock_receive

        # Patch isinstance checks for ResultMessage
        with patch("thenvoi.adapters.claude_sdk.ResultMessage", type(mock_result_msg)):
            await adapter._process_response(mock_client, "room-123", mock_tools)

        # Verify task event was emitted
        mock_tools.send_event.assert_called_once_with(
            content="Claude SDK session",
            message_type="task",
            metadata={"claude_sdk_session_id": "sess-xyz-789"},
        )
        # Verify in-memory cache was updated
        assert adapter._session_ids["room-123"] == "sess-xyz-789"

    @pytest.mark.asyncio
    async def test_uses_history_session_id_for_resume(self, sample_message, mock_tools):
        """Should use history.session_id for resume on bootstrap."""
        adapter = ClaudeSDKAdapter()
        mock_client = MagicMock()
        mock_client.query = AsyncMock()
        mock_manager = AsyncMock()
        mock_manager.get_or_create_session = AsyncMock(return_value=mock_client)

        with (
            patch(
                "thenvoi.adapters.claude_sdk.ClaudeSessionManager",
                return_value=mock_manager,
            ),
            patch.object(adapter, "_process_response", new_callable=AsyncMock),
        ):
            await adapter.on_started(
                agent_name="TestBot", agent_description="A test bot"
            )
            await adapter.on_message(
                msg=sample_message,
                tools=mock_tools,
                history=ClaudeSDKSessionState(
                    text="[Alice]: Hello", session_id="sess-from-history"
                ),
                participants_msg=None,
                contacts_msg=None,
                is_session_bootstrap=True,
                room_id="room-123",
            )

            mock_manager.get_or_create_session.assert_awaited_once_with(
                "room-123", resume_session_id="sess-from-history"
            )

    @pytest.mark.asyncio
    async def test_no_resume_on_non_bootstrap(self, sample_message, mock_tools):
        """Should not attempt resume on non-bootstrap messages."""
        adapter = ClaudeSDKAdapter()
        mock_client = MagicMock()
        mock_client.query = AsyncMock()
        mock_manager = AsyncMock()
        mock_manager.get_or_create_session = AsyncMock(return_value=mock_client)

        with (
            patch(
                "thenvoi.adapters.claude_sdk.ClaudeSessionManager",
                return_value=mock_manager,
            ),
            patch.object(adapter, "_process_response", new_callable=AsyncMock),
        ):
            await adapter.on_started(
                agent_name="TestBot", agent_description="A test bot"
            )
            await adapter.on_message(
                msg=sample_message,
                tools=mock_tools,
                history=ClaudeSDKSessionState(
                    text="", session_id="sess-should-not-use"
                ),
                participants_msg=None,
                contacts_msg=None,
                is_session_bootstrap=False,
                room_id="room-123",
            )

            mock_manager.get_or_create_session.assert_awaited_once_with(
                "room-123", resume_session_id=None
            )

    @pytest.mark.asyncio
    async def test_falls_back_to_new_session_on_resume_failure(
        self, sample_message, mock_tools
    ):
        """Should create new session if resume fails."""
        adapter = ClaudeSDKAdapter()
        mock_client = MagicMock()
        mock_client.query = AsyncMock()
        mock_manager = AsyncMock()
        # First call (with resume) fails, second call (without) succeeds
        mock_manager.get_or_create_session = AsyncMock(
            side_effect=[Exception("Resume failed"), mock_client]
        )

        with (
            patch(
                "thenvoi.adapters.claude_sdk.ClaudeSessionManager",
                return_value=mock_manager,
            ),
            patch.object(adapter, "_process_response", new_callable=AsyncMock),
        ):
            await adapter.on_started(
                agent_name="TestBot", agent_description="A test bot"
            )
            # Should not raise — falls back to new session
            await adapter.on_message(
                msg=sample_message,
                tools=mock_tools,
                history=ClaudeSDKSessionState(text="", session_id="sess-broken"),
                participants_msg=None,
                contacts_msg=None,
                is_session_bootstrap=True,
                room_id="room-123",
            )

            assert mock_manager.get_or_create_session.await_count == 2
            # Second call should be without resume
            second_call = mock_manager.get_or_create_session.call_args_list[1]
            assert second_call == (("room-123",), {"resume_session_id": None})

    @pytest.mark.asyncio
    async def test_task_event_failure_does_not_break_flow(self, mock_tools):
        """Task event emission failure should not break the message flow."""
        adapter = ClaudeSDKAdapter()
        mock_tools.send_event = AsyncMock(side_effect=Exception("Network error"))

        mock_result_msg = MagicMock()
        mock_result_msg.session_id = "sess-xyz"
        mock_result_msg.duration_ms = 100
        mock_result_msg.total_cost_usd = 0.001

        mock_client = MagicMock()

        async def mock_receive():
            yield mock_result_msg

        mock_client.receive_response = mock_receive

        with patch("thenvoi.adapters.claude_sdk.ResultMessage", type(mock_result_msg)):
            # Should not raise despite send_event failure
            await adapter._process_response(mock_client, "room-123", mock_tools)

        # Session ID should still be captured in-memory
        assert adapter._session_ids["room-123"] == "sess-xyz"
