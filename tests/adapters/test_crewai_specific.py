"""CrewAI-specific tests that don't apply to other adapters.

These tests cover CrewAI-specific functionality like:
- Deprecation warnings for system_prompt parameter
- CrewAI-specific parameters (verbose, max_rpm, allow_delegation)
- Context variable-based tool execution
- nest_asyncio handling
- Mentions validation
- Platform instructions
"""

from __future__ import annotations

import contextlib
import json
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel, Field


class MockBaseTool:
    """Mock CrewAI BaseTool for testing."""

    name: str = ""
    description: str = ""


@pytest.fixture
def crewai_mocks(monkeypatch):
    """Set up CrewAI module mocks."""
    mock_crewai_module = MagicMock()
    mock_crewai_tools_module = MagicMock()
    mock_nest_asyncio = MagicMock()

    mock_crewai_module.Agent = MagicMock()
    mock_crewai_module.LLM = MagicMock()
    mock_crewai_tools_module.BaseTool = MockBaseTool

    monkeypatch.setitem(sys.modules, "crewai", mock_crewai_module)
    monkeypatch.setitem(sys.modules, "crewai.tools", mock_crewai_tools_module)
    monkeypatch.setitem(sys.modules, "nest_asyncio", mock_nest_asyncio)

    yield mock_crewai_module

    sys.modules.pop("thenvoi.adapters.crewai", None)


@pytest.fixture
def adapter(crewai_mocks):
    """Create a CrewAI adapter instance."""
    import importlib

    module = importlib.import_module("thenvoi.adapters.crewai")
    return module.CrewAIAdapter()


@pytest.fixture
def mock_tools():
    """Create mock AgentToolsProtocol."""
    tools = AsyncMock()
    tools.get_tool_schemas = MagicMock(return_value=[])
    tools.send_message = AsyncMock(return_value={"status": "sent"})
    tools.send_event = AsyncMock(return_value={"status": "sent"})
    tools.get_participants = AsyncMock(
        return_value=[{"id": "123", "name": "Alice", "type": "User"}]
    )
    return tools


@pytest.fixture
def room_context(mock_tools):
    """Context manager for setting up room context."""
    import importlib

    module = importlib.import_module("thenvoi.adapters.crewai")

    @contextlib.contextmanager
    def _room_context(room_id: str = "room-123"):
        module._current_room_context.set((room_id, mock_tools))
        try:
            yield
        finally:
            module._current_room_context.set(None)

    return _room_context


class TestDeprecation:
    """Tests for deprecated parameters."""

    def test_system_prompt_deprecation_warning(self, crewai_mocks):
        """system_prompt parameter should emit DeprecationWarning."""
        import importlib
        import warnings

        module = importlib.import_module("thenvoi.adapters.crewai")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            adapter = module.CrewAIAdapter(system_prompt="Old style prompt")

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "system_prompt" in str(w[0].message)
            assert adapter.backstory == "Old style prompt"

    def test_system_prompt_does_not_override_backstory(self, crewai_mocks):
        """If both system_prompt and backstory are provided, backstory wins."""
        import importlib
        import warnings

        module = importlib.import_module("thenvoi.adapters.crewai")

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            adapter = module.CrewAIAdapter(
                system_prompt="Old style", backstory="New style"
            )
            assert adapter.backstory == "New style"


class TestParameters:
    """Tests for CrewAI-specific parameters."""

    @pytest.mark.asyncio
    async def test_verbose_mode_passed_to_agent(self, crewai_mocks):
        """verbose parameter should be passed to CrewAI Agent."""
        import importlib

        module = importlib.import_module("thenvoi.adapters.crewai")
        crewai_mocks.Agent.reset_mock()

        adapter = module.CrewAIAdapter(verbose=True)
        await adapter.on_started("TestBot", "Test bot")

        call_kwargs = crewai_mocks.Agent.call_args[1]
        assert call_kwargs["verbose"] is True

    @pytest.mark.asyncio
    async def test_max_rpm_passed_to_agent(self, crewai_mocks):
        """max_rpm parameter should be passed to CrewAI Agent."""
        import importlib

        module = importlib.import_module("thenvoi.adapters.crewai")
        crewai_mocks.Agent.reset_mock()

        adapter = module.CrewAIAdapter(max_rpm=10)
        await adapter.on_started("TestBot", "Test bot")

        call_kwargs = crewai_mocks.Agent.call_args[1]
        assert call_kwargs["max_rpm"] == 10

    @pytest.mark.asyncio
    async def test_allow_delegation_passed_to_agent(self, crewai_mocks):
        """allow_delegation parameter should be passed to CrewAI Agent."""
        import importlib

        module = importlib.import_module("thenvoi.adapters.crewai")
        crewai_mocks.Agent.reset_mock()

        adapter = module.CrewAIAdapter(allow_delegation=True)
        await adapter.on_started("TestBot", "Test bot")

        call_kwargs = crewai_mocks.Agent.call_args[1]
        assert call_kwargs["allow_delegation"] is True

    def test_max_rpm_defaults_to_none(self, adapter):
        """max_rpm should default to None."""
        assert adapter.max_rpm is None

    def test_allow_delegation_defaults_to_false(self, adapter):
        """allow_delegation should default to False."""
        assert adapter.allow_delegation is False


class TestToolExecutionContext:
    """Tests for context variable-based tool execution."""

    def test_tool_returns_error_without_room_context(self, crewai_mocks):
        """Tools return error when called outside message handling."""
        import asyncio
        import importlib

        module = importlib.import_module("thenvoi.adapters.crewai")
        crewai_mocks.Agent.reset_mock()

        adapter = module.CrewAIAdapter()
        asyncio.run(adapter.on_started("TestBot", "Test bot"))

        call_kwargs = crewai_mocks.Agent.call_args[1]
        tools = call_kwargs["tools"]
        send_message_tool = next(t for t in tools if t.name == "thenvoi_send_message")

        result = send_message_tool._run(content="Hello!", mentions="[]")
        result_data = json.loads(result)

        assert result_data["status"] == "error"
        assert "No room context available" in result_data["message"]

    def test_successful_tool_execution_with_room_context(
        self, crewai_mocks, room_context, mock_tools
    ):
        """Tools work when context variable is set."""
        import asyncio
        import importlib

        module = importlib.import_module("thenvoi.adapters.crewai")
        crewai_mocks.Agent.reset_mock()

        adapter = module.CrewAIAdapter()
        asyncio.run(adapter.on_started("TestBot", "Test bot"))

        call_kwargs = crewai_mocks.Agent.call_args[1]
        tools = call_kwargs["tools"]
        get_participants_tool = next(
            t for t in tools if t.name == "thenvoi_get_participants"
        )

        with room_context("room-123"):
            result = get_participants_tool._run()

        result_data = json.loads(result)
        assert result_data["status"] == "success"
        assert result_data["count"] == 1


class TestNestAsyncio:
    """Tests for nest_asyncio handling."""

    def test_nest_asyncio_not_applied_on_import(self, crewai_mocks):
        """nest_asyncio should not be applied on module import."""
        import importlib

        sys.modules.pop("thenvoi.adapters.crewai", None)
        nest_mock = sys.modules["nest_asyncio"]
        nest_mock.reset_mock()

        importlib.import_module("thenvoi.adapters.crewai")

        nest_mock.apply.assert_not_called()

    def test_ensure_nest_asyncio_applies_once(self, crewai_mocks):
        """_ensure_nest_asyncio should only apply the patch once."""
        import importlib

        module = importlib.import_module("thenvoi.adapters.crewai")
        module._nest_asyncio_applied = False

        nest_mock = sys.modules["nest_asyncio"]
        nest_mock.reset_mock()

        module._ensure_nest_asyncio()
        module._ensure_nest_asyncio()

        assert nest_mock.apply.call_count == 1

    def test_nest_asyncio_lock_exists(self, crewai_mocks):
        """Module should have a threading lock for thread-safe application."""
        import importlib
        import threading

        module = importlib.import_module("thenvoi.adapters.crewai")

        assert hasattr(module, "_nest_asyncio_lock")
        assert isinstance(module._nest_asyncio_lock, type(threading.Lock()))


class TestMentionsValidator:
    """Tests for mentions validation."""

    @pytest.mark.asyncio
    async def test_mentions_list_converted_to_json(self, crewai_mocks):
        """List mentions should be converted to JSON string."""
        import importlib

        module = importlib.import_module("thenvoi.adapters.crewai")
        crewai_mocks.Agent.reset_mock()

        adapter = module.CrewAIAdapter()
        await adapter.on_started("TestBot", "Test bot")

        call_kwargs = crewai_mocks.Agent.call_args[1]
        tools = call_kwargs["tools"]
        send_message_tool = next(t for t in tools if t.name == "thenvoi_send_message")

        input_model = send_message_tool.args_schema
        instance = input_model(content="Hello!", mentions=["Alice", "Bob"])

        assert instance.mentions == '["Alice", "Bob"]'

    @pytest.mark.asyncio
    async def test_mentions_none_converted_to_empty_array(self, crewai_mocks):
        """None mentions should be normalized to empty JSON array string."""
        import importlib

        module = importlib.import_module("thenvoi.adapters.crewai")
        crewai_mocks.Agent.reset_mock()

        adapter = module.CrewAIAdapter()
        await adapter.on_started("TestBot", "Test bot")

        call_kwargs = crewai_mocks.Agent.call_args[1]
        tools = call_kwargs["tools"]
        send_message_tool = next(t for t in tools if t.name == "thenvoi_send_message")

        input_model = send_message_tool.args_schema
        instance = input_model(content="Hello!", mentions=None)

        assert instance.mentions == "[]"


class TestPlatformInstructions:
    """Tests for platform instructions constant."""

    def test_platform_instructions_is_constant(self, crewai_mocks):
        """PLATFORM_INSTRUCTIONS should be a non-empty string constant."""
        import importlib

        module = importlib.import_module("thenvoi.adapters.crewai")

        assert hasattr(module, "PLATFORM_INSTRUCTIONS")
        assert isinstance(module.PLATFORM_INSTRUCTIONS, str)
        assert len(module.PLATFORM_INSTRUCTIONS) > 100

    def test_platform_instructions_contains_key_info(self, crewai_mocks):
        """PLATFORM_INSTRUCTIONS should contain key platform information."""
        import importlib

        module = importlib.import_module("thenvoi.adapters.crewai")
        instructions = module.PLATFORM_INSTRUCTIONS

        assert "Environment" in instructions
        assert "thenvoi_send_message" in instructions


class TestToolSchemas:
    """Tests for tool schemas."""

    @pytest.mark.asyncio
    async def test_tools_do_not_require_room_id(self, crewai_mocks):
        """Tools should not require room_id - context is managed via context variable."""
        import importlib

        module = importlib.import_module("thenvoi.adapters.crewai")
        crewai_mocks.Agent.reset_mock()

        adapter = module.CrewAIAdapter()
        await adapter.on_started("TestBot", "Test bot")

        call_kwargs = crewai_mocks.Agent.call_args[1]
        tools = call_kwargs["tools"]

        send_message = next(t for t in tools if t.name == "thenvoi_send_message")
        schema_fields = send_message.args_schema.model_fields
        assert "room_id" not in schema_fields
        assert "content" in schema_fields


class TestCustomToolExecution:
    """Tests for custom tool execution."""

    def test_custom_tool_execution_async(self, crewai_mocks, room_context, mock_tools):
        """Async custom tool should execute correctly."""
        import asyncio
        import importlib

        module = importlib.import_module("thenvoi.adapters.crewai")

        class EchoInput(BaseModel):
            """Echo back the provided message."""

            message: str = Field(description="Message to echo")

        async def echo_message(args: EchoInput) -> str:
            return f"Echo: {args.message}"

        crewai_mocks.Agent.reset_mock()

        adapter = module.CrewAIAdapter(additional_tools=[(EchoInput, echo_message)])
        asyncio.run(adapter.on_started("TestBot", "Test bot"))

        call_kwargs = crewai_mocks.Agent.call_args[1]
        tools = call_kwargs["tools"]
        echo_tool = next(t for t in tools if t.name == "echo")

        with room_context("room-123"):
            result = echo_tool._run(message="Hello world")

        result_data = json.loads(result)
        assert result_data["status"] == "success"
        assert "Echo: Hello world" in result_data["result"]

    def test_custom_tool_without_room_context(self, crewai_mocks):
        """Custom tool should return error when called without room context."""
        import asyncio
        import importlib

        module = importlib.import_module("thenvoi.adapters.crewai")

        class EchoInput(BaseModel):
            """Echo back the provided message."""

            message: str = Field(description="Message to echo")

        async def echo_message(args: EchoInput) -> str:
            return f"Echo: {args.message}"

        crewai_mocks.Agent.reset_mock()

        adapter = module.CrewAIAdapter(additional_tools=[(EchoInput, echo_message)])
        asyncio.run(adapter.on_started("TestBot", "Test bot"))

        call_kwargs = crewai_mocks.Agent.call_args[1]
        tools = call_kwargs["tools"]
        echo_tool = next(t for t in tools if t.name == "echo")

        result = echo_tool._run(message="Hello!")

        result_data = json.loads(result)
        assert result_data["status"] == "error"
        assert "No room context available" in result_data["message"]
