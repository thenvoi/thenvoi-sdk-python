"""Conformance tests for all framework adapters.

These parameterized tests run across all adapter implementations to verify
they conform to expected behaviors. Framework-specific behaviors are handled
through configuration in tests/framework_configs/adapters.py.

Running:
    # All adapters
    uv run pytest tests/adapters/test_adapter_conformance.py -v

    # Specific adapter
    uv run pytest tests/adapters/test_adapter_conformance.py -k "anthropic" -v

Tests covered:
- Initialization (default values, custom params, history converter, model)
- on_started (agent_name, agent_description, system_prompt rendering)
- on_message (history initialization, loading, participants injection)
- on_cleanup (nonexistent room safety, room data cleanup, cleanup_all)
- Custom tools (accepts additional_tools, default empty)
- Error handling (reports errors, preserves state)
- System prompt override

Framework-specific tests are in separate files (e.g., test_crewai.py).
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from typing import Generator
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel, Field

from tests.framework_configs.adapters import (
    ADAPTER_CONFIGS,
    AdapterConfig,
    create_mock_tools,
    create_sample_message,
    setup_adapter_for_on_message,
)


# =============================================================================
# CrewAI Mocking Infrastructure
# =============================================================================
# CrewAI requires module-level mocking because it imports external dependencies
# (crewai, nest_asyncio) at module load time. We mock these before the adapter
# module is imported to avoid ImportError.
#
# The mocks are stored in a module-level dict (not on the config object) to
# avoid mutating the frozen AdapterConfig dataclass.
# =============================================================================

# Store CrewAI mocks separately from config to avoid mutating frozen dataclass
_crewai_test_mocks: dict[str, MagicMock] = {}


class MockBaseTool:
    """Mock CrewAI BaseTool for testing."""

    name: str = ""
    description: str = ""

    def __init__(self):
        pass


def _setup_crewai_mocks(monkeypatch) -> MagicMock:
    """Set up CrewAI module mocks.

    CrewAI adapter imports crewai and nest_asyncio at module level, so we must
    mock these before the adapter module is loaded. The monkeypatch fixture
    ensures mocks are automatically cleaned up after each test.

    Returns:
        The mock crewai module for assertions in tests.
    """
    mock_crewai_module = MagicMock()
    mock_crewai_tools_module = MagicMock()
    mock_nest_asyncio = MagicMock()

    mock_crewai_module.Agent = MagicMock()
    mock_crewai_module.LLM = MagicMock()
    mock_crewai_tools_module.BaseTool = MockBaseTool

    monkeypatch.setitem(sys.modules, "crewai", mock_crewai_module)
    monkeypatch.setitem(sys.modules, "crewai.tools", mock_crewai_tools_module)
    monkeypatch.setitem(sys.modules, "nest_asyncio", mock_nest_asyncio)

    return mock_crewai_module


@pytest.fixture(params=list(ADAPTER_CONFIGS.values()), ids=lambda c: c.name)
def adapter_config(
    request: pytest.FixtureRequest, monkeypatch
) -> Generator[AdapterConfig, None, None]:
    """Parameterized fixture that yields each adapter config.

    Only sets up CrewAI mocks when testing the CrewAI adapter. Mocks are stored
    in a module-level dict rather than on the config to avoid mutating it.
    """
    config = request.param

    # Only set up CrewAI mocks when testing CrewAI adapter
    if config.name == "crewai":
        _crewai_test_mocks["crewai"] = _setup_crewai_mocks(monkeypatch)

    try:
        yield config
    finally:
        # Clean up CrewAI module and mocks if it was loaded
        if config.name == "crewai":
            sys.modules.pop("thenvoi.adapters.crewai", None)
            _crewai_test_mocks.pop("crewai", None)


@pytest.fixture
def adapter(adapter_config: AdapterConfig):
    """Create an adapter instance from config."""
    return adapter_config.factory()


@pytest.fixture
def mock_tools():
    """Create mock AgentToolsProtocol."""
    return create_mock_tools()


@pytest.fixture
def sample_message():
    """Create a sample platform message."""
    return create_sample_message()


class TestInitialization:
    """Tests for adapter initialization across all frameworks."""

    def test_default_initialization_creates_adapter(
        self, adapter, adapter_config: AdapterConfig
    ):
        """Adapter should initialize without errors."""
        assert adapter is not None

    def test_has_history_converter_if_expected(
        self, adapter, adapter_config: AdapterConfig
    ):
        """Adapter should have history_converter if configured."""
        if adapter_config.has_history_converter:
            assert hasattr(adapter, "history_converter")
            assert adapter.history_converter is not None
        else:
            # Verify it either doesn't have it or it's None
            has_converter = hasattr(adapter, "history_converter")
            if has_converter:
                # Some adapters have it but don't use it
                pass  # OK either way

    def test_has_default_model_if_applicable(
        self, adapter, adapter_config: AdapterConfig
    ):
        """Adapter should have correct default model if applicable."""
        if adapter_config.default_model is not None:
            assert hasattr(adapter, "model")
            assert adapter.model == adapter_config.default_model
        else:
            # Adapters without default model may not have model attr or it's None
            if hasattr(adapter, "model"):
                # Model was provided during creation (e.g., pydantic_ai)
                pass

    def test_additional_init_checks(self, adapter, adapter_config: AdapterConfig):
        """Adapter should have correct additional init values."""
        for attr, expected_value in adapter_config.additional_init_checks.items():
            assert hasattr(adapter, attr), f"Missing attribute: {attr}"
            assert getattr(adapter, attr) == expected_value, (
                f"Unexpected value for {attr}: {getattr(adapter, attr)} != {expected_value}"
            )

    def test_accepts_custom_section_parameter(self, adapter_config: AdapterConfig):
        """Adapter should accept custom_section parameter."""
        adapter = adapter_config.factory(custom_section="Be helpful.")
        assert hasattr(adapter, "custom_section")
        assert adapter.custom_section == "Be helpful."

    def test_enable_execution_reporting_parameter(self, adapter_config: AdapterConfig):
        """Test enable_execution_reporting parameter handling."""
        if adapter_config.supports_enable_execution_reporting:
            # Adapter should accept and store the parameter
            adapter = adapter_config.factory(enable_execution_reporting=True)
            assert hasattr(adapter, "enable_execution_reporting")
            assert adapter.enable_execution_reporting is True
        else:
            # Adapter should either not have attribute or ignore parameter
            adapter = adapter_config.factory()
            # Verify it doesn't have enable_execution_reporting or it's False
            if hasattr(adapter, "enable_execution_reporting"):
                assert adapter.enable_execution_reporting is False


class TestOnStarted:
    """Tests for on_started() method across all frameworks."""

    @pytest.mark.asyncio
    async def test_sets_agent_name(self, adapter, adapter_config: AdapterConfig):
        """on_started should set agent_name."""
        await _call_on_started(adapter, adapter_config)
        assert adapter.agent_name == "TestBot"

    @pytest.mark.asyncio
    async def test_sets_agent_description(self, adapter, adapter_config: AdapterConfig):
        """on_started should set agent_description."""
        await _call_on_started(adapter, adapter_config)
        assert adapter.agent_description == "A test bot"

    @pytest.mark.asyncio
    async def test_system_prompt_or_alternative_contains_agent_info(
        self, adapter, adapter_config: AdapterConfig
    ):
        """on_started should set system prompt or alternative prompt with agent info."""
        await _call_on_started(adapter, adapter_config)

        # Check system_prompt_attr if defined
        if adapter_config.system_prompt_attr:
            assert hasattr(adapter, adapter_config.system_prompt_attr)
            prompt = getattr(adapter, adapter_config.system_prompt_attr)
            if adapter_config.system_prompt_contains_name:
                assert "TestBot" in prompt

        # Check alternative_prompt_attr if defined (e.g., backstory for crewai)
        elif adapter_config.alternative_prompt_attr:
            assert hasattr(adapter, adapter_config.alternative_prompt_attr)
            # Alternative prompts may not always contain the name directly

        # For adapters without either, verify they have some initialization
        else:
            # Claude SDK uses session manager, PydanticAI uses agent
            # Just verify on_started completed successfully (no exception)
            pass


class TestSystemPromptOverride:
    """Tests for system_prompt parameter override."""

    @pytest.mark.asyncio
    async def test_system_prompt_override_behavior(self, adapter_config: AdapterConfig):
        """Test system_prompt parameter behavior for each adapter."""
        if adapter_config.supports_system_prompt_override:
            # Adapter should use custom system_prompt
            adapter = adapter_config.factory(system_prompt="Custom prompt here.")
            await _call_on_started(adapter, adapter_config)
            assert adapter._system_prompt == "Custom prompt here."
        else:
            # Adapter uses alternative mechanism - verify it doesn't break
            adapter = adapter_config.factory()
            await _call_on_started(adapter, adapter_config)
            # Just verify no error - adapter uses different prompt mechanism


class TestOnMessage:
    """Tests for on_message() method across all frameworks."""

    @pytest.mark.asyncio
    async def test_initializes_history_on_bootstrap(
        self, adapter_config: AdapterConfig, sample_message, mock_tools
    ):
        """Adapter should initialize room history/session on first message."""
        adapter = adapter_config.factory()
        mocks = await setup_adapter_for_on_message(adapter, adapter_config, mock_tools)

        async with _mock_llm_call(adapter, adapter_config, mocks):
            await adapter.on_message(
                msg=sample_message,
                tools=mock_tools,
                history=[],
                participants_msg=None,
                is_session_bootstrap=True,
                room_id="room-123",
            )

        # Verify storage based on adapter type
        if adapter_config.history_storage_attr:
            storage = getattr(adapter, adapter_config.history_storage_attr, None)
            if storage is not None:
                assert "room-123" in storage
        else:
            # LangGraph uses checkpointer, verify no error occurred
            pass

    @pytest.mark.asyncio
    async def test_injects_participants_message(
        self, adapter_config: AdapterConfig, sample_message, mock_tools
    ):
        """Adapter should handle participants update when provided."""
        adapter = adapter_config.factory()
        mocks = await setup_adapter_for_on_message(adapter, adapter_config, mock_tools)

        captured_input = {}
        participants_msg = "Alice joined the room"

        async with _mock_llm_call(
            adapter, adapter_config, mocks, captured_input=captured_input
        ):
            await adapter.on_message(
                msg=sample_message,
                tools=mock_tools,
                history=[],
                participants_msg=participants_msg,
                is_session_bootstrap=True,
                room_id="room-123",
            )

        # Use verification callback if available
        if adapter_config.verify_participants_injection is not None:
            assert adapter_config.verify_participants_injection(
                adapter, captured_input, participants_msg, mocks
            )


class TestOnCleanup:
    """Tests for on_cleanup() method across all frameworks."""

    @pytest.mark.asyncio
    async def test_cleanup_nonexistent_room_is_safe(
        self, adapter, adapter_config: AdapterConfig
    ):
        """Cleanup of non-existent room should not raise."""
        # Should not raise any exception
        await adapter.on_cleanup("nonexistent-room-xyz")

    @pytest.mark.asyncio
    async def test_cleanup_clears_room_data(
        self, adapter, adapter_config: AdapterConfig
    ):
        """Cleanup should clear room-specific data."""
        # Pre-populate room data based on adapter's storage mechanism
        for storage_attr in adapter_config.cleanup_storage_attrs:
            if hasattr(adapter, storage_attr):
                storage = getattr(adapter, storage_attr)
                if isinstance(storage, dict):
                    storage["room-123"] = {"test": "data"}

        await adapter.on_cleanup("room-123")

        # Verify cleanup
        for storage_attr in adapter_config.cleanup_storage_attrs:
            if hasattr(adapter, storage_attr):
                storage = getattr(adapter, storage_attr)
                if isinstance(storage, dict):
                    assert "room-123" not in storage


class TestCleanupAll:
    """Tests for cleanup_all() method."""

    @pytest.mark.asyncio
    async def test_cleanup_all_behavior(
        self, adapter_config: AdapterConfig, mock_tools
    ):
        """Test cleanup_all behavior - either clears all or method doesn't exist."""
        adapter = adapter_config.factory()

        if adapter_config.supports_cleanup_all:
            # Set up data to clean
            if adapter_config.name == "claude_sdk":
                mock_session_manager = AsyncMock()
                adapter._session_manager = mock_session_manager
                adapter._room_tools["room-1"] = MagicMock()
                adapter._room_tools["room-2"] = MagicMock()

                await adapter.cleanup_all()

                mock_session_manager.stop.assert_awaited_once()
                assert len(adapter._room_tools) == 0

            elif adapter_config.name == "parlant":
                adapter._room_sessions["room-1"] = "session-1"
                adapter._room_sessions["room-2"] = "session-2"
                adapter._room_customers["room-1"] = "customer-1"
                adapter._room_customers["room-2"] = "customer-2"

                await adapter.cleanup_all()

                assert len(adapter._room_sessions) == 0
                assert len(adapter._room_customers) == 0
        else:
            # Adapter doesn't support cleanup_all - verify method doesn't exist
            # or calling it doesn't break anything
            if hasattr(adapter, "cleanup_all"):
                # If it exists but not "supported", it should be safe to call
                await adapter.cleanup_all()


class TestCustomTools:
    """Tests for custom tool support across all frameworks."""

    def test_additional_tools_parameter(self, adapter_config: AdapterConfig):
        """Test additional_tools parameter handling."""

        class EchoInput(BaseModel):
            """Echo the message."""

            message: str = Field(description="Message to echo")

        async def echo(args: EchoInput) -> str:
            return f"Echo: {args.message}"

        if adapter_config.has_custom_tools:
            # Use appropriate tool format based on adapter
            if adapter_config.custom_tool_format == "callable":
                adapter = adapter_config.factory(additional_tools=[echo])
            else:
                adapter = adapter_config.factory(additional_tools=[(EchoInput, echo)])

            # Verify custom tools were stored
            custom_tools_attr = adapter_config.custom_tools_attr
            assert hasattr(adapter, custom_tools_attr)
            custom_tools = getattr(adapter, custom_tools_attr)

            # LangGraph clears additional_tools after baking into factory
            if adapter_config.name == "langgraph":
                assert custom_tools == []
            else:
                assert len(custom_tools) >= 1
        else:
            # Adapter doesn't support custom tools (e.g., parlant uses SDK tools)
            # Verify it has its own tool mechanism
            adapter = adapter_config.factory()
            # Parlant uses Parlant SDK tools directly
            assert adapter is not None

    def test_defaults_to_empty_custom_tools(self, adapter_config: AdapterConfig):
        """Adapter should have empty custom tools by default."""
        if adapter_config.has_custom_tools:
            adapter = adapter_config.factory()
            custom_tools_attr = adapter_config.custom_tools_attr
            assert hasattr(adapter, custom_tools_attr)
            custom_tools = getattr(adapter, custom_tools_attr)
            assert custom_tools == [] or custom_tools is None or len(custom_tools) == 0
        else:
            # Adapter uses different tool mechanism
            adapter = adapter_config.factory()
            # Just verify initialization succeeded
            assert adapter is not None

    def test_multiple_custom_tools(self, adapter_config: AdapterConfig):
        """Adapter should accept multiple custom tools if supported."""

        class EchoInput(BaseModel):
            """Echo the message."""

            message: str = Field(description="Message to echo")

        class CalculatorInput(BaseModel):
            """Perform math calculations."""

            operation: str = Field(description="add, subtract, multiply, divide")
            left: float = Field(description="Left operand")
            right: float = Field(description="Right operand")

        async def echo(args: EchoInput) -> str:
            return f"Echo: {args.message}"

        def calculate(args: CalculatorInput) -> str:
            ops = {
                "add": lambda a, b: a + b,
                "subtract": lambda a, b: a - b,
                "multiply": lambda a, b: a * b,
                "divide": lambda a, b: a / b,
            }
            return str(ops[args.operation](args.left, args.right))

        if adapter_config.has_custom_tools:
            # Use appropriate tool format based on adapter
            if adapter_config.custom_tool_format == "callable":
                adapter = adapter_config.factory(additional_tools=[echo, calculate])
            else:
                adapter = adapter_config.factory(
                    additional_tools=[(EchoInput, echo), (CalculatorInput, calculate)]
                )

            custom_tools_attr = adapter_config.custom_tools_attr
            custom_tools = getattr(adapter, custom_tools_attr)

            # LangGraph clears additional_tools after baking
            if adapter_config.name == "langgraph":
                assert custom_tools == []
            else:
                assert len(custom_tools) >= 2
        else:
            # Adapter doesn't support custom tools
            adapter = adapter_config.factory()
            assert adapter is not None


class TestErrorHandling:
    """Tests for error handling across all frameworks."""

    @pytest.mark.asyncio
    async def test_reports_error_on_failure(
        self, adapter_config: AdapterConfig, sample_message, mock_tools
    ):
        """Adapter should raise/report error when LLM call fails."""
        if not adapter_config.error_setup_callback:
            # Adapter doesn't support error testing - pass without assertion
            # (All current adapters have error_setup_callback configured)
            return

        adapter = adapter_config.factory()
        mocks = await setup_adapter_for_on_message(adapter, adapter_config, mock_tools)

        async with adapter_config.error_setup_callback(adapter, mocks) as error_msg:
            with pytest.raises(Exception, match=error_msg):
                await adapter.on_message(
                    msg=sample_message,
                    tools=mock_tools,
                    history=[],
                    participants_msg=None,
                    is_session_bootstrap=True,
                    room_id="room-123",
                )


# Helper functions


async def _call_on_started(adapter, config: AdapterConfig) -> None:
    """Call on_started with appropriate mocking for each adapter."""
    if config.on_started_callback:
        await config.on_started_callback(adapter, config)
    else:
        await adapter.on_started(agent_name="TestBot", agent_description="A test bot")


@asynccontextmanager
async def _mock_llm_call(
    adapter,
    config: AdapterConfig,
    mocks: dict | None,
    captured_input: dict | None = None,
):
    """Context manager to mock LLM calls for on_message tests."""
    if config.mock_llm_callback:
        async with config.mock_llm_callback(adapter, mocks, captured_input):
            yield
    else:
        yield
