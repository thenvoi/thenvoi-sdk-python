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
- CrewAI-specific: deprecation, context variables, nest_asyncio, mentions, etc.
"""

from __future__ import annotations

import contextlib
import json
import sys
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from tests.framework_configs.adapters import (
    ADAPTER_CONFIGS,
    AdapterConfig,
    create_mock_tools,
    create_sample_message,
    setup_adapter_for_on_message,
)


class MockBaseTool:
    """Mock CrewAI BaseTool for testing."""

    name: str = ""
    description: str = ""

    def __init__(self):
        pass


def _setup_crewai_mocks(monkeypatch) -> MagicMock:
    """Set up CrewAI module mocks. Returns the mock crewai module."""
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
def adapter_config(request: pytest.FixtureRequest, monkeypatch) -> AdapterConfig:
    """Parameterized fixture that yields each adapter config.

    Only sets up CrewAI mocks when testing the CrewAI adapter.
    """
    config = request.param

    # Only set up CrewAI mocks when testing CrewAI adapter
    if config.name == "crewai":
        crewai_mocks = _setup_crewai_mocks(monkeypatch)
        config._crewai_mocks = crewai_mocks

    try:
        yield config
    finally:
        # Clean up CrewAI module if it was loaded
        if config.name == "crewai":
            sys.modules.pop("thenvoi.adapters.crewai", None)


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
                adapter, captured_input, participants_msg
            )
        elif adapter_config.name == "pydantic_ai":
            # PydanticAI needs mocks passed - special case
            if mocks and "agent" in mocks:
                call_args = mocks["agent"].run_stream_events.call_args
                if call_args:
                    call_kwargs = call_args.kwargs
                    message_history = call_kwargs.get("message_history", [])
                    if message_history:
                        found = any(
                            "[System]: Alice joined"
                            in str(getattr(m.parts[0], "content", ""))
                            for m in message_history
                            if hasattr(m, "parts")
                            and m.parts
                            and hasattr(m.parts[0], "content")
                        )
                        assert found


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
        from pydantic import BaseModel, Field

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
        from pydantic import BaseModel, Field

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
        adapter = adapter_config.factory()
        mocks = await setup_adapter_for_on_message(adapter, adapter_config, mock_tools)

        # Set up failure based on adapter type
        if adapter_config.name == "anthropic":
            with patch.object(
                adapter, "_call_anthropic", side_effect=Exception("API Error")
            ):
                with pytest.raises(Exception, match="API Error"):
                    await adapter.on_message(
                        msg=sample_message,
                        tools=mock_tools,
                        history=[],
                        participants_msg=None,
                        is_session_bootstrap=True,
                        room_id="room-123",
                    )

        elif adapter_config.name == "langgraph":

            async def failing_stream(*args, **kwargs):
                raise Exception("Graph error!")
                yield  # Make it async generator

            mock_graph = MagicMock()
            mock_graph.astream_events = failing_stream
            adapter.graph_factory = MagicMock(return_value=mock_graph)

            with patch(
                "thenvoi.integrations.langgraph.langchain_tools.agent_tools_to_langchain"
            ) as mock_convert:
                mock_convert.return_value = []

                with pytest.raises(Exception, match="Graph error!"):
                    await adapter.on_message(
                        msg=sample_message,
                        tools=mock_tools,
                        history=[],
                        participants_msg=None,
                        is_session_bootstrap=True,
                        room_id="room-123",
                    )

        elif adapter_config.name == "pydantic_ai":

            async def failing_stream():
                raise Exception("PydanticAI Error")
                yield  # Never reached

            if mocks and "agent" in mocks:
                mocks["agent"].run_stream_events = MagicMock(
                    return_value=failing_stream()
                )

            with pytest.raises(Exception, match="PydanticAI Error"):
                await adapter.on_message(
                    msg=sample_message,
                    tools=mock_tools,
                    history=[],
                    participants_msg=None,
                    is_session_bootstrap=True,
                    room_id="room-123",
                )

        elif adapter_config.name == "crewai":
            # Mock CrewAI agent to raise error
            if adapter._crewai_agent is not None:
                adapter._crewai_agent.kickoff_async = AsyncMock(
                    side_effect=Exception("CrewAI Error")
                )

            with pytest.raises(Exception, match="CrewAI Error"):
                await adapter.on_message(
                    msg=sample_message,
                    tools=mock_tools,
                    history=[],
                    participants_msg=None,
                    is_session_bootstrap=True,
                    room_id="room-123",
                )

        elif adapter_config.name == "claude_sdk":
            # Claude SDK uses session-based error handling
            # Mock the session manager to return a client that raises error
            adapter._room_tools["room-123"] = mock_tools
            mock_client = AsyncMock()
            mock_client.query = AsyncMock(side_effect=Exception("Claude SDK Error"))
            adapter._session_manager.get_or_create_session = AsyncMock(
                return_value=mock_client
            )

            with pytest.raises(Exception, match="Claude SDK Error"):
                await adapter.on_message(
                    msg=sample_message,
                    tools=mock_tools,
                    history=[],
                    participants_msg=None,
                    is_session_bootstrap=True,
                    room_id="room-123",
                )

        elif adapter_config.name == "parlant":
            # Parlant uses SDK-based error handling
            # Mock session creation to raise error
            adapter._app.sessions.create_customer_message = AsyncMock(
                side_effect=Exception("Parlant Error")
            )

            with pytest.raises(Exception, match="Parlant Error"):
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
    if config.name == "parlant":
        await _setup_parlant_on_started(adapter)
    elif config.name == "claude_sdk":
        with patch("thenvoi.adapters.claude_sdk.ClaudeSessionManager") as mock_manager:
            mock_manager.return_value = MagicMock()
            await adapter.on_started(
                agent_name="TestBot", agent_description="A test bot"
            )
    elif config.name == "pydantic_ai":
        with patch.object(adapter, "_create_agent") as mock_create:
            mock_agent = MagicMock()
            mock_agent._function_tools = {}
            mock_create.return_value = mock_agent
            await adapter.on_started(
                agent_name="TestBot", agent_description="A test bot"
            )
    elif config.name == "crewai":
        crewai_mocks = getattr(config, "_crewai_mocks", None)
        if crewai_mocks:
            crewai_mocks.Agent.reset_mock()
        await adapter.on_started(agent_name="TestBot", agent_description="A test bot")
    else:
        await adapter.on_started(agent_name="TestBot", agent_description="A test bot")


async def _setup_parlant_on_started(adapter) -> None:
    """Set up Parlant adapter for on_started testing."""
    mock_app = MagicMock()
    mock_application_class = MagicMock(name="Application")
    mock_module = MagicMock()
    mock_module.Application = mock_application_class
    adapter._server.container = {mock_application_class: mock_app}

    with patch.dict(sys.modules, {"parlant.core.application": mock_module}):
        await adapter.on_started(agent_name="TestBot", agent_description="A test bot")


@asynccontextmanager
async def _mock_llm_call(
    adapter,
    config: AdapterConfig,
    mocks: dict | None,
    captured_input: dict | None = None,
):
    """Context manager to mock LLM calls for on_message tests.

    Args:
        adapter: The adapter instance being tested
        config: The adapter configuration
        mocks: Dictionary of mock objects from setup
        captured_input: Optional dict to capture LLM input for verification.
                       If provided, input will be captured for assertions.
    """
    if config.name == "anthropic":
        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = []
        with patch.object(adapter, "_call_anthropic", return_value=mock_response):
            yield

    elif config.name == "pydantic_ai":
        from pydantic_ai import AgentRunResultEvent
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        async def make_stream():
            result_event = MagicMock(spec=AgentRunResultEvent)
            result_event.result = MagicMock()
            result_event.result.all_messages.return_value = [
                ModelRequest(parts=[UserPromptPart(content="test")])
            ]
            yield result_event

        if mocks and "agent" in mocks:
            mocks["agent"].run_stream_events = MagicMock(return_value=make_stream())
        yield

    elif config.name == "langgraph":
        if captured_input is not None:
            # Capture mode - record input for verification
            async def capture_stream(graph_input, **kwargs):
                captured_input.update(graph_input)
                return
                yield

            stream_func = capture_stream
        else:
            # Simple mode - just return empty stream
            async def empty_stream(graph_input, **kwargs):
                return
                yield  # Make async generator

            stream_func = empty_stream

        mock_graph = MagicMock()
        mock_graph.astream_events = stream_func
        adapter.graph_factory = MagicMock(return_value=mock_graph)

        with patch(
            "thenvoi.integrations.langgraph.langchain_tools.agent_tools_to_langchain"
        ) as mock_convert:
            mock_convert.return_value = []
            yield

    elif config.name == "parlant":
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
            yield

    elif config.name == "claude_sdk":
        # ClaudeSDK uses session manager and client.query
        mock_client = AsyncMock()
        mock_client.query = AsyncMock(return_value=None)
        if adapter._session_manager:
            adapter._session_manager.get_or_create_session = AsyncMock(
                return_value=mock_client
            )
        with patch.object(adapter, "_process_response", return_value=None):
            yield

    elif config.name == "crewai":
        # CrewAI uses agent kickoff_async, mock it with proper result format
        if adapter._crewai_agent is not None:
            mock_result = MagicMock()
            mock_result.raw = "Test response"
            adapter._crewai_agent.kickoff_async = AsyncMock(return_value=mock_result)
        yield

    else:
        yield


# =============================================================================
# CrewAI-Specific Tests
# =============================================================================
# These tests only run for the CrewAI adapter as they test CrewAI-specific
# functionality like context variables, nest_asyncio, mentions validation, etc.


@pytest.fixture
def crewai_only(adapter_config: AdapterConfig):
    """Skip test if not testing CrewAI adapter."""
    if adapter_config.name != "crewai":
        pytest.skip("CrewAI-specific test")
    return adapter_config


@pytest.fixture
def crewai_room_context(adapter_config: AdapterConfig, mock_tools):
    """Context manager for setting up room context in CrewAI tests."""
    if adapter_config.name != "crewai":
        pytest.skip("CrewAI-specific test")

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


class TestCrewAIDeprecation:
    """CrewAI-specific deprecation tests."""

    def test_system_prompt_deprecation_warning(
        self, adapter_config: AdapterConfig, crewai_only
    ):
        """system_prompt parameter should emit DeprecationWarning."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            adapter = adapter_config.factory(system_prompt="Old style prompt")

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "system_prompt" in str(w[0].message)
            assert "backstory" in str(w[0].message)
            assert adapter.backstory == "Old style prompt"

    def test_system_prompt_does_not_override_backstory(
        self, adapter_config: AdapterConfig, crewai_only
    ):
        """If both system_prompt and backstory are provided, backstory takes precedence."""
        import warnings

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            adapter = adapter_config.factory(
                system_prompt="Old style prompt",
                backstory="New style backstory",
            )
            assert adapter.backstory == "New style backstory"


class TestCrewAIParameters:
    """CrewAI-specific parameter tests."""

    @pytest.mark.asyncio
    async def test_verbose_mode_passed_to_agent(
        self, adapter_config: AdapterConfig, crewai_only
    ):
        """verbose parameter should be passed to CrewAI Agent."""
        crewai_mocks = adapter_config._crewai_mocks
        crewai_mocks.Agent.reset_mock()

        adapter = adapter_config.factory(verbose=True)
        await adapter.on_started("TestBot", "Test bot")

        call_kwargs = crewai_mocks.Agent.call_args[1]
        assert call_kwargs["verbose"] is True

    @pytest.mark.asyncio
    async def test_max_rpm_passed_to_agent(
        self, adapter_config: AdapterConfig, crewai_only
    ):
        """max_rpm parameter should be passed to CrewAI Agent."""
        crewai_mocks = adapter_config._crewai_mocks
        crewai_mocks.Agent.reset_mock()

        adapter = adapter_config.factory(max_rpm=10)
        await adapter.on_started("TestBot", "Test bot")

        call_kwargs = crewai_mocks.Agent.call_args[1]
        assert call_kwargs["max_rpm"] == 10

    @pytest.mark.asyncio
    async def test_max_rpm_defaults_to_none(
        self, adapter_config: AdapterConfig, crewai_only
    ):
        """max_rpm should default to None (no rate limiting)."""
        crewai_mocks = adapter_config._crewai_mocks
        crewai_mocks.Agent.reset_mock()

        adapter = adapter_config.factory()
        await adapter.on_started("TestBot", "Test bot")

        call_kwargs = crewai_mocks.Agent.call_args[1]
        assert call_kwargs["max_rpm"] is None

    def test_max_rpm_stored_on_adapter(
        self, adapter_config: AdapterConfig, crewai_only
    ):
        """max_rpm should be stored on the adapter instance."""
        adapter = adapter_config.factory(max_rpm=60)
        assert adapter.max_rpm == 60

    @pytest.mark.asyncio
    async def test_allow_delegation_passed_to_agent(
        self, adapter_config: AdapterConfig, crewai_only
    ):
        """allow_delegation parameter should be passed to CrewAI Agent."""
        crewai_mocks = adapter_config._crewai_mocks
        crewai_mocks.Agent.reset_mock()

        adapter = adapter_config.factory(allow_delegation=True)
        await adapter.on_started("TestBot", "Test bot")

        call_kwargs = crewai_mocks.Agent.call_args[1]
        assert call_kwargs["allow_delegation"] is True

    @pytest.mark.asyncio
    async def test_allow_delegation_defaults_to_false(
        self, adapter_config: AdapterConfig, crewai_only
    ):
        """allow_delegation should default to False."""
        crewai_mocks = adapter_config._crewai_mocks
        crewai_mocks.Agent.reset_mock()

        adapter = adapter_config.factory()
        await adapter.on_started("TestBot", "Test bot")

        call_kwargs = crewai_mocks.Agent.call_args[1]
        assert call_kwargs["allow_delegation"] is False

    def test_allow_delegation_stored_on_adapter(
        self, adapter_config: AdapterConfig, crewai_only
    ):
        """allow_delegation should be stored on the adapter instance."""
        adapter = adapter_config.factory(allow_delegation=True)
        assert adapter.allow_delegation is True


class TestCrewAIToolExecutionContext:
    """CrewAI-specific tool execution context tests."""

    def test_tool_returns_error_without_room_context(
        self, adapter_config: AdapterConfig, crewai_only
    ):
        """Tools return error when called outside message handling."""
        import asyncio

        crewai_mocks = adapter_config._crewai_mocks
        crewai_mocks.Agent.reset_mock()

        adapter = adapter_config.factory()
        asyncio.run(adapter.on_started("TestBot", "Test bot"))

        call_kwargs = crewai_mocks.Agent.call_args[1]
        tools = call_kwargs["tools"]
        send_message_tool = next(t for t in tools if t.name == "thenvoi_send_message")

        result = send_message_tool._run(content="Hello!", mentions="[]")

        result_data = json.loads(result)
        assert result_data["status"] == "error"
        assert "No room context available" in result_data["message"]

    def test_successful_tool_execution_with_room_context(
        self,
        adapter_config: AdapterConfig,
        crewai_only,
        crewai_room_context,
        mock_tools,
    ):
        """Tools work when context variable is set."""
        import asyncio

        crewai_mocks = adapter_config._crewai_mocks
        crewai_mocks.Agent.reset_mock()

        adapter = adapter_config.factory()
        asyncio.run(adapter.on_started("TestBot", "Test bot"))

        call_kwargs = crewai_mocks.Agent.call_args[1]
        tools = call_kwargs["tools"]
        get_participants_tool = next(
            t for t in tools if t.name == "thenvoi_get_participants"
        )

        with crewai_room_context("room-123"):
            result = get_participants_tool._run()

        result_data = json.loads(result)
        assert result_data["status"] == "success"
        assert "participants" in result_data
        assert result_data["count"] == 1

    def test_tool_execution_handles_exception(
        self,
        adapter_config: AdapterConfig,
        crewai_only,
        crewai_room_context,
        mock_tools,
    ):
        """Tool exceptions should result in error response."""
        import asyncio

        crewai_mocks = adapter_config._crewai_mocks
        crewai_mocks.Agent.reset_mock()

        adapter = adapter_config.factory()
        asyncio.run(adapter.on_started("TestBot", "Test bot"))

        mock_tools.get_participants.side_effect = Exception("Connection failed")

        call_kwargs = crewai_mocks.Agent.call_args[1]
        tools = call_kwargs["tools"]
        get_participants_tool = next(
            t for t in tools if t.name == "thenvoi_get_participants"
        )

        with crewai_room_context("room-123"):
            result = get_participants_tool._run()

        result_data = json.loads(result)
        assert result_data["status"] == "error"
        assert "Connection failed" in result_data["message"]


class TestCrewAINestAsyncio:
    """CrewAI-specific nest_asyncio tests."""

    def test_nest_asyncio_not_applied_on_import(
        self, adapter_config: AdapterConfig, crewai_only
    ):
        """nest_asyncio should not be applied on module import."""
        import importlib

        sys.modules.pop("thenvoi.adapters.crewai", None)

        crewai_mocks_nest = sys.modules["nest_asyncio"]
        crewai_mocks_nest.reset_mock()

        importlib.import_module("thenvoi.adapters.crewai")

        crewai_mocks_nest.apply.assert_not_called()

    def test_ensure_nest_asyncio_applies_once(
        self, adapter_config: AdapterConfig, crewai_only
    ):
        """_ensure_nest_asyncio should only apply the patch once."""
        import importlib

        module = importlib.import_module("thenvoi.adapters.crewai")

        module._nest_asyncio_applied = False
        nest_mock = sys.modules["nest_asyncio"]
        nest_mock.reset_mock()

        module._ensure_nest_asyncio()
        module._ensure_nest_asyncio()

        assert nest_mock.apply.call_count == 1

    def test_nest_asyncio_lock_exists(self, adapter_config: AdapterConfig, crewai_only):
        """Module should have a threading lock for thread-safe nest_asyncio application."""
        import importlib
        import threading

        module = importlib.import_module("thenvoi.adapters.crewai")

        assert hasattr(module, "_nest_asyncio_lock")
        assert isinstance(module._nest_asyncio_lock, type(threading.Lock()))

    def test_ensure_nest_asyncio_is_thread_safe(
        self, adapter_config: AdapterConfig, crewai_only
    ):
        """Multiple threads should only apply nest_asyncio patch once."""
        import concurrent.futures
        import importlib

        module = importlib.import_module("thenvoi.adapters.crewai")

        module._nest_asyncio_applied = False
        nest_mock = sys.modules["nest_asyncio"]
        nest_mock.reset_mock()

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(module._ensure_nest_asyncio) for _ in range(10)]
            concurrent.futures.wait(futures)

        assert nest_mock.apply.call_count == 1


class TestCrewAIRunAsync:
    """CrewAI-specific _run_async tests."""

    def test_run_async_with_running_loop(
        self, adapter_config: AdapterConfig, crewai_only
    ):
        """_run_async should apply nest_asyncio when needed."""
        import importlib

        module = importlib.import_module("thenvoi.adapters.crewai")
        module._nest_asyncio_applied = False

        nest_mock = sys.modules["nest_asyncio"]
        nest_mock.reset_mock()

        async def test_coro() -> str:
            return "result"

        result = module._run_async(test_coro())

        assert result == "result"
        nest_mock.apply.assert_called_once()

    def test_run_async_without_running_loop(
        self, adapter_config: AdapterConfig, crewai_only
    ):
        """_run_async should work without re-applying nest_asyncio."""
        import importlib

        module = importlib.import_module("thenvoi.adapters.crewai")
        module._nest_asyncio_applied = True

        nest_mock = sys.modules["nest_asyncio"]
        nest_mock.reset_mock()

        async def test_coro() -> str:
            return "result"

        result = module._run_async(test_coro())

        assert result == "result"


class TestCrewAIMentionsValidator:
    """CrewAI-specific mentions validator tests."""

    @pytest.mark.asyncio
    async def test_mentions_list_converted_to_json(
        self, adapter_config: AdapterConfig, crewai_only
    ):
        """List mentions should be converted to JSON string."""
        crewai_mocks = adapter_config._crewai_mocks
        crewai_mocks.Agent.reset_mock()

        adapter = adapter_config.factory()
        await adapter.on_started("TestBot", "Test bot")

        call_kwargs = crewai_mocks.Agent.call_args[1]
        tools = call_kwargs["tools"]
        send_message_tool = next(t for t in tools if t.name == "thenvoi_send_message")

        input_model = send_message_tool.args_schema

        instance = input_model(
            content="Hello!",
            mentions=["Alice", "Bob"],
        )

        assert instance.mentions == '["Alice", "Bob"]'

    @pytest.mark.asyncio
    async def test_mentions_string_kept_as_is(
        self, adapter_config: AdapterConfig, crewai_only
    ):
        """String mentions should be kept as-is."""
        crewai_mocks = adapter_config._crewai_mocks
        crewai_mocks.Agent.reset_mock()

        adapter = adapter_config.factory()
        await adapter.on_started("TestBot", "Test bot")

        call_kwargs = crewai_mocks.Agent.call_args[1]
        tools = call_kwargs["tools"]
        send_message_tool = next(t for t in tools if t.name == "thenvoi_send_message")

        input_model = send_message_tool.args_schema

        instance = input_model(
            content="Hello!",
            mentions='["Alice"]',
        )

        assert instance.mentions == '["Alice"]'

    @pytest.mark.asyncio
    async def test_mentions_none_converted_to_empty_array(
        self, adapter_config: AdapterConfig, crewai_only
    ):
        """None mentions should be normalized to empty JSON array string."""
        crewai_mocks = adapter_config._crewai_mocks
        crewai_mocks.Agent.reset_mock()

        adapter = adapter_config.factory()
        await adapter.on_started("TestBot", "Test bot")

        call_kwargs = crewai_mocks.Agent.call_args[1]
        tools = call_kwargs["tools"]
        send_message_tool = next(t for t in tools if t.name == "thenvoi_send_message")

        input_model = send_message_tool.args_schema

        instance = input_model(
            content="Hello!",
            mentions=None,
        )

        assert instance.mentions == "[]"


class TestCrewAIPlatformInstructions:
    """CrewAI-specific platform instructions tests."""

    def test_platform_instructions_is_constant(
        self, adapter_config: AdapterConfig, crewai_only
    ):
        """PLATFORM_INSTRUCTIONS should be a non-empty string constant."""
        import importlib

        module = importlib.import_module("thenvoi.adapters.crewai")

        assert hasattr(module, "PLATFORM_INSTRUCTIONS")
        assert isinstance(module.PLATFORM_INSTRUCTIONS, str)
        assert len(module.PLATFORM_INSTRUCTIONS) > 100

    def test_platform_instructions_contains_key_info(
        self, adapter_config: AdapterConfig, crewai_only
    ):
        """PLATFORM_INSTRUCTIONS should contain key platform information."""
        import importlib

        module = importlib.import_module("thenvoi.adapters.crewai")

        instructions = module.PLATFORM_INSTRUCTIONS

        assert "Environment" in instructions
        assert "thenvoi_send_message" in instructions
        assert "thenvoi_lookup_peers" in instructions
        assert "thenvoi_add_participant" in instructions


class TestCrewAIToolSchemas:
    """CrewAI-specific tool schema tests."""

    @pytest.mark.asyncio
    async def test_all_tools_have_correct_schemas(
        self, adapter_config: AdapterConfig, crewai_only
    ):
        """Tools should not require room_id - context is managed via context variable."""
        crewai_mocks = adapter_config._crewai_mocks
        crewai_mocks.Agent.reset_mock()

        adapter = adapter_config.factory()
        await adapter.on_started("TestBot", "Test bot")

        call_kwargs = crewai_mocks.Agent.call_args[1]
        tools = call_kwargs["tools"]

        # thenvoi_send_message should have content and mentions, but NOT room_id
        send_message = next(t for t in tools if t.name == "thenvoi_send_message")
        assert send_message.args_schema is not None
        schema_fields = send_message.args_schema.model_fields
        assert "room_id" not in schema_fields
        assert "content" in schema_fields
        assert "mentions" in schema_fields

        # thenvoi_add_participant should have participant_name and role, but NOT room_id
        add_participant = next(t for t in tools if t.name == "thenvoi_add_participant")
        schema_fields = add_participant.args_schema.model_fields
        assert "room_id" not in schema_fields
        assert "participant_name" in schema_fields
        assert "role" in schema_fields

        # thenvoi_lookup_peers should have no user-facing parameters
        lookup_peers = next(t for t in tools if t.name == "thenvoi_lookup_peers")
        schema_fields = lookup_peers.args_schema.model_fields
        assert "room_id" not in schema_fields
        assert "page" not in schema_fields
        assert "page_size" not in schema_fields

    @pytest.mark.asyncio
    async def test_send_event_message_type_validation(
        self, adapter_config: AdapterConfig, crewai_only
    ):
        """send_event should have message_type with default 'thought'."""
        crewai_mocks = adapter_config._crewai_mocks
        crewai_mocks.Agent.reset_mock()

        adapter = adapter_config.factory()
        await adapter.on_started("TestBot", "Test bot")

        call_kwargs = crewai_mocks.Agent.call_args[1]
        tools = call_kwargs["tools"]

        send_event = next(t for t in tools if t.name == "thenvoi_send_event")
        schema_fields = send_event.args_schema.model_fields

        assert "message_type" in schema_fields
        message_type_field = schema_fields["message_type"]
        assert message_type_field.default == "thought"


class TestCrewAICustomToolExecution:
    """CrewAI-specific custom tool execution tests."""

    def test_custom_tool_execution_async(
        self,
        adapter_config: AdapterConfig,
        crewai_only,
        crewai_room_context,
        mock_tools,
    ):
        """Async custom tool should execute correctly."""
        import asyncio

        class EchoInput(BaseModel):
            """Echo back the provided message."""

            message: str = Field(description="Message to echo")

        async def echo_message(args: EchoInput) -> str:
            return f"Echo: {args.message}"

        crewai_mocks = adapter_config._crewai_mocks
        crewai_mocks.Agent.reset_mock()

        adapter = adapter_config.factory(
            additional_tools=[(EchoInput, echo_message)],
        )
        asyncio.run(adapter.on_started("TestBot", "Test bot"))

        call_kwargs = crewai_mocks.Agent.call_args[1]
        tools = call_kwargs["tools"]
        echo_tool = next(t for t in tools if t.name == "echo")

        with crewai_room_context("room-123"):
            result = echo_tool._run(message="Hello world")

        result_data = json.loads(result)
        assert result_data["status"] == "success"
        assert "Echo: Hello world" in result_data["result"]

    def test_custom_tool_execution_sync(
        self,
        adapter_config: AdapterConfig,
        crewai_only,
        crewai_room_context,
        mock_tools,
    ):
        """Sync custom tool should execute correctly."""
        import asyncio

        class CalculatorInput(BaseModel):
            """Perform math calculations."""

            operation: str = Field(description="add, subtract, multiply, divide")
            left: float = Field(description="Left operand")
            right: float = Field(description="Right operand")

        def calculate(args: CalculatorInput) -> str:
            ops = {
                "add": lambda a, b: a + b,
                "subtract": lambda a, b: a - b,
                "multiply": lambda a, b: a * b,
                "divide": lambda a, b: a / b,
            }
            return str(ops[args.operation](args.left, args.right))

        crewai_mocks = adapter_config._crewai_mocks
        crewai_mocks.Agent.reset_mock()

        adapter = adapter_config.factory(
            additional_tools=[(CalculatorInput, calculate)],
        )
        asyncio.run(adapter.on_started("TestBot", "Test bot"))

        call_kwargs = crewai_mocks.Agent.call_args[1]
        tools = call_kwargs["tools"]
        calc_tool = next(t for t in tools if t.name == "calculator")

        with crewai_room_context("room-123"):
            result = calc_tool._run(operation="add", left=5.0, right=3.0)

        result_data = json.loads(result)
        assert result_data["status"] == "success"
        assert "8.0" in result_data["result"]

    def test_custom_tool_error_handling(
        self,
        adapter_config: AdapterConfig,
        crewai_only,
        crewai_room_context,
        mock_tools,
    ):
        """Custom tool exception should result in error response."""
        import asyncio

        class EchoInput(BaseModel):
            """Echo back the provided message."""

            message: str = Field(description="Message to echo")

        async def failing_tool(args: EchoInput) -> str:
            raise ValueError("Service unavailable")

        crewai_mocks = adapter_config._crewai_mocks
        crewai_mocks.Agent.reset_mock()

        adapter = adapter_config.factory(
            additional_tools=[(EchoInput, failing_tool)],
        )
        asyncio.run(adapter.on_started("TestBot", "Test bot"))

        call_kwargs = crewai_mocks.Agent.call_args[1]
        tools = call_kwargs["tools"]
        echo_tool = next(t for t in tools if t.name == "echo")

        with crewai_room_context("room-123"):
            result = echo_tool._run(message="test")

        result_data = json.loads(result)
        assert result_data["status"] == "error"
        assert "Service unavailable" in result_data["message"]

    def test_custom_tool_reports_execution_when_enabled(
        self,
        adapter_config: AdapterConfig,
        crewai_only,
        crewai_room_context,
        mock_tools,
    ):
        """Custom tool should report events when execution_reporting is enabled."""
        import asyncio

        class EchoInput(BaseModel):
            """Echo back the provided message."""

            message: str = Field(description="Message to echo")

        async def echo_message(args: EchoInput) -> str:
            return f"Echo: {args.message}"

        crewai_mocks = adapter_config._crewai_mocks
        crewai_mocks.Agent.reset_mock()

        adapter = adapter_config.factory(
            enable_execution_reporting=True,
            additional_tools=[(EchoInput, echo_message)],
        )
        asyncio.run(adapter.on_started("TestBot", "Test bot"))

        call_kwargs = crewai_mocks.Agent.call_args[1]
        tools = call_kwargs["tools"]
        echo_tool = next(t for t in tools if t.name == "echo")

        with crewai_room_context("room-123"):
            echo_tool._run(message="Hello!")

        assert mock_tools.send_event.call_count >= 2

    def test_custom_tool_without_room_context(
        self, adapter_config: AdapterConfig, crewai_only
    ):
        """Custom tool should return error when called without room context."""
        import asyncio

        class EchoInput(BaseModel):
            """Echo back the provided message."""

            message: str = Field(description="Message to echo")

        async def echo_message(args: EchoInput) -> str:
            return f"Echo: {args.message}"

        crewai_mocks = adapter_config._crewai_mocks
        crewai_mocks.Agent.reset_mock()

        adapter = adapter_config.factory(
            additional_tools=[(EchoInput, echo_message)],
        )
        asyncio.run(adapter.on_started("TestBot", "Test bot"))

        call_kwargs = crewai_mocks.Agent.call_args[1]
        tools = call_kwargs["tools"]
        echo_tool = next(t for t in tools if t.name == "echo")

        result = echo_tool._run(message="Hello!")

        result_data = json.loads(result)
        assert result_data["status"] == "error"
        assert "No room context available" in result_data["message"]


class TestCrewAIExecutionReporting:
    """CrewAI-specific execution reporting tests."""

    def test_reports_tool_call_when_enabled(
        self,
        adapter_config: AdapterConfig,
        crewai_only,
        crewai_room_context,
        mock_tools,
    ):
        """Tool calls should be reported when execution_reporting is enabled."""
        import asyncio

        crewai_mocks = adapter_config._crewai_mocks
        crewai_mocks.Agent.reset_mock()

        adapter = adapter_config.factory(enable_execution_reporting=True)
        asyncio.run(adapter.on_started("TestBot", "Test bot"))

        call_kwargs = crewai_mocks.Agent.call_args[1]
        tools = call_kwargs["tools"]
        send_message_tool = next(t for t in tools if t.name == "thenvoi_send_message")

        with crewai_room_context("room-123"):
            send_message_tool._run(content="Hello!", mentions="[]")

        assert mock_tools.send_event.call_count >= 2
