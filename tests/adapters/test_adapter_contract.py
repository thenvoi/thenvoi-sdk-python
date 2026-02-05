"""Contract tests for all framework adapters.

These parameterized tests run across all adapter implementations to verify
common behaviors. Framework-specific behaviors (like tool execution, streaming)
remain in individual test files.

Running:
    # All adapters (except CrewAI which needs special mocks)
    uv run pytest tests/adapters/test_adapter_contract.py -v

    # Specific adapter
    uv run pytest tests/adapters/test_adapter_contract.py -k "anthropic" -v
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tests.framework_configs.adapters import (
    ADAPTER_CONFIGS,
    AdapterConfig,
    create_mock_tools,
    create_sample_message,
)


@pytest.fixture(params=list(ADAPTER_CONFIGS.values()), ids=lambda c: c.name)
def adapter_config(request: pytest.FixtureRequest) -> AdapterConfig:
    """Parameterized fixture that yields each adapter config."""
    return request.param


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

    def test_has_default_model_if_applicable(
        self, adapter, adapter_config: AdapterConfig
    ):
        """Adapter should have correct default model if applicable."""
        if adapter_config.default_model is not None:
            assert hasattr(adapter, "model")
            assert adapter.model == adapter_config.default_model

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

    def test_accepts_enable_execution_reporting_parameter(
        self, adapter_config: AdapterConfig
    ):
        """Adapter should accept enable_execution_reporting parameter."""
        # Skip adapters that don't support this parameter
        if adapter_config.name in ("parlant", "langgraph"):
            pytest.skip(
                f"{adapter_config.name} doesn't support enable_execution_reporting"
            )

        adapter = adapter_config.factory(enable_execution_reporting=True)
        assert hasattr(adapter, "enable_execution_reporting")
        assert adapter.enable_execution_reporting is True


class TestOnStarted:
    """Tests for on_started() method across all frameworks."""

    @pytest.mark.asyncio
    async def test_sets_agent_name(self, adapter, adapter_config: AdapterConfig):
        """on_started should set agent_name."""
        # Parlant needs Application mock
        if adapter_config.name == "parlant":
            await _setup_parlant_on_started(adapter)
        # ClaudeSDK needs session manager mock
        elif adapter_config.name == "claude_sdk":
            with patch(
                "thenvoi.adapters.claude_sdk.ClaudeSessionManager"
            ) as mock_manager:
                mock_manager.return_value = MagicMock()
                await adapter.on_started(
                    agent_name="TestBot", agent_description="A test bot"
                )
        # PydanticAI needs agent mock
        elif adapter_config.name == "pydantic_ai":
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent._function_tools = {}
                mock_create.return_value = mock_agent
                await adapter.on_started(
                    agent_name="TestBot", agent_description="A test bot"
                )
        else:
            await adapter.on_started(
                agent_name="TestBot", agent_description="A test bot"
            )

        assert adapter.agent_name == "TestBot"

    @pytest.mark.asyncio
    async def test_sets_agent_description(self, adapter, adapter_config: AdapterConfig):
        """on_started should set agent_description."""
        if adapter_config.name == "parlant":
            await _setup_parlant_on_started(adapter)
        elif adapter_config.name == "claude_sdk":
            with patch(
                "thenvoi.adapters.claude_sdk.ClaudeSessionManager"
            ) as mock_manager:
                mock_manager.return_value = MagicMock()
                await adapter.on_started(
                    agent_name="TestBot", agent_description="A test bot"
                )
        elif adapter_config.name == "pydantic_ai":
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent._function_tools = {}
                mock_create.return_value = mock_agent
                await adapter.on_started(
                    agent_name="TestBot", agent_description="A test bot"
                )
        else:
            await adapter.on_started(
                agent_name="TestBot", agent_description="A test bot"
            )

        assert adapter.agent_description == "A test bot"

    @pytest.mark.asyncio
    async def test_renders_system_prompt_with_agent_name(
        self, adapter, adapter_config: AdapterConfig
    ):
        """on_started should render system prompt containing agent name."""
        # Skip adapters without _system_prompt attribute
        if adapter_config.name in ("claude_sdk", "pydantic_ai"):
            pytest.skip(f"{adapter_config.name} handles system prompt differently")

        if adapter_config.name == "parlant":
            await _setup_parlant_on_started(adapter)
        else:
            await adapter.on_started(
                agent_name="TestBot", agent_description="A test bot"
            )

        assert hasattr(adapter, "_system_prompt")
        assert "TestBot" in adapter._system_prompt


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
        # Pre-populate room data if adapter has _message_history
        if hasattr(adapter, "_message_history"):
            adapter._message_history["room-123"] = [{"role": "user", "content": "test"}]

            await adapter.on_cleanup("room-123")

            assert "room-123" not in adapter._message_history

        # For Parlant, check session cleanup
        elif adapter_config.name == "parlant":
            adapter._room_sessions["room-123"] = "session-123"
            adapter._room_customers["room-123"] = "customer-123"

            await adapter.on_cleanup("room-123")

            assert "room-123" not in adapter._room_sessions
            assert "room-123" not in adapter._room_customers

        # For ClaudeSDK, check _room_tools cleanup
        elif adapter_config.name == "claude_sdk":
            adapter._room_tools["room-123"] = MagicMock()

            await adapter.on_cleanup("room-123")

            assert "room-123" not in adapter._room_tools


class TestCustomTools:
    """Tests for custom tool support across all frameworks."""

    def test_accepts_additional_tools_parameter(self, adapter_config: AdapterConfig):
        """Adapter should accept additional_tools parameter."""
        if not adapter_config.has_custom_tools:
            pytest.skip(f"{adapter_config.name} doesn't support custom tools")

        # Create a mock custom tool
        from pydantic import BaseModel, Field

        class EchoInput(BaseModel):
            """Echo the message."""

            message: str = Field(description="Message to echo")

        async def echo(args: EchoInput) -> str:
            return f"Echo: {args.message}"

        # PydanticAI uses a different tool format (just callables)
        if adapter_config.name == "pydantic_ai":
            adapter = adapter_config.factory(additional_tools=[echo])
        else:
            adapter = adapter_config.factory(additional_tools=[(EchoInput, echo)])

        # Verify custom tools were stored
        custom_tools_attr = adapter_config.custom_tools_attr
        assert hasattr(adapter, custom_tools_attr)
        custom_tools = getattr(adapter, custom_tools_attr)

        # LangGraph clears additional_tools after baking into factory
        if adapter_config.name == "langgraph":
            # LangGraph bakes tools into factory, so list should be empty
            assert custom_tools == []
        else:
            assert len(custom_tools) >= 1

    def test_defaults_to_empty_custom_tools(self, adapter_config: AdapterConfig):
        """Adapter should have empty custom tools by default."""
        if not adapter_config.has_custom_tools:
            pytest.skip(f"{adapter_config.name} doesn't support custom tools")

        adapter = adapter_config.factory()
        custom_tools_attr = adapter_config.custom_tools_attr
        assert hasattr(adapter, custom_tools_attr)
        custom_tools = getattr(adapter, custom_tools_attr)
        assert custom_tools == [] or custom_tools is None or len(custom_tools) == 0


# Helper functions


async def _setup_parlant_on_started(adapter) -> None:
    """Set up Parlant adapter for on_started testing."""
    import sys

    mock_app = MagicMock()
    mock_application_class = MagicMock(name="Application")
    mock_module = MagicMock()
    mock_module.Application = mock_application_class
    adapter._server.container = {mock_application_class: mock_app}

    with patch.dict(sys.modules, {"parlant.core.application": mock_module}):
        await adapter.on_started(agent_name="TestBot", agent_description="A test bot")
