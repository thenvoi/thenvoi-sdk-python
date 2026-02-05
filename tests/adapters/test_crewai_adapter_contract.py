"""Contract tests for CrewAI adapter.

CrewAI requires module-level mocking, so it has a separate contract test file
that uses the same test patterns as test_adapter_contract.py but with
CrewAI-specific setup.

Running:
    uv run pytest tests/adapters/test_crewai_adapter_contract.py -v
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, Field

from tests.framework_configs.adapters import CREWAI_CONFIG, create_mock_tools


class MockBaseTool:
    """Mock CrewAI BaseTool for testing."""

    name: str = ""
    description: str = ""

    def __init__(self):
        pass


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

    try:
        yield mock_crewai_module
    finally:
        # Clean up the adapter module to force reimport on next test
        sys.modules.pop("thenvoi.adapters.crewai", None)


@pytest.fixture
def CrewAIAdapter(crewai_mocks):
    """Get CrewAIAdapter class with mocks applied."""
    import importlib

    module = importlib.import_module("thenvoi.adapters.crewai")
    return module.CrewAIAdapter


@pytest.fixture
def adapter(CrewAIAdapter):
    """Create a CrewAI adapter instance."""
    return CrewAIAdapter()


@pytest.fixture
def mock_tools():
    """Create mock AgentToolsProtocol."""
    return create_mock_tools()


@pytest.fixture
def adapter_config():
    """Return the CrewAI adapter config."""
    return CREWAI_CONFIG


class TestInitialization:
    """Tests for CrewAI adapter initialization (contract tests)."""

    def test_default_initialization_creates_adapter(self, adapter):
        """Adapter should initialize without errors."""
        assert adapter is not None

    def test_has_history_converter(self, adapter):
        """Adapter should have history_converter."""
        assert hasattr(adapter, "history_converter")
        assert adapter.history_converter is not None

    def test_has_default_model(self, adapter, adapter_config):
        """Adapter should have correct default model."""
        assert hasattr(adapter, "model")
        assert adapter.model == adapter_config.default_model

    def test_additional_init_checks(self, adapter, adapter_config):
        """Adapter should have correct additional init values."""
        for attr, expected_value in adapter_config.additional_init_checks.items():
            assert hasattr(adapter, attr), f"Missing attribute: {attr}"
            assert getattr(adapter, attr) == expected_value, (
                f"Unexpected value for {attr}"
            )

    def test_accepts_custom_section_parameter(self, CrewAIAdapter):
        """Adapter should accept custom_section parameter."""
        adapter = CrewAIAdapter(custom_section="Be helpful.")
        assert hasattr(adapter, "custom_section")
        assert adapter.custom_section == "Be helpful."

    def test_accepts_enable_execution_reporting_parameter(self, CrewAIAdapter):
        """Adapter should accept enable_execution_reporting parameter."""
        adapter = CrewAIAdapter(enable_execution_reporting=True)
        assert hasattr(adapter, "enable_execution_reporting")
        assert adapter.enable_execution_reporting is True


class TestOnStarted:
    """Tests for on_started() method (contract tests)."""

    @pytest.mark.asyncio
    async def test_sets_agent_name(self, adapter, crewai_mocks):
        """on_started should set agent_name."""
        crewai_mocks.Agent.reset_mock()
        await adapter.on_started(agent_name="TestBot", agent_description="A test bot")

        assert adapter.agent_name == "TestBot"

    @pytest.mark.asyncio
    async def test_sets_agent_description(self, adapter, crewai_mocks):
        """on_started should set agent_description."""
        crewai_mocks.Agent.reset_mock()
        await adapter.on_started(agent_name="TestBot", agent_description="A test bot")

        assert adapter.agent_description == "A test bot"


class TestOnCleanup:
    """Tests for on_cleanup() method (contract tests)."""

    @pytest.mark.asyncio
    async def test_cleanup_nonexistent_room_is_safe(self, adapter):
        """Cleanup of non-existent room should not raise."""
        # Should not raise any exception
        await adapter.on_cleanup("nonexistent-room-xyz")

    @pytest.mark.asyncio
    async def test_cleanup_clears_room_data(self, adapter, crewai_mocks):
        """Cleanup should clear room-specific data."""
        await adapter.on_started(agent_name="TestBot", agent_description="A test bot")

        # Pre-populate room data
        adapter._message_history["room-123"] = [{"role": "user", "content": "test"}]

        await adapter.on_cleanup("room-123")

        assert "room-123" not in adapter._message_history


class TestCustomTools:
    """Tests for custom tool support (contract tests)."""

    def test_accepts_additional_tools_parameter(self, CrewAIAdapter):
        """Adapter should accept additional_tools parameter."""

        class EchoInput(BaseModel):
            """Echo the message."""

            message: str = Field(description="Message to echo")

        async def echo(args: EchoInput) -> str:
            return f"Echo: {args.message}"

        adapter = CrewAIAdapter(additional_tools=[(EchoInput, echo)])

        assert hasattr(adapter, "_custom_tools")
        assert len(adapter._custom_tools) == 1

    def test_defaults_to_empty_custom_tools(self, adapter):
        """Adapter should have empty custom tools by default."""
        assert hasattr(adapter, "_custom_tools")
        assert adapter._custom_tools == []
