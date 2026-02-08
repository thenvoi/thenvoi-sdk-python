"""Base test class for adapter conformance testing.

New framework authors can inherit from this class to verify their adapter
implementation without editing the shared config files.

Example usage:

    # tests/adapters/test_my_framework.py
    from tests.framework_configs.base_adapter_tests import BaseAdapterTests
    from thenvoi.adapters.my_framework import MyFrameworkAdapter

    class TestMyFrameworkAdapter(BaseAdapterTests):
        # Required
        adapter_class = MyFrameworkAdapter

        # Optional overrides (defaults shown)
        has_history_converter = True
        has_custom_tools = True
        custom_tools_attr = "_custom_tools"
        custom_tool_format = "tuple"  # or "callable"
        supports_enable_execution_reporting = True
        supports_system_prompt_override = True
        history_storage_attr = "_message_history"
        system_prompt_attr = "_system_prompt"
        cleanup_storage_attrs = ["_message_history"]

        # Override these methods for framework-specific behavior
        def create_adapter(self, **kwargs):
            return MyFrameworkAdapter(**kwargs)

        async def setup_on_started(self, adapter):
            # Any mocking needed before on_started
            await adapter.on_started(agent_name="TestBot", agent_description="A test bot")

        async def setup_on_message(self, adapter, mock_tools):
            # Setup needed before on_message (call on_started, etc.)
            await self.setup_on_started(adapter)
            return {}  # Return any mocks needed

        def mock_llm_call(self, adapter, mocks):
            # Return a context manager that mocks LLM calls
            from contextlib import nullcontext
            return nullcontext()

Running:
    uv run pytest tests/adapters/test_my_framework.py -v
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import nullcontext
from datetime import datetime, timezone
from typing import Any, ClassVar, Literal
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel, Field

from thenvoi.core.types import PlatformMessage


class BaseAdapterTests(ABC):
    """Base class providing conformance tests for adapters.

    Subclasses must set:
        - adapter_class: The adapter class to test

    Subclasses must implement:
        - create_adapter(**kwargs): Factory method to create adapter instances
        - setup_on_started(adapter): Setup and call on_started with any required mocking
        - setup_on_message(adapter, mock_tools): Setup for on_message tests
        - mock_llm_call(adapter, mocks): Return context manager to mock LLM calls

    Subclasses may override:
        - has_history_converter: Whether adapter has history_converter attribute
        - has_custom_tools: Whether adapter supports additional_tools parameter
        - custom_tools_attr: Attribute name storing custom tools
        - custom_tool_format: "tuple" for (Model, func) or "callable" for func
        - supports_enable_execution_reporting: Whether adapter has this parameter
        - supports_system_prompt_override: Whether system_prompt parameter works
        - history_storage_attr: Attribute storing room history (for cleanup tests)
        - system_prompt_attr: Attribute storing rendered system prompt
        - cleanup_storage_attrs: List of attributes to check in cleanup tests
        - default_model: Expected default model value (None if no default)
        - additional_init_checks: Dict of {attr: expected_value} to verify on init
    """

    # Required - must be set by subclass
    adapter_class: ClassVar[type]

    # Optional configuration (sensible defaults)
    has_history_converter: ClassVar[bool] = True
    has_custom_tools: ClassVar[bool] = True
    custom_tools_attr: ClassVar[str] = "_custom_tools"
    custom_tool_format: ClassVar[Literal["tuple", "callable"]] = "tuple"
    supports_enable_execution_reporting: ClassVar[bool] = True
    supports_system_prompt_override: ClassVar[bool] = True
    history_storage_attr: ClassVar[str | None] = "_message_history"
    system_prompt_attr: ClassVar[str | None] = "_system_prompt"
    cleanup_storage_attrs: ClassVar[list[str]] = ["_message_history"]
    default_model: ClassVar[str | None] = None
    additional_init_checks: ClassVar[dict[str, Any]] = {}

    # ==================== Abstract Methods (must implement) ====================

    @abstractmethod
    def create_adapter(self, **kwargs: Any) -> Any:
        """Create an adapter instance. Override to provide required arguments."""
        pass

    @abstractmethod
    async def setup_on_started(self, adapter: Any) -> None:
        """Setup and call on_started with any required mocking.

        Example:
            async def setup_on_started(self, adapter):
                await adapter.on_started(agent_name="TestBot", agent_description="A test bot")
        """
        pass

    @abstractmethod
    async def setup_on_message(self, adapter: Any, mock_tools: Any) -> dict:
        """Setup needed before on_message tests.

        Should call setup_on_started and return any mocks needed for mock_llm_call.

        Example:
            async def setup_on_message(self, adapter, mock_tools):
                await self.setup_on_started(adapter)
                return {}
        """
        pass

    @abstractmethod
    def mock_llm_call(self, adapter: Any, mocks: dict) -> Any:
        """Return a context manager that mocks LLM calls.

        Example:
            def mock_llm_call(self, adapter, mocks):
                from unittest.mock import patch, MagicMock
                mock_response = MagicMock(stop_reason="end_turn", content=[])
                return patch.object(adapter, "_call_llm", return_value=mock_response)
        """
        pass

    # ==================== Fixtures ====================

    @pytest.fixture
    def adapter(self) -> Any:
        """Create an adapter instance."""
        return self.create_adapter()

    @pytest.fixture
    def mock_tools(self) -> AsyncMock:
        """Create mock AgentToolsProtocol."""
        tools = AsyncMock()
        tools.get_tool_schemas = MagicMock(return_value=[])
        tools.get_openai_tool_schemas = MagicMock(return_value=[])
        tools.get_anthropic_tool_schemas = MagicMock(return_value=[])
        tools.send_message = AsyncMock(return_value={"status": "sent"})
        tools.send_event = AsyncMock(return_value={"status": "sent"})
        tools.execute_tool_call = AsyncMock(return_value={"status": "success"})
        tools.add_participant = AsyncMock(return_value={"id": "123", "name": "Test", "status": "added"})
        tools.remove_participant = AsyncMock(return_value={"id": "123", "name": "Test", "status": "removed"})
        tools.get_participants = AsyncMock(return_value=[{"id": "123", "name": "Alice", "type": "User"}])
        tools.lookup_peers = AsyncMock(return_value={"peers": [], "metadata": {"page": 1, "page_size": 50, "total_count": 0, "total_pages": 1}})
        tools.create_chatroom = AsyncMock(return_value="new-room-123")
        return tools

    @pytest.fixture
    def sample_message(self) -> PlatformMessage:
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

    # ==================== Initialization Tests ====================

    def test_default_initialization_creates_adapter(self, adapter):
        """Adapter should initialize without errors."""
        assert adapter is not None

    def test_has_history_converter_if_expected(self, adapter):
        """Adapter should have history_converter if configured."""
        if self.has_history_converter:
            assert hasattr(adapter, "history_converter")
            assert adapter.history_converter is not None

    def test_has_default_model_if_applicable(self, adapter):
        """Adapter should have correct default model if applicable."""
        if self.default_model is not None:
            assert hasattr(adapter, "model")
            assert adapter.model == self.default_model

    def test_additional_init_checks(self, adapter):
        """Adapter should have correct additional init values."""
        for attr, expected_value in self.additional_init_checks.items():
            assert hasattr(adapter, attr), f"Missing attribute: {attr}"
            assert getattr(adapter, attr) == expected_value, (
                f"Unexpected value for {attr}: {getattr(adapter, attr)} != {expected_value}"
            )

    def test_accepts_custom_section_parameter(self):
        """Adapter should accept custom_section parameter."""
        adapter = self.create_adapter(custom_section="Be helpful.")
        assert hasattr(adapter, "custom_section")
        assert adapter.custom_section == "Be helpful."

    def test_enable_execution_reporting_parameter(self):
        """Test enable_execution_reporting parameter handling."""
        if self.supports_enable_execution_reporting:
            adapter = self.create_adapter(enable_execution_reporting=True)
            assert hasattr(adapter, "enable_execution_reporting")
            assert adapter.enable_execution_reporting is True
        else:
            adapter = self.create_adapter()
            if hasattr(adapter, "enable_execution_reporting"):
                assert adapter.enable_execution_reporting is False

    # ==================== on_started Tests ====================

    @pytest.mark.asyncio
    async def test_sets_agent_name(self, adapter):
        """on_started should set agent_name."""
        await self.setup_on_started(adapter)
        assert adapter.agent_name == "TestBot"

    @pytest.mark.asyncio
    async def test_sets_agent_description(self, adapter):
        """on_started should set agent_description."""
        await self.setup_on_started(adapter)
        assert adapter.agent_description == "A test bot"

    @pytest.mark.asyncio
    async def test_system_prompt_contains_agent_info(self, adapter):
        """on_started should set system prompt with agent info."""
        await self.setup_on_started(adapter)

        if self.system_prompt_attr:
            assert hasattr(adapter, self.system_prompt_attr)
            prompt = getattr(adapter, self.system_prompt_attr)
            if prompt:  # Some adapters may have empty prompt initially
                assert "TestBot" in prompt

    @pytest.mark.asyncio
    async def test_system_prompt_override_behavior(self):
        """Test system_prompt parameter behavior."""
        if self.supports_system_prompt_override:
            adapter = self.create_adapter(system_prompt="Custom prompt here.")
            await self.setup_on_started(adapter)
            assert adapter._system_prompt == "Custom prompt here."

    # ==================== on_message Tests ====================

    @pytest.mark.asyncio
    async def test_initializes_history_on_bootstrap(self, sample_message, mock_tools):
        """Adapter should initialize room history/session on first message."""
        adapter = self.create_adapter()
        mocks = await self.setup_on_message(adapter, mock_tools)

        with self.mock_llm_call(adapter, mocks):
            await adapter.on_message(
                msg=sample_message,
                tools=mock_tools,
                history=[],
                participants_msg=None,
                is_session_bootstrap=True,
                room_id="room-123",
            )

        if self.history_storage_attr:
            storage = getattr(adapter, self.history_storage_attr, None)
            if storage is not None:
                assert "room-123" in storage

    # ==================== on_cleanup Tests ====================

    @pytest.mark.asyncio
    async def test_cleanup_nonexistent_room_is_safe(self, adapter):
        """Cleanup of non-existent room should not raise."""
        await adapter.on_cleanup("nonexistent-room-xyz")

    @pytest.mark.asyncio
    async def test_cleanup_clears_room_data(self, adapter):
        """Cleanup should clear room-specific data."""
        # Pre-populate room data
        for storage_attr in self.cleanup_storage_attrs:
            if hasattr(adapter, storage_attr):
                storage = getattr(adapter, storage_attr)
                if isinstance(storage, dict):
                    storage["room-123"] = {"test": "data"}

        await adapter.on_cleanup("room-123")

        # Verify cleanup
        for storage_attr in self.cleanup_storage_attrs:
            if hasattr(adapter, storage_attr):
                storage = getattr(adapter, storage_attr)
                if isinstance(storage, dict):
                    assert "room-123" not in storage

    # ==================== Custom Tools Tests ====================

    def test_additional_tools_parameter(self):
        """Test additional_tools parameter handling."""
        if not self.has_custom_tools:
            adapter = self.create_adapter()
            assert adapter is not None
            return

        class EchoInput(BaseModel):
            """Echo the message."""
            message: str = Field(description="Message to echo")

        async def echo(args: EchoInput) -> str:
            return f"Echo: {args.message}"

        if self.custom_tool_format == "callable":
            adapter = self.create_adapter(additional_tools=[echo])
        else:
            adapter = self.create_adapter(additional_tools=[(EchoInput, echo)])

        assert hasattr(adapter, self.custom_tools_attr)
        custom_tools = getattr(adapter, self.custom_tools_attr)
        # Some adapters clear tools after processing, so just verify no error
        assert custom_tools is not None or custom_tools == []

    def test_defaults_to_empty_custom_tools(self):
        """Adapter should have empty custom tools by default."""
        if not self.has_custom_tools:
            return

        adapter = self.create_adapter()
        assert hasattr(adapter, self.custom_tools_attr)
        custom_tools = getattr(adapter, self.custom_tools_attr)
        assert custom_tools == [] or custom_tools is None or len(custom_tools) == 0

    def test_multiple_custom_tools(self):
        """Adapter should accept multiple custom tools if supported."""
        if not self.has_custom_tools:
            return

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
            ops = {"add": lambda a, b: a + b, "subtract": lambda a, b: a - b, "multiply": lambda a, b: a * b, "divide": lambda a, b: a / b}
            return str(ops[args.operation](args.left, args.right))

        if self.custom_tool_format == "callable":
            adapter = self.create_adapter(additional_tools=[echo, calculate])
        else:
            adapter = self.create_adapter(additional_tools=[(EchoInput, echo), (CalculatorInput, calculate)])

        custom_tools = getattr(adapter, self.custom_tools_attr)
        # Some adapters clear tools after processing
        assert custom_tools is not None or custom_tools == []
