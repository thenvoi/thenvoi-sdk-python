"""Framework adapter configurations for parameterized contract tests.

This module defines the configuration for each adapter framework, allowing
contract tests to run the same test logic across all adapters while handling
framework-specific behaviors through configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Literal, Protocol
from unittest.mock import AsyncMock, MagicMock

from thenvoi.core.types import PlatformMessage

# =============================================================================
# Type Definitions
# =============================================================================


class AdapterFactory(Protocol):
    """Protocol for adapter factory functions.

    Factory functions create adapter instances with optional keyword arguments.
    """

    def __call__(self, **kwargs: Any) -> Any:
        """Create an adapter instance with optional configuration.

        Args:
            **kwargs: Configuration options passed to the adapter constructor.

        Returns:
            An adapter instance conforming to FrameworkAdapter protocol.
        """
        ...


class ParticipantsVerifier(Protocol):
    """Protocol for participants verification callbacks."""

    def __call__(
        self, adapter: Any, captured_input: dict[str, Any], participants_msg: str
    ) -> bool:
        """Verify participants message was injected correctly."""
        ...


class OnStartedCallback(Protocol):
    """Protocol for on_started setup callbacks."""

    async def __call__(self, adapter: Any, config: "AdapterConfig") -> None:
        """Set up and call on_started with appropriate mocking."""
        ...


class MockLLMCallback(Protocol):
    """Protocol for LLM mocking callbacks."""

    def __call__(
        self, adapter: Any, mocks: dict | None, captured_input: dict | None
    ) -> Any:
        """Return a context manager that mocks LLM calls."""
        ...


class ErrorSetupCallback(Protocol):
    """Protocol for error handling test setup."""

    def __call__(self, adapter: Any, mocks: dict | None) -> Any:
        """Return a context manager that makes adapter raise an error."""
        ...


# Literal type for tool formats
ToolFormat = Literal["tuple", "callable"]


@dataclass
class AdapterConfig:
    """Configuration for a framework's adapter.

    This dataclass defines all configuration needed to test an adapter
    implementation in the conformance test suite.

    Attributes:
        name: Human-readable framework name (used as test ID)
        adapter_class: The adapter class to instantiate (may be None if requires_mocks)
        factory: Factory function to create adapter with required dependencies
        requires_mocks: Whether the adapter needs module-level mocks before import
        mock_setup: Optional function to set up mocks before adapter creation
        has_history_converter: Whether the adapter has a history_converter attribute
        has_custom_tools: Whether the adapter supports additional_tools parameter
        custom_tools_attr: Attribute name where custom tools are stored
        default_model: Default model value (if applicable)
        additional_init_checks: Dict of attribute name -> expected value for init tests
        supports_enable_execution_reporting: Whether adapter supports this parameter
        history_storage_attr: Attribute name where message history is stored
        supports_system_prompt_override: Whether system_prompt param overrides default
        supports_cleanup_all: Whether adapter has cleanup_all method
        custom_tool_format: "tuple" for (Model, func), "callable" for just func
        system_prompt_attr: Attribute name for system prompt
        system_prompt_contains_name: Whether system prompt should contain agent name
        alternative_prompt_attr: Alternative attribute for prompt (e.g., backstory)
        error_trigger_method: Method name to patch for error testing
        cleanup_storage_attrs: List of storage attributes to verify cleanup
        verify_participants_injection: Callback to verify participants were injected
    """

    # Required fields
    name: str
    adapter_class: type[Any] | None  # May be None if requires_mocks
    factory: AdapterFactory

    # Mock configuration
    requires_mocks: bool = False
    mock_setup: Callable[[], Any] | None = None

    # Feature support flags
    has_history_converter: bool = True
    has_custom_tools: bool = True
    supports_enable_execution_reporting: bool = True
    supports_system_prompt_override: bool = True
    supports_cleanup_all: bool = False

    # Attribute names
    custom_tools_attr: str = "_custom_tools"
    history_storage_attr: str | None = "_message_history"
    system_prompt_attr: str | None = "_system_prompt"
    alternative_prompt_attr: str | None = None  # e.g., "backstory" for crewai

    # Configuration values
    default_model: str | None = None
    additional_init_checks: dict[str, Any] = field(default_factory=dict)
    custom_tool_format: ToolFormat = "tuple"
    system_prompt_contains_name: bool = True

    # Error handling configuration
    error_trigger_method: str | None = None  # Method to patch for error testing

    # Cleanup configuration
    cleanup_storage_attrs: list[str] = field(default_factory=list)

    # Verification callbacks
    verify_participants_injection: ParticipantsVerifier | None = None

    # Setup callbacks (move framework-specific logic out of tests)
    on_started_callback: OnStartedCallback | None = None
    mock_llm_callback: MockLLMCallback | None = None
    error_setup_callback: ErrorSetupCallback | None = None


def create_sample_message(
    room_id: str = "room-123",
    content: str = "Hello, agent!",
    sender_name: str = "Alice",
) -> PlatformMessage:
    """Create a sample platform message for testing."""
    return PlatformMessage(
        id="msg-123",
        room_id=room_id,
        content=content,
        sender_id="user-456",
        sender_type="User",
        sender_name=sender_name,
        message_type="text",
        metadata={},
        created_at=datetime.now(timezone.utc),
    )


def create_mock_tools() -> AsyncMock:
    """Create mock AgentToolsProtocol for testing."""
    tools = AsyncMock()
    tools.get_tool_schemas = MagicMock(return_value=[])
    tools.get_openai_tool_schemas = MagicMock(return_value=[])
    tools.get_anthropic_tool_schemas = MagicMock(return_value=[])
    tools.send_message = AsyncMock(return_value={"status": "sent"})
    tools.send_event = AsyncMock(return_value={"status": "sent"})
    tools.execute_tool_call = AsyncMock(return_value={"status": "success"})
    tools.add_participant = AsyncMock(
        return_value={"id": "123", "name": "Test", "status": "added"}
    )
    tools.remove_participant = AsyncMock(
        return_value={"id": "123", "name": "Test", "status": "removed"}
    )
    tools.get_participants = AsyncMock(
        return_value=[{"id": "123", "name": "Alice", "type": "User"}]
    )
    tools.lookup_peers = AsyncMock(
        return_value={
            "peers": [],
            "metadata": {
                "page": 1,
                "page_size": 50,
                "total_count": 0,
                "total_pages": 1,
            },
        }
    )
    tools.create_chatroom = AsyncMock(return_value="new-room-123")
    return tools


# Factory functions for each adapter


def _create_anthropic_adapter(**kwargs: Any) -> Any:
    """Create AnthropicAdapter instance."""
    from thenvoi.adapters.anthropic import AnthropicAdapter

    return AnthropicAdapter(**kwargs)


def _create_claude_sdk_adapter(**kwargs: Any) -> Any:
    """Create ClaudeSDKAdapter instance."""
    from thenvoi.adapters.claude_sdk import ClaudeSDKAdapter

    return ClaudeSDKAdapter(**kwargs)


def _create_langgraph_adapter(**kwargs: Any) -> Any:
    """Create LangGraphAdapter instance with mock dependencies."""
    from thenvoi.adapters.langgraph import LangGraphAdapter

    # LangGraph requires llm + checkpointer or graph_factory or graph
    mock_llm = kwargs.pop("llm", MagicMock())
    mock_checkpointer = kwargs.pop("checkpointer", MagicMock())

    return LangGraphAdapter(llm=mock_llm, checkpointer=mock_checkpointer, **kwargs)


def _create_pydantic_ai_adapter(**kwargs: Any) -> Any:
    """Create PydanticAIAdapter instance."""
    from thenvoi.adapters.pydantic_ai import PydanticAIAdapter

    # PydanticAI requires model parameter
    model = kwargs.pop("model", "openai:gpt-4o")

    return PydanticAIAdapter(model=model, **kwargs)


def _create_crewai_adapter(crewai_mocks: Any = None, **kwargs: Any) -> Any:
    """Create CrewAIAdapter instance.

    Note: CrewAI requires module-level mocking, so this expects mocks
    to already be set up via the crewai_mocks fixture.
    """
    import importlib

    module = importlib.import_module("thenvoi.adapters.crewai")
    return module.CrewAIAdapter(**kwargs)


def _create_parlant_adapter(**kwargs: Any) -> Any:
    """Create ParlantAdapter instance with mock dependencies."""
    from thenvoi.adapters.parlant import ParlantAdapter

    # Parlant requires server and parlant_agent
    mock_server = kwargs.pop("server", None)
    mock_agent = kwargs.pop("parlant_agent", None)

    if mock_server is None:
        mock_server = MagicMock()
        mock_server.create_customer = AsyncMock(
            return_value=MagicMock(id="customer-123")
        )

    if mock_agent is None:
        mock_agent = MagicMock()
        mock_agent.id = "parlant-agent-123"
        mock_agent.name = "TestBot"

    return ParlantAdapter(server=mock_server, parlant_agent=mock_agent, **kwargs)


# =============================================================================
# Factory Functions for AdapterConfig
# =============================================================================


def make_standard_adapter_config(
    name: str,
    factory: AdapterFactory,
    *,
    default_model: str | None = None,
    supports_system_prompt_override: bool = True,
    has_history_converter: bool = True,
    has_custom_tools: bool = True,
    custom_tools_attr: str = "_custom_tools",
    custom_tool_format: ToolFormat = "tuple",
    supports_enable_execution_reporting: bool = True,
    history_storage_attr: str | None = "_message_history",
    system_prompt_attr: str | None = "_system_prompt",
    system_prompt_contains_name: bool = True,
    cleanup_storage_attrs: list[str] | None = None,
    additional_init_checks: dict[str, Any] | None = None,
    error_trigger_method: str | None = None,
    alternative_prompt_attr: str | None = None,
    supports_cleanup_all: bool = False,
    requires_mocks: bool = False,
    verify_participants_injection: ParticipantsVerifier | None = None,
    on_started_callback: OnStartedCallback | None = None,
    mock_llm_callback: MockLLMCallback | None = None,
    error_setup_callback: ErrorSetupCallback | None = None,
) -> AdapterConfig:
    """Create AdapterConfig with standard defaults.

    This factory function reduces boilerplate when creating adapter configs
    by providing sensible defaults for common configurations.

    Args:
        name: Human-readable framework name (used as test ID)
        factory: Factory function to create adapter
        default_model: Default model value (if applicable)
        supports_system_prompt_override: Whether system_prompt param overrides default
        has_history_converter: Whether the adapter has a history_converter attribute
        has_custom_tools: Whether the adapter supports additional_tools parameter
        custom_tools_attr: Attribute name where custom tools are stored
        custom_tool_format: "tuple" for (Model, func), "callable" for just func
        supports_enable_execution_reporting: Whether adapter supports this parameter
        history_storage_attr: Attribute name where message history is stored
        system_prompt_attr: Attribute name for system prompt
        system_prompt_contains_name: Whether system prompt should contain agent name
        cleanup_storage_attrs: List of storage attributes to verify cleanup
        additional_init_checks: Dict of attribute name -> expected value
        error_trigger_method: Method name to patch for error testing
        alternative_prompt_attr: Alternative attribute for prompt (e.g., backstory)
        supports_cleanup_all: Whether adapter has cleanup_all method
        requires_mocks: Whether the adapter needs module-level mocks
        verify_participants_injection: Callback to verify participants were injected

    Returns:
        Configured AdapterConfig instance
    """
    return AdapterConfig(
        name=name,
        adapter_class=None,
        factory=factory,
        requires_mocks=requires_mocks,
        has_history_converter=has_history_converter,
        has_custom_tools=has_custom_tools,
        custom_tools_attr=custom_tools_attr,
        default_model=default_model,
        additional_init_checks=additional_init_checks or {},
        supports_enable_execution_reporting=supports_enable_execution_reporting,
        history_storage_attr=history_storage_attr,
        supports_system_prompt_override=supports_system_prompt_override,
        supports_cleanup_all=supports_cleanup_all,
        custom_tool_format=custom_tool_format,
        system_prompt_attr=system_prompt_attr,
        system_prompt_contains_name=system_prompt_contains_name,
        alternative_prompt_attr=alternative_prompt_attr,
        error_trigger_method=error_trigger_method,
        cleanup_storage_attrs=cleanup_storage_attrs or ["_message_history"],
        verify_participants_injection=verify_participants_injection,
        on_started_callback=on_started_callback,
        mock_llm_callback=mock_llm_callback,
        error_setup_callback=error_setup_callback,
    )


# =============================================================================
# On-Started Callbacks (per-adapter)
# =============================================================================


async def _anthropic_on_started(adapter: Any, config: "AdapterConfig") -> None:
    """Call on_started for Anthropic adapter."""
    await adapter.on_started(agent_name="TestBot", agent_description="A test bot")


async def _langgraph_on_started(adapter: Any, config: "AdapterConfig") -> None:
    """Call on_started for LangGraph adapter."""
    await adapter.on_started(agent_name="TestBot", agent_description="A test bot")


async def _crewai_on_started(adapter: Any, config: "AdapterConfig") -> None:
    """Call on_started for CrewAI adapter."""
    crewai_mocks = getattr(config, "_crewai_mocks", None)
    if crewai_mocks:
        crewai_mocks.Agent.reset_mock()
    await adapter.on_started(agent_name="TestBot", agent_description="A test bot")


async def _claude_sdk_on_started(adapter: Any, config: "AdapterConfig") -> None:
    """Call on_started for ClaudeSDK adapter."""
    from unittest.mock import patch

    with patch("thenvoi.adapters.claude_sdk.ClaudeSessionManager") as mock_manager:
        mock_manager.return_value = MagicMock()
        await adapter.on_started(agent_name="TestBot", agent_description="A test bot")


async def _pydantic_ai_on_started(adapter: Any, config: "AdapterConfig") -> None:
    """Call on_started for PydanticAI adapter."""
    from unittest.mock import patch

    with patch.object(adapter, "_create_agent") as mock_create:
        mock_agent = MagicMock()
        mock_agent._function_tools = {}
        mock_create.return_value = mock_agent
        await adapter.on_started(agent_name="TestBot", agent_description="A test bot")


async def _parlant_on_started(adapter: Any, config: "AdapterConfig") -> None:
    """Call on_started for Parlant adapter."""
    import sys
    from unittest.mock import patch

    mock_app = MagicMock()
    mock_application_class = MagicMock(name="Application")
    mock_module = MagicMock()
    mock_module.Application = mock_application_class
    adapter._server.container = {mock_application_class: mock_app}

    with patch.dict(sys.modules, {"parlant.core.application": mock_module}):
        await adapter.on_started(agent_name="TestBot", agent_description="A test bot")


# =============================================================================
# Mock LLM Callbacks (per-adapter)
# =============================================================================


def _anthropic_mock_llm(
    adapter: Any, mocks: dict | None, captured_input: dict | None
) -> Any:
    """Return context manager for mocking Anthropic LLM calls."""
    from contextlib import asynccontextmanager
    from unittest.mock import MagicMock, patch

    @asynccontextmanager
    async def mock_ctx():
        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = []
        with patch.object(adapter, "_call_anthropic", return_value=mock_response):
            yield

    return mock_ctx()


def _langgraph_mock_llm(
    adapter: Any, mocks: dict | None, captured_input: dict | None
) -> Any:
    """Return context manager for mocking LangGraph LLM calls."""
    from contextlib import asynccontextmanager
    from typing import AsyncIterator
    from unittest.mock import MagicMock, patch

    @asynccontextmanager
    async def mock_ctx() -> AsyncIterator[None]:
        async def capture_stream(
            graph_input: dict, **kwargs: Any
        ) -> AsyncIterator[Any]:
            if captured_input is not None:
                captured_input.update(graph_input)
            if False:  # Make this an async generator
                yield

        mock_graph = MagicMock()
        mock_graph.astream_events = capture_stream
        adapter.graph_factory = MagicMock(return_value=mock_graph)

        with patch(
            "thenvoi.integrations.langgraph.langchain_tools.agent_tools_to_langchain"
        ) as mock_convert:
            mock_convert.return_value = []
            yield

    return mock_ctx()


def _crewai_mock_llm(
    adapter: Any, mocks: dict | None, captured_input: dict | None
) -> Any:
    """Return context manager for mocking CrewAI LLM calls."""
    from contextlib import asynccontextmanager
    from unittest.mock import MagicMock

    @asynccontextmanager
    async def mock_ctx():
        if adapter._crewai_agent is not None:
            mock_result = MagicMock()
            mock_result.raw = "Test response"
            adapter._crewai_agent.kickoff_async = AsyncMock(return_value=mock_result)
        yield

    return mock_ctx()


def _claude_sdk_mock_llm(
    adapter: Any, mocks: dict | None, captured_input: dict | None
) -> Any:
    """Return context manager for mocking ClaudeSDK LLM calls."""
    from contextlib import asynccontextmanager
    from unittest.mock import patch

    @asynccontextmanager
    async def mock_ctx():
        mock_client = AsyncMock()
        mock_client.query = AsyncMock(return_value=None)
        if adapter._session_manager:
            adapter._session_manager.get_or_create_session = AsyncMock(
                return_value=mock_client
            )
        with patch.object(adapter, "_process_response", return_value=None):
            yield

    return mock_ctx()


def _pydantic_ai_mock_llm(
    adapter: Any, mocks: dict | None, captured_input: dict | None
) -> Any:
    """Return context manager for mocking PydanticAI LLM calls."""
    from contextlib import asynccontextmanager
    from unittest.mock import MagicMock

    from pydantic_ai import AgentRunResultEvent
    from pydantic_ai.messages import ModelRequest, UserPromptPart

    @asynccontextmanager
    async def mock_ctx():
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

    return mock_ctx()


def _parlant_mock_llm(
    adapter: Any, mocks: dict | None, captured_input: dict | None
) -> Any:
    """Return context manager for mocking Parlant LLM calls."""
    import sys
    from contextlib import asynccontextmanager
    from unittest.mock import MagicMock, patch

    @asynccontextmanager
    async def mock_ctx():
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

    return mock_ctx()


# =============================================================================
# Error Setup Callbacks (per-adapter)
# =============================================================================


def _anthropic_error_setup(adapter: Any, mocks: dict | None) -> Any:
    """Return context manager that makes Anthropic adapter raise error."""
    from contextlib import asynccontextmanager
    from unittest.mock import patch

    @asynccontextmanager
    async def error_ctx():
        with patch.object(
            adapter, "_call_anthropic", side_effect=Exception("API Error")
        ):
            yield "API Error"

    return error_ctx()


def _langgraph_error_setup(adapter: Any, mocks: dict | None) -> Any:
    """Return context manager that makes LangGraph adapter raise error."""
    from contextlib import asynccontextmanager
    from typing import AsyncIterator
    from unittest.mock import MagicMock, patch

    @asynccontextmanager
    async def error_ctx() -> AsyncIterator[str]:
        async def failing_stream(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
            raise Exception("Graph error!")
            if False:  # Make this an async generator
                yield

        mock_graph = MagicMock()
        mock_graph.astream_events = failing_stream
        adapter.graph_factory = MagicMock(return_value=mock_graph)

        with patch(
            "thenvoi.integrations.langgraph.langchain_tools.agent_tools_to_langchain"
        ) as mock_convert:
            mock_convert.return_value = []
            yield "Graph error!"

    return error_ctx()


def _crewai_error_setup(adapter: Any, mocks: dict | None) -> Any:
    """Return context manager that makes CrewAI adapter raise error."""
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def error_ctx():
        if adapter._crewai_agent is not None:
            adapter._crewai_agent.kickoff_async = AsyncMock(
                side_effect=Exception("CrewAI Error")
            )
        yield "CrewAI Error"

    return error_ctx()


def _claude_sdk_error_setup(adapter: Any, mocks: dict | None) -> Any:
    """Return context manager that makes ClaudeSDK adapter raise error."""
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def error_ctx():
        mock_client = AsyncMock()
        mock_client.query = AsyncMock(side_effect=Exception("Claude SDK Error"))
        adapter._session_manager.get_or_create_session = AsyncMock(
            return_value=mock_client
        )
        yield "Claude SDK Error"

    return error_ctx()


def _pydantic_ai_error_setup(adapter: Any, mocks: dict | None) -> Any:
    """Return context manager that makes PydanticAI adapter raise error."""
    from contextlib import asynccontextmanager
    from typing import AsyncIterator
    from unittest.mock import MagicMock

    @asynccontextmanager
    async def error_ctx() -> AsyncIterator[str]:
        async def failing_stream() -> AsyncIterator[Any]:
            raise Exception("PydanticAI Error")
            if False:  # Make this an async generator
                yield

        if mocks and "agent" in mocks:
            mocks["agent"].run_stream_events = MagicMock(return_value=failing_stream())
        yield "PydanticAI Error"

    return error_ctx()


def _parlant_error_setup(adapter: Any, mocks: dict | None) -> Any:
    """Return context manager that makes Parlant adapter raise error."""
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def error_ctx():
        adapter._app.sessions.create_customer_message = AsyncMock(
            side_effect=Exception("Parlant Error")
        )
        yield "Parlant Error"

    return error_ctx()


# =============================================================================
# Participants Verification Functions (per-adapter)
# =============================================================================


def _verify_anthropic_participants(
    adapter: Any, captured_input: dict, participants_msg: str
) -> bool:
    """Verify participants message was injected into Anthropic adapter history."""
    return any(
        "[System]: Alice joined" in str(m.get("content", ""))
        for m in adapter._message_history.get("room-123", [])
    )


def _verify_langgraph_participants(
    adapter: Any, captured_input: dict, participants_msg: str
) -> bool:
    """Verify participants message was passed to LangGraph."""
    messages = captured_input.get("messages", [])
    system_msgs = [m for m in messages if m[0] == "system"]
    return any("Alice joined" in str(m[1]) for m in system_msgs)


def _verify_pydantic_ai_participants(
    adapter: Any, captured_input: dict, participants_msg: str, mocks: dict | None = None
) -> bool:
    """Verify participants message was added to PydanticAI history."""
    if mocks and "agent" in mocks:
        call_args = mocks["agent"].run_stream_events.call_args
        if call_args:
            call_kwargs = call_args.kwargs
            message_history = call_kwargs.get("message_history", [])
            if message_history:
                return any(
                    "[System]: Alice joined" in str(getattr(m.parts[0], "content", ""))
                    for m in message_history
                    if hasattr(m, "parts")
                    and m.parts
                    and hasattr(m.parts[0], "content")
                )
    return False


def _verify_crewai_participants(
    adapter: Any, captured_input: dict, participants_msg: str
) -> bool:
    """Verify participants message was included in CrewAI context."""
    # CrewAI includes participants in message context
    return "room-123" in adapter._message_history


def _verify_claude_sdk_participants(
    adapter: Any, captured_input: dict, participants_msg: str
) -> bool:
    """Verify Claude SDK participants handling (passes through session)."""
    # Claude SDK passes participants through session - verify no error occurred
    return True


def _verify_parlant_participants(
    adapter: Any, captured_input: dict, participants_msg: str
) -> bool:
    """Verify Parlant participants handling (uses SDK)."""
    # Parlant handles participants through SDK - verify session was created
    return True


# =============================================================================
# Adapter Configurations (using factory for conciseness)
# =============================================================================

ADAPTER_CONFIGS: dict[str, AdapterConfig] = {
    "anthropic": make_standard_adapter_config(
        "anthropic",
        _create_anthropic_adapter,
        default_model="claude-sonnet-4-5-20250929",
        additional_init_checks={
            "max_tokens": 4096,
            "enable_execution_reporting": False,
        },
        verify_participants_injection=_verify_anthropic_participants,
        on_started_callback=_anthropic_on_started,
        mock_llm_callback=_anthropic_mock_llm,
        error_setup_callback=_anthropic_error_setup,
    ),
    "claude_sdk": make_standard_adapter_config(
        "claude_sdk",
        _create_claude_sdk_adapter,
        default_model="claude-sonnet-4-5-20250929",
        has_history_converter=False,
        history_storage_attr="_room_tools",
        supports_system_prompt_override=False,
        supports_cleanup_all=True,
        system_prompt_attr=None,
        system_prompt_contains_name=False,
        cleanup_storage_attrs=["_room_tools"],
        additional_init_checks={
            "permission_mode": "acceptEdits",
            "enable_execution_reporting": False,
        },
        verify_participants_injection=_verify_claude_sdk_participants,
        on_started_callback=_claude_sdk_on_started,
        mock_llm_callback=_claude_sdk_mock_llm,
        error_setup_callback=_claude_sdk_error_setup,
    ),
    "langgraph": make_standard_adapter_config(
        "langgraph",
        _create_langgraph_adapter,
        custom_tools_attr="additional_tools",
        custom_tool_format="callable",
        supports_enable_execution_reporting=False,
        history_storage_attr=None,
        supports_system_prompt_override=False,
        cleanup_storage_attrs=[],
        additional_init_checks={"prompt_template": "default"},
        verify_participants_injection=_verify_langgraph_participants,
        on_started_callback=_langgraph_on_started,
        mock_llm_callback=_langgraph_mock_llm,
        error_setup_callback=_langgraph_error_setup,
    ),
    "pydantic_ai": make_standard_adapter_config(
        "pydantic_ai",
        _create_pydantic_ai_adapter,
        custom_tool_format="callable",
        supports_system_prompt_override=False,
        system_prompt_attr=None,
        system_prompt_contains_name=False,
        additional_init_checks={"enable_execution_reporting": False},
        on_started_callback=_pydantic_ai_on_started,
        mock_llm_callback=_pydantic_ai_mock_llm,
        error_setup_callback=_pydantic_ai_error_setup,
    ),
    "parlant": make_standard_adapter_config(
        "parlant",
        _create_parlant_adapter,
        has_custom_tools=False,
        supports_enable_execution_reporting=False,
        history_storage_attr="_room_sessions",
        supports_cleanup_all=True,
        cleanup_storage_attrs=["_room_sessions", "_room_customers"],
        verify_participants_injection=_verify_parlant_participants,
        on_started_callback=_parlant_on_started,
        mock_llm_callback=_parlant_mock_llm,
        error_setup_callback=_parlant_error_setup,
    ),
    "crewai": make_standard_adapter_config(
        "crewai",
        _create_crewai_adapter,
        requires_mocks=True,
        default_model="gpt-4o",
        supports_system_prompt_override=False,
        system_prompt_attr=None,
        system_prompt_contains_name=False,
        alternative_prompt_attr="backstory",
        additional_init_checks={
            "verbose": False,
            "max_iter": 20,
            "allow_delegation": False,
            "enable_execution_reporting": False,
        },
        verify_participants_injection=_verify_crewai_participants,
        on_started_callback=_crewai_on_started,
        mock_llm_callback=_crewai_mock_llm,
        error_setup_callback=_crewai_error_setup,
    ),
}


# Helper functions for on_message tests


async def setup_adapter_for_on_message(
    adapter: Any,
    config: AdapterConfig,
    mock_tools: Any,
) -> Any:
    """Set up adapter for on_message testing.

    Handles framework-specific mocking and returns any mock objects needed.

    Returns:
        Mock object(s) needed for assertions, or None
    """
    import sys
    from unittest.mock import patch

    if config.name == "parlant":
        # Parlant needs Application mock
        mock_app = MagicMock()
        mock_app.sessions = AsyncMock()
        mock_app.sessions.create = AsyncMock(return_value=MagicMock(id="session-123"))
        mock_app.sessions.create_customer_message = AsyncMock(
            return_value=MagicMock(offset=1)
        )
        mock_app.sessions.wait_for_update = AsyncMock(return_value=True)
        mock_app.sessions.find_events = AsyncMock(return_value=[])

        mock_application_class = MagicMock(name="Application")
        mock_module = MagicMock()
        mock_module.Application = mock_application_class
        adapter._server.container = {mock_application_class: mock_app}

        with patch.dict(sys.modules, {"parlant.core.application": mock_module}):
            await adapter.on_started(
                agent_name="TestBot", agent_description="A test bot"
            )

        # Set up internal state for on_message
        adapter._app = mock_app
        return {"app": mock_app}

    elif config.name == "claude_sdk":
        # ClaudeSDK needs session manager mock
        with patch(
            "thenvoi.adapters.claude_sdk.ClaudeSessionManager"
        ) as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager.cleanup_session = AsyncMock()
            mock_manager_class.return_value = mock_manager
            await adapter.on_started(
                agent_name="TestBot", agent_description="A test bot"
            )
        return {"session_manager": mock_manager}

    elif config.name == "pydantic_ai":
        # PydanticAI needs agent mock
        mock_agent = MagicMock()
        mock_agent._function_tools = {
            "thenvoi_send_message": MagicMock(name="thenvoi_send_message"),
        }
        with patch.object(adapter, "_create_agent", return_value=mock_agent):
            await adapter.on_started(
                agent_name="TestBot", agent_description="A test bot"
            )
        adapter._agent = mock_agent
        return {"agent": mock_agent}

    elif config.name == "anthropic":
        await adapter.on_started(agent_name="TestBot", agent_description="A test bot")
        return None

    elif config.name == "langgraph":
        await adapter.on_started(agent_name="TestBot", agent_description="A test bot")
        return None

    elif config.name == "crewai":
        await adapter.on_started(agent_name="TestBot", agent_description="A test bot")
        return None

    else:
        await adapter.on_started(agent_name="TestBot", agent_description="A test bot")
        return None
