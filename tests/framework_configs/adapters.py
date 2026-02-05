"""Framework adapter configurations for parameterized conformance tests."""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Callable, Literal, Protocol
from unittest.mock import AsyncMock, MagicMock, patch

from thenvoi.core.types import PlatformMessage


# =============================================================================
# Type Definitions
# =============================================================================


class AdapterFactory(Protocol):
    """Protocol for adapter factory functions."""

    def __call__(self, **kwargs: Any) -> Any: ...


class ParticipantsVerifier(Protocol):
    """Protocol for participants verification callbacks."""

    def __call__(
        self, adapter: Any, captured_input: dict[str, Any], participants_msg: str
    ) -> bool: ...


class OnStartedCallback(Protocol):
    """Protocol for on_started setup callbacks."""

    async def __call__(self, adapter: Any, config: "AdapterConfig") -> None: ...


class MockLLMCallback(Protocol):
    """Protocol for LLM mocking callbacks."""

    def __call__(
        self, adapter: Any, mocks: dict | None, captured_input: dict | None
    ) -> Any: ...


class ErrorSetupCallback(Protocol):
    """Protocol for error handling test setup."""

    def __call__(self, adapter: Any, mocks: dict | None) -> Any: ...


ToolFormat = Literal["tuple", "callable"]


@dataclass
class AdapterConfig:
    """Configuration for a framework's adapter in conformance tests."""

    # Required fields
    name: str
    adapter_class: type[Any] | None
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
    alternative_prompt_attr: str | None = None

    # Configuration values
    default_model: str | None = None
    additional_init_checks: dict[str, Any] = field(default_factory=dict)
    custom_tool_format: ToolFormat = "tuple"
    system_prompt_contains_name: bool = True

    # Error handling configuration
    error_trigger_method: str | None = None

    # Cleanup configuration
    cleanup_storage_attrs: list[str] = field(default_factory=list)

    # Callbacks
    verify_participants_injection: ParticipantsVerifier | None = None
    on_started_callback: OnStartedCallback | None = None
    mock_llm_callback: MockLLMCallback | None = None
    error_setup_callback: ErrorSetupCallback | None = None


# =============================================================================
# Test Helpers
# =============================================================================


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


# =============================================================================
# Adapter Factory Functions
# =============================================================================


def _create_anthropic_adapter(**kwargs: Any) -> Any:
    from thenvoi.adapters.anthropic import AnthropicAdapter

    return AnthropicAdapter(**kwargs)


def _create_claude_sdk_adapter(**kwargs: Any) -> Any:
    from thenvoi.adapters.claude_sdk import ClaudeSDKAdapter

    return ClaudeSDKAdapter(**kwargs)


def _create_langgraph_adapter(**kwargs: Any) -> Any:
    from thenvoi.adapters.langgraph import LangGraphAdapter

    return LangGraphAdapter(
        llm=kwargs.pop("llm", MagicMock()),
        checkpointer=kwargs.pop("checkpointer", MagicMock()),
        **kwargs,
    )


def _create_pydantic_ai_adapter(**kwargs: Any) -> Any:
    from thenvoi.adapters.pydantic_ai import PydanticAIAdapter

    return PydanticAIAdapter(model=kwargs.pop("model", "openai:gpt-4o"), **kwargs)


def _create_crewai_adapter(crewai_mocks: Any = None, **kwargs: Any) -> Any:
    import importlib

    return importlib.import_module("thenvoi.adapters.crewai").CrewAIAdapter(**kwargs)


def _create_parlant_adapter(**kwargs: Any) -> Any:
    from thenvoi.adapters.parlant import ParlantAdapter

    server = kwargs.pop("server", None)
    agent = kwargs.pop("parlant_agent", None)

    if server is None:
        server = MagicMock()
        server.create_customer = AsyncMock(return_value=MagicMock(id="customer-123"))

    if agent is None:
        agent = MagicMock(id="parlant-agent-123", name="TestBot")

    return ParlantAdapter(server=server, parlant_agent=agent, **kwargs)


# =============================================================================
# Config Factory
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
    """Create AdapterConfig with standard defaults."""
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
# On-Started Callbacks
# =============================================================================


async def _simple_on_started(adapter: Any, config: AdapterConfig) -> None:
    """Default on_started for adapters without special requirements."""
    await adapter.on_started(agent_name="TestBot", agent_description="A test bot")


async def _crewai_on_started(adapter: Any, config: AdapterConfig) -> None:
    crewai_mocks = getattr(config, "_crewai_mocks", None)
    if crewai_mocks:
        crewai_mocks.Agent.reset_mock()
    await adapter.on_started(agent_name="TestBot", agent_description="A test bot")


async def _claude_sdk_on_started(adapter: Any, config: AdapterConfig) -> None:
    with patch("thenvoi.adapters.claude_sdk.ClaudeSessionManager") as mock_manager:
        mock_manager.return_value = MagicMock()
        await adapter.on_started(agent_name="TestBot", agent_description="A test bot")


async def _pydantic_ai_on_started(adapter: Any, config: AdapterConfig) -> None:
    with patch.object(adapter, "_create_agent") as mock_create:
        mock_agent = MagicMock()
        mock_agent._function_tools = {}
        mock_create.return_value = mock_agent
        await adapter.on_started(agent_name="TestBot", agent_description="A test bot")


async def _parlant_on_started(adapter: Any, config: AdapterConfig) -> None:
    mock_app = MagicMock()
    mock_application_class = MagicMock(name="Application")
    mock_module = MagicMock()
    mock_module.Application = mock_application_class
    adapter._server.container = {mock_application_class: mock_app}

    with patch.dict(sys.modules, {"parlant.core.application": mock_module}):
        await adapter.on_started(agent_name="TestBot", agent_description="A test bot")


# =============================================================================
# Mock LLM Callbacks
# =============================================================================


def _anthropic_mock_llm(
    adapter: Any, mocks: dict | None, captured_input: dict | None
) -> Any:
    @asynccontextmanager
    async def ctx() -> AsyncIterator[None]:
        mock_response = MagicMock(stop_reason="end_turn", content=[])
        with patch.object(adapter, "_call_anthropic", return_value=mock_response):
            yield

    return ctx()


def _langgraph_mock_llm(
    adapter: Any, mocks: dict | None, captured_input: dict | None
) -> Any:
    @asynccontextmanager
    async def ctx() -> AsyncIterator[None]:
        async def capture_stream(
            graph_input: dict, **kwargs: Any
        ) -> AsyncIterator[Any]:
            if captured_input is not None:
                captured_input.update(graph_input)
            if False:
                yield

        mock_graph = MagicMock()
        mock_graph.astream_events = capture_stream
        adapter.graph_factory = MagicMock(return_value=mock_graph)

        with patch(
            "thenvoi.integrations.langgraph.langchain_tools.agent_tools_to_langchain"
        ) as mock_convert:
            mock_convert.return_value = []
            yield

    return ctx()


def _crewai_mock_llm(
    adapter: Any, mocks: dict | None, captured_input: dict | None
) -> Any:
    @asynccontextmanager
    async def ctx() -> AsyncIterator[None]:
        if adapter._crewai_agent is not None:
            mock_result = MagicMock(raw="Test response")
            adapter._crewai_agent.kickoff_async = AsyncMock(return_value=mock_result)
        yield

    return ctx()


def _claude_sdk_mock_llm(
    adapter: Any, mocks: dict | None, captured_input: dict | None
) -> Any:
    @asynccontextmanager
    async def ctx() -> AsyncIterator[None]:
        mock_client = AsyncMock()
        mock_client.query = AsyncMock(return_value=None)
        if adapter._session_manager:
            adapter._session_manager.get_or_create_session = AsyncMock(
                return_value=mock_client
            )
        with patch.object(adapter, "_process_response", return_value=None):
            yield

    return ctx()


def _pydantic_ai_mock_llm(
    adapter: Any, mocks: dict | None, captured_input: dict | None
) -> Any:
    from pydantic_ai import AgentRunResultEvent
    from pydantic_ai.messages import ModelRequest, UserPromptPart

    @asynccontextmanager
    async def ctx() -> AsyncIterator[None]:
        async def make_stream() -> AsyncIterator[Any]:
            result_event = MagicMock(spec=AgentRunResultEvent)
            result_event.result = MagicMock()
            result_event.result.all_messages.return_value = [
                ModelRequest(parts=[UserPromptPart(content="test")])
            ]
            yield result_event

        if mocks and "agent" in mocks:
            mocks["agent"].run_stream_events = MagicMock(return_value=make_stream())
        yield

    return ctx()


def _parlant_mock_llm(
    adapter: Any, mocks: dict | None, captured_input: dict | None
) -> Any:
    @asynccontextmanager
    async def ctx() -> AsyncIterator[None]:
        mock_moderation = MagicMock(NONE="none")
        mock_event_source = MagicMock(CUSTOMER="customer", AI_AGENT="ai_agent")
        mock_event_kind = MagicMock(MESSAGE="message")

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

    return ctx()


# =============================================================================
# Error Setup Callbacks
# =============================================================================


def _anthropic_error_setup(adapter: Any, mocks: dict | None) -> Any:
    @asynccontextmanager
    async def ctx() -> AsyncIterator[str]:
        with patch.object(
            adapter, "_call_anthropic", side_effect=Exception("API Error")
        ):
            yield "API Error"

    return ctx()


def _langgraph_error_setup(adapter: Any, mocks: dict | None) -> Any:
    @asynccontextmanager
    async def ctx() -> AsyncIterator[str]:
        async def failing_stream(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
            raise Exception("Graph error!")
            if False:
                yield

        mock_graph = MagicMock()
        mock_graph.astream_events = failing_stream
        adapter.graph_factory = MagicMock(return_value=mock_graph)

        with patch(
            "thenvoi.integrations.langgraph.langchain_tools.agent_tools_to_langchain"
        ) as mock_convert:
            mock_convert.return_value = []
            yield "Graph error!"

    return ctx()


def _crewai_error_setup(adapter: Any, mocks: dict | None) -> Any:
    @asynccontextmanager
    async def ctx() -> AsyncIterator[str]:
        if adapter._crewai_agent is not None:
            adapter._crewai_agent.kickoff_async = AsyncMock(
                side_effect=Exception("CrewAI Error")
            )
        yield "CrewAI Error"

    return ctx()


def _claude_sdk_error_setup(adapter: Any, mocks: dict | None) -> Any:
    @asynccontextmanager
    async def ctx() -> AsyncIterator[str]:
        mock_client = AsyncMock()
        mock_client.query = AsyncMock(side_effect=Exception("Claude SDK Error"))
        adapter._session_manager.get_or_create_session = AsyncMock(
            return_value=mock_client
        )
        yield "Claude SDK Error"

    return ctx()


def _pydantic_ai_error_setup(adapter: Any, mocks: dict | None) -> Any:
    @asynccontextmanager
    async def ctx() -> AsyncIterator[str]:
        async def failing_stream() -> AsyncIterator[Any]:
            raise Exception("PydanticAI Error")
            if False:
                yield

        if mocks and "agent" in mocks:
            mocks["agent"].run_stream_events = MagicMock(return_value=failing_stream())
        yield "PydanticAI Error"

    return ctx()


def _parlant_error_setup(adapter: Any, mocks: dict | None) -> Any:
    @asynccontextmanager
    async def ctx() -> AsyncIterator[str]:
        adapter._app.sessions.create_customer_message = AsyncMock(
            side_effect=Exception("Parlant Error")
        )
        yield "Parlant Error"

    return ctx()


# =============================================================================
# Participants Verification
# =============================================================================


def _verify_anthropic_participants(
    adapter: Any, captured_input: dict, participants_msg: str
) -> bool:
    return any(
        "[System]: Alice joined" in str(m.get("content", ""))
        for m in adapter._message_history.get("room-123", [])
    )


def _verify_langgraph_participants(
    adapter: Any, captured_input: dict, participants_msg: str
) -> bool:
    messages = captured_input.get("messages", [])
    system_msgs = [m for m in messages if m[0] == "system"]
    return any("Alice joined" in str(m[1]) for m in system_msgs)


def _verify_pydantic_ai_participants(
    adapter: Any, captured_input: dict, participants_msg: str, mocks: dict | None = None
) -> bool:
    if mocks and "agent" in mocks:
        call_args = mocks["agent"].run_stream_events.call_args
        if call_args:
            message_history = call_args.kwargs.get("message_history", [])
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
    return "room-123" in adapter._message_history


def _always_true(adapter: Any, captured_input: dict, participants_msg: str) -> bool:
    """Trivial verifier for adapters that handle participants internally."""
    return True


# =============================================================================
# Adapter Configurations
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
        on_started_callback=_simple_on_started,
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
        verify_participants_injection=_always_true,
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
        on_started_callback=_simple_on_started,
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
        verify_participants_injection=_always_true,
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


# =============================================================================
# On-Message Setup Helper
# =============================================================================


async def setup_adapter_for_on_message(
    adapter: Any, config: AdapterConfig, mock_tools: Any
) -> Any:
    """Set up adapter for on_message testing. Returns mocks needed for assertions."""
    if config.name == "parlant":
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
        adapter._app = mock_app
        return {"app": mock_app}

    if config.name == "claude_sdk":
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

    if config.name == "pydantic_ai":
        mock_agent = MagicMock()
        mock_agent._function_tools = {"thenvoi_send_message": MagicMock()}
        with patch.object(adapter, "_create_agent", return_value=mock_agent):
            await adapter.on_started(
                agent_name="TestBot", agent_description="A test bot"
            )
        adapter._agent = mock_agent
        return {"agent": mock_agent}

    await adapter.on_started(agent_name="TestBot", agent_description="A test bot")
    return None
