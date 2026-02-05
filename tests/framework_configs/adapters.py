"""Framework adapter configurations for parameterized contract tests.

This module defines the configuration for each adapter framework, allowing
contract tests to run the same test logic across all adapters while handling
framework-specific behaviors through configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable
from unittest.mock import AsyncMock, MagicMock

from thenvoi.core.types import PlatformMessage


@dataclass
class AdapterConfig:
    """Configuration for a framework's adapter.

    Attributes:
        name: Human-readable framework name (used as test ID)
        adapter_class: The adapter class to instantiate
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
        on_message_requires_setup: Setup function for on_message tests
        custom_tool_format: "tuple" for (Model, func), "callable" for just func
        system_prompt_attr: Attribute name for system prompt (or alternative like backstory)
        system_prompt_contains_name: Whether system prompt should contain agent name
        alternative_prompt_attr: Alternative attribute for prompt (e.g., backstory for crewai)
        error_trigger_method: Method name to patch for error testing
        cleanup_storage_attrs: List of storage attributes to verify cleanup
    """

    name: str
    adapter_class: type | None  # May be None if requires_mocks
    factory: Callable[..., Any]
    requires_mocks: bool = False
    mock_setup: Callable[[], Any] | None = None
    has_history_converter: bool = True
    has_custom_tools: bool = True
    custom_tools_attr: str = "_custom_tools"
    default_model: str | None = None
    additional_init_checks: dict[str, Any] = field(default_factory=dict)
    supports_enable_execution_reporting: bool = True
    history_storage_attr: str | None = "_message_history"
    supports_system_prompt_override: bool = True
    supports_cleanup_all: bool = False
    custom_tool_format: str = "tuple"  # "tuple" or "callable"
    # System prompt configuration
    system_prompt_attr: str | None = "_system_prompt"
    system_prompt_contains_name: bool = True
    alternative_prompt_attr: str | None = None  # e.g., "backstory" for crewai
    # Error handling configuration
    error_trigger_method: str | None = None  # Method to patch for error testing
    # Cleanup configuration
    cleanup_storage_attrs: list[str] = field(default_factory=list)


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


# Adapter configurations
# Note: CrewAI requires special handling due to module-level mocking


def _get_anthropic_adapter_class() -> type:
    from thenvoi.adapters.anthropic import AnthropicAdapter

    return AnthropicAdapter


def _get_claude_sdk_adapter_class() -> type:
    from thenvoi.adapters.claude_sdk import ClaudeSDKAdapter

    return ClaudeSDKAdapter


def _get_langgraph_adapter_class() -> type:
    from thenvoi.adapters.langgraph import LangGraphAdapter

    return LangGraphAdapter


def _get_pydantic_ai_adapter_class() -> type:
    from thenvoi.adapters.pydantic_ai import PydanticAIAdapter

    return PydanticAIAdapter


def _get_parlant_adapter_class() -> type:
    from thenvoi.adapters.parlant import ParlantAdapter

    return ParlantAdapter


# Configs that don't require special mocking
ADAPTER_CONFIGS: dict[str, AdapterConfig] = {
    "anthropic": AdapterConfig(
        name="anthropic",
        adapter_class=None,  # Lazy load
        factory=_create_anthropic_adapter,
        has_history_converter=True,
        has_custom_tools=True,
        custom_tools_attr="_custom_tools",
        default_model="claude-sonnet-4-5-20250929",
        additional_init_checks={
            "max_tokens": 4096,
            "enable_execution_reporting": False,
        },
        supports_enable_execution_reporting=True,
        history_storage_attr="_message_history",
        supports_system_prompt_override=True,
        supports_cleanup_all=False,
        custom_tool_format="tuple",
        system_prompt_attr="_system_prompt",
        system_prompt_contains_name=True,
        error_trigger_method="_call_anthropic",
        cleanup_storage_attrs=["_message_history"],
    ),
    "claude_sdk": AdapterConfig(
        name="claude_sdk",
        adapter_class=None,
        factory=_create_claude_sdk_adapter,
        has_history_converter=False,  # ClaudeSDK uses session manager
        has_custom_tools=True,
        custom_tools_attr="_custom_tools",
        default_model="claude-sonnet-4-5-20250929",
        additional_init_checks={
            "permission_mode": "acceptEdits",
            "enable_execution_reporting": False,
        },
        supports_enable_execution_reporting=True,
        history_storage_attr="_room_tools",  # ClaudeSDK uses _room_tools
        supports_system_prompt_override=False,  # Uses session manager
        supports_cleanup_all=True,
        custom_tool_format="tuple",
        system_prompt_attr=None,  # Uses session manager
        system_prompt_contains_name=False,
        error_trigger_method=None,  # Complex session-based error handling
        cleanup_storage_attrs=["_room_tools"],
    ),
    "langgraph": AdapterConfig(
        name="langgraph",
        adapter_class=None,
        factory=_create_langgraph_adapter,
        has_history_converter=True,
        has_custom_tools=True,
        custom_tools_attr="additional_tools",  # LangGraph clears this after baking
        default_model=None,  # LangGraph uses llm instead of model
        additional_init_checks={"prompt_template": "default"},
        supports_enable_execution_reporting=False,  # LangGraph doesn't support this
        history_storage_attr=None,  # LangGraph uses checkpointer
        supports_system_prompt_override=False,  # Uses prompt_template
        supports_cleanup_all=False,
        custom_tool_format="callable",
        system_prompt_attr="_system_prompt",
        system_prompt_contains_name=True,
        error_trigger_method="graph_factory",  # Mock graph to raise error
        cleanup_storage_attrs=[],
    ),
    "pydantic_ai": AdapterConfig(
        name="pydantic_ai",
        adapter_class=None,
        factory=_create_pydantic_ai_adapter,
        has_history_converter=True,
        has_custom_tools=True,
        custom_tools_attr="_custom_tools",
        default_model=None,  # Model is required, no default
        additional_init_checks={"enable_execution_reporting": False},
        supports_enable_execution_reporting=True,
        history_storage_attr="_message_history",
        supports_system_prompt_override=False,  # Uses agent-level system
        supports_cleanup_all=False,
        custom_tool_format="callable",
        system_prompt_attr=None,  # Uses agent._system_prompt internally
        system_prompt_contains_name=False,
        error_trigger_method="_agent.run_stream_events",
        cleanup_storage_attrs=["_message_history"],
    ),
    "parlant": AdapterConfig(
        name="parlant",
        adapter_class=None,
        factory=_create_parlant_adapter,
        has_history_converter=True,
        has_custom_tools=False,  # Parlant uses Parlant SDK tools
        custom_tools_attr="_custom_tools",
        default_model=None,  # Parlant uses Parlant SDK
        additional_init_checks={},
        supports_enable_execution_reporting=False,  # Parlant doesn't support this
        history_storage_attr="_room_sessions",  # Parlant uses sessions
        supports_system_prompt_override=True,
        supports_cleanup_all=True,
        custom_tool_format="tuple",
        system_prompt_attr="_system_prompt",
        system_prompt_contains_name=True,
        error_trigger_method=None,  # Complex Parlant SDK error handling
        cleanup_storage_attrs=["_room_sessions", "_room_customers"],
    ),
}

# CrewAI config - added to ADAPTER_CONFIGS with requires_mocks=True
ADAPTER_CONFIGS["crewai"] = AdapterConfig(
    name="crewai",
    adapter_class=None,
    factory=_create_crewai_adapter,
    requires_mocks=True,
    has_history_converter=True,
    has_custom_tools=True,
    custom_tools_attr="_custom_tools",
    default_model="gpt-4o",
    additional_init_checks={
        "verbose": False,
        "max_iter": 20,
        "allow_delegation": False,
        "enable_execution_reporting": False,
    },
    supports_enable_execution_reporting=True,
    history_storage_attr="_message_history",
    supports_system_prompt_override=False,  # Uses backstory
    supports_cleanup_all=False,
    custom_tool_format="tuple",
    system_prompt_attr=None,  # Uses backstory in CrewAI agent
    system_prompt_contains_name=False,
    alternative_prompt_attr="backstory",  # CrewAI uses backstory
    error_trigger_method="_crewai_agent.kickoff_async",
    cleanup_storage_attrs=["_message_history"],
)

# Keep CREWAI_CONFIG for backwards compatibility
CREWAI_CONFIG = ADAPTER_CONFIGS["crewai"]


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
