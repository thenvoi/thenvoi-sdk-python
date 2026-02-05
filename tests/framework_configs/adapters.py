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
    ),
}

# CrewAI config - requires special handling
CREWAI_CONFIG = AdapterConfig(
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
)
