"""
Smoke tests - verify basic imports and setup work.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

from thenvoi import (
    Agent,
    AgentConfig,
    ConversationContext,
    PlatformEvent,
    PlatformMessage,
)


def test_can_import_public_root_api() -> None:
    """Verify the intended public root imports still work."""
    assert Agent is not None
    assert PlatformEvent is not None
    assert PlatformMessage is not None
    assert AgentConfig is not None
    assert ConversationContext is not None


def test_runtime_symbols_remain_importable_from_runtime() -> None:
    """Verify removed root exports still exist on thenvoi.runtime."""
    from thenvoi.runtime import AgentRuntime, ExecutionContext

    assert AgentRuntime is not None
    assert ExecutionContext is not None


def test_typed_config_is_importable_from_thenvoi_config() -> None:
    """Verify config-driven API types are importable."""
    from thenvoi.config import AgentConfig as ConfigAgentConfig

    assert ConfigAgentConfig is not None


def test_can_import_letta_adapter_from_lazy_surface() -> None:
    """Verify Letta lazy exports are available."""
    from thenvoi.adapters import LettaAdapter, LettaAdapterConfig

    assert LettaAdapter is not None
    assert LettaAdapterConfig is not None


def test_claude_sdk_adapter_import_guard_message() -> None:
    """Verify lazy Claude SDK import points users to the extra."""
    modules_to_clear = [
        "thenvoi.adapters.claude_sdk",
        "thenvoi.integrations.claude_sdk.prompts",
        "thenvoi.integrations.claude_sdk.session_manager",
        "thenvoi.integrations.claude_sdk.tools",
    ]
    saved_modules = {name: sys.modules.get(name) for name in modules_to_clear}
    for name in modules_to_clear:
        sys.modules.pop(name, None)

    mocked_sys_modules = {
        "claude_agent_sdk": None,
        "claude_agent_sdk._errors": None,
        "claude_agent_sdk.types": None,
        "mcp": MagicMock(),
        "mcp.server.lowlevel": MagicMock(),
        "mcp.server.sse": MagicMock(),
        "mcp.types": MagicMock(),
    }

    try:
        with patch.dict(sys.modules, mocked_sys_modules):
            try:
                getattr(
                    __import__("thenvoi.adapters", fromlist=["ClaudeSDKAdapter"]),
                    "ClaudeSDKAdapter",
                )
            except ImportError as exc:
                assert str(exc) == (
                    "claude-agent-sdk is required for ClaudeSDKAdapter. "
                    "Install with: pip install thenvoi-sdk[claude_sdk] "
                    "or uv add thenvoi-sdk[claude_sdk]"
                )
            else:
                raise AssertionError(
                    "Expected ImportError when claude-agent-sdk is missing"
                )
    finally:
        for name, module in saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_can_import_langgraph_integrations() -> None:
    """Verify we can import LangGraph integration utilities."""
    from thenvoi.integrations.langgraph import (
        agent_tools_to_langchain,
        graph_as_tool,
    )

    assert agent_tools_to_langchain is not None
    assert graph_as_tool is not None


def test_fixtures_work(mock_api_client, mock_websocket, sample_room_message) -> None:
    """Verify our test fixtures are properly configured."""
    assert mock_api_client is not None
    assert mock_websocket is not None
    assert sample_room_message.chat_room_id == "room-123"
    assert sample_room_message.sender_type == "User"
