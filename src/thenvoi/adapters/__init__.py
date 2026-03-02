"""Built-in framework adapters.

Adapters are lazily imported to avoid requiring all optional dependencies.
Install the extra you need::

    uv add thenvoi-sdk[langgraph]
    uv add thenvoi-sdk[anthropic]
    uv add thenvoi-sdk[pydantic_ai]
    uv add thenvoi-sdk[claude_sdk]
    uv add thenvoi-sdk[parlant]
    uv add thenvoi-sdk[crewai]
    uv add thenvoi-sdk[a2a]
    uv add thenvoi-sdk[a2a_gateway]
    uv add thenvoi-sdk[codex]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from thenvoi.adapters.codex.adapter import (
    CodexAdapter as CodexAdapter,
    CodexAdapterConfig as CodexAdapterConfig,
)
from thenvoi.adapters.crewai import (
    CrewAIAdapter as CrewAIAdapter,
    CrewAIAdapterConfig as CrewAIAdapterConfig,
)

# Type-only imports for static analysis (pyrefly, mypy, etc.)
if TYPE_CHECKING:
    from thenvoi.adapters.langgraph import LangGraphAdapter as LangGraphAdapter
    from thenvoi.adapters.anthropic import (
        AnthropicAdapter as AnthropicAdapter,
        AnthropicAdapterConfig as AnthropicAdapterConfig,
    )
    from thenvoi.adapters.pydantic_ai import (
        PydanticAIAdapter as PydanticAIAdapter,
        PydanticAIAdapterConfig as PydanticAIAdapterConfig,
    )
    from thenvoi.adapters.claude_sdk import (
        ClaudeSDKAdapter as ClaudeSDKAdapter,
        ClaudeSDKAdapterConfig as ClaudeSDKAdapterConfig,
    )
    from thenvoi.adapters.parlant import ParlantAdapter as ParlantAdapter
    from thenvoi.adapters.crewai import CrewAIAdapter as CrewAIAdapter
    from thenvoi.adapters.crewai import CrewAIAdapterConfig as CrewAIAdapterConfig
    from thenvoi.integrations.a2a.adapter import A2AAdapter as A2AAdapter
    from thenvoi.integrations.a2a.gateway.adapter import (
        A2AGatewayAdapter as A2AGatewayAdapter,
    )
    from thenvoi.adapters.codex.adapter import CodexAdapter as CodexAdapter
    from thenvoi.adapters.codex.adapter import CodexAdapterConfig as CodexAdapterConfig

__all__ = [
    "LangGraphAdapter",
    "AnthropicAdapter",
    "AnthropicAdapterConfig",
    "PydanticAIAdapter",
    "PydanticAIAdapterConfig",
    "ClaudeSDKAdapter",
    "ClaudeSDKAdapterConfig",
    "ParlantAdapter",
    "CrewAIAdapter",
    "CrewAIAdapterConfig",
    "A2AAdapter",
    "A2AGatewayAdapter",
    "CodexAdapter",
    "CodexAdapterConfig",
]


def __getattr__(name: str) -> type:
    """Lazy import adapters to avoid loading optional dependencies."""
    if name == "LangGraphAdapter":
        from thenvoi.adapters.langgraph import LangGraphAdapter

        return LangGraphAdapter
    elif name == "AnthropicAdapter":
        from thenvoi.adapters.anthropic import AnthropicAdapter

        return AnthropicAdapter
    elif name == "AnthropicAdapterConfig":
        from thenvoi.adapters.anthropic import AnthropicAdapterConfig

        return AnthropicAdapterConfig
    elif name == "PydanticAIAdapter":
        from thenvoi.adapters.pydantic_ai import PydanticAIAdapter

        return PydanticAIAdapter
    elif name == "PydanticAIAdapterConfig":
        from thenvoi.adapters.pydantic_ai import PydanticAIAdapterConfig

        return PydanticAIAdapterConfig
    elif name == "ClaudeSDKAdapter":
        from thenvoi.adapters.claude_sdk import ClaudeSDKAdapter

        return ClaudeSDKAdapter
    elif name == "ClaudeSDKAdapterConfig":
        from thenvoi.adapters.claude_sdk import ClaudeSDKAdapterConfig

        return ClaudeSDKAdapterConfig
    elif name == "ParlantAdapter":
        from thenvoi.adapters.parlant import ParlantAdapter

        return ParlantAdapter
    elif name == "CrewAIAdapter":
        return CrewAIAdapter
    elif name == "CrewAIAdapterConfig":
        return CrewAIAdapterConfig
    elif name == "A2AAdapter":
        from thenvoi.integrations.a2a.adapter import A2AAdapter

        return A2AAdapter
    elif name == "A2AGatewayAdapter":
        from thenvoi.integrations.a2a.gateway.adapter import A2AGatewayAdapter

        return A2AGatewayAdapter
    elif name == "CodexAdapter":
        return CodexAdapter
    elif name == "CodexAdapterConfig":
        return CodexAdapterConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
