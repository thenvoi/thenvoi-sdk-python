"""Built-in framework adapters.

Adapters are lazily imported to avoid requiring all optional dependencies.
Install the extra you need:
    uv add thenvoi-sdk[langgraph]
    uv add thenvoi-sdk[anthropic]
    uv add thenvoi-sdk[pydantic_ai]
    uv add thenvoi-sdk[claude_sdk]
    uv add thenvoi-sdk[parlant]
"""

from typing import TYPE_CHECKING

# Type-only imports for static analysis (pyrefly, mypy, etc.)
if TYPE_CHECKING:
    from thenvoi.adapters.langgraph import LangGraphAdapter as LangGraphAdapter
    from thenvoi.adapters.anthropic import AnthropicAdapter as AnthropicAdapter
    from thenvoi.adapters.pydantic_ai import PydanticAIAdapter as PydanticAIAdapter
    from thenvoi.adapters.claude_sdk import ClaudeSDKAdapter as ClaudeSDKAdapter
    from thenvoi.adapters.parlant import ParlantAdapter as ParlantAdapter

__all__ = [
    "LangGraphAdapter",
    "AnthropicAdapter",
    "PydanticAIAdapter",
    "ClaudeSDKAdapter",
    "ParlantAdapter",
]


def __getattr__(name: str):
    """Lazy import adapters to avoid loading optional dependencies."""
    if name == "LangGraphAdapter":
        from thenvoi.adapters.langgraph import LangGraphAdapter

        return LangGraphAdapter
    elif name == "AnthropicAdapter":
        from thenvoi.adapters.anthropic import AnthropicAdapter

        return AnthropicAdapter
    elif name == "PydanticAIAdapter":
        from thenvoi.adapters.pydantic_ai import PydanticAIAdapter

        return PydanticAIAdapter
    elif name == "ClaudeSDKAdapter":
        from thenvoi.adapters.claude_sdk import ClaudeSDKAdapter

        return ClaudeSDKAdapter
    elif name == "ParlantAdapter":
        from thenvoi.adapters.parlant import ParlantAdapter

        return ParlantAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
