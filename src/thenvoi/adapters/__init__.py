"""Built-in framework adapters.

Adapters are lazily imported to avoid requiring all optional dependencies.
Install the extra you need:
    uv add thenvoi-sdk[langgraph]
    uv add thenvoi-sdk[anthropic]
    uv add thenvoi-sdk[pydantic_ai]
    uv add thenvoi-sdk[claude_sdk]
    uv add thenvoi-sdk[parlant]
    uv add thenvoi-sdk[crewai]
    uv add thenvoi-sdk[a2a]
"""

from typing import TYPE_CHECKING

# Type-only imports for static analysis (pyrefly, mypy, etc.)
if TYPE_CHECKING:
    from thenvoi.adapters.langgraph import LangGraphAdapter as LangGraphAdapter
    from thenvoi.adapters.anthropic import AnthropicAdapter as AnthropicAdapter
    from thenvoi.adapters.pydantic_ai import PydanticAIAdapter as PydanticAIAdapter
    from thenvoi.adapters.claude_sdk import ClaudeSDKAdapter as ClaudeSDKAdapter
    from thenvoi.adapters.parlant import ParlantAdapter as ParlantAdapter
    from thenvoi.adapters.crewai import CrewAIAdapter as CrewAIAdapter
    from thenvoi.adapters.a2a import A2AAdapter as A2AAdapter

__all__ = [
    "LangGraphAdapter",
    "AnthropicAdapter",
    "PydanticAIAdapter",
    "ClaudeSDKAdapter",
    "ParlantAdapter",
    "CrewAIAdapter",
    "A2AAdapter",
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
    elif name == "CrewAIAdapter":
        from thenvoi.adapters.crewai import CrewAIAdapter

        return CrewAIAdapter
    elif name == "A2AAdapter":
        from thenvoi.adapters.a2a import A2AAdapter

        return A2AAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
