"""Converter configuration registry for parameterized conformance tests.

Each ConverterConfig describes a framework's converter properties, behavioral
flags, and factory function so that conformance tests can run identical logic
across all six converters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from tests.framework_configs._output_adapters import (
    DictListOutputAdapter,
    LangChainOutputAdapter,
    OutputAdapter,
    PydanticAIOutputAdapter,
    SimpleDictListOutputAdapter,
    StringOutputAdapter,
)


@dataclass(frozen=True)
class ConverterConfig:
    """Describes a framework converter for parameterized testing."""

    # Identity
    framework_id: str
    display_name: str

    # Factory callable: (**kwargs) -> converter instance
    converter_factory: Callable[..., Any]

    # Output shape
    return_type: str  # "list" or "string"
    empty_result: Any  # [] or ""

    # Behavioral flags
    filters_own_messages: bool = True
    skips_tool_events: bool = False

    # How empty/missing sender_name is handled:
    #   "content_as_is"   -> content returned without prefix
    #   "brackets_empty"  -> "[]: content"
    #   "unknown_prefix"  -> "[Unknown]: content"
    empty_sender_behavior: str = "content_as_is"
    missing_sender_behavior: str = "content_as_is"

    # Edge case flags
    skips_empty_content: bool = False
    has_role_concept: bool = True
    has_missing_sender_name_test: bool = True

    # Output adapter for uniform assertions
    output_adapter: OutputAdapter = field(default_factory=DictListOutputAdapter)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def _anthropic_factory(**kw: Any) -> Any:
    from thenvoi.converters.anthropic import AnthropicHistoryConverter

    return AnthropicHistoryConverter(**kw)


def _langchain_factory(**kw: Any) -> Any:
    from thenvoi.converters.langchain import LangChainHistoryConverter

    return LangChainHistoryConverter(**kw)


def _crewai_factory(**kw: Any) -> Any:
    from thenvoi.converters.crewai import CrewAIHistoryConverter

    return CrewAIHistoryConverter(**kw)


def _claude_sdk_factory(**kw: Any) -> Any:
    from thenvoi.converters.claude_sdk import ClaudeSDKHistoryConverter

    return ClaudeSDKHistoryConverter(**kw)


def _pydantic_ai_factory(**kw: Any) -> Any:
    from thenvoi.converters.pydantic_ai import PydanticAIHistoryConverter

    return PydanticAIHistoryConverter(**kw)


def _parlant_factory(**kw: Any) -> Any:
    from thenvoi.converters.parlant import ParlantHistoryConverter

    return ParlantHistoryConverter(**kw)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

CONVERTER_CONFIGS: list[ConverterConfig] = [
    ConverterConfig(
        framework_id="anthropic",
        display_name="Anthropic",
        converter_factory=_anthropic_factory,
        return_type="list",
        empty_result=[],
        empty_sender_behavior="content_as_is",
        missing_sender_behavior="content_as_is",
        output_adapter=DictListOutputAdapter(),
    ),
    ConverterConfig(
        framework_id="langchain",
        display_name="LangChain",
        converter_factory=_langchain_factory,
        return_type="list",
        empty_result=[],
        empty_sender_behavior="brackets_empty",
        # LangChain uses hist.get("sender_name", ""), so a *missing* key
        # produces the same "[]: content" as an empty string (brackets_empty).
        # This is identical to empty_sender_behavior and doesn't map cleanly
        # to "content_as_is" or "unknown_prefix", so the test is skipped.
        missing_sender_behavior="brackets_empty",
        has_missing_sender_name_test=False,
        output_adapter=LangChainOutputAdapter(),
    ),
    ConverterConfig(
        framework_id="crewai",
        display_name="CrewAI",
        converter_factory=_crewai_factory,
        return_type="list",
        empty_result=[],
        skips_tool_events=True,
        empty_sender_behavior="content_as_is",
        missing_sender_behavior="content_as_is",
        output_adapter=SimpleDictListOutputAdapter(),
    ),
    ConverterConfig(
        framework_id="claude_sdk",
        display_name="ClaudeSDK",
        converter_factory=_claude_sdk_factory,
        return_type="string",
        empty_result="",
        empty_sender_behavior="brackets_empty",
        missing_sender_behavior="unknown_prefix",
        skips_empty_content=True,
        has_role_concept=False,
        output_adapter=StringOutputAdapter(),
    ),
    ConverterConfig(
        framework_id="pydantic_ai",
        display_name="PydanticAI",
        converter_factory=_pydantic_ai_factory,
        return_type="list",
        empty_result=[],
        empty_sender_behavior="content_as_is",
        missing_sender_behavior="content_as_is",
        output_adapter=PydanticAIOutputAdapter(),
    ),
    ConverterConfig(
        framework_id="parlant",
        display_name="Parlant",
        converter_factory=_parlant_factory,
        return_type="list",
        empty_result=[],
        filters_own_messages=False,
        skips_tool_events=True,
        skips_empty_content=True,
        empty_sender_behavior="content_as_is",
        missing_sender_behavior="content_as_is",
        output_adapter=SimpleDictListOutputAdapter(),
    ),
]
