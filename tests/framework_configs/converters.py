"""Converter configuration registry for parameterized conformance tests.

Each ConverterConfig describes a framework's converter properties, behavioral
flags, and factory function so that conformance tests can run identical logic
across all registered converters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable

__all__ = ["ConverterConfig", "CONVERTER_CONFIGS", "SenderBehavior"]

from tests.framework_configs._output_adapters import (
    DictListOutputAdapter,
    LangChainOutputAdapter,
    OutputAdapter,
    PydanticAIOutputAdapter,
    SimpleDictListOutputAdapter,
    StringOutputAdapter,
)


class SenderBehavior(StrEnum):
    """How a converter handles empty or missing sender_name."""

    CONTENT_AS_IS = "content_as_is"  # content returned without prefix
    BRACKETS_EMPTY = "brackets_empty"  # "[]: content"
    UNKNOWN_PREFIX = "unknown_prefix"  # "[Unknown]: content"


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

    # How empty/missing sender_name is handled
    empty_sender_behavior: SenderBehavior = SenderBehavior.CONTENT_AS_IS
    missing_sender_behavior: SenderBehavior = SenderBehavior.CONTENT_AS_IS

    # Edge case flags
    skips_empty_content: bool = False
    has_role_concept: bool = True

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
        empty_sender_behavior=SenderBehavior.CONTENT_AS_IS,
        missing_sender_behavior=SenderBehavior.CONTENT_AS_IS,
        output_adapter=DictListOutputAdapter(),
    ),
    ConverterConfig(
        framework_id="langchain",
        display_name="LangChain",
        converter_factory=_langchain_factory,
        return_type="list",
        empty_result=[],
        empty_sender_behavior=SenderBehavior.BRACKETS_EMPTY,
        # LangChain uses hist.get("sender_name", ""), so a *missing* key
        # produces the same "[]: content" as an empty string (brackets_empty).
        missing_sender_behavior=SenderBehavior.BRACKETS_EMPTY,
        output_adapter=LangChainOutputAdapter(),
    ),
    ConverterConfig(
        framework_id="crewai",
        display_name="CrewAI",
        converter_factory=_crewai_factory,
        return_type="list",
        empty_result=[],
        skips_tool_events=True,
        empty_sender_behavior=SenderBehavior.CONTENT_AS_IS,
        missing_sender_behavior=SenderBehavior.CONTENT_AS_IS,
        output_adapter=SimpleDictListOutputAdapter(),
    ),
    ConverterConfig(
        framework_id="claude_sdk",
        display_name="ClaudeSDK",
        converter_factory=_claude_sdk_factory,
        return_type="string",
        empty_result="",
        empty_sender_behavior=SenderBehavior.BRACKETS_EMPTY,
        missing_sender_behavior=SenderBehavior.UNKNOWN_PREFIX,
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
        empty_sender_behavior=SenderBehavior.CONTENT_AS_IS,
        missing_sender_behavior=SenderBehavior.CONTENT_AS_IS,
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
        empty_sender_behavior=SenderBehavior.CONTENT_AS_IS,
        missing_sender_behavior=SenderBehavior.CONTENT_AS_IS,
        output_adapter=SimpleDictListOutputAdapter(),
    ),
]
