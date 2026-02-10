"""Converter configuration registry for parameterized conformance tests.

Each ConverterConfig describes a framework's converter properties, behavioral
flags, and factory function so that conformance tests can run identical logic
across all registered converters.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable

__all__ = ["ConverterConfig", "CONVERTER_CONFIGS", "SenderBehavior"]

from tests.framework_configs.output_adapters import (
    DictListOutputAdapter,
    LangChainOutputAdapter,
    OutputAdapter,
    PydanticAIOutputAdapter,
    SenderMetadataDictListOutputAdapter,
    StringOutputAdapter,
)

# Populated lazily via __getattr__ to avoid top-level converter imports.
CONVERTER_CONFIGS: list[ConverterConfig]


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

    # Output shape flags
    has_sender_metadata: bool = False  # output includes sender/sender_type fields
    other_agent_output_role: str = "user"  # role assigned to other agents' messages

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
# Registry  (built lazily to avoid top-level converter imports)
# ---------------------------------------------------------------------------


@functools.cache
def _build_converter_configs() -> list[ConverterConfig]:
    """Build configs lazily so converter imports happen only when the
    conformance tests actually need them."""
    return [
        ConverterConfig(
            framework_id="anthropic",
            display_name="Anthropic",
            converter_factory=_anthropic_factory,
            empty_result=[],
            empty_sender_behavior=SenderBehavior.CONTENT_AS_IS,
            missing_sender_behavior=SenderBehavior.CONTENT_AS_IS,
            output_adapter=DictListOutputAdapter(),
        ),
        ConverterConfig(
            framework_id="langchain",
            display_name="LangChain",
            converter_factory=_langchain_factory,
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
            empty_result=[],
            skips_tool_events=True,
            empty_sender_behavior=SenderBehavior.CONTENT_AS_IS,
            missing_sender_behavior=SenderBehavior.CONTENT_AS_IS,
            has_sender_metadata=True,
            # CrewAI treats other agents as peers (assistant role) rather than
            # remapping them to user, because its crew workflow expects all agent
            # outputs to carry the "assistant" role.
            other_agent_output_role="assistant",
            output_adapter=SenderMetadataDictListOutputAdapter(),
        ),
        ConverterConfig(
            framework_id="claude_sdk",
            display_name="ClaudeSDK",
            converter_factory=_claude_sdk_factory,
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
            empty_result=[],
            empty_sender_behavior=SenderBehavior.CONTENT_AS_IS,
            missing_sender_behavior=SenderBehavior.CONTENT_AS_IS,
            output_adapter=PydanticAIOutputAdapter(),
        ),
        ConverterConfig(
            framework_id="parlant",
            display_name="Parlant",
            converter_factory=_parlant_factory,
            empty_result=[],
            filters_own_messages=False,
            skips_tool_events=True,
            skips_empty_content=True,
            empty_sender_behavior=SenderBehavior.CONTENT_AS_IS,
            missing_sender_behavior=SenderBehavior.CONTENT_AS_IS,
            has_sender_metadata=True,
            # Parlant keeps other agents as "assistant" because its server-side
            # session model treats all bot-originated messages uniformly; remapping
            # to "user" would break the Parlant conversation contract.
            other_agent_output_role="assistant",
            output_adapter=SenderMetadataDictListOutputAdapter(),
        ),
    ]


def __getattr__(name: str) -> Any:
    if name == "CONVERTER_CONFIGS":
        configs = _build_converter_configs()
        globals()["CONVERTER_CONFIGS"] = configs
        return configs
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
