"""Framework converter configurations for parameterized contract tests.

This module defines the configuration for each converter framework, allowing
contract tests to run the same test logic across all converters while handling
framework-specific behaviors through configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from thenvoi.converters.anthropic import AnthropicHistoryConverter
from thenvoi.converters.crewai import CrewAIHistoryConverter
from thenvoi.converters.claude_sdk import ClaudeSDKHistoryConverter
from thenvoi.converters.parlant import ParlantHistoryConverter

# LangChain and PydanticAI require optional dependencies
try:
    from thenvoi.converters.langchain import LangChainHistoryConverter

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LangChainHistoryConverter = None  # type: ignore[misc, assignment]
    LANGCHAIN_AVAILABLE = False

try:
    from thenvoi.converters.pydantic_ai import PydanticAIHistoryConverter

    PYDANTIC_AI_AVAILABLE = True
except ImportError:
    PydanticAIHistoryConverter = None  # type: ignore[misc, assignment]
    PYDANTIC_AI_AVAILABLE = False


OutputType = Literal[
    "dict_list",
    "langchain_messages",
    "pydantic_ai_messages",
    "string",
]


@dataclass(frozen=True)
class ConverterConfig:
    """Configuration for a framework's history converter.

    Attributes:
        name: Human-readable framework name (used as test ID)
        converter_class: The converter class to instantiate
        output_type: Type of output the converter produces
        skips_own_messages: Whether this converter skips the agent's own text messages
        converts_other_agents_to_user: Whether other agents' messages become user role
        preserves_other_agents_as_assistant: Whether other agents keep assistant role
        skips_empty_content: Whether messages with empty content are skipped
        empty_sender_prefix_behavior: How empty sender_name is formatted in content
    """

    name: str
    converter_class: type
    output_type: OutputType
    skips_own_messages: bool = True
    converts_other_agents_to_user: bool = True
    preserves_other_agents_as_assistant: bool = False
    skips_empty_content: bool = False
    empty_sender_prefix_behavior: Literal["empty_brackets", "no_prefix"] = "no_prefix"

    def create_converter(self, agent_name: str = "") -> Any:
        """Create an instance of this converter."""
        return self.converter_class(agent_name=agent_name)


def _get_role_from_result(result: Any, config: ConverterConfig, index: int = 0) -> str:
    """Extract role from converter output based on output_type."""
    if config.output_type == "string":
        # String output doesn't have role
        return ""

    if config.output_type == "dict_list":
        if not result or index >= len(result):
            return ""
        return result[index].get("role", "")

    if config.output_type == "langchain_messages":
        if not result or index >= len(result):
            return ""
        msg = result[index]
        # LangChain messages have type-based roles
        type_name = type(msg).__name__
        if "Human" in type_name:
            return "user"
        if "AI" in type_name or "Tool" in type_name:
            return "assistant"
        return ""

    if config.output_type == "pydantic_ai_messages":
        if not result or index >= len(result):
            return ""
        msg = result[index]
        type_name = type(msg).__name__
        if "Request" in type_name:
            return "user"
        if "Response" in type_name:
            return "assistant"
        return ""

    return ""


def _get_result_length(result: Any, config: ConverterConfig) -> int:
    """Get the length/count of items in converter output."""
    if config.output_type == "string":
        # For string, count lines (since multiple messages are newline-separated)
        if not result:
            return 0
        return len(result.split("\n"))

    return len(result) if result else 0


# Assertion helpers for test readability
class ContentAssertion:
    """Helper class for making content assertions on converter output."""

    def __init__(self, result: Any, config: ConverterConfig):
        self.result = result
        self.config = config

    def has_length(self, expected: int) -> bool:
        """Check if result has expected number of items."""
        return _get_result_length(self.result, self.config) == expected

    def is_empty(self) -> bool:
        """Check if result is empty."""
        return _get_result_length(self.result, self.config) == 0

    def get_content(self, index: int = 0) -> str:
        """Get content at index."""
        if self.config.output_type == "string":
            # For string output, split by newline and get line
            lines = self.result.split("\n") if self.result else []
            return lines[index] if index < len(lines) else ""

        if self.config.output_type == "dict_list":
            if not self.result or index >= len(self.result):
                return ""
            return self.result[index].get("content", "")

        if self.config.output_type == "langchain_messages":
            if not self.result or index >= len(self.result):
                return ""
            return self.result[index].content

        if self.config.output_type == "pydantic_ai_messages":
            if not self.result or index >= len(self.result):
                return ""
            msg = self.result[index]
            if not msg.parts:
                return ""
            return msg.parts[0].content

        return ""

    def get_role(self, index: int = 0) -> str:
        """Get role at index."""
        return _get_role_from_result(self.result, self.config, index)

    def content_contains(self, substring: str, index: int = 0) -> bool:
        """Check if content at index contains substring."""
        return substring in self.get_content(index)


# Framework configurations
CONVERTER_CONFIGS: dict[str, ConverterConfig] = {
    "anthropic": ConverterConfig(
        name="anthropic",
        converter_class=AnthropicHistoryConverter,
        output_type="dict_list",
        skips_own_messages=True,
        converts_other_agents_to_user=True,
        preserves_other_agents_as_assistant=False,
        skips_empty_content=False,
        empty_sender_prefix_behavior="no_prefix",
    ),
    "crewai": ConverterConfig(
        name="crewai",
        converter_class=CrewAIHistoryConverter,
        output_type="dict_list",
        skips_own_messages=True,
        converts_other_agents_to_user=False,
        preserves_other_agents_as_assistant=True,
        skips_empty_content=False,
        empty_sender_prefix_behavior="no_prefix",
    ),
    "claude_sdk": ConverterConfig(
        name="claude_sdk",
        converter_class=ClaudeSDKHistoryConverter,
        output_type="string",
        skips_own_messages=True,
        converts_other_agents_to_user=True,
        preserves_other_agents_as_assistant=False,
        skips_empty_content=True,
        empty_sender_prefix_behavior="no_prefix",
    ),
    "parlant": ConverterConfig(
        name="parlant",
        converter_class=ParlantHistoryConverter,
        output_type="dict_list",
        skips_own_messages=False,  # Parlant needs full history
        converts_other_agents_to_user=False,
        preserves_other_agents_as_assistant=True,
        skips_empty_content=True,
        empty_sender_prefix_behavior="no_prefix",
    ),
}

# Add LangChain if available
if LANGCHAIN_AVAILABLE:
    CONVERTER_CONFIGS["langchain"] = ConverterConfig(
        name="langchain",
        converter_class=LangChainHistoryConverter,
        output_type="langchain_messages",
        skips_own_messages=True,
        converts_other_agents_to_user=True,
        preserves_other_agents_as_assistant=False,
        skips_empty_content=False,
        empty_sender_prefix_behavior="empty_brackets",
    )

# Add PydanticAI if available
if PYDANTIC_AI_AVAILABLE:
    CONVERTER_CONFIGS["pydantic_ai"] = ConverterConfig(
        name="pydantic_ai",
        converter_class=PydanticAIHistoryConverter,
        output_type="pydantic_ai_messages",
        skips_own_messages=True,
        converts_other_agents_to_user=True,
        preserves_other_agents_as_assistant=False,
        skips_empty_content=False,
        empty_sender_prefix_behavior="no_prefix",
    )
