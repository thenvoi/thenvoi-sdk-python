"""Framework converter configurations for parameterized conformance tests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

from thenvoi.converters.anthropic import AnthropicHistoryConverter
from thenvoi.converters.crewai import CrewAIHistoryConverter
from thenvoi.converters.claude_sdk import ClaudeSDKHistoryConverter
from thenvoi.converters.parlant import ParlantHistoryConverter

from ._output_adapters import (
    OutputTypeAdapter,
    DictListAdapter,
    StringAdapter,
    LangChainAdapter,
    PydanticAIAdapter,
)

# Optional dependencies
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


# =============================================================================
# Type Definitions
# =============================================================================

OutputType = Literal[
    "dict_list", "langchain_messages", "pydantic_ai_messages", "string"
]
ToolJsonFormat = Literal["anthropic", "langchain"]


class ToolHandlingMode(str, Enum):
    """How a converter handles tool_call and tool_result messages."""

    STRUCTURED = "structured"  # Convert to structured format (Anthropic, PydanticAI)
    LANGCHAIN = "langchain"  # Convert to LangChain message format
    RAW_JSON = "raw_json"  # Include as raw JSON string (ClaudeSDK)
    SKIP = "skip"  # Skip tool messages entirely (CrewAI, Parlant)


# =============================================================================
# Output Type Adapter Registry
# =============================================================================

OUTPUT_ADAPTERS: dict[OutputType, OutputTypeAdapter] = {
    "dict_list": DictListAdapter(),
    "string": StringAdapter(),
}

if LANGCHAIN_AVAILABLE:
    OUTPUT_ADAPTERS["langchain_messages"] = LangChainAdapter()

if PYDANTIC_AI_AVAILABLE:
    OUTPUT_ADAPTERS["pydantic_ai_messages"] = PydanticAIAdapter()


# =============================================================================
# Converter Configuration
# =============================================================================


@dataclass(frozen=True)
class ConverterConfig:
    """Configuration for a framework's history converter."""

    name: str
    converter_class: type
    output_type: OutputType
    skips_own_messages: bool = True
    converts_other_agents_to_user: bool = True
    preserves_other_agents_as_assistant: bool = False
    skips_empty_content: bool = False
    empty_sender_prefix_behavior: Literal["empty_brackets", "no_prefix"] = "no_prefix"
    tool_handling_mode: ToolHandlingMode = ToolHandlingMode.SKIP
    batches_tool_calls: bool = False
    batches_tool_results: bool = False
    supports_is_error: bool = False
    logs_malformed_json: bool = False
    requires_tool_result_for_output: bool = False

    def create_converter(self, agent_name: str = "") -> Any:
        """Create an instance of this converter."""
        return self.converter_class(agent_name=agent_name)

    def format_sender_prefix(self, name: str) -> str:
        """Format sender name prefix (e.g., '[Alice]:')."""
        if not name:
            return (
                "[]:" if self.empty_sender_prefix_behavior == "empty_brackets" else ""
            )
        return f"[{name}]:"


# =============================================================================
# Assertion Helpers
# =============================================================================


def _get_adapter(config: ConverterConfig) -> OutputTypeAdapter:
    adapter = OUTPUT_ADAPTERS.get(config.output_type)
    if adapter is None:
        raise ValueError(f"No adapter for output type: {config.output_type}")
    return adapter


class ContentAssertion:
    """Helper for content assertions on converter output."""

    def __init__(self, result: Any, config: ConverterConfig):
        self.result = result
        self.config = config
        self._adapter = _get_adapter(config)

    def has_length(self, expected: int) -> bool:
        return self._adapter.get_length(self.result) == expected

    def is_empty(self) -> bool:
        return self._adapter.get_length(self.result) == 0

    def get_content(self, index: int = 0) -> str:
        return self._adapter.get_content(self.result, index)

    def get_role(self, index: int = 0) -> str:
        return self._adapter.get_role(self.result, index)

    def content_contains(self, substring: str, index: int = 0) -> bool:
        return substring in self.get_content(index)


class ToolCallAssertion:
    """Helper for tool call assertions on converter output."""

    def __init__(self, result: Any, config: ConverterConfig):
        self.result = result
        self.config = config
        self._adapter = _get_adapter(config)

    def get_length(self) -> int:
        return self._adapter.get_length(self.result)

    def has_tool_call_at(
        self, index: int, name: str | None = None, tool_id: str | None = None
    ) -> bool:
        if self.config.tool_handling_mode == ToolHandlingMode.SKIP:
            return False
        return self._adapter.has_tool_call(self.result, index, name, tool_id)

    def has_tool_result_at(
        self, index: int, tool_id: str | None = None, content: str | None = None
    ) -> bool:
        if self.config.tool_handling_mode == ToolHandlingMode.SKIP:
            return False
        return self._adapter.has_tool_result(self.result, index, tool_id, content)

    def get_tool_call_count_at(self, index: int) -> int:
        return self._adapter.get_tool_call_count(self.result, index)

    def get_tool_result_count_at(self, index: int) -> int:
        return self._adapter.get_tool_result_count(self.result, index)

    def has_is_error_at(self, index: int, expected: bool) -> bool:
        if not self.config.supports_is_error:
            return False
        return self._adapter.has_is_error(self.result, index, expected)


# =============================================================================
# Tool Message Factories
# =============================================================================


def get_tool_format(config: ConverterConfig) -> ToolJsonFormat:
    """Get the tool JSON format expected by a converter."""
    return (
        "langchain"
        if config.tool_handling_mode == ToolHandlingMode.LANGCHAIN
        else "anthropic"
    )


def make_tool_call(
    name: str, args: dict, tool_call_id: str, fmt: ToolJsonFormat = "anthropic"
) -> dict:
    """Create a tool_call message in the appropriate format."""
    if fmt == "langchain":
        content = json.dumps({"name": name, "data": {"input": args}})
    else:
        content = json.dumps({"name": name, "args": args, "tool_call_id": tool_call_id})
    return {"role": "assistant", "content": content, "message_type": "tool_call"}


def make_tool_result(
    name: str,
    output: str,
    tool_call_id: str,
    is_error: bool | None = None,
    fmt: ToolJsonFormat = "anthropic",
) -> dict:
    """Create a tool_result message in the appropriate format."""
    if fmt == "langchain":
        output_with_id = f"{output} tool_call_id='{tool_call_id}'"
        content = json.dumps(
            {"name": name, "data": {"output": output_with_id}, "run_id": tool_call_id}
        )
    else:
        data: dict = {"name": name, "output": output, "tool_call_id": tool_call_id}
        if is_error is not None:
            data["is_error"] = is_error
        content = json.dumps(data)
    return {"role": "assistant", "content": content, "message_type": "tool_result"}


# =============================================================================
# Test Fixtures (Malformed tool messages)
# =============================================================================

MALFORMED_TOOL_CALL = {
    "role": "assistant",
    "content": "not valid json",
    "message_type": "tool_call",
}
MALFORMED_TOOL_RESULT = {
    "role": "assistant",
    "content": "not valid json",
    "message_type": "tool_result",
}
TOOL_CALL_MISSING_ID = {
    "role": "assistant",
    "content": '{"name": "search", "args": {"query": "test"}}',
    "message_type": "tool_call",
}
TOOL_CALL_MISSING_NAME = {
    "role": "assistant",
    "content": '{"args": {"query": "test"}, "tool_call_id": "toolu_123"}',
    "message_type": "tool_call",
}


# =============================================================================
# Converter Configurations
# =============================================================================

CONVERTER_CONFIGS: dict[str, ConverterConfig] = {
    "anthropic": ConverterConfig(
        name="anthropic",
        converter_class=AnthropicHistoryConverter,
        output_type="dict_list",
        tool_handling_mode=ToolHandlingMode.STRUCTURED,
        batches_tool_calls=True,
        batches_tool_results=True,
        supports_is_error=True,
        logs_malformed_json=True,
    ),
    "crewai": ConverterConfig(
        name="crewai",
        converter_class=CrewAIHistoryConverter,
        output_type="dict_list",
        converts_other_agents_to_user=False,
        preserves_other_agents_as_assistant=True,
    ),
    "claude_sdk": ConverterConfig(
        name="claude_sdk",
        converter_class=ClaudeSDKHistoryConverter,
        output_type="string",
        skips_empty_content=True,
        tool_handling_mode=ToolHandlingMode.RAW_JSON,
    ),
    "parlant": ConverterConfig(
        name="parlant",
        converter_class=ParlantHistoryConverter,
        output_type="dict_list",
        skips_own_messages=False,
        converts_other_agents_to_user=False,
        preserves_other_agents_as_assistant=True,
        skips_empty_content=True,
    ),
}

if LANGCHAIN_AVAILABLE and LangChainHistoryConverter is not None:
    CONVERTER_CONFIGS["langchain"] = ConverterConfig(
        name="langchain",
        converter_class=LangChainHistoryConverter,
        output_type="langchain_messages",
        empty_sender_prefix_behavior="empty_brackets",
        tool_handling_mode=ToolHandlingMode.LANGCHAIN,
        requires_tool_result_for_output=True,
    )

if PYDANTIC_AI_AVAILABLE and PydanticAIHistoryConverter is not None:
    CONVERTER_CONFIGS["pydantic_ai"] = ConverterConfig(
        name="pydantic_ai",
        converter_class=PydanticAIHistoryConverter,
        output_type="pydantic_ai_messages",
        tool_handling_mode=ToolHandlingMode.STRUCTURED,
        batches_tool_calls=True,
        batches_tool_results=True,
        supports_is_error=True,
        logs_malformed_json=True,
    )
