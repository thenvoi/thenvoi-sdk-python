"""Framework converter configurations for parameterized contract tests.

This module defines the configuration for each converter framework, allowing
contract tests to run the same test logic across all converters while handling
framework-specific behaviors through configuration.
"""

from __future__ import annotations

import json
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

ToolHandlingMode = Literal[
    "structured",  # Converts to structured format (Anthropic, PydanticAI)
    "langchain",  # LangChain-specific AIMessage/ToolMessage format
    "raw_json",  # Include as raw JSON string (ClaudeSDK)
    "skip",  # Skip tool events entirely (CrewAI, Parlant)
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
        tool_handling_mode: How tool_call/tool_result messages are handled
        batches_tool_calls: Whether consecutive tool_calls are batched
        batches_tool_results: Whether consecutive tool_results are batched
        supports_is_error: Whether the converter handles is_error field
        logs_malformed_json: Whether malformed JSON triggers a log warning
    """

    name: str
    converter_class: type
    output_type: OutputType
    skips_own_messages: bool = True
    converts_other_agents_to_user: bool = True
    preserves_other_agents_as_assistant: bool = False
    skips_empty_content: bool = False
    empty_sender_prefix_behavior: Literal["empty_brackets", "no_prefix"] = "no_prefix"
    # Tool handling configuration
    tool_handling_mode: ToolHandlingMode = "skip"
    batches_tool_calls: bool = False
    batches_tool_results: bool = False
    supports_is_error: bool = False
    logs_malformed_json: bool = False
    # LangChain-specific: only produces output when tool_call has matching tool_result
    requires_tool_result_for_output: bool = False

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
        # For string, count non-empty lines
        if not result:
            return 0
        lines = [line for line in result.split("\n") if line.strip()]
        return len(lines)

    return len(result) if result else 0


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
            lines = [line for line in self.result.split("\n") if line.strip()] if self.result else []
            return lines[index] if index < len(lines) else ""

        if self.config.output_type == "dict_list":
            if not self.result or index >= len(self.result):
                return ""
            content = self.result[index].get("content", "")
            # Handle list content (Anthropic tool blocks)
            if isinstance(content, list):
                return str(content)
            return content

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
            part = msg.parts[0]
            if hasattr(part, "content"):
                return part.content
            return str(part)

        return ""

    def get_role(self, index: int = 0) -> str:
        """Get role at index."""
        return _get_role_from_result(self.result, self.config, index)

    def content_contains(self, substring: str, index: int = 0) -> bool:
        """Check if content at index contains substring."""
        return substring in self.get_content(index)


class ToolCallAssertion:
    """Helper class for making tool call assertions on converter output."""

    def __init__(self, result: Any, config: ConverterConfig):
        self.result = result
        self.config = config

    def get_length(self) -> int:
        """Get number of output items."""
        return _get_result_length(self.result, self.config)

    def has_tool_call_at(self, index: int, name: str = None, tool_id: str = None) -> bool:
        """Check if there's a tool call at the given index with optional name/id match."""
        if self.config.tool_handling_mode == "skip":
            return False

        if self.config.output_type == "dict_list":
            # Anthropic format: {"role": "assistant", "content": [{"type": "tool_use", ...}]}
            if not self.result or index >= len(self.result):
                return False
            msg = self.result[index]
            content = msg.get("content", [])
            if not isinstance(content, list) or not content:
                return False
            tool_block = content[0]
            if tool_block.get("type") != "tool_use":
                return False
            if name and tool_block.get("name") != name:
                return False
            if tool_id and tool_block.get("id") != tool_id:
                return False
            return True

        if self.config.output_type == "langchain_messages":
            from langchain_core.messages import AIMessage
            if not self.result or index >= len(self.result):
                return False
            msg = self.result[index]
            if not isinstance(msg, AIMessage) or not msg.tool_calls:
                return False
            tool_call = msg.tool_calls[0]
            if name and tool_call.get("name") != name:
                return False
            if tool_id and tool_call.get("id") != tool_id:
                return False
            return True

        if self.config.output_type == "pydantic_ai_messages":
            from pydantic_ai.messages import ModelResponse, ToolCallPart
            if not self.result or index >= len(self.result):
                return False
            msg = self.result[index]
            if not isinstance(msg, ModelResponse) or not msg.parts:
                return False
            part = msg.parts[0]
            if not isinstance(part, ToolCallPart):
                return False
            if name and part.tool_name != name:
                return False
            if tool_id and part.tool_call_id != tool_id:
                return False
            return True

        if self.config.output_type == "string":
            # ClaudeSDK: raw JSON string
            if not self.result:
                return False
            lines = [line for line in self.result.split("\n") if line.strip()]
            if index >= len(lines):
                return False
            line = lines[index]
            # Check if it looks like JSON with the tool name
            if name and f'"{name}"' not in line:
                return False
            if tool_id and tool_id not in line:
                return False
            return True

        return False

    def has_tool_result_at(self, index: int, tool_id: str = None, content: str = None) -> bool:
        """Check if there's a tool result at the given index."""
        if self.config.tool_handling_mode == "skip":
            return False

        if self.config.output_type == "dict_list":
            # Anthropic format: {"role": "user", "content": [{"type": "tool_result", ...}]}
            if not self.result or index >= len(self.result):
                return False
            msg = self.result[index]
            msg_content = msg.get("content", [])
            if not isinstance(msg_content, list) or not msg_content:
                return False
            tool_block = msg_content[0]
            if tool_block.get("type") != "tool_result":
                return False
            if tool_id and tool_block.get("tool_use_id") != tool_id:
                return False
            if content and tool_block.get("content") != content:
                return False
            return True

        if self.config.output_type == "langchain_messages":
            from langchain_core.messages import ToolMessage
            if not self.result or index >= len(self.result):
                return False
            msg = self.result[index]
            if not isinstance(msg, ToolMessage):
                return False
            if tool_id and msg.tool_call_id != tool_id:
                return False
            return True

        if self.config.output_type == "pydantic_ai_messages":
            from pydantic_ai.messages import ModelRequest, ToolReturnPart, RetryPromptPart
            if not self.result or index >= len(self.result):
                return False
            msg = self.result[index]
            if not isinstance(msg, ModelRequest) or not msg.parts:
                return False
            part = msg.parts[0]
            if not isinstance(part, (ToolReturnPart, RetryPromptPart)):
                return False
            if tool_id and part.tool_call_id != tool_id:
                return False
            if content and part.content != content:
                return False
            return True

        if self.config.output_type == "string":
            # ClaudeSDK: raw JSON string
            if not self.result:
                return False
            lines = [line for line in self.result.split("\n") if line.strip()]
            if index >= len(lines):
                return False
            line = lines[index]
            if tool_id and tool_id not in line:
                return False
            if content and content not in line:
                return False
            return True

        return False

    def get_tool_call_count_at(self, index: int) -> int:
        """Get number of tool calls batched at the given index."""
        if self.config.output_type == "dict_list":
            if not self.result or index >= len(self.result):
                return 0
            content = self.result[index].get("content", [])
            if not isinstance(content, list):
                return 0
            return sum(1 for block in content if block.get("type") == "tool_use")

        if self.config.output_type == "langchain_messages":
            from langchain_core.messages import AIMessage
            if not self.result or index >= len(self.result):
                return 0
            msg = self.result[index]
            if not isinstance(msg, AIMessage):
                return 0
            return len(msg.tool_calls) if msg.tool_calls else 0

        if self.config.output_type == "pydantic_ai_messages":
            from pydantic_ai.messages import ModelResponse, ToolCallPart
            if not self.result or index >= len(self.result):
                return 0
            msg = self.result[index]
            if not isinstance(msg, ModelResponse):
                return 0
            return sum(1 for part in msg.parts if isinstance(part, ToolCallPart))

        return 0

    def get_tool_result_count_at(self, index: int) -> int:
        """Get number of tool results batched at the given index."""
        if self.config.output_type == "dict_list":
            if not self.result or index >= len(self.result):
                return 0
            content = self.result[index].get("content", [])
            if not isinstance(content, list):
                return 0
            return sum(1 for block in content if block.get("type") == "tool_result")

        if self.config.output_type == "pydantic_ai_messages":
            from pydantic_ai.messages import ModelRequest, ToolReturnPart, RetryPromptPart
            if not self.result or index >= len(self.result):
                return 0
            msg = self.result[index]
            if not isinstance(msg, ModelRequest):
                return 0
            return sum(1 for part in msg.parts if isinstance(part, (ToolReturnPart, RetryPromptPart)))

        return 0

    def has_is_error_at(self, index: int, expected: bool) -> bool:
        """Check if tool result at index has is_error field with expected value."""
        if not self.config.supports_is_error:
            return False

        if self.config.output_type == "dict_list":
            # Anthropic format
            if not self.result or index >= len(self.result):
                return False
            msg = self.result[index]
            content = msg.get("content", [])
            if not isinstance(content, list) or not content:
                return False
            tool_block = content[0]
            if expected:
                return tool_block.get("is_error") is True
            else:
                # When is_error is False, Anthropic omits the field
                return "is_error" not in tool_block

        if self.config.output_type == "pydantic_ai_messages":
            from pydantic_ai.messages import ModelRequest, ToolReturnPart, RetryPromptPart
            if not self.result or index >= len(self.result):
                return False
            msg = self.result[index]
            if not isinstance(msg, ModelRequest) or not msg.parts:
                return False
            part = msg.parts[0]
            if expected:
                return isinstance(part, RetryPromptPart)
            else:
                return isinstance(part, ToolReturnPart)

        return False


# Tool format types - different frameworks expect different JSON structures
ToolFormat = Literal["anthropic", "langchain"]


def make_tool_call(
    name: str, args: dict, tool_call_id: str, fmt: ToolFormat = "anthropic"
) -> dict:
    """Create a tool_call message in the appropriate format."""
    if fmt == "langchain":
        content = json.dumps({"name": name, "data": {"input": args}})
    else:
        content = json.dumps({"name": name, "args": args, "tool_call_id": tool_call_id})

    return {
        "role": "assistant",
        "content": content,
        "message_type": "tool_call",
    }


def make_tool_result(
    name: str,
    output: str,
    tool_call_id: str,
    is_error: bool | None = None,
    fmt: ToolFormat = "anthropic",
) -> dict:
    """Create a tool_result message in the appropriate format."""
    if fmt == "langchain":
        # LangChain extracts tool_call_id from output via regex
        output_with_id = f"{output} tool_call_id='{tool_call_id}'"
        content = json.dumps({"name": name, "data": {"output": output_with_id}, "run_id": tool_call_id})
    else:
        data: dict = {"name": name, "output": output, "tool_call_id": tool_call_id}
        if is_error is not None:
            data["is_error"] = is_error
        content = json.dumps(data)

    return {
        "role": "assistant",
        "content": content,
        "message_type": "tool_result",
    }


def get_tool_format(config: "ConverterConfig") -> ToolFormat:
    """Get the tool format expected by a converter."""
    if config.tool_handling_mode == "langchain":
        return "langchain"
    return "anthropic"


# Shared test input data for tool scenarios (using default anthropic format)
# Tests should use make_tool_call/make_tool_result for framework-specific formats
TOOL_CALL_SIMPLE = {
    "role": "assistant",
    "content": '{"name": "search", "args": {"query": "test"}, "tool_call_id": "toolu_123"}',
    "message_type": "tool_call",
}

TOOL_RESULT_SIMPLE = {
    "role": "assistant",
    "content": '{"name": "search", "output": "result data", "tool_call_id": "toolu_123"}',
    "message_type": "tool_result",
}

TOOL_CALL_1 = {
    "role": "assistant",
    "content": '{"name": "tool1", "args": {}, "tool_call_id": "toolu_1"}',
    "message_type": "tool_call",
}

TOOL_CALL_2 = {
    "role": "assistant",
    "content": '{"name": "tool2", "args": {}, "tool_call_id": "toolu_2"}',
    "message_type": "tool_call",
}

TOOL_RESULT_1 = {
    "role": "assistant",
    "content": '{"name": "tool1", "output": "result1", "tool_call_id": "toolu_1"}',
    "message_type": "tool_result",
}

TOOL_RESULT_2 = {
    "role": "assistant",
    "content": '{"name": "tool2", "output": "result2", "tool_call_id": "toolu_2"}',
    "message_type": "tool_result",
}

TOOL_RESULT_ERROR = {
    "role": "assistant",
    "content": '{"name": "search", "output": "Error: API failed", "tool_call_id": "toolu_123", "is_error": true}',
    "message_type": "tool_result",
}

TOOL_RESULT_SUCCESS = {
    "role": "assistant",
    "content": '{"name": "search", "output": "result", "tool_call_id": "toolu_123", "is_error": false}',
    "message_type": "tool_result",
}

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
        tool_handling_mode="structured",
        batches_tool_calls=True,
        batches_tool_results=True,
        supports_is_error=True,
        logs_malformed_json=True,
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
        tool_handling_mode="skip",
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
        tool_handling_mode="raw_json",
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
        tool_handling_mode="skip",
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
        tool_handling_mode="langchain",
        batches_tool_calls=False,  # LangChain doesn't batch
        batches_tool_results=False,
        supports_is_error=False,
        logs_malformed_json=False,
        requires_tool_result_for_output=True,  # Only outputs when tool_call has matching result
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
        tool_handling_mode="structured",
        batches_tool_calls=True,
        batches_tool_results=True,
        supports_is_error=True,  # Uses RetryPromptPart for errors
        logs_malformed_json=True,
    )
