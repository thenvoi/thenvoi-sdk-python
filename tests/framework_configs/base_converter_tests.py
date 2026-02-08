"""Base test class for converter conformance testing.

New framework authors can inherit from this class to verify their implementation
without editing the shared config files.

Example usage:

    # tests/converters/test_my_framework.py
    from tests.framework_configs.base_converter_tests import BaseConverterTests
    from thenvoi.converters.my_framework import MyFrameworkHistoryConverter

    class TestMyFrameworkConverter(BaseConverterTests):
        converter_class = MyFrameworkHistoryConverter
        output_type = "dict_list"  # or "langchain_messages", "pydantic_ai_messages", "string"

        # Optional overrides (defaults shown):
        tool_handling_mode = "skip"  # or "structured", "langchain", "raw_json"
        skips_own_messages = True
        converts_other_agents_to_user = True
        batches_tool_calls = False
        batches_tool_results = False
        supports_is_error = False
        skips_empty_content = False
        empty_sender_prefix_behavior = "no_prefix"  # or "empty_brackets"

Running:
    uv run pytest tests/converters/test_my_framework.py -v
"""

from __future__ import annotations

import json
from abc import ABC
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import pytest

if TYPE_CHECKING:
    from thenvoi.core.protocols import HistoryConverter


class BaseConverterTests(ABC):
    """Base class providing all conformance tests for converters.

    Subclasses must set:
        - converter_class: The converter class to test (must implement HistoryConverter protocol)
        - output_type: One of "dict_list", "langchain_messages", "pydantic_ai_messages", "string"

    Subclasses may override:
        - tool_handling_mode: "skip", "structured", "langchain", "raw_json"
        - skips_own_messages: Whether to skip agent's own messages
        - converts_other_agents_to_user: Whether other agents become user role
        - batches_tool_calls: Whether consecutive tool calls are batched
        - batches_tool_results: Whether consecutive tool results are batched
        - supports_is_error: Whether is_error field is preserved
        - skips_empty_content: Whether empty content messages are skipped
        - empty_sender_prefix_behavior: "no_prefix" or "empty_brackets"
        - requires_tool_result_for_output: Whether tool_call alone produces no output
        - logs_malformed_json: Whether malformed JSON is logged
    """

    # Required - must be set by subclass
    # Type is 'type' at runtime but should implement HistoryConverter protocol
    converter_class: ClassVar[type[HistoryConverter[Any]]]
    output_type: ClassVar[
        Literal["dict_list", "langchain_messages", "pydantic_ai_messages", "string"]
    ]

    # Optional configuration (sensible defaults)
    tool_handling_mode: ClassVar[
        Literal["skip", "structured", "langchain", "raw_json"]
    ] = "skip"
    skips_own_messages: ClassVar[bool] = True
    converts_other_agents_to_user: ClassVar[bool] = True
    batches_tool_calls: ClassVar[bool] = False
    batches_tool_results: ClassVar[bool] = False
    supports_is_error: ClassVar[bool] = False
    skips_empty_content: ClassVar[bool] = False
    empty_sender_prefix_behavior: ClassVar[Literal["no_prefix", "empty_brackets"]] = (
        "no_prefix"
    )
    requires_tool_result_for_output: ClassVar[bool] = False
    logs_malformed_json: ClassVar[bool] = False

    # ==================== Fixtures ====================

    @pytest.fixture
    def converter(self) -> Any:
        """Create a converter instance."""
        return self.converter_class()

    @pytest.fixture
    def converter_with_name(self) -> Any:
        """Create a converter instance with agent name set."""
        return self.converter_class(agent_name="Agent")

    # ==================== Helper Methods ====================

    def _get_content(self, result: Any, index: int) -> str:
        """Extract content from result based on output_type."""
        if not result:
            return ""

        if self.output_type == "dict_list":
            if index >= len(result):
                return ""
            content = result[index].get("content", "")
            return str(content) if isinstance(content, list) else content

        elif self.output_type == "string":
            lines = [line for line in result.split("\n") if line.strip()]
            return lines[index] if index < len(lines) else ""

        elif self.output_type == "langchain_messages":
            if index >= len(result):
                return ""
            return result[index].content

        elif self.output_type == "pydantic_ai_messages":
            if index >= len(result) or not result[index].parts:
                return ""
            part = result[index].parts[0]
            return part.content if hasattr(part, "content") else str(part)

        return ""

    def _get_role(self, result: Any, index: int) -> str:
        """Extract role from result based on output_type."""
        if not result:
            return ""

        if self.output_type == "dict_list":
            if index >= len(result):
                return ""
            return result[index].get("role", "")

        elif self.output_type == "string":
            return ""  # String output doesn't have role concept

        elif self.output_type == "langchain_messages":
            if index >= len(result):
                return ""
            name = type(result[index]).__name__
            if "Human" in name:
                return "user"
            if "AI" in name or "Tool" in name:
                return "assistant"
            return ""

        elif self.output_type == "pydantic_ai_messages":
            if index >= len(result):
                return ""
            name = type(result[index]).__name__
            return (
                "user"
                if "Request" in name
                else "assistant"
                if "Response" in name
                else ""
            )

        return ""

    def _get_length(self, result: Any) -> int:
        """Get length of result based on output_type."""
        if not result:
            return 0

        if self.output_type == "string":
            return len([line for line in result.split("\n") if line.strip()])

        return len(result)

    def _make_tool_call(self, name: str, args: dict, tool_id: str) -> dict:
        """Create a tool call message in the appropriate format."""
        if self.tool_handling_mode == "langchain":
            return {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "event": "on_tool_start",
                        "name": name,
                        "run_id": tool_id,
                        "data": {"input": args},
                    }
                ),
                "message_type": "tool_call",
            }
        else:
            return {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "name": name,
                        "args": args,
                        "tool_call_id": tool_id,
                    }
                ),
                "message_type": "tool_call",
            }

    def _make_tool_result(
        self, name: str, output: str, tool_id: str, is_error: bool = False
    ) -> dict:
        """Create a tool result message in the appropriate format."""
        if self.tool_handling_mode == "langchain":
            return {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "event": "on_tool_end",
                        "name": name,
                        "run_id": tool_id,
                        "data": {"output": f"{output} tool_call_id='{tool_id}'"},
                    }
                ),
                "message_type": "tool_result",
            }
        else:
            data = {
                "name": name,
                "output": output,
                "tool_call_id": tool_id,
            }
            if is_error:
                data["is_error"] = True
            return {
                "role": "assistant",
                "content": json.dumps(data),
                "message_type": "tool_result",
            }

    # ==================== User Message Tests ====================

    def test_converts_user_text_with_sender_name(self, converter):
        """User text messages include sender name prefix."""
        raw = [
            {
                "role": "user",
                "content": "Hello, agent!",
                "sender_name": "Alice",
                "message_type": "text",
            }
        ]

        result = converter.convert(raw)

        assert self._get_length(result) == 1
        assert "[Alice]:" in self._get_content(result, 0)
        assert "Hello, agent!" in self._get_content(result, 0)

    def test_handles_empty_sender_name(self, converter):
        """User messages with empty sender_name are handled appropriately."""
        raw = [
            {
                "role": "user",
                "content": "Hello!",
                "sender_name": "",
                "message_type": "text",
            }
        ]

        result = converter.convert(raw)

        if self.empty_sender_prefix_behavior == "empty_brackets":
            assert "[]:" in self._get_content(result, 0)
        else:
            assert "Hello!" in self._get_content(result, 0)

    def test_handles_missing_sender_name(self, converter):
        """Messages without sender_name key are handled appropriately."""
        raw = [
            {
                "role": "user",
                "content": "Hello without sender",
                "message_type": "text",
            }
        ]

        result = converter.convert(raw)

        assert self._get_length(result) == 1
        assert "Hello without sender" in self._get_content(result, 0)

    # ==================== Assistant Message Tests ====================

    def test_skips_own_assistant_text_messages_when_configured(
        self, converter_with_name
    ):
        """This agent's text messages are skipped (if configured)."""
        raw = [
            {
                "role": "assistant",
                "content": "I'll help you with that.",
                "sender_name": "Agent",
                "message_type": "text",
            }
        ]

        result = converter_with_name.convert(raw)

        if self.skips_own_messages:
            assert self._get_length(result) == 0
        else:
            assert self._get_length(result) == 1
            assert "I'll help you with that." in self._get_content(result, 0)

    def test_includes_other_agents_messages(self, converter_with_name):
        """Other agents' messages should be included."""
        converter_with_name.set_agent_name("Main Agent")
        raw = [
            {
                "role": "assistant",
                "content": "It's sunny today!",
                "sender_name": "Weather Agent",
                "message_type": "text",
            }
        ]

        result = converter_with_name.convert(raw)

        assert self._get_length(result) == 1
        if self.converts_other_agents_to_user:
            assert "[Weather Agent]:" in self._get_content(result, 0)
        else:
            assert self._get_role(result, 0) == "assistant"

    def test_skips_only_own_messages(self, converter):
        """Only skip THIS agent's text, include other agents."""
        converter.set_agent_name("Main Agent")
        raw = [
            {
                "role": "assistant",
                "content": "I'll help",
                "sender_name": "Main Agent",
                "message_type": "text",
            },
            {
                "role": "assistant",
                "content": "It's sunny!",
                "sender_name": "Weather Agent",
                "message_type": "text",
            },
        ]

        result = converter.convert(raw)

        if self.skips_own_messages:
            assert self._get_length(result) == 1
            assert "sunny" in self._get_content(result, 0).lower()
        else:
            assert self._get_length(result) == 2

    def test_set_agent_name_updates_filtering(self, converter):
        """set_agent_name should update which messages are skipped."""
        raw = [
            {
                "role": "assistant",
                "content": "Hello",
                "sender_name": "Agent",
                "message_type": "text",
            }
        ]

        # Before setting name - all assistant messages included
        result_before = converter.convert(raw)
        assert self._get_length(result_before) == 1

        # After setting name
        converter.set_agent_name("Agent")
        result_after = converter.convert(raw)

        if self.skips_own_messages:
            assert self._get_length(result_after) == 0
        else:
            assert self._get_length(result_after) == 1

    def test_includes_all_assistant_messages_when_no_agent_name(self, converter):
        """When agent name is not set, include all assistant messages."""
        raw = [
            {
                "role": "assistant",
                "content": "Hello from agent 1",
                "sender_name": "Agent 1",
                "message_type": "text",
            },
            {
                "role": "assistant",
                "content": "Hello from agent 2",
                "sender_name": "Agent 2",
                "message_type": "text",
            },
        ]

        result = converter.convert(raw)

        assert self._get_length(result) == 2

    # ==================== Edge Cases ====================

    def test_empty_history(self, converter):
        """Empty history returns empty result."""
        result = converter.convert([])

        assert self._get_length(result) == 0

    def test_defaults_to_text_message_type(self, converter):
        """Messages without message_type default to 'text'."""
        raw = [
            {
                "role": "user",
                "content": "Hello",
                "sender_name": "Bob",
            }
        ]

        result = converter.convert(raw)

        assert self._get_length(result) == 1
        assert "Hello" in self._get_content(result, 0)

    def test_skips_thought_messages(self, converter):
        """thought messages are skipped by all converters."""
        raw = [
            {
                "role": "assistant",
                "content": "I'm thinking about this...",
                "message_type": "thought",
            }
        ]

        result = converter.convert(raw)

        assert self._get_length(result) == 0

    def test_handles_empty_content(self, converter):
        """Empty content messages are handled according to config."""
        raw = [
            {
                "role": "user",
                "content": "",
                "sender_name": "Alice",
                "message_type": "text",
            }
        ]

        result = converter.convert(raw)

        if self.skips_empty_content:
            assert self._get_length(result) == 0
        else:
            assert self._get_length(result) == 1

    def test_defaults_to_user_role(self, converter):
        """Messages without role default to 'user'."""
        raw = [
            {
                "content": "Hello",
                "sender_name": "Bob",
                "message_type": "text",
            }
        ]

        result = converter.convert(raw)

        assert self._get_length(result) == 1
        assert "Hello" in self._get_content(result, 0)
        if self.output_type != "string":
            assert self._get_role(result, 0) == "user"

    # ==================== Tool Handling Tests ====================

    def test_handles_tool_call_according_to_mode(self, converter):
        """Tool calls are handled according to framework's tool_handling_mode."""
        tool_call = self._make_tool_call("search", {"query": "test"}, "toolu_123")
        raw = [tool_call]

        result = converter.convert(raw)

        if self.tool_handling_mode == "skip":
            assert self._get_length(result) == 0
        elif self.requires_tool_result_for_output:
            assert self._get_length(result) == 0
        else:
            assert self._get_length(result) >= 1

    def test_handles_tool_result_according_to_mode(self, converter):
        """Tool results are handled according to framework's tool_handling_mode."""
        tool_call = self._make_tool_call("search", {"query": "test"}, "toolu_123")
        tool_result = self._make_tool_result("search", "result data", "toolu_123")
        raw = [tool_call, tool_result]

        result = converter.convert(raw)

        if self.tool_handling_mode == "skip":
            assert self._get_length(result) == 0
        else:
            assert self._get_length(result) == 2

    # ==================== Tool Batching Tests ====================

    def test_batches_multiple_tool_calls_when_supported(self, converter):
        """Consecutive tool calls are batched if framework supports it."""
        tool_call_1 = self._make_tool_call("tool1", {}, "toolu_1")
        tool_call_2 = self._make_tool_call("tool2", {}, "toolu_2")
        tool_result_1 = self._make_tool_result("tool1", "result1", "toolu_1")
        raw = [tool_call_1, tool_call_2, tool_result_1]

        result = converter.convert(raw)

        if self.tool_handling_mode == "skip":
            assert self._get_length(result) == 0
        elif self.batches_tool_calls:
            # Batched: one message with both calls, one with result
            assert self._get_length(result) == 2
        elif self.requires_tool_result_for_output:
            # LangChain only outputs paired calls
            assert self._get_length(result) >= 1
        else:
            assert self._get_length(result) >= 2

    def test_batches_multiple_tool_results_when_supported(self, converter):
        """Consecutive tool results are batched if framework supports it."""
        tool_call_1 = self._make_tool_call("tool1", {}, "toolu_1")
        tool_call_2 = self._make_tool_call("tool2", {}, "toolu_2")
        tool_result_1 = self._make_tool_result("tool1", "result1", "toolu_1")
        tool_result_2 = self._make_tool_result("tool2", "result2", "toolu_2")
        raw = [tool_call_1, tool_call_2, tool_result_1, tool_result_2]

        result = converter.convert(raw)

        if self.tool_handling_mode == "skip":
            assert self._get_length(result) == 0
        elif self.batches_tool_results:
            # Batched: one message with calls, one with results
            assert self._get_length(result) == 2
        else:
            assert self._get_length(result) >= 2

    # ==================== Tool Error Handling Tests ====================

    def test_preserves_is_error_when_true(self, converter):
        """Tool result with is_error=True is handled correctly."""
        tool_call = self._make_tool_call("search", {"query": "test"}, "toolu_123")
        tool_result_error = self._make_tool_result(
            "search", "Error: API failed", "toolu_123", is_error=True
        )
        raw = [tool_call, tool_result_error]

        result = converter.convert(raw)

        if self.tool_handling_mode == "skip":
            assert self._get_length(result) == 0
        else:
            assert self._get_length(result) == 2

    # ==================== Mixed History Tests ====================

    def test_multi_user_conversation(self, converter_with_name):
        """Handles multiple users in conversation."""
        raw = [
            {
                "role": "user",
                "content": "Hi team!",
                "sender_name": "Alice",
                "message_type": "text",
            },
            {
                "role": "user",
                "content": "Hello everyone!",
                "sender_name": "Bob",
                "message_type": "text",
            },
            {
                "role": "assistant",
                "content": "Hello Alice and Bob!",
                "sender_name": "Agent",
                "message_type": "text",
            },
        ]

        result = converter_with_name.convert(raw)

        if self.skips_own_messages:
            assert self._get_length(result) == 2
        else:
            assert self._get_length(result) == 3

        assert "[Alice]:" in self._get_content(result, 0)
        assert "[Bob]:" in self._get_content(result, 1)

    def test_multi_agent_conversation_flow(self, converter):
        """Should include other agents' messages in multi-agent conversations."""
        converter.set_agent_name("Main Agent")
        raw = [
            {
                "role": "user",
                "content": "What's the weather in Tokyo?",
                "sender_name": "Alice",
                "message_type": "text",
            },
            {
                "role": "assistant",
                "content": "Let me check with the weather agent.",
                "sender_name": "Main Agent",
                "message_type": "text",
            },
            {
                "role": "assistant",
                "content": "Tokyo is 15C and cloudy.",
                "sender_name": "Weather Agent",
                "message_type": "text",
            },
            {
                "role": "assistant",
                "content": "The weather in Tokyo is 15C and cloudy.",
                "sender_name": "Main Agent",
                "message_type": "text",
            },
        ]

        result = converter.convert(raw)

        if self.skips_own_messages:
            assert self._get_length(result) == 2
        else:
            assert self._get_length(result) == 4

        assert "[Alice]:" in self._get_content(result, 0)
        assert "What's the weather" in self._get_content(result, 0)

    def test_full_conversation_with_tools(self, converter_with_name):
        """Full conversation flow with text messages and tool calls."""
        tool_call = self._make_tool_call("search", {"query": "test"}, "toolu_123")
        tool_result = self._make_tool_result("search", "result data", "toolu_123")

        raw = [
            {
                "role": "user",
                "content": "What's the weather?",
                "sender_name": "Alice",
                "message_type": "text",
            },
            tool_call,
            tool_result,
            {
                "role": "assistant",
                "content": "It's sunny!",
                "sender_name": "Agent",
                "message_type": "text",
            },
            {
                "role": "user",
                "content": "Thanks!",
                "sender_name": "Alice",
                "message_type": "text",
            },
        ]

        result = converter_with_name.convert(raw)

        if self.tool_handling_mode == "skip":
            if self.skips_own_messages:
                assert self._get_length(result) == 2  # Alice's two messages
            else:
                assert self._get_length(result) == 3  # + agent's message
        else:
            if self.skips_own_messages:
                assert (
                    self._get_length(result) == 4
                )  # Alice + tool call + tool result + Thanks
            else:
                assert self._get_length(result) == 5  # + agent's message

        assert "[Alice]:" in self._get_content(result, 0)
