"""Pydantic AI converter-specific tests.

Tests for PydanticAI-specific behavior that isn't covered by conformance tests:
- RetryPromptPart for error results (is_error=True)
- ToolReturnPart for success results
- ModelResponse/ModelRequest structure
- Batching of mixed success/error results
"""

from __future__ import annotations

import json

from thenvoi.converters.pydantic_ai import PydanticAIHistoryConverter


class TestToolReturnPartForSuccess:
    """Tests for ToolReturnPart usage in successful results."""

    def test_uses_tool_return_part_for_success_results(self):
        """Successful tool results should use ToolReturnPart."""
        from pydantic_ai.messages import ModelRequest, ToolReturnPart

        converter = PydanticAIHistoryConverter()
        raw = [
            {
                "role": "assistant",
                "content": json.dumps(
                    {"name": "search", "args": {}, "tool_call_id": "toolu_1"}
                ),
                "message_type": "tool_call",
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "name": "search",
                        "output": "Found results",
                        "tool_call_id": "toolu_1",
                        "is_error": False,
                    }
                ),
                "message_type": "tool_result",
            },
        ]

        result = converter.convert(raw)

        # Find the ModelRequest with tool result
        tool_result_msg = [m for m in result if isinstance(m, ModelRequest)][-1]
        part = tool_result_msg.parts[0]

        assert isinstance(part, ToolReturnPart)
        assert part.content == "Found results"
        assert part.tool_call_id == "toolu_1"

    def test_uses_tool_return_part_when_is_error_missing(self):
        """Missing is_error should be treated as success (ToolReturnPart)."""
        from pydantic_ai.messages import ModelRequest, ToolReturnPart

        converter = PydanticAIHistoryConverter()
        raw = [
            {
                "role": "assistant",
                "content": json.dumps(
                    {"name": "calc", "args": {}, "tool_call_id": "toolu_2"}
                ),
                "message_type": "tool_call",
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "name": "calc",
                        "output": "42",
                        "tool_call_id": "toolu_2",
                        # No is_error field
                    }
                ),
                "message_type": "tool_result",
            },
        ]

        result = converter.convert(raw)

        tool_result_msg = [m for m in result if isinstance(m, ModelRequest)][-1]
        part = tool_result_msg.parts[0]

        assert isinstance(part, ToolReturnPart)


class TestRetryPromptPartForErrors:
    """Tests for RetryPromptPart usage in error results."""

    def test_uses_retry_prompt_part_for_error_results(self):
        """Error tool results (is_error=True) should use RetryPromptPart."""
        from pydantic_ai.messages import ModelRequest, RetryPromptPart

        converter = PydanticAIHistoryConverter()
        raw = [
            {
                "role": "assistant",
                "content": json.dumps(
                    {"name": "api_call", "args": {}, "tool_call_id": "toolu_err"}
                ),
                "message_type": "tool_call",
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "name": "api_call",
                        "output": "Error: Connection timeout",
                        "tool_call_id": "toolu_err",
                        "is_error": True,
                    }
                ),
                "message_type": "tool_result",
            },
        ]

        result = converter.convert(raw)

        # Find the ModelRequest with tool result
        tool_result_msg = [m for m in result if isinstance(m, ModelRequest)][-1]
        part = tool_result_msg.parts[0]

        assert isinstance(part, RetryPromptPart)
        assert part.content == "Error: Connection timeout"
        assert part.tool_call_id == "toolu_err"


class TestModelResponseStructure:
    """Tests for ModelResponse structure with ToolCallPart."""

    def test_creates_model_response_with_tool_call_part(self):
        """Tool call should create ModelResponse with ToolCallPart."""
        from pydantic_ai.messages import ModelResponse, ToolCallPart

        converter = PydanticAIHistoryConverter()
        raw = [
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "name": "get_weather",
                        "args": {"location": "NYC"},
                        "tool_call_id": "toolu_weather",
                    }
                ),
                "message_type": "tool_call",
            }
        ]

        result = converter.convert(raw)

        assert len(result) == 1
        msg = result[0]

        assert isinstance(msg, ModelResponse)
        assert len(msg.parts) == 1

        part = msg.parts[0]
        assert isinstance(part, ToolCallPart)
        assert part.tool_name == "get_weather"
        assert part.args == {"location": "NYC"}
        assert part.tool_call_id == "toolu_weather"


class TestMixedSuccessAndErrorResults:
    """Tests for batching mixed success and error results."""

    def test_batches_mixed_success_and_error_results(self):
        """Mixed success/error results should be batched into single ModelRequest."""
        from pydantic_ai.messages import (
            ModelRequest,
            RetryPromptPart,
            ToolReturnPart,
        )

        converter = PydanticAIHistoryConverter()
        raw = [
            # Two tool calls
            {
                "role": "assistant",
                "content": json.dumps(
                    {"name": "tool1", "args": {}, "tool_call_id": "t1"}
                ),
                "message_type": "tool_call",
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    {"name": "tool2", "args": {}, "tool_call_id": "t2"}
                ),
                "message_type": "tool_call",
            },
            # Success result
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "name": "tool1",
                        "output": "success",
                        "tool_call_id": "t1",
                        "is_error": False,
                    }
                ),
                "message_type": "tool_result",
            },
            # Error result
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "name": "tool2",
                        "output": "failed",
                        "tool_call_id": "t2",
                        "is_error": True,
                    }
                ),
                "message_type": "tool_result",
            },
        ]

        result = converter.convert(raw)

        # Should have 2 messages: ModelResponse (2 calls) + ModelRequest (2 results)
        assert len(result) == 2

        # Check results are batched
        results_msg = result[1]
        assert isinstance(results_msg, ModelRequest)
        assert len(results_msg.parts) == 2

        # Check types - one success, one error
        part_types = {type(p) for p in results_msg.parts}
        assert ToolReturnPart in part_types
        assert RetryPromptPart in part_types


class TestToolCallBatching:
    """Tests for tool call batching into single ModelResponse."""

    def test_batches_consecutive_tool_calls(self):
        """Consecutive tool_calls should be batched into one ModelResponse."""
        from pydantic_ai.messages import ModelResponse, ToolCallPart

        converter = PydanticAIHistoryConverter()
        raw = [
            {
                "role": "assistant",
                "content": json.dumps(
                    {"name": "t1", "args": {}, "tool_call_id": "id1"}
                ),
                "message_type": "tool_call",
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    {"name": "t2", "args": {}, "tool_call_id": "id2"}
                ),
                "message_type": "tool_call",
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    {"name": "t3", "args": {}, "tool_call_id": "id3"}
                ),
                "message_type": "tool_call",
            },
        ]

        result = converter.convert(raw)

        assert len(result) == 1
        msg = result[0]

        assert isinstance(msg, ModelResponse)
        assert len(msg.parts) == 3
        assert all(isinstance(p, ToolCallPart) for p in msg.parts)

    def test_flushes_trailing_tool_calls(self):
        """Trailing tool calls at end of history should be flushed."""
        from pydantic_ai.messages import ModelResponse

        converter = PydanticAIHistoryConverter()
        raw = [
            {
                "role": "user",
                "content": "Do something",
                "sender_name": "Alice",
                "message_type": "text",
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    {"name": "action", "args": {}, "tool_call_id": "trailing"}
                ),
                "message_type": "tool_call",
            },
        ]

        result = converter.convert(raw)

        # Should have 2 messages: user text + trailing tool call
        assert len(result) == 2
        assert isinstance(result[1], ModelResponse)
