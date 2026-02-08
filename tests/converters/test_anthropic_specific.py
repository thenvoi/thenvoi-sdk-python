"""Anthropic converter-specific tests.

Tests for Anthropic-specific behavior that isn't covered by conformance tests:
- Output structure validation (tool_use, tool_result content blocks)
- is_error field handling (only included when True, omitted when False)
- Content block structure validation
"""

from __future__ import annotations

import json

from thenvoi.converters.anthropic import AnthropicHistoryConverter


class TestToolUseContentBlocks:
    """Tests for tool_use content block structure."""

    def test_tool_call_creates_tool_use_block_with_correct_structure(self):
        """Tool call should create tool_use block with id, name, input."""
        converter = AnthropicHistoryConverter()
        raw = [
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "name": "get_weather",
                        "args": {"location": "NYC"},
                        "tool_call_id": "toolu_123",
                    }
                ),
                "message_type": "tool_call",
            }
        ]

        result = converter.convert(raw)

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        content = result[0]["content"]
        assert isinstance(content, list)
        assert len(content) == 1

        tool_use_block = content[0]
        assert tool_use_block["type"] == "tool_use"
        assert tool_use_block["id"] == "toolu_123"
        assert tool_use_block["name"] == "get_weather"
        assert tool_use_block["input"] == {"location": "NYC"}

    def test_tool_result_creates_tool_result_block_with_correct_structure(self):
        """Tool result should create tool_result block with tool_use_id, content."""
        converter = AnthropicHistoryConverter()
        raw = [
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "name": "get_weather",
                        "args": {"location": "NYC"},
                        "tool_call_id": "toolu_123",
                    }
                ),
                "message_type": "tool_call",
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "name": "get_weather",
                        "output": "Sunny, 72F",
                        "tool_call_id": "toolu_123",
                    }
                ),
                "message_type": "tool_result",
            },
        ]

        result = converter.convert(raw)

        assert len(result) == 2

        # Check tool_result block
        tool_result_msg = result[1]
        assert tool_result_msg["role"] == "user"
        content = tool_result_msg["content"]
        assert isinstance(content, list)
        assert len(content) == 1

        tool_result_block = content[0]
        assert tool_result_block["type"] == "tool_result"
        assert tool_result_block["tool_use_id"] == "toolu_123"
        assert tool_result_block["content"] == "Sunny, 72F"


class TestIsErrorFieldHandling:
    """Tests for is_error field handling in tool results."""

    def test_is_error_true_is_included_in_output(self):
        """is_error=True should be included in the tool_result block."""
        converter = AnthropicHistoryConverter()
        raw = [
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "name": "get_weather",
                        "args": {"location": "NYC"},
                        "tool_call_id": "toolu_123",
                    }
                ),
                "message_type": "tool_call",
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "name": "get_weather",
                        "output": "Error: API unavailable",
                        "tool_call_id": "toolu_123",
                        "is_error": True,
                    }
                ),
                "message_type": "tool_result",
            },
        ]

        result = converter.convert(raw)

        tool_result_block = result[1]["content"][0]
        assert "is_error" in tool_result_block
        assert tool_result_block["is_error"] is True

    def test_is_error_false_is_omitted_from_output(self):
        """is_error=False should be OMITTED from tool_result block (Anthropic API optimization)."""
        converter = AnthropicHistoryConverter()
        raw = [
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "name": "get_weather",
                        "args": {"location": "NYC"},
                        "tool_call_id": "toolu_123",
                    }
                ),
                "message_type": "tool_call",
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "name": "get_weather",
                        "output": "Sunny, 72F",
                        "tool_call_id": "toolu_123",
                        "is_error": False,
                    }
                ),
                "message_type": "tool_result",
            },
        ]

        result = converter.convert(raw)

        tool_result_block = result[1]["content"][0]
        # is_error should NOT be present when False
        assert "is_error" not in tool_result_block

    def test_is_error_missing_is_treated_as_false(self):
        """Missing is_error should be treated as False (omitted from output)."""
        converter = AnthropicHistoryConverter()
        raw = [
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "name": "get_weather",
                        "args": {"location": "NYC"},
                        "tool_call_id": "toolu_123",
                    }
                ),
                "message_type": "tool_call",
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "name": "get_weather",
                        "output": "Sunny, 72F",
                        "tool_call_id": "toolu_123",
                        # No is_error field
                    }
                ),
                "message_type": "tool_result",
            },
        ]

        result = converter.convert(raw)

        tool_result_block = result[1]["content"][0]
        assert "is_error" not in tool_result_block


class TestToolCallBatching:
    """Tests for tool call batching into single assistant message."""

    def test_consecutive_tool_calls_batched_into_single_message(self):
        """Multiple consecutive tool_calls should be batched into one assistant message."""
        converter = AnthropicHistoryConverter()
        raw = [
            {
                "role": "assistant",
                "content": json.dumps(
                    {"name": "tool1", "args": {}, "tool_call_id": "toolu_1"}
                ),
                "message_type": "tool_call",
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    {"name": "tool2", "args": {}, "tool_call_id": "toolu_2"}
                ),
                "message_type": "tool_call",
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    {"name": "tool3", "args": {}, "tool_call_id": "toolu_3"}
                ),
                "message_type": "tool_call",
            },
        ]

        result = converter.convert(raw)

        # Should be a single assistant message with 3 tool_use blocks
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        content = result[0]["content"]
        assert len(content) == 3
        assert all(block["type"] == "tool_use" for block in content)
        assert [block["id"] for block in content] == ["toolu_1", "toolu_2", "toolu_3"]


class TestToolResultBatching:
    """Tests for tool result batching into single user message."""

    def test_consecutive_tool_results_batched_into_single_message(self):
        """Multiple consecutive tool_results should be batched into one user message."""
        converter = AnthropicHistoryConverter()
        raw = [
            {
                "role": "assistant",
                "content": json.dumps(
                    {"name": "tool1", "args": {}, "tool_call_id": "toolu_1"}
                ),
                "message_type": "tool_call",
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    {"name": "tool2", "args": {}, "tool_call_id": "toolu_2"}
                ),
                "message_type": "tool_call",
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    {"name": "tool1", "output": "result1", "tool_call_id": "toolu_1"}
                ),
                "message_type": "tool_result",
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    {"name": "tool2", "output": "result2", "tool_call_id": "toolu_2"}
                ),
                "message_type": "tool_result",
            },
        ]

        result = converter.convert(raw)

        # Should be 2 messages: assistant (2 tool_use) + user (2 tool_result)
        assert len(result) == 2

        # Check tool results are batched
        tool_results_msg = result[1]
        assert tool_results_msg["role"] == "user"
        content = tool_results_msg["content"]
        assert len(content) == 2
        assert all(block["type"] == "tool_result" for block in content)
