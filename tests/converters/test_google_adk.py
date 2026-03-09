"""Tests for GoogleADKHistoryConverter.

Tests for shared converter behavior (empty input, own-message filtering,
sender formatting, tool event handling) live in
tests/framework_conformance/test_converter_conformance.py.
This file contains Google ADK-specific conversion behavior.
"""

from __future__ import annotations

import json

from thenvoi.converters.google_adk import GoogleADKHistoryConverter


class TestBasicConversion:
    """Tests for basic message conversion."""

    def test_empty_history(self):
        """Should return empty list for empty input."""
        converter = GoogleADKHistoryConverter()
        result = converter.convert([])
        assert result == []

    def test_single_user_message(self):
        """Should convert user text message."""
        converter = GoogleADKHistoryConverter()
        result = converter.convert(
            [
                {
                    "role": "user",
                    "content": "Hello",
                    "sender_name": "Alice",
                    "message_type": "text",
                }
            ]
        )
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "[Alice]: Hello"

    def test_multiple_user_messages(self):
        """Should convert multiple messages."""
        converter = GoogleADKHistoryConverter()
        result = converter.convert(
            [
                {
                    "role": "user",
                    "content": "Hello",
                    "sender_name": "Alice",
                    "message_type": "text",
                },
                {
                    "role": "user",
                    "content": "Hi there",
                    "sender_name": "Bob",
                    "message_type": "text",
                },
            ]
        )
        assert len(result) == 2
        assert "[Alice]: Hello" in result[0]["content"]
        assert "[Bob]: Hi there" in result[1]["content"]


class TestOwnMessageFiltering:
    """Tests for filtering this agent's own messages."""

    def test_filters_own_messages(self):
        """Should skip this agent's text messages."""
        converter = GoogleADKHistoryConverter(agent_name="TestBot")
        result = converter.convert(
            [
                {
                    "role": "assistant",
                    "content": "I'm the bot response",
                    "sender_name": "TestBot",
                    "message_type": "text",
                }
            ]
        )
        assert len(result) == 0

    def test_includes_other_agent_messages(self):
        """Should include messages from other agents."""
        converter = GoogleADKHistoryConverter(agent_name="TestBot")
        result = converter.convert(
            [
                {
                    "role": "assistant",
                    "content": "I'm another bot",
                    "sender_name": "OtherBot",
                    "message_type": "text",
                }
            ]
        )
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert "[OtherBot]" in result[0]["content"]

    def test_set_agent_name(self):
        """Should update agent name."""
        converter = GoogleADKHistoryConverter()
        converter.set_agent_name("TestBot")

        result = converter.convert(
            [
                {
                    "role": "assistant",
                    "content": "My response",
                    "sender_name": "TestBot",
                    "message_type": "text",
                }
            ]
        )
        assert len(result) == 0


class TestToolEvents:
    """Tests for tool call and tool result conversion."""

    def test_tool_call_conversion(self):
        """Should convert tool_call to model message with function_call block."""
        converter = GoogleADKHistoryConverter()
        result = converter.convert(
            [
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "name": "thenvoi_send_message",
                            "args": {"content": "Hello"},
                            "tool_call_id": "tc-1",
                        }
                    ),
                    "sender_name": "TestBot",
                    "message_type": "tool_call",
                }
            ]
        )
        assert len(result) == 1
        assert result[0]["role"] == "model"
        blocks = result[0]["content"]
        assert len(blocks) == 1
        assert blocks[0]["type"] == "function_call"
        assert blocks[0]["name"] == "thenvoi_send_message"
        assert blocks[0]["args"] == {"content": "Hello"}

    def test_tool_result_conversion(self):
        """Should convert tool_result to user message with function_response block."""
        converter = GoogleADKHistoryConverter()
        result = converter.convert(
            [
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "name": "thenvoi_send_message",
                            "output": '{"status": "sent"}',
                            "tool_call_id": "tc-1",
                        }
                    ),
                    "sender_name": "TestBot",
                    "message_type": "tool_result",
                }
            ]
        )
        assert len(result) == 1
        assert result[0]["role"] == "user"
        blocks = result[0]["content"]
        assert len(blocks) == 1
        assert blocks[0]["type"] == "function_response"
        assert blocks[0]["name"] == "thenvoi_send_message"

    def test_tool_call_and_result_pair(self):
        """Should batch tool call and result into separate messages."""
        converter = GoogleADKHistoryConverter()
        result = converter.convert(
            [
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "name": "search",
                            "args": {"q": "test"},
                            "tool_call_id": "tc-1",
                        }
                    ),
                    "sender_name": "TestBot",
                    "message_type": "tool_call",
                },
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "name": "search",
                            "output": "result data",
                            "tool_call_id": "tc-1",
                        }
                    ),
                    "sender_name": "TestBot",
                    "message_type": "tool_result",
                },
            ]
        )
        assert len(result) == 2
        assert result[0]["role"] == "model"
        assert result[1]["role"] == "user"

    def test_batches_multiple_tool_calls(self):
        """Should batch consecutive tool calls into single model message."""
        converter = GoogleADKHistoryConverter()
        result = converter.convert(
            [
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "name": "tool_a",
                            "args": {},
                            "tool_call_id": "tc-1",
                        }
                    ),
                    "sender_name": "TestBot",
                    "message_type": "tool_call",
                },
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "name": "tool_b",
                            "args": {},
                            "tool_call_id": "tc-2",
                        }
                    ),
                    "sender_name": "TestBot",
                    "message_type": "tool_call",
                },
            ]
        )
        assert len(result) == 1
        assert result[0]["role"] == "model"
        assert len(result[0]["content"]) == 2

    def test_error_tool_result(self):
        """Should include is_error flag on error results."""
        converter = GoogleADKHistoryConverter()
        result = converter.convert(
            [
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "name": "failing_tool",
                            "output": "Error: tool failed",
                            "tool_call_id": "tc-1",
                            "is_error": True,
                        }
                    ),
                    "sender_name": "TestBot",
                    "message_type": "tool_result",
                }
            ]
        )
        assert len(result) == 1
        blocks = result[0]["content"]
        assert blocks[0]["is_error"] is True


class TestEdgeCases:
    """Tests for edge cases."""

    def test_missing_sender_name(self):
        """Should handle missing sender_name."""
        converter = GoogleADKHistoryConverter()
        result = converter.convert(
            [
                {
                    "role": "user",
                    "content": "Hello",
                    "message_type": "text",
                }
            ]
        )
        assert len(result) == 1
        assert result[0]["content"] == "Hello"

    def test_empty_sender_name(self):
        """Should handle empty sender_name."""
        converter = GoogleADKHistoryConverter()
        result = converter.convert(
            [
                {
                    "role": "user",
                    "content": "Hello",
                    "sender_name": "",
                    "message_type": "text",
                }
            ]
        )
        assert len(result) == 1
        assert result[0]["content"] == "Hello"

    def test_skips_thought_messages(self):
        """Should skip thought message types."""
        converter = GoogleADKHistoryConverter()
        result = converter.convert(
            [
                {
                    "role": "assistant",
                    "content": "Thinking about the problem...",
                    "sender_name": "TestBot",
                    "message_type": "thought",
                }
            ]
        )
        assert len(result) == 0

    def test_skips_error_messages(self):
        """Should skip error message types."""
        converter = GoogleADKHistoryConverter()
        result = converter.convert(
            [
                {
                    "role": "assistant",
                    "content": "Error: something went wrong",
                    "sender_name": "TestBot",
                    "message_type": "error",
                }
            ]
        )
        assert len(result) == 0

    def test_malformed_tool_call_json(self):
        """Should skip malformed tool call JSON."""
        converter = GoogleADKHistoryConverter()
        result = converter.convert(
            [
                {
                    "role": "assistant",
                    "content": "not json",
                    "sender_name": "TestBot",
                    "message_type": "tool_call",
                }
            ]
        )
        assert len(result) == 0

    def test_mixed_messages_and_tools(self):
        """Should handle interleaved text messages and tool events."""
        converter = GoogleADKHistoryConverter()
        result = converter.convert(
            [
                {
                    "role": "user",
                    "content": "Search for cats",
                    "sender_name": "Alice",
                    "message_type": "text",
                },
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "name": "search",
                            "args": {"q": "cats"},
                            "tool_call_id": "tc-1",
                        }
                    ),
                    "sender_name": "TestBot",
                    "message_type": "tool_call",
                },
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "name": "search",
                            "output": "Found: cats are great",
                            "tool_call_id": "tc-1",
                        }
                    ),
                    "sender_name": "TestBot",
                    "message_type": "tool_result",
                },
                {
                    "role": "user",
                    "content": "Thanks!",
                    "sender_name": "Alice",
                    "message_type": "text",
                },
            ]
        )
        assert len(result) == 4
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "model"
        assert result[2]["role"] == "user"
        assert result[3]["role"] == "user"
