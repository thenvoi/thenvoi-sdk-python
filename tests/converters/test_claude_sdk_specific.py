"""Claude SDK converter-specific tests.

Tests for Claude SDK-specific behavior that isn't covered by conformance tests:
- String output format (newline-joined, not structured messages)
- Raw JSON inclusion for tool events
- Default sender name handling
"""

from __future__ import annotations

import json

from thenvoi.converters.claude_sdk import ClaudeSDKHistoryConverter


class TestStringOutputFormat:
    """Tests for string output format."""

    def test_converts_multiple_messages_with_newlines(self):
        """Multiple messages should be joined with newlines."""
        converter = ClaudeSDKHistoryConverter()
        raw = [
            {
                "role": "user",
                "content": "Hello",
                "sender_name": "Alice",
                "message_type": "text",
            },
            {
                "role": "user",
                "content": "How are you?",
                "sender_name": "Bob",
                "message_type": "text",
            },
            {
                "role": "assistant",
                "content": "I'm fine!",
                "sender_name": "Other Agent",
                "message_type": "text",
            },
        ]

        result = converter.convert(raw)

        # Should be a string with newline separators
        assert isinstance(result, str)
        lines = result.split("\n")
        assert len(lines) == 3
        assert lines[0] == "[Alice]: Hello"
        assert lines[1] == "[Bob]: How are you?"
        assert lines[2] == "[Other Agent]: I'm fine!"

    def test_preserves_message_order(self):
        """Messages should maintain their original order."""
        converter = ClaudeSDKHistoryConverter()
        raw = [
            {
                "role": "user",
                "content": "First",
                "sender_name": "A",
                "message_type": "text",
            },
            {
                "role": "user",
                "content": "Second",
                "sender_name": "B",
                "message_type": "text",
            },
            {
                "role": "user",
                "content": "Third",
                "sender_name": "C",
                "message_type": "text",
            },
        ]

        result = converter.convert(raw)

        lines = result.split("\n")
        assert "First" in lines[0]
        assert "Second" in lines[1]
        assert "Third" in lines[2]

    def test_empty_history_returns_empty_string(self):
        """Empty history should return empty string."""
        converter = ClaudeSDKHistoryConverter()
        result = converter.convert([])
        assert result == ""


class TestToolEventRawJson:
    """Tests for raw JSON inclusion of tool events."""

    def test_includes_tool_call_as_raw_json(self):
        """Tool call content should be included as raw JSON string."""
        converter = ClaudeSDKHistoryConverter()
        tool_call_json = json.dumps(
            {
                "name": "get_weather",
                "args": {"location": "NYC"},
                "tool_call_id": "toolu_123",
            }
        )
        raw = [
            {
                "role": "assistant",
                "content": tool_call_json,
                "message_type": "tool_call",
            }
        ]

        result = converter.convert(raw)

        # Should include the raw JSON string
        assert tool_call_json in result

    def test_includes_tool_result_as_raw_json(self):
        """Tool result content should be included as raw JSON string."""
        converter = ClaudeSDKHistoryConverter()
        tool_result_json = json.dumps(
            {
                "name": "get_weather",
                "output": "Sunny, 72F",
                "tool_call_id": "toolu_123",
            }
        )
        raw = [
            {
                "role": "assistant",
                "content": tool_result_json,
                "message_type": "tool_result",
            }
        ]

        result = converter.convert(raw)

        assert tool_result_json in result

    def test_tool_events_included_regardless_of_agent_name(self):
        """Tool events should be included even when from own agent."""
        converter = ClaudeSDKHistoryConverter(agent_name="MyAgent")
        tool_call_json = json.dumps(
            {"name": "search", "args": {}, "tool_call_id": "toolu_1"}
        )
        raw = [
            {
                "role": "assistant",
                "content": tool_call_json,
                "sender_name": "MyAgent",  # Own agent
                "message_type": "tool_call",
            }
        ]

        result = converter.convert(raw)

        # Tool events should still be included
        assert tool_call_json in result

    def test_mixed_text_and_tool_events(self):
        """Mixed text and tool events should all be included in order."""
        converter = ClaudeSDKHistoryConverter()
        tool_call = json.dumps({"name": "search", "args": {}, "tool_call_id": "t1"})
        tool_result = json.dumps({"output": "found it", "tool_call_id": "t1"})
        raw = [
            {
                "role": "user",
                "content": "Search for X",
                "sender_name": "Alice",
                "message_type": "text",
            },
            {
                "role": "assistant",
                "content": tool_call,
                "message_type": "tool_call",
            },
            {
                "role": "assistant",
                "content": tool_result,
                "message_type": "tool_result",
            },
        ]

        result = converter.convert(raw)
        lines = result.split("\n")

        assert len(lines) == 3
        assert "[Alice]: Search for X" in lines[0]
        assert tool_call == lines[1]
        assert tool_result == lines[2]


class TestEmptyContentHandling:
    """Tests for empty content handling."""

    def test_skips_empty_tool_call_content(self):
        """Empty tool_call content should be skipped."""
        converter = ClaudeSDKHistoryConverter()
        raw = [
            {
                "role": "assistant",
                "content": "",
                "message_type": "tool_call",
            }
        ]

        result = converter.convert(raw)

        assert result == ""

    def test_skips_empty_tool_result_content(self):
        """Empty tool_result content should be skipped."""
        converter = ClaudeSDKHistoryConverter()
        raw = [
            {
                "role": "assistant",
                "content": "",
                "message_type": "tool_result",
            }
        ]

        result = converter.convert(raw)

        assert result == ""

    def test_skips_empty_text_content(self):
        """Empty text content should be skipped."""
        converter = ClaudeSDKHistoryConverter()
        raw = [
            {
                "role": "user",
                "content": "",
                "sender_name": "Alice",
                "message_type": "text",
            }
        ]

        result = converter.convert(raw)

        assert result == ""


class TestSenderNameDefaults:
    """Tests for sender name default handling."""

    def test_defaults_to_unknown_sender(self):
        """Missing sender_name should default to 'Unknown'."""
        converter = ClaudeSDKHistoryConverter()
        raw = [
            {
                "role": "user",
                "content": "Hello",
                "message_type": "text",
                # No sender_name
            }
        ]

        result = converter.convert(raw)

        assert "[Unknown]: Hello" in result
