"""Tests for platform history parser."""

import json

from thenvoi.integrations.history import (
    NormalizedToolExchange,
    NormalizedUserText,
    parse_platform_history,
)


class TestParseUserText:
    """Tests for user text message parsing."""

    def test_parses_user_text_message(self):
        """User text messages become NormalizedUserText."""
        history = [
            {
                "message_type": "text",
                "role": "user",
                "sender_name": "Alice",
                "content": "Hello!",
            }
        ]

        result = parse_platform_history(history)

        assert len(result) == 1
        assert isinstance(result[0], NormalizedUserText)
        assert result[0].sender_name == "Alice"
        assert result[0].content == "Hello!"

    def test_skips_assistant_text_messages(self):
        """Assistant text messages are skipped (redundant with tool calls)."""
        history = [
            {
                "message_type": "text",
                "role": "assistant",
                "sender_name": "Bot",
                "content": "I'll help you with that.",
            }
        ]

        result = parse_platform_history(history)

        assert len(result) == 0

    def test_uses_default_sender_name(self):
        """Missing sender_name defaults to 'Unknown'."""
        history = [
            {
                "message_type": "text",
                "role": "user",
                "content": "Hello!",
            }
        ]

        result = parse_platform_history(history)

        assert result[0].sender_name == "Unknown"


class TestParseToolExchanges:
    """Tests for tool call/result pairing."""

    def test_pairs_tool_call_and_result_by_run_id(self):
        """Tool call and result with same run_id are paired."""
        history = [
            {
                "message_type": "tool_call",
                "content": json.dumps(
                    {
                        "name": "get_weather",
                        "run_id": "run-123",
                        "data": {"input": {"city": "NYC"}},
                    }
                ),
            },
            {
                "message_type": "tool_result",
                "content": json.dumps(
                    {
                        "name": "get_weather",
                        "run_id": "run-123",
                        "data": {"output": "Sunny, 72°F"},
                    }
                ),
            },
        ]

        result = parse_platform_history(history)

        assert len(result) == 1
        assert isinstance(result[0], NormalizedToolExchange)
        assert result[0].tool_name == "get_weather"
        assert result[0].tool_id == "run-123"
        assert result[0].input_args == {"city": "NYC"}
        assert result[0].output == "Sunny, 72°F"
        assert result[0].is_error is False

    def test_handles_out_of_order_results(self):
        """Results arriving out of order still match correctly by run_id."""
        history = [
            {
                "message_type": "tool_call",
                "content": json.dumps(
                    {"name": "tool_a", "run_id": "run-aaa", "data": {"input": {"x": 1}}}
                ),
            },
            {
                "message_type": "tool_call",
                "content": json.dumps(
                    {"name": "tool_b", "run_id": "run-bbb", "data": {"input": {"y": 2}}}
                ),
            },
            # Results arrive in reverse order
            {
                "message_type": "tool_result",
                "content": json.dumps(
                    {"name": "tool_b", "run_id": "run-bbb", "data": {"output": "B"}}
                ),
            },
            {
                "message_type": "tool_result",
                "content": json.dumps(
                    {"name": "tool_a", "run_id": "run-aaa", "data": {"output": "A"}}
                ),
            },
        ]

        result = parse_platform_history(history)

        assert len(result) == 2
        # First exchange is tool_b (result arrived first)
        assert result[0].tool_name == "tool_b"
        assert result[0].output == "B"
        # Second exchange is tool_a
        assert result[1].tool_name == "tool_a"
        assert result[1].output == "A"

    def test_handles_back_to_back_same_tool_calls(self):
        """Multiple calls to same tool match correctly by run_id."""
        history = [
            {
                "message_type": "tool_call",
                "content": json.dumps(
                    {
                        "name": "send_message",
                        "run_id": "run-111",
                        "data": {"input": {"text": "First"}},
                    }
                ),
            },
            {
                "message_type": "tool_call",
                "content": json.dumps(
                    {
                        "name": "send_message",
                        "run_id": "run-222",
                        "data": {"input": {"text": "Second"}},
                    }
                ),
            },
            {
                "message_type": "tool_result",
                "content": json.dumps(
                    {
                        "name": "send_message",
                        "run_id": "run-222",
                        "data": {"output": "ok"},
                    }
                ),
            },
            {
                "message_type": "tool_result",
                "content": json.dumps(
                    {
                        "name": "send_message",
                        "run_id": "run-111",
                        "data": {"output": "ok"},
                    }
                ),
            },
        ]

        result = parse_platform_history(history)

        assert len(result) == 2
        assert result[0].input_args == {"text": "Second"}  # run-222 first
        assert result[1].input_args == {"text": "First"}  # run-111 second

    def test_fallback_to_name_matching_without_run_id(self):
        """Falls back to LIFO name matching when run_id is missing."""
        history = [
            {
                "message_type": "tool_call",
                "content": json.dumps(
                    {"name": "lookup", "data": {"input": {"query": "test"}}}
                ),
            },
            {
                "message_type": "tool_result",
                "content": json.dumps(
                    {"name": "lookup", "data": {"output": "found it"}}
                ),
            },
        ]

        result = parse_platform_history(history)

        assert len(result) == 1
        assert result[0].tool_name == "lookup"
        assert result[0].tool_id == "tool_lookup"  # Generated ID
        assert result[0].output == "found it"

    def test_skips_unmatched_tool_result(self):
        """Tool result without matching call is skipped."""
        history = [
            {
                "message_type": "tool_result",
                "content": json.dumps(
                    {"name": "orphan", "run_id": "orphan-run", "data": {"output": "?"}}
                ),
            },
        ]

        result = parse_platform_history(history)

        assert len(result) == 0

    def test_unmatched_tool_calls_dont_appear_in_output(self):
        """Tool calls without matching results don't produce output."""
        history = [
            {
                "message_type": "tool_call",
                "content": json.dumps(
                    {"name": "pending", "run_id": "pending-run", "data": {"input": {}}}
                ),
            },
        ]

        result = parse_platform_history(history)

        assert len(result) == 0

    def test_handles_error_results(self):
        """Tool results with is_error flag are preserved."""
        history = [
            {
                "message_type": "tool_call",
                "content": json.dumps(
                    {"name": "failing_tool", "run_id": "run-err", "data": {"input": {}}}
                ),
            },
            {
                "message_type": "tool_result",
                "content": json.dumps(
                    {
                        "name": "failing_tool",
                        "run_id": "run-err",
                        "data": {
                            "output": "Error: something went wrong",
                            "is_error": True,
                        },
                    }
                ),
            },
        ]

        result = parse_platform_history(history)

        assert len(result) == 1
        assert result[0].is_error is True
        assert "Error:" in result[0].output


class TestParseMixedHistory:
    """Tests for parsing mixed message types."""

    def test_handles_realistic_conversation(self):
        """Parses a realistic conversation with text and tools."""
        history = [
            {
                "message_type": "text",
                "role": "user",
                "sender_name": "Alice",
                "content": "What's the weather in NYC?",
            },
            {
                "message_type": "tool_call",
                "content": json.dumps(
                    {
                        "name": "get_weather",
                        "run_id": "run-weather",
                        "data": {"input": {"city": "NYC"}},
                    }
                ),
            },
            {
                "message_type": "tool_result",
                "content": json.dumps(
                    {
                        "name": "get_weather",
                        "run_id": "run-weather",
                        "data": {"output": "Sunny, 72°F"},
                    }
                ),
            },
            {
                "message_type": "tool_call",
                "content": json.dumps(
                    {
                        "name": "send_message",
                        "run_id": "run-msg",
                        "data": {"input": {"content": "It's sunny in NYC!"}},
                    }
                ),
            },
            {
                "message_type": "tool_result",
                "content": json.dumps(
                    {
                        "name": "send_message",
                        "run_id": "run-msg",
                        "data": {"output": "sent"},
                    }
                ),
            },
        ]

        result = parse_platform_history(history)

        assert len(result) == 3
        assert isinstance(result[0], NormalizedUserText)
        assert result[0].content == "What's the weather in NYC?"
        assert isinstance(result[1], NormalizedToolExchange)
        assert result[1].tool_name == "get_weather"
        assert isinstance(result[2], NormalizedToolExchange)
        assert result[2].tool_name == "send_message"

    def test_skips_other_message_types(self):
        """Non-text, non-tool message types are skipped."""
        history = [
            {"message_type": "thought", "content": "Thinking..."},
            {"message_type": "error", "content": "An error occurred"},
            {"message_type": "task", "content": "Task created"},
            {
                "message_type": "text",
                "role": "user",
                "sender_name": "Bob",
                "content": "Hello",
            },
        ]

        result = parse_platform_history(history)

        assert len(result) == 1
        assert result[0].content == "Hello"


class TestParseEdgeCases:
    """Tests for edge cases and error handling."""

    def test_handles_empty_history(self):
        """Empty history returns empty list."""
        result = parse_platform_history([])
        assert result == []

    def test_handles_malformed_tool_call_json(self):
        """Malformed JSON in tool_call is skipped."""
        history = [
            {"message_type": "tool_call", "content": "not valid json"},
        ]

        result = parse_platform_history(history)

        assert len(result) == 0

    def test_handles_malformed_tool_result_json(self):
        """Malformed JSON in tool_result is skipped."""
        history = [
            {"message_type": "tool_result", "content": "{broken json"},
        ]

        result = parse_platform_history(history)

        assert len(result) == 0

    def test_handles_missing_data_fields(self):
        """Missing optional fields don't cause errors."""
        history = [
            {
                "message_type": "tool_call",
                "content": json.dumps({"name": "simple", "run_id": "run-simple"}),
            },
            {
                "message_type": "tool_result",
                "content": json.dumps({"name": "simple", "run_id": "run-simple"}),
            },
        ]

        result = parse_platform_history(history)

        assert len(result) == 1
        assert result[0].input_args == {}
        assert result[0].output == ""
