"""Tests for ClaudeSDKHistoryConverter."""

import json

from thenvoi.converters.claude_sdk import ClaudeSDKHistoryConverter


class TestBasicConversion:
    """Tests for basic message conversion to text."""

    def test_converts_single_user_message(self):
        """Single user message is formatted as [sender]: content."""
        converter = ClaudeSDKHistoryConverter()
        raw = [
            {
                "role": "user",
                "content": "Hello, agent!",
                "sender_name": "Alice",
                "message_type": "text",
            }
        ]

        result = converter.convert(raw)

        assert result == "[Alice]: Hello, agent!"

    def test_converts_multiple_messages(self):
        """Multiple messages are joined with newlines."""
        converter = ClaudeSDKHistoryConverter()
        raw = [
            {
                "role": "user",
                "content": "Hello!",
                "sender_name": "Alice",
                "message_type": "text",
            },
            {
                "role": "user",
                "content": "Hi there!",
                "sender_name": "Bob",
                "message_type": "text",
            },
        ]

        result = converter.convert(raw)

        expected = "[Alice]: Hello!\n[Bob]: Hi there!"
        assert result == expected


class TestAgentNameHandling:
    """Tests for agent_name filtering."""

    def test_skips_own_text_messages(self):
        """Should skip this agent's text messages."""
        converter = ClaudeSDKHistoryConverter(agent_name="MyAgent")
        raw = [
            {
                "role": "user",
                "content": "Hello!",
                "sender_name": "Alice",
                "message_type": "text",
            },
            {
                "role": "assistant",
                "content": "Hi there!",
                "sender_name": "MyAgent",
                "message_type": "text",
            },
        ]

        result = converter.convert(raw)

        assert result == "[Alice]: Hello!"
        assert "Hi there!" not in result

    def test_includes_other_agents_messages(self):
        """Should include other agents' messages."""
        converter = ClaudeSDKHistoryConverter(agent_name="MyAgent")
        raw = [
            {
                "role": "assistant",
                "content": "I can help too!",
                "sender_name": "OtherAgent",
                "message_type": "text",
            },
        ]

        result = converter.convert(raw)

        assert result == "[OtherAgent]: I can help too!"

    def test_set_agent_name(self):
        """set_agent_name should update filtering."""
        converter = ClaudeSDKHistoryConverter()
        converter.set_agent_name("MyAgent")
        raw = [
            {
                "role": "assistant",
                "content": "My message",
                "sender_name": "MyAgent",
                "message_type": "text",
            },
        ]

        result = converter.convert(raw)

        assert result == ""

    def test_includes_all_when_no_agent_name(self):
        """Should include all assistant messages when no agent_name set."""
        converter = ClaudeSDKHistoryConverter()
        raw = [
            {
                "role": "assistant",
                "content": "Hello!",
                "sender_name": "Agent",
                "message_type": "text",
            },
        ]

        result = converter.convert(raw)

        assert result == "[Agent]: Hello!"


class TestToolEventHandling:
    """Tests for tool_call and tool_result inclusion."""

    def test_includes_tool_call_as_raw_json(self):
        """tool_call messages are included as raw JSON."""
        converter = ClaudeSDKHistoryConverter(agent_name="MyAgent")
        tool_call_json = json.dumps(
            {
                "name": "get_weather",
                "args": {"city": "NYC"},
                "tool_call_id": "toolu_123",
            }
        )
        raw = [
            {
                "role": "user",
                "content": "Hello",
                "sender_name": "User",
                "message_type": "text",
            },
            {
                "role": "assistant",
                "content": tool_call_json,
                "message_type": "tool_call",
            },
        ]

        result = converter.convert(raw)

        assert "[User]: Hello" in result
        # Raw JSON is included as-is
        assert tool_call_json in result

    def test_includes_tool_result_as_raw_json(self):
        """tool_result messages are included as raw JSON."""
        converter = ClaudeSDKHistoryConverter(agent_name="MyAgent")
        tool_result_json = json.dumps(
            {"output": "sunny, 72F", "tool_call_id": "toolu_123"}
        )
        raw = [
            {
                "role": "assistant",
                "content": tool_result_json,
                "message_type": "tool_result",
            }
        ]

        result = converter.convert(raw)

        # Raw JSON is included as-is
        assert result == tool_result_json

    def test_tool_events_included_regardless_of_agent_name(self):
        """Tool events are included as raw JSON regardless of agent_name."""
        converter = ClaudeSDKHistoryConverter(agent_name="WeatherBot")
        tool_call_json = json.dumps({"name": "search", "args": {}})
        raw = [
            {
                "content": tool_call_json,
                "message_type": "tool_call",
            },
        ]

        result = converter.convert(raw)

        assert result == tool_call_json

    def test_skips_empty_tool_call_content(self):
        """Should skip tool_call with empty content."""
        converter = ClaudeSDKHistoryConverter()
        raw = [
            {
                "content": "",
                "message_type": "tool_call",
            },
        ]

        result = converter.convert(raw)

        assert result == ""

    def test_skips_empty_tool_result_content(self):
        """Should skip tool_result with empty content."""
        converter = ClaudeSDKHistoryConverter()
        raw = [
            {
                "content": "",
                "message_type": "tool_result",
            },
        ]

        result = converter.convert(raw)

        assert result == ""

    def test_skips_thought_messages(self):
        """thought messages are skipped."""
        converter = ClaudeSDKHistoryConverter()
        raw = [
            {
                "role": "assistant",
                "content": "I'm thinking about this...",
                "message_type": "thought",
            }
        ]

        result = converter.convert(raw)

        assert result == ""


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_history(self):
        """Empty history returns empty string."""
        converter = ClaudeSDKHistoryConverter()

        result = converter.convert([])

        assert result == ""

    def test_defaults_to_text_message_type(self):
        """Messages without message_type default to 'text'."""
        converter = ClaudeSDKHistoryConverter()
        raw = [
            {
                "content": "Hello",
                "sender_name": "Bob",
            }
        ]

        result = converter.convert(raw)

        assert result == "[Bob]: Hello"

    def test_defaults_to_unknown_sender(self):
        """Messages without sender_name use 'Unknown'."""
        converter = ClaudeSDKHistoryConverter()
        raw = [
            {
                "content": "Hello",
                "message_type": "text",
            }
        ]

        result = converter.convert(raw)

        assert result == "[Unknown]: Hello"

    def test_skips_empty_content(self):
        """Messages with empty content are skipped."""
        converter = ClaudeSDKHistoryConverter()
        raw = [
            {
                "content": "",
                "sender_name": "Alice",
                "message_type": "text",
            }
        ]

        result = converter.convert(raw)

        assert result == ""


class TestMixedHistory:
    """Integration tests with mixed message types."""

    def test_full_conversation_flow(self):
        """Should handle a realistic conversation with mixed message types."""
        converter = ClaudeSDKHistoryConverter(agent_name="WeatherAgent")
        tool_call_json = json.dumps(
            {
                "name": "get_weather",
                "args": {"city": "NYC"},
                "tool_call_id": "toolu_123",
            }
        )
        tool_result_json = json.dumps(
            {"output": "sunny, 72F", "tool_call_id": "toolu_123"}
        )
        raw = [
            # User asks question
            {
                "role": "user",
                "content": "What's the weather?",
                "sender_name": "Alice",
                "message_type": "text",
            },
            # Agent uses tool
            {
                "role": "assistant",
                "content": tool_call_json,
                "message_type": "tool_call",
            },
            # Tool result
            {
                "role": "assistant",
                "content": tool_result_json,
                "message_type": "tool_result",
            },
            # Agent responds with text (skipped - redundant)
            {
                "role": "assistant",
                "content": "It's sunny today!",
                "sender_name": "WeatherAgent",
                "message_type": "text",
            },
            # User follow-up
            {
                "role": "user",
                "content": "Thanks!",
                "sender_name": "Alice",
                "message_type": "text",
            },
        ]

        result = converter.convert(raw)

        # Should include user messages
        assert "[Alice]: What's the weather?" in result
        assert "[Alice]: Thanks!" in result
        # Should include raw tool event JSON
        assert tool_call_json in result
        assert tool_result_json in result
        # Should NOT include own text message
        assert "It's sunny today!" not in result

    def test_multi_agent_conversation(self):
        """Handles multiple agents in conversation."""
        converter = ClaudeSDKHistoryConverter(agent_name="MyAgent")
        raw = [
            {
                "role": "user",
                "content": "Hi team!",
                "sender_name": "Alice",
                "message_type": "text",
            },
            {
                "role": "assistant",
                "content": "Hello!",
                "sender_name": "OtherAgent",
                "message_type": "text",
            },
            {
                "role": "assistant",
                "content": "Hi Alice!",
                "sender_name": "MyAgent",
                "message_type": "text",
            },
        ]

        result = converter.convert(raw)

        assert "[Alice]: Hi team!" in result
        assert "[OtherAgent]: Hello!" in result
        # Own message skipped
        assert "Hi Alice!" not in result

    def test_preserves_message_order(self):
        """Messages appear in the same order as input."""
        converter = ClaudeSDKHistoryConverter()
        raw = [
            {"content": "First", "sender_name": "A", "message_type": "text"},
            {"content": "Second", "sender_name": "B", "message_type": "text"},
            {"content": "Third", "sender_name": "C", "message_type": "text"},
        ]

        result = converter.convert(raw)
        lines = result.split("\n")

        assert len(lines) == 3
        assert "First" in lines[0]
        assert "Second" in lines[1]
        assert "Third" in lines[2]
