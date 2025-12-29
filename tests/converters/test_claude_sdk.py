"""Tests for ClaudeSDKHistoryConverter."""

from thenvoi.converters.claude_sdk import ClaudeSDKHistoryConverter


class TestBasicConversion:
    """Tests for basic message conversion to text."""

    def test_converts_single_user_message(self):
        """Single user message is formatted with header and room context."""
        converter = ClaudeSDKHistoryConverter(room_id="room-123")
        raw = [
            {
                "role": "user",
                "content": "Hello, agent!",
                "sender_name": "Alice",
                "message_type": "text",
            }
        ]

        result = converter.convert(raw)

        assert (
            result
            == "Previous conversation history:\n[room_id: room-123][Alice]: Hello, agent!"
        )

    def test_converts_multiple_messages(self):
        """Multiple messages are joined with newlines."""
        converter = ClaudeSDKHistoryConverter(room_id="room-123")
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
                "sender_name": "Agent",
                "message_type": "text",
            },
        ]

        result = converter.convert(raw)

        expected = (
            "Previous conversation history:\n"
            "[room_id: room-123][Alice]: Hello!\n"
            "[room_id: room-123][Agent]: Hi there!"
        )
        assert result == expected


class TestRoomIdHandling:
    """Tests for room_id configuration."""

    def test_includes_room_id_when_provided(self):
        """Room ID is included in each line when provided."""
        converter = ClaudeSDKHistoryConverter(room_id="test-room")
        raw = [{"content": "Hello", "sender_name": "User", "message_type": "text"}]

        result = converter.convert(raw)

        assert "[room_id: test-room]" in result

    def test_omits_room_id_when_empty(self):
        """Room ID prefix is omitted when empty string."""
        converter = ClaudeSDKHistoryConverter(room_id="")
        raw = [{"content": "Hello", "sender_name": "User", "message_type": "text"}]

        result = converter.convert(raw)

        assert "[room_id:" not in result
        assert result == "Previous conversation history:\n[User]: Hello"

    def test_default_empty_room_id(self):
        """Default room_id is empty string."""
        converter = ClaudeSDKHistoryConverter()
        raw = [{"content": "Hello", "sender_name": "User", "message_type": "text"}]

        result = converter.convert(raw)

        assert "[room_id:" not in result


class TestToolEventFiltering:
    """Tests for tool_call and tool_result filtering."""

    def test_skips_tool_call_messages(self):
        """tool_call messages are skipped."""
        converter = ClaudeSDKHistoryConverter(room_id="room-123")
        raw = [
            {
                "role": "user",
                "content": "Hello",
                "sender_name": "User",
                "message_type": "text",
            },
            {
                "role": "assistant",
                "content": '{"event": "on_tool_start", "name": "search"}',
                "message_type": "tool_call",
            },
        ]

        result = converter.convert(raw)

        assert "on_tool_start" not in result
        assert "search" not in result

    def test_skips_tool_result_messages(self):
        """tool_result messages are skipped."""
        converter = ClaudeSDKHistoryConverter(room_id="room-123")
        raw = [
            {
                "role": "assistant",
                "content": '{"event": "on_tool_end", "output": "result"}',
                "message_type": "tool_result",
            }
        ]

        result = converter.convert(raw)

        assert result == ""

    def test_skips_thought_messages(self):
        """thought messages are skipped."""
        converter = ClaudeSDKHistoryConverter(room_id="room-123")
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
        converter = ClaudeSDKHistoryConverter(room_id="room-123")

        result = converter.convert([])

        assert result == ""

    def test_defaults_to_text_message_type(self):
        """Messages without message_type default to 'text'."""
        converter = ClaudeSDKHistoryConverter(room_id="room-123")
        raw = [
            {
                "content": "Hello",
                "sender_name": "Bob",
            }
        ]

        result = converter.convert(raw)

        assert "[Bob]: Hello" in result

    def test_defaults_to_unknown_sender(self):
        """Messages without sender_name use 'Unknown'."""
        converter = ClaudeSDKHistoryConverter(room_id="room-123")
        raw = [
            {
                "content": "Hello",
                "message_type": "text",
            }
        ]

        result = converter.convert(raw)

        assert "[Unknown]: Hello" in result

    def test_skips_empty_content(self):
        """Messages with empty content are skipped."""
        converter = ClaudeSDKHistoryConverter(room_id="room-123")
        raw = [
            {
                "content": "",
                "sender_name": "Alice",
                "message_type": "text",
            }
        ]

        result = converter.convert(raw)

        assert result == ""

    def test_returns_empty_when_only_tool_events(self):
        """Returns empty string when history only contains tool events."""
        converter = ClaudeSDKHistoryConverter(room_id="room-123")
        raw = [
            {"content": "tool call", "message_type": "tool_call"},
            {"content": "tool result", "message_type": "tool_result"},
        ]

        result = converter.convert(raw)

        assert result == ""


class TestMixedHistory:
    """Integration tests with mixed message types."""

    def test_full_conversation_flow(self):
        """Should handle a realistic conversation with mixed message types."""
        converter = ClaudeSDKHistoryConverter(room_id="room-123")
        raw = [
            # User asks question
            {
                "role": "user",
                "content": "What's the weather?",
                "sender_name": "Alice",
                "message_type": "text",
            },
            # Agent uses tool (skipped)
            {
                "role": "assistant",
                "content": '{"event": "on_tool_start", "name": "get_weather"}',
                "message_type": "tool_call",
            },
            # Tool result (skipped)
            {
                "role": "assistant",
                "content": '{"event": "on_tool_end", "output": "sunny"}',
                "message_type": "tool_result",
            },
            # Agent responds with text
            {
                "role": "assistant",
                "content": "It's sunny today!",
                "sender_name": "Agent",
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

        expected = (
            "Previous conversation history:\n"
            "[room_id: room-123][Alice]: What's the weather?\n"
            "[room_id: room-123][Agent]: It's sunny today!\n"
            "[room_id: room-123][Alice]: Thanks!"
        )
        assert result == expected

    def test_multi_user_conversation(self):
        """Handles multiple users in conversation."""
        converter = ClaudeSDKHistoryConverter(room_id="room-456")
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

        result = converter.convert(raw)

        assert "[Alice]: Hi team!" in result
        assert "[Bob]: Hello everyone!" in result
        assert "[Agent]: Hello Alice and Bob!" in result

    def test_preserves_message_order(self):
        """Messages appear in the same order as input."""
        converter = ClaudeSDKHistoryConverter(room_id="room-123")
        raw = [
            {"content": "First", "sender_name": "A", "message_type": "text"},
            {"content": "Second", "sender_name": "B", "message_type": "text"},
            {"content": "Third", "sender_name": "C", "message_type": "text"},
        ]

        result = converter.convert(raw)
        lines = result.split("\n")

        # First line is header
        assert "First" in lines[1]
        assert "Second" in lines[2]
        assert "Third" in lines[3]
