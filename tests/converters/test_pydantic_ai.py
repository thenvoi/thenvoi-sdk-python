"""Tests for PydanticAIHistoryConverter."""

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    UserPromptPart,
    TextPart,
)

from thenvoi.converters.pydantic_ai import PydanticAIHistoryConverter


class TestUserMessages:
    """Tests for user message conversion."""

    def test_converts_user_text_to_model_request(self):
        """User text messages become ModelRequest with UserPromptPart."""
        converter = PydanticAIHistoryConverter()
        raw = [
            {
                "role": "user",
                "content": "Hello, agent!",
                "sender_name": "Alice",
                "message_type": "text",
            }
        ]

        result = converter.convert(raw)

        assert len(result) == 1
        assert isinstance(result[0], ModelRequest)
        assert len(result[0].parts) == 1
        assert isinstance(result[0].parts[0], UserPromptPart)
        assert result[0].parts[0].content == "[Alice]: Hello, agent!"

    def test_handles_empty_sender_name(self):
        """User messages without sender_name use content as-is."""
        converter = PydanticAIHistoryConverter()
        raw = [
            {
                "role": "user",
                "content": "Hello!",
                "sender_name": "",
                "message_type": "text",
            }
        ]

        result = converter.convert(raw)

        assert isinstance(result[0], ModelRequest)
        assert result[0].parts[0].content == "Hello!"

    def test_handles_missing_sender_name(self):
        """User messages with no sender_name key use content as-is."""
        converter = PydanticAIHistoryConverter()
        raw = [
            {
                "role": "user",
                "content": "Hello!",
                "message_type": "text",
            }
        ]

        result = converter.convert(raw)

        assert result[0].parts[0].content == "Hello!"


class TestAssistantMessages:
    """Tests for assistant message handling."""

    def test_converts_assistant_text_to_model_response(self):
        """Assistant text messages become ModelResponse with TextPart."""
        converter = PydanticAIHistoryConverter()
        raw = [
            {
                "role": "assistant",
                "content": "I'll help you with that.",
                "sender_name": "Agent",
                "message_type": "text",
            }
        ]

        result = converter.convert(raw)

        assert len(result) == 1
        assert isinstance(result[0], ModelResponse)
        assert len(result[0].parts) == 1
        assert isinstance(result[0].parts[0], TextPart)
        assert result[0].parts[0].content == "I'll help you with that."


class TestToolEventFiltering:
    """Tests for tool_call and tool_result filtering."""

    def test_skips_tool_call_messages(self):
        """tool_call messages are skipped."""
        converter = PydanticAIHistoryConverter()
        raw = [
            {
                "role": "assistant",
                "content": '{"event": "on_tool_start", "name": "search"}',
                "message_type": "tool_call",
            }
        ]

        result = converter.convert(raw)

        assert len(result) == 0

    def test_skips_tool_result_messages(self):
        """tool_result messages are skipped."""
        converter = PydanticAIHistoryConverter()
        raw = [
            {
                "role": "assistant",
                "content": '{"event": "on_tool_end", "output": "result"}',
                "message_type": "tool_result",
            }
        ]

        result = converter.convert(raw)

        assert len(result) == 0

    def test_skips_thought_messages(self):
        """thought messages are skipped."""
        converter = PydanticAIHistoryConverter()
        raw = [
            {
                "role": "assistant",
                "content": "I'm thinking about this...",
                "message_type": "thought",
            }
        ]

        result = converter.convert(raw)

        assert len(result) == 0


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_history(self):
        """Empty history returns empty list."""
        converter = PydanticAIHistoryConverter()

        result = converter.convert([])

        assert result == []

    def test_defaults_to_text_message_type(self):
        """Messages without message_type default to 'text'."""
        converter = PydanticAIHistoryConverter()
        raw = [
            {
                "role": "user",
                "content": "Hello",
                "sender_name": "Bob",
            }
        ]

        result = converter.convert(raw)

        assert len(result) == 1
        assert isinstance(result[0], ModelRequest)

    def test_defaults_to_user_role(self):
        """Messages without role default to 'user'."""
        converter = PydanticAIHistoryConverter()
        raw = [
            {
                "content": "Hello",
                "sender_name": "Bob",
                "message_type": "text",
            }
        ]

        result = converter.convert(raw)

        assert isinstance(result[0], ModelRequest)

    def test_handles_empty_content(self):
        """Handles messages with empty content."""
        converter = PydanticAIHistoryConverter()
        raw = [
            {
                "role": "user",
                "content": "",
                "sender_name": "Alice",
                "message_type": "text",
            }
        ]

        result = converter.convert(raw)

        assert len(result) == 1
        assert result[0].parts[0].content == "[Alice]: "


class TestMixedHistory:
    """Integration tests with mixed message types."""

    def test_full_conversation_flow(self):
        """Should handle a realistic conversation with mixed message types."""
        converter = PydanticAIHistoryConverter()
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

        # Should have: ModelRequest, ModelResponse, ModelRequest (tool events skipped)
        assert len(result) == 3

        assert isinstance(result[0], ModelRequest)
        assert result[0].parts[0].content == "[Alice]: What's the weather?"

        assert isinstance(result[1], ModelResponse)
        assert result[1].parts[0].content == "It's sunny today!"

        assert isinstance(result[2], ModelRequest)
        assert result[2].parts[0].content == "[Alice]: Thanks!"

    def test_multi_user_conversation(self):
        """Handles multiple users in conversation."""
        converter = PydanticAIHistoryConverter()
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

        assert len(result) == 3
        assert result[0].parts[0].content == "[Alice]: Hi team!"
        assert result[1].parts[0].content == "[Bob]: Hello everyone!"
        assert isinstance(result[2], ModelResponse)
        assert result[2].parts[0].content == "Hello Alice and Bob!"

    def test_alternating_messages(self):
        """Properly handles alternating user/assistant messages."""
        converter = PydanticAIHistoryConverter()
        raw = [
            {
                "role": "user",
                "content": "Q1",
                "sender_name": "User",
                "message_type": "text",
            },
            {"role": "assistant", "content": "A1", "message_type": "text"},
            {
                "role": "user",
                "content": "Q2",
                "sender_name": "User",
                "message_type": "text",
            },
            {"role": "assistant", "content": "A2", "message_type": "text"},
        ]

        result = converter.convert(raw)

        assert len(result) == 4
        assert isinstance(result[0], ModelRequest)
        assert isinstance(result[1], ModelResponse)
        assert isinstance(result[2], ModelRequest)
        assert isinstance(result[3], ModelResponse)
