"""CrewAI converter-specific tests.

Tests for CrewAI converter-specific behavior that isn't covered by conformance tests:
- set_agent_name functionality for filtering
- Skipping THIS agent's messages (differs from Parlant which includes ALL)
- Sender metadata in output messages
"""

from __future__ import annotations

from thenvoi.converters.crewai import CrewAIHistoryConverter


class TestSetAgentName:
    """Tests for set_agent_name functionality."""

    def test_set_agent_name_changes_filter(self):
        """set_agent_name should change which messages are filtered."""
        converter = CrewAIHistoryConverter(agent_name="OldBot")

        raw = [
            {
                "role": "assistant",
                "content": "Message from OldBot",
                "sender_name": "OldBot",
                "sender_type": "Agent",
                "message_type": "text",
            },
            {
                "role": "assistant",
                "content": "Message from NewBot",
                "sender_name": "NewBot",
                "sender_type": "Agent",
                "message_type": "text",
            },
        ]

        # With OldBot as agent_name, OldBot's messages are skipped
        result1 = converter.convert(raw)
        assert len(result1) == 1
        assert result1[0]["sender"] == "NewBot"

        # Change agent_name to NewBot
        converter.set_agent_name("NewBot")

        # Now NewBot's messages are skipped
        result2 = converter.convert(raw)
        assert len(result2) == 1
        assert result2[0]["sender"] == "OldBot"

    def test_set_agent_name_empty_string(self):
        """Empty agent_name should include all assistant messages."""
        converter = CrewAIHistoryConverter(agent_name="")

        raw = [
            {
                "role": "assistant",
                "content": "Message from Bot1",
                "sender_name": "Bot1",
                "sender_type": "Agent",
                "message_type": "text",
            },
            {
                "role": "assistant",
                "content": "Message from Bot2",
                "sender_name": "Bot2",
                "sender_type": "Agent",
                "message_type": "text",
            },
        ]

        result = converter.convert(raw)

        # Both messages should be included since no agent_name is set
        assert len(result) == 2


class TestSkipsThisAgentMessages:
    """Tests for skipping THIS agent's messages (unique to CrewAI behavior)."""

    def test_skips_own_agent_assistant_messages(self):
        """Should skip assistant messages from this agent (redundant with CrewAI state)."""
        converter = CrewAIHistoryConverter(agent_name="MyBot")

        raw = [
            {
                "role": "user",
                "content": "Hello",
                "sender_name": "Alice",
                "sender_type": "User",
                "message_type": "text",
            },
            {
                "role": "assistant",
                "content": "Hi! I'm MyBot",
                "sender_name": "MyBot",
                "sender_type": "Agent",
                "message_type": "text",
            },
            {
                "role": "user",
                "content": "Thanks",
                "sender_name": "Alice",
                "sender_type": "User",
                "message_type": "text",
            },
        ]

        result = converter.convert(raw)

        # Should only have the 2 user messages, not MyBot's response
        assert len(result) == 2
        assert all(msg["role"] == "user" for msg in result)

    def test_includes_other_agent_assistant_messages(self):
        """Should include assistant messages from OTHER agents."""
        converter = CrewAIHistoryConverter(agent_name="MyBot")

        raw = [
            {
                "role": "assistant",
                "content": "I'm OtherBot",
                "sender_name": "OtherBot",
                "sender_type": "Agent",
                "message_type": "text",
            },
            {
                "role": "assistant",
                "content": "I'm MyBot",
                "sender_name": "MyBot",
                "sender_type": "Agent",
                "message_type": "text",
            },
        ]

        result = converter.convert(raw)

        # Only OtherBot's message should be included
        assert len(result) == 1
        assert result[0]["sender"] == "OtherBot"


class TestSenderMetadata:
    """Tests for sender metadata in output messages."""

    def test_user_message_includes_sender_info(self):
        """User messages should include sender, sender_type metadata."""
        converter = CrewAIHistoryConverter()

        raw = [
            {
                "role": "user",
                "content": "Hello",
                "sender_name": "Alice",
                "sender_type": "User",
                "message_type": "text",
            }
        ]

        result = converter.convert(raw)

        assert result[0]["sender"] == "Alice"
        assert result[0]["sender_type"] == "User"

    def test_assistant_message_includes_sender_info(self):
        """Other agent messages should include sender, sender_type metadata."""
        converter = CrewAIHistoryConverter(agent_name="MyBot")

        raw = [
            {
                "role": "assistant",
                "content": "I'm OtherBot",
                "sender_name": "OtherBot",
                "sender_type": "Agent",
                "message_type": "text",
            }
        ]

        result = converter.convert(raw)

        assert result[0]["sender"] == "OtherBot"
        assert result[0]["sender_type"] == "Agent"

    def test_preserves_role_in_output(self):
        """Should preserve user/assistant role distinction."""
        converter = CrewAIHistoryConverter(agent_name="MyBot")

        raw = [
            {
                "role": "user",
                "content": "Question",
                "sender_name": "Alice",
                "sender_type": "User",
                "message_type": "text",
            },
            {
                "role": "assistant",
                "content": "Answer from OtherBot",
                "sender_name": "OtherBot",
                "sender_type": "Agent",
                "message_type": "text",
            },
        ]

        result = converter.convert(raw)

        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"


class TestUserMessageFormatting:
    """Tests for user message content formatting."""

    def test_user_message_prefixes_sender_name(self):
        """User messages should have [sender_name]: prefix in content."""
        converter = CrewAIHistoryConverter()

        raw = [
            {
                "role": "user",
                "content": "Hello there",
                "sender_name": "Alice",
                "sender_type": "User",
                "message_type": "text",
            }
        ]

        result = converter.convert(raw)

        assert result[0]["content"] == "[Alice]: Hello there"

    def test_user_message_without_sender_name(self):
        """User message without sender_name should not have prefix."""
        converter = CrewAIHistoryConverter()

        raw = [
            {
                "role": "user",
                "content": "Hello there",
                "sender_name": "",
                "sender_type": "User",
                "message_type": "text",
            }
        ]

        result = converter.convert(raw)

        assert result[0]["content"] == "Hello there"

    def test_assistant_message_no_prefix(self):
        """Assistant messages should NOT have sender prefix in content."""
        converter = CrewAIHistoryConverter(agent_name="MyBot")

        raw = [
            {
                "role": "assistant",
                "content": "This is my response",
                "sender_name": "OtherBot",
                "sender_type": "Agent",
                "message_type": "text",
            }
        ]

        result = converter.convert(raw)

        # Content should be unchanged (no [OtherBot]: prefix)
        assert result[0]["content"] == "This is my response"


class TestMessageTypeFiltering:
    """Tests for message type filtering."""

    def test_skips_tool_call_messages(self):
        """Should skip tool_call message types."""
        converter = CrewAIHistoryConverter()

        raw = [
            {
                "role": "user",
                "content": "Hello",
                "sender_name": "Alice",
                "message_type": "text",
            },
            {
                "role": "assistant",
                "content": '{"name": "search", "args": {}}',
                "message_type": "tool_call",
            },
        ]

        result = converter.convert(raw)

        assert len(result) == 1
        assert result[0]["content"] == "[Alice]: Hello"

    def test_skips_tool_result_messages(self):
        """Should skip tool_result message types."""
        converter = CrewAIHistoryConverter()

        raw = [
            {
                "role": "user",
                "content": "Hello",
                "sender_name": "Alice",
                "message_type": "text",
            },
            {
                "role": "assistant",
                "content": '{"output": "search result"}',
                "message_type": "tool_result",
            },
        ]

        result = converter.convert(raw)

        assert len(result) == 1
