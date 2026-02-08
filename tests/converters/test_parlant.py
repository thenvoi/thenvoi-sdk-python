"""Parlant converter-specific tests.

Tests for Parlant converter-specific behavior that isn't covered by conformance tests:
- Including ALL assistant messages (differs from CrewAI/LangGraph which skip this agent's)
- Empty content skipping (unique to Parlant)
- Full history preservation for session reconstruction
"""

from __future__ import annotations

from thenvoi.converters.parlant import ParlantHistoryConverter


class TestIncludesAllAssistantMessages:
    """Tests for including ALL assistant messages (Parlant-specific behavior).

    Unlike CrewAI and LangGraph converters which skip THIS agent's messages
    (redundant with internal state), Parlant needs the FULL conversation history
    including this agent's own responses to properly reconstruct session state.
    """

    def test_includes_this_agent_messages(self):
        """Should include THIS agent's assistant messages (for session reconstruction)."""
        converter = ParlantHistoryConverter(agent_name="MyBot")

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
        ]

        result = converter.convert(raw)

        # Both messages should be included (Parlant needs full history)
        assert len(result) == 2
        assert result[1]["sender"] == "MyBot"
        assert result[1]["content"] == "Hi! I'm MyBot"

    def test_includes_other_agent_messages(self):
        """Should include other agents' assistant messages."""
        converter = ParlantHistoryConverter(agent_name="MyBot")

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

        # Both should be included
        assert len(result) == 2
        senders = [msg["sender"] for msg in result]
        assert "OtherBot" in senders
        assert "MyBot" in senders

    def test_comparison_with_crewai_behavior(self):
        """Demonstrate difference from CrewAI (which skips this agent's messages)."""
        from thenvoi.converters.crewai import CrewAIHistoryConverter

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
                "content": "I'm the agent",
                "sender_name": "TestBot",
                "sender_type": "Agent",
                "message_type": "text",
            },
        ]

        # Parlant includes the agent's own message
        parlant_converter = ParlantHistoryConverter(agent_name="TestBot")
        parlant_result = parlant_converter.convert(raw)
        assert len(parlant_result) == 2

        # CrewAI skips the agent's own message
        crewai_converter = CrewAIHistoryConverter(agent_name="TestBot")
        crewai_result = crewai_converter.convert(raw)
        assert len(crewai_result) == 1
        assert crewai_result[0]["role"] == "user"


class TestEmptyContentSkipping:
    """Tests for empty content skipping (unique to Parlant)."""

    def test_skips_empty_content_messages(self):
        """Should skip messages with empty content."""
        converter = ParlantHistoryConverter()

        raw = [
            {
                "role": "user",
                "content": "",
                "sender_name": "Alice",
                "message_type": "text",
            },
            {
                "role": "user",
                "content": "Hello",
                "sender_name": "Bob",
                "message_type": "text",
            },
        ]

        result = converter.convert(raw)

        assert len(result) == 1
        assert result[0]["sender"] == "Bob"

    def test_skips_whitespace_only_content(self):
        """Whitespace-only content should NOT be skipped (only truly empty)."""
        converter = ParlantHistoryConverter()

        raw = [
            {
                "role": "user",
                "content": "   ",  # Whitespace only
                "sender_name": "Alice",
                "message_type": "text",
            },
            {
                "role": "user",
                "content": "Hello",
                "sender_name": "Bob",
                "message_type": "text",
            },
        ]

        result = converter.convert(raw)

        # Whitespace content is still truthy, so included
        assert len(result) == 2

    def test_skips_empty_assistant_content(self):
        """Should skip assistant messages with empty content."""
        converter = ParlantHistoryConverter()

        raw = [
            {
                "role": "assistant",
                "content": "",
                "sender_name": "Bot",
                "message_type": "text",
            },
            {
                "role": "assistant",
                "content": "Valid response",
                "sender_name": "Bot",
                "message_type": "text",
            },
        ]

        result = converter.convert(raw)

        assert len(result) == 1
        assert result[0]["content"] == "Valid response"


class TestSetAgentName:
    """Tests for set_agent_name (stored but not used for filtering)."""

    def test_set_agent_name_does_not_affect_filtering(self):
        """set_agent_name should NOT change filtering (Parlant includes all)."""
        converter = ParlantHistoryConverter(agent_name="OldBot")

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

        # With OldBot as agent_name, both messages should be included
        result1 = converter.convert(raw)
        assert len(result1) == 2

        # Change agent_name to NewBot
        converter.set_agent_name("NewBot")

        # Both messages should STILL be included (Parlant needs full history)
        result2 = converter.convert(raw)
        assert len(result2) == 2


class TestSenderMetadata:
    """Tests for sender metadata in output messages."""

    def test_user_message_includes_sender_info(self):
        """User messages should include sender, sender_type metadata."""
        converter = ParlantHistoryConverter()

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
        """Assistant messages should include sender, sender_type metadata."""
        converter = ParlantHistoryConverter()

        raw = [
            {
                "role": "assistant",
                "content": "Response",
                "sender_name": "MyBot",
                "sender_type": "Agent",
                "message_type": "text",
            }
        ]

        result = converter.convert(raw)

        assert result[0]["sender"] == "MyBot"
        assert result[0]["sender_type"] == "Agent"


class TestUserMessageFormatting:
    """Tests for user message content formatting."""

    def test_user_message_prefixes_sender_name(self):
        """User messages should have [sender_name]: prefix in content."""
        converter = ParlantHistoryConverter()

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
        converter = ParlantHistoryConverter()

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
        converter = ParlantHistoryConverter()

        raw = [
            {
                "role": "assistant",
                "content": "This is my response",
                "sender_name": "Bot",
                "sender_type": "Agent",
                "message_type": "text",
            }
        ]

        result = converter.convert(raw)

        # Content should be unchanged (no [Bot]: prefix)
        assert result[0]["content"] == "This is my response"


class TestMessageTypeFiltering:
    """Tests for message type filtering."""

    def test_skips_tool_call_messages(self):
        """Should skip tool_call message types."""
        converter = ParlantHistoryConverter()

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

    def test_skips_tool_result_messages(self):
        """Should skip tool_result message types."""
        converter = ParlantHistoryConverter()

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
