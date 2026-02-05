"""Framework-specific tests for ParlantHistoryConverter.

Common behaviors (user messages, multi-agent, edge cases)
are tested in test_converter_contract.py. This file tests Parlant-specific:
- Includes agent's OWN messages (unlike other converters)
- Tool event filtering (skips tool_call/tool_result)
- Other agents preserve assistant role
- sender/sender_type metadata preservation
- Guideline-based conversation patterns
"""

from thenvoi.converters.parlant import ParlantHistoryConverter


class TestIncludesOwnMessages:
    """Tests for including the agent's own messages.

    Parlant is UNIQUE among converters in that it includes the agent's
    own text messages. This is required to reconstruct session state.
    """

    def test_includes_own_assistant_messages(self):
        """This agent's text messages are included (Parlant needs full history)."""
        converter = ParlantHistoryConverter(agent_name="Agent")
        raw = [
            {
                "role": "assistant",
                "content": "I'll help you with that.",
                "sender_name": "Agent",
                "message_type": "text",
            }
        ]

        result = converter.convert(raw)

        # Parlant needs ALL messages to reconstruct session state
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "I'll help you with that."
        assert result[0]["sender"] == "Agent"

    def test_includes_all_agent_messages(self):
        """Include ALL agent messages (own + others) for Parlant session."""
        converter = ParlantHistoryConverter()
        converter.set_agent_name("Main Agent")
        raw = [
            {
                "role": "assistant",
                "content": "I'll help",
                "sender_name": "Main Agent",
                "message_type": "text",
            },
            {
                "role": "assistant",
                "content": "It's sunny!",
                "sender_name": "Weather Agent",
                "message_type": "text",
            },
        ]

        result = converter.convert(raw)

        # Parlant needs ALL messages
        assert len(result) == 2
        assert result[0]["content"] == "I'll help"
        assert result[1]["content"] == "It's sunny!"

    def test_agent_name_stored_but_not_used_for_filtering(self):
        """set_agent_name stores name but doesn't filter (Parlant needs all)."""
        converter = ParlantHistoryConverter()
        raw = [
            {
                "role": "assistant",
                "content": "Hello",
                "sender_name": "Agent",
                "message_type": "text",
            }
        ]

        # Before setting name - all assistant messages included
        assert len(converter.convert(raw)) == 1

        # After setting name - still all assistant messages included (Parlant needs full history)
        converter.set_agent_name("Agent")
        assert len(converter.convert(raw)) == 1


class TestToolEventFiltering:
    """Tests for tool_call and tool_result filtering."""

    def test_skips_tool_call_messages(self):
        """tool_call messages are skipped."""
        converter = ParlantHistoryConverter()
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
        converter = ParlantHistoryConverter()
        raw = [
            {
                "role": "assistant",
                "content": '{"event": "on_tool_end", "output": "result"}',
                "message_type": "tool_result",
            }
        ]

        result = converter.convert(raw)

        assert len(result) == 0


class TestOtherAgentsAsAssistant:
    """Tests for preserving other agents as assistant role."""

    def test_includes_other_agents_as_assistant(self):
        """Other agents' messages are included with role assistant."""
        converter = ParlantHistoryConverter(agent_name="Main Agent")
        raw = [
            {
                "role": "assistant",
                "content": "Here's my analysis.",
                "sender_name": "Helper Agent",
                "sender_type": "Agent",
                "message_type": "text",
            }
        ]

        result = converter.convert(raw)

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "Here's my analysis."
        assert result[0]["sender"] == "Helper Agent"


class TestSenderMetadata:
    """Tests for sender/sender_type metadata preservation."""

    def test_preserves_sender_type(self):
        """Should preserve sender_type in output."""
        converter = ParlantHistoryConverter()
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
                "content": "Hi there",
                "sender_name": "Helper Agent",
                "sender_type": "Agent",
                "message_type": "text",
            },
        ]

        result = converter.convert(raw)

        assert result[0]["sender_type"] == "User"
        assert result[1]["sender_type"] == "Agent"


class TestEmptyContentHandling:
    """Tests for empty content handling specific to Parlant."""

    def test_skips_empty_content(self):
        """Skips messages with empty content."""
        converter = ParlantHistoryConverter()
        raw = [
            {
                "role": "user",
                "content": "",
                "sender_name": "Alice",
                "message_type": "text",
            }
        ]

        result = converter.convert(raw)

        # Empty content messages are skipped
        assert len(result) == 0


class TestGuidelineBasedConversation:
    """Tests for Parlant-style guideline-based conversations."""

    def test_support_conversation_flow(self):
        """Should handle a customer support conversation flow."""
        converter = ParlantHistoryConverter(agent_name="Support Agent")
        raw = [
            # Customer question
            {
                "role": "user",
                "content": "I want a refund for my order",
                "sender_name": "Customer",
                "message_type": "text",
            },
            # Support agent responds (included for Parlant)
            {
                "role": "assistant",
                "content": "Let me check your order status...",
                "sender_name": "Support Agent",
                "message_type": "text",
            },
            # Tool call event (skipped)
            {
                "role": "assistant",
                "content": '{"tool": "check_order_status"}',
                "message_type": "tool_call",
            },
            # Tool result event (skipped)
            {
                "role": "assistant",
                "content": '{"status": "delivered"}',
                "message_type": "tool_result",
            },
            # Customer follow-up
            {
                "role": "user",
                "content": "The product was damaged",
                "sender_name": "Customer",
                "message_type": "text",
            },
        ]

        result = converter.convert(raw)

        # Should have: 2 customer messages + 1 agent message (tool events skipped)
        assert len(result) == 3

        assert result[0]["role"] == "user"
        assert result[0]["content"] == "[Customer]: I want a refund for my order"

        assert result[1]["role"] == "assistant"
        assert result[1]["content"] == "Let me check your order status..."

        assert result[2]["role"] == "user"
        assert result[2]["content"] == "[Customer]: The product was damaged"

    def test_multi_agent_collaboration(self):
        """Should handle multi-agent collaboration in Parlant-style."""
        converter = ParlantHistoryConverter(agent_name="Coordinator")
        raw = [
            # User request
            {
                "role": "user",
                "content": "Analyze this document",
                "sender_name": "User",
                "message_type": "text",
            },
            # Coordinator delegates (included for Parlant)
            {
                "role": "assistant",
                "content": "I'll have the analyzer look at this.",
                "sender_name": "Coordinator",
                "message_type": "text",
            },
            # Analyzer responds
            {
                "role": "assistant",
                "content": "Analysis complete: ...",
                "sender_name": "Analyzer Agent",
                "message_type": "text",
            },
            # Coordinator summarizes (included for Parlant)
            {
                "role": "assistant",
                "content": "Here's the analysis summary.",
                "sender_name": "Coordinator",
                "message_type": "text",
            },
        ]

        result = converter.convert(raw)

        # All text messages included (Parlant needs full history)
        assert len(result) == 4

        assert result[0]["content"] == "[User]: Analyze this document"
        assert result[1]["content"] == "I'll have the analyzer look at this."
        assert result[2]["content"] == "Analysis complete: ..."
        assert result[3]["content"] == "Here's the analysis summary."
