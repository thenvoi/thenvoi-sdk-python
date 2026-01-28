"""Tests for ParlantHistoryConverter."""

from thenvoi.converters.parlant import ParlantHistoryConverter


class TestUserMessages:
    """Tests for user message conversion."""

    def test_converts_user_text_with_sender_name(self):
        """User text messages include sender name prefix."""
        converter = ParlantHistoryConverter()
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
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "[Alice]: Hello, agent!"
        assert result[0]["sender"] == "Alice"
        assert result[0]["sender_type"] == "User"

    def test_handles_empty_sender_name(self):
        """User messages without sender_name use content as-is."""
        converter = ParlantHistoryConverter()
        raw = [
            {
                "role": "user",
                "content": "Hello!",
                "sender_name": "",
                "message_type": "text",
            }
        ]

        result = converter.convert(raw)

        assert result[0]["content"] == "Hello!"

    def test_handles_missing_sender_name(self):
        """User messages with no sender_name key use content as-is."""
        converter = ParlantHistoryConverter()
        raw = [
            {
                "role": "user",
                "content": "Hello!",
                "message_type": "text",
            }
        ]

        result = converter.convert(raw)

        assert result[0]["content"] == "Hello!"


class TestAssistantMessages:
    """Tests for assistant message handling.

    NOTE: Unlike other converters, ParlantHistoryConverter includes ALL
    assistant messages (including own messages) because Parlant needs
    the full history to reconstruct session state.
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


class TestMultiAgentMessages:
    """Tests for multi-agent message handling.

    NOTE: Parlant converter includes ALL assistant messages (own + others)
    to reconstruct the full session history.
    """

    def test_includes_other_agents_messages(self):
        """Other agents' messages should be included."""
        converter = ParlantHistoryConverter(agent_name="Main Agent")
        raw = [
            {
                "role": "assistant",
                "content": "The weather is sunny.",
                "sender_name": "Weather Agent",
                "message_type": "text",
            }
        ]

        result = converter.convert(raw)

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "The weather is sunny."

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

    def test_skips_thought_messages(self):
        """thought messages are skipped."""
        converter = ParlantHistoryConverter()
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
        converter = ParlantHistoryConverter()

        result = converter.convert([])

        assert result == []

    def test_defaults_to_text_message_type(self):
        """Messages without message_type default to 'text'."""
        converter = ParlantHistoryConverter()
        raw = [
            {
                "role": "user",
                "content": "Hello",
                "sender_name": "Bob",
            }
        ]

        result = converter.convert(raw)

        assert len(result) == 1
        assert result[0]["content"] == "[Bob]: Hello"

    def test_defaults_to_user_role(self):
        """Messages without role default to 'user'."""
        converter = ParlantHistoryConverter()
        raw = [
            {
                "content": "Hello",
                "sender_name": "Bob",
                "message_type": "text",
            }
        ]

        result = converter.convert(raw)

        assert result[0]["role"] == "user"

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
    """Tests for Parlant-style guideline-based conversations.

    NOTE: Parlant converter includes ALL assistant messages (own + others)
    to properly reconstruct session state.
    """

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
