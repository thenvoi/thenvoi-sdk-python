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
    """Tests for assistant message handling."""

    def test_skips_own_assistant_text_messages(self):
        """This agent's text messages are skipped (redundant with internal state)."""
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

        assert len(result) == 0

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
    """Tests for multi-agent message handling."""

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

    def test_skips_only_own_messages(self):
        """Only skip THIS agent's text, include other agents."""
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

        assert len(result) == 1  # Only Weather Agent's message
        assert result[0]["content"] == "It's sunny!"

    def test_set_agent_name_updates_filtering(self):
        """set_agent_name should update which messages are skipped."""
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

        # After setting name - own messages skipped
        converter.set_agent_name("Agent")
        assert len(converter.convert(raw)) == 0


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

    def test_handles_empty_content(self):
        """Handles messages with empty content."""
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

        assert len(result) == 1
        assert result[0]["content"] == "[Alice]: "


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
            # Support agent responds (skipped - own message)
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

        # Should have: 2 customer messages only
        assert len(result) == 2

        assert result[0]["role"] == "user"
        assert result[0]["content"] == "[Customer]: I want a refund for my order"

        assert result[1]["role"] == "user"
        assert result[1]["content"] == "[Customer]: The product was damaged"

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
            # Coordinator delegates (skipped - own message)
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
            # Coordinator summarizes (skipped - own message)
            {
                "role": "assistant",
                "content": "Here's the analysis summary.",
                "sender_name": "Coordinator",
                "message_type": "text",
            },
        ]

        result = converter.convert(raw)

        # User message + Analyzer message (Coordinator messages skipped)
        assert len(result) == 2

        assert result[0]["content"] == "[User]: Analyze this document"
        assert result[1]["content"] == "Analysis complete: ..."
        assert result[1]["sender"] == "Analyzer Agent"

