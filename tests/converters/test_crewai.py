"""Tests for CrewAIHistoryConverter."""

from thenvoi.converters.crewai import CrewAIHistoryConverter


class TestUserMessages:
    """Tests for user message conversion."""

    def test_converts_user_text_with_sender_name(self):
        """User text messages include sender name prefix."""
        converter = CrewAIHistoryConverter()
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
        converter = CrewAIHistoryConverter()
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
        converter = CrewAIHistoryConverter()
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
        converter = CrewAIHistoryConverter(agent_name="Agent")
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
        converter = CrewAIHistoryConverter(agent_name="Main Agent")
        raw = [
            {
                "role": "assistant",
                "content": "Here's my analysis.",
                "sender_name": "Research Agent",
                "sender_type": "Agent",
                "message_type": "text",
            }
        ]

        result = converter.convert(raw)

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "Here's my analysis."
        assert result[0]["sender"] == "Research Agent"


class TestMultiAgentMessages:
    """Tests for multi-agent message handling."""

    def test_includes_other_agents_messages(self):
        """Other agents' messages should be included."""
        converter = CrewAIHistoryConverter(agent_name="Main Agent")
        raw = [
            {
                "role": "assistant",
                "content": "It's sunny today!",
                "sender_name": "Weather Agent",
                "message_type": "text",
            }
        ]

        result = converter.convert(raw)

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "It's sunny today!"

    def test_skips_only_own_messages(self):
        """Only skip THIS agent's text, include other agents."""
        converter = CrewAIHistoryConverter()
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
        converter = CrewAIHistoryConverter()
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
        converter = CrewAIHistoryConverter()
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
        converter = CrewAIHistoryConverter()
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
        converter = CrewAIHistoryConverter()
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
        converter = CrewAIHistoryConverter()

        result = converter.convert([])

        assert result == []

    def test_defaults_to_text_message_type(self):
        """Messages without message_type default to 'text'."""
        converter = CrewAIHistoryConverter()
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
        converter = CrewAIHistoryConverter()
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
        converter = CrewAIHistoryConverter()
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


class TestCrewWorkflow:
    """Tests for CrewAI-style crew workflows."""

    def test_research_crew_conversation(self):
        """Should handle a research crew conversation flow."""
        converter = CrewAIHistoryConverter(agent_name="Research Analyst")
        raw = [
            # User request
            {
                "role": "user",
                "content": "Research AI trends for 2024",
                "sender_name": "Alice",
                "message_type": "text",
            },
            # Research Analyst responds (skipped - own message)
            {
                "role": "assistant",
                "content": "Here are the key AI trends...",
                "sender_name": "Research Analyst",
                "message_type": "text",
            },
            # Writer receives and drafts
            {
                "role": "assistant",
                "content": "Based on the research, here's my draft...",
                "sender_name": "Content Writer",
                "message_type": "text",
            },
            # Editor reviews
            {
                "role": "assistant",
                "content": "Final edited version...",
                "sender_name": "Editor",
                "message_type": "text",
            },
        ]

        result = converter.convert(raw)

        # Should have: user message + 2 other agents' messages
        # (Research Analyst's own message is skipped)
        assert len(result) == 3

        assert result[0]["role"] == "user"
        assert result[0]["content"] == "[Alice]: Research AI trends for 2024"

        assert result[1]["role"] == "assistant"
        assert result[1]["sender"] == "Content Writer"

        assert result[2]["role"] == "assistant"
        assert result[2]["sender"] == "Editor"

    def test_preserves_sender_type(self):
        """Should preserve sender_type in output."""
        converter = CrewAIHistoryConverter()
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
