"""Tests for AnthropicHistoryConverter."""

from thenvoi.converters.anthropic import AnthropicHistoryConverter


class TestUserMessages:
    """Tests for user message conversion."""

    def test_converts_user_text_with_sender_name(self):
        """User text messages include sender name prefix."""
        converter = AnthropicHistoryConverter()
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

    def test_handles_empty_sender_name(self):
        """User messages without sender_name use content as-is."""
        converter = AnthropicHistoryConverter()
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
        converter = AnthropicHistoryConverter()
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
        """This agent's text messages are skipped (redundant with tool results)."""
        converter = AnthropicHistoryConverter(agent_name="Agent")
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


class TestMultiAgentMessages:
    """Tests for multi-agent message handling."""

    def test_includes_other_agents_messages(self):
        """Other agents' messages should be included as user messages."""
        converter = AnthropicHistoryConverter(agent_name="Main Agent")
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
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "[Weather Agent]: It's sunny today!"

    def test_skips_only_own_messages(self):
        """Only skip THIS agent's text, include other agents."""
        converter = AnthropicHistoryConverter()
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
        assert "[Weather Agent]:" in result[0]["content"]

    def test_set_agent_name_updates_filtering(self):
        """set_agent_name should update which messages are skipped."""
        converter = AnthropicHistoryConverter()
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

    def test_includes_all_assistant_messages_when_no_agent_name(self):
        """When agent name is not set, include all assistant messages."""
        converter = AnthropicHistoryConverter()
        raw = [
            {
                "role": "assistant",
                "content": "Hello from agent 1",
                "sender_name": "Agent 1",
                "message_type": "text",
            },
            {
                "role": "assistant",
                "content": "Hello from agent 2",
                "sender_name": "Agent 2",
                "message_type": "text",
            },
        ]

        result = converter.convert(raw)

        assert len(result) == 2
        assert "[Agent 1]:" in result[0]["content"]
        assert "[Agent 2]:" in result[1]["content"]


class TestToolEventConversion:
    """Tests for tool_call and tool_result conversion."""

    def test_converts_tool_call_to_assistant_message(self):
        """tool_call messages become assistant messages with tool_use content."""
        converter = AnthropicHistoryConverter()
        raw = [
            {
                "role": "assistant",
                "content": '{"name": "search", "args": {"query": "test"}, "tool_call_id": "toolu_123"}',
                "message_type": "tool_call",
            }
        ]

        result = converter.convert(raw)

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert isinstance(result[0]["content"], list)
        assert result[0]["content"][0]["type"] == "tool_use"
        assert result[0]["content"][0]["id"] == "toolu_123"
        assert result[0]["content"][0]["name"] == "search"
        assert result[0]["content"][0]["input"] == {"query": "test"}

    def test_converts_tool_result_to_user_message(self):
        """tool_result messages become user messages with tool_result content."""
        converter = AnthropicHistoryConverter()
        raw = [
            {
                "role": "assistant",
                "content": '{"name": "search", "args": {"query": "test"}, "tool_call_id": "toolu_123"}',
                "message_type": "tool_call",
            },
            {
                "role": "assistant",
                "content": '{"name": "search", "output": "result data", "tool_call_id": "toolu_123"}',
                "message_type": "tool_result",
            },
        ]

        result = converter.convert(raw)

        assert len(result) == 2
        # First message is the tool_use
        assert result[0]["role"] == "assistant"
        assert result[0]["content"][0]["type"] == "tool_use"
        # Second message is the tool_result
        assert result[1]["role"] == "user"
        assert isinstance(result[1]["content"], list)
        assert result[1]["content"][0]["type"] == "tool_result"
        assert result[1]["content"][0]["tool_use_id"] == "toolu_123"
        assert result[1]["content"][0]["content"] == "result data"

    def test_batches_multiple_tool_calls(self):
        """Multiple consecutive tool_call messages are batched into one assistant message."""
        converter = AnthropicHistoryConverter()
        raw = [
            {
                "role": "assistant",
                "content": '{"name": "tool1", "args": {}, "tool_call_id": "toolu_1"}',
                "message_type": "tool_call",
            },
            {
                "role": "assistant",
                "content": '{"name": "tool2", "args": {}, "tool_call_id": "toolu_2"}',
                "message_type": "tool_call",
            },
            {
                "role": "assistant",
                "content": '{"name": "tool1", "output": "result1", "tool_call_id": "toolu_1"}',
                "message_type": "tool_result",
            },
        ]

        result = converter.convert(raw)

        # First message should have both tool_use blocks batched
        assert len(result) == 2
        assert result[0]["role"] == "assistant"
        assert len(result[0]["content"]) == 2
        assert result[0]["content"][0]["name"] == "tool1"
        assert result[0]["content"][1]["name"] == "tool2"
        # Second message is the tool_result
        assert result[1]["role"] == "user"

    def test_batches_multiple_tool_results(self):
        """Multiple consecutive tool_result messages are batched into one user message.

        Per Anthropic docs, all tool results must be in a single user message
        to enable parallel tool use.
        """
        converter = AnthropicHistoryConverter()
        raw = [
            {
                "role": "assistant",
                "content": '{"name": "tool1", "args": {}, "tool_call_id": "toolu_1"}',
                "message_type": "tool_call",
            },
            {
                "role": "assistant",
                "content": '{"name": "tool2", "args": {}, "tool_call_id": "toolu_2"}',
                "message_type": "tool_call",
            },
            {
                "role": "assistant",
                "content": '{"name": "tool1", "output": "result1", "tool_call_id": "toolu_1"}',
                "message_type": "tool_result",
            },
            {
                "role": "assistant",
                "content": '{"name": "tool2", "output": "result2", "tool_call_id": "toolu_2"}',
                "message_type": "tool_result",
            },
        ]

        result = converter.convert(raw)

        # Should have: batched tool_use, batched tool_results
        assert len(result) == 2

        # First message: batched tool_use
        assert result[0]["role"] == "assistant"
        assert len(result[0]["content"]) == 2
        assert result[0]["content"][0]["name"] == "tool1"
        assert result[0]["content"][1]["name"] == "tool2"

        # Second message: batched tool_results
        assert result[1]["role"] == "user"
        assert len(result[1]["content"]) == 2
        assert result[1]["content"][0]["tool_use_id"] == "toolu_1"
        assert result[1]["content"][1]["tool_use_id"] == "toolu_2"

    def test_handles_interleaved_tool_calls_and_results(self):
        """Interleaved tool_call and tool_result messages are handled correctly."""
        converter = AnthropicHistoryConverter()
        raw = [
            # First batch of tool calls
            {
                "role": "assistant",
                "content": '{"name": "tool1", "args": {"q": "a"}, "tool_call_id": "toolu_1"}',
                "message_type": "tool_call",
            },
            {
                "role": "assistant",
                "content": '{"name": "tool2", "args": {"q": "b"}, "tool_call_id": "toolu_2"}',
                "message_type": "tool_call",
            },
            # Results for first batch
            {
                "role": "assistant",
                "content": '{"name": "tool1", "output": "result1", "tool_call_id": "toolu_1"}',
                "message_type": "tool_result",
            },
            {
                "role": "assistant",
                "content": '{"name": "tool2", "output": "result2", "tool_call_id": "toolu_2"}',
                "message_type": "tool_result",
            },
            # Second batch of tool calls
            {
                "role": "assistant",
                "content": '{"name": "tool3", "args": {}, "tool_call_id": "toolu_3"}',
                "message_type": "tool_call",
            },
            # Result for second batch
            {
                "role": "assistant",
                "content": '{"name": "tool3", "output": "result3", "tool_call_id": "toolu_3"}',
                "message_type": "tool_result",
            },
        ]

        result = converter.convert(raw)

        # Expected: [batched tool_use (2), batched tool_results (2), tool_use, tool_result]
        assert len(result) == 4

        # First message: batched tool_use for tool1 and tool2
        assert result[0]["role"] == "assistant"
        assert len(result[0]["content"]) == 2
        assert result[0]["content"][0]["name"] == "tool1"
        assert result[0]["content"][1]["name"] == "tool2"

        # Second message: batched tool_results for tool1 and tool2
        assert result[1]["role"] == "user"
        assert len(result[1]["content"]) == 2
        assert result[1]["content"][0]["tool_use_id"] == "toolu_1"
        assert result[1]["content"][1]["tool_use_id"] == "toolu_2"

        # Third message: single tool_use for tool3
        assert result[2]["role"] == "assistant"
        assert len(result[2]["content"]) == 1
        assert result[2]["content"][0]["name"] == "tool3"

        # Fourth message: tool_result for tool3
        assert result[3]["role"] == "user"
        assert len(result[3]["content"]) == 1
        assert result[3]["content"][0]["tool_use_id"] == "toolu_3"

    def test_skips_thought_messages(self):
        """thought messages are skipped."""
        converter = AnthropicHistoryConverter()
        raw = [
            {
                "role": "assistant",
                "content": "I'm thinking about this...",
                "message_type": "thought",
            }
        ]

        result = converter.convert(raw)

        assert len(result) == 0

    def test_handles_malformed_tool_call_json(self, caplog):
        """Malformed tool_call JSON is skipped with warning."""
        converter = AnthropicHistoryConverter()
        raw = [
            {
                "role": "assistant",
                "content": "not valid json",
                "message_type": "tool_call",
            }
        ]

        result = converter.convert(raw)

        assert len(result) == 0
        assert "Failed to parse tool_call" in caplog.text

    def test_handles_malformed_tool_result_json(self, caplog):
        """Malformed tool_result JSON is skipped with warning."""
        converter = AnthropicHistoryConverter()
        raw = [
            {
                "role": "assistant",
                "content": "not valid json",
                "message_type": "tool_result",
            }
        ]

        result = converter.convert(raw)

        assert len(result) == 0
        assert "Failed to parse tool_result" in caplog.text

    def test_skips_tool_call_with_missing_tool_call_id(self, caplog):
        """tool_call without tool_call_id is skipped."""
        converter = AnthropicHistoryConverter()
        raw = [
            {
                "role": "assistant",
                "content": '{"name": "search", "args": {"query": "test"}}',
                "message_type": "tool_call",
            }
        ]

        result = converter.convert(raw)

        assert len(result) == 0
        assert "Skipping tool_call with missing tool_call_id" in caplog.text

    def test_skips_tool_call_with_missing_name(self, caplog):
        """tool_call without name is skipped."""
        converter = AnthropicHistoryConverter()
        raw = [
            {
                "role": "assistant",
                "content": '{"args": {"query": "test"}, "tool_call_id": "toolu_123"}',
                "message_type": "tool_call",
            }
        ]

        result = converter.convert(raw)

        assert len(result) == 0
        assert "Skipping tool_call with missing name" in caplog.text

    def test_skips_tool_result_with_missing_tool_call_id(self, caplog):
        """tool_result without tool_call_id is skipped."""
        converter = AnthropicHistoryConverter()
        raw = [
            {
                "role": "assistant",
                "content": '{"name": "search", "output": "result data"}',
                "message_type": "tool_result",
            }
        ]

        result = converter.convert(raw)

        assert len(result) == 0
        assert "Skipping tool_result with missing tool_call_id" in caplog.text

    def test_skips_tool_result_with_missing_name(self, caplog):
        """tool_result without name is skipped."""
        converter = AnthropicHistoryConverter()
        raw = [
            {
                "role": "assistant",
                "content": '{"output": "result data", "tool_call_id": "toolu_123"}',
                "message_type": "tool_result",
            }
        ]

        result = converter.convert(raw)

        assert len(result) == 0
        assert "Skipping tool_result with missing name" in caplog.text

    def test_flushes_trailing_tool_calls(self):
        """Trailing tool_call messages at end of history are properly flushed."""
        converter = AnthropicHistoryConverter()
        raw = [
            {
                "role": "assistant",
                "content": '{"name": "tool1", "args": {}, "tool_call_id": "toolu_1"}',
                "message_type": "tool_call",
            },
            {
                "role": "assistant",
                "content": '{"name": "tool2", "args": {}, "tool_call_id": "toolu_2"}',
                "message_type": "tool_call",
            },
        ]

        result = converter.convert(raw)

        # Both tool calls should be batched into a single assistant message
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert len(result[0]["content"]) == 2
        assert result[0]["content"][0]["name"] == "tool1"
        assert result[0]["content"][1]["name"] == "tool2"


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_history(self):
        """Empty history returns empty list."""
        converter = AnthropicHistoryConverter()

        result = converter.convert([])

        assert result == []

    def test_defaults_to_text_message_type(self):
        """Messages without message_type default to 'text'."""
        converter = AnthropicHistoryConverter()
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
        converter = AnthropicHistoryConverter()
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
        converter = AnthropicHistoryConverter()
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


class TestMixedHistory:
    """Integration tests with mixed message types."""

    def test_full_conversation_flow(self):
        """Should handle a realistic conversation with mixed message types."""
        converter = AnthropicHistoryConverter(agent_name="Agent")
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
                "content": '{"name": "get_weather", "args": {"location": "NYC"}, "tool_call_id": "toolu_123"}',
                "message_type": "tool_call",
            },
            # Tool result
            {
                "role": "assistant",
                "content": '{"name": "get_weather", "output": "sunny", "tool_call_id": "toolu_123"}',
                "message_type": "tool_result",
            },
            # Agent responds with text (skipped - own message)
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

        # Should have: user message, tool_use, tool_result, user follow-up
        # (agent's own text is skipped)
        assert len(result) == 4

        # User question
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "[Alice]: What's the weather?"

        # Tool use (assistant)
        assert result[1]["role"] == "assistant"
        assert result[1]["content"][0]["type"] == "tool_use"
        assert result[1]["content"][0]["name"] == "get_weather"

        # Tool result (user)
        assert result[2]["role"] == "user"
        assert result[2]["content"][0]["type"] == "tool_result"
        assert result[2]["content"][0]["content"] == "sunny"

        # User follow-up
        assert result[3]["role"] == "user"
        assert result[3]["content"] == "[Alice]: Thanks!"

    def test_multi_user_conversation(self):
        """Handles multiple users in conversation."""
        converter = AnthropicHistoryConverter(agent_name="Agent")
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

        # Agent's own message is skipped
        assert len(result) == 2
        assert result[0]["content"] == "[Alice]: Hi team!"
        assert result[1]["content"] == "[Bob]: Hello everyone!"

    def test_multi_agent_conversation_flow(self):
        """Should include other agents' messages in multi-agent conversations."""
        converter = AnthropicHistoryConverter(agent_name="Main Agent")
        raw = [
            # User asks Main Agent to get weather
            {
                "role": "user",
                "content": "What's the weather in Tokyo?",
                "sender_name": "Alice",
                "message_type": "text",
            },
            # Main Agent asks Weather Agent (skipped - own message)
            {
                "role": "assistant",
                "content": "Let me check with the weather agent.",
                "sender_name": "Main Agent",
                "message_type": "text",
            },
            # Weather Agent responds (included)
            {
                "role": "assistant",
                "content": "Tokyo is 15°C and cloudy.",
                "sender_name": "Weather Agent",
                "message_type": "text",
            },
            # Main Agent relays the response (skipped - own message)
            {
                "role": "assistant",
                "content": "The weather in Tokyo is 15°C and cloudy.",
                "sender_name": "Main Agent",
                "message_type": "text",
            },
        ]

        result = converter.convert(raw)

        # Should have: Alice's message + Weather Agent's message
        # (Main Agent's own messages are skipped)
        assert len(result) == 2

        assert result[0]["role"] == "user"
        assert result[0]["content"] == "[Alice]: What's the weather in Tokyo?"

        assert result[1]["role"] == "user"
        assert result[1]["content"] == "[Weather Agent]: Tokyo is 15°C and cloudy."
