"""Contract tests for all history converters.

These parameterized tests run across all converter implementations to verify
common behaviors. All tests are parameterized by framework, with framework-specific
behaviors handled through configuration in ConverterConfig.

Running:
    # All converters
    uv run pytest tests/converters/test_converter_contract.py -v

    # Specific converter
    uv run pytest tests/converters/test_converter_contract.py -k "anthropic" -v
"""

from __future__ import annotations

import pytest

from tests.framework_configs.converters import (
    CONVERTER_CONFIGS,
    ContentAssertion,
    ConverterConfig,
    ToolCallAssertion,
    get_tool_format,
    make_tool_call,
    make_tool_result,
    MALFORMED_TOOL_CALL,
    MALFORMED_TOOL_RESULT,
    TOOL_CALL_MISSING_ID,
    TOOL_CALL_MISSING_NAME,
)


@pytest.fixture(params=list(CONVERTER_CONFIGS.values()), ids=lambda c: c.name)
def converter_config(request: pytest.FixtureRequest) -> ConverterConfig:
    """Parameterized fixture that yields each converter config."""
    return request.param


@pytest.fixture
def converter(converter_config: ConverterConfig):
    """Create a converter instance from config."""
    return converter_config.create_converter()


@pytest.fixture
def converter_with_name(converter_config: ConverterConfig):
    """Create a converter instance with agent name set."""
    return converter_config.create_converter(agent_name="Agent")


class TestUserMessages:
    """Tests for user message conversion across all frameworks."""

    def test_converts_user_text_with_sender_name(
        self, converter, converter_config: ConverterConfig
    ):
        """User text messages include sender name prefix."""
        raw = [
            {
                "role": "user",
                "content": "Hello, agent!",
                "sender_name": "Alice",
                "message_type": "text",
            }
        ]

        result = converter.convert(raw)
        assertion = ContentAssertion(result, converter_config)

        assert assertion.has_length(1)
        assert assertion.content_contains("[Alice]:")
        assert assertion.content_contains("Hello, agent!")

    def test_handles_empty_sender_name(
        self, converter, converter_config: ConverterConfig
    ):
        """User messages with empty sender_name are handled appropriately."""
        raw = [
            {
                "role": "user",
                "content": "Hello!",
                "sender_name": "",
                "message_type": "text",
            }
        ]

        result = converter.convert(raw)
        assertion = ContentAssertion(result, converter_config)

        # ClaudeSDK skips empty content messages (but not empty sender)
        # LangChain uses empty brackets []:
        if converter_config.empty_sender_prefix_behavior == "empty_brackets":
            assert assertion.content_contains("[]:")
        else:
            # no_prefix: content without brackets, or just content
            content = assertion.get_content(0)
            assert "Hello!" in content


class TestAssistantMessages:
    """Tests for assistant message handling across all frameworks."""

    def test_skips_own_assistant_text_messages_when_configured(
        self, converter_with_name, converter_config: ConverterConfig
    ):
        """This agent's text messages are skipped (if configured)."""
        raw = [
            {
                "role": "assistant",
                "content": "I'll help you with that.",
                "sender_name": "Agent",
                "message_type": "text",
            }
        ]

        result = converter_with_name.convert(raw)
        assertion = ContentAssertion(result, converter_config)

        if converter_config.skips_own_messages:
            assert assertion.is_empty()
        else:
            # Parlant includes own messages
            assert assertion.has_length(1)
            assert assertion.content_contains("I'll help you with that.")


class TestMultiAgentMessages:
    """Tests for multi-agent message handling across all frameworks."""

    def test_includes_other_agents_messages(
        self, converter_with_name, converter_config: ConverterConfig
    ):
        """Other agents' messages should be included."""
        converter_with_name.set_agent_name("Main Agent")
        raw = [
            {
                "role": "assistant",
                "content": "It's sunny today!",
                "sender_name": "Weather Agent",
                "message_type": "text",
            }
        ]

        result = converter_with_name.convert(raw)
        assertion = ContentAssertion(result, converter_config)

        assert assertion.has_length(1)
        if converter_config.converts_other_agents_to_user:
            # Anthropic, LangChain, PydanticAI convert to user role
            assert assertion.content_contains("[Weather Agent]:")
        else:
            # CrewAI, Parlant preserve assistant role
            assert assertion.get_role(0) == "assistant"

    def test_skips_only_own_messages(
        self, converter, converter_config: ConverterConfig
    ):
        """Only skip THIS agent's text, include other agents."""
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
        assertion = ContentAssertion(result, converter_config)

        if converter_config.skips_own_messages:
            # Should have only Weather Agent's message
            assert assertion.has_length(1)
            content = assertion.get_content(0)
            assert "sunny" in content.lower()
        else:
            # Parlant includes both
            assert assertion.has_length(2)

    def test_set_agent_name_updates_filtering(
        self, converter, converter_config: ConverterConfig
    ):
        """set_agent_name should update which messages are skipped."""
        raw = [
            {
                "role": "assistant",
                "content": "Hello",
                "sender_name": "Agent",
                "message_type": "text",
            }
        ]

        # Before setting name - all assistant messages included
        result_before = converter.convert(raw)
        assertion_before = ContentAssertion(result_before, converter_config)
        assert assertion_before.has_length(1)

        # After setting name
        converter.set_agent_name("Agent")
        result_after = converter.convert(raw)
        assertion_after = ContentAssertion(result_after, converter_config)

        if converter_config.skips_own_messages:
            # Own messages now skipped
            assert assertion_after.is_empty()
        else:
            # Parlant still includes
            assert assertion_after.has_length(1)

    def test_includes_all_assistant_messages_when_no_agent_name(
        self, converter, converter_config: ConverterConfig
    ):
        """When agent name is not set, include all assistant messages."""
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
        assertion = ContentAssertion(result, converter_config)

        assert assertion.has_length(2)


class TestEdgeCases:
    """Tests for edge cases and error handling across all frameworks."""

    def test_empty_history(self, converter, converter_config: ConverterConfig):
        """Empty history returns empty result."""
        result = converter.convert([])
        assertion = ContentAssertion(result, converter_config)

        assert assertion.is_empty()

    def test_defaults_to_text_message_type(
        self, converter, converter_config: ConverterConfig
    ):
        """Messages without message_type default to 'text'."""
        raw = [
            {
                "role": "user",
                "content": "Hello",
                "sender_name": "Bob",
            }
        ]

        result = converter.convert(raw)
        assertion = ContentAssertion(result, converter_config)

        assert assertion.has_length(1)
        assert assertion.content_contains("Hello")

    def test_skips_thought_messages(self, converter, converter_config: ConverterConfig):
        """thought messages are skipped by all converters."""
        raw = [
            {
                "role": "assistant",
                "content": "I'm thinking about this...",
                "message_type": "thought",
            }
        ]

        result = converter.convert(raw)
        assertion = ContentAssertion(result, converter_config)

        assert assertion.is_empty()

    def test_handles_missing_sender_name(
        self, converter, converter_config: ConverterConfig
    ):
        """Messages without sender_name key are handled appropriately."""
        raw = [
            {
                "role": "user",
                "content": "Hello without sender",
                "message_type": "text",
            }
        ]

        result = converter.convert(raw)
        assertion = ContentAssertion(result, converter_config)

        # All converters should handle missing sender_name gracefully
        assert assertion.has_length(1)
        assert assertion.content_contains("Hello without sender")

    def test_handles_empty_content(self, converter, converter_config: ConverterConfig):
        """Empty content messages are handled according to config."""
        raw = [
            {
                "role": "user",
                "content": "",
                "sender_name": "Alice",
                "message_type": "text",
            }
        ]

        result = converter.convert(raw)
        assertion = ContentAssertion(result, converter_config)

        if converter_config.skips_empty_content:
            # ClaudeSDK and Parlant skip empty content
            assert assertion.is_empty()
        else:
            # Other converters include empty content messages
            assert assertion.has_length(1)

    def test_defaults_to_user_role(self, converter, converter_config: ConverterConfig):
        """Messages without role default to 'user'."""
        raw = [
            {
                "content": "Hello",
                "sender_name": "Bob",
                "message_type": "text",
            }
        ]

        result = converter.convert(raw)
        assertion = ContentAssertion(result, converter_config)

        assert assertion.has_length(1)
        assert assertion.content_contains("Hello")
        # String output type doesn't have role concept
        if converter_config.output_type != "string":
            assert assertion.get_role(0) == "user"


class TestToolEventHandling:
    """Tests for tool_call and tool_result handling across all frameworks."""

    def test_handles_tool_call_according_to_mode(
        self, converter, converter_config: ConverterConfig
    ):
        """Tool calls are handled according to framework's tool_handling_mode."""
        fmt = get_tool_format(converter_config)
        tool_call = make_tool_call("search", {"query": "test"}, "toolu_123", fmt)
        raw = [tool_call]

        result = converter.convert(raw)
        assertion = ToolCallAssertion(result, converter_config)

        if converter_config.tool_handling_mode == "skip":
            # CrewAI, Parlant skip tool events
            assert assertion.get_length() == 0
        elif converter_config.requires_tool_result_for_output:
            # LangChain: tool_call alone produces no output (needs matching result)
            assert assertion.get_length() == 0
        elif converter_config.tool_handling_mode == "raw_json":
            # ClaudeSDK includes as raw JSON
            assert assertion.get_length() == 1
            assert assertion.has_tool_call_at(0, name="search")
        elif converter_config.tool_handling_mode in ("structured", "langchain"):
            # Anthropic, PydanticAI convert to structured format
            assert assertion.get_length() == 1
            assert assertion.has_tool_call_at(0, name="search", tool_id="toolu_123")

    def test_handles_tool_result_according_to_mode(
        self, converter, converter_config: ConverterConfig
    ):
        """Tool results are handled according to framework's tool_handling_mode."""
        fmt = get_tool_format(converter_config)
        tool_call = make_tool_call("search", {"query": "test"}, "toolu_123", fmt)
        tool_result = make_tool_result("search", "result data", "toolu_123", fmt=fmt)
        raw = [tool_call, tool_result]

        result = converter.convert(raw)
        assertion = ToolCallAssertion(result, converter_config)

        if converter_config.tool_handling_mode == "skip":
            assert assertion.get_length() == 0
        elif converter_config.tool_handling_mode == "raw_json":
            assert assertion.get_length() == 2
        elif converter_config.tool_handling_mode in ("structured", "langchain"):
            assert assertion.get_length() == 2
            assert assertion.has_tool_call_at(0)
            assert assertion.has_tool_result_at(1, tool_id="toolu_123")


class TestToolBatching:
    """Tests for tool call/result batching across frameworks that support it."""

    def test_batches_multiple_tool_calls_when_supported(
        self, converter, converter_config: ConverterConfig
    ):
        """Consecutive tool calls are batched if framework supports it."""
        if converter_config.tool_handling_mode == "skip":
            pytest.skip(f"{converter_config.name} skips tool events")

        fmt = get_tool_format(converter_config)
        tool_call_1 = make_tool_call("tool1", {}, "toolu_1", fmt)
        tool_call_2 = make_tool_call("tool2", {}, "toolu_2", fmt)
        tool_result_1 = make_tool_result("tool1", "result1", "toolu_1", fmt=fmt)
        raw = [tool_call_1, tool_call_2, tool_result_1]

        result = converter.convert(raw)
        assertion = ToolCallAssertion(result, converter_config)

        if converter_config.batches_tool_calls:
            # Anthropic, PydanticAI batch tool calls
            assert assertion.get_length() == 2
            assert assertion.get_tool_call_count_at(0) == 2
        else:
            # LangChain, ClaudeSDK don't batch
            assert assertion.get_length() >= 2

    def test_batches_multiple_tool_results_when_supported(
        self, converter, converter_config: ConverterConfig
    ):
        """Consecutive tool results are batched if framework supports it."""
        if converter_config.tool_handling_mode == "skip":
            pytest.skip(f"{converter_config.name} skips tool events")

        fmt = get_tool_format(converter_config)
        tool_call_1 = make_tool_call("tool1", {}, "toolu_1", fmt)
        tool_call_2 = make_tool_call("tool2", {}, "toolu_2", fmt)
        tool_result_1 = make_tool_result("tool1", "result1", "toolu_1", fmt=fmt)
        tool_result_2 = make_tool_result("tool2", "result2", "toolu_2", fmt=fmt)
        raw = [tool_call_1, tool_call_2, tool_result_1, tool_result_2]

        result = converter.convert(raw)
        assertion = ToolCallAssertion(result, converter_config)

        if converter_config.batches_tool_results:
            # Anthropic, PydanticAI batch tool results
            assert assertion.get_length() == 2
            assert assertion.get_tool_result_count_at(1) == 2
        else:
            # Others don't batch
            assert assertion.get_length() >= 2

    def test_handles_interleaved_tool_calls_and_results(
        self, converter, converter_config: ConverterConfig
    ):
        """Interleaved tool calls and results are handled correctly."""
        if converter_config.tool_handling_mode == "skip":
            pytest.skip(f"{converter_config.name} skips tool events")

        fmt = get_tool_format(converter_config)
        tool_call_1 = make_tool_call("tool1", {}, "toolu_1", fmt)
        tool_call_2 = make_tool_call("tool2", {}, "toolu_2", fmt)
        tool_result_1 = make_tool_result("tool1", "result1", "toolu_1", fmt=fmt)
        tool_result_2 = make_tool_result("tool2", "result2", "toolu_2", fmt=fmt)
        tool_call_3 = make_tool_call("tool3", {}, "toolu_3", fmt)
        tool_result_3 = make_tool_result("tool3", "result3", "toolu_3", fmt=fmt)

        raw = [tool_call_1, tool_call_2, tool_result_1, tool_result_2, tool_call_3, tool_result_3]

        result = converter.convert(raw)
        assertion = ToolCallAssertion(result, converter_config)

        if converter_config.batches_tool_calls:
            # Batched: [calls 1&2] [results 1&2] [call 3] [result 3]
            assert assertion.get_length() == 4
        else:
            # Non-batched: more items
            assert assertion.get_length() >= 4

    def test_flushes_trailing_tool_calls(
        self, converter, converter_config: ConverterConfig
    ):
        """Trailing tool calls at end of history are properly flushed."""
        if converter_config.tool_handling_mode == "skip":
            pytest.skip(f"{converter_config.name} skips tool events")
        if converter_config.requires_tool_result_for_output:
            pytest.skip(f"{converter_config.name} requires matching tool_result for output")

        fmt = get_tool_format(converter_config)
        tool_call_1 = make_tool_call("tool1", {}, "toolu_1", fmt)
        tool_call_2 = make_tool_call("tool2", {}, "toolu_2", fmt)
        raw = [tool_call_1, tool_call_2]

        result = converter.convert(raw)
        assertion = ToolCallAssertion(result, converter_config)

        if converter_config.batches_tool_calls:
            # Both calls batched into one message
            assert assertion.get_length() == 1
            assert assertion.get_tool_call_count_at(0) == 2
        else:
            # Each call separate
            assert assertion.get_length() >= 1


class TestToolErrorHandling:
    """Tests for is_error field handling across frameworks that support it."""

    def test_preserves_is_error_when_true(
        self, converter, converter_config: ConverterConfig
    ):
        """Tool result with is_error=True is handled correctly."""
        if not converter_config.supports_is_error:
            pytest.skip(f"{converter_config.name} doesn't support is_error")

        fmt = get_tool_format(converter_config)
        tool_call = make_tool_call("search", {"query": "test"}, "toolu_123", fmt)
        tool_result_error = make_tool_result("search", "Error: API failed", "toolu_123", is_error=True, fmt=fmt)
        raw = [tool_call, tool_result_error]

        result = converter.convert(raw)
        assertion = ToolCallAssertion(result, converter_config)

        assert assertion.get_length() == 2
        assert assertion.has_is_error_at(1, expected=True)

    def test_handles_is_error_false(
        self, converter, converter_config: ConverterConfig
    ):
        """Tool result with is_error=False is handled correctly."""
        if not converter_config.supports_is_error:
            pytest.skip(f"{converter_config.name} doesn't support is_error")

        fmt = get_tool_format(converter_config)
        tool_call = make_tool_call("search", {"query": "test"}, "toolu_123", fmt)
        tool_result_success = make_tool_result("search", "result", "toolu_123", is_error=False, fmt=fmt)
        raw = [tool_call, tool_result_success]

        result = converter.convert(raw)
        assertion = ToolCallAssertion(result, converter_config)

        assert assertion.get_length() == 2
        assert assertion.has_is_error_at(1, expected=False)


class TestMalformedToolJson:
    """Tests for malformed JSON handling across frameworks."""

    def test_handles_malformed_tool_call_json(
        self, converter, converter_config: ConverterConfig, caplog
    ):
        """Malformed tool_call JSON is skipped."""
        if converter_config.tool_handling_mode == "skip":
            pytest.skip(f"{converter_config.name} skips tool events")

        raw = [MALFORMED_TOOL_CALL]

        result = converter.convert(raw)
        assertion = ToolCallAssertion(result, converter_config)

        if converter_config.tool_handling_mode == "raw_json":
            # ClaudeSDK includes raw JSON as-is (even if malformed)
            assert assertion.get_length() == 1
        else:
            # Others skip malformed JSON
            assert assertion.get_length() == 0

        if converter_config.logs_malformed_json:
            assert "Failed to parse tool_call" in caplog.text

    def test_handles_malformed_tool_result_json(
        self, converter, converter_config: ConverterConfig, caplog
    ):
        """Malformed tool_result JSON is skipped."""
        if converter_config.tool_handling_mode == "skip":
            pytest.skip(f"{converter_config.name} skips tool events")

        raw = [MALFORMED_TOOL_RESULT]

        result = converter.convert(raw)
        assertion = ToolCallAssertion(result, converter_config)

        if converter_config.tool_handling_mode == "raw_json":
            # ClaudeSDK includes raw JSON as-is
            assert assertion.get_length() == 1
        else:
            # Others skip malformed JSON
            assert assertion.get_length() == 0

        if converter_config.logs_malformed_json:
            assert "Failed to parse tool_result" in caplog.text

    def test_skips_tool_call_with_missing_id(
        self, converter, converter_config: ConverterConfig, caplog
    ):
        """Tool call without tool_call_id is skipped."""
        if converter_config.tool_handling_mode not in ("structured",):
            pytest.skip(f"{converter_config.name} doesn't validate tool_call_id")

        raw = [TOOL_CALL_MISSING_ID]

        result = converter.convert(raw)
        assertion = ToolCallAssertion(result, converter_config)

        assert assertion.get_length() == 0
        if converter_config.logs_malformed_json:
            assert "missing tool_call_id" in caplog.text

    def test_skips_tool_call_with_missing_name(
        self, converter, converter_config: ConverterConfig, caplog
    ):
        """Tool call without name is skipped."""
        if converter_config.tool_handling_mode not in ("structured",):
            pytest.skip(f"{converter_config.name} doesn't validate name")

        raw = [TOOL_CALL_MISSING_NAME]

        result = converter.convert(raw)
        assertion = ToolCallAssertion(result, converter_config)

        assert assertion.get_length() == 0
        if converter_config.logs_malformed_json:
            assert "missing name" in caplog.text


class TestMixedHistory:
    """Integration tests with mixed message types across all frameworks."""

    def test_multi_user_conversation(
        self, converter_with_name, converter_config: ConverterConfig
    ):
        """Handles multiple users in conversation."""
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

        result = converter_with_name.convert(raw)
        assertion = ContentAssertion(result, converter_config)

        if converter_config.skips_own_messages:
            # Agent's own message is skipped
            assert assertion.has_length(2)
        else:
            # Parlant includes all
            assert assertion.has_length(3)

        # Check user messages are present
        assert assertion.content_contains("[Alice]:", 0)
        assert assertion.content_contains("[Bob]:", 1)

    def test_multi_agent_conversation_flow(
        self, converter, converter_config: ConverterConfig
    ):
        """Should include other agents' messages in multi-agent conversations."""
        converter.set_agent_name("Main Agent")
        raw = [
            # User asks Main Agent to get weather
            {
                "role": "user",
                "content": "What's the weather in Tokyo?",
                "sender_name": "Alice",
                "message_type": "text",
            },
            # Main Agent asks Weather Agent (skipped if skips_own_messages)
            {
                "role": "assistant",
                "content": "Let me check with the weather agent.",
                "sender_name": "Main Agent",
                "message_type": "text",
            },
            # Weather Agent responds (included)
            {
                "role": "assistant",
                "content": "Tokyo is 15C and cloudy.",
                "sender_name": "Weather Agent",
                "message_type": "text",
            },
            # Main Agent relays the response (skipped if skips_own_messages)
            {
                "role": "assistant",
                "content": "The weather in Tokyo is 15C and cloudy.",
                "sender_name": "Main Agent",
                "message_type": "text",
            },
        ]

        result = converter.convert(raw)
        assertion = ContentAssertion(result, converter_config)

        if converter_config.skips_own_messages:
            # Should have: Alice's message + Weather Agent's message
            # (Main Agent's own messages are skipped)
            assert assertion.has_length(2)
        else:
            # Parlant includes all 4
            assert assertion.has_length(4)

        # First message should always be Alice's
        assert assertion.content_contains("[Alice]:", 0)
        assert assertion.content_contains("What's the weather", 0)

    def test_full_conversation_with_tools(
        self, converter_with_name, converter_config: ConverterConfig
    ):
        """Full conversation flow with text messages and tool calls."""
        fmt = get_tool_format(converter_config)
        tool_call = make_tool_call("search", {"query": "test"}, "toolu_123", fmt)
        tool_result = make_tool_result("search", "result data", "toolu_123", fmt=fmt)

        raw = [
            # User asks question
            {
                "role": "user",
                "content": "What's the weather?",
                "sender_name": "Alice",
                "message_type": "text",
            },
            # Tool call
            tool_call,
            # Tool result
            tool_result,
            # Agent responds (skipped if own message)
            {
                "role": "assistant",
                "content": "It's sunny!",
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

        result = converter_with_name.convert(raw)
        assertion = ContentAssertion(result, converter_config)

        # Count depends on tool handling mode and skips_own_messages
        if converter_config.tool_handling_mode == "skip":
            # Only text messages (minus own if skips_own_messages)
            if converter_config.skips_own_messages:
                assert assertion.has_length(2)  # Alice's two messages
            else:
                assert assertion.has_length(3)  # + agent's message
        else:
            # Text + tool messages
            if converter_config.skips_own_messages:
                assert assertion.has_length(4)  # Alice + tool call + tool result + Thanks
            else:
                assert assertion.has_length(5)  # + agent's message

        # User messages should always be present
        assert assertion.content_contains("[Alice]:", 0)
