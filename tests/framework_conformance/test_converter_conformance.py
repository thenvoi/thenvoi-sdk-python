"""Parameterized conformance tests for all history converters.

These tests verify the shared behavioral contract across all six framework
converters. Framework-specific behavior (tool batching, message object types,
etc.) remains in the per-framework test files under tests/converters/.
"""

from __future__ import annotations

import pytest


class TestUserTextMessages:
    """All converters must handle user text messages consistently."""

    def test_converts_user_text_with_sender_name(
        self, converter_config, make_converter, output
    ):
        """User text messages include sender name prefix in output."""
        converter = make_converter()
        raw = [
            {
                "role": "user",
                "content": "Hello, agent!",
                "sender_name": "Alice",
                "message_type": "text",
            }
        ]

        result = converter.convert(raw)

        assert output.result_length(result) == 1
        assert output.content_contains(result, "[Alice]: Hello, agent!")

    def test_handles_empty_sender_name(self, converter_config, make_converter, output):
        """User messages with empty sender_name are handled per framework."""
        converter = make_converter()
        raw = [
            {
                "role": "user",
                "content": "Hello!",
                "sender_name": "",
                "message_type": "text",
            }
        ]

        result = converter.convert(raw)

        behavior = converter_config.empty_sender_behavior
        if behavior == "content_as_is":
            assert output.content_contains(result, "Hello!")
            # Should NOT have a sender prefix
            assert not output.content_contains(result, "[Alice]")
        elif behavior == "brackets_empty":
            assert output.content_contains(result, "[]: Hello!")
        elif behavior == "unknown_prefix":
            assert output.content_contains(result, "[Unknown]: Hello!")

    def test_handles_missing_sender_name(
        self, converter_config, make_converter, output
    ):
        """User messages with no sender_name key are handled per framework."""
        if not converter_config.has_missing_sender_name_test:
            pytest.skip(
                f"{converter_config.display_name} does not have a missing sender name test"
            )

        converter = make_converter()
        raw = [
            {
                "role": "user",
                "content": "Hello!",
                "message_type": "text",
            }
        ]

        result = converter.convert(raw)

        behavior = converter_config.missing_sender_behavior
        if behavior == "content_as_is":
            assert output.content_contains(result, "Hello!")
        elif behavior == "unknown_prefix":
            assert output.content_contains(result, "[Unknown]: Hello!")


class TestEmptyHistory:
    """All converters must handle empty input."""

    def test_empty_history(self, converter_config, make_converter):
        """Empty history returns the framework's empty result."""
        converter = make_converter()

        result = converter.convert([])

        assert result == converter_config.empty_result


class TestMessageTypeDefaults:
    """All converters default to text message type."""

    def test_defaults_to_text_message_type(
        self, converter_config, make_converter, output
    ):
        """Messages without message_type are treated as text."""
        converter = make_converter()
        raw = [
            {
                "role": "user",
                "content": "Hello",
                "sender_name": "Bob",
            }
        ]

        result = converter.convert(raw)

        assert output.result_length(result) >= 1
        assert output.content_contains(result, "Bob")


class TestThoughtMessageSkipping:
    """All converters must skip thought messages."""

    def test_skips_thought_messages(self, converter_config, make_converter, output):
        """Thought messages are always filtered out."""
        converter = make_converter()
        raw = [
            {
                "role": "assistant",
                "content": "I'm thinking about this...",
                "message_type": "thought",
            }
        ]

        result = converter.convert(raw)

        assert output.is_empty(result)


class TestOwnMessageFiltering:
    """Converters filter own messages (except Parlant which includes all)."""

    def test_own_message_handling(self, converter_config, make_converter, output):
        """Own agent text messages: filtered for most, included for Parlant."""
        converter = make_converter(agent_name="Agent")
        raw = [
            {
                "role": "assistant",
                "content": "I'll help you with that.",
                "sender_name": "Agent",
                "message_type": "text",
            }
        ]

        result = converter.convert(raw)

        if converter_config.filters_own_messages:
            assert output.is_empty(result)
        else:
            # Parlant includes own messages
            assert output.result_length(result) == 1

    def test_includes_other_agents_messages(
        self, converter_config, make_converter, output
    ):
        """Other agents' messages are always included."""
        converter = make_converter(agent_name="Main Agent")
        raw = [
            {
                "role": "assistant",
                "content": "It's sunny today!",
                "sender_name": "Weather Agent",
                "message_type": "text",
            }
        ]

        result = converter.convert(raw)

        assert output.result_length(result) == 1

    def test_skips_only_own_keeps_others(
        self, converter_config, make_converter, output
    ):
        """Only own agent messages are filtered; others are kept."""
        converter = make_converter()
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

        if converter_config.filters_own_messages:
            assert output.result_length(result) == 1  # Only Weather Agent
        else:
            assert output.result_length(result) == 2  # Both kept (Parlant)

    def test_set_agent_name_updates_filtering(
        self, converter_config, make_converter, output
    ):
        """set_agent_name dynamically changes which messages are filtered."""
        converter = make_converter()
        raw = [
            {
                "role": "assistant",
                "content": "Hello",
                "sender_name": "Agent",
                "message_type": "text",
            }
        ]

        # Before setting name - all assistant messages included
        assert output.result_length(converter.convert(raw)) == 1

        # After setting name
        converter.set_agent_name("Agent")
        result_after = converter.convert(raw)

        if converter_config.filters_own_messages:
            assert output.is_empty(result_after)
        else:
            # Parlant still includes own messages
            assert output.result_length(result_after) == 1

    def test_includes_all_when_no_agent_name(
        self, converter_config, make_converter, output
    ):
        """When no agent name is set, all assistant messages are included."""
        converter = make_converter()
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

        assert output.result_length(result) == 2


class TestEdgeCases:
    """Edge cases with config-driven behavior."""

    def test_handles_empty_content(self, converter_config, make_converter, output):
        """Messages with empty content: some converters skip, others keep."""
        converter = make_converter()
        raw = [
            {
                "role": "user",
                "content": "",
                "sender_name": "Alice",
                "message_type": "text",
            }
        ]

        result = converter.convert(raw)

        if converter_config.skips_empty_content:
            assert output.is_empty(result)
        else:
            assert output.result_length(result) == 1

    def test_defaults_to_user_role(self, converter_config, make_converter, output):
        """Messages without role default to 'user'."""
        if not converter_config.has_role_concept:
            pytest.skip(
                f"{converter_config.display_name} returns string (no role concept)"
            )

        converter = make_converter()
        raw = [
            {
                "content": "Hello",
                "sender_name": "Bob",
                "message_type": "text",
            }
        ]

        result = converter.convert(raw)

        assert output.get_role(result, 0) == "user"


class TestToolEventHandling:
    """Converters that skip tool events entirely."""

    def test_tool_events_skipped_for_simple_converters(
        self, converter_config, make_converter, output
    ):
        """CrewAI and Parlant skip tool_call/tool_result messages entirely."""
        if not converter_config.skips_tool_events:
            pytest.skip(
                f"{converter_config.display_name} processes tool events (tested below)"
            )

        converter = make_converter()
        raw = [
            {
                "role": "assistant",
                "content": '{"event": "on_tool_start", "name": "search"}',
                "message_type": "tool_call",
            },
            {
                "role": "assistant",
                "content": '{"event": "on_tool_end", "output": "result"}',
                "message_type": "tool_result",
            },
        ]

        result = converter.convert(raw)

        assert output.is_empty(result)


class TestToolEventConversion:
    """Converters that process tool events must convert them to framework format."""

    def test_converts_tool_call_to_framework_format(
        self, converter_config, make_converter, output
    ):
        """Tool_call is converted and the tool name appears in output."""
        if converter_config.skips_tool_events:
            pytest.skip(f"{converter_config.display_name} skips tool events")

        converter = make_converter()
        # Use paired tool_call + tool_result so converters that require pairing (e.g. LangChain) pass
        raw = [
            {
                "role": "assistant",
                "content": '{"name": "search", "args": {"query": "test"}, "tool_call_id": "tc_1"}',
                "message_type": "tool_call",
            },
            {
                "role": "assistant",
                "content": '{"name": "search", "output": "result data", "tool_call_id": "tc_1"}',
                "message_type": "tool_result",
            },
        ]

        result = converter.convert(raw)

        assert not output.is_empty(result)
        # Tool name must appear somewhere in the converted output
        assert output.content_contains(result, "search")
        # Both tool_call and tool_result should produce output entries
        assert output.result_length(result) >= 2

    def test_converts_tool_result_paired_with_call(
        self, converter_config, make_converter, output
    ):
        """Tool_result output content is present and paired with its tool_call_id."""
        if converter_config.skips_tool_events:
            pytest.skip(f"{converter_config.display_name} skips tool events")

        converter = make_converter()
        raw = [
            {
                "role": "assistant",
                "content": '{"name": "lookup", "args": {"id": "42"}, "tool_call_id": "tc_2"}',
                "message_type": "tool_call",
            },
            {
                "role": "assistant",
                "content": '{"name": "lookup", "output": "found item 42", "tool_call_id": "tc_2"}',
                "message_type": "tool_result",
            },
        ]

        result = converter.convert(raw)

        assert output.result_length(result) >= 1
        # Result output content must be present (focus of this test)
        if not converter_config.tool_result_uses_nested_path:
            assert output.content_contains(result, "found item 42")
        # Tool name must also be present
        assert output.content_contains(result, "lookup")

    def test_mixed_history_includes_user_assistant_tool_messages(
        self, converter_config, make_converter, output
    ):
        """User + assistant + tool_call + tool_result is converted in order."""
        if converter_config.skips_tool_events:
            pytest.skip(f"{converter_config.display_name} skips tool events")

        converter = make_converter()
        raw = [
            {
                "role": "user",
                "content": "Run search",
                "sender_name": "Alice",
                "message_type": "text",
            },
            {
                "role": "assistant",
                "content": "Searching...",
                "sender_name": "Agent",
                "message_type": "text",
            },
            {
                "role": "assistant",
                "content": '{"name": "search", "args": {}, "tool_call_id": "tc_1"}',
                "message_type": "tool_call",
            },
            {
                "role": "assistant",
                "content": '{"name": "search", "output": "found", "tool_call_id": "tc_1"}',
                "message_type": "tool_result",
            },
        ]

        result = converter.convert(raw)

        assert output.result_length(result) >= 2
        assert output.content_contains(result, "Alice")
        assert output.content_contains(result, "search")
        if not converter_config.tool_result_uses_nested_path:
            assert output.content_contains(result, "found")
