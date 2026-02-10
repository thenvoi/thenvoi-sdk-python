"""Unit tests for output adapters.

These test the output adapter logic directly (especially edge cases in
StringOutputAdapter._split_messages) rather than relying on transitive
coverage through the conformance tests.
"""

from __future__ import annotations

import json

import pytest

from tests.framework_configs.converters import CONVERTER_CONFIGS
from tests.framework_configs.fixtures import (
    REQUIRED_TOOL_EVENT_KEYS,
    TOOL_CALL_LOOKUP,
    TOOL_CALL_SEARCH,
    TOOL_CALL_SEARCH_EMPTY,
    TOOL_RESULT_LOOKUP,
    TOOL_RESULT_SEARCH,
    TOOL_RESULT_SEARCH_FOUND,
)
from tests.framework_configs.output_adapters import (
    DictListOutputAdapter,
    SenderMetadataDictListOutputAdapter,
    StringOutputAdapter,
)

# Group fixtures by event type for parameterized key-path validation.
TOOL_CALL_FIXTURES = [TOOL_CALL_SEARCH, TOOL_CALL_LOOKUP, TOOL_CALL_SEARCH_EMPTY]
TOOL_RESULT_FIXTURES = [
    TOOL_RESULT_SEARCH,
    TOOL_RESULT_LOOKUP,
    TOOL_RESULT_SEARCH_FOUND,
]


def _converter_configs_that_process_tools():
    """Return ConverterConfigs that do NOT skip tool events."""
    return [c for c in CONVERTER_CONFIGS if not c.skips_tool_events]


# ---------------------------------------------------------------------------
# StringOutputAdapter
# ---------------------------------------------------------------------------


class TestStringOutputAdapterSplitMessages:
    """Tests for the regex-based message splitting logic."""

    def test_empty_string(self):
        assert StringOutputAdapter._split_messages("") == []

    def test_single_text_message(self):
        assert StringOutputAdapter._split_messages("[Alice]: Hello!") == [
            "[Alice]: Hello!"
        ]

    def test_two_text_messages(self):
        result = StringOutputAdapter._split_messages(
            "[Alice]: Hello!\n[Bob]: Hi there!"
        )
        assert result == ["[Alice]: Hello!", "[Bob]: Hi there!"]

    def test_text_then_json_tool_event(self):
        result = StringOutputAdapter._split_messages(
            '[Alice]: Run search\n{"tool": "search", "args": {}}'
        )
        assert result == ["[Alice]: Run search", '{"tool": "search", "args": {}}']

    def test_preserves_newline_inside_plain_text(self):
        """Newlines not followed by [ or { stay within the message."""
        result = StringOutputAdapter._split_messages("[Alice]: Line one\nLine two")
        assert result == ["[Alice]: Line one\nLine two"]

    def test_json_with_embedded_newline(self):
        """JSON value containing \\n should not split mid-message."""
        # A tool result whose output value contains a plain newline
        msg = '{"tool": "search", "output": "line1\nline2"}'
        result = StringOutputAdapter._split_messages(msg)
        assert result == [msg]

    def test_json_with_embedded_newline_bracket(self):
        """JSON value containing \\n[ should not split when [ is not a sender prefix."""
        msg = '{"output": "before\n[after"}'
        result = StringOutputAdapter._split_messages(msg)
        # The sender-prefix check prevents false splits: "[after"}" does
        # not match the [name]: pattern, so it stays part of the JSON message.
        assert result == [msg]

    def test_multiple_json_events(self):
        result = StringOutputAdapter._split_messages(
            '{"call": "search"}\n{"result": "found"}'
        )
        assert result == ['{"call": "search"}', '{"result": "found"}']

    def test_three_messages_mixed(self):
        raw = '[Alice]: Hi\n{"tool": "lookup"}\n[Bob]: Thanks'
        result = StringOutputAdapter._split_messages(raw)
        assert result == ["[Alice]: Hi", '{"tool": "lookup"}', "[Bob]: Thanks"]


class TestStringOutputAdapterMethods:
    """Tests for the public OutputAdapter interface on StringOutputAdapter."""

    def test_result_length(self):
        adapter = StringOutputAdapter()
        assert adapter.result_length("[A]: Hi\n[B]: Bye") == 2
        assert adapter.result_length("") == 0

    def test_get_content(self):
        adapter = StringOutputAdapter()
        assert adapter.get_content("[A]: Hi\n[B]: Bye", 0) == "[A]: Hi"
        assert adapter.get_content("[A]: Hi\n[B]: Bye", 1) == "[B]: Bye"

    def test_is_empty(self):
        adapter = StringOutputAdapter()
        assert adapter.is_empty("") is True
        assert adapter.is_empty("[A]: Hi") is False

    def test_content_contains(self):
        adapter = StringOutputAdapter()
        assert adapter.content_contains("[A]: Hello world", "Hello") is True
        assert adapter.content_contains("[A]: Hello world", "missing") is False

    def test_get_role_raises(self):
        adapter = StringOutputAdapter()
        with pytest.raises(NotImplementedError):
            adapter.get_role("[A]: Hi", 0)

    def test_assert_sender_metadata_raises(self):
        adapter = StringOutputAdapter()
        with pytest.raises(NotImplementedError):
            adapter.assert_sender_metadata("[A]: Hi", 0, "A")

    def test_assert_element_type_passes_for_string(self):
        adapter = StringOutputAdapter()
        adapter.assert_element_type("[A]: Hi", 0, "user")

    def test_assert_element_type_fails_for_non_string(self):
        adapter = StringOutputAdapter()
        with pytest.raises(AssertionError, match="Expected str"):
            adapter.assert_element_type(["not a string"], 0, "user")


# ---------------------------------------------------------------------------
# DictListOutputAdapter
# ---------------------------------------------------------------------------


class TestDictListOutputAdapter:
    """Tests for Anthropic-style dict list adapter."""

    def test_result_length(self):
        adapter = DictListOutputAdapter()
        assert adapter.result_length([{"role": "user", "content": "hi"}]) == 1
        assert adapter.result_length([]) == 0

    def test_get_content_string(self):
        adapter = DictListOutputAdapter()
        result = [{"role": "user", "content": "Hello"}]
        assert adapter.get_content(result, 0) == "Hello"

    def test_get_content_tool_use_blocks(self):
        adapter = DictListOutputAdapter()
        result = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "search"},
                    {"type": "text", "text": "Searching..."},
                ],
            }
        ]
        assert adapter.get_content(result, 0) == "Searching..."

    def test_get_content_tool_use_blocks_no_text(self):
        adapter = DictListOutputAdapter()
        result = [
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "t1", "name": "search"}],
            }
        ]
        # Falls back to str representation
        content = adapter.get_content(result, 0)
        assert "tool_use" in content

    def test_get_role(self):
        adapter = DictListOutputAdapter()
        result = [{"role": "user", "content": "hi"}]
        assert adapter.get_role(result, 0) == "user"

    def test_content_contains_string_content(self):
        adapter = DictListOutputAdapter()
        result = [{"role": "user", "content": "Hello world"}]
        assert adapter.content_contains(result, "Hello") is True
        assert adapter.content_contains(result, "missing") is False

    def test_content_contains_block_content(self):
        adapter = DictListOutputAdapter()
        result = [
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "name": "search"}],
            }
        ]
        assert adapter.content_contains(result, "search") is True
        assert adapter.content_contains(result, "missing") is False

    def test_assert_element_type_passes(self):
        adapter = DictListOutputAdapter()
        result = [{"role": "user", "content": "hi"}]
        adapter.assert_element_type(result, 0, "user")

    def test_assert_element_type_wrong_role(self):
        adapter = DictListOutputAdapter()
        result = [{"role": "user", "content": "hi"}]
        with pytest.raises(AssertionError, match="Expected role='assistant'"):
            adapter.assert_element_type(result, 0, "assistant")

    def test_assert_sender_metadata_raises(self):
        adapter = DictListOutputAdapter()
        result = [{"role": "user", "content": "hi"}]
        with pytest.raises(NotImplementedError):
            adapter.assert_sender_metadata(result, 0, "Alice")


# ---------------------------------------------------------------------------
# SenderMetadataDictListOutputAdapter
# ---------------------------------------------------------------------------


class TestSenderMetadataDictListOutputAdapter:
    """Tests for CrewAI/Parlant-style dict list adapter."""

    def test_result_length(self):
        adapter = SenderMetadataDictListOutputAdapter()
        result = [
            {"role": "user", "content": "hi", "sender": "A", "sender_type": "User"}
        ]
        assert adapter.result_length(result) == 1

    def test_get_content(self):
        adapter = SenderMetadataDictListOutputAdapter()
        result = [{"role": "user", "content": "Hello", "sender": "A"}]
        assert adapter.get_content(result, 0) == "Hello"

    def test_get_role(self):
        adapter = SenderMetadataDictListOutputAdapter()
        result = [{"role": "assistant", "content": "hi"}]
        assert adapter.get_role(result, 0) == "assistant"

    def test_content_contains(self):
        adapter = SenderMetadataDictListOutputAdapter()
        result = [{"role": "user", "content": "Hello world"}]
        assert adapter.content_contains(result, "Hello") is True
        assert adapter.content_contains(result, "missing") is False

    def test_assert_sender_metadata_passes(self):
        adapter = SenderMetadataDictListOutputAdapter()
        result = [
            {"role": "user", "content": "hi", "sender": "Alice", "sender_type": "User"}
        ]
        adapter.assert_sender_metadata(result, 0, "Alice", "User")

    def test_assert_sender_metadata_wrong_sender(self):
        adapter = SenderMetadataDictListOutputAdapter()
        result = [{"role": "user", "content": "hi", "sender": "Alice"}]
        with pytest.raises(AssertionError, match="Expected sender='Bob'"):
            adapter.assert_sender_metadata(result, 0, "Bob")

    def test_assert_sender_metadata_wrong_sender_type(self):
        adapter = SenderMetadataDictListOutputAdapter()
        result = [
            {"role": "user", "content": "hi", "sender": "Alice", "sender_type": "User"}
        ]
        with pytest.raises(AssertionError, match="Expected sender_type='Agent'"):
            adapter.assert_sender_metadata(result, 0, "Alice", "Agent")

    def test_assert_sender_metadata_skips_type_when_none(self):
        adapter = SenderMetadataDictListOutputAdapter()
        result = [{"role": "user", "content": "hi", "sender": "Alice"}]
        # Should not raise -- sender_type check is skipped when None
        adapter.assert_sender_metadata(result, 0, "Alice")

    def test_assert_element_type_passes(self):
        adapter = SenderMetadataDictListOutputAdapter()
        result = [{"role": "user", "content": "hi"}]
        adapter.assert_element_type(result, 0, "user")

    def test_assert_element_type_wrong_role(self):
        adapter = SenderMetadataDictListOutputAdapter()
        result = [{"role": "user", "content": "hi"}]
        with pytest.raises(AssertionError, match="Expected role='assistant'"):
            adapter.assert_element_type(result, 0, "assistant")


# ---------------------------------------------------------------------------
# Fixture payload validation
# ---------------------------------------------------------------------------


class TestFixturePayloadValidation:
    """Verify that the shared fixture payloads satisfy every non-tool-skipping converter.

    This catches the scenario where a new converter is added but the shared
    tool-event payloads in fixtures.py don't contain the keys it needs.
    """

    @pytest.fixture(
        params=_converter_configs_that_process_tools(), ids=lambda c: c.framework_id
    )
    def payload_converter_config(self, request):
        return request.param

    def test_shared_fixture_produces_nonempty_output(self, payload_converter_config):
        """converter.convert([TOOL_CALL_SEARCH, TOOL_RESULT_SEARCH]) must produce non-empty output."""
        converter = payload_converter_config.converter_factory()
        result = converter.convert([TOOL_CALL_SEARCH, TOOL_RESULT_SEARCH])

        adapter = payload_converter_config.output_adapter
        assert not adapter.is_empty(result), (
            f"{payload_converter_config.display_name} converter produced empty output "
            f"from shared fixture payloads — check that fixtures.py includes "
            f"the keys this converter reads"
        )


# ---------------------------------------------------------------------------
# REQUIRED_TOOL_EVENT_KEYS validation
# ---------------------------------------------------------------------------


def _resolve_dotted_path(data: dict, path: str) -> object:
    """Resolve a dot-notation key path (e.g. 'data.input') into a nested dict."""
    current: object = data
    for segment in path.split("."):
        assert isinstance(current, dict), (
            f"Expected dict at segment {segment!r} of path {path!r}, "
            f"got {type(current).__name__}"
        )
        assert segment in current, (
            f"Key {segment!r} missing at path {path!r} in payload: "
            f"{json.dumps(data, indent=2)}"
        )
        current = current[segment]
    return current


class TestRequiredToolEventKeys:
    """Validate that REQUIRED_TOOL_EVENT_KEYS paths exist in every fixture payload.

    This turns the REQUIRED_TOOL_EVENT_KEYS registry from documentation into an
    enforceable contract: if the fixture schema changes or a key path is wrong,
    the test fails immediately instead of silently drifting.
    """

    @pytest.fixture(
        params=[
            ("tool_call", TOOL_CALL_FIXTURES),
            ("tool_result", TOOL_RESULT_FIXTURES),
        ],
        ids=["tool_call", "tool_result"],
    )
    def event_type_and_fixtures(self, request):
        return request.param

    @pytest.fixture(
        params=list(REQUIRED_TOOL_EVENT_KEYS.items()),
        ids=lambda item: item[0],
    )
    def framework_keys(self, request):
        return request.param

    def test_required_keys_exist_in_fixture_payloads(
        self, framework_keys, event_type_and_fixtures
    ):
        """Every dot-path in REQUIRED_TOOL_EVENT_KEYS resolves in every fixture payload."""
        _framework_id, key_map = framework_keys
        event_type, fixtures = event_type_and_fixtures

        required_paths = key_map.get(event_type, [])
        for fixture in fixtures:
            parsed = json.loads(fixture["content"])
            for path in required_paths:
                _resolve_dotted_path(parsed, path)
