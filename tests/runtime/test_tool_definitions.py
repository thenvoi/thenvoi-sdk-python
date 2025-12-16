"""
Unit tests for tool_definitions - Pydantic models for platform tools.

Tests:
1. Pydantic models validate correctly
2. Required fields are enforced
3. Enum constraints work
4. JSON schema generation is correct
"""

import pytest
from pydantic import ValidationError

from thenvoi.runtime.tools import (
    TOOL_MODELS,
    SendMessageInput,
    SendEventInput,
    AddParticipantInput,
    LookupPeersInput,
)


class TestSendMessageInput:
    """Tests for SendMessageInput model."""

    def test_valid_message(self):
        """Valid message with content and mentions should pass."""
        msg = SendMessageInput(content="Hello", mentions=["Alice"])
        assert msg.content == "Hello"
        assert msg.mentions == ["Alice"]

    def test_requires_content(self):
        """Content is required."""
        with pytest.raises(ValidationError) as exc_info:
            SendMessageInput(mentions=["Alice"])
        assert "content" in str(exc_info.value)

    def test_requires_mentions(self):
        """Mentions is required."""
        with pytest.raises(ValidationError) as exc_info:
            SendMessageInput(content="Hello")
        assert "mentions" in str(exc_info.value)

    def test_mentions_min_length(self):
        """Mentions must have at least one item."""
        with pytest.raises(ValidationError) as exc_info:
            SendMessageInput(content="Hello", mentions=[])
        assert (
            "min_length" in str(exc_info.value).lower()
            or "at least 1" in str(exc_info.value).lower()
        )


class TestSendEventInput:
    """Tests for SendEventInput model."""

    def test_valid_event(self):
        """Valid event with all fields should pass."""
        event = SendEventInput(
            content="Processing...",
            message_type="thought",
            metadata={"step": 1},
        )
        assert event.content == "Processing..."
        assert event.message_type == "thought"
        assert event.metadata == {"step": 1}

    def test_message_type_enum(self):
        """message_type must be one of the allowed values."""
        # Valid values
        for valid_type in ["thought", "error", "task"]:
            event = SendEventInput(content="Test", message_type=valid_type)
            assert event.message_type == valid_type

        # Invalid value
        with pytest.raises(ValidationError) as exc_info:
            SendEventInput(content="Test", message_type="invalid")
        assert "message_type" in str(exc_info.value)

    def test_metadata_optional(self):
        """Metadata should be optional."""
        event = SendEventInput(content="Test", message_type="thought")
        assert event.metadata is None


class TestAddParticipantInput:
    """Tests for AddParticipantInput model."""

    def test_valid_add(self):
        """Valid add with name should pass."""
        add = AddParticipantInput(name="Bob")
        assert add.name == "Bob"
        assert add.role == "member"  # default

    def test_role_enum(self):
        """role must be one of the allowed values."""
        for valid_role in ["owner", "admin", "member"]:
            add = AddParticipantInput(name="Bob", role=valid_role)
            assert add.role == valid_role

        with pytest.raises(ValidationError):
            AddParticipantInput(name="Bob", role="invalid")


class TestLookupPeersInput:
    """Tests for LookupPeersInput model."""

    def test_defaults(self):
        """Default values should be applied."""
        lookup = LookupPeersInput()
        assert lookup.page == 1
        assert lookup.page_size == 50

    def test_page_size_max(self):
        """page_size should have max constraint."""
        with pytest.raises(ValidationError):
            LookupPeersInput(page_size=101)


class TestToolModelsRegistry:
    """Tests for the TOOL_MODELS registry."""

    def test_all_tools_registered(self):
        """All expected tools should be in the registry."""
        expected = {
            "send_message",
            "send_event",
            "add_participant",
            "remove_participant",
            "lookup_peers",
            "get_participants",
            "create_chatroom",
        }
        assert set(TOOL_MODELS.keys()) == expected

    def test_models_have_docstrings(self):
        """All models should have docstrings for LLM descriptions."""
        for name, model in TOOL_MODELS.items():
            assert model.__doc__, f"{name} should have a docstring"

    def test_json_schema_generation(self):
        """All models should generate valid JSON schemas."""
        for name, model in TOOL_MODELS.items():
            schema = model.model_json_schema()
            assert "properties" in schema or "type" in schema, (
                f"{name} should generate valid schema"
            )
