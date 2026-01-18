"""Unit tests for pure formatting functions."""

from thenvoi.runtime.formatters import (
    format_message_for_llm,
    format_history_for_llm,
    build_participants_message,
)


class TestFormatMessageForLlm:
    def test_agent_sender_maps_to_assistant(self):
        msg = {"sender_type": "Agent", "content": "Hello", "sender_name": "Bot"}
        result = format_message_for_llm(msg)
        assert result["role"] == "assistant"
        assert result["content"] == "Hello"
        assert result["sender_name"] == "Bot"

    def test_user_sender_maps_to_user(self):
        msg = {"sender_type": "User", "content": "Hi", "sender_name": "Alice"}
        result = format_message_for_llm(msg)
        assert result["role"] == "user"

    def test_unknown_sender_maps_to_user(self):
        msg = {"sender_type": "", "content": "Test"}
        result = format_message_for_llm(msg)
        assert result["role"] == "user"

    def test_fallback_sender_name_to_type(self):
        # Falls back to sender_type if no name
        msg = {"sender_type": "Agent", "content": ""}
        result = format_message_for_llm(msg)
        assert result["sender_name"] == "Agent"

    def test_fallback_sender_name_to_name_field(self):
        # Falls back to "name" field if sender_name missing
        msg = {"sender_type": "User", "content": "Hi", "name": "Bob"}
        result = format_message_for_llm(msg)
        assert result["sender_name"] == "Bob"

    def test_includes_sender_type(self):
        msg = {"sender_type": "Agent", "content": "Test", "sender_name": "Bot"}
        result = format_message_for_llm(msg)
        assert result["sender_type"] == "Agent"

    def test_preserves_message_type(self):
        # text message
        msg = {"sender_type": "Agent", "content": "Hello", "message_type": "text"}
        result = format_message_for_llm(msg)
        assert result["message_type"] == "text"

        # tool_call message
        msg = {"sender_type": "Agent", "content": "{...}", "message_type": "tool_call"}
        result = format_message_for_llm(msg)
        assert result["message_type"] == "tool_call"

        # tool_result message
        msg = {
            "sender_type": "Agent",
            "content": "{...}",
            "message_type": "tool_result",
        }
        result = format_message_for_llm(msg)
        assert result["message_type"] == "tool_result"

        # thought message
        msg = {
            "sender_type": "Agent",
            "content": "thinking...",
            "message_type": "thought",
        }
        result = format_message_for_llm(msg)
        assert result["message_type"] == "thought"

    def test_defaults_message_type_to_text(self):
        # Missing message_type defaults to "text"
        msg = {"sender_type": "Agent", "content": "Hello"}
        result = format_message_for_llm(msg)
        assert result["message_type"] == "text"

    def test_preserves_metadata(self):
        """Should preserve metadata for adapters that need it (e.g., A2A)."""
        msg = {
            "sender_type": "Agent",
            "content": "A2A task completed",
            "message_type": "task",
            "metadata": {
                "a2a_context_id": "ctx-123",
                "a2a_task_id": "task-456",
                "a2a_task_state": "completed",
            },
        }
        result = format_message_for_llm(msg)
        assert result["metadata"] == {
            "a2a_context_id": "ctx-123",
            "a2a_task_id": "task-456",
            "a2a_task_state": "completed",
        }

    def test_defaults_metadata_to_empty_dict(self):
        """Should default metadata to empty dict if missing."""
        msg = {"sender_type": "Agent", "content": "Hello"}
        result = format_message_for_llm(msg)
        assert result["metadata"] == {}


class TestFormatHistoryForLlm:
    def test_formats_multiple_messages(self):
        messages = [
            {"id": "1", "sender_type": "User", "content": "Hi", "sender_name": "Alice"},
            {
                "id": "2",
                "sender_type": "Agent",
                "content": "Hello",
                "sender_name": "Bot",
            },
        ]
        result = format_history_for_llm(messages)
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"

    def test_excludes_message_by_id(self):
        messages = [
            {"id": "1", "content": "First", "sender_type": "User"},
            {"id": "2", "content": "Second", "sender_type": "User"},
        ]
        result = format_history_for_llm(messages, exclude_id="1")
        assert len(result) == 1
        assert result[0]["content"] == "Second"

    def test_empty_list(self):
        result = format_history_for_llm([])
        assert result == []

    def test_none_exclude_id_includes_all(self):
        messages = [
            {"id": "1", "content": "First", "sender_type": "User"},
            {"id": "2", "content": "Second", "sender_type": "User"},
        ]
        result = format_history_for_llm(messages, exclude_id=None)
        assert len(result) == 2


class TestBuildParticipantsMessage:
    def test_empty_participants(self):
        result = build_participants_message([])
        assert "No other participants" in result

    def test_formats_participants(self):
        participants = [
            {"id": "u1", "name": "Alice", "type": "User"},
            {"id": "a1", "name": "Bot", "type": "Agent"},
        ]
        result = build_participants_message(participants)
        assert "Alice" in result
        assert "Bot" in result
        # IDs are intentionally NOT shown to prevent LLM from using them in mentions
        assert "u1" not in result
        assert "User" in result

    def test_includes_mention_instruction(self):
        participants = [{"id": "1", "name": "Test", "type": "User"}]
        result = build_participants_message(participants)
        assert "send_message" in result
        # Instruction now says to use EXACT name, not ID
        assert "EXACT name" in result

    def test_handles_missing_fields(self):
        participants = [{"id": "1"}]  # Missing name and type
        result = build_participants_message(participants)
        assert "Unknown" in result  # Default for missing name/type
