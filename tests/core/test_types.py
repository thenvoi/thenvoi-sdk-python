"""
Unit tests for AgentTools - focused on SDK logic not validated by REST API.

Tests:
1. _resolve_mentions() - translates names to IDs before API call
2. Schema generation - ensures LLM sees correct tool constraints
"""

import pytest

from thenvoi.agent.core.types import AgentTools


class TestResolveMentions:
    """Tests for _resolve_mentions() - SDK logic that runs before API calls."""

    def test_resolves_string_names_to_dicts(
        self, mock_thenvoi_agent, mock_agent_session
    ):
        """String names should be resolved to {id, name} dicts."""
        mock_agent_session.participants = [
            {"id": "user-456", "name": "Test User"},
            {"id": "agent-789", "name": "Helper Bot"},
        ]
        mock_thenvoi_agent.active_sessions = {"room-123": mock_agent_session}

        tools = AgentTools(room_id="room-123", coordinator=mock_thenvoi_agent)
        result = tools._resolve_mentions(["Test User", "Helper Bot"])

        assert result == [
            {"id": "user-456", "name": "Test User"},
            {"id": "agent-789", "name": "Helper Bot"},
        ]

    def test_raises_for_unknown_participant(
        self, mock_thenvoi_agent, mock_agent_session
    ):
        """Unknown name should raise ValueError with available names."""
        mock_agent_session.participants = [{"id": "user-456", "name": "Test User"}]
        mock_thenvoi_agent.active_sessions = {"room-123": mock_agent_session}

        tools = AgentTools(room_id="room-123", coordinator=mock_thenvoi_agent)

        with pytest.raises(ValueError) as exc_info:
            tools._resolve_mentions(["Unknown Person"])

        assert "Unknown participant 'Unknown Person'" in str(exc_info.value)


class TestExecuteToolCall:
    """Tests for execute_tool_call() dispatcher."""

    async def test_unknown_tool_raises_value_error(self, mock_thenvoi_agent):
        """Unknown tool name should raise ValueError."""
        tools = AgentTools(room_id="room-123", coordinator=mock_thenvoi_agent)

        with pytest.raises(ValueError) as exc_info:
            await tools.execute_tool_call("nonexistent_tool", {})

        assert "Unknown tool: nonexistent_tool" in str(exc_info.value)

    async def test_all_tools_are_mapped(self, mock_thenvoi_agent):
        """All expected tools should be in the dispatcher map."""
        tools = AgentTools(room_id="room-123", coordinator=mock_thenvoi_agent)

        expected_tools = [
            "send_message",
            "send_event",
            "add_participant",
            "remove_participant",
            "lookup_peers",
            "get_participants",
            "create_chatroom",
        ]

        # This will raise ValueError if any tool is missing from the map
        for tool_name in expected_tools:
            # We don't care about the result, just that it doesn't raise "Unknown tool"
            try:
                await tools.execute_tool_call(
                    tool_name,
                    {"content": "test", "message_type": "thought", "name": "Test"},
                )
            except (KeyError, TypeError):
                # Expected - we're passing incomplete args, but tool IS in the map
                pass


class TestSchemaGeneration:
    """Tests for tool schema generation - consumed by LLM, not REST API."""

    def test_langchain_send_event_has_enum_constraint(self, mock_thenvoi_agent):
        """CRITICAL: send_event message_type must have enum so LLM knows valid values."""
        tools = AgentTools(room_id="room-123", coordinator=mock_thenvoi_agent)

        lc_tools = tools.to_langchain_tools()
        send_event = next(t for t in lc_tools if t.name == "send_event")

        schema = send_event.args_schema.model_json_schema()
        assert "enum" in schema["properties"]["message_type"]
        assert set(schema["properties"]["message_type"]["enum"]) == {
            "thought",
            "error",
            "task",
        }

    def test_openai_send_event_has_enum_constraint(self, mock_thenvoi_agent):
        """OpenAI schema should also have enum constraint."""
        tools = AgentTools(room_id="room-123", coordinator=mock_thenvoi_agent)

        openai_tools = tools.to_openai_tools()
        send_event = next(
            t for t in openai_tools if t["function"]["name"] == "send_event"
        )

        params = send_event["function"]["parameters"]
        assert "enum" in params["properties"]["message_type"]
        assert set(params["properties"]["message_type"]["enum"]) == {
            "thought",
            "error",
            "task",
        }
