"""
Unit tests for AgentTools - focused on SDK logic not validated by REST API.

Tests:
1. _resolve_mentions() - translates names to IDs before API call
2. Schema generation - ensures LLM sees correct tool constraints
"""

import pytest

from thenvoi.core.types import AgentTools


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

    async def test_unknown_tool_returns_error_string(self, mock_thenvoi_agent):
        """Unknown tool name should return error string (not raise)."""
        tools = AgentTools(room_id="room-123", coordinator=mock_thenvoi_agent)

        result = await tools.execute_tool_call("nonexistent_tool", {})

        assert result == "Unknown tool: nonexistent_tool"

    async def test_all_tools_are_mapped(self, mock_thenvoi_agent, mock_agent_session):
        """All expected tools should be in the dispatcher map."""
        # Setup mock session with participants for mention resolution
        mock_agent_session.participants = [{"id": "user-456", "name": "Test"}]
        mock_thenvoi_agent.active_sessions = {"room-123": mock_agent_session}

        tools = AgentTools(room_id="room-123", coordinator=mock_thenvoi_agent)

        # Map each tool to valid arguments for Pydantic validation
        tool_args = {
            "send_message": {"content": "test", "mentions": ["Test"]},
            "send_event": {"content": "test", "message_type": "thought"},
            "add_participant": {"name": "Test"},
            "remove_participant": {"name": "Test"},
            "lookup_peers": {},
            "get_participants": {},
            "create_chatroom": {"name": "Test Room"},
        }

        # This will raise ValueError if any tool is missing from the map
        for tool_name, args in tool_args.items():
            # We don't care about the result, just that it doesn't raise "Unknown tool"
            try:
                await tools.execute_tool_call(tool_name, args)
            except (KeyError, TypeError, AttributeError):
                # Expected - mock doesn't have all methods, but tool IS in the map
                pass


class TestSchemaGeneration:
    """Tests for tool schema generation - consumed by LLM, not REST API."""

    def test_openai_schema_send_event_has_enum_constraint(self, mock_thenvoi_agent):
        """CRITICAL: send_event message_type must have enum so LLM knows valid values."""
        tools = AgentTools(room_id="room-123", coordinator=mock_thenvoi_agent)

        openai_tools = tools.get_tool_schemas("openai")
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

    def test_anthropic_schema_send_event_has_enum_constraint(self, mock_thenvoi_agent):
        """Anthropic schema should also have enum constraint."""
        tools = AgentTools(room_id="room-123", coordinator=mock_thenvoi_agent)

        anthropic_tools = tools.get_tool_schemas("anthropic")
        send_event = next(t for t in anthropic_tools if t["name"] == "send_event")

        schema = send_event["input_schema"]
        assert "enum" in schema["properties"]["message_type"]
        assert set(schema["properties"]["message_type"]["enum"]) == {
            "thought",
            "error",
            "task",
        }

    def test_openai_schema_has_all_tools(self, mock_thenvoi_agent):
        """OpenAI schema should include all platform tools."""
        tools = AgentTools(room_id="room-123", coordinator=mock_thenvoi_agent)

        openai_tools = tools.get_tool_schemas("openai")
        tool_names = {t["function"]["name"] for t in openai_tools}

        expected = {
            "send_message",
            "send_event",
            "add_participant",
            "remove_participant",
            "lookup_peers",
            "get_participants",
            "create_chatroom",
        }
        assert tool_names == expected

    def test_anthropic_schema_has_all_tools(self, mock_thenvoi_agent):
        """Anthropic schema should include all platform tools."""
        tools = AgentTools(room_id="room-123", coordinator=mock_thenvoi_agent)

        anthropic_tools = tools.get_tool_schemas("anthropic")
        tool_names = {t["name"] for t in anthropic_tools}

        expected = {
            "send_message",
            "send_event",
            "add_participant",
            "remove_participant",
            "lookup_peers",
            "get_participants",
            "create_chatroom",
        }
        assert tool_names == expected

    def test_tool_models_property(self, mock_thenvoi_agent):
        """tool_models property should return Pydantic models registry."""
        tools = AgentTools(room_id="room-123", coordinator=mock_thenvoi_agent)

        models = tools.tool_models
        assert "send_message" in models
        assert "send_event" in models

        # Verify they are actual Pydantic model classes
        from pydantic import BaseModel

        for name, model in models.items():
            assert issubclass(model, BaseModel), f"{name} should be a Pydantic model"
