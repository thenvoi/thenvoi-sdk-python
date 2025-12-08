"""
Tests for LangGraph tools.

These test the user-facing LangGraph tool APIs that LLMs use to interact
with the Thenvoi platform. Critical behaviors:
- room_id comes from config (thread_id), NOT from LLM parameters
- Required parameters are enforced
- Input validation works correctly
"""

import pytest
from unittest.mock import AsyncMock
from langchain_core.tools import ToolException
from thenvoi.client.rest import AsyncRestClient
from thenvoi.agent.langgraph.tools import get_thenvoi_tools


def test_send_message_signature_does_not_expose_room_id():
    """CRITICAL: LLM should not be able to specify room_id as a parameter."""
    client = AsyncRestClient(api_key="test-key", base_url="http://localhost")
    tools = get_thenvoi_tools(client=client, agent_id="agent-123")

    send_message_tool = tools[0]
    schema = send_message_tool.args_schema.model_json_schema()
    properties = schema.get("properties", {})
    required_fields = schema.get("required", [])

    # Verify tool has content and mentions parameters, but NOT room_id
    assert "content" in properties
    assert "mentions" in properties
    assert "room_id" not in properties

    # Both content and mentions should be required
    assert "content" in required_fields
    assert "mentions" in required_fields, (
        "mentions must be required - prevents agents from forgetting to include mentions"
    )

    # config is NOT in properties - it's hidden from LLM and injected by framework
    assert "config" not in properties


def test_add_participant_signature_does_not_expose_room_id():
    """CRITICAL: LLM should not be able to specify room_id as a parameter."""
    client = AsyncRestClient(api_key="test-key", base_url="http://localhost")
    tools = get_thenvoi_tools(client=client, agent_id="agent-123")

    add_participant_tool = tools[1]
    schema = add_participant_tool.args_schema.model_json_schema()
    properties = schema.get("properties", {})
    required_fields = schema.get("required", [])

    # Verify tool has participant_id and role parameters, but NOT room_id
    assert "participant_id" in properties
    assert "role" in properties
    assert "room_id" not in properties
    assert "config" not in properties

    # participant_id should be required
    assert "participant_id" in required_fields


def test_remove_participant_signature_does_not_expose_room_id():
    """CRITICAL: LLM should not be able to specify room_id as a parameter."""
    client = AsyncRestClient(api_key="test-key", base_url="http://localhost")
    tools = get_thenvoi_tools(client=client, agent_id="agent-123")

    remove_participant_tool = tools[2]
    schema = remove_participant_tool.args_schema.model_json_schema()
    properties = schema.get("properties", {})
    required_fields = schema.get("required", [])

    # Verify tool has participant_id parameter, but NOT room_id
    assert "participant_id" in properties
    assert "room_id" not in properties
    assert "config" not in properties

    # participant_id should be required
    assert "participant_id" in required_fields


def test_get_participants_signature_does_not_expose_room_id():
    """CRITICAL: LLM should not be able to specify room_id as a parameter."""
    client = AsyncRestClient(api_key="test-key", base_url="http://localhost")
    tools = get_thenvoi_tools(client=client, agent_id="agent-123")

    get_participants_tool = tools[3]
    schema = get_participants_tool.args_schema.model_json_schema()
    properties = schema.get("properties", {})

    # This tool takes no LLM parameters - room_id comes only from config
    assert "room_id" not in properties
    assert "config" not in properties


def test_list_available_participants_signature_does_not_expose_room_id():
    """CRITICAL: LLM should not be able to specify room_id as a parameter."""
    client = AsyncRestClient(api_key="test-key", base_url="http://localhost")
    tools = get_thenvoi_tools(client=client, agent_id="agent-123")

    list_available_tool = tools[4]
    schema = list_available_tool.args_schema.model_json_schema()
    properties = schema.get("properties", {})
    required_fields = schema.get("required", [])

    # Should have participant_type parameter, but NOT room_id
    assert "participant_type" in properties
    assert "room_id" not in properties
    assert "config" not in properties

    # participant_type should be required
    assert "participant_type" in required_fields


@pytest.mark.asyncio
async def test_send_message_rejects_empty_mentions():
    """Empty mentions must be rejected - at least one mention is required."""
    # Create a mock client
    mock_client = AsyncMock()
    mock_result = AsyncMock()
    mock_client.chat_messages.create_chat_message = AsyncMock(return_value=mock_result)

    # Create the tools with the mock client
    tools = get_thenvoi_tools(client=mock_client, agent_id="agent-123")

    send_message_tool = tools[0]

    # Create config with thread_id (room_id)
    config = {"configurable": {"thread_id": "room-456"}}

    # Call the tool with empty mentions array - should raise ToolException
    with pytest.raises(ToolException, match="At least one mention is required"):
        await send_message_tool.ainvoke(
            {"content": "Hello", "mentions": "[]"},  # Empty array as JSON string
            config=config,
        )

    # Verify create_chat_message was NOT called
    assert not mock_client.chat_messages.create_chat_message.called


@pytest.mark.asyncio
async def test_send_message_rejects_invalid_json():
    """Invalid JSON in mentions must raise ToolException with helpful message."""
    mock_client = AsyncMock()
    mock_client.chat_messages.create_chat_message = AsyncMock()

    tools = get_thenvoi_tools(client=mock_client, agent_id="agent-123")
    send_message_tool = tools[0]

    config = {"configurable": {"thread_id": "room-456"}}

    # LLM sometimes outputs two arrays instead of one (malformed JSON)
    malformed_mentions = (
        '[{"id":"uuid1","username":"user1"}],[{"id":"uuid2","username":"user2"}]'
    )

    with pytest.raises(ToolException, match="Invalid JSON in mentions parameter"):
        await send_message_tool.ainvoke(
            {"content": "Hello", "mentions": malformed_mentions},
            config=config,
        )

    assert not mock_client.chat_messages.create_chat_message.called
