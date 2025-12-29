"""
Smoke tests - verify basic imports and setup work.
"""

from thenvoi import ThenvoiLink, AgentRuntime, ExecutionContext, AgentTools


def test_can_import_runtime():
    """Verify we can import runtime modules."""
    assert ThenvoiLink is not None
    assert AgentRuntime is not None
    assert ExecutionContext is not None
    assert AgentTools is not None


def test_can_import_langgraph_integrations():
    """Verify we can import LangGraph integration utilities."""
    from thenvoi.integrations.langgraph import (
        agent_tools_to_langchain,
        graph_as_tool,
    )

    assert agent_tools_to_langchain is not None
    assert graph_as_tool is not None


def test_fixtures_work(mock_api_client, mock_websocket, sample_room_message):
    """Verify our test fixtures are properly configured."""
    assert mock_api_client is not None
    assert mock_websocket is not None
    assert sample_room_message.chat_room_id == "room-123"
    assert sample_room_message.sender_type == "User"
