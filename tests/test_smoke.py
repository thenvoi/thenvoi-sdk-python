"""
Smoke tests - verify basic imports and setup work.
"""

from thenvoi.core import ThenvoiAgent, AgentSession, AgentTools


def test_can_import_core():
    """Verify we can import core modules."""
    assert ThenvoiAgent is not None
    assert AgentSession is not None
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
