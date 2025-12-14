"""
Smoke tests - verify basic imports and setup work.
"""

from thenvoi.agent.core import ThenvoiAgent, AgentSession, AgentTools


def test_can_import_core():
    """Verify we can import core modules."""
    assert ThenvoiAgent is not None
    assert AgentSession is not None
    assert AgentTools is not None


def test_can_import_langgraph():
    """Verify we can import LangGraph adapter."""
    from thenvoi.agent.langgraph import LangGraphAdapter, with_langgraph

    assert LangGraphAdapter is not None
    assert with_langgraph is not None


def test_fixtures_work(mock_api_client, mock_websocket, sample_room_message):
    """Verify our test fixtures are properly configured."""
    assert mock_api_client is not None
    assert mock_websocket is not None
    assert sample_room_message.chat_room_id == "room-123"
    assert sample_room_message.sender_type == "User"
