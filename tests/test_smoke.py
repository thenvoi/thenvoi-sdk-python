from thenvoi import ThenvoiLink, AgentRuntime, ExecutionContext, AgentTools


def test_can_import_runtime():
    assert ThenvoiLink
    assert AgentRuntime
    assert ExecutionContext
    assert AgentTools


def test_can_import_langgraph_integrations():
    from thenvoi.integrations.langgraph import (
        agent_tools_to_langchain,
        graph_as_tool,
    )

    assert agent_tools_to_langchain
    assert graph_as_tool


def test_fixtures_work(sample_room_message):
    assert sample_room_message.chat_room_id == "room-123"
    assert sample_room_message.sender_type == "User"
