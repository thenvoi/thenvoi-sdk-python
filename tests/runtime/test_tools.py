"""Tests for AgentTools."""

from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from thenvoi.runtime.tools import (
    TOOL_MODELS,
    AgentTools,
    SendMessageInput,
    SendEventInput,
    AddParticipantInput,
    LookupPeersInput,
    GetParticipantsInput,
    CreateChatroomInput,
)


@pytest.fixture
def mock_rest_client():
    """Mock AsyncRestClient for testing AgentTools."""
    client = MagicMock()
    client.agent_api = MagicMock()

    # Mock create_agent_chat_message
    message_response = MagicMock()
    message_response.data = MagicMock()
    message_response.data.model_dump.return_value = {
        "id": "msg-123",
        "content": "Hello",
        "sender_id": "agent-1",
    }
    client.agent_api.create_agent_chat_message = AsyncMock(
        return_value=message_response
    )

    # Mock create_agent_chat_event
    event_response = MagicMock()
    event_response.data = MagicMock()
    event_response.data.model_dump.return_value = {
        "id": "evt-123",
        "content": "Thinking...",
        "message_type": "thought",
    }
    client.agent_api.create_agent_chat_event = AsyncMock(return_value=event_response)

    # Mock list_agent_chat_participants
    participant1 = MagicMock()
    participant1.id = "user-1"
    participant1.name = "User One"
    participant1.type = "User"
    client.agent_api.list_agent_chat_participants = AsyncMock(
        return_value=MagicMock(data=[participant1])
    )

    # Mock list_agent_peers
    peer1 = MagicMock()
    peer1.id = "agent-2"
    peer1.name = "Agent Two"
    peer1.type = "Agent"
    peer1.description = "Another agent"
    peers_response = MagicMock()
    peers_response.data = [peer1]
    peers_response.metadata = MagicMock()
    peers_response.metadata.page = 1
    peers_response.metadata.page_size = 50
    peers_response.metadata.total_count = 1
    peers_response.metadata.total_pages = 1
    client.agent_api.list_agent_peers = AsyncMock(return_value=peers_response)

    # Mock add_agent_chat_participant
    client.agent_api.add_agent_chat_participant = AsyncMock()

    # Mock remove_agent_chat_participant
    client.agent_api.remove_agent_chat_participant = AsyncMock()

    return client


@pytest.fixture
def participants():
    """Sample participants list."""
    return [
        {"id": "user-1", "name": "User One", "type": "User"},
        {"id": "user-2", "name": "User Two", "type": "User"},
    ]


class TestAgentToolsConstruction:
    """Test AgentTools initialization."""

    def test_init_stores_room_id(self, mock_rest_client):
        """Should store room_id."""
        tools = AgentTools("room-123", mock_rest_client)

        assert tools.room_id == "room-123"

    def test_init_stores_rest_client(self, mock_rest_client):
        """Should store REST client."""
        tools = AgentTools("room-123", mock_rest_client)

        assert tools.rest is mock_rest_client

    def test_init_empty_participants_by_default(self, mock_rest_client):
        """Should have empty participants by default."""
        tools = AgentTools("room-123", mock_rest_client)

        assert tools._participants == []

    def test_init_with_participants(self, mock_rest_client, participants):
        """Should accept participants."""
        tools = AgentTools("room-123", mock_rest_client, participants)

        assert tools._participants == participants


class TestAgentToolsFromContext:
    """Test AgentTools.from_context() factory method."""

    def test_from_context_creates_tools(self, mock_rest_client, participants):
        """from_context() should create AgentTools from ExecutionContext."""
        # Mock ExecutionContext
        mock_ctx = MagicMock()
        mock_ctx.room_id = "room-456"
        mock_ctx.link = MagicMock()
        mock_ctx.link.rest = mock_rest_client
        mock_ctx.participants = participants

        tools = AgentTools.from_context(mock_ctx)

        assert tools.room_id == "room-456"
        assert tools.rest is mock_rest_client
        assert tools._participants == participants


class TestAgentToolsSendMessage:
    """Test send_message tool."""

    async def test_send_message_success(self, mock_rest_client, participants):
        """send_message() should send via REST."""
        tools = AgentTools("room-123", mock_rest_client, participants)

        result = await tools.send_message("Hello!", mentions=["User One"])

        assert result["id"] == "msg-123"
        mock_rest_client.agent_api.create_agent_chat_message.assert_called_once()

    async def test_send_message_resolves_mentions(self, mock_rest_client, participants):
        """send_message() should resolve mention names to IDs."""
        tools = AgentTools("room-123", mock_rest_client, participants)

        await tools.send_message("Hello @User One!", mentions=["User One"])

        call_args = mock_rest_client.agent_api.create_agent_chat_message.call_args
        message = call_args.kwargs["message"]
        assert len(message.mentions) == 1
        assert message.mentions[0].id == "user-1"
        assert message.mentions[0].name == "User One"

    async def test_send_message_unknown_mention_raises(
        self, mock_rest_client, participants
    ):
        """send_message() should raise for unknown mention."""
        tools = AgentTools("room-123", mock_rest_client, participants)

        with pytest.raises(ValueError, match="Unknown participant 'Unknown'"):
            await tools.send_message("Hello!", mentions=["Unknown"])

    async def test_send_message_no_response_raises(
        self, mock_rest_client, participants
    ):
        """send_message() should raise if no response data."""
        mock_rest_client.agent_api.create_agent_chat_message.return_value = MagicMock(
            data=None
        )
        tools = AgentTools("room-123", mock_rest_client, participants)

        with pytest.raises(RuntimeError, match="Failed to send message"):
            await tools.send_message("Hello!", mentions=["User One"])


class TestAgentToolsSendEvent:
    """Test send_event tool."""

    async def test_send_event_success(self, mock_rest_client):
        """send_event() should send via REST."""
        tools = AgentTools("room-123", mock_rest_client)

        result = await tools.send_event("Thinking...", "thought")

        assert result["message_type"] == "thought"
        mock_rest_client.agent_api.create_agent_chat_event.assert_called_once()

    async def test_send_event_with_metadata(self, mock_rest_client):
        """send_event() should pass metadata."""
        tools = AgentTools("room-123", mock_rest_client)

        await tools.send_event("Error!", "error", metadata={"code": 500})

        call_args = mock_rest_client.agent_api.create_agent_chat_event.call_args
        event = call_args.kwargs["event"]
        assert event.metadata == {"code": 500}

    async def test_send_event_no_response_raises(self, mock_rest_client):
        """send_event() should raise if no response data."""
        mock_rest_client.agent_api.create_agent_chat_event.return_value = MagicMock(
            data=None
        )
        tools = AgentTools("room-123", mock_rest_client)

        with pytest.raises(RuntimeError, match="Failed to send event"):
            await tools.send_event("Error!", "error")


class TestAgentToolsAddParticipant:
    """Test add_participant tool."""

    async def test_add_participant_success(self, mock_rest_client):
        """add_participant() should lookup and add via REST."""
        tools = AgentTools("room-123", mock_rest_client)

        result = await tools.add_participant("Agent Two", role="member")

        assert result["id"] == "agent-2"
        assert result["name"] == "Agent Two"
        assert result["role"] == "member"
        assert result["status"] == "added"
        mock_rest_client.agent_api.add_agent_chat_participant.assert_called_once()

    async def test_add_participant_not_found_raises(self, mock_rest_client):
        """add_participant() should raise if peer not found."""
        # Return empty peers
        mock_rest_client.agent_api.list_agent_peers.return_value = MagicMock(
            data=[], metadata=MagicMock(total_pages=1)
        )
        tools = AgentTools("room-123", mock_rest_client)

        with pytest.raises(ValueError, match="Participant 'Unknown' not found"):
            await tools.add_participant("Unknown")


class TestAgentToolsRemoveParticipant:
    """Test remove_participant tool."""

    async def test_remove_participant_success(self, mock_rest_client):
        """remove_participant() should lookup and remove via REST."""
        tools = AgentTools("room-123", mock_rest_client)

        result = await tools.remove_participant("User One")

        assert result["id"] == "user-1"
        assert result["name"] == "User One"
        assert result["status"] == "removed"
        mock_rest_client.agent_api.remove_agent_chat_participant.assert_called_once()

    async def test_remove_participant_not_found_raises(self, mock_rest_client):
        """remove_participant() should raise if not in room."""
        # Return empty participants
        mock_rest_client.agent_api.list_agent_chat_participants.return_value = (
            MagicMock(data=[])
        )
        tools = AgentTools("room-123", mock_rest_client)

        with pytest.raises(ValueError, match="not found in this room"):
            await tools.remove_participant("Unknown")


class TestAgentToolsLookupPeers:
    """Test lookup_peers tool."""

    async def test_lookup_peers_success(self, mock_rest_client):
        """lookup_peers() should return formatted results."""
        tools = AgentTools("room-123", mock_rest_client)

        result = await tools.lookup_peers(page=1, page_size=50)

        assert len(result["peers"]) == 1
        assert result["peers"][0]["name"] == "Agent Two"
        assert result["metadata"]["page"] == 1

    async def test_lookup_peers_filters_by_room(self, mock_rest_client):
        """lookup_peers() should filter by not_in_chat."""
        tools = AgentTools("room-123", mock_rest_client)

        await tools.lookup_peers()

        call_args = mock_rest_client.agent_api.list_agent_peers.call_args
        assert call_args.kwargs["not_in_chat"] == "room-123"


class TestAgentToolsGetParticipants:
    """Test get_participants tool."""

    async def test_get_participants_success(self, mock_rest_client):
        """get_participants() should return formatted participants."""
        tools = AgentTools("room-123", mock_rest_client)

        result = await tools.get_participants()

        assert len(result) == 1
        assert result[0]["name"] == "User One"

    async def test_get_participants_empty(self, mock_rest_client):
        """get_participants() should return empty list if none."""
        mock_rest_client.agent_api.list_agent_chat_participants.return_value = (
            MagicMock(data=None)
        )
        tools = AgentTools("room-123", mock_rest_client)

        result = await tools.get_participants()

        assert result == []


class TestAgentToolsCreateChatroom:
    """Test create_chatroom tool."""

    async def test_create_chatroom_success(self, mock_rest_client):
        """create_chatroom() should call REST API and return room ID."""
        mock_response = Mock()
        mock_response.data.id = "room-123"
        mock_rest_client.agent_api.create_agent_chat = AsyncMock(
            return_value=mock_response
        )

        tools = AgentTools("room-456", mock_rest_client)
        result = await tools.create_chatroom(task_id="task-789")

        assert result == "room-123"
        mock_rest_client.agent_api.create_agent_chat.assert_called_once()

    async def test_create_chatroom_without_task_id(self, mock_rest_client):
        """create_chatroom() should work without task_id."""
        mock_response = Mock()
        mock_response.data.id = "room-abc"
        mock_rest_client.agent_api.create_agent_chat = AsyncMock(
            return_value=mock_response
        )

        tools = AgentTools("room-456", mock_rest_client)
        result = await tools.create_chatroom()

        assert result == "room-abc"


class TestAgentToolsSchemas:
    """Test tool schema generation."""

    def test_tool_models_registry(self):
        """TOOL_MODELS should contain all tool input models."""
        assert "send_message" in TOOL_MODELS
        assert "send_event" in TOOL_MODELS
        assert "add_participant" in TOOL_MODELS
        assert "remove_participant" in TOOL_MODELS
        assert "lookup_peers" in TOOL_MODELS
        assert "get_participants" in TOOL_MODELS
        assert "create_chatroom" in TOOL_MODELS

    def test_tool_models_property(self, mock_rest_client):
        """tool_models property should return registry."""
        tools = AgentTools("room-123", mock_rest_client)

        assert tools.tool_models is TOOL_MODELS

    def test_get_tool_schemas_openai(self, mock_rest_client):
        """get_tool_schemas('openai') should return OpenAI format."""
        tools = AgentTools("room-123", mock_rest_client)

        schemas = tools.get_tool_schemas("openai")

        assert len(schemas) == 7
        send_msg = next(s for s in schemas if s["function"]["name"] == "send_message")
        assert send_msg["type"] == "function"
        assert "parameters" in send_msg["function"]
        assert "description" in send_msg["function"]

    def test_get_tool_schemas_anthropic(self, mock_rest_client):
        """get_tool_schemas('anthropic') should return Anthropic format."""
        tools = AgentTools("room-123", mock_rest_client)

        schemas = tools.get_tool_schemas("anthropic")

        assert len(schemas) == 7
        send_msg = next(s for s in schemas if s["name"] == "send_message")
        assert "input_schema" in send_msg
        assert "description" in send_msg


class TestAgentToolsExecuteToolCall:
    """Test execute_tool_call dispatch."""

    async def test_execute_send_message(self, mock_rest_client, participants):
        """execute_tool_call() should dispatch send_message."""
        tools = AgentTools("room-123", mock_rest_client, participants)

        result = await tools.execute_tool_call(
            "send_message", {"content": "Hello!", "mentions": ["User One"]}
        )

        assert result["id"] == "msg-123"

    async def test_execute_send_event(self, mock_rest_client):
        """execute_tool_call() should dispatch send_event."""
        tools = AgentTools("room-123", mock_rest_client)

        result = await tools.execute_tool_call(
            "send_event", {"content": "Thinking...", "message_type": "thought"}
        )

        assert result["message_type"] == "thought"

    async def test_execute_lookup_peers(self, mock_rest_client):
        """execute_tool_call() should dispatch lookup_peers."""
        tools = AgentTools("room-123", mock_rest_client)

        result = await tools.execute_tool_call("lookup_peers", {"page": 1})

        assert "peers" in result
        assert "metadata" in result

    async def test_execute_get_participants(self, mock_rest_client):
        """execute_tool_call() should dispatch get_participants."""
        tools = AgentTools("room-123", mock_rest_client)

        result = await tools.execute_tool_call("get_participants", {})

        assert isinstance(result, list)

    async def test_execute_unknown_tool(self, mock_rest_client):
        """execute_tool_call() should return error for unknown tool."""
        tools = AgentTools("room-123", mock_rest_client)

        result = await tools.execute_tool_call("unknown_tool", {})

        assert "Unknown tool" in result

    async def test_execute_validation_error(self, mock_rest_client):
        """execute_tool_call() should return validation error."""
        tools = AgentTools("room-123", mock_rest_client)

        # Missing required field
        result = await tools.execute_tool_call("send_message", {"content": "Hello"})

        assert "Error validating" in result

    async def test_execute_runtime_error(self, mock_rest_client, participants):
        """execute_tool_call() should return execution error."""
        mock_rest_client.agent_api.create_agent_chat_message.side_effect = Exception(
            "Network error"
        )
        tools = AgentTools("room-123", mock_rest_client, participants)

        result = await tools.execute_tool_call(
            "send_message", {"content": "Hello!", "mentions": ["User One"]}
        )

        assert "Error executing" in result


class TestMentionResolution:
    """Test mention resolution logic."""

    def test_resolve_string_mentions(self, mock_rest_client, participants):
        """Should resolve string mentions to dicts."""
        tools = AgentTools("room-123", mock_rest_client, participants)

        resolved = tools._resolve_mentions(["User One", "User Two"])

        assert len(resolved) == 2
        assert resolved[0] == {"id": "user-1", "name": "User One"}
        assert resolved[1] == {"id": "user-2", "name": "User Two"}

    def test_resolve_dict_mentions_with_id(self, mock_rest_client, participants):
        """Should pass through dict mentions with ID."""
        tools = AgentTools("room-123", mock_rest_client, participants)

        resolved = tools._resolve_mentions([{"id": "custom-id", "name": "Custom"}])

        assert resolved[0] == {"id": "custom-id", "name": "Custom"}

    def test_resolve_dict_mentions_without_id(self, mock_rest_client, participants):
        """Should resolve dict mentions without ID."""
        tools = AgentTools("room-123", mock_rest_client, participants)

        resolved = tools._resolve_mentions([{"name": "User One"}])

        assert resolved[0] == {"id": "user-1", "name": "User One"}

    def test_resolve_unknown_raises(self, mock_rest_client, participants):
        """Should raise for unknown mention."""
        tools = AgentTools("room-123", mock_rest_client, participants)

        with pytest.raises(ValueError, match="Unknown participant"):
            tools._resolve_mentions(["Unknown Person"])


class TestToolInputModels:
    """Test Pydantic tool input models."""

    def test_send_message_input_validation(self):
        """SendMessageInput should validate fields."""
        model = SendMessageInput(content="Hello", mentions=["User"])
        assert model.content == "Hello"
        assert model.mentions == ["User"]

    def test_send_message_input_requires_mention(self):
        """SendMessageInput should require at least one mention."""
        with pytest.raises(Exception):  # Pydantic validation
            SendMessageInput(content="Hello", mentions=[])

    def test_send_event_input_validation(self):
        """SendEventInput should validate fields."""
        model = SendEventInput(content="Thinking", message_type="thought")
        assert model.message_type == "thought"

    def test_send_event_input_validates_type(self):
        """SendEventInput should validate message_type."""
        with pytest.raises(Exception):
            SendEventInput(content="Test", message_type="invalid")

    def test_add_participant_input_defaults(self):
        """AddParticipantInput should have default role."""
        model = AddParticipantInput(name="User")
        assert model.role == "member"

    def test_lookup_peers_input_defaults(self):
        """LookupPeersInput should have defaults."""
        model = LookupPeersInput()
        assert model.page == 1
        assert model.page_size == 50

    def test_get_participants_input_no_fields(self):
        """GetParticipantsInput should have no required fields."""
        model = GetParticipantsInput()
        assert model is not None

    def test_create_chatroom_input_validation(self):
        """CreateChatroomInput should allow optional task_id."""
        model = CreateChatroomInput(task_id="task-123")
        assert model.task_id == "task-123"

    def test_create_chatroom_input_no_task_id(self):
        """CreateChatroomInput should work without task_id."""
        model = CreateChatroomInput()
        assert model.task_id is None
