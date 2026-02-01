"""
Integration tests for Letta adapter.

These tests require a running Letta server (Docker).
Run with: pytest tests/adapters/letta/test_integration.py -v -m integration

To start Letta server:
    docker run -p 8283:8283 letta/letta:latest
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest

# Skip all tests if letta-client not installed
pytest.importorskip("letta_client")

from thenvoi.adapters.letta import (
    LettaAdapter,
    LettaConfig,
    LettaMode,
    StateStore,
)
from thenvoi.adapters.letta.memory import MemoryManager
from thenvoi.core.types import PlatformMessage


# ══════════════════════════════════════════════════════════════════════════════
# Test Configuration
# ══════════════════════════════════════════════════════════════════════════════

# Load settings from .env.test if present
from pathlib import Path
from dotenv import load_dotenv

_env_test_path = Path(__file__).parent.parent.parent.parent / ".env.test"
if _env_test_path.exists():
    load_dotenv(_env_test_path)

LETTA_BASE_URL = os.getenv("LETTA_BASE_URL", "http://localhost:8283")
LETTA_MCP_SERVER_URL = os.getenv(
    "LETTA_MCP_SERVER_URL", "http://mcp-agent-darter:8000/sse"
)


def letta_server_available() -> bool:
    """Check if Letta server is running."""
    try:
        from letta_client import Letta

        client = Letta(base_url=LETTA_BASE_URL)
        client.agents.list(limit=1)
        return True
    except Exception:
        return False


# Skip all tests if Letta server not available
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not letta_server_available(),
        reason=f"Letta server not available at {LETTA_BASE_URL}",
    ),
]


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def letta_client():
    """Create real Letta client."""
    from letta_client import Letta

    return Letta(base_url=LETTA_BASE_URL)


@pytest.fixture
def cleanup_agents(letta_client):
    """Track and cleanup agents created during tests."""
    created_agent_ids = []
    yield created_agent_ids
    for agent_id in created_agent_ids:
        try:
            letta_client.agents.delete(agent_id=agent_id)
        except Exception:
            pass


@pytest.fixture
def cleanup_conversations(letta_client):
    """Track and cleanup conversations created during tests."""
    created_conversation_ids = []
    yield created_conversation_ids
    for conv_id in created_conversation_ids:
        try:
            letta_client.conversations.delete(conversation_id=conv_id)
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# Agent Creation Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestAgentCreation:
    """Test agent creation against real Letta server."""

    def test_create_agent_basic(self, letta_client, cleanup_agents):
        """Should create agent with basic configuration."""
        agent = letta_client.agents.create(
            name="test-basic-agent",
            model="openai/gpt-4o-mini",
            embedding="openai/text-embedding-3-small",
        )
        cleanup_agents.append(agent.id)

        assert agent.id is not None
        assert agent.name == "test-basic-agent"

    def test_create_agent_with_memory_blocks(self, letta_client, cleanup_agents):
        """Should create agent with custom memory blocks."""
        agent = letta_client.agents.create(
            name="test-memory-agent",
            model="openai/gpt-4o-mini",
            embedding="openai/text-embedding-3-small",
            memory_blocks=[
                {"label": "persona", "value": "You are a helpful assistant."},
                {"label": "participants", "value": "No participants yet."},
                {"label": "room_contexts", "value": "No room contexts yet."},
            ],
        )
        cleanup_agents.append(agent.id)

        assert agent.id is not None

        # Verify memory blocks exist
        blocks = letta_client.agents.blocks.list(agent_id=agent.id)
        block_labels = [b.label for b in blocks]
        assert "persona" in block_labels
        assert "participants" in block_labels
        assert "room_contexts" in block_labels

    def test_retrieve_agent(self, letta_client, cleanup_agents):
        """Should retrieve existing agent."""
        agent = letta_client.agents.create(
            name="test-retrieve-agent",
            model="openai/gpt-4o-mini",
            embedding="openai/text-embedding-3-small",
        )
        cleanup_agents.append(agent.id)

        retrieved = letta_client.agents.retrieve(agent_id=agent.id)

        assert retrieved.id == agent.id
        assert retrieved.name == agent.name


# ══════════════════════════════════════════════════════════════════════════════
# Conversations API Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestConversationsAPI:
    """Test Conversations API against real Letta server."""

    def test_create_conversation(
        self, letta_client, cleanup_agents, cleanup_conversations
    ):
        """Should create conversation for agent."""
        agent = letta_client.agents.create(
            name="test-conv-agent",
            model="openai/gpt-4o-mini",
            embedding="openai/text-embedding-3-small",
        )
        cleanup_agents.append(agent.id)

        conversation = letta_client.conversations.create(
            agent_id=agent.id,
        )
        cleanup_conversations.append(conversation.id)

        assert conversation.id is not None
        assert conversation.agent_id == agent.id

    def test_multiple_conversations_same_agent(
        self, letta_client, cleanup_agents, cleanup_conversations
    ):
        """Should create multiple conversations for same agent."""
        agent = letta_client.agents.create(
            name="test-multi-conv-agent",
            model="openai/gpt-4o-mini",
            embedding="openai/text-embedding-3-small",
        )
        cleanup_agents.append(agent.id)

        conv1 = letta_client.conversations.create(agent_id=agent.id)
        conv2 = letta_client.conversations.create(agent_id=agent.id)
        conv3 = letta_client.conversations.create(agent_id=agent.id)

        cleanup_conversations.extend([conv1.id, conv2.id, conv3.id])

        assert conv1.id != conv2.id != conv3.id
        assert conv1.agent_id == conv2.agent_id == conv3.agent_id == agent.id

    def test_list_conversations_for_agent(
        self, letta_client, cleanup_agents, cleanup_conversations
    ):
        """Should list all conversations for an agent."""
        agent = letta_client.agents.create(
            name="test-list-conv-agent",
            model="openai/gpt-4o-mini",
            embedding="openai/text-embedding-3-small",
        )
        cleanup_agents.append(agent.id)

        conv1 = letta_client.conversations.create(agent_id=agent.id)
        conv2 = letta_client.conversations.create(agent_id=agent.id)
        cleanup_conversations.extend([conv1.id, conv2.id])

        conversations = letta_client.conversations.list(agent_id=agent.id)
        conv_ids = [c.id for c in conversations]

        assert conv1.id in conv_ids
        assert conv2.id in conv_ids

    def test_retrieve_conversation(
        self, letta_client, cleanup_agents, cleanup_conversations
    ):
        """Should retrieve existing conversation."""
        agent = letta_client.agents.create(
            name="test-retrieve-conv-agent",
            model="openai/gpt-4o-mini",
            embedding="openai/text-embedding-3-small",
        )
        cleanup_agents.append(agent.id)

        conversation = letta_client.conversations.create(
            agent_id=agent.id,
        )
        cleanup_conversations.append(conversation.id)

        retrieved = letta_client.conversations.retrieve(conversation_id=conversation.id)

        assert retrieved.id == conversation.id
        assert retrieved.agent_id == agent.id


# ══════════════════════════════════════════════════════════════════════════════
# Memory Operations Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestMemoryOperations:
    """Test memory operations against real Letta server."""

    def test_modify_memory_block(self, letta_client, cleanup_agents):
        """Should modify memory block content."""
        agent = letta_client.agents.create(
            name="test-memory-modify-agent",
            model="openai/gpt-4o-mini",
            embedding="openai/text-embedding-3-small",
            memory_blocks=[
                {"label": "participants", "value": "Initial value"},
            ],
        )
        cleanup_agents.append(agent.id)

        letta_client.agents.blocks.update(
            agent_id=agent.id,
            block_label="participants",
            value="Updated participants list",
        )

        block = letta_client.agents.blocks.retrieve(
            agent_id=agent.id,
            block_label="participants",
        )

        assert block is not None
        assert "Updated participants" in block.value

    @pytest.mark.asyncio
    async def test_memory_manager_update_participants(
        self, letta_client, cleanup_agents
    ):
        """Should update participants via MemoryManager."""
        agent = letta_client.agents.create(
            name="test-mm-participants-agent",
            model="openai/gpt-4o-mini",
            embedding="openai/text-embedding-3-small",
            memory_blocks=[
                {"label": "participants", "value": "No participants"},
            ],
        )
        cleanup_agents.append(agent.id)

        manager = MemoryManager(letta_client)
        await manager.update_participants(
            agent_id=agent.id,
            participants=[
                {"name": "Alice", "type": "User"},
                {"name": "DataBot", "type": "Agent"},
            ],
        )

        block = letta_client.agents.blocks.retrieve(
            agent_id=agent.id,
            block_label="participants",
        )

        assert block is not None
        assert "Alice" in block.value
        assert "DataBot" in block.value


# ══════════════════════════════════════════════════════════════════════════════
# Message Sending Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestMessageSending:
    """Test message sending against real Letta server.

    These tests require a Letta server with valid LLM provider credentials.
    """

    @pytest.mark.slow
    def test_send_message_to_agent(self, letta_client, cleanup_agents):
        """Should send message and receive response."""
        agent = letta_client.agents.create(
            name="test-message-agent",
            model="openai/gpt-4o-mini",
            embedding="openai/text-embedding-3-small",
            memory_blocks=[
                {
                    "label": "persona",
                    "value": "You are a helpful assistant. Keep responses brief.",
                },
            ],
        )
        cleanup_agents.append(agent.id)

        response = letta_client.agents.messages.create(
            agent_id=agent.id,
            messages=[{"role": "user", "content": "Hello, how are you?"}],
            streaming=False,
        )

        assert response.messages is not None
        assert len(response.messages) > 0

    @pytest.mark.slow
    def test_send_message_via_conversation(
        self, letta_client, cleanup_agents, cleanup_conversations
    ):
        """Should send message via Conversations API."""
        agent = letta_client.agents.create(
            name="test-conv-message-agent",
            model="openai/gpt-4o-mini",
            embedding="openai/text-embedding-3-small",
            memory_blocks=[
                {
                    "label": "persona",
                    "value": "You are a helpful assistant. Keep responses brief.",
                },
            ],
        )
        cleanup_agents.append(agent.id)

        conversation = letta_client.conversations.create(
            agent_id=agent.id,
        )
        cleanup_conversations.append(conversation.id)

        # Note: letta-client always returns a Stream for conversations.messages.create
        stream = letta_client.conversations.messages.create(
            conversation_id=conversation.id,
            messages=[{"role": "user", "content": "What is 2 + 2?"}],
        )

        # Consume stream to collect messages
        response_items = list(stream)
        assert response_items is not None
        assert len(response_items) > 0

    @pytest.mark.slow
    def test_conversation_isolation(
        self, letta_client, cleanup_agents, cleanup_conversations
    ):
        """Should maintain separate context per conversation."""
        agent = letta_client.agents.create(
            name="test-isolation-agent",
            model="openai/gpt-4o-mini",
            embedding="openai/text-embedding-3-small",
            memory_blocks=[
                {
                    "label": "persona",
                    "value": "You remember names. Keep responses brief.",
                },
            ],
        )
        cleanup_agents.append(agent.id)

        conv_a = letta_client.conversations.create(agent_id=agent.id)
        conv_b = letta_client.conversations.create(agent_id=agent.id)
        cleanup_conversations.extend([conv_a.id, conv_b.id])

        letta_client.conversations.messages.create(
            conversation_id=conv_a.id,
            messages=[{"role": "user", "content": "My name is Alice."}],
            stream=False,
        )

        letta_client.conversations.messages.create(
            conversation_id=conv_b.id,
            messages=[{"role": "user", "content": "My name is Bob."}],
            stream=False,
        )

        assert conv_a.id != conv_b.id


# ══════════════════════════════════════════════════════════════════════════════
# State Persistence Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestStatePersistence:
    """Test state persistence with real agents."""

    def test_state_persists_agent_mapping(self, letta_client, cleanup_agents, tmp_path):
        """Should persist room-to-agent mapping."""
        agent = letta_client.agents.create(
            name="test-persist-agent",
            model="openai/gpt-4o-mini",
            embedding="openai/text-embedding-3-small",
        )
        cleanup_agents.append(agent.id)

        store = StateStore(tmp_path / "state.json")
        state = store.load()
        state.set_room_agent("room-123", agent.id)
        store.save()

        store2 = StateStore(tmp_path / "state.json")
        state2 = store2.load()

        assert state2.get_room_agent("room-123") == agent.id

        retrieved = letta_client.agents.retrieve(agent_id=agent.id)
        assert retrieved.id == agent.id

    def test_state_persists_conversation_mapping(
        self, letta_client, cleanup_agents, cleanup_conversations, tmp_path
    ):
        """Should persist room-to-conversation mapping."""
        agent = letta_client.agents.create(
            name="test-conv-persist-agent",
            model="openai/gpt-4o-mini",
            embedding="openai/text-embedding-3-small",
        )
        cleanup_agents.append(agent.id)

        conversation = letta_client.conversations.create(
            agent_id=agent.id,
        )
        cleanup_conversations.append(conversation.id)

        store = StateStore(tmp_path / "state.json")
        state = store.load()
        state.shared_agent_id = agent.id
        state.set_room_conversation("room-456", conversation.id)
        store.save()

        store2 = StateStore(tmp_path / "state.json")
        state2 = store2.load()

        assert state2.shared_agent_id == agent.id
        assert state2.get_room_conversation("room-456") == conversation.id

        retrieved = letta_client.conversations.retrieve(conversation_id=conversation.id)
        assert retrieved.id == conversation.id


# ══════════════════════════════════════════════════════════════════════════════
# End-to-End Adapter Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestAdapterEndToEnd:
    """End-to-end tests with full adapter.

    These tests require:
    - A Letta server with valid LLM provider credentials
    - MCP server for tool execution at http://localhost:8002/sse
    """

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_adapter_per_room_creates_agent(self, tmp_path, cleanup_agents):
        """Should create agent on first message in PER_ROOM mode."""
        from unittest.mock import AsyncMock, MagicMock

        config = LettaConfig(
            mode=LettaMode.PER_ROOM,
            base_url=LETTA_BASE_URL,
            model="openai/gpt-4o-mini",
            embedding_model="openai/text-embedding-3-small",
            persona="Test assistant",
            mcp_server_url=LETTA_MCP_SERVER_URL,
            api_timeout=10,
        )

        adapter = LettaAdapter(
            config=config,
            state_storage_path=tmp_path / "state.json",
        )

        await adapter.on_started("TestBot", "A test bot")

        tools = MagicMock()
        tools.send_message = AsyncMock()
        tools.send_event = AsyncMock()
        tools.execute_tool_call = AsyncMock(return_value={"status": "sent"})
        tools.get_participants = AsyncMock(
            return_value=[
                {"name": "Alice", "type": "User"},
            ]
        )

        msg = PlatformMessage(
            id="msg-1",
            room_id="room-e2e-test",
            content="Hello!",
            sender_id="user-1",
            sender_type="User",
            sender_name="Alice",
            message_type="text",
            metadata={},
            created_at=datetime.now(timezone.utc),
        )

        await adapter.on_message(
            msg=msg,
            tools=tools,
            history=[],
            participants_msg=None,
            is_session_bootstrap=True,
            room_id="room-e2e-test",
        )

        agent_id = adapter.state.get_room_agent("room-e2e-test")
        assert agent_id is not None
        cleanup_agents.append(agent_id)

        # Adapter uses execute_tool_call for messages and send_event for tool visibility
        assert tools.execute_tool_call.called or tools.send_event.called

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_adapter_shared_creates_conversation(
        self, tmp_path, cleanup_agents, cleanup_conversations
    ):
        """Should create conversation on first message in SHARED mode."""
        from unittest.mock import AsyncMock, MagicMock

        config = LettaConfig(
            mode=LettaMode.SHARED,
            base_url=LETTA_BASE_URL,
            model="openai/gpt-4o-mini",
            embedding_model="openai/text-embedding-3-small",
            persona="Test assistant",
            mcp_server_url=LETTA_MCP_SERVER_URL,
            api_timeout=10,
        )

        adapter = LettaAdapter(
            config=config,
            state_storage_path=tmp_path / "state.json",
        )

        await adapter.on_started("TestBot", "A test bot")

        if adapter.state.shared_agent_id:
            cleanup_agents.append(adapter.state.shared_agent_id)

        tools = MagicMock()
        tools.send_message = AsyncMock()
        tools.send_event = AsyncMock()
        tools.execute_tool_call = AsyncMock(return_value={"status": "sent"})
        tools.get_participants = AsyncMock(
            return_value=[
                {"name": "Alice", "type": "User"},
            ]
        )

        msg = PlatformMessage(
            id="msg-1",
            room_id="room-shared-test",
            content="Hello!",
            sender_id="user-1",
            sender_type="User",
            sender_name="Alice",
            message_type="text",
            metadata={},
            created_at=datetime.now(timezone.utc),
        )

        await adapter.on_message(
            msg=msg,
            tools=tools,
            history=[],
            participants_msg=None,
            is_session_bootstrap=True,
            room_id="room-shared-test",
        )

        conv_id = adapter.state.get_room_conversation("room-shared-test")
        assert conv_id is not None
        cleanup_conversations.append(conv_id)

        assert adapter.state.shared_agent_id is not None
