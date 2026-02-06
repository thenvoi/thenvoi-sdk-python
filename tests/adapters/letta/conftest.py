"""
Shared fixtures for Letta adapter tests.

Provides common test fixtures including:
- Mock Letta client
- Mock Thenvoi tools
- Sample platform messages
- Temporary state storage
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

import pytest

from thenvoi.adapters.letta import LettaConfig, LettaMode
from thenvoi.core.types import PlatformMessage


# ══════════════════════════════════════════════════════════════════════════════
# Configuration Fixtures
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def per_room_config() -> LettaConfig:
    """Configuration for PER_ROOM mode testing."""
    return LettaConfig(
        api_key="test-api-key",
        mode=LettaMode.PER_ROOM,
        base_url="http://localhost:8283",
        model="openai/gpt-4o-mini",
        embedding_model="openai/text-embedding-3-small",
        persona="You are a helpful test assistant.",
    )


@pytest.fixture
def shared_config() -> LettaConfig:
    """Configuration for SHARED mode testing."""
    return LettaConfig(
        api_key="test-api-key",
        mode=LettaMode.SHARED,
        base_url="http://localhost:8283",
        model="openai/gpt-4o-mini",
        embedding_model="openai/text-embedding-3-small",
        persona="You are a personal assistant for testing.",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Mock Client Fixtures
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def mock_letta_client() -> MagicMock:
    """
    Create a fully mocked Letta client.

    Provides mocks for:
    - agents.create(), retrieve(), messages.create()
    - conversations.create(), retrieve(), messages.create()
    - agents.core_memory.modify(), retrieve()
    """
    client = MagicMock()

    # Mock agent response
    mock_agent = MagicMock()
    mock_agent.id = "agent-123"
    mock_agent.name = "TestAgent"
    client.agents.create.return_value = mock_agent
    client.agents.retrieve.return_value = mock_agent

    # Mock conversation response
    mock_conversation = MagicMock()
    mock_conversation.id = "conv-456"
    mock_conversation.name = "room-test"
    client.conversations.create.return_value = mock_conversation
    client.conversations.retrieve.return_value = mock_conversation

    # Mock message response
    mock_response = MagicMock()
    mock_message = MagicMock()
    mock_message.message_type = "assistant_message"
    mock_message.content = "Hello! How can I help you?"
    mock_response.messages = [mock_message]
    client.agents.messages.create.return_value = mock_response
    client.conversations.messages.create.return_value = mock_response

    # Mock core memory
    mock_memory = MagicMock()
    mock_memory.blocks = []
    client.agents.core_memory.retrieve.return_value = mock_memory
    client.agents.core_memory.modify.return_value = None

    return client


@pytest.fixture
def mock_tools() -> MagicMock:
    """
    Create mock Thenvoi AgentToolsProtocol.

    All methods are AsyncMock for proper async testing.
    """
    tools = MagicMock()
    tools.send_message = AsyncMock(return_value={"id": "msg-123", "status": "sent"})
    tools.send_event = AsyncMock(return_value={"id": "event-123"})
    tools.add_participant = AsyncMock(return_value={"success": True})
    tools.remove_participant = AsyncMock(return_value={"success": True})
    tools.get_participants = AsyncMock(
        return_value=[
            {"name": "Alice", "type": "User", "id": "user-1"},
            {"name": "TestBot", "type": "Agent", "id": "agent-1"},
        ]
    )
    tools.lookup_peers = AsyncMock(
        return_value={
            "items": [
                {"name": "DataBot", "type": "Agent", "description": "Data analysis"},
                {"name": "Bob", "type": "User", "description": "Team member"},
            ],
            "total": 2,
            "page": 1,
        }
    )
    tools.create_chatroom = AsyncMock(return_value="room-new-123")
    return tools


# ══════════════════════════════════════════════════════════════════════════════
# Message Fixtures
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def sample_message() -> PlatformMessage:
    """Create a sample platform message for testing."""
    return PlatformMessage(
        id="msg-test-123",
        room_id="room-test-456",
        content="Hello, can you help me with something?",
        sender_id="user-alice-789",
        sender_type="User",
        sender_name="Alice",
        message_type="text",
        metadata={},
        created_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def bootstrap_message() -> PlatformMessage:
    """Create a message that triggers session bootstrap."""
    return PlatformMessage(
        id="msg-bootstrap-001",
        room_id="room-new-session",
        content="Hi there!",
        sender_id="user-bob-123",
        sender_type="User",
        sender_name="Bob",
        message_type="text",
        metadata={"is_bootstrap": True},
        created_at=datetime.now(timezone.utc),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Storage Fixtures
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def state_storage_path(tmp_path: Path) -> Path:
    """Provide temporary path for state storage."""
    return tmp_path / "letta_state.json"


@pytest.fixture
def state_storage_dir(tmp_path: Path) -> Path:
    """Provide temporary directory for state storage."""
    state_dir = tmp_path / "letta_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir


# ══════════════════════════════════════════════════════════════════════════════
# Integration Test Markers
# ══════════════════════════════════════════════════════════════════════════════


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test (requires Letta server)",
    )
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow running",
    )
