"""Integration tests using dynamically created agents.

These tests use the User API to:
1. Register a new external agent (getting API key)
2. Run tests using that agent
3. Clean up by deleting the agent

IMPORTANT SAFETY NOTE:
    Only agents CREATED BY THESE TESTS are deleted. The pre-existing agents
    defined in .env.test (THENVOI_API_KEY, THENVOI_API_KEY_2) are NEVER deleted.
    The cleanup only deletes the agent created via register_my_agent() during
    the test run.

Run with: uv run pytest tests/integration/test_dynamic_agent.py -v -s

Prerequisites:
- THENVOI_API_KEY_USER must be set in .env.test
- thenvoi-rest SDK must have delete_my_agent() method
"""

import uuid
from dataclasses import dataclass

import pytest
from thenvoi_rest import AsyncRestClient, ChatRoomRequest

from tests.integration.conftest import (
    get_base_url,
    get_user_api_key,
    requires_user_api,
)


@dataclass
class DynamicAgent:
    """Container for dynamically created agent credentials."""

    agent_id: str
    agent_name: str
    api_key: str


# Module-level storage for the created agent
_dynamic_agent: DynamicAgent | None = None


@pytest.fixture(scope="module")
def event_loop():
    """Create an event loop for the module scope.

    Required for module-scoped async fixtures.
    """
    import asyncio

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()
    asyncio.set_event_loop(None)


@pytest.fixture(scope="module")
def module_user_api_client():
    """Create user API client for the module scope."""
    api_key = get_user_api_key()
    if not api_key:
        return None
    return AsyncRestClient(api_key=api_key, base_url=get_base_url())


@pytest.fixture(scope="module")
async def dynamic_agent(module_user_api_client):
    """Create an agent for this test module and delete it after.

    Uses module scope so the agent is created once and reused across all tests.
    The API key is stored and can be used to create agent API clients.
    """
    global _dynamic_agent

    if module_user_api_client is None:
        pytest.skip("THENVOI_API_KEY_USER not set")

    # Check if delete_my_agent exists (SDK may not be updated yet)
    if not hasattr(module_user_api_client.human_api, "delete_my_agent"):
        pytest.skip(
            "thenvoi-rest SDK does not have delete_my_agent() method yet. "
            "Please update the SDK."
        )

    # Import AgentRegisterRequest for registering agents
    try:
        from thenvoi_rest.types import AgentRegisterRequest
    except ImportError:
        pytest.skip("Cannot import AgentRegisterRequest from thenvoi_rest")

    # Create agent via User API with unique name to avoid conflicts
    unique_suffix = uuid.uuid4().hex[:8]
    agent_name = f"SDK Dynamic Test Agent {unique_suffix}"

    response = await module_user_api_client.human_api.register_my_agent(
        agent=AgentRegisterRequest(
            name=agent_name,
            description="Created by SDK integration tests - will be deleted",
        )
    )

    agent = response.data.agent
    credentials = response.data.credentials

    _dynamic_agent = DynamicAgent(
        agent_id=agent.id,
        agent_name=agent.name,
        api_key=credentials.api_key,
    )

    print(f"\nCreated dynamic agent: {agent.name} (ID: {agent.id})")

    yield _dynamic_agent

    # Cleanup: Delete ONLY the agent we created in this test
    # SAFETY: We only delete agent.id which was returned by register_my_agent()
    # above. We NEVER touch the pre-existing agents from .env.test.
    print(f"\nDeleting dynamic agent: {agent.id} (created by this test)")
    try:
        await module_user_api_client.human_api.delete_my_agent(
            id=agent.id,
            force=True,  # Delete any executions too
        )
        print("Agent deleted successfully")
    except Exception as e:
        print(f"Warning: Failed to delete agent: {e}")
    _dynamic_agent = None


@pytest.fixture
def dynamic_agent_client(dynamic_agent: DynamicAgent) -> AsyncRestClient:
    """Create API client using the dynamically created agent's API key."""
    return AsyncRestClient(
        api_key=dynamic_agent.api_key,
        base_url=get_base_url(),
    )


@requires_user_api
class TestDynamicAgentWorkflow:
    """Test complete workflow using a dynamically created agent."""

    async def test_agent_can_fetch_identity(self, dynamic_agent_client, dynamic_agent):
        """Verify the dynamic agent can authenticate and fetch its identity."""
        response = await dynamic_agent_client.agent_api.get_agent_me()

        assert response.data is not None
        assert response.data.id == dynamic_agent.agent_id
        assert response.data.name == dynamic_agent.agent_name
        print(f"Agent verified: {response.data.name}")

    async def test_agent_can_list_peers(self, dynamic_agent_client):
        """Verify the dynamic agent can list available peers."""
        response = await dynamic_agent_client.agent_api.list_agent_peers()

        assert response.data is not None
        print(f"Agent can see {len(response.data)} peers")

    async def test_agent_can_create_chat(self, dynamic_agent_client):
        """Verify the dynamic agent can create a chat room."""
        from thenvoi_rest import ChatMessageRequest
        from thenvoi_rest.types import ChatMessageRequestMentionsItem as Mention
        from thenvoi_rest.types import ParticipantRequest

        response = await dynamic_agent_client.agent_api.create_agent_chat(
            chat=ChatRoomRequest()
        )

        assert response.data is not None
        assert response.data.id is not None
        chat_id = response.data.id
        print(f"Created chat: {chat_id}")

        # Get a peer to add to the room
        peers_response = await dynamic_agent_client.agent_api.list_agent_peers()
        if peers_response.data:
            peer = peers_response.data[0]
            await dynamic_agent_client.agent_api.add_agent_chat_participant(
                chat_id,
                participant=ParticipantRequest(participant_id=peer.id, role="member"),
            )
            print(f"Added peer: {peer.name}")

            # Add descriptive message (triggers auto-title)
            await dynamic_agent_client.agent_api.create_agent_chat_message(
                chat_id,
                message=ChatMessageRequest(
                    content=f"Dynamic agent test: @{peer.name} verifying agent can create a chat room",
                    mentions=[Mention(id=peer.id, name=peer.name)],
                ),
            )

    async def test_full_messaging_workflow(self, dynamic_agent_client, dynamic_agent):
        """Test complete messaging flow with dynamic agent."""
        from thenvoi_rest import ChatEventRequest, ChatMessageRequest
        from thenvoi_rest.types import ChatMessageRequestMentionsItem as Mention
        from thenvoi_rest.types import ParticipantRequest

        # 1. Create chat
        chat_response = await dynamic_agent_client.agent_api.create_agent_chat(
            chat=ChatRoomRequest()
        )
        chat_id = chat_response.data.id
        print(f"Created chat: {chat_id}")

        # 2. Get peers to find someone to mention
        peers_response = await dynamic_agent_client.agent_api.list_agent_peers()
        if not peers_response.data:
            pytest.skip("No peers available for messaging test")

        peer = peers_response.data[0]
        print(f"Using peer: {peer.name} (ID: {peer.id})")

        # 3. Add peer to chat
        await dynamic_agent_client.agent_api.add_agent_chat_participant(
            chat_id,
            participant=ParticipantRequest(participant_id=peer.id, role="member"),
        )
        print("Added peer to chat")

        # 4. Send message
        msg_response = await dynamic_agent_client.agent_api.create_agent_chat_message(
            chat_id,
            message=ChatMessageRequest(
                content=f"Hello @{peer.name} from dynamic agent!",
                mentions=[Mention(id=peer.id, name=peer.name)],
            ),
        )
        assert msg_response.data.id is not None
        print(f"Sent message: {msg_response.data.id}")

        # 5. Send thought event
        event_response = await dynamic_agent_client.agent_api.create_agent_chat_event(
            chat_id,
            event=ChatEventRequest(
                content="Dynamic agent is thinking...",
                message_type="thought",
            ),
        )
        assert event_response.data.id is not None
        print(f"Sent thought event: {event_response.data.id}")

        print(f"Complete workflow passed for dynamic agent {dynamic_agent.agent_id}")

    async def test_agent_to_agent_communication(
        self, dynamic_agent_client, dynamic_agent
    ):
        """Test agent-to-agent messaging flow.

        This test verifies that a dynamically created agent can:
        1. Create a chat room
        2. Find another Agent peer (not a User)
        3. Add that agent to the chat
        4. Send a message mentioning the other agent

        This is a critical test for validating agent-to-agent collaboration.
        """
        from thenvoi_rest import ChatEventRequest, ChatMessageRequest
        from thenvoi_rest.types import ChatMessageRequestMentionsItem as Mention
        from thenvoi_rest.types import ParticipantRequest

        # 1. Get peers and find an Agent peer (not a User)
        peers_response = await dynamic_agent_client.agent_api.list_agent_peers()
        if not peers_response.data:
            pytest.skip("No peers available")

        # Filter for Agent type peers
        agent_peers = [p for p in peers_response.data if p.type == "Agent"]
        if not agent_peers:
            pytest.skip("No Agent peers available for agent-to-agent test")

        target_agent = agent_peers[0]
        print(
            f"Target agent for communication: {target_agent.name} (ID: {target_agent.id})"
        )

        # 2. Create chat room for agent-to-agent communication
        chat_response = await dynamic_agent_client.agent_api.create_agent_chat(
            chat=ChatRoomRequest()
        )
        chat_id = chat_response.data.id
        print(f"Created chat room: {chat_id}")

        # 3. Add the target agent to the chat
        await dynamic_agent_client.agent_api.add_agent_chat_participant(
            chat_id,
            participant=ParticipantRequest(
                participant_id=target_agent.id, role="member"
            ),
        )
        print(f"Added agent {target_agent.name} to chat")

        # 4. Verify participants include both agents
        participants_response = (
            await dynamic_agent_client.agent_api.list_agent_chat_participants(chat_id)
        )
        participant_ids = [p.id for p in participants_response.data]
        assert dynamic_agent.agent_id in participant_ids, "Dynamic agent not in chat"
        assert target_agent.id in participant_ids, "Target agent not in chat"
        print(
            f"Verified both agents are participants: {len(participants_response.data)} total"
        )

        # 5. Send message from dynamic agent to target agent
        msg_content = (
            f"Hello @{target_agent.name}! This is agent-to-agent communication test."
        )
        msg_response = await dynamic_agent_client.agent_api.create_agent_chat_message(
            chat_id,
            message=ChatMessageRequest(
                content=msg_content,
                mentions=[Mention(id=target_agent.id, name=target_agent.name)],
            ),
        )
        assert msg_response.data.id is not None
        print(f"Sent message: {msg_response.data.id}")

        # 6. Send a thought event (simulating agent processing)
        event_response = await dynamic_agent_client.agent_api.create_agent_chat_event(
            chat_id,
            event=ChatEventRequest(
                content="Processing agent-to-agent communication...",
                message_type="thought",
            ),
        )
        assert event_response.data.id is not None
        print(f"Sent thought event: {event_response.data.id}")

        print(
            f"Agent-to-agent communication test passed: "
            f"{dynamic_agent.agent_name} -> {target_agent.name}"
        )


@requires_user_api
class TestUserAgentManagement:
    """Test User API agent management operations.

    These tests verify User API functionality using the user_api_client fixture.
    Note: These use function-scoped fixtures to avoid event loop issues.
    """

    async def test_user_can_list_owned_agents(self, user_api_client):
        """User should be able to list their owned agents."""
        if user_api_client is None:
            pytest.skip("THENVOI_API_KEY_USER not set")

        response = await user_api_client.human_api.list_my_agents()

        assert response.data is not None
        print(f"User owns {len(response.data)} agents")

    async def test_user_can_list_peers(self, user_api_client):
        """User should be able to list available peers."""
        if user_api_client is None:
            pytest.skip("THENVOI_API_KEY_USER not set")

        response = await user_api_client.human_api.list_my_peers()

        assert response.data is not None
        print(f"User can see {len(response.data)} peers")
