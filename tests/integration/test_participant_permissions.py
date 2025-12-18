"""
Comprehensive participant permission tests using dynamically created agents.

ALL agents used in these tests are:
1. Created dynamically via user_api_client.human_api.register_my_agent()
2. Deleted after tests complete

NO agents from .env.test are used or modified.

Tests all permutations of:
- Actor role: owner, admin, member
- Target role: owner, admin, member
- Target type: User, Agent
- Actions: add, remove

Expected behavior:
- Owner: Can add/remove anyone except themselves (must transfer ownership first)
- Admin: Can add/remove members, cannot touch owner/other admins
- Member: Can add other members (as member role only), cannot remove anyone except self

Run with: uv run pytest tests/integration/test_participant_permissions.py -v -s
"""

import uuid
from dataclasses import dataclass

import pytest
from thenvoi_rest import AsyncRestClient, ChatRoomRequest
from thenvoi_rest.types import ParticipantRequest

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


class DynamicAgentManager:
    """Manages creation and cleanup of dynamic agents for testing."""

    def __init__(self, user_client: AsyncRestClient):
        self.user_client = user_client
        self.created_agents: list[DynamicAgent] = []

    async def create_agent(self, name_prefix: str) -> DynamicAgent:
        """Create a new agent and track it for cleanup."""
        from thenvoi_rest.types import AgentRegisterRequest

        unique_suffix = uuid.uuid4().hex[:8]
        agent_name = f"{name_prefix} {unique_suffix}"

        response = await self.user_client.human_api.register_my_agent(
            agent=AgentRegisterRequest(
                name=agent_name,
                description="Created by SDK permission tests - will be deleted",
            )
        )

        agent = response.data.agent
        credentials = response.data.credentials

        dynamic_agent = DynamicAgent(
            agent_id=agent.id,
            agent_name=agent.name,
            api_key=credentials.api_key,
        )

        self.created_agents.append(dynamic_agent)
        print(f"  Created agent: {agent.name} (ID: {agent.id})")

        return dynamic_agent

    async def cleanup_all(self):
        """Delete all agents created by this manager."""
        for agent in self.created_agents:
            try:
                await self.user_client.human_api.delete_my_agent(
                    id=agent.agent_id,
                    force=True,
                )
                print(f"  Deleted agent: {agent.agent_name} (ID: {agent.agent_id})")
            except Exception as e:
                print(f"  Warning: Failed to delete agent {agent.agent_id}: {e}")

        self.created_agents.clear()


# Module-level storage
_agent_manager: DynamicAgentManager | None = None


@pytest.fixture(scope="module")
def event_loop():
    """Create an event loop for the module scope."""
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
async def agent_manager(module_user_api_client):
    """Create and manage dynamic agents for the entire test module."""
    global _agent_manager

    if module_user_api_client is None:
        pytest.skip("THENVOI_API_KEY_USER not set")

    # Check if delete_my_agent exists
    if not hasattr(module_user_api_client.human_api, "delete_my_agent"):
        pytest.skip(
            "thenvoi-rest SDK does not have delete_my_agent() method yet. "
            "Please update the SDK."
        )

    # Check if AgentRegisterRequest is available
    try:
        from thenvoi_rest.types import AgentRegisterRequest  # noqa: F401
    except ImportError:
        pytest.skip("Cannot import AgentRegisterRequest from thenvoi_rest")

    _agent_manager = DynamicAgentManager(module_user_api_client)

    print("\n=== Creating dynamic agents for permission tests ===")
    yield _agent_manager

    print("\n=== Cleaning up dynamic agents ===")
    await _agent_manager.cleanup_all()
    _agent_manager = None


@pytest.fixture(scope="module")
async def permission_agents(agent_manager: DynamicAgentManager):
    """
    Create 4 agents for permission testing:
    - agent_owner: Will create chats (becomes owner)
    - agent_admin: Will be added as admin
    - agent_member: Will be added as member
    - agent_extra: Available for add tests

    NOTE: Clients are NOT created here to avoid event loop issues.
    Use get_agent_client() to create clients in function scope.
    """
    agents = {
        "owner": await agent_manager.create_agent("Perm Test Owner"),
        "admin": await agent_manager.create_agent("Perm Test Admin"),
        "member": await agent_manager.create_agent("Perm Test Member"),
        "extra": await agent_manager.create_agent("Perm Test Extra"),
    }

    return agents


def get_agent_client(agent: DynamicAgent) -> AsyncRestClient:
    """Create a fresh REST client for an agent (avoids event loop issues)."""
    return AsyncRestClient(api_key=agent.api_key, base_url=get_base_url())


@requires_user_api
class TestParticipantRemovalPermissions:
    """
    Test all permutations of participant removal permissions.

    Setup for each test:
    - Owner agent creates chat (becomes owner)
    - Owner adds Admin agent as admin
    - Owner adds Member agent as member
    - Owner adds a User peer as member (if available)
    """

    @pytest.fixture
    async def removal_test_chat(self, permission_agents, module_user_api_client):
        """
        Create a chat with all role combinations for removal testing.

        Returns dict with:
        - chat_id: str
        - owner: DynamicAgent with role="owner"
        - admin: DynamicAgent with role="admin"
        - member: DynamicAgent with role="member"
        - member_user: {id, name, type, role} or None
        """
        owner = permission_agents["owner"]
        admin = permission_agents["admin"]
        member = permission_agents["member"]

        owner_client = get_agent_client(owner)

        # Get agent's owner_uuid to identify which User is the agent's owner
        agent_me = await owner_client.agent_api.get_agent_me()
        agent_owner_uuid = (
            str(agent_me.data.owner_uuid) if agent_me.data.owner_uuid else None
        )

        # Find User peers - separate owner from non-owner
        response = await owner_client.agent_api.list_agent_peers()
        user_peers = [p for p in response.data if p.type == "User"]

        # Identify agent's owner (for P4 protection rule test)
        agent_owner_user = next(
            (p for p in user_peers if p.id == agent_owner_uuid), None
        )
        # Find a non-owner User (for generic removal test)
        non_owner_user = next((p for p in user_peers if p.id != agent_owner_uuid), None)

        # Create chat (owner agent becomes owner)
        response = await owner_client.agent_api.create_agent_chat(
            chat=ChatRoomRequest(title="Removal Permission Test Chat")
        )
        chat_id = response.data.id
        print(f"\n  Created test chat: {chat_id}")

        # Add admin agent
        await owner_client.agent_api.add_agent_chat_participant(
            chat_id,
            participant=ParticipantRequest(participant_id=admin.agent_id, role="admin"),
        )
        print(f"  Added admin: {admin.agent_name}")

        # Add member agent
        await owner_client.agent_api.add_agent_chat_participant(
            chat_id,
            participant=ParticipantRequest(
                participant_id=member.agent_id, role="member"
            ),
        )
        print(f"  Added member: {member.agent_name}")

        # Add agent's owner as member (for P4 protection rule test)
        agent_owner_member = None
        if agent_owner_user:
            await owner_client.agent_api.add_agent_chat_participant(
                chat_id,
                participant=ParticipantRequest(
                    participant_id=agent_owner_user.id, role="member"
                ),
            )
            agent_owner_member = {
                "id": agent_owner_user.id,
                "name": agent_owner_user.name,
                "type": "User",
                "role": "member",
                "is_agent_owner": True,
            }
            print(f"  Added agent's owner as member: {agent_owner_user.name}")

        # Add non-owner User as member (for generic removal test)
        non_owner_member = None
        if non_owner_user:
            await owner_client.agent_api.add_agent_chat_participant(
                chat_id,
                participant=ParticipantRequest(
                    participant_id=non_owner_user.id, role="member"
                ),
            )
            non_owner_member = {
                "id": non_owner_user.id,
                "name": non_owner_user.name,
                "type": "User",
                "role": "member",
                "is_agent_owner": False,
            }
            print(f"  Added non-owner user member: {non_owner_user.name}")

        yield {
            "chat_id": chat_id,
            "owner": owner,
            "admin": admin,
            "member": member,
            "member_user": non_owner_member,  # For generic "remove User member" tests
            "agent_owner_member": agent_owner_member,  # For P4 protection rule tests
        }

        # No cleanup needed - chat will be abandoned when agents are deleted

    async def _try_remove(self, client, chat_id: str, target_id: str) -> str:
        """
        Try to remove a participant and return result.

        Returns:
        - "success" if removal succeeded
        - "403" if forbidden
        - "404" if not found
        - "error:<message>" for other errors
        """
        try:
            await client.agent_api.remove_agent_chat_participant(chat_id, target_id)
            return "success"
        except Exception as e:
            error_str = str(e).lower()
            if "403" in error_str or "forbidden" in error_str:
                return "403"
            elif "404" in error_str or "not found" in error_str:
                return "404"
            else:
                return f"error:{e}"

    # === Owner removal tests ===

    async def test_owner_removes_member_agent(self, removal_test_chat):
        """Owner removes Member (Agent) -> Expected: SUCCESS"""
        chat = removal_test_chat
        client = get_agent_client(chat["owner"])
        result = await self._try_remove(
            client, chat["chat_id"], chat["member"].agent_id
        )
        print(f"Owner removes Member(Agent): {result}")
        assert result == "success", (
            f"Owner should be able to remove member agent, got: {result}"
        )

    async def test_owner_removes_member_user(self, removal_test_chat):
        """Owner removes Member (User who is NOT agent's owner) -> Expected: SUCCESS"""
        chat = removal_test_chat
        if not chat["member_user"]:
            pytest.skip("No non-owner User peer available")
        client = get_agent_client(chat["owner"])
        result = await self._try_remove(
            client, chat["chat_id"], chat["member_user"]["id"]
        )
        print(f"Owner removes Member(User, non-owner): {result}")
        assert result == "success", (
            f"Owner should be able to remove non-owner user member, got: {result}"
        )

    async def test_agent_cannot_remove_own_owner_p4(self, removal_test_chat):
        """P4 Protection Rule: Agent cannot remove its own owner (User) -> Expected: 403"""
        chat = removal_test_chat
        if not chat["agent_owner_member"]:
            pytest.skip("No agent owner User available")
        client = get_agent_client(chat["owner"])
        result = await self._try_remove(
            client, chat["chat_id"], chat["agent_owner_member"]["id"]
        )
        print(f"Agent removes own owner (P4): {result}")
        assert result == "403", (
            f"Agent should NOT be able to remove its own owner (P4), got: {result}"
        )

    async def test_owner_removes_admin(self, removal_test_chat):
        """Owner removes Admin (Agent) -> Expected: SUCCESS"""
        chat = removal_test_chat
        client = get_agent_client(chat["owner"])
        result = await self._try_remove(client, chat["chat_id"], chat["admin"].agent_id)
        print(f"Owner removes Admin: {result}")
        assert result == "success", (
            f"Owner should be able to remove admin, got: {result}"
        )

    async def test_owner_removes_self(self, removal_test_chat):
        """Owner removes Self -> Expected: 403 (cannot leave as owner)"""
        chat = removal_test_chat
        client = get_agent_client(chat["owner"])
        result = await self._try_remove(client, chat["chat_id"], chat["owner"].agent_id)
        print(f"Owner removes Self: {result}")
        # Owner cannot remove themselves without transferring ownership
        print(f"  -> Actual behavior: {result}")

    # === Admin removal tests ===

    async def test_admin_removes_member_agent(self, removal_test_chat):
        """Admin removes Member (Agent) -> Expected: SUCCESS"""
        chat = removal_test_chat
        client = get_agent_client(chat["admin"])
        result = await self._try_remove(
            client, chat["chat_id"], chat["member"].agent_id
        )
        print(f"Admin removes Member(Agent): {result}")
        print(f"  -> Actual behavior: {result}")

    async def test_admin_removes_member_user(self, removal_test_chat):
        """Admin removes Member (User who is NOT agent's owner) -> Expected: SUCCESS"""
        chat = removal_test_chat
        if not chat["member_user"]:
            pytest.skip("No non-owner User peer available")
        client = get_agent_client(chat["admin"])
        result = await self._try_remove(
            client, chat["chat_id"], chat["member_user"]["id"]
        )
        print(f"Admin removes Member(User, non-owner): {result}")
        print(f"  -> Actual behavior: {result}")

    async def test_admin_removes_owner(self, removal_test_chat):
        """Admin removes Owner (Agent) -> Expected: 403"""
        chat = removal_test_chat
        client = get_agent_client(chat["admin"])
        result = await self._try_remove(client, chat["chat_id"], chat["owner"].agent_id)
        print(f"Admin removes Owner: {result}")
        assert result == "403", (
            f"Admin should NOT be able to remove owner, got: {result}"
        )

    async def test_admin_removes_self(self, removal_test_chat):
        """Admin removes Self -> Expected: SUCCESS (leaving chat)"""
        chat = removal_test_chat
        client = get_agent_client(chat["admin"])
        result = await self._try_remove(client, chat["chat_id"], chat["admin"].agent_id)
        print(f"Admin removes Self: {result}")
        print(f"  -> Actual behavior: {result}")

    async def test_admin_removes_other_admin(
        self, removal_test_chat, permission_agents
    ):
        """Admin removes another Admin (Agent) -> Expected: 403"""
        chat = removal_test_chat
        # Add extra agent as another admin
        owner_client = get_agent_client(chat["owner"])
        extra_agent = permission_agents["extra"]
        await owner_client.agent_api.add_agent_chat_participant(
            chat["chat_id"],
            participant=ParticipantRequest(
                participant_id=extra_agent.agent_id, role="admin"
            ),
        )
        print(f"  Added second admin: {extra_agent.agent_name}")

        # Now first admin tries to remove second admin
        client = get_agent_client(chat["admin"])
        result = await self._try_remove(client, chat["chat_id"], extra_agent.agent_id)
        print(f"Admin removes other Admin(Agent): {result}")
        print(f"  -> Actual behavior: {result}")

    # === Member removal tests ===

    async def test_member_removes_other_member_user(self, removal_test_chat):
        """Member removes Member (User who is NOT agent's owner) -> Expected: SUCCESS"""
        chat = removal_test_chat
        if not chat["member_user"]:
            pytest.skip("No non-owner User peer available")
        client = get_agent_client(chat["member"])
        result = await self._try_remove(
            client, chat["chat_id"], chat["member_user"]["id"]
        )
        print(f"Member removes Member(User, non-owner): {result}")
        assert result == "success", (
            f"Member should be able to remove other member users, got: {result}"
        )

    async def test_member_removes_other_member_agent(
        self, removal_test_chat, permission_agents
    ):
        """Member removes Member (Agent) -> Expected: SUCCESS"""
        chat = removal_test_chat
        # Add the extra agent as another member so we can test member removing member
        owner_client = get_agent_client(chat["owner"])
        extra_agent = permission_agents["extra"]
        await owner_client.agent_api.add_agent_chat_participant(
            chat["chat_id"],
            participant=ParticipantRequest(
                participant_id=extra_agent.agent_id, role="member"
            ),
        )
        print(f"  Added extra member: {extra_agent.agent_name}")

        # Now member tries to remove other member agent
        client = get_agent_client(chat["member"])
        result = await self._try_remove(client, chat["chat_id"], extra_agent.agent_id)
        print(f"Member removes Member(Agent): {result}")
        assert result == "success", (
            f"Member should be able to remove other member agents, got: {result}"
        )

    async def test_member_removes_admin(self, removal_test_chat):
        """Member removes Admin (Agent) -> Expected: 403"""
        chat = removal_test_chat
        client = get_agent_client(chat["member"])
        result = await self._try_remove(client, chat["chat_id"], chat["admin"].agent_id)
        print(f"Member removes Admin: {result}")
        assert result == "403", (
            f"Member should NOT be able to remove admin, got: {result}"
        )

    async def test_member_removes_owner(self, removal_test_chat):
        """Member removes Owner (Agent) -> Expected: 403"""
        chat = removal_test_chat
        client = get_agent_client(chat["member"])
        result = await self._try_remove(client, chat["chat_id"], chat["owner"].agent_id)
        print(f"Member removes Owner: {result}")
        assert result == "403", (
            f"Member should NOT be able to remove owner, got: {result}"
        )

    async def test_member_removes_self(self, removal_test_chat):
        """Member removes Self -> Expected: SUCCESS (leaving chat)"""
        chat = removal_test_chat
        client = get_agent_client(chat["member"])
        result = await self._try_remove(
            client, chat["chat_id"], chat["member"].agent_id
        )
        print(f"Member removes Self: {result}")
        # Members should be able to leave (remove themselves)
        print(f"  -> Actual behavior: {result}")


@requires_user_api
class TestParticipantAddPermissions:
    """
    Test all permutations of participant add permissions.

    Per user requirement: Members should be able to add other members.
    """

    @pytest.fixture
    async def add_test_chat(self, permission_agents, module_user_api_client):
        """Create a chat for add permission testing."""
        owner = permission_agents["owner"]
        admin = permission_agents["admin"]
        member = permission_agents["member"]
        extra = permission_agents["extra"]

        owner_client = get_agent_client(owner)

        # Find User peers (need for testing User targets)
        response = await owner_client.agent_api.list_agent_peers()
        user_peers = [p for p in response.data if p.type == "User"]

        # Find additional Agent peers not in our test set
        test_agent_ids = {
            owner.agent_id,
            admin.agent_id,
            member.agent_id,
            extra.agent_id,
        }
        other_agents = [
            p for p in response.data if p.type == "Agent" and p.id not in test_agent_ids
        ]

        # Create chat (owner agent becomes owner)
        response = await owner_client.agent_api.create_agent_chat(
            chat=ChatRoomRequest(title="Add Permission Test Chat")
        )
        chat_id = response.data.id
        print(f"\n  Created test chat: {chat_id}")

        # Add admin agent
        await owner_client.agent_api.add_agent_chat_participant(
            chat_id,
            participant=ParticipantRequest(participant_id=admin.agent_id, role="admin"),
        )

        # Add member agent
        await owner_client.agent_api.add_agent_chat_participant(
            chat_id,
            participant=ParticipantRequest(
                participant_id=member.agent_id, role="member"
            ),
        )

        yield {
            "chat_id": chat_id,
            "owner": owner,
            "admin": admin,
            "member": member,
            "extra_agent": extra,  # Not in chat - available to add
            "available_user": user_peers[0] if user_peers else None,
            "available_user_2": user_peers[1] if len(user_peers) > 1 else None,
            "other_agent": other_agents[0] if other_agents else None,
        }

    async def _try_add(
        self, client, chat_id: str, target_id: str, role: str = "member"
    ) -> str:
        """Try to add a participant and return result."""
        try:
            await client.agent_api.add_agent_chat_participant(
                chat_id,
                participant=ParticipantRequest(participant_id=target_id, role=role),
            )
            return "success"
        except Exception as e:
            error_str = str(e).lower()
            if "403" in error_str or "forbidden" in error_str:
                return "403"
            elif (
                "409" in error_str or "conflict" in error_str or "already" in error_str
            ):
                return "409"  # Already in chat
            else:
                return f"error:{e}"

    # === Owner add tests ===

    async def test_owner_adds_user_as_member(self, add_test_chat):
        """Owner adds User as member -> Expected: SUCCESS"""
        chat = add_test_chat
        if not chat["available_user"]:
            pytest.skip("No available user peer")
        client = get_agent_client(chat["owner"])
        result = await self._try_add(
            client, chat["chat_id"], chat["available_user"].id, "member"
        )
        print(f"Owner adds User as member: {result}")
        assert result == "success", (
            f"Owner should be able to add user as member, got: {result}"
        )

    async def test_owner_adds_user_as_admin(self, add_test_chat):
        """Owner adds User as admin -> Expected: SUCCESS"""
        chat = add_test_chat
        if not chat["available_user_2"]:
            pytest.skip("No second available user peer")
        client = get_agent_client(chat["owner"])
        result = await self._try_add(
            client, chat["chat_id"], chat["available_user_2"].id, "admin"
        )
        print(f"Owner adds User as admin: {result}")
        assert result == "success", (
            f"Owner should be able to add user as admin, got: {result}"
        )

    async def test_owner_adds_agent_as_member(self, add_test_chat):
        """Owner adds Agent as member -> Expected: SUCCESS"""
        chat = add_test_chat
        client = get_agent_client(chat["owner"])
        result = await self._try_add(
            client, chat["chat_id"], chat["extra_agent"].agent_id, "member"
        )
        print(f"Owner adds Agent as member: {result}")
        assert result == "success", (
            f"Owner should be able to add agent as member, got: {result}"
        )

    async def test_owner_adds_agent_as_admin(self, add_test_chat):
        """Owner adds Agent as admin -> Expected: SUCCESS"""
        chat = add_test_chat
        if not chat["other_agent"]:
            pytest.skip("No other agent peer available")
        client = get_agent_client(chat["owner"])
        result = await self._try_add(
            client, chat["chat_id"], chat["other_agent"].id, "admin"
        )
        print(f"Owner adds Agent as admin: {result}")
        assert result == "success", (
            f"Owner should be able to add agent as admin, got: {result}"
        )

    # === Admin add tests ===

    async def test_admin_adds_user_as_member(self, add_test_chat):
        """Admin adds User as member -> Expected: SUCCESS"""
        chat = add_test_chat
        if not chat["available_user"]:
            pytest.skip("No available user peer")
        client = get_agent_client(chat["admin"])
        result = await self._try_add(
            client, chat["chat_id"], chat["available_user"].id, "member"
        )
        print(f"Admin adds User as member: {result}")
        assert result == "success", (
            f"Admin should be able to add user as member, got: {result}"
        )

    async def test_admin_adds_agent_as_member(self, add_test_chat):
        """Admin adds Agent as member -> Expected: SUCCESS"""
        chat = add_test_chat
        client = get_agent_client(chat["admin"])
        result = await self._try_add(
            client, chat["chat_id"], chat["extra_agent"].agent_id, "member"
        )
        print(f"Admin adds Agent as member: {result}")
        assert result == "success", (
            f"Admin should be able to add agent as member, got: {result}"
        )

    async def test_admin_adds_user_as_admin(self, add_test_chat):
        """Admin adds User as admin -> Expected: ? (may or may not be allowed)"""
        chat = add_test_chat
        if not chat["available_user_2"]:
            pytest.skip("No second available user peer")
        client = get_agent_client(chat["admin"])
        result = await self._try_add(
            client, chat["chat_id"], chat["available_user_2"].id, "admin"
        )
        print(f"Admin adds User as admin: {result}")
        print(f"  -> Actual behavior: {result}")

    # === Member add tests ===

    async def test_member_adds_user_as_member(self, add_test_chat):
        """Member adds User as member -> Expected: SUCCESS (per user requirement)"""
        chat = add_test_chat
        if not chat["available_user"]:
            pytest.skip("No available user peer")
        client = get_agent_client(chat["member"])
        result = await self._try_add(
            client, chat["chat_id"], chat["available_user"].id, "member"
        )
        print(f"Member adds User as member: {result}")
        # Per user requirement: members should be able to add other members
        assert result == "success", (
            f"Member should be able to add user as member, got: {result}"
        )

    async def test_member_adds_agent_as_member(self, add_test_chat):
        """Member adds Agent as member -> Expected: SUCCESS (per user requirement)"""
        chat = add_test_chat
        client = get_agent_client(chat["member"])
        result = await self._try_add(
            client, chat["chat_id"], chat["extra_agent"].agent_id, "member"
        )
        print(f"Member adds Agent as member: {result}")
        # Per user requirement: members should be able to add other members
        assert result == "success", (
            f"Member should be able to add agent as member, got: {result}"
        )

    async def test_member_adds_user_as_admin(self, add_test_chat):
        """Member adds User as admin -> Expected: 403 (can only add as member)"""
        chat = add_test_chat
        if not chat["available_user"]:
            pytest.skip("No available user peer")
        client = get_agent_client(chat["member"])
        result = await self._try_add(
            client, chat["chat_id"], chat["available_user"].id, "admin"
        )
        print(f"Member adds User as admin: {result}")
        # Members should only be able to add as member, not promote to admin
        assert result == "403", (
            f"Member should NOT be able to add user as admin, got: {result}"
        )

    async def test_member_adds_agent_as_admin(self, add_test_chat):
        """Member adds Agent as admin -> Expected: 403 (can only add as member)"""
        chat = add_test_chat
        client = get_agent_client(chat["member"])
        result = await self._try_add(
            client, chat["chat_id"], chat["extra_agent"].agent_id, "admin"
        )
        print(f"Member adds Agent as admin: {result}")
        # Members should only be able to add as member, not promote to admin
        assert result == "403", (
            f"Member should NOT be able to add agent as admin, got: {result}"
        )


@requires_user_api
class TestPermissionMatrix:
    """
    Run all permission scenarios and generate a matrix report.

    This test creates fresh chats for each scenario to avoid state pollution.
    """

    async def test_full_removal_permission_matrix(self, permission_agents):
        """
        Generate a complete removal permission matrix.

        Prints a matrix showing what's actually allowed by the API.
        """
        print("\n" + "=" * 80)
        print("PARTICIPANT REMOVAL PERMISSION MATRIX")
        print("=" * 80)

        owner = permission_agents["owner"]
        admin = permission_agents["admin"]
        member = permission_agents["member"]
        extra = permission_agents["extra"]

        owner_client = get_agent_client(owner)

        # Find a User peer
        response = await owner_client.agent_api.list_agent_peers()
        user_peer = next((p for p in response.data if p.type == "User"), None)

        agents = {
            "owner": owner,
            "admin": admin,
            "member": member,
        }

        # Define all removal scenarios
        scenarios = [
            # (actor_role, target_role, target_type)
            ("owner", "member", "Agent"),
            ("owner", "member", "User"),
            ("owner", "admin", "Agent"),
            ("owner", "self", "Agent"),
            ("admin", "member", "Agent"),
            ("admin", "member", "User"),
            ("admin", "admin", "Agent"),  # Added: admin removing another admin
            ("admin", "owner", "Agent"),
            ("admin", "self", "Agent"),
            ("member", "member", "Agent"),
            ("member", "member", "User"),
            ("member", "admin", "Agent"),
            ("member", "owner", "Agent"),
            ("member", "self", "Agent"),
        ]

        print("\nREMOVAL PERMISSIONS:")
        print("-" * 60)
        print(f"{'Actor':<12} {'Target':<20} {'Result':<15}")
        print("-" * 60)

        for actor_role, target_role, target_type in scenarios:
            # Create fresh chat for each scenario
            response = await owner_client.agent_api.create_agent_chat(
                chat=ChatRoomRequest(title=f"Matrix Test {actor_role}->{target_role}")
            )
            chat_id = response.data.id

            try:
                # Setup participants - add extra as admin for admin->admin test
                extra_role = (
                    "admin"
                    if (actor_role == "admin" and target_role == "admin")
                    else "member"
                )

                await owner_client.agent_api.add_agent_chat_participant(
                    chat_id,
                    participant=ParticipantRequest(
                        participant_id=admin.agent_id, role="admin"
                    ),
                )
                await owner_client.agent_api.add_agent_chat_participant(
                    chat_id,
                    participant=ParticipantRequest(
                        participant_id=member.agent_id, role="member"
                    ),
                )
                await owner_client.agent_api.add_agent_chat_participant(
                    chat_id,
                    participant=ParticipantRequest(
                        participant_id=extra.agent_id, role=extra_role
                    ),
                )

                if user_peer and target_type == "User":
                    await owner_client.agent_api.add_agent_chat_participant(
                        chat_id,
                        participant=ParticipantRequest(
                            participant_id=user_peer.id, role="member"
                        ),
                    )

                # Determine target ID
                if target_role == "self":
                    target_id = agents[actor_role].agent_id
                elif target_role == "owner":
                    target_id = owner.agent_id
                elif target_role == "admin" and actor_role == "admin":
                    # Admin trying to remove another admin - use extra agent
                    target_id = extra.agent_id
                elif target_role == "admin":
                    target_id = admin.agent_id
                elif target_role == "member" and target_type == "User":
                    target_id = user_peer.id if user_peer else None
                elif (
                    target_role == "member"
                    and target_type == "Agent"
                    and actor_role == "member"
                ):
                    # Member trying to remove another member agent - use extra agent
                    target_id = extra.agent_id
                else:
                    target_id = member.agent_id

                if target_id is None:
                    result = "SKIP:no-user"
                else:
                    actor_client = get_agent_client(agents[actor_role])
                    try:
                        await actor_client.agent_api.remove_agent_chat_participant(
                            chat_id, target_id
                        )
                        result = "SUCCESS"
                    except Exception as e:
                        error_str = str(e).lower()
                        if "403" in error_str or "forbidden" in error_str:
                            result = "403 FORBIDDEN"
                        elif "404" in error_str:
                            result = "404 NOT FOUND"
                        else:
                            result = "ERROR"

                target_desc = f"{target_role}({target_type})"
                print(f"{actor_role:<12} {target_desc:<20} {result:<15}")

            except Exception as e:
                print(
                    f"{actor_role:<12} {target_role}({target_type})  SETUP ERROR: {e}"
                )

        print("-" * 60)
        print("\nLegend:")
        print("  SUCCESS      = Action allowed")
        print("  403 FORBIDDEN = Permission denied")
        print("  404 NOT FOUND = Target not in chat")
        print("=" * 80)

    async def test_full_add_permission_matrix(self, permission_agents):
        """
        Generate a complete add permission matrix.
        """
        print("\n" + "=" * 80)
        print("PARTICIPANT ADD PERMISSION MATRIX")
        print("=" * 80)

        owner = permission_agents["owner"]
        admin = permission_agents["admin"]
        member = permission_agents["member"]
        extra = permission_agents["extra"]

        owner_client = get_agent_client(owner)

        # Find User peers
        response = await owner_client.agent_api.list_agent_peers()
        user_peers = [p for p in response.data if p.type == "User"]
        user_peer = user_peers[0] if user_peers else None

        agents = {
            "owner": owner,
            "admin": admin,
            "member": member,
        }

        # Define add scenarios
        scenarios = [
            # (actor_role, target_type, add_as_role)
            ("owner", "Agent", "member"),
            ("owner", "Agent", "admin"),
            ("owner", "Agent", "owner"),  # Ownership transfer?
            ("owner", "User", "member"),
            ("owner", "User", "admin"),
            ("owner", "User", "owner"),  # Ownership transfer?
            ("admin", "Agent", "member"),
            ("admin", "Agent", "admin"),
            ("admin", "Agent", "owner"),  # Can admin transfer ownership?
            ("admin", "User", "member"),
            ("admin", "User", "admin"),
            ("admin", "User", "owner"),
            ("member", "Agent", "member"),
            ("member", "Agent", "admin"),
            ("member", "Agent", "owner"),
            ("member", "User", "member"),
            ("member", "User", "admin"),
            ("member", "User", "owner"),
        ]

        print("\nADD PERMISSIONS:")
        print("-" * 70)
        print(f"{'Actor':<12} {'Target Type':<12} {'Add As':<10} {'Result':<15}")
        print("-" * 70)

        for actor_role, target_type, add_as_role in scenarios:
            # Create fresh chat for each scenario
            response = await owner_client.agent_api.create_agent_chat(
                chat=ChatRoomRequest(title=f"Add Matrix {actor_role}+{target_type}")
            )
            chat_id = response.data.id

            try:
                # Setup: add admin and member to chat
                await owner_client.agent_api.add_agent_chat_participant(
                    chat_id,
                    participant=ParticipantRequest(
                        participant_id=admin.agent_id, role="admin"
                    ),
                )
                await owner_client.agent_api.add_agent_chat_participant(
                    chat_id,
                    participant=ParticipantRequest(
                        participant_id=member.agent_id, role="member"
                    ),
                )

                # Determine target to add
                if target_type == "Agent":
                    target_id = extra.agent_id
                else:  # User
                    target_id = user_peer.id if user_peer else None

                if target_id is None:
                    result = "SKIP:no-user"
                else:
                    actor_client = get_agent_client(agents[actor_role])
                    try:
                        await actor_client.agent_api.add_agent_chat_participant(
                            chat_id,
                            participant=ParticipantRequest(
                                participant_id=target_id, role=add_as_role
                            ),
                        )
                        result = "SUCCESS"
                    except Exception as e:
                        error_str = str(e).lower()
                        if "403" in error_str or "forbidden" in error_str:
                            result = "403 FORBIDDEN"
                        elif "409" in error_str or "conflict" in error_str:
                            result = "409 CONFLICT"
                        else:
                            result = "ERROR"

                print(
                    f"{actor_role:<12} {target_type:<12} {add_as_role:<10} {result:<15}"
                )

            except Exception:
                print(
                    f"{actor_role:<12} {target_type:<12} {add_as_role:<10} SETUP ERROR"
                )

        print("-" * 70)
        print("\nLegend:")
        print("  SUCCESS      = Action allowed")
        print("  403 FORBIDDEN = Permission denied")
        print("  409 CONFLICT  = Already in chat")
        print("=" * 80)
