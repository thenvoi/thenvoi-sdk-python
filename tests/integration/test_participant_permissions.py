"""
Comprehensive participant permission tests using dynamically created agents.

ALL agents used in these tests are:
1. Created dynamically via user_api_client.human_api_agents.register_my_agent()
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

import logging
import uuid
from dataclasses import dataclass

import pytest
from thenvoi_rest import AsyncRestClient, ChatMessageRequest, ChatRoomRequest
from thenvoi_rest.core.api_error import ApiError
from thenvoi_rest.types import (
    ChatMessageRequestMentionsItem as Mention,
    ParticipantRequest,
)

from tests.integration.conftest import (
    get_base_url,
    get_user_api_key,
    is_no_clean_mode,
    requires_user_api,
)

logger = logging.getLogger(__name__)


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

        try:
            response = await self.user_client.human_api_agents.register_my_agent(
                agent=AgentRegisterRequest(
                    name=agent_name,
                    description="Created by SDK permission tests - will be deleted",
                )
            )
        except ApiError as e:
            if e.status_code == 403:
                pytest.skip("Enterprise plan required for Human API agent registration")
            raise

        agent = response.data.agent
        credentials = response.data.credentials

        dynamic_agent = DynamicAgent(
            agent_id=agent.id,
            agent_name=agent.name,
            api_key=credentials.api_key,
        )

        self.created_agents.append(dynamic_agent)
        logger.info("  Created agent: %s (ID: %s)", agent.name, agent.id)

        return dynamic_agent

    async def cleanup_all(self):
        """Delete all agents created by this manager."""
        for agent in self.created_agents:
            try:
                await self.user_client.human_api_agents.delete_my_agent(
                    id=agent.agent_id,
                    force=True,
                )
                logger.info(
                    f"  Deleted agent: {agent.agent_name} (ID: {agent.agent_id})"
                )
            except Exception as e:
                logger.warning("Failed to delete agent %s: %s", agent.agent_id, e)

        self.created_agents.clear()


@pytest.fixture(scope="module")
def module_user_api_client():
    """Create user API client for the module scope."""
    api_key = get_user_api_key()
    if not api_key:
        return None
    return AsyncRestClient(api_key=api_key, base_url=get_base_url())


@pytest.fixture(scope="module")
async def agent_manager(module_user_api_client, request):
    """Create and manage dynamic agents for the entire test module."""
    if module_user_api_client is None:
        pytest.skip("THENVOI_API_KEY_USER not set")

    # Check if delete_my_agent exists
    if not hasattr(module_user_api_client.human_api_agents, "delete_my_agent"):
        pytest.skip(
            "thenvoi-rest SDK does not have delete_my_agent() method yet. "
            "Please update the SDK."
        )

    # Check if AgentRegisterRequest is available
    try:
        from thenvoi_rest.types import AgentRegisterRequest  # noqa: F401
    except ImportError:
        pytest.skip("Cannot import AgentRegisterRequest from thenvoi_rest")

    manager = DynamicAgentManager(module_user_api_client)

    logger.info("\n=== Creating dynamic agents for permission tests ===")
    yield manager

    # Cleanup: delete the agents (unless --no-clean mode)
    if not is_no_clean_mode(request):
        logger.info("\n=== Cleaning up dynamic agents ===")
        await manager.cleanup_all()
    else:
        agent_names = [a.agent_name for a in manager.created_agents]
        logger.info(
            "[NO-CLEAN MODE] Skipping cleanup of %d agents: %s",
            len(agent_names),
            ", ".join(agent_names),
        )


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
        agent_me = await owner_client.agent_api_identity.get_agent_me()
        agent_owner_uuid = (
            str(agent_me.data.owner_uuid) if agent_me.data.owner_uuid else None
        )

        # Find User peers - separate owner from non-owner
        response = await owner_client.agent_api_peers.list_agent_peers()
        user_peers = [p for p in response.data if p.type == "User"]

        # Identify agent's owner (for P4 protection rule test)
        agent_owner_user = next(
            (p for p in user_peers if p.id == agent_owner_uuid), None
        )
        # Find a non-owner User (for generic removal test)
        non_owner_user = next((p for p in user_peers if p.id != agent_owner_uuid), None)

        # Create chat (owner agent becomes owner)
        response = await owner_client.agent_api_chats.create_agent_chat(
            chat=ChatRoomRequest()
        )
        chat_id = response.data.id
        logger.info("\n  Created test chat: %s", chat_id)

        # Add admin agent
        await owner_client.agent_api_participants.add_agent_chat_participant(
            chat_id,
            participant=ParticipantRequest(participant_id=admin.agent_id, role="admin"),
        )
        logger.info("  Added admin: %s", admin.agent_name)

        # Add descriptive message (triggers auto-title)
        await owner_client.agent_api_messages.create_agent_chat_message(
            chat_id,
            message=ChatMessageRequest(
                content=f"Participant removal permission test: @{admin.agent_name} testing removal permissions for owner/admin/member roles",
                mentions=[Mention(id=admin.agent_id, name=admin.agent_name)],
            ),
        )

        # Add member agent
        await owner_client.agent_api_participants.add_agent_chat_participant(
            chat_id,
            participant=ParticipantRequest(
                participant_id=member.agent_id, role="member"
            ),
        )
        logger.info("  Added member: %s", member.agent_name)

        # Add agent's owner as member (for P4 protection rule test)
        agent_owner_member = None
        if agent_owner_user:
            await owner_client.agent_api_participants.add_agent_chat_participant(
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
            logger.info("  Added agent's owner as member: %s", agent_owner_user.name)

        # Add non-owner User as member (for generic removal test)
        non_owner_member = None
        if non_owner_user:
            await owner_client.agent_api_participants.add_agent_chat_participant(
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
            logger.info("  Added non-owner user member: %s", non_owner_user.name)

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
            await client.agent_api_participants.remove_agent_chat_participant(
                chat_id, target_id
            )
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
        logger.info("Owner removes Member(Agent): %s", result)
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
        logger.info("Owner removes Member(User, non-owner): %s", result)
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
        logger.info("Agent removes own owner (P4): %s", result)
        assert result == "403", (
            f"Agent should NOT be able to remove its own owner (P4), got: {result}"
        )

    async def test_owner_removes_admin(self, removal_test_chat):
        """Owner removes Admin (Agent) -> Expected: SUCCESS"""
        chat = removal_test_chat
        client = get_agent_client(chat["owner"])
        result = await self._try_remove(client, chat["chat_id"], chat["admin"].agent_id)
        logger.info("Owner removes Admin: %s", result)
        assert result == "success", (
            f"Owner should be able to remove admin, got: {result}"
        )

    async def test_owner_removes_self(self, removal_test_chat):
        """Owner removes Self -> Expected: 403 (cannot leave as owner)"""
        chat = removal_test_chat
        client = get_agent_client(chat["owner"])
        result = await self._try_remove(client, chat["chat_id"], chat["owner"].agent_id)
        logger.info("Owner removes Self: %s", result)
        # Owner cannot remove themselves without transferring ownership
        logger.info("  -> Actual behavior: %s", result)

    # === Admin removal tests ===

    async def test_admin_removes_member_agent(self, removal_test_chat):
        """Admin removes Member (Agent) -> Expected: SUCCESS"""
        chat = removal_test_chat
        client = get_agent_client(chat["admin"])
        result = await self._try_remove(
            client, chat["chat_id"], chat["member"].agent_id
        )
        logger.info("Admin removes Member(Agent): %s", result)
        logger.info("  -> Actual behavior: %s", result)

    async def test_admin_removes_member_user(self, removal_test_chat):
        """Admin removes Member (User who is NOT agent's owner) -> Expected: SUCCESS"""
        chat = removal_test_chat
        if not chat["member_user"]:
            pytest.skip("No non-owner User peer available")
        client = get_agent_client(chat["admin"])
        result = await self._try_remove(
            client, chat["chat_id"], chat["member_user"]["id"]
        )
        logger.info("Admin removes Member(User, non-owner): %s", result)
        logger.info("  -> Actual behavior: %s", result)

    async def test_admin_removes_owner(self, removal_test_chat):
        """Admin removes Owner (Agent) -> Expected: 403"""
        chat = removal_test_chat
        client = get_agent_client(chat["admin"])
        result = await self._try_remove(client, chat["chat_id"], chat["owner"].agent_id)
        logger.info("Admin removes Owner: %s", result)
        assert result == "403", (
            f"Admin should NOT be able to remove owner, got: {result}"
        )

    async def test_admin_removes_self(self, removal_test_chat):
        """Admin removes Self -> Expected: SUCCESS (leaving chat)"""
        chat = removal_test_chat
        client = get_agent_client(chat["admin"])
        result = await self._try_remove(client, chat["chat_id"], chat["admin"].agent_id)
        logger.info("Admin removes Self: %s", result)
        logger.info("  -> Actual behavior: %s", result)

    async def test_admin_removes_other_admin(
        self, removal_test_chat, permission_agents
    ):
        """Admin removes another Admin (Agent) -> Expected: 403"""
        chat = removal_test_chat
        # Add extra agent as another admin
        owner_client = get_agent_client(chat["owner"])
        extra_agent = permission_agents["extra"]
        await owner_client.agent_api_participants.add_agent_chat_participant(
            chat["chat_id"],
            participant=ParticipantRequest(
                participant_id=extra_agent.agent_id, role="admin"
            ),
        )
        logger.info("  Added second admin: %s", extra_agent.agent_name)

        # Now first admin tries to remove second admin
        client = get_agent_client(chat["admin"])
        result = await self._try_remove(client, chat["chat_id"], extra_agent.agent_id)
        logger.info("Admin removes other Admin(Agent): %s", result)
        logger.info("  -> Actual behavior: %s", result)

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
        logger.info("Member removes Member(User, non-owner): %s", result)
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
        await owner_client.agent_api_participants.add_agent_chat_participant(
            chat["chat_id"],
            participant=ParticipantRequest(
                participant_id=extra_agent.agent_id, role="member"
            ),
        )
        logger.info("  Added extra member: %s", extra_agent.agent_name)

        # Now member tries to remove other member agent
        client = get_agent_client(chat["member"])
        result = await self._try_remove(client, chat["chat_id"], extra_agent.agent_id)
        logger.info("Member removes Member(Agent): %s", result)
        assert result == "success", (
            f"Member should be able to remove other member agents, got: {result}"
        )

    async def test_member_removes_admin(self, removal_test_chat):
        """Member removes Admin (Agent) -> Expected: 403"""
        chat = removal_test_chat
        client = get_agent_client(chat["member"])
        result = await self._try_remove(client, chat["chat_id"], chat["admin"].agent_id)
        logger.info("Member removes Admin: %s", result)
        assert result == "403", (
            f"Member should NOT be able to remove admin, got: {result}"
        )

    async def test_member_removes_owner(self, removal_test_chat):
        """Member removes Owner (Agent) -> Expected: 403"""
        chat = removal_test_chat
        client = get_agent_client(chat["member"])
        result = await self._try_remove(client, chat["chat_id"], chat["owner"].agent_id)
        logger.info("Member removes Owner: %s", result)
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
        logger.info("Member removes Self: %s", result)
        # Members should be able to leave (remove themselves)
        logger.info("  -> Actual behavior: %s", result)


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
        response = await owner_client.agent_api_peers.list_agent_peers()
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
        response = await owner_client.agent_api_chats.create_agent_chat(
            chat=ChatRoomRequest()
        )
        chat_id = response.data.id
        logger.info("\n  Created test chat: %s", chat_id)

        # Add admin agent
        await owner_client.agent_api_participants.add_agent_chat_participant(
            chat_id,
            participant=ParticipantRequest(participant_id=admin.agent_id, role="admin"),
        )

        # Add descriptive message (triggers auto-title)
        await owner_client.agent_api_messages.create_agent_chat_message(
            chat_id,
            message=ChatMessageRequest(
                content=f"Participant add permission test: @{admin.agent_name} testing add permissions for owner/admin/member roles",
                mentions=[Mention(id=admin.agent_id, name=admin.agent_name)],
            ),
        )

        # Add member agent
        await owner_client.agent_api_participants.add_agent_chat_participant(
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
            await client.agent_api_participants.add_agent_chat_participant(
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
        logger.info("Owner adds User as member: %s", result)
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
        logger.info("Owner adds User as admin: %s", result)
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
        logger.info("Owner adds Agent as member: %s", result)
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
        logger.info("Owner adds Agent as admin: %s", result)
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
        logger.info("Admin adds User as member: %s", result)
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
        logger.info("Admin adds Agent as member: %s", result)
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
        logger.info("Admin adds User as admin: %s", result)
        logger.info("  -> Actual behavior: %s", result)

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
        logger.info("Member adds User as member: %s", result)
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
        logger.info("Member adds Agent as member: %s", result)
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
        logger.info("Member adds User as admin: %s", result)
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
        logger.info("Member adds Agent as admin: %s", result)
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
        logger.info("\n" + "=" * 80)
        logger.info("PARTICIPANT REMOVAL PERMISSION MATRIX")
        logger.info("=" * 80)

        owner = permission_agents["owner"]
        admin = permission_agents["admin"]
        member = permission_agents["member"]
        extra = permission_agents["extra"]

        owner_client = get_agent_client(owner)

        # Find a User peer
        response = await owner_client.agent_api_peers.list_agent_peers()
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

        logger.info("\nREMOVAL PERMISSIONS:")
        logger.info("-" * 60)
        logger.info("%-12s %-20s %-15s", "Actor", "Target", "Result")
        logger.info("-" * 60)

        for actor_role, target_role, target_type in scenarios:
            # Create fresh chat for each scenario
            response = await owner_client.agent_api_chats.create_agent_chat(
                chat=ChatRoomRequest()
            )
            chat_id = response.data.id

            try:
                # Setup participants - add extra as admin for admin->admin test
                extra_role = (
                    "admin"
                    if (actor_role == "admin" and target_role == "admin")
                    else "member"
                )

                await owner_client.agent_api_participants.add_agent_chat_participant(
                    chat_id,
                    participant=ParticipantRequest(
                        participant_id=admin.agent_id, role="admin"
                    ),
                )

                # Add descriptive message (triggers auto-title)
                await owner_client.agent_api_messages.create_agent_chat_message(
                    chat_id,
                    message=ChatMessageRequest(
                        content=f"Removal matrix: @{admin.agent_name} {actor_role} removing {target_role}({target_type})",
                        mentions=[Mention(id=admin.agent_id, name=admin.agent_name)],
                    ),
                )

                await owner_client.agent_api_participants.add_agent_chat_participant(
                    chat_id,
                    participant=ParticipantRequest(
                        participant_id=member.agent_id, role="member"
                    ),
                )
                await owner_client.agent_api_participants.add_agent_chat_participant(
                    chat_id,
                    participant=ParticipantRequest(
                        participant_id=extra.agent_id, role=extra_role
                    ),
                )

                if user_peer and target_type == "User":
                    await (
                        owner_client.agent_api_participants.add_agent_chat_participant(
                            chat_id,
                            participant=ParticipantRequest(
                                participant_id=user_peer.id, role="member"
                            ),
                        )
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
                        await actor_client.agent_api_participants.remove_agent_chat_participant(
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
                logger.info("%-12s %-20s %-15s", actor_role, target_desc, result)

            except Exception as e:
                logger.info(
                    f"{actor_role:<12} {target_role}({target_type})  SETUP ERROR: {e}"
                )

        logger.info("-" * 60)
        logger.info("\nLegend:")
        logger.info("  SUCCESS      = Action allowed")
        logger.info("  403 FORBIDDEN = Permission denied")
        logger.info("  404 NOT FOUND = Target not in chat")
        logger.info("=" * 80)

    async def test_full_add_permission_matrix(self, permission_agents):
        """
        Generate a complete add permission matrix.
        """
        logger.info("\n" + "=" * 80)
        logger.info("PARTICIPANT ADD PERMISSION MATRIX")
        logger.info("=" * 80)

        owner = permission_agents["owner"]
        admin = permission_agents["admin"]
        member = permission_agents["member"]
        extra = permission_agents["extra"]

        owner_client = get_agent_client(owner)

        # Find User peers
        response = await owner_client.agent_api_peers.list_agent_peers()
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

        logger.info("\nADD PERMISSIONS:")
        logger.info("-" * 70)
        logger.info(
            "%-12s %-12s %-10s %-15s", "Actor", "Target Type", "Add As", "Result"
        )
        logger.info("-" * 70)

        for actor_role, target_type, add_as_role in scenarios:
            # Create fresh chat for each scenario
            response = await owner_client.agent_api_chats.create_agent_chat(
                chat=ChatRoomRequest()
            )
            chat_id = response.data.id

            try:
                # Setup: add admin and member to chat
                await owner_client.agent_api_participants.add_agent_chat_participant(
                    chat_id,
                    participant=ParticipantRequest(
                        participant_id=admin.agent_id, role="admin"
                    ),
                )

                # Add descriptive message (triggers auto-title)
                await owner_client.agent_api_messages.create_agent_chat_message(
                    chat_id,
                    message=ChatMessageRequest(
                        content=f"Add matrix: @{admin.agent_name} {actor_role} adding {target_type} as {add_as_role}",
                        mentions=[Mention(id=admin.agent_id, name=admin.agent_name)],
                    ),
                )

                await owner_client.agent_api_participants.add_agent_chat_participant(
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
                        await actor_client.agent_api_participants.add_agent_chat_participant(
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

                logger.info(
                    f"{actor_role:<12} {target_type:<12} {add_as_role:<10} {result:<15}"
                )

            except Exception:
                logger.info(
                    f"{actor_role:<12} {target_type:<12} {add_as_role:<10} SETUP ERROR"
                )

        logger.info("-" * 70)
        logger.info("\nLegend:")
        logger.info("  SUCCESS      = Action allowed")
        logger.info("  403 FORBIDDEN = Permission denied")
        logger.info("  409 CONFLICT  = Already in chat")
        logger.info("=" * 80)
