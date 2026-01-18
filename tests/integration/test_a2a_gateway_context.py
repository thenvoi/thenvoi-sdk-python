"""E2E tests for A2A Gateway context_id flow.

Tests the complete flow:
1. Gateway receives HTTP request with contextId
2. Gateway creates/reuses chat room based on contextId
3. Same contextId -> same room, different contextId -> different room

Unit tests use mocks (fast, no network).
Integration tests use real Thenvoi platform (requires API keys in .env.test).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from thenvoi.integrations.a2a.gateway import A2AGatewayAdapter
from thenvoi_rest import AsyncRestClient, ChatMessageRequest, Peer
from thenvoi_rest.types import ChatMessageRequestMentionsItem as Mention

from .conftest import requires_api


class TestA2AGatewayContextIdFlow:
    """Unit tests for context_id persistence in A2A Gateway (mock-based)."""

    @pytest.fixture
    def gateway_adapter_with_mocks(self) -> A2AGatewayAdapter:
        """Create gateway adapter with mocked REST client for testing."""
        adapter = A2AGatewayAdapter(
            rest_url="http://localhost:4000",
            api_key="test-key",
            gateway_url="http://localhost:10000",
            port=10000,
        )

        # Mock peer
        weather_peer = Peer(id="uuid-weather", name="Weather Agent", type="agent")
        adapter._peers = {"weather-agent": weather_peer}
        adapter._peers_by_uuid = {"uuid-weather": weather_peer}

        # Track room creation
        rooms_created: list[str] = []

        def track_room_creation(*args, **kwargs):
            room_id = f"room-{len(rooms_created) + 1}"
            rooms_created.append(room_id)
            response = MagicMock()
            response.data = MagicMock(id=room_id)
            return response

        adapter._rest.agent_api.create_agent_chat = AsyncMock(
            side_effect=track_room_creation
        )
        adapter._rest.agent_api.add_agent_chat_participant = AsyncMock()
        adapter._rest.agent_api.create_agent_chat_message = AsyncMock()
        adapter._rest.agent_api.create_agent_chat_event = AsyncMock()
        adapter._rooms_created = rooms_created  # Expose for assertions

        return adapter

    @pytest.mark.asyncio
    async def test_same_context_id_twice_uses_same_room(
        self, gateway_adapter_with_mocks: A2AGatewayAdapter
    ) -> None:
        """Same contextId sent twice should reuse the same chat room."""
        adapter = gateway_adapter_with_mocks

        # First request with context_id="ctx-user-session"
        room_1, ctx_1 = await adapter._get_or_create_room(
            "ctx-user-session", "uuid-weather"
        )

        # Second request with SAME context_id
        room_2, ctx_2 = await adapter._get_or_create_room(
            "ctx-user-session", "uuid-weather"
        )

        # Assertions
        assert room_1 == room_2, f"Expected same room, got {room_1} vs {room_2}"
        assert ctx_1 == ctx_2 == "ctx-user-session"
        assert len(adapter._rooms_created) == 1, "Should only create 1 room"
        assert adapter._rest.agent_api.create_agent_chat.call_count == 1

    @pytest.mark.asyncio
    async def test_different_context_ids_create_different_rooms(
        self, gateway_adapter_with_mocks: A2AGatewayAdapter
    ) -> None:
        """Different contextIds should create separate chat rooms."""
        adapter = gateway_adapter_with_mocks

        # First context
        room_a, ctx_a = await adapter._get_or_create_room(
            "ctx-session-a", "uuid-weather"
        )

        # Second context (different)
        room_b, ctx_b = await adapter._get_or_create_room(
            "ctx-session-b", "uuid-weather"
        )

        # Assertions
        assert room_a != room_b, f"Expected different rooms, got same: {room_a}"
        assert ctx_a == "ctx-session-a"
        assert ctx_b == "ctx-session-b"
        assert len(adapter._rooms_created) == 2, "Should create 2 separate rooms"

    @pytest.mark.asyncio
    async def test_same_context_different_peers_same_room_adds_peer(
        self, gateway_adapter_with_mocks: A2AGatewayAdapter
    ) -> None:
        """Same contextId with different peers should use same room, add peer."""
        adapter = gateway_adapter_with_mocks

        # Add second peer
        data_peer = Peer(id="uuid-data", name="Data Agent", type="agent")
        adapter._peers["data-agent"] = data_peer
        adapter._peers_by_uuid["uuid-data"] = data_peer

        # First peer
        room_1, _ = await adapter._get_or_create_room("ctx-multi", "uuid-weather")

        # Second peer, same context
        room_2, _ = await adapter._get_or_create_room("ctx-multi", "uuid-data")

        # Assertions
        assert room_1 == room_2, "Same context should use same room"
        assert len(adapter._rooms_created) == 1, "Should only create 1 room"
        assert "uuid-weather" in adapter._room_participants[room_1]
        assert "uuid-data" in adapter._room_participants[room_1]
        assert adapter._rest.agent_api.add_agent_chat_participant.call_count == 2


@requires_api
class TestA2AGatewayContextIdWithPlatform:
    """E2E tests using real Thenvoi platform (requires API keys in .env.test).

    These tests actually:
    1. Create rooms on the real platform
    2. Add participants (peers) to rooms
    3. Send real messages with @mentions
    4. Query room context to verify messages arrive
    5. Verify same contextId -> same room with shared messages
    """

    @pytest.mark.asyncio
    async def test_same_context_routes_messages_to_same_room(
        self, api_client: AsyncRestClient, integration_settings
    ) -> None:
        """Same context_id twice should route to same room with shared messages.

        Flow:
        1. Create room with context_id via _get_or_create_room()
        2. Send message 1 to that room
        3. Call _get_or_create_room() again with SAME context_id
        4. Verify same room is returned
        5. Send message 2 to that room
        6. Query room context - BOTH messages should be present
        """
        # Get agent identity
        response = await api_client.agent_api.get_agent_me()
        agent_id = response.data.id

        # Get a peer to add to rooms
        response = await api_client.agent_api.list_agent_peers()
        if not response.data:
            pytest.skip("No peers available")
        peer = response.data[0]

        # Create adapter with real REST client credentials
        adapter = A2AGatewayAdapter(
            rest_url=integration_settings.thenvoi_base_url,
            api_key=integration_settings.thenvoi_api_key,
            gateway_url="http://localhost:10000",
            port=10000,
        )
        adapter._peers = {peer.name.lower().replace(" ", "-"): peer}
        adapter._peers_by_uuid = {peer.id: peer}

        # ===== Step 1: First request with context_id =====
        context_id = f"e2e-context-{agent_id[:8]}"
        room_1, ctx_1 = await adapter._get_or_create_room(context_id, peer.id)

        await asyncio.sleep(0.5)  # Platform consistency

        # Send message to room 1
        msg1_response = await api_client.agent_api.create_agent_chat_message(
            room_1,
            message=ChatMessageRequest(
                content=f"Message 1 from context {context_id}",
                mentions=[Mention(id=peer.id, name=peer.name)],
            ),
        )
        msg1_id = msg1_response.data.id

        # ===== Step 2: Second request with SAME context_id =====
        room_2, ctx_2 = await adapter._get_or_create_room(context_id, peer.id)

        # Verify same room
        assert room_1 == room_2, (
            f"Same context should use same room: {room_1} vs {room_2}"
        )
        assert ctx_1 == ctx_2 == context_id

        # Send second message
        msg2_response = await api_client.agent_api.create_agent_chat_message(
            room_2,
            message=ChatMessageRequest(
                content=f"Message 2 from same context {context_id}",
                mentions=[Mention(id=peer.id, name=peer.name)],
            ),
        )
        msg2_id = msg2_response.data.id

        await asyncio.sleep(0.5)

        # ===== Step 3: Verify BOTH messages in SAME room =====
        context_response = await api_client.agent_api.get_agent_chat_context(room_1)
        context_items = context_response.data or []

        msg_ids_in_context = [item.id for item in context_items if hasattr(item, "id")]

        assert msg1_id in msg_ids_in_context, (
            f"Message 1 ({msg1_id}) should be in room context"
        )
        assert msg2_id in msg_ids_in_context, (
            f"Message 2 ({msg2_id}) should be in room context"
        )

    @pytest.mark.asyncio
    async def test_different_contexts_create_separate_rooms_with_isolated_messages(
        self, api_client: AsyncRestClient, integration_settings
    ) -> None:
        """Different context_ids should create separate rooms with isolated messages.

        Flow:
        1. Create room A with context_a, send message A
        2. Create room B with context_b, send message B
        3. Verify room A != room B
        4. Verify room A contains message A but NOT message B
        5. Verify room B contains message B but NOT message A
        """
        # Get agent and peer
        response = await api_client.agent_api.get_agent_me()
        agent_id = response.data.id

        response = await api_client.agent_api.list_agent_peers()
        if not response.data:
            pytest.skip("No peers available")
        peer = response.data[0]

        # Create adapter
        adapter = A2AGatewayAdapter(
            rest_url=integration_settings.thenvoi_base_url,
            api_key=integration_settings.thenvoi_api_key,
            gateway_url="http://localhost:10000",
            port=10000,
        )
        adapter._peers = {peer.name.lower().replace(" ", "-"): peer}
        adapter._peers_by_uuid = {peer.id: peer}

        # ===== Context A: Create room, send message =====
        context_a = f"e2e-ctx-a-{agent_id[:8]}"
        room_a, _ = await adapter._get_or_create_room(context_a, peer.id)

        await asyncio.sleep(0.3)

        msg_a_response = await api_client.agent_api.create_agent_chat_message(
            room_a,
            message=ChatMessageRequest(
                content="Message for context A only",
                mentions=[Mention(id=peer.id, name=peer.name)],
            ),
        )
        msg_a_id = msg_a_response.data.id

        # ===== Context B: Create DIFFERENT room, send message =====
        context_b = f"e2e-ctx-b-{agent_id[:8]}"
        room_b, _ = await adapter._get_or_create_room(context_b, peer.id)

        # Verify different rooms
        assert room_a != room_b, "Different contexts should create different rooms"

        await asyncio.sleep(0.3)

        msg_b_response = await api_client.agent_api.create_agent_chat_message(
            room_b,
            message=ChatMessageRequest(
                content="Message for context B only",
                mentions=[Mention(id=peer.id, name=peer.name)],
            ),
        )
        msg_b_id = msg_b_response.data.id

        await asyncio.sleep(0.5)

        # ===== Verify Room A has only message A =====
        context_a_response = await api_client.agent_api.get_agent_chat_context(room_a)
        context_a_items = context_a_response.data or []
        msg_ids_in_a = [item.id for item in context_a_items if hasattr(item, "id")]

        assert msg_a_id in msg_ids_in_a, "Room A should contain message A"
        assert msg_b_id not in msg_ids_in_a, "Room A should NOT contain message B"

        # ===== Verify Room B has only message B =====
        context_b_response = await api_client.agent_api.get_agent_chat_context(room_b)
        context_b_items = context_b_response.data or []
        msg_ids_in_b = [item.id for item in context_b_items if hasattr(item, "id")]

        assert msg_b_id in msg_ids_in_b, "Room B should contain message B"
        assert msg_a_id not in msg_ids_in_b, "Room B should NOT contain message A"

    @pytest.mark.asyncio
    async def test_same_context_multiple_peers_shares_room(
        self, api_client: AsyncRestClient, integration_settings
    ) -> None:
        """Same context_id with different peers should use same room, add all peers.

        Flow:
        1. Get 2 different peers
        2. Create room with context_id for peer_1
        3. Call _get_or_create_room with SAME context_id for peer_2
        4. Verify same room is returned
        5. Verify both peers are in the room's participant list
        """
        # Get agent
        response = await api_client.agent_api.get_agent_me()
        agent_id = response.data.id

        # Get multiple peers
        response = await api_client.agent_api.list_agent_peers()
        if not response.data or len(response.data) < 2:
            pytest.skip("Need at least 2 peers for this test")
        peer_1 = response.data[0]
        peer_2 = response.data[1]

        # Create adapter with both peers
        adapter = A2AGatewayAdapter(
            rest_url=integration_settings.thenvoi_base_url,
            api_key=integration_settings.thenvoi_api_key,
            gateway_url="http://localhost:10000",
            port=10000,
        )
        adapter._peers = {
            peer_1.name.lower().replace(" ", "-"): peer_1,
            peer_2.name.lower().replace(" ", "-"): peer_2,
        }
        adapter._peers_by_uuid = {peer_1.id: peer_1, peer_2.id: peer_2}

        # Same context, first peer
        context_id = f"e2e-multi-peer-{agent_id[:8]}"
        room_1, _ = await adapter._get_or_create_room(context_id, peer_1.id)

        await asyncio.sleep(0.3)

        # Same context, second peer
        room_2, _ = await adapter._get_or_create_room(context_id, peer_2.id)

        # Verify same room
        assert room_1 == room_2, "Same context should use same room for different peers"

        await asyncio.sleep(0.3)

        # Verify both peers are participants
        response = await api_client.agent_api.list_agent_chat_participants(room_1)
        participant_ids = [p.id for p in response.data]

        assert peer_1.id in participant_ids, f"Peer 1 ({peer_1.name}) should be in room"
        assert peer_2.id in participant_ids, f"Peer 2 ({peer_2.name}) should be in room"
