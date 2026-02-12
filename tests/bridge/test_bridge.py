"""Tests for bridge configuration, initialization, and runtime."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from thenvoi.client.streaming import (
    Mention,
    MessageCreatedPayload,
    MessageMetadata,
    RoomAddedPayload,
    RoomOwner,
    RoomRemovedPayload,
)
from thenvoi.platform.event import (
    MessageEvent,
    ParticipantAddedEvent,
    RoomAddedEvent,
    RoomRemovedEvent,
)

from bridge_core.bridge import BridgeConfig, ReconnectConfig, ThenvoiBridge


class TestBridgeConfig:
    def test_validate_missing_agent_id(self) -> None:
        with pytest.raises(ValueError, match="THENVOI_AGENT_ID"):
            BridgeConfig(agent_id="", api_key="key", agent_mapping="a:b")

    def test_validate_missing_api_key(self) -> None:
        with pytest.raises(ValueError, match="THENVOI_API_KEY"):
            BridgeConfig(agent_id="id", api_key="", agent_mapping="a:b")

    def test_validate_missing_agent_mapping(self) -> None:
        with pytest.raises(ValueError, match="AGENT_MAPPING"):
            BridgeConfig(agent_id="id", api_key="key", agent_mapping="")

    def test_validate_missing_agent_mapping_required(self) -> None:
        """agent_mapping is a required field (no default)."""
        with pytest.raises(Exception):
            BridgeConfig(agent_id="id", api_key="key")  # type: ignore[call-arg]

    def test_validate_success(self) -> None:
        # Should not raise — valid config
        config = BridgeConfig(agent_id="id", api_key="key", agent_mapping="a:b")
        assert config.agent_id == "id"

    def test_api_key_hidden_in_repr(self) -> None:
        config = BridgeConfig(
            agent_id="id", api_key="secret-key-123", agent_mapping="a:b"
        )
        config_repr = repr(config)
        assert "secret-key-123" not in config_repr

    def test_session_ttl_zero_disables_eviction(self) -> None:
        config = BridgeConfig(
            agent_id="id", api_key="key", agent_mapping="a:b", session_ttl=0
        )
        assert config.session_ttl == 0

    @pytest.mark.parametrize("ttl", [-1, -100.5])
    def test_invalid_session_ttl(self, ttl: float) -> None:
        with pytest.raises(ValueError, match="SESSION_TTL must be non-negative"):
            BridgeConfig(
                agent_id="id", api_key="key", agent_mapping="a:b", session_ttl=ttl
            )

    def test_invalid_health_port_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="HEALTH_PORT must be between 1 and 65535"):
            BridgeConfig(
                agent_id="id", api_key="key", agent_mapping="a:b", health_port=0
            )

    def test_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("THENVOI_AGENT_ID", "test-agent")
        monkeypatch.setenv("THENVOI_API_KEY", "test-key")
        monkeypatch.setenv("AGENT_MAPPING", "alice:handler_a")
        monkeypatch.setenv("HEALTH_PORT", "9090")
        monkeypatch.setenv("HEALTH_HOST", "127.0.0.1")

        config = BridgeConfig.from_env()
        assert config.agent_id == "test-agent"
        assert config.api_key == "test-key"
        assert config.agent_mapping == "alice:handler_a"
        assert config.health_port == 9090
        assert config.health_host == "127.0.0.1"

    def test_from_env_with_session_ttl(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("THENVOI_AGENT_ID", "test-agent")
        monkeypatch.setenv("THENVOI_API_KEY", "test-key")
        monkeypatch.setenv("AGENT_MAPPING", "alice:handler_a")
        monkeypatch.setenv("SESSION_TTL", "3600")

        config = BridgeConfig.from_env()
        assert config.session_ttl == 3600.0

    def test_from_env_invalid_session_ttl(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("THENVOI_AGENT_ID", "test-agent")
        monkeypatch.setenv("THENVOI_API_KEY", "test-key")
        monkeypatch.setenv("AGENT_MAPPING", "alice:handler_a")
        monkeypatch.setenv("SESSION_TTL", "not-a-number")

        with pytest.raises(ValueError, match="SESSION_TTL must be a valid number"):
            BridgeConfig.from_env()

    def test_from_env_session_ttl_zero_disables_eviction(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("THENVOI_AGENT_ID", "test-agent")
        monkeypatch.setenv("THENVOI_API_KEY", "test-key")
        monkeypatch.setenv("AGENT_MAPPING", "alice:handler_a")
        monkeypatch.setenv("SESSION_TTL", "0")

        config = BridgeConfig.from_env()
        assert config.session_ttl == 0

    @pytest.mark.parametrize("ttl", ["-1", "-100.5"])
    def test_from_env_invalid_session_ttl_negative(
        self, monkeypatch: pytest.MonkeyPatch, ttl: str
    ) -> None:
        monkeypatch.setenv("THENVOI_AGENT_ID", "test-agent")
        monkeypatch.setenv("THENVOI_API_KEY", "test-key")
        monkeypatch.setenv("AGENT_MAPPING", "alice:handler_a")
        monkeypatch.setenv("SESSION_TTL", ttl)

        with pytest.raises(ValueError, match="SESSION_TTL must be non-negative"):
            BridgeConfig.from_env()

    def test_from_env_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("THENVOI_AGENT_ID", "test-agent")
        monkeypatch.setenv("THENVOI_API_KEY", "test-key")
        monkeypatch.setenv("AGENT_MAPPING", "alice:handler_a")
        # Ensure optional vars are not set so model defaults are used
        for var in (
            "THENVOI_WS_URL",
            "THENVOI_REST_URL",
            "HEALTH_PORT",
            "HEALTH_HOST",
            "SESSION_TTL",
        ):
            monkeypatch.delenv(var, raising=False)

        config = BridgeConfig.from_env()
        assert config.ws_url == "wss://app.thenvoi.com/api/v1/socket/websocket"
        assert config.rest_url == "https://app.thenvoi.com"
        assert config.health_port == 8080
        assert config.health_host == "0.0.0.0"
        assert config.session_ttl == 86400.0

    def test_from_env_invalid_health_port(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("THENVOI_AGENT_ID", "test-agent")
        monkeypatch.setenv("THENVOI_API_KEY", "test-key")
        monkeypatch.setenv("AGENT_MAPPING", "alice:handler_a")
        monkeypatch.setenv("HEALTH_PORT", "not-a-number")

        with pytest.raises(ValueError, match="HEALTH_PORT must be a valid integer"):
            BridgeConfig.from_env()

    @pytest.mark.parametrize("port", ["0", "-1", "65536", "99999"])
    def test_from_env_health_port_out_of_range(
        self, monkeypatch: pytest.MonkeyPatch, port: str
    ) -> None:
        monkeypatch.setenv("THENVOI_AGENT_ID", "test-agent")
        monkeypatch.setenv("THENVOI_API_KEY", "test-key")
        monkeypatch.setenv("AGENT_MAPPING", "alice:handler_a")
        monkeypatch.setenv("HEALTH_PORT", port)

        with pytest.raises(ValueError, match="HEALTH_PORT must be between 1 and 65535"):
            BridgeConfig.from_env()

    def test_from_env_missing_required(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("THENVOI_AGENT_ID", raising=False)
        monkeypatch.delenv("THENVOI_API_KEY", raising=False)
        monkeypatch.delenv("AGENT_MAPPING", raising=False)

        with pytest.raises(ValueError):
            BridgeConfig.from_env()


class TestReconnectConfig:
    def test_defaults_are_valid(self) -> None:
        config = ReconnectConfig()
        assert config.initial_delay == 1.0
        assert config.max_delay == 60.0
        assert config.multiplier == 2.0
        assert config.jitter == 0.5
        assert config.max_retries == 0

    @pytest.mark.parametrize("value", [0, -1, -0.5])
    def test_invalid_initial_delay(self, value: float) -> None:
        with pytest.raises(ValueError, match="initial_delay must be positive"):
            ReconnectConfig(initial_delay=value)

    @pytest.mark.parametrize("value", [0, -1])
    def test_invalid_max_delay(self, value: float) -> None:
        with pytest.raises(ValueError, match="max_delay must be positive"):
            ReconnectConfig(max_delay=value)

    @pytest.mark.parametrize("value", [0.5, 0, -1])
    def test_invalid_multiplier(self, value: float) -> None:
        with pytest.raises(ValueError, match="multiplier must be >= 1"):
            ReconnectConfig(multiplier=value)

    def test_invalid_jitter(self) -> None:
        with pytest.raises(ValueError, match="jitter must be non-negative"):
            ReconnectConfig(jitter=-0.1)

    def test_invalid_max_retries(self) -> None:
        with pytest.raises(ValueError, match="max_retries must be non-negative"):
            ReconnectConfig(max_retries=-1)

    def test_jitter_zero_is_valid(self) -> None:
        config = ReconnectConfig(jitter=0)
        assert config.jitter == 0

    def test_multiplier_one_is_valid(self) -> None:
        config = ReconnectConfig(multiplier=1.0)
        assert config.multiplier == 1.0


class TestThenvoiBridgeInit:
    def _make_config(self, mapping: str = "alice:handler_a") -> BridgeConfig:
        return BridgeConfig(
            agent_id="agent-1",
            api_key="key-1",
            agent_mapping=mapping,
        )

    def test_init_valid(self) -> None:
        handler = AsyncMock()
        bridge = ThenvoiBridge(
            config=self._make_config(),
            handlers={"handler_a": handler},
        )
        assert bridge._agent_mapping == {"alice": "handler_a"}

    def test_init_missing_handler_raises(self) -> None:
        with pytest.raises(ValueError, match="no handler with that name"):
            ThenvoiBridge(
                config=self._make_config("alice:missing_handler"),
                handlers={"handler_a": AsyncMock()},
            )

    def test_init_invalid_mapping_raises(self) -> None:
        with pytest.raises(ValueError):
            ThenvoiBridge(
                config=self._make_config("invalid_no_colon"),
                handlers={},
            )

    def test_init_multiple_handlers(self) -> None:
        bridge = ThenvoiBridge(
            config=self._make_config("alice:handler_a,bob:handler_b"),
            handlers={"handler_a": AsyncMock(), "handler_b": AsyncMock()},
        )
        assert bridge._agent_mapping == {"alice": "handler_a", "bob": "handler_b"}


class TestThenvoiBridgeHandleEvent:
    """Tests for _handle_event dispatch."""

    async def test_room_added_subscribes(
        self, bridge_with_mock_link: ThenvoiBridge
    ) -> None:
        bridge = bridge_with_mock_link
        event = RoomAddedEvent(
            room_id="room-new",
            payload=RoomAddedPayload(
                id="room-new",
                title="New Room",
                owner=RoomOwner(id="user-1", name="User", type="User"),
                status="active",
                type="direct",
                created_at="2024-01-01T00:00:00Z",
                participant_role="member",
            ),
        )

        await bridge._handle_event(event)

        bridge._link.subscribe_room.assert_called_once_with("room-new")

    async def test_room_removed_unsubscribes_and_cleans_session(
        self, bridge_with_mock_link: ThenvoiBridge
    ) -> None:
        bridge = bridge_with_mock_link
        # Pre-populate a session
        await bridge._session_store.get_or_create("room-old")

        event = RoomRemovedEvent(
            room_id="room-old",
            payload=RoomRemovedPayload(
                id="room-old",
                status="removed",
                type="direct",
                title="Old Room",
                removed_at="2024-01-01T00:00:00Z",
            ),
        )

        await bridge._handle_event(event)

        bridge._link.unsubscribe_room.assert_called_once_with("room-old")
        session = await bridge._session_store.get("room-old")
        assert session is None

    async def test_room_added_subscribe_failure_does_not_propagate(
        self, bridge_with_mock_link: ThenvoiBridge
    ) -> None:
        bridge = bridge_with_mock_link
        bridge._link.subscribe_room = AsyncMock(
            side_effect=ConnectionError("subscribe failed")
        )
        event = RoomAddedEvent(
            room_id="room-new",
            payload=RoomAddedPayload(
                id="room-new",
                title="New Room",
                owner=RoomOwner(id="user-1", name="User", type="User"),
                status="active",
                type="direct",
                created_at="2024-01-01T00:00:00Z",
                participant_role="member",
            ),
        )

        # Should not raise
        await bridge._handle_event(event)

    async def test_room_removed_unsubscribe_failure_still_cleans_session(
        self, bridge_with_mock_link: ThenvoiBridge
    ) -> None:
        bridge = bridge_with_mock_link
        bridge._link.unsubscribe_room = AsyncMock(
            side_effect=ConnectionError("unsubscribe failed")
        )
        # Pre-populate a session
        await bridge._session_store.get_or_create("room-old")

        event = RoomRemovedEvent(
            room_id="room-old",
            payload=RoomRemovedPayload(
                id="room-old",
                status="removed",
                type="direct",
                title="Old Room",
                removed_at="2024-01-01T00:00:00Z",
            ),
        )

        # Should not raise, and session should still be cleaned up
        await bridge._handle_event(event)
        session = await bridge._session_store.get("room-old")
        assert session is None

    async def test_unhandled_event_does_not_raise(
        self, bridge_with_mock_link: ThenvoiBridge
    ) -> None:
        from thenvoi.client.streaming import ParticipantAddedPayload

        bridge = bridge_with_mock_link
        event = ParticipantAddedEvent(
            room_id="room-1",
            payload=ParticipantAddedPayload(id="user-1", name="User", type="User"),
        )
        # Should not raise
        await bridge._handle_event(event)

    async def test_message_event_routes_to_handler(
        self, bridge_with_mock_link: ThenvoiBridge
    ) -> None:
        bridge = bridge_with_mock_link
        payload = MessageCreatedPayload(
            id="msg-1",
            content="@alice hello",
            message_type="text",
            sender_id="user-1",
            sender_type="User",
            chat_room_id="room-1",
            inserted_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            metadata=MessageMetadata(
                mentions=[Mention(id="alice-id", username="alice")], status="sent"
            ),
        )
        event = MessageEvent(room_id="room-1", payload=payload)

        # Mock participant fetch
        mock_response = MagicMock()
        mock_response.data = []
        bridge._link.rest.agent_api.list_agent_chat_participants = AsyncMock(
            return_value=mock_response
        )
        bridge._link.mark_processing = AsyncMock()
        bridge._link.mark_processed = AsyncMock()

        await bridge._handle_event(event)

        # The handler should have been called via the router
        bridge._handlers["handler_a"].handle.assert_called_once()
        # Message lifecycle marks should be called once (via router)
        bridge._link.mark_processing.assert_called_once_with("room-1", "msg-1")
        bridge._link.mark_processed.assert_called_once_with("room-1", "msg-1")

    async def test_message_event_with_none_payload_ignored(
        self, bridge_with_mock_link: ThenvoiBridge
    ) -> None:
        bridge = bridge_with_mock_link
        event = MessageEvent(room_id="room-1", payload=None)

        # Should not raise or route
        await bridge._handle_event(event)


class TestThenvoiBridgeFetchExistingRooms:
    async def test_returns_room_ids(self, bridge_config: BridgeConfig) -> None:
        bridge = ThenvoiBridge(
            config=bridge_config, handlers={"handler_a": AsyncMock()}
        )

        mock_room_1 = MagicMock()
        mock_room_1.id = "room-1"
        mock_room_2 = MagicMock()
        mock_room_2.id = "room-2"

        mock_response = MagicMock()
        mock_response.data = [mock_room_1, mock_room_2]
        bridge._link.rest.agent_api.list_agent_chats = AsyncMock(
            return_value=mock_response
        )

        rooms = await bridge._fetch_existing_rooms()
        assert rooms == ["room-1", "room-2"]

    async def test_returns_empty_on_error(self, bridge_config: BridgeConfig) -> None:
        bridge = ThenvoiBridge(
            config=bridge_config, handlers={"handler_a": AsyncMock()}
        )

        bridge._link.rest.agent_api.list_agent_chats = AsyncMock(
            side_effect=RuntimeError("connection failed")
        )

        rooms = await bridge._fetch_existing_rooms()
        assert rooms == []

    async def test_returns_empty_on_no_data(self, bridge_config: BridgeConfig) -> None:
        bridge = ThenvoiBridge(
            config=bridge_config, handlers={"handler_a": AsyncMock()}
        )

        mock_response = MagicMock()
        mock_response.data = None
        bridge._link.rest.agent_api.list_agent_chats = AsyncMock(
            return_value=mock_response
        )

        rooms = await bridge._fetch_existing_rooms()
        assert rooms == []


class TestThenvoiBridgeReconnect:
    async def test_reconnect_exits_cleanly_on_success(
        self, bridge_config: BridgeConfig
    ) -> None:
        bridge = ThenvoiBridge(
            config=bridge_config, handlers={"handler_a": AsyncMock()}
        )

        # Make _connect_and_consume succeed immediately (simulating clean exit)
        bridge._connect_and_consume = AsyncMock()

        await bridge._run_with_reconnect()

        bridge._connect_and_consume.assert_called_once()

    async def test_reconnect_stops_on_shutdown(
        self, bridge_config: BridgeConfig
    ) -> None:
        bridge = ThenvoiBridge(
            config=bridge_config, handlers={"handler_a": AsyncMock()}
        )
        bridge._request_shutdown()

        bridge._connect_and_consume = AsyncMock()

        await bridge._run_with_reconnect()

        bridge._connect_and_consume.assert_not_called()

    async def test_reconnect_retries_on_error(
        self, bridge_config: BridgeConfig
    ) -> None:
        bridge = ThenvoiBridge(
            config=bridge_config, handlers={"handler_a": AsyncMock()}
        )
        bridge._link = MagicMock()
        bridge._link.disconnect = AsyncMock()

        # First call fails, second triggers shutdown
        call_count = 0

        async def fail_then_shutdown() -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("lost connection")
            bridge._request_shutdown()
            raise ConnectionError("still lost")

        bridge._connect_and_consume = AsyncMock(side_effect=fail_then_shutdown)

        with patch("bridge_core.bridge.asyncio.sleep", new_callable=AsyncMock):
            await bridge._run_with_reconnect()

        assert call_count == 2
        bridge._link.disconnect.assert_called()

    async def test_reconnect_stops_after_max_retries(
        self, bridge_config: BridgeConfig
    ) -> None:
        reconnect = ReconnectConfig(max_retries=3)
        bridge = ThenvoiBridge(
            config=bridge_config,
            handlers={"handler_a": AsyncMock()},
            reconnect_config=reconnect,
        )
        bridge._link = MagicMock()
        bridge._link.disconnect = AsyncMock()

        bridge._connect_and_consume = AsyncMock(
            side_effect=ConnectionError("always fails")
        )

        with patch("bridge_core.bridge.asyncio.sleep", new_callable=AsyncMock):
            await bridge._run_with_reconnect()

        assert bridge._connect_and_consume.call_count == 3

    async def test_reconnect_resets_delay_after_successful_connection(
        self, bridge_config: BridgeConfig
    ) -> None:
        """After a successful connection that later drops, backoff should reset."""
        reconnect = ReconnectConfig(initial_delay=1.0, multiplier=2.0, jitter=0)
        bridge = ThenvoiBridge(
            config=bridge_config,
            handlers={"handler_a": AsyncMock()},
            reconnect_config=reconnect,
        )
        bridge._link = MagicMock()
        bridge._link.disconnect = AsyncMock()

        call_count = 0
        sleep_delays: list[float] = []

        async def connect_then_fail() -> None:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                # First two attempts: fail without connecting (backoff should escalate)
                raise ConnectionError("connect failed")
            if call_count == 3:
                # Third attempt: connect successfully, then drop later
                bridge._connected = True
                raise ConnectionError("runtime disconnect")
            # Fourth attempt: fail without connecting — delay should be reset
            bridge._request_shutdown()
            raise ConnectionError("connect failed again")

        bridge._connect_and_consume = AsyncMock(side_effect=connect_then_fail)

        async def capture_sleep(delay: float) -> None:
            sleep_delays.append(delay)

        with patch("bridge_core.bridge.asyncio.sleep", side_effect=capture_sleep):
            await bridge._run_with_reconnect()

        assert call_count == 4
        # sleep_delays[0]: after 1st failure (no connection), delay=1.0
        # sleep_delays[1]: after 2nd failure (no connection), delay escalated to 2.0
        # sleep_delays[2]: after 3rd failure (was connected), delay reset to 1.0
        assert sleep_delays[0] == 1.0
        assert sleep_delays[1] == 2.0
        assert sleep_delays[2] == 1.0


class TestThenvoiBridgeShutdown:
    def test_request_shutdown_sets_flag(self, bridge_config: BridgeConfig) -> None:
        bridge = ThenvoiBridge(
            config=bridge_config, handlers={"handler_a": AsyncMock()}
        )

        assert bridge._shutting_down is False
        bridge._request_shutdown()
        assert bridge._shutting_down is True

    def test_request_shutdown_idempotent(self, bridge_config: BridgeConfig) -> None:
        bridge = ThenvoiBridge(
            config=bridge_config, handlers={"handler_a": AsyncMock()}
        )

        bridge._request_shutdown()
        bridge._request_shutdown()
        assert bridge._shutting_down is True

    async def test_shutdown_disconnects_and_stops_health(
        self, bridge_config: BridgeConfig
    ) -> None:
        bridge = ThenvoiBridge(
            config=bridge_config, handlers={"handler_a": AsyncMock()}
        )
        bridge._link = MagicMock()
        bridge._link.disconnect = AsyncMock()
        bridge._health = MagicMock()
        bridge._health.stop = AsyncMock()

        await bridge._shutdown()

        bridge._link.disconnect.assert_called_once()
        bridge._health.stop.assert_called_once()

    async def test_shutdown_stops_health_even_if_disconnect_raises(
        self, bridge_config: BridgeConfig
    ) -> None:
        bridge = ThenvoiBridge(
            config=bridge_config, handlers={"handler_a": AsyncMock()}
        )
        bridge._link = MagicMock()
        bridge._link.disconnect = AsyncMock(side_effect=RuntimeError("disconnect boom"))
        bridge._health = MagicMock()
        bridge._health.stop = AsyncMock()

        await bridge._shutdown()

        bridge._link.disconnect.assert_called_once()
        bridge._health.stop.assert_called_once()


class TestConnectAndConsume:
    """Tests for _connect_and_consume event loop logic."""

    def _make_bridge(self, bridge_config: BridgeConfig) -> ThenvoiBridge:
        handler = AsyncMock()
        b = ThenvoiBridge(config=bridge_config, handlers={"handler_a": handler})
        mock_link = MagicMock()
        mock_link.connect = AsyncMock()
        mock_link.subscribe_agent_rooms = AsyncMock()
        mock_link.subscribe_room = AsyncMock()
        mock_link.rest = MagicMock()
        mock_link.rest.agent_api.list_agent_chats = AsyncMock(
            return_value=MagicMock(data=None)
        )
        b._link = mock_link
        b._router._link = mock_link
        return b

    async def test_consumes_events_until_shutdown(
        self, bridge_config: BridgeConfig
    ) -> None:
        bridge = self._make_bridge(bridge_config)
        events_delivered: list[object] = []

        event1 = RoomAddedEvent(
            room_id="room-1",
            payload=RoomAddedPayload(
                id="room-1",
                title="Room",
                owner=RoomOwner(id="u1", name="User", type="User"),
                status="active",
                type="direct",
                created_at="2024-01-01T00:00:00Z",
                participant_role="member",
            ),
        )
        event2 = RoomAddedEvent(
            room_id="room-2",
            payload=RoomAddedPayload(
                id="room-2",
                title="Room 2",
                owner=RoomOwner(id="u1", name="User", type="User"),
                status="active",
                type="direct",
                created_at="2024-01-01T00:00:00Z",
                participant_role="member",
            ),
        )

        call_count = 0

        async def fake_anext(_iter: object) -> object:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return event1
            if call_count == 2:
                return event2
            # After 2 events, trigger shutdown
            bridge._request_shutdown()
            # Return a future that never resolves — shutdown_fut will win the race
            await AsyncMock(side_effect=lambda: None)()
            return MagicMock()

        original_handle = bridge._handle_event

        async def tracking_handle(event: object) -> None:
            events_delivered.append(event)
            await original_handle(event)

        bridge._handle_event = tracking_handle  # type: ignore[assignment]

        with patch("bridge_core.bridge.anext", side_effect=fake_anext):
            await bridge._connect_and_consume()

        assert len(events_delivered) == 2
        bridge._link.connect.assert_called_once()
        bridge._link.subscribe_agent_rooms.assert_called_once()

    async def test_shutdown_cancels_pending_next_fut(
        self, bridge_config: BridgeConfig
    ) -> None:
        bridge = self._make_bridge(bridge_config)

        # Trigger shutdown immediately so shutdown_fut wins the race
        bridge._request_shutdown()

        never_resolving = AsyncMock(side_effect=lambda: None)

        async def fake_anext(_iter: object) -> object:
            await never_resolving()
            return MagicMock()

        with patch("bridge_core.bridge.anext", side_effect=fake_anext):
            await bridge._connect_and_consume()

        # Should exit cleanly without processing any events
        bridge._link.connect.assert_called_once()

    async def test_stop_async_iteration_exits_cleanly(
        self, bridge_config: BridgeConfig
    ) -> None:
        bridge = self._make_bridge(bridge_config)

        async def fake_anext(_iter: object) -> object:
            raise StopAsyncIteration

        with patch("bridge_core.bridge.anext", side_effect=fake_anext):
            await bridge._connect_and_consume()

        bridge._link.connect.assert_called_once()

    async def test_runtime_error_wrapping_stop_async_iteration_via_context(
        self, bridge_config: BridgeConfig
    ) -> None:
        """CPython wraps StopAsyncIteration in RuntimeError via __context__."""
        bridge = self._make_bridge(bridge_config)

        async def fake_anext(_iter: object) -> object:
            err = RuntimeError("coroutine raised StopAsyncIteration")
            err.__context__ = StopAsyncIteration()
            raise err

        with patch("bridge_core.bridge.anext", side_effect=fake_anext):
            await bridge._connect_and_consume()

        bridge._link.connect.assert_called_once()

    async def test_runtime_error_wrapping_stop_async_iteration_via_cause(
        self, bridge_config: BridgeConfig
    ) -> None:
        """Explicit chaining: raise RuntimeError from StopAsyncIteration."""
        bridge = self._make_bridge(bridge_config)

        async def fake_anext(_iter: object) -> object:
            err = RuntimeError("iterator stopped")
            err.__cause__ = StopAsyncIteration()
            raise err

        with patch("bridge_core.bridge.anext", side_effect=fake_anext):
            await bridge._connect_and_consume()

        bridge._link.connect.assert_called_once()

    async def test_unrelated_runtime_error_propagates(
        self, bridge_config: BridgeConfig
    ) -> None:
        bridge = self._make_bridge(bridge_config)

        async def fake_anext(_iter: object) -> object:
            raise RuntimeError("unrelated error")

        with patch("bridge_core.bridge.anext", side_effect=fake_anext):
            with pytest.raises(RuntimeError, match="unrelated error"):
                await bridge._connect_and_consume()

    async def test_handle_event_exception_does_not_break_loop(
        self, bridge_config: BridgeConfig
    ) -> None:
        """An exception in _handle_event should be logged, not trigger reconnect."""
        bridge = self._make_bridge(bridge_config)
        events_delivered: list[object] = []

        event1 = RoomAddedEvent(
            room_id="room-1",
            payload=RoomAddedPayload(
                id="room-1",
                title="Room",
                owner=RoomOwner(id="u1", name="User", type="User"),
                status="active",
                type="direct",
                created_at="2024-01-01T00:00:00Z",
                participant_role="member",
            ),
        )
        event2 = RoomAddedEvent(
            room_id="room-2",
            payload=RoomAddedPayload(
                id="room-2",
                title="Room 2",
                owner=RoomOwner(id="u1", name="User", type="User"),
                status="active",
                type="direct",
                created_at="2024-01-01T00:00:00Z",
                participant_role="member",
            ),
        )

        call_count = 0

        async def fake_anext(_iter: object) -> object:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return event1
            if call_count == 2:
                return event2
            bridge._request_shutdown()
            await AsyncMock(side_effect=lambda: None)()
            return MagicMock()

        handle_call_count = 0

        async def failing_then_ok_handle(event: object) -> None:
            nonlocal handle_call_count
            handle_call_count += 1
            events_delivered.append(event)
            if handle_call_count == 1:
                raise RuntimeError("handler blew up")

        bridge._handle_event = failing_then_ok_handle  # type: ignore[assignment]

        with patch("bridge_core.bridge.anext", side_effect=fake_anext):
            # Should NOT raise — the exception is caught and logged
            await bridge._connect_and_consume()

        # Both events should have been delivered despite the first one raising
        assert len(events_delivered) == 2


class TestThenvoiBridgeMain:
    """Tests for the main() entry point."""

    async def test_main_loads_env_and_runs_bridge(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("THENVOI_AGENT_ID", "test-agent")
        monkeypatch.setenv("THENVOI_API_KEY", "test-key")
        monkeypatch.setenv("AGENT_MAPPING", "alice:handler_a")

        mock_run = AsyncMock()

        with (
            patch("bridge_core.bridge.ThenvoiBridge.run", mock_run),
            patch("dotenv.load_dotenv") as mock_dotenv,
        ):
            from bridge_core.bridge import main

            handler = AsyncMock()
            await main(handlers={"handler_a": handler})

            mock_dotenv.assert_called_once()
            mock_run.assert_called_once()


class TestModuleMain:
    """Tests for python -m bridge_core entry point."""

    def test_module_main_exits_with_error(self) -> None:
        from bridge_core.__main__ import _main

        with pytest.raises(SystemExit, match="requires handlers to be registered"):
            _main()
