"""Tests for bridge @mention router."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from thenvoi.client.streaming import (
    Mention,
    MessageCreatedPayload,
    MessageMetadata,
)

from core.router import MentionRouter
from core.session import InMemorySessionStore


class TestParseAgentMapping:
    def test_valid_single_entry(self) -> None:
        result = MentionRouter.parse_agent_mapping("alice:alice_handler")
        assert result == {"alice": "alice_handler"}

    def test_valid_multiple_entries(self) -> None:
        result = MentionRouter.parse_agent_mapping("alice:handler_a,bob:handler_b")
        assert result == {"alice": "handler_a", "bob": "handler_b"}

    def test_strips_whitespace(self) -> None:
        result = MentionRouter.parse_agent_mapping(
            " alice : handler_a , bob : handler_b "
        )
        assert result == {"alice": "handler_a", "bob": "handler_b"}

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            MentionRouter.parse_agent_mapping("")

    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            MentionRouter.parse_agent_mapping("   ")

    def test_missing_colon_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid AGENT_MAPPING entry"):
            MentionRouter.parse_agent_mapping("alice_handler")

    def test_empty_key_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid AGENT_MAPPING entry"):
            MentionRouter.parse_agent_mapping(":handler")

    def test_empty_value_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid AGENT_MAPPING entry"):
            MentionRouter.parse_agent_mapping("alice:")

    def test_too_many_colons_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid AGENT_MAPPING entry"):
            MentionRouter.parse_agent_mapping("alice:handler:extra")

    def test_skips_empty_entries(self) -> None:
        result = MentionRouter.parse_agent_mapping("alice:handler_a,,bob:handler_b,")
        assert result == {"alice": "handler_a", "bob": "handler_b"}


def _make_payload(
    sender_id: str = "user-1",
    sender_type: str = "User",
    content: str = "hello",
    msg_id: str = "msg-1",
    mentions: list[Mention] | None = None,
    thread_id: str | None = None,
) -> MessageCreatedPayload:
    return MessageCreatedPayload(
        id=msg_id,
        content=content,
        message_type="text",
        sender_id=sender_id,
        sender_type=sender_type,
        chat_room_id="room-1",
        thread_id=thread_id,
        inserted_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        metadata=MessageMetadata(mentions=mentions or [], status="sent"),
    )


@pytest.fixture
def mock_link() -> AsyncMock:
    link = AsyncMock()
    link.mark_processing = AsyncMock()
    link.mark_processed = AsyncMock()
    link.mark_failed = AsyncMock()
    return link


class TestMentionRouterRoute:
    @pytest.fixture
    def mock_handler(self) -> AsyncMock:
        return AsyncMock()

    @pytest.fixture
    def session_store(self) -> InMemorySessionStore:
        return InMemorySessionStore()

    @pytest.fixture
    def router(
        self,
        mock_handler: AsyncMock,
        session_store: InMemorySessionStore,
        mock_link: AsyncMock,
    ) -> MentionRouter:
        return MentionRouter(
            agent_mapping={"alice": "alice_handler"},
            handlers={"alice_handler": mock_handler},
            session_store=session_store,
            agent_id="bridge-agent-id",
            link=mock_link,
        )

    async def test_routes_to_correct_handler(
        self, router: MentionRouter, mock_handler: AsyncMock
    ) -> None:
        payload = _make_payload(mentions=[Mention(id="alice-id", username="alice")])
        tools = MagicMock()

        await router.route(payload, "room-1", tools)

        mock_handler.handle.assert_called_once_with(
            content="hello",
            room_id="room-1",
            thread_id="room-1",
            message_id="msg-1",
            sender_id="user-1",
            sender_type="User",
            mentioned_agent="alice",
            tools=tools,
        )

    async def test_skips_self_messages(
        self, router: MentionRouter, mock_handler: AsyncMock
    ) -> None:
        payload = _make_payload(
            sender_id="bridge-agent-id",
            mentions=[Mention(id="alice-id", username="alice")],
        )
        tools = MagicMock()

        await router.route(payload, "room-1", tools)

        mock_handler.handle.assert_not_called()

    async def test_skips_unmapped_mentions(
        self, router: MentionRouter, mock_handler: AsyncMock
    ) -> None:
        payload = _make_payload(
            mentions=[Mention(id="unknown-id", username="unknown_agent")]
        )
        tools = MagicMock()

        await router.route(payload, "room-1", tools)

        mock_handler.handle.assert_not_called()

    async def test_duplicate_mentions_dispatched_once(
        self, router: MentionRouter, mock_handler: AsyncMock, mock_link: AsyncMock
    ) -> None:
        payload = _make_payload(
            mentions=[
                Mention(id="alice-id-1", username="alice"),
                Mention(id="alice-id-2", username="alice"),
            ]
        )
        tools = MagicMock()

        await router.route(payload, "room-1", tools)

        # Handler should be called once despite duplicate @alice mentions
        mock_handler.handle.assert_called_once()
        mock_link.mark_processing.assert_called_once_with("room-1", "msg-1")
        mock_link.mark_processed.assert_called_once_with("room-1", "msg-1")

    async def test_multi_mention_routing(
        self, session_store: InMemorySessionStore, mock_link: AsyncMock
    ) -> None:
        handler_a = AsyncMock()
        handler_b = AsyncMock()

        router = MentionRouter(
            agent_mapping={"alice": "handler_a", "bob": "handler_b"},
            handlers={"handler_a": handler_a, "handler_b": handler_b},
            session_store=session_store,
            agent_id="bridge-agent-id",
            link=mock_link,
        )

        payload = _make_payload(
            mentions=[
                Mention(id="alice-id", username="alice"),
                Mention(id="bob-id", username="bob"),
            ]
        )
        tools = MagicMock()

        await router.route(payload, "room-1", tools)

        handler_a.handle.assert_called_once()
        handler_b.handle.assert_called_once()
        # Lifecycle marks called once per message, not per handler
        mock_link.mark_processing.assert_called_once_with("room-1", "msg-1")
        mock_link.mark_processed.assert_called_once_with("room-1", "msg-1")

    async def test_creates_session_on_route(
        self,
        router: MentionRouter,
        mock_handler: AsyncMock,
        session_store: InMemorySessionStore,
    ) -> None:
        payload = _make_payload(mentions=[Mention(id="alice-id", username="alice")])
        tools = MagicMock()

        await router.route(payload, "room-1", tools)

        session = await session_store.get("room-1")
        assert session is not None

    async def test_uses_thread_id_from_payload(
        self,
        router: MentionRouter,
        mock_handler: AsyncMock,
        session_store: InMemorySessionStore,
    ) -> None:
        payload = _make_payload(
            thread_id="thread-42",
            mentions=[Mention(id="alice-id", username="alice")],
        )
        tools = MagicMock()

        await router.route(payload, "room-1", tools)

        mock_handler.handle.assert_called_once()
        call_kwargs = mock_handler.handle.call_args.kwargs
        assert call_kwargs["thread_id"] == "thread-42"

    async def test_thread_id_from_payload_not_session(
        self,
        router: MentionRouter,
        mock_handler: AsyncMock,
        session_store: InMemorySessionStore,
    ) -> None:
        """Thread ID should come from each message's payload, not the session."""
        payload_1 = _make_payload(
            msg_id="msg-1",
            thread_id="thread-1",
            mentions=[Mention(id="alice-id", username="alice")],
        )
        payload_2 = _make_payload(
            msg_id="msg-2",
            thread_id="thread-2",
            mentions=[Mention(id="alice-id", username="alice")],
        )
        tools = MagicMock()

        await router.route(payload_1, "room-1", tools)
        await router.route(payload_2, "room-1", tools)

        calls = mock_handler.handle.call_args_list
        assert calls[0].kwargs["thread_id"] == "thread-1"
        assert calls[1].kwargs["thread_id"] == "thread-2"

    async def test_no_mentions_does_nothing(
        self, router: MentionRouter, mock_handler: AsyncMock
    ) -> None:
        payload = _make_payload(mentions=[])
        tools = MagicMock()

        await router.route(payload, "room-1", tools)

        mock_handler.handle.assert_not_called()

    async def test_marks_processing_before_handler(
        self, router: MentionRouter, mock_handler: AsyncMock, mock_link: AsyncMock
    ) -> None:
        payload = _make_payload(mentions=[Mention(id="alice-id", username="alice")])
        tools = MagicMock()

        await router.route(payload, "room-1", tools)

        mock_link.mark_processing.assert_called_with("room-1", "msg-1")

    async def test_marks_processed_on_success(
        self, router: MentionRouter, mock_handler: AsyncMock, mock_link: AsyncMock
    ) -> None:
        payload = _make_payload(mentions=[Mention(id="alice-id", username="alice")])
        tools = MagicMock()

        await router.route(payload, "room-1", tools)

        mock_link.mark_processed.assert_called_with("room-1", "msg-1")

    async def test_marks_failed_on_handler_error(
        self, router: MentionRouter, mock_handler: AsyncMock, mock_link: AsyncMock
    ) -> None:
        mock_handler.handle.side_effect = RuntimeError("handler exploded")

        payload = _make_payload(mentions=[Mention(id="alice-id", username="alice")])
        tools = MagicMock()
        tools.send_event = AsyncMock()

        # Should not raise
        await router.route(payload, "room-1", tools)

        mock_link.mark_failed.assert_called_with(
            "room-1", "msg-1", "'alice_handler' (@alice): handler exploded"
        )

    async def test_sends_error_event_on_handler_failure(
        self, router: MentionRouter, mock_handler: AsyncMock, mock_link: AsyncMock
    ) -> None:
        mock_handler.handle.side_effect = RuntimeError("handler exploded")

        payload = _make_payload(mentions=[Mention(id="alice-id", username="alice")])
        tools = MagicMock()
        tools.send_event = AsyncMock()

        await router.route(payload, "room-1", tools)

        tools.send_event.assert_called_once_with(
            content="Handler failures: 'alice_handler' (@alice): handler exploded",
            message_type="error",
        )

    async def test_handler_error_does_not_propagate(
        self, router: MentionRouter, mock_handler: AsyncMock
    ) -> None:
        mock_handler.handle.side_effect = RuntimeError("handler exploded")

        payload = _make_payload(mentions=[Mention(id="alice-id", username="alice")])
        tools = MagicMock()
        tools.send_event = AsyncMock()

        # Should not raise
        await router.route(payload, "room-1", tools)

    async def test_mark_processing_exception_does_not_propagate(
        self, router: MentionRouter, mock_handler: AsyncMock, mock_link: AsyncMock
    ) -> None:
        mock_link.mark_processing.side_effect = ConnectionError("API down")

        payload = _make_payload(mentions=[Mention(id="alice-id", username="alice")])
        tools = MagicMock()

        # Should not raise even when mark_processing fails
        await router.route(payload, "room-1", tools)

        # Handler should still be called despite mark_processing failure
        mock_handler.handle.assert_called_once()

    async def test_mark_failed_exception_does_not_propagate(
        self, router: MentionRouter, mock_handler: AsyncMock, mock_link: AsyncMock
    ) -> None:
        mock_handler.handle.side_effect = RuntimeError("handler exploded")
        mock_link.mark_failed.side_effect = ConnectionError("API down")

        payload = _make_payload(mentions=[Mention(id="alice-id", username="alice")])
        tools = MagicMock()
        tools.send_event = AsyncMock()

        # Should not raise even when mark_failed fails
        await router.route(payload, "room-1", tools)

    async def test_mark_processed_exception_does_not_propagate(
        self, router: MentionRouter, mock_handler: AsyncMock, mock_link: AsyncMock
    ) -> None:
        mock_link.mark_processed.side_effect = ConnectionError("API down")

        payload = _make_payload(mentions=[Mention(id="alice-id", username="alice")])
        tools = MagicMock()

        # Should not raise even when mark_processed fails
        await router.route(payload, "room-1", tools)

    async def test_multi_handler_errors_aggregated(
        self, session_store: InMemorySessionStore, mock_link: AsyncMock
    ) -> None:
        handler_a = AsyncMock()
        handler_a.handle.side_effect = RuntimeError("error A")
        handler_b = AsyncMock()
        handler_b.handle.side_effect = RuntimeError("error B")

        router = MentionRouter(
            agent_mapping={"alice": "handler_a", "bob": "handler_b"},
            handlers={"handler_a": handler_a, "handler_b": handler_b},
            session_store=session_store,
            agent_id="bridge-agent-id",
            link=mock_link,
        )

        payload = _make_payload(
            mentions=[
                Mention(id="alice-id", username="alice"),
                Mention(id="bob-id", username="bob"),
            ]
        )
        tools = MagicMock()
        tools.send_event = AsyncMock()

        await router.route(payload, "room-1", tools)

        # mark_failed should contain both errors
        mark_failed_msg = mock_link.mark_failed.call_args[0][2]
        assert "'handler_a' (@alice): error A" in mark_failed_msg
        assert "'handler_b' (@bob): error B" in mark_failed_msg

        # Error event should also contain both
        event_content = tools.send_event.call_args.kwargs["content"]
        assert "handler_a" in event_content
        assert "handler_b" in event_content

    async def test_passes_sender_type(
        self, router: MentionRouter, mock_handler: AsyncMock
    ) -> None:
        payload = _make_payload(
            sender_type="Agent",
            mentions=[Mention(id="alice-id", username="alice")],
        )
        tools = MagicMock()

        await router.route(payload, "room-1", tools)

        call_kwargs = mock_handler.handle.call_args.kwargs
        assert call_kwargs["sender_type"] == "Agent"

    async def test_passes_message_id(
        self, router: MentionRouter, mock_handler: AsyncMock
    ) -> None:
        payload = _make_payload(
            msg_id="custom-msg-id",
            mentions=[Mention(id="alice-id", username="alice")],
        )
        tools = MagicMock()

        await router.route(payload, "room-1", tools)

        call_kwargs = mock_handler.handle.call_args.kwargs
        assert call_kwargs["message_id"] == "custom-msg-id"
