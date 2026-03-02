"""Tests for bridge @mention router."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

from thenvoi.client.streaming import (
    Mention,
    MessageCreatedPayload,
    MessageMetadata,
)

from thenvoi.integrations.a2a_bridge.router import MentionRouter
from thenvoi.integrations.a2a_bridge.handler import HandlerResult
from thenvoi.integrations.a2a_bridge.session import InMemorySessionStore

if TYPE_CHECKING:
    from thenvoi.runtime.tools import AgentTools
    from thenvoi.runtime.types import PlatformMessage


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

    def test_duplicate_username_raises(self) -> None:
        with pytest.raises(ValueError, match="Duplicate username 'alice'"):
            MentionRouter.parse_agent_mapping("alice:handler_a,alice:handler_b")


def _make_payload(
    **overrides: object,
) -> MessageCreatedPayload:
    payload_data: dict[str, object] = {
        "id": "msg-1",
        "content": "hello",
        "message_type": "text",
        "sender_id": "user-1",
        "sender_type": "User",
        "chat_room_id": "room-1",
        "thread_id": None,
        "inserted_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
    }
    mention_values: list[Mention] | None = None
    unknown_keys: list[str] = []
    for key, value in overrides.items():
        canonical_key = {"msg_id": "id", "room_id": "chat_room_id"}.get(key, key)
        if canonical_key == "mentions":
            mention_values = value if isinstance(value, list) else None
            continue
        if canonical_key not in payload_data:
            unknown_keys.append(key)
            continue
        payload_data[canonical_key] = value
    if unknown_keys:
        unknown = ", ".join(sorted(unknown_keys))
        raise TypeError(f"_make_payload() got unexpected keyword arguments: {unknown}")
    payload_data["metadata"] = MessageMetadata(mentions=mention_values or [], status="sent")
    return MessageCreatedPayload(
        **payload_data,
    )


def _called_message(handler: AsyncMock):
    """Return the normalized PlatformMessage passed to handler.handle()."""
    return handler.handle.call_args.kwargs["message"]


class RecordingHandler:
    """Concrete handler used to validate router/BaseHandler contract arguments."""

    def __init__(self) -> None:
        self.calls: list[tuple["PlatformMessage", str, "AgentTools"]] = []

    async def handle(
        self,
        message: "PlatformMessage",
        mentioned_agent: str,
        tools: "AgentTools",
    ) -> HandlerResult:
        self.calls.append((message, mentioned_agent, tools))
        return HandlerResult.handled()


class IgnoringHandler:
    """Handler that intentionally ignores routed messages."""

    async def handle(
        self,
        message: "PlatformMessage",
        mentioned_agent: str,
        tools: "AgentTools",
    ) -> HandlerResult:
        _ = message
        _ = mentioned_agent
        _ = tools
        return HandlerResult.ignored("already handled elsewhere")


class ErrorResultHandler:
    """Handler that reports errors via structured outcome contract."""

    async def handle(
        self,
        message: "PlatformMessage",
        mentioned_agent: str,
        tools: "AgentTools",
    ) -> HandlerResult:
        _ = message
        _ = mentioned_agent
        _ = tools
        return HandlerResult.error("structured failure")


class InvalidStatusHandler:
    """Handler that returns an invalid structured status."""

    async def handle(
        self,
        message: "PlatformMessage",
        mentioned_agent: str,
        tools: "AgentTools",
    ) -> HandlerResult:
        _ = message
        _ = mentioned_agent
        _ = tools
        return HandlerResult(status="invalid")  # type: ignore[arg-type]


class MalformedHandler:
    """Intentionally malformed handler used to verify contract failure behavior."""

    async def handle(
        self,
        message: "PlatformMessage",
        tools: "AgentTools",
    ) -> None:
        _ = message
        _ = tools


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
        handler = AsyncMock()
        handler.handle.return_value = HandlerResult.handled()
        return handler

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

        mock_handler.handle.assert_called_once()
        call_kwargs = mock_handler.handle.call_args.kwargs
        assert call_kwargs["mentioned_agent"] == "alice"
        assert call_kwargs["tools"] is tools

        message = _called_message(mock_handler)
        assert message.id == "msg-1"
        assert message.content == "hello"
        assert message.room_id == "room-1"
        assert message.thread_id == "room-1"
        assert message.sender_id == "user-1"
        assert message.sender_type == "User"
        assert message.sender_name is None

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
        handler_a.handle.return_value = HandlerResult.handled()
        handler_b = AsyncMock()
        handler_b.handle.return_value = HandlerResult.handled()

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

    async def test_router_refreshes_expired_session_ttl(
        self,
        mock_link: AsyncMock,
    ) -> None:
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)

        def _clock() -> datetime:
            return now

        session_store = InMemorySessionStore(session_ttl=60.0, clock=_clock)
        handler = AsyncMock()
        handler.handle.return_value = HandlerResult.handled()
        router = MentionRouter(
            agent_mapping={"alice": "alice_handler"},
            handlers={"alice_handler": handler},
            session_store=session_store,
            agent_id="bridge-agent-id",
            link=mock_link,
        )
        tools = MagicMock()

        await router.route(
            _make_payload(
                msg_id="msg-1", mentions=[Mention(id="alice-id", username="alice")]
            ),
            "room-1",
            tools,
        )
        first_session = await session_store.get("room-1")
        assert first_session is not None

        now = now + timedelta(seconds=61)

        await router.route(
            _make_payload(
                msg_id="msg-2", mentions=[Mention(id="alice-id", username="alice")]
            ),
            "room-1",
            tools,
        )
        second_session = await session_store.get("room-1")
        assert second_session is not None
        assert second_session is not first_session
        assert handler.handle.await_count == 2

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
        assert _called_message(mock_handler).thread_id == "thread-42"

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
        assert calls[0].kwargs["message"].thread_id == "thread-1"
        assert calls[1].kwargs["message"].thread_id == "thread-2"

    async def test_skips_mentions_with_none_username(
        self, router: MentionRouter, mock_handler: AsyncMock, mock_link: AsyncMock
    ) -> None:
        """Mentions with username=None should be silently skipped."""
        payload = _make_payload(
            mentions=[
                Mention(id="null-id", username=None),
                Mention(id="alice-id", username="alice"),
            ]
        )
        tools = MagicMock()

        await router.route(payload, "room-1", tools)

        # Only alice should be dispatched; the None-username mention is skipped
        mock_handler.handle.assert_called_once()
        call_kwargs = mock_handler.handle.call_args.kwargs
        assert call_kwargs["mentioned_agent"] == "alice"

    async def test_all_mentions_none_username_does_nothing(
        self, router: MentionRouter, mock_handler: AsyncMock, mock_link: AsyncMock
    ) -> None:
        """If all mentions have username=None, no handler should be dispatched."""
        payload = _make_payload(
            mentions=[
                Mention(id="null-id-1", username=None),
                Mention(id="null-id-2", username=None),
            ]
        )
        tools = MagicMock()

        await router.route(payload, "room-1", tools)

        mock_handler.handle.assert_not_called()
        mock_link.mark_processing.assert_not_called()

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
            "room-1",
            "msg-1",
            "'alice_handler' (@alice): handler exploded",
        )

    async def test_sends_sanitized_error_event_on_handler_failure(
        self, router: MentionRouter, mock_handler: AsyncMock, mock_link: AsyncMock
    ) -> None:
        """User-facing error event should not expose internal handler names."""
        mock_handler.handle.side_effect = RuntimeError("handler exploded")

        payload = _make_payload(mentions=[Mention(id="alice-id", username="alice")])
        tools = MagicMock()
        tools.send_event = AsyncMock()

        await router.route(payload, "room-1", tools)

        event_content = tools.send_event.call_args.kwargs["content"]
        assert "alice_handler" not in event_content
        assert "@alice" in event_content

    async def test_sends_error_event_on_handler_failure(
        self, router: MentionRouter, mock_handler: AsyncMock, mock_link: AsyncMock
    ) -> None:
        mock_handler.handle.side_effect = RuntimeError("handler exploded")

        payload = _make_payload(mentions=[Mention(id="alice-id", username="alice")])
        tools = MagicMock()
        tools.send_event = AsyncMock()

        await router.route(payload, "room-1", tools)

        tools.send_event.assert_called_once_with(
            content="Handler failures: @alice: processing failed",
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

        # mark_failed should contain both errors with full internal details
        mark_failed_msg = mock_link.mark_failed.call_args[0][2]
        assert "'handler_a' (@alice): error A" in mark_failed_msg
        assert "'handler_b' (@bob): error B" in mark_failed_msg

        # User-facing error event should contain @usernames but not handler names
        event_content = tools.send_event.call_args.kwargs["content"]
        assert "@alice" in event_content
        assert "@bob" in event_content
        assert "handler_a" not in event_content
        assert "handler_b" not in event_content

    async def test_partial_success_marks_processed_and_sends_error_event(
        self, session_store: InMemorySessionStore, mock_link: AsyncMock
    ) -> None:
        """When some handlers succeed and some fail, mark_processed is called (not mark_failed)."""
        handler_a = AsyncMock()  # succeeds
        handler_a.handle.return_value = HandlerResult.handled()
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

        # handler_a should have succeeded
        handler_a.handle.assert_called_once()
        # Partial success: mark_processed (not mark_failed)
        mock_link.mark_processed.assert_called_once_with("room-1", "msg-1")
        mock_link.mark_failed.assert_not_called()
        # User-facing error event should still report the failure
        tools.send_event.assert_called_once()
        event_content = tools.send_event.call_args.kwargs["content"]
        assert "@bob" in event_content

    async def test_long_error_message_truncated_in_mark_failed(
        self, session_store: InMemorySessionStore, mock_link: AsyncMock
    ) -> None:
        """Error messages longer than 500 chars should be truncated in mark_failed."""
        long_error = "x" * 1000
        handler = AsyncMock()
        handler.handle.side_effect = RuntimeError(long_error)

        router = MentionRouter(
            agent_mapping={"alice": "handler_a"},
            handlers={"handler_a": handler},
            session_store=session_store,
            agent_id="bridge-agent-id",
            link=mock_link,
        )

        payload = _make_payload(mentions=[Mention(id="alice-id", username="alice")])
        tools = MagicMock()
        tools.send_event = AsyncMock()

        await router.route(payload, "room-1", tools)

        mark_failed_msg = mock_link.mark_failed.call_args[0][2]
        # The error portion should be truncated to 500 chars
        assert len(mark_failed_msg) < 600
        assert "x" * 500 in mark_failed_msg
        assert "x" * 501 not in mark_failed_msg

    async def test_passes_sender_type(
        self, router: MentionRouter, mock_handler: AsyncMock
    ) -> None:
        payload = _make_payload(
            sender_type="Agent",
            mentions=[Mention(id="alice-id", username="alice")],
        )
        tools = MagicMock()

        await router.route(payload, "room-1", tools)

        assert _called_message(mock_handler).sender_type == "Agent"

    async def test_passes_message_id(
        self, router: MentionRouter, mock_handler: AsyncMock
    ) -> None:
        payload = _make_payload(
            msg_id="custom-msg-id",
            mentions=[Mention(id="alice-id", username="alice")],
        )
        tools = MagicMock()

        await router.route(payload, "room-1", tools)

        assert _called_message(mock_handler).id == "custom-msg-id"

    async def test_passes_sender_name(
        self, router: MentionRouter, mock_handler: AsyncMock
    ) -> None:
        payload = _make_payload(
            mentions=[Mention(id="alice-id", username="alice")],
        )
        tools = MagicMock()

        await router.route(payload, "room-1", tools, sender_name="Jane Doe")

        assert _called_message(mock_handler).sender_name == "Jane Doe"

    async def test_sender_name_defaults_to_none(
        self, router: MentionRouter, mock_handler: AsyncMock
    ) -> None:
        payload = _make_payload(
            mentions=[Mention(id="alice-id", username="alice")],
        )
        tools = MagicMock()

        await router.route(payload, "room-1", tools)

        assert _called_message(mock_handler).sender_name is None


class TestBaseHandlerContract:
    def test_handler_result_rejects_invalid_status(self) -> None:
        with pytest.raises(ValueError, match="Invalid handler status"):
            HandlerResult(status="invalid")  # type: ignore[arg-type]

    async def test_concrete_handler_receives_contract_arguments(
        self, mock_link: AsyncMock
    ) -> None:
        handler = RecordingHandler()
        router = MentionRouter(
            agent_mapping={"alice": "concrete"},
            handlers={"concrete": handler},
            session_store=InMemorySessionStore(),
            agent_id="bridge-agent-id",
            link=mock_link,
        )
        payload = _make_payload(mentions=[Mention(id="alice-id", username="alice")])
        tools = MagicMock()
        tools.send_event = AsyncMock()

        await router.route(payload, "room-1", tools)

        assert len(handler.calls) == 1
        message, mentioned_agent, received_tools = handler.calls[0]
        assert message.id == "msg-1"
        assert message.room_id == "room-1"
        assert mentioned_agent == "alice"
        assert received_tools is tools
        mock_link.mark_processing.assert_called_once_with("room-1", "msg-1")
        mock_link.mark_processed.assert_called_once_with("room-1", "msg-1")
        mock_link.mark_failed.assert_not_called()

    async def test_malformed_handler_signature_is_marked_failed(
        self, mock_link: AsyncMock
    ) -> None:
        router = MentionRouter(
            agent_mapping={"alice": "bad_handler"},
            handlers={"bad_handler": MalformedHandler()},
            session_store=InMemorySessionStore(),
            agent_id="bridge-agent-id",
            link=mock_link,
        )
        payload = _make_payload(mentions=[Mention(id="alice-id", username="alice")])
        tools = MagicMock()
        tools.send_event = AsyncMock()

        await router.route(payload, "room-1", tools)

        mock_link.mark_processing.assert_called_once_with("room-1", "msg-1")
        mock_link.mark_processed.assert_not_called()
        mock_link.mark_failed.assert_called_once()

        failed_call = mock_link.mark_failed.call_args
        assert failed_call.args[0] == "room-1"
        assert failed_call.args[1] == "msg-1"
        assert "bad_handler" in failed_call.args[2]
        assert "mentioned_agent" in failed_call.args[2]
        tools.send_event.assert_awaited_once()

    async def test_handler_result_ignored_marks_processed(
        self, mock_link: AsyncMock
    ) -> None:
        router = MentionRouter(
            agent_mapping={"alice": "ignorer"},
            handlers={"ignorer": IgnoringHandler()},
            session_store=InMemorySessionStore(),
            agent_id="bridge-agent-id",
            link=mock_link,
        )
        payload = _make_payload(mentions=[Mention(id="alice-id", username="alice")])
        tools = MagicMock()
        tools.send_event = AsyncMock()

        await router.route(payload, "room-1", tools)

        mock_link.mark_processing.assert_called_once_with("room-1", "msg-1")
        mock_link.mark_processed.assert_called_once_with("room-1", "msg-1")
        mock_link.mark_failed.assert_not_called()
        tools.send_event.assert_not_called()

    async def test_handler_result_error_marks_failed(
        self, mock_link: AsyncMock
    ) -> None:
        router = MentionRouter(
            agent_mapping={"alice": "error_handler"},
            handlers={"error_handler": ErrorResultHandler()},
            session_store=InMemorySessionStore(),
            agent_id="bridge-agent-id",
            link=mock_link,
        )
        payload = _make_payload(mentions=[Mention(id="alice-id", username="alice")])
        tools = MagicMock()
        tools.send_event = AsyncMock()

        await router.route(payload, "room-1", tools)

        mock_link.mark_processing.assert_called_once_with("room-1", "msg-1")
        mock_link.mark_processed.assert_not_called()
        mock_link.mark_failed.assert_called_once()
        failed_message = mock_link.mark_failed.call_args.args[2]
        assert "structured failure" in failed_message
        tools.send_event.assert_awaited_once()

    async def test_invalid_handler_status_marks_failed(
        self, mock_link: AsyncMock
    ) -> None:
        router = MentionRouter(
            agent_mapping={"alice": "bad_status"},
            handlers={"bad_status": InvalidStatusHandler()},
            session_store=InMemorySessionStore(),
            agent_id="bridge-agent-id",
            link=mock_link,
        )
        payload = _make_payload(mentions=[Mention(id="alice-id", username="alice")])
        tools = MagicMock()
        tools.send_event = AsyncMock()

        await router.route(payload, "room-1", tools)

        mock_link.mark_failed.assert_called_once()
        failed_message = mock_link.mark_failed.call_args.args[2]
        assert "Invalid handler status" in failed_message


class TestMentionRouterTimeout:
    """Tests for handler execution timeout."""

    @pytest.fixture
    def session_store(self) -> InMemorySessionStore:
        return InMemorySessionStore()

    @pytest.fixture
    def mock_link(self) -> AsyncMock:
        link = AsyncMock()
        link.mark_processing = AsyncMock()
        link.mark_processed = AsyncMock()
        link.mark_failed = AsyncMock()
        return link

    async def test_handler_timeout_fires(
        self, session_store: InMemorySessionStore, mock_link: AsyncMock
    ) -> None:
        """A handler that exceeds the timeout should be reported as failed."""
        handler = AsyncMock()

        async def slow_handler(**kwargs: object) -> None:
            await asyncio.sleep(10)

        handler.handle = slow_handler

        router = MentionRouter(
            agent_mapping={"alice": "handler_a"},
            handlers={"handler_a": handler},
            session_store=session_store,
            agent_id="bridge-agent-id",
            link=mock_link,
            handler_timeout=0.01,
        )

        payload = _make_payload(mentions=[Mention(id="alice-id", username="alice")])
        tools = MagicMock()
        tools.send_event = AsyncMock()

        await router.route(payload, "room-1", tools)

        # Should be marked as failed due to timeout
        mock_link.mark_failed.assert_called_once()
        mark_failed_msg = mock_link.mark_failed.call_args[0][2]
        assert "timed out" in mark_failed_msg

        # Error event should be sent to chat
        tools.send_event.assert_called_once()
        event_content = tools.send_event.call_args.kwargs["content"]
        assert "@alice" in event_content

    async def test_no_timeout_when_disabled(
        self, session_store: InMemorySessionStore, mock_link: AsyncMock
    ) -> None:
        """With handler_timeout=None, handlers run without timeout."""
        handler = AsyncMock()
        handler.handle.return_value = HandlerResult.handled()

        router = MentionRouter(
            agent_mapping={"alice": "handler_a"},
            handlers={"handler_a": handler},
            session_store=session_store,
            agent_id="bridge-agent-id",
            link=mock_link,
            handler_timeout=None,
        )

        payload = _make_payload(mentions=[Mention(id="alice-id", username="alice")])
        tools = MagicMock()

        await router.route(payload, "room-1", tools)

        handler.handle.assert_called_once()
        mock_link.mark_processed.assert_called_once()

    async def test_timeout_partial_success(
        self, session_store: InMemorySessionStore, mock_link: AsyncMock
    ) -> None:
        """When one handler times out but another succeeds, partial success applies."""
        fast_handler = AsyncMock()
        fast_handler.handle.return_value = HandlerResult.handled()

        slow_handler = AsyncMock()

        async def slow_handle(**kwargs: object) -> None:
            await asyncio.sleep(10)

        slow_handler.handle = slow_handle

        router = MentionRouter(
            agent_mapping={"alice": "handler_a", "bob": "handler_b"},
            handlers={"handler_a": fast_handler, "handler_b": slow_handler},
            session_store=session_store,
            agent_id="bridge-agent-id",
            link=mock_link,
            handler_timeout=0.01,
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

        # Partial success: mark_processed (not mark_failed)
        fast_handler.handle.assert_called_once()
        mock_link.mark_processed.assert_called_once_with("room-1", "msg-1")
        mock_link.mark_failed.assert_not_called()
        # Error event should report the timeout
        tools.send_event.assert_called_once()
        event_content = tools.send_event.call_args.kwargs["content"]
        assert "@bob" in event_content


class TestMentionRouterCancellation:
    """Tests that CancelledError propagates for shutdown cancellation."""

    @pytest.fixture
    def session_store(self) -> InMemorySessionStore:
        return InMemorySessionStore()

    @pytest.fixture
    def mock_link(self) -> AsyncMock:
        link = AsyncMock()
        link.mark_processing = AsyncMock()
        link.mark_processed = AsyncMock()
        link.mark_failed = AsyncMock()
        return link

    async def test_cancelled_handler_propagates(
        self, session_store: InMemorySessionStore, mock_link: AsyncMock
    ) -> None:
        """CancelledError in a handler should propagate out of route()."""
        handler = AsyncMock()
        handler.handle.side_effect = asyncio.CancelledError()

        router = MentionRouter(
            agent_mapping={"alice": "handler_a"},
            handlers={"handler_a": handler},
            session_store=session_store,
            agent_id="bridge-agent-id",
            link=mock_link,
        )

        payload = _make_payload(mentions=[Mention(id="alice-id", username="alice")])
        tools = MagicMock()
        tools.send_event = AsyncMock()

        with pytest.raises(asyncio.CancelledError):
            await router.route(payload, "room-1", tools)

        # Lifecycle marks should NOT be called (shutdown path)
        mock_link.mark_processed.assert_not_called()
        mock_link.mark_failed.assert_not_called()

    async def test_cancellation_cancels_sibling_handlers(
        self, session_store: InMemorySessionStore, mock_link: AsyncMock
    ) -> None:
        """When one handler is cancelled, sibling handlers are also cancelled."""
        handler_a_cancelled = False
        handler_b_started = asyncio.Event()

        async def slow_handler_a(**kwargs: object) -> None:
            nonlocal handler_a_cancelled
            try:
                await asyncio.sleep(300)
            except asyncio.CancelledError:
                handler_a_cancelled = True
                raise

        async def cancel_handler_b(**kwargs: object) -> None:
            handler_b_started.set()
            raise asyncio.CancelledError()

        handler_a = AsyncMock()
        handler_a.handle = slow_handler_a
        handler_b = AsyncMock()
        handler_b.handle = cancel_handler_b

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

        with pytest.raises(asyncio.CancelledError):
            await router.route(payload, "room-1", tools)
