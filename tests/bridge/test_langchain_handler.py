"""Tests for LangChainHandler — HTTP POST to LangChain agent endpoints."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from handlers.chain import LangChainHandler, _DEFAULT_TIMEOUT, _MAX_RESPONSE_BYTES

from .conftest import make_tools


def _make_stream_context(
    status_code: int = 200,
    json_data: dict | None = None,
    text: str = "",
    content: bytes | None = None,
    raise_for_status_error: Exception | None = None,
) -> MagicMock:
    """Create a mock async context manager for ``client.stream()``.

    Simulates httpx streaming: the returned context manager yields a
    mock response whose ``aiter_bytes`` produces the body in a single chunk.
    """
    if json_data is not None:
        body = json.dumps(json_data).encode("utf-8")
    elif content is not None:
        body = content
    else:
        body = text.encode("utf-8")

    mock_response = MagicMock()
    mock_response.status_code = status_code

    if raise_for_status_error:
        mock_response.raise_for_status.side_effect = raise_for_status_error

    async def aiter_bytes(chunk_size: int | None = None):  # noqa: ARG001
        if body:
            yield body

    mock_response.aiter_bytes = aiter_bytes

    mock_cm = MagicMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
    mock_cm.__aexit__ = AsyncMock(return_value=False)
    return mock_cm


# ---------------------------------------------------------------------------
# TestLangChainHandlerInit
# ---------------------------------------------------------------------------


class TestLangChainHandlerInit:
    def test_valid_construction_with_base_url(self) -> None:
        handler = LangChainHandler(base_url="http://localhost:8000")
        assert handler._base_url == "http://localhost:8000"
        assert handler._urls is None
        assert handler._timeout == _DEFAULT_TIMEOUT

    def test_valid_construction_with_urls(self) -> None:
        handler = LangChainHandler(
            urls={"alice": "http://localhost:8000", "bob": "http://localhost:8001"}
        )
        assert handler._base_url is None
        assert handler._urls == {
            "alice": "http://localhost:8000",
            "bob": "http://localhost:8001",
        }

    def test_custom_timeout(self) -> None:
        handler = LangChainHandler(base_url="http://localhost:8000", timeout=30.0)
        assert handler._timeout == 30.0

    def test_no_url_raises(self) -> None:
        with pytest.raises(ValueError, match="Either base_url or urls"):
            LangChainHandler()

    def test_both_url_and_urls_raises(self) -> None:
        with pytest.raises(ValueError, match="not both"):
            LangChainHandler(
                base_url="http://localhost:8000",
                urls={"alice": "http://localhost:8001"},
            )

    def test_empty_base_url_raises(self) -> None:
        with pytest.raises(ValueError, match="base_url must be a non-empty"):
            LangChainHandler(base_url="   ")

    def test_empty_urls_dict_raises(self) -> None:
        with pytest.raises(ValueError, match="urls must be a non-empty"):
            LangChainHandler(urls={})

    def test_empty_agent_name_in_urls_raises(self) -> None:
        with pytest.raises(ValueError, match="Agent name"):
            LangChainHandler(urls={"  ": "http://localhost:8000"})

    def test_empty_url_in_urls_raises(self) -> None:
        with pytest.raises(ValueError, match="URL for agent"):
            LangChainHandler(urls={"alice": "   "})

    def test_zero_timeout_raises(self) -> None:
        with pytest.raises(ValueError, match="timeout"):
            LangChainHandler(base_url="http://localhost:8000", timeout=0)

    def test_negative_timeout_raises(self) -> None:
        with pytest.raises(ValueError, match="timeout"):
            LangChainHandler(base_url="http://localhost:8000", timeout=-5)

    def test_strips_whitespace_from_base_url(self) -> None:
        handler = LangChainHandler(base_url="  http://localhost:8000  ")
        assert handler._base_url == "http://localhost:8000"

    def test_strips_whitespace_from_urls(self) -> None:
        handler = LangChainHandler(urls={"  alice  ": "  http://localhost:8000  "})
        assert handler._urls == {"alice": "http://localhost:8000"}

    def test_default_timeout(self) -> None:
        handler = LangChainHandler(base_url="http://localhost:8000")
        assert handler._timeout == 120.0

    def test_custom_max_response_bytes(self) -> None:
        handler = LangChainHandler(
            base_url="http://localhost:8000",
            max_response_bytes=2_097_152,
        )
        assert handler._max_response_bytes == 2_097_152

    def test_default_max_response_bytes(self) -> None:
        handler = LangChainHandler(base_url="http://localhost:8000")
        assert handler._max_response_bytes == _MAX_RESPONSE_BYTES

    def test_zero_max_response_bytes_raises(self) -> None:
        with pytest.raises(ValueError, match="max_response_bytes"):
            LangChainHandler(
                base_url="http://localhost:8000",
                max_response_bytes=0,
            )

    def test_negative_max_response_bytes_raises(self) -> None:
        with pytest.raises(ValueError, match="max_response_bytes"):
            LangChainHandler(
                base_url="http://localhost:8000",
                max_response_bytes=-1,
            )

    def test_owns_client_when_not_injected(self) -> None:
        handler = LangChainHandler(base_url="http://localhost:8000")
        assert handler._owns_client is True

    def test_does_not_own_injected_client(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        handler = LangChainHandler(
            base_url="http://localhost:8000",
            httpx_client=mock_client,
        )
        assert handler._owns_client is False


# ---------------------------------------------------------------------------
# TestClose
# ---------------------------------------------------------------------------


class TestClose:
    async def test_close_lazily_created_client(self) -> None:
        handler = LangChainHandler(base_url="http://localhost:8000")
        # Trigger lazy client creation
        handler._get_client()
        assert handler._httpx_client is not None

        await handler.close()
        assert handler._httpx_client is None

    async def test_close_does_not_close_injected_client(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        handler = LangChainHandler(
            base_url="http://localhost:8000",
            httpx_client=mock_client,
        )

        await handler.close()
        # Injected client should NOT be closed
        mock_client.aclose.assert_not_called()
        assert handler._httpx_client is mock_client

    async def test_close_is_idempotent(self) -> None:
        handler = LangChainHandler(base_url="http://localhost:8000")
        handler._get_client()

        await handler.close()
        await handler.close()  # Should not raise
        assert handler._httpx_client is None

    async def test_close_before_client_created(self) -> None:
        handler = LangChainHandler(base_url="http://localhost:8000")
        await handler.close()  # Should not raise
        assert handler._httpx_client is None


# ---------------------------------------------------------------------------
# TestGetClient
# ---------------------------------------------------------------------------


class TestGetClient:
    def test_returns_injected_client(self) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        handler = LangChainHandler(
            base_url="http://localhost:8000",
            httpx_client=mock_client,
        )
        assert handler._get_client() is mock_client

    def test_lazy_creates_client(self) -> None:
        handler = LangChainHandler(base_url="http://localhost:8000")
        client = handler._get_client()
        assert isinstance(client, httpx.AsyncClient)

    def test_import_error_when_httpx_missing(self) -> None:
        handler = LangChainHandler(base_url="http://localhost:8000")
        handler._httpx_client = None

        with patch.dict("sys.modules", {"httpx": None}):
            with pytest.raises(ImportError, match="httpx is required"):
                handler._get_client()


# ---------------------------------------------------------------------------
# TestResolveUrl
# ---------------------------------------------------------------------------


class TestResolveUrl:
    def test_base_url_appends_invoke(self) -> None:
        handler = LangChainHandler(base_url="http://localhost:8000")
        assert handler._resolve_url("alice") == "http://localhost:8000/invoke"

    def test_base_url_appends_invoke_any_agent(self) -> None:
        handler = LangChainHandler(base_url="http://host.docker.internal:8000")
        assert handler._resolve_url("bob") == "http://host.docker.internal:8000/invoke"

    def test_base_url_strips_trailing_slash(self) -> None:
        handler = LangChainHandler(base_url="http://localhost:8000/")
        assert handler._resolve_url("alice") == "http://localhost:8000/invoke"

    def test_urls_returns_agent_specific(self) -> None:
        handler = LangChainHandler(
            urls={
                "alice": "http://localhost:8000/invoke",
                "bob": "http://localhost:8001/invoke",
            }
        )
        assert handler._resolve_url("alice") == "http://localhost:8000/invoke"
        assert handler._resolve_url("bob") == "http://localhost:8001/invoke"

    def test_urls_unknown_agent_raises(self) -> None:
        handler = LangChainHandler(urls={"alice": "http://localhost:8000/invoke"})
        with pytest.raises(ValueError, match="No URL configured.*charlie"):
            handler._resolve_url("charlie")

    def test_urls_error_lists_known_agents(self) -> None:
        handler = LangChainHandler(
            urls={"alice": "http://a:8000", "bob": "http://b:8001"}
        )
        with pytest.raises(ValueError, match="alice.*bob"):
            handler._resolve_url("charlie")


# ---------------------------------------------------------------------------
# TestBuildPayload
# ---------------------------------------------------------------------------


class TestBuildPayload:
    def test_payload_structure(self) -> None:
        handler = LangChainHandler(base_url="http://localhost:8000")
        payload = handler._build_payload(
            content="Hello agent",
            room_id="room-123",
            thread_id="thread-456",
            message_id="msg-789",
            sender_id="user-456",
            sender_name="Alice",
            sender_type="User",
        )
        assert payload == {
            "input": "Hello agent",
            "config": {
                "configurable": {
                    "thread_id": "thread-456",
                },
            },
            "metadata": {
                "thenvoi_room_id": "room-123",
                "thenvoi_message_id": "msg-789",
                "thenvoi_sender_id": "user-456",
                "thenvoi_sender_type": "User",
                "thenvoi_sender_name": "Alice",
            },
        }

    def test_payload_omits_sender_name_when_none(self) -> None:
        handler = LangChainHandler(base_url="http://localhost:8000")
        payload = handler._build_payload(
            content="Hello",
            room_id="room-123",
            thread_id="thread-123",
            message_id="msg-1",
            sender_id="user-456",
            sender_name=None,
            sender_type="User",
        )
        assert "thenvoi_sender_name" not in payload["metadata"]

    def test_thread_id_used_for_configurable(self) -> None:
        handler = LangChainHandler(base_url="http://localhost:8000")
        payload = handler._build_payload(
            content="test",
            room_id="room-uuid-42",
            thread_id="thread-uuid-99",
            message_id="msg-1",
            sender_id="user-1",
            sender_name=None,
            sender_type="User",
        )
        assert payload["config"]["configurable"]["thread_id"] == "thread-uuid-99"
        assert payload["metadata"]["thenvoi_room_id"] == "room-uuid-42"


# ---------------------------------------------------------------------------
# TestExtractResponse
# ---------------------------------------------------------------------------


class TestExtractResponse:
    @pytest.fixture
    def handler(self) -> LangChainHandler:
        return LangChainHandler(base_url="http://localhost:8000")

    def test_extracts_output_key(self, handler: LangChainHandler) -> None:
        assert handler._extract_response({"output": "Hello"}) == "Hello"

    def test_extracts_response_key(self, handler: LangChainHandler) -> None:
        assert handler._extract_response({"response": "Reply"}) == "Reply"

    def test_extracts_text_key(self, handler: LangChainHandler) -> None:
        assert handler._extract_response({"text": "Some text"}) == "Some text"

    def test_extracts_content_key(self, handler: LangChainHandler) -> None:
        assert handler._extract_response({"content": "Content"}) == "Content"

    def test_extracts_message_key(self, handler: LangChainHandler) -> None:
        assert handler._extract_response({"message": "Msg"}) == "Msg"

    def test_priority_order_output_first(self, handler: LangChainHandler) -> None:
        result = handler._extract_response(
            {"output": "from output", "response": "from response"}
        )
        assert result == "from output"

    def test_unknown_keys_returns_json_string(self, handler: LangChainHandler) -> None:
        result = handler._extract_response({"unknown_key": "value"})
        parsed = json.loads(result)
        assert parsed == {"unknown_key": "value"}

    def test_non_string_value_converted(self, handler: LangChainHandler) -> None:
        assert handler._extract_response({"output": 42}) == "42"


# ---------------------------------------------------------------------------
# TestLangChainHandlerHandle
# ---------------------------------------------------------------------------


class TestLangChainHandlerHandle:
    @pytest.fixture
    def mock_httpx_client(self) -> AsyncMock:
        client = AsyncMock(spec=httpx.AsyncClient)
        stream_cm = _make_stream_context(json_data={"output": "Agent response"})
        client.stream = MagicMock(return_value=stream_cm)
        return client

    @pytest.fixture
    def handler(self, mock_httpx_client: AsyncMock) -> LangChainHandler:
        return LangChainHandler(
            base_url="http://localhost:8000",
            httpx_client=mock_httpx_client,
        )

    @pytest.fixture
    def tools(self) -> MagicMock:
        return make_tools(
            participants=[
                {"id": "user-1", "name": "Alice", "type": "User", "handle": "alice_h"}
            ]
        )

    async def test_successful_invocation(
        self,
        handler: LangChainHandler,
        tools: MagicMock,
        mock_httpx_client: AsyncMock,
    ) -> None:
        await handler.handle(
            content="Hello",
            room_id="room-1",
            thread_id="thread-1",
            message_id="msg-1",
            sender_id="user-1",
            sender_name="Alice",
            sender_handle="alice_h",
            sender_type="User",
            mentioned_agent="alice",
            tools=tools,
        )

        # Thought event sent
        tools.send_event.assert_called_once_with(
            content="Invoking LangChain agent for @alice...",
            message_type="thought",
        )

        # HTTP POST made via streaming
        mock_httpx_client.stream.assert_called_once()
        call_args = mock_httpx_client.stream.call_args
        assert call_args[0][0] == "POST"
        assert call_args[0][1] == "http://localhost:8000/invoke"
        payload = call_args[1]["json"]
        assert payload["input"] == "Hello"
        assert payload["config"]["configurable"]["thread_id"] == "thread-1"
        assert payload["metadata"]["thenvoi_room_id"] == "room-1"
        assert payload["metadata"]["thenvoi_message_id"] == "msg-1"
        assert payload["metadata"]["thenvoi_sender_id"] == "user-1"
        assert payload["metadata"]["thenvoi_sender_type"] == "User"
        assert payload["metadata"]["thenvoi_sender_name"] == "Alice"

        # Response sent back with sender's handle as mention
        tools.send_message.assert_called_once_with(
            content="Agent response",
            mentions=["alice_h"],
        )

    async def test_per_agent_urls(self, mock_httpx_client: AsyncMock) -> None:
        handler = LangChainHandler(
            urls={
                "alice": "http://localhost:8000/invoke",
                "bob": "http://localhost:8001/invoke",
            },
            httpx_client=mock_httpx_client,
        )
        tools = make_tools(
            participants=[
                {"id": "user-1", "name": "Alice", "type": "User", "handle": "alice_h"}
            ]
        )

        await handler.handle(
            content="Hello",
            room_id="room-1",
            thread_id="thread-1",
            message_id="msg-1",
            sender_id="user-1",
            sender_name="Alice",
            sender_handle="alice_h",
            sender_type="User",
            mentioned_agent="bob",
            tools=tools,
        )

        call_args = mock_httpx_client.stream.call_args
        assert call_args[0][1] == "http://localhost:8001/invoke"

    async def test_thought_event_failure_is_non_fatal(
        self,
        handler: LangChainHandler,
        mock_httpx_client: AsyncMock,
    ) -> None:
        tools = make_tools(
            participants=[
                {"id": "user-1", "name": "Alice", "type": "User", "handle": "alice_h"}
            ],
            send_event_side_effect=RuntimeError("event API down"),
        )

        await handler.handle(
            content="Hello",
            room_id="room-1",
            thread_id="thread-1",
            message_id="msg-1",
            sender_id="user-1",
            sender_name="Alice",
            sender_handle="alice_h",
            sender_type="User",
            mentioned_agent="alice",
            tools=tools,
        )

        mock_httpx_client.stream.assert_called_once()
        tools.send_message.assert_called_once()

    async def test_httpx_timeout_raises_timeout_error(self) -> None:
        client = AsyncMock(spec=httpx.AsyncClient)
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        client.stream = MagicMock(return_value=mock_cm)

        handler = LangChainHandler(
            base_url="http://localhost:8000",
            timeout=10.0,
            httpx_client=client,
        )
        tools = make_tools(
            participants=[
                {"id": "user-1", "name": "Alice", "type": "User", "handle": "alice_h"}
            ]
        )

        with pytest.raises(TimeoutError, match="timed out.*10.0s"):
            await handler.handle(
                content="Hello",
                room_id="room-1",
                thread_id="thread-1",
                message_id="msg-1",
                sender_id="user-1",
                sender_name="Alice",
                sender_handle="alice_h",
                sender_type="User",
                mentioned_agent="alice",
                tools=tools,
            )

    async def test_http_error_propagates(self) -> None:
        client = AsyncMock(spec=httpx.AsyncClient)
        stream_cm = _make_stream_context(
            status_code=500,
            text="Internal Server Error",
            raise_for_status_error=httpx.HTTPStatusError(
                "Server Error",
                request=httpx.Request("POST", "http://test/invoke"),
                response=httpx.Response(500),
            ),
        )
        client.stream = MagicMock(return_value=stream_cm)

        handler = LangChainHandler(
            base_url="http://localhost:8000",
            httpx_client=client,
        )
        tools = make_tools(
            participants=[
                {"id": "user-1", "name": "Alice", "type": "User", "handle": "alice_h"}
            ]
        )

        with pytest.raises(RuntimeError, match="HTTP 500"):
            await handler.handle(
                content="Hello",
                room_id="room-1",
                thread_id="thread-1",
                message_id="msg-1",
                sender_id="user-1",
                sender_name="Alice",
                sender_handle="alice_h",
                sender_type="User",
                mentioned_agent="alice",
                tools=tools,
            )

    async def test_empty_response_body_raises(self) -> None:
        client = AsyncMock(spec=httpx.AsyncClient)
        stream_cm = _make_stream_context(status_code=200, content=b"")
        client.stream = MagicMock(return_value=stream_cm)

        handler = LangChainHandler(
            base_url="http://localhost:8000",
            httpx_client=client,
        )
        tools = make_tools(
            participants=[
                {"id": "user-1", "name": "Alice", "type": "User", "handle": "alice_h"}
            ]
        )

        with pytest.raises(RuntimeError, match="empty response"):
            await handler.handle(
                content="Hello",
                room_id="room-1",
                thread_id="thread-1",
                message_id="msg-1",
                sender_id="user-1",
                sender_name="Alice",
                sender_handle="alice_h",
                sender_type="User",
                mentioned_agent="alice",
                tools=tools,
            )

    async def test_sender_not_in_participants_sends_without_mention(
        self,
        handler: LangChainHandler,
    ) -> None:
        tools = make_tools(participants=[])

        await handler.handle(
            content="Hello",
            room_id="room-1",
            thread_id="thread-1",
            message_id="msg-1",
            sender_id="user-unknown",
            sender_name=None,
            sender_handle=None,
            sender_type="User",
            mentioned_agent="alice",
            tools=tools,
        )

        tools.send_message.assert_called_once_with(
            content="Agent response",
        )

    async def test_unresolvable_sender_omits_name_from_payload(
        self,
        mock_httpx_client: AsyncMock,
    ) -> None:
        handler = LangChainHandler(
            base_url="http://localhost:8000",
            httpx_client=mock_httpx_client,
        )
        tools = make_tools(participants=[])

        await handler.handle(
            content="Hello",
            room_id="room-1",
            thread_id="thread-1",
            message_id="msg-1",
            sender_id="user-unknown",
            sender_name=None,
            sender_handle=None,
            sender_type="User",
            mentioned_agent="alice",
            tools=tools,
        )

        call_args = mock_httpx_client.stream.call_args
        payload = call_args[1]["json"]
        assert "thenvoi_sender_name" not in payload["metadata"]

    async def test_handle_preferred_over_name_for_mention(
        self,
        handler: LangChainHandler,
        mock_httpx_client: AsyncMock,
    ) -> None:
        tools = make_tools(
            participants=[
                {"id": "user-1", "name": "Alice", "type": "User", "handle": "alice_h"}
            ]
        )

        await handler.handle(
            content="Hello",
            room_id="room-1",
            thread_id="thread-1",
            message_id="msg-1",
            sender_id="user-1",
            sender_name="Alice",
            sender_handle="alice_h",
            sender_type="User",
            mentioned_agent="bob",
            tools=tools,
        )

        tools.send_message.assert_called_once_with(
            content="Agent response",
            mentions=["alice_h"],
        )

    async def test_name_used_when_handle_missing(
        self,
        handler: LangChainHandler,
        mock_httpx_client: AsyncMock,
    ) -> None:
        tools = make_tools(
            participants=[
                {"id": "user-1", "name": "Alice", "type": "User", "handle": None}
            ]
        )

        await handler.handle(
            content="Hello",
            room_id="room-1",
            thread_id="thread-1",
            message_id="msg-1",
            sender_id="user-1",
            sender_name="Alice",
            sender_handle=None,
            sender_type="User",
            mentioned_agent="bob",
            tools=tools,
        )

        tools.send_message.assert_called_once_with(
            content="Agent response",
            mentions=["Alice"],
        )

    async def test_send_message_failure_propagates(
        self,
        handler: LangChainHandler,
    ) -> None:
        tools = make_tools(
            participants=[
                {"id": "user-1", "name": "Alice", "type": "User", "handle": "alice_h"}
            ]
        )
        tools.send_message = AsyncMock(side_effect=RuntimeError("send failed"))

        with pytest.raises(RuntimeError, match="send failed"):
            await handler.handle(
                content="Hello",
                room_id="room-1",
                thread_id="thread-1",
                message_id="msg-1",
                sender_id="user-1",
                sender_name="Alice",
                sender_handle="alice_h",
                sender_type="User",
                mentioned_agent="alice",
                tools=tools,
            )

    async def test_unknown_agent_with_urls_raises(self) -> None:
        client = AsyncMock(spec=httpx.AsyncClient)
        handler = LangChainHandler(
            urls={"alice": "http://localhost:8000/invoke"},
            httpx_client=client,
        )
        tools = make_tools(
            participants=[
                {"id": "user-1", "name": "Alice", "type": "User", "handle": "alice_h"}
            ]
        )

        with pytest.raises(ValueError, match="No URL configured.*bob"):
            await handler.handle(
                content="Hello",
                room_id="room-1",
                thread_id="thread-1",
                message_id="msg-1",
                sender_id="user-1",
                sender_name="Alice",
                sender_handle="alice_h",
                sender_type="User",
                mentioned_agent="bob",
                tools=tools,
            )

    async def test_raw_text_response(self) -> None:
        client = AsyncMock(spec=httpx.AsyncClient)
        stream_cm = _make_stream_context(status_code=200, text="Plain text reply")
        client.stream = MagicMock(return_value=stream_cm)

        handler = LangChainHandler(
            base_url="http://localhost:8000",
            httpx_client=client,
        )
        tools = make_tools(
            participants=[
                {"id": "user-1", "name": "Alice", "type": "User", "handle": "alice_h"}
            ]
        )

        await handler.handle(
            content="Hello",
            room_id="room-1",
            thread_id="thread-1",
            message_id="msg-1",
            sender_id="user-1",
            sender_name="Alice",
            sender_handle="alice_h",
            sender_type="User",
            mentioned_agent="alice",
            tools=tools,
        )

        tools.send_message.assert_called_once_with(
            content="Plain text reply",
            mentions=["alice_h"],
        )

    async def test_host_docker_internal_url(self) -> None:
        client = AsyncMock(spec=httpx.AsyncClient)
        stream_cm = _make_stream_context(json_data={"output": "Docker response"})
        client.stream = MagicMock(return_value=stream_cm)

        handler = LangChainHandler(
            base_url="http://host.docker.internal:8000",
            httpx_client=client,
        )
        tools = make_tools(
            participants=[
                {"id": "user-1", "name": "Alice", "type": "User", "handle": "alice_h"}
            ]
        )

        await handler.handle(
            content="Hello",
            room_id="room-1",
            thread_id="thread-1",
            message_id="msg-1",
            sender_id="user-1",
            sender_name="Alice",
            sender_handle="alice_h",
            sender_type="User",
            mentioned_agent="alice",
            tools=tools,
        )

        call_args = client.stream.call_args
        assert call_args[0][1] == "http://host.docker.internal:8000/invoke"

    async def test_response_size_limit_exceeded_raises(self) -> None:
        client = AsyncMock(spec=httpx.AsyncClient)
        oversized_body = b"x" * (_MAX_RESPONSE_BYTES + 1)
        stream_cm = _make_stream_context(status_code=200, content=oversized_body)
        client.stream = MagicMock(return_value=stream_cm)

        handler = LangChainHandler(
            base_url="http://localhost:8000",
            httpx_client=client,
        )
        tools = make_tools(
            participants=[
                {"id": "user-1", "name": "Alice", "type": "User", "handle": "alice_h"}
            ]
        )

        with pytest.raises(RuntimeError, match="exceeds.*byte limit"):
            await handler.handle(
                content="Hello",
                room_id="room-1",
                thread_id="thread-1",
                message_id="msg-1",
                sender_id="user-1",
                sender_name="Alice",
                sender_handle="alice_h",
                sender_type="User",
                mentioned_agent="alice",
                tools=tools,
            )

    async def test_custom_response_size_limit(self) -> None:
        client = AsyncMock(spec=httpx.AsyncClient)
        stream_cm = _make_stream_context(json_data={"output": "ok"})
        client.stream = MagicMock(return_value=stream_cm)

        handler = LangChainHandler(
            base_url="http://localhost:8000",
            max_response_bytes=200,
            httpx_client=client,
        )
        tools = make_tools(
            participants=[
                {"id": "user-1", "name": "Alice", "type": "User", "handle": "alice_h"}
            ]
        )

        await handler.handle(
            content="Hello",
            room_id="room-1",
            thread_id="thread-1",
            message_id="msg-1",
            sender_id="user-1",
            sender_name="Alice",
            sender_handle="alice_h",
            sender_type="User",
            mentioned_agent="alice",
            tools=tools,
        )

        tools.send_message.assert_called_once()

    async def test_thread_id_differs_from_room_id_in_payload(
        self,
        mock_httpx_client: AsyncMock,
    ) -> None:
        handler = LangChainHandler(
            base_url="http://localhost:8000",
            httpx_client=mock_httpx_client,
        )
        tools = make_tools(
            participants=[
                {"id": "user-1", "name": "Alice", "type": "User", "handle": "alice_h"}
            ]
        )

        await handler.handle(
            content="Hello",
            room_id="room-1",
            thread_id="thread-99",
            message_id="msg-1",
            sender_id="user-1",
            sender_name="Alice",
            sender_handle="alice_h",
            sender_type="User",
            mentioned_agent="alice",
            tools=tools,
        )

        call_args = mock_httpx_client.stream.call_args
        payload = call_args[1]["json"]
        # thread_id goes to LangChain config, room_id goes to metadata
        assert payload["config"]["configurable"]["thread_id"] == "thread-99"
        assert payload["metadata"]["thenvoi_room_id"] == "room-1"

    async def test_streaming_incremental_size_check(self) -> None:
        """Size limit is enforced incrementally during streaming, not after
        the full response is buffered."""
        client = AsyncMock(spec=httpx.AsyncClient)

        # Build a mock that yields multiple chunks, exceeding the limit mid-stream
        mock_response = MagicMock()
        mock_response.status_code = 200

        chunk_size = 65_536
        limit = chunk_size * 2  # 128 KB

        async def aiter_bytes(chunk_size: int | None = None):  # noqa: ARG001
            # Yield 3 chunks — the 3rd pushes total over the limit
            yield b"x" * chunk_size if chunk_size else b"x" * 65_536
            yield b"x" * 65_536
            yield b"x" * 65_536  # Should trigger the limit

        mock_response.aiter_bytes = aiter_bytes

        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        client.stream = MagicMock(return_value=mock_cm)

        handler = LangChainHandler(
            base_url="http://localhost:8000",
            max_response_bytes=limit,
            httpx_client=client,
        )
        tools = make_tools(
            participants=[
                {"id": "user-1", "name": "Alice", "type": "User", "handle": "alice_h"}
            ]
        )

        with pytest.raises(RuntimeError, match="exceeds.*byte limit"):
            await handler.handle(
                content="Hello",
                room_id="room-1",
                thread_id="thread-1",
                message_id="msg-1",
                sender_id="user-1",
                sender_name="Alice",
                sender_handle="alice_h",
                sender_type="User",
                mentioned_agent="alice",
                tools=tools,
            )


# ---------------------------------------------------------------------------
# TestFromEnv
# ---------------------------------------------------------------------------


class TestFromEnv:
    def test_single_url(self) -> None:
        handler = LangChainHandler.from_env("http://localhost:8000/invoke")
        assert handler._base_url == "http://localhost:8000/invoke"
        assert handler._urls is None

    def test_single_url_with_whitespace(self) -> None:
        handler = LangChainHandler.from_env("  http://localhost:8000/invoke  ")
        assert handler._base_url == "http://localhost:8000/invoke"

    def test_per_agent_urls(self) -> None:
        handler = LangChainHandler.from_env(
            "alice:http://localhost:8000,bob:http://localhost:8001"
        )
        assert handler._urls == {
            "alice": "http://localhost:8000",
            "bob": "http://localhost:8001",
        }
        assert handler._base_url is None

    def test_per_agent_urls_with_whitespace(self) -> None:
        handler = LangChainHandler.from_env(
            " alice : http://localhost:8000 , bob : http://localhost:8001 "
        )
        assert handler._urls == {
            "alice": "http://localhost:8000",
            "bob": "http://localhost:8001",
        }

    def test_empty_value_raises(self) -> None:
        with pytest.raises(ValueError, match="LANGCHAIN_URLS must be non-empty"):
            LangChainHandler.from_env("")

    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(ValueError, match="LANGCHAIN_URLS must be non-empty"):
            LangChainHandler.from_env("   ")

    def test_invalid_agent_url_format_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid LANGCHAIN_URLS"):
            LangChainHandler.from_env("alice:http://a:8000,invalid_entry")

    def test_forwards_kwargs(self) -> None:
        handler = LangChainHandler.from_env(
            "http://localhost:8000/invoke",
            timeout=30.0,
        )
        assert handler._timeout == 30.0

    def test_https_url_treated_as_single(self) -> None:
        handler = LangChainHandler.from_env("https://agent.example.com/invoke")
        assert handler._base_url == "https://agent.example.com/invoke"
        assert handler._urls is None

    def test_multiple_plain_urls_raises(self) -> None:
        with pytest.raises(ValueError, match="must use.*agent:url.*format"):
            LangChainHandler.from_env("http://localhost:8000,http://localhost:8001")

    def test_single_agent_url(self) -> None:
        handler = LangChainHandler.from_env("alice:http://localhost:8000")
        assert handler._urls == {"alice": "http://localhost:8000"}

    def test_per_agent_with_port_in_url(self) -> None:
        handler = LangChainHandler.from_env(
            "alice:http://host.docker.internal:8000/invoke"
        )
        assert handler._urls == {"alice": "http://host.docker.internal:8000/invoke"}
