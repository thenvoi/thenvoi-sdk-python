"""Tests for AgentCoreHandler — AWS Bedrock AgentCore agent runtime handler."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from handlers.agentcore import AgentCoreHandler, _MAX_RESPONSE_BYTES, _READ_CHUNK_SIZE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_streaming_body(data: str | bytes) -> MagicMock:
    """Create a mock StreamingBody that returns ``data`` via chunked reads."""
    body = MagicMock()
    raw = data.encode("utf-8") if isinstance(data, str) else data
    # Simulate chunked reads: return data on first call, then b"" to signal EOF
    body.read = MagicMock(side_effect=[raw, b""])
    body.close = MagicMock()
    return body


def _make_tools(
    participants: list[dict] | None = None,
    send_event_side_effect: Exception | None = None,
) -> MagicMock:
    """Create a mock AgentTools with common defaults."""
    tools = MagicMock()
    tools.send_message = AsyncMock()
    tools.send_event = AsyncMock(side_effect=send_event_side_effect)
    tools.participants = participants or []
    return tools


# ---------------------------------------------------------------------------
# TestAgentCoreHandlerInit
# ---------------------------------------------------------------------------


class TestAgentCoreHandlerInit:
    def test_valid_construction(self) -> None:
        handler = AgentCoreHandler(
            agent_runtime_arn="arn:aws:bedrock:us-east-1:123:agent/abc",
            region="us-east-1",
        )
        assert handler._agent_runtime_arn == "arn:aws:bedrock:us-east-1:123:agent/abc"
        assert handler._region == "us-east-1"
        assert handler._timeout == 120.0
        assert handler._mcp_tool_name == "chat"

    def test_custom_timeout(self) -> None:
        handler = AgentCoreHandler(
            agent_runtime_arn="arn:abc",
            region="us-west-2",
            timeout=30.0,
        )
        assert handler._timeout == 30.0

    def test_custom_mcp_tool_name(self) -> None:
        handler = AgentCoreHandler(
            agent_runtime_arn="arn:abc",
            region="us-east-1",
            mcp_tool_name="echo",
        )
        assert handler._mcp_tool_name == "echo"

    def test_empty_arn_raises(self) -> None:
        with pytest.raises(ValueError, match="agent_runtime_arn"):
            AgentCoreHandler(agent_runtime_arn="", region="us-east-1")

    def test_whitespace_arn_raises(self) -> None:
        with pytest.raises(ValueError, match="agent_runtime_arn"):
            AgentCoreHandler(agent_runtime_arn="   ", region="us-east-1")

    def test_empty_region_raises(self) -> None:
        with pytest.raises(ValueError, match="region"):
            AgentCoreHandler(agent_runtime_arn="arn:abc", region="")

    def test_whitespace_region_raises(self) -> None:
        with pytest.raises(ValueError, match="region"):
            AgentCoreHandler(agent_runtime_arn="arn:abc", region="  ")

    def test_zero_timeout_raises(self) -> None:
        with pytest.raises(ValueError, match="timeout"):
            AgentCoreHandler(agent_runtime_arn="arn:abc", region="us-east-1", timeout=0)

    def test_negative_timeout_raises(self) -> None:
        with pytest.raises(ValueError, match="timeout"):
            AgentCoreHandler(
                agent_runtime_arn="arn:abc", region="us-east-1", timeout=-5
            )

    def test_strips_whitespace_from_arn_and_region(self) -> None:
        handler = AgentCoreHandler(
            agent_runtime_arn="  arn:abc  ",
            region="  us-east-1  ",
        )
        assert handler._agent_runtime_arn == "arn:abc"
        assert handler._region == "us-east-1"

    def test_custom_max_response_bytes(self) -> None:
        handler = AgentCoreHandler(
            agent_runtime_arn="arn:abc",
            region="us-east-1",
            max_response_bytes=2_097_152,
        )
        assert handler._max_response_bytes == 2_097_152

    def test_default_max_response_bytes(self) -> None:
        handler = AgentCoreHandler(
            agent_runtime_arn="arn:abc",
            region="us-east-1",
        )
        assert handler._max_response_bytes == _MAX_RESPONSE_BYTES

    def test_zero_max_response_bytes_raises(self) -> None:
        with pytest.raises(ValueError, match="max_response_bytes"):
            AgentCoreHandler(
                agent_runtime_arn="arn:abc",
                region="us-east-1",
                max_response_bytes=0,
            )

    def test_negative_max_response_bytes_raises(self) -> None:
        with pytest.raises(ValueError, match="max_response_bytes"):
            AgentCoreHandler(
                agent_runtime_arn="arn:abc",
                region="us-east-1",
                max_response_bytes=-1,
            )


# ---------------------------------------------------------------------------
# TestAgentCoreHandlerGetClient
# ---------------------------------------------------------------------------


class TestAgentCoreHandlerGetClient:
    def test_returns_injected_client(self) -> None:
        mock_client = MagicMock()
        handler = AgentCoreHandler(
            agent_runtime_arn="arn:abc",
            region="us-east-1",
            boto3_client=mock_client,
        )
        assert handler._get_client() is mock_client

    def test_lazy_creates_client(self) -> None:
        mock_boto3 = MagicMock()
        mock_boto3.client.return_value = MagicMock()

        handler = AgentCoreHandler(
            agent_runtime_arn="arn:abc",
            region="us-east-1",
        )

        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            client = handler._get_client()

        mock_boto3.client.assert_called_once_with(
            "bedrock-agentcore", region_name="us-east-1"
        )
        assert client is mock_boto3.client.return_value

    def test_import_error_when_boto3_missing(self) -> None:
        handler = AgentCoreHandler(
            agent_runtime_arn="arn:abc",
            region="us-east-1",
        )

        with patch.dict("sys.modules", {"boto3": None}):
            with pytest.raises(ImportError, match="boto3 is required"):
                handler._get_client()


# ---------------------------------------------------------------------------
# TestBuildPayload
# ---------------------------------------------------------------------------


class TestBuildPayload:
    def test_default_chat_tool(self) -> None:
        handler = AgentCoreHandler(
            agent_runtime_arn="arn:abc",
            region="us-east-1",
            boto3_client=MagicMock(),
        )
        payload = handler._build_payload(
            content="Hello agent",
            thread_id="thread-456",
        )
        assert payload == {
            "jsonrpc": "2.0",
            "id": "thread-456",
            "method": "tools/call",
            "params": {
                "name": "chat",
                "arguments": {"message": "Hello agent"},
            },
        }

    def test_custom_tool_name(self) -> None:
        handler = AgentCoreHandler(
            agent_runtime_arn="arn:abc",
            region="us-east-1",
            mcp_tool_name="echo",
            boto3_client=MagicMock(),
        )
        payload = handler._build_payload(
            content="Hello",
            thread_id="thread-456",
        )
        assert payload["params"]["name"] == "echo"

    def test_thread_id_used_as_jsonrpc_id(self) -> None:
        handler = AgentCoreHandler(
            agent_runtime_arn="arn:abc",
            region="us-east-1",
            boto3_client=MagicMock(),
        )
        payload = handler._build_payload(content="Hi", thread_id="t-99")
        assert payload["id"] == "t-99"


# ---------------------------------------------------------------------------
# TestReadStreamingResponse
# ---------------------------------------------------------------------------


class TestReadStreamingResponse:
    @pytest.fixture
    def handler(self) -> AgentCoreHandler:
        return AgentCoreHandler(
            agent_runtime_arn="arn:abc",
            region="us-east-1",
            boto3_client=MagicMock(),
        )

    # --- Generic JSON keys ---

    def test_extracts_output_key(self, handler: AgentCoreHandler) -> None:
        body = _make_streaming_body(json.dumps({"output": "Hello from agent"}))
        result = handler._read_streaming_response({"response": body})
        assert result == "Hello from agent"

    def test_extracts_response_key(self, handler: AgentCoreHandler) -> None:
        body = _make_streaming_body(json.dumps({"response": "Agent reply"}))
        result = handler._read_streaming_response({"response": body})
        assert result == "Agent reply"

    def test_extracts_text_key(self, handler: AgentCoreHandler) -> None:
        body = _make_streaming_body(json.dumps({"text": "Some text"}))
        result = handler._read_streaming_response({"response": body})
        assert result == "Some text"

    def test_extracts_content_key(self, handler: AgentCoreHandler) -> None:
        body = _make_streaming_body(json.dumps({"content": "Content value"}))
        result = handler._read_streaming_response({"response": body})
        assert result == "Content value"

    def test_extracts_message_key(self, handler: AgentCoreHandler) -> None:
        body = _make_streaming_body(json.dumps({"message": "Message value"}))
        result = handler._read_streaming_response({"response": body})
        assert result == "Message value"

    def test_priority_order_output_first(self, handler: AgentCoreHandler) -> None:
        body = _make_streaming_body(
            json.dumps({"output": "from output", "response": "from response"})
        )
        result = handler._read_streaming_response({"response": body})
        assert result == "from output"

    # --- MCP JSON-RPC responses ---

    def test_extracts_mcp_result_content(self, handler: AgentCoreHandler) -> None:
        mcp_response = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "content": [{"type": "text", "text": "Echo: hello"}],
                "isError": False,
            },
        }
        body = _make_streaming_body(json.dumps(mcp_response))
        result = handler._read_streaming_response({"response": body})
        assert result == "Echo: hello"

    def test_extracts_mcp_multi_content(self, handler: AgentCoreHandler) -> None:
        mcp_response = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "content": [
                    {"type": "text", "text": "Line 1"},
                    {"type": "image", "data": "..."},
                    {"type": "text", "text": "Line 2"},
                ],
            },
        }
        body = _make_streaming_body(json.dumps(mcp_response))
        result = handler._read_streaming_response({"response": body})
        assert result == "Line 1\nLine 2"

    def test_extracts_mcp_error(self, handler: AgentCoreHandler) -> None:
        mcp_response = {
            "jsonrpc": "2.0",
            "id": 1,
            "error": {"code": -32010, "message": "Runtime error"},
        }
        body = _make_streaming_body(json.dumps(mcp_response))
        result = handler._read_streaming_response({"response": body})
        assert result == "AgentCore error: Runtime error"

    # --- SSE format ---

    def test_extracts_from_sse_wrapped_mcp(self, handler: AgentCoreHandler) -> None:
        mcp_response = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "content": [{"type": "text", "text": "SSE echo"}],
            },
        }
        sse = f"event: message\ndata: {json.dumps(mcp_response)}"
        body = _make_streaming_body(sse)
        result = handler._read_streaming_response({"response": body})
        assert result == "SSE echo"

    def test_extracts_from_sse_wrapped_generic(self, handler: AgentCoreHandler) -> None:
        sse = 'event: message\ndata: {"output": "SSE output"}'
        body = _make_streaming_body(sse)
        result = handler._read_streaming_response({"response": body})
        assert result == "SSE output"

    # --- Fallbacks and edge cases ---

    def test_raw_text_fallback(self, handler: AgentCoreHandler) -> None:
        body = _make_streaming_body("Just plain text")
        result = handler._read_streaming_response({"response": body})
        assert result == "Just plain text"

    def test_json_without_known_keys_returns_raw(
        self, handler: AgentCoreHandler
    ) -> None:
        body = _make_streaming_body(json.dumps({"unknown_key": "value"}))
        result = handler._read_streaming_response({"response": body})
        assert "unknown_key" in result

    def test_missing_body_raises(self, handler: AgentCoreHandler) -> None:
        with pytest.raises(RuntimeError, match="missing 'response'"):
            handler._read_streaming_response({})

    def test_size_limit_exceeded_raises(self, handler: AgentCoreHandler) -> None:
        body = MagicMock()
        chunk = b"x" * _READ_CHUNK_SIZE
        num_full_chunks = _MAX_RESPONSE_BYTES // _READ_CHUNK_SIZE
        body.read = MagicMock(side_effect=[chunk] * (num_full_chunks + 1))
        body.close = MagicMock()

        with pytest.raises(RuntimeError, match="exceeds.*limit"):
            handler._read_streaming_response({"response": body})

        body.close.assert_called_once()

    def test_read_error_closes_body(self, handler: AgentCoreHandler) -> None:
        body = MagicMock()
        body.read = MagicMock(side_effect=IOError("read failed"))
        body.close = MagicMock()

        with pytest.raises(IOError, match="read failed"):
            handler._read_streaming_response({"response": body})

        body.close.assert_called_once()

    def test_non_string_json_value_converted(self, handler: AgentCoreHandler) -> None:
        body = _make_streaming_body(json.dumps({"output": 42}))
        result = handler._read_streaming_response({"response": body})
        assert result == "42"

    def test_body_closed_on_success(self, handler: AgentCoreHandler) -> None:
        body = _make_streaming_body(json.dumps({"output": "ok"}))
        handler._read_streaming_response({"response": body})
        body.close.assert_called_once()

    def test_invalid_utf8_raises(self, handler: AgentCoreHandler) -> None:
        body = _make_streaming_body(b"\xff\xfe invalid utf-8")
        with pytest.raises(UnicodeDecodeError):
            handler._read_streaming_response({"response": body})
        body.close.assert_called_once()

    def test_chunked_reads(self, handler: AgentCoreHandler) -> None:
        body = MagicMock()
        chunk1 = b'{"output": "'
        chunk2 = b'hello from chunks"}'
        body.read = MagicMock(side_effect=[chunk1, chunk2, b""])
        body.close = MagicMock()

        result = handler._read_streaming_response({"response": body})
        assert result == "hello from chunks"
        assert body.read.call_count == 3
        body.close.assert_called_once()

    def test_body_key_fallback(self, handler: AgentCoreHandler) -> None:
        body = _make_streaming_body(json.dumps({"output": "from body key"}))
        result = handler._read_streaming_response({"body": body})
        assert result == "from body key"


# ---------------------------------------------------------------------------
# TestResolveSender
# ---------------------------------------------------------------------------


class TestResolveSender:
    @pytest.fixture
    def handler(self) -> AgentCoreHandler:
        return AgentCoreHandler(
            agent_runtime_arn="arn:abc",
            region="us-east-1",
            boto3_client=MagicMock(),
        )

    def test_found_returns_name_and_handle(self, handler: AgentCoreHandler) -> None:
        tools = _make_tools(
            participants=[
                {"id": "user-1", "name": "Alice", "type": "User", "handle": "alice"},
                {"id": "user-2", "name": "Bob", "type": "User", "handle": "bob"},
            ]
        )
        name, handle = handler._resolve_sender("user-1", tools)
        assert name == "Alice"
        assert handle == "alice"

    def test_found_without_handle(self, handler: AgentCoreHandler) -> None:
        tools = _make_tools(
            participants=[
                {"id": "user-1", "name": "Alice", "type": "User", "handle": None},
            ]
        )
        name, handle = handler._resolve_sender("user-1", tools)
        assert name == "Alice"
        assert handle is None

    def test_not_found_returns_none(self, handler: AgentCoreHandler) -> None:
        tools = _make_tools(
            participants=[
                {"id": "user-2", "name": "Bob", "type": "User", "handle": "bob"}
            ]
        )
        name, handle = handler._resolve_sender("user-1", tools)
        assert name is None
        assert handle is None

    def test_empty_participants_returns_none(self, handler: AgentCoreHandler) -> None:
        tools = _make_tools(participants=[])
        name, handle = handler._resolve_sender("user-1", tools)
        assert name is None
        assert handle is None


# ---------------------------------------------------------------------------
# TestAgentCoreHandlerHandle
# ---------------------------------------------------------------------------


class TestAgentCoreHandlerHandle:
    @pytest.fixture
    def mock_boto3_client(self) -> MagicMock:
        client = MagicMock()
        body = _make_streaming_body(json.dumps({"output": "Agent response"}))
        client.invoke_agent_runtime.return_value = {"response": body}
        return client

    @pytest.fixture
    def handler(self, mock_boto3_client: MagicMock) -> AgentCoreHandler:
        return AgentCoreHandler(
            agent_runtime_arn="arn:abc",
            region="us-east-1",
            boto3_client=mock_boto3_client,
        )

    @pytest.fixture
    def tools(self) -> MagicMock:
        return _make_tools(
            participants=[
                {"id": "user-1", "name": "Alice", "type": "User", "handle": "alice_h"}
            ]
        )

    async def test_successful_invocation(
        self,
        handler: AgentCoreHandler,
        tools: MagicMock,
        mock_boto3_client: MagicMock,
    ) -> None:
        await handler.handle(
            content="Hello",
            room_id="room-1",
            thread_id="thread-1",
            message_id="msg-1",
            sender_id="user-1",
            sender_name="Alice",
            sender_type="User",
            mentioned_agent="alice",
            tools=tools,
        )

        # Thought event sent
        tools.send_event.assert_called_once_with(
            content="Invoking AgentCore for @alice...",
            message_type="thought",
        )

        # boto3 client invoked with correct API params
        mock_boto3_client.invoke_agent_runtime.assert_called_once()
        call_kwargs = mock_boto3_client.invoke_agent_runtime.call_args.kwargs
        assert call_kwargs["agentRuntimeArn"] == "arn:abc"
        assert call_kwargs["runtimeSessionId"] == "thread-1"
        assert call_kwargs["contentType"] == "application/json"
        assert isinstance(call_kwargs["payload"], bytes)

        # Payload is MCP tools/call with default "chat" tool
        payload_data = json.loads(call_kwargs["payload"])
        assert payload_data["method"] == "tools/call"
        assert payload_data["params"]["name"] == "chat"
        assert payload_data["params"]["arguments"]["message"] == "Hello"

        # Response sent back with sender's handle as mention
        tools.send_message.assert_called_once_with(
            content="Agent response",
            mentions=["alice_h"],
        )

    async def test_thought_event_failure_is_non_fatal(
        self,
        handler: AgentCoreHandler,
        mock_boto3_client: MagicMock,
    ) -> None:
        tools = _make_tools(
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
            sender_type="User",
            mentioned_agent="alice",
            tools=tools,
        )

        mock_boto3_client.invoke_agent_runtime.assert_called_once()
        tools.send_message.assert_called_once()

    async def test_timeout_raises_timeout_error(self) -> None:
        client = MagicMock()

        handler = AgentCoreHandler(
            agent_runtime_arn="arn:abc",
            region="us-east-1",
            timeout=0.05,
            boto3_client=client,
        )

        import time

        client.invoke_agent_runtime = MagicMock(side_effect=lambda **kw: time.sleep(5))

        tools = _make_tools(
            participants=[
                {"id": "user-1", "name": "Alice", "type": "User", "handle": "alice_h"}
            ]
        )

        with pytest.raises(TimeoutError, match="timed out"):
            await handler.handle(
                content="Hello",
                room_id="room-1",
                thread_id="thread-1",
                message_id="msg-1",
                sender_id="user-1",
                sender_name="Alice",
                sender_type="User",
                mentioned_agent="alice",
                tools=tools,
            )

    async def test_empty_response_raises(self, mock_boto3_client: MagicMock) -> None:
        body = _make_streaming_body("")
        mock_boto3_client.invoke_agent_runtime.return_value = {"response": body}

        handler = AgentCoreHandler(
            agent_runtime_arn="arn:abc",
            region="us-east-1",
            boto3_client=mock_boto3_client,
        )
        tools = _make_tools(
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
                sender_type="User",
                mentioned_agent="alice",
                tools=tools,
            )

    async def test_whitespace_only_response_raises(
        self, mock_boto3_client: MagicMock
    ) -> None:
        body = _make_streaming_body("   \n  ")
        mock_boto3_client.invoke_agent_runtime.return_value = {"response": body}

        handler = AgentCoreHandler(
            agent_runtime_arn="arn:abc",
            region="us-east-1",
            boto3_client=mock_boto3_client,
        )
        tools = _make_tools(
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
                sender_type="User",
                mentioned_agent="alice",
                tools=tools,
            )

    async def test_boto3_error_propagates(self, mock_boto3_client: MagicMock) -> None:
        mock_boto3_client.invoke_agent_runtime.side_effect = Exception("AWS error")

        handler = AgentCoreHandler(
            agent_runtime_arn="arn:abc",
            region="us-east-1",
            boto3_client=mock_boto3_client,
        )
        tools = _make_tools(
            participants=[
                {"id": "user-1", "name": "Alice", "type": "User", "handle": "alice_h"}
            ]
        )

        with pytest.raises(Exception, match="AWS error"):
            await handler.handle(
                content="Hello",
                room_id="room-1",
                thread_id="thread-1",
                message_id="msg-1",
                sender_id="user-1",
                sender_name="Alice",
                sender_type="User",
                mentioned_agent="alice",
                tools=tools,
            )

    async def test_send_message_failure_propagates(
        self,
        handler: AgentCoreHandler,
    ) -> None:
        tools = _make_tools(
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
                sender_type="User",
                mentioned_agent="alice",
                tools=tools,
            )

    async def test_sender_not_in_participants_sends_without_mention(
        self,
        handler: AgentCoreHandler,
    ) -> None:
        tools = _make_tools(participants=[])

        await handler.handle(
            content="Hello",
            room_id="room-1",
            thread_id="thread-1",
            message_id="msg-1",
            sender_id="user-unknown",
            sender_name=None,
            sender_type="User",
            mentioned_agent="alice",
            tools=tools,
        )

        tools.send_message.assert_called_once_with(
            content="Agent response",
        )

    async def test_handle_preferred_over_name_for_mention(
        self,
        handler: AgentCoreHandler,
        mock_boto3_client: MagicMock,
    ) -> None:
        tools = _make_tools(
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
        handler: AgentCoreHandler,
        mock_boto3_client: MagicMock,
    ) -> None:
        tools = _make_tools(
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
            sender_type="User",
            mentioned_agent="bob",
            tools=tools,
        )

        tools.send_message.assert_called_once_with(
            content="Agent response",
            mentions=["Alice"],
        )

    async def test_session_id_is_thread_id(
        self,
        handler: AgentCoreHandler,
        tools: MagicMock,
        mock_boto3_client: MagicMock,
    ) -> None:
        await handler.handle(
            content="Hello",
            room_id="room-1",
            thread_id="thread-99",
            message_id="msg-1",
            sender_id="user-1",
            sender_name="Alice",
            sender_type="User",
            mentioned_agent="alice",
            tools=tools,
        )

        call_kwargs = mock_boto3_client.invoke_agent_runtime.call_args.kwargs
        assert call_kwargs["runtimeSessionId"] == "thread-99"

    async def test_custom_mcp_tool_in_payload(
        self,
        mock_boto3_client: MagicMock,
    ) -> None:
        handler = AgentCoreHandler(
            agent_runtime_arn="arn:abc",
            region="us-east-1",
            mcp_tool_name="echo",
            boto3_client=mock_boto3_client,
        )
        tools = _make_tools(
            participants=[
                {"id": "user-1", "name": "Alice", "type": "User", "handle": "alice_h"}
            ]
        )

        await handler.handle(
            content="test message",
            room_id="room-1",
            thread_id="thread-1",
            message_id="msg-1",
            sender_id="user-1",
            sender_name="Alice",
            sender_type="User",
            mentioned_agent="echo_agent",
            tools=tools,
        )

        call_kwargs = mock_boto3_client.invoke_agent_runtime.call_args.kwargs
        payload_data = json.loads(call_kwargs["payload"])
        assert payload_data["params"]["name"] == "echo"
        assert payload_data["params"]["arguments"]["message"] == "test message"

    async def test_mcp_response_parsed_end_to_end(
        self,
        mock_boto3_client: MagicMock,
    ) -> None:
        """Full handle() flow with MCP JSON-RPC response in SSE format."""
        mcp_response = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "content": [{"type": "text", "text": "I'm doing well, thanks!"}],
                "isError": False,
            },
        }
        sse = f"event: message\ndata: {json.dumps(mcp_response)}"
        body = _make_streaming_body(sse)
        mock_boto3_client.invoke_agent_runtime.return_value = {"response": body}

        handler = AgentCoreHandler(
            agent_runtime_arn="arn:abc",
            region="us-east-1",
            boto3_client=mock_boto3_client,
        )
        tools = _make_tools(
            participants=[
                {"id": "user-1", "name": "Alice", "type": "User", "handle": "alice_h"}
            ]
        )

        await handler.handle(
            content="How are you?",
            room_id="room-1",
            thread_id="thread-1",
            message_id="msg-1",
            sender_id="user-1",
            sender_name="Alice",
            sender_type="User",
            mentioned_agent="llm_agent",
            tools=tools,
        )

        tools.send_message.assert_called_once_with(
            content="I'm doing well, thanks!",
            mentions=["alice_h"],
        )
