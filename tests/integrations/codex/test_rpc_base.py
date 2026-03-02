"""Direct tests for shared Codex JSON-RPC base client helpers."""

from __future__ import annotations

import asyncio
import json
from typing import Any, cast

import pytest

from thenvoi.integrations.codex.rpc_base import BaseJsonRpcClient, RpcEvent


class _DummyRpcClient(BaseJsonRpcClient):
    def __init__(self) -> None:
        super().__init__()
        self.sent_payloads: list[dict[str, Any]] = []

    async def connect(self) -> None:
        self._connected = True

    async def close(self) -> None:
        self._connected = False

    async def _send_json(self, payload: dict[str, Any]) -> None:
        self.sent_payloads.append(payload)


@pytest.mark.asyncio
async def test_fail_pending_records_drop_when_event_queue_is_full() -> None:
    client = _DummyRpcClient()
    client._events = asyncio.Queue(maxsize=1)
    client._events.put_nowait(
        RpcEvent(
            kind="notification",
            method="already/queued",
            params={},
            id=None,
            raw={"method": "already/queued", "params": {}},
        )
    )

    loop = asyncio.get_running_loop()
    pending_future: asyncio.Future[dict[str, Any]] = loop.create_future()
    client._pending[1] = pending_future

    client._fail_pending("transport down")

    assert pending_future.done() is True
    with pytest.raises(RuntimeError, match="transport down"):
        pending_future.result()
    assert client.dropped_events
    assert client.dropped_events[0]["kind"] == "transport/closed"


@pytest.mark.asyncio
async def test_dispatch_request_records_drop_when_queue_is_full() -> None:
    client = _DummyRpcClient()
    client._events = asyncio.Queue(maxsize=1)
    client._events.put_nowait(
        RpcEvent(
            kind="notification",
            method="already/queued",
            params={},
            id=None,
            raw={"method": "already/queued", "params": {}},
        )
    )

    await client._dispatch_rpc_message(
        json.dumps({"id": "srv-1", "method": "item/tool/call", "params": {"a": 1}})
    )

    assert client.dropped_events
    assert client.dropped_events[0]["kind"] == "request"
    assert client.dropped_events[0]["method"] == "item/tool/call"


@pytest.mark.asyncio
async def test_dispatch_notification_records_drop_when_queue_is_full() -> None:
    client = _DummyRpcClient()
    client._events = asyncio.Queue(maxsize=1)
    client._events.put_nowait(
        RpcEvent(
            kind="notification",
            method="already/queued",
            params={},
            id=None,
            raw={"method": "already/queued", "params": {}},
        )
    )

    await client._dispatch_rpc_message(
        json.dumps({"method": "turn/completed", "params": {"turn": {"id": "t-1"}}})
    )

    assert client.dropped_events
    assert client.dropped_events[0]["kind"] == "notification"
    assert client.dropped_events[0]["method"] == "turn/completed"


@pytest.mark.asyncio
async def test_dispatch_response_resolves_pending_future() -> None:
    client = _DummyRpcClient()
    loop = asyncio.get_running_loop()
    pending_future: asyncio.Future[dict[str, Any]] = loop.create_future()
    client._pending[7] = pending_future

    await client._dispatch_rpc_message(json.dumps({"id": 7, "result": {"ok": True}}))

    assert pending_future.done() is True
    result = cast(dict[str, Any], pending_future.result())
    assert result["result"]["ok"] is True
