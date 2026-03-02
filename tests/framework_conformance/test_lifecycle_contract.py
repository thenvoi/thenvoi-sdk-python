"""Shared lifecycle contract tests for framework adapters."""

from __future__ import annotations

import json
from typing import Any

import pytest


class _RecordingTools:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def send_event(
        self,
        content: str,
        message_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "content": content,
                "message_type": message_type,
                "metadata": metadata,
            }
        )
        return {"status": "sent"}


class _FailingTools:
    async def send_event(
        self,
        content: str,
        message_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del content, message_type, metadata
        raise RuntimeError("send_event failed")


def test_metadata_update_contract(adapter_config) -> None:
    """All adapters use canonical metadata formatting and order."""
    adapter = adapter_config.adapter_factory()
    updates = adapter.build_metadata_updates(
        participants_msg="Alice joined",
        contacts_msg="Connected to @bob",
    )
    assert updates == [
        "[System]: Alice joined",
        "[System]: Connected to @bob",
    ]


@pytest.mark.asyncio
async def test_tool_event_contract(adapter_config) -> None:
    """All adapters share a consistent tool event payload path."""
    adapter = adapter_config.adapter_factory()
    tools = _RecordingTools()

    sent = await adapter.send_tool_call_event(
        tools,
        payload={
            "name": "thenvoi_send_message",
            "args": {"content": "hello"},
            "tool_call_id": "call-1",
        },
    )

    assert sent is True
    assert tools.calls
    call = tools.calls[0]
    assert call["message_type"] == "tool_call"
    assert json.loads(call["content"]) == {
        "name": "thenvoi_send_message",
        "args": {"content": "hello"},
        "tool_call_id": "call-1",
    }


@pytest.mark.asyncio
async def test_error_reporting_is_best_effort(adapter_config) -> None:
    """All adapters swallow send_event failures during error reporting."""
    adapter = adapter_config.adapter_factory()
    sent = await adapter.send_error_event(_FailingTools(), error="boom")
    assert sent is False
