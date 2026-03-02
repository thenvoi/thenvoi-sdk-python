"""Direct contract tests for bridge handler outcomes and compatibility paths."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pytest

from thenvoi.integrations.a2a_bridge.handler import HandlerResult
from thenvoi.integrations.a2a_bridge.route_dispatch import (
    DispatchTarget,
    _normalize_handler_result,
    execute_dispatch_targets,
)
from thenvoi.runtime.types import PlatformMessage


def _message() -> PlatformMessage:
    return PlatformMessage(
        id="msg-1",
        room_id="room-1",
        thread_id="room-1",
        content="hello",
        sender_id="user-1",
        sender_type="User",
        sender_name="Alice",
        message_type="text",
        metadata={},
        created_at=datetime.now(timezone.utc),
    )


def test_handler_result_defaults_to_handled() -> None:
    result = HandlerResult()
    assert result.status == "handled"
    assert result.detail is None


def test_handler_result_factories_match_constructor_contract() -> None:
    assert HandlerResult.handled("ok") == HandlerResult(status="handled", detail="ok")
    assert HandlerResult.ignored("skip") == HandlerResult(status="ignored", detail="skip")
    assert HandlerResult.error("boom") == HandlerResult(status="error", detail="boom")


def test_handler_result_rejects_invalid_status() -> None:
    with pytest.raises(ValueError, match="Invalid handler status"):
        HandlerResult(status="unknown")  # type: ignore[arg-type]


def test_normalize_handler_result_rejects_none_legacy_path() -> None:
    with pytest.raises(TypeError, match="must return HandlerResult, got None"):
        _normalize_handler_result(None)


def test_normalize_handler_result_rejects_non_contract_payload() -> None:
    with pytest.raises(TypeError, match="must return HandlerResult"):
        _normalize_handler_result({"legacy": True})


@dataclass
class _LegacyNoneHandler:
    async def handle(
        self,
        message: PlatformMessage,
        mentioned_agent: str,
        tools: Any,
    ) -> None:
        _ = message
        _ = mentioned_agent
        _ = tools
        return None


@dataclass
class _LegacyPayloadHandler:
    async def handle(
        self,
        message: PlatformMessage,
        mentioned_agent: str,
        tools: Any,
    ) -> dict[str, str]:
        _ = message
        _ = mentioned_agent
        _ = tools
        return {"status": "legacy"}


@pytest.mark.asyncio
async def test_execute_dispatch_targets_treats_none_as_failure() -> None:
    targets = [
        DispatchTarget(
            username="alice",
            handler_name="legacy_none",
            handler=_LegacyNoneHandler(),
        )
    ]

    failures = await execute_dispatch_targets(
        targets,
        platform_message=_message(),
        tools=object(),
        room_id="room-1",
        handler_timeout=None,
        logger=logging.getLogger(__name__),
    )
    assert len(failures) == 1
    assert failures[0].handler_name == "legacy_none"
    assert "must return HandlerResult" in str(failures[0].error)


@pytest.mark.asyncio
async def test_execute_dispatch_targets_rejects_legacy_payload() -> None:
    targets = [
        DispatchTarget(
            username="alice",
            handler_name="legacy_payload",
            handler=_LegacyPayloadHandler(),
        )
    ]

    failures = await execute_dispatch_targets(
        targets,
        platform_message=_message(),
        tools=object(),
        room_id="room-1",
        handler_timeout=None,
        logger=logging.getLogger(__name__),
    )
    assert len(failures) == 1
    assert failures[0].handler_name == "legacy_payload"
    assert "must return HandlerResult" in str(failures[0].error)
