from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from thenvoi.adapters.codex.adapter_commands import (
    clear_pending_approval,
    clear_pending_approvals_for_room,
)
from thenvoi.adapters.codex.approval import PendingApproval


@dataclass
class _Adapter:
    _pending_approvals: dict[str, dict[str, PendingApproval]] = field(default_factory=dict)
    config: Any = None
    _client: Any = None
    _room_threads: dict[str, str] = field(default_factory=dict)
    _selected_model: str | None = None
    _model_explicitly_set: bool = False
    _task_titles_by_id: dict[str, str] = field(default_factory=dict)
    _max_task_titles: int = 10

    @staticmethod
    def _visible_model_ids(result: dict[str, Any]) -> list[str]:
        del result
        return []

    def _record_nonfatal_error(
        self,
        category: str,
        error: Exception,
        **context: Any,
    ) -> None:
        del category, error, context


def test_clear_pending_approval_removes_only_selected_token() -> None:
    loop = asyncio.new_event_loop()
    try:
        first = PendingApproval(
            request_id="1",
            method="m",
            summary="s1",
            created_at=datetime.now(timezone.utc),
            future=loop.create_future(),
        )
        second = PendingApproval(
            request_id="2",
            method="m",
            summary="s2",
            created_at=datetime.now(timezone.utc),
            future=loop.create_future(),
        )
        adapter = _Adapter(_pending_approvals={"room": {"a": first, "b": second}})
        clear_pending_approval(adapter, "room", "a")
        assert set(adapter._pending_approvals["room"].keys()) == {"b"}
    finally:
        loop.close()


def test_clear_pending_approvals_for_room_declines_all() -> None:
    loop = asyncio.new_event_loop()
    try:
        pending = PendingApproval(
            request_id="1",
            method="m",
            summary="s1",
            created_at=datetime.now(timezone.utc),
            future=loop.create_future(),
        )
        adapter = _Adapter(_pending_approvals={"room": {"token": pending}})
        clear_pending_approvals_for_room(adapter, "room")
        assert "room" not in adapter._pending_approvals
        assert pending.future.done()
        assert pending.future.result() == "decline"
    finally:
        loop.close()
