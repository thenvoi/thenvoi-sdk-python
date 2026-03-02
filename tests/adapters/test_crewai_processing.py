"""Tests for CrewAI message-processing helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from thenvoi.adapters.crewai_processing import build_backstory, process_message


@dataclass
class _FakeMessage:
    id: str = "msg-1"
    text: str = "hello world"

    def format_for_llm(self) -> str:
        return self.text


@dataclass
class _FakeAgent:
    responses: list[str]
    calls: list[list[dict[str, str]]] = field(default_factory=list)

    async def kickoff_async(self, messages: list[dict[str, str]]) -> Any:
        self.calls.append(messages)
        return SimpleNamespace(raw=self.responses.pop(0) if self.responses else "")


@dataclass
class _FakeAdapter:
    _crewai_agent: _FakeAgent
    _message_history: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    reported_errors: list[str] = field(default_factory=list)

    def stage_room_history(
        self,
        history_by_room: dict[str, list[dict[str, Any]]],
        *,
        room_id: str,
        is_session_bootstrap: bool,
        hydrated_history: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        if is_session_bootstrap and hydrated_history is not None:
            history_by_room[room_id] = list(hydrated_history)
        return history_by_room.setdefault(room_id, [])

    def apply_metadata_updates(
        self,
        messages: list[dict[str, str]],
        *,
        participants_msg: str | None,
        contacts_msg: str | None,
        make_entry: Any,
    ) -> int:
        count = 0
        for update in (participants_msg, contacts_msg):
            if update:
                messages.append(make_entry(update))
                count += 1
        return count

    async def report_adapter_error(
        self,
        tools: Any,
        *,
        error: Exception,
        operation: str,
    ) -> None:
        del tools
        self.reported_errors.append(f"{operation}:{error}")


def test_build_backstory_includes_default_identity_and_custom_section() -> None:
    rendered = build_backstory(
        agent_name="Helper",
        backstory=None,
        custom_section="Custom",
        platform_instructions="Platform",
    )
    assert "You are Helper" in rendered
    assert "Custom" in rendered
    assert rendered.endswith("Platform")


@pytest.mark.asyncio
async def test_process_message_bootstrap_injects_history_and_updates() -> None:
    adapter = _FakeAdapter(_crewai_agent=_FakeAgent(responses=["assistant reply"]))
    msg = _FakeMessage(text="[user]: ping")
    tools = object()

    await process_message(
        adapter,
        msg=msg,
        tools=tools,
        history=[{"role": "assistant", "content": "previous"}],
        participants_msg="participants update",
        contacts_msg="contacts update",
        is_session_bootstrap=True,
        room_id="room-1",
    )

    sent_messages = adapter._crewai_agent.calls[0]
    assert sent_messages[0]["content"].startswith("[Previous conversation:]")
    assert any(m["content"] == "participants update" for m in sent_messages)
    assert any(m["content"] == "contacts update" for m in sent_messages)
    assert adapter._message_history["room-1"][-1]["content"] == "assistant reply"

