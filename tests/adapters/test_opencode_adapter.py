"""Tests for OpencodeAdapter."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any, cast
from uuid import uuid4

import httpx
import pytest

from thenvoi.adapters.opencode import OpencodeAdapter, OpencodeAdapterConfig
from thenvoi.core.protocols import AgentToolsProtocol
from thenvoi.core.types import PlatformMessage
from thenvoi.integrations.opencode.types import OpencodeSessionState
from thenvoi.testing import FakeAgentTools


def make_platform_message(
    room_id: str = "room-1",
    content: str = "hello",
    sender_id: str = "user-1",
    sender_name: str = "Alice",
) -> PlatformMessage:
    return PlatformMessage(
        id=str(uuid4()),
        room_id=room_id,
        content=content,
        sender_id=sender_id,
        sender_type="User",
        sender_name=sender_name,
        message_type="text",
        metadata={},
        created_at=datetime.now(),
    )


def event_message_updated(session_id: str, message_id: str) -> dict[str, Any]:
    return {
        "type": "message.updated",
        "properties": {
            "info": {
                "id": message_id,
                "sessionID": session_id,
                "role": "assistant",
            }
        },
    }


def event_text_part(session_id: str, message_id: str, text: str) -> dict[str, Any]:
    return {
        "type": "message.part.updated",
        "properties": {
            "part": {
                "id": f"part-{message_id}",
                "sessionID": session_id,
                "messageID": message_id,
                "type": "text",
                "text": text,
            }
        },
    }


def event_reasoning_part(
    session_id: str, message_id: str, part_id: str = "reasoning-part"
) -> dict[str, Any]:
    return {
        "type": "message.part.updated",
        "properties": {
            "part": {
                "id": part_id,
                "sessionID": session_id,
                "messageID": message_id,
                "type": "reasoning",
                "text": "",
            }
        },
    }


def event_part_delta(
    session_id: str, message_id: str, part_id: str, delta: str
) -> dict[str, Any]:
    return {
        "type": "message.part.delta",
        "properties": {
            "sessionID": session_id,
            "messageID": message_id,
            "partID": part_id,
            "field": "text",
            "delta": delta,
        },
    }


def event_user_message_updated(session_id: str, message_id: str) -> dict[str, Any]:
    return {
        "type": "message.updated",
        "properties": {
            "info": {
                "id": message_id,
                "sessionID": session_id,
                "role": "user",
            }
        },
    }


def event_tool_part(
    session_id: str,
    message_id: str,
    *,
    tool: str,
    call_id: str,
    status: str,
    input_data: dict[str, Any],
    output: str | None = None,
) -> dict[str, Any]:
    state: dict[str, Any] = {"status": status, "input": input_data}
    if status == "running":
        state["time"] = {"start": 1}
    if status == "completed":
        state["output"] = output or ""
        state["title"] = tool
        state["metadata"] = {}
        state["time"] = {"start": 1, "end": 2}

    return {
        "type": "message.part.updated",
        "properties": {
            "part": {
                "id": f"tool-{call_id}-{status}",
                "sessionID": session_id,
                "messageID": message_id,
                "type": "tool",
                "tool": tool,
                "callID": call_id,
                "state": state,
            }
        },
    }


def event_permission(session_id: str, request_id: str) -> dict[str, Any]:
    return {
        "type": "permission.asked",
        "properties": {
            "id": request_id,
            "sessionID": session_id,
            "permission": "bash",
            "patterns": ["rm -rf tmp"],
        },
    }


def event_question(session_id: str, request_id: str, *questions: str) -> dict[str, Any]:
    return {
        "type": "question.asked",
        "properties": {
            "id": request_id,
            "sessionID": session_id,
            "questions": [{"question": question} for question in questions],
        },
    }


def event_session_idle(session_id: str) -> dict[str, Any]:
    return {"type": "session.idle", "properties": {"sessionID": session_id}}


def event_session_error(session_id: str, message: str) -> dict[str, Any]:
    return {
        "type": "session.error",
        "properties": {
            "sessionID": session_id,
            "error": {"name": "APIError", "data": {"message": message}},
        },
    }


def tools_protocol(tools: FakeAgentTools) -> AgentToolsProtocol:
    return cast(AgentToolsProtocol, tools)


class FakeOpencodeClient:
    def __init__(
        self,
        *,
        prompt_event_sequences: list[list[dict[str, Any]]] | None = None,
        reply_permission_events: dict[str, list[dict[str, Any]]] | None = None,
        reply_question_events: dict[str, list[dict[str, Any]]] | None = None,
        get_session_missing: set[str] | None = None,
        prompt_exceptions: list[Exception] | None = None,
    ) -> None:
        self.created_sessions: list[dict[str, Any]] = []
        self.prompt_calls: list[dict[str, Any]] = []
        self.permission_replies: list[dict[str, Any]] = []
        self.question_replies: list[dict[str, Any]] = []
        self.question_rejections: list[str] = []
        self.aborted_sessions: list[str] = []
        self.closed = False
        self._session_counter = 0
        self._queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self._prompt_event_sequences = list(prompt_event_sequences or [])
        self._reply_permission_events = reply_permission_events or {}
        self._reply_question_events = reply_question_events or {}
        self._get_session_missing = get_session_missing or set()
        self._prompt_exceptions = list(prompt_exceptions or [])

    async def create_session(
        self,
        *,
        title: str | None = None,
        permission: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        self._session_counter += 1
        session = {
            "id": f"sess-{self._session_counter}",
            "title": title or "",
            "permission": permission or [],
        }
        self.created_sessions.append(session)
        return session

    async def get_session(self, session_id: str) -> dict[str, Any]:
        if session_id in self._get_session_missing:
            request = AnyHTTPStatusError(404, session_id)
            raise request
        return {"id": session_id, "title": "existing"}

    async def prompt_async(
        self,
        session_id: str,
        *,
        parts: list[dict[str, Any]],
        system: str | None = None,
        model: dict[str, str] | None = None,
        agent: str | None = None,
        variant: str | None = None,
    ) -> None:
        self.prompt_calls.append(
            {
                "session_id": session_id,
                "parts": parts,
                "system": system,
                "model": model,
                "agent": agent,
                "variant": variant,
            }
        )
        if self._prompt_exceptions:
            raise self._prompt_exceptions.pop(0)
        if self._prompt_event_sequences:
            for event in self._prompt_event_sequences.pop(0):
                await self._queue.put(event)

    async def reply_permission(
        self,
        request_id: str,
        *,
        reply: str,
        message: str | None = None,
    ) -> None:
        self.permission_replies.append(
            {"request_id": request_id, "reply": reply, "message": message}
        )
        for event in self._reply_permission_events.get(request_id, []):
            await self._queue.put(event)

    async def reply_question(
        self, request_id: str, *, answers: list[list[str]]
    ) -> None:
        self.question_replies.append({"request_id": request_id, "answers": answers})
        for event in self._reply_question_events.get(request_id, []):
            await self._queue.put(event)

    async def reject_question(self, request_id: str) -> None:
        self.question_rejections.append(request_id)

    async def abort_session(self, session_id: str) -> None:
        self.aborted_sessions.append(session_id)

    async def iter_events(self):
        while True:
            event = await self._queue.get()
            if event is None:
                return
            yield event

    async def close(self) -> None:
        self.closed = True
        await self._queue.put(None)


class AnyHTTPStatusError(httpx.HTTPStatusError):
    def __init__(self, status_code: int, session_id: str) -> None:
        request = httpx.Request("GET", f"http://localhost/session/{session_id}")
        response = httpx.Response(status_code=status_code, request=request)
        super().__init__("status error", request=request, response=response)


async def wait_for(predicate, timeout_s: float = 1.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_s
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    pytest.fail("Timed out waiting for condition")


class TestOpencodeAdapter:
    @pytest.mark.asyncio
    async def test_bootstrap_creates_session_relays_text_and_persists_task(
        self,
    ) -> None:
        fake_client = FakeOpencodeClient(
            prompt_event_sequences=[
                [
                    event_message_updated("sess-1", "msg-1"),
                    event_text_part("sess-1", "msg-1", "OpenCode says hi"),
                    event_session_idle("sess-1"),
                ]
            ]
        )
        adapter = OpencodeAdapter(client_factory=lambda _config: fake_client)
        tools = FakeAgentTools()

        await adapter.on_started("OpenCode Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(),
            tools_protocol(tools),
            OpencodeSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        assert fake_client.created_sessions[0]["id"] == "sess-1"
        assert tools.messages_sent[0]["content"] == "OpenCode says hi"
        assert tools.messages_sent[0]["mentions"] == [{"id": "user-1"}]
        task_events = [e for e in tools.events_sent if e["message_type"] == "task"]
        assert task_events
        assert task_events[0]["metadata"]["opencode_session_id"] == "sess-1"

    @pytest.mark.asyncio
    async def test_reuses_persisted_session(self) -> None:
        fake_client = FakeOpencodeClient(
            prompt_event_sequences=[
                [
                    event_message_updated("sess-existing", "msg-2"),
                    event_text_part("sess-existing", "msg-2", "Reused session"),
                    event_session_idle("sess-existing"),
                ]
            ]
        )
        adapter = OpencodeAdapter(client_factory=lambda _config: fake_client)
        tools = FakeAgentTools()

        await adapter.on_started("OpenCode Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(),
            tools_protocol(tools),
            OpencodeSessionState(session_id="sess-existing", room_id="room-1"),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        assert fake_client.created_sessions == []
        assert fake_client.prompt_calls[0]["session_id"] == "sess-existing"
        assert tools.messages_sent[0]["content"] == "Reused session"

    @pytest.mark.asyncio
    async def test_manual_permission_reply_from_follow_up_message(self) -> None:
        fake_client = FakeOpencodeClient(
            prompt_event_sequences=[[event_permission("sess-1", "req-1")]],
            reply_permission_events={
                "req-1": [
                    event_message_updated("sess-1", "msg-3"),
                    event_text_part("sess-1", "msg-3", "Approved and done"),
                    event_session_idle("sess-1"),
                ]
            },
        )
        adapter = OpencodeAdapter(client_factory=lambda _config: fake_client)
        tools = FakeAgentTools()

        await adapter.on_started("OpenCode Agent", "A coding agent")
        first_turn = asyncio.create_task(
            adapter.on_message(
                make_platform_message(content="Please continue"),
                tools_protocol(tools),
                OpencodeSessionState(),
                participants_msg=None,
                contacts_msg=None,
                is_session_bootstrap=True,
                room_id="room-1",
            )
        )

        await wait_for(
            lambda: any(
                "approval requested" in m["content"].lower()
                for m in tools.messages_sent
            )
        )
        await wait_for(lambda: first_turn.done())
        assert all(msg["content"] != "Approved and done" for msg in tools.messages_sent)

        await adapter.on_message(
            make_platform_message(content="approve req-1"),
            tools_protocol(tools),
            OpencodeSessionState(session_id="sess-1", room_id="room-1"),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=False,
            room_id="room-1",
        )
        await first_turn
        await wait_for(
            lambda: any(
                msg["content"] == "Approved and done" for msg in tools.messages_sent
            )
        )

        assert fake_client.permission_replies == [
            {"request_id": "req-1", "reply": "once", "message": None}
        ]
        assert any(msg["content"] == "Approved and done" for msg in tools.messages_sent)

    @pytest.mark.asyncio
    async def test_manual_question_reply_from_follow_up_message(self) -> None:
        fake_client = FakeOpencodeClient(
            prompt_event_sequences=[
                [event_question("sess-1", "q-1", "What should I do next?")]
            ],
            reply_question_events={
                "q-1": [
                    event_message_updated("sess-1", "msg-4"),
                    event_text_part("sess-1", "msg-4", "Question answered"),
                    event_session_idle("sess-1"),
                ]
            },
        )
        adapter = OpencodeAdapter(client_factory=lambda _config: fake_client)
        tools = FakeAgentTools()

        await adapter.on_started("OpenCode Agent", "A coding agent")
        first_turn = asyncio.create_task(
            adapter.on_message(
                make_platform_message(content="Need an answer"),
                tools_protocol(tools),
                OpencodeSessionState(),
                participants_msg=None,
                contacts_msg=None,
                is_session_bootstrap=True,
                room_id="room-1",
            )
        )

        await wait_for(
            lambda: any(
                "asked question" in message["content"].lower()
                for message in tools.messages_sent
            )
        )
        await wait_for(lambda: first_turn.done())

        await adapter.on_message(
            make_platform_message(content="Ship the adapter"),
            tools_protocol(tools),
            OpencodeSessionState(session_id="sess-1", room_id="room-1"),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=False,
            room_id="room-1",
        )

        await wait_for(
            lambda: any(
                message["content"] == "Question answered"
                for message in tools.messages_sent
            )
        )
        assert fake_client.question_replies == [
            {"request_id": "q-1", "answers": [["Ship the adapter"]]}
        ]

    @pytest.mark.asyncio
    async def test_prompt_submission_failure_does_not_leave_room_stuck(self) -> None:
        fake_client = FakeOpencodeClient(
            prompt_event_sequences=[
                [
                    event_message_updated("sess-1", "msg-5"),
                    event_text_part("sess-1", "msg-5", "Recovered after failure"),
                    event_session_idle("sess-1"),
                ]
            ],
            prompt_exceptions=[AnyHTTPStatusError(500, "sess-1")],
        )
        adapter = OpencodeAdapter(client_factory=lambda _config: fake_client)
        tools = FakeAgentTools()

        await adapter.on_started("OpenCode Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(content="first try"),
            tools_protocol(tools),
            OpencodeSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        await adapter.on_message(
            make_platform_message(content="second try"),
            tools_protocol(tools),
            OpencodeSessionState(session_id="sess-1", room_id="room-1"),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=False,
            room_id="room-1",
        )

        assert len(fake_client.prompt_calls) == 2
        assert not any(
            "still processing the previous request" in event["content"].lower()
            for event in tools.events_sent
        )
        assert any(
            message["content"] == "Recovered after failure"
            for message in tools.messages_sent
        )

    @pytest.mark.asyncio
    async def test_missing_session_replays_history_into_new_prompt(self) -> None:
        fake_client = FakeOpencodeClient(
            prompt_event_sequences=[
                [
                    event_message_updated("sess-1", "msg-6"),
                    event_text_part("sess-1", "msg-6", "Session recreated"),
                    event_session_idle("sess-1"),
                ]
            ],
            get_session_missing={"sess-missing"},
        )
        adapter = OpencodeAdapter(client_factory=lambda _config: fake_client)
        tools = FakeAgentTools()

        await adapter.on_started("OpenCode Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(content="Continue from before"),
            tools_protocol(tools),
            OpencodeSessionState(
                session_id="sess-missing",
                room_id="room-1",
                replay_messages=[
                    "[Alice]: Earlier question",
                    "[OpenCode Agent]: Earlier answer",
                ],
            ),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=False,
            room_id="room-1",
        )

        prompt_text = fake_client.prompt_calls[0]["parts"][0]["text"]
        assert fake_client.created_sessions[0]["id"] == "sess-1"
        assert "Recovered room history" in prompt_text
        assert "[Alice]: Earlier question" in prompt_text
        assert "[OpenCode Agent]: Earlier answer" in prompt_text

    @pytest.mark.asyncio
    async def test_reports_tool_events_when_enabled(self) -> None:
        fake_client = FakeOpencodeClient(
            prompt_event_sequences=[
                [
                    event_tool_part(
                        "sess-1",
                        "msg-4",
                        tool="bash",
                        call_id="call-1",
                        status="running",
                        input_data={"command": "pytest"},
                    ),
                    event_tool_part(
                        "sess-1",
                        "msg-4",
                        tool="bash",
                        call_id="call-1",
                        status="completed",
                        input_data={"command": "pytest"},
                        output="ok",
                    ),
                    event_session_idle("sess-1"),
                ]
            ]
        )
        adapter = OpencodeAdapter(
            config=OpencodeAdapterConfig(enable_execution_reporting=True),
            client_factory=lambda _config: fake_client,
        )
        tools = FakeAgentTools()

        await adapter.on_started("OpenCode Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(),
            tools_protocol(tools),
            OpencodeSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        tool_calls = [e for e in tools.events_sent if e["message_type"] == "tool_call"]
        tool_results = [
            e for e in tools.events_sent if e["message_type"] == "tool_result"
        ]
        assert len(tool_calls) == 1
        assert len(tool_results) == 1
        assert json.loads(tool_calls[0]["content"])["name"] == "bash"
        assert json.loads(tool_results[0]["content"])["output"] == "ok"

    @pytest.mark.asyncio
    async def test_does_not_echo_user_text_parts_as_assistant_output(self) -> None:
        fake_client = FakeOpencodeClient(
            prompt_event_sequences=[
                [
                    event_user_message_updated("sess-1", "msg-user"),
                    event_text_part("sess-1", "msg-user", "user prompt text"),
                    event_session_idle("sess-1"),
                ]
            ]
        )
        adapter = OpencodeAdapter(
            config=OpencodeAdapterConfig(provider_id="openai", model_id="gpt-5.2"),
            client_factory=lambda _config: fake_client,
        )
        tools = FakeAgentTools()

        await adapter.on_started("OpenCode Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(content="user prompt text"),
            tools_protocol(tools),
            OpencodeSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        assert tools.messages_sent[0]["content"] == (
            "OpenCode completed the turn without a text reply."
        )

    @pytest.mark.asyncio
    async def test_ignores_reasoning_deltas_and_relays_final_text_only(self) -> None:
        fake_client = FakeOpencodeClient(
            prompt_event_sequences=[
                [
                    event_message_updated("sess-1", "msg-assistant"),
                    event_reasoning_part(
                        "sess-1",
                        "msg-assistant",
                        part_id="part-reasoning",
                    ),
                    event_part_delta(
                        "sess-1",
                        "msg-assistant",
                        "part-reasoning",
                        'The user wants "pong".',
                    ),
                    event_text_part("sess-1", "msg-assistant", ""),
                    event_part_delta(
                        "sess-1",
                        "msg-assistant",
                        "part-msg-assistant",
                        "pong",
                    ),
                    event_session_idle("sess-1"),
                ]
            ]
        )
        adapter = OpencodeAdapter(client_factory=lambda _config: fake_client)
        tools = FakeAgentTools()

        await adapter.on_started("OpenCode Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(content="Reply with exactly: pong"),
            tools_protocol(tools),
            OpencodeSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        assert tools.messages_sent[0]["content"] == "pong"

    @pytest.mark.asyncio
    async def test_session_error_emits_error_event(self) -> None:
        fake_client = FakeOpencodeClient(
            prompt_event_sequences=[[event_session_error("sess-1", "boom")]]
        )
        adapter = OpencodeAdapter(client_factory=lambda _config: fake_client)
        tools = FakeAgentTools()

        await adapter.on_started("OpenCode Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(),
            tools_protocol(tools),
            OpencodeSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        error_events = [e for e in tools.events_sent if e["message_type"] == "error"]
        assert error_events
        assert "boom" in error_events[0]["content"].lower()

    @pytest.mark.asyncio
    async def test_cleanup_is_idempotent(self) -> None:
        fake_client = FakeOpencodeClient(
            prompt_event_sequences=[
                [
                    event_message_updated("sess-1", "msg-5"),
                    event_text_part("sess-1", "msg-5", "done"),
                    event_session_idle("sess-1"),
                ]
            ]
        )
        adapter = OpencodeAdapter(client_factory=lambda _config: fake_client)
        tools = FakeAgentTools()

        await adapter.on_started("OpenCode Agent", "A coding agent")
        await adapter.on_message(
            make_platform_message(),
            tools_protocol(tools),
            OpencodeSessionState(),
            participants_msg=None,
            contacts_msg=None,
            is_session_bootstrap=True,
            room_id="room-1",
        )

        await adapter.on_cleanup("room-1")
        await adapter.on_cleanup("room-1")
        assert fake_client.closed is True
