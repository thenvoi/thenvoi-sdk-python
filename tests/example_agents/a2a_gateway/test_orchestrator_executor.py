"""Unit tests for orchestrator executor error/status handling."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from a2a.types import TaskState
from a2a.utils.errors import ServerError

from thenvoi.integrations.a2a_gateway.orchestrator.agent_executor import (
    OrchestratorAgentExecutor,
)
from thenvoi.integrations.a2a_gateway.orchestrator.remote_agent import (
    GatewayRequestError,
)


class _EventQueue:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def enqueue_event(self, event: Any) -> None:
        self.events.append(event)


class _TaskUpdaterSpy:
    def __init__(self, event_queue: Any, task_id: str, context_id: str) -> None:
        self.event_queue = event_queue
        self.task_id = task_id
        self.context_id = context_id
        self.status_updates: list[dict[str, Any]] = []
        self.artifacts: list[dict[str, Any]] = []
        self.completed = False

    async def update_status(
        self,
        state: TaskState,
        message: Any,
        final: bool = False,
    ) -> None:
        self.status_updates.append(
            {"state": state, "message": message, "final": final}
        )

    async def add_artifact(self, parts: Any, name: str) -> None:
        self.artifacts.append({"parts": parts, "name": name})

    async def complete(self) -> None:
        self.completed = True


class _Context:
    def __init__(self, context_id: str = "ctx-1") -> None:
        self.current_task = SimpleNamespace(id="task-1", context_id=context_id)
        self.message = {"id": "msg-1"}

    def get_user_input(self) -> str:
        return "hello"


class _AgentSuccess:
    async def stream(self, query: str, context_id: str):  # noqa: ANN201
        yield {
            "is_task_complete": False,
            "require_user_input": False,
            "content": "working",
        }
        yield {
            "is_task_complete": True,
            "require_user_input": False,
            "content": "done",
        }


class _AgentGatewayError:
    async def stream(self, query: str, context_id: str):  # noqa: ANN201
        raise GatewayRequestError(
            "temporary gateway failure",
            code="send_message_failed",
            peer_id="weather",
            retryable=True,
        )
        yield  # pragma: no cover


class _AgentUnexpectedError:
    async def stream(self, query: str, context_id: str):  # noqa: ANN201
        raise RuntimeError("boom")
        yield  # pragma: no cover


@pytest.fixture
def _patch_updater(monkeypatch: pytest.MonkeyPatch) -> list[_TaskUpdaterSpy]:
    updater_spies: list[_TaskUpdaterSpy] = []

    def _task_updater_factory(
        event_queue: Any,
        task_id: str,
        context_id: str,
    ) -> _TaskUpdaterSpy:
        updater_spy = _TaskUpdaterSpy(
            event_queue=event_queue,
            task_id=task_id,
            context_id=context_id,
        )
        updater_spies.append(updater_spy)
        return updater_spy

    monkeypatch.setattr(
        "thenvoi.integrations.a2a_gateway.orchestrator.agent_executor.TaskUpdater",
        _task_updater_factory,
    )
    monkeypatch.setattr(
        "thenvoi.integrations.a2a_gateway.orchestrator.agent_executor.new_agent_text_message",
        lambda content, context_id, task_id: {
            "content": content,
            "context_id": context_id,
            "task_id": task_id,
        },
    )
    return updater_spies


@pytest.mark.asyncio
async def test_execute_marks_working_then_completes(
    _patch_updater: list[_TaskUpdaterSpy],
) -> None:
    executor = OrchestratorAgentExecutor(_AgentSuccess())
    context = _Context()
    event_queue = _EventQueue()

    await executor.execute(context, event_queue)

    updater = _patch_updater[0]
    assert updater.status_updates[0]["state"] == TaskState.working
    assert updater.artifacts
    assert updater.completed is True


@pytest.mark.asyncio
async def test_execute_maps_gateway_error_to_input_required(
    _patch_updater: list[_TaskUpdaterSpy],
) -> None:
    executor = OrchestratorAgentExecutor(_AgentGatewayError())
    context = _Context()
    event_queue = _EventQueue()

    await executor.execute(context, event_queue)

    updater = _patch_updater[0]
    assert len(updater.status_updates) == 1
    assert updater.status_updates[0]["state"] == TaskState.input_required
    assert updater.status_updates[0]["final"] is True


@pytest.mark.asyncio
async def test_execute_raises_server_error_for_unexpected_failures(
    _patch_updater: list[_TaskUpdaterSpy],
) -> None:
    executor = OrchestratorAgentExecutor(_AgentUnexpectedError())
    context = _Context()
    event_queue = _EventQueue()

    with pytest.raises(ServerError):
        await executor.execute(context, event_queue)
