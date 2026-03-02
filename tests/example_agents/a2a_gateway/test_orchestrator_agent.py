"""Unit tests for OrchestratorAgent response/status handling."""

from __future__ import annotations

import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

import thenvoi.integrations.a2a_gateway.orchestrator.agent as orchestrator_module
from thenvoi.integrations.a2a_gateway.orchestrator.agent import (
    OrchestratorAgent,
    ResponseFormat,
)

# Ensure ChatOpenAI can be constructed in tests.
os.environ.setdefault("OPENAI_API_KEY", "test-key-for-unit-tests")


@dataclass
class _State:
    values: dict[str, Any]


class _GraphDouble:
    """Minimal graph double for stream/state tests."""

    def __init__(
        self,
        *,
        stream_items: list[dict[str, Any]] | None = None,
        structured_response: Any = None,
    ) -> None:
        self.stream_items = stream_items or []
        self.structured_response = structured_response
        self.stream_calls: list[dict[str, Any]] = []

    async def astream(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any],
        stream_mode: str,
    ):
        self.stream_calls.append(
            {"inputs": inputs, "config": config, "stream_mode": stream_mode}
        )
        for item in self.stream_items:
            yield item

    async def aget_state(self, config: dict[str, Any]) -> _State:
        return _State(values={"structured_response": self.structured_response})


@pytest.fixture
def make_agent(monkeypatch: pytest.MonkeyPatch):
    """Factory that creates OrchestratorAgent instances with injected graph/gateway."""

    def _build(graph: _GraphDouble) -> tuple[OrchestratorAgent, MagicMock]:
        mock_gateway = MagicMock()
        mock_gateway.close = AsyncMock()

        monkeypatch.setattr(orchestrator_module, "GatewayClient", lambda _url: mock_gateway)
        monkeypatch.setattr(
            orchestrator_module,
            "ChatOpenAI",
            lambda model: MagicMock(name=f"llm-{model}"),
        )
        monkeypatch.setattr(
            orchestrator_module,
            "MemorySaver",
            lambda: MagicMock(name="memory"),
        )
        monkeypatch.setattr(
            orchestrator_module,
            "create_react_agent",
            lambda *args, **kwargs: graph,
        )

        agent = OrchestratorAgent(
            gateway_url="http://gateway.local",
            available_peers=["weather", "research"],
            model="gpt-4o-mini",
        )
        return agent, mock_gateway

    return _build


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "expected_complete", "expected_input"),
    [
        ("input_required", False, True),
        ("error", False, True),
        ("completed", True, False),
    ],
)
async def test_aget_agent_response_maps_structured_statuses(
    make_agent,
    status: str,
    expected_complete: bool,
    expected_input: bool,
) -> None:
    """Each structured status should map to the expected executor flags."""
    graph = _GraphDouble(
        structured_response=ResponseFormat(status=status, message=f"status={status}")
    )
    agent, _ = make_agent(graph)

    result = await agent._aget_agent_response({"configurable": {"thread_id": "ctx-1"}})

    assert result == {
        "is_task_complete": expected_complete,
        "require_user_input": expected_input,
        "content": f"status={status}",
    }


@pytest.mark.asyncio
async def test_aget_agent_response_falls_back_for_unknown_structured_payload(
    make_agent,
) -> None:
    """Unknown structured payload types should return stable fallback content."""
    graph = _GraphDouble(structured_response={"status": "completed", "message": "ok"})
    agent, _ = make_agent(graph)

    result = await agent._aget_agent_response({"configurable": {"thread_id": "ctx-1"}})

    assert result == {
        "is_task_complete": False,
        "require_user_input": True,
        "content": "Unable to process your request. Please try again.",
    }


@pytest.mark.asyncio
async def test_stream_emits_interim_tool_call_event_then_final_result(make_agent) -> None:
    """Tool call messages should emit routing progress before final response."""
    tool_call_message = SimpleNamespace(tool_calls=[{"name": "call_peer_agent"}])
    plain_message = SimpleNamespace(tool_calls=[])
    graph = _GraphDouble(
        stream_items=[
            {"messages": [tool_call_message]},
            {"messages": [plain_message]},
        ],
        structured_response=None,
    )
    agent, _ = make_agent(graph)

    expected_final = {
        "is_task_complete": True,
        "require_user_input": False,
        "content": "done",
    }
    agent._aget_agent_response = AsyncMock(return_value=expected_final)

    result = [item async for item in agent.stream("find weather", "ctx-user-1")]

    assert result == [
        {
            "is_task_complete": False,
            "require_user_input": False,
            "content": "Routing request to peer agent...",
        },
        expected_final,
    ]
    assert graph.stream_calls == [
        {
            "inputs": {"messages": [("user", "find weather")]},
            "config": {"configurable": {"thread_id": "ctx-user-1"}},
            "stream_mode": "values",
        }
    ]
    agent._aget_agent_response.assert_awaited_once_with(
        {"configurable": {"thread_id": "ctx-user-1"}}
    )


@pytest.mark.asyncio
async def test_close_closes_gateway_client(make_agent) -> None:
    """close() should delegate cleanup to gateway client."""
    graph = _GraphDouble()
    agent, mock_gateway = make_agent(graph)

    await agent.close()

    mock_gateway.close.assert_awaited_once()
