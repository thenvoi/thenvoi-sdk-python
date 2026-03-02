"""Direct unit tests for A2A integration type models."""

from __future__ import annotations

from thenvoi.integrations.a2a.types import A2AAuth, A2ASessionState


class TestA2AAuthTypes:
    """Tests for A2AAuth header generation behavior."""

    def test_to_headers_empty(self) -> None:
        auth = A2AAuth()
        assert auth.to_headers() == {}

    def test_to_headers_merged(self) -> None:
        auth = A2AAuth(
            api_key="api-key",
            bearer_token="token",
            headers={"X-Custom": "value"},
        )
        assert auth.to_headers() == {
            "X-API-Key": "api-key",
            "Authorization": "Bearer token",
            "X-Custom": "value",
        }


class TestA2ASessionStateTypes:
    """Tests for A2ASessionState defaults and assignment."""

    def test_defaults(self) -> None:
        state = A2ASessionState()
        assert state.context_id is None
        assert state.task_id is None
        assert state.task_state is None

    def test_values(self) -> None:
        state = A2ASessionState(
            context_id="ctx-1",
            task_id="task-1",
            task_state="completed",
        )
        assert state.context_id == "ctx-1"
        assert state.task_id == "task-1"
        assert state.task_state == "completed"
