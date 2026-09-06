"""Tests for shared AgentToolsProtocol helpers."""

from __future__ import annotations

import pytest
from band_sdk_core import AgentFailure

from band.core.protocols import to_failure_event


class TestToFailureEvent:
    """``to_failure_event`` is the one place both SDKs put the failure ->
    room-event shape, so its parity string and metadata key are load-bearing."""

    def test_carries_the_message_and_failure_metadata(self) -> None:
        failure = AgentFailure("codex", "boom", "timeout", {"http_status": 500})

        content, metadata = to_failure_event(failure)

        assert content == "boom"
        assert metadata == {
            "failure": {
                "provider": "codex",
                "code": "timeout",
                "message": "boom",
                "detail": {"http_status": 500},
            }
        }

    @pytest.mark.parametrize("blank_message", ["", "   ", "\n\t"])
    def test_blank_message_falls_back_to_a_generic_one(
        self, blank_message: str
    ) -> None:
        """A provider message can arrive blank -- the platform rejects a blank
        chat event, so an unguarded blank message would make the failure
        vanish from the room entirely. The fallback string must match TS's
        ``toFailureEvent`` exactly for cross-SDK parity."""
        content, _metadata = to_failure_event(AgentFailure("acp", blank_message))

        assert content == "acp failed without an error message."

    def test_generic_fallback_has_no_code_or_detail(self) -> None:
        """A provider/adapter that gives no structured signal at all still
        produces a valid failure -- code and detail default to None rather
        than an invented value."""
        content, metadata = to_failure_event(AgentFailure("anthropic", "boom"))

        assert content == "boom"
        assert metadata["failure"]["code"] is None
        assert metadata["failure"]["detail"] is None
