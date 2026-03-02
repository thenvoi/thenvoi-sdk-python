"""Tests for data-driven run_agent runner registry."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from examples.run_agent import (
    RunnerSpec,
    _build_runner_kwargs,
    _resolve_codex_custom_section,
)


def _make_args() -> argparse.Namespace:
    return argparse.Namespace(
        custom_section="custom prompt",
        streaming=True,
        thinking=True,
        codex_transport="ws",
        codex_ws_url="ws://codex.local",
        codex_model="gpt-codex",
        codex_personality="pragmatic",
        codex_approval_policy="never",
        codex_approval_mode="manual",
        codex_turn_task_markers=True,
        codex_cwd="/tmp/project",
        codex_sandbox="workspace-write",
        codex_reasoning_effort="medium",
        a2a_url="http://a2a.local",
        gateway_port=11000,
        debug=True,
    )


def test_build_runner_kwargs_with_standard_feature_flags() -> None:
    """Standard framework options should be mapped from declarative flags."""
    spec = RunnerSpec(
        default_agent="simple_agent",
        runner=MagicMock(),
        uses_model=True,
        uses_custom_section=True,
        uses_streaming=True,
        uses_contact_config=True,
        uses_thinking=True,
    )

    kwargs = _build_runner_kwargs(
        spec=spec,
        args=_make_args(),
        agent_id="agent-1",
        api_key="api-1",
        rest_url="https://rest",
        ws_url="wss://ws",
        model="model-x",
        contact_config=object(),
        logger=MagicMock(),
    )

    assert kwargs["agent_id"] == "agent-1"
    assert kwargs["api_key"] == "api-1"
    assert kwargs["rest_url"] == "https://rest"
    assert kwargs["ws_url"] == "wss://ws"
    assert kwargs["model"] == "model-x"
    assert kwargs["custom_section"] == "custom prompt"
    assert kwargs["enable_streaming"] is True
    assert kwargs["enable_thinking"] is True
    assert "contact_config" in kwargs


def test_build_runner_kwargs_uses_custom_section_resolver() -> None:
    """Codex-style role prompt resolution should override raw custom_section."""
    spec = RunnerSpec(
        default_agent="simple_agent",
        runner=MagicMock(),
        uses_custom_section=True,
        custom_section_resolver=lambda _args, _logger: "role prompt",
    )

    kwargs = _build_runner_kwargs(
        spec=spec,
        args=_make_args(),
        agent_id="agent-1",
        api_key="api-1",
        rest_url="https://rest",
        ws_url="wss://ws",
        model="unused",
        contact_config=None,
        logger=MagicMock(),
    )

    assert kwargs["custom_section"] == "role prompt"


def test_build_runner_kwargs_maps_codex_and_gateway_options() -> None:
    """Codex and A2A/Gateway feature flags should map dedicated options."""
    spec = RunnerSpec(
        default_agent="simple_agent",
        runner=MagicMock(),
        uses_codex_options=True,
        uses_a2a_url=True,
        uses_gateway_port=True,
    )

    kwargs = _build_runner_kwargs(
        spec=spec,
        args=_make_args(),
        agent_id="agent-1",
        api_key="api-1",
        rest_url="https://rest",
        ws_url="wss://ws",
        model="unused",
        contact_config=None,
        logger=MagicMock(),
    )

    assert kwargs["codex_transport"] == "ws"
    assert kwargs["codex_ws_url"] == "ws://codex.local"
    assert kwargs["codex_model"] == "gpt-codex"
    assert kwargs["codex_reasoning_effort"] == "medium"
    assert kwargs["a2a_url"] == "http://a2a.local"
    assert kwargs["gateway_port"] == 11000
    assert kwargs["enable_debug"] is True


def test_resolve_codex_custom_section_wraps_prompt_read_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Codex prompt read failures should be normalized to ValueError."""
    args = _make_args()
    args.codex_role = "reviewer"
    logger = logging.getLogger(__name__)

    monkeypatch.setattr(Path, "exists", lambda _self: True)

    def _failing_read_text(_self: Path, *, encoding: str = "utf-8") -> str:
        raise OSError("read denied")

    monkeypatch.setattr(Path, "read_text", _failing_read_text)

    with pytest.raises(ValueError, match="Failed to read Codex role prompt file"):
        _resolve_codex_custom_section(args, logger)
