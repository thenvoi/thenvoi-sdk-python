"""Tests for thenvoi.example_support.scenarios."""

from __future__ import annotations

import pytest

from thenvoi.example_support.scenarios import (
    BASIC_ASSISTANT_PROMPT,
    basic_adapter_kwargs,
    basic_agent_key,
    basic_startup_message,
)


def test_basic_registry_agent_keys_and_messages() -> None:
    assert basic_agent_key("anthropic") == "anthropic_agent"
    assert basic_agent_key("langgraph") == "simple_agent"
    assert basic_startup_message("pydantic_ai") == "Starting Pydantic AI agent..."


def test_basic_adapter_kwargs_include_shared_prompt() -> None:
    kwargs = basic_adapter_kwargs("claude_sdk")
    assert kwargs["custom_section"] == BASIC_ASSISTANT_PROMPT
    assert kwargs["enable_execution_reporting"] is True


def test_basic_adapter_kwargs_rejects_unsupported_framework() -> None:
    with pytest.raises(ValueError, match="not defined"):
        basic_adapter_kwargs("langgraph")
