"""Shared scenario registry used by example scripts and tests."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

FrameworkName = Literal[
    "anthropic",
    "claude_sdk",
    "crewai",
    "langgraph",
    "parlant",
    "pydantic_ai",
]

BASIC_ASSISTANT_PROMPT = "You are a helpful assistant. Be concise and friendly."

_BASIC_AGENT_KEYS: dict[FrameworkName, str] = {
    "anthropic": "anthropic_agent",
    "claude_sdk": "claude_sdk_agent",
    "crewai": "crewai_agent",
    "langgraph": "simple_agent",
    "parlant": "parlant_agent",
    "pydantic_ai": "pydantic_agent",
}

_BASIC_STARTUP_MESSAGES: dict[FrameworkName, str] = {
    "anthropic": "Starting Anthropic agent...",
    "claude_sdk": "Starting Claude SDK agent...",
    "crewai": "Starting CrewAI agent...",
    "langgraph": "Starting LangGraph agent...",
    "parlant": "Starting Thenvoi agent with Parlant SDK (full tools)...",
    "pydantic_ai": "Starting Pydantic AI agent...",
}

_BASIC_ADAPTER_KWARGS: dict[str, Mapping[str, Any]] = {
    "anthropic": {
        "model": "claude-sonnet-4-5-20250929",
        "custom_section": BASIC_ASSISTANT_PROMPT,
    },
    "claude_sdk": {
        "model": "claude-sonnet-4-5-20250929",
        "custom_section": BASIC_ASSISTANT_PROMPT,
        "enable_execution_reporting": True,
    },
    "crewai": {
        "model": "gpt-4o",
        "custom_section": BASIC_ASSISTANT_PROMPT,
    },
    "pydantic_ai": {
        "model": "openai:gpt-4o",
        "custom_section": BASIC_ASSISTANT_PROMPT,
    },
}


def basic_agent_key(framework: FrameworkName) -> str:
    """Return the canonical basic-scenario agent key for a framework."""
    return _BASIC_AGENT_KEYS[framework]


def basic_startup_message(framework: FrameworkName) -> str:
    """Return the canonical startup log message for the basic scenario."""
    return _BASIC_STARTUP_MESSAGES[framework]


def basic_adapter_kwargs(framework: str) -> dict[str, Any]:
    """Return adapter kwargs for basic scenarios on model-string adapters."""
    kwargs = _BASIC_ADAPTER_KWARGS.get(framework)
    if kwargs is None:
        raise ValueError(f"Basic adapter kwargs not defined for framework: {framework}")
    return dict(kwargs)


__all__ = [
    "BASIC_ASSISTANT_PROMPT",
    "FrameworkName",
    "basic_adapter_kwargs",
    "basic_agent_key",
    "basic_startup_message",
]
