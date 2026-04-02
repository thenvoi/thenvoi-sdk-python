"""Adapter registry for config-driven agent creation."""

from __future__ import annotations

from difflib import get_close_matches
from enum import Enum
from typing import Any

from thenvoi.core.exceptions import ThenvoiConfigError


def _normalize_enum_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _normalize_enum_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_enum_value(item) for item in value]
    if isinstance(value, Enum):
        return value
    return value


def _build_anthropic_adapter(config: dict[str, Any]) -> Any:
    from thenvoi.adapters.anthropic import AnthropicAdapter

    return AnthropicAdapter(**{k: _normalize_enum_value(v) for k, v in config.items()})


def _build_gemini_adapter(config: dict[str, Any]) -> Any:
    from thenvoi.adapters.gemini import GeminiAdapter

    normalized = {k: _normalize_enum_value(v) for k, v in config.items()}
    if "api_key" in normalized:
        normalized["gemini_api_key"] = normalized.pop("api_key")
    if "prompt" in normalized:
        normalized["custom_section"] = normalized.pop("prompt")
    if "max_tokens" in normalized:
        normalized["max_output_tokens"] = normalized.pop("max_tokens")
    return GeminiAdapter(**normalized)


def _build_claude_sdk_adapter(config: dict[str, Any]) -> Any:
    from thenvoi.adapters.claude_sdk import ClaudeSDKAdapter

    normalized = {k: _normalize_enum_value(v) for k, v in config.items()}
    if "prompt" in normalized:
        normalized["custom_section"] = normalized.pop("prompt")
    return ClaudeSDKAdapter(**normalized)


def _build_codex_adapter(config: dict[str, Any]) -> Any:
    from thenvoi.adapters.codex import CodexAdapter, CodexAdapterConfig

    normalized = {k: _normalize_enum_value(v) for k, v in config.items()}
    if "prompt" in normalized:
        normalized["custom_section"] = normalized.pop("prompt")
    if "adapter_config" in normalized:
        raise ThenvoiConfigError(
            "Nested adapter_config is not supported. Put Codex options directly under adapter:."
        )
    return CodexAdapter(config=CodexAdapterConfig(**normalized))


def _build_langgraph_adapter(config: dict[str, Any]) -> Any:
    raise ThenvoiConfigError(
        "LangGraph requires a Python-built adapter. Pass adapter=LangGraphAdapter(...) to Agent.from_config()."
    )


def _build_crewai_adapter(config: dict[str, Any]) -> Any:
    from thenvoi.adapters.crewai import CrewAIAdapter

    normalized = {k: _normalize_enum_value(v) for k, v in config.items()}
    if "prompt" in normalized:
        normalized["custom_section"] = normalized.pop("prompt")
    return CrewAIAdapter(**normalized)


def _build_pydantic_ai_adapter(config: dict[str, Any]) -> Any:
    from thenvoi.adapters.pydantic_ai import PydanticAIAdapter

    normalized = {k: _normalize_enum_value(v) for k, v in config.items()}
    if "prompt" in normalized:
        normalized["custom_section"] = normalized.pop("prompt")
    return PydanticAIAdapter(**normalized)


def _build_google_adk_adapter(config: dict[str, Any]) -> Any:
    from thenvoi.adapters.google_adk import GoogleADKAdapter

    normalized = {k: _normalize_enum_value(v) for k, v in config.items()}
    if "prompt" in normalized:
        normalized["custom_section"] = normalized.pop("prompt")
    return GoogleADKAdapter(**normalized)


def _build_parlant_adapter(config: dict[str, Any]) -> Any:
    from thenvoi.adapters.parlant import ParlantAdapter

    normalized = {k: _normalize_enum_value(v) for k, v in config.items()}
    if "prompt" in normalized:
        normalized["custom_section"] = normalized.pop("prompt")
    return ParlantAdapter(**normalized)


def _build_letta_adapter(config: dict[str, Any]) -> Any:
    from thenvoi.adapters.letta import LettaAdapter, LettaAdapterConfig

    normalized = {k: _normalize_enum_value(v) for k, v in config.items()}
    if "prompt" in normalized:
        normalized["custom_section"] = normalized.pop("prompt")
    return LettaAdapter(config=LettaAdapterConfig(**normalized))


_ADAPTER_BUILDERS = {
    "anthropic": _build_anthropic_adapter,
    "gemini": _build_gemini_adapter,
    "claude_sdk": _build_claude_sdk_adapter,
    "codex": _build_codex_adapter,
    "langgraph": _build_langgraph_adapter,
    "crewai": _build_crewai_adapter,
    "pydantic_ai": _build_pydantic_ai_adapter,
    "google_adk": _build_google_adk_adapter,
    "parlant": _build_parlant_adapter,
    "letta": _build_letta_adapter,
}


def build_adapter_from_config(adapter_config: dict[str, Any]) -> Any:
    """Build an adapter instance from YAML adapter config."""
    adapter_type = adapter_config.get("type")
    if not adapter_type:
        raise ThenvoiConfigError(
            "Missing adapter type. Add adapter.type to your config."
        )

    builder = _ADAPTER_BUILDERS.get(str(adapter_type))
    if builder is None:
        candidates = ", ".join(sorted(_ADAPTER_BUILDERS))
        suggestion = get_close_matches(str(adapter_type), _ADAPTER_BUILDERS.keys(), n=1)
        hint = f" Did you mean '{suggestion[0]}'?" if suggestion else ""
        raise ThenvoiConfigError(
            f"Unknown adapter type '{adapter_type}'. Use one of: {candidates}.{hint}"
        )

    kwargs = {k: v for k, v in adapter_config.items() if k != "type"}
    return builder(kwargs)
