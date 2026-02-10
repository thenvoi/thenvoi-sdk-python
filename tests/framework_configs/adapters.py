"""Adapter configuration registry for parameterized conformance tests.

Each AdapterConfig describes a framework adapter's properties, default values,
custom initialization kwargs, and factory function so that conformance tests can
run identical logic across all registered adapters.
"""

from __future__ import annotations

import functools
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable
from unittest.mock import MagicMock

__all__ = ["AdapterConfig", "ADAPTER_CONFIGS"]

# Populated lazily via __getattr__ to avoid top-level adapter imports.
ADAPTER_CONFIGS: list[AdapterConfig]


class _MissingSentinel:
    __slots__ = ()

    def __repr__(self) -> str:
        return "<MISSING>"


_MISSING = _MissingSentinel()


def _default_from_init(cls: type, param: str, fallback: Any = _MISSING) -> Any:
    """Extract the default value of *param* from *cls.__init__* signature.

    Keeps test configs in sync with adapter source automatically — no need
    to hard-code model strings or other defaults here.

    Relies on ``inspect.signature(cls.__init__)``. It does not work for
    classes that use ``__init_subclass__``, custom metaclasses, or
    constructor patterns that hide defaults (e.g. attrs, Pydantic). If an
    adapter switches to such a pattern, expected_initial_values must be
    updated manually or a fallback passed.

    If the parameter is not found (e.g. the adapter accepts ``**kwargs``
    and forwards it to an underlying client), *fallback* is returned when
    provided.  Without a fallback, ``ValueError`` is raised.
    """
    sig = inspect.signature(cls.__init__)
    p = sig.parameters.get(param)
    if p is None or p.default is inspect.Parameter.empty:
        if fallback is not _MISSING:
            return fallback
        raise ValueError(
            f"{cls.__name__}.__init__ has no default for {param!r}. "
            f"If the adapter uses **kwargs, pass an explicit fallback value."
        )
    return p.default


@dataclass(frozen=True)
class AdapterConfig:
    """Describes a framework adapter for parameterized testing."""

    # Identity
    framework_id: str
    display_name: str

    # Factory: (**kwargs) -> adapter instance (handles mocking internally)
    adapter_factory: Callable[..., Any]

    # {attr_name: expected_value} verified by test_default_initialization.
    # For most adapters these are true defaults from __init__; for PydanticAI
    # ``model`` is a required kwarg injected by the factory (not a real default).
    expected_initial_values: dict[str, Any] = field(default_factory=dict)

    # For test_custom_initialization
    custom_kwargs: dict[str, Any] = field(default_factory=dict)
    custom_expected: dict[str, Any] = field(default_factory=dict)

    # Custom tools support
    has_custom_tools_attr: bool = True
    custom_tools_attr: str = "_custom_tools"

    # History converter presence
    has_history_converter: bool = True

    # Skip on_started conformance test when adapter needs live client (e.g. PydanticAI)
    skip_on_started_conformance: bool = False


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def _anthropic_factory(**kw: Any) -> Any:
    from thenvoi.adapters.anthropic import AnthropicAdapter

    return AnthropicAdapter(**kw)


def _langgraph_factory(**kw: Any) -> Any:
    from thenvoi.adapters.langgraph import LangGraphAdapter

    if "llm" not in kw and "graph_factory" not in kw and "graph" not in kw:
        kw["llm"] = MagicMock()
        kw["checkpointer"] = MagicMock()
    return LangGraphAdapter(**kw)


# CrewAI conformance instances are ONLY safe for inspecting primitive
# attributes (model, role, etc.).  Runtime methods (on_message, _invoke_crew)
# are guarded because the class is imported with mocked crewai dependencies.
# For runtime tests, use monkeypatch fixtures in tests/adapters/test_crewai_adapter.py.


class _MockBaseTool:
    """Minimal stand-in for ``crewai.tools.BaseTool`` at import time."""

    name: str = ""
    description: str = ""

    def __init__(self):
        pass


def _get_crewai_adapter_cls() -> type:
    """Import CrewAIAdapter with mocked crewai dependencies.

    Uses ``patch.dict(sys.modules, ...)`` to temporarily inject mock modules,
    imports the adapter, then cleans up.  The returned class retains mock
    references in its module globals — safe for attribute inspection only.
    """
    import importlib
    import sys
    from unittest.mock import patch

    adapter_module_name = "thenvoi.adapters.crewai"

    # Clear any prior imports to ensure a clean mock-based import.
    for mod in (adapter_module_name, "crewai", "crewai.tools", "nest_asyncio"):
        sys.modules.pop(mod, None)

    mock_crewai = MagicMock()
    mock_crewai.Agent = MagicMock()
    mock_crewai.LLM = MagicMock()
    mock_crewai_tools = MagicMock()
    mock_crewai_tools.BaseTool = _MockBaseTool

    mock_entries = {
        "crewai": mock_crewai,
        "crewai.tools": mock_crewai_tools,
        "nest_asyncio": MagicMock(),
    }

    with patch.dict(sys.modules, mock_entries):
        cls = importlib.import_module(adapter_module_name).CrewAIAdapter

    # Remove adapter module so non-conformance imports aren't polluted.
    sys.modules.pop(adapter_module_name, None)
    return cls


async def _crewai_conformance_guard(*_args: Any, **_kw: Any) -> None:
    raise RuntimeError(
        "CrewAI conformance instance has mocked dependencies — "
        "use tests/adapters/test_crewai_adapter.py fixtures for runtime tests."
    )


def _crewai_factory(**kw: Any) -> Any:
    cls = _get_crewai_adapter_cls()
    instance = cls(**kw)
    # Guard runtime methods that would silently operate on MagicMock objects.
    for method_name in ("on_message", "_invoke_crew"):
        if hasattr(instance, method_name):
            setattr(instance, method_name, _crewai_conformance_guard)
    return instance


def _claude_sdk_factory(**kw: Any) -> Any:
    from thenvoi.adapters.claude_sdk import ClaudeSDKAdapter

    return ClaudeSDKAdapter(**kw)


def _pydantic_ai_factory(**kw: Any) -> Any:
    from thenvoi.adapters.pydantic_ai import PydanticAIAdapter

    if "model" not in kw:
        kw["model"] = _PYDANTIC_AI_INJECTED_MODEL
    return PydanticAIAdapter(**kw)


def _parlant_factory(**kw: Any) -> Any:
    from thenvoi.adapters.parlant import ParlantAdapter

    if "server" not in kw:
        kw["server"] = MagicMock()
    if "parlant_agent" not in kw:
        mock_agent = MagicMock()
        mock_agent.id = "parlant-agent-123"
        mock_agent.name = "TestBot"
        kw["parlant_agent"] = mock_agent
    return ParlantAdapter(**kw)


# ---------------------------------------------------------------------------
# Registry  (built lazily to avoid top-level adapter imports)
# ---------------------------------------------------------------------------

# PydanticAI requires ``model`` as a mandatory kwarg (no default in __init__).
# The conformance factory injects this value so the adapter can be instantiated
# without a real API key.  ``expected_initial_values["model"]`` then verifies
# the factory injection, NOT a real adapter default.
_PYDANTIC_AI_INJECTED_MODEL = "openai:gpt-4o"


def _build_anthropic_config() -> AdapterConfig:
    from thenvoi.adapters.anthropic import AnthropicAdapter

    return AdapterConfig(
        framework_id="anthropic",
        display_name="Anthropic",
        adapter_factory=_anthropic_factory,
        expected_initial_values={
            "model": _default_from_init(AnthropicAdapter, "model"),
            "max_tokens": _default_from_init(AnthropicAdapter, "max_tokens"),
            "enable_execution_reporting": _default_from_init(
                AnthropicAdapter, "enable_execution_reporting"
            ),
        },
        custom_kwargs={
            "model": "claude-opus-4-20250514",
            "max_tokens": 8192,
            "custom_section": "Be helpful.",
            "enable_execution_reporting": True,
        },
        custom_expected={
            "model": "claude-opus-4-20250514",
            "max_tokens": 8192,
            "custom_section": "Be helpful.",
            "enable_execution_reporting": True,
        },
    )


def _build_langgraph_config() -> AdapterConfig:
    from thenvoi.adapters.langgraph import LangGraphAdapter

    return AdapterConfig(
        framework_id="langgraph",
        display_name="LangGraph",
        adapter_factory=_langgraph_factory,
        expected_initial_values={
            "prompt_template": _default_from_init(LangGraphAdapter, "prompt_template"),
            "custom_section": _default_from_init(LangGraphAdapter, "custom_section"),
        },
        custom_kwargs={
            "custom_section": "Be helpful.",
        },
        custom_expected={
            "custom_section": "Be helpful.",
        },
        has_custom_tools_attr=True,
        custom_tools_attr="additional_tools",
        has_history_converter=True,
    )


def _build_crewai_config() -> AdapterConfig:
    crewai_cls = _get_crewai_adapter_cls()

    return AdapterConfig(
        framework_id="crewai",
        display_name="CrewAI",
        adapter_factory=_crewai_factory,
        expected_initial_values={
            "model": _default_from_init(crewai_cls, "model"),
            "role": _default_from_init(crewai_cls, "role"),
            "goal": _default_from_init(crewai_cls, "goal"),
            "backstory": _default_from_init(crewai_cls, "backstory"),
            "enable_execution_reporting": _default_from_init(
                crewai_cls, "enable_execution_reporting"
            ),
            "verbose": _default_from_init(crewai_cls, "verbose"),
            "max_iter": _default_from_init(crewai_cls, "max_iter"),
            "allow_delegation": _default_from_init(crewai_cls, "allow_delegation"),
        },
        custom_kwargs={
            "model": "gpt-4o-mini",
            "role": "Research Analyst",
            "goal": "Find and analyze information",
            "backstory": "Expert researcher",
            "custom_section": "Be thorough.",
            "enable_execution_reporting": True,
            "verbose": True,
            "max_iter": 30,
            "max_rpm": 10,
            "allow_delegation": True,
        },
        custom_expected={
            "model": "gpt-4o-mini",
            "role": "Research Analyst",
            "goal": "Find and analyze information",
            "backstory": "Expert researcher",
            "custom_section": "Be thorough.",
            "enable_execution_reporting": True,
            "verbose": True,
            "max_iter": 30,
            "max_rpm": 10,
            "allow_delegation": True,
        },
    )


def _build_claude_sdk_config() -> AdapterConfig:
    from thenvoi.adapters.claude_sdk import ClaudeSDKAdapter

    return AdapterConfig(
        framework_id="claude_sdk",
        display_name="ClaudeSDK",
        adapter_factory=_claude_sdk_factory,
        expected_initial_values={
            "model": _default_from_init(ClaudeSDKAdapter, "model"),
            "custom_section": _default_from_init(ClaudeSDKAdapter, "custom_section"),
            "max_thinking_tokens": _default_from_init(
                ClaudeSDKAdapter, "max_thinking_tokens"
            ),
            "permission_mode": _default_from_init(ClaudeSDKAdapter, "permission_mode"),
            "enable_execution_reporting": _default_from_init(
                ClaudeSDKAdapter, "enable_execution_reporting"
            ),
        },
        custom_kwargs={
            "model": "claude-opus-4-20250514",
            "custom_section": "Be helpful.",
            "max_thinking_tokens": 10000,
            "permission_mode": "bypassPermissions",
            "enable_execution_reporting": True,
        },
        custom_expected={
            "model": "claude-opus-4-20250514",
            "custom_section": "Be helpful.",
            "max_thinking_tokens": 10000,
            "permission_mode": "bypassPermissions",
            "enable_execution_reporting": True,
        },
        skip_on_started_conformance=True,  # on_started creates real MCP server + ClaudeSessionManager; tested in test_claude_sdk_adapter
    )


def _build_pydantic_ai_config() -> AdapterConfig:
    from thenvoi.adapters.pydantic_ai import PydanticAIAdapter

    return AdapterConfig(
        framework_id="pydantic_ai",
        display_name="PydanticAI",
        adapter_factory=_pydantic_ai_factory,
        expected_initial_values={
            # Injected by _pydantic_ai_factory, not a real __init__ default.
            # Verifies that the factory injection is stored correctly.
            "model": _PYDANTIC_AI_INJECTED_MODEL,
            "system_prompt": _default_from_init(PydanticAIAdapter, "system_prompt"),
            "custom_section": _default_from_init(PydanticAIAdapter, "custom_section"),
            "enable_execution_reporting": _default_from_init(
                PydanticAIAdapter, "enable_execution_reporting"
            ),
        },
        custom_kwargs={
            "model": "anthropic:claude-sonnet-4-5-20250929",
            "system_prompt": "You are a helpful bot.",
            "custom_section": "Be concise.",
            "enable_execution_reporting": True,
        },
        custom_expected={
            "model": "anthropic:claude-sonnet-4-5-20250929",
            "system_prompt": "You are a helpful bot.",
            "custom_section": "Be concise.",
            "enable_execution_reporting": True,
        },
        skip_on_started_conformance=True,  # on_started creates real OpenAI client; tested in test_pydantic_ai_adapter
    )


def _build_parlant_config() -> AdapterConfig:
    from thenvoi.adapters.parlant import ParlantAdapter

    return AdapterConfig(
        framework_id="parlant",
        display_name="Parlant",
        adapter_factory=_parlant_factory,
        expected_initial_values={
            "system_prompt": _default_from_init(ParlantAdapter, "system_prompt"),
            "custom_section": _default_from_init(ParlantAdapter, "custom_section"),
        },
        custom_kwargs={
            "system_prompt": "Custom system prompt",
            "custom_section": "Be helpful.",
        },
        custom_expected={
            "system_prompt": "Custom system prompt",
            "custom_section": "Be helpful.",
        },
        has_custom_tools_attr=False,
    )


_ADAPTER_CONFIG_BUILDERS: list[Callable[[], AdapterConfig]] = [
    _build_anthropic_config,
    _build_langgraph_config,
    _build_crewai_config,
    _build_claude_sdk_config,
    _build_pydantic_ai_config,
    _build_parlant_config,
]


@functools.lru_cache(maxsize=1)
def _build_adapter_configs() -> list[AdapterConfig]:
    """Build configs lazily so adapter imports happen only when needed.

    Each framework config is built independently so that an import failure
    in one framework does not prevent the remaining frameworks from being
    tested.  Uses ``lru_cache(maxsize=1)`` so ``.cache_clear()`` is available.
    """
    import logging

    logger = logging.getLogger(__name__)
    configs: list[AdapterConfig] = []
    for builder in _ADAPTER_CONFIG_BUILDERS:
        try:
            configs.append(builder())
        except Exception as exc:
            logger.warning("Skipping adapter config from %s: %s", builder.__name__, exc)
    return configs


def __getattr__(name: str) -> Any:
    if name == "ADAPTER_CONFIGS":
        configs = _build_adapter_configs()
        globals()["ADAPTER_CONFIGS"] = configs
        return configs
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
