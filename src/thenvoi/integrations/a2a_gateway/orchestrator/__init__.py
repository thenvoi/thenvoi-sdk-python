"""Demo A2A Gateway orchestrator public namespace."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = ("OrchestratorAgent", "OrchestratorAgentExecutor", "GatewayClient")

_EXPORT_MODULES: dict[str, str] = {
    "OrchestratorAgent": "thenvoi.integrations.a2a_gateway.orchestrator.agent",
    "OrchestratorAgentExecutor": "thenvoi.integrations.a2a_gateway.orchestrator.agent_executor",
    "GatewayClient": "thenvoi.integrations.a2a_gateway.orchestrator.remote_agent",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
