"""Shared bootstrap helpers for example scripts."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

from thenvoi import Agent
from thenvoi.config.runtime import resolve_agent_credentials, resolve_platform_urls

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExampleRuntimeConfig:
    """Runtime credentials and platform endpoints for an example agent."""

    agent_key: str
    agent_id: str
    api_key: str
    ws_url: str
    rest_url: str


@dataclass(frozen=True)
class BootstrappedAgent:
    """A ready-to-run agent paired with the runtime config used to create it."""

    runtime: ExampleRuntimeConfig
    agent: Agent


def load_platform_urls(
    *,
    ws_default: str | None = None,
    rest_default: str | None = None,
    load_env: bool = True,
    env_loader: Callable[[], None] | None = None,
) -> tuple[str, str]:
    """Resolve Thenvoi endpoint URLs with optional defaults."""
    if load_env:
        (env_loader or load_dotenv)()
    return resolve_platform_urls(
        ws_default=ws_default,
        rest_default=rest_default,
    )


def load_runtime_config(
    agent_key: str,
    *,
    ws_default: str | None = None,
    rest_default: str | None = None,
    load_env: bool = True,
    url_resolver: Callable[..., tuple[str, str]] | None = None,
    credentials_resolver: Callable[[str], tuple[str, str]] | None = None,
) -> ExampleRuntimeConfig:
    """Load runtime endpoints and credentials for an example agent."""
    resolve_urls = url_resolver or load_platform_urls
    ws_url, rest_url = resolve_urls(
        ws_default=ws_default,
        rest_default=rest_default,
        load_env=load_env,
    )

    resolve_credentials = credentials_resolver or resolve_agent_credentials
    agent_id, api_key = resolve_credentials(agent_key)
    runtime = ExampleRuntimeConfig(
        agent_key=agent_key,
        agent_id=agent_id,
        api_key=api_key,
        ws_url=ws_url,
        rest_url=rest_url,
    )
    logger.debug(
        "Loaded example runtime config",
        extra={"agent_key": runtime.agent_key, "agent_id": runtime.agent_id},
    )
    return runtime


def create_agent_from_runtime(
    runtime: ExampleRuntimeConfig,
    adapter: Any,
    *,
    agent_factory: Callable[..., Agent] | None = None,
) -> BootstrappedAgent:
    """Create a Thenvoi agent from a resolved runtime config."""
    factory = agent_factory or Agent.create
    agent = factory(
        adapter=adapter,
        agent_id=runtime.agent_id,
        api_key=runtime.api_key,
        ws_url=runtime.ws_url,
        rest_url=runtime.rest_url,
    )
    return BootstrappedAgent(runtime=runtime, agent=agent)


def bootstrap_agent(
    agent_key: str,
    adapter: Any,
    *,
    runtime_loader: Callable[[str], ExampleRuntimeConfig] | None = None,
    agent_builder: Callable[[ExampleRuntimeConfig, Any], BootstrappedAgent] | None = None,
) -> BootstrappedAgent:
    """Create a Thenvoi Agent from shared example bootstrap steps."""
    runtime = (runtime_loader or load_runtime_config)(agent_key)
    if agent_builder is not None:
        return agent_builder(runtime, adapter)
    return create_agent_from_runtime(runtime, adapter)


__all__ = [
    "BootstrappedAgent",
    "ExampleRuntimeConfig",
    "bootstrap_agent",
    "create_agent_from_runtime",
    "load_platform_urls",
    "load_runtime_config",
]

