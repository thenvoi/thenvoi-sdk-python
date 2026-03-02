"""Basic Codex adapter example implementation."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from thenvoi.example_support.bootstrap import bootstrap_agent
from thenvoi.adapters import CodexAdapter, CodexAdapterConfig
from thenvoi.testing.example_logging import setup_logging_profile

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _load_role_prompt(codex_role: str | None) -> str:
    custom_section = "You are a helpful assistant. Keep responses concise."
    if not codex_role:
        return custom_section

    prompt_file = Path(__file__).resolve().parent / "prompts" / f"{codex_role}.md"
    if prompt_file.exists():
        logger.info("Using role prompt from: %s", prompt_file)
        return prompt_file.read_text(encoding="utf-8")

    logger.warning("Role '%s' specified but no prompt file at %s", codex_role, prompt_file)
    return custom_section


async def main() -> None:
    setup_logging_profile("codex")

    agent_key = os.getenv("AGENT_KEY", "darter")
    codex_transport = os.getenv("CODEX_TRANSPORT", "stdio")
    if codex_transport not in {"stdio", "ws"}:
        raise ValueError("CODEX_TRANSPORT must be 'stdio' or 'ws'")

    codex_role = os.getenv("CODEX_ROLE")
    custom_section = _load_role_prompt(codex_role)

    adapter = CodexAdapter(
        config=CodexAdapterConfig(
            transport=codex_transport,  # type: ignore[arg-type]  # str from env, validated at runtime
            codex_ws_url=os.getenv("CODEX_WS_URL", "ws://127.0.0.1:8765"),
            model=os.getenv("CODEX_MODEL") or None,
            cwd=os.getenv("CODEX_CWD", os.getcwd()),
            approval_policy=os.getenv("CODEX_APPROVAL_POLICY", "never"),
            approval_mode=os.getenv("CODEX_APPROVAL_MODE", "manual"),  # type: ignore[arg-type]  # str from env, validated at runtime
            personality="pragmatic",
            custom_section=custom_section,
            include_base_instructions=True,
            enable_task_events=True,
            emit_turn_task_markers=_env_bool("CODEX_TURN_TASK_MARKERS", False),
            fallback_send_agent_text=True,
        )
    )

    session = bootstrap_agent(agent_key=agent_key, adapter=adapter)
    logger.info(
        "Starting Codex agent: agent_key=%s transport=%s role=%s",
        agent_key,
        codex_transport,
        codex_role or "none",
    )
    await session.agent.run()
