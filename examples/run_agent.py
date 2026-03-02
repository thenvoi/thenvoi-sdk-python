#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "thenvoi-sdk[langgraph,anthropic,pydantic-ai,claude_sdk,parlant,crewai,a2a,codex]",
#   "python-dotenv>=1.1.1",
# ]
#
# [tool.uv.sources]
# thenvoi-sdk = { git = "https://github.com/thenvoi/thenvoi-sdk-python.git" }
# ///
"""Run Thenvoi SDK example agents from one CLI."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from thenvoi.example_support.bootstrap import load_runtime_config
from thenvoi.testing.example_logging import setup_logging_profile
from thenvoi.testing.example_runners import (
    _DEFAULT_MODELS,
    ExampleRunnerSpec,
    build_example_runner_kwargs as _build_runner_kwargs,
    run_a2a_agent,
    run_a2a_gateway_agent,
    run_anthropic_agent,
    run_claude_sdk_agent,
    run_codex_agent,
    run_contacts_auto_agent,
    run_contacts_broadcast_agent,
    run_contacts_hub_agent,
    run_crewai_agent,
    run_langgraph_agent,
    run_parlant_agent,
    run_pydantic_ai_agent,
    run_pydantic_ai_contacts_agent,
)
from thenvoi.platform.event import ContactEvent, ContactRequestReceivedEvent
from thenvoi.runtime.contacts.contact_tools import ContactTools
from thenvoi.runtime.types import ContactEventConfig, ContactEventStrategy

# Load environment from .env so parser defaults can use environment-backed values.
load_dotenv()


def build_contact_config(
    mode: str | None,
    logger: logging.Logger,
) -> ContactEventConfig | None:
    """Build ContactEventConfig based on mode."""
    if mode is None or mode == "none":
        return None

    if mode == "auto":

        async def auto_approve(event: ContactEvent, tools: ContactTools) -> None:
            """Auto-approve all contact requests."""
            if isinstance(event, ContactRequestReceivedEvent) and event.payload:
                logger.info(
                    "Auto-approving contact request from %s",
                    event.payload.from_handle,
                )
                await tools.respond_contact_request(
                    "approve", request_id=event.payload.id
                )

        return ContactEventConfig(
            strategy=ContactEventStrategy.CALLBACK,
            on_event=auto_approve,
            broadcast_changes=True,
        )

    if mode == "hub":
        return ContactEventConfig(
            strategy=ContactEventStrategy.HUB_ROOM,
            broadcast_changes=True,
        )

    if mode == "broadcast":
        return ContactEventConfig(
            strategy=ContactEventStrategy.DISABLED,
            broadcast_changes=True,
        )

    raise ValueError(f"Unknown contacts mode: {mode}")


def _log_level(level: str) -> int:
    """Parse log level string with a safe INFO fallback."""
    return getattr(logging, level.upper(), logging.INFO)


def _logging_profile_for_example(example: str) -> str:
    """Map runner example keys to shared logging profiles."""
    if example in {"pydantic_ai_contacts", "contacts_auto", "contacts_hub", "contacts_broadcast"}:
        return "pydantic_ai"
    if example in {"a2a", "a2a_gateway"}:
        return "a2a_gateway"
    return example


def setup_logging(*, example: str, level: str = "INFO", a2a_debug: bool = False) -> logging.Logger:
    """Configure logging using shared example logging profiles."""
    setup_logging_profile(
        _logging_profile_for_example(example),
        level=_log_level(level),
        a2a_debug=a2a_debug,
    )
    return logging.getLogger(__name__)


def _resolve_codex_custom_section(
    args: argparse.Namespace, logger: logging.Logger
) -> str:
    """Load codex role prompt when requested; fall back to custom section."""
    if not args.codex_role:
        return args.custom_section

    prompt_file = Path(__file__).parent / "codex" / "prompts" / f"{args.codex_role}.md"
    if prompt_file.exists():
        try:
            content = prompt_file.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(
                f"Failed to read Codex role prompt file '{prompt_file}': {exc}"
            ) from exc
        logger.info("Using role prompt from: %s", prompt_file)
        return content

    logger.warning(
        "Role '%s' specified but no prompt file at %s",
        args.codex_role,
        prompt_file,
    )
    return args.custom_section


RunnerSpec = ExampleRunnerSpec


RUNNER_SPECS: dict[str, RunnerSpec] = {
    "langgraph": RunnerSpec(
        default_agent="simple_agent",
        runner=run_langgraph_agent,
        uses_custom_section=True,
    ),
    "pydantic_ai": RunnerSpec(
        default_agent="simple_agent",
        runner=run_pydantic_ai_agent,
        uses_model=True,
        uses_custom_section=True,
        uses_streaming=True,
        uses_contact_config=True,
    ),
    "pydantic_ai_contacts": RunnerSpec(
        default_agent="simple_agent",
        runner=run_pydantic_ai_contacts_agent,
        uses_model=True,
    ),
    "contacts_auto": RunnerSpec(
        default_agent="simple_agent",
        runner=run_contacts_auto_agent,
        uses_model=True,
    ),
    "contacts_hub": RunnerSpec(
        default_agent="simple_agent",
        runner=run_contacts_hub_agent,
        uses_model=True,
    ),
    "contacts_broadcast": RunnerSpec(
        default_agent="simple_agent",
        runner=run_contacts_broadcast_agent,
        uses_model=True,
    ),
    "anthropic": RunnerSpec(
        default_agent="anthropic_agent",
        runner=run_anthropic_agent,
        uses_model=True,
        uses_custom_section=True,
        uses_streaming=True,
        uses_contact_config=True,
    ),
    "claude_sdk": RunnerSpec(
        default_agent="anthropic_agent",
        runner=run_claude_sdk_agent,
        uses_model=True,
        uses_custom_section=True,
        uses_streaming=True,
        uses_contact_config=True,
        uses_thinking=True,
    ),
    "parlant": RunnerSpec(
        default_agent="parlant_agent",
        runner=run_parlant_agent,
        uses_model=True,
        uses_custom_section=True,
        uses_streaming=True,
    ),
    "crewai": RunnerSpec(
        default_agent="crewai_agent",
        runner=run_crewai_agent,
        uses_model=True,
        uses_custom_section=True,
        uses_streaming=True,
    ),
    "codex": RunnerSpec(
        default_agent="simple_agent",
        runner=run_codex_agent,
        uses_custom_section=True,
        uses_codex_options=True,
        custom_section_resolver=_resolve_codex_custom_section,
    ),
    "a2a": RunnerSpec(
        default_agent="a2a_agent",
        runner=run_a2a_agent,
        uses_a2a_url=True,
    ),
    "a2a_gateway": RunnerSpec(
        default_agent="a2a_gateway_agent",
        runner=run_a2a_gateway_agent,
        uses_gateway_port=True,
    ),
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a Thenvoi SDK test agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--example",
        "-e",
        choices=list(RUNNER_SPECS),
        default="langgraph",
        help="Which example agent to run (default: langgraph)",
    )
    parser.add_argument(
        "--agent",
        "-g",
        default=None,
        help="Agent key from agent_config.yaml (default: based on --example)",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="openai:gpt-4o",
        help="Model for Pydantic AI/Anthropic examples (default: openai:gpt-4o)",
    )
    parser.add_argument(
        "--custom-section",
        "-c",
        default="You are a helpful assistant. Keep responses concise.",
        help="Custom instructions for the agent",
    )
    parser.add_argument(
        "--log-level",
        "-l",
        default=os.getenv("LOG_LEVEL", "INFO"),
        help="Logging level (default: INFO or LOG_LEVEL env var)",
    )
    parser.add_argument(
        "--thinking",
        "-t",
        action="store_true",
        help="Enable extended thinking for Claude SDK (default: False)",
    )
    parser.add_argument(
        "--streaming",
        "-s",
        action="store_true",
        help="Enable tool call/result visibility for anthropic/claude_sdk/parlant/crewai (default: False)",
    )
    parser.add_argument(
        "--codex-transport",
        choices=["stdio", "ws"],
        default="stdio",
        help="Codex transport mode (default: stdio)",
    )
    parser.add_argument(
        "--codex-ws-url",
        default=os.getenv("CODEX_WS_URL", "ws://127.0.0.1:8765"),
        help="Codex WebSocket URL when --codex-transport=ws",
    )
    parser.add_argument(
        "--codex-role",
        default=None,
        help="Role name; loads prompt from prompts/{role}.md into custom_section",
    )
    parser.add_argument(
        "--codex-personality",
        choices=["friendly", "pragmatic", "none"],
        default="pragmatic",
        help="Codex personality (default: pragmatic)",
    )
    parser.add_argument(
        "--codex-model",
        default=None,
        help="Codex model id override (default: auto-select via model/list)",
    )
    parser.add_argument(
        "--codex-cwd",
        default=os.getcwd(),
        help="Working directory given to Codex app-server (default: current directory)",
    )
    parser.add_argument(
        "--codex-reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        default=None,
        help="Codex reasoning effort level",
    )
    parser.add_argument(
        "--codex-sandbox",
        default=os.getenv("CODEX_SANDBOX", "external-sandbox"),
        help=(
            "Optional Codex sandbox override. "
            "Examples: external-sandbox, workspace-write, danger-full-access, "
            "or legacy aliases like workspaceWrite."
        ),
    )
    parser.add_argument(
        "--codex-approval-policy",
        default="never",
        help="Codex approvalPolicy value (default: never)",
    )
    parser.add_argument(
        "--codex-approval-mode",
        choices=["manual", "auto_accept", "auto_decline"],
        default="manual",
        help="How adapter answers Codex approval requests (default: manual)",
    )
    parser.add_argument(
        "--codex-turn-task-markers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Emit synthetic Codex turn started/completed task events (default: False)",
    )
    parser.add_argument(
        "--a2a-url",
        default=os.getenv("A2A_AGENT_URL", "http://localhost:10000"),
        help="URL of the remote A2A agent (default: http://localhost:10000 or A2A_AGENT_URL env var)",
    )
    parser.add_argument(
        "--gateway-port",
        type=int,
        default=int(os.getenv("GATEWAY_PORT", "10000")),
        help="Port for A2A Gateway (default: 10000 or GATEWAY_PORT env var)",
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Enable debug logging for adapter internals (e.g., A2A context_id tracing)",
    )
    parser.add_argument(
        "--contacts",
        choices=["none", "auto", "hub", "broadcast"],
        default="none",
        help="Contact handling strategy: none (disabled), auto (auto-approve), hub (LLM decides in hub room), broadcast (awareness only)",
    )
    return parser


async def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    spec = RUNNER_SPECS[args.example]
    if args.agent is None:
        args.agent = spec.default_agent
    logger = setup_logging(
        example=args.example,
        level=args.log_level,
        a2a_debug=args.debug,
    )

    try:
        runtime = load_runtime_config(args.agent, load_env=False)
    except Exception as e:  # pragma: no cover - argparse exits
        parser.error(f"Failed to load runtime config for '{args.agent}': {e}")

    agent_id = runtime.agent_id
    api_key = runtime.api_key
    rest_url = runtime.rest_url
    ws_url = runtime.ws_url

    logger.info("Agent: %s (%s)", args.agent, agent_id)
    logger.info("Example: %s", args.example)
    logger.info("REST URL: %s", rest_url)
    logger.info("WS URL: %s", ws_url)

    contact_config = build_contact_config(args.contacts, logger)
    if contact_config:
        logger.info(
            "Contacts: %s (broadcast=%s)",
            contact_config.strategy.value,
            contact_config.broadcast_changes,
        )

    model = args.model
    if model == "openai:gpt-4o" and args.example in _DEFAULT_MODELS:
        model = _DEFAULT_MODELS[args.example]

    runner_kwargs = _build_runner_kwargs(
        spec=spec,
        args=args,
        agent_id=agent_id,
        api_key=api_key,
        rest_url=rest_url,
        ws_url=ws_url,
        model=model,
        contact_config=contact_config,
        logger=logger,
    )
    await spec.runner(**runner_kwargs)


if __name__ == "__main__":
    asyncio.run(main())
