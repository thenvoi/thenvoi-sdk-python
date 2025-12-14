#!/usr/bin/env python3
"""
Run Thenvoi SDK agents.

Usage:
    uv run python examples/run_agent.py                    # Default: langgraph
    uv run python examples/run_agent.py --example langgraph
    uv run python examples/run_agent.py --example pydantic_ai
    uv run python examples/run_agent.py --example pydantic_ai --model anthropic:claude-sonnet-4-5
    uv run python examples/run_agent.py --example anthropic
    uv run python examples/run_agent.py --example anthropic --model claude-sonnet-4-5-20250929
    uv run python examples/run_agent.py --example claude_sdk
    uv run python examples/run_agent.py --example claude_sdk --thinking  # Enable extended thinking

Configure agent in agent_config.yaml:
    uv run python examples/run_agent.py --agent test_agent
    uv run python examples/run_agent.py --agent my_custom_agent

Setup:
1. Copy .env.example to .env and configure:
   - THENVOI_REST_API_URL (default: production, change for local dev)
   - THENVOI_WS_URL (default: production, change for local dev)
   - OPENAI_API_KEY (required for langgraph/openai models)
   - ANTHROPIC_API_KEY (required for anthropic models)

2. Configure agent in agent_config.yaml
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add examples directories to path for agent imports
examples_root = Path(__file__).parent
sys.path.insert(0, str(examples_root / "langgraph"))
sys.path.insert(0, str(examples_root / "pydantic_ai"))
sys.path.insert(0, str(examples_root / "anthropic"))
sys.path.insert(0, str(examples_root / "claude_sdk"))

from thenvoi.config import load_agent_config  # noqa: E402

# Load environment from .env
load_dotenv()


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure logging."""
    log_level = level.upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)


def create_langgraph_factory():
    """
    Create a LangGraph graph factory with a persistent checkpointer.

    The checkpointer is created ONCE and reused for all messages.
    Each room's conversation is isolated by thread_id (room_id).
    """
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent
    from langgraph.checkpoint.memory import MemorySaver

    checkpointer = MemorySaver()
    llm = ChatOpenAI(model="gpt-4o")

    def factory(tools):
        return create_react_agent(llm, tools, checkpointer=checkpointer)

    factory.checkpointer = checkpointer
    return factory


async def run_langgraph_agent(
    agent_id: str,
    api_key: str,
    rest_url: str,
    ws_url: str,
    custom_section: str,
    logger: logging.Logger,
):
    """Run the LangGraph agent."""
    from thenvoi_langgraph_agent import ThenvoiLangGraphAgent

    agent = ThenvoiLangGraphAgent(
        graph_factory=create_langgraph_factory(),
        agent_id=agent_id,
        api_key=api_key,
        ws_url=ws_url,
        rest_url=rest_url,
        prompt_template="default",
        custom_section=custom_section,
    )

    logger.info("Starting LangGraph agent...")
    await agent.run()


async def run_pydantic_ai_agent(
    agent_id: str,
    api_key: str,
    rest_url: str,
    ws_url: str,
    model: str,
    custom_section: str,
    logger: logging.Logger,
):
    """Run the Pydantic AI agent."""
    from thenvoi_pydantic_agent import ThenvoiPydanticAgent

    agent = ThenvoiPydanticAgent(
        model=model,
        agent_id=agent_id,
        api_key=api_key,
        ws_url=ws_url,
        rest_url=rest_url,
        custom_section=custom_section,
    )

    logger.info(f"Starting Pydantic AI agent with model: {model}")
    await agent.run()


async def run_anthropic_agent(
    agent_id: str,
    api_key: str,
    rest_url: str,
    ws_url: str,
    model: str,
    custom_section: str,
    logger: logging.Logger,
):
    """Run the Anthropic SDK agent."""
    from thenvoi_anthropic_agent import ThenvoiAnthropicAgent

    agent = ThenvoiAnthropicAgent(
        model=model,
        agent_id=agent_id,
        api_key=api_key,
        ws_url=ws_url,
        rest_url=rest_url,
        custom_section=custom_section,
    )

    logger.info(f"Starting Anthropic agent with model: {model}")
    await agent.run()


async def run_claude_sdk_agent(
    agent_id: str,
    api_key: str,
    rest_url: str,
    ws_url: str,
    model: str,
    custom_section: str,
    enable_thinking: bool,
    logger: logging.Logger,
):
    """Run the Claude Agent SDK agent."""
    from thenvoi_claude_sdk_agent import ThenvoiClaudeSDKAgent

    agent = ThenvoiClaudeSDKAgent(
        model=model,
        agent_id=agent_id,
        api_key=api_key,
        ws_url=ws_url,
        rest_url=rest_url,
        custom_section=custom_section,
        max_thinking_tokens=10000 if enable_thinking else None,
    )

    thinking_str = " with extended thinking" if enable_thinking else ""
    logger.info(f"Starting Claude SDK agent with model: {model}{thinking_str}")
    await agent.run()


async def main():
    parser = argparse.ArgumentParser(
        description="Run a Thenvoi SDK test agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # LangGraph with test_agent
  %(prog)s --example pydantic_ai              # Pydantic AI with OpenAI
  %(prog)s --example pydantic_ai --model anthropic:claude-3-5-sonnet-latest
  %(prog)s --agent my_custom_agent            # Use different agent config
  %(prog)s --log-level DEBUG                  # Enable debug logging
        """,
    )
    parser.add_argument(
        "--example",
        "-e",
        choices=["langgraph", "pydantic_ai", "anthropic", "claude_sdk"],
        default="langgraph",
        help="Which example agent to run (default: langgraph)",
    )
    parser.add_argument(
        "--agent",
        "-g",
        default="test_agent",
        help="Agent key from agent_config.yaml (default: test_agent)",
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

    args = parser.parse_args()

    logger = setup_logging(args.log_level)

    # Load URLs from environment
    rest_url = os.getenv("THENVOI_REST_API_URL")
    ws_url = os.getenv("THENVOI_WS_URL")

    if not rest_url:
        parser.error("THENVOI_REST_API_URL environment variable is required")
    if not ws_url:
        parser.error("THENVOI_WS_URL environment variable is required")

    # Load agent credentials
    try:
        agent_id, api_key = load_agent_config(args.agent)
    except Exception as e:
        parser.error(f"Failed to load agent config '{args.agent}': {e}")

    logger.info(f"Agent: {args.agent} ({agent_id})")
    logger.info(f"Example: {args.example}")
    logger.info(f"REST URL: {rest_url}")
    logger.info(f"WS URL: {ws_url}")

    try:
        if args.example == "langgraph":
            await run_langgraph_agent(
                agent_id=agent_id,
                api_key=api_key,
                rest_url=rest_url,
                ws_url=ws_url,
                custom_section=args.custom_section,
                logger=logger,
            )
        elif args.example == "pydantic_ai":
            await run_pydantic_ai_agent(
                agent_id=agent_id,
                api_key=api_key,
                rest_url=rest_url,
                ws_url=ws_url,
                model=args.model,
                custom_section=args.custom_section,
                logger=logger,
            )
        elif args.example == "anthropic":
            # For Anthropic example, use claude model format if default model is still set
            model = args.model
            if model == "openai:gpt-4o":
                model = "claude-sonnet-4-5-20250929"
            await run_anthropic_agent(
                agent_id=agent_id,
                api_key=api_key,
                rest_url=rest_url,
                ws_url=ws_url,
                model=model,
                custom_section=args.custom_section,
                logger=logger,
            )
        elif args.example == "claude_sdk":
            # For Claude SDK example, use claude model format if default model is still set
            model = args.model
            if model == "openai:gpt-4o":
                model = "claude-sonnet-4-5-20250929"
            await run_claude_sdk_agent(
                agent_id=agent_id,
                api_key=api_key,
                rest_url=rest_url,
                ws_url=ws_url,
                model=model,
                custom_section=args.custom_section,
                enable_thinking=args.thinking,
                logger=logger,
            )
    except KeyboardInterrupt:
        logger.info("Interrupted by user")


if __name__ == "__main__":
    asyncio.run(main())
