#!/usr/bin/env python3
"""
Run Thenvoi SDK agents using the composition pattern.

Usage:
    uv run python examples/run_agent.py                    # Default: langgraph
    uv run python examples/run_agent.py --example langgraph
    uv run python examples/run_agent.py --example pydantic_ai
    uv run python examples/run_agent.py --example pydantic_ai --streaming  # With tool_call/tool_result events
    uv run python examples/run_agent.py --example pydantic_ai --model anthropic:claude-sonnet-4-5
    uv run python examples/run_agent.py --example anthropic
    uv run python examples/run_agent.py --example anthropic --streaming  # With tool_call/tool_result events
    uv run python examples/run_agent.py --example anthropic --model claude-sonnet-4-5-20250929
    uv run python examples/run_agent.py --example claude_sdk
    uv run python examples/run_agent.py --example claude_sdk --thinking  # Enable extended thinking
    uv run python examples/run_agent.py --example parlant
    uv run python examples/run_agent.py --example crewai
    uv run python examples/run_agent.py --example crewai --streaming  # Show tool calls

Configure agent in agent_config.yaml:
    uv run python examples/run_agent.py --agent test_agent
    uv run python examples/run_agent.py --agent my_custom_agent

Setup:
1. Copy .env.example to .env and configure:
   - THENVOI_REST_API_URL (default: production, change for local dev)
   - THENVOI_WS_URL (default: production, change for local dev)
   - OPENAI_API_KEY (required for langgraph/openai/parlant/crewai models)
   - ANTHROPIC_API_KEY (required for anthropic models)

2. Configure agent in agent_config.yaml
"""

import argparse
import asyncio
import logging
import os

from dotenv import load_dotenv

from thenvoi import Agent
from thenvoi.config import load_agent_config

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


PYDANTIC_AI_INSTRUCTIONS = """
## CRITICAL: Your Capabilities and Limitations

**You have NO internet access and NO real-time data.**
- You CANNOT look up weather, news, stock prices, or any current information
- You MUST NOT invent or guess factual information like temperatures, prices, or dates
- For real-time data (weather, etc.), you MUST delegate to specialized agents (e.g., Weather Agent)

If you don't know something and can't delegate to another agent, say "I don't know" - never make up information.
"""

PARLANT_GUIDELINES = [
    {
        "condition": "User asks for help",
        "action": "Acknowledge and clarify before helping",
    },
    {"condition": "User says goodbye", "action": "Summarize and offer further help"},
]

CREWAI_DEFAULTS = {
    "role": "Research Assistant",
    "goal": "Help users find, analyze, and synthesize information",
    "backstory": "Expert researcher with attention to detail and ability to break down "
    "complex topics into understandable insights.",
}


async def run_langgraph_agent(
    agent_id: str,
    api_key: str,
    rest_url: str,
    ws_url: str,
    custom_section: str,
    logger: logging.Logger,
):
    """Run the LangGraph agent."""
    from langchain_openai import ChatOpenAI
    from langgraph.checkpoint.memory import InMemorySaver

    from thenvoi.adapters import LangGraphAdapter

    adapter = LangGraphAdapter(
        llm=ChatOpenAI(model="gpt-4o"),
        checkpointer=InMemorySaver(),
        custom_section=custom_section,
    )

    agent = Agent.create(
        adapter=adapter,
        agent_id=agent_id,
        api_key=api_key,
        ws_url=ws_url,
        rest_url=rest_url,
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
    enable_streaming: bool,
    logger: logging.Logger,
):
    """Run the Pydantic AI agent."""
    from thenvoi.adapters import PydanticAIAdapter

    adapter = PydanticAIAdapter(
        model=model,
        custom_section=custom_section,
        enable_execution_reporting=enable_streaming,
    )

    agent = Agent.create(
        adapter=adapter,
        agent_id=agent_id,
        api_key=api_key,
        ws_url=ws_url,
        rest_url=rest_url,
    )

    streaming_str = " with execution reporting" if enable_streaming else ""
    logger.info(f"Starting Pydantic AI agent with model: {model}{streaming_str}")
    await agent.run()


async def run_anthropic_agent(
    agent_id: str,
    api_key: str,
    rest_url: str,
    ws_url: str,
    model: str,
    custom_section: str,
    enable_streaming: bool,
    logger: logging.Logger,
):
    """Run the Anthropic SDK agent."""
    from thenvoi.adapters import AnthropicAdapter

    adapter = AnthropicAdapter(
        model=model,
        custom_section=custom_section,
        enable_execution_reporting=enable_streaming,
    )

    agent = Agent.create(
        adapter=adapter,
        agent_id=agent_id,
        api_key=api_key,
        ws_url=ws_url,
        rest_url=rest_url,
    )

    streaming_str = " with execution reporting" if enable_streaming else ""
    logger.info(f"Starting Anthropic agent with model: {model}{streaming_str}")
    await agent.run()


async def run_claude_sdk_agent(
    agent_id: str,
    api_key: str,
    rest_url: str,
    ws_url: str,
    model: str,
    custom_section: str,
    enable_thinking: bool,
    enable_streaming: bool,
    logger: logging.Logger,
):
    """Run the Claude Agent SDK agent."""
    from thenvoi.adapters import ClaudeSDKAdapter

    adapter = ClaudeSDKAdapter(
        model=model,
        custom_section=custom_section,
        max_thinking_tokens=10000 if enable_thinking else None,
        enable_execution_reporting=enable_streaming,
    )

    agent = Agent.create(
        adapter=adapter,
        agent_id=agent_id,
        api_key=api_key,
        ws_url=ws_url,
        rest_url=rest_url,
    )

    options = []
    if enable_thinking:
        options.append("extended thinking")
    if enable_streaming:
        options.append("execution reporting")
    options_str = f" with {', '.join(options)}" if options else ""
    logger.info(f"Starting Claude SDK agent with model: {model}{options_str}")
    await agent.run()


async def run_parlant_agent(
    agent_id: str,
    api_key: str,
    rest_url: str,
    ws_url: str,
    model: str,
    custom_section: str,
    enable_streaming: bool,
    logger: logging.Logger,
):
    """Run the Parlant agent."""
    from thenvoi.adapters import ParlantAdapter

    adapter = ParlantAdapter(
        model=model,
        custom_section=custom_section,
        guidelines=PARLANT_GUIDELINES,
        enable_execution_reporting=enable_streaming,
    )

    agent = Agent.create(
        adapter=adapter,
        agent_id=agent_id,
        api_key=api_key,
        ws_url=ws_url,
        rest_url=rest_url,
    )

    logger.info(f"Starting Parlant agent with model: {model}")
    await agent.run()


async def run_crewai_agent(
    agent_id: str,
    api_key: str,
    rest_url: str,
    ws_url: str,
    model: str,
    custom_section: str,
    enable_streaming: bool,
    logger: logging.Logger,
):
    """Run the CrewAI agent."""
    from thenvoi.adapters import CrewAIAdapter

    adapter = CrewAIAdapter(
        model=model,
        role=CREWAI_DEFAULTS["role"],
        goal=CREWAI_DEFAULTS["goal"],
        backstory=CREWAI_DEFAULTS["backstory"],
        custom_section=custom_section,
        enable_execution_reporting=enable_streaming,
    )

    agent = Agent.create(
        adapter=adapter,
        agent_id=agent_id,
        api_key=api_key,
        ws_url=ws_url,
        rest_url=rest_url,
    )

    logger.info(f"Starting CrewAI agent with model: {model}")
    await agent.run()


async def main():
    parser = argparse.ArgumentParser(
        description="Run a Thenvoi SDK test agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python examples/run_agent.py                                     # LangGraph (default)
  uv run python examples/run_agent.py --example langgraph                 # LangGraph with OpenAI
  uv run python examples/run_agent.py --example pydantic_ai               # Pydantic AI with OpenAI
  uv run python examples/run_agent.py --example pydantic_ai --streaming   # With tool_call/tool_result events
  uv run python examples/run_agent.py --example pydantic_ai --model anthropic:claude-sonnet-4-5
  uv run python examples/run_agent.py --example anthropic                 # Anthropic SDK
  uv run python examples/run_agent.py --example anthropic --streaming     # With tool_call/tool_result events
  uv run python examples/run_agent.py --example claude_sdk                # Claude Agent SDK
  uv run python examples/run_agent.py --example claude_sdk --streaming    # With tool_call/tool_result events
  uv run python examples/run_agent.py --example claude_sdk --thinking     # With extended thinking
  uv run python examples/run_agent.py --example parlant                   # Parlant adapter
  uv run python examples/run_agent.py --example crewai                    # CrewAI adapter
  uv run python examples/run_agent.py --example anthropic --streaming  # With tool visibility
  uv run python examples/run_agent.py --example claude_sdk --streaming  # With tool visibility
  uv run python examples/run_agent.py --example parlant --streaming     # With tool visibility
  uv run python examples/run_agent.py --example crewai --streaming      # With tool visibility
  uv run python examples/run_agent.py --agent my_custom_agent             # Use different agent config
  uv run python examples/run_agent.py --log-level DEBUG                   # Enable debug logging
        """,
    )
    parser.add_argument(
        "--example",
        "-e",
        choices=[
            "langgraph",
            "pydantic_ai",
            "anthropic",
            "claude_sdk",
            "parlant",
            "crewai",
        ],
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
    parser.add_argument(
        "--streaming",
        "-s",
        action="store_true",
        help="Enable tool call/result visibility for anthropic/claude_sdk/parlant/crewai (default: False)",
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
                enable_streaming=args.streaming,
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
                enable_streaming=args.streaming,
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
                enable_streaming=args.streaming,
                logger=logger,
            )
        elif args.example == "parlant":
            # For Parlant example, use OpenAI model format
            model = args.model
            if model == "openai:gpt-4o":
                model = "gpt-4o"
            await run_parlant_agent(
                agent_id=agent_id,
                api_key=api_key,
                rest_url=rest_url,
                ws_url=ws_url,
                model=model,
                custom_section=args.custom_section,
                enable_streaming=args.streaming,
                logger=logger,
            )
        elif args.example == "crewai":
            # For CrewAI example, use OpenAI model format
            model = args.model
            if model == "openai:gpt-4o":
                model = "gpt-4o"
            await run_crewai_agent(
                agent_id=agent_id,
                api_key=api_key,
                rest_url=rest_url,
                ws_url=ws_url,
                model=model,
                custom_section=args.custom_section,
                enable_streaming=args.streaming,
                logger=logger,
            )
    except KeyboardInterrupt:
        logger.info("Interrupted by user")


if __name__ == "__main__":
    asyncio.run(main())
