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
    uv run python examples/run_agent.py --example claude_sdk --streaming  # With tool_call/tool_result events
    uv run python examples/run_agent.py --example claude_sdk --thinking   # Enable extended thinking
    uv run python examples/run_agent.py --example parlant
    uv run python examples/run_agent.py --example crewai
    uv run python examples/run_agent.py --example crewai --streaming  # Show tool calls
    uv run python examples/run_agent.py --example letta                    # Letta PER_ROOM mode
    uv run python examples/run_agent.py --example letta_shared             # Letta SHARED mode
    uv run python examples/run_agent.py --example letta --letta-url http://localhost:8283
    uv run python examples/run_agent.py --example a2a --a2a-url http://localhost:10000  # A2A bridge
    uv run python examples/run_agent.py --example a2a_gateway              # A2A Gateway (exposes peers)
    uv run python examples/run_agent.py --example a2a_gateway --gateway-port 8080  # Custom port

Configure agent in agent_config.yaml:
    uv run python examples/run_agent.py --agent test_agent
    uv run python examples/run_agent.py --agent my_custom_agent

Setup:
1. Copy .env.example to .env and configure:
   - THENVOI_REST_URL (default: production, change for local dev)
   - THENVOI_WS_URL (default: production, change for local dev)
   - OPENAI_API_KEY (required for langgraph/openai/parlant/crewai/letta models)
   - ANTHROPIC_API_KEY (required for anthropic models)

2. Configure agent in agent_config.yaml

3. For A2A example, start a remote A2A agent first (e.g., LangGraph currency agent)

4. For A2A Gateway example, the gateway exposes Thenvoi platform peers as A2A endpoints

5. For Letta examples, start a Letta server first:
   docker run -p 8283:8283 --env-file .env letta/letta:latest
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
    logger.info("Starting Pydantic AI agent with model: %s%s", model, streaming_str)
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
    logger.info("Starting Anthropic agent with model: %s%s", model, streaming_str)
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
    logger.info("Starting Claude SDK agent with model: %s%s", model, options_str)
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

    logger.info("Starting Parlant agent with model: %s", model)
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

    logger.info("Starting CrewAI agent with model: %s", model)
    await agent.run()


async def run_a2a_agent(
    agent_id: str,
    api_key: str,
    rest_url: str,
    ws_url: str,
    a2a_url: str,
    enable_debug: bool,
    logger: logging.Logger,
):
    """Run the A2A bridge agent."""
    from thenvoi.adapters import A2AAdapter

    # Enable debug logging for A2A adapter to trace context_id and rehydration
    if enable_debug:
        logging.getLogger("thenvoi.integrations.a2a").setLevel(logging.DEBUG)
        logging.getLogger("thenvoi.converters.a2a").setLevel(logging.DEBUG)

    adapter = A2AAdapter(
        remote_url=a2a_url,
        streaming=True,
    )

    agent = Agent.create(
        adapter=adapter,
        agent_id=agent_id,
        api_key=api_key,
        ws_url=ws_url,
        rest_url=rest_url,
    )

    logger.info("Starting A2A bridge agent (forwarding to %s)...", a2a_url)
    await agent.run()


async def run_letta_agent(
    agent_id: str,
    api_key: str,
    rest_url: str,
    ws_url: str,
    letta_url: str,
    letta_model: str,
    mcp_url: str | None,
    persona: str,
    shared_mode: bool,
    mcp_tools: list[str] | None,
    letta_base_tools: list[str] | None,
    logger: logging.Logger,
):
    """Run the Letta agent."""
    from thenvoi.adapters.letta import LettaAdapter, LettaConfig, LettaMode
    from thenvoi.runtime import run_with_graceful_shutdown
    from thenvoi.runtime.types import SessionConfig

    mode = LettaMode.SHARED if shared_mode else LettaMode.PER_ROOM
    state_file = (
        "letta_shared_state.json" if shared_mode else "letta_per_room_state.json"
    )

    adapter = LettaAdapter(
        config=LettaConfig(
            mode=mode,
            base_url=letta_url,
            mcp_server_url=mcp_url,
            model=letta_model,
            embedding_model="openai/text-embedding-3-small",
            persona=persona,
            mcp_tools=mcp_tools,
            letta_base_tools=letta_base_tools,
        ),
        state_storage_path=f"~/.thenvoi/{state_file}",
    )

    agent = Agent.create(
        adapter=adapter,
        agent_id=agent_id,
        api_key=api_key,
        ws_url=ws_url,
        rest_url=rest_url,
        # IMPORTANT: Letta manages its own conversation history
        session_config=SessionConfig(enable_context_hydration=False),
    )

    mode_str = "SHARED" if shared_mode else "PER_ROOM"
    logger.info(f"Starting Letta agent in {mode_str} mode")
    logger.info(f"  Letta server: {letta_url}")
    logger.info(f"  Model: {letta_model}")
    if mcp_url:
        logger.info(f"  MCP server: {mcp_url}")
    else:
        logger.info("  MCP server: Not configured (tools will use stubs)")
    if shared_mode:
        logger.info("  Architecture: One agent serves all rooms via conversations")
    else:
        logger.info("  Architecture: Each room gets its own dedicated agent")
    logger.info("  Press Ctrl+C once for graceful shutdown (waits for cleanup)")

    # Use graceful shutdown to allow memory consolidation on exit
    await run_with_graceful_shutdown(agent, timeout=60.0)


async def run_a2a_gateway_agent(
    agent_id: str,
    api_key: str,
    rest_url: str,
    ws_url: str,
    gateway_port: int,
    enable_debug: bool,
    logger: logging.Logger,
):
    """Run the A2A Gateway agent.

    The gateway connects to Thenvoi platform and exposes discovered peers
    as A2A endpoints. External A2A agents can call these peers via standard
    A2A protocol.
    """
    from thenvoi.adapters import A2AGatewayAdapter

    # Enable debug logging for gateway adapter
    if enable_debug:
        logging.getLogger("thenvoi.integrations.a2a.gateway").setLevel(logging.DEBUG)

    gateway_url = f"http://localhost:{gateway_port}"

    adapter = A2AGatewayAdapter(
        rest_url=rest_url,
        api_key=api_key,
        gateway_url=gateway_url,
        port=gateway_port,
    )

    agent = Agent.create(
        adapter=adapter,
        agent_id=agent_id,
        api_key=api_key,
        ws_url=ws_url,
        rest_url=rest_url,
    )

    logger.info("Starting A2A Gateway on %s...", gateway_url)
    logger.info("Peers will be exposed at:")
    logger.info(
        "  - %s/agents/{peer_id}/.well-known/agent.json (discovery)", gateway_url
    )
    logger.info("  - %s/agents/{peer_id}/v1/message:stream (messaging)", gateway_url)
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
  uv run python examples/run_agent.py --example parlant --streaming       # With tool visibility
  uv run python examples/run_agent.py --example crewai                    # CrewAI adapter
  uv run python examples/run_agent.py --example crewai --streaming        # With tool visibility
  uv run python examples/run_agent.py --example letta                     # Letta PER_ROOM mode (agent per room)
  uv run python examples/run_agent.py --example letta_shared              # Letta SHARED mode (one agent, many rooms)
  uv run python examples/run_agent.py --example letta --letta-url http://localhost:8283
  uv run python examples/run_agent.py --example letta --letta-model openai/gpt-4o
  uv run python examples/run_agent.py --example a2a                       # A2A bridge (default: localhost:10000)
  uv run python examples/run_agent.py --example a2a --debug               # A2A with debug logging (context_id tracing)
  uv run python examples/run_agent.py --example a2a --a2a-url http://remote:8080  # A2A with custom URL
  uv run python examples/run_agent.py --example a2a_gateway               # A2A Gateway (exposes peers)
  uv run python examples/run_agent.py --example a2a_gateway --debug       # A2A Gateway with debug logging
  uv run python examples/run_agent.py --example a2a_gateway --gateway-port 8080  # Custom gateway port
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
            "letta",
            "letta_shared",
            "a2a",
            "a2a_gateway",
        ],
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
        "--letta-url",
        default=os.getenv("LETTA_BASE_URL", "http://localhost:8283"),
        help="Letta server URL (default: http://localhost:8283 or LETTA_BASE_URL env var)",
    )
    parser.add_argument(
        "--letta-model",
        default=os.getenv("LETTA_MODEL", "openai/gpt-4o"),
        help="Letta LLM model (default: openai/gpt-4o or LETTA_MODEL env var)",
    )
    parser.add_argument(
        "--mcp-url",
        default=os.getenv("MCP_SERVER_URL"),
        help="MCP server URL for Letta tools (e.g., http://localhost:8002/sse)",
    )
    parser.add_argument(
        "--persona",
        default="You are a helpful assistant with persistent memory.",
        help="Agent persona for Letta examples",
    )
    parser.add_argument(
        "--letta-mcp-tools",
        type=str,
        default=None,
        help="Comma-separated MCP tool names for Letta (default: all agent tools)",
    )
    parser.add_argument(
        "--letta-base-tools",
        type=str,
        default=None,
        help="Comma-separated Letta base tools (default: memory,conversation_search,archival_memory_insert,archival_memory_search)",
    )

    args = parser.parse_args()

    logger = setup_logging(args.log_level)

    # Set default agent based on example type if not specified
    default_agents = {
        "langgraph": "simple_agent",
        "pydantic_ai": "pydantic_agent",
        "anthropic": "anthropic_agent",
        "claude_sdk": "anthropic_agent",
        "parlant": "parlant_agent",
        "crewai": "crewai_agent",
        "letta": "letta_agent",
        "letta_shared": "letta_agent",
        "a2a": "a2a_agent",
        "a2a_gateway": "a2a_gateway_agent",
    }
    if args.agent is None:
        args.agent = default_agents.get(args.example, "simple_agent")

    # Load URLs from environment
    rest_url = os.getenv("THENVOI_REST_URL")
    ws_url = os.getenv("THENVOI_WS_URL")

    if not rest_url:
        parser.error("THENVOI_REST_URL environment variable is required")
    if not ws_url:
        parser.error("THENVOI_WS_URL environment variable is required")

    # Load agent credentials
    try:
        agent_id, api_key = load_agent_config(args.agent)
    except Exception as e:
        parser.error(f"Failed to load agent config '{args.agent}': {e}")

    logger.info("Agent: %s (%s)", args.agent, agent_id)
    logger.info("Example: %s", args.example)
    logger.info("REST URL: %s", rest_url)
    logger.info("WS URL: %s", ws_url)

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
        elif args.example in ("letta", "letta_shared"):
            # Parse comma-separated tool lists (filter empty strings)
            mcp_tools = None
            if args.letta_mcp_tools:
                mcp_tools = [
                    t.strip() for t in args.letta_mcp_tools.split(",") if t.strip()
                ]
                mcp_tools = mcp_tools or None  # Convert empty list to None

            letta_base_tools = None
            if args.letta_base_tools:
                letta_base_tools = [
                    t.strip() for t in args.letta_base_tools.split(",") if t.strip()
                ]
                letta_base_tools = letta_base_tools or None

            await run_letta_agent(
                agent_id=agent_id,
                api_key=api_key,
                rest_url=rest_url,
                ws_url=ws_url,
                letta_url=args.letta_url,
                letta_model=args.letta_model,
                mcp_url=args.mcp_url,
                persona=args.persona,
                shared_mode=(args.example == "letta_shared"),
                mcp_tools=mcp_tools,
                letta_base_tools=letta_base_tools,
                logger=logger,
            )
        elif args.example == "a2a":
            await run_a2a_agent(
                agent_id=agent_id,
                api_key=api_key,
                rest_url=rest_url,
                ws_url=ws_url,
                a2a_url=args.a2a_url,
                enable_debug=args.debug,
                logger=logger,
            )
        elif args.example == "a2a_gateway":
            await run_a2a_gateway_agent(
                agent_id=agent_id,
                api_key=api_key,
                rest_url=rest_url,
                ws_url=ws_url,
                gateway_port=args.gateway_port,
                enable_debug=args.debug,
                logger=logger,
            )
    except KeyboardInterrupt:
        logger.info("Interrupted by user")


if __name__ == "__main__":
    asyncio.run(main())
