"""
Example: Custom Agent with System Instructions using LangGraph's create_agent()

This example shows how to use LangGraph's create_agent() helper to build
an agent with custom system instructions and connect it to Thenvoi.

Use this approach when:
- You want to customize the agent's behavior with system prompts
- You want simpler setup than manually building the graph
- You want to leverage the platform's default instructions

Compare with:
- 21_custom_graph.py: Manual graph construction (full control)

Key difference: create_agent() handles graph construction, you just provide
LLM, tools, and system prompt.
"""

import asyncio
import logging
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from setup_logging import setup_logging
from thenvoi.agent.langgraph import connect_graph_to_platform, get_thenvoi_tools
from thenvoi.agent.langgraph.prompts import generate_langgraph_agent_prompt
from thenvoi.agent.core import ThenvoiPlatformClient
from thenvoi.config import load_agent_config

setup_logging()

logger = logging.getLogger(__name__)


async def main():
    load_dotenv()
    ws_url = os.getenv("THENVOI_WS_URL")
    thenvoi_restapi_url = os.getenv("THENVOI_REST_API_URL")

    if not ws_url or not thenvoi_restapi_url:
        raise ValueError("THENVOI_WS_URL and THENVOI_REST_API_URL are required")

    # Load agent credentials
    agent_id, api_key = load_agent_config("custom_agent_with_instructions")

    logger.info("=" * 60)
    logger.info("STEP 1: Create Platform Client")
    logger.info("=" * 60)

    # Create platform client - single client for all operations
    platform_client = ThenvoiPlatformClient(
        agent_id=agent_id,
        api_key=api_key,
        ws_url=ws_url,
        thenvoi_restapi_url=thenvoi_restapi_url,
    )

    logger.info("✓ Platform client created")
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 2: Fetch Agent Metadata")
    logger.info("=" * 60)

    # Fetch agent metadata to get the agent's name
    await platform_client.fetch_agent_metadata()

    logger.info(f"✓ Agent name: {platform_client.name}")
    logger.info(f"✓ Agent description: {platform_client.description}")
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 3: Get Thenvoi Platform Tools")
    logger.info("=" * 60)

    # Get platform tools using the same client
    platform_tools = get_thenvoi_tools(
        client=platform_client.api_client, agent_id=agent_id
    )

    logger.info(f"✓ Got {len(platform_tools)} platform tools:")
    for tool in platform_tools:
        logger.info(f"  - {tool.name}: {tool.description}")

    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 4: Generate Custom System Instructions")
    logger.info("=" * 60)

    # Get the platform's default system instructions
    # These include critical rules about:
    # - How to respond using send_message
    # - How to handle mentions
    # - How to manage room participants
    # - Multi-agent coordination
    #
    # Use the agent's name from the platform
    system_instructions = generate_langgraph_agent_prompt(platform_client.name)

    logger.info("✓ Generated system instructions from prompts.py")
    logger.info(f"✓ Using agent name: {platform_client.name}")
    logger.info(
        f"✓ Instructions include {len(system_instructions)} characters of guidance"
    )
    logger.info("")
    logger.info("Key instructions included:")
    logger.info("  - MUST use send_message to respond to users")
    logger.info("  - Proper mention formatting with @username")
    logger.info("  - Room participant management")
    logger.info("  - Multi-agent coordination rules")

    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 5: Build Agent Graph using create_agent()")
    logger.info("=" * 60)

    # Create LLM
    llm = ChatOpenAI(model="gpt-4o")

    # Use LangGraph's create_agent helper
    # This automatically builds the graph with:
    # - Agent node (LLM with tools and system prompt)
    # - Tool node (executes tools)
    # - Conditional edges (routing logic)
    #
    # The system_prompt parameter sets the system instructions
    my_graph = create_agent(
        model=llm,
        tools=platform_tools,
        system_prompt=system_instructions,  # <-- Custom system instructions!
        checkpointer=InMemorySaver(),
    )

    logger.info("✓ Graph created with create_agent()")
    logger.info("✓ Custom system prompt configured via system_prompt parameter")
    logger.info("✓ Platform tools integrated")
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 6: Connect Graph to Platform")
    logger.info("=" * 60)

    # Connect your graph to Thenvoi using the same platform_client
    await connect_graph_to_platform(
        graph=my_graph,
        platform_client=platform_client,
    )

    logger.info("✓ Connected! Agent features:")
    logger.info("  - Receives chat messages")
    logger.info("  - Follows Thenvoi platform instructions")
    logger.info("  - Uses platform tools with proper mention formatting")
    logger.info("  - Manages multi-participant rooms")
    logger.info("")
    logger.info("The agent is now running and will:")
    logger.info("  1. Always respond using send_message tool")
    logger.info("  2. Include @mentions in messages")
    logger.info("  3. Handle room participants correctly")
    logger.info("  4. Follow all platform coordination rules")


if __name__ == "__main__":
    asyncio.run(main())
