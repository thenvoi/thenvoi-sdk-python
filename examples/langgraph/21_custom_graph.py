"""
Example: Connect Your Own Custom LangGraph to Thenvoi

This example shows how to bring your own LangGraph and connect it to the
Thenvoi platform using connect_graph_to_platform().

Use this approach when you want full control over your agent's graph architecture.

Architecture:
1. Get Thenvoi platform tools (send_message, add_participant, etc.)
2. Build your custom LangGraph with those tools
3. Connect your graph to Thenvoi platform

Compare with:
- 20_custom_agent_with_instructions.py: Use create_agent() helper (simpler setup)
- 02_custom_tools.py: Start with Thenvoi's built-in agent, add your custom tools
"""

import asyncio
import logging
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver

from setup_logging import setup_logging
from thenvoi.agent.langgraph import connect_graph_to_platform, get_thenvoi_tools
from thenvoi.agent.langgraph.prompts import generate_langgraph_agent_prompt
from thenvoi.agent.core import ThenvoiPlatformClient
from thenvoi.config import load_agent_config

setup_logging()

logger = logging.getLogger(__name__)


def create_agent_graph_with_platform_tools(platform_tools, system_instructions: str):
    """
    Create a LangGraph that uses Thenvoi platform tools with custom system instructions.

    This is the standard LangGraph pattern:
    1. LLM node - decides what to do (with system prompt)
    2. Tool node - executes tools
    3. Conditional edges - route based on LLM's decision

    Args:
        platform_tools: List of Thenvoi tools (send_message, add_participant, etc.)
        system_instructions: System prompt for the agent
    """

    # Create LLM and bind tools to it
    llm = ChatOpenAI(model="gpt-4o")
    llm_with_tools = llm.bind_tools(platform_tools)

    # Define the agent node (LLM decides what to do)
    async def agent_node(state: MessagesState):
        """LLM node - analyzes messages and decides which tools to call."""
        messages = state["messages"]

        # Add system message at the beginning if not already present
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_instructions)] + messages

        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}

    # Define conditional routing
    def should_continue(state: MessagesState):
        """
        Decides whether to call tools or end.

        If LLM wants to call a tool -> go to "tools"
        Otherwise -> end
        """
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        return "end"

    # Build the graph
    graph = StateGraph(MessagesState)

    # Add nodes
    graph.add_node("agent", agent_node)
    # THIS is where platform tools are used! ToolNode executes whatever tool the LLM chose
    graph.add_node("tools", ToolNode(platform_tools))

    # Define edges
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",  # LLM wants to call a tool
            "end": END,  # LLM is done
        },
    )
    graph.add_edge("tools", "agent")  # After tool runs, go back to LLM

    # Compile with checkpointer
    return graph.compile(checkpointer=InMemorySaver())


async def main():
    load_dotenv()
    ws_url = os.getenv("THENVOI_WS_URL")
    thenvoi_restapi_url = os.getenv("THENVOI_REST_API_URL")

    if not ws_url or not thenvoi_restapi_url:
        raise ValueError("THENVOI_WS_URL and THENVOI_REST_API_URL are required")

    # Load agent credentials
    agent_id, api_key = load_agent_config("custom_graph_agent")

    logger.info("=" * 60)
    logger.info("STEP 1: Create Platform Client")
    logger.info("=" * 60)

    # Create platform client - this will be shared across all components
    platform_client = ThenvoiPlatformClient(
        agent_id=agent_id,
        api_key=api_key,
        ws_url=ws_url,
        thenvoi_restapi_url=thenvoi_restapi_url,
    )

    logger.info("✓ Platform client created (single client for all operations)")
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

    # Generate system instructions with agent's name from platform
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
    logger.info("STEP 5: Build Your Graph with Tools and Instructions")
    logger.info("=" * 60)

    # Build your graph, passing the tools and system instructions
    my_graph = create_agent_graph_with_platform_tools(
        platform_tools, system_instructions
    )

    logger.info("✓ Graph built with platform tools integrated")
    logger.info("✓ System instructions added to agent node")
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 6: Connect Graph to Platform (Reusing Same Client)")
    logger.info("=" * 60)

    # Connect your graph to Thenvoi using the same platform_client
    await connect_graph_to_platform(
        graph=my_graph,
        platform_client=platform_client,  # Reuse the same client!
    )

    logger.info("✓ Connected! Agent can now:")
    logger.info("  - Receive chat messages")
    logger.info("  - Follow Thenvoi platform instructions")
    logger.info("  - Use platform tools with proper mention formatting")
    logger.info("  - Manage multi-participant rooms")


if __name__ == "__main__":
    asyncio.run(main())
