"""
Example showing how to add custom tools to a Thenvoi agent.

The new architecture makes it trivial to add your own tools alongside
the platform tools.
"""

import asyncio
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from setup_logging import setup_logging
from thenvoi.integrations.langgraph import create_langgraph_agent
from thenvoi.config import load_agent_config

setup_logging()


# Define custom tools
@tool
def calculate(operation: str, left: float, right: float) -> str:
    """Perform a mathematical calculation safely.

    Args:
        operation: The operation to perform: "add", "subtract", "multiply", "divide", or "power"
        left: The first number
        right: The second number
    """
    try:
        if operation == "add":
            result = left + right
        elif operation == "subtract":
            result = left - right
        elif operation == "multiply":
            result = left * right
        elif operation == "divide":
            if right == 0:
                return "Error: Cannot divide by zero"
            result = left / right
        elif operation == "power":
            result = left**right
        else:
            return f"Error: Unknown operation '{operation}'. Use: add, subtract, multiply, divide, or power"

        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"


@tool
def get_weather(city: str) -> str:
    """Get weather for a city (mock implementation).

    Args:
        city: Name of the city
    """
    # In real implementation, call weather API
    return f"Weather in {city}: Sunny, 72°F"


async def main():
    load_dotenv()

    ws_url = os.getenv("THENVOI_WS_URL")
    thenvoi_restapi_url = os.getenv("THENVOI_REST_API_URL")

    if not ws_url:
        raise ValueError("THENVOI_WS_URL environment variable is required")
    if not thenvoi_restapi_url:
        raise ValueError("THENVOI_REST_API_URL environment variable is required")

    # Load agent credentials from agent_config.yaml
    agent_id, api_key = load_agent_config("custom_tools_agent")

    # Create agent with custom tools
    await create_langgraph_agent(
        agent_id=agent_id,
        api_key=api_key,
        llm=ChatOpenAI(model="gpt-4o"),
        checkpointer=InMemorySaver(),
        ws_url=ws_url,
        thenvoi_restapi_url=thenvoi_restapi_url,
        additional_tools=[calculate, get_weather],  # Add your tools here
        custom_instructions="""You are a helpful assistant with access to:
        - Platform tools (send_message, add_participant, etc.)
        - Calculator tool for math
        - Weather tool for weather info

        When users ask math questions, use the calculator.
        When users ask about weather, use get_weather.
        Always send your response using send_message.""",
    )


if __name__ == "__main__":
    asyncio.run(main())
