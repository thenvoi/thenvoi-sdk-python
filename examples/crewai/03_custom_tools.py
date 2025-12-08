"""
CrewAI agent with custom tools.

This example shows how to add your own custom tools to the agent
in addition to the platform tools.
"""

import asyncio
import os
from typing import Type
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from crewai.tools import BaseTool

from setup_logging import setup_logging
from thenvoi.agent.crewai import create_crewai_agent
from thenvoi.config import load_agent_config

setup_logging()


# Example: Custom calculator tool
class CalculatorInput(BaseModel):
    """Input for calculator."""

    operation: str = Field(description="The operation: add, subtract, multiply, divide")
    a: float = Field(description="First number")
    b: float = Field(description="Second number")


class CalculatorTool(BaseTool):
    name: str = "calculator"
    description: str = """Perform basic math operations.
    
    Args:
        operation: The operation (add, subtract, multiply, divide)
        a: First number
        b: Second number
    
    Returns:
        The result of the calculation
    """
    args_schema: Type[BaseModel] = CalculatorInput

    def _run(self, operation: str, a: float, b: float) -> str:
        if operation == "add":
            result = a + b
        elif operation == "subtract":
            result = a - b
        elif operation == "multiply":
            result = a * b
        elif operation == "divide":
            if b == 0:
                return "Error: Division by zero"
            result = a / b
        else:
            return f"Error: Unknown operation '{operation}'"
        
        return f"Result: {a} {operation} {b} = {result}"


# Example: Weather lookup tool (mock)
class WeatherInput(BaseModel):
    """Input for weather lookup."""

    city: str = Field(description="The city to get weather for")


class WeatherTool(BaseTool):
    name: str = "get_weather"
    description: str = """Get current weather for a city.
    
    Args:
        city: The city name
    
    Returns:
        Current weather conditions
    """
    args_schema: Type[BaseModel] = WeatherInput

    def _run(self, city: str) -> str:
        # Mock weather data - in production, call a real API
        mock_weather = {
            "new york": "72°F, Partly Cloudy",
            "london": "59°F, Rainy",
            "tokyo": "68°F, Clear",
            "paris": "64°F, Overcast",
        }
        
        weather = mock_weather.get(city.lower(), f"Weather data not available for {city}")
        return f"Current weather in {city}: {weather}"


async def main():
    load_dotenv()

    ws_url = os.getenv("THENVOI_WS_URL")
    thenvoi_restapi_url = os.getenv("THENVOI_REST_API_URL")

    if not ws_url:
        raise ValueError("THENVOI_WS_URL environment variable is required")
    if not thenvoi_restapi_url:
        raise ValueError("THENVOI_REST_API_URL environment variable is required")

    # Load agent credentials from agent_config.yaml
    agent_id, api_key = load_agent_config("tools_crewai_agent")

    # Create custom tools
    custom_tools = [
        CalculatorTool(),
        WeatherTool(),
    ]

    # Create agent with custom tools
    await create_crewai_agent(
        agent_id=agent_id,
        api_key=api_key,
        llm=ChatOpenAI(model="gpt-4o"),
        ws_url=ws_url,
        thenvoi_restapi_url=thenvoi_restapi_url,
        additional_tools=custom_tools,
        custom_instructions="""
        You have access to a calculator and weather lookup tools.
        Use them when users ask for calculations or weather information.
        Always use the send_message tool to communicate your findings to users.
        """,
    )


if __name__ == "__main__":
    asyncio.run(main())

