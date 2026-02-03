"""Example tools - copy what you need to __init__.py"""

from claude_agent_sdk import tool


@tool("calculator", "Evaluate math expressions", {"expression": str})
async def calculator(args: dict) -> dict:
    """Example: calculator("2 + 2") → "4" """
    try:
        allowed = {"abs": abs, "round": round, "min": min, "max": max, "pow": pow}
        result = eval(args["expression"], {"__builtins__": allowed}, {})
        return {"content": [{"type": "text", "text": str(result)}]}
    except Exception as e:
        return {"content": [{"type": "text", "text": f"Error: {e}"}], "is_error": True}


@tool("get_time", "Get current date/time", {})
async def get_time(args: dict) -> dict:
    """Returns current time in ISO format"""
    from datetime import datetime

    return {"content": [{"type": "text", "text": datetime.now().isoformat()}]}


@tool("random_number", "Generate random number", {"min": int, "max": int})
async def random_number(args: dict) -> dict:
    """Example: random_number(1, 100) → "42" """
    import random

    result = random.randint(args.get("min", 1), args.get("max", 100))
    return {"content": [{"type": "text", "text": str(result)}]}
