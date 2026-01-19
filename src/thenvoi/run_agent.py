"""Run agent from YAML config."""
import asyncio
import logging
import os
import sys
from pathlib import Path
import yaml
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_tools(names, config_path):
    """Load tools from tools/ folder next to config."""
    if not names:
        return []
    
    # Tools are in the same folder as the config file
    config_dir = Path(config_path).parent
    sys.path.insert(0, str(config_dir))
    
    try:
        from tools import TOOL_REGISTRY
        loaded = [TOOL_REGISTRY[n] for n in names if n in TOOL_REGISTRY]
        missing = [n for n in names if n not in TOOL_REGISTRY]
        if missing:
            logger.warning(f"Tools not found: {missing}")
        return loaded
    except ImportError as e:
        logger.warning(f"Could not import tools: {e}")
        return []


async def main():
    load_dotenv()
    
    config_path = os.getenv("AGENT_CONFIG", "user_agents/example_agent.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loading: {config_path}")
    
    from thenvoi import Agent
    from thenvoi.adapters import ClaudeSDKAdapter
    
    tools = load_tools(config.get("tools", []), config_path)
    
    adapter = ClaudeSDKAdapter(
        model=config.get("model", "claude-sonnet-4-5-20250929"),
        custom_section=config.get("prompt"),
        custom_tools=tools if tools else None,
        max_thinking_tokens=config.get("thinking_tokens"),
        enable_execution_reporting=True,
    )
    
    agent = Agent.create(
        adapter=adapter,
        agent_id=config["agent_id"],
        api_key=config["api_key"],
        ws_url=os.getenv("THENVOI_WS_URL"),
        rest_url=os.getenv("THENVOI_REST_URL"),
    )
    
    logger.info(f"Agent running ({len(tools)} tools). Ctrl+C to stop.")
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
