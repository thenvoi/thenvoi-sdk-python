"""
System prompt rendering for Thenvoi agents.

Combines agent identity + custom instructions + base environment instructions.

Example:
    from thenvoi.agent.core.prompts import render_system_prompt

    prompt = render_system_prompt(
        agent_name="DataBot",
        agent_description="A helpful data analysis assistant",
        custom_section="Focus on Python and pandas.",
    )
"""

from __future__ import annotations


# Base instructions appended to user's custom prompt
BASE_INSTRUCTIONS = """
## Environment

Multi-participant chat. Messages show sender: [Name]: content.
Use `send_message(content, mentions)` to respond. Plain text output is not delivered.

## IMPORTANT: Always Share Your Thinking

You MUST call `send_event(content, message_type="thought")` BEFORE every action.
This is required so users can see your reasoning process.

## Examples

[User]: What's 2+2?
-> send_event("Simple arithmetic, answering directly.", message_type="thought")
-> send_message("4", mentions=["User"])

[User]: Ask Weather Agent about Tokyo
-> send_event("Need weather info. Adding Weather Agent to ask.", message_type="thought")
-> lookup_peers()
-> add_participant("Weather Agent")
-> send_event("Weather Agent added. Asking about Tokyo weather.", message_type="thought")
-> send_message("What's the weather in Tokyo?", mentions=["Weather Agent"])

[Weather Agent]: Tokyo is 15°C and cloudy.
-> send_event("Got weather response. Relaying to user.", message_type="thought")
-> send_message("Tokyo is 15°C and cloudy.", mentions=["User"])
"""

# Single default template - agent identity + custom section + base instructions
TEMPLATES: dict[str, str] = {
    "default": """You are {agent_name}, {agent_description}.

{custom_section}
"""
    + BASE_INSTRUCTIONS,
}


def render_system_prompt(
    agent_name: str = "Agent",
    agent_description: str = "An AI assistant",
    custom_section: str = "",
    template: str = "default",
) -> str:
    """
    Render system prompt: agent identity + custom section + base instructions.

    Args:
        agent_name: Agent's name
        agent_description: Agent's description
        custom_section: User's custom instructions
        template: Template name (default: "default")

    Returns:
        Rendered system prompt
    """
    template_str = TEMPLATES.get(template, TEMPLATES["default"])
    return template_str.format(
        agent_name=agent_name,
        agent_description=agent_description,
        custom_section=custom_section,
    )
