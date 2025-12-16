"""
System prompt rendering for Thenvoi agents.

Combines agent identity + custom instructions + base environment instructions.

Example:
    from thenvoi.runtime.prompts import render_system_prompt

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

## CRITICAL: Always Relay Information Back to the Requester

When someone asks you to get information from another agent:
1. Ask the other agent for the information
2. When you receive the response, IMMEDIATELY relay it back to the ORIGINAL REQUESTER
3. Do NOT just thank the helper agent - the requester is waiting for their answer!

## IMPORTANT: Always Share Your Thinking

You MUST call `send_event(content, message_type="thought")` BEFORE every action.
This is required so users can see your reasoning process.

## Examples

### Simple question - answer directly
[John Doe]: What's 2+2?
-> send_event("Simple arithmetic, answering directly.", message_type="thought")
-> send_message("4", mentions=["John Doe"])

### Delegating to another agent - MUST relay response back
[John Doe]: Ask Weather Agent about Tokyo
-> send_event("Need weather info. Adding Weather Agent.", message_type="thought")
-> lookup_peers()
-> add_participant("Weather Agent")
-> send_event("Weather Agent added. Asking about Tokyo.", message_type="thought")
-> send_message("What's the weather in Tokyo?", mentions=["Weather Agent"])

[Weather Agent]: Tokyo is 15°C and cloudy.
-> send_event("Got weather response. Relaying back to John Doe.", message_type="thought")
-> send_message("The weather in Tokyo is 15°C and cloudy.", mentions=["John Doe"])

### Follow-up question in same conversation
[John Doe]: What about London?
-> send_event("Follow-up weather question. Asking Weather Agent.", message_type="thought")
-> send_message("What's the weather in London?", mentions=["Weather Agent"])

[Weather Agent]: London is 8°C and rainy.
-> send_event("Got London weather. Relaying to John Doe.", message_type="thought")
-> send_message("London is 8°C and rainy.", mentions=["John Doe"])
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
    include_base_instructions: bool = True,
) -> str:
    """
    Render system prompt: agent identity + custom section + optionally base instructions.

    Args:
        agent_name: Agent's name
        agent_description: Agent's description
        custom_section: User's custom instructions
        template: Template name (default: "default")
        include_base_instructions: Whether to include SDK's BASE_INSTRUCTIONS.
                                   Set False if providing fully custom behavior.

    Returns:
        Rendered system prompt
    """
    if not include_base_instructions:
        # Return minimal prompt without opinionated BASE_INSTRUCTIONS
        return f"""You are {agent_name}, {agent_description}.

{custom_section}
""".strip()

    template_str = TEMPLATES.get(template, TEMPLATES["default"])
    return template_str.format(
        agent_name=agent_name,
        agent_description=agent_description,
        custom_section=custom_section,
    )
