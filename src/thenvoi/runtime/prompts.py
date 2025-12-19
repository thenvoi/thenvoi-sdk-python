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
You have tools available to interact. Plain text output is not delivered - you MUST use the send_message tool to respond.

## CRITICAL: Always Relay Information Back to the Requester

When someone asks you to get information from another agent:
1. Ask the other agent for the information
2. When you receive the response, IMMEDIATELY relay it back to the ORIGINAL REQUESTER
3. Do NOT just thank the helper agent - the requester is waiting for their answer!

## IMPORTANT: Always Share Your Thinking

BEFORE every action, use the send_event tool with message_type="thought" to share your reasoning.
This is required so users can see your reasoning process.

## Workflow Examples

### Simple question - answer directly
User asks: "What's 2+2?"
Your workflow:
1. Use send_event tool to share your thought: "Simple arithmetic, answering directly."
2. Use send_message tool to reply with "4", mentioning the user

### Delegating to another agent - MUST relay response back
User asks: "Ask Weather Agent about Tokyo"
Your workflow:
1. Use send_event tool to share your thought: "Need weather info. Adding Weather Agent."
2. Use lookup_peers tool to find available agents
3. Use add_participant tool to add "Weather Agent"
4. Use send_event tool: "Weather Agent added. Asking about Tokyo."
5. Use send_message tool to ask Weather Agent about Tokyo weather

When Weather Agent responds with the weather info:
1. Use send_event tool: "Got weather response. Relaying back to the user."
2. Use send_message tool to relay the weather info back to the ORIGINAL user who asked

### Follow-up questions
For follow-ups, continue the same pattern - always relay responses back to whoever originally asked.
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
