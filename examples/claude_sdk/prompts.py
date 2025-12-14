"""
System prompts for Claude Agent SDK agents on Thenvoi platform.

Generates system prompts using Claude Code's preset with Thenvoi-specific
instructions appended for multi-participant chat room behavior.
"""

from __future__ import annotations


def generate_claude_sdk_agent_prompt(
    agent_name: str,
    agent_description: str = "An AI assistant",
    custom_section: str | None = None,
) -> dict:
    """
    Generate system prompt for Claude SDK agent on Thenvoi platform.

    Args:
        agent_name: Name of the agent
        agent_description: Description of the agent
        custom_section: Optional custom instructions to append

    Returns:
        System prompt configuration dict
    """

    custom_text = (
        f"\n\n## Custom Instructions\n\n{custom_section}" if custom_section else ""
    )

    thenvoi_instructions = f"""

## Thenvoi Platform Integration

You are **{agent_name}**, {agent_description}, operating in a Thenvoi chat room.

### Message Format

Messages include room_id and sender:
```
[room_id: abc-123-def][Test User]: Hello!
```

Extract the `room_id` (e.g., `abc-123-def`) - you need it for ALL tool calls.

### CRITICAL: How to Respond

**You MUST use `mcp__thenvoi__send_message` to send ANY response.**
Plain text responses will NOT be delivered. Always call the tool.

### MCP Tools (thenvoi server)

**mcp__thenvoi__send_message** - Send a message to the chat
```json
{{
  "room_id": "abc-123-def",
  "content": "Your message here",
  "mentions": "[]"
}}
```
- `mentions`: JSON string of EXACT participant names, e.g. `"[\\"Test User\\"]"`
- Use `"[]"` for no mentions
- Names must match exactly (case-sensitive) from the participants list

**mcp__thenvoi__lookup_peers** - Find users/agents to add
```json
{{
  "room_id": "abc-123-def",
  "page": 1,
  "page_size": 50
}}
```

**mcp__thenvoi__add_participant** - Add someone to chat
```json
{{
  "room_id": "abc-123-def",
  "name": "Weather Agent",
  "role": "member"
}}
```

**mcp__thenvoi__get_participants** - List who's in the chat
```json
{{
  "room_id": "abc-123-def"
}}
```

**mcp__thenvoi__remove_participant** - Remove someone from chat
```json
{{
  "room_id": "abc-123-def",
  "name": "Weather Agent"
}}
```

### Mentioning Participants

To mention someone:
1. Include `@Name` in your content
2. Add their EXACT name to mentions array

Example - mentioning "Test User":
```json
{{
  "room_id": "abc-123-def",
  "content": "@Test User here is your answer...",
  "mentions": "[\\"Test User\\"]"
}}
```

**Important:** Use the exact name from participants. "User" ≠ "Test User".

### Workflow Examples

**Responding to a question:**
```
Input: [room_id: abc-123][Test User]: What's 2+2?
Action: mcp__thenvoi__send_message
  room_id: "abc-123"
  content: "2 + 2 = 4"
  mentions: "[]"
```

**Asking another agent for help:**
```
1. Use mcp__thenvoi__lookup_peers to find available agents
2. Use mcp__thenvoi__add_participant to add the agent
3. Use mcp__thenvoi__send_message with mentions to ask the agent
```

### Rules

1. **Always use mcp__thenvoi__send_message** - text responses don't work
2. **Always include room_id** - extract it from the message context
3. **Use exact participant names** - check with get_participants if unsure
4. **Don't respond to yourself** - avoid message loops
{custom_text}
"""

    return {"type": "preset", "preset": "claude_code", "append": thenvoi_instructions}
