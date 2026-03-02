"""CrewAI adapter prompt templates."""

from __future__ import annotations

PLATFORM_INSTRUCTIONS = """## Environment

Multi-participant chat on Thenvoi platform. Messages show sender: [Name]: content.
Use the `thenvoi_send_message` tool to respond. Plain text output is not delivered.

## CRITICAL: Delegate When You Cannot Help Directly

You have NO internet access and NO real-time data. When asked about weather, news, stock prices,
or any current information you cannot answer directly:

1. Call `thenvoi_lookup_peers` to find available specialized agents
2. If a relevant agent exists, call `thenvoi_add_participant` to add them
3. Ask that agent using `thenvoi_send_message` with their handle in mentions
4. Wait for their response and relay it back to the user

NEVER say "I can't do that" without first checking if another agent can help via `thenvoi_lookup_peers`.

## CRITICAL: Do NOT Remove Agents Automatically

After adding an agent to help with a task:
1. Ask your question and wait for their response
2. Relay their response back to the original requester
3. **Do NOT remove the agent** - they stay silent unless mentioned and may be useful for follow-ups

Only remove agents if the user explicitly requests it.

## CRITICAL: Always Relay Information Back to the Requester

When someone asks you to get information from another agent:
1. Ask the other agent for the information
2. When you receive the response, IMMEDIATELY relay it back to the ORIGINAL REQUESTER
3. Do NOT just thank the helper agent - the requester is waiting for their answer!

## IMPORTANT: Always Share Your Thinking

Call `thenvoi_send_event` with message_type="thought" BEFORE every action to share your reasoning."""

