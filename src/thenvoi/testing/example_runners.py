"""Framework-specific example runner executors."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from thenvoi import Agent
from thenvoi.adapters.crewai import CrewAIAdapter
from thenvoi.adapters.codex.adapter import CodexAdapter, CodexAdapterConfig
from thenvoi.platform.event import ContactEvent
from thenvoi.runtime.contacts.contact_tools import ContactTools
from thenvoi.runtime.types import ContactEventConfig, ContactEventStrategy

RunnerFunction = Callable[..., Awaitable[None]]
CustomSectionResolver = Callable[[argparse.Namespace, logging.Logger], str]


@dataclass(frozen=True)
class ExampleRunnerSpec:
    """Declarative registry entry for one runnable example."""

    default_agent: str
    runner: RunnerFunction
    uses_model: bool = False
    uses_custom_section: bool = False
    uses_streaming: bool = False
    uses_contact_config: bool = False
    uses_thinking: bool = False
    uses_codex_options: bool = False
    uses_a2a_url: bool = False
    uses_gateway_port: bool = False
    custom_section_resolver: CustomSectionResolver | None = None


def build_example_runner_kwargs(
    *,
    spec: ExampleRunnerSpec,
    args: argparse.Namespace,
    agent_id: str,
    api_key: str,
    rest_url: str,
    ws_url: str,
    model: str,
    contact_config: ContactEventConfig | None,
    logger: logging.Logger,
) -> dict[str, object]:
    """Build runner kwargs from declarative runner capabilities."""
    kwargs: dict[str, object] = {
        "agent_id": agent_id,
        "api_key": api_key,
        "rest_url": rest_url,
        "ws_url": ws_url,
        "logger": logger,
    }

    if spec.uses_model:
        kwargs["model"] = model
    if spec.uses_custom_section:
        section_resolver = spec.custom_section_resolver or (
            lambda parsed_args, _logger: parsed_args.custom_section
        )
        kwargs["custom_section"] = section_resolver(args, logger)
    if spec.uses_streaming:
        kwargs["enable_streaming"] = args.streaming
    if spec.uses_contact_config:
        kwargs["contact_config"] = contact_config
    if spec.uses_thinking:
        kwargs["enable_thinking"] = args.thinking
    if spec.uses_codex_options:
        kwargs["codex_transport"] = args.codex_transport
        kwargs["codex_ws_url"] = args.codex_ws_url
        kwargs["codex_model"] = args.codex_model
        kwargs["codex_personality"] = args.codex_personality
        kwargs["codex_approval_policy"] = args.codex_approval_policy
        kwargs["codex_approval_mode"] = args.codex_approval_mode
        kwargs["codex_turn_task_markers"] = args.codex_turn_task_markers
        kwargs["codex_cwd"] = args.codex_cwd
        kwargs["codex_sandbox"] = args.codex_sandbox
        kwargs["codex_reasoning_effort"] = args.codex_reasoning_effort
    if spec.uses_a2a_url:
        kwargs["a2a_url"] = args.a2a_url
        kwargs["enable_debug"] = args.debug
    if spec.uses_gateway_port:
        kwargs["gateway_port"] = args.gateway_port
        kwargs["enable_debug"] = args.debug

    return kwargs

PYDANTIC_AI_INSTRUCTIONS = """
## CRITICAL: Your Capabilities and Limitations

**You have NO internet access and NO real-time data.**
- You CANNOT look up weather, news, stock prices, or any current information
- You MUST NOT invent or guess factual information like temperatures, prices, or dates
- For real-time data (weather, etc.), you MUST delegate to specialized agents (e.g., Weather Agent)

If you don't know something and can't delegate to another agent, say "I don't know" - never make up information.
"""

PARLANT_GUIDELINES = [
    {
        "condition": "User asks for help",
        "action": "Acknowledge and clarify before helping",
    },
    {"condition": "User says goodbye", "action": "Summarize and offer further help"},
]

CREWAI_DEFAULTS = {
    "role": "Research Assistant",
    "goal": "Help users find, analyze, and synthesize information",
    "backstory": "Expert researcher with attention to detail and ability to break down "
    "complex topics into understandable insights.",
}

# When --model is left at the default "openai:gpt-4o", these examples override
# it to a framework-appropriate model. The user can always pass --model
# explicitly to bypass this.
_DEFAULT_MODELS: dict[str, str] = {
    "pydantic_ai_contacts": "anthropic:claude-sonnet-4-5",
    "contacts_auto": "anthropic:claude-sonnet-4-5",
    "contacts_hub": "anthropic:claude-sonnet-4-5",
    "contacts_broadcast": "anthropic:claude-sonnet-4-5",
    "anthropic": "claude-sonnet-4-5-20250929",
    "claude_sdk": "claude-sonnet-4-5-20250929",
    "parlant": "gpt-4o",
    "crewai": "gpt-4o",
}

CONTACTS_INSTRUCTIONS = """
## Contact Management Assistant

You help manage contacts and contact requests on the Thenvoi platform.

### Available Actions

When the user asks you to manage contacts, use these tools:

1. **List contacts**: Use `thenvoi_list_contacts` to show current contacts
2. **Check requests**: Use `thenvoi_list_contact_requests` to see pending requests
3. **Approve request**: Use `thenvoi_respond_contact_request` with action="approve"
4. **Reject request**: Use `thenvoi_respond_contact_request` with action="reject"
5. **Add contact**: Use `thenvoi_add_contact` to send a contact request
6. **Remove contact**: Use `thenvoi_remove_contact` to remove an existing contact

### Response Format

When listing contacts or requests, format the information clearly:
- Show names and handles
- For requests, show who sent them and any message included
- Confirm actions after completing them

### Examples

User: "check my contact requests"
→ Call thenvoi_list_contact_requests()

User: "approve alice's request"
→ Call thenvoi_respond_contact_request(action="approve", handle="alice")

User: "reject the request from bob"
→ Call thenvoi_respond_contact_request(action="reject", handle="bob")

User: "list my contacts"
→ Call thenvoi_list_contacts()

User: "add john as a contact"
→ Call thenvoi_add_contact(handle="john")
"""


async def run_langgraph_agent(
    agent_id: str,
    api_key: str,
    rest_url: str,
    ws_url: str,
    custom_section: str,
    logger: logging.Logger,
) -> None:
    """Run the LangGraph agent."""
    from langchain_openai import ChatOpenAI
    from langgraph.checkpoint.memory import InMemorySaver

    from thenvoi.adapters import LangGraphAdapter

    adapter = LangGraphAdapter(
        llm=ChatOpenAI(model="gpt-4o"),
        checkpointer=InMemorySaver(),
        custom_section=custom_section,
    )

    agent = Agent.create(
        adapter=adapter,
        agent_id=agent_id,
        api_key=api_key,
        ws_url=ws_url,
        rest_url=rest_url,
    )

    logger.info("Starting LangGraph agent...")
    await agent.run()


async def run_pydantic_ai_agent(
    agent_id: str,
    api_key: str,
    rest_url: str,
    ws_url: str,
    model: str,
    custom_section: str,
    enable_streaming: bool,
    contact_config: ContactEventConfig | None,
    logger: logging.Logger,
) -> None:
    """Run the Pydantic AI agent."""
    from thenvoi.adapters import PydanticAIAdapter

    # Augment custom_section for contact modes
    section = custom_section
    if contact_config:
        if contact_config.strategy == ContactEventStrategy.CALLBACK:
            section += "\n\nContact requests are handled automatically."
        elif contact_config.strategy == ContactEventStrategy.HUB_ROOM:
            section += "\n\nYou will receive contact requests in a hub room. Use tools to approve/reject."
        if contact_config.broadcast_changes:
            section += (
                "\nYou will see system messages when contacts are added or removed."
            )

    adapter = PydanticAIAdapter(
        model=model,
        custom_section=section,
        enable_execution_reporting=enable_streaming,
    )

    agent = Agent.create(
        adapter=adapter,
        agent_id=agent_id,
        api_key=api_key,
        ws_url=ws_url,
        rest_url=rest_url,
        contact_config=contact_config,
    )

    streaming_str = " with execution reporting" if enable_streaming else ""
    contacts_str = ""
    if contact_config:
        contacts_str = f", contacts={contact_config.strategy.value}"
        if contact_config.broadcast_changes:
            contacts_str += "+broadcast"
    logger.info(
        "Starting Pydantic AI agent with model: %s%s%s",
        model,
        streaming_str,
        contacts_str,
    )
    await agent.run()


async def run_anthropic_agent(
    agent_id: str,
    api_key: str,
    rest_url: str,
    ws_url: str,
    model: str,
    custom_section: str,
    enable_streaming: bool,
    contact_config: ContactEventConfig | None,
    logger: logging.Logger,
) -> None:
    """Run the Anthropic SDK agent."""
    from thenvoi.adapters import AnthropicAdapter

    adapter = AnthropicAdapter(
        model=model,
        custom_section=custom_section,
        enable_execution_reporting=enable_streaming,
    )

    agent = Agent.create(
        adapter=adapter,
        agent_id=agent_id,
        api_key=api_key,
        ws_url=ws_url,
        rest_url=rest_url,
        contact_config=contact_config,
    )

    streaming_str = " with execution reporting" if enable_streaming else ""
    contacts_str = (
        f", contacts={contact_config.strategy.value}" if contact_config else ""
    )
    logger.info(
        "Starting Anthropic agent with model: %s%s%s",
        model,
        streaming_str,
        contacts_str,
    )
    await agent.run()


async def run_claude_sdk_agent(
    agent_id: str,
    api_key: str,
    rest_url: str,
    ws_url: str,
    model: str,
    custom_section: str,
    enable_thinking: bool,
    enable_streaming: bool,
    contact_config: ContactEventConfig | None,
    logger: logging.Logger,
) -> None:
    """Run the Claude Agent SDK agent."""
    from thenvoi.adapters import ClaudeSDKAdapter

    adapter = ClaudeSDKAdapter(
        model=model,
        custom_section=custom_section,
        max_thinking_tokens=10000 if enable_thinking else None,
        enable_execution_reporting=enable_streaming,
    )

    agent = Agent.create(
        adapter=adapter,
        agent_id=agent_id,
        api_key=api_key,
        ws_url=ws_url,
        rest_url=rest_url,
        contact_config=contact_config,
    )

    options = []
    if enable_thinking:
        options.append("extended thinking")
    if enable_streaming:
        options.append("execution reporting")
    if contact_config:
        options.append(f"contacts={contact_config.strategy.value}")
    options_str = f" with {', '.join(options)}" if options else ""
    logger.info("Starting Claude SDK agent with model: %s%s", model, options_str)
    await agent.run()


async def run_parlant_agent(
    agent_id: str,
    api_key: str,
    rest_url: str,
    ws_url: str,
    model: str,
    custom_section: str,
    enable_streaming: bool,
    logger: logging.Logger,
) -> None:
    """Run the Parlant agent."""
    import parlant.sdk as p

    from thenvoi.adapters import ParlantAdapter

    async with p.Server() as server:
        parlant_agent = await server.create_agent(
            name="Thenvoi Parlant Agent",
            description=custom_section or "Parlant assistant on Thenvoi",
        )
        adapter = ParlantAdapter(
            server=server,
            parlant_agent=parlant_agent,
            custom_section=custom_section,
        )

        agent = Agent.create(
            adapter=adapter,
            agent_id=agent_id,
            api_key=api_key,
            ws_url=ws_url,
            rest_url=rest_url,
        )

        if enable_streaming:
            logger.info(
                "Parlant adapter execution reporting is handled by Thenvoi tools/events."
            )
        logger.info("Starting Parlant agent (model hint: %s)", model)
        await agent.run()


async def run_crewai_agent(
    agent_id: str,
    api_key: str,
    rest_url: str,
    ws_url: str,
    model: str,
    custom_section: str,
    enable_streaming: bool,
    logger: logging.Logger,
) -> None:
    """Run the CrewAI agent."""
    adapter = CrewAIAdapter(
        model=model,
        role=CREWAI_DEFAULTS["role"],
        goal=CREWAI_DEFAULTS["goal"],
        backstory=CREWAI_DEFAULTS["backstory"],
        custom_section=custom_section,
        enable_execution_reporting=enable_streaming,
    )

    agent = Agent.create(
        adapter=adapter,
        agent_id=agent_id,
        api_key=api_key,
        ws_url=ws_url,
        rest_url=rest_url,
    )

    logger.info("Starting CrewAI agent with model: %s", model)
    await agent.run()


async def run_codex_agent(
    agent_id: str,
    api_key: str,
    rest_url: str,
    ws_url: str,
    custom_section: str,
    codex_transport: str,
    codex_ws_url: str,
    codex_model: str | None,
    codex_personality: str,
    codex_approval_policy: str,
    codex_approval_mode: str,
    codex_turn_task_markers: bool,
    codex_cwd: str,
    codex_sandbox: str | None,
    codex_reasoning_effort: str | None,
    logger: logging.Logger,
) -> None:
    """Run the Codex app-server adapter."""
    adapter = CodexAdapter(
        config=CodexAdapterConfig(
            transport=codex_transport,  # type: ignore[arg-type]  # str from CLI args, validated at runtime
            cwd=codex_cwd,
            model=codex_model,
            personality=codex_personality,  # type: ignore[arg-type]  # str from CLI args, validated at runtime
            approval_policy=codex_approval_policy,
            approval_mode=codex_approval_mode,  # type: ignore[arg-type]  # str from CLI args, validated at runtime
            sandbox=codex_sandbox,
            reasoning_effort=codex_reasoning_effort,  # type: ignore[arg-type]  # str from CLI args, validated at runtime
            codex_ws_url=codex_ws_url,
            custom_section=custom_section,
            include_base_instructions=True,
            enable_task_events=True,
            emit_turn_task_markers=codex_turn_task_markers,
            enable_execution_reporting=False,
            emit_thought_events=False,
            fallback_send_agent_text=True,
            experimental_api=True,
        )
    )

    agent = Agent.create(
        adapter=adapter,
        agent_id=agent_id,
        api_key=api_key,
        ws_url=ws_url,
        rest_url=rest_url,
    )

    logger.info(
        "Starting Codex agent (transport=%s, model=%s, cwd=%s)",
        codex_transport,
        codex_model or "auto",
        codex_cwd,
    )
    await agent.run()


async def run_pydantic_ai_contacts_agent(
    agent_id: str,
    api_key: str,
    rest_url: str,
    ws_url: str,
    model: str,
    logger: logging.Logger,
) -> None:
    """Run Pydantic AI agent with contact management focus.

    This example demonstrates using contact tools via natural language:
    - "check my contact requests"
    - "list my contacts"
    - "approve alice's request"
    - "reject bob"
    - "add john as a contact"
    """
    from thenvoi.adapters import PydanticAIAdapter

    adapter = PydanticAIAdapter(
        model=model,
        custom_section=CONTACTS_INSTRUCTIONS,
        enable_execution_reporting=True,  # Show tool calls
    )

    agent = Agent.create(
        adapter=adapter,
        agent_id=agent_id,
        api_key=api_key,
        ws_url=ws_url,
        rest_url=rest_url,
    )

    logger.info("Starting Pydantic AI contacts agent with model: %s", model)
    logger.info("Try: 'check my contact requests', 'list contacts', 'approve X'")
    await agent.run()


async def run_contacts_auto_agent(
    agent_id: str,
    api_key: str,
    rest_url: str,
    ws_url: str,
    model: str,
    logger: logging.Logger,
) -> None:
    """Run agent with CALLBACK strategy that auto-approves contact requests.

    This example demonstrates:
    - ContactEventConfig with CALLBACK strategy
    - Auto-approve logic for contact requests
    - broadcast_changes=True to notify all rooms of contact updates
    """
    from thenvoi.adapters import PydanticAIAdapter
    from thenvoi.platform.event import ContactRequestReceivedEvent

    async def auto_approve(event: "ContactEvent", tools: "ContactTools") -> None:
        """Auto-approve all contact requests."""
        if isinstance(event, ContactRequestReceivedEvent):
            if event.payload:
                logger.info(
                    "Auto-approving contact request from %s", event.payload.from_handle
                )
                await tools.respond_contact_request(
                    "approve", request_id=event.payload.id
                )

    config = ContactEventConfig(
        strategy=ContactEventStrategy.CALLBACK,
        on_event=auto_approve,
        broadcast_changes=True,  # All rooms see "[Contacts]: @X is now a contact"
    )

    adapter = PydanticAIAdapter(
        model=model,
        custom_section="""You are a helpful assistant. Contact requests are handled automatically.
When you see system messages about new contacts, acknowledge them to the user.""",
        enable_execution_reporting=True,
    )

    agent = Agent.create(
        adapter=adapter,
        agent_id=agent_id,
        api_key=api_key,
        ws_url=ws_url,
        rest_url=rest_url,
        contact_config=config,
    )

    logger.info("Starting contacts auto-approve agent with model: %s", model)
    logger.info("Contact requests will be automatically approved")
    logger.info("All rooms will see broadcast: '@handle (name) is now a contact'")
    await agent.run()


async def run_contacts_hub_agent(
    agent_id: str,
    api_key: str,
    rest_url: str,
    ws_url: str,
    model: str,
    logger: logging.Logger,
) -> None:
    """Run agent with HUB_ROOM strategy for LLM-driven contact decisions.

    This example demonstrates:
    - ContactEventConfig with HUB_ROOM strategy
    - Contact events routed to a dedicated hub room
    - Agent can reason about requests and respond using tools
    - broadcast_changes=True to notify all rooms of outcomes
    """
    from thenvoi.adapters import PydanticAIAdapter

    config = ContactEventConfig(
        strategy=ContactEventStrategy.HUB_ROOM,
        hub_task_id="contacts-hub",  # Custom task ID for the hub room
        broadcast_changes=True,
    )

    adapter = PydanticAIAdapter(
        model=model,
        custom_section="""You are a helpful assistant that also manages contact requests.

When you receive contact request notifications in the hub room:
1. Review the request details (who sent it, any message included)
2. Decide whether to approve or reject based on the context
3. Use thenvoi_respond_contact_request tool to take action
4. Explain your decision to the user

Contact request format:
- [Contact Request] name (@handle) wants to connect. Message: "..."

Actions available:
- thenvoi_respond_contact_request(action="approve", handle="...")
- thenvoi_respond_contact_request(action="reject", handle="...")
""",
        enable_execution_reporting=True,
    )

    agent = Agent.create(
        adapter=adapter,
        agent_id=agent_id,
        api_key=api_key,
        ws_url=ws_url,
        rest_url=rest_url,
        contact_config=config,
    )

    logger.info("Starting contacts hub room agent with model: %s", model)
    logger.info("Contact events will appear in hub room for LLM reasoning")
    logger.info("All rooms will see broadcasts when contacts change")
    await agent.run()


async def run_contacts_broadcast_agent(
    agent_id: str,
    api_key: str,
    rest_url: str,
    ws_url: str,
    model: str,
    logger: logging.Logger,
) -> None:
    """Run agent with broadcast-only contact notifications.

    This example demonstrates:
    - ContactEventConfig with DISABLED strategy (no auto-handling)
    - broadcast_changes=True for awareness in all rooms
    - User can manually manage contacts via chat commands
    """
    from thenvoi.adapters import PydanticAIAdapter

    config = ContactEventConfig(
        strategy=ContactEventStrategy.DISABLED,  # No auto-handling
        broadcast_changes=True,  # But broadcast updates to all rooms
    )

    adapter = PydanticAIAdapter(
        model=model,
        custom_section=CONTACTS_INSTRUCTIONS
        + """

## System Messages
You will receive system messages when contacts are added or removed.
These appear as "[Contacts]: @handle (name) is now a contact" or similar.
Acknowledge these updates to the user when you see them.
""",
        enable_execution_reporting=True,
    )

    agent = Agent.create(
        adapter=adapter,
        agent_id=agent_id,
        api_key=api_key,
        ws_url=ws_url,
        rest_url=rest_url,
        contact_config=config,
    )

    logger.info("Starting contacts broadcast-only agent with model: %s", model)
    logger.info("Contact changes will be broadcast to all rooms")
    logger.info("Use chat commands to manually manage contacts")
    await agent.run()


async def run_a2a_agent(
    agent_id: str,
    api_key: str,
    rest_url: str,
    ws_url: str,
    a2a_url: str,
    enable_debug: bool,
    logger: logging.Logger,
) -> None:
    """Run the A2A bridge agent."""
    from thenvoi.adapters import A2AAdapter

    # Enable debug logging for A2A adapter to trace context_id and rehydration
    if enable_debug:
        logging.getLogger("thenvoi.integrations.a2a").setLevel(logging.DEBUG)
        logging.getLogger("thenvoi.converters.a2a").setLevel(logging.DEBUG)

    adapter = A2AAdapter(
        remote_url=a2a_url,
        streaming=True,
    )

    agent = Agent.create(
        adapter=adapter,
        agent_id=agent_id,
        api_key=api_key,
        ws_url=ws_url,
        rest_url=rest_url,
    )

    logger.info("Starting A2A bridge agent (forwarding to %s)...", a2a_url)
    await agent.run()


async def run_a2a_gateway_agent(
    agent_id: str,
    api_key: str,
    rest_url: str,
    ws_url: str,
    gateway_port: int,
    enable_debug: bool,
    logger: logging.Logger,
) -> None:
    """Run the A2A Gateway agent.

    The gateway connects to Thenvoi platform and exposes discovered peers
    as A2A endpoints. External A2A agents can call these peers via standard
    A2A protocol.
    """
    from thenvoi.adapters import A2AGatewayAdapter

    # Enable debug logging for gateway adapter
    if enable_debug:
        logging.getLogger("thenvoi.integrations.a2a.gateway").setLevel(logging.DEBUG)

    gateway_url = f"http://localhost:{gateway_port}"

    adapter = A2AGatewayAdapter(
        rest_url=rest_url,
        api_key=api_key,
        gateway_url=gateway_url,
        port=gateway_port,
    )

    agent = Agent.create(
        adapter=adapter,
        agent_id=agent_id,
        api_key=api_key,
        ws_url=ws_url,
        rest_url=rest_url,
    )

    logger.info("Starting A2A Gateway on %s...", gateway_url)
    logger.info("Peers will be exposed at:")
    logger.info(
        "  - %s/agents/{peer_id}/.well-known/agent.json (discovery)", gateway_url
    )
    logger.info("  - %s/agents/{peer_id}/v1/message:stream (messaging)", gateway_url)
    await agent.run()
