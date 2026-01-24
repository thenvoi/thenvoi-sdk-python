"""CrewAI adapter using SimpleAdapter pattern with official CrewAI SDK."""

from __future__ import annotations

import json
import logging
from typing import Any, Type

try:
    from crewai import Agent as CrewAIAgent
    from crewai import LLM
    from crewai.tools import BaseTool
    from pydantic import BaseModel, Field
except ImportError as e:
    raise ImportError(
        "crewai is required for CrewAI adapter.\n"
        "Install with: pip install crewai\n"
        "Or: uv add crewai"
    ) from e

from thenvoi.core.protocols import AgentToolsProtocol
from thenvoi.core.simple_adapter import SimpleAdapter
from thenvoi.core.types import PlatformMessage
from thenvoi.converters.crewai import CrewAIHistoryConverter, CrewAIMessages
from thenvoi.runtime.tools import get_tool_description

logger = logging.getLogger(__name__)


class CrewAIAdapter(SimpleAdapter[CrewAIMessages]):
    """
    CrewAI adapter using the official CrewAI SDK.

    Integrates the CrewAI framework (https://docs.crewai.com/) with Thenvoi
    platform for building collaborative multi-agent systems.

    CrewAI provides:
    - Agent collaboration with defined roles and goals
    - Task orchestration with sequential/hierarchical processes
    - Memory and knowledge management
    - Built-in tool integration

    Example:
        adapter = CrewAIAdapter(
            model="gpt-4o",
            role="Research Assistant",
            goal="Help users find and analyze information",
            backstory="Expert researcher with deep knowledge across domains",
        )
        agent = Agent.create(adapter=adapter, agent_id="...", api_key="...")
        await agent.run()
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        role: str | None = None,
        goal: str | None = None,
        backstory: str | None = None,
        custom_section: str | None = None,
        enable_execution_reporting: bool = False,
        verbose: bool = False,
        max_iter: int = 20,
        max_rpm: int | None = None,
        allow_delegation: bool = False,
        history_converter: CrewAIHistoryConverter | None = None,
    ):
        """
        Initialize the CrewAI adapter.

        Args:
            model: Model name to use (e.g., "gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet")
            role: Agent's role in the crew (e.g., "Research Assistant")
            goal: Agent's primary goal or objective
            backstory: Agent's background and expertise description
            custom_section: Custom instructions added to the agent's backstory
            enable_execution_reporting: If True, sends tool_call/tool_result events
            verbose: If True, enables detailed logging from CrewAI
            max_iter: Maximum iterations for the agent (default: 20)
            max_rpm: Maximum requests per minute (rate limiting)
            allow_delegation: Whether to allow task delegation to other agents
            history_converter: Custom history converter (optional)
        """
        super().__init__(
            history_converter=history_converter or CrewAIHistoryConverter()
        )

        self.model = model
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.custom_section = custom_section
        self.enable_execution_reporting = enable_execution_reporting
        self.verbose = verbose
        self.max_iter = max_iter
        self.max_rpm = max_rpm
        self.allow_delegation = allow_delegation

        # CrewAI agent (created after start)
        self._crewai_agent: CrewAIAgent | None = None

        # Per-room tools storage (like Claude SDK adapter)
        self._room_tools: dict[str, AgentToolsProtocol] = {}

        # Per-room conversation history
        self._message_history: dict[str, list[dict[str, Any]]] = {}

    async def on_started(self, agent_name: str, agent_description: str) -> None:
        """Initialize CrewAI agent after metadata is fetched."""
        await super().on_started(agent_name, agent_description)

        # Build role, goal, backstory from params or platform metadata
        role = self.role or agent_name
        goal = self.goal or agent_description or "Help users accomplish their tasks"

        # Build backstory with custom section if provided
        backstory_parts = []
        if self.backstory:
            backstory_parts.append(self.backstory)
        else:
            backstory_parts.append(
                f"You are {agent_name}, a collaborative AI agent on the Thenvoi platform."
            )

        if self.custom_section:
            backstory_parts.append(self.custom_section)

        # Add platform-specific instructions
        backstory_parts.append(self._get_platform_instructions())

        backstory = "\n\n".join(backstory_parts)

        # Create CrewAI tools for Thenvoi platform
        tools = self._create_crewai_tools()

        # Create the CrewAI agent
        self._crewai_agent = CrewAIAgent(
            role=role,
            goal=goal,
            backstory=backstory,
            llm=LLM(model=self.model),
            tools=tools,
            verbose=self.verbose,
            max_iter=self.max_iter,
            max_rpm=self.max_rpm,
            allow_delegation=self.allow_delegation,
        )

        logger.info(
            f"CrewAI adapter started for agent: {agent_name} "
            f"(model={self.model}, role={role})"
        )

    def _get_platform_instructions(self) -> str:
        """Get platform-specific instructions for the agent's backstory."""
        return """## Environment

Multi-participant chat on Thenvoi platform. Messages show sender: [Name]: content.
Use the `send_message` tool to respond. Plain text output is not delivered.

## CRITICAL: Delegate When You Cannot Help Directly

You have NO internet access and NO real-time data. When asked about weather, news, stock prices,
or any current information you cannot answer directly:

1. Call `lookup_peers` to find available specialized agents
2. If a relevant agent exists (e.g., Weather Agent), call `add_participant` to add them
3. Ask that agent using `send_message` with their name in mentions
4. Wait for their response and relay it back to the user

NEVER say "I can't do that" without first checking if another agent can help via `lookup_peers`.

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

Call `send_event` with message_type="thought" BEFORE every action to share your reasoning."""

    def _create_crewai_tools(self) -> list[BaseTool]:
        """Create CrewAI-compatible tools for Thenvoi platform."""
        adapter = self  # Capture reference for tool closures

        # Define input schemas for each tool
        class SendMessageInput(BaseModel):
            room_id: str = Field(..., description="The room ID to send the message to")
            content: str = Field(..., description="The message content to send")
            mentions: str = Field(
                default="[]",
                description="JSON array of participant names to @mention",
            )

        class SendEventInput(BaseModel):
            room_id: str = Field(..., description="The room ID to send the event to")
            content: str = Field(..., description="Human-readable event content")
            message_type: str = Field(
                default="thought",
                description="Type of event: 'thought', 'error', or 'task'",
            )

        class AddParticipantInput(BaseModel):
            room_id: str = Field(
                ..., description="The room ID to add the participant to"
            )
            participant_name: str = Field(
                ...,
                description="Name of participant to add (must match from lookup_peers)",
            )
            role: str = Field(
                default="member", description="Role: 'owner', 'admin', or 'member'"
            )

        class RemoveParticipantInput(BaseModel):
            room_id: str = Field(
                ..., description="The room ID to remove the participant from"
            )
            participant_name: str = Field(
                ..., description="Name of the participant to remove"
            )

        class GetParticipantsInput(BaseModel):
            room_id: str = Field(
                ..., description="The room ID to get participants from"
            )

        class LookupPeersInput(BaseModel):
            room_id: str = Field(..., description="The room ID for context")
            page: int = Field(default=1, description="Page number")
            page_size: int = Field(default=50, description="Items per page (max 100)")

        class CreateChatroomInput(BaseModel):
            room_id: str = Field(..., description="The current room ID for context")
            task_id: str = Field(
                default="", description="Associated task ID (optional)"
            )

        # Define tools with args_schema
        class SendMessageTool(BaseTool):
            name: str = "send_message"
            description: str = get_tool_description("send_message")
            args_schema: Type[BaseModel] = SendMessageInput

            def _run(self, room_id: str, content: str, mentions: str = "[]") -> str:
                import asyncio

                async def _execute():
                    tools = adapter._room_tools.get(room_id)
                    if not tools:
                        return json.dumps(
                            {"status": "error", "message": f"No tools for room {room_id}"}
                        )

                    try:
                        mention_list = json.loads(mentions) if mentions else []
                    except json.JSONDecodeError:
                        mention_list = []

                    if adapter.enable_execution_reporting:
                        await tools.send_event(
                            content=json.dumps(
                                {
                                    "tool": "send_message",
                                    "input": {"content": content, "mentions": mention_list},
                                }
                            ),
                            message_type="tool_call",
                        )

                    try:
                        await tools.send_message(content, mention_list)
                        if adapter.enable_execution_reporting:
                            await tools.send_event(
                                content=json.dumps(
                                    {"tool": "send_message", "result": "success"}
                                ),
                                message_type="tool_result",
                            )
                        return json.dumps({"status": "success", "message": "Message sent"})
                    except Exception as e:
                        error_msg = str(e)
                        if adapter.enable_execution_reporting:
                            await tools.send_event(
                                content=json.dumps(
                                    {"tool": "send_message", "error": error_msg}
                                ),
                                message_type="tool_result",
                            )
                        return json.dumps({"status": "error", "message": error_msg})

                try:
                    loop = asyncio.get_running_loop()
                    return loop.run_until_complete(_execute())
                except RuntimeError:
                    return asyncio.run(_execute())

        class SendEventTool(BaseTool):
            name: str = "send_event"
            description: str = get_tool_description("send_event")
            args_schema: Type[BaseModel] = SendEventInput

            def _run(
                self, room_id: str, content: str, message_type: str = "thought"
            ) -> str:
                import asyncio

                async def _execute():
                    tools = adapter._room_tools.get(room_id)
                    if not tools:
                        return json.dumps(
                            {"status": "error", "message": f"No tools for room {room_id}"}
                        )

                    try:
                        await tools.send_event(content, message_type)
                        return json.dumps({"status": "success", "message": "Event sent"})
                    except Exception as e:
                        return json.dumps({"status": "error", "message": str(e)})

                try:
                    loop = asyncio.get_running_loop()
                    return loop.run_until_complete(_execute())
                except RuntimeError:
                    return asyncio.run(_execute())

        class AddParticipantTool(BaseTool):
            name: str = "add_participant"
            description: str = get_tool_description("add_participant")
            args_schema: Type[BaseModel] = AddParticipantInput

            def _run(
                self, room_id: str, participant_name: str, role: str = "member"
            ) -> str:
                import asyncio

                async def _execute():
                    tools = adapter._room_tools.get(room_id)
                    if not tools:
                        return json.dumps(
                            {"status": "error", "message": f"No tools for room {room_id}"}
                        )

                    if adapter.enable_execution_reporting:
                        await tools.send_event(
                            content=json.dumps(
                                {
                                    "tool": "add_participant",
                                    "input": {"name": participant_name, "role": role},
                                }
                            ),
                            message_type="tool_call",
                        )

                    try:
                        result = await tools.add_participant(participant_name, role)
                        if adapter.enable_execution_reporting:
                            await tools.send_event(
                                content=json.dumps(
                                    {"tool": "add_participant", "result": result}
                                ),
                                message_type="tool_result",
                            )
                        return json.dumps({"status": "success", **result})
                    except Exception as e:
                        error_msg = str(e)
                        if adapter.enable_execution_reporting:
                            await tools.send_event(
                                content=json.dumps(
                                    {"tool": "add_participant", "error": error_msg}
                                ),
                                message_type="tool_result",
                            )
                        return json.dumps({"status": "error", "message": error_msg})

                try:
                    loop = asyncio.get_running_loop()
                    return loop.run_until_complete(_execute())
                except RuntimeError:
                    return asyncio.run(_execute())

        class RemoveParticipantTool(BaseTool):
            name: str = "remove_participant"
            description: str = get_tool_description("remove_participant")
            args_schema: Type[BaseModel] = RemoveParticipantInput

            def _run(self, room_id: str, participant_name: str) -> str:
                import asyncio

                async def _execute():
                    tools = adapter._room_tools.get(room_id)
                    if not tools:
                        return json.dumps(
                            {"status": "error", "message": f"No tools for room {room_id}"}
                        )

                    if adapter.enable_execution_reporting:
                        await tools.send_event(
                            content=json.dumps(
                                {
                                    "tool": "remove_participant",
                                    "input": {"name": participant_name},
                                }
                            ),
                            message_type="tool_call",
                        )

                    try:
                        result = await tools.remove_participant(participant_name)
                        if adapter.enable_execution_reporting:
                            await tools.send_event(
                                content=json.dumps(
                                    {"tool": "remove_participant", "result": result}
                                ),
                                message_type="tool_result",
                            )
                        return json.dumps({"status": "success", **result})
                    except Exception as e:
                        error_msg = str(e)
                        if adapter.enable_execution_reporting:
                            await tools.send_event(
                                content=json.dumps(
                                    {"tool": "remove_participant", "error": error_msg}
                                ),
                                message_type="tool_result",
                            )
                        return json.dumps({"status": "error", "message": error_msg})

                try:
                    loop = asyncio.get_running_loop()
                    return loop.run_until_complete(_execute())
                except RuntimeError:
                    return asyncio.run(_execute())

        class GetParticipantsTool(BaseTool):
            name: str = "get_participants"
            description: str = get_tool_description("get_participants")
            args_schema: Type[BaseModel] = GetParticipantsInput

            def _run(self, room_id: str) -> str:
                import asyncio

                async def _execute():
                    tools = adapter._room_tools.get(room_id)
                    if not tools:
                        return json.dumps(
                            {"status": "error", "message": f"No tools for room {room_id}"}
                        )

                    try:
                        participants = await tools.get_participants()
                        return json.dumps(
                            {
                                "status": "success",
                                "participants": participants,
                                "count": len(participants),
                            }
                        )
                    except Exception as e:
                        return json.dumps({"status": "error", "message": str(e)})

                try:
                    loop = asyncio.get_running_loop()
                    return loop.run_until_complete(_execute())
                except RuntimeError:
                    return asyncio.run(_execute())

        class LookupPeersTool(BaseTool):
            name: str = "lookup_peers"
            description: str = get_tool_description("lookup_peers")
            args_schema: Type[BaseModel] = LookupPeersInput

            def _run(self, room_id: str, page: int = 1, page_size: int = 50) -> str:
                import asyncio

                async def _execute():
                    tools = adapter._room_tools.get(room_id)
                    if not tools:
                        return json.dumps(
                            {"status": "error", "message": f"No tools for room {room_id}"}
                        )

                    try:
                        result = await tools.lookup_peers(page, page_size)
                        return json.dumps({"status": "success", **result})
                    except Exception as e:
                        return json.dumps({"status": "error", "message": str(e)})

                try:
                    loop = asyncio.get_running_loop()
                    return loop.run_until_complete(_execute())
                except RuntimeError:
                    return asyncio.run(_execute())

        class CreateChatroomTool(BaseTool):
            name: str = "create_chatroom"
            description: str = get_tool_description("create_chatroom")
            args_schema: Type[BaseModel] = CreateChatroomInput

            def _run(self, room_id: str, task_id: str = "") -> str:
                import asyncio

                async def _execute():
                    tools = adapter._room_tools.get(room_id)
                    if not tools:
                        return json.dumps(
                            {"status": "error", "message": f"No tools for room {room_id}"}
                        )

                    try:
                        new_room_id = await tools.create_chatroom(task_id or None)
                        return json.dumps(
                            {
                                "status": "success",
                                "message": "Chat room created",
                                "room_id": new_room_id,
                            }
                        )
                    except Exception as e:
                        return json.dumps({"status": "error", "message": str(e)})

                try:
                    loop = asyncio.get_running_loop()
                    return loop.run_until_complete(_execute())
                except RuntimeError:
                    return asyncio.run(_execute())

        return [
            SendMessageTool(),
            SendEventTool(),
            AddParticipantTool(),
            RemoveParticipantTool(),
            GetParticipantsTool(),
            LookupPeersTool(),
            CreateChatroomTool(),
        ]

    async def on_message(
        self,
        msg: PlatformMessage,
        tools: AgentToolsProtocol,
        history: CrewAIMessages,
        participants_msg: str | None,
        *,
        is_session_bootstrap: bool,
        room_id: str,
    ) -> None:
        """
        Handle incoming message using CrewAI agent.

        Uses the CrewAI SDK's kickoff_async() for message processing with
        the platform tools for collaboration.
        """
        logger.debug(f"Handling message {msg.id} in room {room_id}")

        if not self._crewai_agent:
            logger.error("CrewAI agent not initialized")
            return

        # Store tools for CrewAI tool access
        self._room_tools[room_id] = tools

        # Initialize history for this room on first message
        if is_session_bootstrap:
            if history:
                self._message_history[room_id] = [
                    {"role": h["role"], "content": h["content"]} for h in history
                ]
                logger.info(
                    f"Room {room_id}: Loaded {len(history)} historical messages"
                )
            else:
                self._message_history[room_id] = []
                logger.info(f"Room {room_id}: No historical messages found")
        elif room_id not in self._message_history:
            self._message_history[room_id] = []

        # Build message list for CrewAI
        messages = []

        # Add room_id context (CrewAI needs this for tool calls)
        room_context = f"[room_id: {room_id}]"

        # Include historical context on first message
        if is_session_bootstrap and self._message_history.get(room_id):
            history_text = "\n".join(
                f"{m['role']}: {m['content']}" for m in self._message_history[room_id]
            )
            messages.append(
                {
                    "role": "user",
                    "content": f"{room_context}[Previous conversation context:]\n{history_text}",
                }
            )

        # Inject participants message if changed
        if participants_msg:
            messages.append(
                {
                    "role": "user",
                    "content": f"{room_context}[System]: {participants_msg}",
                }
            )
            logger.info(f"Room {room_id}: Participants updated")

        # Add current message with room_id context
        user_message = f"{room_context}{msg.format_for_llm()}"
        messages.append({"role": "user", "content": user_message})

        # Store the current message in history
        self._message_history[room_id].append(
            {
                "role": "user",
                "content": msg.format_for_llm(),
            }
        )

        # Log message count
        total_messages = len(self._message_history[room_id])
        logger.info(
            f"Room {room_id}: Processing with {total_messages} messages "
            f"(first_msg={is_session_bootstrap})"
        )

        try:
            # Use CrewAI's kickoff_async with message list
            result = await self._crewai_agent.kickoff_async(messages)

            # Store the response in history
            if result and result.raw:
                self._message_history[room_id].append(
                    {
                        "role": "assistant",
                        "content": result.raw,
                    }
                )

            logger.info(
                f"Room {room_id}: CrewAI agent completed "
                f"(output_length={len(result.raw) if result and result.raw else 0})"
            )

        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            await self._report_error(tools, str(e))
            raise

        logger.debug(
            f"Message {msg.id} processed successfully "
            f"(history now has {len(self._message_history[room_id])} messages)"
        )

    async def on_cleanup(self, room_id: str) -> None:
        """Clean up message history and tools when agent leaves a room."""
        if room_id in self._message_history:
            del self._message_history[room_id]
        if room_id in self._room_tools:
            del self._room_tools[room_id]
        logger.debug(f"Room {room_id}: Cleaned up CrewAI session")

    async def _report_error(self, tools: AgentToolsProtocol, error: str) -> None:
        """Send error event (best effort)."""
        try:
            await tools.send_event(content=f"Error: {error}", message_type="error")
        except Exception:
            pass
