"""CrewAI adapter using SimpleAdapter pattern with official CrewAI SDK."""

from __future__ import annotations

import asyncio
import json
import logging
from contextvars import ContextVar
from typing import Any, Coroutine, Literal, Type, TypeVar

from pydantic import BaseModel, Field, field_validator

try:
    from crewai import Agent as CrewAIAgent
    from crewai import LLM
    from crewai.tools import BaseTool
    import nest_asyncio
except ImportError as e:
    raise ImportError(
        "crewai is required for CrewAI adapter.\n"
        "Install with: pip install 'thenvoi-sdk[crewai]'\n"
        "Or: uv add crewai nest-asyncio"
    ) from e

from thenvoi.core.protocols import AgentToolsProtocol
from thenvoi.core.simple_adapter import SimpleAdapter
from thenvoi.core.types import PlatformMessage
from thenvoi.converters.crewai import CrewAIHistoryConverter, CrewAIMessages
from thenvoi.runtime.custom_tools import CustomToolDef, get_custom_tool_name
from thenvoi.runtime.tools import get_tool_description

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Module-level state for nest_asyncio patch.
# See module docstring for important notes about global process impact.
_nest_asyncio_applied = False

# Context variable for thread-safe room context access.
# Set automatically when processing messages, accessed by tools.
_current_room_context: ContextVar[tuple[str, AgentToolsProtocol] | None] = ContextVar(
    "_current_room_context", default=None
)

MessageType = Literal["thought", "error", "task"]

PLATFORM_INSTRUCTIONS = """## Environment

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


def _ensure_nest_asyncio() -> None:
    """Apply nest_asyncio patch lazily on first use.

    See module docstring for important notes about global process impact.
    """
    global _nest_asyncio_applied
    if not _nest_asyncio_applied:
        nest_asyncio.apply()
        _nest_asyncio_applied = True
        logger.debug("Applied nest_asyncio patch for nested event loops")


def _run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Run an async coroutine from sync context.

    CrewAI tools are synchronous but need to call async platform methods.
    With nest_asyncio applied, we can safely run coroutines even when
    an event loop is already running.
    """
    _ensure_nest_asyncio()

    try:
        loop = asyncio.get_running_loop()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


class CrewAIAdapter(SimpleAdapter[CrewAIMessages]):
    """CrewAI adapter using the official CrewAI SDK.

    Integrates the CrewAI framework (https://docs.crewai.com/) with Thenvoi
    platform for building collaborative multi-agent systems.

    Example:
        adapter = CrewAIAdapter(
            model="gpt-4o",
            role="Research Assistant",
            goal="Help users find and analyze information",
            backstory="Expert researcher with deep knowledge across domains",
        )
        agent = Agent.create(adapter=adapter, agent_id="...", api_key="...")
        await agent.run()

    Note:
        API keys are configured through environment variables as expected by
        the CrewAI LLM class (e.g., OPENAI_API_KEY, ANTHROPIC_API_KEY).
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
        additional_tools: list[CustomToolDef] | None = None,
    ):
        """Initialize the CrewAI adapter.

        Args:
            model: Model name (e.g., "gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet").
                   API keys are read from environment variables by CrewAI's LLM class.
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
            additional_tools: List of custom tools as (InputModel, callable) tuples.
                Each InputModel is a Pydantic model defining the tool's input schema,
                and the callable is the function to execute (sync or async).
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

        self._crewai_agent: CrewAIAgent | None = None
        self._room_tools: dict[str, AgentToolsProtocol] = {}
        self._message_history: dict[str, list[dict[str, Any]]] = {}
        self._custom_tools: list[CustomToolDef] = additional_tools or []

    async def on_started(self, agent_name: str, agent_description: str) -> None:
        """Initialize CrewAI agent after metadata is fetched."""
        await super().on_started(agent_name, agent_description)

        role = self.role or agent_name
        goal = self.goal or agent_description or "Help users accomplish their tasks"

        backstory_parts = []
        if self.backstory:
            backstory_parts.append(self.backstory)
        else:
            backstory_parts.append(
                f"You are {agent_name}, a collaborative AI agent on the Thenvoi platform."
            )

        if self.custom_section:
            backstory_parts.append(self.custom_section)

        backstory_parts.append(PLATFORM_INSTRUCTIONS)
        backstory = "\n\n".join(backstory_parts)

        tools = self._create_crewai_tools()

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

    def _get_current_room_context(self) -> tuple[str, AgentToolsProtocol] | None:
        """Get current room context from context variable.

        Returns:
            Tuple of (room_id, tools) if context is set, None otherwise.
        """
        return _current_room_context.get()

    def _execute_tool(
        self,
        tool_name: str,
        coro_factory: Any,
    ) -> str:
        """Execute a tool with common error handling.

        Args:
            tool_name: Name of the tool for error messages
            coro_factory: Callable that takes tools and returns a coroutine

        Returns:
            JSON string with status and result/error
        """
        context = self._get_current_room_context()
        if not context:
            return json.dumps(
                {
                    "status": "error",
                    "message": "No room context available - tool called outside message handling",
                }
            )

        room_id, tools = context

        async def _execute() -> str:
            try:
                return await coro_factory(tools)
            except Exception as e:
                error_msg = str(e)
                logger.error(f"{tool_name} failed in room {room_id}: {error_msg}")
                return json.dumps({"status": "error", "message": error_msg})

        return _run_async(_execute())

    async def _report_tool_call(
        self,
        tools: AgentToolsProtocol,
        tool_name: str,
        input_data: dict[str, Any],
    ) -> None:
        """Report tool call event if execution reporting is enabled."""
        if self.enable_execution_reporting:
            await tools.send_event(
                content=json.dumps({"tool": tool_name, "input": input_data}),
                message_type="tool_call",
            )

    async def _report_tool_result(
        self,
        tools: AgentToolsProtocol,
        tool_name: str,
        result: Any,
        is_error: bool = False,
    ) -> None:
        """Report tool result event if execution reporting is enabled."""
        if self.enable_execution_reporting:
            if is_error:
                await tools.send_event(
                    content=json.dumps({"tool": tool_name, "error": result}),
                    message_type="tool_result",
                )
            else:
                await tools.send_event(
                    content=json.dumps({"tool": tool_name, "result": result}),
                    message_type="tool_result",
                )

    def _convert_custom_tools_to_crewai(self) -> list[BaseTool]:
        """Convert CustomToolDef tuples to CrewAI BaseTool instances.

        Each custom tool is wrapped in a dynamically created BaseTool subclass
        that handles sync/async bridging and error handling.

        Returns:
            List of CrewAI BaseTool instances.
        """
        adapter = self
        crewai_tools: list[BaseTool] = []

        for input_model, func in self._custom_tools:
            tool_name = get_custom_tool_name(input_model)
            tool_description = input_model.__doc__ or f"Execute {tool_name}"

            # Create a closure to capture the current input_model and func
            def make_tool(
                tool_name_param: str,
                tool_desc_param: str,
                model: type[BaseModel],
                handler: Any,
            ) -> BaseTool:
                # Capture values in local variables for the closure
                _tool_name = tool_name_param
                _tool_desc = tool_desc_param

                class CustomCrewAITool(BaseTool):
                    name: str = _tool_name  # type: ignore[misc]
                    description: str = _tool_desc  # type: ignore[misc]
                    args_schema: Type[BaseModel] = model

                    def _run(self, **kwargs: Any) -> str:
                        async def execute(_tools: AgentToolsProtocol) -> str:
                            try:
                                # Validate input using Pydantic model
                                validated = model.model_validate(kwargs)

                                # Report tool call if enabled
                                await adapter._report_tool_call(
                                    _tools, _tool_name, kwargs
                                )

                                # Execute the handler (sync or async)
                                if asyncio.iscoroutinefunction(handler):
                                    result = await handler(validated)
                                else:
                                    result = handler(validated)

                                # Report tool result if enabled
                                await adapter._report_tool_result(
                                    _tools, _tool_name, result
                                )

                                # Return JSON-serialized result
                                if isinstance(result, str):
                                    return json.dumps(
                                        {"status": "success", "result": result}
                                    )
                                return json.dumps(
                                    {"status": "success", "result": result},
                                    default=str,
                                )
                            except Exception as e:
                                error_msg = str(e)
                                logger.error(
                                    f"Custom tool {_tool_name} failed: {error_msg}"
                                )
                                await adapter._report_tool_result(
                                    _tools, _tool_name, error_msg, is_error=True
                                )
                                return json.dumps(
                                    {"status": "error", "message": error_msg}
                                )

                        return adapter._execute_tool(_tool_name, execute)

                return CustomCrewAITool()

            crewai_tools.append(
                make_tool(tool_name, tool_description, input_model, func)
            )

        return crewai_tools

    def _create_crewai_tools(self) -> list[BaseTool]:
        """Create CrewAI-compatible tools for Thenvoi platform.

        Tools access the current room context via a context variable that is set
        automatically when processing messages. This removes the need for the LLM
        to pass room_id explicitly, making tool calls more reliable.
        """
        adapter = self

        class SendMessageInput(BaseModel):
            content: str = Field(..., description="The message content to send")
            mentions: str = Field(
                default="[]",
                description="JSON array of participant names to @mention",
            )

            @field_validator("mentions", mode="before")
            @classmethod
            def normalize_mentions(cls, v: Any) -> str:
                if isinstance(v, list):
                    return json.dumps(v)
                return v

        class SendEventInput(BaseModel):
            content: str = Field(..., description="Human-readable event content")
            message_type: MessageType = Field(
                default="thought",
                description="Type of event: 'thought', 'error', or 'task'",
            )

        class AddParticipantInput(BaseModel):
            participant_name: str = Field(
                ...,
                description="Name of participant to add (must match from lookup_peers)",
            )
            role: str = Field(
                default="member", description="Role: 'owner', 'admin', or 'member'"
            )

        class RemoveParticipantInput(BaseModel):
            participant_name: str = Field(
                ..., description="Name of the participant to remove"
            )

        class GetParticipantsInput(BaseModel):
            pass  # No parameters needed - room context from context variable

        class LookupPeersInput(BaseModel):
            page: int = Field(default=1, description="Page number")
            page_size: int = Field(default=50, description="Items per page (max 100)")

        class CreateChatroomInput(BaseModel):
            task_id: str = Field(
                default="", description="Associated task ID (optional)"
            )

        class SendMessageTool(BaseTool):
            name: str = "send_message"
            description: str = get_tool_description("send_message")
            args_schema: Type[BaseModel] = SendMessageInput

            def _run(self, **kwargs: Any) -> str:
                content: str = kwargs.get("content", "")
                mentions: str = kwargs.get("mentions", "[]")

                try:
                    mention_list = json.loads(mentions) if mentions else []
                except json.JSONDecodeError:
                    mention_list = []

                async def execute(tools: AgentToolsProtocol) -> str:
                    await adapter._report_tool_call(
                        tools,
                        "send_message",
                        {"content": content, "mentions": mention_list},
                    )
                    await tools.send_message(content, mention_list)
                    await adapter._report_tool_result(tools, "send_message", "success")
                    return json.dumps({"status": "success", "message": "Message sent"})

                return adapter._execute_tool("send_message", execute)

        class SendEventTool(BaseTool):
            name: str = "send_event"
            description: str = get_tool_description("send_event")
            args_schema: Type[BaseModel] = SendEventInput

            def _run(self, **kwargs: Any) -> str:
                content: str = kwargs.get("content", "")
                message_type: str = kwargs.get("message_type", "thought")

                async def execute(tools: AgentToolsProtocol) -> str:
                    await tools.send_event(content, message_type)
                    return json.dumps({"status": "success", "message": "Event sent"})

                return adapter._execute_tool("send_event", execute)

        class AddParticipantTool(BaseTool):
            name: str = "add_participant"
            description: str = get_tool_description("add_participant")
            args_schema: Type[BaseModel] = AddParticipantInput

            def _run(self, **kwargs: Any) -> str:
                participant_name: str = kwargs.get("participant_name", "")
                role: str = kwargs.get("role", "member")

                async def execute(tools: AgentToolsProtocol) -> str:
                    await adapter._report_tool_call(
                        tools,
                        "add_participant",
                        {"name": participant_name, "role": role},
                    )
                    result = await tools.add_participant(participant_name, role)
                    await adapter._report_tool_result(tools, "add_participant", result)
                    return json.dumps({"status": "success", **result})

                return adapter._execute_tool("add_participant", execute)

        class RemoveParticipantTool(BaseTool):
            name: str = "remove_participant"
            description: str = get_tool_description("remove_participant")
            args_schema: Type[BaseModel] = RemoveParticipantInput

            def _run(self, **kwargs: Any) -> str:
                participant_name: str = kwargs.get("participant_name", "")

                async def execute(tools: AgentToolsProtocol) -> str:
                    await adapter._report_tool_call(
                        tools,
                        "remove_participant",
                        {"name": participant_name},
                    )
                    result = await tools.remove_participant(participant_name)
                    await adapter._report_tool_result(
                        tools, "remove_participant", result
                    )
                    return json.dumps({"status": "success", **result})

                return adapter._execute_tool("remove_participant", execute)

        class GetParticipantsTool(BaseTool):
            name: str = "get_participants"
            description: str = get_tool_description("get_participants")
            args_schema: Type[BaseModel] = GetParticipantsInput

            def _run(self, **_kwargs: Any) -> str:
                async def execute(tools: AgentToolsProtocol) -> str:
                    participants = await tools.get_participants()
                    return json.dumps(
                        {
                            "status": "success",
                            "participants": participants,
                            "count": len(participants),
                        }
                    )

                return adapter._execute_tool("get_participants", execute)

        class LookupPeersTool(BaseTool):
            name: str = "lookup_peers"
            description: str = get_tool_description("lookup_peers")
            args_schema: Type[BaseModel] = LookupPeersInput

            def _run(self, **kwargs: Any) -> str:
                page: int = kwargs.get("page", 1)
                page_size: int = kwargs.get("page_size", 50)

                async def execute(tools: AgentToolsProtocol) -> str:
                    result = await tools.lookup_peers(page, page_size)
                    return json.dumps({"status": "success", **result})

                return adapter._execute_tool("lookup_peers", execute)

        class CreateChatroomTool(BaseTool):
            name: str = "create_chatroom"
            description: str = get_tool_description("create_chatroom")
            args_schema: Type[BaseModel] = CreateChatroomInput

            def _run(self, **kwargs: Any) -> str:
                task_id: str = kwargs.get("task_id", "")

                async def execute(tools: AgentToolsProtocol) -> str:
                    new_room_id = await tools.create_chatroom(task_id or None)
                    return json.dumps(
                        {
                            "status": "success",
                            "message": "Chat room created",
                            "room_id": new_room_id,
                        }
                    )

                return adapter._execute_tool("create_chatroom", execute)

        platform_tools: list[BaseTool] = [
            SendMessageTool(),
            SendEventTool(),
            AddParticipantTool(),
            RemoveParticipantTool(),
            GetParticipantsTool(),
            LookupPeersTool(),
            CreateChatroomTool(),
        ]

        # Add custom tools converted to CrewAI format
        custom_tools = self._convert_custom_tools_to_crewai()
        if custom_tools:
            logger.debug(
                f"Added {len(custom_tools)} custom tools: "
                f"{[t.name for t in custom_tools]}"
            )

        return platform_tools + custom_tools

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
        """Handle incoming message using CrewAI agent."""
        logger.debug(f"Handling message {msg.id} in room {room_id}")

        if not self._crewai_agent:
            logger.error("CrewAI agent not initialized")
            return

        # Set context variable for tool access (thread-safe room context)
        _current_room_context.set((room_id, tools))

        # Keep reference for cleanup
        self._room_tools[room_id] = tools

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

        messages = []

        if is_session_bootstrap and self._message_history.get(room_id):
            history_text = "\n".join(
                f"{m['role']}: {m['content']}" for m in self._message_history[room_id]
            )
            messages.append(
                {
                    "role": "user",
                    "content": f"[Previous conversation:]\n{history_text}",
                }
            )

        if participants_msg:
            messages.append(
                {
                    "role": "user",
                    "content": f"[System]: {participants_msg}",
                }
            )
            logger.info(f"Room {room_id}: Participants updated")

        user_message = msg.format_for_llm()
        messages.append({"role": "user", "content": user_message})

        self._message_history[room_id].append(
            {
                "role": "user",
                "content": user_message,
            }
        )

        total_messages = len(self._message_history[room_id])
        logger.info(
            f"Room {room_id}: Processing with {total_messages} messages "
            f"(first_msg={is_session_bootstrap})"
        )

        try:
            # CrewAI's kickoff_async expects a string prompt but also accepts
            # a list of message dicts for multi-turn context. The type stub
            # is overly restrictive, hence the ignore.
            result = await self._crewai_agent.kickoff_async(messages)  # type: ignore[arg-type]

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
        finally:
            # Clear context after processing (defensive, not strictly necessary
            # since context vars are task-local in asyncio)
            _current_room_context.set(None)

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
