"""CrewAI adapter using SimpleAdapter pattern."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from thenvoi.core.protocols import AgentToolsProtocol
from thenvoi.core.simple_adapter import SimpleAdapter
from thenvoi.core.types import PlatformMessage
from thenvoi.converters.crewai import CrewAIHistoryConverter, CrewAIMessages
from thenvoi.runtime.prompts import render_system_prompt

logger = logging.getLogger(__name__)


class CrewAIAdapter(SimpleAdapter[CrewAIMessages]):
    """
    CrewAI adapter using SimpleAdapter pattern.

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
        system_prompt: str | None = None,
        custom_section: str | None = None,
        openai_api_key: str | None = None,
        enable_execution_reporting: bool = False,
        verbose: bool = False,
        history_converter: CrewAIHistoryConverter | None = None,
    ):
        """
        Initialize the CrewAI adapter.

        Args:
            model: Model name to use (e.g., "gpt-4o", "gpt-4o-mini")
            role: Agent's role in the crew (e.g., "Research Assistant")
            goal: Agent's primary goal or objective
            backstory: Agent's background and expertise description
            system_prompt: Full system prompt override (optional)
            custom_section: Custom instructions added to default prompt
            openai_api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)
            enable_execution_reporting: If True, sends tool_call/tool_result events
            verbose: If True, enables detailed logging
            history_converter: Custom history converter (optional)
        """
        super().__init__(
            history_converter=history_converter or CrewAIHistoryConverter()
        )

        self.model = model
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.system_prompt = system_prompt
        self.custom_section = custom_section
        self.openai_api_key = openai_api_key
        self.enable_execution_reporting = enable_execution_reporting
        self.verbose = verbose

        # Per-room conversation history
        self._message_history: dict[str, list[dict[str, Any]]] = {}
        # Rendered system prompt (set after start)
        self._system_prompt: str = ""
        # Max tool iterations to prevent infinite loops
        self._max_tool_iterations = 10

    async def on_started(self, agent_name: str, agent_description: str) -> None:
        """Initialize CrewAI agent after metadata is fetched."""
        await super().on_started(agent_name, agent_description)

        # Build system prompt with CrewAI-style agent definition
        if self.system_prompt:
            self._system_prompt = self.system_prompt
        else:
            # Use role/goal/backstory if provided, otherwise use platform metadata
            role = self.role or agent_name
            goal = self.goal or agent_description or "Help users accomplish their tasks"
            backstory = self.backstory or f"You are {agent_name}, a collaborative AI agent."

            base_prompt = render_system_prompt(
                agent_name=agent_name,
                agent_description=agent_description,
                custom_section=self.custom_section or "",
            )

            # Add CrewAI-style agent definition
            crewai_section = f"""
## Agent Profile

**Role**: {role}

**Goal**: {goal}

**Backstory**: {backstory}

## Operating Guidelines

1. Stay focused on your role and goal
2. Collaborate effectively with other participants
3. Use available tools when they help accomplish tasks
4. Communicate clearly and provide actionable responses
5. Ask clarifying questions when the request is ambiguous
"""
            self._system_prompt = base_prompt + crewai_section

        logger.info(f"CrewAI adapter started for agent: {agent_name}")

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
        Handle incoming message using CrewAI-style processing.

        Implements a tool loop similar to CrewAI's agent execution,
        with platform tools for collaboration.
        """
        logger.debug(f"Handling message {msg.id} in room {room_id}")

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

        # Inject participants message if changed
        if participants_msg:
            self._message_history[room_id].append(
                {
                    "role": "system",
                    "content": f"[Crew Update]: {participants_msg}",
                }
            )
            logger.info(f"Room {room_id}: Participants updated")

        # Add current message
        user_message = msg.format_for_llm()
        self._message_history[room_id].append(
            {
                "role": "user",
                "content": user_message,
            }
        )

        # Log message count
        total_messages = len(self._message_history[room_id])
        logger.info(
            f"Room {room_id}: Processing with {total_messages} messages "
            f"(first_msg={is_session_bootstrap})"
        )

        # Get tool schemas in OpenAI format
        tool_schemas = tools.get_openai_tool_schemas()

        # Build messages for LLM call
        messages = self._build_messages(room_id)

        # Tool loop (similar to CrewAI's agent execution)
        iteration = 0
        while iteration < self._max_tool_iterations:
            iteration += 1

            if self.verbose:
                logger.info(f"Room {room_id}: Iteration {iteration}")

            try:
                response = await self._call_llm(messages, tool_schemas)
            except Exception as e:
                logger.error(f"Error calling LLM: {e}", exc_info=True)
                await self._report_error(tools, str(e))
                raise

            # Check for tool calls
            tool_calls = response.get("tool_calls", [])

            if not tool_calls:
                # No more tool calls - extract content
                content = response.get("content", "")
                if content:
                    self._message_history[room_id].append(
                        {
                            "role": "assistant",
                            "content": content,
                        }
                    )
                logger.debug(f"Room {room_id}: Completed without tool calls")
                break

            # Add assistant response with tool calls to history
            self._message_history[room_id].append(
                {
                    "role": "assistant",
                    "content": response.get("content"),
                    "tool_calls": tool_calls,
                }
            )

            # Process tool calls
            tool_results = await self._process_tool_calls(tool_calls, tools)

            # Add tool results to history
            for result in tool_results:
                self._message_history[room_id].append(result)

            # Update messages for next iteration
            messages = self._build_messages(room_id)

        if iteration >= self._max_tool_iterations:
            logger.warning(
                f"Room {room_id}: Hit max tool iterations ({self._max_tool_iterations})"
            )

        logger.debug(
            f"Message {msg.id} processed successfully "
            f"(history now has {len(self._message_history[room_id])} messages)"
        )

    def _build_messages(self, room_id: str) -> list[dict[str, Any]]:
        """Build messages list with system prompt and history."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt}
        ]

        # Add conversation history
        messages.extend(self._message_history.get(room_id, []))

        return messages

    async def _call_llm(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Call the LLM with messages and tools.

        Uses OpenAI-compatible API format.
        """
        try:
            import openai
        except ImportError:
            raise ImportError(
                "OpenAI package required for CrewAIAdapter. "
                "Install with: pip install openai"
            )

        # Get API key from parameter or environment
        api_key = self.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass openai_api_key parameter to CrewAIAdapter."
            )

        client = openai.AsyncOpenAI(api_key=api_key)

        # Build request
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }

        if tools:
            request_kwargs["tools"] = tools
            request_kwargs["tool_choice"] = "auto"

        response = await client.chat.completions.create(**request_kwargs)

        # Extract response
        choice = response.choices[0]
        message = choice.message

        result: dict[str, Any] = {
            "content": message.content,
            "tool_calls": [],
        }

        if message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]

        return result

    async def _process_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
        tools: AgentToolsProtocol,
    ) -> list[dict[str, Any]]:
        """
        Process tool calls and execute them.

        Returns tool result messages in OpenAI format.
        """
        results: list[dict[str, Any]] = []

        for tc in tool_calls:
            tool_id = tc.get("id", "")
            function = tc.get("function", {})
            tool_name = function.get("name", "")
            arguments_str = function.get("arguments", "{}")

            try:
                arguments = json.loads(arguments_str)
            except json.JSONDecodeError:
                arguments = {}

            logger.debug(f"Executing tool: {tool_name} with args: {arguments}")

            # Report tool call if enabled
            if self.enable_execution_reporting:
                await tools.send_event(
                    content=f"Calling {tool_name}",
                    message_type="tool_call",
                    metadata={"tool": tool_name, "input": arguments},
                )

            # Execute tool
            try:
                result = await tools.execute_tool_call(tool_name, arguments)
                result_str = (
                    json.dumps(result, default=str)
                    if not isinstance(result, str)
                    else result
                )
                is_error = False
            except Exception as e:
                result_str = f"Error: {e}"
                is_error = True
                logger.error(f"Tool {tool_name} failed: {e}")

            # Report tool result if enabled
            if self.enable_execution_reporting:
                await tools.send_event(
                    content=(
                        f"Result: {result_str[:200]}..."
                        if len(result_str) > 200
                        else f"Result: {result_str}"
                    ),
                    message_type="tool_result",
                    metadata={"tool": tool_name, "is_error": is_error},
                )

            # Add to results in OpenAI tool message format
            results.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": result_str,
                }
            )

        return results

    async def on_cleanup(self, room_id: str) -> None:
        """Clean up message history when agent leaves a room."""
        if room_id in self._message_history:
            del self._message_history[room_id]
            logger.debug(f"Room {room_id}: Cleaned up message history")

    async def _report_error(self, tools: AgentToolsProtocol, error: str) -> None:
        """Send error event (best effort)."""
        try:
            await tools.send_event(content=f"Error: {error}", message_type="error")
        except Exception:
            pass

