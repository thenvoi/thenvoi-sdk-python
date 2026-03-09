"""
Google ADK adapter using SimpleAdapter pattern.

Integrates with the Google Agent Development Kit (ADK) to run Gemini-powered
agents on the Thenvoi platform. Uses ADK's built-in Runner for tool loop
management and session handling.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import TYPE_CHECKING, Any

from thenvoi.core.protocols import AgentToolsProtocol
from thenvoi.core.simple_adapter import SimpleAdapter
from thenvoi.core.types import PlatformMessage
from thenvoi.converters.google_adk import GoogleADKHistoryConverter, GoogleADKMessages
from thenvoi.runtime.custom_tools import (
    CustomToolDef,
    custom_tools_to_schemas,
    execute_custom_tool,
    find_custom_tool,
)
from thenvoi.runtime.prompts import render_system_prompt

if TYPE_CHECKING:
    from google.adk.tools import ToolContext

logger = logging.getLogger(__name__)

_APP_NAME = "thenvoi"


def _import_adk() -> tuple:
    """Lazily import Google ADK dependencies.

    Returns:
        (ADKAgent, InMemoryRunner, BaseTool, types) tuple.

    Raises:
        ImportError: If google-adk is not installed.
    """
    try:
        from google.adk import Agent as ADKAgent
        from google.adk.runners import InMemoryRunner
        from google.adk.tools import BaseTool
        from google.genai import types
    except ImportError as exc:
        raise ImportError(
            "google-adk is required for GoogleADKAdapter. "
            "Install with: pip install thenvoi-sdk[google_adk]"
        ) from exc
    return ADKAgent, InMemoryRunner, BaseTool, types


# Resolve base class lazily — BaseTool is needed at class-definition time
# for _ThenvoiToolBridge, so we import it eagerly but guard with a clear error.
_ADKAgent, _InMemoryRunner, _BaseTool, _types = _import_adk()


class _ThenvoiToolBridge(_BaseTool):
    """Bridges a Thenvoi platform tool to Google ADK.

    Wraps a tool schema from AgentToolsProtocol into a BaseTool that ADK
    can register with its agent. Execution delegates to the platform's
    execute_tool_call method.
    """

    # Inherited from BaseTool (declared for pyrefly visibility)
    name: str
    description: str

    def __init__(
        self,
        tool_name: str,
        tool_description: str,
        parameters_schema: dict[str, Any],
        tools_ref: list[AgentToolsProtocol | None],
        custom_tools_ref: list[list[CustomToolDef]],
    ):
        super().__init__(name=tool_name, description=tool_description)
        self._parameters_schema = parameters_schema
        # Mutable lists holding current references (updated per on_message)
        self._tools_ref = tools_ref
        self._custom_tools_ref = custom_tools_ref

    def _get_declaration(self) -> _types.FunctionDeclaration:
        """Build a FunctionDeclaration from the OpenAI-format schema."""
        return _types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=self._convert_parameters(self._parameters_schema),
        )

    async def run_async(
        self,
        *,
        args: dict[str, Any],
        tool_context: ToolContext,
    ) -> Any:
        """Execute the tool via Thenvoi's AgentToolsProtocol."""
        tools = self._tools_ref[0]
        if tools is None:
            return {"error": "Tool execution context not available"}

        custom_tools = self._custom_tools_ref[0]

        try:
            custom_tool = find_custom_tool(custom_tools, self.name)
            if custom_tool:
                result = await execute_custom_tool(custom_tool, args)
            else:
                result = await tools.execute_tool_call(self.name, args)

            if not isinstance(result, str):
                return json.dumps(result, default=str)
            return result
        except Exception as e:
            logger.error("Tool %s failed: %s", self.name, e)
            return f"Error: {e}"

    @staticmethod
    def _convert_parameters(openai_params: dict[str, Any]) -> dict[str, Any]:
        """Convert OpenAI JSON Schema parameters to Gemini format.

        Gemini does not support the ``additionalProperties`` key in function
        parameter schemas.  Passing it causes ``google.genai`` to reject the
        declaration with a validation error.  This helper strips the key
        recursively so the schema is compatible.
        """
        if not isinstance(openai_params, dict):
            return openai_params

        cleaned: dict[str, Any] = {}
        for key, value in openai_params.items():
            if key == "additionalProperties":
                continue
            if isinstance(value, dict):
                cleaned[key] = _ThenvoiToolBridge._convert_parameters(value)
            elif isinstance(value, list):
                cleaned[key] = [
                    _ThenvoiToolBridge._convert_parameters(item)
                    if isinstance(item, dict)
                    else item
                    for item in value
                ]
            else:
                cleaned[key] = value
        return cleaned


class GoogleADKAdapter(SimpleAdapter[GoogleADKMessages]):
    """
    Google ADK adapter using SimpleAdapter pattern.

    Uses Google's Agent Development Kit with Gemini models for agent
    interactions, with automatic tool bridging and session management.

    Tool bridges (``_ThenvoiToolBridge``) are created each time a runner is
    built and need access to the *current* ``AgentToolsProtocol`` for the
    message being handled.  Because the bridges are instantiated before
    ``on_message`` receives the ``tools`` argument, the adapter stores
    ``_tools_ref`` and ``_custom_tools_ref`` as single-element mutable lists
    so the bridges can read an up-to-date reference without being recreated.

    Example:
        adapter = GoogleADKAdapter(
            model="gemini-2.5-flash",
            custom_section="You are a helpful assistant.",
        )
        agent = Agent.create(adapter=adapter, agent_id="...", api_key="...")
        await agent.run()
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        system_prompt: str | None = None,
        custom_section: str | None = None,
        enable_execution_reporting: bool = False,
        enable_memory_tools: bool = False,
        history_converter: GoogleADKHistoryConverter | None = None,
        additional_tools: list[CustomToolDef] | None = None,
    ):
        super().__init__(
            history_converter=history_converter or GoogleADKHistoryConverter()
        )

        self.model = model
        self.system_prompt = system_prompt
        self.custom_section = custom_section
        self.enable_execution_reporting = enable_execution_reporting
        self.enable_memory_tools = enable_memory_tools

        # Custom tools (user-provided)
        self._custom_tools: list[CustomToolDef] = additional_tools or []

        # Rendered system prompt (set after start)
        self._system_prompt: str = ""

        # Per-room session IDs
        self._room_sessions: dict[str, str] = {}

        # Mutable reference holders for tool bridge (updated per on_message)
        self._tools_ref: list[AgentToolsProtocol | None] = [None]
        self._custom_tools_ref: list[list[CustomToolDef]] = [self._custom_tools]

    async def on_started(self, agent_name: str, agent_description: str) -> None:
        """Render system prompt and create ADK agent after metadata is fetched."""
        await super().on_started(agent_name, agent_description)
        self._system_prompt = self.system_prompt or render_system_prompt(
            agent_name=agent_name,
            agent_description=agent_description,
            custom_section=self.custom_section or "",
        )

        logger.info("Google ADK adapter started for agent: %s", agent_name)

    def _build_adk_tools(self, tools: AgentToolsProtocol) -> list[_ThenvoiToolBridge]:
        """Build ADK tool bridges from Thenvoi tool schemas."""
        openai_schemas = tools.get_openai_tool_schemas(
            include_memory=self.enable_memory_tools
        )

        adk_tools: list[_ThenvoiToolBridge] = []
        for schema in openai_schemas:
            func_def = schema["function"]
            adk_tools.append(
                _ThenvoiToolBridge(
                    tool_name=func_def["name"],
                    tool_description=func_def.get("description", ""),
                    parameters_schema=func_def.get("parameters", {}),
                    tools_ref=self._tools_ref,
                    custom_tools_ref=self._custom_tools_ref,
                )
            )

        # Add custom tool bridges
        if self._custom_tools:
            custom_schemas = custom_tools_to_schemas(self._custom_tools, "openai")
            for schema in custom_schemas:
                func_def = schema["function"]
                adk_tools.append(
                    _ThenvoiToolBridge(
                        tool_name=func_def["name"],
                        tool_description=func_def.get("description", ""),
                        parameters_schema=func_def.get("parameters", {}),
                        tools_ref=self._tools_ref,
                        custom_tools_ref=self._custom_tools_ref,
                    )
                )

        return adk_tools

    def _create_runner(self, tools: AgentToolsProtocol) -> _InMemoryRunner:
        """Create a fresh ADK InMemoryRunner with current tools."""
        adk_tools = self._build_adk_tools(tools)

        adk_agent = _ADKAgent(
            name=self.agent_name or "thenvoi_agent",
            model=self.model,
            instruction=self._system_prompt,
            tools=adk_tools,
        )

        return _InMemoryRunner(
            agent=adk_agent,
            app_name=_APP_NAME,
        )

    async def on_message(
        self,
        msg: PlatformMessage,
        tools: AgentToolsProtocol,
        history: GoogleADKMessages,
        participants_msg: str | None,
        contacts_msg: str | None,
        *,
        is_session_bootstrap: bool,
        room_id: str,
    ) -> None:
        """
        Handle incoming message via Google ADK.

        Uses ADK's Runner for the full tool loop. The runner handles
        LLM calls, tool execution, and conversation management.
        """
        logger.debug("Handling message %s in room %s", msg.id, room_id)

        # Update mutable tool references for the bridge
        self._tools_ref[0] = tools
        self._custom_tools_ref[0] = self._custom_tools

        # A fresh runner is created per message because InMemoryRunner
        # accumulates session history internally and tool schemas may change
        # between calls.  History is injected as a text transcript instead.
        runner = self._create_runner(tools)

        # Get or create session for this room
        if is_session_bootstrap or room_id not in self._room_sessions:
            session_id = str(uuid.uuid4())
            self._room_sessions[room_id] = session_id
            logger.info("Room %s: Created new ADK session %s", room_id, session_id)
        else:
            session_id = self._room_sessions[room_id]

        # Build the user message content
        parts: list[str] = []

        # Inject history context on bootstrap
        if is_session_bootstrap and history:
            transcript = self._format_history_transcript(history)
            if transcript:
                parts.append(
                    f"[Previous conversation context]\n{transcript}\n"
                    f"[End of previous context]\n\n"
                )
            logger.info(
                "Room %s: Injected %s historical messages as context",
                room_id,
                len(history),
            )

        # Inject participants update
        if participants_msg:
            parts.append(f"[System]: {participants_msg}")
            logger.info("Room %s: Participants updated", room_id)

        # Inject contacts update
        if contacts_msg:
            parts.append(f"[System]: {contacts_msg}")
            logger.info("Room %s: Contacts broadcast received", room_id)

        # Add the actual message
        parts.append(msg.format_for_llm())

        user_content = _types.Content(
            role="user",
            parts=[_types.Part.from_text(text="\n".join(parts))],
        )

        logger.info(
            "Room %s: Running ADK agent (bootstrap=%s)",
            room_id,
            is_session_bootstrap,
        )

        # Run the ADK agent - it handles the full tool loop
        try:
            async for event in runner.run_async(
                user_id=room_id,
                session_id=session_id,
                new_message=user_content,
            ):
                # Report tool calls/results if enabled
                if self.enable_execution_reporting:
                    await self._report_event(event, tools)

                if event.is_final_response():
                    logger.debug(
                        "Room %s: ADK agent completed with final response",
                        room_id,
                    )
        except Exception as e:
            logger.error("Error running ADK agent: %s", e, exc_info=True)
            await self._report_error(tools, str(e))
            raise
        finally:
            await runner.close()

        logger.debug("Message %s processed successfully", msg.id)

    async def on_cleanup(self, room_id: str) -> None:
        """Clean up session when agent leaves a room."""
        if room_id in self._room_sessions:
            del self._room_sessions[room_id]
            logger.debug("Room %s: Cleaned up ADK session", room_id)

    def _format_history_transcript(self, history: GoogleADKMessages) -> str:
        """Format converted history as a text transcript for context injection."""
        lines: list[str] = []
        for msg in history:
            content = msg.get("content", "")
            if isinstance(content, str):
                lines.append(content)
            elif isinstance(content, list):
                # Tool call/result blocks - summarize
                for block in content:
                    if isinstance(block, dict):
                        block_type = block.get("type", "")
                        if block_type == "function_call":
                            lines.append(
                                f"[Tool Call] {block.get('name', 'unknown')}"
                                f" ({json.dumps(block.get('args', {}), default=str)})"
                            )
                        elif block_type == "function_response":
                            output = str(block.get("output", ""))
                            truncated = (
                                output[:200] + "..." if len(output) > 200 else output
                            )
                            lines.append(
                                f"[Tool Result] {block.get('name', 'unknown')}: "
                                f"{truncated}"
                            )
        return "\n".join(lines)

    async def _report_event(self, event: Any, tools: AgentToolsProtocol) -> None:
        """Report ADK event as tool_call/tool_result if applicable."""
        try:
            function_calls = event.get_function_calls()
            if function_calls:
                for fc in function_calls:
                    try:
                        await tools.send_event(
                            content=json.dumps(
                                {
                                    "name": fc.name,
                                    "args": dict(fc.args) if fc.args else {},
                                    "tool_call_id": fc.id if hasattr(fc, "id") else "",
                                }
                            ),
                            message_type="tool_call",
                        )
                    except Exception as e:
                        logger.warning("Failed to send tool_call event: %s", e)

            function_responses = event.get_function_responses()
            if function_responses:
                for fr in function_responses:
                    try:
                        await tools.send_event(
                            content=json.dumps(
                                {
                                    "name": fr.name,
                                    "output": str(fr.response) if fr.response else "",
                                    "tool_call_id": fr.id if hasattr(fr, "id") else "",
                                }
                            ),
                            message_type="tool_result",
                        )
                    except Exception as e:
                        logger.warning("Failed to send tool_result event: %s", e)
        except AttributeError:
            # Event may not have get_function_calls/get_function_responses — that's OK
            pass

    async def _report_error(self, tools: AgentToolsProtocol, error: str) -> None:
        """Send error event (best effort)."""
        try:
            await tools.send_event(content=f"Error: {error}", message_type="error")
        except Exception as e:
            logger.warning("Failed to send error event: %s", e)
