"""Gemini adapter using the official google-genai SDK."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, cast

import httpx
from pydantic import ValidationError

try:
    from google import genai
    from google.genai import types
    from google.genai.errors import ServerError
except ImportError as e:
    raise ImportError(
        "google-genai is required for Gemini adapter.\n"
        "Install with: pip install 'thenvoi-sdk[gemini]'\n"
        "Or: uv add google-genai"
    ) from e

from thenvoi.core.protocols import AgentToolsProtocol
from thenvoi.core.simple_adapter import SimpleAdapter
from thenvoi.core.types import PlatformMessage
from thenvoi.converters.gemini import GeminiHistoryConverter, GeminiMessages
from thenvoi.runtime.custom_tools import (
    CustomToolDef,
    execute_custom_tool,
    find_custom_tool,
)
from thenvoi.runtime.prompts import render_system_prompt

logger = logging.getLogger(__name__)


class GeminiAdapter(SimpleAdapter[GeminiMessages]):
    """
    Gemini SDK adapter using SimpleAdapter pattern.

    Uses the official google-genai Python SDK with explicit tool-loop control.
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        gemini_api_key: str | None = None,
        system_prompt: str | None = None,
        custom_section: str | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        enable_execution_reporting: bool = False,
        enable_memory_tools: bool = False,
        max_tool_rounds: int = 20,
        max_retries: int = 2,
        retry_base_delay_s: float = 1.0,
        max_history_messages: int = 200,
        history_converter: GeminiHistoryConverter | None = None,
        additional_tools: list[CustomToolDef] | None = None,
    ) -> None:
        super().__init__(
            history_converter=history_converter or GeminiHistoryConverter()
        )

        self.model = model
        self.system_prompt = system_prompt
        self.custom_section = custom_section
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.enable_execution_reporting = enable_execution_reporting
        self.enable_memory_tools = enable_memory_tools
        self.max_tool_rounds = max_tool_rounds
        self.max_retries = max_retries
        self.retry_base_delay_s = retry_base_delay_s
        self.max_history_messages = max_history_messages

        self._gemini_api_key = gemini_api_key
        self.client: genai.Client | None = None
        self._message_history: dict[str, GeminiMessages] = {}
        self._system_prompt: str = ""
        self._custom_tools: list[CustomToolDef] = additional_tools or []

    async def on_started(self, agent_name: str, agent_description: str) -> None:
        """Render system prompt after agent metadata is fetched."""
        await super().on_started(agent_name, agent_description)
        self._system_prompt = self.system_prompt or render_system_prompt(
            agent_name=agent_name,
            agent_description=agent_description,
            custom_section=self.custom_section or "",
        )
        logger.info("Gemini adapter started for agent: %s", agent_name)

    async def on_message(
        self,
        msg: PlatformMessage,
        tools: AgentToolsProtocol,
        history: GeminiMessages,
        participants_msg: str | None,
        contacts_msg: str | None,
        *,
        is_session_bootstrap: bool,
        room_id: str,
    ) -> None:
        """Handle incoming message with explicit function-calling loop."""
        if is_session_bootstrap:
            if history:
                self._message_history[room_id] = list(history)
                logger.info(
                    "Room %s: Loaded %s historical Gemini messages",
                    room_id,
                    len(history),
                )
            else:
                self._message_history[room_id] = []
                logger.info("Room %s: No historical messages found", room_id)
        elif room_id not in self._message_history:
            self._message_history[room_id] = []

        if participants_msg:
            self._message_history[room_id].append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=f"[System]: {participants_msg}")],
                )
            )

        if contacts_msg:
            self._message_history[room_id].append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=f"[System]: {contacts_msg}")],
                )
            )

        self._message_history[room_id].append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=msg.format_for_llm())],
            )
        )

        self._trim_history(room_id)
        gemini_tools = self._build_gemini_tools(tools)
        tool_rounds = 0
        while True:
            if tool_rounds >= self.max_tool_rounds:
                raise RuntimeError(
                    f"Exceeded max tool rounds ({self.max_tool_rounds}) in room {room_id}"
                )

            try:
                response = await self._call_gemini(
                    contents=self._message_history[room_id], tools=gemini_tools
                )
            except Exception as e:
                logger.exception("Error calling Gemini: %s", e)
                await self._report_error(tools, str(e))
                raise

            candidate_content = self._extract_candidate_content(response)
            if candidate_content is not None:
                self._message_history[room_id].append(candidate_content)

            function_calls = list(response.function_calls or [])
            if not function_calls:
                break

            tool_response_parts = await self._process_function_calls(
                function_calls=function_calls,
                tools=tools,
            )
            if tool_response_parts:
                self._message_history[room_id].append(
                    types.Content(role="user", parts=tool_response_parts)
                )

            tool_rounds += 1

    async def on_cleanup(self, room_id: str) -> None:
        """Clean up message history when the agent leaves a room."""
        self._message_history.pop(room_id, None)
        logger.debug("Room %s: Cleaned up Gemini history", room_id)

    def _trim_history(self, room_id: str) -> None:
        """Trim message history to stay within ``max_history_messages``."""
        history = self._message_history.get(room_id)
        if history and len(history) > self.max_history_messages:
            trimmed = len(history) - self.max_history_messages
            self._message_history[room_id] = history[-self.max_history_messages :]
            logger.debug(
                "Room %s: Trimmed %s oldest messages (kept %s)",
                room_id,
                trimmed,
                self.max_history_messages,
            )

    def _ensure_client(self) -> genai.Client:
        """Create client lazily to avoid requiring API key during adapter init."""
        if self.client is not None:
            return self.client

        try:
            self.client = genai.Client(api_key=self._gemini_api_key)
        except ValueError as e:
            raise ValueError(
                "Gemini client initialization failed. Provide GEMINI_API_KEY or "
                "pass gemini_api_key explicitly."
            ) from e
        return self.client

    async def _call_gemini(
        self,
        contents: GeminiMessages,
        tools: list[types.Tool],
    ) -> types.GenerateContentResponse:
        """
        Call Gemini API with bounded retries for transient transport/server failures.
        """
        config = types.GenerateContentConfig(
            system_instruction=self._system_prompt,
            tools=tools or None,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        )
        if self.max_output_tokens is not None:
            config.max_output_tokens = self.max_output_tokens
        if self.temperature is not None:
            config.temperature = self.temperature

        max_attempts = self.max_retries + 1
        client = self._ensure_client()
        for attempt in range(1, max_attempts + 1):
            try:
                return await client.aio.models.generate_content(
                    model=self.model,
                    contents=cast(Any, contents),
                    config=config,
                )
            except (ServerError, httpx.TimeoutException, httpx.TransportError) as e:
                if attempt >= max_attempts:
                    raise
                delay_s = self.retry_base_delay_s * (2 ** (attempt - 1))
                logger.warning(
                    "Gemini transient error on attempt %s/%s: %s (retrying in %.2fs)",
                    attempt,
                    max_attempts,
                    e,
                    delay_s,
                )
                await asyncio.sleep(delay_s)
        raise RuntimeError("Gemini call retry loop exhausted unexpectedly")

    def _build_gemini_tools(self, tools: AgentToolsProtocol) -> list[types.Tool]:
        """Build Gemini function declarations from platform and custom tools."""
        declarations: list[types.FunctionDeclaration] = []

        openai_schemas = tools.get_openai_tool_schemas(
            include_memory=self.enable_memory_tools
        )
        for schema in openai_schemas:
            function = schema.get("function", {})
            name = function.get("name")
            if not name:
                continue
            parameters = function.get(
                "parameters", {"type": "object", "properties": {}}
            )
            declarations.append(
                types.FunctionDeclaration(
                    name=name,
                    description=function.get("description", "") or "",
                    parameters_json_schema=parameters,
                )
            )

        for input_model, _func in self._custom_tools:
            schema = input_model.model_json_schema()
            schema.pop("title", None)
            tool_name = input_model.__name__
            if tool_name.endswith("Input"):
                tool_name = tool_name[:-5]
            tool_name = tool_name.lower()
            declarations.append(
                types.FunctionDeclaration(
                    name=tool_name,
                    description=input_model.__doc__ or "",
                    parameters_json_schema=schema,
                )
            )

        if not declarations:
            return []
        return [types.Tool(function_declarations=declarations)]

    def _extract_candidate_content(
        self, response: types.GenerateContentResponse
    ) -> types.Content | None:
        """Extract the model output content for history persistence."""
        if response.candidates and response.candidates[0].content:
            return response.candidates[0].content

        function_calls = response.function_calls or []
        if function_calls:
            parts: list[types.Part] = []
            for call in function_calls:
                if call.name:
                    parts.append(
                        types.Part(
                            function_call=types.FunctionCall(
                                id=call.id,
                                name=call.name,
                                args=dict(call.args or {}),
                            )
                        )
                    )
            if parts:
                return types.Content(role="model", parts=parts)
        return None

    async def _process_function_calls(
        self,
        function_calls: list[types.FunctionCall],
        tools: AgentToolsProtocol,
    ) -> list[types.Part]:
        """Execute model function calls and return function_response parts."""
        tool_response_parts: list[types.Part] = []

        for index, function_call in enumerate(function_calls):
            tool_name = function_call.name or ""
            tool_input = dict(function_call.args or {})
            tool_call_id = function_call.id or f"gemini_tool_call_{index}"

            if self.enable_execution_reporting:
                try:
                    await tools.send_event(
                        content=json.dumps(
                            {
                                "name": tool_name,
                                "args": tool_input,
                                "tool_call_id": tool_call_id,
                            }
                        ),
                        message_type="tool_call",
                    )
                except Exception as e:
                    logger.warning("Failed to send tool_call event: %s", e)

            try:
                custom_tool = find_custom_tool(self._custom_tools, tool_name)
                if custom_tool:
                    result = await execute_custom_tool(custom_tool, tool_input)
                else:
                    result = await tools.execute_tool_call(tool_name, tool_input)
                result_str = (
                    json.dumps(result, default=str)
                    if not isinstance(result, str)
                    else result
                )
                is_error = False
            except ValidationError as exc:
                errors = "; ".join(
                    f"{err['loc'][0]}: {err['msg']}" for err in exc.errors()
                )
                result_str = f"Invalid arguments for {tool_name}: {errors}"
                is_error = True
                logger.warning("Validation error for tool %s: %s", tool_name, errors)
            except Exception as e:
                result_str = f"Error: {e}"
                is_error = True
                logger.exception("Tool %s failed: %s", tool_name, e)

            if self.enable_execution_reporting:
                try:
                    await tools.send_event(
                        content=json.dumps(
                            {
                                "name": tool_name,
                                "output": result_str,
                                "tool_call_id": tool_call_id,
                                "is_error": is_error,
                            }
                        ),
                        message_type="tool_result",
                    )
                except Exception as e:
                    logger.warning("Failed to send tool_result event: %s", e)

            response_payload = (
                {"error": result_str} if is_error else {"output": result_str}
            )
            tool_response_parts.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        id=tool_call_id,
                        name=tool_name,
                        response=response_payload,
                    )
                )
            )

        return tool_response_parts

    async def _report_error(self, tools: AgentToolsProtocol, error: str) -> None:
        """Send error event (best effort)."""
        try:
            await tools.send_event(content=f"Error: {error}", message_type="error")
        except Exception as e:
            logger.warning("Failed to send error event: %s", e)
