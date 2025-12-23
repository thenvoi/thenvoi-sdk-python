"""
Anthropic tool execution and event reporting.

Handles processing tool_use blocks, executing tools, and reporting
events to the platform for history reconstruction.
"""

from __future__ import annotations

import json
import logging

from anthropic.types import Message, ToolResultBlockParam, ToolUseBlock

from thenvoi.runtime import AgentTools

logger = logging.getLogger(__name__)


class AnthropicToolExecutor:
    """
    Executes tool calls and reports events to the platform.

    Responsibilities:
    - Process tool_use blocks from Anthropic responses
    - Execute tools via AgentTools
    - Report tool_call and tool_result events for platform history
    """

    async def process_tool_calls(
        self, response: Message, tools: AgentTools
    ) -> list[ToolResultBlockParam]:
        """
        Process tool_use blocks from response and execute tools.

        Args:
            response: Anthropic Message with tool_use blocks
            tools: AgentTools instance for execution

        Returns:
            List of tool_result content blocks for next API call
        """
        tool_results: list[ToolResultBlockParam] = []

        for block in response.content:
            if not isinstance(block, ToolUseBlock):
                continue

            tool_name = block.name
            tool_input = block.input
            tool_use_id = block.id

            logger.debug(f"Executing tool: {tool_name} with input: {tool_input}")

            # Report tool call (required for history reconstruction)
            await self._report_tool_call(tools, tool_use_id, tool_name, tool_input)

            # Execute tool
            try:
                result = await tools.execute_tool_call(tool_name, tool_input)
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

            # Report tool result (required for history reconstruction)
            await self._report_tool_result(
                tools, tool_use_id, tool_name, result_str, is_error
            )

            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": result_str,
                    "is_error": is_error,
                }
            )

        return tool_results

    async def _report_tool_call(
        self,
        tools: AgentTools,
        tool_use_id: str,
        name: str,
        input_data: object,
    ) -> None:
        """Report tool_call event to platform."""
        await tools.send_event(
            content=json.dumps(
                {
                    "run_id": tool_use_id,
                    "name": name,
                    "data": {"input": input_data},
                },
                default=str,
            ),
            message_type="tool_call",
            metadata=None,
        )

    async def _report_tool_result(
        self,
        tools: AgentTools,
        tool_use_id: str,
        name: str,
        output: str,
        is_error: bool,
    ) -> None:
        """Report tool_result event to platform."""
        await tools.send_event(
            content=json.dumps(
                {
                    "run_id": tool_use_id,
                    "name": name,
                    "data": {"output": output, "is_error": is_error},
                },
                default=str,
            ),
            message_type="tool_result",
            metadata=None,
        )
