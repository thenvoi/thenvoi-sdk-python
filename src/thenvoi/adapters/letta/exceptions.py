"""Letta adapter exceptions."""

from __future__ import annotations


class LettaAdapterError(Exception):
    """Base exception for Letta adapter errors."""

    pass


class LettaAgentNotFoundError(LettaAdapterError):
    """Raised when a Letta agent is not found."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        super().__init__(f"Letta agent not found: {agent_id}")


class LettaConnectionError(LettaAdapterError):
    """Raised when unable to connect to Letta server."""

    pass


class LettaToolExecutionError(LettaAdapterError):
    """Raised when tool execution fails."""

    def __init__(self, tool_name: str, error: str):
        self.tool_name = tool_name
        self.error = error
        super().__init__(f"Tool '{tool_name}' failed: {error}")


class LettaMemoryError(LettaAdapterError):
    """Raised when memory operations fail."""

    pass


class LettaTimeoutError(LettaAdapterError):
    """Raised when Letta API call times out."""

    def __init__(self, operation: str, timeout: int):
        self.operation = operation
        self.timeout = timeout
        super().__init__(f"Letta operation '{operation}' timed out after {timeout}s")


class LettaConfigurationError(LettaAdapterError):
    """Raised when configuration is invalid."""

    pass
