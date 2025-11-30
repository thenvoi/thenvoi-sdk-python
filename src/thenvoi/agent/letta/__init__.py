"""Letta adapter for Thenvoi - Coming in future release."""


def __getattr__(name):
    raise NotImplementedError(
        "Letta adapter is not yet implemented. "
        "This adapter is planned for a future release. "
        "Currently available: thenvoi.agent.langgraph"
    )
