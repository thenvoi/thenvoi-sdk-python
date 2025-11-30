"""Basic agent without any framework.

This module provides a simple agent implementation that demonstrates
the use of the client SDKs without any AI framework.

Coming in a future release.
"""


def __getattr__(name):
    raise NotImplementedError(
        "Basic agent is not yet implemented. "
        "This module is planned for a future release. "
        "Currently available: thenvoi.agent.langgraph"
    )
