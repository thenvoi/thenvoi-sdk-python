"""Message formatters for LangGraph adapter inputs."""

from typing import Dict, Any, Protocol
from thenvoi.client.streaming import MessageCreatedPayload


class MessageFormatter(Protocol):
    """Callable contract for mapping platform payloads into graph state."""

    def __call__(
        self, message: MessageCreatedPayload, sender_name: str
    ) -> Dict[str, Any]:
        """Convert a platform message payload into graph input state."""
        ...


def default_messages_state_formatter(
    message: MessageCreatedPayload, sender_name: str
) -> Dict[str, Any]:
    """Default formatter for LangGraph `MessagesState` style inputs."""
    formatted_content = (
        f"Message from {sender_name} ({message.sender_type}, ID: {message.sender_id}): "
        f"{message.content}"
    )

    return {"messages": [{"role": "user", "content": formatted_content}]}
