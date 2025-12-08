"""
CrewAI integration for Thenvoi platform.

Uses composition with PlatformClient and RoomManager for clean separation.
Only implements CrewAI-specific logic.
"""

import logging
from typing import Optional, Any, List, Callable

from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool

from thenvoi.agent.core import ThenvoiPlatformClient, RoomManager
from thenvoi.client.streaming import MessageCreatedPayload
from thenvoi.agent.crewai.prompts import generate_crewai_agent_prompt
from thenvoi.agent.crewai.tools import get_thenvoi_tools, ThenvoiContext
from thenvoi.client.rest import ChatEventRequest

logger = logging.getLogger(__name__)


# Type alias for message formatters
MessageFormatter = Callable[[MessageCreatedPayload, str], str]


def default_message_formatter(message: MessageCreatedPayload, sender_name: str) -> str:
    """Default formatter for CrewAI message input.

    Formats messages into a structured string that the CrewAI agent can understand.

    Args:
        message: The message payload from WebSocket
        sender_name: Display name of the sender

    Returns:
        Formatted message string
    """
    return (
        f"A new Message received on chat_room_id: {message.chat_room_id} "
        f"from {sender_name} (ID: {message.sender_id}, sender_type: {message.sender_type}): "
        f"{message.content}"
    )


async def _send_platform_event(
    api_client, room_id: str, content: str, message_type: str
):
    """Send an event to the platform via API (shared utility)."""
    event_request = ChatEventRequest(
        content=content,
        message_type=message_type,
    )

    await api_client.chat_events.create_chat_event(chat_id=room_id, event=event_request)


async def create_crewai_agent(
    agent_id: str,
    api_key: str,
    llm: Any,
    ws_url: str,
    thenvoi_restapi_url: str,
    additional_tools: Optional[List[BaseTool]] = None,
    custom_instructions: Optional[str] = None,
    role: Optional[str] = None,
    goal: Optional[str] = None,
    backstory: Optional[str] = None,
) -> "ThenvoiCrewAIAgent":
    """
    Create and start a Thenvoi CrewAI agent (functional API).

    This is a convenience function that creates a ThenvoiCrewAIAgent
    and starts it immediately.

    Args:
        agent_id: Agent ID from platform (agent must be created externally)
        api_key: Agent-specific API key from platform
        llm: Language model instance (e.g., from langchain_openai.ChatOpenAI)
        ws_url: WebSocket URL
        thenvoi_restapi_url: REST API base URL
        additional_tools: Custom tools to add to agent (in addition to platform tools)
        custom_instructions: Additional instructions appended to base system prompt
        role: Optional custom role for the CrewAI agent
        goal: Optional custom goal for the CrewAI agent
        backstory: Optional custom backstory for the CrewAI agent

    Returns:
        ThenvoiCrewAIAgent instance (already running)

    Example:
        >>> from langchain_openai import ChatOpenAI
        >>> agent_id, api_key = load_agent_config("my_agent")
        >>> agent = await create_crewai_agent(
        ...     agent_id=agent_id,
        ...     api_key=api_key,
        ...     llm=ChatOpenAI(model="gpt-4o"),
        ...     ws_url=os.getenv("THENVOI_WS_URL"),
        ...     thenvoi_restapi_url=os.getenv("THENVOI_REST_API_URL"),
        ...     custom_instructions="You are a friendly assistant with a sense of humor.",
        ... )
        # Agent is now listening for messages
    """
    agent = ThenvoiCrewAIAgent(
        agent_id=agent_id,
        api_key=api_key,
        llm=llm,
        ws_url=ws_url,
        thenvoi_restapi_url=thenvoi_restapi_url,
        additional_tools=additional_tools,
        custom_instructions=custom_instructions,
        role=role,
        goal=goal,
        backstory=backstory,
    )
    await agent.start()
    return agent


class ThenvoiCrewAIAgent:
    """
    CrewAI agent integration with Thenvoi platform.

    Builds a CrewAI Crew with a single agent that responds to messages
    from the Thenvoi platform. Uses platform tools for communication.
    """

    def __init__(
        self,
        agent_id: str,
        api_key: str,
        llm: Any,
        ws_url: str,
        thenvoi_restapi_url: str,
        additional_tools: Optional[List[BaseTool]] = None,
        custom_instructions: Optional[str] = None,
        role: Optional[str] = None,
        goal: Optional[str] = None,
        backstory: Optional[str] = None,
    ):
        """
        Initialize a Thenvoi CrewAI agent.

        Args:
            agent_id: Agent ID from platform (agent must be created externally)
            api_key: Agent-specific API key from platform
            llm: Language model instance
            ws_url: WebSocket URL
            thenvoi_restapi_url: REST API base URL
            additional_tools: Custom tools to add to agent (in addition to platform tools)
            custom_instructions: Additional instructions appended to base system prompt
            role: Optional custom role for the CrewAI agent
            goal: Optional custom goal for the CrewAI agent  
            backstory: Optional custom backstory for the CrewAI agent
        """
        self.agent_id = agent_id
        self.api_key = api_key
        self.ws_url = ws_url
        self.thenvoi_restapi_url = thenvoi_restapi_url
        self.llm = llm
        self.additional_tools = additional_tools or []
        self.custom_instructions = custom_instructions
        self.role = role
        self.goal = goal
        self.backstory = backstory

        self._platform_client: Optional[ThenvoiPlatformClient] = None
        self.room_manager: Optional[RoomManager] = None
        self.crewai_agent: Optional[Agent] = None

    def _build_backstory(self, agent_name: str) -> str:
        """Build backstory: base prompt + optional custom instructions."""
        base_prompt = generate_crewai_agent_prompt(agent_name)

        if self.custom_instructions:
            return f"{base_prompt}\n\nAdditional Instructions:\n{self.custom_instructions}"

        return base_prompt

    def _build_agent(self, platform_client: ThenvoiPlatformClient) -> Agent:
        """Build the CrewAI agent with all tools.

        Args:
            platform_client: ThenvoiPlatformClient with fetched metadata
        """
        # Create platform tools
        platform_tools = get_thenvoi_tools(
            client=platform_client.api_client, agent_id=platform_client.agent_id
        )

        all_tools = platform_tools + self.additional_tools

        logger.debug(f"Building CrewAI agent with {len(all_tools)} tools")

        # Build backstory using agent name from platform client
        backstory = self.backstory or self._build_backstory(platform_client.name)

        # Create the CrewAI agent
        agent = Agent(
            role=self.role or f"Thenvoi Platform Agent - {platform_client.name}",
            goal=self.goal
            or "Assist users with their requests and coordinate with other agents when needed",
            backstory=backstory,
            tools=all_tools,
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
        )

        logger.debug("CrewAI agent built")
        return agent

    async def _handle_room_message(self, message: MessageCreatedPayload):
        """Handle incoming message - invoke CrewAI agent."""
        logger.debug(f"Received message in room {message.chat_room_id}")

        # Set room context for tools
        ThenvoiContext.set_room_id(message.chat_room_id)

        try:
            # Get sender name for formatting
            sender_name = await self.room_manager.get_participant_name(
                message.sender_id, message.sender_type, message.chat_room_id
            )

            # Format message
            formatted_message = default_message_formatter(message, sender_name)

            logger.debug(f"Invoking CrewAI agent in room {message.chat_room_id}")
            logger.debug(f"Formatted message: {formatted_message}")

            # Create a task for the CrewAI agent
            task = Task(
                description=formatted_message,
                expected_output="A helpful response to the user's message, sent via the send_message tool",
                agent=self.crewai_agent,
            )

            # Create a crew with just this agent and task
            crew = Crew(
                agents=[self.crewai_agent],
                tasks=[task],
                process=Process.sequential,
                verbose=True,
            )

            # Execute the crew (this is sync in CrewAI but we're in async context)
            # CrewAI handles async internally for tool execution
            result = crew.kickoff()

            logger.debug(f"CrewAI agent processed message: {result}")

        except Exception as e:
            logger.error(
                f"Error invoking CrewAI agent: {type(e).__name__}: {str(e)}",
                exc_info=True,
            )
            # Send error event to chat room
            error_content = f"Error processing message: {type(e).__name__}: {str(e)}"
            await _send_platform_event(
                self._platform_client.api_client,
                message.chat_room_id,
                error_content,
                "error",
            )
        finally:
            # Clear context after processing
            ThenvoiContext.clear()

    async def _on_room_removed(self, room_id: str):
        """Handle room removal (optional cleanup)."""
        logger.debug(f"Room removed: {room_id}")

    async def start(self):
        """
        Start the agent.

        Steps:
        1. Fetch agent metadata (to get agent name for backstory)
        2. Build the CrewAI agent
        3. Connect to WebSocket and start listening
        """
        # Step 1: Fetch agent metadata to get agent name
        self._platform_client = ThenvoiPlatformClient(
            agent_id=self.agent_id,
            api_key=self.api_key,
            ws_url=self.ws_url,
            thenvoi_restapi_url=self.thenvoi_restapi_url,
        )
        await self._platform_client.fetch_agent_metadata()

        logger.debug(
            f"Building CrewAI agent for '{self._platform_client.name}'"
        )

        # Step 2: Build the CrewAI agent
        self.crewai_agent = self._build_agent(self._platform_client)

        logger.info(f"Agent '{self._platform_client.name}' validated on platform")

        # Step 3: Connect WebSocket
        ws_client = await self._platform_client.connect_websocket()

        try:
            async with ws_client:
                logger.debug("WebSocket connected")

                # Step 4: Create room manager with our message handler
                self.room_manager = RoomManager(
                    agent_id=self._platform_client.agent_id,
                    agent_name=self._platform_client.name,
                    api_client=self._platform_client.api_client,
                    ws_client=ws_client,
                    message_handler=self._handle_room_message,
                    on_room_removed=self._on_room_removed,
                )

                # Step 5: Subscribe to all rooms and room events
                room_count = await self.room_manager.subscribe_to_all_rooms()
                await self.room_manager.subscribe_to_room_events()

                if room_count == 0:
                    logger.warning(
                        "No rooms found. Add agent to a room via the platform."
                    )

                # Keep running
                logger.info(
                    f"Agent '{self._platform_client.name}' is now listening for messages..."
                )
                await ws_client.run_forever()
        except Exception as e:
            logger.error(
                f"Agent '{self._platform_client.name}' disconnected: {type(e).__name__}: {e}",
                exc_info=True,
            )
            raise
        finally:
            logger.info(f"Agent '{self._platform_client.name}' stopped")


class ConnectedCrewAgent:
    """
    Runs a user-provided CrewAI Crew directly with Thenvoi platform.

    This allows users to bring their own Crew and connect it
    directly to chat room messages.

    The user's crew receives chat messages as tasks and can use Thenvoi
    platform tools from any agent.
    """

    def __init__(
        self,
        crew: Crew,
        platform_client: ThenvoiPlatformClient,
        task_builder: Optional[Callable[[str, str], Task]] = None,
        message_formatter: MessageFormatter = default_message_formatter,
    ):
        """
        Initialize a connected crew agent.

        Args:
            crew: User's CrewAI Crew
            platform_client: ThenvoiPlatformClient instance for platform communication
            task_builder: Optional function to build tasks from messages.
                         Signature: (formatted_message: str, room_id: str) -> Task
                         Default: Creates a simple response task
            message_formatter: Function to convert platform messages to string input.
                             Default: converts to structured message format
        """
        self.platform = platform_client
        self.crew = crew
        self.task_builder = task_builder
        self.message_formatter = message_formatter

    def _default_task_builder(self, formatted_message: str, room_id: str) -> Task:
        """Build a default task from the message."""
        # Use the first agent in the crew
        agent = self.crew.agents[0] if self.crew.agents else None
        return Task(
            description=formatted_message,
            expected_output="A helpful response to the user's message",
            agent=agent,
        )

    async def _handle_room_message(self, message: MessageCreatedPayload):
        """Handle incoming message - invoke user's crew."""
        logger.debug(f"Received message in room {message.chat_room_id}")

        # Set room context for tools
        ThenvoiContext.set_room_id(message.chat_room_id)

        try:
            # Get sender name for formatting
            sender_name = await self.room_manager.get_participant_name(
                message.sender_id, message.sender_type, message.chat_room_id
            )

            # Format message
            formatted_message = self.message_formatter(message, sender_name)

            logger.debug(f"Invoking user crew in room {message.chat_room_id}")
            logger.debug(f"Formatted message: {formatted_message}")

            # Build task using provided builder or default
            if self.task_builder:
                task = self.task_builder(formatted_message, message.chat_room_id)
            else:
                task = self._default_task_builder(
                    formatted_message, message.chat_room_id
                )

            # Execute with the task
            result = self.crew.kickoff(inputs={"message": formatted_message})

            logger.debug(f"User crew processed message: {result}")

        except Exception as e:
            logger.error(
                f"Error invoking user crew: {type(e).__name__}: {str(e)}",
                exc_info=True,
            )
            # Send error event to chat room
            error_content = f"Error processing message: {type(e).__name__}: {str(e)}"
            await _send_platform_event(
                self.platform.api_client,
                message.chat_room_id,
                error_content,
                "error",
            )
        finally:
            ThenvoiContext.clear()

    async def _on_room_removed(self, room_id: str):
        """Handle room removal (optional cleanup)."""
        logger.debug(f"Room removed: {room_id}")

    async def start(self):
        """
        Start the agent.

        Steps:
        1. Validate agent on platform
        2. Connect to WebSocket
        3. Subscribe to rooms
        4. Listen for messages
        """
        # Step 1: Fetch agent metadata from platform
        await self.platform.fetch_agent_metadata()
        logger.info(f"Agent '{self.platform.name}' validated on platform")

        # Step 2: Connect WebSocket
        ws_client = await self.platform.connect_websocket()

        try:
            async with ws_client:
                logger.debug("WebSocket connected")

                # Step 3: Create room manager with our message handler
                self.room_manager = RoomManager(
                    agent_id=self.platform.agent_id,
                    agent_name=self.platform.name,
                    api_client=self.platform.api_client,
                    ws_client=ws_client,
                    message_handler=self._handle_room_message,
                    on_room_removed=self._on_room_removed,
                )

                # Step 4: Subscribe to all rooms and room events
                room_count = await self.room_manager.subscribe_to_all_rooms()
                await self.room_manager.subscribe_to_room_events()

                if room_count == 0:
                    logger.warning(
                        "No rooms found. Add agent to a room via the platform."
                    )

                # Keep running
                logger.info(
                    f"Agent '{self.platform.name}' is now listening for messages..."
                )
                logger.info("Messages will be passed directly to user's crew")
                await ws_client.run_forever()
        except Exception as e:
            logger.error(
                f"Agent '{self.platform.name}' disconnected: {type(e).__name__}: {e}",
                exc_info=True,
            )
            raise
        finally:
            logger.info(f"Agent '{self.platform.name}' stopped")


async def connect_crew_to_platform(
    crew: Crew,
    platform_client: ThenvoiPlatformClient,
    task_builder: Optional[Callable[[str, str], Task]] = None,
    message_formatter: MessageFormatter = default_message_formatter,
) -> ConnectedCrewAgent:
    """
    Connect a custom CrewAI Crew to the Thenvoi platform.

    This function provides the input from Thenvoi chat rooms to your crew.
    Inside your crew, you decide when and how to use platform tools.

    Args:
        crew: Your CrewAI Crew instance
        platform_client: ThenvoiPlatformClient instance for platform communication
        task_builder: Optional function to build tasks from messages.
                     Signature: (formatted_message: str, room_id: str) -> Task
        message_formatter: Function to convert platform messages to string input.
                          Default: converts to structured message format

    Returns:
        ConnectedCrewAgent instance (already running)

    Example:
        >>> from crewai import Agent, Task, Crew, Process
        >>> from thenvoi.agent.crewai import connect_crew_to_platform, get_thenvoi_tools
        >>> from thenvoi.agent.core import ThenvoiPlatformClient
        >>>
        >>> # Create platform client
        >>> platform_client = ThenvoiPlatformClient(
        ...     agent_id=agent_id,
        ...     api_key=api_key,
        ...     ws_url=ws_url,
        ...     thenvoi_restapi_url=thenvoi_restapi_url,
        ... )
        >>>
        >>> # Get platform tools
        >>> platform_tools = get_thenvoi_tools(
        ...     client=platform_client.api_client,
        ...     agent_id=agent_id
        ... )
        >>>
        >>> # Create your agents with platform tools
        >>> researcher = Agent(
        ...     role="Researcher",
        ...     goal="Research topics thoroughly",
        ...     backstory="Expert researcher",
        ...     tools=platform_tools,
        ...     llm=llm,
        ... )
        >>>
        >>> # Create your crew
        >>> my_crew = Crew(
        ...     agents=[researcher],
        ...     tasks=[],  # Tasks are created dynamically from messages
        ...     process=Process.sequential,
        ... )
        >>>
        >>> agent = await connect_crew_to_platform(
        ...     crew=my_crew,
        ...     platform_client=platform_client,
        ... )
    """
    agent = ConnectedCrewAgent(
        crew=crew,
        platform_client=platform_client,
        task_builder=task_builder,
        message_formatter=message_formatter,
    )
    await agent.start()
    return agent

