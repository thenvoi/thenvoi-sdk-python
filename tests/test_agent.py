"""Tests for Agent compositor."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from thenvoi.agent import Agent
from thenvoi.core.simple_adapter import SimpleAdapter
from thenvoi.core.types import AgentInput
from thenvoi.runtime.types import AgentConfig, SessionConfig
from thenvoi.preprocessing.default import DefaultPreprocessor


@pytest.fixture
def mock_adapter():
    """Create mock adapter."""
    adapter = AsyncMock()
    adapter.on_started = AsyncMock()
    adapter.on_cleanup = AsyncMock()
    adapter.on_event = AsyncMock()
    return adapter


@pytest.fixture
def mock_runtime():
    """Create mock PlatformRuntime."""
    runtime = MagicMock()
    runtime.agent_name = "TestBot"
    runtime.agent_description = "A test bot"
    runtime.agent_id = "agent-123"
    runtime.start = AsyncMock()
    runtime.stop = AsyncMock()
    runtime.run_forever = AsyncMock()
    return runtime


@pytest.fixture
def mock_preprocessor():
    """Create mock Preprocessor."""
    preprocessor = AsyncMock()
    preprocessor.process = AsyncMock(return_value=None)
    return preprocessor


class TestInitialization:
    """Tests for Agent initialization."""

    def test_full_composition(self, mock_runtime, mock_adapter, mock_preprocessor):
        """Should accept runtime, adapter, and preprocessor."""
        agent = Agent(
            runtime=mock_runtime,
            adapter=mock_adapter,
            preprocessor=mock_preprocessor,
        )

        assert agent._runtime is mock_runtime
        assert agent._adapter is mock_adapter
        assert agent._preprocessor is mock_preprocessor

    def test_default_preprocessor(self, mock_runtime, mock_adapter):
        """Should use DefaultPreprocessor if none provided."""
        agent = Agent(
            runtime=mock_runtime,
            adapter=mock_adapter,
        )

        assert isinstance(agent._preprocessor, DefaultPreprocessor)


class TestCreateFactory:
    """Tests for Agent.create() factory method."""

    def test_creates_with_default_urls(self, mock_adapter):
        """Should create agent with default URLs."""
        with patch("thenvoi.agent.PlatformRuntime") as mock_runtime_class:
            mock_runtime = MagicMock()
            mock_runtime_class.return_value = mock_runtime

            Agent.create(
                adapter=mock_adapter,
                agent_id="agent-123",
                api_key="test-key",
            )

            mock_runtime_class.assert_called_once_with(
                agent_id="agent-123",
                api_key="test-key",
                ws_url="wss://api.thenvoi.com/ws",
                rest_url="https://api.thenvoi.com",
                config=None,
                session_config=None,
            )

    def test_creates_with_custom_urls(self, mock_adapter):
        """Should accept custom URLs."""
        with patch("thenvoi.agent.PlatformRuntime") as mock_runtime_class:
            mock_runtime = MagicMock()
            mock_runtime_class.return_value = mock_runtime

            Agent.create(
                adapter=mock_adapter,
                agent_id="agent-123",
                api_key="test-key",
                ws_url="wss://custom.example.com/ws",
                rest_url="https://custom.example.com",
            )

            call_kwargs = mock_runtime_class.call_args.kwargs
            assert call_kwargs["ws_url"] == "wss://custom.example.com/ws"
            assert call_kwargs["rest_url"] == "https://custom.example.com"

    def test_creates_with_configs(self, mock_adapter):
        """Should accept custom configs."""
        config = AgentConfig()
        session_config = SessionConfig()

        with patch("thenvoi.agent.PlatformRuntime") as mock_runtime_class:
            mock_runtime = MagicMock()
            mock_runtime_class.return_value = mock_runtime

            Agent.create(
                adapter=mock_adapter,
                agent_id="agent-123",
                api_key="test-key",
                config=config,
                session_config=session_config,
            )

            call_kwargs = mock_runtime_class.call_args.kwargs
            assert call_kwargs["config"] is config
            assert call_kwargs["session_config"] is session_config

    def test_creates_with_custom_preprocessor(self, mock_adapter, mock_preprocessor):
        """Should accept custom preprocessor."""
        with patch("thenvoi.agent.PlatformRuntime") as mock_runtime_class:
            mock_runtime = MagicMock()
            mock_runtime_class.return_value = mock_runtime

            agent = Agent.create(
                adapter=mock_adapter,
                agent_id="agent-123",
                api_key="test-key",
                preprocessor=mock_preprocessor,
            )

            assert agent._preprocessor is mock_preprocessor


class TestProperties:
    """Tests for Agent properties."""

    def test_runtime_property(self, mock_runtime, mock_adapter):
        """Should expose runtime property."""
        agent = Agent(runtime=mock_runtime, adapter=mock_adapter)
        assert agent.runtime is mock_runtime

    def test_agent_name_property(self, mock_runtime, mock_adapter):
        """Should delegate agent_name to runtime."""
        agent = Agent(runtime=mock_runtime, adapter=mock_adapter)
        assert agent.agent_name == "TestBot"

    def test_agent_description_property(self, mock_runtime, mock_adapter):
        """Should delegate agent_description to runtime."""
        agent = Agent(runtime=mock_runtime, adapter=mock_adapter)
        assert agent.agent_description == "A test bot"


class TestStart:
    """Tests for Agent.start() method."""

    @pytest.mark.asyncio
    async def test_starts_runtime(self, mock_runtime, mock_adapter):
        """Should start the platform runtime."""
        agent = Agent(runtime=mock_runtime, adapter=mock_adapter)

        await agent.start()

        mock_runtime.start.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_calls_adapter_on_started(self, mock_runtime, mock_adapter):
        """Should call adapter.on_started with agent metadata."""
        agent = Agent(runtime=mock_runtime, adapter=mock_adapter)

        await agent.start()

        mock_adapter.on_started.assert_awaited_once_with("TestBot", "A test bot")

    @pytest.mark.asyncio
    async def test_passes_on_execute_to_runtime(self, mock_runtime, mock_adapter):
        """Should pass _on_execute to runtime.start()."""
        agent = Agent(runtime=mock_runtime, adapter=mock_adapter)

        await agent.start()

        call_kwargs = mock_runtime.start.call_args.kwargs
        assert call_kwargs["on_execute"] == agent._on_execute

    @pytest.mark.asyncio
    async def test_passes_on_cleanup_to_runtime(self, mock_runtime, mock_adapter):
        """Should pass adapter.on_cleanup to runtime.start()."""
        agent = Agent(runtime=mock_runtime, adapter=mock_adapter)

        await agent.start()

        call_kwargs = mock_runtime.start.call_args.kwargs
        assert call_kwargs["on_cleanup"] == mock_adapter.on_cleanup


class TestStop:
    """Tests for Agent.stop() method."""

    @pytest.mark.asyncio
    async def test_stops_runtime(self, mock_runtime, mock_adapter):
        """Should stop the platform runtime."""
        agent = Agent(runtime=mock_runtime, adapter=mock_adapter)

        await agent.stop()

        mock_runtime.stop.assert_awaited_once()


class TestRun:
    """Tests for Agent.run() method."""

    @pytest.mark.asyncio
    async def test_starts_then_runs_forever(self, mock_runtime, mock_adapter):
        """Should start and run forever."""
        agent = Agent(runtime=mock_runtime, adapter=mock_adapter)

        await agent.run()

        mock_runtime.start.assert_awaited_once()
        mock_runtime.run_forever.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stops_on_completion(self, mock_runtime, mock_adapter):
        """Should stop runtime after run_forever completes."""
        agent = Agent(runtime=mock_runtime, adapter=mock_adapter)

        await agent.run()

        mock_runtime.stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stops_on_exception(self, mock_runtime, mock_adapter):
        """Should stop runtime even if run_forever raises."""
        mock_runtime.run_forever = AsyncMock(side_effect=Exception("Connection lost"))
        agent = Agent(runtime=mock_runtime, adapter=mock_adapter)

        with pytest.raises(Exception, match="Connection lost"):
            await agent.run()

        # Should still have called stop
        mock_runtime.stop.assert_awaited_once()


class TestOnExecute:
    """Tests for Agent._on_execute() method."""

    @pytest.mark.asyncio
    async def test_processes_event_with_preprocessor(
        self, mock_runtime, mock_adapter, mock_preprocessor
    ):
        """Should pass event to preprocessor."""
        agent = Agent(
            runtime=mock_runtime,
            adapter=mock_adapter,
            preprocessor=mock_preprocessor,
        )

        mock_ctx = MagicMock()
        mock_event = MagicMock()

        await agent._on_execute(mock_ctx, mock_event)

        mock_preprocessor.process.assert_awaited_once_with(
            ctx=mock_ctx,
            event=mock_event,
            agent_id="agent-123",
        )

    @pytest.mark.asyncio
    async def test_skips_when_preprocessor_returns_none(
        self, mock_runtime, mock_adapter, mock_preprocessor
    ):
        """Should not call adapter when preprocessor returns None."""
        mock_preprocessor.process.return_value = None
        agent = Agent(
            runtime=mock_runtime,
            adapter=mock_adapter,
            preprocessor=mock_preprocessor,
        )

        mock_ctx = MagicMock()
        mock_event = MagicMock()

        await agent._on_execute(mock_ctx, mock_event)

        mock_adapter.on_event.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_calls_adapter_on_event(
        self, mock_runtime, mock_adapter, mock_preprocessor
    ):
        """Should call adapter.on_event with AgentInput."""
        mock_input = MagicMock(spec=AgentInput)
        mock_preprocessor.process.return_value = mock_input

        agent = Agent(
            runtime=mock_runtime,
            adapter=mock_adapter,
            preprocessor=mock_preprocessor,
        )

        mock_ctx = MagicMock()
        mock_event = MagicMock()

        await agent._on_execute(mock_ctx, mock_event)

        mock_adapter.on_event.assert_awaited_once_with(mock_input)


class TestSimpleAdapterIntegration:
    """Tests for SimpleAdapter integration."""

    @pytest.mark.asyncio
    async def test_works_with_simple_adapter(self, mock_runtime):
        """Should work with SimpleAdapter subclass."""
        # Create a mock SimpleAdapter
        adapter = MagicMock(spec=SimpleAdapter)
        adapter.on_started = AsyncMock()
        adapter.on_cleanup = AsyncMock()
        adapter.on_event = AsyncMock()

        agent = Agent(runtime=mock_runtime, adapter=adapter)

        await agent.start()

        adapter.on_started.assert_awaited_once()


class TestDefaultPreprocessorIntegration:
    """Tests for DefaultPreprocessor behavior."""

    @pytest.mark.asyncio
    async def test_default_preprocessor_filters_non_message_events(
        self, mock_runtime, mock_adapter
    ):
        """DefaultPreprocessor should filter non-message events."""
        agent = Agent(runtime=mock_runtime, adapter=mock_adapter)

        # Create a non-MessageEvent (e.g., RoomAddedEvent)
        from thenvoi.platform.event import RoomAddedEvent
        from thenvoi.client.streaming import RoomAddedPayload, RoomOwner

        mock_ctx = MagicMock()
        mock_event = RoomAddedEvent(
            room_id="room-123",
            payload=RoomAddedPayload(
                id="room-123",
                title="Test Room",
                owner=RoomOwner(id="user-1", name="Owner", type="User"),
                status="active",
                type="direct",
                created_at="2024-01-01T00:00:00Z",
                participant_role="member",
            ),
        )

        await agent._on_execute(mock_ctx, mock_event)

        # Should not call adapter (event was filtered)
        mock_adapter.on_event.assert_not_awaited()
