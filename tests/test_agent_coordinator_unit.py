"""Unit tests for Streams-Based Agent Coordination.

These tests use mocks to verify behavior without Redis.
"""

from unittest.mock import AsyncMock, patch

import pytest


class TestAgentCoordinatorInit:
    """Tests for AgentCoordinator initialization."""

    def test_init_sets_stream_name(self) -> None:
        """Should set stream name."""
        from redis_openai_agents.coordinator import AgentCoordinator

        coordinator = AgentCoordinator(
            redis_url="redis://localhost:6379",
            stream_name="my_stream",
        )

        assert coordinator._stream_name == "my_stream"

    def test_init_generates_consumer_name(self) -> None:
        """Should auto-generate consumer name if not provided."""
        from redis_openai_agents.coordinator import AgentCoordinator

        coordinator = AgentCoordinator(
            redis_url="redis://localhost:6379",
            stream_name="events",
            consumer_group="workers",
        )

        assert coordinator._consumer_name is not None
        assert coordinator._consumer_name.startswith("consumer_")

    def test_init_uses_provided_consumer_name(self) -> None:
        """Should use provided consumer name."""
        from redis_openai_agents.coordinator import AgentCoordinator

        coordinator = AgentCoordinator(
            redis_url="redis://localhost:6379",
            stream_name="events",
            consumer_group="workers",
            consumer_name="my_worker",
        )

        assert coordinator._consumer_name == "my_worker"


class TestAgentCoordinatorPublish:
    """Tests for event publishing."""

    @pytest.mark.asyncio
    async def test_publish_handoff_ready_calls_xadd(self) -> None:
        """Should call XADD with correct event data."""
        from redis_openai_agents.coordinator import AgentCoordinator

        mock_client = AsyncMock()
        mock_client.xadd = AsyncMock(return_value="1234567890-0")

        with patch("redis_openai_agents.coordinator.aioredis") as mock_aioredis:
            mock_aioredis.from_url.return_value = mock_client

            coordinator = AgentCoordinator(
                redis_url="redis://localhost:6379",
                stream_name="events",
            )
            await coordinator.initialize()

            await coordinator.publish_handoff_ready(
                from_agent="agent_a",
                to_agent="agent_b",
                session_id="sess_1",
                context={"reason": "done"},
            )

            mock_client.xadd.assert_called_once()
            call_args = mock_client.xadd.call_args
            assert call_args[0][0] == "events"  # stream name

            data = call_args[0][1]
            assert data["type"] == "handoff_ready"
            assert data["from_agent"] == "agent_a"
            assert data["to_agent"] == "agent_b"
            assert data["session_id"] == "sess_1"
            assert "context" in data

    @pytest.mark.asyncio
    async def test_publish_tool_result_includes_metadata(self) -> None:
        """Should include execution time and timestamp."""
        from redis_openai_agents.coordinator import AgentCoordinator

        mock_client = AsyncMock()
        mock_client.xadd = AsyncMock(return_value="1234567890-0")

        with patch("redis_openai_agents.coordinator.aioredis") as mock_aioredis:
            mock_aioredis.from_url.return_value = mock_client

            coordinator = AgentCoordinator(
                redis_url="redis://localhost:6379",
                stream_name="events",
            )
            await coordinator.initialize()

            await coordinator.publish_tool_result(
                tool_name="search",
                session_id="sess_1",
                result={"data": "value"},
                execution_time_ms=250.5,
            )

            call_args = mock_client.xadd.call_args
            data = call_args[0][1]

            assert data["type"] == "tool_result"
            assert data["tool_name"] == "search"
            assert data["execution_time_ms"] == "250.5"
            assert "timestamp" in data

    @pytest.mark.asyncio
    async def test_publish_agent_completed_includes_metrics(self) -> None:
        """Should include duration and tokens."""
        from redis_openai_agents.coordinator import AgentCoordinator

        mock_client = AsyncMock()
        mock_client.xadd = AsyncMock(return_value="1234567890-0")

        with patch("redis_openai_agents.coordinator.aioredis") as mock_aioredis:
            mock_aioredis.from_url.return_value = mock_client

            coordinator = AgentCoordinator(
                redis_url="redis://localhost:6379",
                stream_name="events",
            )
            await coordinator.initialize()

            await coordinator.publish_agent_completed(
                agent_name="research",
                session_id="sess_1",
                output_summary="Found results",
                duration_ms=1500.0,
                tokens_used=100,
            )

            call_args = mock_client.xadd.call_args
            data = call_args[0][1]

            assert data["type"] == "agent_completed"
            assert data["duration_ms"] == "1500.0"
            assert data["tokens_used"] == "100"


class TestAgentCoordinatorSubscribe:
    """Tests for event subscription."""

    @pytest.mark.asyncio
    async def test_subscribe_creates_consumer_group(self) -> None:
        """Should create consumer group if not exists."""
        from redis_openai_agents.coordinator import AgentCoordinator

        mock_client = AsyncMock()
        mock_client.xgroup_create = AsyncMock()
        mock_client.xreadgroup = AsyncMock(return_value=[])

        with patch("redis_openai_agents.coordinator.aioredis") as mock_aioredis:
            mock_aioredis.from_url.return_value = mock_client

            coordinator = AgentCoordinator(
                redis_url="redis://localhost:6379",
                stream_name="events",
                consumer_group="workers",
            )
            await coordinator.initialize()

            # xgroup_create should be called during initialization
            mock_client.xgroup_create.assert_called()

    @pytest.mark.asyncio
    async def test_subscribe_calls_xreadgroup(self) -> None:
        """Should use XREADGROUP for consumer group reading."""
        from redis_openai_agents.coordinator import AgentCoordinator

        mock_client = AsyncMock()
        mock_client.xgroup_create = AsyncMock()
        mock_client.xreadgroup = AsyncMock(
            return_value=[
                (
                    "events",
                    [
                        ("1234-0", {"type": "handoff_ready", "from_agent": "a", "to_agent": "b"}),
                    ],
                )
            ]
        )
        mock_client.xack = AsyncMock()

        with patch("redis_openai_agents.coordinator.aioredis") as mock_aioredis:
            mock_aioredis.from_url.return_value = mock_client

            coordinator = AgentCoordinator(
                redis_url="redis://localhost:6379",
                stream_name="events",
                consumer_group="workers",
                consumer_name="worker_1",
            )
            await coordinator.initialize()

            events = []
            async for event in coordinator.subscribe(timeout_ms=100, max_events=1):
                events.append(event)

            mock_client.xreadgroup.assert_called()
            call_args = mock_client.xreadgroup.call_args
            assert call_args[0][0] == "workers"  # group name
            assert call_args[0][1] == "worker_1"  # consumer name

    @pytest.mark.asyncio
    async def test_subscribe_acknowledges_processed_events(self) -> None:
        """Should call XACK after processing."""
        from redis_openai_agents.coordinator import AgentCoordinator

        mock_client = AsyncMock()
        mock_client.xgroup_create = AsyncMock()
        mock_client.xreadgroup = AsyncMock(
            return_value=[
                (
                    "events",
                    [
                        ("1234-0", {"type": "tool_result", "tool_name": "test"}),
                    ],
                )
            ]
        )
        mock_client.xack = AsyncMock()

        with patch("redis_openai_agents.coordinator.aioredis") as mock_aioredis:
            mock_aioredis.from_url.return_value = mock_client

            coordinator = AgentCoordinator(
                redis_url="redis://localhost:6379",
                stream_name="events",
                consumer_group="workers",
            )
            await coordinator.initialize()

            async for _event in coordinator.subscribe(timeout_ms=100, max_events=1):
                pass  # Process event

            mock_client.xack.assert_called_once()
            call_args = mock_client.xack.call_args
            assert call_args[0][0] == "events"  # stream name
            assert call_args[0][1] == "workers"  # group name
            assert call_args[0][2] == "1234-0"  # message id

    @pytest.mark.asyncio
    async def test_subscribe_filters_by_event_type(self) -> None:
        """Should only yield events matching filter."""
        from redis_openai_agents.coordinator import AgentCoordinator

        mock_client = AsyncMock()
        mock_client.xgroup_create = AsyncMock()
        # Return messages once, then empty (simulating stream exhausted)
        mock_client.xreadgroup = AsyncMock(
            side_effect=[
                [
                    (
                        "events",
                        [
                            ("1234-0", {"type": "handoff_ready", "data": "1"}),
                            ("1234-1", {"type": "tool_result", "data": "2"}),
                            ("1234-2", {"type": "handoff_ready", "data": "3"}),
                        ],
                    )
                ],
                [],  # Empty on second call
            ]
        )
        mock_client.xack = AsyncMock()

        with patch("redis_openai_agents.coordinator.aioredis") as mock_aioredis:
            mock_aioredis.from_url.return_value = mock_client

            coordinator = AgentCoordinator(
                redis_url="redis://localhost:6379",
                stream_name="events",
                consumer_group="workers",
            )
            await coordinator.initialize()

            events = []
            async for event in coordinator.subscribe(
                event_types=["handoff_ready"],
                timeout_ms=100,
                max_events=10,
            ):
                events.append(event)

            # Should only get handoff_ready events
            assert len(events) == 2
            assert all(e["type"] == "handoff_ready" for e in events)


class TestAgentCoordinatorClaim:
    """Tests for claiming abandoned messages."""

    @pytest.mark.asyncio
    async def test_claim_calls_xpending_and_xclaim(self) -> None:
        """Should use XPENDING and XCLAIM for recovery."""
        from redis_openai_agents.coordinator import AgentCoordinator

        mock_client = AsyncMock()
        mock_client.xgroup_create = AsyncMock()
        mock_client.xpending_range = AsyncMock(
            return_value=[
                {
                    "message_id": "1234-0",
                    "consumer": "dead_worker",
                    "time_since_delivered": 600000,  # 10 minutes
                    "times_delivered": 1,
                }
            ]
        )
        mock_client.xclaim = AsyncMock(
            return_value=[("1234-0", {"type": "handoff_ready", "from_agent": "a"})]
        )

        with patch("redis_openai_agents.coordinator.aioredis") as mock_aioredis:
            mock_aioredis.from_url.return_value = mock_client

            coordinator = AgentCoordinator(
                redis_url="redis://localhost:6379",
                stream_name="events",
                consumer_group="workers",
                consumer_name="recovery_worker",
            )
            await coordinator.initialize()

            claimed = await coordinator.claim_abandoned_messages(min_idle_ms=300000)

            mock_client.xpending_range.assert_called_once()
            mock_client.xclaim.assert_called_once()

            assert len(claimed) == 1
            assert claimed[0]["type"] == "handoff_ready"

    @pytest.mark.asyncio
    async def test_claim_skips_recent_messages(self) -> None:
        """Should not claim messages that aren't old enough."""
        from redis_openai_agents.coordinator import AgentCoordinator

        mock_client = AsyncMock()
        mock_client.xgroup_create = AsyncMock()
        mock_client.xpending_range = AsyncMock(
            return_value=[
                {
                    "message_id": "1234-0",
                    "consumer": "slow_worker",
                    "time_since_delivered": 10000,  # Only 10 seconds
                    "times_delivered": 1,
                }
            ]
        )
        mock_client.xclaim = AsyncMock(return_value=[])

        with patch("redis_openai_agents.coordinator.aioredis") as mock_aioredis:
            mock_aioredis.from_url.return_value = mock_client

            coordinator = AgentCoordinator(
                redis_url="redis://localhost:6379",
                stream_name="events",
                consumer_group="workers",
            )
            await coordinator.initialize()

            # Min idle is 5 minutes, messages are only 10 seconds old
            claimed = await coordinator.claim_abandoned_messages(min_idle_ms=300000)

            # Should not claim anything
            mock_client.xclaim.assert_not_called()
            assert len(claimed) == 0


class TestAgentCoordinatorStreamInfo:
    """Tests for stream information retrieval."""

    @pytest.mark.asyncio
    async def test_get_stream_info_returns_stats(self) -> None:
        """Should return stream statistics."""
        from redis_openai_agents.coordinator import AgentCoordinator

        mock_client = AsyncMock()
        mock_client.xinfo_stream = AsyncMock(
            return_value={
                "length": 100,
                "first-entry": ("1234-0", {}),
                "last-entry": ("1234-99", {}),
            }
        )
        mock_client.xinfo_groups = AsyncMock(
            return_value=[{"name": "workers", "consumers": 3, "pending": 5}]
        )

        with patch("redis_openai_agents.coordinator.aioredis") as mock_aioredis:
            mock_aioredis.from_url.return_value = mock_client

            coordinator = AgentCoordinator(
                redis_url="redis://localhost:6379",
                stream_name="events",
            )
            await coordinator.initialize()

            info = await coordinator.get_stream_info()

            assert info["length"] == 100
            assert "groups" in info
            assert len(info["groups"]) == 1
            assert info["groups"][0]["name"] == "workers"

    @pytest.mark.asyncio
    async def test_trim_stream_calls_xtrim(self) -> None:
        """Should call XTRIM with MAXLEN."""
        from redis_openai_agents.coordinator import AgentCoordinator

        mock_client = AsyncMock()
        mock_client.xtrim = AsyncMock(return_value=90)  # Trimmed 90 entries

        with patch("redis_openai_agents.coordinator.aioredis") as mock_aioredis:
            mock_aioredis.from_url.return_value = mock_client

            coordinator = AgentCoordinator(
                redis_url="redis://localhost:6379",
                stream_name="events",
            )
            await coordinator.initialize()

            trimmed = await coordinator.trim_stream(max_length=10)

            mock_client.xtrim.assert_called_once()
            call_args = mock_client.xtrim.call_args
            assert call_args[0][0] == "events"
            assert call_args[1]["maxlen"] == 10

            assert trimmed == 90


class TestEventTypes:
    """Tests for event type constants."""

    def test_event_types_defined(self) -> None:
        """Should have standard event types."""
        from redis_openai_agents.coordinator import EventType

        assert EventType.HANDOFF_READY == "handoff_ready"
        assert EventType.TOOL_RESULT == "tool_result"
        assert EventType.STATE_CHANGED == "state_changed"
        assert EventType.AGENT_STARTED == "agent_started"
        assert EventType.AGENT_COMPLETED == "agent_completed"
        assert EventType.ERROR == "error"
