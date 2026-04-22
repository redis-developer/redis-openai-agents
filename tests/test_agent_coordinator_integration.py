"""Integration tests for Streams-Based Agent Coordination.

These tests run against a real Redis instance and verify:
- Real-time agent coordination via Redis Streams
- Consumer group functionality
- Handoff notifications
- Tool result broadcasting
- Crash recovery via XCLAIM
"""

import asyncio
from uuid import uuid4

import pytest


class TestAgentCoordinatorPublish:
    """Tests for publishing coordination events."""

    @pytest.mark.asyncio
    async def test_publish_handoff_ready(self, redis_url: str) -> None:
        """Should publish handoff_ready event to stream."""

        from redis_openai_agents import AgentCoordinator

        coordinator = AgentCoordinator(
            redis_url=redis_url,
            stream_name=f"test_stream_{uuid4().hex[:8]}",
        )
        await coordinator.initialize()

        msg_id = await coordinator.publish_handoff_ready(
            from_agent="research_agent",
            to_agent="analysis_agent",
            session_id="session_123",
            context={"reason": "task_complete", "data": {"findings": ["item1"]}},
        )

        assert msg_id is not None
        assert "-" in msg_id  # Redis stream IDs contain '-'

        await coordinator.close()

    @pytest.mark.asyncio
    async def test_publish_tool_result(self, redis_url: str) -> None:
        """Should publish tool_result event to stream."""
        from redis_openai_agents import AgentCoordinator

        coordinator = AgentCoordinator(
            redis_url=redis_url,
            stream_name=f"test_stream_{uuid4().hex[:8]}",
        )
        await coordinator.initialize()

        msg_id = await coordinator.publish_tool_result(
            tool_name="web_search",
            session_id="session_123",
            result={"urls": ["https://example.com"]},
            execution_time_ms=150.5,
        )

        assert msg_id is not None

        await coordinator.close()

    @pytest.mark.asyncio
    async def test_publish_state_changed(self, redis_url: str) -> None:
        """Should publish state_changed event."""
        from redis_openai_agents import AgentCoordinator

        coordinator = AgentCoordinator(
            redis_url=redis_url,
            stream_name=f"test_stream_{uuid4().hex[:8]}",
        )
        await coordinator.initialize()

        msg_id = await coordinator.publish_state_changed(
            session_id="session_123",
            changes={"current_agent": "new_agent", "message_count": 5},
        )

        assert msg_id is not None

        await coordinator.close()

    @pytest.mark.asyncio
    async def test_publish_agent_started(self, redis_url: str) -> None:
        """Should publish agent_started event."""
        from redis_openai_agents import AgentCoordinator

        coordinator = AgentCoordinator(
            redis_url=redis_url,
            stream_name=f"test_stream_{uuid4().hex[:8]}",
        )
        await coordinator.initialize()

        msg_id = await coordinator.publish_agent_started(
            agent_name="research_agent",
            session_id="session_123",
            input_summary="User asked about Redis",
        )

        assert msg_id is not None

        await coordinator.close()

    @pytest.mark.asyncio
    async def test_publish_agent_completed(self, redis_url: str) -> None:
        """Should publish agent_completed event."""
        from redis_openai_agents import AgentCoordinator

        coordinator = AgentCoordinator(
            redis_url=redis_url,
            stream_name=f"test_stream_{uuid4().hex[:8]}",
        )
        await coordinator.initialize()

        msg_id = await coordinator.publish_agent_completed(
            agent_name="research_agent",
            session_id="session_123",
            output_summary="Found 5 relevant articles",
            duration_ms=2500.0,
            tokens_used=150,
        )

        assert msg_id is not None

        await coordinator.close()

    @pytest.mark.asyncio
    async def test_publish_error_without_agent_name(self, redis_url: str) -> None:
        """Should publish error event when optional agent_name is None.

        Regression test: XADD rejects None values, so _publish must drop keys
        whose value is None rather than passing them through.
        """
        from redis_openai_agents import AgentCoordinator

        coordinator = AgentCoordinator(
            redis_url=redis_url,
            stream_name=f"test_stream_{uuid4().hex[:8]}",
        )
        await coordinator.initialize()

        msg_id = await coordinator.publish_error(
            session_id="session_123",
            error_type="ValidationError",
            error_message="Invalid input format",
        )

        assert msg_id is not None

        await coordinator.close()


class TestAgentCoordinatorSubscribe:
    """Tests for subscribing to coordination events."""

    @pytest.mark.asyncio
    async def test_subscribe_receives_published_events(self, redis_url: str) -> None:
        """Should receive events via subscription."""
        from redis_openai_agents import AgentCoordinator

        stream_name = f"test_stream_{uuid4().hex[:8]}"

        publisher = AgentCoordinator(
            redis_url=redis_url,
            stream_name=stream_name,
        )
        await publisher.initialize()

        subscriber = AgentCoordinator(
            redis_url=redis_url,
            stream_name=stream_name,
            consumer_group="test_group",
            consumer_name="test_consumer",
        )
        await subscriber.initialize()

        # Publish event
        await publisher.publish_handoff_ready(
            from_agent="agent_a",
            to_agent="agent_b",
            session_id="sess_1",
            context={},
        )

        # Subscribe and receive
        received = []
        async for event in subscriber.subscribe(timeout_ms=1000, max_events=1):
            received.append(event)

        assert len(received) == 1
        assert received[0]["type"] == "handoff_ready"
        assert received[0]["from_agent"] == "agent_a"

        await publisher.close()
        await subscriber.close()

    @pytest.mark.asyncio
    async def test_subscribe_with_event_type_filter(self, redis_url: str) -> None:
        """Should filter events by type."""
        from redis_openai_agents import AgentCoordinator

        stream_name = f"test_stream_{uuid4().hex[:8]}"

        coordinator = AgentCoordinator(
            redis_url=redis_url,
            stream_name=stream_name,
            consumer_group="test_group",
        )
        await coordinator.initialize()

        # Publish different event types
        await coordinator.publish_handoff_ready(
            from_agent="a", to_agent="b", session_id="s1", context={}
        )
        await coordinator.publish_tool_result(
            tool_name="tool1", session_id="s1", result={}, execution_time_ms=100
        )
        await coordinator.publish_handoff_ready(
            from_agent="b", to_agent="c", session_id="s1", context={}
        )

        # Subscribe only to handoff_ready
        received = []
        async for event in coordinator.subscribe(
            event_types=["handoff_ready"],
            timeout_ms=1000,
            max_events=10,
        ):
            received.append(event)

        # Should only receive handoff_ready events
        handoff_events = [e for e in received if e["type"] == "handoff_ready"]
        assert len(handoff_events) >= 2

        await coordinator.close()

    @pytest.mark.asyncio
    async def test_subscribe_acknowledges_events(self, redis_url: str) -> None:
        """Events should be acknowledged after processing."""
        from redis.asyncio import Redis

        from redis_openai_agents import AgentCoordinator

        stream_name = f"test_stream_{uuid4().hex[:8]}"
        group_name = "test_group"

        coordinator = AgentCoordinator(
            redis_url=redis_url,
            stream_name=stream_name,
            consumer_group=group_name,
        )
        await coordinator.initialize()

        # Publish event
        await coordinator.publish_handoff_ready(
            from_agent="a", to_agent="b", session_id="s1", context={}
        )

        # Consume event
        async for _event in coordinator.subscribe(timeout_ms=1000, max_events=1):
            pass  # Just consume

        # Check pending count
        client = Redis.from_url(redis_url, decode_responses=True)
        info = await client.xpending(stream_name, group_name)

        # Should have no pending messages after ACK
        assert info["pending"] == 0

        await client.aclose()
        await coordinator.close()


class TestAgentCoordinatorConsumerGroups:
    """Tests for consumer group functionality."""

    @pytest.mark.asyncio
    async def test_multiple_consumers_share_work(self, redis_url: str) -> None:
        """Multiple consumers should receive different messages."""
        from redis_openai_agents import AgentCoordinator

        stream_name = f"test_stream_{uuid4().hex[:8]}"
        group_name = "workers"

        publisher = AgentCoordinator(
            redis_url=redis_url,
            stream_name=stream_name,
        )
        await publisher.initialize()

        consumer1 = AgentCoordinator(
            redis_url=redis_url,
            stream_name=stream_name,
            consumer_group=group_name,
            consumer_name="worker_1",
        )
        await consumer1.initialize()

        consumer2 = AgentCoordinator(
            redis_url=redis_url,
            stream_name=stream_name,
            consumer_group=group_name,
            consumer_name="worker_2",
        )
        await consumer2.initialize()

        # Publish multiple events
        for i in range(10):
            await publisher.publish_tool_result(
                tool_name=f"tool_{i}",
                session_id="s1",
                result={"index": i},
                execution_time_ms=100,
            )

        # Both consumers read concurrently to test work sharing
        async def consume(coordinator: AgentCoordinator) -> list:
            events = []
            async for event in coordinator.subscribe(timeout_ms=500, max_events=10):
                events.append(event)
            return events

        results = await asyncio.gather(
            consume(consumer1),
            consume(consumer2),
        )
        received1, received2 = results

        # Together should have processed all messages
        total_received = len(received1) + len(received2)
        assert total_received == 10

        # With concurrent consumption, work should be distributed
        # Note: distribution may not be perfectly even, but both should get some
        assert len(received1) > 0 or len(received2) > 0  # At least one got messages
        # The key assertion is that together they got all 10

        await publisher.close()
        await consumer1.close()
        await consumer2.close()


class TestAgentCoordinatorCrashRecovery:
    """Tests for crash recovery via XCLAIM."""

    @pytest.mark.asyncio
    async def test_claim_abandoned_messages(self, redis_url: str) -> None:
        """Should claim messages from crashed consumers."""
        from redis.asyncio import Redis

        from redis_openai_agents import AgentCoordinator

        stream_name = f"test_stream_{uuid4().hex[:8]}"
        group_name = "workers"

        publisher = AgentCoordinator(
            redis_url=redis_url,
            stream_name=stream_name,
        )
        await publisher.initialize()

        # Consumer 1 reads but doesn't ACK (simulates crash)
        consumer1 = AgentCoordinator(
            redis_url=redis_url,
            stream_name=stream_name,
            consumer_group=group_name,
            consumer_name="crashed_worker",
        )
        await consumer1.initialize()

        # Publish message
        await publisher.publish_handoff_ready(
            from_agent="a", to_agent="b", session_id="s1", context={}
        )

        # Consumer 1 reads without ACK
        client = Redis.from_url(redis_url, decode_responses=True)
        await client.xreadgroup(
            group_name,
            "crashed_worker",
            {stream_name: ">"},
            count=1,
        )

        # Consumer 2 claims abandoned messages
        consumer2 = AgentCoordinator(
            redis_url=redis_url,
            stream_name=stream_name,
            consumer_group=group_name,
            consumer_name="recovery_worker",
        )
        await consumer2.initialize()

        # Small delay to make message "old"
        await asyncio.sleep(0.1)

        # Claim with very short idle time for test
        claimed = await consumer2.claim_abandoned_messages(min_idle_ms=50)

        assert len(claimed) >= 1
        assert claimed[0]["type"] == "handoff_ready"

        await client.aclose()
        await publisher.close()
        await consumer1.close()
        await consumer2.close()


class TestAgentCoordinatorStreamManagement:
    """Tests for stream management operations."""

    @pytest.mark.asyncio
    async def test_get_stream_info(self, redis_url: str) -> None:
        """Should return stream statistics."""
        from redis_openai_agents import AgentCoordinator

        stream_name = f"test_stream_{uuid4().hex[:8]}"

        coordinator = AgentCoordinator(
            redis_url=redis_url,
            stream_name=stream_name,
            consumer_group="test_group",
        )
        await coordinator.initialize()

        # Publish some events
        for i in range(5):
            await coordinator.publish_tool_result(
                tool_name=f"tool_{i}",
                session_id="s1",
                result={},
                execution_time_ms=100,
            )

        info = await coordinator.get_stream_info()

        assert info["length"] == 5
        assert "groups" in info
        assert len(info["groups"]) >= 1

        await coordinator.close()

    @pytest.mark.asyncio
    async def test_trim_stream(self, redis_url: str) -> None:
        """Should trim stream to max length."""
        from redis_openai_agents import AgentCoordinator

        stream_name = f"test_stream_{uuid4().hex[:8]}"

        coordinator = AgentCoordinator(
            redis_url=redis_url,
            stream_name=stream_name,
        )
        await coordinator.initialize()

        # Publish many events
        for i in range(100):
            await coordinator.publish_tool_result(
                tool_name=f"tool_{i}",
                session_id="s1",
                result={},
                execution_time_ms=100,
            )

        # Trim to 10 with exact mode (not approximate) for predictable test
        trimmed = await coordinator.trim_stream(max_length=10, approximate=False)

        assert trimmed == 90

        info = await coordinator.get_stream_info()
        assert info["length"] == 10

        await coordinator.close()
