"""Integration tests for RobustStreamProcessor.

These tests run against a real Redis instance and verify:
- Automatic pending message recovery via XCLAIM
- Dead-letter queue (DLQ) for failed messages after max retries
- Message replay from DLQ back to main stream
- Health statistics and monitoring
"""

import asyncio
from uuid import uuid4

import pytest
from redis.exceptions import ResponseError


class TestRobustStreamProcessorBasicProcessing:
    """Tests for basic message processing."""

    @pytest.mark.asyncio
    async def test_process_messages_successfully(self, redis_url: str) -> None:
        """Should process messages and acknowledge them."""
        from redis.asyncio import Redis

        from redis_openai_agents.robust_processor import RobustStreamProcessor

        stream_name = f"test_stream_{uuid4().hex[:8]}"
        group_name = "test_group"

        processor = RobustStreamProcessor(
            redis_url=redis_url,
            stream_name=stream_name,
            consumer_group=group_name,
        )
        await processor.initialize()

        # Publish test messages
        client = Redis.from_url(redis_url, decode_responses=True)
        try:
            await client.xgroup_create(stream_name, group_name, id="0", mkstream=True)
        except ResponseError:
            pass  # BUSYGROUP — group already exists

        for i in range(5):
            await client.xadd(stream_name, {"data": f"message_{i}"})

        # Process messages
        processed = []

        async def handler(msg: dict) -> bool:
            processed.append(msg)
            return True  # Success

        # Process with timeout
        await asyncio.wait_for(
            processor.process_batch(handler, batch_size=5, max_batches=1),
            timeout=5.0,
        )

        assert len(processed) == 5

        # Verify no pending messages
        pending_info = await client.xpending(stream_name, group_name)
        assert pending_info["pending"] == 0

        await client.aclose()
        await processor.close()

    @pytest.mark.asyncio
    async def test_failed_messages_stay_pending(self, redis_url: str) -> None:
        """Messages that fail processing should stay pending for retry."""
        from redis.asyncio import Redis

        from redis_openai_agents.robust_processor import RobustStreamProcessor

        stream_name = f"test_stream_{uuid4().hex[:8]}"
        group_name = "test_group"

        processor = RobustStreamProcessor(
            redis_url=redis_url,
            stream_name=stream_name,
            consumer_group=group_name,
        )
        await processor.initialize()

        client = Redis.from_url(redis_url, decode_responses=True)
        try:
            await client.xgroup_create(stream_name, group_name, id="0", mkstream=True)
        except ResponseError:
            pass  # BUSYGROUP — group already exists

        await client.xadd(stream_name, {"data": "will_fail"})

        async def failing_handler(msg: dict) -> bool:
            return False  # Failure

        await asyncio.wait_for(
            processor.process_batch(failing_handler, batch_size=1, max_batches=1),
            timeout=5.0,
        )

        # Message should still be pending
        pending_info = await client.xpending(stream_name, group_name)
        assert pending_info["pending"] == 1

        await client.aclose()
        await processor.close()


class TestRobustStreamProcessorPendingRecovery:
    """Tests for pending message recovery."""

    @pytest.mark.asyncio
    async def test_claim_pending_from_crashed_consumer(self, redis_url: str) -> None:
        """Should claim pending messages from crashed consumers."""
        from redis.asyncio import Redis

        from redis_openai_agents.robust_processor import RobustStreamProcessor

        stream_name = f"test_stream_{uuid4().hex[:8]}"
        group_name = "test_group"

        client = Redis.from_url(redis_url, decode_responses=True)

        # Create group and add message
        try:
            await client.xgroup_create(stream_name, group_name, id="0", mkstream=True)
        except ResponseError:
            pass  # BUSYGROUP — group already exists

        await client.xadd(stream_name, {"data": "stuck_message"})

        # Crashed consumer reads but doesn't ACK
        await client.xreadgroup(
            group_name,
            "crashed_consumer",
            {stream_name: ">"},
            count=1,
        )

        # Small delay to make message "old"
        await asyncio.sleep(0.1)

        # New processor should claim the message
        processor = RobustStreamProcessor(
            redis_url=redis_url,
            stream_name=stream_name,
            consumer_group=group_name,
            consumer_name="recovery_consumer",
            claim_timeout_ms=50,  # Very short for testing
        )
        await processor.initialize()

        claimed = await processor.claim_pending_messages()

        assert claimed >= 1

        await client.aclose()
        await processor.close()


class TestRobustStreamProcessorDLQ:
    """Tests for dead-letter queue functionality."""

    @pytest.mark.asyncio
    async def test_message_moved_to_dlq_after_max_retries(self, redis_url: str) -> None:
        """Messages exceeding max retries should be moved to DLQ."""
        from redis.asyncio import Redis

        from redis_openai_agents.robust_processor import RobustStreamProcessor

        stream_name = f"test_stream_{uuid4().hex[:8]}"
        group_name = "test_group"
        dlq_stream = f"{stream_name}:dlq"

        processor = RobustStreamProcessor(
            redis_url=redis_url,
            stream_name=stream_name,
            consumer_group=group_name,
            dlq_stream=dlq_stream,
            max_retries=2,
            claim_timeout_ms=50,
        )
        await processor.initialize()

        client = Redis.from_url(redis_url, decode_responses=True)
        try:
            await client.xgroup_create(stream_name, group_name, id="0", mkstream=True)
        except ResponseError:
            pass  # BUSYGROUP — group already exists

        # Add message
        await client.xadd(stream_name, {"data": "will_fail_repeatedly"})

        # Simulate multiple delivery attempts by reading without ACK multiple times
        for i in range(3):  # Exceed max_retries of 2
            await client.xreadgroup(
                group_name,
                f"consumer_{i}",
                {stream_name: ">"},
                count=1,
            )
            await asyncio.sleep(0.1)

            # Claim and process, but fail
            await processor.claim_pending_messages()

        # Message should now be in DLQ
        await processor.get_dlq_messages()

        # The message should eventually end up in DLQ after exceeding retries.
        # Poll briefly to allow async DLQ move to complete.
        dlq_len = 0
        for _ in range(20):
            dlq_len = await client.xlen(dlq_stream)
            if dlq_len >= 1:
                break
            await asyncio.sleep(0.1)
        assert dlq_len >= 1, f"Expected at least 1 DLQ message, got {dlq_len}"

        await client.aclose()
        await processor.close()

    @pytest.mark.asyncio
    async def test_get_dlq_messages(self, redis_url: str) -> None:
        """Should retrieve messages from DLQ."""
        from redis.asyncio import Redis

        from redis_openai_agents.robust_processor import RobustStreamProcessor

        stream_name = f"test_stream_{uuid4().hex[:8]}"
        dlq_stream = f"{stream_name}:dlq"

        processor = RobustStreamProcessor(
            redis_url=redis_url,
            stream_name=stream_name,
            consumer_group="test_group",
            dlq_stream=dlq_stream,
        )
        await processor.initialize()

        # Add messages directly to DLQ for testing
        client = Redis.from_url(redis_url, decode_responses=True)
        await client.xadd(
            dlq_stream,
            {
                "data": "failed_message",
                "failure_reason": "test",
                "attempts": "3",
            },
        )

        dlq_messages = await processor.get_dlq_messages()

        assert len(dlq_messages) == 1
        assert dlq_messages[0]["data"] == "failed_message"

        await client.aclose()
        await processor.close()

    @pytest.mark.asyncio
    async def test_replay_dlq_message(self, redis_url: str) -> None:
        """Should replay DLQ message back to main stream."""
        from redis.asyncio import Redis

        from redis_openai_agents.robust_processor import RobustStreamProcessor

        stream_name = f"test_stream_{uuid4().hex[:8]}"
        dlq_stream = f"{stream_name}:dlq"

        processor = RobustStreamProcessor(
            redis_url=redis_url,
            stream_name=stream_name,
            consumer_group="test_group",
            dlq_stream=dlq_stream,
        )
        await processor.initialize()

        client = Redis.from_url(redis_url, decode_responses=True)

        # Add message to DLQ
        dlq_msg_id = await client.xadd(
            dlq_stream,
            {
                "type": "test_event",
                "data": "replay_me",
                "original_stream": stream_name,
                "original_id": "1234-0",
                "failure_reason": "temporary_error",
                "attempts": "2",
                "dlq_timestamp": "1234567890",
            },
        )

        # Replay the message
        new_msg_id = await processor.replay_dlq_message(dlq_msg_id)

        assert new_msg_id is not None
        assert "-" in new_msg_id

        # Verify message is in main stream
        messages = await client.xrange(stream_name, new_msg_id, new_msg_id)
        assert len(messages) == 1
        _, data = messages[0]
        assert data["type"] == "test_event"
        assert data["data"] == "replay_me"
        assert "replayed_from_dlq" in data

        # Verify message removed from DLQ
        dlq_len = await client.xlen(dlq_stream)
        assert dlq_len == 0

        await client.aclose()
        await processor.close()


class TestRobustStreamProcessorHealthStats:
    """Tests for health statistics."""

    @pytest.mark.asyncio
    async def test_get_health_stats(self, redis_url: str) -> None:
        """Should return health statistics."""
        from redis.asyncio import Redis

        from redis_openai_agents.robust_processor import RobustStreamProcessor

        stream_name = f"test_stream_{uuid4().hex[:8]}"
        group_name = "test_group"

        processor = RobustStreamProcessor(
            redis_url=redis_url,
            stream_name=stream_name,
            consumer_group=group_name,
        )
        await processor.initialize()

        client = Redis.from_url(redis_url, decode_responses=True)
        try:
            await client.xgroup_create(stream_name, group_name, id="0", mkstream=True)
        except ResponseError:
            pass  # BUSYGROUP — group already exists

        # Add some messages
        for i in range(5):
            await client.xadd(stream_name, {"data": f"msg_{i}"})

        stats = await processor.get_health_stats()

        assert "stream_length" in stats
        assert stats["stream_length"] == 5
        assert "pending_messages" in stats
        assert "dlq_length" in stats
        assert "consumers" in stats

        await client.aclose()
        await processor.close()

    @pytest.mark.asyncio
    async def test_health_stats_with_pending_messages(self, redis_url: str) -> None:
        """Should show pending message count in stats."""
        from redis.asyncio import Redis

        from redis_openai_agents.robust_processor import RobustStreamProcessor

        stream_name = f"test_stream_{uuid4().hex[:8]}"
        group_name = "test_group"

        processor = RobustStreamProcessor(
            redis_url=redis_url,
            stream_name=stream_name,
            consumer_group=group_name,
        )
        await processor.initialize()

        client = Redis.from_url(redis_url, decode_responses=True)
        try:
            await client.xgroup_create(stream_name, group_name, id="0", mkstream=True)
        except ResponseError:
            pass  # BUSYGROUP — group already exists

        # Add and read message without ACK
        await client.xadd(stream_name, {"data": "pending_msg"})
        await client.xreadgroup(
            group_name,
            "test_consumer",
            {stream_name: ">"},
            count=1,
        )

        stats = await processor.get_health_stats()

        assert stats["pending_messages"] == 1
        assert stats["consumers"] >= 1

        await client.aclose()
        await processor.close()
