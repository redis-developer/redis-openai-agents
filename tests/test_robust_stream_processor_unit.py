"""Unit tests for RobustStreamProcessor.

These tests use mocks to verify behavior without Redis.
"""

from unittest.mock import AsyncMock, patch

import pytest


class TestRobustStreamProcessorInit:
    """Tests for RobustStreamProcessor initialization."""

    def test_init_sets_stream_name(self) -> None:
        """Should set stream name."""
        from redis_openai_agents.robust_processor import RobustStreamProcessor

        processor = RobustStreamProcessor(
            redis_url="redis://localhost:6379",
            stream_name="my_stream",
            consumer_group="my_group",
        )

        assert processor._stream_name == "my_stream"

    def test_init_sets_consumer_group(self) -> None:
        """Should set consumer group."""
        from redis_openai_agents.robust_processor import RobustStreamProcessor

        processor = RobustStreamProcessor(
            redis_url="redis://localhost:6379",
            stream_name="events",
            consumer_group="workers",
        )

        assert processor._consumer_group == "workers"

    def test_init_generates_consumer_name(self) -> None:
        """Should auto-generate consumer name if not provided."""
        from redis_openai_agents.robust_processor import RobustStreamProcessor

        processor = RobustStreamProcessor(
            redis_url="redis://localhost:6379",
            stream_name="events",
            consumer_group="workers",
        )

        assert processor._consumer_name is not None
        assert processor._consumer_name.startswith("consumer_")

    def test_init_uses_provided_consumer_name(self) -> None:
        """Should use provided consumer name."""
        from redis_openai_agents.robust_processor import RobustStreamProcessor

        processor = RobustStreamProcessor(
            redis_url="redis://localhost:6379",
            stream_name="events",
            consumer_group="workers",
            consumer_name="my_worker",
        )

        assert processor._consumer_name == "my_worker"

    def test_init_default_dlq_stream(self) -> None:
        """Should create default DLQ stream name."""
        from redis_openai_agents.robust_processor import RobustStreamProcessor

        processor = RobustStreamProcessor(
            redis_url="redis://localhost:6379",
            stream_name="events",
            consumer_group="workers",
        )

        assert processor._dlq_stream == "events:dlq"

    def test_init_custom_dlq_stream(self) -> None:
        """Should use custom DLQ stream name."""
        from redis_openai_agents.robust_processor import RobustStreamProcessor

        processor = RobustStreamProcessor(
            redis_url="redis://localhost:6379",
            stream_name="events",
            consumer_group="workers",
            dlq_stream="custom_dlq",
        )

        assert processor._dlq_stream == "custom_dlq"

    def test_init_default_max_retries(self) -> None:
        """Should default max_retries to 3."""
        from redis_openai_agents.robust_processor import RobustStreamProcessor

        processor = RobustStreamProcessor(
            redis_url="redis://localhost:6379",
            stream_name="events",
            consumer_group="workers",
        )

        assert processor._max_retries == 3

    def test_init_custom_max_retries(self) -> None:
        """Should accept custom max_retries."""
        from redis_openai_agents.robust_processor import RobustStreamProcessor

        processor = RobustStreamProcessor(
            redis_url="redis://localhost:6379",
            stream_name="events",
            consumer_group="workers",
            max_retries=5,
        )

        assert processor._max_retries == 5


class TestRobustStreamProcessorProcessing:
    """Tests for message processing."""

    @pytest.mark.asyncio
    async def test_process_batch_calls_xreadgroup(self) -> None:
        """Should call XREADGROUP to get messages."""
        from redis_openai_agents.robust_processor import RobustStreamProcessor

        mock_client = AsyncMock()
        mock_client.xgroup_create = AsyncMock()
        mock_client.xreadgroup = AsyncMock(
            return_value=[
                (
                    "events",
                    [
                        ("1234-0", {"data": "test"}),
                    ],
                )
            ]
        )
        mock_client.xack = AsyncMock()
        mock_client.xpending_range = AsyncMock(return_value=[])

        with patch("redis_openai_agents.robust_processor.aioredis") as mock_aioredis:
            mock_aioredis.from_url.return_value = mock_client

            processor = RobustStreamProcessor(
                redis_url="redis://localhost:6379",
                stream_name="events",
                consumer_group="workers",
                consumer_name="worker_1",
            )
            await processor.initialize()

            handler = AsyncMock(return_value=True)
            await processor.process_batch(handler, batch_size=10, max_batches=1)

            mock_client.xreadgroup.assert_called()
            handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_batch_acks_successful_messages(self) -> None:
        """Should ACK messages when handler returns True."""
        from redis_openai_agents.robust_processor import RobustStreamProcessor

        mock_client = AsyncMock()
        mock_client.xgroup_create = AsyncMock()
        mock_client.xreadgroup = AsyncMock(
            return_value=[
                (
                    "events",
                    [
                        ("1234-0", {"data": "test"}),
                    ],
                )
            ]
        )
        mock_client.xack = AsyncMock()
        mock_client.xpending_range = AsyncMock(return_value=[])

        with patch("redis_openai_agents.robust_processor.aioredis") as mock_aioredis:
            mock_aioredis.from_url.return_value = mock_client

            processor = RobustStreamProcessor(
                redis_url="redis://localhost:6379",
                stream_name="events",
                consumer_group="workers",
            )
            await processor.initialize()

            handler = AsyncMock(return_value=True)
            await processor.process_batch(handler, batch_size=10, max_batches=1)

            mock_client.xack.assert_called_once_with("events", "workers", "1234-0")

    @pytest.mark.asyncio
    async def test_process_batch_no_ack_on_failure(self) -> None:
        """Should not ACK messages when handler returns False."""
        from redis_openai_agents.robust_processor import RobustStreamProcessor

        mock_client = AsyncMock()
        mock_client.xgroup_create = AsyncMock()
        mock_client.xreadgroup = AsyncMock(
            return_value=[
                (
                    "events",
                    [
                        ("1234-0", {"data": "test"}),
                    ],
                )
            ]
        )
        mock_client.xack = AsyncMock()
        mock_client.xpending_range = AsyncMock(return_value=[])

        with patch("redis_openai_agents.robust_processor.aioredis") as mock_aioredis:
            mock_aioredis.from_url.return_value = mock_client

            processor = RobustStreamProcessor(
                redis_url="redis://localhost:6379",
                stream_name="events",
                consumer_group="workers",
            )
            await processor.initialize()

            handler = AsyncMock(return_value=False)  # Failure
            await processor.process_batch(handler, batch_size=10, max_batches=1)

            mock_client.xack.assert_not_called()


class TestRobustStreamProcessorClaiming:
    """Tests for pending message claiming."""

    @pytest.mark.asyncio
    async def test_claim_pending_calls_xpending_range(self) -> None:
        """Should call XPENDING to get pending messages."""
        from redis_openai_agents.robust_processor import RobustStreamProcessor

        mock_client = AsyncMock()
        mock_client.xgroup_create = AsyncMock()
        mock_client.xpending_range = AsyncMock(return_value=[])

        with patch("redis_openai_agents.robust_processor.aioredis") as mock_aioredis:
            mock_aioredis.from_url.return_value = mock_client

            processor = RobustStreamProcessor(
                redis_url="redis://localhost:6379",
                stream_name="events",
                consumer_group="workers",
            )
            await processor.initialize()

            await processor.claim_pending_messages()

            mock_client.xpending_range.assert_called_once()

    @pytest.mark.asyncio
    async def test_claim_pending_calls_xclaim_for_old_messages(self) -> None:
        """Should call XCLAIM for messages older than timeout."""
        from redis_openai_agents.robust_processor import RobustStreamProcessor

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
        mock_client.xclaim = AsyncMock(return_value=[("1234-0", {"data": "test"})])

        with patch("redis_openai_agents.robust_processor.aioredis") as mock_aioredis:
            mock_aioredis.from_url.return_value = mock_client

            processor = RobustStreamProcessor(
                redis_url="redis://localhost:6379",
                stream_name="events",
                consumer_group="workers",
                consumer_name="recovery_worker",
                claim_timeout_ms=300000,  # 5 minutes
            )
            await processor.initialize()

            claimed = await processor.claim_pending_messages()

            mock_client.xclaim.assert_called_once()
            assert claimed == 1

    @pytest.mark.asyncio
    async def test_claim_moves_to_dlq_after_max_retries(self) -> None:
        """Should move message to DLQ after exceeding max retries."""
        from redis_openai_agents.robust_processor import RobustStreamProcessor

        mock_client = AsyncMock()
        mock_client.xgroup_create = AsyncMock()
        mock_client.xpending_range = AsyncMock(
            return_value=[
                {
                    "message_id": "1234-0",
                    "consumer": "dead_worker",
                    "time_since_delivered": 600000,
                    "times_delivered": 5,  # Exceeds max_retries
                }
            ]
        )
        mock_client.xrange = AsyncMock(return_value=[("1234-0", {"data": "failed_message"})])
        mock_client.xadd = AsyncMock(return_value="dlq-1234-0")
        mock_client.xack = AsyncMock()

        with patch("redis_openai_agents.robust_processor.aioredis") as mock_aioredis:
            mock_aioredis.from_url.return_value = mock_client

            processor = RobustStreamProcessor(
                redis_url="redis://localhost:6379",
                stream_name="events",
                consumer_group="workers",
                max_retries=3,
                claim_timeout_ms=300000,
            )
            await processor.initialize()

            await processor.claim_pending_messages()

            # Should have called xadd to add to DLQ
            mock_client.xadd.assert_called_once()
            call_args = mock_client.xadd.call_args
            assert call_args[0][0] == "events:dlq"

            # Should have ACKed the original message
            mock_client.xack.assert_called_once()


class TestRobustStreamProcessorDLQ:
    """Tests for DLQ operations."""

    @pytest.mark.asyncio
    async def test_get_dlq_messages(self) -> None:
        """Should retrieve messages from DLQ."""
        from redis_openai_agents.robust_processor import RobustStreamProcessor

        mock_client = AsyncMock()
        mock_client.xgroup_create = AsyncMock()
        mock_client.xrange = AsyncMock(
            return_value=[
                ("dlq-1234-0", {"data": "failed", "failure_reason": "error"}),
                ("dlq-1234-1", {"data": "failed2", "failure_reason": "timeout"}),
            ]
        )

        with patch("redis_openai_agents.robust_processor.aioredis") as mock_aioredis:
            mock_aioredis.from_url.return_value = mock_client

            processor = RobustStreamProcessor(
                redis_url="redis://localhost:6379",
                stream_name="events",
                consumer_group="workers",
            )
            await processor.initialize()

            messages = await processor.get_dlq_messages(count=10)

            assert len(messages) == 2
            assert messages[0]["id"] == "dlq-1234-0"
            assert messages[0]["data"] == "failed"

    @pytest.mark.asyncio
    async def test_replay_dlq_message(self) -> None:
        """Should replay DLQ message to main stream."""
        from redis_openai_agents.robust_processor import RobustStreamProcessor

        mock_client = AsyncMock()
        mock_client.xgroup_create = AsyncMock()
        mock_client.xrange = AsyncMock(
            return_value=[
                (
                    "dlq-1234-0",
                    {
                        "type": "event",
                        "data": "replay_me",
                        "original_stream": "events",
                        "original_id": "1234-0",
                        "failure_reason": "temp_error",
                        "attempts": "2",
                        "dlq_timestamp": "1234567890",
                    },
                )
            ]
        )
        mock_client.xadd = AsyncMock(return_value="5678-0")
        mock_client.xdel = AsyncMock()

        with patch("redis_openai_agents.robust_processor.aioredis") as mock_aioredis:
            mock_aioredis.from_url.return_value = mock_client

            processor = RobustStreamProcessor(
                redis_url="redis://localhost:6379",
                stream_name="events",
                consumer_group="workers",
            )
            await processor.initialize()

            new_id = await processor.replay_dlq_message("dlq-1234-0")

            assert new_id == "5678-0"

            # Verify xadd called without DLQ metadata
            xadd_call = mock_client.xadd.call_args
            assert xadd_call[0][0] == "events"  # Main stream
            data = xadd_call[0][1]
            assert data["type"] == "event"
            assert data["data"] == "replay_me"
            assert "replayed_from_dlq" in data
            assert "original_stream" not in data  # Removed
            assert "failure_reason" not in data  # Removed

            # Verify xdel called on DLQ
            mock_client.xdel.assert_called_once_with("events:dlq", "dlq-1234-0")


class TestRobustStreamProcessorHealthStats:
    """Tests for health statistics."""

    @pytest.mark.asyncio
    async def test_get_health_stats(self) -> None:
        """Should return health statistics."""
        from redis_openai_agents.robust_processor import RobustStreamProcessor

        mock_client = AsyncMock()
        mock_client.xgroup_create = AsyncMock()
        mock_client.xinfo_stream = AsyncMock(
            return_value={
                "length": 100,
                "first-entry": ("1234-0", {}),
                "last-entry": ("1234-99", {}),
            }
        )
        mock_client.xinfo_groups = AsyncMock(
            return_value=[
                {"name": "workers", "consumers": 3, "pending": 5, "last-delivered-id": "1234-95"}
            ]
        )
        mock_client.xlen = AsyncMock(return_value=2)

        with patch("redis_openai_agents.robust_processor.aioredis") as mock_aioredis:
            mock_aioredis.from_url.return_value = mock_client

            processor = RobustStreamProcessor(
                redis_url="redis://localhost:6379",
                stream_name="events",
                consumer_group="workers",
            )
            await processor.initialize()

            stats = await processor.get_health_stats()

            assert stats["stream_length"] == 100
            assert stats["pending_messages"] == 5
            assert stats["consumers"] == 3
            assert stats["dlq_length"] == 2
            assert stats["last_delivered_id"] == "1234-95"
