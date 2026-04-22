"""Unit tests for DeduplicationService.

These tests use mocks to verify behavior without Redis.
"""

from unittest.mock import AsyncMock, patch

import pytest


class TestDeduplicationServiceInit:
    """Tests for DeduplicationService initialization."""

    def test_init_sets_redis_url(self) -> None:
        """Should set Redis URL."""
        from redis_openai_agents.deduplication import DeduplicationService

        dedup = DeduplicationService(redis_url="redis://localhost:6379")

        assert dedup._redis_url == "redis://localhost:6379"

    def test_init_sets_prefix(self) -> None:
        """Should set custom prefix."""
        from redis_openai_agents.deduplication import DeduplicationService

        dedup = DeduplicationService(
            redis_url="redis://localhost:6379",
            prefix="custom_dedup",
        )

        assert dedup._prefix == "custom_dedup"

    def test_init_default_prefix(self) -> None:
        """Should use default prefix 'dedup'."""
        from redis_openai_agents.deduplication import DeduplicationService

        dedup = DeduplicationService(redis_url="redis://localhost:6379")

        assert dedup._prefix == "dedup"

    def test_init_sets_error_rate(self) -> None:
        """Should set custom error rate."""
        from redis_openai_agents.deduplication import DeduplicationService

        dedup = DeduplicationService(
            redis_url="redis://localhost:6379",
            default_error_rate=0.001,
        )

        assert dedup._error_rate == 0.001


class TestBloomFilterOperationsUnit:
    """Unit tests for Bloom filter operations."""

    @pytest.mark.asyncio
    async def test_create_filter_calls_bf_reserve(self) -> None:
        """Should call BF.RESERVE to create filter."""
        from redis_openai_agents.deduplication import DeduplicationService

        mock_client = AsyncMock()
        mock_bf = AsyncMock()
        mock_bf.reserve = AsyncMock()
        # bf() returns synchronously, not as a coroutine
        mock_client.bf = lambda: mock_bf

        with patch("redis_openai_agents.deduplication.aioredis") as mock_aioredis:
            mock_aioredis.from_url.return_value = mock_client

            dedup = DeduplicationService(redis_url="redis://localhost:6379")
            await dedup.initialize()

            await dedup.create_filter("test_filter", capacity=1000, error_rate=0.01)

            mock_bf.reserve.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_item_calls_bf_add(self) -> None:
        """Should call BF.ADD to add item."""
        from redis_openai_agents.deduplication import DeduplicationService

        mock_client = AsyncMock()
        mock_bf = AsyncMock()
        mock_bf.add = AsyncMock()
        mock_client.bf = lambda: mock_bf

        with patch("redis_openai_agents.deduplication.aioredis") as mock_aioredis:
            mock_aioredis.from_url.return_value = mock_client

            dedup = DeduplicationService(redis_url="redis://localhost:6379")
            await dedup.initialize()

            await dedup.add_item("test_filter", "test_item")

            mock_bf.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_exists_calls_bf_exists(self) -> None:
        """Should call BF.EXISTS to check item."""
        from redis_openai_agents.deduplication import DeduplicationService

        mock_client = AsyncMock()
        mock_bf = AsyncMock()
        mock_bf.exists = AsyncMock(return_value=True)
        mock_client.bf = lambda: mock_bf

        with patch("redis_openai_agents.deduplication.aioredis") as mock_aioredis:
            mock_aioredis.from_url.return_value = mock_client

            dedup = DeduplicationService(redis_url="redis://localhost:6379")
            await dedup.initialize()

            result = await dedup.check_exists("test_filter", "test_item")

            assert result is True
            mock_bf.exists.assert_called_once()


class TestDuplicateToolCallDetectionUnit:
    """Unit tests for duplicate tool call detection."""

    @pytest.mark.asyncio
    async def test_is_duplicate_tool_call_new(self) -> None:
        """Should return False for new tool call."""
        from redis_openai_agents.deduplication import DeduplicationService

        mock_client = AsyncMock()
        mock_bf = AsyncMock()
        mock_bf.reserve = AsyncMock()
        mock_bf.exists = AsyncMock(return_value=False)
        mock_bf.add = AsyncMock()
        mock_client.bf = lambda: mock_bf
        mock_client.expire = AsyncMock()

        with patch("redis_openai_agents.deduplication.aioredis") as mock_aioredis:
            mock_aioredis.from_url.return_value = mock_client

            dedup = DeduplicationService(redis_url="redis://localhost:6379")
            await dedup.initialize()

            result = await dedup.is_duplicate_tool_call(
                tool_name="test_tool",
                params={"arg": "value"},
                window_minutes=5,
            )

            assert result is False
            mock_bf.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_is_duplicate_tool_call_duplicate(self) -> None:
        """Should return True for duplicate tool call."""
        from redis_openai_agents.deduplication import DeduplicationService

        mock_client = AsyncMock()
        mock_bf = AsyncMock()
        mock_bf.reserve = AsyncMock()
        mock_bf.exists = AsyncMock(return_value=True)
        mock_client.bf = lambda: mock_bf

        with patch("redis_openai_agents.deduplication.aioredis") as mock_aioredis:
            mock_aioredis.from_url.return_value = mock_client

            dedup = DeduplicationService(redis_url="redis://localhost:6379")
            await dedup.initialize()

            result = await dedup.is_duplicate_tool_call(
                tool_name="test_tool",
                params={"arg": "value"},
                window_minutes=5,
            )

            assert result is True
            mock_bf.add.assert_not_called()


class TestCacheStampedePreventionUnit:
    """Unit tests for cache stampede prevention."""

    @pytest.mark.asyncio
    async def test_prevent_cache_stampede_acquires_lock(self) -> None:
        """Should acquire lock when available."""
        from redis_openai_agents.deduplication import DeduplicationService

        mock_client = AsyncMock()
        mock_client.set = AsyncMock(return_value=True)

        with patch("redis_openai_agents.deduplication.aioredis") as mock_aioredis:
            mock_aioredis.from_url.return_value = mock_client

            dedup = DeduplicationService(redis_url="redis://localhost:6379")
            await dedup.initialize()

            result = await dedup.prevent_cache_stampede("query_hash", timeout_seconds=30)

            assert result is True
            mock_client.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_prevent_cache_stampede_blocked(self) -> None:
        """Should return False when lock not available."""
        from redis_openai_agents.deduplication import DeduplicationService

        mock_client = AsyncMock()
        mock_client.set = AsyncMock(return_value=False)

        with patch("redis_openai_agents.deduplication.aioredis") as mock_aioredis:
            mock_aioredis.from_url.return_value = mock_client

            dedup = DeduplicationService(redis_url="redis://localhost:6379")
            await dedup.initialize()

            result = await dedup.prevent_cache_stampede("query_hash", timeout_seconds=30)

            assert result is False

    @pytest.mark.asyncio
    async def test_release_cache_lock_calls_delete(self) -> None:
        """Should call DELETE to release lock."""
        from redis_openai_agents.deduplication import DeduplicationService

        mock_client = AsyncMock()
        mock_client.delete = AsyncMock()

        with patch("redis_openai_agents.deduplication.aioredis") as mock_aioredis:
            mock_aioredis.from_url.return_value = mock_client

            dedup = DeduplicationService(redis_url="redis://localhost:6379")
            await dedup.initialize()

            await dedup.release_cache_lock("query_hash")

            mock_client.delete.assert_called_once()


class TestRequestIdempotencyUnit:
    """Unit tests for request idempotency."""

    @pytest.mark.asyncio
    async def test_mark_request_processed_new(self) -> None:
        """Should return True for new request."""
        from redis_openai_agents.deduplication import DeduplicationService

        mock_client = AsyncMock()
        mock_bf = AsyncMock()
        mock_bf.reserve = AsyncMock()
        mock_bf.exists = AsyncMock(return_value=False)
        mock_bf.add = AsyncMock()
        mock_client.bf = lambda: mock_bf

        with patch("redis_openai_agents.deduplication.aioredis") as mock_aioredis:
            mock_aioredis.from_url.return_value = mock_client

            dedup = DeduplicationService(redis_url="redis://localhost:6379")
            await dedup.initialize()

            result = await dedup.mark_request_processed("request_123")

            assert result is True
            mock_bf.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_mark_request_processed_duplicate(self) -> None:
        """Should return False for duplicate request."""
        from redis_openai_agents.deduplication import DeduplicationService

        mock_client = AsyncMock()
        mock_bf = AsyncMock()
        mock_bf.reserve = AsyncMock()
        mock_bf.exists = AsyncMock(return_value=True)
        mock_client.bf = lambda: mock_bf

        with patch("redis_openai_agents.deduplication.aioredis") as mock_aioredis:
            mock_aioredis.from_url.return_value = mock_client

            dedup = DeduplicationService(redis_url="redis://localhost:6379")
            await dedup.initialize()

            result = await dedup.mark_request_processed("request_123")

            assert result is False
            mock_bf.add.assert_not_called()


class TestMessageDeduplicationUnit:
    """Unit tests for message deduplication."""

    @pytest.mark.asyncio
    async def test_is_duplicate_message_new(self) -> None:
        """Should return False for new message."""
        from redis_openai_agents.deduplication import DeduplicationService

        mock_client = AsyncMock()
        mock_bf = AsyncMock()
        mock_bf.reserve = AsyncMock()
        mock_bf.exists = AsyncMock(return_value=False)
        mock_bf.add = AsyncMock()
        mock_client.bf = lambda: mock_bf

        with patch("redis_openai_agents.deduplication.aioredis") as mock_aioredis:
            mock_aioredis.from_url.return_value = mock_client

            dedup = DeduplicationService(redis_url="redis://localhost:6379")
            await dedup.initialize()

            result = await dedup.is_duplicate_message("session_1", "Hello world")

            assert result is False
            mock_bf.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_is_duplicate_message_duplicate(self) -> None:
        """Should return True for duplicate message."""
        from redis_openai_agents.deduplication import DeduplicationService

        mock_client = AsyncMock()
        mock_bf = AsyncMock()
        mock_bf.reserve = AsyncMock()
        mock_bf.exists = AsyncMock(return_value=True)
        mock_client.bf = lambda: mock_bf

        with patch("redis_openai_agents.deduplication.aioredis") as mock_aioredis:
            mock_aioredis.from_url.return_value = mock_client

            dedup = DeduplicationService(redis_url="redis://localhost:6379")
            await dedup.initialize()

            result = await dedup.is_duplicate_message("session_1", "Hello world")

            assert result is True
            mock_bf.add.assert_not_called()
