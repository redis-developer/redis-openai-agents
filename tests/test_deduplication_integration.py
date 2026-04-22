"""Integration tests for DeduplicationService.

These tests run against a real Redis instance and verify:
- Bloom filter creation and existence checking
- Duplicate tool call detection
- Cache stampede prevention
- Request idempotency marking
"""

import asyncio
from uuid import uuid4

import pytest


class TestBloomFilterBasics:
    """Tests for basic Bloom filter operations."""

    @pytest.mark.asyncio
    async def test_create_filter(self, redis_url: str) -> None:
        """Should create a Bloom filter without error."""
        from redis_openai_agents.deduplication import DeduplicationService

        dedup = DeduplicationService(redis_url=redis_url)
        await dedup.initialize()

        # Should not raise
        await dedup.create_filter("test_filter", capacity=1000)

        await dedup.close()

    @pytest.mark.asyncio
    async def test_create_filter_idempotent(self, redis_url: str) -> None:
        """Creating the same filter twice should not error."""
        from redis_openai_agents.deduplication import DeduplicationService

        dedup = DeduplicationService(redis_url=redis_url)
        await dedup.initialize()

        # Create twice - should not raise
        await dedup.create_filter("idempotent_filter", capacity=1000)
        await dedup.create_filter("idempotent_filter", capacity=1000)

        await dedup.close()

    @pytest.mark.asyncio
    async def test_add_and_check_item(self, redis_url: str) -> None:
        """Should add item and detect it as existing."""
        from redis_openai_agents.deduplication import DeduplicationService

        dedup = DeduplicationService(redis_url=redis_url)
        await dedup.initialize()

        filter_name = "add_check_test"
        await dedup.create_filter(filter_name, capacity=1000)

        # Item should not exist initially
        exists_before = await dedup.check_exists(filter_name, "test_item")
        assert exists_before is False

        # Add item
        await dedup.add_item(filter_name, "test_item")

        # Item should exist now
        exists_after = await dedup.check_exists(filter_name, "test_item")
        assert exists_after is True

        await dedup.close()


class TestDuplicateToolCallDetection:
    """Tests for duplicate tool call detection."""

    @pytest.mark.asyncio
    async def test_first_call_not_duplicate(self, redis_url: str) -> None:
        """First tool call should not be detected as duplicate."""
        from redis_openai_agents.deduplication import DeduplicationService

        dedup = DeduplicationService(redis_url=redis_url)
        await dedup.initialize()

        is_dup = await dedup.is_duplicate_tool_call(
            tool_name="web_search",
            params={"query": "redis bloom filters"},
            window_minutes=5,
        )

        assert is_dup is False

        await dedup.close()

    @pytest.mark.asyncio
    async def test_second_call_same_params_is_duplicate(self, redis_url: str) -> None:
        """Second call with same params should be detected as duplicate."""
        from redis_openai_agents.deduplication import DeduplicationService

        dedup = DeduplicationService(redis_url=redis_url)
        await dedup.initialize()

        params = {"query": f"unique_query_{uuid4()}"}

        # First call
        is_dup1 = await dedup.is_duplicate_tool_call(
            tool_name="web_search",
            params=params,
            window_minutes=5,
        )
        assert is_dup1 is False

        # Second call with same params
        is_dup2 = await dedup.is_duplicate_tool_call(
            tool_name="web_search",
            params=params,
            window_minutes=5,
        )
        assert is_dup2 is True

        await dedup.close()

    @pytest.mark.asyncio
    async def test_different_params_not_duplicate(self, redis_url: str) -> None:
        """Calls with different params should not be duplicates."""
        from redis_openai_agents.deduplication import DeduplicationService

        dedup = DeduplicationService(redis_url=redis_url)
        await dedup.initialize()

        # First call
        is_dup1 = await dedup.is_duplicate_tool_call(
            tool_name="web_search",
            params={"query": f"first_query_{uuid4()}"},
            window_minutes=5,
        )
        assert is_dup1 is False

        # Second call with different params
        is_dup2 = await dedup.is_duplicate_tool_call(
            tool_name="web_search",
            params={"query": f"different_query_{uuid4()}"},
            window_minutes=5,
        )
        assert is_dup2 is False

        await dedup.close()

    @pytest.mark.asyncio
    async def test_different_tools_not_duplicate(self, redis_url: str) -> None:
        """Same params for different tools should not be duplicates."""
        from redis_openai_agents.deduplication import DeduplicationService

        dedup = DeduplicationService(redis_url=redis_url)
        await dedup.initialize()

        params = {"query": f"shared_query_{uuid4()}"}

        # First tool
        is_dup1 = await dedup.is_duplicate_tool_call(
            tool_name="tool_a",
            params=params,
            window_minutes=5,
        )
        assert is_dup1 is False

        # Different tool, same params
        is_dup2 = await dedup.is_duplicate_tool_call(
            tool_name="tool_b",
            params=params,
            window_minutes=5,
        )
        assert is_dup2 is False

        await dedup.close()


class TestCacheStampedePrevention:
    """Tests for cache stampede prevention."""

    @pytest.mark.asyncio
    async def test_first_caller_acquires_lock(self, redis_url: str) -> None:
        """First caller should acquire the lock."""
        from redis_openai_agents.deduplication import DeduplicationService

        dedup = DeduplicationService(redis_url=redis_url)
        await dedup.initialize()

        query_hash = f"query_{uuid4()}"
        acquired = await dedup.prevent_cache_stampede(query_hash, timeout_seconds=30)

        assert acquired is True

        await dedup.close()

    @pytest.mark.asyncio
    async def test_second_caller_blocked(self, redis_url: str) -> None:
        """Second caller should not acquire the lock."""
        from redis_openai_agents.deduplication import DeduplicationService

        dedup = DeduplicationService(redis_url=redis_url)
        await dedup.initialize()

        query_hash = f"query_{uuid4()}"

        # First caller acquires lock
        acquired1 = await dedup.prevent_cache_stampede(query_hash, timeout_seconds=30)
        assert acquired1 is True

        # Second caller blocked
        acquired2 = await dedup.prevent_cache_stampede(query_hash, timeout_seconds=30)
        assert acquired2 is False

        await dedup.close()

    @pytest.mark.asyncio
    async def test_lock_expires_after_timeout(self, redis_url: str) -> None:
        """Lock should expire after timeout, allowing new acquisition."""
        from redis_openai_agents.deduplication import DeduplicationService

        dedup = DeduplicationService(redis_url=redis_url)
        await dedup.initialize()

        query_hash = f"query_{uuid4()}"

        # First caller acquires lock with short timeout
        acquired1 = await dedup.prevent_cache_stampede(query_hash, timeout_seconds=1)
        assert acquired1 is True

        # Wait for lock to expire
        await asyncio.sleep(1.5)

        # Second caller should now be able to acquire
        acquired2 = await dedup.prevent_cache_stampede(query_hash, timeout_seconds=30)
        assert acquired2 is True

        await dedup.close()

    @pytest.mark.asyncio
    async def test_release_lock(self, redis_url: str) -> None:
        """Should be able to explicitly release the lock."""
        from redis_openai_agents.deduplication import DeduplicationService

        dedup = DeduplicationService(redis_url=redis_url)
        await dedup.initialize()

        query_hash = f"query_{uuid4()}"

        # First caller acquires lock
        acquired1 = await dedup.prevent_cache_stampede(query_hash, timeout_seconds=30)
        assert acquired1 is True

        # Release the lock
        await dedup.release_cache_lock(query_hash)

        # Second caller should now be able to acquire
        acquired2 = await dedup.prevent_cache_stampede(query_hash, timeout_seconds=30)
        assert acquired2 is True

        await dedup.close()


class TestRequestIdempotency:
    """Tests for request idempotency marking."""

    @pytest.mark.asyncio
    async def test_new_request_returns_true(self, redis_url: str) -> None:
        """New request should be marked as new (return True)."""
        from redis_openai_agents.deduplication import DeduplicationService

        dedup = DeduplicationService(redis_url=redis_url)
        await dedup.initialize()

        request_id = f"req_{uuid4()}"
        is_new = await dedup.mark_request_processed(request_id)

        assert is_new is True

        await dedup.close()

    @pytest.mark.asyncio
    async def test_duplicate_request_returns_false(self, redis_url: str) -> None:
        """Duplicate request should return False."""
        from redis_openai_agents.deduplication import DeduplicationService

        dedup = DeduplicationService(redis_url=redis_url)
        await dedup.initialize()

        request_id = f"req_{uuid4()}"

        # First time - new
        is_new1 = await dedup.mark_request_processed(request_id)
        assert is_new1 is True

        # Second time - duplicate
        is_new2 = await dedup.mark_request_processed(request_id)
        assert is_new2 is False

        await dedup.close()

    @pytest.mark.asyncio
    async def test_different_requests_both_new(self, redis_url: str) -> None:
        """Different requests should both be marked as new."""
        from redis_openai_agents.deduplication import DeduplicationService

        dedup = DeduplicationService(redis_url=redis_url)
        await dedup.initialize()

        request_id1 = f"req_{uuid4()}"
        request_id2 = f"req_{uuid4()}"

        is_new1 = await dedup.mark_request_processed(request_id1)
        is_new2 = await dedup.mark_request_processed(request_id2)

        assert is_new1 is True
        assert is_new2 is True

        await dedup.close()


class TestMessageDeduplication:
    """Tests for message deduplication."""

    @pytest.mark.asyncio
    async def test_is_duplicate_message_new(self, redis_url: str) -> None:
        """New message should not be detected as duplicate."""
        from redis_openai_agents.deduplication import DeduplicationService

        dedup = DeduplicationService(redis_url=redis_url)
        await dedup.initialize()

        session_id = f"session_{uuid4()}"
        message_content = "Hello, this is a unique message"

        is_dup = await dedup.is_duplicate_message(session_id, message_content)

        assert is_dup is False

        await dedup.close()

    @pytest.mark.asyncio
    async def test_is_duplicate_message_same_content(self, redis_url: str) -> None:
        """Same message content should be detected as duplicate."""
        from redis_openai_agents.deduplication import DeduplicationService

        dedup = DeduplicationService(redis_url=redis_url)
        await dedup.initialize()

        session_id = f"session_{uuid4()}"
        message_content = f"Hello, this is message {uuid4()}"

        # First message
        is_dup1 = await dedup.is_duplicate_message(session_id, message_content)
        assert is_dup1 is False

        # Same content again
        is_dup2 = await dedup.is_duplicate_message(session_id, message_content)
        assert is_dup2 is True

        await dedup.close()

    @pytest.mark.asyncio
    async def test_different_sessions_not_duplicate(self, redis_url: str) -> None:
        """Same message in different sessions should not be duplicate."""
        from redis_openai_agents.deduplication import DeduplicationService

        dedup = DeduplicationService(redis_url=redis_url)
        await dedup.initialize()

        message_content = f"Hello, shared content {uuid4()}"

        # Message in session 1
        is_dup1 = await dedup.is_duplicate_message("session_a", message_content)
        assert is_dup1 is False

        # Same message in session 2
        is_dup2 = await dedup.is_duplicate_message("session_b", message_content)
        assert is_dup2 is False

        await dedup.close()
