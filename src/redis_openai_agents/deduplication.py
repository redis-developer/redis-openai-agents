"""DeduplicationService - Bloom filter-based deduplication for agent operations.

This module provides deduplication capabilities using Redis Bloom Filters:
- Duplicate tool call detection within time windows
- Cache stampede prevention via distributed locks
- Request idempotency marking
- Message deduplication per session

Key Features:
- Space-efficient probabilistic data structure (millions of items in KB)
- O(1) operations for add/check
- No false negatives - "not in set" is always correct
- Configurable false positive rate
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import TYPE_CHECKING

from redis import asyncio as aioredis

if TYPE_CHECKING:
    from redis.asyncio import Redis


class DeduplicationService:
    """
    Bloom filter-based deduplication for agent operations.

    Prevents:
    - Duplicate tool executions
    - Duplicate message storage
    - Cache stampede (multiple concurrent cache-miss handlers)
    - Request replay attacks

    Example:
        >>> dedup = DeduplicationService(redis_url="redis://localhost:6379")
        >>> await dedup.initialize()
        >>> is_dup = await dedup.is_duplicate_tool_call("search", {"q": "redis"})
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        prefix: str = "dedup",
        default_error_rate: float = 0.01,
    ) -> None:
        """
        Initialize DeduplicationService.

        Args:
            redis_url: Redis connection URL
            prefix: Key prefix for all deduplication keys
            default_error_rate: Default false positive rate for Bloom filters
        """
        self._redis_url = redis_url
        self._prefix = prefix
        self._error_rate = default_error_rate
        self._client: Redis | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the Redis connection."""
        if self._initialized:
            return

        self._client = aioredis.from_url(self._redis_url, decode_responses=True)
        self._initialized = True

    async def _get_client(self) -> Redis:
        """Get Redis client, ensuring initialization."""
        if not self._initialized or self._client is None:
            await self.initialize()
        return self._client  # type: ignore[return-value]

    # --- Bloom Filter Basic Operations ---

    async def create_filter(
        self,
        name: str,
        capacity: int = 100000,
        error_rate: float | None = None,
    ) -> None:
        """
        Create a new Bloom filter.

        Args:
            name: Filter name (will be prefixed)
            capacity: Expected number of items
            error_rate: False positive rate (default: instance default)
        """
        client = await self._get_client()
        key = f"{self._prefix}:{name}"
        rate = error_rate or self._error_rate

        try:
            await client.bf().reserve(key, rate, capacity, expansion=2)  # type: ignore[no-untyped-call]
        except Exception:
            # Filter already exists - this is fine
            pass

    async def add_item(self, filter_name: str, item: str) -> None:
        """
        Add an item to a Bloom filter.

        Args:
            filter_name: Name of the filter
            item: Item to add
        """
        client = await self._get_client()
        key = f"{self._prefix}:{filter_name}"
        await client.bf().add(key, item)  # type: ignore[no-untyped-call]

    async def check_exists(self, filter_name: str, item: str) -> bool:
        """
        Check if an item exists in a Bloom filter.

        Args:
            filter_name: Name of the filter
            item: Item to check

        Returns:
            True if item might exist, False if definitely not
        """
        client = await self._get_client()
        key = f"{self._prefix}:{filter_name}"
        result = await client.bf().exists(key, item)  # type: ignore[no-untyped-call]
        return bool(result)

    # --- Duplicate Tool Call Detection ---

    async def is_duplicate_tool_call(
        self,
        tool_name: str,
        params: dict,
        window_minutes: int = 5,
    ) -> bool:
        """
        Check if tool was recently called with same params.

        Uses time-windowed filter to allow same call after window expires.

        Args:
            tool_name: Name of the tool
            params: Tool parameters
            window_minutes: Time window for deduplication

        Returns:
            True if this is a duplicate call, False if new
        """
        client = await self._get_client()

        # Create hash of tool + params
        params_hash = hashlib.sha256(
            f"{tool_name}:{json.dumps(params, sort_keys=True)}".encode()
        ).hexdigest()[:32]

        # Use time-window bucket
        bucket = int(time.time() // (window_minutes * 60))
        filter_name = f"tool_calls:{bucket}"
        key = f"{self._prefix}:{filter_name}"

        # Ensure filter exists
        await self.create_filter(filter_name, capacity=10000)

        # Check if exists
        exists = await client.bf().exists(key, params_hash)  # type: ignore[no-untyped-call]
        if exists:
            return True

        # Add to filter
        await client.bf().add(key, params_hash)  # type: ignore[no-untyped-call]

        # Set expiry on bucket (2x window for safety)
        await client.expire(key, window_minutes * 60 * 2)

        return False

    # --- Cache Stampede Prevention ---

    async def prevent_cache_stampede(
        self,
        query_hash: str,
        timeout_seconds: int = 30,
    ) -> bool:
        """
        Acquire lock to prevent cache stampede.

        When multiple processes cache-miss simultaneously,
        only one should compute the response.

        Args:
            query_hash: Hash of the query being processed
            timeout_seconds: Lock timeout

        Returns:
            True if this process should compute, False otherwise
        """
        client = await self._get_client()
        lock_key = f"{self._prefix}:cache_lock:{query_hash}"

        # Try to acquire lock with NX (only if not exists) and EX (expiry)
        acquired = await client.set(lock_key, "1", nx=True, ex=timeout_seconds)

        return bool(acquired)

    async def release_cache_lock(self, query_hash: str) -> None:
        """
        Release a cache stampede lock.

        Args:
            query_hash: Hash of the query being processed
        """
        client = await self._get_client()
        lock_key = f"{self._prefix}:cache_lock:{query_hash}"
        await client.delete(lock_key)

    # --- Request Idempotency ---

    async def mark_request_processed(
        self,
        request_id: str,
    ) -> bool:
        """
        Mark request as processed (for idempotency).

        Args:
            request_id: Unique request identifier

        Returns:
            True if this is a new request, False if duplicate
        """
        client = await self._get_client()

        filter_name = "requests"
        key = f"{self._prefix}:{filter_name}"

        # Ensure filter exists (high capacity for daily requests)
        await self.create_filter(filter_name, capacity=1000000)

        # Check if exists
        exists = await client.bf().exists(key, request_id)  # type: ignore[no-untyped-call]
        if exists:
            return False

        # Add to filter
        await client.bf().add(key, request_id)  # type: ignore[no-untyped-call]
        return True

    # --- Message Deduplication ---

    async def is_duplicate_message(
        self,
        session_id: str,
        message_content: str,
    ) -> bool:
        """
        Check if message was already added to session.

        Args:
            session_id: Session identifier
            message_content: Message content to check

        Returns:
            True if this is a duplicate message, False if new
        """
        client = await self._get_client()

        # Create content hash
        content_hash = hashlib.sha256(message_content.encode()).hexdigest()[:32]

        # Per-session filter
        filter_name = f"messages:{session_id}"
        key = f"{self._prefix}:{filter_name}"

        # Ensure filter exists
        await self.create_filter(filter_name, capacity=10000)

        # Check if exists
        exists = await client.bf().exists(key, content_hash)  # type: ignore[no-untyped-call]
        if exists:
            return True

        # Add to filter
        await client.bf().add(key, content_hash)  # type: ignore[no-untyped-call]
        return False

    async def close(self) -> None:
        """Close the Redis connection."""
        if self._client:
            await self._client.aclose()
            self._client = None
            self._initialized = False
