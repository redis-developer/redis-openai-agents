"""RankedOperations - Sorted set-based ranking for agent operations.

This module provides ranking and prioritization capabilities using Redis Sorted Sets:
- Agent performance ranking by task type
- Session LRU tracking for cleanup
- Token budget rate limiting per user
- Tool effectiveness ranking

Key Features:
- Atomic score updates via ZADD
- Efficient range queries with ZREVRANGE/ZRANGEBYSCORE
- Natural expiration with ZREMRANGEBYSCORE
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from redis import asyncio as aioredis

if TYPE_CHECKING:
    from redis.asyncio import Redis


class RankedOperations:
    """
    Sorted set-based ranking for agent operations.

    Use cases:
    - Agent success rate leaderboards
    - Tool effectiveness ranking
    - LRU session tracking
    - Rate limit budgets

    Example:
        >>> ranking = RankedOperations(redis_url="redis://localhost:6379")
        >>> await ranking.initialize()
        >>> await ranking.record_agent_success("agent_1", "research", True, 150.0)
        >>> top_agents = await ranking.get_best_agents("research", limit=5)
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        prefix: str = "rank",
    ) -> None:
        """
        Initialize RankedOperations.

        Args:
            redis_url: Redis connection URL
            prefix: Key prefix for all sorted sets
        """
        self._redis_url = redis_url
        self._prefix = prefix
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

    async def _update_score(
        self,
        *,
        stats_prefix: str,
        ranking_key: str,
        member: str,
        success: bool,
        latency_ms: float,
    ) -> None:
        """Update a performance score from success/latency counters.

        Shared logic for agent and tool score updates.

        Args:
            stats_prefix: Key prefix for the success/total/latency counters
            ranking_key: Sorted set key to update
            member: Member name in the sorted set
            success: Whether the operation succeeded
            latency_ms: Operation time in milliseconds
        """
        client = await self._get_client()

        success_key = f"{stats_prefix}:success"
        total_key = f"{stats_prefix}:total"
        latency_key = f"{stats_prefix}:latency_sum"

        await client.incr(total_key)
        if success:
            await client.incr(success_key)
        await client.incrbyfloat(latency_key, latency_ms)

        success_count = int(await client.get(success_key) or 0)
        total_count = int(await client.get(total_key) or 1)
        latency_sum = float(await client.get(latency_key) or 1)

        success_rate = success_count / total_count
        avg_latency = latency_sum / total_count

        latency_factor = 1 - min(avg_latency / 10000, 1)
        score = success_rate * 0.7 + latency_factor * 0.3

        await client.zadd(ranking_key, {member: score})

    # --- Agent Performance Ranking ---

    async def record_agent_success(
        self,
        agent_id: str,
        task_type: str,
        success: bool,
        latency_ms: float,
    ) -> None:
        """
        Update agent performance score.

        Score = success_rate * 0.7 + latency_factor * 0.3
        Higher = better

        Args:
            agent_id: Unique agent identifier
            task_type: Type of task (e.g., "research", "analysis")
            success: Whether the task succeeded
            latency_ms: Task completion time in milliseconds
        """
        await self._update_score(
            stats_prefix=f"{self._prefix}:agent_stats:{agent_id}:{task_type}",
            ranking_key=f"{self._prefix}:agents:{task_type}",
            member=agent_id,
            success=success,
            latency_ms=latency_ms,
        )

    async def get_best_agents(
        self,
        task_type: str,
        limit: int = 5,
    ) -> list[tuple[str, float]]:
        """
        Get top-performing agents for task type.

        Args:
            task_type: Type of task to rank
            limit: Maximum number of agents to return

        Returns:
            List of (agent_id, score) tuples in descending order
        """
        client = await self._get_client()

        key = f"{self._prefix}:agents:{task_type}"
        result = await client.zrevrange(key, 0, limit - 1, withscores=True)
        return list(result)

    # --- Tool Effectiveness Ranking ---

    async def record_tool_success(
        self,
        tool_name: str,
        success: bool,
        latency_ms: float,
    ) -> None:
        """
        Update tool effectiveness score.

        Args:
            tool_name: Name of the tool
            success: Whether the tool call succeeded
            latency_ms: Execution time in milliseconds
        """
        await self._update_score(
            stats_prefix=f"{self._prefix}:tool_stats:{tool_name}",
            ranking_key=f"{self._prefix}:tools",
            member=tool_name,
            success=success,
            latency_ms=latency_ms,
        )

    async def get_best_tools(
        self,
        limit: int = 5,
    ) -> list[tuple[str, float]]:
        """
        Get top-performing tools.

        Args:
            limit: Maximum number of tools to return

        Returns:
            List of (tool_name, score) tuples in descending order
        """
        client = await self._get_client()

        key = f"{self._prefix}:tools"
        result = await client.zrevrange(key, 0, limit - 1, withscores=True)
        return list(result)

    # --- Session LRU Tracking ---

    async def touch_session(self, session_id: str) -> None:
        """
        Mark session as recently used.

        Args:
            session_id: Session identifier
        """
        client = await self._get_client()

        key = f"{self._prefix}:session_lru"
        await client.zadd(key, {session_id: time.time()})

    async def get_stale_sessions(
        self,
        max_age_seconds: int = 86400,
        limit: int = 100,
    ) -> list[str]:
        """
        Get sessions not accessed within max_age.

        Args:
            max_age_seconds: Maximum age in seconds (default 24 hours)
            limit: Maximum number of sessions to return

        Returns:
            List of stale session IDs
        """
        client = await self._get_client()

        key = f"{self._prefix}:session_lru"
        cutoff = time.time() - max_age_seconds

        result = await client.zrangebyscore(key, "-inf", cutoff, start=0, num=limit)
        return list(result)

    async def evict_stale_sessions(
        self,
        max_age_seconds: int = 86400,
        batch_size: int = 100,
    ) -> int:
        """
        Evict sessions not accessed within max_age.

        Deletes session data and removes from LRU tracking.

        Args:
            max_age_seconds: Maximum age in seconds
            batch_size: Maximum sessions to evict in one call

        Returns:
            Number of sessions evicted
        """
        client = await self._get_client()

        stale = await self.get_stale_sessions(max_age_seconds, batch_size)

        if not stale:
            return 0

        # Delete session data
        session_keys = [f"session:{sid}" for sid in stale]
        await client.delete(*session_keys)

        # Remove from LRU tracking
        key = f"{self._prefix}:session_lru"
        await client.zrem(key, *stale)

        return len(stale)

    # --- Rate Limiting ---

    async def check_token_budget(
        self,
        user_id: str,
        tokens_needed: int,
        budget_per_hour: int = 100000,
    ) -> tuple[bool, int]:
        """
        Check if user has token budget remaining.

        Uses hourly buckets with TTL for automatic cleanup.

        Args:
            user_id: User identifier
            tokens_needed: Number of tokens requested
            budget_per_hour: Maximum tokens allowed per hour

        Returns:
            Tuple of (allowed, remaining_tokens)
        """
        client = await self._get_client()

        hour_bucket = int(time.time() // 3600)
        key = f"{self._prefix}:token_budget:{user_id}:{hour_bucket}"

        # Get current usage
        current = int(await client.get(key) or 0)
        remaining = budget_per_hour - current

        if tokens_needed > remaining:
            return False, remaining

        # Increment usage and set TTL (24 hours for analytics retention)
        await client.incrby(key, tokens_needed)
        await client.expire(key, 86400)  # 24 hours TTL

        return True, remaining - tokens_needed

    async def get_token_usage(
        self,
        user_id: str,
        budget_per_hour: int = 100000,
    ) -> dict[str, Any]:
        """
        Get token usage statistics for user.

        Args:
            user_id: User identifier
            budget_per_hour: Budget limit for remaining calculation

        Returns:
            Dictionary with usage statistics
        """
        client = await self._get_client()

        hour_bucket = int(time.time() // 3600)
        key = f"{self._prefix}:token_budget:{user_id}:{hour_bucket}"

        current = int(await client.get(key) or 0)

        return {
            "current_hour": current,
            "remaining": budget_per_hour - current,
            "budget_per_hour": budget_per_hour,
            "hour_bucket": hour_bucket,
        }

    async def close(self) -> None:
        """Close the Redis connection."""
        if self._client:
            await self._client.aclose()
            self._client = None
            self._initialized = False
