"""AgentMetrics - Time-series metrics using RedisTimeSeries.

This module provides built-in observability for agent operations using
Redis TimeSeries for efficient time-series data storage and querying.

Features:
- Record latency, token counts, and cache hits
- Query aggregated statistics
- Time range queries for analysis
- Automatic retention policies
"""

import logging
import threading
import time
from typing import TYPE_CHECKING, Any, Optional

from redis import Redis
from redis import asyncio as aioredis

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .pool import RedisConnectionPool

# Thread-safe counter for unique timestamps
_ts_lock = threading.Lock()
_last_ts = 0


def _get_unique_ts() -> int:
    """Get a strictly monotonically increasing timestamp in milliseconds."""
    global _last_ts
    with _ts_lock:
        current_ms = int(time.time() * 1000)
        # Ensure we always return a value greater than the last one
        if current_ms <= _last_ts:
            _last_ts += 1
        else:
            _last_ts = current_ms
        return _last_ts


class AgentMetrics:
    """Time-series metrics collector using RedisTimeSeries.

    Provides built-in observability for agent operations without
    requiring a separate time-series database.

    Example:
        >>> metrics = AgentMetrics(name="my_agent")
        >>> metrics.record(latency_ms=150.0, input_tokens=100)
        >>> stats = metrics.get_stats()
        >>> print(f"Average latency: {stats['latency_avg']}")

    Args:
        name: Metrics namespace/agent name
        redis_url: Redis connection URL
        retention_ms: Data retention period in milliseconds (default: 1 hour)
    """

    def __init__(
        self,
        name: str,
        redis_url: str = "redis://localhost:6379",
        retention_ms: int = 3600000,  # 1 hour default
        pool: Optional["RedisConnectionPool"] = None,
    ) -> None:
        """Initialize the metrics collector.

        Args:
            name: Metrics namespace/agent name
            redis_url: Redis connection URL
            retention_ms: Data retention period in milliseconds
            pool: Optional shared connection pool
        """
        self._name = name
        self._retention_ms = retention_ms
        self._pool = pool

        # Use pool's client if provided
        if pool is not None:
            self._redis_url = pool.redis_url
            self._client = pool.get_sync_client()
        else:
            self._redis_url = redis_url
            self._client = Redis.from_url(redis_url, decode_responses=True)

        # Create time series keys for each metric
        self._keys = {
            "latency": f"metrics:{name}:latency",
            "input_tokens": f"metrics:{name}:input_tokens",
            "output_tokens": f"metrics:{name}:output_tokens",
            "cache_hit": f"metrics:{name}:cache_hit",
            "count": f"metrics:{name}:count",
        }

        # Ensure time series exist with retention
        self._ensure_timeseries()

    def _ensure_timeseries(self) -> None:
        """Create time series keys if they don't exist."""
        for key in self._keys.values():
            try:
                self._client.execute_command(  # type: ignore[no-untyped-call]
                    "TS.CREATE",
                    key,
                    "RETENTION",
                    self._retention_ms,
                    "DUPLICATE_POLICY",
                    "LAST",
                )
            except Exception as exc:
                # Key may already exist
                logger.debug("TS.CREATE for %s may already exist: %s", key, exc)

    @property
    def name(self) -> str:
        """Metrics namespace/agent name."""
        return self._name

    def record(
        self,
        latency_ms: float | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cache_hit: bool | None = None,
    ) -> None:
        """Record metrics for an agent request.

        Args:
            latency_ms: Request processing time in milliseconds
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cache_hit: Whether the request was a cache hit
        """
        # Get unique timestamp to avoid collisions when recording multiple samples
        timestamp = _get_unique_ts()

        # Record each metric if provided
        if latency_ms is not None:
            self._client.execute_command(  # type: ignore[no-untyped-call]
                "TS.ADD", self._keys["latency"], timestamp, latency_ms
            )

        if input_tokens is not None:
            self._client.execute_command(  # type: ignore[no-untyped-call]
                "TS.ADD", self._keys["input_tokens"], timestamp, input_tokens
            )

        if output_tokens is not None:
            self._client.execute_command(  # type: ignore[no-untyped-call]
                "TS.ADD", self._keys["output_tokens"], timestamp, output_tokens
            )

        if cache_hit is not None:
            self._client.execute_command(  # type: ignore[no-untyped-call]
                "TS.ADD", self._keys["cache_hit"], timestamp, 1 if cache_hit else 0
            )

        # Always increment count
        self._client.execute_command(  # type: ignore[no-untyped-call]
            "TS.ADD", self._keys["count"], timestamp, 1
        )

    def get_stats(self) -> dict[str, Any]:
        """Get aggregated statistics for all metrics.

        Returns:
            Dictionary with aggregated stats:
            - count: Total number of requests
            - latency_avg/min/max: Latency statistics
            - input_tokens_sum/output_tokens_sum: Total tokens
            - cache_hit_rate: Percentage of cache hits
        """
        stats: dict[str, Any] = {
            "count": 0,
            "latency_avg": 0.0,
            "latency_min": 0.0,
            "latency_max": 0.0,
            "input_tokens_sum": 0.0,
            "output_tokens_sum": 0.0,
            "cache_hit_rate": 0.0,
        }

        try:
            # Get count
            count_info = self._client.execute_command(  # type: ignore[no-untyped-call]
                "TS.INFO", self._keys["count"]
            )
            if count_info:
                # Parse info response
                info_dict = self._parse_ts_info(count_info)
                stats["count"] = int(info_dict.get("totalSamples", 0))

            if stats["count"] == 0:
                return stats

            # Get latency stats using TS.RANGE with aggregation
            from_time = 0
            to_time = int(time.time() * 1000) + 1000

            try:
                latency_data = self._client.execute_command(  # type: ignore[no-untyped-call]
                    "TS.RANGE", self._keys["latency"], from_time, to_time
                )
                if latency_data:
                    values = [float(v) for _, v in latency_data]
                    if values:
                        stats["latency_avg"] = sum(values) / len(values)
                        stats["latency_min"] = min(values)
                        stats["latency_max"] = max(values)
            except Exception as exc:
                logger.debug("failed to query metric: %s", exc)

            # Get token sums
            try:
                input_data = self._client.execute_command(  # type: ignore[no-untyped-call]
                    "TS.RANGE", self._keys["input_tokens"], from_time, to_time
                )
                if input_data:
                    stats["input_tokens_sum"] = sum(float(v) for _, v in input_data)
            except Exception as exc:
                logger.debug("failed to query metric: %s", exc)

            try:
                output_data = self._client.execute_command(  # type: ignore[no-untyped-call]
                    "TS.RANGE", self._keys["output_tokens"], from_time, to_time
                )
                if output_data:
                    stats["output_tokens_sum"] = sum(float(v) for _, v in output_data)
            except Exception as exc:
                logger.debug("failed to query metric: %s", exc)

            # Calculate cache hit rate
            try:
                cache_data = self._client.execute_command(  # type: ignore[no-untyped-call]
                    "TS.RANGE", self._keys["cache_hit"], from_time, to_time
                )
                if cache_data:
                    hits = sum(float(v) for _, v in cache_data)
                    total = len(cache_data)
                    stats["cache_hit_rate"] = hits / total if total > 0 else 0.0
            except Exception as exc:
                logger.debug("failed to query metric: %s", exc)

        except Exception as exc:
            logger.debug("get_stats failed: %s", exc)

        return stats

    def _parse_ts_info(self, info: list[Any]) -> dict[str, Any]:
        """Parse TS.INFO response into a dictionary."""
        result: dict[str, Any] = {}
        it = iter(info)
        for key in it:
            try:
                value = next(it)
                if isinstance(key, bytes):
                    key = key.decode()
                result[key] = value
            except StopIteration:
                break
        return result

    def range(
        self,
        metric: str,
        from_time: int,
        to_time: int,
    ) -> list[tuple[int, float]]:
        """Query metric data over a time range.

        Args:
            metric: Metric name (latency, input_tokens, output_tokens, cache_hit)
            from_time: Start timestamp in milliseconds
            to_time: End timestamp in milliseconds

        Returns:
            List of (timestamp, value) tuples
        """
        key = self._keys.get(metric)
        if not key:
            return []

        try:
            data = self._client.execute_command(  # type: ignore[no-untyped-call]
                "TS.RANGE", key, from_time, to_time
            )
            if data:
                return [(int(ts), float(v)) for ts, v in data]
        except Exception as exc:
            logger.debug("range query failed: %s", exc)

        return []

    def delete(self) -> None:
        """Delete all metrics data."""
        for key in self._keys.values():
            try:
                self._client.delete(key)
            except Exception as exc:
                logger.debug("failed to delete metric key: %s", exc)

        # Recreate empty time series
        self._ensure_timeseries()

    def close(self) -> None:
        """Close the Redis connection."""
        self._client.close()

    # Async methods

    def _get_async_redis(self) -> aioredis.Redis:
        """Get or create async Redis client."""
        if not hasattr(self, "_async_client"):
            self._async_client: aioredis.Redis = aioredis.from_url(
                self._redis_url, decode_responses=True
            )
        return self._async_client

    async def arecord(
        self,
        latency_ms: float | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cache_hit: bool | None = None,
    ) -> None:
        """Async version of record() - record metrics for an agent request.

        Args:
            latency_ms: Request processing time in milliseconds
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cache_hit: Whether the request was a cache hit
        """
        client = self._get_async_redis()
        timestamp = _get_unique_ts()

        if latency_ms is not None:
            await client.execute_command(  # type: ignore[no-untyped-call]
                "TS.ADD", self._keys["latency"], timestamp, latency_ms
            )

        if input_tokens is not None:
            await client.execute_command(  # type: ignore[no-untyped-call]
                "TS.ADD", self._keys["input_tokens"], timestamp, input_tokens
            )

        if output_tokens is not None:
            await client.execute_command(  # type: ignore[no-untyped-call]
                "TS.ADD", self._keys["output_tokens"], timestamp, output_tokens
            )

        if cache_hit is not None:
            await client.execute_command(  # type: ignore[no-untyped-call]
                "TS.ADD", self._keys["cache_hit"], timestamp, 1 if cache_hit else 0
            )

        await client.execute_command(  # type: ignore[no-untyped-call]
            "TS.ADD", self._keys["count"], timestamp, 1
        )
