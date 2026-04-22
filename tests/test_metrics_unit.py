"""Unit tests for AgentMetrics - TDD RED phase."""

from redis_openai_agents import AgentMetrics


class TestMetricsBasics:
    """Test basic metrics operations."""

    def test_create_metrics_collector(self, redis_url: str) -> None:
        """Can create a metrics collector."""
        metrics = AgentMetrics(
            name="test_agent",
            redis_url=redis_url,
        )
        assert metrics is not None
        assert metrics.name == "test_agent"

    def test_create_with_custom_retention(self, redis_url: str) -> None:
        """Can create with custom retention period."""
        metrics = AgentMetrics(
            name="test_retention",
            redis_url=redis_url,
            retention_ms=3600000,  # 1 hour
        )
        assert metrics is not None


class TestRecordMetrics:
    """Test recording metrics."""

    def test_record_latency(self, redis_url: str) -> None:
        """Can record latency metric."""
        metrics = AgentMetrics(
            name="test_latency",
            redis_url=redis_url,
        )

        metrics.record(latency_ms=150.5)

        stats = metrics.get_stats()
        assert stats["count"] >= 1

    def test_record_tokens(self, redis_url: str) -> None:
        """Can record token counts."""
        metrics = AgentMetrics(
            name="test_tokens",
            redis_url=redis_url,
        )

        metrics.record(input_tokens=100, output_tokens=200)

        stats = metrics.get_stats()
        assert stats["input_tokens_sum"] >= 100
        assert stats["output_tokens_sum"] >= 200

    def test_record_cache_hit(self, redis_url: str) -> None:
        """Can record cache hit."""
        metrics = AgentMetrics(
            name="test_cache",
            redis_url=redis_url,
        )

        metrics.record(cache_hit=True)
        metrics.record(cache_hit=False)

        stats = metrics.get_stats()
        assert "cache_hit_rate" in stats

    def test_record_multiple_metrics(self, redis_url: str) -> None:
        """Can record multiple metrics at once."""
        metrics = AgentMetrics(
            name="test_multi",
            redis_url=redis_url,
        )

        metrics.record(
            latency_ms=200.0,
            input_tokens=50,
            output_tokens=100,
            cache_hit=False,
        )

        stats = metrics.get_stats()
        assert stats["count"] >= 1


class TestGetStats:
    """Test getting aggregated statistics."""

    def test_get_latency_stats(self, redis_url: str) -> None:
        """Can get latency statistics."""
        metrics = AgentMetrics(
            name="test_latency_stats",
            redis_url=redis_url,
        )

        metrics.record(latency_ms=100.0)
        metrics.record(latency_ms=200.0)
        metrics.record(latency_ms=300.0)

        stats = metrics.get_stats()

        assert "latency_avg" in stats
        assert "latency_min" in stats
        assert "latency_max" in stats

    def test_stats_values_reasonable(self, redis_url: str) -> None:
        """Stats values are reasonable."""
        metrics = AgentMetrics(
            name="test_stats_values",
            redis_url=redis_url,
        )

        metrics.record(latency_ms=100.0)
        metrics.record(latency_ms=300.0)

        stats = metrics.get_stats()

        assert stats["latency_min"] <= stats["latency_avg"]
        assert stats["latency_avg"] <= stats["latency_max"]

    def test_cache_hit_rate_calculation(self, redis_url: str) -> None:
        """Cache hit rate is calculated correctly."""
        import uuid

        unique_name = f"test_hit_rate_{uuid.uuid4().hex[:8]}"
        metrics = AgentMetrics(
            name=unique_name,
            redis_url=redis_url,
        )

        # 3 hits, 1 miss = 75% hit rate
        metrics.record(cache_hit=True)
        metrics.record(cache_hit=True)
        metrics.record(cache_hit=True)
        metrics.record(cache_hit=False)

        stats = metrics.get_stats()

        assert 0.7 <= stats["cache_hit_rate"] <= 0.8

        # Clean up
        metrics.delete()


class TestRangeQuery:
    """Test time range queries."""

    def test_query_latency_range(self, redis_url: str) -> None:
        """Can query latency over time range."""
        import time

        metrics = AgentMetrics(
            name="test_range",
            redis_url=redis_url,
        )
        # Clean up any leftover data
        metrics.delete()

        metrics.record(latency_ms=100.0)
        metrics.record(latency_ms=200.0)

        # Query last minute
        from_time = int((time.time() - 60) * 1000)
        to_time = int(time.time() * 1000) + 1000

        data = metrics.range(
            metric="latency",
            from_time=from_time,
            to_time=to_time,
        )

        assert len(data) >= 2
        assert all(isinstance(point, tuple) for point in data)

    def test_range_returns_timestamp_value_pairs(self, redis_url: str) -> None:
        """Range query returns (timestamp, value) pairs."""
        import time

        metrics = AgentMetrics(
            name="test_range_format",
            redis_url=redis_url,
        )

        metrics.record(latency_ms=150.0)

        from_time = int((time.time() - 60) * 1000)
        to_time = int(time.time() * 1000) + 1000

        data = metrics.range(
            metric="latency",
            from_time=from_time,
            to_time=to_time,
        )

        assert len(data) >= 1
        ts, value = data[0]
        assert isinstance(ts, int)
        assert isinstance(value, float)


class TestMetricsDeletion:
    """Test metrics deletion."""

    def test_delete_metrics(self, redis_url: str) -> None:
        """Can delete all metrics."""
        metrics = AgentMetrics(
            name="test_delete",
            redis_url=redis_url,
        )

        metrics.record(latency_ms=100.0)
        stats_before = metrics.get_stats()
        assert stats_before["count"] >= 1

        metrics.delete()

        stats_after = metrics.get_stats()
        assert stats_after["count"] == 0
