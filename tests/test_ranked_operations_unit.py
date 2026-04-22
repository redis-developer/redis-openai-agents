"""Unit tests for RankedOperations.

These tests use mocks to verify behavior without Redis.
"""

from unittest.mock import AsyncMock, patch

import pytest


class TestRankedOperationsInit:
    """Tests for RankedOperations initialization."""

    def test_init_sets_redis_url(self) -> None:
        """Should set Redis URL."""
        from redis_openai_agents.ranking import RankedOperations

        ranking = RankedOperations(redis_url="redis://localhost:6379")

        assert ranking._redis_url == "redis://localhost:6379"

    def test_init_sets_prefix(self) -> None:
        """Should set prefix."""
        from redis_openai_agents.ranking import RankedOperations

        ranking = RankedOperations(
            redis_url="redis://localhost:6379",
            prefix="custom_prefix",
        )

        assert ranking._prefix == "custom_prefix"

    def test_init_default_prefix(self) -> None:
        """Should use default prefix 'rank'."""
        from redis_openai_agents.ranking import RankedOperations

        ranking = RankedOperations(redis_url="redis://localhost:6379")

        assert ranking._prefix == "rank"


class TestAgentPerformanceRankingUnit:
    """Unit tests for agent performance ranking."""

    @pytest.mark.asyncio
    async def test_record_agent_success_calls_redis(self) -> None:
        """Should call Redis commands for agent success recording."""
        from redis_openai_agents.ranking import RankedOperations

        mock_client = AsyncMock()
        mock_client.incr = AsyncMock()
        mock_client.incrbyfloat = AsyncMock()
        mock_client.get = AsyncMock(side_effect=["1", "1", "100"])
        mock_client.zadd = AsyncMock()

        with patch("redis_openai_agents.ranking.aioredis") as mock_aioredis:
            mock_aioredis.from_url.return_value = mock_client

            ranking = RankedOperations(redis_url="redis://localhost:6379")
            await ranking.initialize()

            await ranking.record_agent_success(
                agent_id="agent_1",
                task_type="research",
                success=True,
                latency_ms=100.0,
            )

            # Should increment counters
            mock_client.incr.assert_called()
            mock_client.incrbyfloat.assert_called()
            # Should update sorted set
            mock_client.zadd.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_best_agents_calls_zrevrange(self) -> None:
        """Should call ZREVRANGE to get top agents."""
        from redis_openai_agents.ranking import RankedOperations

        mock_client = AsyncMock()
        mock_client.zrevrange = AsyncMock(
            return_value=[
                ("agent_1", 0.9),
                ("agent_2", 0.7),
            ]
        )

        with patch("redis_openai_agents.ranking.aioredis") as mock_aioredis:
            mock_aioredis.from_url.return_value = mock_client

            ranking = RankedOperations(redis_url="redis://localhost:6379")
            await ranking.initialize()

            agents = await ranking.get_best_agents("research", limit=5)

            mock_client.zrevrange.assert_called_once()
            assert len(agents) == 2
            assert agents[0][0] == "agent_1"


class TestSessionLRUTrackingUnit:
    """Unit tests for session LRU tracking."""

    @pytest.mark.asyncio
    async def test_touch_session_calls_zadd(self) -> None:
        """Should call ZADD with current timestamp."""
        from redis_openai_agents.ranking import RankedOperations

        mock_client = AsyncMock()
        mock_client.zadd = AsyncMock()

        with patch("redis_openai_agents.ranking.aioredis") as mock_aioredis:
            mock_aioredis.from_url.return_value = mock_client

            ranking = RankedOperations(redis_url="redis://localhost:6379")
            await ranking.initialize()

            await ranking.touch_session("session_1")

            mock_client.zadd.assert_called_once()
            call_args = mock_client.zadd.call_args
            assert "rank:session_lru" == call_args[0][0]
            assert "session_1" in call_args[0][1]

    @pytest.mark.asyncio
    async def test_get_stale_sessions_calls_zrangebyscore(self) -> None:
        """Should call ZRANGEBYSCORE with cutoff time."""
        from redis_openai_agents.ranking import RankedOperations

        mock_client = AsyncMock()
        mock_client.zrangebyscore = AsyncMock(return_value=["old_session"])

        with patch("redis_openai_agents.ranking.aioredis") as mock_aioredis:
            mock_aioredis.from_url.return_value = mock_client

            ranking = RankedOperations(redis_url="redis://localhost:6379")
            await ranking.initialize()

            stale = await ranking.get_stale_sessions(max_age_seconds=3600)

            mock_client.zrangebyscore.assert_called_once()
            assert "old_session" in stale

    @pytest.mark.asyncio
    async def test_evict_stale_sessions_deletes_data(self) -> None:
        """Should delete session data and remove from LRU."""
        from redis_openai_agents.ranking import RankedOperations

        mock_client = AsyncMock()
        mock_client.zrangebyscore = AsyncMock(return_value=["stale_1", "stale_2"])
        mock_client.delete = AsyncMock()
        mock_client.zrem = AsyncMock()

        with patch("redis_openai_agents.ranking.aioredis") as mock_aioredis:
            mock_aioredis.from_url.return_value = mock_client

            ranking = RankedOperations(redis_url="redis://localhost:6379")
            await ranking.initialize()

            count = await ranking.evict_stale_sessions(max_age_seconds=3600)

            assert count == 2
            mock_client.delete.assert_called_once()
            mock_client.zrem.assert_called_once()


class TestTokenBudgetRateLimitingUnit:
    """Unit tests for token budget rate limiting."""

    @pytest.mark.asyncio
    async def test_check_token_budget_allowed(self) -> None:
        """Should allow when within budget."""
        from redis_openai_agents.ranking import RankedOperations

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=None)  # No usage yet
        mock_client.incrby = AsyncMock(return_value=1000)
        mock_client.expire = AsyncMock()

        with patch("redis_openai_agents.ranking.aioredis") as mock_aioredis:
            mock_aioredis.from_url.return_value = mock_client

            ranking = RankedOperations(redis_url="redis://localhost:6379")
            await ranking.initialize()

            allowed, remaining = await ranking.check_token_budget(
                user_id="user_1",
                tokens_needed=1000,
                budget_per_hour=10000,
            )

            assert allowed is True
            assert remaining == 9000

    @pytest.mark.asyncio
    async def test_check_token_budget_denied(self) -> None:
        """Should deny when budget exceeded."""
        from redis_openai_agents.ranking import RankedOperations

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value="9500")  # Most budget used
        mock_client.incrby = AsyncMock()
        mock_client.expire = AsyncMock()

        with patch("redis_openai_agents.ranking.aioredis") as mock_aioredis:
            mock_aioredis.from_url.return_value = mock_client

            ranking = RankedOperations(redis_url="redis://localhost:6379")
            await ranking.initialize()

            allowed, remaining = await ranking.check_token_budget(
                user_id="user_1",
                tokens_needed=1000,  # Need 1000, only 500 remaining
                budget_per_hour=10000,
            )

            assert allowed is False
            assert remaining == 500
            mock_client.incrby.assert_not_called()


class TestToolEffectivenessRankingUnit:
    """Unit tests for tool effectiveness ranking."""

    @pytest.mark.asyncio
    async def test_record_tool_success_calls_redis(self) -> None:
        """Should call Redis commands for tool success recording."""
        from redis_openai_agents.ranking import RankedOperations

        mock_client = AsyncMock()
        mock_client.incr = AsyncMock()
        mock_client.incrbyfloat = AsyncMock()
        mock_client.get = AsyncMock(side_effect=["1", "1", "100"])
        mock_client.zadd = AsyncMock()

        with patch("redis_openai_agents.ranking.aioredis") as mock_aioredis:
            mock_aioredis.from_url.return_value = mock_client

            ranking = RankedOperations(redis_url="redis://localhost:6379")
            await ranking.initialize()

            await ranking.record_tool_success(
                tool_name="web_search",
                success=True,
                latency_ms=200.0,
            )

            mock_client.zadd.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_best_tools_calls_zrevrange(self) -> None:
        """Should call ZREVRANGE to get top tools."""
        from redis_openai_agents.ranking import RankedOperations

        mock_client = AsyncMock()
        mock_client.zrevrange = AsyncMock(
            return_value=[
                ("tool_1", 0.95),
                ("tool_2", 0.80),
            ]
        )

        with patch("redis_openai_agents.ranking.aioredis") as mock_aioredis:
            mock_aioredis.from_url.return_value = mock_client

            ranking = RankedOperations(redis_url="redis://localhost:6379")
            await ranking.initialize()

            tools = await ranking.get_best_tools(limit=5)

            mock_client.zrevrange.assert_called_once()
            assert len(tools) == 2
            assert tools[0][0] == "tool_1"
