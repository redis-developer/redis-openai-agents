"""Integration tests for RankedOperations.

These tests run against a real Redis instance and verify:
- Agent performance ranking using sorted sets
- Session LRU tracking
- Token budget rate limiting
"""

import time

import pytest


class TestAgentPerformanceRanking:
    """Tests for agent performance ranking."""

    @pytest.mark.asyncio
    async def test_record_agent_success(self, redis_url: str) -> None:
        """Should record successful agent execution."""
        from redis_openai_agents.ranking import RankedOperations

        ranking = RankedOperations(redis_url=redis_url)
        await ranking.initialize()

        await ranking.record_agent_success(
            agent_id="agent_1",
            task_type="research",
            success=True,
            latency_ms=150.0,
        )

        # Agent should now appear in rankings
        top_agents = await ranking.get_best_agents("research", limit=5)

        assert len(top_agents) == 1
        assert top_agents[0][0] == "agent_1"
        assert top_agents[0][1] > 0  # Has a positive score

        await ranking.close()

    @pytest.mark.asyncio
    async def test_multiple_agents_ranked_correctly(self, redis_url: str) -> None:
        """Agents with better performance should rank higher."""
        from redis_openai_agents.ranking import RankedOperations

        ranking = RankedOperations(redis_url=redis_url)
        await ranking.initialize()

        # Agent 1: 80% success, fast
        for _ in range(8):
            await ranking.record_agent_success("agent_1", "analysis", True, 100.0)
        for _ in range(2):
            await ranking.record_agent_success("agent_1", "analysis", False, 100.0)

        # Agent 2: 60% success, slower
        for _ in range(6):
            await ranking.record_agent_success("agent_2", "analysis", True, 500.0)
        for _ in range(4):
            await ranking.record_agent_success("agent_2", "analysis", False, 500.0)

        # Agent 3: 100% success but VERY slow (penalty should outweigh perfect success)
        for _ in range(5):
            await ranking.record_agent_success("agent_3", "analysis", True, 8000.0)

        top_agents = await ranking.get_best_agents("analysis", limit=3)

        # Agent 1 should rank highest (good success + fast)
        assert top_agents[0][0] == "agent_1"
        # Scores should be in descending order
        assert top_agents[0][1] >= top_agents[1][1]
        assert top_agents[1][1] >= top_agents[2][1]

        await ranking.close()

    @pytest.mark.asyncio
    async def test_separate_rankings_per_task_type(self, redis_url: str) -> None:
        """Different task types should have separate rankings."""
        from redis_openai_agents.ranking import RankedOperations

        ranking = RankedOperations(redis_url=redis_url)
        await ranking.initialize()

        # Agent 1 is good at research
        await ranking.record_agent_success("agent_1", "research", True, 100.0)
        await ranking.record_agent_success("agent_1", "research", True, 100.0)

        # Agent 2 is good at analysis
        await ranking.record_agent_success("agent_2", "analysis", True, 100.0)
        await ranking.record_agent_success("agent_2", "analysis", True, 100.0)

        research_top = await ranking.get_best_agents("research", limit=5)
        analysis_top = await ranking.get_best_agents("analysis", limit=5)

        # Different agents should lead each category
        assert len(research_top) == 1
        assert research_top[0][0] == "agent_1"
        assert len(analysis_top) == 1
        assert analysis_top[0][0] == "agent_2"

        await ranking.close()


class TestSessionLRUTracking:
    """Tests for session LRU tracking."""

    @pytest.mark.asyncio
    async def test_touch_session(self, redis_url: str) -> None:
        """Should mark session as recently used."""
        from redis_openai_agents.ranking import RankedOperations

        ranking = RankedOperations(redis_url=redis_url)
        await ranking.initialize()

        await ranking.touch_session("session_1")

        # Session should not be stale (just touched)
        stale = await ranking.get_stale_sessions(max_age_seconds=3600)
        assert "session_1" not in stale

        await ranking.close()

    @pytest.mark.asyncio
    async def test_get_stale_sessions(self, redis_url: str) -> None:
        """Should return sessions not accessed within max_age."""
        from redis_openai_agents.ranking import RankedOperations

        ranking = RankedOperations(redis_url=redis_url)
        await ranking.initialize()

        # Touch sessions with different "ages"
        current_time = time.time()

        # Manually add sessions with old timestamps for testing
        from redis.asyncio import Redis

        client = Redis.from_url(redis_url, decode_responses=True)

        key = "rank:session_lru"
        # Old session (1 hour ago)
        await client.zadd(key, {"old_session": current_time - 3700})
        # Recent session
        await client.zadd(key, {"recent_session": current_time})

        stale = await ranking.get_stale_sessions(max_age_seconds=3600, limit=100)

        assert "old_session" in stale
        assert "recent_session" not in stale

        await client.aclose()
        await ranking.close()

    @pytest.mark.asyncio
    async def test_evict_stale_sessions(self, redis_url: str) -> None:
        """Should evict sessions not accessed within max_age."""
        from redis.asyncio import Redis

        from redis_openai_agents.ranking import RankedOperations

        ranking = RankedOperations(redis_url=redis_url)
        await ranking.initialize()

        client = Redis.from_url(redis_url, decode_responses=True)

        # Create some session data
        await client.hset("session:stale_1", mapping={"data": "test"})
        await client.hset("session:stale_2", mapping={"data": "test"})

        # Add to LRU with old timestamps
        current_time = time.time()
        key = "rank:session_lru"
        await client.zadd(
            key,
            {
                "stale_1": current_time - 100000,
                "stale_2": current_time - 100000,
                "recent": current_time,
            },
        )

        evicted = await ranking.evict_stale_sessions(max_age_seconds=3600, batch_size=100)

        assert evicted == 2

        # Stale sessions should be removed
        assert not await client.exists("session:stale_1")
        assert not await client.exists("session:stale_2")

        # Recent session should still be tracked
        stale = await ranking.get_stale_sessions(max_age_seconds=3600)
        assert "recent" not in stale

        await client.aclose()
        await ranking.close()


class TestTokenBudgetRateLimiting:
    """Tests for token budget rate limiting."""

    @pytest.mark.asyncio
    async def test_check_token_budget_allowed(self, redis_url: str) -> None:
        """Should allow tokens within budget."""
        from redis_openai_agents.ranking import RankedOperations

        ranking = RankedOperations(redis_url=redis_url)
        await ranking.initialize()

        allowed, remaining = await ranking.check_token_budget(
            user_id="user_1",
            tokens_needed=1000,
            budget_per_hour=10000,
        )

        assert allowed is True
        assert remaining == 9000

        await ranking.close()

    @pytest.mark.asyncio
    async def test_check_token_budget_exceeded(self, redis_url: str) -> None:
        """Should deny tokens exceeding budget."""
        from redis_openai_agents.ranking import RankedOperations

        ranking = RankedOperations(redis_url=redis_url)
        await ranking.initialize()

        # Use most of budget
        await ranking.check_token_budget("user_2", 9500, budget_per_hour=10000)

        # Try to use more than remaining
        allowed, remaining = await ranking.check_token_budget(
            user_id="user_2",
            tokens_needed=1000,
            budget_per_hour=10000,
        )

        assert allowed is False
        assert remaining == 500

        await ranking.close()

    @pytest.mark.asyncio
    async def test_token_budget_accumulates(self, redis_url: str) -> None:
        """Token usage should accumulate correctly."""
        from redis_openai_agents.ranking import RankedOperations

        ranking = RankedOperations(redis_url=redis_url)
        await ranking.initialize()

        # Multiple requests
        await ranking.check_token_budget("user_3", 1000, budget_per_hour=10000)
        await ranking.check_token_budget("user_3", 2000, budget_per_hour=10000)
        await ranking.check_token_budget("user_3", 3000, budget_per_hour=10000)

        allowed, remaining = await ranking.check_token_budget(
            user_id="user_3",
            tokens_needed=1000,
            budget_per_hour=10000,
        )

        assert allowed is True
        assert remaining == 3000  # 10000 - 1000 - 2000 - 3000 - 1000 = 3000

        await ranking.close()

    @pytest.mark.asyncio
    async def test_get_token_usage(self, redis_url: str) -> None:
        """Should return token usage statistics."""
        from redis_openai_agents.ranking import RankedOperations

        ranking = RankedOperations(redis_url=redis_url)
        await ranking.initialize()

        # Use some tokens
        await ranking.check_token_budget("user_4", 5000, budget_per_hour=10000)

        usage = await ranking.get_token_usage("user_4")

        assert usage["current_hour"] == 5000
        assert "remaining" in usage

        await ranking.close()


class TestToolEffectivenessRanking:
    """Tests for tool effectiveness ranking."""

    @pytest.mark.asyncio
    async def test_record_tool_success(self, redis_url: str) -> None:
        """Should record tool execution success."""
        from redis_openai_agents.ranking import RankedOperations

        ranking = RankedOperations(redis_url=redis_url)
        await ranking.initialize()

        await ranking.record_tool_success(
            tool_name="web_search",
            success=True,
            latency_ms=200.0,
        )

        top_tools = await ranking.get_best_tools(limit=5)

        assert len(top_tools) == 1
        assert top_tools[0][0] == "web_search"

        await ranking.close()

    @pytest.mark.asyncio
    async def test_tools_ranked_by_effectiveness(self, redis_url: str) -> None:
        """Tools should be ranked by success rate and latency."""
        from redis_openai_agents.ranking import RankedOperations

        ranking = RankedOperations(redis_url=redis_url)
        await ranking.initialize()

        # Tool 1: 90% success, fast
        for _ in range(9):
            await ranking.record_tool_success("tool_1", True, 100.0)
        await ranking.record_tool_success("tool_1", False, 100.0)

        # Tool 2: 50% success, slow
        for _ in range(5):
            await ranking.record_tool_success("tool_2", True, 1000.0)
        for _ in range(5):
            await ranking.record_tool_success("tool_2", False, 1000.0)

        top_tools = await ranking.get_best_tools(limit=2)

        assert top_tools[0][0] == "tool_1"
        assert top_tools[0][1] > top_tools[1][1]

        await ranking.close()
