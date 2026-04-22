"""Integration tests for RedisRateLimitGuardrail - TDD RED phase.

Tests the Redis-backed rate limiting guardrail for OpenAI Agents SDK.
"""

import time


class TestRedisRateLimitGuardrailRateLimiting:
    """Test rate limiting functionality."""

    async def test_allows_requests_under_limit(self, redis_url: str) -> None:
        """Requests under rate limit are allowed."""
        from redis_openai_agents import RedisRateLimitGuardrail

        guardrail = RedisRateLimitGuardrail(
            redis_url=redis_url,
            requests_per_minute=10,
            key_prefix="test_under_limit",
        )
        await guardrail.initialize()

        # Make a few requests - should all pass
        for i in range(5):
            result = await guardrail.check_rate_limit(user_id=f"user_{i}")
            assert result.tripwire_triggered is False

        await guardrail.close()

    async def test_blocks_requests_over_limit(self, redis_url: str) -> None:
        """Requests over rate limit are blocked."""
        from redis_openai_agents import RedisRateLimitGuardrail

        guardrail = RedisRateLimitGuardrail(
            redis_url=redis_url,
            requests_per_minute=3,
            key_prefix="test_over_limit",
        )
        await guardrail.initialize()

        user_id = "user_rate_limited"

        # Make requests up to limit
        for _ in range(3):
            result = await guardrail.check_rate_limit(user_id=user_id)
            assert result.tripwire_triggered is False

        # Next request should be blocked
        result = await guardrail.check_rate_limit(user_id=user_id)
        assert result.tripwire_triggered is True

        await guardrail.close()

    async def test_different_users_have_separate_limits(self, redis_url: str) -> None:
        """Each user has their own rate limit counter."""
        from redis_openai_agents import RedisRateLimitGuardrail

        guardrail = RedisRateLimitGuardrail(
            redis_url=redis_url,
            requests_per_minute=2,
            key_prefix="test_separate",
        )
        await guardrail.initialize()

        # User A makes 2 requests (at limit)
        for _ in range(2):
            result = await guardrail.check_rate_limit(user_id="user_a")
            assert result.tripwire_triggered is False

        # User A's third request blocked
        result = await guardrail.check_rate_limit(user_id="user_a")
        assert result.tripwire_triggered is True

        # User B should still be allowed
        result = await guardrail.check_rate_limit(user_id="user_b")
        assert result.tripwire_triggered is False

        await guardrail.close()


class TestRedisRateLimitGuardrailWindowTypes:
    """Test different rate limiting window types."""

    async def test_sliding_window(self, redis_url: str) -> None:
        """Sliding window rate limiting."""
        from redis_openai_agents import RedisRateLimitGuardrail

        guardrail = RedisRateLimitGuardrail(
            redis_url=redis_url,
            requests_per_minute=5,
            window_type="sliding",
            key_prefix="test_sliding",
        )
        await guardrail.initialize()

        user_id = "sliding_user"

        # Make 5 requests
        for _ in range(5):
            result = await guardrail.check_rate_limit(user_id=user_id)
            assert result.tripwire_triggered is False

        # 6th request blocked
        result = await guardrail.check_rate_limit(user_id=user_id)
        assert result.tripwire_triggered is True

        await guardrail.close()

    async def test_fixed_window(self, redis_url: str) -> None:
        """Fixed window rate limiting."""
        from redis_openai_agents import RedisRateLimitGuardrail

        guardrail = RedisRateLimitGuardrail(
            redis_url=redis_url,
            requests_per_minute=3,
            window_type="fixed",
            key_prefix="test_fixed",
        )
        await guardrail.initialize()

        user_id = "fixed_user"

        # Make requests
        for _ in range(3):
            await guardrail.check_rate_limit(user_id=user_id)

        # Should be blocked
        result = await guardrail.check_rate_limit(user_id=user_id)
        assert result.tripwire_triggered is True

        await guardrail.close()


class TestRedisRateLimitGuardrailTokenLimiting:
    """Test token-based rate limiting."""

    async def test_token_rate_limiting(self, redis_url: str) -> None:
        """Can limit by tokens per minute."""
        from redis_openai_agents import RedisRateLimitGuardrail

        guardrail = RedisRateLimitGuardrail(
            redis_url=redis_url,
            tokens_per_minute=1000,
            key_prefix="test_tokens",
        )
        await guardrail.initialize()

        user_id = "token_user"

        # Use some tokens
        result = await guardrail.check_rate_limit(
            user_id=user_id,
            tokens_used=500,
        )
        assert result.tripwire_triggered is False

        # Use more tokens (total 900)
        result = await guardrail.check_rate_limit(
            user_id=user_id,
            tokens_used=400,
        )
        assert result.tripwire_triggered is False

        # This would exceed (total 1100)
        result = await guardrail.check_rate_limit(
            user_id=user_id,
            tokens_used=200,
        )
        assert result.tripwire_triggered is True

        await guardrail.close()


class TestRedisRateLimitGuardrailCombinedLimits:
    """Test combined request and token limits."""

    async def test_combined_limits(self, redis_url: str) -> None:
        """Can enforce both request and token limits."""
        from redis_openai_agents import RedisRateLimitGuardrail

        guardrail = RedisRateLimitGuardrail(
            redis_url=redis_url,
            requests_per_minute=10,
            tokens_per_minute=500,
            key_prefix="test_combined",
        )
        await guardrail.initialize()

        user_id = "combined_user"

        # Make requests with high token usage
        result = await guardrail.check_rate_limit(
            user_id=user_id,
            tokens_used=200,
        )
        assert result.tripwire_triggered is False

        result = await guardrail.check_rate_limit(
            user_id=user_id,
            tokens_used=200,
        )
        assert result.tripwire_triggered is False

        # Token limit hit before request limit
        result = await guardrail.check_rate_limit(
            user_id=user_id,
            tokens_used=200,
        )
        assert result.tripwire_triggered is True

        await guardrail.close()


class TestRedisRateLimitGuardrailInfo:
    """Test rate limit info retrieval."""

    async def test_get_remaining_requests(self, redis_url: str) -> None:
        """Can get remaining requests."""
        from redis_openai_agents import RedisRateLimitGuardrail

        guardrail = RedisRateLimitGuardrail(
            redis_url=redis_url,
            requests_per_minute=10,
            key_prefix="test_remaining",
        )
        await guardrail.initialize()

        user_id = "remaining_user"

        # Check initial remaining
        info = await guardrail.get_rate_limit_info(user_id)
        assert info["remaining_requests"] == 10

        # Make a request
        await guardrail.check_rate_limit(user_id=user_id)

        # Check remaining after request
        info = await guardrail.get_rate_limit_info(user_id)
        assert info["remaining_requests"] == 9

        await guardrail.close()

    async def test_get_reset_time(self, redis_url: str) -> None:
        """Can get time until rate limit reset."""
        from redis_openai_agents import RedisRateLimitGuardrail

        guardrail = RedisRateLimitGuardrail(
            redis_url=redis_url,
            requests_per_minute=5,
            key_prefix="test_reset",
        )
        await guardrail.initialize()

        user_id = "reset_user"

        # Make a request to start the window
        await guardrail.check_rate_limit(user_id=user_id)

        info = await guardrail.get_rate_limit_info(user_id)
        assert "reset_at" in info
        assert info["reset_at"] > time.time()

        await guardrail.close()


class TestRedisRateLimitGuardrailSDKIntegration:
    """Test integration with OpenAI Agents SDK guardrail interface."""

    async def test_returns_guardrail_output(self, redis_url: str) -> None:
        """Returns GuardrailFunctionOutput compatible result."""
        from redis_openai_agents import RedisRateLimitGuardrail

        guardrail = RedisRateLimitGuardrail(
            redis_url=redis_url,
            requests_per_minute=10,
            key_prefix="test_output",
        )
        await guardrail.initialize()

        result = await guardrail.check_rate_limit(user_id="sdk_user")

        # Should have GuardrailFunctionOutput structure
        assert hasattr(result, "tripwire_triggered")
        assert hasattr(result, "output_info")
        assert isinstance(result.tripwire_triggered, bool)

        await guardrail.close()

    async def test_output_info_contains_details(self, redis_url: str) -> None:
        """Output info contains rate limit details."""
        from redis_openai_agents import RedisRateLimitGuardrail

        guardrail = RedisRateLimitGuardrail(
            redis_url=redis_url,
            requests_per_minute=5,
            key_prefix="test_details",
        )
        await guardrail.initialize()

        # Make requests to hit limit
        for _ in range(5):
            await guardrail.check_rate_limit(user_id="detail_user")

        result = await guardrail.check_rate_limit(user_id="detail_user")

        assert result.tripwire_triggered is True
        # Output info should explain why blocked
        assert result.output_info is not None
        info = result.output_info
        assert "limit" in str(info).lower() or "exceeded" in str(info).lower()

        await guardrail.close()
