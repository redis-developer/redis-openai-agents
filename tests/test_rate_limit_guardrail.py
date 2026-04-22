"""Unit tests for RedisRateLimitGuardrail."""

import time
from unittest.mock import AsyncMock, MagicMock

from redis_openai_agents.rate_limit_guardrail import (
    GuardrailFunctionOutput,
    RedisRateLimitGuardrail,
)


class TestGuardrailFunctionOutput:
    """Test GuardrailFunctionOutput dataclass."""

    def test_allowed_output(self) -> None:
        """GuardrailFunctionOutput for allowed request."""
        output = GuardrailFunctionOutput(
            output_info={"allowed": True},
            tripwire_triggered=False,
        )

        assert output.tripwire_triggered is False
        assert output.output_info["allowed"] is True

    def test_blocked_output(self) -> None:
        """GuardrailFunctionOutput for blocked request."""
        output = GuardrailFunctionOutput(
            output_info={"reason": "Rate limit exceeded"},
            tripwire_triggered=True,
        )

        assert output.tripwire_triggered is True
        assert "reason" in output.output_info


class TestRedisRateLimitGuardrailInit:
    """Test RedisRateLimitGuardrail initialization."""

    def test_default_values(self) -> None:
        """Guardrail has correct default values."""
        guardrail = RedisRateLimitGuardrail()

        assert guardrail._redis_url == "redis://localhost:6379"
        assert guardrail._key_prefix == "rate_limit"
        assert guardrail._requests_per_minute is None
        assert guardrail._tokens_per_minute is None
        assert guardrail._window_type == "sliding"
        assert guardrail._window_seconds == 60
        assert guardrail._client is None

    def test_custom_values(self) -> None:
        """Guardrail accepts custom configuration."""
        guardrail = RedisRateLimitGuardrail(
            redis_url="redis://custom:6380",
            key_prefix="custom_limit",
            requests_per_minute=100,
            tokens_per_minute=10000,
            window_type="fixed",
            window_seconds=30,
        )

        assert guardrail._redis_url == "redis://custom:6380"
        assert guardrail._key_prefix == "custom_limit"
        assert guardrail._requests_per_minute == 100
        assert guardrail._tokens_per_minute == 10000
        assert guardrail._window_type == "fixed"
        assert guardrail._window_seconds == 30


class TestRedisRateLimitGuardrailName:
    """Test guardrail name property."""

    def test_name_property(self) -> None:
        """Guardrail has name property."""
        guardrail = RedisRateLimitGuardrail()

        assert guardrail.name == "redis_rate_limit"


class TestRedisRateLimitGuardrailWindowKey:
    """Test window key generation."""

    def test_sliding_window_key(self) -> None:
        """Sliding window uses single key."""
        guardrail = RedisRateLimitGuardrail(
            key_prefix="test",
            window_type="sliding",
        )

        key = guardrail._get_window_key("user_123", "requests")

        assert key == "test:user_123:requests"

    def test_fixed_window_key_includes_bucket(self) -> None:
        """Fixed window key includes time bucket."""
        guardrail = RedisRateLimitGuardrail(
            key_prefix="test",
            window_type="fixed",
            window_seconds=60,
        )

        key = guardrail._get_window_key("user_123", "requests")

        # Key should have format: test:user_123:requests:bucket_id
        parts = key.split(":")
        assert len(parts) == 4
        assert parts[0] == "test"
        assert parts[1] == "user_123"
        assert parts[2] == "requests"
        # Bucket ID should be a number
        assert parts[3].isdigit()

    def test_token_key(self) -> None:
        """Token key generated correctly."""
        guardrail = RedisRateLimitGuardrail(
            key_prefix="test",
            window_type="sliding",
        )

        key = guardrail._get_window_key("user_123", "tokens")

        assert key == "test:user_123:tokens"


class TestRedisRateLimitGuardrailCheckRateLimit:
    """Test rate limit checking."""

    async def test_no_client_returns_allowed(self) -> None:
        """Returns allowed when Redis not initialized."""
        guardrail = RedisRateLimitGuardrail(
            requests_per_minute=10,
        )

        result = await guardrail.check_rate_limit(user_id="user_123")

        assert result.tripwire_triggered is False
        assert "error" in result.output_info

    async def test_no_limits_configured(self) -> None:
        """Returns allowed when no limits configured."""
        guardrail = RedisRateLimitGuardrail()

        mock_client = AsyncMock()
        guardrail._client = mock_client

        result = await guardrail.check_rate_limit(user_id="user_123")

        assert result.tripwire_triggered is False

    async def test_sliding_window_under_limit(self) -> None:
        """Request allowed under limit with sliding window."""
        guardrail = RedisRateLimitGuardrail(
            requests_per_minute=10,
            window_type="sliding",
        )

        mock_client = AsyncMock()
        mock_client.zremrangebyscore.return_value = 0
        mock_client.zcard.return_value = 5  # Under limit
        mock_client.zadd.return_value = 1
        mock_client.expire.return_value = True
        guardrail._client = mock_client

        result = await guardrail.check_rate_limit(user_id="user_123")

        assert result.tripwire_triggered is False
        mock_client.zadd.assert_called_once()

    async def test_sliding_window_at_limit(self) -> None:
        """Request blocked at limit with sliding window."""
        guardrail = RedisRateLimitGuardrail(
            requests_per_minute=10,
            window_type="sliding",
        )

        mock_client = AsyncMock()
        mock_client.zremrangebyscore.return_value = 0
        mock_client.zcard.return_value = 10  # At limit
        guardrail._client = mock_client

        result = await guardrail.check_rate_limit(user_id="user_123")

        assert result.tripwire_triggered is True
        assert "exceeded" in result.output_info["reason"].lower()

    async def test_fixed_window_under_limit(self) -> None:
        """Request allowed under limit with fixed window."""
        guardrail = RedisRateLimitGuardrail(
            requests_per_minute=10,
            window_type="fixed",
        )

        mock_client = AsyncMock()
        mock_client.incr.return_value = 5  # Under limit
        mock_client.expire.return_value = True
        guardrail._client = mock_client

        result = await guardrail.check_rate_limit(user_id="user_123")

        assert result.tripwire_triggered is False

    async def test_fixed_window_over_limit(self) -> None:
        """Request blocked over limit with fixed window."""
        guardrail = RedisRateLimitGuardrail(
            requests_per_minute=10,
            window_type="fixed",
        )

        mock_client = AsyncMock()
        mock_client.incr.return_value = 11  # Over limit
        guardrail._client = mock_client

        result = await guardrail.check_rate_limit(user_id="user_123")

        assert result.tripwire_triggered is True

    async def test_token_limit_under(self) -> None:
        """Request allowed under token limit."""
        guardrail = RedisRateLimitGuardrail(
            tokens_per_minute=1000,
            window_type="sliding",
        )

        mock_client = AsyncMock()
        mock_client.zremrangebyscore.return_value = 0
        mock_client.zrange.return_value = [("500:1234.5", 1234.5)]  # 500 tokens used
        mock_client.zadd.return_value = 1
        mock_client.expire.return_value = True
        guardrail._client = mock_client

        result = await guardrail.check_rate_limit(
            user_id="user_123",
            tokens_used=200,  # Total would be 700, under 1000
        )

        assert result.tripwire_triggered is False

    async def test_token_limit_over(self) -> None:
        """Request blocked over token limit."""
        guardrail = RedisRateLimitGuardrail(
            tokens_per_minute=1000,
            window_type="sliding",
        )

        mock_client = AsyncMock()
        mock_client.zremrangebyscore.return_value = 0
        mock_client.zrange.return_value = [("800:1234.5", 1234.5)]  # 800 tokens used
        guardrail._client = mock_client

        result = await guardrail.check_rate_limit(
            user_id="user_123",
            tokens_used=300,  # Total would be 1100, over 1000
        )

        assert result.tripwire_triggered is True


class TestRedisRateLimitGuardrailGetInfo:
    """Test rate limit info retrieval."""

    async def test_no_client_returns_error(self) -> None:
        """Returns error when Redis not initialized."""
        guardrail = RedisRateLimitGuardrail()

        info = await guardrail.get_rate_limit_info("user_123")

        assert "error" in info

    async def test_returns_remaining_requests(self) -> None:
        """Returns remaining requests info."""
        guardrail = RedisRateLimitGuardrail(
            requests_per_minute=10,
            window_type="fixed",
        )

        mock_client = AsyncMock()
        mock_client.get.return_value = "3"  # 3 requests made
        guardrail._client = mock_client

        info = await guardrail.get_rate_limit_info("user_123")

        assert info["current_requests"] == 3
        assert info["remaining_requests"] == 7
        assert info["requests_limit"] == 10

    async def test_returns_reset_time(self) -> None:
        """Returns reset time info."""
        guardrail = RedisRateLimitGuardrail(
            requests_per_minute=10,
            window_type="fixed",
            window_seconds=60,
        )

        mock_client = AsyncMock()
        mock_client.get.return_value = "5"
        guardrail._client = mock_client

        info = await guardrail.get_rate_limit_info("user_123")

        assert "reset_at" in info
        assert info["reset_at"] > time.time()


class TestRedisRateLimitGuardrailGuardrailFunction:
    """Test SDK guardrail function integration."""

    async def test_extracts_user_id_from_context(self) -> None:
        """Extracts user_id from context."""
        guardrail = RedisRateLimitGuardrail(
            requests_per_minute=10,
        )

        mock_client = AsyncMock()
        mock_client.zremrangebyscore.return_value = 0
        mock_client.zcard.return_value = 0
        mock_client.zadd.return_value = 1
        mock_client.expire.return_value = True
        guardrail._client = mock_client

        mock_context = MagicMock()
        mock_context.context.user_id = "extracted_user"

        result = await guardrail.guardrail_function(
            context=mock_context,
            agent=MagicMock(),
            input_data="Hello",
        )

        assert result.tripwire_triggered is False

    async def test_extracts_user_id_from_dict_context(self) -> None:
        """Extracts user_id from dict context."""
        guardrail = RedisRateLimitGuardrail(
            requests_per_minute=10,
        )

        mock_client = AsyncMock()
        mock_client.zremrangebyscore.return_value = 0
        mock_client.zcard.return_value = 0
        mock_client.zadd.return_value = 1
        mock_client.expire.return_value = True
        guardrail._client = mock_client

        mock_context = MagicMock()
        mock_context.context = {"user_id": "dict_user"}

        result = await guardrail.guardrail_function(
            context=mock_context,
            agent=MagicMock(),
            input_data="Hello",
        )

        assert result.tripwire_triggered is False

    async def test_uses_default_user_id(self) -> None:
        """Uses default user_id when not in context."""
        guardrail = RedisRateLimitGuardrail(
            requests_per_minute=10,
        )

        mock_client = AsyncMock()
        mock_client.zremrangebyscore.return_value = 0
        mock_client.zcard.return_value = 0
        mock_client.zadd.return_value = 1
        mock_client.expire.return_value = True
        guardrail._client = mock_client

        mock_context = MagicMock()
        mock_context.context = {}

        result = await guardrail.guardrail_function(
            context=mock_context,
            agent=MagicMock(),
            input_data="Hello",
        )

        assert result.tripwire_triggered is False


class TestRedisRateLimitGuardrailLifecycle:
    """Test guardrail lifecycle methods."""

    async def test_initialize_creates_client(self) -> None:
        """Initialize creates Redis client."""
        guardrail = RedisRateLimitGuardrail(
            redis_url="redis://nonexistent:9999",
        )

        # Create client but don't connect yet (lazy connection)
        await guardrail.initialize()

        # Client should be created
        assert guardrail._client is not None

        await guardrail.close()

    async def test_close_clears_client(self) -> None:
        """Close clears client reference."""
        guardrail = RedisRateLimitGuardrail()

        mock_client = AsyncMock()
        guardrail._client = mock_client

        await guardrail.close()

        assert guardrail._client is None
        mock_client.aclose.assert_called_once()

    async def test_close_when_no_client(self) -> None:
        """Close handles no client gracefully."""
        guardrail = RedisRateLimitGuardrail()

        # Should not raise
        await guardrail.close()
