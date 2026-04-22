"""Unit tests for retry/backoff functionality."""

from __future__ import annotations

import pytest
from redis.exceptions import ConnectionError, TimeoutError


class TestRetryDecorator:
    """Tests for @with_retry decorator."""

    def test_success_no_retry(self) -> None:
        """Successful call should not retry."""
        from redis_openai_agents.retry import with_retry

        call_count = 0

        @with_retry(max_retries=3)
        def successful_fn() -> str:
            nonlocal call_count
            call_count += 1
            return "success"

        result = successful_fn()

        assert result == "success"
        assert call_count == 1

    def test_retry_on_connection_error(self) -> None:
        """Should retry on ConnectionError."""
        from redis_openai_agents.retry import with_retry

        call_count = 0

        @with_retry(max_retries=3, base_delay=0.01)
        def failing_then_success() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Connection lost")
            return "success"

        result = failing_then_success()

        assert result == "success"
        assert call_count == 3

    def test_retry_on_timeout_error(self) -> None:
        """Should retry on TimeoutError."""
        from redis_openai_agents.retry import with_retry

        call_count = 0

        @with_retry(max_retries=3, base_delay=0.01)
        def timeout_then_success() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise TimeoutError("Timeout")
            return "success"

        result = timeout_then_success()

        assert result == "success"
        assert call_count == 2

    def test_max_retries_exceeded(self) -> None:
        """Should raise after max retries exceeded."""
        from redis_openai_agents.retry import with_retry

        call_count = 0

        @with_retry(max_retries=2, base_delay=0.01)
        def always_fails() -> str:
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Always fails")

        with pytest.raises(ConnectionError):
            always_fails()

        assert call_count == 3  # Initial + 2 retries

    def test_no_retry_on_other_exceptions(self) -> None:
        """Should not retry on non-retryable exceptions."""
        from redis_openai_agents.retry import with_retry

        call_count = 0

        @with_retry(max_retries=3, base_delay=0.01)
        def raises_value_error() -> str:
            nonlocal call_count
            call_count += 1
            raise ValueError("Bad value")

        with pytest.raises(ValueError):
            raises_value_error()

        assert call_count == 1  # No retries


class TestAsyncRetryDecorator:
    """Tests for @with_async_retry decorator."""

    @pytest.mark.asyncio
    async def test_async_success_no_retry(self) -> None:
        """Successful async call should not retry."""
        from redis_openai_agents.retry import with_async_retry

        call_count = 0

        @with_async_retry(max_retries=3)
        async def successful_fn() -> str:
            nonlocal call_count
            call_count += 1
            return "success"

        result = await successful_fn()

        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_async_retry_on_connection_error(self) -> None:
        """Should retry async on ConnectionError."""
        from redis_openai_agents.retry import with_async_retry

        call_count = 0

        @with_async_retry(max_retries=3, base_delay=0.01)
        async def failing_then_success() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Connection lost")
            return "success"

        result = await failing_then_success()

        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_async_max_retries_exceeded(self) -> None:
        """Should raise after max async retries exceeded."""
        from redis_openai_agents.retry import with_async_retry

        call_count = 0

        @with_async_retry(max_retries=2, base_delay=0.01)
        async def always_fails() -> str:
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Always fails")

        with pytest.raises(ConnectionError):
            await always_fails()

        assert call_count == 3


class TestRetryConfig:
    """Tests for RetryConfig class."""

    def test_default_config(self) -> None:
        """Default config should have sensible values."""
        from redis_openai_agents.retry import RetryConfig

        config = RetryConfig()

        assert config.max_retries == 3
        assert config.base_delay == 0.1
        assert config.max_delay == 10.0
        assert config.exponential_base == 2

    def test_custom_config(self) -> None:
        """Custom config should override defaults."""
        from redis_openai_agents.retry import RetryConfig

        config = RetryConfig(
            max_retries=5,
            base_delay=0.5,
            max_delay=30.0,
            exponential_base=3,
        )

        assert config.max_retries == 5
        assert config.base_delay == 0.5
        assert config.max_delay == 30.0
        assert config.exponential_base == 3

    def test_calculate_delay(self) -> None:
        """Should calculate exponential backoff delay."""
        from redis_openai_agents.retry import RetryConfig

        # Disable jitter for predictable testing
        config = RetryConfig(base_delay=1.0, max_delay=10.0, exponential_base=2, jitter=False)

        # delay = base_delay * (exponential_base ** attempt)
        assert config.calculate_delay(0) == 1.0  # 1 * 2^0 = 1
        assert config.calculate_delay(1) == 2.0  # 1 * 2^1 = 2
        assert config.calculate_delay(2) == 4.0  # 1 * 2^2 = 4
        assert config.calculate_delay(3) == 8.0  # 1 * 2^3 = 8
        assert config.calculate_delay(4) == 10.0  # Capped at max_delay


class TestRetryableExceptions:
    """Tests for configurable retryable exceptions."""

    def test_custom_retryable_exceptions(self) -> None:
        """Should retry on custom exception types."""
        from redis_openai_agents.retry import with_retry

        class CustomError(Exception):
            pass

        call_count = 0

        @with_retry(
            max_retries=2,
            base_delay=0.01,
            retryable_exceptions=(CustomError,),
        )
        def raises_custom() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise CustomError("Custom error")
            return "success"

        result = raises_custom()

        assert result == "success"
        assert call_count == 2
