"""Retry and backoff functionality for Redis operations.

This module provides retry logic with exponential backoff for
handling transient Redis connection failures.

Features:
- Configurable retry count and delays
- Exponential backoff with jitter
- Sync and async decorators
- Configurable retryable exceptions
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import Any, TypeVar

from redis.exceptions import ConnectionError, TimeoutError

T = TypeVar("T")

# Default exceptions that trigger retry
DEFAULT_RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (
    ConnectionError,
    TimeoutError,
)


@dataclass
class RetryConfig:
    """Configuration for retry behavior.

    Attributes:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds for exponential backoff
        max_delay: Maximum delay cap in seconds
        exponential_base: Base for exponential backoff (delay = base_delay * exponential_base^attempt)
        jitter: Whether to add random jitter to delays
    """

    max_retries: int = 3
    base_delay: float = 0.1
    max_delay: float = 10.0
    exponential_base: float = 2
    jitter: bool = True

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for a given attempt number.

        Uses exponential backoff: delay = base_delay * (exponential_base ^ attempt)

        Args:
            attempt: The attempt number (0-indexed)

        Returns:
            Delay in seconds
        """
        delay = self.base_delay * (self.exponential_base**attempt)
        delay = min(delay, self.max_delay)

        if self.jitter:
            # Add up to 25% random jitter
            jitter_range = delay * 0.25
            delay += random.uniform(-jitter_range, jitter_range)
            delay = max(0, delay)

        return delay


def with_retry(
    max_retries: int = 3,
    base_delay: float = 0.1,
    max_delay: float = 10.0,
    exponential_base: float = 2,
    jitter: bool = True,
    retryable_exceptions: tuple[type[Exception], ...] = DEFAULT_RETRYABLE_EXCEPTIONS,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator that adds retry logic to a sync function.

    Example:
        >>> @with_retry(max_retries=3, base_delay=0.1)
        ... def fetch_from_redis():
        ...     return client.get("key")

    Args:
        max_retries: Maximum retry attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay cap
        exponential_base: Exponential backoff base
        jitter: Whether to add random jitter
        retryable_exceptions: Tuple of exception types to retry

    Returns:
        Decorated function with retry logic
    """
    config = RetryConfig(
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        jitter=jitter,
    )

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Exception = Exception("No attempts made")

            for attempt in range(max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e

                    if attempt < max_retries:
                        delay = config.calculate_delay(attempt)
                        time.sleep(delay)
                except Exception:
                    # Non-retryable exception, re-raise immediately
                    raise

            # All retries exhausted
            raise last_exception

        return wrapper

    return decorator


def with_async_retry(
    max_retries: int = 3,
    base_delay: float = 0.1,
    max_delay: float = 10.0,
    exponential_base: float = 2,
    jitter: bool = True,
    retryable_exceptions: tuple[type[Exception], ...] = DEFAULT_RETRYABLE_EXCEPTIONS,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator that adds retry logic to an async function.

    Example:
        >>> @with_async_retry(max_retries=3, base_delay=0.1)
        ... async def fetch_from_redis():
        ...     return await client.get("key")

    Args:
        max_retries: Maximum retry attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay cap
        exponential_base: Exponential backoff base
        jitter: Whether to add random jitter
        retryable_exceptions: Tuple of exception types to retry

    Returns:
        Decorated async function with retry logic
    """
    config = RetryConfig(
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        jitter=jitter,
    )

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Exception = Exception("No attempts made")

            for attempt in range(max_retries + 1):
                try:
                    return await fn(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e

                    if attempt < max_retries:
                        delay = config.calculate_delay(attempt)
                        await asyncio.sleep(delay)
                except Exception:
                    # Non-retryable exception, re-raise immediately
                    raise

            # All retries exhausted
            raise last_exception

        return wrapper

    return decorator


# Global default configuration
_default_config = RetryConfig()


def get_retry_config() -> RetryConfig:
    """Get the default retry configuration."""
    return _default_config


def configure_retry(
    max_retries: int = 3,
    base_delay: float = 0.1,
    max_delay: float = 10.0,
    exponential_base: float = 2,
    jitter: bool = True,
) -> None:
    """Configure the global default retry settings.

    Args:
        max_retries: Maximum retry attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay cap
        exponential_base: Exponential backoff base
        jitter: Whether to add random jitter
    """
    global _default_config
    _default_config = RetryConfig(
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        jitter=jitter,
    )
