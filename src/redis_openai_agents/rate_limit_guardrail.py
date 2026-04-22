"""RedisRateLimitGuardrail - Rate limiting guardrail for OpenAI Agents SDK.

This module provides a guardrail that uses Redis to enforce rate limits
on agent requests. Supports both request count and token-based limiting.

Features:
- Per-user rate limiting
- Request count limits
- Token usage limits
- Sliding and fixed window algorithms
- SDK-compatible guardrail interface

Example:
    >>> from redis_openai_agents import RedisRateLimitGuardrail
    >>> from agents import Agent, InputGuardrail
    >>>
    >>> # Create guardrail
    >>> rate_limiter = RedisRateLimitGuardrail(
    ...     redis_url="redis://localhost:6379",
    ...     requests_per_minute=60,
    ...     tokens_per_minute=10000,
    ... )
    >>> await rate_limiter.initialize()
    >>>
    >>> # Use with agent
    >>> guardrail = InputGuardrail(
    ...     guardrail_function=rate_limiter.guardrail_function,
    ...     name="rate_limit",
    ... )
    >>> agent = Agent(
    ...     name="MyAgent",
    ...     input_guardrails=[guardrail],
    ... )
"""

import time
from dataclasses import dataclass
from typing import Any

import redis.asyncio as redis


@dataclass
class GuardrailFunctionOutput:
    """Output from a guardrail function.

    Matches the OpenAI Agents SDK GuardrailFunctionOutput structure.
    """

    output_info: Any
    """Optional metadata about the guardrail check."""

    tripwire_triggered: bool
    """If True, the request is blocked."""


class RedisRateLimitGuardrail:
    """Redis-backed rate limiting guardrail for OpenAI Agents SDK.

    Enforces rate limits using Redis counters with configurable
    time windows and limit types (requests, tokens, or both).

    Attributes:
        requests_per_minute: Maximum requests allowed per minute per user.
        tokens_per_minute: Maximum tokens allowed per minute per user.
        window_type: Rate limiting algorithm ("sliding" or "fixed").
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        key_prefix: str = "rate_limit",
        requests_per_minute: int | None = None,
        tokens_per_minute: int | None = None,
        window_type: str = "sliding",
        window_seconds: int = 60,
    ) -> None:
        """Initialize the rate limit guardrail.

        Args:
            redis_url: Redis connection URL.
            key_prefix: Prefix for rate limit keys.
            requests_per_minute: Max requests per minute (None = unlimited).
            tokens_per_minute: Max tokens per minute (None = unlimited).
            window_type: "sliding" or "fixed" window algorithm.
            window_seconds: Duration of rate limit window.
        """
        self._redis_url = redis_url
        self._key_prefix = key_prefix
        self._requests_per_minute = requests_per_minute
        self._tokens_per_minute = tokens_per_minute
        self._window_type = window_type
        self._window_seconds = window_seconds

        self._client: redis.Redis | None = None

    async def initialize(self) -> None:
        """Initialize Redis connection."""
        self._client = redis.from_url(self._redis_url, decode_responses=True)

    async def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _get_window_key(self, user_id: str, key_type: str) -> str:
        """Get the Redis key for a rate limit counter.

        Args:
            user_id: User identifier.
            key_type: Type of counter ("requests" or "tokens").

        Returns:
            Redis key string.
        """
        if self._window_type == "fixed":
            # Fixed window uses current minute bucket
            window_id = int(time.time() // self._window_seconds)
            return f"{self._key_prefix}:{user_id}:{key_type}:{window_id}"
        else:
            # Sliding window uses single key
            return f"{self._key_prefix}:{user_id}:{key_type}"

    async def check_rate_limit(
        self,
        user_id: str,
        tokens_used: int = 0,
    ) -> GuardrailFunctionOutput:
        """Check if request is within rate limits.

        Args:
            user_id: User identifier for rate limiting.
            tokens_used: Number of tokens for this request.

        Returns:
            GuardrailFunctionOutput with tripwire_triggered=True if blocked.
        """
        if not self._client:
            return GuardrailFunctionOutput(
                output_info={"error": "Redis not initialized"},
                tripwire_triggered=False,
            )

        now = time.time()
        blocked = False
        block_reason = None

        # Check request limit
        if self._requests_per_minute is not None:
            request_key = self._get_window_key(user_id, "requests")

            if self._window_type == "sliding":
                # Sliding window: use sorted set with timestamps
                # Remove old entries
                window_start = now - self._window_seconds
                await self._client.zremrangebyscore(request_key, 0, window_start)

                # Count current requests
                current_count = await self._client.zcard(request_key)

                if current_count >= self._requests_per_minute:
                    blocked = True
                    block_reason = (
                        f"Request limit exceeded: {current_count}/{self._requests_per_minute} "
                        f"requests in the last {self._window_seconds} seconds"
                    )
                else:
                    # Add this request
                    await self._client.zadd(request_key, {str(now): now})
                    await self._client.expire(request_key, self._window_seconds * 2)
            else:
                # Fixed window: simple counter
                current_count = await self._client.incr(request_key)

                if current_count == 1:
                    # First request in window, set expiry
                    await self._client.expire(request_key, self._window_seconds)

                if current_count > self._requests_per_minute:
                    blocked = True
                    block_reason = (
                        f"Request limit exceeded: {current_count}/{self._requests_per_minute} "
                        f"requests in current window"
                    )

        # Check token limit (only if not already blocked)
        if not blocked and self._tokens_per_minute is not None and tokens_used > 0:
            token_key = self._get_window_key(user_id, "tokens")

            if self._window_type == "sliding":
                # For tokens, we need to track cumulative usage
                window_start = now - self._window_seconds
                await self._client.zremrangebyscore(token_key, 0, window_start)

                # Get current token usage
                token_entries = await self._client.zrange(token_key, 0, -1, withscores=True)
                current_tokens = sum(int(entry[0].split(":")[0]) for entry in token_entries)

                if current_tokens + tokens_used > self._tokens_per_minute:
                    blocked = True
                    block_reason = (
                        f"Token limit exceeded: {current_tokens + tokens_used}/"
                        f"{self._tokens_per_minute} tokens"
                    )
                else:
                    # Add this usage
                    entry_id = f"{tokens_used}:{now}"
                    await self._client.zadd(token_key, {entry_id: now})
                    await self._client.expire(token_key, self._window_seconds * 2)
            else:
                # Fixed window
                current_tokens = await self._client.incrby(token_key, tokens_used)

                if current_tokens == tokens_used:
                    # First request in window
                    await self._client.expire(token_key, self._window_seconds)

                if current_tokens > self._tokens_per_minute:
                    blocked = True
                    block_reason = (
                        f"Token limit exceeded: {current_tokens}/{self._tokens_per_minute} tokens"
                    )

        if blocked:
            return GuardrailFunctionOutput(
                output_info={
                    "reason": block_reason,
                    "user_id": user_id,
                    "timestamp": now,
                },
                tripwire_triggered=True,
            )

        return GuardrailFunctionOutput(
            output_info={
                "allowed": True,
                "user_id": user_id,
            },
            tripwire_triggered=False,
        )

    async def get_rate_limit_info(self, user_id: str) -> dict:
        """Get current rate limit status for a user.

        Args:
            user_id: User identifier.

        Returns:
            Dictionary with rate limit information.
        """
        if not self._client:
            return {"error": "Redis not initialized"}

        now = time.time()
        info: dict[str, Any] = {
            "user_id": user_id,
            "timestamp": now,
        }

        # Get request count
        if self._requests_per_minute is not None:
            request_key = self._get_window_key(user_id, "requests")

            if self._window_type == "sliding":
                window_start = now - self._window_seconds
                await self._client.zremrangebyscore(request_key, 0, window_start)
                current_count = await self._client.zcard(request_key)
            else:
                current_count_str = await self._client.get(request_key)
                current_count = int(current_count_str) if current_count_str else 0

            info["current_requests"] = current_count
            info["remaining_requests"] = max(0, self._requests_per_minute - current_count)
            info["requests_limit"] = self._requests_per_minute

        # Get token count
        if self._tokens_per_minute is not None:
            token_key = self._get_window_key(user_id, "tokens")

            if self._window_type == "sliding":
                window_start = now - self._window_seconds
                await self._client.zremrangebyscore(token_key, 0, window_start)
                token_entries = await self._client.zrange(token_key, 0, -1, withscores=True)
                current_tokens = sum(int(entry[0].split(":")[0]) for entry in token_entries)
            else:
                current_tokens_str = await self._client.get(token_key)
                current_tokens = int(current_tokens_str) if current_tokens_str else 0

            info["current_tokens"] = current_tokens
            info["remaining_tokens"] = max(0, self._tokens_per_minute - current_tokens)
            info["tokens_limit"] = self._tokens_per_minute

        # Calculate reset time
        if self._window_type == "fixed":
            window_id = int(now // self._window_seconds)
            info["reset_at"] = (window_id + 1) * self._window_seconds
        else:
            info["reset_at"] = now + self._window_seconds

        return info

    async def guardrail_function(
        self,
        context: Any,
        agent: Any,
        input_data: Any,
    ) -> GuardrailFunctionOutput:
        """Guardrail function compatible with OpenAI Agents SDK.

        This method can be used directly with InputGuardrail.

        Args:
            context: Run context wrapper.
            agent: The agent being run.
            input_data: Input to the agent.

        Returns:
            GuardrailFunctionOutput indicating whether request is allowed.
        """
        # Extract user_id from context if available
        user_id = "default"
        if hasattr(context, "context"):
            ctx = context.context
            if hasattr(ctx, "user_id"):
                user_id = ctx.user_id
            elif isinstance(ctx, dict) and "user_id" in ctx:
                user_id = ctx["user_id"]

        return await self.check_rate_limit(user_id=user_id)

    @property
    def name(self) -> str:
        """Guardrail name for SDK registration."""
        return "redis_rate_limit"
