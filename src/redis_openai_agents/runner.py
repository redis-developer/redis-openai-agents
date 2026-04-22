"""Runner integration hooks for OpenAI Agents SDK.

This module provides automatic caching and metrics collection
that integrates with the OpenAI Agents SDK Runner.

Features:
- cached_run(): Check cache before calling LLM
- with_metrics(): Decorator for automatic metrics recording
- RedisAgentRunner: Full integration wrapper for cache, metrics, and session
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, TypeVar

from .cache import SemanticCache
from .metrics import AgentMetrics
from .session import AgentSession

T = TypeVar("T")


@dataclass
class CachedRunResult:
    """Result from a cached run.

    Attributes:
        response: The response string (from cache or LLM)
        cache_hit: Whether the response came from cache
        similarity: Similarity score if from cache (1.0 = exact match)
        original_result: The original SDK result (if not from cache)
    """

    response: str
    cache_hit: bool = False
    similarity: float = 0.0
    original_result: Any | None = field(default=None, repr=False)


def extract_query_from_input(
    input: str | list[dict[str, Any]],  # noqa: A002
) -> str:
    """Extract query string from various input formats.

    Args:
        input: String query or list of messages

    Returns:
        The query string for cache lookup
    """
    if isinstance(input, str):
        return input

    if isinstance(input, list):
        if not input:
            return ""

        # Find last user message
        for msg in reversed(input):
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                # Handle content as list of parts
                if isinstance(content, list):
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and "text" in part:
                            text_parts.append(part["text"])
                        elif hasattr(part, "text"):
                            text_parts.append(part.text)
                    return " ".join(text_parts)

        # Fallback to last message content
        last_msg = input[-1]
        if isinstance(last_msg, dict):
            return str(last_msg.get("content", ""))

    return str(input)


def extract_response_from_result(result: Any) -> str:
    """Extract response string from SDK result.

    Args:
        result: The result from Runner.run()

    Returns:
        The response string for caching
    """
    if isinstance(result, str):
        return result

    if hasattr(result, "final_output"):
        output = result.final_output
        if isinstance(output, str):
            return output
        return str(output)

    return str(result)


def cached_run(
    query: str,
    run_fn: Callable[[], T],
    cache: SemanticCache,
    response_extractor: Callable[[T], str] | None = None,
) -> CachedRunResult | T:
    """Execute run function with cache checking.

    Checks the semantic cache before calling the run function.
    On cache miss, calls the function and caches the result.

    Args:
        query: The query string for cache lookup
        run_fn: Function to call on cache miss (e.g., lambda: Runner.run(...))
        cache: SemanticCache instance
        response_extractor: Optional function to extract response from result

    Returns:
        CachedRunResult on cache hit, or the original result on miss
    """
    # Check cache first
    cached = cache.get(query)

    if cached is not None:
        return CachedRunResult(
            response=cached.response,
            cache_hit=True,
            similarity=cached.similarity,
        )

    # Cache miss - call the run function
    result = run_fn()

    # Extract response for caching
    if response_extractor:
        response = response_extractor(result)
    else:
        response = extract_response_from_result(result)

    # Cache the response
    cache.set(query=query, response=response)

    return result


def with_metrics(
    metrics: AgentMetrics,
    token_extractor: Callable[[Any], tuple[int, int]] | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator that records metrics for a run function.

    Records latency, cache hit status, and token counts.

    Args:
        metrics: AgentMetrics instance
        token_extractor: Optional function to extract (input_tokens, output_tokens)

    Returns:
        Decorated function
    """

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            start_time = time.time()

            result = fn(*args, **kwargs)

            latency_ms = (time.time() - start_time) * 1000

            # Determine if cache hit
            cache_hit = False
            if isinstance(result, CachedRunResult):
                cache_hit = result.cache_hit

            # Extract token counts
            input_tokens = 0
            output_tokens = 0

            if token_extractor:
                input_tokens, output_tokens = token_extractor(result)
            elif hasattr(result, "input_tokens"):
                input_tokens = result.input_tokens
            elif hasattr(result, "output_tokens"):
                output_tokens = result.output_tokens

            # For non-CachedRunResult, check attributes directly
            if not isinstance(result, CachedRunResult):
                if hasattr(result, "input_tokens"):
                    input_tokens = result.input_tokens
                if hasattr(result, "output_tokens"):
                    output_tokens = result.output_tokens

            metrics.record(
                latency_ms=latency_ms,
                input_tokens=input_tokens if input_tokens > 0 else None,
                output_tokens=output_tokens if output_tokens > 0 else None,
                cache_hit=cache_hit,
            )

            return result

        return wrapper

    return decorator


def _call_sdk_runner(
    agent: Any,
    input: str | list[dict[str, Any]],  # noqa: A002
    **kwargs: Any,
) -> Any:
    """Call the OpenAI Agents SDK Runner synchronously.

    This is a separate function to allow easy mocking in tests.

    Args:
        agent: The Agent instance
        input: The input to run
        **kwargs: Additional arguments for Runner.run_sync

    Returns:
        The result from Runner.run_sync
    """
    # Import here to avoid circular imports and allow optional SDK
    try:
        from agents import Runner
    except ImportError as err:
        raise ImportError(
            "openai-agents SDK is required for Runner integration. "
            "Install with: pip install openai-agents"
        ) from err

    return Runner.run_sync(agent, input=input, **kwargs)  # type: ignore[arg-type]


class RedisAgentRunner:
    """Integrated runner with automatic caching, metrics, and session management.

    This class provides a unified interface that combines:
    - SemanticCache: Check/store LLM responses
    - AgentMetrics: Record latency, tokens, cache hits
    - AgentSession: Store conversation history

    Example:
        >>> from redis_openai_agents import (
        ...     RedisAgentRunner, SemanticCache, AgentMetrics, AgentSession
        ... )
        >>>
        >>> runner = RedisAgentRunner(
        ...     cache=SemanticCache(redis_url="redis://localhost:6379"),
        ...     metrics=AgentMetrics(name="my_agent"),
        ...     session=AgentSession(user_id="user_123"),
        ... )
        >>>
        >>> result = runner.run(agent, "What is Redis?")
        >>> # Cache checked, metrics recorded, session updated automatically

    Args:
        cache: Optional SemanticCache for response caching
        metrics: Optional AgentMetrics for observability
        session: Optional AgentSession for conversation history
    """

    def __init__(
        self,
        cache: SemanticCache | None = None,
        metrics: AgentMetrics | None = None,
        session: AgentSession | None = None,
    ) -> None:
        """Initialize the runner with Redis components.

        Args:
            cache: Optional SemanticCache for response caching
            metrics: Optional AgentMetrics for observability
            session: Optional AgentSession for conversation history
        """
        self.cache = cache
        self.metrics = metrics
        self.session = session

    def run(
        self,
        agent: Any,
        input: str | list[dict[str, Any]],  # noqa: A002
        **kwargs: Any,
    ) -> CachedRunResult | Any:
        """Run an agent with automatic caching and metrics.

        On cache hit, returns immediately without calling the LLM.
        On cache miss, calls the SDK runner and caches the result.

        Args:
            agent: The Agent instance to run
            input: The input query or messages
            **kwargs: Additional arguments for Runner.run

        Returns:
            CachedRunResult on cache hit, or SDK result on miss
        """
        start_time = time.time()
        cache_hit = False
        result: Any = None

        # Extract query for cache lookup
        query = extract_query_from_input(input)

        # Check cache first (if configured)
        if self.cache is not None and query:
            cached = self.cache.get(query)
            if cached is not None:
                cache_hit = True
                result = CachedRunResult(
                    response=cached.response,
                    cache_hit=True,
                    similarity=cached.similarity,
                )

        # Cache miss - call SDK runner
        if result is None:
            sdk_result = _call_sdk_runner(agent, input, **kwargs)
            result = sdk_result

            # Cache the response
            if self.cache is not None and query:
                response = extract_response_from_result(sdk_result)
                self.cache.set(query=query, response=response)

            # Store in session (if configured)
            if self.session is not None:
                self.session.store_agent_result(sdk_result)

        # Record metrics (if configured)
        if self.metrics is not None:
            latency_ms = (time.time() - start_time) * 1000

            # Extract token counts if available
            input_tokens = None
            output_tokens = None
            if not cache_hit and hasattr(result, "input_tokens"):
                input_tokens = result.input_tokens
            if not cache_hit and hasattr(result, "output_tokens"):
                output_tokens = result.output_tokens

            self.metrics.record(
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_hit=cache_hit,
            )

        return result

    def run_streamed(
        self,
        agent: Any,
        input: str | list[dict[str, Any]],  # noqa: A002
        **kwargs: Any,
    ) -> Any:
        """Run an agent with streaming (no caching for streamed responses).

        Note: Streaming responses are not cached since the full response
        is not available upfront. Use run() for cacheable requests.

        Args:
            agent: The Agent instance to run
            input: The input query or messages
            **kwargs: Additional arguments for Runner.run_streamed

        Returns:
            Streaming result from Runner.run_streamed
        """
        try:
            from agents import Runner
        except ImportError as err:
            raise ImportError(
                "openai-agents SDK is required for Runner integration. "
                "Install with: pip install openai-agents"
            ) from err

        result = Runner.run_streamed(agent, input=input, **kwargs)  # type: ignore[arg-type]

        # Note: For streaming, we can only record metrics after the stream
        # is consumed. The caller should call record_stream_complete() or
        # use session.store_agent_result() which will handle this.

        return result

    async def arun(
        self,
        agent: Any,
        input: str | list[dict[str, Any]],  # noqa: A002
        **kwargs: Any,
    ) -> CachedRunResult | Any:
        """Async version of run() - run agent with automatic caching and metrics.

        Args:
            agent: The Agent instance to run
            input: The input query or messages
            **kwargs: Additional arguments for Runner.run

        Returns:
            CachedRunResult on cache hit, or SDK result on miss
        """
        start_time = time.time()
        cache_hit = False
        result: Any = None

        # Extract query for cache lookup
        query = extract_query_from_input(input)

        # Check cache first (if configured)
        if self.cache is not None and query:
            cached = await self.cache.aget(query)
            if cached is not None:
                cache_hit = True
                result = CachedRunResult(
                    response=cached.response,
                    cache_hit=True,
                    similarity=cached.similarity,
                )

        # Cache miss - call SDK runner
        if result is None:
            sdk_result = await _acall_sdk_runner(agent, input, **kwargs)
            result = sdk_result

            # Cache the response
            if self.cache is not None and query:
                response = extract_response_from_result(sdk_result)
                await self.cache.aset(query=query, response=response)

            # Store in session (if configured)
            if self.session is not None:
                await self.session.astore_agent_result(sdk_result)

        # Record metrics (if configured)
        if self.metrics is not None:
            latency_ms = (time.time() - start_time) * 1000

            input_tokens = None
            output_tokens = None
            if not cache_hit and hasattr(result, "input_tokens"):
                input_tokens = result.input_tokens
            if not cache_hit and hasattr(result, "output_tokens"):
                output_tokens = result.output_tokens

            await self.metrics.arecord(
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_hit=cache_hit,
            )

        return result


async def _acall_sdk_runner(
    agent: Any,
    input: str | list[dict[str, Any]],  # noqa: A002
    **kwargs: Any,
) -> Any:
    """Async version of _call_sdk_runner.

    Args:
        agent: The Agent instance
        input: The input to run
        **kwargs: Additional arguments for Runner.run

    Returns:
        The result from Runner.run
    """
    try:
        from agents import Runner
    except ImportError as err:
        raise ImportError(
            "openai-agents SDK is required for Runner integration. "
            "Install with: pip install openai-agents"
        ) from err

    # The SDK Runner.run() is actually async
    return await Runner.run(agent, input=input, **kwargs)  # type: ignore[arg-type]
