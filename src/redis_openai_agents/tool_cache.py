"""cached_tool - memoize deterministic tool results in Redis.

The OpenAI Agents SDK routes tool execution through the Runner, not through
the ``Model`` interface, so tool caching cannot be implemented as a regular
middleware. ``cached_tool`` is a decorator that wraps the underlying
callable before it becomes a ``function_tool``, hashing its arguments to
produce a cache key and serving repeat calls from Redis.

Inspired by the ``ToolResultCacheMiddleware`` in ``langgraph-redis``.

Example::

    from agents import function_tool
    from redis_openai_agents import cached_tool

    @function_tool
    @cached_tool(
        name="weather",
        redis_url="redis://localhost:6379",
        ttl=3600,
        volatile_arg_names={"timestamp"},
    )
    async def get_weather(city: str) -> str:
        return fetch_forecast(city)
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import pickle
from collections.abc import Callable, Set
from functools import wraps
from typing import Any, TypeVar

from redis import Redis

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=Callable[..., Any])

DEFAULT_VOLATILE_ARG_NAMES: frozenset[str] = frozenset(
    {
        "timestamp",
        "current_time",
        "now",
        "date",
        "today",
        "current_date",
        "current_timestamp",
    }
)

DEFAULT_SIDE_EFFECT_PREFIXES: tuple[str, ...] = (
    "send_",
    "delete_",
    "create_",
    "update_",
    "remove_",
    "write_",
    "post_",
    "put_",
    "patch_",
)


def cached_tool(
    *,
    name: str,
    redis_url: str = "redis://localhost:6379",
    ttl: int | None = None,
    key_prefix: str = "tool_cache",
    volatile_arg_names: Set[str] | None = None,
    ignored_arg_names: Set[str] | None = None,
    side_effect_prefixes: tuple[str, ...] | None = None,
) -> Callable[[T], T]:
    """Decorator that memoizes a callable's return value in Redis.

    Args:
        name: Logical tool name. Also used as part of the cache key and for
            checking side-effect prefixes.
        redis_url: Redis connection URL. The underlying Redis client is
            created once at decoration time and shared across calls.
        ttl: Optional TTL in seconds for cache entries. ``None`` keeps them
            indefinitely.
        key_prefix: Prefix for the Redis key; useful for namespacing across
            environments or deployments.
        volatile_arg_names: Set of argument names whose presence bypasses
            the cache entirely (even if the value is ``None``). Use this for
            arguments that always change (timestamps, trace IDs that must
            be observed, random seeds). Defaults to
            :data:`DEFAULT_VOLATILE_ARG_NAMES`.
        ignored_arg_names: Set of argument names to strip from the cache
            key before hashing. Use for values that do not change the
            result but vary between calls (trace IDs, request IDs).
        side_effect_prefixes: Tuple of name prefixes that mark the tool as
            side-effecting, in which case caching is disabled. Defaults to
            :data:`DEFAULT_SIDE_EFFECT_PREFIXES`.

    Returns:
        A decorator.
    """
    volatile = (
        frozenset(volatile_arg_names)
        if volatile_arg_names is not None
        else DEFAULT_VOLATILE_ARG_NAMES
    )
    ignored = frozenset(ignored_arg_names) if ignored_arg_names is not None else frozenset()
    prefixes = (
        tuple(side_effect_prefixes)
        if side_effect_prefixes is not None
        else DEFAULT_SIDE_EFFECT_PREFIXES
    )

    is_side_effecting = name.startswith(prefixes)
    client = Redis.from_url(redis_url)

    def decorator(fn: T) -> T:
        signature = inspect.signature(fn)

        def make_key(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str | None:
            """Normalize args to a canonical key; return ``None`` to skip the cache."""
            try:
                bound = signature.bind_partial(*args, **kwargs)
                bound.apply_defaults()
            except TypeError:
                # Let the underlying call raise the real TypeError.
                return None

            arguments = dict(bound.arguments)
            if _contains_volatile(arguments, volatile):
                return None
            for arg_name in ignored:
                arguments.pop(arg_name, None)

            canonical = _canonicalize(arguments)
            digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            return f"{key_prefix}:{name}:{digest}"

        def load(key: str) -> tuple[bool, Any]:
            try:
                raw = client.get(key)
            except Exception as exc:
                logger.debug("cache read failed for %s: %s", key, exc)
                return False, None
            if raw is None or not isinstance(raw, (bytes, bytearray)):
                return False, None
            try:
                return True, pickle.loads(raw)
            except Exception as exc:
                logger.debug("cache payload decode failed for %s: %s", key, exc)
                return False, None

        def store(key: str, value: Any) -> None:
            try:
                payload = pickle.dumps(value)
                if ttl is not None:
                    client.setex(key, ttl, payload)
                else:
                    client.set(key, payload)
            except Exception as exc:
                logger.debug("cache write failed for %s: %s", key, exc)

        if asyncio.iscoroutinefunction(fn):

            @wraps(fn)
            async def async_wrapped(*args: Any, **kwargs: Any) -> Any:
                if is_side_effecting:
                    return await fn(*args, **kwargs)
                key = make_key(args, kwargs)
                if key is None:
                    return await fn(*args, **kwargs)
                hit, value = await asyncio.to_thread(load, key)
                if hit:
                    return value
                result = await fn(*args, **kwargs)
                await asyncio.to_thread(store, key, result)
                return result

            return async_wrapped  # type: ignore[return-value]

        @wraps(fn)
        def sync_wrapped(*args: Any, **kwargs: Any) -> Any:
            if is_side_effecting:
                return fn(*args, **kwargs)
            key = make_key(args, kwargs)
            if key is None:
                return fn(*args, **kwargs)
            hit, value = load(key)
            if hit:
                return value
            result = fn(*args, **kwargs)
            store(key, result)
            return result

        return sync_wrapped  # type: ignore[return-value]

    return decorator


def _contains_volatile(value: Any, volatile: Set[str]) -> bool:
    """Recursively check whether any volatile arg name appears in ``value``."""
    if not volatile:
        return False
    if isinstance(value, dict):
        for k, v in value.items():
            if k in volatile:
                return True
            if _contains_volatile(v, volatile):
                return True
        return False
    if isinstance(value, (list, tuple, set)):
        return any(_contains_volatile(v, volatile) for v in value)
    return False


def _canonicalize(arguments: dict[str, Any]) -> str:
    """Serialize arguments to a deterministic JSON string.

    Falls back to ``repr`` for values that are not JSON-serializable so
    the hash still stays stable for a given input shape.
    """

    def default(o: Any) -> Any:
        if isinstance(o, (set, frozenset)):
            return sorted(o, key=repr)
        return repr(o)

    return json.dumps(arguments, sort_keys=True, default=default, separators=(",", ":"))
