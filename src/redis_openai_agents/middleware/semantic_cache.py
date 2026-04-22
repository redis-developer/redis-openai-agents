"""SemanticCacheMiddleware - cache model responses by semantic similarity.

Wraps the project's ``SemanticCache`` (which itself layers an L1 exact-match
hash and an L2 RedisVL vector cache) and exposes it as an
:class:`~redis_openai_agents.middleware.AgentMiddleware` so it can compose
with other middlewares in a ``MiddlewareStack``.
"""

from __future__ import annotations

import base64
import pickle
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .base import ModelCallHandler, ModelRequest

if TYPE_CHECKING:
    from ..cache import SemanticCache


Serializer = Callable[[Any], str]
Deserializer = Callable[[str], Any]


def _default_serialize(response: Any) -> str:
    """Pickle the response and base64-encode the bytes.

    Pickle handles arbitrary Python values including dataclasses and
    Pydantic models, at the cost of being unsafe to load from untrusted
    sources. Callers caching responses across trust boundaries should
    supply a structured serializer (e.g. ``json.dumps`` with a custom
    ``default``).
    """
    return base64.b64encode(pickle.dumps(response)).decode("ascii")


def _default_deserialize(payload: str) -> Any:
    return pickle.loads(base64.b64decode(payload.encode("ascii")))


class SemanticCacheMiddleware:
    """Cache LLM responses keyed by the semantic similarity of the input.

    Guards against non-deterministic call contexts - requests are not
    cached when any of ``tools``, ``handoffs``, or ``output_schema`` are
    present, because the response typically depends on those side
    conditions and may not repeat.

    Args:
        cache: A :class:`SemanticCache` instance managing the underlying
            two-level cache.
        serializer: Callable that converts a model response to a string
            for storage. Defaults to pickle+base64.
        deserializer: Inverse of ``serializer``.
        cacheable: Optional predicate that returns False to skip caching
            for a given request even if the default guards would allow
            it. Applied after the default guards.
    """

    def __init__(
        self,
        cache: SemanticCache,
        *,
        serializer: Serializer = _default_serialize,
        deserializer: Deserializer = _default_deserialize,
        cacheable: Callable[[ModelRequest], bool] | None = None,
    ) -> None:
        self._cache = cache
        self._serialize = serializer
        self._deserialize = deserializer
        self._cacheable = cacheable

    async def awrap_model_call(self, request: ModelRequest, handler: ModelCallHandler) -> Any:
        if not self._is_cacheable(request):
            return await handler(request)

        prompt = self._build_prompt(request)
        if not prompt:
            return await handler(request)

        hit = await self._lookup(prompt)
        if hit is not None:
            return hit

        response = await handler(request)
        await self._store(prompt, response)
        return response

    def _is_cacheable(self, request: ModelRequest) -> bool:
        """Default guards plus optional user predicate."""
        if request.tools or request.handoffs or request.output_schema:
            return False
        if self._cacheable is not None:
            return self._cacheable(request)
        return True

    def _build_prompt(self, request: ModelRequest) -> str:
        """Build the cache key from the request's textual content.

        Includes system_instructions to distinguish otherwise-identical
        user inputs run with different system prompts.
        """
        parts: list[str] = []
        if request.system_instructions:
            parts.append(f"sys:{request.system_instructions}")

        if isinstance(request.input, str):
            parts.append(f"input:{request.input}")
        elif isinstance(request.input, list):
            for item in request.input:
                parts.append(f"item:{self._stringify(item)}")
        else:
            parts.append(f"input:{self._stringify(request.input)}")

        return "\n".join(parts)

    @staticmethod
    def _stringify(value: Any) -> str:
        if isinstance(value, dict):
            # Stable order for keys so cache keys are deterministic.
            return "{" + ",".join(f"{k}={value[k]}" for k in sorted(value)) + "}"
        return str(value)

    async def _lookup(self, prompt: str) -> Any | None:
        hit = await self._cache.aget(prompt)
        if hit is None:
            return None
        try:
            return self._deserialize(hit.response)
        except Exception:
            return None

    async def _store(self, prompt: str, response: Any) -> None:
        try:
            payload = self._serialize(response)
        except Exception:
            return
        await self._cache.aset(prompt, payload)
