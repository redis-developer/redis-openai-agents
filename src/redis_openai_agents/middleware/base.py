"""Middleware protocol and request snapshot.

Mirrors LangChain's ``AgentMiddleware`` pattern for the OpenAI Agents SDK.
A middleware receives a :class:`ModelRequest` and a ``handler`` (next link
in the chain) and may observe, mutate, or short-circuit the LLM call.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    pass


@dataclass
class ModelRequest:
    """Snapshot of a ``Model.get_response`` invocation.

    Packaging the call as a dataclass lets middlewares inspect and selectively
    override fields without touching the rest of the request shape.
    """

    system_instructions: str | None
    input: Any
    model_settings: Any
    tools: list[Any]
    output_schema: Any
    handoffs: list[Any]
    tracing: Any
    previous_response_id: str | None = None
    conversation_id: str | None = None
    prompt: Any = None


ModelCallHandler = Callable[[ModelRequest], Awaitable[Any]]
"""Signature of the ``handler`` (next link) passed to middleware."""


@runtime_checkable
class AgentMiddleware(Protocol):
    """Protocol for around-style middleware in the OpenAI Agents SDK.

    An implementation receives the request and a handler. It may:

    * Inspect the request, then call ``await handler(request)`` to delegate.
    * Short-circuit by returning a response without calling the handler.
    * Mutate the response produced by the handler before returning it.
    * Swap the request (e.g. inject context) before calling the handler.

    Implementations should be idempotent with respect to repeated calls and
    must not raise for cache or infrastructure errors unless the middleware
    is explicitly configured to fail closed.
    """

    async def awrap_model_call(self, request: ModelRequest, handler: ModelCallHandler) -> Any:
        """Wrap a model call.

        Args:
            request: The packaged request.
            handler: The next link in the chain. Call it with ``request``
                (or a modified copy) to delegate; skip to short-circuit.

        Returns:
            The final model response to return to the Runner.
        """
        ...
