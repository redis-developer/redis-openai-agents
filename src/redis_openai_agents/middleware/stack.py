"""MiddlewareStack - composes middleware around a base Model.

The stack implements the OpenAI Agents SDK ``Model`` interface so it can be
passed directly to ``Agent(model=stack)`` or ``Runner.run(..., model=stack)``.
Middlewares are applied outer-to-inner for the request path and inner-to-outer
for the response path.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import TYPE_CHECKING, Any

from agents.models.interface import Model

from .base import AgentMiddleware, ModelCallHandler, ModelRequest

if TYPE_CHECKING:
    pass


class MiddlewareStack(Model):
    """Compose middleware around an OpenAI Agents SDK ``Model``.

    Example::

        base = OpenAIResponsesModel(model="gpt-4o")
        stack = MiddlewareStack(
            model=base,
            middlewares=[SemanticCacheMiddleware(cache), MetricsMiddleware(metrics)],
        )
        result = await Runner.run(agent, "Hello", model=stack)

    Args:
        model: The inner ``Model`` instance that performs the actual LLM call.
        middlewares: Sequence of middleware applied outermost-first.
    """

    def __init__(
        self,
        model: Any,
        middlewares: Sequence[AgentMiddleware],
    ) -> None:
        self._inner = model
        self._middlewares: list[AgentMiddleware] = list(middlewares)

    @property
    def inner(self) -> Any:
        """The wrapped ``Model`` instance."""
        return self._inner

    @property
    def middlewares(self) -> list[AgentMiddleware]:
        """Applied middlewares, in outer-to-inner order."""
        return list(self._middlewares)

    async def get_response(
        self,
        system_instructions: str | None,
        input: Any,
        model_settings: Any,
        tools: list[Any],
        output_schema: Any,
        handoffs: list[Any],
        tracing: Any,
        *,
        previous_response_id: str | None = None,
        conversation_id: str | None = None,
        prompt: Any = None,
    ) -> Any:
        """Run the middleware chain then the underlying model."""
        request = ModelRequest(
            system_instructions=system_instructions,
            input=input,
            model_settings=model_settings,
            tools=tools,
            output_schema=output_schema,
            handoffs=handoffs,
            tracing=tracing,
            previous_response_id=previous_response_id,
            conversation_id=conversation_id,
            prompt=prompt,
        )

        async def terminal(req: ModelRequest) -> Any:
            return await self._inner.get_response(
                system_instructions=req.system_instructions,
                input=req.input,
                model_settings=req.model_settings,
                tools=req.tools,
                output_schema=req.output_schema,
                handoffs=req.handoffs,
                tracing=req.tracing,
                previous_response_id=req.previous_response_id,
                conversation_id=req.conversation_id,
                prompt=req.prompt,
            )

        chain: ModelCallHandler = terminal
        for middleware in reversed(self._middlewares):
            chain = _wrap(middleware, chain)

        return await chain(request)

    def stream_response(
        self,
        system_instructions: str | None,
        input: Any,
        model_settings: Any,
        tools: list[Any],
        output_schema: Any,
        handoffs: list[Any],
        tracing: Any,
        *,
        previous_response_id: str | None = None,
        conversation_id: str | None = None,
        prompt: Any = None,
    ) -> AsyncIterator[Any]:
        """Delegate streaming directly to the inner model.

        Streaming middleware is a separate concern (frame-level interception).
        For now, middlewares only see complete responses via ``get_response``.
        """
        iterator: AsyncIterator[Any] = self._inner.stream_response(
            system_instructions=system_instructions,
            input=input,
            model_settings=model_settings,
            tools=tools,
            output_schema=output_schema,
            handoffs=handoffs,
            tracing=tracing,
            previous_response_id=previous_response_id,
            conversation_id=conversation_id,
            prompt=prompt,
        )
        return iterator

    async def close(self) -> None:
        close = getattr(self._inner, "close", None)
        if close is not None:
            await close()


def _wrap(middleware: AgentMiddleware, next_handler: ModelCallHandler) -> ModelCallHandler:
    async def wrapped(request: ModelRequest) -> Any:
        return await middleware.awrap_model_call(request, next_handler)

    return wrapped
