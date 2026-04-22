"""SemanticRouterMiddleware - short-circuit model calls by matched intent.

Routes the user input through a :class:`SemanticRouter`. If the input
matches a known route with a configured canned response, that response is
returned immediately and the LLM call is skipped. Otherwise, the request
is forwarded to the inner model.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from ._response import is_model_response, text_response
from ._utils import extract_user_text
from .base import ModelCallHandler, ModelRequest

if TYPE_CHECKING:
    from ..semantic_router import RouteMatch, SemanticRouter


ResponseFactory = Callable[["RouteMatch"], Any]


class SemanticRouterMiddleware:
    """Short-circuit the LLM call when the input matches a known intent.

    Args:
        router: A :class:`SemanticRouter` configured with the intents to
            recognize.
        responses: Optional mapping from route name to canned response.
            Used when ``response_factory`` is not provided.
        response_factory: Optional callable that receives the
            :class:`RouteMatch` and returns the response. Takes
            precedence over ``responses`` when supplied.

    Either ``responses`` or ``response_factory`` must yield a value for
    the short-circuit to trigger. If neither produces a response (for
    instance, an unmapped route name), the request delegates to the
    inner model.
    """

    def __init__(
        self,
        router: SemanticRouter,
        *,
        responses: Mapping[str, Any] | None = None,
        response_factory: ResponseFactory | None = None,
        auto_wrap: bool = False,
    ) -> None:
        self._router = router
        self._responses = dict(responses) if responses else {}
        self._response_factory = response_factory
        self._auto_wrap = auto_wrap

    async def awrap_model_call(self, request: ModelRequest, handler: ModelCallHandler) -> Any:
        statement = self._extract_statement(request)
        if not statement:
            return await handler(request)

        match = await self._router(statement)
        if match.name is None:
            return await handler(request)

        response = self._resolve_response(match)
        if response is _SENTINEL:
            return await handler(request)
        # Plain strings are auto-wrapped so the Runner receives a real
        # ModelResponse. Pre-built responses are passed through.
        if isinstance(response, str):
            return text_response(response)
        if not is_model_response(response) and self._auto_wrap:
            return text_response(str(response))
        return response

    @staticmethod
    def _extract_statement(request: ModelRequest) -> str:
        """Pick the most recent user text out of the request input."""
        return extract_user_text(request.input, fallback_to_last=True)

    def _resolve_response(self, match: RouteMatch) -> Any:
        if self._response_factory is not None:
            try:
                return self._response_factory(match)
            except Exception:
                return _SENTINEL
        if match.name in self._responses:
            return self._responses[match.name]
        return _SENTINEL


# Sentinel distinguishing "no canned response" from a user-supplied None.
_SENTINEL: Any = object()
