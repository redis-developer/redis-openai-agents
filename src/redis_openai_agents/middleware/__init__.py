"""Middleware package for the OpenAI Agents SDK.

Provides an around-style middleware protocol modelled on LangChain's
``AgentMiddleware`` and a ``MiddlewareStack`` that implements the OpenAI
Agents SDK ``Model`` interface so middlewares plug directly into the Runner.
"""

from ._response import text_response
from .base import AgentMiddleware, ModelCallHandler, ModelRequest
from .conversation_memory import ConversationMemoryMiddleware
from .semantic_cache import SemanticCacheMiddleware
from .semantic_router import SemanticRouterMiddleware
from .stack import MiddlewareStack

__all__ = [
    "AgentMiddleware",
    "ConversationMemoryMiddleware",
    "ModelCallHandler",
    "ModelRequest",
    "MiddlewareStack",
    "SemanticCacheMiddleware",
    "SemanticRouterMiddleware",
    "text_response",
]
