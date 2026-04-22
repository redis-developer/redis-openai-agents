---
myst:
  html_meta:
    "description lang=en": |
        API documentation for Redis OpenAI Agents
---

# Redis OpenAI Agents API

Reference documentation for the Redis OpenAI Agents API.

## Sessions & Memory

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :recursive:

   redis_openai_agents.AgentSession
   redis_openai_agents.JSONSession
```

## Caching

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :recursive:

   redis_openai_agents.SemanticCache
   redis_openai_agents.CacheResult
   redis_openai_agents.RedisCachingModel
   redis_openai_agents.CachedModelResponse
   redis_openai_agents.CachedUsage
```

## Search

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :recursive:

   redis_openai_agents.RedisVectorStore
   redis_openai_agents.RedisFullTextSearch
   redis_openai_agents.HybridSearchService
   redis_openai_agents.SearchResult
```

## Routing

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :recursive:

   redis_openai_agents.SemanticRouter
   redis_openai_agents.Route
   redis_openai_agents.RouteMatch
```

## Streaming

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :recursive:

   redis_openai_agents.RedisStreamTransport
   redis_openai_agents.RobustStreamProcessor
   redis_openai_agents.ResumableStreamRunner
   redis_openai_agents.StreamingEventPublisher
```

## Coordination

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :recursive:

   redis_openai_agents.AgentCoordinator
   redis_openai_agents.EventType
   redis_openai_agents.AtomicOperations
   redis_openai_agents.HandoffInProgressError
```

## Observability

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :recursive:

   redis_openai_agents.AgentMetrics
   redis_openai_agents.PrometheusExporter
   redis_openai_agents.RedisTracingProcessor
```

## SDK Integration

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :recursive:

   redis_openai_agents.RedisAgentRunner
   redis_openai_agents.CachedRunResult
   redis_openai_agents.RedisFileSearchTool
   redis_openai_agents.RedisRateLimitGuardrail
   redis_openai_agents.cached_run
   redis_openai_agents.with_metrics
   redis_openai_agents.create_redis_file_search_tool
```

## Tool Caching

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :recursive:

   redis_openai_agents.cached_tool
   redis_openai_agents.DEFAULT_SIDE_EFFECT_PREFIXES
   redis_openai_agents.DEFAULT_VOLATILE_ARG_NAMES
```

## Middleware

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :recursive:

   redis_openai_agents.MiddlewareStack
   redis_openai_agents.AgentMiddleware
   redis_openai_agents.ModelRequest
   redis_openai_agents.middleware.SemanticCacheMiddleware
   redis_openai_agents.middleware.SemanticRouterMiddleware
   redis_openai_agents.middleware.ConversationMemoryMiddleware
```

## Infrastructure

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :recursive:

   redis_openai_agents.RedisConnectionPool
   redis_openai_agents.configure_pool
   redis_openai_agents.get_pool
   redis_openai_agents.reset_pool
   redis_openai_agents.RetryConfig
   redis_openai_agents.with_retry
   redis_openai_agents.with_async_retry
   redis_openai_agents.configure_retry
   redis_openai_agents.get_retry_config
   redis_openai_agents.RankedOperations
   redis_openai_agents.DeduplicationService
```

