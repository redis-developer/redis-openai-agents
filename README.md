<div align="center">
    <img width="300" src="https://raw.githubusercontent.com/redis/redis-vl-python/main/docs/_static/Redis_Logo_Red_RGB.svg" alt="Redis">
    <h1>Redis OpenAI Agents</h1>
    <p><strong>Production-ready Redis integrations for the OpenAI Agents SDK</strong></p>
</div>

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](https://mypy-lang.org/)

**[Documentation](https://redis.github.io/redis-openai-agents)** • **[Examples](#examples)** • **[GitHub](https://github.com/redis/redis-openai-agents)**

</div>

---

## Introduction

Redis OpenAI Agents is a production-ready Python library that provides Redis-powered infrastructure for the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python). **Replace 5+ separate systems with a single Redis deployment.**

<div align="center">

| **Sessions & Memory** | **Caching & Search** | **Streaming & Coordination** |
|:---:|:---:|:---:|
| **[AgentSession](#agent-sessions)**<br/>*Persistent conversation storage* | **[SemanticCache](#semantic-caching)**<br/>*Reduce LLM costs by 25%+* | **[RedisStreamTransport](#token-streaming)**<br/>*Reliable, replayable streaming* |
| **[JSONSession](#agent-sessions)**<br/>*Complex nested data storage* | **[RedisVectorStore](#vector-search-rag)**<br/>*Fast similarity search* | **[AgentCoordinator](#agent-coordination)**<br/>*Multi-agent orchestration* |
| **[SemanticRouter](#semantic-routing)**<br/>*Intent-based agent routing* | **[HybridSearchService](#hybrid-search)**<br/>*BM25 + vector combined* | **[RobustStreamProcessor](#token-streaming)**<br/>*Consumer groups & replay* |

</div>

### **Built for OpenAI Agents SDK**

- **Drop-in Session Storage** → Replace SQLite with distributed Redis sessions
- **Cost Reduction** → Semantic caching reduces LLM API calls by 25%+
- **Production Streaming** → Redis Streams for reliable token delivery
- **Multi-Agent Systems** → Coordinate agents with atomic operations

---

## Getting Started

### Installation

Install `redis-openai-agents` into your Python (>=3.10) environment:

```bash
pip install redis-openai-agents
```

### Redis

Choose from multiple Redis deployment options:

1. **[Redis Cloud](https://redis.io/try-free)**: Managed cloud database (free tier available)
2. **Redis (Docker)**: The official `redis:8` image ships with Search, JSON, Time Series, and Bloom filters built in - no separate stack image required.

    ```bash
    docker run -d --name redis -p 6379:6379 redis:8
    ```

3. **[Redis Enterprise](https://redis.io/enterprise/)**: Commercial, self-hosted database

> Want a GUI? Run [Redis Insight](https://redis.io/insight/) separately: `docker run -d --name redisinsight -p 5540:5540 redis/redisinsight:latest`.

---

## Overview

### Agent Sessions

Replace SQLite sessions with Redis for persistent, distributed conversation storage:

```python
from agents import Agent, Runner
from redis_openai_agents import AgentSession

# Create a session
session = AgentSession.create(
    user_id="user_123",
    redis_url="redis://localhost:6379"
)

# Define your agent
agent = Agent(name="assistant", instructions="You are a helpful assistant.")

# Run the agent
result = await Runner.run(agent, input="Hello!")

# Store the conversation
session.store_agent_result(result)

# Later: Load and continue the conversation
session = AgentSession.load(
    conversation_id=session.conversation_id,
    user_id="user_123",
    redis_url="redis://localhost:6379"
)

# Get conversation history in SDK format
history = session.to_agent_inputs()
result = await Runner.run(agent, input=history + [{"role": "user", "content": "Follow up"}])
```

> An async-compatible JSON session is also available: `JSONSession` for complex nested data.

### Semantic Caching

Reduce LLM costs by caching responses for similar queries:

```python
from redis_openai_agents import SemanticCache

cache = SemanticCache(
    redis_url="redis://localhost:6379",
    distance_threshold=0.1,  # Similarity threshold (lower = stricter)
    ttl=3600                  # 1 hour TTL
)

# Check cache before calling LLM
result = cache.check(query="What is the capital of France?")
if result:
    print(f"Cache hit: {result.response}")
else:
    # Call LLM and store result
    response = "Paris is the capital of France."
    cache.store(query="What is the capital of France?", response=response)
```

> Learn more about [semantic caching](docs/user_guide/02_semantic_cache.ipynb).

### Semantic Routing

Route queries to the appropriate agent using vector similarity - no LLM calls required:

```python
from redis_openai_agents import SemanticRouter, Route

router = SemanticRouter(
    name="support-router",
    redis_url="redis://localhost:6379",
    routes=[
        Route(
            name="billing",
            references=["payment issue", "invoice", "refund request"],
            metadata={"agent": "billing_agent"},
            distance_threshold=0.3
        ),
        Route(
            name="technical",
            references=["bug report", "error message", "not working"],
            metadata={"agent": "tech_agent"},
            distance_threshold=0.3
        ),
    ]
)

# Route a query (vector lookup, not LLM call)
match = router.route("I need help with my payment")
print(f"Route to: {match.name}")  # "billing"
```

> Learn more about [semantic routing](docs/user_guide/03_semantic_router.ipynb).

### Vector Search (RAG)

Build RAG applications with Redis vector search:

```python
from redis_openai_agents import RedisVectorStore

store = RedisVectorStore(
    name="knowledge-base",
    redis_url="redis://localhost:6379"
)

# Add documents
store.add_documents([
    {"content": "Redis is an in-memory data store.", "source": "docs"},
    {"content": "Python is a programming language.", "source": "wiki"},
])

# Search with metadata filtering
results = store.search(
    query="What is Redis?",
    k=5,
    filter={"source": "docs"}
)

for result in results:
    print(f"{result.content} (score: {result.score})")
```

### Hybrid Search

Combine vector similarity with BM25 full-text search for better retrieval:

```python
from redis_openai_agents import HybridSearchService

search = HybridSearchService(
    name="hybrid-search",
    redis_url="redis://localhost:6379"
)

# Search with both vector and text matching
results = search.search(
    query="Redis performance optimization",
    k=10,
    vector_weight=0.7,  # 70% vector similarity
    text_weight=0.3     # 30% BM25 text match
)
```

### Token Streaming

Reliable, replayable token streaming via Redis Streams:

```python
from redis_openai_agents import RedisStreamTransport, RobustStreamProcessor

# Publisher side
transport = RedisStreamTransport(
    stream_name="agent-output",
    redis_url="redis://localhost:6379"
)

await transport.publish({"type": "token", "data": {"text": "Hello"}})
await transport.publish({"type": "token", "data": {"text": " world!"}})
await transport.publish({"type": "complete", "data": {}})

# Consumer side with automatic recovery
processor = RobustStreamProcessor(
    stream_name="agent-output",
    consumer_group="clients",
    redis_url="redis://localhost:6379"
)

async for event in processor.process():
    if event["type"] == "token":
        print(event["data"]["text"], end="")
```

> Supports consumer groups, automatic acknowledgment, and replay from any position.

### Agent Coordination

Coordinate multiple agents with Redis pub/sub and atomic operations:

```python
from redis_openai_agents import AgentCoordinator, EventType

coordinator = AgentCoordinator(
    session_id="multi-agent-session",
    redis_url="redis://localhost:6379"
)

# Agent 1: Signal handoff ready
await coordinator.publish(EventType.HANDOFF_READY, {
    "from_agent": "triage",
    "to_agent": "specialist",
    "context": {"topic": "billing"}
})

# Agent 2: Listen for handoffs
async for event in coordinator.subscribe():
    if event.type == EventType.HANDOFF_READY:
        print(f"Handoff from {event.data['from_agent']}")
```

### Middleware for the Model Call

Compose cross-cutting concerns around the agent's LLM call with an
around-style middleware protocol modelled on LangChain's `AgentMiddleware`:

```python
from agents import Agent, Runner
from agents.models.openai_responses import OpenAIResponsesModel
from openai import AsyncOpenAI

from redis_openai_agents import (
    MiddlewareStack, Route, SemanticCache, SemanticRouter,
)
from redis_openai_agents.middleware import (
    SemanticCacheMiddleware, SemanticRouterMiddleware,
)

router = SemanticRouter(
    name="support-router", redis_url="redis://localhost:6379",
    routes=[Route(name="greeting", references=["hello", "hi"])],
)
router_mw = SemanticRouterMiddleware(router=router, responses={"greeting": "Hi!"})

cache = SemanticCache(redis_url="redis://localhost:6379", similarity_threshold=0.92)
cache_mw = SemanticCacheMiddleware(cache=cache)

stack = MiddlewareStack(
    model=OpenAIResponsesModel(model="gpt-4o-mini", openai_client=AsyncOpenAI()),
    middlewares=[router_mw, cache_mw],  # outer-to-inner
)

agent = Agent(name="assistant", instructions="Be concise.", model=stack)
result = await Runner.run(agent, "hello")  # short-circuited by router
```

Ships with `SemanticCacheMiddleware`, `SemanticRouterMiddleware`, and
`ConversationMemoryMiddleware`. Write your own: any object with an async
`awrap_model_call(request, handler)` coroutine is a middleware.

### Tool Result Caching

Memoize a tool's Python callable in Redis, keyed by argument hash. Side-effect
prefixes (`send_`, `delete_`, ...) and volatile args (`timestamp`, `now`, ...)
bypass the cache automatically.

```python
from agents import function_tool
from redis_openai_agents import cached_tool


@function_tool
@cached_tool(name="lookup_company", redis_url="redis://localhost:6379", ttl=3600)
async def lookup_company(ticker: str) -> str:
    return await _hit_paid_api(ticker)
```

### Metrics & Observability

Built-in observability with RedisTimeSeries and Prometheus:

```python
from redis_openai_agents import AgentMetrics, PrometheusExporter

metrics = AgentMetrics(redis_url="redis://localhost:6379")

# Record metrics
await metrics.record_latency("agent_run", 150.5)
await metrics.record_tokens("gpt-4", input_tokens=100, output_tokens=50)
await metrics.record_cache_hit("semantic_cache")

# Get statistics
stats = await metrics.get_stats("latency", aggregation="avg", time_range="1h")

# Prometheus export (http://localhost:9090/metrics)
exporter = PrometheusExporter(metrics)
await exporter.start_server(port=9090)
```

---

## Components

### Sessions & Memory

| Component | Description |
|-----------|-------------|
| `AgentSession` | Hash-based session storage built on RedisVL MessageHistory |
| `JSONSession` | JSON document storage for complex nested session data |
| `SemanticRouter` | Vector-based intent routing without LLM calls |

### Caching & Search

| Component | Description |
|-----------|-------------|
| `SemanticCache` | Two-level cache (exact match + semantic similarity) |
| `RedisCachingModel` | Model wrapper with automatic response caching |
| `RedisVectorStore` | HNSW vector search for RAG applications |
| `RedisFullTextSearch` | BM25 full-text search with filters |
| `HybridSearchService` | Combined vector + text search with configurable weights |

### Streaming & Coordination

| Component | Description |
|-----------|-------------|
| `RedisStreamTransport` | Redis Streams-based event transport |
| `RobustStreamProcessor` | Consumer groups with automatic recovery |
| `ResumableStreamRunner` | Checkpoint-based stream resumption |
| `AgentCoordinator` | Multi-agent coordination via pub/sub |
| `AtomicOperations` | Lua script-based atomic Redis operations |

### Observability

| Component | Description |
|-----------|-------------|
| `AgentMetrics` | RedisTimeSeries metrics collection |
| `PrometheusExporter` | Prometheus metrics endpoint |
| `RedisTracingProcessor` | SDK-compatible trace storage in Redis Streams |

### SDK Integration

| Component | Description |
|-----------|-------------|
| `RedisAgentRunner` | Enhanced runner with caching and metrics |
| `RedisFileSearchTool` | Drop-in replacement for OpenAI file search |
| `RedisRateLimitGuardrail` | SDK guardrail with Redis-backed rate limiting |
| `MiddlewareStack` | Around-style middleware wrapping the SDK `Model` interface |
| `SemanticCacheMiddleware` | Cache LLM responses by input similarity |
| `SemanticRouterMiddleware` | Short-circuit matched intents with canned responses |
| `ConversationMemoryMiddleware` | Inject semantically relevant past messages |
| `cached_tool` | Decorator that memoizes a tool callable's result in Redis |

### Advanced Features

| Component | Description |
|-----------|-------------|
| `RankedOperations` | Sorted set rankings for agents and tools |
| `DeduplicationService` | Bloom filter request deduplication |
| `RedisConnectionPool` | Connection pooling with retry logic |

---

## Examples

| Example | Description |
|---------|-------------|
| [01-routing-agents](examples/01-routing-agents.ipynb) | Multi-agent routing with handoffs |
| [02-semantic-cache](examples/02-semantic-cache.ipynb) | Reduce LLM costs with caching |
| [03-vector-search](examples/03-vector-search.ipynb) | Build RAG applications |
| [04-full-text-search](examples/04-full-text-search.ipynb) | BM25 full-text search |
| [05-token-streaming](examples/05-token-streaming.ipynb) | Real-time streaming with Redis Streams |
| [06-time-series-metrics](examples/06-time-series-metrics.ipynb) | Observability with TimeSeries |
| [07-full-stack-integration](examples/07-full-stack-integration.ipynb) | Complete integration example |
| [08-runner-integration](examples/08-runner-integration.ipynb) | RedisAgentRunner usage |
| [09-hybrid-search](examples/09-hybrid-search.ipynb) | Combined vector + full-text search |
| [10-agent-ranking](examples/10-agent-ranking.ipynb) | Sorted set rankings |
| [11-deduplication](examples/11-deduplication.ipynb) | Bloom filter deduplication |
| [12-agent-coordinator](examples/12-agent-coordinator.ipynb) | Multi-agent orchestration |
| [13-robust-streaming](examples/13-robust-streaming.ipynb) | Consumer groups & recovery |
| [14-atomic-operations](examples/14-atomic-operations.ipynb) | Lua script atomicity |
| [15-semantic-router](examples/15-semantic-router.ipynb) | Intent-based routing |
| [16-middleware](examples/16-middleware.ipynb) | Cache + router + composition around the Model |
| [17-tool-caching](examples/17-tool-caching.ipynb) | `@cached_tool` for idempotent tools |

---

## Why Redis OpenAI Agents?

| Challenge | Without Redis | With Redis OpenAI Agents |
|-----------|--------------|-------------------------|
| **Session Storage** | SQLite (single-node) | Distributed Redis sessions |
| **Caching** | None or external service | Built-in semantic cache |
| **Vector Search** | Pinecone, Qdrant ($70+/mo) | Redis Vector Search (free) |
| **Streaming** | Custom WebSocket code | Redis Streams (reliable) |
| **Metrics** | Prometheus + Grafana setup | Built-in TimeSeries |
| **Total Services** | 5+ separate systems | **1 Redis deployment** |

---

## Development

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install dependencies
uv sync --all-extras --group dev

# Run tests
uv run pytest --run-api-tests

# Format and lint
make format
make lint

# Type check
make mypy

# Build documentation
make docs
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
    <p>Built with ❤️ by Redis for the OpenAI Agents SDK community</p>
</div>
