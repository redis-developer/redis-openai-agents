---
myst:
  html_meta:
    "description lang=en": |
      Top-level documentation for Redis OpenAI Agents, with links to the rest
      of the site.
html_theme.sidebar_secondary.remove: false
---

# Redis OpenAI Agents

Production-ready Redis integrations for the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python). Replace 5+ separate systems with a single Redis deployment.

```{gallery-grid}
:grid-columns: 1 2 2 3

- header: "{fas}`database;pst-color-primary` Session Storage"
  content: "Persistent, distributed conversation storage with AgentSession and JSONSession."
  link: "user_guide/01_sessions"
- header: "{fas}`bolt;pst-color-primary` Semantic Caching"
  content: "Reduce LLM costs by 25%+ with two-level semantic caching."
  link: "user_guide/02_semantic_cache"
- header: "{fas}`route;pst-color-primary` Semantic Routing"
  content: "Route queries to agents using vector similarity - no LLM calls required."
  link: "user_guide/03_semantic_router"
- header: "{fas}`magnifying-glass;pst-color-primary` Vector Search"
  content: "Build RAG applications with Redis vector search (HNSW)."
  link: "user_guide/04_vector_search"
- header: "{fas}`stream;pst-color-primary` Token Streaming"
  content: "Reliable, replayable token streaming via Redis Streams."
  link: "user_guide/05_streaming"
- header: "{fas}`chart-line;pst-color-primary` Observability"
  content: "Built-in metrics with RedisTimeSeries and Prometheus."
  link: "user_guide/06_metrics"
- header: "{fas}`layer-group;pst-color-primary` Middleware"
  content: "Around-style middleware for the Agents SDK: cache, router, composition."
  link: "user_guide/09_middleware"
```

## Installation

Install `redis-openai-agents` into your Python (>=3.10) environment using `pip`:

```bash
pip install redis-openai-agents
```

Then make sure to have [Redis](https://redis.io) accessible with Search & Query features enabled on [Redis Cloud](https://redis.io/cloud) or locally in docker. The official `redis:8` image includes Search, JSON, Time Series, and Bloom filters:

```bash
docker run -d --name redis -p 6379:6379 redis:8
```

For a GUI, run [Redis Insight](https://redis.io/insight/) separately: `docker run -d --name redisinsight -p 5540:5540 redis/redisinsight:latest`.


## Quick Start

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
```

### Semantic Caching

Reduce LLM costs by caching responses for similar queries:

```python
from redis_openai_agents import SemanticCache

cache = SemanticCache(
    redis_url="redis://localhost:6379",
    similarity_threshold=0.9,
    ttl=3600
)

# Check cache before calling LLM
result = cache.get(query="What is the capital of France?")
if result:
    print(f"Cache hit: {result.response}")
else:
    response = "Paris is the capital of France."
    cache.set(query="What is the capital of France?", response=response)
```

### Semantic Routing

Route queries to the appropriate agent using vector similarity:

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

## Why Redis OpenAI Agents?

| Challenge | Without Redis | With Redis OpenAI Agents |
|-----------|--------------|-------------------------|
| **Session Storage** | SQLite (single-node) | Distributed Redis sessions |
| **Caching** | None or external service | Built-in semantic cache |
| **Vector Search** | Pinecone, Qdrant ($70+/mo) | Redis Vector Search (free) |
| **Streaming** | Custom WebSocket code | Redis Streams (reliable) |
| **Metrics** | Prometheus + Grafana setup | Built-in TimeSeries |
| **Total Services** | 5+ separate systems | **1 Redis deployment** |


## Table of Contents

```{toctree}
:maxdepth: 2

Overview <overview/index>
API <api/index>
User Guides <user_guide/index>
Example Gallery <examples/index>
```

```{toctree}
:hidden:

Changelog <https://github.com/redis/redis-openai-agents/releases>
```
