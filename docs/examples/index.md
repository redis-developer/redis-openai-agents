---
myst:
  html_meta:
    "description lang=en": |
        Examples for Redis OpenAI Agents users
---


# Example Gallery

Explore examples of Redis OpenAI Agents in action.

```{note}
If you are using Redis OpenAI Agents, please consider adding your example to this page by
opening a Pull Request on [GitHub](https://github.com/redis/redis-openai-agents)
```

## Quick Start Examples

### Session Persistence

Save, load, and continue conversations across agent runs.

```python
from agents import Agent, Runner
from redis_openai_agents import AgentSession

# Create a session
session = AgentSession.create(
    user_id="user_123",
    redis_url="redis://localhost:6379"
)

agent = Agent(name="assistant", instructions="You are a helpful assistant.")

# First conversation
result = await Runner.run(agent, input="My name is Alice")
session.store_agent_result(result)

# Later: Continue the conversation
session = AgentSession.load(
    conversation_id=session.conversation_id,
    user_id="user_123",
    redis_url="redis://localhost:6379"
)

history = session.to_agent_inputs()
result = await Runner.run(agent, input=history + [{"role": "user", "content": "What's my name?"}])
# Agent remembers: "Your name is Alice"
```

### Semantic Caching

Reduce LLM costs by caching similar queries.

```python
from redis_openai_agents import SemanticCache

cache = SemanticCache(
    redis_url="redis://localhost:6379",
    similarity_threshold=0.9,
    ttl=3600
)

# First query - cache miss, call LLM
result = cache.get(query="What is the capital of France?")
if not result:
    response = await call_llm("What is the capital of France?")
    cache.set(query="What is the capital of France?", response=response)

# Similar query - cache hit!
result = cache.get(query="What's France's capital city?")
if result:
    print(f"Cache hit: {result.response}")  # Returns cached response
```

### Agent Routing

Route queries to specialized agents without LLM calls.

```python
from redis_openai_agents import SemanticRouter, Route

router = SemanticRouter(
    name="support-router",
    redis_url="redis://localhost:6379",
    routes=[
        Route(
            name="billing",
            references=["payment", "invoice", "refund", "subscription"],
            metadata={"agent": "billing_agent"}
        ),
        Route(
            name="technical",
            references=["bug", "error", "crash", "not working"],
            metadata={"agent": "tech_agent"}
        ),
        Route(
            name="sales",
            references=["pricing", "demo", "enterprise", "upgrade"],
            metadata={"agent": "sales_agent"}
        ),
    ]
)

# Route queries to appropriate agents
match = router.route("I need help with my subscription payment")
print(f"Route to: {match.metadata['agent']}")  # billing_agent
```

### RAG with Vector Search

Build retrieval-augmented generation applications.

```python
from redis_openai_agents import RedisVectorStore

store = RedisVectorStore(
    name="knowledge-base",
    redis_url="redis://localhost:6379"
)

# Index documents
store.add_documents([
    {"content": "Redis is an in-memory data store.", "source": "docs"},
    {"content": "Vector search enables semantic similarity.", "source": "docs"},
    {"content": "Agents can use tools to accomplish tasks.", "source": "guide"},
])

# Search with metadata filtering
results = store.search(
    query="How does Redis store data?",
    k=3,
    filter={"source": "docs"}
)

# Use results in agent context
context = "\n".join([r.content for r in results])
agent = Agent(
    name="rag-agent",
    instructions=f"Answer using this context:\n{context}"
)
```

### Real-time Token Streaming

Stream tokens reliably with automatic recovery.

```python
from redis_openai_agents import RedisStreamTransport
import asyncio

# Publisher side
async def publish_tokens():
    transport = RedisStreamTransport(
        stream_name="agent-output",
        redis_url="redis://localhost:6379"
    )

    for word in ["Hello", " ", "world", "!"]:
        await transport.apublish({"type": "token", "text": word})
        await asyncio.sleep(0.1)

    await transport.apublish({"type": "complete"})

# Consumer side — read all events from the stream
async def consume_tokens():
    transport = RedisStreamTransport(
        stream_name="agent-output",
        redis_url="redis://localhost:6379",
        consumer_group="clients",
    )

    events = await transport.asubscribe(timeout_ms=5000)
    for event in events:
        if event.get("type") == "token":
            print(event.get("text", ""), end="", flush=True)
        elif event.get("type") == "complete":
            print("\nStream complete!")
```

### Multi-Agent Coordination

Orchestrate multiple agents with handoffs.

```python
from redis_openai_agents import AgentCoordinator, EventType

coordinator = AgentCoordinator(
    session_id="support-session",
    redis_url="redis://localhost:6379"
)
await coordinator.initialize()

# Triage agent signals handoff
async def triage_agent():
    await coordinator.publish_handoff_ready(
        from_agent="triage",
        to_agent="billing_specialist",
        session_id="support-session",
        context={"issue": "refund_request", "order_id": "12345"},
    )

# Specialist agent listens
async def specialist_agent():
    async for event in coordinator.subscribe():
        if event.get("event_type") == EventType.HANDOFF_READY.value:
            print(f"Received handoff: {event}")
```

### Metrics & Observability

Track agent performance with built-in metrics.

```python
from redis_openai_agents import AgentMetrics, PrometheusExporter

metrics = AgentMetrics(name="my-agent", redis_url="redis://localhost:6379")

# Record metrics during agent execution
async def run_with_metrics(agent, query):
    import time

    start = time.time()
    result = await Runner.run(agent, input=query)
    latency_ms = (time.time() - start) * 1000

    # Record latency, tokens, and cache status in one call
    await metrics.arecord(
        latency_ms=latency_ms,
        input_tokens=result.usage.input_tokens,
        output_tokens=result.usage.output_tokens,
        cache_hit=False,
    )

    return result

# Export to Prometheus
exporter = PrometheusExporter(metrics)
print(exporter.generate())  # Prometheus text format output
```

### Middleware: Cache + Router around an Agent

Wrap the agent's model call in a composable pipeline. The router
short-circuits matched intents with a canned response; the cache serves
repeat queries without a second LLM call.

```python
from agents import Agent, Runner
from agents.models.openai_responses import OpenAIResponsesModel
from openai import AsyncOpenAI

from redis_openai_agents import (
    MiddlewareStack,
    Route,
    SemanticCache,
    SemanticRouter,
)
from redis_openai_agents.middleware import (
    SemanticCacheMiddleware,
    SemanticRouterMiddleware,
)

router = SemanticRouter(
    name="support-router",
    redis_url="redis://localhost:6379",
    routes=[
        Route(name="greeting", references=["hello", "hi", "hey"]),
        Route(name="thanks", references=["thank you", "thanks"]),
    ],
)
router_mw = SemanticRouterMiddleware(
    router=router,
    responses={
        "greeting": "Hello! How can I help?",
        "thanks": "You're welcome!",
    },
)

cache = SemanticCache(redis_url="redis://localhost:6379", similarity_threshold=0.92)
cache_mw = SemanticCacheMiddleware(cache=cache)

base_model = OpenAIResponsesModel(model="gpt-4o-mini", openai_client=AsyncOpenAI())
stack = MiddlewareStack(model=base_model, middlewares=[router_mw, cache_mw])

agent = Agent(name="assistant", instructions="Be concise.", model=stack)
result = await Runner.run(agent, "hello")  # short-circuited by router
```

## Full Application Examples

See the [examples/](https://github.com/redis/redis-openai-agents/tree/main/examples) directory for complete applications:

- **Customer Support Bot** - Multi-agent system with routing, handoffs, and session persistence
- **RAG Chatbot** - Document Q&A with hybrid search and caching
- **Streaming Dashboard** - Real-time token streaming with multiple consumers
