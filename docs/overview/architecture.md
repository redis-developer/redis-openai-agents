---
myst:
  html_meta:
    "description lang=en": |
      Architecture overview for Redis OpenAI Agents
---

# Architecture

Redis OpenAI Agents provides a unified Redis-powered infrastructure layer for the OpenAI Agents SDK. This document explains the core architecture and how the components work together.

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Your Application                              │
├─────────────────────────────────────────────────────────────────────┤
│                      OpenAI Agents SDK                               │
│   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐               │
│   │  Agent  │  │  Tools  │  │Guardrails│  │ Handoffs│               │
│   └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘               │
├────────┼────────────┼───────────┼────────────┼─────────────────────┤
│        │   Redis OpenAI Agents Integration Layer                    │
│   ┌────▼────────────▼───────────▼────────────▼────┐                │
│   │              Redis OpenAI Agents               │                │
│   │  ┌──────────┐ ┌──────────┐ ┌──────────┐      │                │
│   │  │ Sessions │ │  Cache   │ │ Routing  │      │                │
│   │  └────┬─────┘ └────┬─────┘ └────┬─────┘      │                │
│   │  ┌────▼─────┐ ┌────▼─────┐ ┌────▼─────┐      │                │
│   │  │ Streams  │ │  Search  │ │ Metrics  │      │                │
│   │  └────┬─────┘ └────┬─────┘ └────┬─────┘      │                │
│   └───────┼────────────┼────────────┼────────────┘                │
├───────────┼────────────┼────────────┼─────────────────────────────┤
│           │            │            │                              │
│   ┌───────▼────────────▼────────────▼───────┐                     │
│   │                 Redis 8                  │                     │
│   │  ┌─────────┐ ┌─────────┐ ┌─────────────┐│                     │
│   │  │  JSON   │ │ Search  │ │ TimeSeries  ││                     │
│   │  │  Hash   │ │ Vector  │ │   Streams   ││                     │
│   │  └─────────┘ └─────────┘ └─────────────┘│                     │
│   └─────────────────────────────────────────┘                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Component Categories

### Sessions & Memory

Store and retrieve conversation history across distributed systems.

| Component | Redis Feature | Purpose |
|-----------|---------------|---------|
| `AgentSession` | Hash + MessageHistory | Persistent conversation storage |
| `JSONSession` | JSON | Complex nested session data |

**How it works:**
1. `AgentSession` wraps RedisVL's `MessageHistory` for SDK-compatible storage
2. Messages are stored with timestamps for ordering
3. Sessions support TTL for automatic cleanup
4. Load/save operations are atomic

### Caching & Search

Reduce costs and latency with intelligent caching and fast retrieval.

| Component | Redis Feature | Purpose |
|-----------|---------------|---------|
| `SemanticCache` | Hash + Vector Search | Two-level response caching |
| `RedisCachingModel` | SemanticCache | Model wrapper with caching |
| `RedisVectorStore` | Vector Search (HNSW) | RAG vector storage |
| `RedisFullTextSearch` | FT.SEARCH (BM25) | Keyword retrieval |
| `HybridSearchService` | Vector + BM25 | Combined search |

**Two-Level Cache Architecture:**
```
Query → Exact Hash Match (L1) → Semantic Vector Match (L2) → LLM
         ↓ hit                    ↓ hit
       Return                   Return
```

### Streaming & Coordination

Reliable event streaming and multi-agent orchestration.

| Component | Redis Feature | Purpose |
|-----------|---------------|---------|
| `RedisStreamTransport` | Streams | Event publishing |
| `RobustStreamProcessor` | Streams + Consumer Groups | Reliable consumption |
| `ResumableStreamRunner` | Streams + Checkpoints | Stream resumption |
| `AgentCoordinator` | Pub/Sub + Streams | Multi-agent coordination |

**Stream Processing Flow:**
```
Producer → Redis Stream → Consumer Group → Multiple Consumers
              ↓                               ↓
         Persistence                    Acknowledgment
              ↓                               ↓
           Replay                        Recovery
```

### Routing

Route queries to appropriate agents without LLM calls.

| Component | Redis Feature | Purpose |
|-----------|---------------|---------|
| `SemanticRouter` | Vector Search | Intent-based routing |
| `Route` | Vector Index | Route definitions |
| `RouteMatch` | Query Result | Matched route with metadata |

**Routing Flow:**
```
Query → Embed → Vector Search → Best Match → Agent
                    ↓
              Route Metadata
```

### Observability

Built-in metrics and tracing.

| Component | Redis Feature | Purpose |
|-----------|---------------|---------|
| `AgentMetrics` | TimeSeries | Metric collection |
| `PrometheusExporter` | TimeSeries | Prometheus export |
| `RedisTracingProcessor` | Streams | Trace storage |

### SDK Integration

Direct integrations with OpenAI Agents SDK patterns.

| Component | SDK Pattern | Purpose |
|-----------|-------------|---------|
| `RedisAgentRunner` | Runner | Enhanced runner with caching |
| `RedisFileSearchTool` | Tool | Vector-backed file search |
| `RedisRateLimitGuardrail` | Guardrail | Rate limiting |

## Data Flow Examples

### Session Persistence

```python
# 1. Create session
session = AgentSession.create(user_id="user_123", redis_url="redis://...")

# 2. Run agent
result = await Runner.run(agent, input="Hello")

# 3. Store result (atomic write to Redis Hash)
session.store_agent_result(result)

# 4. Later: Load session
session = AgentSession.load(conversation_id=..., redis_url="redis://...")

# 5. Get history in SDK format
history = session.to_agent_inputs()
```

### Cached Agent Execution

```python
# 1. Check cache
cache = SemanticCache(redis_url="redis://...")
cached = cache.check(query="What is Redis?")

if cached:
    # 2a. Cache hit - return immediately
    return cached.response
else:
    # 2b. Cache miss - call LLM
    result = await Runner.run(agent, input="What is Redis?")

    # 3. Store in cache
    cache.store(query="What is Redis?", response=result.output)
    return result.output
```

### Multi-Agent Coordination

```python
# Agent 1: Triage
coordinator = AgentCoordinator(session_id="session_123", redis_url="redis://...")

# Determine handoff target
await coordinator.publish(EventType.HANDOFF_READY, {
    "from_agent": "triage",
    "to_agent": "specialist",
    "context": {"topic": "billing"}
})

# Agent 2: Specialist (listening)
async for event in coordinator.subscribe():
    if event.type == EventType.HANDOFF_READY:
        # Handle handoff
        await process_handoff(event.data)
```

## Redis Features Used

| Feature | Components | Purpose |
|---------|------------|---------|
| **Hash** | Sessions, Cache L1 | Fast key-value storage |
| **JSON** | JSONSession | Complex nested data |
| **Vector Search** | Cache L2, Router, VectorStore | Semantic similarity |
| **FT.SEARCH** | FullTextSearch, HybridSearch | BM25 text search |
| **Streams** | Transport, Processor, Tracing | Event streaming |
| **TimeSeries** | Metrics, Prometheus | Time-based metrics |
| **Pub/Sub** | Coordinator | Real-time events |
| **Sorted Sets** | Rankings | Leaderboards |
| **Bloom Filter** | Deduplication | Probabilistic dedup |

## Performance Characteristics

| Operation | Typical Latency | Notes |
|-----------|----------------|-------|
| Session Load | < 5ms | Hash read |
| Cache L1 Check | < 1ms | Exact hash lookup |
| Cache L2 Check | < 10ms | Vector search |
| Router Match | < 10ms | Vector search |
| Stream Publish | < 1ms | Async write |
| Metrics Write | < 1ms | TimeSeries insert |

## Scaling Considerations

### Horizontal Scaling
- **Sessions**: Use Redis Cluster for sharding by conversation_id
- **Cache**: Vector indices can be distributed across shards
- **Streams**: Consumer groups enable parallel processing

### High Availability
- **Sentinel**: Automatic failover support
- **Cluster**: Built-in replication
- **Streams**: Durable with acknowledgment

### Connection Management
- Use `RedisConnectionPool` for connection reuse
- Configure `RetryConfig` for transient failures
- Set appropriate TTLs to manage memory
