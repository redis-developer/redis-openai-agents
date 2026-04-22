# Examples - Redis OpenAI Agents

Jupyter notebook examples demonstrating Redis integrations with OpenAI Agents SDK.

## Quick Start

### Prerequisites

- Docker and Docker Compose
- OpenAI API key

### Setup

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your OpenAI API key:
   ```
   OPENAI_API_KEY=sk-...
   ```

3. Start the notebook environment:
   ```bash
   docker compose up
   ```

4. Open your browser to http://localhost:8888

5. Redis Insight is available at http://localhost:8001

### Stopping

```bash
docker compose down
```

## Available Examples

| Notebook | Complexity | Topics |
|----------|------------|--------|
| **01-routing-agents** | Medium | Multi-agent systems, handoffs, streaming |
| **02-semantic-cache** | Low | LLM response caching, semantic similarity |
| **03-vector-search** | Medium | Vector embeddings, RAG, similarity search |
| **04-full-text-search** | Low | BM25 search, text indexing |
| **05-token-streaming** | Medium | Redis Streams, real-time tokens |
| **06-time-series-metrics** | Low | RedisTimeSeries, metrics collection |
| **07-full-stack-integration** | High | Complete integration of all components |
| **08-runner-integration** | Medium | RedisAgentRunner, caching, metrics |
| **09-hybrid-search** | Medium | Combined vector + full-text search |
| **10-agent-ranking** | Low | Sorted sets, leaderboards |
| **11-deduplication** | Low | Bloom filters, request deduplication |
| **12-agent-coordinator** | Medium | Multi-agent state, pub/sub coordination |
| **13-robust-streaming** | Medium | Fault-tolerant stream processing |
| **14-atomic-operations** | Medium | Lua scripts, atomic Redis operations |
| **15-semantic-router** | Medium | Intent classification, query routing |
| **16-middleware** | Medium | Around-style middleware: cache, router, composition |
| **17-tool-caching** | Low | `@cached_tool` decorator for idempotent tools |

### Notebook Descriptions

**01-routing-agents** - Demonstrates the routing pattern with language-specific agents. Triage agent routes requests to French/Spanish/English agents with streaming responses and conversation continuity.

**02-semantic-cache** - Shows 2-level caching (exact match + semantic similarity) to reduce LLM API calls and latency.

**03-vector-search** - Implements RAG (Retrieval Augmented Generation) using RedisVL for vector similarity search.

**04-full-text-search** - BM25 lexical search for keyword-based retrieval using RediSearch.

**05-token-streaming** - Real-time token streaming via Redis Streams with consumer groups and replay capability.

**06-time-series-metrics** - Metrics collection and visualization using RedisTimeSeries.

**07-full-stack-integration** - Complete example combining sessions, caching, search, streaming, and metrics.

**08-runner-integration** - RedisAgentRunner wrapper with automatic caching and metrics collection.

**09-hybrid-search** - Combined vector and full-text search with configurable weights.

**10-agent-ranking** - Sorted set-based ranking for agent performance leaderboards.

**11-deduplication** - Bloom filter-based request deduplication to prevent duplicate processing.

**12-agent-coordinator** - Multi-agent state coordination with Redis pub/sub.

**13-robust-streaming** - Fault-tolerant stream processing with automatic recovery.

**14-atomic-operations** - Lua script-based atomic operations for complex Redis transactions.

**15-semantic-router** - Intent classification using vector similarity search. Route queries to specialized agents based on semantic matching to reference phrases.

**16-middleware** - Around-style middleware that wraps the model call. Composes a `SemanticRouterMiddleware` (short-circuits matched intents with canned responses) and a `SemanticCacheMiddleware` (serves repeat queries from Redis) into a single `MiddlewareStack` that plugs directly into `Agent(model=...)`. See also `ConversationMemoryMiddleware` for semantic history injection.

**17-tool-caching** - Demonstrates `@cached_tool`, a decorator that memoises a tool's Python callable in Redis. Covers sync and async tools, volatile-argument bypass, ignored arguments, and side-effect prefix exclusions. Tool caching is a separate primitive from model middleware because the SDK does not route tool execution through the `Model` interface.

## Architecture

```
┌─────────────────────────────────────────┐
│  Jupyter Notebook (localhost:8888)      │
│  - Python 3.12                          │
│  - OpenAI Agents SDK                    │
│  - redis-openai-agents (editable)       │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│  Redis 8 (localhost:6379)               │
│  - Search & Query                       │
│  - JSON                                 │
│  - Time Series                          │
│  - Bloom filters                        │
└─────────────────────────────────────────┘
```

## Development

The `redis-openai-agents` library is mounted as an editable install, so changes to the library source code are immediately available in notebooks without rebuilding.

### Rebuilding After Dockerfile Changes

```bash
docker compose up --build
```

### Viewing Logs

```bash
docker compose logs -f jupyter
```

### Accessing Redis CLI

```bash
docker compose exec redis redis-cli
```

## Troubleshooting

### "OPENAI_API_KEY not found"

Make sure you've created a `.env` file with your API key:
```bash
cp .env.example .env
# Edit .env and add your key
```

### Jupyter won't start

Check logs:
```bash
docker compose logs jupyter
```

Rebuild if needed:
```bash
docker compose down
docker compose up --build
```

### Can't connect to Redis

Verify Redis is healthy:
```bash
docker compose ps
docker compose exec redis redis-cli ping
```

Should return `PONG`.

## Resources

- [OpenAI Agents SDK Documentation](https://openai.github.io/openai-agents-python/)
- [Redis Documentation](https://redis.io/docs/)
- [RedisVL Documentation](https://redisvl.com/)
- [Project Documentation](../docs/)
