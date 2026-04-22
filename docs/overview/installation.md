---
myst:
  html_meta:
    "description lang=en": |
      Installation instructions for Redis OpenAI Agents
---

# Install Redis OpenAI Agents

There are a few ways to install Redis OpenAI Agents. The easiest way is to use pip.

## Install with Pip

Install `redis-openai-agents` into your Python (>=3.10) environment using `pip`:

```bash
$ pip install -U redis-openai-agents
```

Redis OpenAI Agents comes with a few dependencies that are automatically installed, however, a few dependencies are optional:

```bash
$ pip install redis-openai-agents[all]  # install all optional dependencies
$ pip install redis-openai-agents[dev]  # install dev dependencies
```

If you use ZSH, remember to escape the brackets:

```bash
$ pip install redis-openai-agents\[all\]
```

## Install from Source

To install Redis OpenAI Agents from source, clone the repository and install using `uv`:

```bash
$ git clone https://github.com/redis/redis-openai-agents.git && cd redis-openai-agents
$ uv sync --all-extras --group dev
```

For a standard pip installation:

```bash
$ pip install -e .
```

## Installing Redis

Redis OpenAI Agents requires a distribution of Redis that supports [Search and Query](https://redis.io/docs/interact/search-and-query/) and optionally Time Series. The official `redis:8` Docker image includes both, along with JSON and Bloom filters. Three deployment options:

1. [Redis Cloud](https://redis.io/cloud), a fully managed cloud offering
2. The `redis:8` Docker image for local development and testing
3. [Redis Enterprise](https://redis.com/redis-enterprise/), a commercial self-hosted offering

### Redis Cloud

Redis Cloud is the easiest way to get started. You can sign up for a free account [here](https://redis.io/cloud). Make sure to enable:
- **Search and Query** - Required for semantic caching, routing, and vector search
- **TimeSeries** - Optional, for metrics collection

### Redis 8 (local development)

For local development and testing, the official `redis:8` image provides all required features:

```bash
docker run -d --name redis -p 6379:6379 redis:8
```

Want a GUI? Run [Redis Insight](https://redis.io/insight/) separately:

```bash
docker run -d --name redisinsight -p 5540:5540 redis/redisinsight:latest
```

Then open `http://localhost:5540`.

### DevContainer Setup

For a complete development environment, use the provided DevContainer configuration:

```bash
# Clone the repository
git clone https://github.com/redis/redis-openai-agents.git
cd redis-openai-agents

# Open in VS Code with DevContainers extension
code .
# Then: Cmd/Ctrl+Shift+P -> "Dev Containers: Reopen in Container"
```

The DevContainer includes:
- Python 3.11 with all dependencies
- Redis 8 (on host port 16379)
- Redis CLI tools
- Pre-configured environment

### Redis Enterprise (self-hosted)

Redis Enterprise is a commercial offering that can be self-hosted. Download the latest version [here](https://redis.io/downloads/).

For Kubernetes deployments, use the [Redis Enterprise Operator](https://docs.redis.com/latest/kubernetes/).

### Redis Sentinel

For high availability deployments, Redis OpenAI Agents supports connecting through Sentinel. Use the `redis+sentinel://` URL scheme:

```python
from redis_openai_agents import AgentSession

# Connect via Sentinel
# Format: redis+sentinel://[username:password@]host1:port1,host2:port2/service_name[/db]
session = AgentSession.create(
    user_id="user_123",
    redis_url="redis+sentinel://sentinel1:26379,sentinel2:26379/mymaster"
)
```

The Sentinel URL format supports:
- Multiple sentinel hosts (comma-separated)
- Optional authentication (username:password)
- Service name (required - the name of the Redis master)
- Optional database number (defaults to 0)

## Verifying Installation

After installation, verify everything is working:

```python
from redis_openai_agents import AgentSession, SemanticCache

# Test connection
session = AgentSession.create(
    user_id="test",
    redis_url="redis://localhost:6379"
)
print(f"Session created: {session.conversation_id}")

# Clean up
session.clear()
```

## Dependencies

Redis OpenAI Agents depends on:
- [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) - Core agent framework
- [RedisVL](https://github.com/redis/redis-vl-python) - Redis vector library
- [redis-py](https://github.com/redis/redis-py) - Redis Python client

Optional dependencies:
- [prometheus-client](https://github.com/prometheus/client_python) - For Prometheus metrics export
