"""Pytest configuration and fixtures for redis-openai-agents tests."""

import logging
import os
import socket
import time
from collections.abc import Generator

import pytest
from dotenv import load_dotenv
from redis import Redis
from redis.asyncio import Redis as AsyncRedis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()


def _get_env_redis_url() -> str | None:
    """Check for environment-provided Redis URL (e.g., from DevContainer)."""
    # DevContainer provides Redis on redis:6379 (internal) or localhost:16379 (host)
    env_url = os.environ.get("REDIS_URL")
    if env_url:
        return env_url

    # Check if DevContainer Redis is available
    for host, port in [("redis", 6379), ("localhost", 16379)]:
        try:
            with socket.create_connection((host, port), timeout=1):
                return f"redis://{host}:{port}"
        except OSError:
            continue

    return None


@pytest.fixture(autouse=True)
def set_tokenizers_parallelism() -> None:
    """Disable tokenizers parallelism in tests to avoid deadlocks."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


@pytest.fixture(scope="session")
def redis_url() -> Generator[str, None, None]:
    """
    Get Redis connection URL for testing.

    Priority:
    1. REDIS_URL environment variable
    2. DevContainer Redis (redis:6379 or localhost:16379)
    3. Start testcontainers Redis

    Returns:
        Redis connection URL (e.g., "redis://localhost:6379")
    """
    # Try environment/DevContainer Redis first
    env_url = _get_env_redis_url()
    if env_url:
        yield env_url
        return

    # Fall back to testcontainers
    try:
        from testcontainers.redis import RedisContainer

        container = RedisContainer("redis:8")
        container.start()

        # Wait for Redis to be ready
        host = container.get_container_host_ip()
        port = container.get_exposed_port(6379)
        url = f"redis://{host}:{port}"

        deadline = time.time() + 15
        while time.time() < deadline:
            try:
                with socket.create_connection((host, int(port)), timeout=1):
                    break
            except OSError:
                time.sleep(0.5)
        else:
            pytest.skip("Redis container failed to become ready")

        yield url

        container.stop()
    except Exception as e:
        pytest.skip(f"Could not start Redis container: {e}")


@pytest.fixture
def redis_client(redis_url: str) -> Redis:
    """
    Create a sync Redis client for direct inspection.

    Args:
        redis_url: Redis connection URL from redis_url fixture

    Yields:
        Redis client instance
    """
    client = Redis.from_url(redis_url)
    yield client
    client.close()


@pytest.fixture
async def async_redis_client(redis_url: str) -> AsyncRedis:
    """
    Create an async Redis client for async operations.

    Args:
        redis_url: Redis connection URL from redis_url fixture

    Yields:
        Async Redis client instance
    """
    client = AsyncRedis.from_url(redis_url)
    yield client
    await client.aclose()


@pytest.fixture(autouse=True)
async def clear_redis(redis_url: str) -> None:
    """
    Clear Redis before each test.

    This ensures test isolation by flushing all data between tests.
    """
    try:
        client = AsyncRedis.from_url(redis_url)
        await client.flushall()
        await client.aclose()
    except (ConnectionError, OSError, RedisError) as exc:
        logger.warning("Failed to flush Redis before test: %s", exc)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add custom pytest command line options."""
    parser.addoption(
        "--run-api-tests",
        action="store_true",
        default=False,
        help="Run tests that require API keys (OpenAI, etc.)",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "requires_api_keys: mark test as requiring API keys (OpenAI, etc.)"
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip tests marked with requires_api_keys unless --run-api-tests is passed."""
    if config.getoption("--run-api-tests"):
        return

    skip_api = pytest.mark.skip(
        reason="Skipping test because API keys are not provided. Use --run-api-tests to run these tests."
    )

    for item in items:
        if item.get_closest_marker("requires_api_keys"):
            item.add_marker(skip_api)


# ---------------------------------------------------------------------------
# Shared mock objects for tracing tests
# ---------------------------------------------------------------------------


class MockTrace:
    """Mock trace object for testing."""

    def __init__(self, trace_id: str, name: str) -> None:
        self.trace_id = trace_id
        self.name = name
        self.started_at = time.time()
        self.completed_at: float | None = None
        self.error: str | None = None


class MockSpanData:
    """Mock span data object for testing."""

    def __init__(
        self,
        type: str,
        name: str,
        input: dict | None = None,
        output: dict | None = None,
    ) -> None:
        self.type = type
        self._name = name
        self._input = input or {}
        self._output = output

    def export(self) -> dict:
        return {
            "type": self.type,
            "name": self._name,
            "input": self._input,
            "output": self._output,
        }


class MockSpan:
    """Mock span object for testing."""

    def __init__(
        self,
        trace_id: str,
        span_id: str,
        name: str,
        span_type: str = "function",
        parent_id: str | None = None,
    ) -> None:
        self.trace_id = trace_id
        self.span_id = span_id
        self.parent_id = parent_id
        self.name = name
        self.started_at = time.time()
        self.finished_at: float | None = None
        self.error: str | None = None
        self.span_data = MockSpanData(type=span_type, name=name)
