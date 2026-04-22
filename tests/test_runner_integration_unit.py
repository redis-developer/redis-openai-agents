"""Unit tests for Runner integration hooks - auto-caching and metrics."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch


# Mock classes for testing without actual OpenAI Agents SDK
@dataclass
class MockRunResult:
    """Mock result from Runner.run()."""

    final_output: str
    input_tokens: int = 100
    output_tokens: int = 50

    def to_input_list(self) -> list[dict[str, str]]:
        return [{"role": "assistant", "content": self.final_output}]


class TestCachedRun:
    """Tests for cached_run() function."""

    def test_cache_miss_calls_run_function(self) -> None:
        """When cache misses, the run function should be called."""
        from redis_openai_agents.runner import cached_run

        mock_cache = MagicMock()
        mock_cache.get.return_value = None  # Cache miss

        mock_run = MagicMock(return_value=MockRunResult(final_output="Hello!"))

        result = cached_run(
            query="What is Redis?",
            run_fn=mock_run,
            cache=mock_cache,
        )

        mock_cache.get.assert_called_once_with("What is Redis?")
        mock_run.assert_called_once()
        mock_cache.set.assert_called_once()
        assert result.final_output == "Hello!"

    def test_cache_hit_skips_run_function(self) -> None:
        """When cache hits, the run function should be skipped."""
        from redis_openai_agents.runner import CachedRunResult, cached_run

        mock_cache = MagicMock()
        mock_cache_result = MagicMock()
        mock_cache_result.response = "Cached response"
        mock_cache_result.similarity = 0.98
        mock_cache.get.return_value = mock_cache_result

        mock_run = MagicMock()

        result = cached_run(
            query="What is Redis?",
            run_fn=mock_run,
            cache=mock_cache,
        )

        mock_cache.get.assert_called_once()
        mock_run.assert_not_called()  # Should NOT be called
        mock_cache.set.assert_not_called()  # Should NOT cache again

        assert isinstance(result, CachedRunResult)
        assert result.response == "Cached response"
        assert result.cache_hit is True

    def test_response_extractor_custom(self) -> None:
        """Custom response extractor should be used when provided."""
        from redis_openai_agents.runner import cached_run

        mock_cache = MagicMock()
        mock_cache.get.return_value = None  # Cache miss

        mock_result = MockRunResult(final_output="Custom output")
        mock_run = MagicMock(return_value=mock_result)

        def custom_extractor(result: Any) -> str:
            return f"Extracted: {result.final_output}"

        cached_run(
            query="Test query",
            run_fn=mock_run,
            cache=mock_cache,
            response_extractor=custom_extractor,
        )

        # Check that set was called with extracted response
        mock_cache.set.assert_called_once()
        call_args = mock_cache.set.call_args
        assert call_args[1]["response"] == "Extracted: Custom output"


class TestWithMetrics:
    """Tests for with_metrics() decorator/wrapper."""

    def test_records_latency(self) -> None:
        """Metrics should record latency after run."""
        from redis_openai_agents.runner import with_metrics

        mock_metrics = MagicMock()

        @with_metrics(mock_metrics)
        def run_fn() -> MockRunResult:
            time.sleep(0.01)  # Small delay
            return MockRunResult(final_output="Response")

        run_fn()

        mock_metrics.record.assert_called_once()
        call_kwargs = mock_metrics.record.call_args[1]
        assert "latency_ms" in call_kwargs
        assert call_kwargs["latency_ms"] >= 10  # At least 10ms

    def test_records_cache_hit(self) -> None:
        """Metrics should record cache hit status."""
        from redis_openai_agents.runner import CachedRunResult, with_metrics

        mock_metrics = MagicMock()

        @with_metrics(mock_metrics)
        def run_fn() -> CachedRunResult:
            return CachedRunResult(response="Cached", cache_hit=True)

        run_fn()

        call_kwargs = mock_metrics.record.call_args[1]
        assert call_kwargs["cache_hit"] is True

    def test_records_token_counts_from_result(self) -> None:
        """Metrics should record token counts when available."""
        from redis_openai_agents.runner import with_metrics

        mock_metrics = MagicMock()

        @with_metrics(mock_metrics)
        def run_fn() -> MockRunResult:
            return MockRunResult(
                final_output="Response",
                input_tokens=150,
                output_tokens=75,
            )

        run_fn()

        call_kwargs = mock_metrics.record.call_args[1]
        assert call_kwargs["input_tokens"] == 150
        assert call_kwargs["output_tokens"] == 75


class TestRedisAgentRunner:
    """Tests for the RedisAgentRunner integration class."""

    def test_init_with_all_components(self) -> None:
        """Runner should accept cache, metrics, and session."""
        from redis_openai_agents.runner import RedisAgentRunner

        mock_cache = MagicMock()
        mock_metrics = MagicMock()
        mock_session = MagicMock()

        runner = RedisAgentRunner(
            cache=mock_cache,
            metrics=mock_metrics,
            session=mock_session,
        )

        assert runner.cache is mock_cache
        assert runner.metrics is mock_metrics
        assert runner.session is mock_session

    def test_init_with_optional_components(self) -> None:
        """Runner should work with just some components."""
        from redis_openai_agents.runner import RedisAgentRunner

        runner = RedisAgentRunner()

        assert runner.cache is None
        assert runner.metrics is None
        assert runner.session is None

    def test_run_with_cache_hit(self) -> None:
        """Runner.run should return cached response on cache hit."""
        from redis_openai_agents.runner import RedisAgentRunner

        mock_cache = MagicMock()
        mock_cache_result = MagicMock()
        mock_cache_result.response = "Cached answer"
        mock_cache_result.similarity = 0.99
        mock_cache.get.return_value = mock_cache_result

        mock_metrics = MagicMock()

        runner = RedisAgentRunner(cache=mock_cache, metrics=mock_metrics)

        result = runner.run(
            agent=MagicMock(),
            input="What is Redis?",
        )

        assert result.response == "Cached answer"
        assert result.cache_hit is True

        # Metrics should be recorded
        mock_metrics.record.assert_called_once()
        call_kwargs = mock_metrics.record.call_args[1]
        assert call_kwargs["cache_hit"] is True

    def test_run_with_cache_miss_calls_sdk(self) -> None:
        """Runner.run should call SDK runner on cache miss."""
        from redis_openai_agents.runner import RedisAgentRunner

        mock_cache = MagicMock()
        mock_cache.get.return_value = None  # Cache miss

        mock_metrics = MagicMock()
        mock_session = MagicMock()

        # Mock the SDK runner
        mock_sdk_result = MagicMock()
        mock_sdk_result.final_output = "SDK response"

        with patch(
            "redis_openai_agents.runner._call_sdk_runner",
            return_value=mock_sdk_result,
        ):
            runner = RedisAgentRunner(
                cache=mock_cache,
                metrics=mock_metrics,
                session=mock_session,
            )

            runner.run(
                agent=MagicMock(),
                input="What is Redis?",
            )

        # Cache should be populated
        mock_cache.set.assert_called_once()

        # Session should store result
        mock_session.store_agent_result.assert_called_once_with(mock_sdk_result)

        # Metrics recorded
        mock_metrics.record.assert_called_once()
        call_kwargs = mock_metrics.record.call_args[1]
        assert call_kwargs["cache_hit"] is False

    def test_run_without_cache_always_calls_sdk(self) -> None:
        """When no cache configured, always call SDK."""
        from redis_openai_agents.runner import RedisAgentRunner

        mock_metrics = MagicMock()

        mock_sdk_result = MagicMock()
        mock_sdk_result.final_output = "SDK response"

        with patch(
            "redis_openai_agents.runner._call_sdk_runner",
            return_value=mock_sdk_result,
        ):
            runner = RedisAgentRunner(metrics=mock_metrics)

            result = runner.run(
                agent=MagicMock(),
                input="What is Redis?",
            )

        assert result.final_output == "SDK response"
        mock_metrics.record.assert_called_once()


class TestExtractQueryFromInput:
    """Tests for extracting query string from various input formats."""

    def test_string_input(self) -> None:
        """String input should be returned as-is."""
        from redis_openai_agents.runner import extract_query_from_input

        result = extract_query_from_input("What is Redis?")
        assert result == "What is Redis?"

    def test_single_message_list(self) -> None:
        """Single message list should extract content."""
        from redis_openai_agents.runner import extract_query_from_input

        result = extract_query_from_input([{"role": "user", "content": "Hello"}])
        assert result == "Hello"

    def test_multiple_messages_uses_last_user(self) -> None:
        """Multiple messages should use last user message."""
        from redis_openai_agents.runner import extract_query_from_input

        result = extract_query_from_input(
            [
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": "First answer"},
                {"role": "user", "content": "Second question"},
            ]
        )
        assert result == "Second question"

    def test_empty_list_returns_empty(self) -> None:
        """Empty list should return empty string."""
        from redis_openai_agents.runner import extract_query_from_input

        result = extract_query_from_input([])
        assert result == ""


class TestExtractResponseFromResult:
    """Tests for extracting response string from SDK results."""

    def test_final_output_attribute(self) -> None:
        """Result with final_output attribute should extract it."""
        from redis_openai_agents.runner import extract_response_from_result

        result = MagicMock()
        result.final_output = "The response"

        extracted = extract_response_from_result(result)
        assert extracted == "The response"

    def test_string_result(self) -> None:
        """String result should be returned as-is."""
        from redis_openai_agents.runner import extract_response_from_result

        extracted = extract_response_from_result("Direct string")
        assert extracted == "Direct string"

    def test_fallback_to_str(self) -> None:
        """Unknown result type should fallback to str()."""
        from redis_openai_agents.runner import extract_response_from_result

        result = {"custom": "format"}
        extracted = extract_response_from_result(result)
        assert "custom" in extracted
