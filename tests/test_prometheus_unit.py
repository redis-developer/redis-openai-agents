"""Unit tests for Prometheus metrics export."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch


def _make_mock_metrics(
    name: str = "test",
    stats: dict[str, Any] | None = None,
) -> Any:
    """Create a mock AgentMetrics with given stats."""
    from redis_openai_agents import AgentMetrics

    default_stats = {
        "count": 100,
        "latency_avg": 100.0,
        "latency_min": 50.0,
        "latency_max": 200.0,
        "input_tokens_sum": 1000,
        "output_tokens_sum": 500,
        "cache_hit_rate": 0.5,
    }
    if stats:
        default_stats.update(stats)

    with patch.object(AgentMetrics, "__init__", return_value=None):
        metrics = AgentMetrics.__new__(AgentMetrics)
        metrics._name = name
        metrics.get_stats = MagicMock(return_value=default_stats)
    return metrics


class TestPrometheusExporter:
    """Tests for PrometheusExporter class."""

    def test_create_exporter(self) -> None:
        """Exporter should be created with metrics instance."""
        from redis_openai_agents.prometheus import PrometheusExporter

        metrics = _make_mock_metrics("test_agent", {"count": 100, "cache_hit_rate": 0.25})
        exporter = PrometheusExporter(metrics)

        assert exporter.metrics is metrics

    def test_generate_text_format(self) -> None:
        """Should generate Prometheus text format."""
        from redis_openai_agents.prometheus import PrometheusExporter

        metrics = _make_mock_metrics("test_agent")
        exporter = PrometheusExporter(metrics)
        output = exporter.generate()

        assert "# HELP" in output
        assert "# TYPE" in output
        assert "agent_requests_total" in output
        assert "agent_latency_milliseconds" in output
        assert "agent_tokens_total" in output
        assert "agent_cache_hit_ratio" in output

    def test_includes_agent_label(self) -> None:
        """Output should include agent name as label."""
        from redis_openai_agents.prometheus import PrometheusExporter

        metrics = _make_mock_metrics("my_special_agent", {"count": 50})
        exporter = PrometheusExporter(metrics)
        output = exporter.generate()

        assert 'agent="my_special_agent"' in output

    def test_values_are_correct(self) -> None:
        """Metric values should match stats."""
        from redis_openai_agents.prometheus import PrometheusExporter

        metrics = _make_mock_metrics(
            stats={
                "count": 42,
                "latency_avg": 123.45,
                "latency_min": 10.0,
                "latency_max": 500.0,
                "input_tokens_sum": 1000,
                "output_tokens_sum": 500,
                "cache_hit_rate": 0.75,
            }
        )
        exporter = PrometheusExporter(metrics)
        output = exporter.generate()

        # Check values appear on their expected metric lines
        lines = output.strip().split("\n")
        metric_lines = [line for line in lines if not line.startswith("#")]
        metric_text = {line.split("{")[0].strip(): line for line in metric_lines if line.strip()}
        assert "42" in metric_text.get("agent_requests_total", "")
        assert "0.75" in metric_text.get("agent_cache_hit_ratio", "")
        # Token lines should contain the expected values
        token_lines = [line for line in metric_lines if "agent_tokens_total" in line]
        token_text = " ".join(token_lines)
        assert "1000" in token_text
        assert "500" in token_text


class TestPrometheusMetricTypes:
    """Tests for proper Prometheus metric types."""

    def test_counter_type_for_totals(self) -> None:
        """Totals should be typed as counter."""
        from redis_openai_agents.prometheus import PrometheusExporter

        metrics = _make_mock_metrics()
        exporter = PrometheusExporter(metrics)
        output = exporter.generate()

        assert "# TYPE agent_requests_total counter" in output
        assert "# TYPE agent_tokens_total counter" in output

    def test_gauge_type_for_ratios(self) -> None:
        """Ratios should be typed as gauge."""
        from redis_openai_agents.prometheus import PrometheusExporter

        metrics = _make_mock_metrics()
        exporter = PrometheusExporter(metrics)
        output = exporter.generate()

        assert "# TYPE agent_cache_hit_ratio gauge" in output


class TestPrometheusHTTPHandler:
    """Tests for HTTP handler for /metrics endpoint."""

    def test_handler_returns_text_format(self) -> None:
        """Handler should return text/plain content type."""
        from redis_openai_agents.prometheus import create_metrics_handler

        metrics = _make_mock_metrics()
        handler = create_metrics_handler(metrics)
        content, content_type = handler()

        assert content_type == "text/plain; version=0.0.4; charset=utf-8"
        assert "agent_requests_total" in content
