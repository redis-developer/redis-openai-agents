"""Prometheus metrics export for AgentMetrics.

This module provides Prometheus-compatible metrics export,
allowing integration with Prometheus/Grafana monitoring.

Features:
- Prometheus text format export
- Standard metric types (counter, gauge)
- HTTP handler for /metrics endpoint
- Integration with AgentMetrics
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .metrics import AgentMetrics


class PrometheusExporter:
    """Exports AgentMetrics in Prometheus text format.

    Generates Prometheus-compatible text format metrics that can
    be scraped by Prometheus or other monitoring systems.

    Example:
        >>> from redis_openai_agents import AgentMetrics
        >>> from redis_openai_agents.prometheus import PrometheusExporter
        >>>
        >>> metrics = AgentMetrics(name="my_agent")
        >>> exporter = PrometheusExporter(metrics)
        >>> print(exporter.generate())

    Args:
        metrics: AgentMetrics instance to export
        prefix: Metric name prefix (default: "agent")
    """

    def __init__(
        self,
        metrics: AgentMetrics,
        prefix: str = "agent",
    ) -> None:
        """Initialize the exporter.

        Args:
            metrics: AgentMetrics instance to export
            prefix: Metric name prefix
        """
        self.metrics = metrics
        self.prefix = prefix

    def generate(self) -> str:
        """Generate Prometheus text format output.

        Returns:
            Prometheus text format string
        """
        stats = self.metrics.get_stats()
        agent_name = self.metrics._name

        lines = []

        # Request counter
        lines.extend(
            [
                f"# HELP {self.prefix}_requests_total Total number of agent requests",
                f"# TYPE {self.prefix}_requests_total counter",
                f'{self.prefix}_requests_total{{agent="{agent_name}"}} {stats["count"]}',
            ]
        )

        # Latency metrics
        lines.extend(
            [
                f"# HELP {self.prefix}_latency_milliseconds Request latency in milliseconds",
                f"# TYPE {self.prefix}_latency_milliseconds gauge",
                f'{self.prefix}_latency_milliseconds{{agent="{agent_name}",stat="avg"}} {stats["latency_avg"]}',
                f'{self.prefix}_latency_milliseconds{{agent="{agent_name}",stat="min"}} {stats["latency_min"]}',
                f'{self.prefix}_latency_milliseconds{{agent="{agent_name}",stat="max"}} {stats["latency_max"]}',
            ]
        )

        # Token counters
        lines.extend(
            [
                f"# HELP {self.prefix}_tokens_total Total tokens processed",
                f"# TYPE {self.prefix}_tokens_total counter",
                f'{self.prefix}_tokens_total{{agent="{agent_name}",type="input"}} {int(stats["input_tokens_sum"])}',
                f'{self.prefix}_tokens_total{{agent="{agent_name}",type="output"}} {int(stats["output_tokens_sum"])}',
            ]
        )

        # Cache hit ratio
        lines.extend(
            [
                f"# HELP {self.prefix}_cache_hit_ratio Cache hit ratio (0.0 to 1.0)",
                f"# TYPE {self.prefix}_cache_hit_ratio gauge",
                f'{self.prefix}_cache_hit_ratio{{agent="{agent_name}"}} {stats["cache_hit_rate"]}',
            ]
        )

        return "\n".join(lines) + "\n"


def create_metrics_handler(
    metrics: AgentMetrics,
    prefix: str = "agent",
) -> Callable[[], tuple[str, str]]:
    """Create a handler function for /metrics HTTP endpoint.

    The handler returns Prometheus text format with appropriate content type.

    Example:
        >>> from http.server import HTTPServer, BaseHTTPRequestHandler
        >>> handler = create_metrics_handler(metrics)
        >>>
        >>> class MetricsHandler(BaseHTTPRequestHandler):
        ...     def do_GET(self):
        ...         content, content_type = handler()
        ...         self.send_response(200)
        ...         self.send_header("Content-Type", content_type)
        ...         self.end_headers()
        ...         self.wfile.write(content.encode())

    Args:
        metrics: AgentMetrics instance
        prefix: Metric name prefix

    Returns:
        Function that returns (content, content_type) tuple
    """
    exporter = PrometheusExporter(metrics, prefix=prefix)

    def handler() -> tuple[str, str]:
        content = exporter.generate()
        content_type = "text/plain; version=0.0.4; charset=utf-8"
        return content, content_type

    return handler


def start_metrics_server(
    metrics: AgentMetrics,
    port: int = 9090,
    prefix: str = "agent",
) -> None:
    """Start a simple HTTP server for Prometheus scraping.

    Starts a basic HTTP server that exposes /metrics endpoint.
    For production use, integrate with your existing web framework.

    Args:
        metrics: AgentMetrics instance
        port: Port to listen on
        prefix: Metric name prefix
    """
    from http.server import BaseHTTPRequestHandler, HTTPServer

    handler_fn = create_metrics_handler(metrics, prefix=prefix)

    class MetricsHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/metrics":
                content, content_type = handler_fn()
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.end_headers()
                self.wfile.write(content.encode())
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            # Suppress request logging
            pass

    server = HTTPServer(("", port), MetricsHandler)
    logger.info("Prometheus metrics server started on port %d", port)
    server.serve_forever()
