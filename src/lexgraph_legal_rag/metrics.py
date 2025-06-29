"""Prometheus metrics for monitoring search operations."""

from __future__ import annotations

import os

from prometheus_client import Counter, Histogram, start_http_server


SEARCH_REQUESTS = Counter(
    "search_requests_total", "Total number of search queries processed"
)
SEARCH_LATENCY = Histogram(
    "search_latency_seconds", "Time spent processing search queries"
)


def start_metrics_server(port: int | None = None) -> None:
    """Expose metrics on ``port`` if provided or via ``METRICS_PORT`` env var."""

    if port is None:
        port = int(os.environ.get("METRICS_PORT", "0"))
    if port:
        start_http_server(port)
