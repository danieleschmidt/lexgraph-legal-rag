"""Prometheus metrics for monitoring search operations."""

from __future__ import annotations

import os

from prometheus_client import Counter, Histogram, Gauge, Info, start_http_server


# Search metrics
SEARCH_REQUESTS = Counter(
    "search_requests_total", "Total number of search queries processed",
    ["search_type"]  # semantic, vector, cached
)
SEARCH_LATENCY = Histogram(
    "search_latency_seconds", "Time spent processing search queries",
    ["search_type"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

# Cache metrics
CACHE_HITS = Counter(
    "cache_hits_total", "Total number of cache hits"
)
CACHE_MISSES = Counter(
    "cache_misses_total", "Total number of cache misses"
)
CACHE_SIZE = Gauge(
    "cache_size_entries", "Current number of entries in cache"
)
CACHE_HIT_RATE = Gauge(
    "cache_hit_rate", "Cache hit rate (0-1)"
)

# Index metrics
INDEX_SIZE = Gauge(
    "index_size_documents", "Number of documents in the index"
)
INDEX_POOL_AVAILABLE = Gauge(
    "index_pool_available", "Number of available indices in the pool"
)
INDEX_POOL_IN_USE = Gauge(
    "index_pool_in_use", "Number of indices currently in use"
)

# API metrics
API_KEY_ROTATIONS = Counter(
    "api_key_rotations_total", "Total number of API key rotations"
)
RATE_LIMIT_EXCEEDED = Counter(
    "rate_limit_exceeded_total", "Total number of rate limit violations"
)

# System metrics
MEMORY_USAGE = Gauge(
    "memory_usage_percent", "Memory usage percentage"
)

# Application info
APP_INFO = Info(
    "application_info", "Application version and build information"
)


def start_metrics_server(port: int | None = None) -> None:
    """Expose metrics on ``port`` if provided or via ``METRICS_PORT`` env var."""
    if port is None:
        port = int(os.environ.get("METRICS_PORT", "0"))
    if port:
        # Set application info
        APP_INFO.info({
            "version": "1.0.0",
            "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
            "environment": os.environ.get("ENVIRONMENT", "development"),
        })
        start_http_server(port)


def update_cache_metrics(cache_stats: dict) -> None:
    """Update cache-related metrics."""
    CACHE_SIZE.set(cache_stats.get("size", 0))
    hit_rate = cache_stats.get("hit_rate", 0.0)
    CACHE_HIT_RATE.set(hit_rate)


def update_index_metrics(document_count: int, pool_stats: dict = None) -> None:
    """Update index-related metrics."""
    INDEX_SIZE.set(document_count)
    
    if pool_stats:
        INDEX_POOL_AVAILABLE.set(pool_stats.get("available", 0))
        INDEX_POOL_IN_USE.set(pool_stats.get("in_use", 0))


def update_memory_metrics() -> None:
    """Update system memory metrics."""
    try:
        import psutil
        memory_percent = psutil.virtual_memory().percent
        MEMORY_USAGE.set(memory_percent)
    except ImportError:
        pass  # psutil not available


def record_cache_hit() -> None:
    """Record a cache hit."""
    CACHE_HITS.inc()


def record_cache_miss() -> None:
    """Record a cache miss."""
    CACHE_MISSES.inc()


def record_api_key_rotation() -> None:
    """Record an API key rotation."""
    API_KEY_ROTATIONS.inc()


def record_rate_limit_exceeded() -> None:
    """Record a rate limit violation."""
    RATE_LIMIT_EXCEEDED.inc()
