"""Prometheus metrics for monitoring search operations."""

from __future__ import annotations

import os
import threading
import sys

from prometheus_client import Counter, Histogram, Gauge, Info, start_http_server, CollectorRegistry, REGISTRY


# Thread-safe metric initialization
_metrics_lock = threading.Lock()
_metrics_initialized = False
_test_registry = None

# Metric storage
_metrics = {}


def _get_registry():
    """Get the appropriate registry for metrics."""
    global _test_registry
    
    # Check if we're in a test environment
    if ("pytest" in sys.modules or 
        "PYTEST_CURRENT_TEST" in os.environ or
        any("pytest" in arg for arg in sys.argv)):
        if _test_registry is None:
            _test_registry = CollectorRegistry()
        return _test_registry
    return REGISTRY


def _init_metrics():
    """Initialize metrics lazily and thread-safely."""
    global _metrics_initialized
    
    with _metrics_lock:
        if _metrics_initialized:
            return
            
        registry = _get_registry()
        
        try:
            # Search metrics
            _metrics['SEARCH_REQUESTS'] = Counter(
                "search_requests_total", "Total number of search queries processed",
                ["search_type"],  # semantic, vector, cached
                registry=registry
            )
            _metrics['SEARCH_LATENCY'] = Histogram(
                "search_latency_seconds", "Time spent processing search queries",
                ["search_type"],
                buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
                registry=registry
            )
            
            # Cache metrics
            _metrics['CACHE_HITS'] = Counter(
                "cache_hits_total", "Total number of cache hits",
                registry=registry
            )
            _metrics['CACHE_MISSES'] = Counter(
                "cache_misses_total", "Total number of cache misses",
                registry=registry
            )
            _metrics['CACHE_SIZE'] = Gauge(
                "cache_size_entries", "Current number of entries in cache",
                registry=registry
            )
            _metrics['CACHE_HIT_RATE'] = Gauge(
                "cache_hit_rate", "Cache hit rate (0-1)",
                registry=registry
            )
            
            # Index metrics
            _metrics['INDEX_SIZE'] = Gauge(
                "index_size_documents", "Number of documents in the index",
                registry=registry
            )
            _metrics['INDEX_POOL_AVAILABLE'] = Gauge(
                "index_pool_available", "Number of available indices in the pool",
                registry=registry
            )
            _metrics['INDEX_POOL_IN_USE'] = Gauge(
                "index_pool_in_use", "Number of indices currently in use",
                registry=registry
            )
            
            # HTTP metrics for alerting
            _metrics['HTTP_REQUESTS_TOTAL'] = Counter(
                "http_requests_total", "Total number of HTTP requests",
                ["method", "endpoint", "status"],
                registry=registry
            )
            _metrics['HTTP_REQUESTS_ERRORS_TOTAL'] = Counter(
                "http_requests_errors_total", "Total number of HTTP error responses",
                ["method", "endpoint", "status"],
                registry=registry
            )
            _metrics['HTTP_REQUEST_DURATION_SECONDS'] = Histogram(
                "http_request_duration_seconds", "HTTP request duration in seconds",
                ["method", "endpoint", "status"],
                buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
                registry=registry
            )
            
            # API metrics
            _metrics['API_KEY_ROTATIONS'] = Counter(
                "api_key_rotations_total", "Total number of API key rotations",
                registry=registry
            )
            _metrics['RATE_LIMIT_EXCEEDED'] = Counter(
                "rate_limit_exceeded_total", "Total number of rate limit violations",
                registry=registry
            )
            
            # System metrics
            _metrics['MEMORY_USAGE'] = Gauge(
                "memory_usage_percent", "Memory usage percentage",
                registry=registry
            )
            
            # Application info
            _metrics['APP_INFO'] = Info(
                "application_info", "Application version and build information",
                registry=registry
            )
            
            _metrics_initialized = True
            
        except ValueError as e:
            # If we get a duplicate metric error, we're probably in tests
            # Just mark as initialized to avoid re-registering
            if "Duplicated timeseries" in str(e):
                _metrics_initialized = True
            else:
                raise


def _get_metric(name: str):
    """Get a metric by name, initializing if necessary."""
    if not _metrics_initialized:
        _init_metrics()
    return _metrics.get(name)


# Lazy metric accessors
@property
def SEARCH_REQUESTS():
    """Get the search requests counter metric."""
    return _get_metric('SEARCH_REQUESTS')

@property  
def SEARCH_LATENCY():
    """Get the search latency histogram metric."""
    return _get_metric('SEARCH_LATENCY')

@property
def CACHE_HITS():
    """Get the cache hits counter metric."""
    return _get_metric('CACHE_HITS')

@property
def CACHE_MISSES():
    """Get the cache misses counter metric."""
    return _get_metric('CACHE_MISSES')

@property
def CACHE_SIZE():
    """Get the cache size gauge metric."""
    return _get_metric('CACHE_SIZE')

@property
def CACHE_HIT_RATE():
    """Get the cache hit rate gauge metric."""
    return _get_metric('CACHE_HIT_RATE')

@property
def INDEX_SIZE():
    """Get the index size gauge metric."""
    return _get_metric('INDEX_SIZE')

@property
def INDEX_POOL_AVAILABLE():
    """Get the index pool available gauge metric."""
    return _get_metric('INDEX_POOL_AVAILABLE')

@property
def INDEX_POOL_IN_USE():
    """Get the index pool in use gauge metric."""
    return _get_metric('INDEX_POOL_IN_USE')

@property
def HTTP_REQUESTS_TOTAL():
    """Get the HTTP requests total counter metric."""
    return _get_metric('HTTP_REQUESTS_TOTAL')

@property
def HTTP_REQUESTS_ERRORS_TOTAL():
    """Get the HTTP request errors total counter metric."""
    return _get_metric('HTTP_REQUESTS_ERRORS_TOTAL')

@property
def HTTP_REQUEST_DURATION_SECONDS():
    """Get the HTTP request duration histogram metric."""
    return _get_metric('HTTP_REQUEST_DURATION_SECONDS')

@property
def API_KEY_ROTATIONS():
    """Get the API key rotations counter metric."""
    return _get_metric('API_KEY_ROTATIONS')

@property
def RATE_LIMIT_EXCEEDED():
    """Get the rate limit exceeded counter metric."""
    return _get_metric('RATE_LIMIT_EXCEEDED')

@property
def MEMORY_USAGE():
    """Get the memory usage gauge metric."""
    return _get_metric('MEMORY_USAGE')

@property
def APP_INFO():
    """Get the application info metric."""
    return _get_metric('APP_INFO')


def start_metrics_server(port: int | None = None) -> None:
    """Expose metrics on ``port`` if provided or via ``METRICS_PORT`` env var."""
    _init_metrics()  # Ensure metrics are initialized
    
    if port is None:
        port = int(os.environ.get("METRICS_PORT", "0"))
    if port:
        # Set application info
        app_info = _get_metric('APP_INFO')
        if app_info:
            app_info.info({
                "version": "1.0.0",
                "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
                "environment": os.environ.get("ENVIRONMENT", "development"),
            })
        start_http_server(port)


def update_cache_metrics(cache_stats: dict) -> None:
    """Update cache-related metrics."""
    cache_size = _get_metric('CACHE_SIZE')
    cache_hit_rate = _get_metric('CACHE_HIT_RATE')
    
    if cache_size:
        cache_size.set(cache_stats.get("size", 0))
    if cache_hit_rate:
        hit_rate = cache_stats.get("hit_rate", 0.0)
        cache_hit_rate.set(hit_rate)


def update_index_metrics(document_count: int, pool_stats: dict = None) -> None:
    """Update index-related metrics."""
    index_size = _get_metric('INDEX_SIZE')
    if index_size:
        index_size.set(document_count)
    
    if pool_stats:
        pool_available = _get_metric('INDEX_POOL_AVAILABLE')
        pool_in_use = _get_metric('INDEX_POOL_IN_USE')
        if pool_available:
            pool_available.set(pool_stats.get("available", 0))
        if pool_in_use:
            pool_in_use.set(pool_stats.get("in_use", 0))


def update_memory_metrics() -> None:
    """Update system memory metrics."""
    try:
        import psutil
        memory_percent = psutil.virtual_memory().percent
        memory_usage = _get_metric('MEMORY_USAGE')
        if memory_usage:
            memory_usage.set(memory_percent)
    except ImportError:
        pass  # psutil not available


def record_cache_hit() -> None:
    """Record a cache hit."""
    cache_hits = _get_metric('CACHE_HITS')
    if cache_hits:
        cache_hits.inc()


def record_cache_miss() -> None:
    """Record a cache miss."""
    cache_misses = _get_metric('CACHE_MISSES')
    if cache_misses:
        cache_misses.inc()


def record_api_key_rotation() -> None:
    """Record an API key rotation."""
    api_key_rotations = _get_metric('API_KEY_ROTATIONS')
    if api_key_rotations:
        api_key_rotations.inc()


def record_rate_limit_exceeded() -> None:
    """Record a rate limit violation."""
    rate_limit_exceeded = _get_metric('RATE_LIMIT_EXCEEDED')
    if rate_limit_exceeded:
        rate_limit_exceeded.inc()


def record_http_request(method: str, endpoint: str, status_code: int, duration: float) -> None:
    """Record HTTP request metrics for alerting."""
    status_str = str(status_code)
    
    # Record total requests
    http_requests_total = _get_metric('HTTP_REQUESTS_TOTAL')
    if http_requests_total:
        http_requests_total.labels(method=method, endpoint=endpoint, status=status_str).inc()
    
    # Record request duration
    http_request_duration = _get_metric('HTTP_REQUEST_DURATION_SECONDS')
    if http_request_duration:
        http_request_duration.labels(method=method, endpoint=endpoint, status=status_str).observe(duration)
    
    # Record errors (4xx and 5xx responses)
    if status_code >= 400:
        http_requests_errors = _get_metric('HTTP_REQUESTS_ERRORS_TOTAL')
        if http_requests_errors:
            http_requests_errors.labels(method=method, endpoint=endpoint, status=status_str).inc()


def record_http_error(method: str, endpoint: str, status_code: int, error_type: str = "server_error") -> None:
    """Record HTTP error for alerting purposes."""
    status_str = str(status_code)
    http_requests_errors = _get_metric('HTTP_REQUESTS_ERRORS_TOTAL')
    if http_requests_errors:
        http_requests_errors.labels(method=method, endpoint=endpoint, status=status_str).inc()


def get_error_rate_metrics() -> dict:
    """Get current error rate metrics for monitoring."""
    try:
        # Get current metric values - this is a simplified version
        # In production, you'd query Prometheus directly
        return {
            "total_requests": "Available via Prometheus /metrics endpoint",
            "total_errors": "Available via Prometheus /metrics endpoint", 
            "error_rate": "Calculate via Prometheus: rate(http_requests_errors_total[5m]) / rate(http_requests_total[5m])"
        }
    except Exception as e:
        return {"error": f"Failed to get metrics: {e}"}


# For backward compatibility, create module-level variables that access the properties
import sys
current_module = sys.modules[__name__]

# Create module-level variables dynamically
for metric_name in ['SEARCH_REQUESTS', 'SEARCH_LATENCY', 'CACHE_HITS', 'CACHE_MISSES', 
                   'CACHE_SIZE', 'CACHE_HIT_RATE', 'INDEX_SIZE', 'INDEX_POOL_AVAILABLE', 
                   'INDEX_POOL_IN_USE', 'HTTP_REQUESTS_TOTAL', 'HTTP_REQUESTS_ERRORS_TOTAL',
                   'HTTP_REQUEST_DURATION_SECONDS', 'API_KEY_ROTATIONS', 'RATE_LIMIT_EXCEEDED',
                   'MEMORY_USAGE', 'APP_INFO']:
    
    class MetricAccessor:
        def __init__(self, name):
            self.name = name
            
        def __getattr__(self, attr):
            metric = _get_metric(self.name)
            if metric is None:
                raise AttributeError(f"Metric {self.name} not initialized")
            return getattr(metric, attr)
    
    setattr(current_module, metric_name, MetricAccessor(metric_name))