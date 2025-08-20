"""Advanced performance optimization patterns for Legal RAG system."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import defaultdict
from collections import deque
from dataclasses import dataclass
from dataclasses import field
from functools import wraps
from typing import Any
from typing import Callable
from typing import TypeVar


logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking."""

    operation_name: str
    execution_time_ms: float
    throughput_ops_per_sec: float
    memory_usage_mb: float
    cache_hit_rate: float
    timestamp: float = field(default_factory=time.time)


class AdaptiveCache:
    """Intelligent cache with adaptive sizing and eviction."""

    def __init__(
        self,
        initial_size: int = 1000,
        max_size: int = 10000,
        ttl_seconds: float = 3600,
        hit_rate_threshold: float = 0.8,
    ):
        self.initial_size = initial_size
        self.current_size = initial_size
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.hit_rate_threshold = hit_rate_threshold

        self._cache: dict[str, Any] = {}
        self._access_times: dict[str, float] = {}
        self._access_counts: dict[str, int] = defaultdict(int)
        self._creation_times: dict[str, float] = {}

        # Performance tracking
        self.hits = 0
        self.misses = 0
        self.evictions = 0

        self._lock = threading.RLock()

    def get(self, key: str) -> Any | None:
        """Get value from cache with LFU/TTL eviction."""
        with self._lock:
            current_time = time.time()

            # Check if key exists and is not expired
            if key in self._cache:
                if current_time - self._creation_times[key] > self.ttl_seconds:
                    # Expired, remove
                    self._remove_key(key)
                    self.misses += 1
                    return None

                # Update access statistics
                self._access_times[key] = current_time
                self._access_counts[key] += 1
                self.hits += 1

                return self._cache[key]

            self.misses += 1
            return None

    def put(self, key: str, value: Any) -> None:
        """Put value in cache with adaptive sizing."""
        with self._lock:
            current_time = time.time()

            # If key already exists, update it
            if key in self._cache:
                self._cache[key] = value
                self._creation_times[key] = current_time
                self._access_times[key] = current_time
                self._access_counts[key] += 1
                return

            # Check if we need to make space
            if len(self._cache) >= self.current_size:
                self._evict_entries()

            # Add new entry
            self._cache[key] = value
            self._creation_times[key] = current_time
            self._access_times[key] = current_time
            self._access_counts[key] = 1

            # Adapt cache size based on hit rate
            self._adapt_cache_size()

    def _remove_key(self, key: str) -> None:
        """Remove key and all associated metadata."""
        self._cache.pop(key, None)
        self._access_times.pop(key, None)
        self._access_counts.pop(key, None)
        self._creation_times.pop(key, None)

    def _evict_entries(self) -> None:
        """Evict entries using LFU with TTL considerations."""
        current_time = time.time()
        evict_count = max(1, len(self._cache) // 10)  # Evict 10% at a time

        # First, remove expired entries
        expired_keys = [
            key
            for key, creation_time in self._creation_times.items()
            if current_time - creation_time > self.ttl_seconds
        ]

        for key in expired_keys:
            self._remove_key(key)
            self.evictions += 1

        # If we still need to evict more, use LFU
        remaining_to_evict = evict_count - len(expired_keys)
        if remaining_to_evict > 0 and self._cache:
            # Sort by access count (LFU) then by access time (LRU as tiebreaker)
            sorted_keys = sorted(
                self._cache.keys(),
                key=lambda k: (self._access_counts[k], self._access_times[k]),
            )

            for key in sorted_keys[:remaining_to_evict]:
                self._remove_key(key)
                self.evictions += 1

    def _adapt_cache_size(self) -> None:
        """Adapt cache size based on hit rate performance."""
        if self.hits + self.misses < 100:  # Need enough samples
            return

        current_hit_rate = self.hits / (self.hits + self.misses)

        if (
            current_hit_rate > self.hit_rate_threshold
            and self.current_size < self.max_size
        ):
            # Good hit rate, can increase cache size
            self.current_size = min(self.max_size, int(self.current_size * 1.2))
            logger.debug(
                f"Increased cache size to {self.current_size} (hit rate: {current_hit_rate:.3f})"
            )
        elif (
            current_hit_rate < self.hit_rate_threshold * 0.8
            and self.current_size > self.initial_size
        ):
            # Poor hit rate, decrease cache size
            self.current_size = max(self.initial_size, int(self.current_size * 0.8))
            logger.debug(
                f"Decreased cache size to {self.current_size} (hit rate: {current_hit_rate:.3f})"
            )

    def get_stats(self) -> dict[str, Any]:
        """Get cache performance statistics."""
        with self._lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0

            return {
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "evictions": self.evictions,
                "current_size": len(self._cache),
                "max_size": self.current_size,
                "total_requests": total_requests,
            }

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            self._access_counts.clear()
            self._creation_times.clear()


class ConnectionPool:
    """Generic connection pool for resource management."""

    def __init__(
        self,
        factory: Callable[[], T],
        max_size: int = 10,
        min_size: int = 2,
        max_idle_time: float = 300.0,
    ):
        self.factory = factory
        self.max_size = max_size
        self.min_size = min_size
        self.max_idle_time = max_idle_time

        self._pool: deque = deque()
        self._in_use: set = set()
        self._created_at: dict[int, float] = {}
        self._lock = asyncio.Lock()

        # Initialize minimum connections
        asyncio.create_task(self._initialize_pool())

    async def _initialize_pool(self) -> None:
        """Initialize the connection pool with minimum connections."""
        async with self._lock:
            while len(self._pool) < self.min_size:
                try:
                    conn = self.factory()
                    self._pool.append(conn)
                    self._created_at[id(conn)] = time.time()
                    logger.debug(
                        f"Created new pooled connection, total: {len(self._pool)}"
                    )
                except Exception as e:
                    logger.error(f"Failed to create pooled connection: {e}")
                    break

    async def acquire(self) -> T:
        """Acquire a connection from the pool."""
        async with self._lock:
            current_time = time.time()

            # Clean up expired connections
            while self._pool:
                conn = self._pool[0]
                if (
                    current_time - self._created_at.get(id(conn), 0)
                    > self.max_idle_time
                ):
                    expired_conn = self._pool.popleft()
                    self._created_at.pop(id(expired_conn), None)
                    logger.debug("Removed expired connection from pool")
                else:
                    break

            # Get connection from pool or create new one
            if self._pool:
                conn = self._pool.popleft()
                self._in_use.add(id(conn))
                return conn
            elif len(self._in_use) < self.max_size:
                # Create new connection if under limit
                conn = self.factory()
                self._in_use.add(id(conn))
                self._created_at[id(conn)] = current_time
                logger.debug(f"Created new connection, in use: {len(self._in_use)}")
                return conn
            else:
                raise RuntimeError("Connection pool exhausted")

    async def release(self, conn: T) -> None:
        """Release a connection back to the pool."""
        async with self._lock:
            conn_id = id(conn)
            if conn_id not in self._in_use:
                logger.warning("Attempting to release connection not in use")
                return

            self._in_use.remove(conn_id)

            # Return to pool if under max size, otherwise discard
            if len(self._pool) < self.max_size:
                self._pool.append(conn)
                logger.debug(
                    f"Returned connection to pool, available: {len(self._pool)}"
                )
            else:
                self._created_at.pop(conn_id, None)
                logger.debug("Discarded excess connection")


class BatchProcessor:
    """Intelligent batch processing for improved throughput."""

    def __init__(
        self,
        process_func: Callable[[list[T]], list[Any]],
        max_batch_size: int = 50,
        max_wait_time: float = 0.1,
        min_batch_size: int = 1,
    ):
        self.process_func = process_func
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.min_batch_size = min_batch_size

        self._pending_items: list[T] = []
        self._pending_futures: list[asyncio.Future] = []
        self._batch_timer: asyncio.Task | None = None
        self._lock = asyncio.Lock()

        # Performance tracking
        self.batches_processed = 0
        self.items_processed = 0
        self.total_wait_time = 0.0

    async def submit(self, item: T) -> Any:
        """Submit an item for batch processing."""
        future = asyncio.Future()

        async with self._lock:
            self._pending_items.append(item)
            self._pending_futures.append(future)

            # Process immediately if batch is full
            if len(self._pending_items) >= self.max_batch_size:
                await self._process_batch()
            else:
                # Start timer if this is the first item in batch
                if len(self._pending_items) == 1:
                    self._batch_timer = asyncio.create_task(self._wait_and_process())

        return await future

    async def _wait_and_process(self) -> None:
        """Wait for max_wait_time then process batch."""
        try:
            await asyncio.sleep(self.max_wait_time)
            async with self._lock:
                if self._pending_items:  # Check if items still pending
                    await self._process_batch()
        except asyncio.CancelledError:
            pass  # Timer was cancelled, batch was processed early

    async def _process_batch(self) -> None:
        """Process current batch of items."""
        if not self._pending_items:
            return

        items = self._pending_items.copy()
        futures = self._pending_futures.copy()

        # Clear pending lists
        self._pending_items.clear()
        self._pending_futures.clear()

        # Cancel timer if running
        if self._batch_timer and not self._batch_timer.done():
            self._batch_timer.cancel()
            self._batch_timer = None

        # Process batch
        try:
            start_time = time.time()
            results = await asyncio.get_event_loop().run_in_executor(
                None, self.process_func, items
            )
            processing_time = time.time() - start_time

            # Set results on futures
            for future, result in zip(futures, results):
                if not future.cancelled():
                    future.set_result(result)

            # Update metrics
            self.batches_processed += 1
            self.items_processed += len(items)
            self.total_wait_time += processing_time

            logger.debug(
                f"Processed batch of {len(items)} items in {processing_time:.3f}s"
            )

        except Exception as e:
            # Set exception on all futures
            for future in futures:
                if not future.cancelled():
                    future.set_exception(e)
            logger.error(f"Batch processing failed: {e}")

    def get_stats(self) -> dict[str, Any]:
        """Get batch processing statistics."""
        avg_batch_size = (
            self.items_processed / self.batches_processed
            if self.batches_processed > 0
            else 0
        )
        avg_processing_time = (
            self.total_wait_time / self.batches_processed
            if self.batches_processed > 0
            else 0
        )

        return {
            "batches_processed": self.batches_processed,
            "items_processed": self.items_processed,
            "avg_batch_size": avg_batch_size,
            "avg_processing_time_ms": avg_processing_time * 1000,
            "current_pending": len(self._pending_items),
        }


class PerformanceProfiler:
    """Profiling and performance monitoring."""

    def __init__(self):
        self.metrics: dict[str, list[PerformanceMetrics]] = defaultdict(list)
        self.max_metrics_per_operation = 1000
        self._lock = threading.Lock()

    def record_operation(
        self,
        operation_name: str,
        execution_time_ms: float,
        memory_usage_mb: float = 0.0,
        cache_hit_rate: float = 0.0,
        throughput_ops_per_sec: float = 0.0,
    ) -> None:
        """Record performance metrics for an operation."""
        metric = PerformanceMetrics(
            operation_name=operation_name,
            execution_time_ms=execution_time_ms,
            throughput_ops_per_sec=throughput_ops_per_sec,
            memory_usage_mb=memory_usage_mb,
            cache_hit_rate=cache_hit_rate,
        )

        with self._lock:
            self.metrics[operation_name].append(metric)

            # Keep only recent metrics
            if len(self.metrics[operation_name]) > self.max_metrics_per_operation:
                self.metrics[operation_name].pop(0)

    def get_operation_stats(
        self, operation_name: str, minutes: int = 5
    ) -> dict[str, Any]:
        """Get statistics for a specific operation."""
        cutoff_time = time.time() - (minutes * 60)

        with self._lock:
            recent_metrics = [
                m
                for m in self.metrics.get(operation_name, [])
                if m.timestamp >= cutoff_time
            ]

        if not recent_metrics:
            return {}

        execution_times = [m.execution_time_ms for m in recent_metrics]

        return {
            "operation_name": operation_name,
            "sample_count": len(recent_metrics),
            "timespan_minutes": minutes,
            "execution_time_ms": {
                "avg": sum(execution_times) / len(execution_times),
                "min": min(execution_times),
                "max": max(execution_times),
                "p95": self._percentile(execution_times, 95),
                "p99": self._percentile(execution_times, 99),
            },
            "throughput_ops_per_sec": sum(
                m.throughput_ops_per_sec for m in recent_metrics
            )
            / len(recent_metrics),
            "avg_memory_usage_mb": sum(m.memory_usage_mb for m in recent_metrics)
            / len(recent_metrics),
            "avg_cache_hit_rate": sum(m.cache_hit_rate for m in recent_metrics)
            / len(recent_metrics),
        }

    def _percentile(self, values: list[float], percentile: int) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0

        sorted_values = sorted(values)
        index = int((percentile / 100.0) * len(sorted_values))
        index = min(index, len(sorted_values) - 1)
        return sorted_values[index]

    def get_all_stats(self, minutes: int = 5) -> dict[str, Any]:
        """Get statistics for all operations."""
        return {
            operation: self.get_operation_stats(operation, minutes)
            for operation in self.metrics
        }


def performance_monitor(operation_name: str | None = None):
    """Decorator to monitor function performance."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        op_name = operation_name or f"{func.__module__}.{func.__name__}"

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = _get_memory_usage()

            try:
                result = await func(*args, **kwargs)

                end_time = time.time()
                end_memory = _get_memory_usage()

                execution_time_ms = (end_time - start_time) * 1000
                memory_delta_mb = end_memory - start_memory

                # Record metrics
                profiler = _get_global_profiler()
                profiler.record_operation(
                    operation_name=op_name,
                    execution_time_ms=execution_time_ms,
                    memory_usage_mb=memory_delta_mb,
                )

                return result

            except Exception as e:
                # Still record the timing even for failures
                end_time = time.time()
                execution_time_ms = (end_time - start_time) * 1000

                profiler = _get_global_profiler()
                profiler.record_operation(
                    operation_name=f"{op_name}_error",
                    execution_time_ms=execution_time_ms,
                    memory_usage_mb=0.0,
                )

                raise e

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = _get_memory_usage()

            try:
                result = func(*args, **kwargs)

                end_time = time.time()
                end_memory = _get_memory_usage()

                execution_time_ms = (end_time - start_time) * 1000
                memory_delta_mb = end_memory - start_memory

                # Record metrics
                profiler = _get_global_profiler()
                profiler.record_operation(
                    operation_name=op_name,
                    execution_time_ms=execution_time_ms,
                    memory_usage_mb=memory_delta_mb,
                )

                return result

            except Exception as e:
                # Still record the timing even for failures
                end_time = time.time()
                execution_time_ms = (end_time - start_time) * 1000

                profiler = _get_global_profiler()
                profiler.record_operation(
                    operation_name=f"{op_name}_error",
                    execution_time_ms=execution_time_ms,
                    memory_usage_mb=0.0,
                )

                raise e

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def _get_memory_usage() -> float:
    """Get current memory usage in MB."""
    try:
        import psutil

        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


# Global profiler instance
_global_profiler: PerformanceProfiler | None = None


def _get_global_profiler() -> PerformanceProfiler:
    """Get the global profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler()
    return _global_profiler


def get_performance_stats(minutes: int = 5) -> dict[str, Any]:
    """Get comprehensive performance statistics."""
    profiler = _get_global_profiler()
    return profiler.get_all_stats(minutes)
