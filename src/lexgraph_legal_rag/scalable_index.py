"""Scalable index pool with auto-scaling and load balancing."""

from __future__ import annotations

import asyncio
import threading
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from pathlib import Path
import weakref

from .document_pipeline import VectorIndex, LegalDocumentPipeline
from .models import LegalDocument
from .performance_optimization import AdaptiveCache, performance_monitor

logger = logging.getLogger(__name__)


class IndexStatus(Enum):
    """Status of an index instance."""
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    RETIRED = "retired"


@dataclass
class IndexMetrics:
    """Metrics for an index instance."""
    query_count: int = 0
    avg_response_time_ms: float = 0.0
    error_count: int = 0
    memory_usage_mb: float = 0.0
    last_used: float = field(default_factory=time.time)
    cpu_time_ms: float = 0.0


@dataclass
class ScalableIndexInstance:
    """A scalable index instance with metrics tracking."""
    index_id: str
    index: VectorIndex
    status: IndexStatus = IndexStatus.INITIALIZING
    metrics: IndexMetrics = field(default_factory=IndexMetrics)
    created_at: float = field(default_factory=time.time)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    
    def update_metrics(self, response_time_ms: float, error: bool = False) -> None:
        """Update performance metrics."""
        with self._lock:
            self.metrics.query_count += 1
            self.metrics.last_used = time.time()
            
            if error:
                self.metrics.error_count += 1
            else:
                # Update rolling average
                alpha = 0.1  # Smoothing factor
                self.metrics.avg_response_time_ms = (
                    alpha * response_time_ms + 
                    (1 - alpha) * self.metrics.avg_response_time_ms
                )
    
    def get_load_score(self) -> float:
        """Calculate load score for load balancing."""
        with self._lock:
            if self.status != IndexStatus.READY:
                return float('inf')  # Don't route to non-ready instances
            
            # Base score on response time and error rate
            error_rate = self.metrics.error_count / max(1, self.metrics.query_count)
            load_score = self.metrics.avg_response_time_ms * (1 + error_rate * 10)
            
            return load_score


class IndexPool:
    """Auto-scaling pool of vector indices."""
    
    def __init__(
        self,
        min_instances: int = 2,
        max_instances: int = 10,
        target_response_time_ms: float = 100.0,
        scale_up_threshold: float = 150.0,
        scale_down_threshold: float = 50.0,
        scale_check_interval: float = 30.0
    ):
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.target_response_time_ms = target_response_time_ms
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.scale_check_interval = scale_check_interval
        
        self.instances: Dict[str, ScalableIndexInstance] = {}
        self.master_index_path: Optional[Path] = None
        self.is_initialized = False
        
        self._pool_lock = threading.RLock()
        self._scaling_task: Optional[asyncio.Task] = None
        self._instance_counter = 0
        
        # Performance cache
        self.query_cache = AdaptiveCache(
            initial_size=1000,
            max_size=10000,
            ttl_seconds=1800  # 30 minutes
        )
    
    async def initialize(self, index_path: Optional[Path] = None) -> None:
        """Initialize the index pool."""
        if self.is_initialized:
            return
        
        logger.info("Initializing scalable index pool...")
        
        self.master_index_path = index_path
        
        # Create initial instances
        for i in range(self.min_instances):
            await self._create_instance()
        
        # Start auto-scaling task
        self._scaling_task = asyncio.create_task(self._auto_scaling_loop())
        
        self.is_initialized = True
        logger.info(f"Index pool initialized with {len(self.instances)} instances")
    
    async def _create_instance(self) -> str:
        """Create a new index instance."""
        instance_id = f"index_{self._instance_counter}"
        self._instance_counter += 1
        
        try:
            # Create new vector index
            index = VectorIndex()
            
            # Load data from master index if available
            if self.master_index_path and self.master_index_path.exists():
                index.load(self.master_index_path)
            
            instance = ScalableIndexInstance(
                index_id=instance_id,
                index=index,
                status=IndexStatus.READY
            )
            
            with self._pool_lock:
                self.instances[instance_id] = instance
            
            logger.info(f"Created index instance: {instance_id}")
            return instance_id
            
        except Exception as e:
            logger.error(f"Failed to create index instance {instance_id}: {e}")
            # Create error instance to track the failure
            error_instance = ScalableIndexInstance(
                index_id=instance_id,
                index=VectorIndex(),  # Empty index
                status=IndexStatus.ERROR
            )
            
            with self._pool_lock:
                self.instances[instance_id] = error_instance
            
            raise
    
    def _select_best_instance(self) -> Optional[ScalableIndexInstance]:
        """Select the best instance for a query using load balancing."""
        with self._pool_lock:
            ready_instances = [
                instance for instance in self.instances.values()
                if instance.status == IndexStatus.READY
            ]
        
        if not ready_instances:
            return None
        
        # Select instance with lowest load score
        best_instance = min(ready_instances, key=lambda x: x.get_load_score())
        return best_instance
    
    @performance_monitor("index_pool_search")
    async def search(
        self, 
        query: str, 
        top_k: int = 5, 
        use_cache: bool = True
    ) -> List[Tuple[LegalDocument, float]]:
        """Search across the index pool with load balancing."""
        if not self.is_initialized:
            await self.initialize()
        
        # Check cache first
        if use_cache:
            cache_key = f"{query}:{top_k}"
            cached_result = self.query_cache.get(cache_key)
            if cached_result is not None:
                logger.debug("Cache hit for query: %s", query[:50])
                return cached_result
        
        # Select best instance
        instance = self._select_best_instance()
        if instance is None:
            logger.error("No ready index instances available")
            return []
        
        # Execute search with timing
        start_time = time.time()
        try:
            instance.status = IndexStatus.BUSY
            results = instance.index.search(query, top_k)
            
            response_time_ms = (time.time() - start_time) * 1000
            instance.update_metrics(response_time_ms, error=False)
            
            # Cache results
            if use_cache and results:
                cache_key = f"{query}:{top_k}"
                self.query_cache.put(cache_key, results)
            
            logger.debug(
                f"Search completed on {instance.index_id} in {response_time_ms:.1f}ms"
            )
            
            return results
            
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            instance.update_metrics(response_time_ms, error=True)
            logger.error(f"Search failed on {instance.index_id}: {e}")
            raise
            
        finally:
            instance.status = IndexStatus.READY
    
    @performance_monitor("index_pool_batch_search")
    async def batch_search(
        self, 
        queries: List[str], 
        top_k: int = 5, 
        use_cache: bool = True
    ) -> List[List[Tuple[LegalDocument, float]]]:
        """Batch search with intelligent load distribution."""
        if not queries:
            return []
        
        if not self.is_initialized:
            await self.initialize()
        
        # Check cache for all queries
        cached_results = {}
        uncached_queries = []
        
        if use_cache:
            for i, query in enumerate(queries):
                cache_key = f"{query}:{top_k}"
                cached_result = self.query_cache.get(cache_key)
                if cached_result is not None:
                    cached_results[i] = cached_result
                else:
                    uncached_queries.append((i, query))
        else:
            uncached_queries = list(enumerate(queries))
        
        # Process uncached queries in parallel across instances
        if uncached_queries:
            batch_results = await self._parallel_batch_search(
                uncached_queries, top_k, use_cache
            )
        else:
            batch_results = {}
        
        # Combine cached and uncached results
        final_results = []
        for i in range(len(queries)):
            if i in cached_results:
                final_results.append(cached_results[i])
            elif i in batch_results:
                final_results.append(batch_results[i])
            else:
                final_results.append([])  # Fallback
        
        return final_results
    
    async def _parallel_batch_search(
        self,
        indexed_queries: List[Tuple[int, str]],
        top_k: int,
        use_cache: bool
    ) -> Dict[int, List[Tuple[LegalDocument, float]]]:
        """Execute batch search in parallel across available instances."""
        with self._pool_lock:
            ready_instances = [
                instance for instance in self.instances.values()
                if instance.status == IndexStatus.READY
            ]
        
        if not ready_instances:
            logger.error("No ready instances for batch search")
            return {}
        
        # Distribute queries across instances
        query_batches = self._distribute_queries(indexed_queries, len(ready_instances))
        
        # Execute searches in parallel
        tasks = []
        for i, query_batch in enumerate(query_batches):
            if query_batch and i < len(ready_instances):
                instance = ready_instances[i]
                task = asyncio.create_task(
                    self._execute_batch_on_instance(instance, query_batch, top_k, use_cache)
                )
                tasks.append(task)
        
        # Collect results
        all_results = {}
        try:
            batch_results_list = await asyncio.gather(*tasks, return_exceptions=True)
            
            for batch_results in batch_results_list:
                if isinstance(batch_results, dict):
                    all_results.update(batch_results)
                elif isinstance(batch_results, Exception):
                    logger.error(f"Batch search failed: {batch_results}")
        
        except Exception as e:
            logger.error(f"Parallel batch search failed: {e}")
        
        return all_results
    
    def _distribute_queries(
        self, 
        indexed_queries: List[Tuple[int, str]], 
        num_instances: int
    ) -> List[List[Tuple[int, str]]]:
        """Distribute queries evenly across instances."""
        batches = [[] for _ in range(num_instances)]
        
        for i, query_item in enumerate(indexed_queries):
            batch_index = i % num_instances
            batches[batch_index].append(query_item)
        
        return batches
    
    async def _execute_batch_on_instance(
        self,
        instance: ScalableIndexInstance,
        query_batch: List[Tuple[int, str]],
        top_k: int,
        use_cache: bool
    ) -> Dict[int, List[Tuple[LegalDocument, float]]]:
        """Execute a batch of queries on a specific instance."""
        results = {}
        
        try:
            instance.status = IndexStatus.BUSY
            
            for query_index, query in query_batch:
                start_time = time.time()
                
                try:
                    search_results = instance.index.search(query, top_k)
                    response_time_ms = (time.time() - start_time) * 1000
                    
                    instance.update_metrics(response_time_ms, error=False)
                    results[query_index] = search_results
                    
                    # Cache individual results
                    if use_cache and search_results:
                        cache_key = f"{query}:{top_k}"
                        self.query_cache.put(cache_key, search_results)
                
                except Exception as e:
                    response_time_ms = (time.time() - start_time) * 1000
                    instance.update_metrics(response_time_ms, error=True)
                    logger.error(f"Query failed on {instance.index_id}: {e}")
                    results[query_index] = []
        
        finally:
            instance.status = IndexStatus.READY
        
        return results
    
    async def _auto_scaling_loop(self) -> None:
        """Auto-scaling loop that adjusts pool size based on performance."""
        while True:
            try:
                await asyncio.sleep(self.scale_check_interval)
                await self._check_and_scale()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Auto-scaling check failed: {e}")
    
    async def _check_and_scale(self) -> None:
        """Check metrics and scale the pool if needed."""
        with self._pool_lock:
            ready_instances = [
                instance for instance in self.instances.values()
                if instance.status == IndexStatus.READY
            ]
        
        if not ready_instances:
            logger.warning("No ready instances available")
            return
        
        # Calculate average response time
        total_response_time = sum(
            instance.metrics.avg_response_time_ms for instance in ready_instances
        )
        avg_response_time = total_response_time / len(ready_instances)
        
        logger.debug(f"Pool metrics: {len(ready_instances)} instances, "
                    f"avg response time: {avg_response_time:.1f}ms")
        
        # Scale up if response time is too high
        if (avg_response_time > self.scale_up_threshold and 
            len(self.instances) < self.max_instances):
            
            logger.info(f"Scaling up: avg response time {avg_response_time:.1f}ms > "
                       f"threshold {self.scale_up_threshold}ms")
            await self._create_instance()
        
        # Scale down if response time is very low and we have excess instances
        elif (avg_response_time < self.scale_down_threshold and 
              len(ready_instances) > self.min_instances):
            
            # Find the least used instance to retire
            least_used = min(ready_instances, key=lambda x: x.metrics.query_count)
            
            logger.info(f"Scaling down: avg response time {avg_response_time:.1f}ms < "
                       f"threshold {self.scale_down_threshold}ms, retiring {least_used.index_id}")
            
            with self._pool_lock:
                least_used.status = IndexStatus.RETIRED
                del self.instances[least_used.index_id]
    
    def update_all_instances(self, new_documents: List[LegalDocument]) -> None:
        """Update all instances with new documents."""
        with self._pool_lock:
            for instance in self.instances.values():
                if instance.status == IndexStatus.READY:
                    try:
                        instance.status = IndexStatus.BUSY
                        instance.index.add(new_documents)
                        instance.status = IndexStatus.READY
                        logger.debug(f"Updated {instance.index_id} with {len(new_documents)} documents")
                    except Exception as e:
                        logger.error(f"Failed to update {instance.index_id}: {e}")
                        instance.status = IndexStatus.ERROR
        
        # Invalidate cache since indices have changed
        self.query_cache.clear()
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get comprehensive pool statistics."""
        with self._pool_lock:
            instances_by_status = {}
            for status in IndexStatus:
                instances_by_status[status.value] = len([
                    i for i in self.instances.values() if i.status == status
                ])
            
            total_queries = sum(i.metrics.query_count for i in self.instances.values())
            total_errors = sum(i.metrics.error_count for i in self.instances.values())
            
            ready_instances = [
                i for i in self.instances.values() if i.status == IndexStatus.READY
            ]
            avg_response_time = (
                sum(i.metrics.avg_response_time_ms for i in ready_instances) / 
                len(ready_instances) if ready_instances else 0
            )
        
        cache_stats = self.query_cache.get_stats()
        
        return {
            "total_instances": len(self.instances),
            "instances_by_status": instances_by_status,
            "total_queries": total_queries,
            "total_errors": total_errors,
            "error_rate": total_errors / max(1, total_queries),
            "avg_response_time_ms": avg_response_time,
            "cache_stats": cache_stats,
            "min_instances": self.min_instances,
            "max_instances": self.max_instances,
            "target_response_time_ms": self.target_response_time_ms
        }
    
    async def shutdown(self) -> None:
        """Shutdown the index pool."""
        if self._scaling_task:
            self._scaling_task.cancel()
            try:
                await self._scaling_task
            except asyncio.CancelledError:
                pass
        
        with self._pool_lock:
            self.instances.clear()
        
        logger.info("Index pool shutdown complete")


# Global index pool instance
_global_index_pool: Optional[IndexPool] = None


def get_index_pool() -> IndexPool:
    """Get the global index pool instance."""
    global _global_index_pool
    if _global_index_pool is None:
        _global_index_pool = IndexPool()
    return _global_index_pool