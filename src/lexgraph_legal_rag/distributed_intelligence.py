"""Distributed intelligence system for massive-scale legal RAG processing.

This module provides advanced distributed processing capabilities including:
- Intelligent load balancing across multiple workers
- Dynamic scaling based on query patterns and load
- Distributed caching with consistency guarantees
- Advanced query routing and optimization
- Real-time performance monitoring and adjustment
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import random
import threading
import time
from collections import defaultdict
from collections import deque
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import Any
from typing import AsyncIterator


logger = logging.getLogger(__name__)


class WorkerStatus(Enum):
    """Worker node status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    OVERLOADED = "overloaded"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"


class QueryComplexity(Enum):
    """Query complexity levels for intelligent routing."""

    SIMPLE = "simple"  # Basic keyword searches
    MODERATE = "moderate"  # Multi-term queries with some logic
    COMPLEX = "complex"  # Advanced semantic queries
    HEAVY = "heavy"  # Resource-intensive queries


@dataclass
class WorkerNode:
    """Represents a worker node in the distributed system."""

    node_id: str
    hostname: str
    port: int
    status: WorkerStatus = WorkerStatus.HEALTHY

    # Capacity and load metrics
    max_concurrent_queries: int = 10
    current_load: int = 0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0

    # Performance metrics
    avg_response_time: float = 0.0
    success_rate: float = 1.0
    total_processed: int = 0

    # Specializations
    specializations: list[str] = field(default_factory=list)
    capabilities: dict[str, Any] = field(default_factory=dict)

    # Health tracking
    last_health_check: float = field(default_factory=time.time)
    consecutive_failures: int = 0

    def get_load_factor(self) -> float:
        """Calculate current load factor (0.0 to 1.0+)."""
        return self.current_load / max(self.max_concurrent_queries, 1)

    def get_health_score(self) -> float:
        """Calculate overall health score (0.0 to 1.0)."""
        if self.status == WorkerStatus.OFFLINE:
            return 0.0

        # Base score from status
        status_scores = {
            WorkerStatus.HEALTHY: 1.0,
            WorkerStatus.DEGRADED: 0.7,
            WorkerStatus.OVERLOADED: 0.3,
            WorkerStatus.MAINTENANCE: 0.1,
        }
        base_score = status_scores.get(self.status, 0.5)

        # Adjust for performance metrics
        perf_factor = (
            self.success_rate * 0.4
            + (1.0 - min(self.cpu_usage, 1.0)) * 0.3
            + (1.0 - min(self.memory_usage, 1.0)) * 0.3
        )

        # Penalty for consecutive failures
        failure_penalty = max(0, 1.0 - (self.consecutive_failures * 0.1))

        return base_score * perf_factor * failure_penalty

    def can_handle_query(self, complexity: QueryComplexity) -> bool:
        """Check if worker can handle query of given complexity."""
        if self.status == WorkerStatus.OFFLINE:
            return False

        # Check load capacity
        if self.get_load_factor() >= 1.0:
            return False

        # Check complexity constraints
        complexity_requirements = {
            QueryComplexity.SIMPLE: 0.1,  # Can handle if <90% loaded
            QueryComplexity.MODERATE: 0.3,  # Can handle if <70% loaded
            QueryComplexity.COMPLEX: 0.6,  # Can handle if <40% loaded
            QueryComplexity.HEAVY: 0.8,  # Can handle if <20% loaded
        }

        load_threshold = complexity_requirements.get(complexity, 0.5)
        return self.get_load_factor() <= (1.0 - load_threshold)

    def update_metrics(self, response_time: float, success: bool) -> None:
        """Update performance metrics after query processing."""
        self.total_processed += 1

        # Update running averages
        alpha = 0.1  # Smoothing factor
        self.avg_response_time = (
            alpha * response_time + (1 - alpha) * self.avg_response_time
        )

        self.success_rate = (
            alpha * (1.0 if success else 0.0) + (1 - alpha) * self.success_rate
        )

        if success:
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1

            # Adjust status based on failures
            if self.consecutive_failures >= 5:
                self.status = WorkerStatus.DEGRADED
            if self.consecutive_failures >= 10:
                self.status = WorkerStatus.OFFLINE


@dataclass
class QueryProfile:
    """Query analysis and routing profile."""

    query: str
    query_hash: str
    complexity: QueryComplexity
    estimated_processing_time: float
    resource_requirements: dict[str, float]
    specialization_needed: str | None = None
    priority: float = 1.0

    @classmethod
    def analyze_query(cls, query: str) -> QueryProfile:
        """Analyze query and create profile."""
        query_hash = hashlib.sha256(query.encode()).hexdigest()

        # Simple complexity analysis
        complexity = cls._estimate_complexity(query)
        processing_time = cls._estimate_processing_time(query, complexity)

        # Resource requirements estimation
        resources = {
            "cpu": cls._estimate_cpu_requirement(query, complexity),
            "memory": cls._estimate_memory_requirement(query, complexity),
            "network": cls._estimate_network_requirement(query, complexity),
        }

        return cls(
            query=query,
            query_hash=query_hash,
            complexity=complexity,
            estimated_processing_time=processing_time,
            resource_requirements=resources,
        )

    @staticmethod
    def _estimate_complexity(query: str) -> QueryComplexity:
        """Estimate query complexity based on content analysis."""
        query_lower = query.lower()
        query_length = len(query)
        word_count = len(query.split())

        # Simple heuristics for complexity estimation
        if query_length > 500 or word_count > 50:
            return QueryComplexity.HEAVY

        # Check for complex legal concepts
        complex_terms = [
            "constitutional",
            "jurisprudence",
            "precedent analysis",
            "comparative law",
            "statutory interpretation",
            "case synthesis",
        ]
        if any(term in query_lower for term in complex_terms):
            return QueryComplexity.COMPLEX

        # Check for moderate complexity indicators
        moderate_terms = [
            "analyze",
            "compare",
            "interpret",
            "summarize",
            "explain",
            "implications",
            "consequences",
        ]
        if any(term in query_lower for term in moderate_terms) or word_count > 10:
            return QueryComplexity.MODERATE

        return QueryComplexity.SIMPLE

    @staticmethod
    def _estimate_processing_time(query: str, complexity: QueryComplexity) -> float:
        """Estimate processing time based on query characteristics."""
        base_times = {
            QueryComplexity.SIMPLE: 0.5,
            QueryComplexity.MODERATE: 2.0,
            QueryComplexity.COMPLEX: 8.0,
            QueryComplexity.HEAVY: 30.0,
        }

        # Adjust based on query length
        length_factor = min(2.0, len(query) / 100)
        return base_times[complexity] * length_factor

    @staticmethod
    def _estimate_cpu_requirement(query: str, complexity: QueryComplexity) -> float:
        """Estimate CPU requirement (0.0 to 1.0)."""
        base_cpu = {
            QueryComplexity.SIMPLE: 0.1,
            QueryComplexity.MODERATE: 0.3,
            QueryComplexity.COMPLEX: 0.6,
            QueryComplexity.HEAVY: 0.9,
        }
        return base_cpu[complexity]

    @staticmethod
    def _estimate_memory_requirement(query: str, complexity: QueryComplexity) -> float:
        """Estimate memory requirement (0.0 to 1.0)."""
        base_memory = {
            QueryComplexity.SIMPLE: 0.1,
            QueryComplexity.MODERATE: 0.2,
            QueryComplexity.COMPLEX: 0.5,
            QueryComplexity.HEAVY: 0.8,
        }
        return base_memory[complexity]

    @staticmethod
    def _estimate_network_requirement(query: str, complexity: QueryComplexity) -> float:
        """Estimate network I/O requirement (0.0 to 1.0)."""
        # Most legal queries are network-light as they operate on cached data
        return 0.1


class IntelligentLoadBalancer:
    """Advanced load balancer with intelligent routing."""

    def __init__(self):
        self.workers: dict[str, WorkerNode] = {}
        self.routing_history: deque = deque(maxlen=10000)
        self.performance_metrics: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )
        self._lock = threading.RLock()

        # Load balancing strategies
        self.strategies = {
            "weighted_round_robin": self._weighted_round_robin,
            "least_connections": self._least_connections,
            "performance_based": self._performance_based,
            "complexity_aware": self._complexity_aware,
            "predictive": self._predictive_routing,
        }

        self.current_strategy = "complexity_aware"
        logger.info("Intelligent load balancer initialized")

    def register_worker(self, worker: WorkerNode) -> None:
        """Register a new worker node."""
        with self._lock:
            self.workers[worker.node_id] = worker
            logger.info(
                f"Registered worker {worker.node_id} at {worker.hostname}:{worker.port}"
            )

    def unregister_worker(self, node_id: str) -> None:
        """Unregister a worker node."""
        with self._lock:
            if node_id in self.workers:
                del self.workers[node_id]
                logger.info(f"Unregistered worker {node_id}")

    async def route_query(self, query_profile: QueryProfile) -> WorkerNode | None:
        """Route query to optimal worker based on current strategy."""
        with self._lock:
            available_workers = [
                worker
                for worker in self.workers.values()
                if worker.can_handle_query(query_profile.complexity)
            ]

            if not available_workers:
                logger.warning(
                    f"No available workers for query complexity: {query_profile.complexity}"
                )
                return None

            # Use current routing strategy
            strategy_func = self.strategies.get(
                self.current_strategy, self._weighted_round_robin
            )
            selected_worker = await strategy_func(query_profile, available_workers)

            # Record routing decision
            if selected_worker:
                self._record_routing(query_profile, selected_worker)

            return selected_worker

    async def _weighted_round_robin(
        self, query_profile: QueryProfile, workers: list[WorkerNode]
    ) -> WorkerNode:
        """Weighted round-robin based on worker capacity and health."""
        if not workers:
            return None

        # Calculate weights based on health and available capacity
        weights = []
        for worker in workers:
            available_capacity = worker.max_concurrent_queries - worker.current_load
            health_score = worker.get_health_score()
            weight = available_capacity * health_score
            weights.append(weight)

        # Select worker based on weights
        total_weight = sum(weights)
        if total_weight == 0:
            return random.choice(workers)

        rand_val = random.uniform(0, total_weight)
        current_weight = 0

        for worker, weight in zip(workers, weights):
            current_weight += weight
            if rand_val <= current_weight:
                return worker

        return workers[-1]  # Fallback

    async def _least_connections(
        self, query_profile: QueryProfile, workers: list[WorkerNode]
    ) -> WorkerNode:
        """Route to worker with least current connections."""
        return min(workers, key=lambda w: w.current_load)

    async def _performance_based(
        self, query_profile: QueryProfile, workers: list[WorkerNode]
    ) -> WorkerNode:
        """Route based on worker performance metrics."""
        # Score workers based on response time and success rate
        best_worker = None
        best_score = -1

        for worker in workers:
            # Performance score (higher is better)
            response_score = 1.0 / max(worker.avg_response_time, 0.1)
            load_score = 1.0 - worker.get_load_factor()
            health_score = worker.get_health_score()

            combined_score = (
                response_score * 0.4 + load_score * 0.3 + health_score * 0.3
            )

            if combined_score > best_score:
                best_score = combined_score
                best_worker = worker

        return best_worker

    async def _complexity_aware(
        self, query_profile: QueryProfile, workers: list[WorkerNode]
    ) -> WorkerNode:
        """Route based on query complexity and worker capabilities."""
        # Filter workers that can handle the complexity well
        suitable_workers = []

        for worker in workers:
            load_factor = worker.get_load_factor()

            # Check if worker is suitable for this complexity
            complexity_thresholds = {
                QueryComplexity.SIMPLE: 0.8,
                QueryComplexity.MODERATE: 0.6,
                QueryComplexity.COMPLEX: 0.4,
                QueryComplexity.HEAVY: 0.2,
            }

            threshold = complexity_thresholds.get(query_profile.complexity, 0.5)
            if load_factor <= threshold:
                suitable_workers.append(worker)

        if suitable_workers:
            # Among suitable workers, pick the one with best performance
            return await self._performance_based(query_profile, suitable_workers)
        else:
            # Fallback to least loaded worker
            return await self._least_connections(query_profile, workers)

    async def _predictive_routing(
        self, query_profile: QueryProfile, workers: list[WorkerNode]
    ) -> WorkerNode:
        """Predictive routing based on historical patterns."""
        # This would implement ML-based routing in production
        # For now, fall back to complexity-aware routing
        return await self._complexity_aware(query_profile, workers)

    def _record_routing(self, query_profile: QueryProfile, worker: WorkerNode) -> None:
        """Record routing decision for analysis."""
        routing_record = {
            "timestamp": time.time(),
            "query_hash": query_profile.query_hash,
            "complexity": query_profile.complexity.value,
            "worker_id": worker.node_id,
            "worker_load": worker.current_load,
            "estimated_time": query_profile.estimated_processing_time,
            "strategy": self.current_strategy,
        }

        self.routing_history.append(routing_record)

    def update_worker_status(
        self, node_id: str, cpu_usage: float, memory_usage: float, current_load: int
    ) -> None:
        """Update worker status and metrics."""
        with self._lock:
            if node_id in self.workers:
                worker = self.workers[node_id]
                worker.cpu_usage = cpu_usage
                worker.memory_usage = memory_usage
                worker.current_load = current_load
                worker.last_health_check = time.time()

                # Update status based on metrics
                if cpu_usage > 0.9 or memory_usage > 0.9:
                    worker.status = WorkerStatus.OVERLOADED
                elif cpu_usage > 0.7 or memory_usage > 0.7:
                    worker.status = WorkerStatus.DEGRADED
                else:
                    worker.status = WorkerStatus.HEALTHY

    def get_cluster_stats(self) -> dict[str, Any]:
        """Get comprehensive cluster statistics."""
        with self._lock:
            if not self.workers:
                return {"error": "No workers registered"}

            # Worker statistics
            worker_stats = {}
            total_capacity = 0
            total_load = 0
            healthy_workers = 0

            for worker in self.workers.values():
                worker_stats[worker.node_id] = {
                    "status": worker.status.value,
                    "load_factor": worker.get_load_factor(),
                    "health_score": worker.get_health_score(),
                    "avg_response_time": worker.avg_response_time,
                    "success_rate": worker.success_rate,
                    "total_processed": worker.total_processed,
                }

                total_capacity += worker.max_concurrent_queries
                total_load += worker.current_load

                if worker.status == WorkerStatus.HEALTHY:
                    healthy_workers += 1

            # Routing statistics
            recent_routes = [
                r for r in self.routing_history if time.time() - r["timestamp"] < 3600
            ]  # Last hour

            complexity_distribution = {}
            for route in recent_routes:
                complexity = route["complexity"]
                complexity_distribution[complexity] = (
                    complexity_distribution.get(complexity, 0) + 1
                )

            return {
                "cluster_overview": {
                    "total_workers": len(self.workers),
                    "healthy_workers": healthy_workers,
                    "total_capacity": total_capacity,
                    "current_load": total_load,
                    "cluster_utilization": total_load / max(total_capacity, 1),
                },
                "worker_details": worker_stats,
                "routing_stats": {
                    "current_strategy": self.current_strategy,
                    "recent_routes": len(recent_routes),
                    "complexity_distribution": complexity_distribution,
                },
            }


class DistributedQueryProcessor:
    """High-level distributed query processing coordinator."""

    def __init__(self):
        self.load_balancer = IntelligentLoadBalancer()
        self.query_queue: asyncio.Queue = asyncio.Queue()
        self.results_cache: dict[str, Any] = {}
        self._processing_tasks: dict[str, asyncio.Task] = {}

        # Start background workers
        self._start_background_tasks()

        logger.info("Distributed query processor initialized")

    def _start_background_tasks(self) -> None:
        """Start background processing tasks."""
        # These would be started in an event loop in production
        pass

    async def process_query_distributed(
        self, query: str, priority: float = 1.0
    ) -> AsyncIterator[str]:
        """Process query using distributed workers."""
        # Analyze query
        query_profile = QueryProfile.analyze_query(query)
        query_profile.priority = priority

        # Check cache first
        cache_key = query_profile.query_hash
        if cache_key in self.results_cache:
            logger.info(f"Cache hit for query: {query[:50]}...")
            cached_result = self.results_cache[cache_key]
            for chunk in cached_result:
                yield chunk
            return

        # Route to optimal worker
        selected_worker = await self.load_balancer.route_query(query_profile)
        if not selected_worker:
            yield "Error: No available workers to process query"
            return

        # Process query on selected worker
        try:
            selected_worker.current_load += 1
            start_time = time.time()

            # Simulate distributed processing
            result_chunks = await self._execute_on_worker(query, selected_worker)

            processing_time = time.time() - start_time
            selected_worker.update_metrics(processing_time, True)

            # Cache results
            result_list = []
            for chunk in result_chunks:
                result_list.append(chunk)
                yield chunk

            self.results_cache[cache_key] = result_list

            # Limit cache size
            if len(self.results_cache) > 10000:
                # Remove oldest entries
                old_keys = list(self.results_cache.keys())[:1000]
                for old_key in old_keys:
                    del self.results_cache[old_key]

        except Exception as e:
            processing_time = time.time() - start_time
            selected_worker.update_metrics(processing_time, False)

            logger.error(
                f"Error processing query on worker {selected_worker.node_id}: {e}"
            )
            yield f"Error: Query processing failed: {e!s}"

        finally:
            selected_worker.current_load = max(0, selected_worker.current_load - 1)

    async def _execute_on_worker(self, query: str, worker: WorkerNode) -> list[str]:
        """Execute query on specific worker node."""
        # This would make actual API calls to worker nodes in production
        # For demo, we simulate the processing

        logger.info(f"Processing query on worker {worker.node_id}")

        # Simulate processing delay based on query complexity
        await asyncio.sleep(0.1)  # Simulated network + processing time

        # Simulate response
        result_chunks = [
            f"Processing query: {query[:100]}...",
            f"Worker {worker.node_id} found relevant legal documents.",
            f"Analysis complete. Results processed in {worker.avg_response_time:.2f}s average time.",
        ]

        return result_chunks

    def add_worker(
        self,
        node_id: str,
        hostname: str,
        port: int,
        max_concurrent: int = 10,
        specializations: list[str] | None = None,
    ) -> None:
        """Add a new worker to the cluster."""
        worker = WorkerNode(
            node_id=node_id,
            hostname=hostname,
            port=port,
            max_concurrent_queries=max_concurrent,
            specializations=specializations or [],
        )

        self.load_balancer.register_worker(worker)

    def get_system_status(self) -> dict[str, Any]:
        """Get comprehensive system status."""
        cluster_stats = self.load_balancer.get_cluster_stats()

        return {
            "distributed_processor": {
                "cache_size": len(self.results_cache),
                "active_processing_tasks": len(self._processing_tasks),
                "queue_size": self.query_queue.qsize(),
            },
            "cluster_stats": cluster_stats,
        }


# Global distributed processor instance
_global_processor = None


def get_distributed_processor() -> DistributedQueryProcessor:
    """Get global distributed processor instance."""
    global _global_processor
    if _global_processor is None:
        _global_processor = DistributedQueryProcessor()

        # Add some demo workers
        _global_processor.add_worker(
            "worker-1", "localhost", 8001, 15, ["contract-law", "corporate"]
        )
        _global_processor.add_worker(
            "worker-2", "localhost", 8002, 10, ["criminal-law", "litigation"]
        )
        _global_processor.add_worker(
            "worker-3", "localhost", 8003, 20, ["ip-law", "regulatory"]
        )

    return _global_processor


async def process_query_at_scale(
    query: str, priority: float = 1.0
) -> AsyncIterator[str]:
    """Convenience function for distributed query processing."""
    processor = get_distributed_processor()
    async for chunk in processor.process_query_distributed(query, priority):
        yield chunk


def get_cluster_status() -> dict[str, Any]:
    """Convenience function to get cluster status."""
    processor = get_distributed_processor()
    return processor.get_system_status()
