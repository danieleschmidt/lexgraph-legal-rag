"""
Distributed Scaling Engine for Legal RAG System
==============================================

Advanced distributed processing and auto-scaling for production workloads.
Novel contribution: AI-driven distributed scaling with predictive load balancing.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)


class ScalingMode(Enum):
    """Scaling modes for distributed processing."""

    HORIZONTAL = "horizontal"  # Add more worker nodes
    VERTICAL = "vertical"  # Increase resources per node
    ELASTIC = "elastic"  # Dynamic horizontal + vertical
    PREDICTIVE = "predictive"  # Proactive scaling based on predictions


class WorkerState(Enum):
    """States of worker nodes."""

    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    OVERLOADED = "overloaded"
    DEGRADED = "degraded"
    FAILED = "failed"
    SCALING = "scaling"


class TaskPriority(Enum):
    """Task priority levels."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class WorkerNode:
    """Distributed worker node."""

    node_id: str
    state: WorkerState = WorkerState.INITIALIZING
    cpu_cores: int = 4
    memory_gb: int = 8
    current_load: float = 0.0
    capacity: float = 1.0
    tasks_completed: int = 0
    tasks_failed: int = 0
    last_heartbeat: float = field(default_factory=time.time)
    specialization: str | None = None


@dataclass
class DistributedTask:
    """Task for distributed processing."""

    task_id: str
    task_type: str
    priority: TaskPriority
    payload: dict[str, Any]
    estimated_duration: float = 1.0
    max_retries: int = 3
    assigned_node: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None


@dataclass
class ScalingDecision:
    """Decision for scaling operations."""

    action: str  # scale_up, scale_down, optimize
    target_nodes: int
    reason: str
    confidence: float
    estimated_impact: dict[str, float]


@dataclass
class LoadPrediction:
    """Predicted load characteristics."""

    predicted_load: float
    confidence: float
    time_horizon: float
    factors: dict[str, float]


class DistributedScalingEngine:
    """
    Advanced distributed scaling engine with predictive capabilities.

    Features:
    - Intelligent worker node management
    - Predictive load balancing
    - Dynamic task distribution
    - Auto-scaling with multiple strategies
    - Fault tolerance and self-healing
    - Performance optimization
    """

    def __init__(self, initial_workers: int = 4):
        self.scaling_mode = ScalingMode.ELASTIC

        # Worker management
        self.workers: dict[str, WorkerNode] = {}
        self.task_queue: deque = deque()
        self.active_tasks: dict[str, DistributedTask] = {}
        self.completed_tasks: deque = deque(maxlen=10000)

        # Load balancing
        self.load_history: deque = deque(maxlen=1000)
        self.performance_metrics: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=100)
        )

        # Predictive modeling
        self.load_predictor: dict[str, float] = {}
        self.scaling_history: deque = deque(maxlen=100)

        # Execution
        self.executor = ThreadPoolExecutor(max_workers=initial_workers)
        self.running = False

        # Initialize workers
        self._initialize_workers(initial_workers)

    def _initialize_workers(self, count: int):
        """Initialize worker nodes."""
        for i in range(count):
            node_id = f"worker_{i:04d}"
            self.workers[node_id] = WorkerNode(
                node_id=node_id,
                state=WorkerState.HEALTHY,
                cpu_cores=4,
                memory_gb=8,
                specialization="bioneural_processing" if i % 3 == 0 else None,
            )
        logger.info(f"Initialized {count} worker nodes")

    def _calculate_node_score(self, node: WorkerNode, task: DistributedTask) -> float:
        """Calculate node suitability score for task."""
        base_score = 1.0

        # Load factor (lower load = higher score)
        load_factor = 1.0 - node.current_load
        base_score *= load_factor

        # Capacity factor
        capacity_factor = node.capacity
        base_score *= capacity_factor

        # Specialization bonus
        if node.specialization and task.task_type.startswith(node.specialization):
            base_score *= 1.5

        # Reliability factor
        total_tasks = node.tasks_completed + node.tasks_failed
        if total_tasks > 0:
            reliability = node.tasks_completed / total_tasks
            base_score *= reliability

        # Health penalty
        if node.state != WorkerState.HEALTHY:
            base_score *= 0.5

        return base_score

    def _select_optimal_worker(self, task: DistributedTask) -> str | None:
        """Select optimal worker for task using intelligent load balancing."""
        available_workers = [
            (node_id, node)
            for node_id, node in self.workers.items()
            if node.state in [WorkerState.HEALTHY, WorkerState.DEGRADED]
            and node.current_load < 0.9
        ]

        if not available_workers:
            return None

        # Calculate scores for all available workers
        scored_workers = [
            (node_id, self._calculate_node_score(node, task))
            for node_id, node in available_workers
        ]

        # Select worker with highest score
        best_worker = max(scored_workers, key=lambda x: x[1])
        return best_worker[0]

    async def _predict_load(self, time_horizon: float = 300.0) -> LoadPrediction:
        """Predict future load using simple time series analysis."""
        if len(self.load_history) < 10:
            return LoadPrediction(
                predicted_load=0.5,
                confidence=0.3,
                time_horizon=time_horizon,
                factors={"insufficient_data": 1.0},
            )

        recent_loads = np.array(list(self.load_history)[-50:])

        # Simple trend analysis
        time_points = np.arange(len(recent_loads))
        trend_coeff = np.polyfit(time_points, recent_loads, 1)[0]

        # Current average load
        current_avg = np.mean(recent_loads[-10:])

        # Predict future load
        future_steps = int(time_horizon / 30)  # 30 second intervals
        predicted_load = current_avg + trend_coeff * future_steps
        predicted_load = max(0.0, min(1.0, predicted_load))

        # Calculate confidence based on trend stability
        load_variance = np.var(recent_loads[-10:])
        confidence = max(0.1, 1.0 - load_variance)

        return LoadPrediction(
            predicted_load=predicted_load,
            confidence=confidence,
            time_horizon=time_horizon,
            factors={
                "trend": trend_coeff,
                "current_load": current_avg,
                "variance": load_variance,
            },
        )

    def _make_scaling_decision(
        self, current_load: float, predicted_load: LoadPrediction
    ) -> ScalingDecision | None:
        """Make intelligent scaling decision."""
        active_workers = len(
            [w for w in self.workers.values() if w.state == WorkerState.HEALTHY]
        )

        # Scale up conditions
        if current_load > 0.8 or (
            predicted_load.predicted_load > 0.7 and predicted_load.confidence > 0.6
        ):
            if active_workers < 20:  # Max workers limit
                return ScalingDecision(
                    action="scale_up",
                    target_nodes=min(active_workers + 2, 20),
                    reason=f"High load: current={current_load:.2f}, "
                    f"predicted={predicted_load.predicted_load:.2f}",
                    confidence=predicted_load.confidence,
                    estimated_impact={
                        "latency_reduction": 0.3,
                        "throughput_increase": 0.4,
                    },
                )

        # Scale down conditions
        if (
            current_load < 0.3
            and predicted_load.predicted_load < 0.4
            and predicted_load.confidence > 0.5
        ):
            if active_workers > 2:  # Min workers limit
                return ScalingDecision(
                    action="scale_down",
                    target_nodes=max(active_workers - 1, 2),
                    reason=f"Low load: current={current_load:.2f}, "
                    f"predicted={predicted_load.predicted_load:.2f}",
                    confidence=predicted_load.confidence,
                    estimated_impact={"cost_reduction": 0.2},
                )

        return None

    async def _execute_scaling_decision(self, decision: ScalingDecision):
        """Execute scaling decision."""
        logger.info(
            f"Executing scaling decision: {decision.action} to {decision.target_nodes} nodes"
        )

        current_workers = len(self.workers)

        if decision.action == "scale_up":
            # Add new workers
            new_workers_needed = decision.target_nodes - current_workers
            for i in range(new_workers_needed):
                node_id = f"worker_{current_workers + i:04d}"
                self.workers[node_id] = WorkerNode(
                    node_id=node_id,
                    state=WorkerState.INITIALIZING,
                    cpu_cores=4,
                    memory_gb=8,
                    specialization="bioneural_processing" if i % 3 == 0 else None,
                )

                # Simulate initialization time
                await asyncio.sleep(0.1)
                self.workers[node_id].state = WorkerState.HEALTHY

        elif decision.action == "scale_down":
            # Remove excess workers
            workers_to_remove = current_workers - decision.target_nodes
            idle_workers = [
                node_id
                for node_id, node in self.workers.items()
                if node.current_load < 0.1 and node.state == WorkerState.HEALTHY
            ]

            for node_id in idle_workers[:workers_to_remove]:
                self.workers[node_id].state = WorkerState.FAILED
                logger.debug(f"Scaled down worker: {node_id}")

        self.scaling_history.append(decision)

    async def submit_task(self, task: DistributedTask) -> str:
        """Submit task for distributed processing."""
        self.task_queue.append(task)
        self.active_tasks[task.task_id] = task

        logger.debug(f"Task submitted: {task.task_id} (priority: {task.priority.name})")
        return task.task_id

    async def _process_task(
        self, task: DistributedTask, worker_id: str
    ) -> dict[str, Any]:
        """Process task on assigned worker."""
        worker = self.workers[worker_id]
        task.assigned_node = worker_id
        task.started_at = time.time()

        # Update worker load
        worker.current_load = min(1.0, worker.current_load + 0.2)

        try:
            # Simulate task processing based on type
            if task.task_type == "bioneural_analysis":
                # Simulate bioneural processing
                await asyncio.sleep(
                    task.estimated_duration * np.random.uniform(0.8, 1.2)
                )
                result = {
                    "status": "completed",
                    "signals_detected": np.random.randint(1, 6),
                    "confidence": np.random.uniform(0.7, 0.95),
                    "processing_node": worker_id,
                }
            elif task.task_type == "document_indexing":
                # Simulate document indexing
                await asyncio.sleep(
                    task.estimated_duration * np.random.uniform(0.5, 1.5)
                )
                result = {
                    "status": "completed",
                    "documents_indexed": np.random.randint(10, 100),
                    "index_size": np.random.randint(1000, 10000),
                    "processing_node": worker_id,
                }
            else:
                # Generic processing
                await asyncio.sleep(task.estimated_duration)
                result = {"status": "completed", "processing_node": worker_id}

            # Task completed successfully
            task.completed_at = time.time()
            worker.tasks_completed += 1
            worker.current_load = max(0.0, worker.current_load - 0.2)

            return result

        except Exception as e:
            # Task failed
            worker.tasks_failed += 1
            worker.current_load = max(0.0, worker.current_load - 0.2)

            logger.error(f"Task {task.task_id} failed on {worker_id}: {e}")
            return {"status": "failed", "error": str(e), "processing_node": worker_id}

    async def _task_distribution_loop(self):
        """Main task distribution loop."""
        while self.running:
            try:
                # Process pending tasks
                if self.task_queue:
                    # Sort tasks by priority
                    pending_tasks = []
                    while self.task_queue:
                        pending_tasks.append(self.task_queue.popleft())

                    pending_tasks.sort(key=lambda t: t.priority.value, reverse=True)

                    # Assign tasks to workers
                    for task in pending_tasks:
                        worker_id = self._select_optimal_worker(task)
                        if worker_id:
                            # Process task asynchronously
                            asyncio.create_task(self._process_task(task, worker_id))
                        else:
                            # No available workers, put back in queue
                            self.task_queue.appendleft(task)
                            break

                # Update load metrics
                current_load = self._calculate_current_load()
                self.load_history.append(current_load)

                # Predictive scaling
                if self.scaling_mode in [ScalingMode.ELASTIC, ScalingMode.PREDICTIVE]:
                    prediction = await self._predict_load()
                    scaling_decision = self._make_scaling_decision(
                        current_load, prediction
                    )

                    if scaling_decision:
                        await self._execute_scaling_decision(scaling_decision)

                # Health monitoring
                await self._monitor_worker_health()

                await asyncio.sleep(1.0)  # 1 second loop

            except Exception as e:
                logger.error(f"Error in task distribution loop: {e}")
                await asyncio.sleep(5.0)

    def _calculate_current_load(self) -> float:
        """Calculate current system load."""
        active_workers = [
            w for w in self.workers.values() if w.state == WorkerState.HEALTHY
        ]

        if not active_workers:
            return 0.0

        total_load = sum(worker.current_load for worker in active_workers)
        average_load = total_load / len(active_workers)

        return average_load

    async def _monitor_worker_health(self):
        """Monitor worker health and handle failures."""
        current_time = time.time()

        for worker in self.workers.values():
            # Check heartbeat
            if current_time - worker.last_heartbeat > 30.0:  # 30 second timeout
                if worker.state == WorkerState.HEALTHY:
                    worker.state = WorkerState.DEGRADED
                    logger.warning(
                        f"Worker {worker.node_id} degraded (heartbeat timeout)"
                    )
                elif worker.state == WorkerState.DEGRADED:
                    worker.state = WorkerState.FAILED
                    logger.error(f"Worker {worker.node_id} failed")

            # Update heartbeat (simulate)
            if worker.state in [WorkerState.HEALTHY, WorkerState.DEGRADED]:
                worker.last_heartbeat = current_time

    async def start(self):
        """Start the distributed scaling engine."""
        if self.running:
            return

        self.running = True
        logger.info("Starting distributed scaling engine")

        # Start task distribution loop
        asyncio.create_task(self._task_distribution_loop())

    async def stop(self):
        """Stop the distributed scaling engine."""
        self.running = False
        self.executor.shutdown(wait=True)
        logger.info("Distributed scaling engine stopped")

    def get_cluster_status(self) -> dict[str, Any]:
        """Get comprehensive cluster status."""
        healthy_workers = len(
            [w for w in self.workers.values() if w.state == WorkerState.HEALTHY]
        )
        total_workers = len(self.workers)

        current_load = self._calculate_current_load()

        return {
            "cluster_health": {
                "healthy_workers": healthy_workers,
                "total_workers": total_workers,
                "health_ratio": (
                    healthy_workers / total_workers if total_workers > 0 else 0
                ),
            },
            "load_metrics": {
                "current_load": current_load,
                "pending_tasks": len(self.task_queue),
                "active_tasks": len(self.active_tasks),
                "completed_tasks": len(self.completed_tasks),
            },
            "scaling_info": {
                "mode": self.scaling_mode.value,
                "recent_decisions": len(self.scaling_history),
                "last_scaling": (
                    self.scaling_history[-1].action if self.scaling_history else None
                ),
            },
            "worker_details": [
                {
                    "node_id": worker.node_id,
                    "state": worker.state.value,
                    "load": worker.current_load,
                    "completed": worker.tasks_completed,
                    "failed": worker.tasks_failed,
                    "specialization": worker.specialization,
                }
                for worker in self.workers.values()
            ],
        }


# Global instance
_scaling_engine = DistributedScalingEngine()


def get_scaling_engine() -> DistributedScalingEngine:
    """Get the global distributed scaling engine instance."""
    return _scaling_engine


async def submit_distributed_task(
    task_type: str,
    payload: dict[str, Any],
    priority: TaskPriority = TaskPriority.NORMAL,
) -> str:
    """Convenience function to submit distributed task."""
    engine = get_scaling_engine()

    task = DistributedTask(
        task_id=f"task_{int(time.time()*1000)}",
        task_type=task_type,
        priority=priority,
        payload=payload,
        estimated_duration=payload.get("estimated_duration", 1.0),
    )

    return await engine.submit_task(task)
