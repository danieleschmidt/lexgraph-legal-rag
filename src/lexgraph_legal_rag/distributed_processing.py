"""
Distributed Processing Engine
High-performance distributed computing for legal document processing and analysis
"""

import os
import json
import time
import logging
import asyncio
import threading
import multiprocessing as mp
from typing import Dict, List, Any, Optional, Callable, Union, Generator
from dataclasses import dataclass, asdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
from queue import Queue, Empty
import pickle
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class ProcessingTask:
    """Represents a processing task for distributed execution."""
    id: str
    task_type: str
    payload: Dict[str, Any]
    priority: int = 0
    created_at: float = 0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    result: Optional[Any] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at == 0:
            self.created_at = time.time()


@dataclass
class WorkerStats:
    """Statistics for a distributed worker."""
    worker_id: str
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_processing_time: float = 0
    average_task_time: float = 0
    last_active: float = 0
    status: str = "idle"  # idle, busy, error


@dataclass 
class ClusterStats:
    """Statistics for the entire processing cluster."""
    total_workers: int
    active_workers: int
    idle_workers: int
    total_tasks_processed: int
    total_tasks_queued: int
    average_processing_time: float
    throughput_per_second: float
    cluster_efficiency: float


class TaskProcessor:
    """Base class for task processors."""
    
    def process(self, task: ProcessingTask) -> Any:
        """Process a task and return the result."""
        raise NotImplementedError
    
    def can_handle(self, task_type: str) -> bool:
        """Check if processor can handle the given task type."""
        raise NotImplementedError


class DocumentIndexProcessor(TaskProcessor):
    """Processor for document indexing tasks."""
    
    def can_handle(self, task_type: str) -> bool:
        return task_type in ['index_document', 'reindex_document', 'update_index']
    
    def process(self, task: ProcessingTask) -> Dict[str, Any]:
        """Process document indexing task."""
        payload = task.payload
        document_id = payload.get('document_id')
        content = payload.get('content', '')
        metadata = payload.get('metadata', {})
        
        # Simulate document processing with realistic computation
        processing_time = len(content) / 10000  # Simulate processing time based on content length
        time.sleep(min(processing_time, 2.0))  # Cap at 2 seconds for demo
        
        # Generate mock index entry
        index_entry = {
            'document_id': document_id,
            'content_hash': hashlib.md5(content.encode()).hexdigest(),
            'word_count': len(content.split()),
            'indexed_at': datetime.now().isoformat(),
            'metadata': metadata,
            'vector_embeddings_computed': True,
            'keywords_extracted': len(content.split()) // 10  # Mock keyword count
        }
        
        return {
            'success': True,
            'document_id': document_id,
            'index_entry': index_entry,
            'processing_time': processing_time
        }


class SemanticSearchProcessor(TaskProcessor):
    """Processor for semantic search tasks."""
    
    def can_handle(self, task_type: str) -> bool:
        return task_type in ['semantic_search', 'similarity_search', 'vector_search']
    
    def process(self, task: ProcessingTask) -> Dict[str, Any]:
        """Process semantic search task."""
        payload = task.payload
        query = payload.get('query', '')
        top_k = payload.get('top_k', 10)
        filters = payload.get('filters', {})
        
        # Simulate search processing
        search_complexity = len(query.split()) * top_k
        processing_time = search_complexity / 1000
        time.sleep(min(processing_time, 1.0))
        
        # Generate mock search results
        results = []
        for i in range(min(top_k, 8)):  # Return up to 8 results
            results.append({
                'document_id': f"doc_{i}_{hashlib.md5(query.encode()).hexdigest()[:8]}",
                'similarity_score': 0.95 - (i * 0.1),
                'title': f"Legal Document {i+1} matching '{query[:20]}...'",
                'snippet': f"Relevant content snippet {i+1} for query: {query[:50]}...",
                'metadata': {
                    'document_type': 'legal_case' if i % 2 == 0 else 'statute',
                    'jurisdiction': 'federal' if i % 3 == 0 else 'state',
                    'year': 2020 + (i % 4)
                }
            })
        
        return {
            'success': True,
            'query': query,
            'results': results,
            'total_found': len(results),
            'processing_time': processing_time
        }


class LegalAnalysisProcessor(TaskProcessor):
    """Processor for complex legal analysis tasks."""
    
    def can_handle(self, task_type: str) -> bool:
        return task_type in ['legal_analysis', 'precedent_analysis', 'clause_extraction']
    
    def process(self, task: ProcessingTask) -> Dict[str, Any]:
        """Process legal analysis task."""
        payload = task.payload
        document_text = payload.get('document_text', '')
        analysis_type = payload.get('analysis_type', 'general')
        
        # Simulate complex analysis
        analysis_complexity = len(document_text) / 5000 + len(analysis_type)
        processing_time = analysis_complexity * 0.1
        time.sleep(min(processing_time, 3.0))
        
        # Generate mock analysis results
        analysis_results = {
            'analysis_type': analysis_type,
            'key_clauses': [
                {
                    'clause_id': f"clause_{i}",
                    'text': f"Important legal clause {i+1} extracted from document",
                    'importance_score': 0.9 - (i * 0.15),
                    'clause_type': ['liability', 'indemnification', 'termination', 'payment'][i % 4]
                }
                for i in range(min(6, len(document_text) // 1000))
            ],
            'legal_concepts': [
                {'concept': 'contract_formation', 'confidence': 0.88},
                {'concept': 'breach_of_contract', 'confidence': 0.75},
                {'concept': 'damages', 'confidence': 0.82}
            ],
            'precedent_citations': [
                {
                    'case_name': f"Sample Case {i+1}",
                    'citation': f"123 F.3d {400 + i*10} ({2015 + i})",
                    'relevance_score': 0.85 - (i * 0.1)
                }
                for i in range(3)
            ]
        }
        
        return {
            'success': True,
            'analysis_results': analysis_results,
            'processing_time': processing_time,
            'document_length': len(document_text)
        }


class DistributedTaskQueue:
    """High-performance distributed task queue with priority scheduling."""
    
    def __init__(self, max_size: int = 10000):
        self.queue = Queue(maxsize=max_size)
        self.processing_queue = Queue()
        self.completed_tasks = {}
        self.failed_tasks = {}
        self.task_status = {}
        self.lock = threading.RLock()
        
        # Performance tracking
        self.total_tasks_submitted = 0
        self.total_tasks_completed = 0
        self.start_time = time.time()
    
    def submit_task(self, task: ProcessingTask) -> str:
        """Submit a task to the queue."""
        with self.lock:
            self.total_tasks_submitted += 1
            self.task_status[task.id] = 'queued'
        
        # Priority queue simulation (higher priority first)
        self.queue.put((task.priority, task.created_at, task))
        logger.debug(f"Task {task.id} submitted with priority {task.priority}")
        return task.id
    
    def get_next_task(self, timeout: float = 1.0) -> Optional[ProcessingTask]:
        """Get the next task to process."""
        try:
            priority, created_at, task = self.queue.get(timeout=timeout)
            with self.lock:
                self.task_status[task.id] = 'processing'
                task.started_at = time.time()
            
            self.processing_queue.put(task)
            return task
        except Empty:
            return None
    
    def complete_task(self, task: ProcessingTask, result: Any) -> None:
        """Mark a task as completed."""
        task.completed_at = time.time()
        task.result = result
        
        with self.lock:
            self.completed_tasks[task.id] = task
            self.task_status[task.id] = 'completed'
            self.total_tasks_completed += 1
        
        logger.debug(f"Task {task.id} completed in {task.completed_at - task.started_at:.2f}s")
    
    def fail_task(self, task: ProcessingTask, error: str) -> None:
        """Mark a task as failed."""
        task.error = error
        task.retry_count += 1
        
        if task.retry_count < task.max_retries:
            # Requeue for retry with lower priority
            task.priority -= 1
            self.submit_task(task)
            logger.warning(f"Task {task.id} failed, requeuing (retry {task.retry_count})")
        else:
            with self.lock:
                self.failed_tasks[task.id] = task
                self.task_status[task.id] = 'failed'
            
            logger.error(f"Task {task.id} failed permanently: {error}")
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        with self.lock:
            uptime = time.time() - self.start_time
            throughput = self.total_tasks_completed / uptime if uptime > 0 else 0
            
            return {
                'queued_tasks': self.queue.qsize(),
                'processing_tasks': self.processing_queue.qsize(),
                'completed_tasks': len(self.completed_tasks),
                'failed_tasks': len(self.failed_tasks),
                'total_submitted': self.total_tasks_submitted,
                'throughput_per_second': throughput,
                'uptime_seconds': uptime
            }


class DistributedWorker:
    """Distributed worker for processing tasks."""
    
    def __init__(self, worker_id: str, task_queue: DistributedTaskQueue, 
                 processors: List[TaskProcessor]):
        self.worker_id = worker_id
        self.task_queue = task_queue
        self.processors = {p.__class__.__name__: p for p in processors}
        self.stats = WorkerStats(worker_id=worker_id)
        self.running = False
        self.worker_thread = None
    
    def start(self) -> None:
        """Start the worker thread."""
        if not self.running:
            self.running = True
            self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.worker_thread.start()
            logger.info(f"Worker {self.worker_id} started")
    
    def stop(self) -> None:
        """Stop the worker thread."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        logger.info(f"Worker {self.worker_id} stopped")
    
    def _worker_loop(self) -> None:
        """Main worker processing loop."""
        while self.running:
            try:
                # Get next task
                task = self.task_queue.get_next_task(timeout=1.0)
                if not task:
                    self.stats.status = "idle"
                    continue
                
                self.stats.status = "busy"
                self.stats.last_active = time.time()
                
                # Find appropriate processor
                processor = self._find_processor(task.task_type)
                if not processor:
                    self.task_queue.fail_task(task, f"No processor for task type: {task.task_type}")
                    continue
                
                # Process the task
                try:
                    start_time = time.time()
                    result = processor.process(task)
                    processing_time = time.time() - start_time
                    
                    # Update statistics
                    self.stats.tasks_completed += 1
                    self.stats.total_processing_time += processing_time
                    self.stats.average_task_time = (
                        self.stats.total_processing_time / self.stats.tasks_completed
                    )
                    
                    self.task_queue.complete_task(task, result)
                    
                except Exception as e:
                    self.stats.tasks_failed += 1
                    self.task_queue.fail_task(task, str(e))
                    logger.error(f"Worker {self.worker_id} failed to process task {task.id}: {e}")
                
            except Exception as e:
                logger.error(f"Worker {self.worker_id} error: {e}")
                self.stats.status = "error"
                time.sleep(1)  # Brief pause on error
        
        self.stats.status = "stopped"
    
    def _find_processor(self, task_type: str) -> Optional[TaskProcessor]:
        """Find appropriate processor for task type."""
        for processor in self.processors.values():
            if processor.can_handle(task_type):
                return processor
        return None
    
    def get_stats(self) -> WorkerStats:
        """Get worker statistics."""
        return WorkerStats(
            worker_id=self.stats.worker_id,
            tasks_completed=self.stats.tasks_completed,
            tasks_failed=self.stats.tasks_failed,
            total_processing_time=self.stats.total_processing_time,
            average_task_time=self.stats.average_task_time,
            last_active=self.stats.last_active,
            status=self.stats.status
        )


class DistributedProcessingCluster:
    """Main distributed processing cluster manager."""
    
    def __init__(self, num_workers: int = None, max_queue_size: int = 10000):
        if num_workers is None:
            num_workers = min(mp.cpu_count(), 8)  # Reasonable default
        
        self.num_workers = num_workers
        self.task_queue = DistributedTaskQueue(max_size=max_queue_size)
        self.workers = []
        self.is_running = False
        
        # Initialize processors
        self.processors = [
            DocumentIndexProcessor(),
            SemanticSearchProcessor(), 
            LegalAnalysisProcessor()
        ]
        
        logger.info(f"Initialized distributed cluster with {num_workers} workers")
    
    def start_cluster(self) -> None:
        """Start all workers in the cluster."""
        if self.is_running:
            logger.warning("Cluster is already running")
            return
        
        # Create and start workers
        for i in range(self.num_workers):
            worker_id = f"worker_{i}"
            worker = DistributedWorker(
                worker_id=worker_id,
                task_queue=self.task_queue,
                processors=self.processors.copy()
            )
            worker.start()
            self.workers.append(worker)
        
        self.is_running = True
        logger.info(f"Distributed cluster started with {len(self.workers)} workers")
    
    def stop_cluster(self) -> None:
        """Stop all workers in the cluster."""
        if not self.is_running:
            return
        
        logger.info("Stopping distributed cluster...")
        for worker in self.workers:
            worker.stop()
        
        self.is_running = False
        logger.info("Distributed cluster stopped")
    
    def submit_task(self, task_type: str, payload: Dict[str, Any], 
                    priority: int = 0) -> str:
        """Submit a task to the cluster."""
        task_id = hashlib.md5(f"{task_type}_{time.time()}_{priority}".encode()).hexdigest()[:16]
        
        task = ProcessingTask(
            id=task_id,
            task_type=task_type,
            payload=payload,
            priority=priority
        )
        
        return self.task_queue.submit_task(task)
    
    def submit_batch_tasks(self, tasks: List[Dict[str, Any]]) -> List[str]:
        """Submit multiple tasks in batch."""
        task_ids = []
        for task_data in tasks:
            task_id = self.submit_task(
                task_type=task_data['task_type'],
                payload=task_data['payload'],
                priority=task_data.get('priority', 0)
            )
            task_ids.append(task_id)
        
        logger.info(f"Submitted batch of {len(tasks)} tasks")
        return task_ids
    
    def get_task_result(self, task_id: str, timeout: float = 30.0) -> Optional[Any]:
        """Get result of a completed task."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if task_id in self.task_queue.completed_tasks:
                return self.task_queue.completed_tasks[task_id].result
            elif task_id in self.task_queue.failed_tasks:
                error = self.task_queue.failed_tasks[task_id].error
                raise Exception(f"Task failed: {error}")
            
            time.sleep(0.1)  # Poll interval
        
        raise TimeoutError(f"Task {task_id} did not complete within {timeout}s")
    
    def wait_for_completion(self, task_ids: List[str], timeout: float = 60.0) -> Dict[str, Any]:
        """Wait for multiple tasks to complete."""
        results = {}
        start_time = time.time()
        
        while task_ids and (time.time() - start_time) < timeout:
            completed_ids = []
            
            for task_id in task_ids:
                if task_id in self.task_queue.completed_tasks:
                    results[task_id] = self.task_queue.completed_tasks[task_id].result
                    completed_ids.append(task_id)
                elif task_id in self.task_queue.failed_tasks:
                    results[task_id] = {
                        'error': self.task_queue.failed_tasks[task_id].error,
                        'failed': True
                    }
                    completed_ids.append(task_id)
            
            # Remove completed tasks from waiting list
            for task_id in completed_ids:
                task_ids.remove(task_id)
            
            if task_ids:
                time.sleep(0.1)
        
        return results
    
    def get_cluster_stats(self) -> ClusterStats:
        """Get comprehensive cluster statistics."""
        queue_stats = self.task_queue.get_queue_stats()
        worker_stats = [worker.get_stats() for worker in self.workers]
        
        active_workers = len([w for w in worker_stats if w.status in ['busy', 'idle']])
        idle_workers = len([w for w in worker_stats if w.status == 'idle'])
        
        total_processing_time = sum(w.total_processing_time for w in worker_stats)
        total_tasks_completed = sum(w.tasks_completed for w in worker_stats)
        
        avg_processing_time = (total_processing_time / total_tasks_completed 
                             if total_tasks_completed > 0 else 0)
        
        cluster_efficiency = (active_workers / len(self.workers) 
                            if self.workers else 0)
        
        return ClusterStats(
            total_workers=len(self.workers),
            active_workers=active_workers,
            idle_workers=idle_workers,
            total_tasks_processed=total_tasks_completed,
            total_tasks_queued=queue_stats['queued_tasks'],
            average_processing_time=avg_processing_time,
            throughput_per_second=queue_stats['throughput_per_second'],
            cluster_efficiency=cluster_efficiency
        )
    
    def scale_cluster(self, target_workers: int) -> None:
        """Dynamically scale the cluster up or down."""
        current_workers = len(self.workers)
        
        if target_workers > current_workers:
            # Scale up
            for i in range(current_workers, target_workers):
                worker_id = f"worker_{i}"
                worker = DistributedWorker(
                    worker_id=worker_id,
                    task_queue=self.task_queue,
                    processors=self.processors.copy()
                )
                worker.start()
                self.workers.append(worker)
            
            logger.info(f"Scaled cluster up to {target_workers} workers")
        
        elif target_workers < current_workers:
            # Scale down
            workers_to_stop = self.workers[target_workers:]
            for worker in workers_to_stop:
                worker.stop()
            
            self.workers = self.workers[:target_workers]
            logger.info(f"Scaled cluster down to {target_workers} workers")
    
    def save_cluster_report(self, filepath: Optional[str] = None) -> str:
        """Save comprehensive cluster performance report."""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"cluster_report_{timestamp}.json"
        
        stats = self.get_cluster_stats()
        queue_stats = self.task_queue.get_queue_stats()
        worker_stats = [asdict(worker.get_stats()) for worker in self.workers]
        
        report = {
            'cluster_stats': asdict(stats),
            'queue_stats': queue_stats,
            'worker_stats': worker_stats,
            'processors': [p.__class__.__name__ for p in self.processors],
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Cluster report saved to {filepath}")
        return filepath


async def run_distributed_benchmark(cluster: DistributedProcessingCluster, 
                                   num_tasks: int = 100) -> Dict[str, Any]:
    """Run performance benchmark on the distributed cluster."""
    logger.info(f"Starting distributed benchmark with {num_tasks} tasks")
    
    # Generate diverse test tasks
    tasks = []
    task_types = ['index_document', 'semantic_search', 'legal_analysis']
    
    for i in range(num_tasks):
        task_type = task_types[i % len(task_types)]
        
        if task_type == 'index_document':
            payload = {
                'document_id': f"doc_{i}",
                'content': f"Sample legal document content {i} " * (50 + i % 100),
                'metadata': {'type': 'contract', 'year': 2020 + i % 5}
            }
        elif task_type == 'semantic_search':
            payload = {
                'query': f"legal query {i} about contracts and agreements",
                'top_k': 10,
                'filters': {'year': 2020 + i % 5}
            }
        else:  # legal_analysis
            payload = {
                'document_text': f"Legal document for analysis {i} " * (20 + i % 50),
                'analysis_type': ['general', 'contract', 'liability'][i % 3]
            }
        
        tasks.append({
            'task_type': task_type,
            'payload': payload,
            'priority': i % 5  # Vary priority
        })
    
    # Submit tasks and measure performance
    start_time = time.time()
    task_ids = cluster.submit_batch_tasks(tasks)
    submission_time = time.time() - start_time
    
    # Wait for all tasks to complete
    results = cluster.wait_for_completion(task_ids, timeout=120.0)
    total_time = time.time() - start_time
    
    # Analyze results
    completed_tasks = len([r for r in results.values() if not r.get('failed', False)])
    failed_tasks = len([r for r in results.values() if r.get('failed', False)])
    
    benchmark_results = {
        'total_tasks': num_tasks,
        'completed_tasks': completed_tasks,
        'failed_tasks': failed_tasks,
        'success_rate': completed_tasks / num_tasks if num_tasks > 0 else 0,
        'total_time_seconds': total_time,
        'submission_time_seconds': submission_time,
        'processing_time_seconds': total_time - submission_time,
        'throughput_tasks_per_second': completed_tasks / total_time if total_time > 0 else 0,
        'cluster_stats': asdict(cluster.get_cluster_stats()),
        'timestamp': datetime.now().isoformat()
    }
    
    logger.info(f"Benchmark completed: {completed_tasks}/{num_tasks} tasks in {total_time:.2f}s "
                f"({benchmark_results['throughput_tasks_per_second']:.2f} tasks/s)")
    
    return benchmark_results


def main():
    """Main entry point for distributed processing demonstration."""
    logging.basicConfig(level=logging.INFO)
    
    # Create distributed cluster
    cluster = DistributedProcessingCluster(num_workers=4)
    
    try:
        # Start the cluster
        cluster.start_cluster()
        print("🚀 DISTRIBUTED PROCESSING CLUSTER STARTED")
        
        # Run benchmark
        print("📊 Running performance benchmark...")
        
        # Use asyncio for the benchmark
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        benchmark_results = loop.run_until_complete(
            run_distributed_benchmark(cluster, num_tasks=50)
        )
        
        print(f"✅ Benchmark completed:")
        print(f"   - Tasks completed: {benchmark_results['completed_tasks']}")
        print(f"   - Success rate: {benchmark_results['success_rate']:.2%}")
        print(f"   - Throughput: {benchmark_results['throughput_tasks_per_second']:.2f} tasks/s")
        
        # Save cluster report
        report_file = cluster.save_cluster_report()
        print(f"📋 Cluster report saved: {report_file}")
        
        # Demonstrate dynamic scaling
        print("🔧 Demonstrating dynamic scaling...")
        cluster.scale_cluster(6)
        time.sleep(2)
        cluster.scale_cluster(2)
        
        print("✅ DISTRIBUTED PROCESSING DEMONSTRATION COMPLETED")
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping cluster...")
    finally:
        cluster.stop_cluster()


if __name__ == "__main__":
    main()