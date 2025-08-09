#!/usr/bin/env python3
"""Scalable legal RAG pipeline with performance optimization and concurrent processing."""

import asyncio
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
import structlog
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import queue
import json

from src.lexgraph_legal_rag.document_pipeline import LegalDocumentPipeline
from src.lexgraph_legal_rag.multi_agent import MultiAgentGraph
from src.lexgraph_legal_rag.logging_config import configure_logging
from src.lexgraph_legal_rag.metrics import start_metrics_server
from src.lexgraph_legal_rag.validation_fixed import validate_query_input, validate_document_content
from src.lexgraph_legal_rag.cache import get_query_cache


class ScalableLegalRAGPipeline:
    """High-performance, scalable legal RAG pipeline with concurrent processing."""
    
    def __init__(
        self,
        use_semantic: bool = True,
        enable_caching: bool = True,
        enable_monitoring: bool = True,
        metrics_port: Optional[int] = None,
        max_workers: int = 4,
        batch_size: int = 10,
        enable_prefetching: bool = True,
        cache_ttl: int = 3600,
        auto_scale: bool = True
    ):
        # Configure structured logging
        configure_logging(level="INFO")
        self.logger = structlog.get_logger(__name__)
        
        # Performance configuration
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.enable_prefetching = enable_prefetching
        self.cache_ttl = cache_ttl
        self.auto_scale = auto_scale
        
        # Threading and concurrency
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.query_queue = queue.Queue()
        self.result_cache = {}
        self.cache_lock = threading.Lock()
        
        # Performance metrics
        self.query_count = 0
        self.avg_response_time = 0
        self.cache_hit_rate = 0
        self.load_metrics = {"cpu": 0, "memory": 0, "queries_per_second": 0}
        
        # Initialize metrics server if requested
        if enable_monitoring and metrics_port:
            start_metrics_server(metrics_port)
            self.logger.info("Started metrics server", port=metrics_port)
        
        # Initialize core components with pooling
        try:
            self.pipeline_pool = self._create_pipeline_pool()
            self.agent_pool = self._create_agent_pool()
            self.enable_caching = enable_caching
            
            # Initialize cache if enabled
            if enable_caching:
                self.cache = get_query_cache()
                self.logger.info("Query caching enabled with TTL", ttl=cache_ttl)
            
            # Start background workers
            self._start_background_workers()
            
            self.logger.info(
                "Scalable Legal RAG Pipeline initialized",
                semantic_search=use_semantic,
                caching_enabled=enable_caching,
                monitoring_enabled=enable_monitoring,
                max_workers=max_workers,
                batch_size=batch_size,
                auto_scale=auto_scale
            )
            
        except Exception as e:
            self.logger.error("Failed to initialize scalable pipeline", error=str(e))
            raise
    
    def _create_pipeline_pool(self) -> List[LegalDocumentPipeline]:
        """Create a pool of pipeline instances for concurrent processing."""
        pool = []
        for i in range(self.max_workers):
            pipeline = LegalDocumentPipeline(use_semantic=True)
            pool.append(pipeline)
        return pool
    
    def _create_agent_pool(self) -> List[MultiAgentGraph]:
        """Create a pool of agent graphs for concurrent processing."""
        pool = []
        for pipeline in self.pipeline_pool:
            agent_graph = MultiAgentGraph(pipeline=pipeline)
            pool.append(agent_graph)
        return pool
    
    def _start_background_workers(self):
        """Start background workers for performance optimization."""
        # Performance monitoring worker
        self.monitoring_thread = threading.Thread(target=self._monitor_performance, daemon=True)
        self.monitoring_thread.start()
        
        # Cache cleanup worker
        if self.enable_caching:
            self.cache_cleanup_thread = threading.Thread(target=self._cleanup_cache, daemon=True)
            self.cache_cleanup_thread.start()
        
        # Auto-scaling worker
        if self.auto_scale:
            self.scaling_thread = threading.Thread(target=self._auto_scale_monitor, daemon=True)
            self.scaling_thread.start()
    
    def _monitor_performance(self):
        """Monitor system performance and adjust parameters."""
        while True:
            try:
                import psutil
                
                # Collect system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_percent = psutil.virtual_memory().percent
                
                # Calculate queries per second
                current_time = time.time()
                if hasattr(self, '_last_metric_time'):
                    time_diff = current_time - self._last_metric_time
                    if time_diff > 0:
                        qps = (self.query_count - self._last_query_count) / time_diff
                        self.load_metrics["queries_per_second"] = qps
                
                self._last_metric_time = current_time
                self._last_query_count = self.query_count
                
                self.load_metrics.update({
                    "cpu": cpu_percent,
                    "memory": memory_percent,
                })
                
                # Log performance metrics
                self.logger.debug("Performance metrics", **self.load_metrics)
                
                # Adjust batch size based on load
                if cpu_percent > 80:
                    self.batch_size = max(5, self.batch_size - 1)
                elif cpu_percent < 40 and self.batch_size < 20:
                    self.batch_size += 1
                
                time.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                self.logger.error("Performance monitoring error", error=str(e))
                time.sleep(60)
    
    def _cleanup_cache(self):
        """Clean up expired cache entries."""
        while True:
            try:
                time.sleep(300)  # Clean every 5 minutes
                
                current_time = time.time()
                expired_keys = []
                
                with self.cache_lock:
                    for key, (timestamp, _) in self.result_cache.items():
                        if current_time - timestamp > self.cache_ttl:
                            expired_keys.append(key)
                    
                    for key in expired_keys:
                        del self.result_cache[key]
                
                if expired_keys:
                    self.logger.debug("Cleaned expired cache entries", count=len(expired_keys))
                    
            except Exception as e:
                self.logger.error("Cache cleanup error", error=str(e))
    
    def _auto_scale_monitor(self):
        """Monitor load and trigger auto-scaling recommendations."""
        while True:
            try:
                time.sleep(60)  # Check every minute
                
                cpu = self.load_metrics.get("cpu", 0)
                memory = self.load_metrics.get("memory", 0)
                qps = self.load_metrics.get("queries_per_second", 0)
                
                # Auto-scaling triggers
                if cpu > 85 or memory > 85:
                    self.logger.warning("High resource usage detected", cpu=cpu, memory=memory)
                    self._trigger_scale_up()
                elif cpu < 30 and memory < 30 and qps < 1:
                    self._trigger_scale_down()
                
                # Performance optimization triggers
                if qps > 10 and not self.enable_prefetching:
                    self.enable_prefetching = True
                    self.logger.info("Enabled prefetching due to high query rate", qps=qps)
                    
            except Exception as e:
                self.logger.error("Auto-scaling monitor error", error=str(e))
    
    def _trigger_scale_up(self):
        """Trigger scale-up recommendations."""
        self.logger.info("Scale-up recommendation", 
                        current_workers=self.max_workers,
                        recommended_workers=min(self.max_workers + 2, 16))
        
        # In a real implementation, this would trigger container scaling
        # For demo purposes, we increase batch processing efficiency
        self.batch_size = min(self.batch_size + 2, 20)
    
    def _trigger_scale_down(self):
        """Trigger scale-down recommendations."""
        if self.max_workers > 2:
            self.logger.info("Scale-down recommendation",
                           current_workers=self.max_workers,
                           recommended_workers=self.max_workers - 1)
    
    async def ingest_documents_parallel(
        self, 
        docs_paths: List[Path], 
        chunk_size: int = 512
    ) -> Dict[str, Any]:
        """Ingest multiple document directories in parallel."""
        self.logger.info("Starting parallel document ingestion", 
                        paths_count=len(docs_paths), chunk_size=chunk_size)
        
        start_time = time.time()
        stats = {
            "total_files_processed": 0,
            "total_documents_created": 0,
            "processing_time": 0,
            "parallel_jobs": len(docs_paths)
        }
        
        # Process directories in parallel
        tasks = []
        for i, docs_path in enumerate(docs_paths):
            pipeline = self.pipeline_pool[i % len(self.pipeline_pool)]
            task = asyncio.create_task(
                self._ingest_directory_async(pipeline, docs_path, chunk_size)
            )
            tasks.append(task)
        
        # Wait for all ingestion tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Aggregate results
        for result in results:
            if isinstance(result, Exception):
                self.logger.error("Parallel ingestion error", error=str(result))
            else:
                stats["total_files_processed"] += result.get("files_processed", 0)
                stats["total_documents_created"] += result.get("documents_created", 0)
        
        # Merge indices from all pipelines
        await self._merge_pipeline_indices()
        
        stats["processing_time"] = time.time() - start_time
        self.logger.info("Parallel document ingestion completed", **stats)
        
        return stats
    
    async def _ingest_directory_async(self, pipeline: LegalDocumentPipeline, docs_path: Path, chunk_size: int) -> Dict[str, Any]:
        """Asynchronously ingest a single directory."""
        loop = asyncio.get_event_loop()
        
        def sync_ingest():
            num_files = pipeline.ingest_directory(docs_path, chunk_size=chunk_size, enable_semantic=True)
            return {
                "files_processed": num_files,
                "documents_created": len(pipeline.documents)
            }
        
        return await loop.run_in_executor(self.executor, sync_ingest)
    
    async def _merge_pipeline_indices(self):
        """Merge indices from all pipeline instances."""
        # For demo purposes, we use the first pipeline as primary
        # In production, this would involve sophisticated index merging
        primary_pipeline = self.pipeline_pool[0]
        
        # Collect all documents from secondary pipelines
        all_documents = []
        for pipeline in self.pipeline_pool[1:]:
            all_documents.extend(pipeline.documents)
        
        if all_documents:
            primary_pipeline.index.add(all_documents)
            if primary_pipeline.semantic:
                primary_pipeline.semantic.ingest(all_documents)
        
        # Update all agent pools to use primary pipeline
        for agent_graph in self.agent_pool:
            agent_graph.pipeline = primary_pipeline
            if hasattr(agent_graph.retriever, 'pipeline'):
                agent_graph.retriever.pipeline = primary_pipeline
        
        self.logger.info("Pipeline indices merged", total_documents=len(primary_pipeline.documents))
    
    async def query_batch(
        self, 
        queries: List[str], 
        include_citations: bool = True,
        timeout: float = 30.0
    ) -> List[Dict[str, Any]]:
        """Process multiple queries in parallel with optimized batching."""
        start_time = time.time()
        self.logger.info("Processing query batch", batch_size=len(queries))
        
        # Split into optimal batch sizes
        batches = [queries[i:i + self.batch_size] for i in range(0, len(queries), self.batch_size)]
        
        # Process batches concurrently
        all_results = []
        tasks = []
        
        for batch in batches:
            task = asyncio.create_task(self._process_query_batch(batch, include_citations, timeout))
            tasks.append(task)
        
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results
        for batch_result in batch_results:
            if isinstance(batch_result, Exception):
                self.logger.error("Batch processing error", error=str(batch_result))
                # Add error results for failed batch
                all_results.extend([
                    {"query": "", "error": str(batch_result), "processing_time": 0}
                    for _ in range(self.batch_size)
                ])
            else:
                all_results.extend(batch_result)
        
        processing_time = time.time() - start_time
        self.query_count += len(queries)
        
        # Update average response time
        self.avg_response_time = (
            (self.avg_response_time * (self.query_count - len(queries)) + processing_time) / 
            self.query_count
        )
        
        self.logger.info("Query batch completed", 
                        queries_processed=len(queries),
                        total_time=processing_time,
                        avg_time_per_query=processing_time/len(queries))
        
        return all_results[:len(queries)]  # Ensure exact count
    
    async def _process_query_batch(
        self, 
        queries: List[str], 
        include_citations: bool, 
        timeout: float
    ) -> List[Dict[str, Any]]:
        """Process a batch of queries using available agent pools."""
        results = []
        
        # Use different agent instances for each query in the batch
        tasks = []
        for i, query in enumerate(queries):
            agent_graph = self.agent_pool[i % len(self.agent_pool)]
            task = asyncio.create_task(self._process_single_query_optimized(query, agent_graph, include_citations, timeout))
            tasks.append(task)
        
        # Wait for all queries in this batch
        query_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for query, result in zip(queries, query_results):
            if isinstance(result, Exception):
                results.append({
                    "query": query,
                    "answer": None,
                    "citations": [],
                    "error": str(result),
                    "processing_time": 0,
                    "cached": False
                })
            else:
                results.append(result)
        
        return results
    
    async def _process_single_query_optimized(
        self, 
        query: str, 
        agent_graph: MultiAgentGraph, 
        include_citations: bool, 
        timeout: float
    ) -> Dict[str, Any]:
        """Process a single query with all optimizations."""
        start_time = time.time()
        query_id = f"query_{int(start_time * 1000)}"
        
        # Input validation (fast path)
        validation_result = validate_query_input(query)
        if not validation_result.is_valid:
            return {
                "query_id": query_id,
                "query": query,
                "answer": None,
                "citations": [],
                "errors": validation_result.errors,
                "processing_time": time.time() - start_time,
                "cached": False
            }
        
        # Cache lookup with lock-free check
        cache_key = f"batch:{hash(query)}"
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            cached_result.update({
                "query_id": query_id,
                "query": query,
                "processing_time": time.time() - start_time,
                "cached": True
            })
            return cached_result
        
        try:
            # Process with optimized agent
            if include_citations:
                # Collect streaming citations efficiently
                answer_parts = []
                citations = []
                
                async for chunk in agent_graph.run_with_citations(query, agent_graph.pipeline, top_k=3):
                    if chunk.startswith("Citations:"):
                        citations.append(chunk)
                    else:
                        answer_parts.append(chunk)
                
                answer = ''.join(answer_parts)
            else:
                answer = await asyncio.wait_for(agent_graph.run(query), timeout=timeout)
            
            result = {
                "query_id": query_id,
                "query": query,
                "answer": answer,
                "citations": citations if include_citations else [],
                "errors": [],
                "processing_time": time.time() - start_time,
                "cached": False
            }
            
            # Cache successful results
            self._put_in_cache(cache_key, {
                "answer": answer,
                "citations": citations if include_citations else [],
                "timestamp": time.time()
            })
            
            return result
            
        except asyncio.TimeoutError:
            return {
                "query_id": query_id,
                "query": query,
                "answer": None,
                "citations": [],
                "errors": [f"Query processing timed out after {timeout}s"],
                "processing_time": time.time() - start_time,
                "cached": False
            }
        except Exception as e:
            return {
                "query_id": query_id,
                "query": query,
                "answer": None,
                "citations": [],
                "errors": [f"Query processing failed: {str(e)}"],
                "processing_time": time.time() - start_time,
                "cached": False
            }
    
    def _get_from_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """Thread-safe cache lookup."""
        with self.cache_lock:
            if key in self.result_cache:
                timestamp, data = self.result_cache[key]
                if time.time() - timestamp < self.cache_ttl:
                    return data.copy()
                else:
                    del self.result_cache[key]
        return None
    
    def _put_in_cache(self, key: str, data: Dict[str, Any]):
        """Thread-safe cache storage."""
        with self.cache_lock:
            self.result_cache[key] = (time.time(), data.copy())
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        cache_entries = len(self.result_cache)
        cache_hit_rate = 0
        if self.query_count > 0:
            # Rough estimation of cache hit rate
            cache_hit_rate = min(cache_entries / self.query_count, 1.0)
        
        return {
            "query_count": self.query_count,
            "avg_response_time": self.avg_response_time,
            "cache_hit_rate": cache_hit_rate,
            "cache_entries": cache_entries,
            "system_load": self.load_metrics,
            "pipeline_workers": len(self.pipeline_pool),
            "agent_workers": len(self.agent_pool),
            "current_batch_size": self.batch_size,
            "timestamp": time.time()
        }
    
    def save_indices(self, base_path: str):
        """Save all pipeline indices."""
        for i, pipeline in enumerate(self.pipeline_pool):
            index_path = f"{base_path}_worker_{i}.bin"
            pipeline.save_index(index_path)
        
        self.logger.info("All pipeline indices saved", count=len(self.pipeline_pool))
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)


async def demo_scalable_pipeline():
    """Demo the scalable pipeline capabilities."""
    print("🚀 Initializing Scalable Legal RAG Pipeline...")
    
    # Initialize with high-performance configuration
    pipeline = ScalableLegalRAGPipeline(
        use_semantic=True,
        enable_caching=True,
        enable_monitoring=True,
        metrics_port=9091,
        max_workers=4,
        batch_size=8,
        enable_prefetching=True,
        cache_ttl=1800,  # 30 minutes
        auto_scale=True
    )
    
    # Test parallel document ingestion
    docs_paths = [Path('demo_documents')]
    if all(path.exists() for path in docs_paths):
        print("📄 Ingesting documents in parallel...")
        stats = await pipeline.ingest_documents_parallel(docs_paths)
        print(f"✅ Parallel ingestion completed: {stats}")
    
    # Test high-throughput query processing
    test_queries = [
        "What are liability limits in commercial leases?",
        "Explain California Civil Code 1542",
        "Find termination clauses in contracts",
        "What is indemnification?",
        "How do warranty disclaimers work?",
        "What are the governing law provisions?",
        "Explain intellectual property rights",
        "What are breach remedies?",
        "How does force majeure apply?",
        "What are the payment terms?"
    ] * 2  # 20 queries for batch testing
    
    print(f"\n🔥 Processing {len(test_queries)} queries in parallel batches...")
    start_time = time.time()
    
    results = await pipeline.query_batch(test_queries, include_citations=False, timeout=15.0)
    
    processing_time = time.time() - start_time
    successful_queries = len([r for r in results if not r.get("errors")])
    
    print(f"✅ Batch processing completed:")
    print(f"   Total queries: {len(test_queries)}")
    print(f"   Successful: {successful_queries}")
    print(f"   Failed: {len(test_queries) - successful_queries}")
    print(f"   Total time: {processing_time:.2f}s")
    print(f"   Queries/second: {len(test_queries) / processing_time:.2f}")
    
    # Show performance metrics
    print("\n📊 Performance Metrics:")
    metrics = pipeline.get_performance_metrics()
    for key, value in metrics.items():
        if key != "system_load":
            print(f"   {key}: {value}")
    
    print(f"   System Load: CPU {metrics['system_load']['cpu']:.1f}%, "
          f"Memory {metrics['system_load']['memory']:.1f}%, "
          f"QPS {metrics['system_load']['queries_per_second']:.2f}")
    
    # Test cache effectiveness
    print("\n🔄 Testing cache effectiveness...")
    cache_test_queries = test_queries[:5]
    
    # First run (populate cache)
    await pipeline.query_batch(cache_test_queries, include_citations=False)
    
    # Second run (should hit cache)
    start_time = time.time()
    cached_results = await pipeline.query_batch(cache_test_queries, include_citations=False)
    cache_time = time.time() - start_time
    
    cached_count = len([r for r in cached_results if r.get("cached")])
    print(f"✅ Cache test: {cached_count}/{len(cache_test_queries)} queries served from cache")
    print(f"   Cache response time: {cache_time:.3f}s ({cache_time/len(cache_test_queries)*1000:.1f}ms per query)")
    
    # Save optimized indices
    pipeline.save_indices("scalable_index")
    
    print("\n🎉 Scalable Pipeline Demo Complete!")


if __name__ == "__main__":
    asyncio.run(demo_scalable_pipeline())