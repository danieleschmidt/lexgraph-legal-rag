#!/usr/bin/env python3
"""Scalable bioneural olfactory fusion engine with performance optimization."""

import sys
import os
import asyncio
import concurrent.futures
import multiprocessing
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import hashlib
import threading
from collections import defaultdict
import pickle

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

class ProcessingMode(Enum):
    """Processing modes for scalability."""
    SINGLE_THREADED = "single"
    MULTI_THREADED = "threaded"
    MULTI_PROCESS = "process"
    ASYNC_CONCURRENT = "async"
    HYBRID = "hybrid"

@dataclass
class PerformanceMetrics:
    """Performance tracking metrics."""
    documents_processed: int = 0
    total_processing_time: float = 0.0
    average_time_per_doc: float = 0.0
    throughput_docs_per_sec: float = 0.0
    memory_usage_mb: float = 0.0
    cache_hit_rate: float = 0.0
    concurrent_workers: int = 1
    processing_mode: str = "single"

@dataclass  
class CacheEntry:
    """Cache entry for processed documents."""
    document_hash: str
    scent_profile: Dict[str, Any]
    timestamp: float
    access_count: int = 0

class BioneuroCache:
    """Intelligent caching system for bioneural analysis results."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
    
    def get(self, document_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached scent profile."""
        with self.lock:
            if document_hash in self.cache:
                entry = self.cache[document_hash]
                
                # Check TTL
                if time.time() - entry.timestamp > self.ttl_seconds:
                    self._remove(document_hash)
                    self.misses += 1
                    return None
                
                # Update access
                entry.access_count += 1
                if document_hash in self.access_order:
                    self.access_order.remove(document_hash)
                self.access_order.append(document_hash)
                
                self.hits += 1
                return entry.scent_profile
            
            self.misses += 1
            return None
    
    def put(self, document_hash: str, scent_profile: Dict[str, Any]):
        """Cache scent profile."""
        with self.lock:
            # Evict if necessary
            while len(self.cache) >= self.max_size and self.access_order:
                oldest = self.access_order.pop(0)
                self._remove(oldest)
            
            entry = CacheEntry(
                document_hash=document_hash,
                scent_profile=scent_profile,
                timestamp=time.time()
            )
            
            self.cache[document_hash] = entry
            if document_hash not in self.access_order:
                self.access_order.append(document_hash)
    
    def _remove(self, document_hash: str):
        """Remove entry from cache."""
        if document_hash in self.cache:
            del self.cache[document_hash]
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
            self.hits = 0
            self.misses = 0

class ScalableBioneuroEngine:
    """Scalable bioneural olfactory fusion engine."""
    
    def __init__(self, 
                 processing_mode: ProcessingMode = ProcessingMode.HYBRID,
                 max_workers: int = None,
                 enable_caching: bool = True,
                 cache_size: int = 1000):
        
        self.processing_mode = processing_mode
        self.max_workers = max_workers or min(8, multiprocessing.cpu_count())
        self.enable_caching = enable_caching
        
        # Initialize cache
        self.cache = BioneuroCache(max_size=cache_size) if enable_caching else None
        
        # Performance tracking
        self.metrics = PerformanceMetrics(
            concurrent_workers=self.max_workers,
            processing_mode=processing_mode.value
        )
        
        # Threading/async setup
        self.executor = None
        self.setup_processing()
    
    def setup_processing(self):
        """Setup processing based on mode."""
        if self.processing_mode == ProcessingMode.MULTI_THREADED:
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        elif self.processing_mode == ProcessingMode.MULTI_PROCESS:
            self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers)
    
    def _compute_document_hash(self, text: str) -> str:
        """Compute hash for document caching."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
    
    def _enhanced_bioneural_analysis(self, text: str) -> Dict[str, Any]:
        """Enhanced bioneural analysis with optimizations."""
        
        # Import analysis functions locally to avoid import issues in multiprocessing
        from minimal_working_demo import simple_legal_text_analysis, bioneural_scent_simulation
        
        # Perform analysis
        traditional = simple_legal_text_analysis(text)
        bioneural = bioneural_scent_simulation(text)
        
        # Enhanced features for scalability
        text_lower = text.lower()
        
        # Advanced pattern detection
        advanced_patterns = {
            'clause_density': text.count(',') / len(text.split()) if len(text.split()) > 0 else 0,
            'legal_phrases': sum(1 for phrase in ['pursuant to', 'notwithstanding', 'heretofore', 'whereas'] if phrase in text_lower),
            'citation_complexity': text.count('(') + text.count('[') + text.count('§'),
            'conditional_clauses': text_lower.count('if') + text_lower.count('unless') + text_lower.count('provided that'),
            'temporal_references': text_lower.count('shall') + text_lower.count('will') + text_lower.count('may')
        }
        
        # Combine analyses
        enhanced_profile = {
            **bioneural,
            'traditional_analysis': traditional,
            'advanced_patterns': advanced_patterns,
            'processing_metadata': {
                'analysis_version': '2.0',
                'enhanced_features': True,
                'processing_time': time.time()
            }
        }
        
        return enhanced_profile
    
    def analyze_document(self, text: str, document_id: str = None) -> Dict[str, Any]:
        """Analyze single document with caching."""
        
        start_time = time.time()
        
        # Check cache first
        if self.cache:
            doc_hash = self._compute_document_hash(text)
            cached_result = self.cache.get(doc_hash)
            if cached_result:
                processing_time = time.time() - start_time
                self._update_metrics(1, processing_time)
                return cached_result
        
        # Perform analysis
        result = self._enhanced_bioneural_analysis(text)
        
        # Cache result
        if self.cache:
            self.cache.put(doc_hash, result)
        
        # Update metrics
        processing_time = time.time() - start_time
        self._update_metrics(1, processing_time)
        
        return result
    
    async def analyze_document_async(self, text: str, document_id: str = None) -> Dict[str, Any]:
        """Async document analysis."""
        
        loop = asyncio.get_event_loop()
        
        # Check cache in async context
        if self.cache:
            doc_hash = self._compute_document_hash(text)
            cached_result = self.cache.get(doc_hash)
            if cached_result:
                return cached_result
        
        # Run analysis in thread pool
        result = await loop.run_in_executor(
            self.executor, 
            self._enhanced_bioneural_analysis, 
            text
        )
        
        # Cache result
        if self.cache:
            doc_hash = self._compute_document_hash(text)
            self.cache.put(doc_hash, result)
        
        return result
    
    def analyze_batch(self, documents: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        """Analyze batch of documents with optimal processing mode."""
        
        start_time = time.time()
        results = []
        
        if self.processing_mode == ProcessingMode.SINGLE_THREADED:
            results = [self.analyze_document(doc[0], doc[1]) for doc in documents]
            
        elif self.processing_mode in [ProcessingMode.MULTI_THREADED, ProcessingMode.MULTI_PROCESS]:
            futures = []
            for doc_text, doc_id in documents:
                future = self.executor.submit(self.analyze_document, doc_text, doc_id)
                futures.append(future)
            
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
        elif self.processing_mode == ProcessingMode.ASYNC_CONCURRENT:
            results = asyncio.run(self._analyze_batch_async(documents))
            
        elif self.processing_mode == ProcessingMode.HYBRID:
            # Use different strategies based on batch size
            if len(documents) < 10:
                results = [self.analyze_document(doc[0], doc[1]) for doc in documents]
            else:
                results = asyncio.run(self._analyze_batch_async(documents))
        
        # Update batch metrics
        processing_time = time.time() - start_time
        self._update_metrics(len(documents), processing_time)
        
        return results
    
    async def _analyze_batch_async(self, documents: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        """Async batch processing."""
        
        # Create semaphore to limit concurrent operations
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def bounded_analysis(doc_text: str, doc_id: str):
            async with semaphore:
                return await self.analyze_document_async(doc_text, doc_id)
        
        # Process all documents concurrently
        tasks = [bounded_analysis(doc[0], doc[1]) for doc in documents]
        results = await asyncio.gather(*tasks)
        
        return results
    
    def _update_metrics(self, docs_processed: int, processing_time: float):
        """Update performance metrics."""
        
        self.metrics.documents_processed += docs_processed
        self.metrics.total_processing_time += processing_time
        
        if self.metrics.documents_processed > 0:
            self.metrics.average_time_per_doc = (
                self.metrics.total_processing_time / self.metrics.documents_processed
            )
            self.metrics.throughput_docs_per_sec = (
                self.metrics.documents_processed / self.metrics.total_processing_time
            )
        
        if self.cache:
            self.metrics.cache_hit_rate = self.cache.get_hit_rate()
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        
        # Update memory usage
        try:
            import psutil
            process = psutil.Process()
            self.metrics.memory_usage_mb = process.memory_info().rss / 1024 / 1024
        except ImportError:
            pass  # psutil not available
        
        return self.metrics
    
    def optimize_performance(self) -> Dict[str, Any]:
        """Auto-optimize performance based on current metrics."""
        
        metrics = self.get_performance_metrics()
        optimizations = []
        
        # Cache optimization
        if self.cache and metrics.cache_hit_rate < 0.3:
            self.cache.max_size = min(self.cache.max_size * 2, 5000)
            optimizations.append("Increased cache size")
        
        # Processing mode optimization
        if metrics.throughput_docs_per_sec < 100:
            if self.processing_mode == ProcessingMode.SINGLE_THREADED:
                self.processing_mode = ProcessingMode.MULTI_THREADED
                self.setup_processing()
                optimizations.append("Switched to multi-threading")
        
        # Worker optimization
        if metrics.throughput_docs_per_sec > 1000 and self.max_workers < 16:
            self.max_workers = min(self.max_workers * 2, 16)
            self.setup_processing()
            optimizations.append(f"Increased workers to {self.max_workers}")
        
        return {
            'optimizations_applied': optimizations,
            'new_metrics': self.get_performance_metrics()
        }
    
    def benchmark(self, test_documents: List[str] = None, iterations: int = 3) -> Dict[str, Any]:
        """Run performance benchmark."""
        
        if not test_documents:
            test_documents = [
                "This Software License Agreement governs use of proprietary software.",
                "Commercial lease agreement between landlord and tenant for retail space.",
                "Employment contract with confidentiality and intellectual property clauses.",
                "Purchase agreement for acquisition of business assets and goodwill.",
                "15 U.S.C. § 1681 requires disclosure of consumer credit information."
            ] * 20  # 100 test documents
        
        # Prepare test data
        documents = [(doc, f"test_{i}") for i, doc in enumerate(test_documents)]
        
        print(f"🏃 Running benchmark with {len(documents)} documents, {iterations} iterations")
        print(f"   Processing mode: {self.processing_mode.value}")
        print(f"   Max workers: {self.max_workers}")
        print(f"   Caching: {'enabled' if self.enable_caching else 'disabled'}")
        
        # Reset metrics
        self.metrics = PerformanceMetrics(
            concurrent_workers=self.max_workers,
            processing_mode=self.processing_mode.value
        )
        if self.cache:
            self.cache.clear()
        
        # Run benchmark iterations
        iteration_results = []
        
        for iteration in range(iterations):
            print(f"   Iteration {iteration + 1}/{iterations}...")
            
            start_time = time.time()
            results = self.analyze_batch(documents)
            end_time = time.time()
            
            iteration_time = end_time - start_time
            iteration_throughput = len(documents) / iteration_time
            
            iteration_results.append({
                'iteration': iteration + 1,
                'processing_time': iteration_time,
                'throughput': iteration_throughput,
                'documents_processed': len(results)
            })
            
            print(f"      Time: {iteration_time:.2f}s, Throughput: {iteration_throughput:.0f} docs/sec")
        
        # Calculate aggregate metrics
        total_docs = sum(r['documents_processed'] for r in iteration_results)
        total_time = sum(r['processing_time'] for r in iteration_results)
        avg_throughput = sum(r['throughput'] for r in iteration_results) / len(iteration_results)
        
        final_metrics = self.get_performance_metrics()
        
        benchmark_results = {
            'total_documents_processed': total_docs,
            'total_processing_time': total_time,
            'average_throughput_docs_per_sec': avg_throughput,
            'cache_hit_rate': final_metrics.cache_hit_rate,
            'memory_usage_mb': final_metrics.memory_usage_mb,
            'processing_mode': self.processing_mode.value,
            'max_workers': self.max_workers,
            'iterations': iteration_results
        }
        
        print(f"\n📊 Benchmark Results:")
        print(f"   Average throughput: {avg_throughput:.0f} docs/sec")
        print(f"   Cache hit rate: {final_metrics.cache_hit_rate:.1%}")
        print(f"   Memory usage: {final_metrics.memory_usage_mb:.1f} MB")
        
        return benchmark_results
    
    def __del__(self):
        """Cleanup resources."""
        if self.executor:
            self.executor.shutdown(wait=True)

def demonstrate_scalability():
    """Demonstrate scalable bioneural engine."""
    
    print("⚡ Scalable Bioneural Olfactory Fusion Engine")
    print("=" * 60)
    
    # Test different processing modes
    modes_to_test = [
        ProcessingMode.SINGLE_THREADED,
        ProcessingMode.MULTI_THREADED,
        ProcessingMode.ASYNC_CONCURRENT,
        ProcessingMode.HYBRID
    ]
    
    results = {}
    
    for mode in modes_to_test:
        print(f"\n🧪 Testing {mode.value} mode:")
        print("-" * 40)
        
        engine = ScalableBioneuroEngine(
            processing_mode=mode,
            max_workers=4,
            enable_caching=True,
            cache_size=500
        )
        
        # Run benchmark
        benchmark_results = engine.benchmark(iterations=2)
        results[mode.value] = benchmark_results
        
        # Clean up
        del engine
    
    # Performance comparison
    print(f"\n📈 Performance Comparison:")
    print("-" * 40)
    for mode, result in results.items():
        throughput = result['average_throughput_docs_per_sec']
        cache_hit = result['cache_hit_rate']
        memory = result['memory_usage_mb']
        
        print(f"{mode:15}: {throughput:6.0f} docs/sec | {cache_hit:5.1%} cache | {memory:5.1f} MB")
    
    # Save results
    output_file = Path('scalability_benchmark_results.json')
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': time.time(),
            'benchmark_results': results,
            'system_info': {
                'cpu_count': multiprocessing.cpu_count(),
                'python_version': sys.version
            }
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    results = demonstrate_scalability()
    print(f"\n✅ Scalability demonstration completed!")
    print(f"🚀 Best performance: {max(r['average_throughput_docs_per_sec'] for r in results.values()):.0f} docs/sec")