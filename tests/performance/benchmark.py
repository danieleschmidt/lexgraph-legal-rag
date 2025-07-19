#!/usr/bin/env python3
"""
Performance benchmarking script for LexGraph Legal RAG.
Tests various components and generates detailed performance reports.
"""

import asyncio
import time
import statistics
import json
import sys
from typing import List, Dict, Any
from dataclasses import dataclass
import tempfile
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lexgraph_legal_rag.cache import LRUCache, QueryResultCache
from lexgraph_legal_rag.document_pipeline import VectorIndex, LegalDocumentPipeline
from lexgraph_legal_rag.faiss_index import FaissVectorIndex
from lexgraph_legal_rag.models import LegalDocument


@dataclass
class BenchmarkResult:
    """Results from a benchmark test."""
    name: str
    avg_time: float
    min_time: float
    max_time: float
    p95_time: float
    throughput: float
    error_rate: float
    metadata: Dict[str, Any] = None


class PerformanceBenchmark:
    """Performance benchmark suite for LexGraph Legal RAG."""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        
    def create_test_documents(self, count: int = 1000) -> List[LegalDocument]:
        """Create test documents for benchmarking."""
        documents = []
        legal_texts = [
            "This contract establishes the terms and conditions for employment.",
            "The party agrees to indemnify and hold harmless the other party.",
            "Intellectual property rights shall remain with the original creator.",
            "Confidentiality obligations shall survive termination of this agreement.",
            "Any disputes shall be resolved through binding arbitration.",
            "The governing law for this contract shall be the laws of California.",
            "Force majeure events include natural disasters and acts of war.",
            "Payment terms require net 30 days from invoice date.",
            "Termination may occur with 30 days written notice.",
            "All modifications must be in writing and signed by both parties.",
        ]
        
        for i in range(count):
            text_index = i % len(legal_texts)
            doc = LegalDocument(
                id=f"doc_{i:04d}",
                text=f"{legal_texts[text_index]} Document {i} additional content for variety.",
                metadata={"path": f"/test/doc_{i:04d}.txt", "category": f"cat_{i % 5}"}
            )
            documents.append(doc)
        
        return documents
    
    def benchmark_cache_performance(self, iterations: int = 10000) -> BenchmarkResult:
        """Benchmark cache operations."""
        cache = LRUCache(max_size=1000, ttl_seconds=3600)
        
        # Pre-populate cache
        for i in range(500):
            cache.put(f"key_{i}", f"value_{i}")
        
        times = []
        errors = 0
        
        start_time = time.time()
        
        for i in range(iterations):
            operation_start = time.perf_counter()
            
            try:
                if i % 4 == 0:  # 25% writes
                    cache.put(f"bench_key_{i}", f"bench_value_{i}")
                else:  # 75% reads
                    key = f"key_{i % 500}"
                    cache.get(key)
            except Exception:
                errors += 1
            
            operation_end = time.perf_counter()
            times.append((operation_end - operation_start) * 1000)  # Convert to ms
        
        total_time = time.time() - start_time
        
        return BenchmarkResult(
            name="Cache Operations",
            avg_time=statistics.mean(times),
            min_time=min(times),
            max_time=max(times),
            p95_time=self._percentile(times, 95),
            throughput=iterations / total_time,
            error_rate=errors / iterations,
            metadata={"cache_size": len(cache._cache), "hit_rate": cache.get_stats()["hit_rate"]}
        )
    
    def benchmark_vector_index(self, doc_count: int = 1000, query_count: int = 100) -> BenchmarkResult:
        """Benchmark vector index operations."""
        documents = self.create_test_documents(doc_count)
        index = VectorIndex()
        
        # Index creation time
        index_start = time.perf_counter()
        index.add(documents)
        index_time = time.perf_counter() - index_start
        
        # Search performance
        queries = [
            "employment contract terms",
            "intellectual property rights",
            "indemnification clause",
            "confidentiality agreement",
            "termination notice",
        ]
        
        search_times = []
        errors = 0
        
        start_time = time.time()
        
        for i in range(query_count):
            query = queries[i % len(queries)]
            
            search_start = time.perf_counter()
            try:
                results = index.search(query, top_k=10)
                if not results:
                    errors += 1
            except Exception:
                errors += 1
            search_end = time.perf_counter()
            
            search_times.append((search_end - search_start) * 1000)
        
        total_time = time.time() - start_time
        
        return BenchmarkResult(
            name="Vector Index Search",
            avg_time=statistics.mean(search_times),
            min_time=min(search_times),
            max_time=max(search_times),
            p95_time=self._percentile(search_times, 95),
            throughput=query_count / total_time,
            error_rate=errors / query_count,
            metadata={
                "document_count": len(documents),
                "index_creation_time_ms": index_time * 1000,
                "matrix_shape": index._matrix.shape if index._matrix is not None else None
            }
        )
    
    def benchmark_faiss_index(self, doc_count: int = 1000, query_count: int = 100) -> BenchmarkResult:
        """Benchmark FAISS index operations."""
        documents = self.create_test_documents(doc_count)
        index = FaissVectorIndex()
        
        # Index creation
        index_start = time.perf_counter()
        index.add(documents)
        index_time = time.perf_counter() - index_start
        
        # Search performance
        queries = [
            "contract employment terms conditions",
            "intellectual property creator rights",
            "indemnify hold harmless party",
            "confidentiality survive termination",
            "disputes arbitration resolution",
        ]
        
        search_times = []
        errors = 0
        
        start_time = time.time()
        
        for i in range(query_count):
            query = queries[i % len(queries)]
            
            search_start = time.perf_counter()
            try:
                results = index.search(query, top_k=10)
                if not results:
                    errors += 1
            except Exception:
                errors += 1
            search_end = time.perf_counter()
            
            search_times.append((search_end - search_start) * 1000)
        
        total_time = time.time() - start_time
        
        return BenchmarkResult(
            name="FAISS Index Search",
            avg_time=statistics.mean(search_times),
            min_time=min(search_times),
            max_time=max(search_times),
            p95_time=self._percentile(search_times, 95),
            throughput=query_count / total_time,
            error_rate=errors / query_count,
            metadata={
                "document_count": len(documents),
                "index_creation_time_ms": index_time * 1000,
                "index_type": type(index.index).__name__ if index.index else None
            }
        )
    
    def benchmark_batch_operations(self, doc_count: int = 500, batch_size: int = 20) -> BenchmarkResult:
        """Benchmark batch search operations."""
        documents = self.create_test_documents(doc_count)
        index = VectorIndex()
        index.add(documents)
        
        queries = [
            "employment contract",
            "intellectual property",
            "indemnification clause",
            "confidentiality terms",
            "termination conditions",
            "payment terms net",
            "governing law california",
            "force majeure events",
            "arbitration disputes",
            "written modifications",
        ]
        
        # Single query baseline
        single_times = []
        for query in queries:
            start = time.perf_counter()
            index.search(query, top_k=5)
            end = time.perf_counter()
            single_times.append((end - start) * 1000)
        
        # Batch query performance
        batch_times = []
        iterations = 50
        
        for _ in range(iterations):
            batch_queries = queries[:batch_size]
            
            start = time.perf_counter()
            batch_results = index.batch_search(batch_queries, top_k=5)
            end = time.perf_counter()
            
            batch_times.append((end - start) * 1000)
        
        avg_single = statistics.mean(single_times)
        avg_batch = statistics.mean(batch_times)
        speedup = (avg_single * batch_size) / avg_batch
        
        return BenchmarkResult(
            name="Batch Search Operations",
            avg_time=avg_batch,
            min_time=min(batch_times),
            max_time=max(batch_times),
            p95_time=self._percentile(batch_times, 95),
            throughput=batch_size * iterations / sum(batch_times) * 1000,
            error_rate=0.0,
            metadata={
                "batch_size": batch_size,
                "single_query_avg_ms": avg_single,
                "batch_speedup": speedup,
                "efficiency": speedup / batch_size
            }
        )
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of a dataset."""
        data_sorted = sorted(data)
        index = int(len(data_sorted) * percentile / 100)
        return data_sorted[min(index, len(data_sorted) - 1)]
    
    def run_all_benchmarks(self) -> List[BenchmarkResult]:
        """Run all benchmark tests."""
        print("Starting performance benchmarks...")
        
        # Cache benchmark
        print("Running cache performance benchmark...")
        cache_result = self.benchmark_cache_performance()
        self.results.append(cache_result)
        
        # Vector index benchmark
        print("Running vector index benchmark...")
        vector_result = self.benchmark_vector_index()
        self.results.append(vector_result)
        
        # FAISS index benchmark
        print("Running FAISS index benchmark...")
        faiss_result = self.benchmark_faiss_index()
        self.results.append(faiss_result)
        
        # Batch operations benchmark
        print("Running batch operations benchmark...")
        batch_result = self.benchmark_batch_operations()
        self.results.append(batch_result)
        
        return self.results
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive performance report."""
        report = {
            "timestamp": time.time(),
            "benchmarks": []
        }
        
        for result in self.results:
            benchmark_data = {
                "name": result.name,
                "metrics": {
                    "avg_time_ms": round(result.avg_time, 3),
                    "min_time_ms": round(result.min_time, 3),
                    "max_time_ms": round(result.max_time, 3),
                    "p95_time_ms": round(result.p95_time, 3),
                    "throughput_ops_per_sec": round(result.throughput, 2),
                    "error_rate": round(result.error_rate, 4),
                },
                "metadata": result.metadata or {}
            }
            report["benchmarks"].append(benchmark_data)
        
        return report
    
    def print_results(self):
        """Print benchmark results to console."""
        print("\n" + "="*80)
        print("LEXGRAPH LEGAL RAG PERFORMANCE BENCHMARK RESULTS")
        print("="*80)
        
        for result in self.results:
            print(f"\n{result.name}")
            print("-" * len(result.name))
            print(f"Average Time:    {result.avg_time:.3f} ms")
            print(f"95th Percentile: {result.p95_time:.3f} ms")
            print(f"Throughput:      {result.throughput:.2f} ops/sec")
            print(f"Error Rate:      {result.error_rate:.2%}")
            
            if result.metadata:
                print("Metadata:")
                for key, value in result.metadata.items():
                    print(f"  {key}: {value}")


def main():
    """Run the performance benchmark suite."""
    benchmark = PerformanceBenchmark()
    
    try:
        results = benchmark.run_all_benchmarks()
        benchmark.print_results()
        
        # Save detailed report
        report = benchmark.generate_report()
        
        output_file = "performance-benchmark-report.json"
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nDetailed report saved to: {output_file}")
        
        # Check if performance meets expectations
        critical_metrics = []
        for result in results:
            if result.p95_time > 1000:  # More than 1 second is concerning
                critical_metrics.append(f"{result.name}: {result.p95_time:.3f}ms")
            if result.error_rate > 0.01:  # More than 1% error rate
                critical_metrics.append(f"{result.name}: {result.error_rate:.2%} errors")
        
        if critical_metrics:
            print("\n⚠️  PERFORMANCE CONCERNS:")
            for metric in critical_metrics:
                print(f"  - {metric}")
            return 1
        else:
            print("\n✅ All performance benchmarks passed!")
            return 0
            
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())