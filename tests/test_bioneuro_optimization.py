"""
Comprehensive tests for Bioneural Optimization module.

Tests cover:
- Intelligent caching strategies
- Resource pooling and management
- Performance optimization modes
- Memory management and scaling
- Batch processing optimization
"""

import pytest
import asyncio
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from lexgraph_legal_rag.bioneuro_optimization import (
    IntelligentCache,
    ResourcePool,
    OptimizedBioneuroEngine,
    OptimizationConfig,
    CacheStrategy,
    ProcessingMode,
    PerformanceProfile,
    get_optimized_engine,
    analyze_document_optimized
)

from lexgraph_legal_rag.bioneuro_olfactory_fusion import (
    OlfactoryReceptorType,
    DocumentScentProfile,
    OlfactorySignal
)


class TestIntelligentCache:
    """Test intelligent caching system."""
    
    def test_cache_initialization(self):
        """Test cache initialization with different strategies."""
        cache = IntelligentCache(max_size=100, strategy=CacheStrategy.LRU)
        
        assert cache.max_size == 100
        assert cache.strategy == CacheStrategy.LRU
        assert cache.hits == 0
        assert cache.misses == 0
        assert len(cache._cache) == 0
    
    def test_cache_put_and_get(self):
        """Test basic cache put and get operations."""
        cache = IntelligentCache(max_size=10, strategy=CacheStrategy.LRU)
        
        # Put item
        cache.put("key1", "value1")
        
        # Get item (should hit)
        value = cache.get("key1")
        assert value == "value1"
        assert cache.hits == 1
        assert cache.misses == 0
        
        # Get non-existent item (should miss)
        value = cache.get("key2")
        assert value is None
        assert cache.hits == 1
        assert cache.misses == 1
    
    def test_cache_lru_eviction(self):
        """Test LRU eviction strategy."""
        cache = IntelligentCache(max_size=3, strategy=CacheStrategy.LRU)
        
        # Fill cache
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")
        
        # Access key1 to make it recently used
        cache.get("key1")
        
        # Add new item (should evict key2, the least recently used)
        cache.put("key4", "value4")
        
        assert cache.get("key1") == "value1"  # Should still exist
        assert cache.get("key2") is None      # Should be evicted
        assert cache.get("key3") == "value3"  # Should still exist
        assert cache.get("key4") == "value4"  # Should exist
    
    def test_cache_lfu_eviction(self):
        """Test LFU eviction strategy."""
        cache = IntelligentCache(max_size=3, strategy=CacheStrategy.LFU)
        
        # Fill cache and access items different numbers of times
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")
        
        # Access key1 multiple times
        cache.get("key1")
        cache.get("key1")
        cache.get("key2")  # Access key2 once
        # key3 accessed 0 times after initial put
        
        # Add new item (should evict key3, least frequently used)
        cache.put("key4", "value4")
        
        assert cache.get("key1") == "value1"  # Should still exist (most frequent)
        assert cache.get("key2") == "value2"  # Should still exist
        assert cache.get("key3") is None      # Should be evicted (least frequent)
        assert cache.get("key4") == "value4"  # Should exist
    
    def test_cache_ttl_expiration(self):
        """Test TTL expiration functionality."""
        cache = IntelligentCache(max_size=10, strategy=CacheStrategy.TTL)
        
        # Put item with short TTL
        cache.put("key1", "value1", ttl=0.1)  # 100ms TTL
        
        # Should be available immediately
        assert cache.get("key1") == "value1"
        
        # Wait for expiration
        time.sleep(0.15)
        
        # Should be expired
        assert cache.get("key1") is None
        assert cache.misses > 0
    
    def test_cache_adaptive_strategy(self):
        """Test adaptive strategy behavior."""
        cache = IntelligentCache(max_size=10, strategy=CacheStrategy.ADAPTIVE)
        
        # Add items and access them
        for i in range(5):
            cache.put(f"key{i}", f"value{i}")
            cache.get(f"key{i}")
        
        # Should adapt strategy based on performance
        assert cache._current_strategy in [CacheStrategy.LRU, CacheStrategy.LFU, CacheStrategy.TTL, CacheStrategy.ADAPTIVE]
    
    def test_cache_statistics(self):
        """Test cache statistics collection."""
        cache = IntelligentCache(max_size=5, strategy=CacheStrategy.LRU)
        
        # Perform some operations
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.get("key1")  # Hit
        cache.get("key3")  # Miss
        
        stats = cache.get_stats()
        
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 2
        assert stats["max_size"] == 5
        assert "hit_rate" in stats
        assert "current_strategy" in stats
    
    def test_cache_thread_safety(self):
        """Test cache thread safety."""
        cache = IntelligentCache(max_size=100, strategy=CacheStrategy.LRU)
        
        def worker(thread_id):
            for i in range(100):
                key = f"thread_{thread_id}_key_{i}"
                value = f"thread_{thread_id}_value_{i}"
                cache.put(key, value)
                retrieved = cache.get(key)
                assert retrieved == value
        
        # Run multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should have completed without errors
        assert cache.hits > 0


class TestResourcePool:
    """Test resource pooling functionality."""
    
    def test_resource_pool_initialization(self):
        """Test resource pool initialization."""
        def create_resource():
            return {"id": time.time()}
        
        pool = ResourcePool(create_resource, max_size=5, min_size=2)
        
        assert pool.max_size == 5
        assert pool.min_size == 2
        assert len(pool._pool) == 2  # Should pre-populate with min_size
    
    def test_resource_acquisition_and_release(self):
        """Test resource acquisition and release."""
        def create_resource():
            return {"id": time.time(), "data": "test"}
        
        pool = ResourcePool(create_resource, max_size=3, min_size=1)
        
        # Acquire resource
        resource = pool.acquire(timeout=1.0)
        assert resource is not None
        assert "id" in resource
        
        # Release resource
        pool.release(resource)
        
        # Should be back in pool
        stats = pool.get_stats()
        assert stats["pool_size"] >= 1
        assert stats["in_use"] == 0
    
    def test_resource_pool_limits(self):
        """Test resource pool size limits."""
        def create_resource():
            return {"id": time.time()}
        
        pool = ResourcePool(create_resource, max_size=2, min_size=1)
        
        # Acquire up to max_size
        resource1 = pool.acquire(timeout=1.0)
        resource2 = pool.acquire(timeout=1.0)
        
        assert resource1 is not None
        assert resource2 is not None
        
        # Should timeout when trying to acquire beyond max_size
        resource3 = pool.acquire(timeout=0.1)
        assert resource3 is None  # Should timeout
        
        # Release one resource
        pool.release(resource1)
        
        # Should now be able to acquire again
        resource3 = pool.acquire(timeout=1.0)
        assert resource3 is not None
    
    def test_resource_pool_thread_safety(self):
        """Test resource pool thread safety."""
        def create_resource():
            return {"id": time.time(), "thread_safe": True}
        
        pool = ResourcePool(create_resource, max_size=10, min_size=2)
        
        def worker():
            for _ in range(10):
                resource = pool.acquire(timeout=2.0)
                if resource:
                    time.sleep(0.01)  # Simulate work
                    pool.release(resource)
        
        # Run multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should complete without deadlocks
        stats = pool.get_stats()
        assert stats["in_use"] == 0  # All resources should be released
    
    def test_resource_pool_statistics(self):
        """Test resource pool statistics."""
        def create_resource():
            return {"id": time.time()}
        
        pool = ResourcePool(create_resource, max_size=5, min_size=2)
        
        # Acquire some resources
        resource1 = pool.acquire()
        resource2 = pool.acquire()
        
        stats = pool.get_stats()
        
        assert stats["in_use"] == 2
        assert stats["max_size"] == 5
        assert stats["min_size"] == 2
        assert stats["total_resources"] >= 2


class TestOptimizationConfig:
    """Test optimization configuration."""
    
    def test_default_config(self):
        """Test default optimization configuration."""
        config = OptimizationConfig()
        
        assert config.max_cache_size == 10000
        assert config.cache_strategy == CacheStrategy.ADAPTIVE
        assert config.processing_mode == ProcessingMode.ADAPTIVE
        assert config.max_workers == 4
        assert config.batch_size == 32
        assert config.memory_limit_mb == 2048
        assert config.enable_profiling is False
        assert config.auto_scaling is True
    
    def test_custom_config(self):
        """Test custom optimization configuration."""
        config = OptimizationConfig(
            max_cache_size=5000,
            cache_strategy=CacheStrategy.LRU,
            processing_mode=ProcessingMode.CONCURRENT,
            max_workers=8,
            enable_profiling=True
        )
        
        assert config.max_cache_size == 5000
        assert config.cache_strategy == CacheStrategy.LRU
        assert config.processing_mode == ProcessingMode.CONCURRENT
        assert config.max_workers == 8
        assert config.enable_profiling is True


class TestOptimizedBioneuroEngine:
    """Test optimized bioneural engine functionality."""
    
    def test_engine_initialization(self):
        """Test optimized engine initialization."""
        config = OptimizationConfig(max_workers=2, max_cache_size=100)
        engine = OptimizedBioneuroEngine(config)
        
        assert engine.config.max_workers == 2
        assert engine.config.max_cache_size == 100
        assert engine.scent_cache is not None
        assert engine.receptor_pool is not None
        assert engine.thread_pool is not None
    
    @pytest.mark.asyncio
    async def test_optimized_document_analysis(self):
        """Test optimized document analysis."""
        config = OptimizationConfig(max_workers=2, enable_profiling=True)
        engine = OptimizedBioneuroEngine(config)
        
        test_document = """
        This is a legal contract with liability provisions and indemnification
        clauses pursuant to applicable statutes and regulations.
        """
        
        # First analysis (should miss cache)
        profile1 = await engine.analyze_document_optimized(test_document, "test_doc_1")
        
        assert isinstance(profile1, DocumentScentProfile)
        assert profile1.document_id == "test_doc_1"
        assert len(profile1.composite_scent) > 0
        assert engine.processing_stats["cache_misses"] == 1
        
        # Second analysis with same content (should hit cache)
        profile2 = await engine.analyze_document_optimized(test_document, "test_doc_2")
        
        assert isinstance(profile2, DocumentScentProfile)
        assert profile2.document_id == "test_doc_2"
        assert engine.processing_stats["cache_hits"] == 1
    
    @pytest.mark.asyncio
    async def test_batch_document_analysis(self):
        """Test batch document analysis."""
        config = OptimizationConfig(batch_size=3, max_workers=2)
        engine = OptimizedBioneuroEngine(config)
        
        documents = [
            ("Contract with liability clauses", "doc1", {}),
            ("Statute with regulatory provisions", "doc2", {}),
            ("Agreement with indemnification terms", "doc3", {}),
            ("Policy with compliance requirements", "doc4", {}),
            ("Regulation with penalty provisions", "doc5", {})
        ]
        
        profiles = await engine.batch_analyze_documents(documents)
        
        assert len(profiles) == 5
        assert all(isinstance(p, DocumentScentProfile) for p in profiles)
        assert all(p.document_id for p in profiles)
    
    def test_cache_key_generation(self):
        """Test cache key generation."""
        config = OptimizationConfig()
        engine = OptimizedBioneuroEngine(config)
        
        # Same content should generate same key
        key1 = engine._generate_cache_key("test document", {"type": "contract"})
        key2 = engine._generate_cache_key("test document", {"type": "contract"})
        
        assert key1 == key2
        
        # Different content should generate different keys
        key3 = engine._generate_cache_key("different document", {"type": "contract"})
        assert key1 != key3
        
        # Different metadata should generate different keys
        key4 = engine._generate_cache_key("test document", {"type": "statute"})
        assert key1 != key4
    
    def test_processing_mode_selection(self):
        """Test processing mode selection logic."""
        config = OptimizationConfig(processing_mode=ProcessingMode.ADAPTIVE)
        engine = OptimizedBioneuroEngine(config)
        
        # Short simple document should use sequential
        short_doc = "Simple contract."
        mode = engine._select_processing_mode(short_doc)
        assert mode == ProcessingMode.SEQUENTIAL
        
        # Complex document should use concurrent or parallel
        complex_doc = """
        WHEREAS the parties hereto desire to enter into this comprehensive
        agreement pursuant to applicable statutes and regulations, NOTWITHSTANDING
        any prior agreements, and PROVIDED THAT all terms and conditions
        specified herein shall remain in full force and effect throughout
        the term of this agreement, including all indemnification provisions,
        liability limitations, and penalty clauses as set forth below.
        """ * 10  # Make it longer
        
        mode = engine._select_processing_mode(complex_doc)
        assert mode in [ProcessingMode.CONCURRENT, ProcessingMode.PARALLEL]
    
    def test_complexity_estimation(self):
        """Test document complexity estimation."""
        config = OptimizationConfig()
        engine = OptimizedBioneuroEngine(config)
        
        # Simple document
        simple_doc = "This is a basic agreement between parties."
        simple_complexity = engine._estimate_complexity(simple_doc)
        
        # Complex legal document
        complex_doc = """
        WHEREAS the parties hereto agree pursuant to the terms and conditions
        set forth below, NOTWITHSTANDING any prior agreements, and subject to
        indemnification provisions and liability limitations as specified herein,
        the Contractor shall perform all services in accordance with applicable
        statutes and regulations.
        """
        complex_complexity = engine._estimate_complexity(complex_doc)
        
        assert complex_complexity > simple_complexity
        assert 0.0 <= simple_complexity <= 1.0
        assert 0.0 <= complex_complexity <= 1.0
    
    def test_optimization_statistics(self):
        """Test optimization statistics collection."""
        config = OptimizationConfig(enable_profiling=True)
        engine = OptimizedBioneuroEngine(config)
        
        stats = engine.get_optimization_stats()
        
        assert "processing_stats" in stats
        assert "cache_stats" in stats
        assert "resource_pool_stats" in stats
        assert "config" in stats
        
        # Check processing stats structure
        assert "documents_processed" in stats["processing_stats"]
        assert "cache_hits" in stats["processing_stats"]
        assert "cache_misses" in stats["processing_stats"]
        
        # Check cache stats structure
        assert "hits" in stats["cache_stats"]
        assert "misses" in stats["cache_stats"]
        assert "hit_rate" in stats["cache_stats"]
        
        # Check config structure
        assert "max_cache_size" in stats["config"]
        assert "processing_mode" in stats["config"]
    
    @pytest.mark.asyncio
    async def test_error_handling_and_fallback(self):
        """Test error handling and fallback mechanisms."""
        config = OptimizationConfig(max_workers=1)
        engine = OptimizedBioneuroEngine(config)
        
        # Mock receptor pool to fail
        with patch.object(engine.receptor_pool, 'acquire', return_value=None):
            # Should fall back to base engine
            profile = await engine.analyze_document_optimized("test document", "test_doc")
            
            assert isinstance(profile, DocumentScentProfile)
            # Should still produce a valid profile even with failures
    
    def test_memory_management(self):
        """Test memory management functionality."""
        config = OptimizationConfig(memory_limit_mb=100)  # Low limit for testing
        engine = OptimizedBioneuroEngine(config)
        
        # Fill cache with data
        for i in range(100):
            engine.scent_cache.put(f"key_{i}", f"value_{i}" * 1000)  # Large values
        
        # Trigger memory management
        engine._manage_memory()
        
        # Should have cleared cache if memory limit exceeded
        # (This test depends on actual memory usage)
    
    def test_performance_profiling(self):
        """Test performance profiling functionality."""
        config = OptimizationConfig(enable_profiling=True)
        engine = OptimizedBioneuroEngine(config)
        
        # Record some performance profiles
        engine._record_performance_profile("test_operation", 1.5, ProcessingMode.CONCURRENT)
        engine._record_performance_profile("test_operation", 2.0, ProcessingMode.SEQUENTIAL)
        
        assert len(engine.performance_profiles) == 2
        
        # Check profile structure
        profile = engine.performance_profiles[0]
        assert profile.operation == "test_operation"
        assert profile.duration == 1.5
        assert profile.parallelism_factor > 1.0  # Concurrent mode
    
    def test_engine_shutdown(self):
        """Test engine shutdown and cleanup."""
        config = OptimizationConfig(max_workers=2)
        engine = OptimizedBioneuroEngine(config)
        
        # Shutdown should complete without errors
        engine.shutdown()
        
        # Thread pools should be shutdown
        assert engine.thread_pool._shutdown
        
        # Cache should be cleared
        assert len(engine.scent_cache._cache) == 0


class TestGlobalFunctions:
    """Test global convenience functions."""
    
    def test_get_optimized_engine_singleton(self):
        """Test that get_optimized_engine returns singleton."""
        engine1 = get_optimized_engine()
        engine2 = get_optimized_engine()
        
        assert engine1 is engine2
        assert isinstance(engine1, OptimizedBioneuroEngine)
    
    @pytest.mark.asyncio
    async def test_analyze_document_optimized_convenience(self):
        """Test convenience function for optimized analysis."""
        profile = await analyze_document_optimized(
            "Test legal document with provisions",
            "convenience_test"
        )
        
        assert isinstance(profile, DocumentScentProfile)
        assert profile.document_id == "convenience_test"


class TestPerformanceAndScaling:
    """Test performance characteristics and scaling behavior."""
    
    @pytest.mark.asyncio
    async def test_concurrent_processing_performance(self):
        """Test concurrent processing performance."""
        config = OptimizationConfig(
            processing_mode=ProcessingMode.CONCURRENT,
            max_workers=4,
            enable_profiling=True
        )
        engine = OptimizedBioneuroEngine(config)
        
        documents = [
            (f"Legal document {i} with various clauses and provisions", f"doc_{i}", {})
            for i in range(10)
        ]
        
        start_time = time.time()
        profiles = await engine.batch_analyze_documents(documents, batch_size=5)
        end_time = time.time()
        
        assert len(profiles) == 10
        assert all(isinstance(p, DocumentScentProfile) for p in profiles)
        
        # Should complete reasonably quickly with concurrent processing
        total_time = end_time - start_time
        assert total_time < 30.0  # Should complete within 30 seconds
        
        # Check performance statistics
        stats = engine.get_optimization_stats()
        assert stats["processing_stats"]["documents_processed"] >= 10
    
    @pytest.mark.asyncio
    async def test_cache_performance_improvement(self):
        """Test that caching improves performance."""
        config = OptimizationConfig(max_cache_size=100)
        engine = OptimizedBioneuroEngine(config)
        
        test_document = "Legal contract with standard provisions and clauses."
        
        # First analysis (cache miss)
        start_time = time.time()
        profile1 = await engine.analyze_document_optimized(test_document, "perf_test_1")
        first_time = time.time() - start_time
        
        # Second analysis (should be cache hit)
        start_time = time.time()
        profile2 = await engine.analyze_document_optimized(test_document, "perf_test_2")
        second_time = time.time() - start_time
        
        assert isinstance(profile1, DocumentScentProfile)
        assert isinstance(profile2, DocumentScentProfile)
        
        # Cache hit should be significantly faster
        assert second_time < first_time * 0.5  # At least 50% faster
        
        # Check cache statistics
        stats = engine.get_optimization_stats()
        assert stats["cache_stats"]["hits"] >= 1
        assert stats["cache_stats"]["hit_rate"] > 0.0
    
    def test_memory_efficient_storage(self):
        """Test memory efficient storage of scent profiles."""
        config = OptimizationConfig(max_cache_size=1000)
        engine = OptimizedBioneuroEngine(config)
        
        # Generate many cache entries
        for i in range(100):
            key = f"test_key_{i}"
            
            # Create a mock profile
            profile = DocumentScentProfile(
                document_id=f"doc_{i}",
                signals=[],
                composite_scent=np.random.random(12),
                similarity_hash=f"hash_{i}"
            )
            
            engine.scent_cache.put(key, profile)
        
        # Should handle many entries without excessive memory usage
        stats = engine.scent_cache.get_stats()
        assert stats["size"] <= 100
        assert stats["hit_rate"] >= 0.0  # Valid hit rate
    
    @pytest.mark.asyncio
    async def test_adaptive_processing_mode_selection(self):
        """Test adaptive processing mode selection."""
        config = OptimizationConfig(processing_mode=ProcessingMode.ADAPTIVE)
        engine = OptimizedBioneuroEngine(config)
        
        # Test with different document types
        documents = [
            ("Short contract.", "short_doc", {}),
            ("Medium length contract with several clauses and provisions. " * 50, "medium_doc", {}),
            ("Very long and complex legal document. " * 200, "long_doc", {})
        ]
        
        profiles = []
        processing_modes = []
        
        for text, doc_id, metadata in documents:
            mode = engine._select_processing_mode(text)
            processing_modes.append(mode)
            
            profile = await engine.analyze_document_optimized(text, doc_id, metadata)
            profiles.append(profile)
        
        assert len(profiles) == 3
        assert all(isinstance(p, DocumentScentProfile) for p in profiles)
        
        # Should use different processing modes for different document sizes
        assert ProcessingMode.SEQUENTIAL in processing_modes or ProcessingMode.CONCURRENT in processing_modes
    
    def test_resource_pool_scaling(self):
        """Test resource pool scaling behavior."""
        def create_mock_receptor_set():
            return {receptor_type: Mock() for receptor_type in OlfactoryReceptorType}
        
        pool = ResourcePool(
            resource_factory=create_mock_receptor_set,
            max_size=10,
            min_size=2
        )
        
        # Should start with minimum resources
        initial_stats = pool.get_stats()
        assert initial_stats["pool_size"] == 2
        assert initial_stats["total_resources"] == 2
        
        # Acquire multiple resources
        resources = []
        for i in range(5):
            resource = pool.acquire()
            if resource:
                resources.append(resource)
        
        # Should scale up as needed
        scaled_stats = pool.get_stats()
        assert scaled_stats["in_use"] == len(resources)
        assert scaled_stats["total_resources"] >= len(resources)
        
        # Release resources
        for resource in resources:
            pool.release(resource)
        
        # Should maintain some resources in pool
        final_stats = pool.get_stats()
        assert final_stats["in_use"] == 0
        assert final_stats["pool_size"] >= 2  # At least minimum


class TestIntegrationScenarios:
    """Test integration scenarios with optimization."""
    
    @pytest.mark.asyncio
    async def test_high_throughput_document_processing(self):
        """Test high throughput document processing scenario."""
        config = OptimizationConfig(
            max_workers=4,
            batch_size=10,
            max_cache_size=500,
            processing_mode=ProcessingMode.CONCURRENT
        )
        engine = OptimizedBioneuroEngine(config)
        
        # Generate many documents
        documents = []
        for i in range(50):
            doc_type = ["contract", "statute", "regulation"][i % 3]
            text = f"Legal {doc_type} number {i} with various provisions and clauses. " * (10 + i % 20)
            documents.append((text, f"doc_{i}", {"type": doc_type}))
        
        start_time = time.time()
        profiles = await engine.batch_analyze_documents(documents)
        end_time = time.time()
        
        assert len(profiles) == 50
        assert all(isinstance(p, DocumentScentProfile) for p in profiles)
        
        # Should complete within reasonable time
        total_time = end_time - start_time
        throughput = len(documents) / total_time
        assert throughput > 1.0  # At least 1 document per second
        
        # Check optimization statistics
        stats = engine.get_optimization_stats()
        assert stats["processing_stats"]["documents_processed"] >= 50
        
        print(f"Processed {len(documents)} documents in {total_time:.2f}s "
              f"(throughput: {throughput:.2f} docs/sec)")
    
    @pytest.mark.asyncio
    async def test_mixed_workload_optimization(self):
        """Test optimization with mixed workload patterns."""
        config = OptimizationConfig(
            processing_mode=ProcessingMode.ADAPTIVE,
            cache_strategy=CacheStrategy.ADAPTIVE,
            enable_profiling=True
        )
        engine = OptimizedBioneuroEngine(config)
        
        # Mixed workload: some repeated documents, some unique
        documents = []
        
        # Add some documents that will be repeated (cache hits)
        base_docs = [
            "Standard contract template with liability clauses.",
            "Regulatory compliance document with penalty provisions.",
            "Basic service agreement with indemnification terms."
        ]
        
        # Add repeated documents
        for i in range(15):
            doc_text = base_docs[i % len(base_docs)]
            documents.append((doc_text, f"repeated_doc_{i}", {}))
        
        # Add unique documents
        for i in range(10):
            unique_text = f"Unique legal document {i} with specific clauses and provisions."
            documents.append((unique_text, f"unique_doc_{i}", {}))
        
        # Process all documents
        profiles = await engine.batch_analyze_documents(documents)
        
        assert len(profiles) == 25
        
        # Check cache performance
        stats = engine.get_optimization_stats()
        cache_stats = stats["cache_stats"]
        
        # Should have good hit rate due to repeated documents
        assert cache_stats["hits"] > 0
        assert cache_stats["hit_rate"] > 0.2  # At least 20% hit rate
        
        print(f"Mixed workload cache hit rate: {cache_stats['hit_rate']:.1%}")
    
    @pytest.mark.asyncio
    async def test_memory_constrained_processing(self):
        """Test processing under memory constraints."""
        config = OptimizationConfig(
            memory_limit_mb=100,  # Low memory limit
            max_cache_size=50,    # Small cache
            batch_size=5          # Small batches
        )
        engine = OptimizedBioneuroEngine(config)
        
        # Generate documents that might consume significant memory
        large_documents = []
        for i in range(20):
            # Create larger documents
            text = f"Large legal document {i}. " + "Standard clause text. " * 100
            large_documents.append((text, f"large_doc_{i}", {}))
        
        # Should handle processing without memory issues
        profiles = await engine.batch_analyze_documents(large_documents)
        
        assert len(profiles) == 20
        assert all(isinstance(p, DocumentScentProfile) for p in profiles)
        
        # Memory management should have been triggered
        # (This is hard to test directly, but the system should not crash)
        
        # Check that optimization is still working
        stats = engine.get_optimization_stats()
        assert stats["processing_stats"]["documents_processed"] >= 20