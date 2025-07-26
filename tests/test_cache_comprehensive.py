"""Comprehensive tests for cache module to achieve 80%+ coverage."""

import time
import threading
from unittest.mock import patch, MagicMock
import pytest

from src.lexgraph_legal_rag.cache import (
    CacheEntry,
    LRUCache,
    QueryResultCache,
    get_query_cache,
    configure_cache,
)
from src.lexgraph_legal_rag.models import LegalDocument


class TestCacheEntry:
    """Test CacheEntry dataclass functionality."""
    
    def test_cache_entry_creation(self):
        """Test basic cache entry creation."""
        entry = CacheEntry(value="test_value", timestamp=1234567890.0)
        assert entry.value == "test_value"
        assert entry.timestamp == 1234567890.0
        assert entry.hit_count == 0
        assert isinstance(entry.last_accessed, float)
        assert entry.last_accessed > 0


class TestLRUCache:
    """Test LRU cache implementation with edge cases."""
    
    def test_lru_cache_basic_operations(self):
        """Test basic put/get operations."""
        cache = LRUCache(max_size=3, ttl_seconds=60.0)
        
        # Initially empty
        assert cache.get("nonexistent") is None
        
        # Put and get
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Stats tracking
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1
        assert stats["max_size"] == 3
    
    def test_lru_eviction_behavior(self):
        """Test that LRU items are evicted when cache is full."""
        cache = LRUCache(max_size=2, ttl_seconds=60.0)
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        # Access key1 to make it recently used
        assert cache.get("key1") == "value1"
        
        # Add key3, should evict key2 (least recently used)
        cache.put("key3", "value3")
        
        assert cache.get("key1") == "value1"  # Still present
        assert cache.get("key2") is None      # Evicted
        assert cache.get("key3") == "value3"  # Present
    
    def test_ttl_expiration(self):
        """Test that expired entries are not returned."""
        cache = LRUCache(max_size=10, ttl_seconds=0.1)  # 100ms TTL
        
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Wait for expiration
        time.sleep(0.2)
        
        assert cache.get("key1") is None
        
        # Should count as miss (1 miss from expired check)
        stats = cache.get_stats()
        assert stats["misses"] == 1
    
    def test_cache_update_existing_key(self):
        """Test updating an existing key moves it to end and updates value."""
        cache = LRUCache(max_size=2, ttl_seconds=60.0)
        
        cache.put("key1", "value1")
        cache.put("key2", "value2") 
        
        # Update key1 - should move to end
        cache.put("key1", "updated_value1")
        
        # Add key3 - should evict key2, not key1
        cache.put("key3", "value3")
        
        assert cache.get("key1") == "updated_value1"
        assert cache.get("key2") is None  # Evicted
        assert cache.get("key3") == "value3"
    
    def test_invalidate_operations(self):
        """Test cache invalidation functionality."""
        cache = LRUCache(max_size=5, ttl_seconds=60.0)
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        # Invalidate existing key
        assert cache.invalidate("key1") is True
        assert cache.get("key1") is None
        
        # Invalidate non-existent key
        assert cache.invalidate("nonexistent") is False
    
    def test_clear_operations(self):
        """Test cache clearing functionality."""
        cache = LRUCache(max_size=5, ttl_seconds=60.0)
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.get("key1")  # Generate some hits
        
        stats_before = cache.get_stats()
        assert stats_before["size"] == 2
        assert stats_before["hits"] == 1
        
        cache.clear()
        
        stats_after = cache.get_stats()
        assert stats_after["size"] == 0
        assert stats_after["hits"] == 0
        assert stats_after["misses"] == 0
    
    def test_cleanup_expired_entries(self):
        """Test manual cleanup of expired entries."""
        cache = LRUCache(max_size=10, ttl_seconds=0.1)
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        # Wait for expiration
        time.sleep(0.2)
        
        # Add non-expired entry
        cache.put("key3", "value3")
        
        # Manual cleanup should remove 2 expired entries
        removed_count = cache.cleanup_expired()
        assert removed_count == 2
        
        # Only key3 should remain
        assert cache.get_stats()["size"] == 1
        assert cache.get("key3") == "value3"
    
    def test_thread_safety(self):
        """Test cache behavior under concurrent access."""
        cache = LRUCache(max_size=100, ttl_seconds=60.0)
        errors = []
        
        def worker(thread_id):
            try:
                for i in range(50):
                    key = f"thread_{thread_id}_key_{i}"
                    value = f"thread_{thread_id}_value_{i}"
                    cache.put(key, value)
                    retrieved = cache.get(key)
                    if retrieved != value:
                        errors.append(f"Thread {thread_id}: Expected {value}, got {retrieved}")
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        assert len(errors) == 0, f"Thread safety errors: {errors}"
    
    @patch('src.lexgraph_legal_rag.cache.logger')
    def test_logging_behavior(self, mock_logger):
        """Test that appropriate log messages are generated."""
        cache = LRUCache(max_size=2, ttl_seconds=0.1)
        
        # Test eviction logging
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")  # Should trigger eviction
        
        mock_logger.debug.assert_called()
        
        # Test expiration logging
        time.sleep(0.2)
        cache.get("key2")  # Should log expiration
        
        # Test clear logging
        cache.clear()
        mock_logger.info.assert_called_with("Cache cleared")
    
    def test_hit_count_and_access_tracking(self):
        """Test that hit counts and access times are tracked correctly."""
        cache = LRUCache(max_size=5, ttl_seconds=60.0)
        cache.put("key1", "value1")
        
        start_time = time.time()
        
        # Multiple accesses should increment hit count
        cache.get("key1")
        cache.get("key1")
        cache.get("key1")
        
        # Access the internal entry to check hit count
        entry = cache._cache["key1"]
        assert entry.hit_count == 3
        assert entry.last_accessed >= start_time


class TestQueryResultCache:
    """Test query-specific cache functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.doc1 = LegalDocument(id="doc1", text="Legal document 1", metadata={"type": "case"})
        self.doc2 = LegalDocument(id="doc2", text="Legal document 2", metadata={"type": "statute"})
        self.results = [(self.doc1, 0.95), (self.doc2, 0.85)]
    
    def test_query_cache_basic_operations(self):
        """Test basic query caching operations."""
        cache = QueryResultCache(max_size=10, ttl_seconds=60.0)
        
        # Initially empty
        assert cache.get("test query", top_k=5) is None
        
        # Cache results
        cache.put("test query", top_k=5, results=self.results)
        
        # Retrieve results
        cached_results = cache.get("test query", top_k=5)
        assert cached_results == self.results
    
    def test_query_normalization(self):
        """Test that queries are normalized for consistent caching."""
        cache = QueryResultCache()
        
        # These should all resolve to the same cache key
        queries = [
            "Test Query",
            "test query",
            "  test   query  ",
            "TEST\t\tQUERY",
        ]
        
        # Cache with first variant
        cache.put(queries[0], top_k=5, results=self.results)
        
        # All variants should retrieve the same results
        for query in queries:
            cached_results = cache.get(query, top_k=5)
            assert cached_results == self.results
    
    def test_cache_key_generation(self):
        """Test that cache keys are generated consistently."""
        cache = QueryResultCache()
        
        # Different parameters should generate different keys
        cache.put("query1", top_k=5, results=self.results[:1])
        cache.put("query1", top_k=10, results=self.results)  # Different top_k
        cache.put("query1", top_k=5, results=self.results, semantic=True)  # Different semantic flag
        
        # Each should be cached separately
        assert len(cache.get("query1", top_k=5)) == 1
        assert len(cache.get("query1", top_k=10)) == 2
        assert len(cache.get("query1", top_k=5, semantic=True)) == 2
    
    def test_extra_params_in_cache_key(self):
        """Test that extra parameters affect cache key generation."""
        cache = QueryResultCache()
        
        extra_params1 = {"jurisdiction": "US", "date_range": "2020-2023"}
        extra_params2 = {"jurisdiction": "UK", "date_range": "2020-2023"}
        
        cache.put("legal query", top_k=5, results=self.results[:1], extra_params=extra_params1)
        cache.put("legal query", top_k=5, results=self.results, extra_params=extra_params2)
        
        # Different params should retrieve different results
        results1 = cache.get("legal query", top_k=5, extra_params=extra_params1)
        results2 = cache.get("legal query", top_k=5, extra_params=extra_params2)
        
        assert len(results1) == 1
        assert len(results2) == 2
    
    def test_invalidate_pattern(self):
        """Test pattern-based cache invalidation."""
        cache = QueryResultCache()
        
        cache.put("query1", top_k=5, results=self.results)
        cache.put("query2", top_k=5, results=self.results)
        
        # Invalidation should clear entire cache (current implementation)
        invalidated_count = cache.invalidate_pattern("*")
        
        assert invalidated_count == 1  # Returns 1 as per current implementation
        assert cache.get("query1", top_k=5) is None
        assert cache.get("query2", top_k=5) is None
    
    def test_cache_stats_with_query_metrics(self):
        """Test that query cache provides enhanced statistics."""
        cache = QueryResultCache(enable_semantic_cache=True)
        
        cache.put("query1", top_k=5, results=self.results)
        cache.get("query1", top_k=5)  # Generate hit
        cache.get("query2", top_k=5)  # Generate miss
        
        stats = cache.get_stats()
        
        assert "semantic_cache_enabled" in stats
        assert stats["semantic_cache_enabled"] is True
        assert "query_variations" in stats
        assert stats["hits"] >= 1
        assert stats["misses"] >= 1
    
    @patch('src.lexgraph_legal_rag.cache.logger')
    def test_query_cache_logging(self, mock_logger):
        """Test that query cache operations are logged appropriately."""
        cache = QueryResultCache()
        
        cache.put("test query", top_k=5, results=self.results)
        mock_logger.debug.assert_called()
        
        cache.invalidate_pattern("*")
        mock_logger.info.assert_called()


class TestGlobalCacheManagement:
    """Test global cache instance management."""
    
    def test_get_query_cache_singleton(self):
        """Test that get_query_cache returns singleton instance."""
        cache1 = get_query_cache()
        cache2 = get_query_cache()
        
        assert cache1 is cache2
    
    def test_configure_cache(self):
        """Test cache configuration functionality."""
        # Configure with custom settings
        configure_cache(max_size=100, ttl_seconds=300.0, enable_semantic_cache=False)
        
        cache = get_query_cache()
        stats = cache.get_stats()
        
        assert stats["max_size"] == 100  
        assert stats["ttl_seconds"] == 300.0
        assert stats["semantic_cache_enabled"] is False
    
    @patch('src.lexgraph_legal_rag.cache.logger')
    def test_configure_cache_logging(self, mock_logger):
        """Test that cache configuration is logged."""
        configure_cache(max_size=200, ttl_seconds=600.0)
        
        mock_logger.info.assert_called_with(
            "Query cache configured: max_size=200, ttl=600.0s"
        )


class TestCacheMemoryManagement:
    """Test cache behavior under memory pressure and load."""
    
    def test_large_cache_operations(self):
        """Test cache performance with large number of entries."""
        cache = LRUCache(max_size=1000, ttl_seconds=60.0)
        
        # Add many entries
        for i in range(1500):  # More than max_size
            cache.put(f"key_{i}", f"value_{i}")
        
        # Cache should be at max size
        stats = cache.get_stats()
        assert stats["size"] == 1000
        
        # Oldest entries should be evicted
        assert cache.get("key_0") is None
        assert cache.get("key_499") is None
        
        # Recent entries should be present
        assert cache.get("key_1499") == "value_1499"
        assert cache.get("key_1000") == "value_1000"
    
    def test_memory_efficient_key_hashing(self):
        """Test that cache keys are efficiently hashed."""
        cache = QueryResultCache()
        
        # Very long query should generate short, consistent key
        long_query = "a" * 10000
        cache.put(long_query, top_k=5, results=[(LegalDocument("test", "test"), 1.0)])
        
        # Should still be retrievable
        results = cache.get(long_query, top_k=5)
        assert results is not None
        assert len(results) == 1
    
    def test_hit_rate_optimization(self):
        """Test that cache hit rates are optimized through LRU behavior."""
        cache = LRUCache(max_size=3, ttl_seconds=60.0)
        
        # Add entries
        cache.put("frequent", "value")
        cache.put("occasional", "value") 
        cache.put("rare", "value")
        
        # Simulate access patterns
        for _ in range(10):
            cache.get("frequent")  # Very frequent access
        
        for _ in range(3):
            cache.get("occasional")  # Occasional access
        
        # Add new entry - should evict "rare" (least recently used)
        cache.put("new", "value")
        
        assert cache.get("frequent") == "value"    # Should remain
        assert cache.get("occasional") == "value"  # Should remain  
        assert cache.get("rare") is None          # Should be evicted
        assert cache.get("new") == "value"        # Should be present
        
        # Verify hit rate is reasonable
        stats = cache.get_stats()
        assert stats["hit_rate"] > 0.5  # Should have decent hit rate


class TestCacheIntegrationWithMetrics:
    """Test cache integration with metrics system."""
    
    @patch('src.lexgraph_legal_rag.metrics.record_cache_hit')
    @patch('src.lexgraph_legal_rag.metrics.record_cache_miss')
    def test_metrics_integration_success(self, mock_record_miss, mock_record_hit):
        """Test that cache integrates with metrics when available."""
        cache = LRUCache(max_size=5, ttl_seconds=60.0)
        
        cache.put("key1", "value1")
        cache.get("key1")  # Should record hit
        cache.get("nonexistent")  # Should record miss
        
        mock_record_hit.assert_called_once()
        mock_record_miss.assert_called_once()
    
    def test_metrics_integration_graceful_failure(self):
        """Test that cache works gracefully when metrics are unavailable."""
        # This should not raise ImportError even if metrics module is missing
        cache = LRUCache(max_size=5, ttl_seconds=60.0)
        
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        assert cache.get("nonexistent") is None
        
        # Should still track internal stats
        stats = cache.get_stats() 
        assert stats["hits"] == 1
        assert stats["misses"] == 1