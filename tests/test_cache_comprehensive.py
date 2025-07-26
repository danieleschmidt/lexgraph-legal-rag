"""Comprehensive test coverage for the cache module."""

import time
import pytest
import threading
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Tuple, Optional

from lexgraph_legal_rag.cache import (
    CacheEntry,
    LRUCache,
    QueryResultCache,
    get_query_cache,
    configure_cache,
    _query_cache,
    _cache_lock
)
from lexgraph_legal_rag.models import LegalDocument


class TestCacheEntry:
    """Test cases for CacheEntry dataclass."""
    
    def test_cache_entry_creation(self):
        """Test cache entry creation with default values."""
        entry = CacheEntry(value="test_value", timestamp=123.45)
        
        assert entry.value == "test_value"
        assert entry.timestamp == 123.45
        assert entry.hit_count == 0
        assert isinstance(entry.last_accessed, float)
        assert entry.last_accessed > 0
    
    def test_cache_entry_with_custom_values(self):
        """Test cache entry creation with custom values."""
        custom_time = time.time()
        entry = CacheEntry(
            value={"key": "value"}, 
            timestamp=123.45,
            hit_count=5,
            last_accessed=custom_time
        )
        
        assert entry.value == {"key": "value"}
        assert entry.timestamp == 123.45
        assert entry.hit_count == 5
        assert entry.last_accessed == custom_time


class TestLRUCache:
    """Test cases for LRUCache implementation."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.cache = LRUCache(max_size=3, ttl_seconds=60.0)
    
    def test_cache_initialization(self):
        """Test cache initialization with default and custom values."""
        # Default values
        default_cache = LRUCache()
        assert default_cache.max_size == 1000
        assert default_cache.ttl_seconds == 3600.0
        assert len(default_cache._cache) == 0
        assert default_cache._hits == 0
        assert default_cache._misses == 0
        
        # Custom values
        custom_cache = LRUCache(max_size=500, ttl_seconds=1800.0)
        assert custom_cache.max_size == 500
        assert custom_cache.ttl_seconds == 1800.0
    
    def test_put_and_get_basic(self):
        """Test basic put and get operations."""
        self.cache.put("key1", "value1")
        result = self.cache.get("key1")
        
        assert result == "value1"
        assert self.cache._hits == 1
        assert self.cache._misses == 0
    
    def test_get_nonexistent_key(self):
        """Test getting a key that doesn't exist."""
        result = self.cache.get("nonexistent")
        
        assert result is None
        assert self.cache._hits == 0
        assert self.cache._misses == 1
    
    def test_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        # Fill cache to capacity
        self.cache.put("key1", "value1")
        self.cache.put("key2", "value2")
        self.cache.put("key3", "value3")
        
        # Add one more item, should evict oldest (key1)
        self.cache.put("key4", "value4")
        
        assert self.cache.get("key1") is None  # Evicted
        assert self.cache.get("key2") == "value2"
        assert self.cache.get("key3") == "value3"
        assert self.cache.get("key4") == "value4"
    
    def test_lru_ordering_on_access(self):
        """Test that LRU ordering is maintained on access."""
        self.cache.put("key1", "value1")
        self.cache.put("key2", "value2")
        self.cache.put("key3", "value3")
        
        # Access key1 to make it most recently used
        self.cache.get("key1")
        
        # Add another item, should evict key2 (oldest unaccessed)
        self.cache.put("key4", "value4")
        
        assert self.cache.get("key1") == "value1"  # Still there
        assert self.cache.get("key2") is None  # Evicted
        assert self.cache.get("key3") == "value3"
        assert self.cache.get("key4") == "value4"
    
    def test_update_existing_key(self):
        """Test updating an existing key."""
        self.cache.put("key1", "value1")
        self.cache.put("key1", "updated_value")
        
        result = self.cache.get("key1")
        assert result == "updated_value"
        assert len(self.cache._cache) == 1
    
    def test_ttl_expiration(self):
        """Test TTL-based expiration."""
        short_ttl_cache = LRUCache(max_size=10, ttl_seconds=0.1)
        short_ttl_cache.put("key1", "value1")
        
        # Should be accessible immediately
        assert short_ttl_cache.get("key1") == "value1"
        
        # Wait for expiration
        time.sleep(0.2)
        
        # Should be expired now
        assert short_ttl_cache.get("key1") is None
        assert short_ttl_cache._misses == 1
    
    def test_cache_invalidation(self):
        """Test cache invalidation."""
        self.cache.put("key1", "value1")
        self.cache.put("key2", "value2")
        
        # Invalidate existing key
        result = self.cache.invalidate("key1")
        assert result is True
        assert self.cache.get("key1") is None
        assert self.cache.get("key2") == "value2"
        
        # Invalidate non-existent key
        result = self.cache.invalidate("nonexistent")
        assert result is False
    
    def test_cache_clear(self):
        """Test cache clearing."""
        self.cache.put("key1", "value1")
        self.cache.put("key2", "value2")
        self.cache.get("key1")  # Generate some hits
        
        self.cache.clear()
        
        assert len(self.cache._cache) == 0
        assert self.cache._hits == 0
        assert self.cache._misses == 0
        assert self.cache.get("key1") is None
    
    def test_get_stats(self):
        """Test cache statistics."""
        self.cache.put("key1", "value1")
        self.cache.put("key2", "value2")
        self.cache.get("key1")  # Hit
        self.cache.get("key3")  # Miss
        
        stats = self.cache.get_stats()
        
        assert stats["size"] == 2
        assert stats["max_size"] == 3
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
        assert stats["ttl_seconds"] == 60.0
    
    def test_get_stats_empty_cache(self):
        """Test statistics with empty cache."""
        stats = self.cache.get_stats()
        
        assert stats["size"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0.0
    
    def test_cleanup_expired(self):
        """Test cleanup of expired entries."""
        # Use very short TTL for testing
        short_ttl_cache = LRUCache(max_size=10, ttl_seconds=0.1)
        
        short_ttl_cache.put("key1", "value1")
        short_ttl_cache.put("key2", "value2")
        short_ttl_cache.put("key3", "value3")
        
        # Wait for some entries to expire
        time.sleep(0.15)
        
        # Add a fresh entry
        short_ttl_cache.put("key4", "value4")
        
        # Clean up expired entries
        expired_count = short_ttl_cache.cleanup_expired()
        
        assert expired_count == 3  # key1, key2, key3 should be expired
        assert short_ttl_cache.get("key4") == "value4"  # Fresh entry should remain
    
    def test_hit_count_tracking(self):
        """Test hit count tracking per entry."""
        self.cache.put("key1", "value1")
        
        # Access multiple times
        self.cache.get("key1")
        self.cache.get("key1")
        self.cache.get("key1")
        
        # Check internal entry
        entry = self.cache._cache["key1"]
        assert entry.hit_count == 3
        assert isinstance(entry.last_accessed, float)
    
    def test_thread_safety(self):
        """Test thread safety of cache operations."""
        results = []
        errors = []
        
        def cache_worker(worker_id: int):
            try:
                for i in range(10):
                    key = f"worker_{worker_id}_key_{i}"
                    value = f"worker_{worker_id}_value_{i}"
                    self.cache.put(key, value)
                    retrieved = self.cache.get(key)
                    results.append((key, value, retrieved))
            except Exception as e:
                errors.append(e)
        
        # Run multiple threads concurrently
        threads = []
        for i in range(5):
            thread = threading.Thread(target=cache_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        assert len(errors) == 0
        assert len(results) == 50  # 5 workers * 10 operations
        
        # Verify all stored values are correct
        for key, expected_value, retrieved_value in results:
            assert retrieved_value == expected_value
    
    def test_metrics_integration(self):
        """Test integration with metrics recording."""
        with patch('lexgraph_legal_rag.cache.record_cache_hit') as mock_hit:
            with patch('lexgraph_legal_rag.cache.record_cache_miss') as mock_miss:
                self.cache.put("key1", "value1")
                
                # Hit should record metric
                self.cache.get("key1")
                mock_hit.assert_called_once()
                
                # Miss should record metric
                self.cache.get("nonexistent")
                mock_miss.assert_called_once()
    
    def test_metrics_import_error_handling(self):
        """Test graceful handling when metrics module is not available."""
        with patch('lexgraph_legal_rag.cache.record_cache_hit', side_effect=ImportError):
            with patch('lexgraph_legal_rag.cache.record_cache_miss', side_effect=ImportError):
                self.cache.put("key1", "value1")
                
                # Should not raise error despite ImportError
                result = self.cache.get("key1")
                assert result == "value1"
                
                # Should not raise error for miss either
                result = self.cache.get("nonexistent")
                assert result is None


class TestQueryResultCache:
    """Test cases for QueryResultCache."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.query_cache = QueryResultCache(max_size=10, ttl_seconds=60.0)
        
        # Create test documents
        self.doc1 = LegalDocument(
            id="doc1",
            title="Test Document 1",
            content="Test content 1",
            document_type="case",
            jurisdiction="federal",
            metadata={}
        )
        self.doc2 = LegalDocument(
            id="doc2", 
            title="Test Document 2",
            content="Test content 2",
            document_type="statute",
            jurisdiction="state",
            metadata={}
        )
        
        self.test_results = [
            (self.doc1, 0.95),
            (self.doc2, 0.87)
        ]
    
    def test_query_cache_initialization(self):
        """Test query cache initialization."""
        # Default initialization
        default_cache = QueryResultCache()
        assert default_cache.cache.max_size == 500
        assert default_cache.cache.ttl_seconds == 1800.0
        assert default_cache.enable_semantic_cache is True
        
        # Custom initialization
        custom_cache = QueryResultCache(
            max_size=100,
            ttl_seconds=600.0,
            enable_semantic_cache=False
        )
        assert custom_cache.cache.max_size == 100
        assert custom_cache.cache.ttl_seconds == 600.0
        assert custom_cache.enable_semantic_cache is False
    
    def test_query_normalization(self):
        """Test query normalization."""
        test_cases = [
            ("  Hello World  ", "hello world"),
            ("UPPER CASE", "upper case"),
            ("Multiple   Spaces", "multiple spaces"),
            ("Mixed   Case   Query", "mixed case query"),
        ]
        
        for input_query, expected in test_cases:
            result = self.query_cache._normalize_query(input_query)
            assert result == expected
    
    def test_cache_key_generation(self):
        """Test cache key generation."""
        key1 = self.query_cache._create_cache_key("test query", 10, True)
        key2 = self.query_cache._create_cache_key("test query", 10, True)
        key3 = self.query_cache._create_cache_key("different query", 10, True)
        key4 = self.query_cache._create_cache_key("test query", 5, True)
        
        # Same parameters should generate same key
        assert key1 == key2
        
        # Different parameters should generate different keys
        assert key1 != key3
        assert key1 != key4
        
        # Keys should be reasonably short
        assert len(key1) == 16
    
    def test_cache_key_with_extra_params(self):
        """Test cache key generation with extra parameters."""
        extra_params = {"filter": "cases", "jurisdiction": "federal"}
        
        key1 = self.query_cache._create_cache_key(
            "test query", 10, True, extra_params
        )
        key2 = self.query_cache._create_cache_key(
            "test query", 10, True, extra_params
        )
        key3 = self.query_cache._create_cache_key(
            "test query", 10, True, {"filter": "statutes"}
        )
        
        assert key1 == key2  # Same params
        assert key1 != key3  # Different extra params
    
    def test_put_and_get_query_results(self):
        """Test putting and getting query results."""
        query = "test legal query"
        top_k = 5
        
        # Cache results
        self.query_cache.put(query, top_k, self.test_results)
        
        # Retrieve results
        cached_results = self.query_cache.get(query, top_k)
        
        assert cached_results == self.test_results
        assert len(cached_results) == 2
        assert cached_results[0][0].id == "doc1"
        assert cached_results[1][0].id == "doc2"
    
    def test_query_cache_miss(self):
        """Test cache miss for non-existent query."""
        result = self.query_cache.get("non-existent query", 5)
        assert result is None
    
    def test_semantic_parameter_distinction(self):
        """Test that semantic parameter creates different cache keys."""
        query = "legal precedent"
        top_k = 10
        
        # Cache with semantic=False
        self.query_cache.put(query, top_k, self.test_results, semantic=False)
        
        # Should miss with semantic=True
        result_semantic = self.query_cache.get(query, top_k, semantic=True)
        assert result_semantic is None
        
        # Should hit with semantic=False
        result_non_semantic = self.query_cache.get(query, top_k, semantic=False)
        assert result_non_semantic == self.test_results
    
    def test_invalidate_pattern(self):
        """Test pattern-based cache invalidation."""
        # Cache multiple queries
        self.query_cache.put("query1", 5, self.test_results)
        self.query_cache.put("query2", 5, self.test_results)
        
        # Verify both are cached
        assert self.query_cache.get("query1", 5) is not None
        assert self.query_cache.get("query2", 5) is not None
        
        # Invalidate by pattern (currently clears all)
        count = self.query_cache.invalidate_pattern("*")
        
        assert count == 1  # Returns 1 as per current implementation
        assert self.query_cache.get("query1", 5) is None
        assert self.query_cache.get("query2", 5) is None
    
    def test_get_stats_query_cache(self):
        """Test query cache statistics."""
        self.query_cache.put("query1", 5, self.test_results)
        self.query_cache.get("query1", 5)  # Hit
        self.query_cache.get("query2", 5)  # Miss
        
        stats = self.query_cache.get_stats()
        
        assert "size" in stats
        assert "hits" in stats
        assert "misses" in stats
        assert "semantic_cache_enabled" in stats
        assert "query_variations" in stats
        assert stats["semantic_cache_enabled"] is True
        assert stats["query_variations"] == 0
    
    def test_cache_key_collision_resistance(self):
        """Test that different queries don't create colliding cache keys."""
        keys = set()
        
        test_queries = [
            ("query about contracts", 10, False),
            ("contract query about", 10, False),
            ("different legal topic", 5, True),
            ("legal different topic", 5, True),
            ("same query", 10, False),
            ("same query", 10, True),  # Different semantic flag
        ]
        
        for query, top_k, semantic in test_queries:
            key = self.query_cache._create_cache_key(query, top_k, semantic)
            keys.add(key)
        
        # All keys should be unique
        assert len(keys) == len(test_queries)


class TestGlobalCacheFunctions:
    """Test cases for global cache management functions."""
    
    def setup_method(self):
        """Reset global cache state."""
        global _query_cache
        with _cache_lock:
            _query_cache = None
    
    def test_get_query_cache_singleton(self):
        """Test that get_query_cache returns singleton instance."""
        cache1 = get_query_cache()
        cache2 = get_query_cache()
        
        assert cache1 is cache2
        assert isinstance(cache1, QueryResultCache)
    
    def test_configure_cache(self):
        """Test cache configuration."""
        configure_cache(max_size=200, ttl_seconds=900.0, enable_semantic_cache=False)
        
        cache = get_query_cache()
        assert cache.cache.max_size == 200
        assert cache.cache.ttl_seconds == 900.0
        assert cache.enable_semantic_cache is False
    
    def test_reconfigure_cache(self):
        """Test reconfiguring existing cache."""
        # First configuration
        configure_cache(max_size=100, ttl_seconds=600.0)
        cache1 = get_query_cache()
        
        # Reconfigure
        configure_cache(max_size=300, ttl_seconds=1200.0)
        cache2 = get_query_cache()
        
        # Should be different instance with new config
        assert cache2 is not cache1
        assert cache2.cache.max_size == 300
        assert cache2.cache.ttl_seconds == 1200.0
    
    def test_global_cache_thread_safety(self):
        """Test thread safety of global cache access."""
        cache_instances = []
        errors = []
        
        def get_cache_worker():
            try:
                cache = get_query_cache()
                cache_instances.append(cache)
            except Exception as e:
                errors.append(e)
        
        # Run multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=get_cache_worker)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        assert len(errors) == 0
        assert len(cache_instances) == 10
        
        # All should be the same instance
        first_cache = cache_instances[0]
        for cache in cache_instances:
            assert cache is first_cache


class TestCacheIntegration:
    """Integration tests for cache components working together."""
    
    def test_query_cache_with_lru_behavior(self):
        """Test that QueryResultCache exhibits LRU behavior."""
        # Create small cache for testing eviction
        small_cache = QueryResultCache(max_size=2, ttl_seconds=60.0)
        
        doc1 = LegalDocument(id="1", title="Doc 1", content="Content 1",
                           document_type="case", jurisdiction="federal", metadata={})
        doc2 = LegalDocument(id="2", title="Doc 2", content="Content 2", 
                           document_type="case", jurisdiction="federal", metadata={})
        doc3 = LegalDocument(id="3", title="Doc 3", content="Content 3",
                           document_type="case", jurisdiction="federal", metadata={})
        
        results1 = [(doc1, 0.9)]
        results2 = [(doc2, 0.8)]
        results3 = [(doc3, 0.7)]
        
        # Fill cache
        small_cache.put("query1", 5, results1)
        small_cache.put("query2", 5, results2)
        
        # Both should be cached
        assert small_cache.get("query1", 5) == results1
        assert small_cache.get("query2", 5) == results2
        
        # Add third item, should evict first
        small_cache.put("query3", 5, results3)
        
        assert small_cache.get("query1", 5) is None  # Evicted
        assert small_cache.get("query2", 5) == results2
        assert small_cache.get("query3", 5) == results3
    
    def test_cache_with_complex_documents(self):
        """Test cache with complex document structures."""
        complex_doc = LegalDocument(
            id="complex_1",
            title="Complex Legal Document",
            content="Very long content with special characters: àáâãäåæçèéêë",
            document_type="regulation",
            jurisdiction="international",
            metadata={
                "court": "Supreme Court",
                "date": "2023-01-15",
                "citations": ["123 F.3d 456", "789 U.S. 101"],
                "tags": ["constitutional", "civil rights"],
                "nested": {
                    "level1": {
                        "level2": "deep value"
                    }
                }
            }
        )
        
        results = [(complex_doc, 0.95)]
        
        self.query_cache = QueryResultCache()
        self.query_cache.put("complex query", 10, results)
        
        cached_results = self.query_cache.get("complex query", 10)
        assert cached_results == results
        
        # Verify document structure is preserved
        cached_doc = cached_results[0][0]
        assert cached_doc.id == "complex_1"
        assert cached_doc.metadata["court"] == "Supreme Court"
        assert cached_doc.metadata["nested"]["level1"]["level2"] == "deep value"
    
    def test_error_handling_and_logging(self):
        """Test error handling and logging functionality."""
        with patch('lexgraph_legal_rag.cache.logger') as mock_logger:
            cache = LRUCache(max_size=2, ttl_seconds=60.0)
            
            # Test eviction logging
            cache.put("key1", "value1")
            cache.put("key2", "value2")
            cache.put("key3", "value3")  # Should evict key1
            
            # Check that eviction was logged
            mock_logger.debug.assert_called()
            
            # Test invalidation logging
            cache.invalidate("key2")
            mock_logger.debug.assert_called()
            
            # Test clear logging
            cache.clear()
            mock_logger.info.assert_called_with("Cache cleared")


class TestCachePerformance:
    """Performance-related tests for cache functionality."""
    
    def test_large_cache_operations(self):
        """Test cache performance with large number of items."""
        large_cache = LRUCache(max_size=1000, ttl_seconds=3600.0)
        
        # Insert many items
        start_time = time.time()
        for i in range(500):
            large_cache.put(f"key_{i}", f"value_{i}")
        insert_time = time.time() - start_time
        
        # Retrieve many items
        start_time = time.time()
        for i in range(500):
            result = large_cache.get(f"key_{i}")
            assert result == f"value_{i}"
        retrieve_time = time.time() - start_time
        
        # Operations should complete in reasonable time
        assert insert_time < 1.0  # Should complete in under 1 second
        assert retrieve_time < 1.0
        
        # Verify cache statistics
        stats = large_cache.get_stats()
        assert stats["size"] == 500
        assert stats["hits"] == 500
        assert stats["misses"] == 0
    
    def test_cleanup_performance(self):
        """Test performance of cleanup operations."""
        cache = LRUCache(max_size=100, ttl_seconds=0.1)  # Very short TTL
        
        # Add many items
        for i in range(50):
            cache.put(f"key_{i}", f"value_{i}")
        
        # Wait for expiration
        time.sleep(0.2)
        
        # Cleanup should be efficient
        start_time = time.time()
        expired_count = cache.cleanup_expired()
        cleanup_time = time.time() - start_time
        
        assert expired_count == 50
        assert cleanup_time < 0.1  # Should be very fast
        assert len(cache._cache) == 0


class TestCacheEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_zero_max_size_cache(self):
        """Test cache with zero max size."""
        zero_cache = LRUCache(max_size=0, ttl_seconds=60.0)
        
        # Should not store anything
        zero_cache.put("key1", "value1")
        result = zero_cache.get("key1")
        
        assert result is None
        assert len(zero_cache._cache) == 0
    
    def test_negative_ttl(self):
        """Test cache with negative TTL."""
        negative_ttl_cache = LRUCache(max_size=10, ttl_seconds=-1.0)
        
        negative_ttl_cache.put("key1", "value1")
        
        # Should expire immediately due to negative TTL
        result = negative_ttl_cache.get("key1")
        assert result is None
    
    def test_empty_string_keys(self):
        """Test cache with empty string keys."""
        cache = LRUCache()
        
        cache.put("", "empty_key_value")
        result = cache.get("")
        
        assert result == "empty_key_value"
    
    def test_special_character_keys(self):
        """Test cache with special character keys."""
        cache = LRUCache()
        special_keys = [
            "key with spaces",
            "key|with|pipes",
            "key:with:colons",
            "key/with/slashes",
            "key\nwith\nnewlines",
            "key\twith\ttabs",
            "🚀emoji🔥key🎉"
        ]
        
        for key in special_keys:
            cache.put(key, f"value_for_{key}")
        
        for key in special_keys:
            result = cache.get(key)
            assert result == f"value_for_{key}"
    
    def test_none_values(self):
        """Test cache with None values."""
        cache = LRUCache()
        
        cache.put("none_key", None)
        result = cache.get("none_key")
        
        assert result is None
        # Note: This is ambiguous with cache miss, but it's the expected behavior
    
    def test_concurrent_modification_during_cleanup(self):
        """Test cleanup behavior during concurrent modifications."""
        cache = LRUCache(max_size=10, ttl_seconds=0.1)
        
        def put_worker():
            for i in range(20):
                cache.put(f"concurrent_{i}", f"value_{i}")
                time.sleep(0.01)
        
        def cleanup_worker():
            time.sleep(0.05)  # Let some items be added first
            for _ in range(5):
                cache.cleanup_expired()
                time.sleep(0.02)
        
        # Run concurrent operations
        put_thread = threading.Thread(target=put_worker)
        cleanup_thread = threading.Thread(target=cleanup_worker)
        
        put_thread.start()
        cleanup_thread.start()
        
        put_thread.join()
        cleanup_thread.join()
        
        # Should not crash and maintain data integrity
        stats = cache.get_stats()
        assert isinstance(stats["size"], int)
        assert stats["size"] >= 0