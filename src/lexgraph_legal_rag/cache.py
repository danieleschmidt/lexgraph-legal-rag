"""Query result caching for improved performance."""

from __future__ import annotations

import time
import hashlib
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from threading import Lock
from collections import OrderedDict

from .models import LegalDocument

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with timestamp and result data."""
    value: Any
    timestamp: float
    hit_count: int = 0
    last_accessed: float = field(default_factory=time.time)


class LRUCache:
    """Thread-safe LRU cache implementation."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: float = 3600.0) -> None:
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = Lock()
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if it exists and is not expired."""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                # Record cache miss metric
                try:
                    from .metrics import record_cache_miss
                    record_cache_miss()
                except ImportError:
                    pass
                return None
            
            entry = self._cache[key]
            
            # Check if entry is expired
            if time.time() - entry.timestamp > self.ttl_seconds:
                del self._cache[key]
                self._misses += 1
                # Record cache miss metric
                try:
                    from .metrics import record_cache_miss
                    record_cache_miss()
                except ImportError:
                    pass
                logger.debug(f"Cache entry expired: {key[:20]}...")
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.hit_count += 1
            entry.last_accessed = time.time()
            self._hits += 1
            
            # Record cache hit metric
            try:
                from .metrics import record_cache_hit
                record_cache_hit()
            except ImportError:
                pass
            
            logger.debug(f"Cache hit: {key[:20]}...")
            return entry.value
    
    def put(self, key: str, value: Any) -> None:
        """Put value in cache, evicting LRU items if necessary."""
        with self._lock:
            current_time = time.time()
            
            if key in self._cache:
                # Update existing entry
                self._cache[key].value = value
                self._cache[key].timestamp = current_time
                self._cache[key].last_accessed = current_time
                self._cache.move_to_end(key)
            else:
                # Add new entry
                entry = CacheEntry(value=value, timestamp=current_time)
                self._cache[key] = entry
                
                # Evict oldest items if cache is full
                while len(self._cache) > self.max_size:
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]
                    logger.debug(f"Evicted cache entry: {oldest_key[:20]}...")
    
    def invalidate(self, key: str) -> bool:
        """Remove specific key from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                logger.debug(f"Invalidated cache entry: {key[:20]}...")
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Union[int, float]]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
            
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "ttl_seconds": self.ttl_seconds,
            }
    
    def cleanup_expired(self) -> int:
        """Remove expired entries and return count of removed items."""
        current_time = time.time()
        expired_keys = []
        
        with self._lock:
            for key, entry in self._cache.items():
                if current_time - entry.timestamp > self.ttl_seconds:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self._cache[key]
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
        
        return len(expired_keys)


class QueryResultCache:
    """Specialized cache for search query results."""
    
    def __init__(
        self,
        max_size: int = 500,
        ttl_seconds: float = 1800.0,  # 30 minutes
        enable_semantic_cache: bool = True,
    ) -> None:
        self.cache = LRUCache(max_size=max_size, ttl_seconds=ttl_seconds)
        self.enable_semantic_cache = enable_semantic_cache
        self._query_variations: Dict[str, str] = {}  # Map similar queries to canonical form
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for consistent caching."""
        # Convert to lowercase and strip whitespace
        normalized = query.lower().strip()
        
        # Remove extra whitespace
        import re
        normalized = re.sub(r'\s+', ' ', normalized)
        
        return normalized
    
    def _create_cache_key(
        self,
        query: str,
        top_k: int,
        semantic: bool = False,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a deterministic cache key for the query parameters."""
        normalized_query = self._normalize_query(query)
        
        # Create key components
        key_parts = [
            f"q:{normalized_query}",
            f"k:{top_k}",
            f"s:{semantic}",
        ]
        
        if extra_params:
            # Sort params for consistent key generation
            sorted_params = sorted(extra_params.items())
            key_parts.extend([f"{k}:{v}" for k, v in sorted_params])
        
        key_string = "|".join(key_parts)
        
        # Hash to create shorter, consistent keys
        return hashlib.sha256(key_string.encode()).hexdigest()[:16]
    
    def get(
        self,
        query: str,
        top_k: int,
        semantic: bool = False,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Optional[List[Tuple[LegalDocument, float]]]:
        """Get cached results for a query."""
        cache_key = self._create_cache_key(query, top_k, semantic, extra_params)
        return self.cache.get(cache_key)
    
    def put(
        self,
        query: str,
        top_k: int,
        results: List[Tuple[LegalDocument, float]],
        semantic: bool = False,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Cache results for a query."""
        cache_key = self._create_cache_key(query, top_k, semantic, extra_params)
        self.cache.put(cache_key, results)
        
        logger.debug(f"Cached results for query: {query[:50]}... (key: {cache_key})")
    
    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching a pattern (e.g., when index is updated)."""
        # For now, we'll clear the entire cache when invalidation is requested
        # In a more sophisticated implementation, we could track which documents
        # are referenced in which cache entries
        self.cache.clear()
        logger.info(f"Cache invalidated due to pattern: {pattern}")
        return 1  # Return count of invalidated entries
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics including query-specific metrics."""
        base_stats = self.cache.get_stats()
        base_stats.update({
            "semantic_cache_enabled": self.enable_semantic_cache,
            "query_variations": len(self._query_variations),
        })
        return base_stats


# Global cache instance
_query_cache: Optional[QueryResultCache] = None
_cache_lock = Lock()


def get_query_cache() -> QueryResultCache:
    """Get the global query result cache instance."""
    global _query_cache
    with _cache_lock:
        if _query_cache is None:
            _query_cache = QueryResultCache()
        return _query_cache


def configure_cache(
    max_size: int = 500,
    ttl_seconds: float = 1800.0,
    enable_semantic_cache: bool = True,
) -> None:
    """Configure the global query cache."""
    global _query_cache
    with _cache_lock:
        _query_cache = QueryResultCache(
            max_size=max_size,
            ttl_seconds=ttl_seconds,
            enable_semantic_cache=enable_semantic_cache,
        )
        logger.info(f"Query cache configured: max_size={max_size}, ttl={ttl_seconds}s")