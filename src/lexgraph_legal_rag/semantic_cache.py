"""Advanced semantic caching system for legal RAG queries.

This module provides intelligent caching that understands semantic similarity
between legal queries, enabling cache hits even for differently worded but
semantically similar questions.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
import logging
import asyncio
import threading
from collections import defaultdict
import math

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Semantic cache entry with metadata."""
    query: str
    query_hash: str
    response: str
    timestamp: float
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    semantic_keywords: Set[str] = field(default_factory=set)
    intent: Optional[str] = None
    confidence: float = 0.0
    response_time: float = 0.0  # Original response time
    
    def update_access(self) -> None:
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = time.time()
    
    def is_expired(self, ttl_seconds: int) -> bool:
        """Check if entry is expired."""
        return time.time() - self.timestamp > ttl_seconds
    
    def get_freshness_score(self) -> float:
        """Get freshness score (0.0 to 1.0, higher is fresher)."""
        age_seconds = time.time() - self.timestamp
        # Exponential decay with half-life of 1 hour
        return math.exp(-age_seconds / 3600)


class SemanticSimilarityMatcher:
    """Simple semantic similarity matching for legal queries."""
    
    def __init__(self):
        # Legal term weights for similarity calculation
        self.legal_term_weights = {
            'contract': 3.0, 'liability': 3.0, 'breach': 3.0, 'damages': 3.0,
            'indemnify': 2.5, 'warranty': 2.5, 'terminate': 2.5, 'jurisdiction': 2.5,
            'precedent': 2.5, 'compliance': 2.5, 'statute': 2.5, 'negligence': 2.5,
            'consideration': 2.0, 'fiduciary': 2.0, 'tort': 2.0, 'remedy': 2.0,
        }
        
        # Common legal stopwords to ignore in similarity
        self.legal_stopwords = {
            'law', 'legal', 'case', 'court', 'judge', 'lawyer', 'attorney',
            'document', 'file', 'record', 'section', 'clause', 'article'
        }
    
    def extract_keywords(self, query: str) -> Set[str]:
        """Extract semantic keywords from query."""
        import re
        
        # Convert to lowercase and extract words
        words = set(re.findall(r'\b\w+\b', query.lower()))
        
        # Remove common stopwords but keep legal terms
        filtered_words = set()
        for word in words:
            if word not in self.legal_stopwords or word in self.legal_term_weights:
                filtered_words.add(word)
        
        return filtered_words
    
    def calculate_similarity(self, keywords1: Set[str], keywords2: Set[str]) -> float:
        """Calculate semantic similarity between two keyword sets."""
        if not keywords1 or not keywords2:
            return 0.0
        
        # Calculate weighted Jaccard similarity
        intersection = keywords1.intersection(keywords2)
        union = keywords1.union(keywords2)
        
        if not intersection:
            return 0.0
        
        # Apply legal term weights
        weighted_intersection = sum(
            self.legal_term_weights.get(word, 1.0) for word in intersection
        )
        weighted_union = sum(
            self.legal_term_weights.get(word, 1.0) for word in union
        )
        
        # Jaccard similarity with weights
        similarity = weighted_intersection / weighted_union
        
        # Boost similarity if legal terms match
        legal_terms_match = intersection.intersection(self.legal_term_weights.keys())
        if legal_terms_match:
            boost = min(0.2, len(legal_terms_match) * 0.1)
            similarity += boost
        
        return min(similarity, 1.0)  # Cap at 1.0


class SemanticQueryCache:
    """Advanced semantic-aware caching system."""
    
    def __init__(self, 
                 max_size: int = 1000,
                 ttl_seconds: int = 3600,
                 similarity_threshold: float = 0.7,
                 enable_analytics: bool = True):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.similarity_threshold = similarity_threshold
        self.enable_analytics = enable_analytics
        
        self._cache: Dict[str, CacheEntry] = {}
        self._keyword_index: Dict[str, Set[str]] = defaultdict(set)  # keyword -> query_hashes
        self._access_lock = threading.RLock()
        self.similarity_matcher = SemanticSimilarityMatcher()
        
        # Analytics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'semantic_hits': 0,  # Cache hits found through semantic similarity
            'evictions': 0,
            'total_queries': 0,
            'avg_similarity_score': 0.0
        }
        
        logger.info(f"Initialized semantic cache: max_size={max_size}, ttl={ttl_seconds}s, threshold={similarity_threshold}")
    
    def _generate_cache_key(self, query: str) -> str:
        """Generate cache key for query."""
        # Normalize query for consistent hashing
        normalized = query.strip().lower()
        return hashlib.sha256(normalized.encode()).hexdigest()
    
    def _update_keyword_index(self, query_hash: str, keywords: Set[str]) -> None:
        """Update the keyword index for semantic lookups."""
        for keyword in keywords:
            self._keyword_index[keyword].add(query_hash)
    
    def _remove_from_keyword_index(self, query_hash: str, keywords: Set[str]) -> None:
        """Remove entry from keyword index."""
        for keyword in keywords:
            self._keyword_index[keyword].discard(query_hash)
            if not self._keyword_index[keyword]:  # Clean up empty sets
                del self._keyword_index[keyword]
    
    def _find_similar_entries(self, keywords: Set[str]) -> List[Tuple[str, float]]:
        """Find semantically similar cache entries."""
        candidate_hashes = set()
        
        # Get all entries that share at least one keyword
        for keyword in keywords:
            candidate_hashes.update(self._keyword_index.get(keyword, set()))
        
        # Calculate similarity scores
        similar_entries = []
        for candidate_hash in candidate_hashes:
            if candidate_hash in self._cache:
                entry = self._cache[candidate_hash]
                similarity = self.similarity_matcher.calculate_similarity(
                    keywords, entry.semantic_keywords
                )
                if similarity >= self.similarity_threshold:
                    similar_entries.append((candidate_hash, similarity))
        
        # Sort by similarity score (highest first)
        return sorted(similar_entries, key=lambda x: x[1], reverse=True)
    
    def _evict_expired(self) -> int:
        """Evict expired entries and return count."""
        expired_keys = [
            key for key, entry in self._cache.items() 
            if entry.is_expired(self.ttl_seconds)
        ]
        
        for key in expired_keys:
            self._evict_entry(key)
        
        return len(expired_keys)
    
    def _evict_lru(self, count: int) -> int:
        """Evict least recently used entries."""
        if count <= 0:
            return 0
        
        # Sort by last accessed time (oldest first)
        entries_by_access = sorted(
            self._cache.items(),
            key=lambda x: x[1].last_accessed
        )
        
        evicted = 0
        for key, _ in entries_by_access[:count]:
            self._evict_entry(key)
            evicted += 1
        
        return evicted
    
    def _evict_entry(self, query_hash: str) -> None:
        """Evict a specific entry."""
        if query_hash in self._cache:
            entry = self._cache[query_hash]
            self._remove_from_keyword_index(query_hash, entry.semantic_keywords)
            del self._cache[query_hash]
            self.stats['evictions'] += 1
    
    async def get(self, query: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Retrieve cached response for query with semantic matching."""
        with self._access_lock:
            self.stats['total_queries'] += 1
            
            # Clean up expired entries periodically
            if self.stats['total_queries'] % 100 == 0:
                expired_count = self._evict_expired()
                if expired_count > 0:
                    logger.debug(f"Evicted {expired_count} expired cache entries")
            
            query_hash = self._generate_cache_key(query)
            keywords = self.similarity_matcher.extract_keywords(query)
            
            # Try exact match first
            if query_hash in self._cache:
                entry = self._cache[query_hash]
                if not entry.is_expired(self.ttl_seconds):
                    entry.update_access()
                    self.stats['hits'] += 1
                    
                    metadata = {
                        'cache_hit_type': 'exact',
                        'similarity_score': 1.0,
                        'access_count': entry.access_count,
                        'freshness_score': entry.get_freshness_score(),
                        'original_response_time': entry.response_time
                    }
                    
                    logger.debug(f"Exact cache hit for query: {query[:50]}...")
                    return entry.response, metadata
                else:
                    # Remove expired entry
                    self._evict_entry(query_hash)
            
            # Try semantic similarity matching
            similar_entries = self._find_similar_entries(keywords)
            
            if similar_entries:
                best_hash, similarity_score = similar_entries[0]
                entry = self._cache[best_hash]
                
                if not entry.is_expired(self.ttl_seconds):
                    entry.update_access()
                    self.stats['hits'] += 1
                    self.stats['semantic_hits'] += 1
                    
                    # Update running average similarity score
                    current_avg = self.stats['avg_similarity_score']
                    total_semantic = self.stats['semantic_hits']
                    self.stats['avg_similarity_score'] = (
                        (current_avg * (total_semantic - 1) + similarity_score) / total_semantic
                    )
                    
                    metadata = {
                        'cache_hit_type': 'semantic',
                        'similarity_score': similarity_score,
                        'access_count': entry.access_count,
                        'freshness_score': entry.get_freshness_score(),
                        'original_response_time': entry.response_time,
                        'original_query': entry.query
                    }
                    
                    logger.info(f"Semantic cache hit (similarity: {similarity_score:.3f}) for query: {query[:50]}...")
                    return entry.response, metadata
                else:
                    # Remove expired entry
                    self._evict_entry(best_hash)
            
            # Cache miss
            self.stats['misses'] += 1
            logger.debug(f"Cache miss for query: {query[:50]}...")
            return None
    
    async def put(self, query: str, response: str, response_time: float = 0.0, 
                  intent: Optional[str] = None, confidence: float = 0.0) -> None:
        """Store response in cache with semantic indexing."""
        with self._access_lock:
            query_hash = self._generate_cache_key(query)
            keywords = self.similarity_matcher.extract_keywords(query)
            
            # Check if we need to evict entries
            if len(self._cache) >= self.max_size:
                # Evict 20% of entries to make room
                evict_count = max(1, self.max_size // 5)
                evicted = self._evict_lru(evict_count)
                logger.debug(f"Evicted {evicted} LRU cache entries")
            
            # Create cache entry
            entry = CacheEntry(
                query=query,
                query_hash=query_hash,
                response=response,
                timestamp=time.time(),
                semantic_keywords=keywords,
                intent=intent,
                confidence=confidence,
                response_time=response_time
            )
            
            # Store entry and update index
            self._cache[query_hash] = entry
            self._update_keyword_index(query_hash, keywords)
            
            logger.debug(f"Cached response for query: {query[:50]}... (keywords: {len(keywords)})")
    
    def invalidate(self, pattern: Optional[str] = None) -> int:
        """Invalidate cache entries matching pattern."""
        with self._access_lock:
            if pattern is None:
                # Clear all
                count = len(self._cache)
                self._cache.clear()
                self._keyword_index.clear()
                return count
            
            # Pattern matching (simple substring match)
            to_remove = []
            for query_hash, entry in self._cache.items():
                if pattern.lower() in entry.query.lower():
                    to_remove.append(query_hash)
            
            for query_hash in to_remove:
                self._evict_entry(query_hash)
            
            logger.info(f"Invalidated {len(to_remove)} cache entries matching pattern: {pattern}")
            return len(to_remove)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._access_lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0.0
            semantic_hit_rate = self.stats['semantic_hits'] / total_requests if total_requests > 0 else 0.0
            
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'hit_rate': hit_rate,
                'semantic_hit_rate': semantic_hit_rate,
                'total_hits': self.stats['hits'],
                'total_misses': self.stats['misses'],
                'semantic_hits': self.stats['semantic_hits'],
                'evictions': self.stats['evictions'],
                'avg_similarity_score': self.stats['avg_similarity_score'],
                'keyword_index_size': len(self._keyword_index),
                'ttl_seconds': self.ttl_seconds,
                'similarity_threshold': self.similarity_threshold
            }
    
    def get_top_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top queries by access count."""
        with self._access_lock:
            sorted_entries = sorted(
                self._cache.values(),
                key=lambda x: x.access_count,
                reverse=True
            )
            
            return [
                {
                    'query': entry.query[:100] + ('...' if len(entry.query) > 100 else ''),
                    'access_count': entry.access_count,
                    'last_accessed': entry.last_accessed,
                    'freshness_score': entry.get_freshness_score(),
                    'keywords_count': len(entry.semantic_keywords)
                }
                for entry in sorted_entries[:limit]
            ]


# Global cache instance
_global_cache = None

def get_semantic_cache() -> SemanticQueryCache:
    """Get global semantic cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = SemanticQueryCache()
    return _global_cache


async def cached_query(query: str, response_func, **kwargs) -> Tuple[str, Dict[str, Any]]:
    """Decorator-style cached query execution."""
    cache = get_semantic_cache()
    
    # Try cache first
    cached_result = await cache.get(query)
    if cached_result is not None:
        return cached_result
    
    # Execute query and cache result
    start_time = time.time()
    try:
        response = await response_func(query, **kwargs)
        response_time = time.time() - start_time
        
        # Cache the result
        await cache.put(query, response, response_time)
        
        metadata = {
            'cache_hit_type': 'miss',
            'response_time': response_time
        }
        
        return response, metadata
        
    except Exception as e:
        logger.error(f"Error executing query function: {e}")
        raise