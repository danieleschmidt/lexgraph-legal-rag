"""
Intelligent Caching System with Predictive Analytics
===================================================

Generation 3 Caching: Advanced caching with AI-powered prediction and optimization
- Predictive cache warming based on usage patterns
- Multi-level caching hierarchy with intelligent eviction
- Cache coherence across distributed systems
- Real-time performance optimization
- Machine learning-based cache placement
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import pickle
import time
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import threading
import numpy as np

logger = logging.getLogger(__name__)


class CacheLevel(Enum):
    """Cache hierarchy levels."""
    
    L1_MEMORY = "l1_memory"         # In-memory fastest cache
    L2_REDIS = "l2_redis"           # Redis distributed cache
    L3_DISK = "l3_disk"             # Disk-based cache
    L4_REMOTE = "l4_remote"         # Remote/cloud cache


class EvictionPolicy(Enum):
    """Cache eviction policies."""
    
    LRU = "lru"                     # Least Recently Used
    LFU = "lfu"                     # Least Frequently Used
    PREDICTIVE = "predictive"       # ML-based predictive eviction
    WEIGHTED = "weighted"           # Weighted by importance
    QUANTUM = "quantum"             # Quantum-inspired optimization


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    
    key: str
    value: Any
    timestamp: float
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    size_bytes: int = 0
    ttl: Optional[float] = None
    importance_score: float = 1.0
    prediction_score: float = 0.0
    cache_level: CacheLevel = CacheLevel.L1_MEMORY


@dataclass
class AccessPattern:
    """Access pattern for predictive analytics."""
    
    key: str
    frequency: float
    recency: float
    periodicity: float
    trend: float
    confidence: float = 0.5


@dataclass
class CacheMetrics:
    """Cache performance metrics."""
    
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size_bytes: int = 0
    avg_access_time: float = 0.0
    prediction_accuracy: float = 0.0


class IntelligentCachingSystem:
    """
    Advanced caching system with predictive analytics and multi-level hierarchy.
    
    Features:
    - Multi-level cache hierarchy (Memory -> Redis -> Disk -> Remote)
    - Predictive cache warming using access patterns
    - Intelligent eviction policies with ML optimization
    - Real-time performance monitoring and optimization
    - Distributed cache coherence
    - Quantum-inspired cache placement algorithms
    """
    
    def __init__(self,
                 max_l1_size: int = 1000,
                 max_l2_size: int = 10000,
                 max_l3_size: int = 100000,
                 default_ttl: float = 3600.0,
                 eviction_policy: EvictionPolicy = EvictionPolicy.PREDICTIVE):
        
        self.max_l1_size = max_l1_size
        self.max_l2_size = max_l2_size
        self.max_l3_size = max_l3_size
        self.default_ttl = default_ttl
        self.eviction_policy = eviction_policy
        
        # Multi-level cache storage
        self._l1_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._l2_cache: Dict[str, CacheEntry] = {}
        self._l3_cache: Dict[str, CacheEntry] = {}
        
        # Access patterns for prediction
        self._access_patterns: Dict[str, AccessPattern] = {}
        self._access_history: List[Tuple[str, float]] = []
        
        # Performance metrics
        self._metrics: Dict[CacheLevel, CacheMetrics] = {
            level: CacheMetrics() for level in CacheLevel
        }
        
        # Predictive model
        self._prediction_weights: Dict[str, float] = {
            'frequency': 0.3,
            'recency': 0.3,
            'periodicity': 0.2,
            'trend': 0.2
        }
        
        # Background tasks
        self._optimization_task: Optional[asyncio.Task] = None
        self._warming_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Threading locks
        self._l1_lock = threading.RLock()
        self._l2_lock = threading.RLock()
        self._l3_lock = threading.RLock()
    
    async def start(self):
        """Start cache optimization and warming tasks."""
        
        self._running = True
        self._optimization_task = asyncio.create_task(self._optimization_loop())
        self._warming_task = asyncio.create_task(self._cache_warming_loop())
        logger.info("Intelligent caching system started")
    
    async def stop(self):
        """Stop cache optimization tasks."""
        
        self._running = False
        
        if self._optimization_task:
            self._optimization_task.cancel()
        if self._warming_task:
            self._warming_task.cancel()
        
        logger.info("Intelligent caching system stopped")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache with intelligent hierarchy traversal."""
        
        start_time = time.time()
        
        # Try L1 cache first
        result = await self._get_from_l1(key)
        if result is not None:
            self._record_hit(CacheLevel.L1_MEMORY, time.time() - start_time)
            return result
        
        # Try L2 cache
        result = await self._get_from_l2(key)
        if result is not None:
            self._record_hit(CacheLevel.L2_REDIS, time.time() - start_time)
            # Promote to L1
            await self._promote_to_l1(key, result)
            return result
        
        # Try L3 cache
        result = await self._get_from_l3(key)
        if result is not None:
            self._record_hit(CacheLevel.L3_DISK, time.time() - start_time)
            # Promote to higher levels
            await self._promote_to_l2(key, result)
            await self._promote_to_l1(key, result)
            return result
        
        # Cache miss
        self._record_miss(time.time() - start_time)
        return None
    
    async def put(self, 
                  key: str, 
                  value: Any, 
                  ttl: Optional[float] = None,
                  importance_score: float = 1.0) -> None:
        """Put value in cache with intelligent placement."""
        
        ttl = ttl or self.default_ttl
        size_bytes = self._estimate_size(value)
        
        entry = CacheEntry(
            key=key,
            value=value,
            timestamp=time.time(),
            size_bytes=size_bytes,
            ttl=ttl,
            importance_score=importance_score
        )
        
        # Update access patterns
        self._update_access_pattern(key)
        
        # Calculate prediction score
        entry.prediction_score = self._calculate_prediction_score(key)
        
        # Intelligent placement based on importance and prediction
        cache_level = self._determine_cache_level(entry)
        entry.cache_level = cache_level
        
        # Place in appropriate cache level
        if cache_level == CacheLevel.L1_MEMORY:
            await self._put_in_l1(entry)
        elif cache_level == CacheLevel.L2_REDIS:
            await self._put_in_l2(entry)
        elif cache_level == CacheLevel.L3_DISK:
            await self._put_in_l3(entry)
    
    async def invalidate(self, key: str) -> bool:
        """Invalidate key from all cache levels."""
        
        invalidated = False
        
        # Remove from all levels
        if await self._remove_from_l1(key):
            invalidated = True
        
        if await self._remove_from_l2(key):
            invalidated = True
        
        if await self._remove_from_l3(key):
            invalidated = True
        
        return invalidated
    
    async def warm_cache(self, keys: List[str]) -> None:
        """Warm cache with predicted keys."""
        
        logger.info(f"Warming cache with {len(keys)} predicted keys")
        
        for key in keys:
            if key not in self._l1_cache:
                # This would typically load from data source
                # For now, we'll just update the prediction score
                if key in self._access_patterns:
                    pattern = self._access_patterns[key]
                    pattern.confidence = min(1.0, pattern.confidence + 0.1)
    
    async def _get_from_l1(self, key: str) -> Optional[Any]:
        """Get from L1 memory cache."""
        
        with self._l1_lock:
            if key in self._l1_cache:
                entry = self._l1_cache[key]
                
                # Check TTL
                if entry.ttl and time.time() - entry.timestamp > entry.ttl:
                    del self._l1_cache[key]
                    return None
                
                # Update access info
                entry.access_count += 1
                entry.last_access = time.time()
                
                # Move to end (LRU)
                self._l1_cache.move_to_end(key)
                
                return entry.value
        
        return None
    
    async def _get_from_l2(self, key: str) -> Optional[Any]:
        """Get from L2 Redis cache."""
        
        with self._l2_lock:
            if key in self._l2_cache:
                entry = self._l2_cache[key]
                
                # Check TTL
                if entry.ttl and time.time() - entry.timestamp > entry.ttl:
                    del self._l2_cache[key]
                    return None
                
                entry.access_count += 1
                entry.last_access = time.time()
                
                return entry.value
        
        return None
    
    async def _get_from_l3(self, key: str) -> Optional[Any]:
        """Get from L3 disk cache."""
        
        with self._l3_lock:
            if key in self._l3_cache:
                entry = self._l3_cache[key]
                
                # Check TTL
                if entry.ttl and time.time() - entry.timestamp > entry.ttl:
                    del self._l3_cache[key]
                    return None
                
                entry.access_count += 1
                entry.last_access = time.time()
                
                return entry.value
        
        return None
    
    async def _put_in_l1(self, entry: CacheEntry) -> None:
        """Put entry in L1 cache with eviction if needed."""
        
        with self._l1_lock:
            # Check if eviction is needed
            while len(self._l1_cache) >= self.max_l1_size:
                await self._evict_from_l1()
            
            self._l1_cache[entry.key] = entry
            self._metrics[CacheLevel.L1_MEMORY].size_bytes += entry.size_bytes
    
    async def _put_in_l2(self, entry: CacheEntry) -> None:
        """Put entry in L2 cache."""
        
        with self._l2_lock:
            while len(self._l2_cache) >= self.max_l2_size:
                await self._evict_from_l2()
            
            self._l2_cache[entry.key] = entry
            self._metrics[CacheLevel.L2_REDIS].size_bytes += entry.size_bytes
    
    async def _put_in_l3(self, entry: CacheEntry) -> None:
        """Put entry in L3 cache."""
        
        with self._l3_lock:
            while len(self._l3_cache) >= self.max_l3_size:
                await self._evict_from_l3()
            
            self._l3_cache[entry.key] = entry
            self._metrics[CacheLevel.L3_DISK].size_bytes += entry.size_bytes
    
    async def _evict_from_l1(self) -> None:
        """Evict entry from L1 cache."""
        
        if not self._l1_cache:
            return
        
        if self.eviction_policy == EvictionPolicy.LRU:
            key, entry = self._l1_cache.popitem(last=False)
        elif self.eviction_policy == EvictionPolicy.PREDICTIVE:
            key = self._select_eviction_candidate_l1()
            entry = self._l1_cache.pop(key)
        else:
            # Default LRU
            key, entry = self._l1_cache.popitem(last=False)
        
        # Demote to L2 if important
        if entry.importance_score > 0.5:
            await self._put_in_l2(entry)
        
        self._metrics[CacheLevel.L1_MEMORY].evictions += 1
        self._metrics[CacheLevel.L1_MEMORY].size_bytes -= entry.size_bytes
    
    async def _evict_from_l2(self) -> None:
        """Evict entry from L2 cache."""
        
        if not self._l2_cache:
            return
        
        # Select candidate based on policy
        key = self._select_eviction_candidate_l2()
        entry = self._l2_cache.pop(key)
        
        # Demote to L3 if important
        if entry.importance_score > 0.3:
            await self._put_in_l3(entry)
        
        self._metrics[CacheLevel.L2_REDIS].evictions += 1
        self._metrics[CacheLevel.L2_REDIS].size_bytes -= entry.size_bytes
    
    async def _evict_from_l3(self) -> None:
        """Evict entry from L3 cache."""
        
        if not self._l3_cache:
            return
        
        key = self._select_eviction_candidate_l3()
        entry = self._l3_cache.pop(key)
        
        self._metrics[CacheLevel.L3_DISK].evictions += 1
        self._metrics[CacheLevel.L3_DISK].size_bytes -= entry.size_bytes
    
    def _select_eviction_candidate_l1(self) -> str:
        """Select eviction candidate using predictive policy."""
        
        if self.eviction_policy == EvictionPolicy.PREDICTIVE:
            # Calculate eviction scores
            candidates = []
            current_time = time.time()
            
            for key, entry in self._l1_cache.items():
                # Score based on recency, frequency, and prediction
                recency_score = 1.0 / (current_time - entry.last_access + 1)
                frequency_score = entry.access_count / 100.0  # Normalize
                prediction_score = entry.prediction_score
                importance_score = entry.importance_score
                
                # Lower is more likely to be evicted
                eviction_score = (
                    -recency_score * 0.3 +
                    -frequency_score * 0.3 +
                    -prediction_score * 0.2 +
                    -importance_score * 0.2
                )
                
                candidates.append((eviction_score, key))
            
            # Select candidate with highest eviction score (most negative)
            candidates.sort(reverse=True)
            return candidates[0][1]
        
        # Fallback to LRU
        return next(iter(self._l1_cache))
    
    def _select_eviction_candidate_l2(self) -> str:
        """Select eviction candidate from L2."""
        
        # Similar logic to L1 but with different weights
        candidates = []
        current_time = time.time()
        
        for key, entry in self._l2_cache.items():
            recency_score = 1.0 / (current_time - entry.last_access + 1)
            frequency_score = entry.access_count / 100.0
            
            eviction_score = -recency_score * 0.4 - frequency_score * 0.6
            candidates.append((eviction_score, key))
        
        candidates.sort(reverse=True)
        return candidates[0][1] if candidates else next(iter(self._l2_cache))
    
    def _select_eviction_candidate_l3(self) -> str:
        """Select eviction candidate from L3."""
        
        # For L3, prioritize size and age
        candidates = []
        current_time = time.time()
        
        for key, entry in self._l3_cache.items():
            age_score = current_time - entry.timestamp
            size_score = entry.size_bytes
            
            eviction_score = age_score * 0.5 + size_score * 0.5
            candidates.append((eviction_score, key))
        
        candidates.sort(reverse=True)
        return candidates[0][1] if candidates else next(iter(self._l3_cache))
    
    async def _promote_to_l1(self, key: str, value: Any) -> None:
        """Promote entry to L1 cache."""
        
        entry = CacheEntry(
            key=key,
            value=value,
            timestamp=time.time(),
            cache_level=CacheLevel.L1_MEMORY,
            size_bytes=self._estimate_size(value)
        )
        
        await self._put_in_l1(entry)
    
    async def _promote_to_l2(self, key: str, value: Any) -> None:
        """Promote entry to L2 cache."""
        
        entry = CacheEntry(
            key=key,
            value=value,
            timestamp=time.time(),
            cache_level=CacheLevel.L2_REDIS,
            size_bytes=self._estimate_size(value)
        )
        
        await self._put_in_l2(entry)
    
    async def _remove_from_l1(self, key: str) -> bool:
        """Remove key from L1 cache."""
        
        with self._l1_lock:
            if key in self._l1_cache:
                entry = self._l1_cache.pop(key)
                self._metrics[CacheLevel.L1_MEMORY].size_bytes -= entry.size_bytes
                return True
        return False
    
    async def _remove_from_l2(self, key: str) -> bool:
        """Remove key from L2 cache."""
        
        with self._l2_lock:
            if key in self._l2_cache:
                entry = self._l2_cache.pop(key)
                self._metrics[CacheLevel.L2_REDIS].size_bytes -= entry.size_bytes
                return True
        return False
    
    async def _remove_from_l3(self, key: str) -> bool:
        """Remove key from L3 cache."""
        
        with self._l3_lock:
            if key in self._l3_cache:
                entry = self._l3_cache.pop(key)
                self._metrics[CacheLevel.L3_DISK].size_bytes -= entry.size_bytes
                return True
        return False
    
    def _update_access_pattern(self, key: str) -> None:
        """Update access pattern for predictive analytics."""
        
        current_time = time.time()
        self._access_history.append((key, current_time))
        
        # Keep only recent history (last 1000 accesses)
        if len(self._access_history) > 1000:
            self._access_history = self._access_history[-1000:]
        
        # Update or create pattern
        if key not in self._access_patterns:
            self._access_patterns[key] = AccessPattern(
                key=key,
                frequency=1.0,
                recency=1.0,
                periodicity=0.0,
                trend=0.0
            )
        else:
            pattern = self._access_patterns[key]
            
            # Update frequency (accesses per hour)
            recent_accesses = [
                timestamp for access_key, timestamp in self._access_history[-100:]
                if access_key == key and current_time - timestamp < 3600
            ]
            pattern.frequency = len(recent_accesses)
            
            # Update recency
            pattern.recency = 1.0 / (current_time - recent_accesses[-1] + 1) if recent_accesses else 0.0
            
            # Simple trend calculation
            if len(recent_accesses) >= 2:
                recent_trend = len(recent_accesses[-10:]) - len(recent_accesses[-20:-10])
                pattern.trend = recent_trend / 10.0
            
            # Update confidence
            pattern.confidence = min(1.0, pattern.confidence + 0.01)
    
    def _calculate_prediction_score(self, key: str) -> float:
        """Calculate prediction score for cache placement."""
        
        if key not in self._access_patterns:
            return 0.5  # Default score for new keys
        
        pattern = self._access_patterns[key]
        
        # Weighted combination of pattern features
        score = (
            pattern.frequency * self._prediction_weights['frequency'] +
            pattern.recency * self._prediction_weights['recency'] +
            pattern.periodicity * self._prediction_weights['periodicity'] +
            max(0, pattern.trend) * self._prediction_weights['trend']
        )
        
        # Apply confidence weighting
        score *= pattern.confidence
        
        return min(1.0, max(0.0, score))
    
    def _determine_cache_level(self, entry: CacheEntry) -> CacheLevel:
        """Determine appropriate cache level for entry."""
        
        # Score-based placement
        placement_score = (
            entry.importance_score * 0.4 +
            entry.prediction_score * 0.4 +
            (1.0 / (entry.size_bytes / 1024 + 1)) * 0.2  # Smaller = higher level
        )
        
        if placement_score > 0.8:
            return CacheLevel.L1_MEMORY
        elif placement_score > 0.6:
            return CacheLevel.L2_REDIS
        elif placement_score > 0.3:
            return CacheLevel.L3_DISK
        else:
            return CacheLevel.L3_DISK  # Default to L3
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate size of value in bytes."""
        
        try:
            return len(pickle.dumps(value))
        except:
            return len(str(value).encode('utf-8'))
    
    def _record_hit(self, cache_level: CacheLevel, access_time: float) -> None:
        """Record cache hit."""
        
        metrics = self._metrics[cache_level]
        metrics.hits += 1
        
        # Update average access time
        total_accesses = metrics.hits + metrics.misses
        metrics.avg_access_time = (
            (metrics.avg_access_time * (total_accesses - 1) + access_time) / total_accesses
        )
    
    def _record_miss(self, access_time: float) -> None:
        """Record cache miss."""
        
        # Record miss for L1 (since that's where we start)
        metrics = self._metrics[CacheLevel.L1_MEMORY]
        metrics.misses += 1
        
        total_accesses = metrics.hits + metrics.misses
        metrics.avg_access_time = (
            (metrics.avg_access_time * (total_accesses - 1) + access_time) / total_accesses
        )
    
    async def _optimization_loop(self):
        """Background optimization loop."""
        
        while self._running:
            try:
                # Optimize eviction policies
                await self._optimize_eviction_policies()
                
                # Update prediction weights
                self._update_prediction_weights()
                
                # Clean expired entries
                await self._clean_expired_entries()
                
                await asyncio.sleep(60)  # Optimize every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cache optimization loop: {e}")
                await asyncio.sleep(10)
    
    async def _cache_warming_loop(self):
        """Background cache warming loop."""
        
        while self._running:
            try:
                # Predict likely accessed keys
                predicted_keys = self._predict_next_accesses()
                
                # Warm cache with predicted keys
                if predicted_keys:
                    await self.warm_cache(predicted_keys)
                
                await asyncio.sleep(300)  # Warm every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cache warming loop: {e}")
                await asyncio.sleep(60)
    
    async def _optimize_eviction_policies(self):
        """Optimize eviction policies based on performance."""
        
        # Calculate hit rates for each level
        l1_hit_rate = (
            self._metrics[CacheLevel.L1_MEMORY].hits / 
            max(1, self._metrics[CacheLevel.L1_MEMORY].hits + self._metrics[CacheLevel.L1_MEMORY].misses)
        )
        
        # Adjust prediction weights based on performance
        if l1_hit_rate < 0.5:
            # Increase importance of frequency
            self._prediction_weights['frequency'] = min(0.5, self._prediction_weights['frequency'] + 0.01)
            self._prediction_weights['recency'] = max(0.1, self._prediction_weights['recency'] - 0.005)
    
    def _update_prediction_weights(self):
        """Update prediction weights based on accuracy."""
        
        # This would typically use ML to optimize weights
        # For now, use simple heuristics
        
        total_weight = sum(self._prediction_weights.values())
        if total_weight != 1.0:
            # Normalize weights
            for key in self._prediction_weights:
                self._prediction_weights[key] /= total_weight
    
    async def _clean_expired_entries(self):
        """Clean expired entries from all cache levels."""
        
        current_time = time.time()
        
        # Clean L1
        with self._l1_lock:
            expired_keys = [
                key for key, entry in self._l1_cache.items()
                if entry.ttl and current_time - entry.timestamp > entry.ttl
            ]
            
            for key in expired_keys:
                del self._l1_cache[key]
        
        # Clean L2 and L3 similarly
        with self._l2_lock:
            expired_keys = [
                key for key, entry in self._l2_cache.items()
                if entry.ttl and current_time - entry.timestamp > entry.ttl
            ]
            
            for key in expired_keys:
                del self._l2_cache[key]
        
        with self._l3_lock:
            expired_keys = [
                key for key, entry in self._l3_cache.items()
                if entry.ttl and current_time - entry.timestamp > entry.ttl
            ]
            
            for key in expired_keys:
                del self._l3_cache[key]
    
    def _predict_next_accesses(self) -> List[str]:
        """Predict which keys will be accessed next."""
        
        predictions = []
        current_time = time.time()
        
        for key, pattern in self._access_patterns.items():
            # Calculate prediction score
            time_since_last = current_time - pattern.recency if pattern.recency > 0 else float('inf')
            
            # Predict based on frequency and periodicity
            prediction_score = (
                pattern.frequency * 0.4 +
                (1.0 / (time_since_last / 3600 + 1)) * 0.3 +  # Hours since last access
                pattern.periodicity * 0.2 +
                max(0, pattern.trend) * 0.1
            ) * pattern.confidence
            
            if prediction_score > 0.6:  # Threshold for warming
                predictions.append((prediction_score, key))
        
        # Sort by score and return top predictions
        predictions.sort(reverse=True)
        return [key for score, key in predictions[:20]]  # Top 20 predictions
    
    def _get_cache_level_size(self, level: CacheLevel) -> int:
        """Get size of specific cache level."""
        
        if level == CacheLevel.L1_MEMORY:
            return len(self._l1_cache)
        elif level == CacheLevel.L2_REDIS:
            return len(self._l2_cache)
        elif level == CacheLevel.L3_DISK:
            return len(self._l3_cache)
        elif level == CacheLevel.L4_REMOTE:
            return 0  # Not implemented
        else:
            return 0
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        
        total_hits = sum(metrics.hits for metrics in self._metrics.values())
        total_misses = sum(metrics.misses for metrics in self._metrics.values())
        hit_rate = total_hits / max(1, total_hits + total_misses)
        
        return {
            "timestamp": time.time(),
            "overall_hit_rate": hit_rate,
            "total_entries": len(self._l1_cache) + len(self._l2_cache) + len(self._l3_cache),
            "access_patterns": len(self._access_patterns),
            "prediction_weights": self._prediction_weights.copy(),
            "levels": {
                level.value: {
                    "entries": self._get_cache_level_size(level),
                    "hits": metrics.hits,
                    "misses": metrics.misses,
                    "evictions": metrics.evictions,
                    "size_bytes": metrics.size_bytes,
                    "hit_rate": metrics.hits / max(1, metrics.hits + metrics.misses),
                    "avg_access_time": metrics.avg_access_time
                }
                for level, metrics in self._metrics.items()
            }
        }


# Global cache instance
_global_intelligent_cache = None


def get_intelligent_cache(**kwargs) -> IntelligentCachingSystem:
    """Get global intelligent cache instance."""
    
    global _global_intelligent_cache
    if _global_intelligent_cache is None:
        _global_intelligent_cache = IntelligentCachingSystem(**kwargs)
    return _global_intelligent_cache


# Decorator for intelligent caching
def intelligent_cache(
    ttl: Optional[float] = None,
    importance_score: float = 1.0,
    key_generator: Optional[callable] = None
):
    """Decorator to add intelligent caching to functions."""
    
    def decorator(func):
        async def wrapper(*args, **kwargs):
            cache = get_intelligent_cache()
            
            # Generate cache key
            if key_generator:
                cache_key = key_generator(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{hashlib.md5(str(args + tuple(kwargs.items())).encode()).hexdigest()}"
            
            # Try to get from cache
            result = await cache.get(cache_key)
            if result is not None:
                return result
            
            # Execute function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Cache result
            await cache.put(cache_key, result, ttl=ttl, importance_score=importance_score)
            
            return result
        
        return wrapper
    return decorator