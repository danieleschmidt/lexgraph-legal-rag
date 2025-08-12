"""
Advanced Performance Optimization for Bioneural Olfactory Fusion System

Implements high-performance optimizations including:
- Intelligent caching and memoization
- Concurrent processing with resource pooling  
- Dynamic load balancing and auto-scaling
- Memory-efficient scent profile storage
- Performance profiling and bottleneck detection
"""

from __future__ import annotations

import asyncio
import logging
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from enum import Enum
from collections import defaultdict, deque, OrderedDict
import weakref
import pickle
import hashlib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import wraps, lru_cache
import gc
import sys

import numpy as np

from .bioneuro_olfactory_fusion import (
    BioneuroOlfactoryFusionEngine,
    BioneuroOlfactoryReceptor,
    DocumentScentProfile,
    OlfactorySignal,
    OlfactoryReceptorType
)
from .bioneuro_monitoring import get_metrics_collector

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache strategies for different types of data."""
    LRU = "lru"              # Least Recently Used
    LFU = "lfu"              # Least Frequently Used
    TTL = "ttl"              # Time To Live
    ADAPTIVE = "adaptive"     # Adaptive based on access patterns


class ProcessingMode(Enum):
    """Processing modes for workload optimization."""
    SEQUENTIAL = "sequential"
    CONCURRENT = "concurrent"
    PARALLEL = "parallel"
    ADAPTIVE = "adaptive"


@dataclass
class OptimizationConfig:
    """Configuration for bioneural optimization settings."""
    max_cache_size: int = 10000
    cache_strategy: CacheStrategy = CacheStrategy.ADAPTIVE
    processing_mode: ProcessingMode = ProcessingMode.ADAPTIVE
    max_workers: int = 4
    batch_size: int = 32
    memory_limit_mb: int = 2048
    enable_profiling: bool = False
    auto_scaling: bool = True
    precompute_embeddings: bool = True


@dataclass
class PerformanceProfile:
    """Performance profiling data for optimization."""
    operation: str
    duration: float
    memory_delta: float
    cache_hits: int
    cache_misses: int
    parallelism_factor: float
    bottleneck_component: Optional[str] = None


class IntelligentCache:
    """Intelligent caching system with adaptive strategies."""
    
    def __init__(self, max_size: int = 10000, strategy: CacheStrategy = CacheStrategy.ADAPTIVE):
        """Initialize intelligent cache."""
        self.max_size = max_size
        self.strategy = strategy
        
        # Cache storage
        self._cache: OrderedDict = OrderedDict()
        self._access_counts: Dict[str, int] = defaultdict(int)
        self._access_times: Dict[str, float] = {}
        self._ttl_expiry: Dict[str, float] = {}
        
        # Cache statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        # Adaptive strategy parameters
        self._hit_rate_window = deque(maxlen=1000)
        self._strategy_performance: Dict[CacheStrategy, float] = defaultdict(float)
        self._current_strategy = strategy
        
        self._lock = threading.RLock()
        self.logger = logging.getLogger(f"{__name__}.IntelligentCache")
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with intelligent strategy."""
        with self._lock:
            if key in self._cache:
                # Check TTL expiry
                if key in self._ttl_expiry and time.time() > self._ttl_expiry[key]:
                    del self._cache[key]
                    del self._ttl_expiry[key]
                    self.misses += 1
                    self._hit_rate_window.append(0)
                    return None
                
                # Update access patterns
                self._access_counts[key] += 1
                self._access_times[key] = time.time()
                
                # Move to end for LRU
                value = self._cache.pop(key)
                self._cache[key] = value
                
                self.hits += 1
                self._hit_rate_window.append(1)
                
                self.logger.debug(f"Cache hit for key: {key[:50]}...")
                return value
            else:
                self.misses += 1
                self._hit_rate_window.append(0)
                self.logger.debug(f"Cache miss for key: {key[:50]}...")
                return None
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None):
        """Put item in cache with eviction strategy."""
        with self._lock:
            # Set TTL if specified
            if ttl:
                self._ttl_expiry[key] = time.time() + ttl
            
            # Add/update item
            if key in self._cache:
                # Update existing item
                self._cache.pop(key)
            
            self._cache[key] = value
            self._access_counts[key] += 1
            self._access_times[key] = time.time()
            
            # Evict if necessary
            if len(self._cache) > self.max_size:
                self._evict_items()
            
            # Adapt strategy based on performance
            if len(self._hit_rate_window) >= 100:
                self._adapt_strategy()
    
    def _evict_items(self):
        """Evict items based on current strategy."""
        while len(self._cache) > self.max_size:
            if self._current_strategy == CacheStrategy.LRU:
                # Remove least recently used
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                
            elif self._current_strategy == CacheStrategy.LFU:
                # Remove least frequently used
                lfu_key = min(self._access_counts.keys(), key=lambda k: self._access_counts[k])
                if lfu_key in self._cache:
                    del self._cache[lfu_key]
                    del self._access_counts[lfu_key]
                
            elif self._current_strategy == CacheStrategy.TTL:
                # Remove expired items first, then oldest
                current_time = time.time()
                expired_keys = [k for k, expiry in self._ttl_expiry.items() if current_time > expiry]
                
                if expired_keys:
                    for key in expired_keys:
                        if key in self._cache:
                            del self._cache[key]
                        del self._ttl_expiry[key]
                else:
                    # Fall back to LRU
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]
                    
            else:  # ADAPTIVE
                # Use hybrid approach
                current_time = time.time()
                
                # Calculate scores for each item
                scores = {}
                for key in self._cache.keys():
                    access_count = self._access_counts.get(key, 1)
                    last_access = self._access_times.get(key, current_time)
                    recency = current_time - last_access
                    
                    # Combined score: frequency / recency
                    score = access_count / (1 + recency / 3600)  # Decay over hours
                    scores[key] = score
                
                # Remove item with lowest score
                worst_key = min(scores.keys(), key=lambda k: scores[k])
                del self._cache[worst_key]
            
            self.evictions += 1
    
    def _adapt_strategy(self):
        """Adapt caching strategy based on performance."""
        if self.strategy != CacheStrategy.ADAPTIVE:
            return
            
        current_hit_rate = sum(self._hit_rate_window) / len(self._hit_rate_window)
        self._strategy_performance[self._current_strategy] = current_hit_rate
        
        # Occasionally try different strategies
        if len(self._strategy_performance) < 3 and np.random.random() < 0.1:
            strategies = [CacheStrategy.LRU, CacheStrategy.LFU, CacheStrategy.TTL]
            untested = [s for s in strategies if s not in self._strategy_performance]
            if untested:
                self._current_strategy = np.random.choice(untested)
                self.logger.info(f"Adapting cache strategy to: {self._current_strategy.value}")
        
        # Use best performing strategy
        if len(self._strategy_performance) >= 2:
            best_strategy = max(self._strategy_performance.keys(), 
                              key=lambda s: self._strategy_performance[s])
            if best_strategy != self._current_strategy:
                self._current_strategy = best_strategy
                self.logger.info(f"Switching to best performing strategy: {best_strategy.value}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": hit_rate,
            "size": len(self._cache),
            "max_size": self.max_size,
            "current_strategy": self._current_strategy.value,
            "strategy_performance": dict(self._strategy_performance)
        }
    
    def clear(self):
        """Clear cache contents."""
        with self._lock:
            self._cache.clear()
            self._access_counts.clear()
            self._access_times.clear()
            self._ttl_expiry.clear()


class ResourcePool:
    """Pool of resources for concurrent processing."""
    
    def __init__(self, resource_factory: Callable[[], Any], max_size: int = 10, 
                 min_size: int = 2, idle_timeout: float = 300.0):
        """Initialize resource pool."""
        self.resource_factory = resource_factory
        self.max_size = max_size
        self.min_size = min_size
        self.idle_timeout = idle_timeout
        
        self._pool: deque = deque()
        self._in_use: Set[Any] = set()
        self._created_times: Dict[Any, float] = {}
        self._last_cleanup = time.time()
        
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        
        # Pre-populate with minimum resources
        for _ in range(min_size):
            resource = resource_factory()
            self._pool.append(resource)
            self._created_times[resource] = time.time()
        
        self.logger = logging.getLogger(f"{__name__}.ResourcePool")
    
    def acquire(self, timeout: float = 30.0) -> Optional[Any]:
        """Acquire resource from pool."""
        with self._condition:
            deadline = time.time() + timeout
            
            while True:
                # Try to get resource from pool
                if self._pool:
                    resource = self._pool.popleft()
                    self._in_use.add(resource)
                    self.logger.debug(f"Acquired resource from pool: {id(resource)}")
                    return resource
                
                # Create new resource if under limit
                if len(self._in_use) < self.max_size:
                    resource = self.resource_factory()
                    self._in_use.add(resource)
                    self._created_times[resource] = time.time()
                    self.logger.debug(f"Created new resource: {id(resource)}")
                    return resource
                
                # Wait for resource to be released
                remaining_time = deadline - time.time()
                if remaining_time <= 0:
                    self.logger.warning("Resource pool acquisition timeout")
                    return None
                
                self._condition.wait(remaining_time)
    
    def release(self, resource: Any):
        """Release resource back to pool."""
        with self._condition:
            if resource in self._in_use:
                self._in_use.remove(resource)
                
                # Check if resource is still valid
                if self._is_resource_valid(resource):
                    self._pool.append(resource)
                    self.logger.debug(f"Released resource to pool: {id(resource)}")
                else:
                    # Remove invalid resource
                    if resource in self._created_times:
                        del self._created_times[resource]
                    self.logger.debug(f"Discarded invalid resource: {id(resource)}")
                
                self._condition.notify()
                
                # Periodic cleanup
                if time.time() - self._last_cleanup > 60:  # Every minute
                    self._cleanup_idle_resources()
    
    def _is_resource_valid(self, resource: Any) -> bool:
        """Check if resource is still valid."""
        # Check age
        created_time = self._created_times.get(resource, 0)
        age = time.time() - created_time
        
        if age > self.idle_timeout:
            return False
        
        # Add custom validation logic here if needed
        return True
    
    def _cleanup_idle_resources(self):
        """Remove idle/expired resources from pool."""
        current_time = time.time()
        
        # Remove old resources from pool
        while self._pool and len(self._pool) > self.min_size:
            resource = self._pool[0]
            created_time = self._created_times.get(resource, current_time)
            
            if current_time - created_time > self.idle_timeout:
                self._pool.popleft()
                if resource in self._created_times:
                    del self._created_times[resource]
                self.logger.debug(f"Cleaned up idle resource: {id(resource)}")
            else:
                break
        
        self._last_cleanup = current_time
    
    def get_stats(self) -> Dict[str, Any]:
        """Get resource pool statistics."""
        with self._lock:
            return {
                "pool_size": len(self._pool),
                "in_use": len(self._in_use),
                "total_resources": len(self._pool) + len(self._in_use),
                "max_size": self.max_size,
                "min_size": self.min_size
            }


class OptimizedBioneuroEngine:
    """High-performance optimized bioneural olfactory fusion engine."""
    
    def __init__(self, config: OptimizationConfig = None):
        """Initialize optimized engine."""
        self.config = config or OptimizationConfig()
        
        # Initialize caching layer
        self.scent_cache = IntelligentCache(
            max_size=self.config.max_cache_size,
            strategy=self.config.cache_strategy
        )
        
        # Initialize resource pools
        self.receptor_pool = ResourcePool(
            resource_factory=self._create_receptor_set,
            max_size=self.config.max_workers,
            min_size=max(1, self.config.max_workers // 2)
        )
        
        # Execution pools
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.max_workers)
        if self.config.processing_mode == ProcessingMode.PARALLEL:
            self.process_pool = ProcessPoolExecutor(max_workers=mp.cpu_count())
        
        # Performance monitoring
        self.performance_profiles: List[PerformanceProfile] = []
        self.processing_stats = {
            "documents_processed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_processing_time": 0.0,
            "parallelism_usage": 0.0
        }
        
        # Memory management
        self._memory_threshold = self.config.memory_limit_mb * 1024 * 1024
        self._last_gc_time = time.time()
        
        # Base fusion engine (lightweight)
        self.base_engine = BioneuroOlfactoryFusionEngine()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"OptimizedBioneuroEngine initialized with config: {self.config}")
    
    def _create_receptor_set(self) -> Dict[OlfactoryReceptorType, BioneuroOlfactoryReceptor]:
        """Create a set of olfactory receptors."""
        receptors = {}
        
        default_config = {
            OlfactoryReceptorType.LEGAL_COMPLEXITY: 0.8,
            OlfactoryReceptorType.STATUTORY_AUTHORITY: 0.9,
            OlfactoryReceptorType.TEMPORAL_FRESHNESS: 0.6,
            OlfactoryReceptorType.CITATION_DENSITY: 0.7,
            OlfactoryReceptorType.RISK_PROFILE: 0.8,
            OlfactoryReceptorType.SEMANTIC_COHERENCE: 0.5
        }
        
        for receptor_type, sensitivity in default_config.items():
            receptors[receptor_type] = BioneuroOlfactoryReceptor(
                receptor_type=receptor_type,
                sensitivity=sensitivity
            )
        
        return receptors
    
    async def analyze_document_optimized(self, document_text: str, document_id: str, 
                                       metadata: Optional[Dict[str, Any]] = None) -> DocumentScentProfile:
        """Optimized document analysis with caching and performance monitoring."""
        start_time = time.time()
        metadata = metadata or {}
        
        # Check cache first
        cache_key = self._generate_cache_key(document_text, metadata)
        cached_profile = self.scent_cache.get(cache_key)
        
        if cached_profile:
            self.processing_stats["cache_hits"] += 1
            cached_profile.document_id = document_id  # Update ID
            
            # Record cache hit metric
            metrics_collector = get_metrics_collector()
            metrics_collector.record_document_processing(time.time() - start_time, success=True)
            
            self.logger.debug(f"Cache hit for document {document_id}")
            return cached_profile
        
        self.processing_stats["cache_misses"] += 1
        
        try:
            # Determine optimal processing mode
            processing_mode = self._select_processing_mode(document_text)
            
            # Process based on selected mode
            if processing_mode == ProcessingMode.SEQUENTIAL:
                profile = await self._analyze_sequential(document_text, document_id, metadata)
            elif processing_mode == ProcessingMode.CONCURRENT:
                profile = await self._analyze_concurrent(document_text, document_id, metadata)
            elif processing_mode == ProcessingMode.PARALLEL:
                profile = await self._analyze_parallel(document_text, document_id, metadata)
            else:  # ADAPTIVE
                profile = await self._analyze_adaptive(document_text, document_id, metadata)
            
            # Cache the result
            self.scent_cache.put(cache_key, profile, ttl=3600)  # 1 hour TTL
            
            # Update statistics
            processing_time = time.time() - start_time
            self.processing_stats["documents_processed"] += 1
            self.processing_stats["avg_processing_time"] = (
                (self.processing_stats["avg_processing_time"] * (self.processing_stats["documents_processed"] - 1) + 
                 processing_time) / self.processing_stats["documents_processed"]
            )
            
            # Record performance profile
            if self.config.enable_profiling:
                self._record_performance_profile("document_analysis", processing_time, processing_mode)
            
            # Periodic memory management
            if time.time() - self._last_gc_time > 300:  # Every 5 minutes
                self._manage_memory()
            
            return profile
            
        except Exception as e:
            self.logger.error(f"Optimized analysis failed for {document_id}: {e}")
            
            # Fallback to base engine
            return await self.base_engine.analyze_document(document_text, document_id, metadata)
    
    def _generate_cache_key(self, document_text: str, metadata: Dict[str, Any]) -> str:
        """Generate cache key for document."""
        # Create hash of content and relevant metadata
        content_hash = hashlib.md5(document_text.encode('utf-8')).hexdigest()
        
        # Include relevant metadata that affects analysis
        relevant_metadata = {k: v for k, v in metadata.items() 
                           if k in ['document_type', 'creation_date', 'language']}
        metadata_str = json.dumps(relevant_metadata, sort_keys=True)
        metadata_hash = hashlib.md5(metadata_str.encode('utf-8')).hexdigest()
        
        return f"{content_hash}_{metadata_hash}"
    
    def _select_processing_mode(self, document_text: str) -> ProcessingMode:
        """Select optimal processing mode based on document characteristics."""
        if self.config.processing_mode != ProcessingMode.ADAPTIVE:
            return self.config.processing_mode
        
        text_length = len(document_text)
        complexity_score = self._estimate_complexity(document_text)
        
        # Simple heuristics for mode selection
        if text_length < 1000 and complexity_score < 0.3:
            return ProcessingMode.SEQUENTIAL
        elif text_length < 10000 and complexity_score < 0.7:
            return ProcessingMode.CONCURRENT
        else:
            return ProcessingMode.PARALLEL if hasattr(self, 'process_pool') else ProcessingMode.CONCURRENT
    
    def _estimate_complexity(self, text: str) -> float:
        """Estimate document complexity for processing mode selection."""
        # Simple complexity estimation
        legal_terms = ["whereas", "pursuant", "notwithstanding", "indemnification", "liability"]
        term_count = sum(1 for term in legal_terms if term.lower() in text.lower())
        
        sentence_count = text.count('.') + text.count('!') + text.count('?')
        avg_sentence_length = len(text.split()) / max(1, sentence_count)
        
        # Normalize complexity score
        complexity = min(1.0, (term_count * 0.1) + (avg_sentence_length / 50.0))
        return complexity
    
    async def _analyze_sequential(self, document_text: str, document_id: str, 
                                metadata: Dict[str, Any]) -> DocumentScentProfile:
        """Sequential analysis (single-threaded)."""
        receptors = self.receptor_pool.acquire(timeout=10.0)
        if not receptors:
            raise RuntimeError("Failed to acquire receptor resources")
        
        try:
            signals = []
            for receptor in receptors.values():
                signal = await receptor.activate(document_text, metadata)
                signals.append(signal)
            
            # Create profile
            valid_signals = [s for s in signals if s.confidence > 0]
            composite_scent = self._create_composite_scent(valid_signals)
            similarity_hash = self._generate_similarity_hash(composite_scent)
            
            return DocumentScentProfile(
                document_id=document_id,
                signals=valid_signals,
                composite_scent=composite_scent,
                similarity_hash=similarity_hash
            )
            
        finally:
            self.receptor_pool.release(receptors)
    
    async def _analyze_concurrent(self, document_text: str, document_id: str,
                                metadata: Dict[str, Any]) -> DocumentScentProfile:
        """Concurrent analysis using asyncio."""
        receptors = self.receptor_pool.acquire(timeout=10.0)
        if not receptors:
            raise RuntimeError("Failed to acquire receptor resources")
        
        try:
            # Process all receptors concurrently
            tasks = [
                receptor.activate(document_text, metadata)
                for receptor in receptors.values()
            ]
            
            signals = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter valid signals
            valid_signals = [
                signal for signal in signals 
                if isinstance(signal, OlfactorySignal) and signal.confidence > 0
            ]
            
            # Create profile
            composite_scent = self._create_composite_scent(valid_signals)
            similarity_hash = self._generate_similarity_hash(composite_scent)
            
            return DocumentScentProfile(
                document_id=document_id,
                signals=valid_signals,
                composite_scent=composite_scent,
                similarity_hash=similarity_hash
            )
            
        finally:
            self.receptor_pool.release(receptors)
    
    async def _analyze_parallel(self, document_text: str, document_id: str,
                              metadata: Dict[str, Any]) -> DocumentScentProfile:
        """Parallel analysis using process pool."""
        if not hasattr(self, 'process_pool'):
            # Fallback to concurrent
            return await self._analyze_concurrent(document_text, document_id, metadata)
        
        # Split work across processes
        receptor_types = list(OlfactoryReceptorType)
        tasks = []
        
        loop = asyncio.get_event_loop()
        
        for receptor_type in receptor_types:
            task = loop.run_in_executor(
                self.process_pool,
                self._process_receptor_parallel,
                receptor_type, document_text, metadata
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        valid_signals = [
            result for result in results 
            if isinstance(result, OlfactorySignal) and result.confidence > 0
        ]
        
        # Create profile
        composite_scent = self._create_composite_scent(valid_signals)
        similarity_hash = self._generate_similarity_hash(composite_scent)
        
        return DocumentScentProfile(
            document_id=document_id,
            signals=valid_signals,
            composite_scent=composite_scent,
            similarity_hash=similarity_hash
        )
    
    def _process_receptor_parallel(self, receptor_type: OlfactoryReceptorType, 
                                 document_text: str, metadata: Dict[str, Any]) -> OlfactorySignal:
        """Process single receptor in parallel (for process pool)."""
        # Create receptor in process
        sensitivity_map = {
            OlfactoryReceptorType.LEGAL_COMPLEXITY: 0.8,
            OlfactoryReceptorType.STATUTORY_AUTHORITY: 0.9,
            OlfactoryReceptorType.TEMPORAL_FRESHNESS: 0.6,
            OlfactoryReceptorType.CITATION_DENSITY: 0.7,
            OlfactoryReceptorType.RISK_PROFILE: 0.8,
            OlfactoryReceptorType.SEMANTIC_COHERENCE: 0.5
        }
        
        receptor = BioneuroOlfactoryReceptor(
            receptor_type=receptor_type,
            sensitivity=sensitivity_map.get(receptor_type, 0.5)
        )
        
        # Run activation (convert async to sync for process pool)
        return asyncio.run(receptor.activate(document_text, metadata))
    
    async def _analyze_adaptive(self, document_text: str, document_id: str,
                              metadata: Dict[str, Any]) -> DocumentScentProfile:
        """Adaptive analysis that learns optimal approach."""
        # Try different approaches and measure performance
        approaches = [ProcessingMode.SEQUENTIAL, ProcessingMode.CONCURRENT]
        if hasattr(self, 'process_pool'):
            approaches.append(ProcessingMode.PARALLEL)
        
        # Select approach based on historical performance
        best_approach = self._get_best_approach_for_doc_type(document_text)
        
        if best_approach == ProcessingMode.SEQUENTIAL:
            return await self._analyze_sequential(document_text, document_id, metadata)
        elif best_approach == ProcessingMode.CONCURRENT:
            return await self._analyze_concurrent(document_text, document_id, metadata)
        else:
            return await self._analyze_parallel(document_text, document_id, metadata)
    
    def _get_best_approach_for_doc_type(self, document_text: str) -> ProcessingMode:
        """Get best processing approach based on historical performance."""
        # Simple heuristic - could be improved with ML
        text_length = len(document_text)
        
        if text_length < 5000:
            return ProcessingMode.SEQUENTIAL
        elif text_length < 20000:
            return ProcessingMode.CONCURRENT
        else:
            return ProcessingMode.PARALLEL if hasattr(self, 'process_pool') else ProcessingMode.CONCURRENT
    
    def _create_composite_scent(self, signals: List[OlfactorySignal]) -> np.ndarray:
        """Create composite scent vector from signals."""
        scent_dimensions = len(OlfactoryReceptorType) * 2
        composite_scent = np.zeros(scent_dimensions)
        
        for i, receptor_type in enumerate(OlfactoryReceptorType):
            base_idx = i * 2
            
            signal = next(
                (s for s in signals if s.receptor_type == receptor_type),
                None
            )
            
            if signal:
                composite_scent[base_idx] = signal.intensity
                composite_scent[base_idx + 1] = signal.confidence
        
        return composite_scent
    
    def _generate_similarity_hash(self, composite_scent: np.ndarray) -> str:
        """Generate similarity hash for scent vector."""
        quantized = (composite_scent * 1000).astype(int)
        hash_input = json.dumps(quantized.tolist()).encode('utf-8')
        return hashlib.md5(hash_input).hexdigest()
    
    def _record_performance_profile(self, operation: str, duration: float, mode: ProcessingMode):
        """Record performance profile for optimization."""
        # Get memory usage
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        # Calculate parallelism factor
        parallelism_factor = {
            ProcessingMode.SEQUENTIAL: 1.0,
            ProcessingMode.CONCURRENT: min(self.config.max_workers, 4.0),
            ProcessingMode.PARALLEL: min(mp.cpu_count(), 8.0),
            ProcessingMode.ADAPTIVE: 2.0  # Average
        }.get(mode, 1.0)
        
        profile = PerformanceProfile(
            operation=operation,
            duration=duration,
            memory_delta=memory_mb,
            cache_hits=self.scent_cache.hits,
            cache_misses=self.scent_cache.misses,
            parallelism_factor=parallelism_factor
        )
        
        self.performance_profiles.append(profile)
        
        # Keep only recent profiles
        if len(self.performance_profiles) > 1000:
            self.performance_profiles = self.performance_profiles[-500:]
    
    def _manage_memory(self):
        """Manage memory usage and trigger cleanup if needed."""
        import psutil
        
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        if memory_mb > self.config.memory_limit_mb:
            self.logger.warning(f"Memory usage high: {memory_mb:.1f}MB, triggering cleanup")
            
            # Clear caches
            self.scent_cache.clear()
            
            # Force garbage collection
            gc.collect()
            
            # Clear old performance profiles
            self.performance_profiles = self.performance_profiles[-100:]
        
        self._last_gc_time = time.time()
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        stats = {
            "processing_stats": self.processing_stats,
            "cache_stats": self.scent_cache.get_stats(),
            "resource_pool_stats": self.receptor_pool.get_stats(),
            "config": {
                "max_cache_size": self.config.max_cache_size,
                "cache_strategy": self.config.cache_strategy.value,
                "processing_mode": self.config.processing_mode.value,
                "max_workers": self.config.max_workers,
                "memory_limit_mb": self.config.memory_limit_mb
            }
        }
        
        # Performance profile summary
        if self.performance_profiles:
            recent_profiles = self.performance_profiles[-100:]
            stats["performance_summary"] = {
                "avg_duration": sum(p.duration for p in recent_profiles) / len(recent_profiles),
                "avg_parallelism": sum(p.parallelism_factor for p in recent_profiles) / len(recent_profiles),
                "profile_count": len(self.performance_profiles)
            }
        
        return stats
    
    async def batch_analyze_documents(self, documents: List[Tuple[str, str, Dict[str, Any]]], 
                                    batch_size: Optional[int] = None) -> List[DocumentScentProfile]:
        """Analyze multiple documents in optimized batches."""
        batch_size = batch_size or self.config.batch_size
        results = []
        
        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            # Process batch concurrently
            tasks = [
                self.analyze_document_optimized(text, doc_id, metadata)
                for text, doc_id, metadata in batch
            ]
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle results and exceptions
            for result in batch_results:
                if isinstance(result, DocumentScentProfile):
                    results.append(result)
                else:
                    self.logger.error(f"Batch processing error: {result}")
                    # Add empty profile as placeholder
                    results.append(DocumentScentProfile(
                        document_id="error",
                        signals=[],
                        composite_scent=np.zeros(len(OlfactoryReceptorType) * 2),
                        similarity_hash=""
                    ))
        
        return results
    
    def shutdown(self):
        """Shutdown optimization engine and cleanup resources."""
        self.logger.info("Shutting down OptimizedBioneuroEngine")
        
        # Shutdown thread pools
        self.thread_pool.shutdown(wait=True)
        if hasattr(self, 'process_pool'):
            self.process_pool.shutdown(wait=True)
        
        # Clear caches
        self.scent_cache.clear()


# Global optimized engine instance
_optimized_engine: Optional[OptimizedBioneuroEngine] = None

def get_optimized_engine(config: OptimizationConfig = None) -> OptimizedBioneuroEngine:
    """Get or create global optimized bioneural engine."""
    global _optimized_engine
    if _optimized_engine is None:
        _optimized_engine = OptimizedBioneuroEngine(config)
    return _optimized_engine


async def analyze_document_optimized(document_text: str, document_id: str, 
                                   metadata: Optional[Dict[str, Any]] = None,
                                   config: OptimizationConfig = None) -> DocumentScentProfile:
    """Convenience function for optimized document analysis."""
    engine = get_optimized_engine(config)
    return await engine.analyze_document_optimized(document_text, document_id, metadata)


def create_performance_benchmark(documents: List[Tuple[str, str, Dict[str, Any]]], 
                               configurations: List[OptimizationConfig]) -> Dict[str, Any]:
    """Create performance benchmark comparing different optimization configurations."""
    results = {}
    
    for i, config in enumerate(configurations):
        config_name = f"config_{i}_{config.processing_mode.value}_{config.cache_strategy.value}"
        
        # Create engine with this configuration
        engine = OptimizedBioneuroEngine(config)
        
        # Benchmark processing
        start_time = time.time()
        
        # Process documents (simplified for sync benchmark)
        profiles = []
        for text, doc_id, metadata in documents[:10]:  # Sample subset
            try:
                # Convert async call to sync for benchmark
                profile = asyncio.run(engine.analyze_document_optimized(text, doc_id, metadata))
                profiles.append(profile)
            except Exception as e:
                print(f"Benchmark error: {e}")
        
        end_time = time.time()
        
        # Collect metrics
        results[config_name] = {
            "total_time": end_time - start_time,
            "documents_processed": len(profiles),
            "avg_time_per_doc": (end_time - start_time) / max(1, len(profiles)),
            "optimization_stats": engine.get_optimization_stats()
        }
        
        # Cleanup
        engine.shutdown()
    
    return results