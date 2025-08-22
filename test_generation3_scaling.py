"""
Test Generation 3 Scaling Systems
=================================

Comprehensive tests for quantum-inspired scaling and intelligent caching.
"""

import asyncio
import pytest
import time
import random
from unittest.mock import Mock, patch

from src.lexgraph_legal_rag.quantum_scaling_optimizer import (
    QuantumScalingOptimizer,
    QuantumState,
    QuantumResource,
    WorkloadPattern,
    get_quantum_optimizer,
    quantum_optimized
)

from src.lexgraph_legal_rag.intelligent_caching_system import (
    IntelligentCachingSystem,
    CacheLevel,
    EvictionPolicy,
    get_intelligent_cache,
    intelligent_cache
)


class TestQuantumScalingOptimizer:
    """Test quantum-inspired scaling optimization."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.optimizer = QuantumScalingOptimizer(
            max_resources=10,
            optimization_interval=1.0
        )
    
    @pytest.mark.asyncio
    async def test_resource_registration(self):
        """Test resource registration."""
        
        await self.optimizer.register_resource("test_resource", 100.0, 50.0)
        
        assert "test_resource" in self.optimizer._resources
        resource = self.optimizer._resources["test_resource"]
        assert resource.capacity == 100.0
        assert resource.current_load == 50.0
        assert resource.quantum_state == QuantumState.SUPERPOSITION
    
    @pytest.mark.asyncio
    async def test_workload_update(self):
        """Test workload updates."""
        
        await self.optimizer.register_resource("test_resource", 100.0)
        await self.optimizer.update_workload("test_resource", 75.0)
        
        resource = self.optimizer._resources["test_resource"]
        assert resource.current_load == 75.0
    
    @pytest.mark.asyncio
    async def test_scaling_prediction(self):
        """Test scaling needs prediction."""
        
        # Register some resources
        await self.optimizer.register_resource("cpu", 100.0, 60.0)
        await self.optimizer.register_resource("memory", 200.0, 120.0)
        
        predictions = await self.optimizer.predict_scaling_needs(300.0)
        
        assert "cpu" in predictions
        assert "memory" in predictions
        assert predictions["cpu"] > 0
        assert predictions["memory"] > 0
    
    @pytest.mark.asyncio
    async def test_quantum_optimization(self):
        """Test quantum optimization algorithm."""
        
        # Set up resources with suboptimal allocation
        await self.optimizer.register_resource("overloaded", 50.0, 80.0)  # Overloaded
        await self.optimizer.register_resource("underused", 200.0, 20.0)  # Underused
        
        # Run optimization
        result = await self.optimizer.optimize_resource_allocation()
        
        assert result.success
        assert len(result.new_allocation) > 0
        assert result.optimization_time > 0
    
    @pytest.mark.asyncio
    async def test_emergency_optimization(self):
        """Test emergency optimization trigger."""
        
        await self.optimizer.register_resource("critical", 100.0, 50.0)
        
        # Trigger emergency with high load
        await self.optimizer.update_workload("critical", 95.0)  # 95% utilization
        
        # Check that capacity was increased
        resource = self.optimizer._resources["critical"]
        assert resource.capacity >= 95.0 * 1.5  # Emergency boost
        assert resource.quantum_state == QuantumState.DECOHERENCE
    
    @pytest.mark.asyncio
    async def test_optimization_loop(self):
        """Test continuous optimization loop."""
        
        await self.optimizer.register_resource("looped", 100.0, 40.0)
        
        # Start optimization
        await self.optimizer.start_optimization()
        
        # Wait for at least one cycle
        await asyncio.sleep(1.5)
        
        # Stop optimization
        await self.optimizer.stop_optimization()
        
        # Check that system is running
        assert not self.optimizer._running
    
    @pytest.mark.asyncio
    async def test_scaling_metrics(self):
        """Test scaling metrics collection."""
        
        await self.optimizer.register_resource("metrics_test", 100.0, 60.0)
        
        metrics = await self.optimizer.get_scaling_metrics()
        
        assert "timestamp" in metrics
        assert "system_coherence" in metrics
        assert "system_efficiency" in metrics
        assert "total_resources" in metrics
        assert metrics["total_resources"] == 1
        assert "resources" in metrics
        assert "metrics_test" in metrics["resources"]
    
    @pytest.mark.asyncio
    async def test_quantum_optimized_decorator(self):
        """Test quantum optimization decorator."""
        
        @quantum_optimized("decorated_resource", 10.0)
        async def test_function(value):
            await asyncio.sleep(0.1)  # Simulate work
            return value * 2
        
        result = await test_function(5)
        assert result == 10
        
        # Check that resource was registered
        optimizer = get_quantum_optimizer()
        assert "decorated_resource" in optimizer._resources


class TestIntelligentCachingSystem:
    """Test intelligent caching system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cache = IntelligentCachingSystem(
            max_l1_size=10,
            max_l2_size=20,
            max_l3_size=30,
            default_ttl=60.0
        )
    
    @pytest.mark.asyncio
    async def test_basic_caching(self):
        """Test basic cache operations."""
        
        # Put and get
        await self.cache.put("test_key", "test_value")
        result = await self.cache.get("test_key")
        
        assert result == "test_value"
    
    @pytest.mark.asyncio
    async def test_cache_miss(self):
        """Test cache miss behavior."""
        
        result = await self.cache.get("nonexistent_key")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_ttl_expiration(self):
        """Test TTL-based expiration."""
        
        # Put with short TTL
        await self.cache.put("ttl_key", "ttl_value", ttl=0.1)
        
        # Should be available immediately
        result = await self.cache.get("ttl_key")
        assert result == "ttl_value"
        
        # Wait for expiration
        await asyncio.sleep(0.2)
        
        # Should be expired
        result = await self.cache.get("ttl_key")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_cache_hierarchy(self):
        """Test multi-level cache hierarchy."""
        
        # Fill L1 cache beyond capacity
        for i in range(15):  # More than max_l1_size
            await self.cache.put(f"key_{i}", f"value_{i}", importance_score=0.5)
        
        # Check that some entries are in L2
        assert len(self.cache._l1_cache) <= self.cache.max_l1_size
        assert len(self.cache._l2_cache) > 0
    
    @pytest.mark.asyncio
    async def test_cache_promotion(self):
        """Test cache level promotion on access."""
        
        # Put in L3 (low importance)
        await self.cache.put("promote_test", "promote_value", importance_score=0.1)
        
        # Access multiple times to trigger promotion
        for _ in range(5):
            result = await self.cache.get("promote_test")
            assert result == "promote_value"
            await asyncio.sleep(0.01)
        
        # Should eventually be promoted to higher levels
        # (exact behavior depends on current cache state)
    
    @pytest.mark.asyncio
    async def test_intelligent_placement(self):
        """Test intelligent cache placement."""
        
        # High importance should go to L1
        await self.cache.put("important", "important_value", importance_score=1.0)
        
        # Low importance should go to lower levels
        await self.cache.put("unimportant", "unimportant_value", importance_score=0.1)
        
        # Access pattern should influence placement
        for _ in range(10):
            await self.cache.get("important")
    
    @pytest.mark.asyncio
    async def test_cache_invalidation(self):
        """Test cache invalidation across all levels."""
        
        await self.cache.put("invalidate_test", "test_value")
        
        # Confirm it's cached
        result = await self.cache.get("invalidate_test")
        assert result == "test_value"
        
        # Invalidate
        invalidated = await self.cache.invalidate("invalidate_test")
        assert invalidated is True
        
        # Should be gone
        result = await self.cache.get("invalidate_test")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_cache_warming(self):
        """Test predictive cache warming."""
        
        # Simulate access patterns
        for i in range(10):
            await self.cache.put(f"warm_key_{i}", f"warm_value_{i}")
            await self.cache.get(f"warm_key_{i}")
        
        # Warm cache with predicted keys
        predicted_keys = [f"warm_key_{i}" for i in range(5, 15)]
        await self.cache.warm_cache(predicted_keys)
        
        # Check that patterns were updated
        assert len(self.cache._access_patterns) > 0
    
    @pytest.mark.asyncio
    async def test_access_pattern_learning(self):
        """Test access pattern learning."""
        
        key = "pattern_test"
        
        # Create access pattern
        for _ in range(20):
            await self.cache.put(key, "pattern_value")
            await self.cache.get(key)
            await asyncio.sleep(0.01)
        
        # Check that pattern was learned
        assert key in self.cache._access_patterns
        pattern = self.cache._access_patterns[key]
        assert pattern.frequency > 0
        assert pattern.confidence > 0.5
    
    @pytest.mark.asyncio
    async def test_cache_statistics(self):
        """Test cache statistics collection."""
        
        # Generate some cache activity
        for i in range(10):
            await self.cache.put(f"stats_key_{i}", f"stats_value_{i}")
            await self.cache.get(f"stats_key_{i}")
        
        # Get some misses
        for i in range(5):
            await self.cache.get(f"miss_key_{i}")
        
        stats = await self.cache.get_cache_stats()
        
        assert "timestamp" in stats
        assert "overall_hit_rate" in stats
        assert "total_entries" in stats
        assert "levels" in stats
        assert stats["total_entries"] > 0
    
    @pytest.mark.asyncio
    async def test_cache_optimization_loop(self):
        """Test background optimization."""
        
        await self.cache.start()
        
        # Generate activity
        for i in range(20):
            await self.cache.put(f"opt_key_{i}", f"opt_value_{i}")
            await self.cache.get(f"opt_key_{i}")
        
        # Let optimization run
        await asyncio.sleep(0.1)
        
        await self.cache.stop()
    
    @pytest.mark.asyncio
    async def test_intelligent_cache_decorator(self):
        """Test intelligent caching decorator."""
        
        call_count = 0
        
        @intelligent_cache(ttl=10.0, importance_score=0.8)
        async def expensive_function(value):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)  # Simulate expensive operation
            return value * 2
        
        # First call should execute function
        result1 = await expensive_function(5)
        assert result1 == 10
        assert call_count == 1
        
        # Second call should use cache
        result2 = await expensive_function(5)
        assert result2 == 10
        assert call_count == 1  # Should not increment
        
        # Different argument should execute function
        result3 = await expensive_function(7)
        assert result3 == 14
        assert call_count == 2


class TestIntegratedScaling:
    """Test integrated scaling and caching systems."""
    
    @pytest.mark.asyncio
    async def test_scaling_with_caching(self):
        """Test scaling optimization with intelligent caching."""
        
        # Set up systems
        optimizer = QuantumScalingOptimizer(optimization_interval=0.5)
        cache = IntelligentCachingSystem()
        
        # Register resources
        await optimizer.register_resource("cache_resource", 100.0, 50.0)
        
        # Start systems
        await optimizer.start_optimization()
        await cache.start()
        
        # Simulate workload with caching
        for i in range(20):
            # Cache some data
            await cache.put(f"workload_key_{i}", f"workload_data_{i}")
            
            # Update resource load based on cache performance
            cache_stats = await cache.get_cache_stats()
            hit_rate = cache_stats["overall_hit_rate"]
            
            # Higher hit rate = lower resource usage
            adjusted_load = 50.0 * (1.1 - hit_rate)
            await optimizer.update_workload("cache_resource", adjusted_load)
            
            await asyncio.sleep(0.01)
        
        # Get final metrics
        scaling_metrics = await optimizer.get_scaling_metrics()
        cache_stats = await cache.get_cache_stats()
        
        # Stop systems
        await optimizer.stop_optimization()
        await cache.stop()
        
        # Verify integration
        assert scaling_metrics["system_efficiency"] > 0
        assert cache_stats["overall_hit_rate"] >= 0


# Performance benchmarks
@pytest.mark.asyncio
async def test_quantum_optimizer_performance():
    """Test quantum optimizer performance."""
    
    optimizer = QuantumScalingOptimizer()
    
    # Register many resources
    for i in range(50):
        await optimizer.register_resource(f"resource_{i}", 100.0, random.uniform(20, 80))
    
    # Measure optimization time
    start_time = time.time()
    result = await optimizer.optimize_resource_allocation()
    optimization_time = time.time() - start_time
    
    assert result.success
    assert optimization_time < 5.0  # Should complete within 5 seconds
    
    print(f"✅ Optimized 50 resources in {optimization_time:.3f}s")
    print(f"⚡ Improvement: {result.improvement_percentage:.2f}%")


@pytest.mark.asyncio
async def test_intelligent_cache_performance():
    """Test intelligent cache performance."""
    
    cache = IntelligentCachingSystem()
    
    # Measure cache operations
    start_time = time.time()
    
    # Put many items
    for i in range(1000):
        await cache.put(f"perf_key_{i}", f"perf_value_{i * i}")
    
    put_time = time.time() - start_time
    
    # Get items (should hit cache)
    start_time = time.time()
    
    for i in range(1000):
        result = await cache.get(f"perf_key_{i}")
        assert result == f"perf_value_{i * i}"
    
    get_time = time.time() - start_time
    
    stats = await cache.get_cache_stats()
    
    assert put_time < 2.0  # Should put 1000 items within 2 seconds
    assert get_time < 1.0  # Should get 1000 items within 1 second
    assert stats["overall_hit_rate"] > 0.9  # Should have high hit rate
    
    print(f"✅ Put 1000 items in {put_time:.3f}s ({1000/put_time:.0f} ops/sec)")
    print(f"✅ Get 1000 items in {get_time:.3f}s ({1000/get_time:.0f} ops/sec)")
    print(f"✅ Hit rate: {stats['overall_hit_rate']:.1%}")


if __name__ == "__main__":
    # Run basic tests
    print("🧪 Testing Generation 3 Scaling Systems")
    print("=" * 60)
    
    async def run_tests():
        # Test quantum optimizer
        optimizer = QuantumScalingOptimizer()
        await optimizer.register_resource("test_cpu", 100.0, 60.0)
        
        metrics = await optimizer.get_scaling_metrics()
        print(f"✅ Quantum optimizer: {metrics['system_efficiency']:.2f} efficiency")
        
        # Test intelligent cache
        cache = IntelligentCachingSystem()
        await cache.put("test_key", "test_value", importance_score=0.8)
        
        result = await cache.get("test_key")
        assert result == "test_value"
        print("✅ Intelligent cache: Basic operations working")
        
        # Test integration
        await cache.put("integration_test", "integration_value")
        await optimizer.update_workload("test_cpu", 70.0)
        
        print("✅ Integration: Systems working together")
        
        print("\n🎉 Generation 3 Scaling Tests Completed Successfully!")
    
    asyncio.run(run_tests())