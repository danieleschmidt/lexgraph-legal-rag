"""
Adaptive Performance Optimization Engine
Self-learning system that continuously optimizes performance based on usage patterns
"""

import json
import logging
import threading
import time
from collections import defaultdict
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import psutil


logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Represents a performance metric measurement."""

    name: str
    value: float
    timestamp: str
    context: Dict[str, Any]
    threshold: Optional[float] = None


@dataclass
class OptimizationRule:
    """Represents an adaptive optimization rule."""

    id: str
    condition: str
    action: str
    priority: int
    success_rate: float
    last_applied: Optional[str] = None
    application_count: int = 0
    impact_score: float = 0.0


@dataclass
class SystemProfile:
    """System performance profile with usage patterns."""

    cpu_pattern: List[float]
    memory_pattern: List[float]
    query_patterns: Dict[str, int]
    peak_hours: List[int]
    resource_bottlenecks: List[str]
    optimization_opportunities: List[str]


class AdaptiveCache:
    """Self-optimizing cache with usage pattern learning."""

    def __init__(self, initial_size: int = 1000):
        self.cache: Dict[str, Any] = {}
        self.access_counts = defaultdict(int)
        self.access_times = defaultdict(float)
        self.max_size = initial_size
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_strategy = "lru"

        # Adaptive parameters
        self.access_history = deque(maxlen=10000)
        self.size_adjustment_threshold = 0.9
        self.last_optimization = time.time()

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with adaptive learning."""
        if key in self.cache:
            self.hit_count += 1
            self.access_counts[key] += 1
            self.access_times[key] = time.time()
            self.access_history.append(("hit", key, time.time()))
            return self.cache[key]
        else:
            self.miss_count += 1
            self.access_history.append(("miss", key, time.time()))
            return None

    def put(self, key: str, value: Any) -> None:
        """Put item in cache with adaptive eviction."""
        if len(self.cache) >= self.max_size:
            self._adaptive_eviction()

        self.cache[key] = value
        self.access_counts[key] = 1
        self.access_times[key] = time.time()

        # Trigger optimization check
        self._check_optimization_trigger()

    def _adaptive_eviction(self) -> None:
        """Perform adaptive cache eviction based on learned patterns."""
        if not self.cache:
            return

        # Analyze access patterns to choose eviction strategy
        if self._should_use_lfu():
            self.eviction_strategy = "lfu"
            victim = min(self.cache.keys(), key=lambda k: self.access_counts[k])
        else:
            self.eviction_strategy = "lru"
            victim = min(self.cache.keys(), key=lambda k: self.access_times[k])

        del self.cache[victim]
        del self.access_counts[victim]
        del self.access_times[victim]

    def _should_use_lfu(self) -> bool:
        """Determine if LFU eviction is better than LRU based on patterns."""
        if len(self.access_history) < 100:
            return False

        # Analyze recent access patterns
        recent_accesses = list(self.access_history)[-100:]
        unique_keys = len({access[1] for access in recent_accesses})

        # If high key reuse, LFU is better
        return unique_keys < len(recent_accesses) * 0.5

    def _check_optimization_trigger(self) -> None:
        """Check if cache should be optimized."""
        if time.time() - self.last_optimization > 300:  # 5 minutes
            self.optimize()
            self.last_optimization = time.time()

    def optimize(self) -> Dict[str, Any]:
        """Optimize cache parameters based on learned patterns."""
        optimization_actions = []

        # Adjust cache size based on hit rate
        hit_rate = (
            self.hit_count / (self.hit_count + self.miss_count)
            if (self.hit_count + self.miss_count) > 0
            else 0
        )

        if hit_rate < 0.7 and self.max_size < 10000:
            new_size = min(self.max_size * 1.2, 10000)
            optimization_actions.append(
                f"Increased cache size: {self.max_size} -> {new_size}"
            )
            self.max_size = int(new_size)
        elif hit_rate > 0.95 and self.max_size > 100:
            new_size = max(self.max_size * 0.9, 100)
            optimization_actions.append(
                f"Decreased cache size: {self.max_size} -> {new_size}"
            )
            self.max_size = int(new_size)

        return {
            "hit_rate": hit_rate,
            "cache_size": self.max_size,
            "eviction_strategy": self.eviction_strategy,
            "optimizations": optimization_actions,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.hit_count + self.miss_count
        return {
            "hit_rate": self.hit_count / total_requests if total_requests > 0 else 0,
            "total_requests": total_requests,
            "cache_size": len(self.cache),
            "max_size": self.max_size,
            "eviction_strategy": self.eviction_strategy,
        }


class QueryOptimizer:
    """Adaptive query optimization based on usage patterns."""

    def __init__(self):
        self.query_patterns = defaultdict(int)
        self.response_times = defaultdict(list)
        self.optimization_cache = {}
        self.pattern_learning_enabled = True

    def analyze_query(self, query: str, response_time: float) -> Dict[str, Any]:
        """Analyze query for optimization opportunities."""
        query_hash = hash(query) % 10000  # Simplified hash
        self.query_patterns[query_hash] += 1
        self.response_times[query_hash].append(response_time)

        # Keep only recent response times
        if len(self.response_times[query_hash]) > 100:
            self.response_times[query_hash] = self.response_times[query_hash][-50:]

        optimization_suggestions = []

        # Check for slow queries
        avg_time = sum(self.response_times[query_hash]) / len(
            self.response_times[query_hash]
        )
        if avg_time > 2.0:  # Slow query threshold
            optimization_suggestions.append("consider_query_caching")
            optimization_suggestions.append("analyze_index_optimization")

        # Check for frequent queries
        if self.query_patterns[query_hash] > 10:
            optimization_suggestions.append("high_frequency_query_caching")

        return {
            "query_hash": query_hash,
            "frequency": self.query_patterns[query_hash],
            "avg_response_time": avg_time,
            "optimizations": optimization_suggestions,
        }

    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get query optimization recommendations."""
        recommendations = []

        # Analyze top frequent queries
        top_queries = sorted(
            self.query_patterns.items(), key=lambda x: x[1], reverse=True
        )[:10]

        for query_hash, frequency in top_queries:
            if query_hash in self.response_times:
                avg_time = sum(self.response_times[query_hash]) / len(
                    self.response_times[query_hash]
                )

                if avg_time > 1.0 or frequency > 50:
                    recommendations.append(
                        {
                            "query_hash": query_hash,
                            "frequency": frequency,
                            "avg_response_time": avg_time,
                            "recommendation": "implement_result_caching",
                            "priority": "high" if avg_time > 2.0 else "medium",
                        }
                    )

        return recommendations


class ResourceMonitor:
    """Monitors system resources and triggers adaptive optimizations."""

    def __init__(self):
        self.metrics_history = deque(maxlen=1000)
        self.monitoring_active = False
        self.monitor_thread = None
        self.alert_thresholds = {
            "cpu_percent": 85.0,
            "memory_percent": 80.0,
            "disk_percent": 90.0,
        }

    def start_monitoring(self) -> None:
        """Start resource monitoring in background thread."""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(
                target=self._monitor_loop, daemon=True
            )
            self.monitor_thread.start()
            logger.info("Resource monitoring started")

    def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Resource monitoring stopped")

    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect system metrics
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)

                # Check for alerts
                alerts = self._check_alerts(metrics)
                if alerts:
                    logger.warning(f"Resource alerts triggered: {alerts}")

                time.sleep(10)  # Monitor every 10 seconds

            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(30)  # Back off on error

    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics."""
        return {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage("/").percent,
            "network_io": (
                dict(psutil.net_io_counters()._asdict())
                if hasattr(psutil, "net_io_counters")
                else {}
            ),
            "process_count": len(psutil.pids()),
        }

    def _check_alerts(self, metrics: Dict[str, Any]) -> List[str]:
        """Check metrics against alert thresholds."""
        alerts = []

        for metric, threshold in self.alert_thresholds.items():
            if metric in metrics and metrics[metric] > threshold:
                alerts.append(f"{metric}: {metrics[metric]:.1f}% > {threshold}%")

        return alerts

    def get_performance_profile(self) -> SystemProfile:
        """Generate system performance profile."""
        if len(self.metrics_history) < 10:
            return SystemProfile([], [], {}, [], [], [])

        metrics_list = list(self.metrics_history)

        # Extract patterns
        cpu_pattern = [m["cpu_percent"] for m in metrics_list]
        memory_pattern = [m["memory_percent"] for m in metrics_list]

        # Analyze peak usage hours (simplified)
        peak_hours = []
        if metrics_list:
            timestamps = [datetime.fromisoformat(m["timestamp"]) for m in metrics_list]
            for ts in timestamps:
                if any(
                    m["cpu_percent"] > 70
                    for m in metrics_list
                    if datetime.fromisoformat(m["timestamp"]).hour == ts.hour
                ):
                    peak_hours.append(ts.hour)

        # Identify bottlenecks
        bottlenecks = []
        avg_cpu = sum(cpu_pattern) / len(cpu_pattern)
        avg_memory = sum(memory_pattern) / len(memory_pattern)

        if avg_cpu > 60:
            bottlenecks.append("high_cpu_usage")
        if avg_memory > 70:
            bottlenecks.append("high_memory_usage")

        # Generate optimization opportunities
        opportunities = []
        if avg_cpu > 50:
            opportunities.append("cpu_optimization")
        if avg_memory > 60:
            opportunities.append("memory_optimization")
        if len(set(peak_hours)) > 4:
            opportunities.append("load_balancing")

        return SystemProfile(
            cpu_pattern=cpu_pattern,
            memory_pattern=memory_pattern,
            query_patterns={},
            peak_hours=list(set(peak_hours)),
            resource_bottlenecks=bottlenecks,
            optimization_opportunities=opportunities,
        )


class AdaptiveOptimizationEngine:
    """Main adaptive optimization engine coordinating all optimizations."""

    def __init__(self, repo_path: str = "/root/repo"):
        self.repo_path = Path(repo_path)
        self.optimization_dir = self.repo_path / "optimization"
        self.optimization_dir.mkdir(exist_ok=True)

        # Components
        self.cache = AdaptiveCache()
        self.query_optimizer = QueryOptimizer()
        self.resource_monitor = ResourceMonitor()

        # Optimization rules
        self.optimization_rules: List[OptimizationRule] = []
        self._initialize_optimization_rules()

        # Performance tracking
        self.performance_metrics = deque(maxlen=1000)
        self.optimization_history = []

    def _initialize_optimization_rules(self) -> None:
        """Initialize adaptive optimization rules."""
        rules = [
            OptimizationRule(
                id="cache_size_increase",
                condition="cache_hit_rate < 0.7 AND memory_available > 50%",
                action="increase_cache_size",
                priority=2,
                success_rate=0.8,
            ),
            OptimizationRule(
                id="query_caching_enable",
                condition="avg_query_time > 2.0 AND query_frequency > 10",
                action="enable_query_result_caching",
                priority=1,
                success_rate=0.9,
            ),
            OptimizationRule(
                id="memory_cleanup",
                condition="memory_usage > 80%",
                action="trigger_garbage_collection",
                priority=3,
                success_rate=0.7,
            ),
            OptimizationRule(
                id="index_optimization",
                condition="slow_queries_count > 5",
                action="analyze_index_optimization",
                priority=2,
                success_rate=0.6,
            ),
        ]

        self.optimization_rules = rules

    def start_adaptive_optimization(self) -> None:
        """Start autonomous adaptive optimization."""
        logger.info("🚀 Starting adaptive optimization engine")

        # Start resource monitoring
        self.resource_monitor.start_monitoring()

        # Start optimization loop in background
        optimization_thread = threading.Thread(
            target=self._optimization_loop, daemon=True
        )
        optimization_thread.start()

        logger.info("✅ Adaptive optimization engine started")

    def stop_adaptive_optimization(self) -> None:
        """Stop adaptive optimization."""
        self.resource_monitor.stop_monitoring()
        logger.info("🛑 Adaptive optimization engine stopped")

    def _optimization_loop(self) -> None:
        """Main optimization loop."""
        while True:
            try:
                # Collect current performance state
                performance_state = self._collect_performance_state()

                # Evaluate optimization rules
                applicable_rules = self._evaluate_optimization_rules(performance_state)

                # Apply optimizations
                for rule in applicable_rules:
                    self._apply_optimization(rule, performance_state)

                # Log performance metrics
                self.performance_metrics.append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "state": performance_state,
                        "optimizations_applied": len(applicable_rules),
                    }
                )

                # Sleep before next optimization cycle
                time.sleep(60)  # Run every minute

            except Exception as e:
                logger.error(f"Optimization loop error: {e}")
                time.sleep(120)  # Back off on error

    def _collect_performance_state(self) -> Dict[str, Any]:
        """Collect current system performance state."""
        cache_stats = self.cache.get_stats()
        system_profile = self.resource_monitor.get_performance_profile()
        query_recommendations = self.query_optimizer.get_optimization_recommendations()

        return {
            "cache_hit_rate": cache_stats["hit_rate"],
            "cache_size": cache_stats["cache_size"],
            "memory_usage": psutil.virtual_memory().percent,
            "cpu_usage": psutil.cpu_percent(interval=1),
            "slow_queries_count": len(
                [r for r in query_recommendations if r.get("priority") == "high"]
            ),
            "avg_query_time": (
                np.mean([r["avg_response_time"] for r in query_recommendations])
                if query_recommendations
                else 0
            ),
            "system_bottlenecks": system_profile.resource_bottlenecks,
            "optimization_opportunities": system_profile.optimization_opportunities,
        }

    def _evaluate_optimization_rules(
        self, state: Dict[str, Any]
    ) -> List[OptimizationRule]:
        """Evaluate which optimization rules should be applied."""
        applicable_rules = []

        for rule in self.optimization_rules:
            if self._evaluate_rule_condition(rule.condition, state):
                # Check if rule was recently applied
                if rule.last_applied:
                    last_applied_time = datetime.fromisoformat(rule.last_applied)
                    if datetime.now() - last_applied_time < timedelta(minutes=30):
                        continue  # Skip recently applied rules

                applicable_rules.append(rule)

        # Sort by priority and success rate
        applicable_rules.sort(key=lambda r: (r.priority, r.success_rate), reverse=True)

        return applicable_rules[:3]  # Apply max 3 optimizations per cycle

    def _evaluate_rule_condition(self, condition: str, state: Dict[str, Any]) -> bool:
        """Evaluate if a rule condition is met."""
        # Simplified condition evaluation - would use proper expression parser in production
        try:
            if (
                "cache_hit_rate < 0.7" in condition
                and state.get("cache_hit_rate", 1.0) < 0.7
            ):
                return True
            if (
                "memory_available > 50%" in condition
                and state.get("memory_usage", 100) < 50
            ):
                return True
            if (
                "avg_query_time > 2.0" in condition
                and state.get("avg_query_time", 0) > 2.0
            ):
                return True
            if "memory_usage > 80%" in condition and state.get("memory_usage", 0) > 80:
                return True
            return bool(
                "slow_queries_count > 5" in condition
                and state.get("slow_queries_count", 0) > 5
            )
        except Exception as e:
            logger.error(f"Error evaluating condition '{condition}': {e}")
            return False

    def _apply_optimization(
        self, rule: OptimizationRule, state: Dict[str, Any]
    ) -> None:
        """Apply an optimization rule."""
        logger.info(f"🔧 Applying optimization: {rule.action}")

        try:
            optimization_result = None

            if rule.action == "increase_cache_size":
                optimization_result = self.cache.optimize()
            elif rule.action == "enable_query_result_caching":
                optimization_result = self._enable_query_caching()
            elif rule.action == "trigger_garbage_collection":
                optimization_result = self._trigger_garbage_collection()
            elif rule.action == "analyze_index_optimization":
                optimization_result = self._analyze_index_optimization()

            # Update rule statistics
            rule.application_count += 1
            rule.last_applied = datetime.now().isoformat()

            # Record optimization history
            self.optimization_history.append(
                {
                    "rule_id": rule.id,
                    "action": rule.action,
                    "timestamp": rule.last_applied,
                    "system_state": state.copy(),
                    "result": optimization_result,
                }
            )

            logger.info(f"✅ Optimization applied successfully: {rule.action}")

        except Exception as e:
            logger.error(f"Failed to apply optimization {rule.action}: {e}")

    def _enable_query_caching(self) -> Dict[str, Any]:
        """Enable enhanced query result caching."""
        # In a real implementation, this would configure query caching
        return {
            "action": "query_caching_enabled",
            "cache_ttl": 300,
            "max_cached_queries": 1000,
        }

    def _trigger_garbage_collection(self) -> Dict[str, Any]:
        """Trigger Python garbage collection."""
        import gc

        collected = gc.collect()
        return {
            "action": "garbage_collection",
            "objects_collected": collected,
            "memory_freed_estimate": collected * 0.001,  # Rough estimate
        }

    def _analyze_index_optimization(self) -> Dict[str, Any]:
        """Analyze and suggest index optimizations."""
        # In a real implementation, this would analyze query patterns and suggest index optimizations
        return {
            "action": "index_analysis",
            "recommendations": [
                "consider_composite_index_on_frequent_filters",
                "analyze_unused_indexes",
                "optimize_vector_index_parameters",
            ],
        }

    def generate_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report."""
        if not self.optimization_history:
            return {"message": "No optimizations have been applied yet"}

        # Calculate optimization impact
        total_optimizations = len(self.optimization_history)
        successful_optimizations = len(
            [o for o in self.optimization_history if o.get("result")]
        )

        # Analyze rule effectiveness
        rule_effectiveness = {}
        for rule in self.optimization_rules:
            if rule.application_count > 0:
                rule_effectiveness[rule.id] = {
                    "applications": rule.application_count,
                    "success_rate": rule.success_rate,
                    "last_applied": rule.last_applied,
                }

        # Performance trends
        recent_metrics = (
            list(self.performance_metrics)[-100:] if self.performance_metrics else []
        )
        performance_trend = "stable"
        if len(recent_metrics) > 10:
            recent_cpu = [m["state"]["cpu_usage"] for m in recent_metrics[-10:]]
            earlier_cpu = [m["state"]["cpu_usage"] for m in recent_metrics[-20:-10]]
            if np.mean(recent_cpu) < np.mean(earlier_cpu) * 0.9:
                performance_trend = "improving"
            elif np.mean(recent_cpu) > np.mean(earlier_cpu) * 1.1:
                performance_trend = "degrading"

        report = {
            "optimization_summary": {
                "total_optimizations": total_optimizations,
                "successful_optimizations": successful_optimizations,
                "success_rate": (
                    successful_optimizations / total_optimizations
                    if total_optimizations > 0
                    else 0
                ),
                "performance_trend": performance_trend,
            },
            "rule_effectiveness": rule_effectiveness,
            "current_performance": self._collect_performance_state(),
            "recommendations": self._generate_optimization_recommendations(),
            "timestamp": datetime.now().isoformat(),
        }

        return report

    def _generate_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on patterns."""
        recommendations = []

        current_state = self._collect_performance_state()

        if current_state["cache_hit_rate"] < 0.8:
            recommendations.append(
                "Consider increasing cache size or improving cache key strategy"
            )

        if current_state["memory_usage"] > 75:
            recommendations.append(
                "High memory usage detected - consider memory optimization"
            )

        if current_state["slow_queries_count"] > 3:
            recommendations.append(
                "Multiple slow queries detected - analyze query optimization opportunities"
            )

        if not recommendations:
            recommendations.append(
                "System performance appears optimal - continue monitoring"
            )

        return recommendations

    def save_optimization_report(self) -> str:
        """Save optimization report to file."""
        report = self.generate_optimization_report()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.optimization_dir / f"optimization_report_{timestamp}.json"

        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"📊 Optimization report saved: {report_file}")
        return str(report_file)


def main():
    """Main entry point for adaptive optimization engine."""
    logging.basicConfig(level=logging.INFO)

    engine = AdaptiveOptimizationEngine()

    try:
        # Start adaptive optimization
        engine.start_adaptive_optimization()

        # Run for demonstration (in production, this would run continuously)
        print("🔧 ADAPTIVE OPTIMIZATION ENGINE STARTED")
        print("📊 Monitoring system performance and applying optimizations...")

        # Let it run for a short time to demonstrate
        time.sleep(30)

        # Generate and save report
        report_file = engine.save_optimization_report()
        print(f"📋 Optimization report saved: {report_file}")

        # Stop optimization
        engine.stop_adaptive_optimization()
        print("✅ ADAPTIVE OPTIMIZATION COMPLETED")

    except KeyboardInterrupt:
        print("\n🛑 Stopping adaptive optimization...")
        engine.stop_adaptive_optimization()


if __name__ == "__main__":
    main()
