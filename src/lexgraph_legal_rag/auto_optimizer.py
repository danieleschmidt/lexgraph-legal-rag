"""Autonomous optimization system for legal RAG performance.

This module provides intelligent, self-improving optimization that learns from
query patterns and automatically adjusts system parameters for better performance.
"""

from __future__ import annotations

import time
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import logging
import asyncio
import threading
from collections import deque, defaultdict
import json
import math

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Available optimization strategies."""
    AGGRESSIVE = "aggressive"  # Fast optimization, higher risk
    CONSERVATIVE = "conservative"  # Slow optimization, lower risk
    BALANCED = "balanced"  # Default balanced approach
    ADAPTIVE = "adaptive"  # Adapts based on system performance


@dataclass
class PerformanceMetric:
    """Performance metric with statistical tracking."""
    name: str
    value: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def age_seconds(self) -> float:
        """Get age of metric in seconds."""
        return time.time() - self.timestamp


@dataclass 
class OptimizationAction:
    """Represents an optimization action taken by the system."""
    parameter: str
    old_value: Any
    new_value: Any
    timestamp: float
    reason: str
    expected_impact: str
    actual_impact: Optional[float] = None
    
    def evaluate_success(self, performance_improvement: float) -> bool:
        """Evaluate if the optimization was successful."""
        self.actual_impact = performance_improvement
        return performance_improvement > 0.05  # 5% improvement threshold


class PerformanceMonitor:
    """Monitors system performance metrics."""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=history_size))
        self.thresholds: Dict[str, Dict[str, float]] = {
            'response_time': {'warning': 2.0, 'critical': 5.0},
            'cache_hit_rate': {'warning': 0.6, 'critical': 0.4},
            'error_rate': {'warning': 0.05, 'critical': 0.1},
            'memory_usage': {'warning': 0.8, 'critical': 0.9},
            'similarity_score': {'warning': 0.6, 'critical': 0.4}
        }
        self._lock = threading.RLock()
    
    def record_metric(self, name: str, value: float, metadata: Dict[str, Any] = None) -> None:
        """Record a performance metric."""
        with self._lock:
            metric = PerformanceMetric(
                name=name,
                value=value, 
                timestamp=time.time(),
                metadata=metadata or {}
            )
            self.metrics[name].append(metric)
    
    def get_recent_average(self, metric_name: str, window_seconds: int = 300) -> Optional[float]:
        """Get average value for metric over recent time window."""
        with self._lock:
            if metric_name not in self.metrics:
                return None
            
            cutoff_time = time.time() - window_seconds
            recent_values = [
                m.value for m in self.metrics[metric_name] 
                if m.timestamp >= cutoff_time
            ]
            
            return statistics.mean(recent_values) if recent_values else None
    
    def detect_performance_issues(self) -> List[Dict[str, Any]]:
        """Detect performance issues based on thresholds."""
        issues = []
        
        with self._lock:
            for metric_name, thresholds in self.thresholds.items():
                if metric_name not in self.metrics or not self.metrics[metric_name]:
                    continue
                
                recent_avg = self.get_recent_average(metric_name)
                if recent_avg is None:
                    continue
                
                # Check for issues based on metric type
                if metric_name in ['response_time', 'error_rate', 'memory_usage']:
                    # Higher is worse
                    if recent_avg >= thresholds['critical']:
                        severity = 'critical'
                    elif recent_avg >= thresholds['warning']:
                        severity = 'warning'
                    else:
                        continue
                else:
                    # Lower is worse (cache_hit_rate, similarity_score)
                    if recent_avg <= thresholds['critical']:
                        severity = 'critical'
                    elif recent_avg <= thresholds['warning']:
                        severity = 'warning'
                    else:
                        continue
                
                issues.append({
                    'metric': metric_name,
                    'value': recent_avg,
                    'threshold': thresholds[severity],
                    'severity': severity,
                    'timestamp': time.time()
                })
        
        return issues
    
    def get_performance_trends(self) -> Dict[str, str]:
        """Analyze performance trends over time."""
        trends = {}
        
        with self._lock:
            for metric_name in self.metrics:
                if len(self.metrics[metric_name]) < 10:
                    trends[metric_name] = 'insufficient_data'
                    continue
                
                # Compare recent vs older performance
                recent_avg = self.get_recent_average(metric_name, 300)  # 5 minutes
                older_avg = self.get_recent_average(metric_name, 1800)  # 30 minutes
                
                if recent_avg is None or older_avg is None:
                    trends[metric_name] = 'insufficient_data'
                    continue
                
                change_pct = (recent_avg - older_avg) / older_avg * 100
                
                if abs(change_pct) < 2:
                    trends[metric_name] = 'stable'
                elif change_pct > 10:
                    trends[metric_name] = 'degrading' if metric_name in ['response_time', 'error_rate'] else 'improving'
                elif change_pct < -10:
                    trends[metric_name] = 'improving' if metric_name in ['response_time', 'error_rate'] else 'degrading'
                else:
                    trends[metric_name] = 'stable'
        
        return trends


class AdaptiveOptimizer:
    """Self-improving optimization engine."""
    
    def __init__(self, strategy: OptimizationStrategy = OptimizationStrategy.BALANCED):
        self.strategy = strategy
        self.performance_monitor = PerformanceMonitor()
        self.optimization_history: List[OptimizationAction] = []
        self.current_parameters = self._get_default_parameters()
        self.last_optimization = 0
        self._lock = threading.RLock()
        
        # Strategy-specific settings
        self.strategy_settings = {
            OptimizationStrategy.AGGRESSIVE: {
                'optimization_interval': 60,  # seconds
                'parameter_change_rate': 0.3,  # 30% changes
                'min_data_points': 10
            },
            OptimizationStrategy.CONSERVATIVE: {
                'optimization_interval': 600,  # 10 minutes
                'parameter_change_rate': 0.1,  # 10% changes
                'min_data_points': 50
            },
            OptimizationStrategy.BALANCED: {
                'optimization_interval': 300,  # 5 minutes
                'parameter_change_rate': 0.2,  # 20% changes
                'min_data_points': 25
            },
            OptimizationStrategy.ADAPTIVE: {
                'optimization_interval': 180,  # 3 minutes
                'parameter_change_rate': 0.15,  # 15% changes
                'min_data_points': 20
            }
        }
        
        logger.info(f"Initialized adaptive optimizer with strategy: {strategy.value}")
    
    def _get_default_parameters(self) -> Dict[str, Any]:
        """Get default system parameters."""
        return {
            'cache_size': 1000,
            'cache_ttl': 3600,
            'similarity_threshold': 0.7,
            'top_k_results': 3,
            'max_query_length': 500,
            'batch_size': 10,
            'timeout_seconds': 30,
            'retry_attempts': 3,
            'circuit_breaker_threshold': 0.5,
            'rate_limit': 60
        }
    
    async def record_query_performance(self, query: str, response_time: float, 
                                     success: bool, metadata: Dict[str, Any] = None) -> None:
        """Record query performance for optimization."""
        self.performance_monitor.record_metric('response_time', response_time, metadata)
        self.performance_monitor.record_metric('error_rate', 0.0 if success else 1.0, metadata)
        
        # Record cache metrics if available
        if metadata:
            if 'cache_hit_type' in metadata:
                hit = 1.0 if metadata['cache_hit_type'] != 'miss' else 0.0
                self.performance_monitor.record_metric('cache_hit_rate', hit)
            
            if 'similarity_score' in metadata:
                self.performance_monitor.record_metric('similarity_score', metadata['similarity_score'])
            
            if 'memory_usage_pct' in metadata:
                self.performance_monitor.record_metric('memory_usage', metadata['memory_usage_pct'] / 100)
        
        # Trigger optimization if needed
        await self._maybe_optimize()
    
    async def _maybe_optimize(self) -> None:
        """Check if optimization should be triggered."""
        current_time = time.time()
        settings = self.strategy_settings[self.strategy]
        
        # Check if enough time has passed
        if current_time - self.last_optimization < settings['optimization_interval']:
            return
        
        # Check if we have enough data
        response_time_data = self.performance_monitor.metrics.get('response_time', deque())
        if len(response_time_data) < settings['min_data_points']:
            return
        
        await self.optimize()
    
    async def optimize(self) -> List[OptimizationAction]:
        """Perform autonomous optimization."""
        with self._lock:
            logger.info("Starting autonomous optimization cycle")
            
            # Detect performance issues
            issues = self.performance_monitor.detect_performance_issues()
            trends = self.performance_monitor.get_performance_trends()
            
            actions = []
            
            # Address performance issues
            for issue in issues:
                action = await self._address_performance_issue(issue, trends)
                if action:
                    actions.append(action)
                    self.optimization_history.append(action)
            
            # Proactive optimizations based on trends
            proactive_actions = await self._proactive_optimizations(trends)
            actions.extend(proactive_actions)
            
            self.last_optimization = time.time()
            
            logger.info(f"Optimization cycle complete. Applied {len(actions)} optimizations")
            return actions
    
    async def _address_performance_issue(self, issue: Dict[str, Any], 
                                       trends: Dict[str, str]) -> Optional[OptimizationAction]:
        """Address a specific performance issue."""
        metric = issue['metric']
        severity = issue['severity']
        value = issue['value']
        
        settings = self.strategy_settings[self.strategy]
        change_rate = settings['parameter_change_rate']
        
        # Adjust change rate based on severity
        if severity == 'critical':
            change_rate *= 2
        
        if metric == 'response_time':
            # Reduce cache TTL to get fresher data, increase top_k for better results
            if value > 3.0:  # Very slow responses
                new_ttl = max(600, int(self.current_parameters['cache_ttl'] * (1 - change_rate)))
                action = OptimizationAction(
                    parameter='cache_ttl',
                    old_value=self.current_parameters['cache_ttl'],
                    new_value=new_ttl,
                    timestamp=time.time(),
                    reason=f'Slow response time: {value:.2f}s',
                    expected_impact='Reduce cache staleness, improve relevance'
                )
                self.current_parameters['cache_ttl'] = new_ttl
                return action
        
        elif metric == 'cache_hit_rate':
            # Increase cache size and lower similarity threshold
            if value < 0.5:  # Very low hit rate
                new_size = min(5000, int(self.current_parameters['cache_size'] * (1 + change_rate)))
                new_threshold = max(0.5, self.current_parameters['similarity_threshold'] - 0.1)
                
                self.current_parameters['cache_size'] = new_size
                self.current_parameters['similarity_threshold'] = new_threshold
                
                return OptimizationAction(
                    parameter='cache_config',
                    old_value={'size': self.current_parameters['cache_size'], 
                              'threshold': self.current_parameters['similarity_threshold'] + 0.1},
                    new_value={'size': new_size, 'threshold': new_threshold},
                    timestamp=time.time(),
                    reason=f'Low cache hit rate: {value:.2%}',
                    expected_impact='Increase cache capacity and sensitivity'
                )
        
        elif metric == 'error_rate':
            # Increase timeout and retry attempts
            if value > 0.05:  # >5% error rate
                new_timeout = min(60, int(self.current_parameters['timeout_seconds'] * (1 + change_rate)))
                new_retries = min(5, self.current_parameters['retry_attempts'] + 1)
                
                self.current_parameters['timeout_seconds'] = new_timeout
                self.current_parameters['retry_attempts'] = new_retries
                
                return OptimizationAction(
                    parameter='reliability_config',
                    old_value={'timeout': self.current_parameters['timeout_seconds'] - int(self.current_parameters['timeout_seconds'] * change_rate),
                              'retries': new_retries - 1},
                    new_value={'timeout': new_timeout, 'retries': new_retries},
                    timestamp=time.time(),
                    reason=f'High error rate: {value:.2%}',
                    expected_impact='Increase fault tolerance'
                )
        
        return None
    
    async def _proactive_optimizations(self, trends: Dict[str, str]) -> List[OptimizationAction]:
        """Apply proactive optimizations based on trends."""
        actions = []
        
        # If response time is improving but cache hit rate is stable,
        # we can be more aggressive with cache parameters
        if (trends.get('response_time') == 'improving' and 
            trends.get('cache_hit_rate') == 'stable'):
            
            # Slightly increase similarity threshold for better precision
            current_threshold = self.current_parameters['similarity_threshold']
            if current_threshold < 0.8:
                new_threshold = min(0.85, current_threshold + 0.05)
                self.current_parameters['similarity_threshold'] = new_threshold
                
                actions.append(OptimizationAction(
                    parameter='similarity_threshold',
                    old_value=current_threshold,
                    new_value=new_threshold,
                    timestamp=time.time(),
                    reason='Response time improving, optimizing for precision',
                    expected_impact='Better result quality with maintained performance'
                ))
        
        # If cache hit rate is improving, we can increase TTL slightly
        if trends.get('cache_hit_rate') == 'improving':
            current_ttl = self.current_parameters['cache_ttl']
            if current_ttl < 7200:  # Less than 2 hours
                new_ttl = min(7200, int(current_ttl * 1.1))
                self.current_parameters['cache_ttl'] = new_ttl
                
                actions.append(OptimizationAction(
                    parameter='cache_ttl',
                    old_value=current_ttl,
                    new_value=new_ttl,
                    timestamp=time.time(),
                    reason='Cache performance improving, extending TTL',
                    expected_impact='Reduced computational load'
                ))
        
        return actions
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report."""
        with self._lock:
            recent_actions = [
                action for action in self.optimization_history
                if time.time() - action.timestamp < 3600  # Last hour
            ]
            
            successful_optimizations = sum(
                1 for action in recent_actions 
                if action.actual_impact and action.actual_impact > 0
            )
            
            performance_issues = self.performance_monitor.detect_performance_issues()
            trends = self.performance_monitor.get_performance_trends()
            
            return {
                'strategy': self.strategy.value,
                'current_parameters': self.current_parameters.copy(),
                'recent_optimizations': len(recent_actions),
                'successful_optimizations': successful_optimizations,
                'success_rate': successful_optimizations / len(recent_actions) if recent_actions else 0,
                'active_issues': len(performance_issues),
                'performance_trends': trends,
                'last_optimization': self.last_optimization,
                'next_optimization_due': self.last_optimization + self.strategy_settings[self.strategy]['optimization_interval']
            }
    
    def get_parameter(self, name: str) -> Any:
        """Get current value of optimization parameter."""
        return self.current_parameters.get(name)
    
    def force_optimization(self) -> None:
        """Force an optimization cycle."""
        self.last_optimization = 0  # Reset timer
    
    async def evaluate_optimization_impact(self, action: OptimizationAction, 
                                         duration_seconds: int = 300) -> float:
        """Evaluate the impact of an optimization action."""
        # This would typically compare performance before and after the optimization
        # For now, we'll simulate the evaluation
        
        await asyncio.sleep(1)  # Simulate evaluation time
        
        # Simple heuristic: assume optimization was successful if no new issues
        current_issues = self.performance_monitor.detect_performance_issues()
        impact_score = max(0, 0.1 - len(current_issues) * 0.02)
        
        action.evaluate_success(impact_score)
        return impact_score


# Global optimizer instance
_global_optimizer = None

def get_auto_optimizer() -> AdaptiveOptimizer:
    """Get global optimizer instance."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = AdaptiveOptimizer()
    return _global_optimizer


async def record_performance(query: str, response_time: float, success: bool, **metadata) -> None:
    """Convenience function to record performance data."""
    optimizer = get_auto_optimizer()
    await optimizer.record_query_performance(query, response_time, success, metadata)


def get_optimal_parameter(name: str, default: Any = None) -> Any:
    """Get optimally tuned parameter value."""
    optimizer = get_auto_optimizer()
    return optimizer.get_parameter(name) or default


async def force_optimization() -> List[OptimizationAction]:
    """Force an optimization cycle and return actions taken."""
    optimizer = get_auto_optimizer()
    optimizer.force_optimization()
    return await optimizer.optimize()