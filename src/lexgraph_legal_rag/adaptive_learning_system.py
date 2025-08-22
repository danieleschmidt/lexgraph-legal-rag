"""
Adaptive Learning System with Self-Improvement
==============================================

Self-improving patterns with adaptive learning and autonomous optimization:
- Machine learning model adaptation based on usage patterns
- Autonomous system optimization and tuning
- Feedback-driven improvement mechanisms  
- Predictive performance optimization
- Self-healing and auto-correction capabilities
- Continuous learning from user interactions
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import pickle
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)


class LearningMode(Enum):
    """Learning modes for the adaptive system."""
    
    PASSIVE = "passive"           # Learn from observations only
    ACTIVE = "active"             # Actively seek learning opportunities
    REINFORCEMENT = "reinforcement"  # Learn from rewards/penalties
    SUPERVISED = "supervised"     # Learn from labeled feedback
    UNSUPERVISED = "unsupervised"  # Learn patterns autonomously


class OptimizationObjective(Enum):
    """Optimization objectives for self-improvement."""
    
    PERFORMANCE = "performance"   # Optimize for speed/throughput
    ACCURACY = "accuracy"         # Optimize for result quality
    EFFICIENCY = "efficiency"     # Optimize resource usage
    USER_SATISFACTION = "user_satisfaction"  # Optimize user experience
    BALANCED = "balanced"         # Multi-objective optimization


@dataclass
class LearningEvent:
    """Event captured for learning purposes."""
    
    event_id: str
    timestamp: datetime
    event_type: str
    context: Dict[str, Any]
    outcome: Any
    feedback_score: Optional[float] = None
    learning_value: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetric:
    """Performance metric tracking."""
    
    metric_name: str
    value: float
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)
    trend: Optional[str] = None  # "improving", "degrading", "stable"


@dataclass
class AdaptationRule:
    """Rule for adaptive system behavior."""
    
    rule_id: str
    condition: Callable[[Dict[str, Any]], bool]
    action: Callable[[Dict[str, Any]], Any]
    priority: int = 1
    success_count: int = 0
    failure_count: int = 0
    last_applied: Optional[datetime] = None
    is_active: bool = True


@dataclass
class LearningModel:
    """Simple learning model for pattern recognition."""
    
    model_id: str
    model_type: str
    parameters: Dict[str, float]
    training_data: List[Tuple[Any, Any]]
    accuracy: float = 0.0
    last_trained: Optional[datetime] = None
    prediction_count: int = 0
    correct_predictions: int = 0


class AdaptiveLearningSystem:
    """
    Comprehensive adaptive learning system with self-improvement capabilities.
    
    Features:
    - Continuous learning from system interactions
    - Adaptive parameter tuning and optimization
    - Self-healing and auto-correction mechanisms
    - Predictive performance optimization
    - Feedback-driven improvement loops
    - Autonomous system evolution
    """
    
    def __init__(self, 
                 learning_rate: float = 0.1,
                 optimization_interval: float = 300.0,  # 5 minutes
                 max_history_size: int = 10000):
        
        self.learning_rate = learning_rate
        self.optimization_interval = optimization_interval
        self.max_history_size = max_history_size
        
        # Learning components
        self._learning_events: deque[LearningEvent] = deque(maxlen=max_history_size)
        self._performance_metrics: Dict[str, deque[PerformanceMetric]] = defaultdict(
            lambda: deque(maxlen=1000)
        )
        self._adaptation_rules: Dict[str, AdaptationRule] = {}
        self._learning_models: Dict[str, LearningModel] = {}
        
        # System state
        self._system_parameters: Dict[str, float] = {}
        self._optimization_objectives: List[OptimizationObjective] = [OptimizationObjective.BALANCED]
        self._learning_mode = LearningMode.ACTIVE
        
        # Performance tracking
        self._baseline_metrics: Dict[str, float] = {}
        self._improvement_history: List[Dict[str, Any]] = []
        
        # Background tasks
        self._learning_task: Optional[asyncio.Task] = None
        self._optimization_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Initialize default components
        self._initialize_default_rules()
        self._initialize_learning_models()
    
    async def start_learning(self):
        """Start adaptive learning and optimization."""
        
        self._running = True
        self._learning_task = asyncio.create_task(self._learning_loop())
        self._optimization_task = asyncio.create_task(self._optimization_loop())
        
        logger.info("Adaptive learning system started")
    
    async def stop_learning(self):
        """Stop adaptive learning."""
        
        self._running = False
        
        if self._learning_task:
            self._learning_task.cancel()
        if self._optimization_task:
            self._optimization_task.cancel()
        
        logger.info("Adaptive learning system stopped")
    
    def record_event(self, 
                    event_type: str,
                    context: Dict[str, Any],
                    outcome: Any,
                    feedback_score: Optional[float] = None) -> str:
        """Record a learning event."""
        
        event_id = f"{event_type}_{int(time.time() * 1000)}"
        
        event = LearningEvent(
            event_id=event_id,
            timestamp=datetime.now(),
            event_type=event_type,
            context=context,
            outcome=outcome,
            feedback_score=feedback_score,
            learning_value=self._calculate_learning_value(event_type, context, outcome)
        )
        
        self._learning_events.append(event)
        
        # Trigger immediate learning if significant event
        if feedback_score and abs(feedback_score) > 0.8:
            asyncio.create_task(self._process_immediate_learning(event))
        
        return event_id
    
    def record_performance_metric(self, 
                                 metric_name: str,
                                 value: float,
                                 context: Optional[Dict[str, Any]] = None):
        """Record a performance metric."""
        
        metric = PerformanceMetric(
            metric_name=metric_name,
            value=value,
            timestamp=datetime.now(),
            context=context or {}
        )
        
        # Calculate trend
        metric.trend = self._calculate_metric_trend(metric_name, value)
        
        self._performance_metrics[metric_name].append(metric)
        
        # Update baseline if this is a new metric
        if metric_name not in self._baseline_metrics:
            self._baseline_metrics[metric_name] = value
    
    def add_adaptation_rule(self, 
                           rule_id: str,
                           condition: Callable[[Dict[str, Any]], bool],
                           action: Callable[[Dict[str, Any]], Any],
                           priority: int = 1):
        """Add an adaptation rule."""
        
        rule = AdaptationRule(
            rule_id=rule_id,
            condition=condition,
            action=action,
            priority=priority
        )
        
        self._adaptation_rules[rule_id] = rule
        logger.info(f"Added adaptation rule: {rule_id}")
    
    async def predict_performance(self, 
                                 context: Dict[str, Any],
                                 time_horizon: float = 300.0) -> Dict[str, float]:
        """Predict future performance based on current context."""
        
        predictions = {}
        
        for metric_name, metrics in self._performance_metrics.items():
            if len(metrics) < 5:  # Need minimum data
                continue
            
            # Simple linear regression prediction
            recent_metrics = list(metrics)[-10:]  # Last 10 measurements
            
            # Extract time series data
            times = [(m.timestamp - recent_metrics[0].timestamp).total_seconds() 
                    for m in recent_metrics]
            values = [m.value for m in recent_metrics]
            
            if len(times) >= 3:
                # Simple linear trend calculation
                slope = self._calculate_trend_slope(times, values)
                current_value = values[-1]
                predicted_value = current_value + (slope * time_horizon)
                
                predictions[metric_name] = predicted_value
        
        return predictions
    
    async def optimize_parameters(self, 
                                 objective: OptimizationObjective = OptimizationObjective.BALANCED) -> Dict[str, Any]:
        """Optimize system parameters based on learning."""
        
        logger.info(f"Starting parameter optimization for objective: {objective.value}")
        
        optimization_result = {
            "objective": objective.value,
            "parameters_changed": [],
            "performance_improvement": 0.0,
            "timestamp": datetime.now()
        }
        
        # Collect recent performance data
        recent_performance = self._get_recent_performance()
        
        # Apply learning-based optimization
        if objective == OptimizationObjective.PERFORMANCE:
            await self._optimize_for_performance(recent_performance, optimization_result)
        elif objective == OptimizationObjective.ACCURACY:
            await self._optimize_for_accuracy(recent_performance, optimization_result)
        elif objective == OptimizationObjective.EFFICIENCY:
            await self._optimize_for_efficiency(recent_performance, optimization_result)
        elif objective == OptimizationObjective.BALANCED:
            await self._optimize_balanced(recent_performance, optimization_result)
        
        # Record improvement
        self._improvement_history.append(optimization_result)
        
        return optimization_result
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights from learning data."""
        
        insights = {
            "total_events": len(self._learning_events),
            "learning_rate": self.learning_rate,
            "active_rules": len([r for r in self._adaptation_rules.values() if r.is_active]),
            "performance_trends": {},
            "top_learning_patterns": [],
            "optimization_history": self._improvement_history[-10:],  # Last 10 optimizations
            "model_performance": {}
        }
        
        # Analyze performance trends
        for metric_name, metrics in self._performance_metrics.items():
            if len(metrics) >= 5:
                recent_values = [m.value for m in list(metrics)[-10:]]
                trend = "stable"
                
                if len(recent_values) >= 3:
                    slope = self._calculate_trend_slope(
                        list(range(len(recent_values))), 
                        recent_values
                    )
                    
                    if slope > 0.1:
                        trend = "improving"
                    elif slope < -0.1:
                        trend = "degrading"
                
                insights["performance_trends"][metric_name] = {
                    "current_value": recent_values[-1] if recent_values else 0,
                    "trend": trend,
                    "data_points": len(metrics)
                }
        
        # Analyze learning patterns
        event_types = defaultdict(int)
        for event in self._learning_events:
            event_types[event.event_type] += 1
        
        insights["top_learning_patterns"] = [
            {"event_type": event_type, "count": count}
            for event_type, count in sorted(event_types.items(), key=lambda x: x[1], reverse=True)[:5]
        ]
        
        # Model performance
        for model_id, model in self._learning_models.items():
            if model.prediction_count > 0:
                accuracy = model.correct_predictions / model.prediction_count
                insights["model_performance"][model_id] = {
                    "accuracy": accuracy,
                    "predictions": model.prediction_count,
                    "last_trained": model.last_trained.isoformat() if model.last_trained else None
                }
        
        return insights
    
    async def adapt_to_feedback(self, 
                               event_id: str,
                               feedback_score: float,
                               feedback_details: Optional[Dict[str, Any]] = None):
        """Adapt system based on feedback."""
        
        # Find the event
        event = None
        for e in self._learning_events:
            if e.event_id == event_id:
                event = e
                break
        
        if not event:
            logger.warning(f"Event not found for feedback: {event_id}")
            return
        
        # Update event with feedback
        event.feedback_score = feedback_score
        if feedback_details:
            event.metadata.update(feedback_details)
        
        # Learn from feedback
        await self._learn_from_feedback(event, feedback_score)
        
        logger.info(f"Adapted to feedback for event {event_id}: score={feedback_score}")
    
    def _calculate_learning_value(self, 
                                 event_type: str,
                                 context: Dict[str, Any],
                                 outcome: Any) -> float:
        """Calculate learning value of an event."""
        
        base_value = 1.0
        
        # Higher value for rare events
        event_count = sum(1 for e in self._learning_events if e.event_type == event_type)
        if event_count < 10:
            base_value *= 1.5
        
        # Higher value for complex contexts
        context_complexity = len(context)
        if context_complexity > 5:
            base_value *= 1.2
        
        # Higher value for outcomes with measurable results
        if isinstance(outcome, (int, float)) and outcome != 0:
            base_value *= 1.3
        
        return min(5.0, base_value)  # Cap at 5.0
    
    def _calculate_metric_trend(self, metric_name: str, current_value: float) -> str:
        """Calculate trend for a metric."""
        
        if metric_name not in self._performance_metrics:
            return "stable"
        
        recent_metrics = list(self._performance_metrics[metric_name])[-5:]
        
        if len(recent_metrics) < 3:
            return "stable"
        
        values = [m.value for m in recent_metrics] + [current_value]
        times = list(range(len(values)))
        
        slope = self._calculate_trend_slope(times, values)
        
        if slope > 0.1:
            return "improving"
        elif slope < -0.1:
            return "degrading"
        else:
            return "stable"
    
    def _calculate_trend_slope(self, x_values: List[float], y_values: List[float]) -> float:
        """Calculate slope of trend line."""
        
        if len(x_values) != len(y_values) or len(x_values) < 2:
            return 0.0
        
        n = len(x_values)
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_x2 = sum(x * x for x in x_values)
        
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope
    
    def _initialize_default_rules(self):
        """Initialize default adaptation rules."""
        
        # Performance degradation rule
        def performance_degradation_condition(context: Dict[str, Any]) -> bool:
            # Check if recent performance has degraded
            for metric_name, metrics in self._performance_metrics.items():
                if len(metrics) >= 5:
                    recent_values = [m.value for m in list(metrics)[-5:]]
                    if len(recent_values) >= 3:
                        slope = self._calculate_trend_slope(
                            list(range(len(recent_values))), 
                            recent_values
                        )
                        if slope < -0.2:  # Significant degradation
                            return True
            return False
        
        def performance_degradation_action(context: Dict[str, Any]) -> Any:
            # Trigger immediate optimization
            asyncio.create_task(self.optimize_parameters(OptimizationObjective.PERFORMANCE))
            return "triggered_performance_optimization"
        
        self.add_adaptation_rule(
            "performance_degradation_response",
            performance_degradation_condition,
            performance_degradation_action,
            priority=1
        )
        
        # High error rate rule
        def high_error_rate_condition(context: Dict[str, Any]) -> bool:
            error_events = [e for e in list(self._learning_events)[-100:] 
                           if e.event_type.endswith("_error")]
            return len(error_events) > 10  # More than 10% error rate
        
        def high_error_rate_action(context: Dict[str, Any]) -> Any:
            # Reduce learning rate and increase stability
            self.learning_rate *= 0.8
            return "reduced_learning_rate"
        
        self.add_adaptation_rule(
            "high_error_rate_response",
            high_error_rate_condition,
            high_error_rate_action,
            priority=2
        )
    
    def _initialize_learning_models(self):
        """Initialize learning models."""
        
        # Performance prediction model
        self._learning_models["performance_predictor"] = LearningModel(
            model_id="performance_predictor",
            model_type="linear_regression",
            parameters={"weight": 1.0, "bias": 0.0},
            training_data=[]
        )
        
        # User satisfaction model
        self._learning_models["satisfaction_predictor"] = LearningModel(
            model_id="satisfaction_predictor",
            model_type="feedback_classifier",
            parameters={"threshold": 0.7, "learning_rate": 0.1},
            training_data=[]
        )
    
    async def _learning_loop(self):
        """Main learning loop."""
        
        while self._running:
            try:
                # Process recent events for learning
                await self._process_learning_batch()
                
                # Apply adaptation rules
                await self._apply_adaptation_rules()
                
                # Update learning models
                await self._update_learning_models()
                
                # Sleep
                await asyncio.sleep(60)  # Learn every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in learning loop: {e}")
                await asyncio.sleep(10)
    
    async def _optimization_loop(self):
        """Main optimization loop."""
        
        while self._running:
            try:
                # Run periodic optimization
                await self.optimize_parameters()
                
                # Sleep until next optimization
                await asyncio.sleep(self.optimization_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                await asyncio.sleep(60)
    
    async def _process_learning_batch(self):
        """Process a batch of recent events for learning."""
        
        recent_events = list(self._learning_events)[-100:]  # Last 100 events
        
        if len(recent_events) < 10:
            return
        
        # Analyze patterns
        patterns = self._analyze_event_patterns(recent_events)
        
        # Update system parameters based on patterns
        for pattern, strength in patterns.items():
            if strength > 0.7:  # Strong pattern
                await self._adapt_to_pattern(pattern, strength)
    
    def _analyze_event_patterns(self, events: List[LearningEvent]) -> Dict[str, float]:
        """Analyze patterns in events."""
        
        patterns = {}
        
        # Event type frequency patterns
        event_type_counts = defaultdict(int)
        for event in events:
            event_type_counts[event.event_type] += 1
        
        total_events = len(events)
        for event_type, count in event_type_counts.items():
            frequency = count / total_events
            patterns[f"frequent_{event_type}"] = frequency
        
        # Feedback score patterns
        feedback_events = [e for e in events if e.feedback_score is not None]
        if feedback_events:
            avg_feedback = sum(e.feedback_score for e in feedback_events) / len(feedback_events)
            patterns["positive_feedback"] = max(0, avg_feedback)
            patterns["negative_feedback"] = max(0, -avg_feedback)
        
        # Context complexity patterns
        complex_events = [e for e in events if len(e.context) > 5]
        patterns["high_complexity"] = len(complex_events) / total_events
        
        return patterns
    
    async def _adapt_to_pattern(self, pattern: str, strength: float):
        """Adapt system behavior to detected pattern."""
        
        if pattern.startswith("frequent_"):
            event_type = pattern.replace("frequent_", "")
            # Optimize for this frequent event type
            logger.info(f"Adapting to frequent event type: {event_type}")
        
        elif pattern == "positive_feedback":
            # Increase learning rate for positive feedback
            self.learning_rate = min(0.3, self.learning_rate * 1.1)
        
        elif pattern == "negative_feedback":
            # Decrease learning rate for negative feedback
            self.learning_rate = max(0.01, self.learning_rate * 0.9)
        
        elif pattern == "high_complexity":
            # Increase processing capacity for complex events
            logger.info("Adapting to high complexity events")
    
    async def _apply_adaptation_rules(self):
        """Apply adaptation rules based on current context."""
        
        context = {
            "recent_events": list(self._learning_events)[-50:],
            "performance_metrics": dict(self._performance_metrics),
            "system_parameters": self._system_parameters
        }
        
        # Sort rules by priority
        sorted_rules = sorted(
            self._adaptation_rules.values(),
            key=lambda r: r.priority,
            reverse=True
        )
        
        for rule in sorted_rules:
            if not rule.is_active:
                continue
            
            try:
                if rule.condition(context):
                    # Apply rule action
                    result = rule.action(context)
                    rule.success_count += 1
                    rule.last_applied = datetime.now()
                    
                    logger.info(f"Applied adaptation rule: {rule.rule_id}")
            
            except Exception as e:
                rule.failure_count += 1
                logger.error(f"Failed to apply rule {rule.rule_id}: {e}")
    
    async def _update_learning_models(self):
        """Update learning models with recent data."""
        
        for model_id, model in self._learning_models.items():
            try:
                await self._train_model(model)
            except Exception as e:
                logger.error(f"Failed to update model {model_id}: {e}")
    
    async def _train_model(self, model: LearningModel):
        """Train a learning model."""
        
        if model.model_type == "linear_regression":
            await self._train_linear_regression(model)
        elif model.model_type == "feedback_classifier":
            await self._train_feedback_classifier(model)
    
    async def _train_linear_regression(self, model: LearningModel):
        """Train linear regression model."""
        
        # Collect training data from performance metrics
        training_data = []
        
        for metric_name, metrics in self._performance_metrics.items():
            if len(metrics) >= 5:
                recent_metrics = list(metrics)[-10:]
                for i, metric in enumerate(recent_metrics[:-1]):
                    x = [i, metric.value, len(metric.context)]  # Simple features
                    y = recent_metrics[i + 1].value  # Next value
                    training_data.append((x, y))
        
        if len(training_data) >= 5:
            # Simple gradient descent update
            for x, y in training_data[-5:]:  # Use last 5 samples
                prediction = sum(w * f for w, f in zip(model.parameters.values(), x + [1]))
                error = y - prediction
                
                # Update weights
                learning_rate = 0.01
                model.parameters["weight"] += learning_rate * error * x[0]
                model.parameters["bias"] += learning_rate * error
            
            model.last_trained = datetime.now()
    
    async def _train_feedback_classifier(self, model: LearningModel):
        """Train feedback classifier model."""
        
        # Collect feedback data
        feedback_events = [e for e in self._learning_events 
                          if e.feedback_score is not None]
        
        if len(feedback_events) >= 10:
            # Simple threshold adjustment
            positive_scores = [e.feedback_score for e in feedback_events[-20:] 
                             if e.feedback_score > 0]
            negative_scores = [e.feedback_score for e in feedback_events[-20:] 
                             if e.feedback_score < 0]
            
            if positive_scores and negative_scores:
                avg_positive = sum(positive_scores) / len(positive_scores)
                avg_negative = sum(negative_scores) / len(negative_scores)
                
                # Adjust threshold
                new_threshold = (avg_positive + abs(avg_negative)) / 2
                model.parameters["threshold"] = new_threshold
                
                model.last_trained = datetime.now()
    
    async def _process_immediate_learning(self, event: LearningEvent):
        """Process immediate learning from significant event."""
        
        if event.feedback_score and abs(event.feedback_score) > 0.8:
            # Strong feedback - immediate adaptation
            if event.feedback_score > 0.8:
                # Very positive feedback - reinforce behavior
                for param_name in self._system_parameters:
                    if param_name in event.context:
                        current_value = self._system_parameters[param_name]
                        self._system_parameters[param_name] = current_value * 1.05
            
            elif event.feedback_score < -0.8:
                # Very negative feedback - adjust behavior
                for param_name in self._system_parameters:
                    if param_name in event.context:
                        current_value = self._system_parameters[param_name]
                        self._system_parameters[param_name] = current_value * 0.95
    
    async def _learn_from_feedback(self, event: LearningEvent, feedback_score: float):
        """Learn from feedback on an event."""
        
        # Update learning models with feedback
        for model in self._learning_models.values():
            if model.model_type == "feedback_classifier":
                # Add to training data
                features = [
                    len(event.context),
                    event.learning_value,
                    float(event.event_type.endswith("_error"))
                ]
                model.training_data.append((features, feedback_score))
                
                # Keep only recent training data
                model.training_data = model.training_data[-100:]
        
        # Adjust system parameters based on feedback
        adjustment_factor = feedback_score * self.learning_rate
        
        for param_name, param_value in self._system_parameters.items():
            if param_name in event.context:
                new_value = param_value * (1 + adjustment_factor)
                self._system_parameters[param_name] = max(0.1, min(10.0, new_value))
    
    def _get_recent_performance(self) -> Dict[str, List[float]]:
        """Get recent performance data."""
        
        recent_performance = {}
        
        for metric_name, metrics in self._performance_metrics.items():
            recent_values = [m.value for m in list(metrics)[-20:]]  # Last 20 values
            if recent_values:
                recent_performance[metric_name] = recent_values
        
        return recent_performance
    
    async def _optimize_for_performance(self, 
                                       recent_performance: Dict[str, List[float]],
                                       result: Dict[str, Any]):
        """Optimize system for performance."""
        
        # Look for performance-related metrics
        perf_metrics = ["response_time", "throughput", "processing_speed"]
        
        for metric_name in perf_metrics:
            if metric_name in recent_performance:
                values = recent_performance[metric_name]
                if len(values) >= 3:
                    slope = self._calculate_trend_slope(list(range(len(values))), values)
                    
                    if slope < 0:  # Performance degrading
                        # Increase relevant parameters
                        if "cache_size" in self._system_parameters:
                            self._system_parameters["cache_size"] *= 1.1
                            result["parameters_changed"].append("cache_size")
                        
                        if "worker_threads" in self._system_parameters:
                            self._system_parameters["worker_threads"] *= 1.05
                            result["parameters_changed"].append("worker_threads")
    
    async def _optimize_for_accuracy(self, 
                                    recent_performance: Dict[str, List[float]],
                                    result: Dict[str, Any]):
        """Optimize system for accuracy."""
        
        accuracy_metrics = ["prediction_accuracy", "classification_accuracy", "f1_score"]
        
        for metric_name in accuracy_metrics:
            if metric_name in recent_performance:
                values = recent_performance[metric_name]
                current_accuracy = values[-1] if values else 0
                
                if current_accuracy < 0.8:  # Low accuracy
                    # Increase model complexity or training
                    if "model_complexity" in self._system_parameters:
                        self._system_parameters["model_complexity"] *= 1.05
                        result["parameters_changed"].append("model_complexity")
    
    async def _optimize_for_efficiency(self, 
                                      recent_performance: Dict[str, List[float]],
                                      result: Dict[str, Any]):
        """Optimize system for efficiency."""
        
        efficiency_metrics = ["cpu_usage", "memory_usage", "energy_consumption"]
        
        for metric_name in efficiency_metrics:
            if metric_name in recent_performance:
                values = recent_performance[metric_name]
                current_usage = values[-1] if values else 0
                
                if current_usage > 0.8:  # High resource usage
                    # Reduce resource consumption
                    if "batch_size" in self._system_parameters:
                        self._system_parameters["batch_size"] *= 0.95
                        result["parameters_changed"].append("batch_size")
    
    async def _optimize_balanced(self, 
                                recent_performance: Dict[str, List[float]],
                                result: Dict[str, Any]):
        """Optimize system with balanced objectives."""
        
        # Balanced optimization considers multiple factors
        await self._optimize_for_performance(recent_performance, result)
        await self._optimize_for_accuracy(recent_performance, result)
        await self._optimize_for_efficiency(recent_performance, result)
        
        # Apply conservative adjustments for balanced approach
        for param_name in result["parameters_changed"]:
            if param_name in self._system_parameters:
                current_value = self._system_parameters[param_name]
                # Reduce adjustment magnitude for balanced optimization
                if current_value > 1.0:
                    self._system_parameters[param_name] = 1.0 + (current_value - 1.0) * 0.5
                else:
                    self._system_parameters[param_name] = current_value + (1.0 - current_value) * 0.5


# Global adaptive learning system
_global_adaptive_learning = None


def get_adaptive_learning_system(**kwargs) -> AdaptiveLearningSystem:
    """Get global adaptive learning system instance."""
    
    global _global_adaptive_learning
    if _global_adaptive_learning is None:
        _global_adaptive_learning = AdaptiveLearningSystem(**kwargs)
    return _global_adaptive_learning


# Decorator for adaptive learning
def adaptive_learning(event_type: str, learning_value: float = 1.0):
    """Decorator to enable adaptive learning for functions."""
    
    def decorator(func):
        async def wrapper(*args, **kwargs):
            learning_system = get_adaptive_learning_system()
            
            start_time = time.time()
            
            try:
                # Execute function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Record successful event
                execution_time = time.time() - start_time
                context = {
                    "function_name": func.__name__,
                    "execution_time": execution_time,
                    "args_count": len(args),
                    "kwargs_count": len(kwargs)
                }
                
                learning_system.record_event(
                    event_type=event_type,
                    context=context,
                    outcome={"success": True, "execution_time": execution_time},
                    feedback_score=None
                )
                
                # Record performance metric
                learning_system.record_performance_metric(
                    f"{event_type}_execution_time",
                    execution_time,
                    context
                )
                
                return result
                
            except Exception as e:
                # Record error event
                execution_time = time.time() - start_time
                context = {
                    "function_name": func.__name__,
                    "execution_time": execution_time,
                    "error_type": type(e).__name__
                }
                
                learning_system.record_event(
                    event_type=f"{event_type}_error",
                    context=context,
                    outcome={"success": False, "error": str(e)},
                    feedback_score=-0.8  # Negative feedback for errors
                )
                
                raise
        
        return wrapper
    return decorator