"""
Adaptive Learning System Demo
=============================

Demonstrate self-improving patterns and adaptive learning capabilities.
"""

import asyncio
import random
import time
from datetime import datetime

from src.lexgraph_legal_rag.adaptive_learning_system import (
    AdaptiveLearningSystem,
    LearningMode,
    OptimizationObjective,
    get_adaptive_learning_system,
    adaptive_learning
)


class AdaptiveLearningDemo:
    """Demonstrate adaptive learning capabilities."""
    
    def __init__(self):
        self.learning_system = AdaptiveLearningSystem(
            learning_rate=0.15,
            optimization_interval=30.0,  # 30 seconds for demo
            max_history_size=1000
        )
        
        # Initialize some system parameters
        self.learning_system._system_parameters = {
            "cache_size": 1.0,
            "worker_threads": 1.0,
            "batch_size": 1.0,
            "model_complexity": 1.0
        }
    
    async def run_demo(self):
        """Run comprehensive adaptive learning demonstration."""
        
        print("🧠 ADAPTIVE LEARNING SYSTEM DEMONSTRATION")
        print("=" * 60)
        print("Novel self-improving AI with autonomous optimization")
        print()
        
        # Start learning system
        await self.learning_system.start_learning()
        
        try:
            # Phase 1: Basic Learning
            print("📊 Phase 1: Basic Learning and Event Recording")
            print("-" * 50)
            await self._demonstrate_basic_learning()
            
            # Phase 2: Performance Tracking
            print("\n📈 Phase 2: Performance Tracking and Trend Analysis")
            print("-" * 50)
            await self._demonstrate_performance_tracking()
            
            # Phase 3: Feedback Learning
            print("\n🔄 Phase 3: Feedback-Driven Learning")
            print("-" * 50)
            await self._demonstrate_feedback_learning()
            
            # Phase 4: Adaptive Optimization
            print("\n⚡ Phase 4: Autonomous Parameter Optimization")
            print("-" * 50)
            await self._demonstrate_adaptive_optimization()
            
            # Phase 5: Pattern Recognition
            print("\n🔍 Phase 5: Pattern Recognition and Adaptation")
            print("-" * 50)
            await self._demonstrate_pattern_recognition()
            
            # Phase 6: Self-Improvement
            print("\n🚀 Phase 6: Autonomous Self-Improvement")
            print("-" * 50)
            await self._demonstrate_self_improvement()
            
            # Final Insights
            print("\n📋 Learning Insights Summary")
            print("-" * 50)
            await self._show_learning_insights()
            
        finally:
            await self.learning_system.stop_learning()
    
    async def _demonstrate_basic_learning(self):
        """Demonstrate basic learning event recording."""
        
        # Record various types of events
        event_types = [
            "document_processing",
            "query_execution", 
            "cache_access",
            "model_inference",
            "user_interaction"
        ]
        
        print("Recording learning events...")
        
        for i in range(20):
            event_type = random.choice(event_types)
            
            # Simulate event context
            context = {
                "document_size": random.randint(100, 10000),
                "complexity": random.uniform(0.1, 1.0),
                "user_type": random.choice(["expert", "novice", "intermediate"]),
                "timestamp": time.time()
            }
            
            # Simulate outcome
            success = random.random() > 0.1  # 90% success rate
            outcome = {
                "success": success,
                "processing_time": random.uniform(0.1, 2.0),
                "accuracy": random.uniform(0.7, 0.95) if success else random.uniform(0.3, 0.6)
            }
            
            # Record event
            event_id = self.learning_system.record_event(
                event_type=event_type,
                context=context,
                outcome=outcome
            )
            
            if i % 5 == 0:
                print(f"  ✅ Recorded event {i+1}/20: {event_type}")
            
            await asyncio.sleep(0.1)  # Small delay
        
        print(f"✅ Recorded 20 learning events")
        print(f"📊 Total events in system: {len(self.learning_system._learning_events)}")
    
    async def _demonstrate_performance_tracking(self):
        """Demonstrate performance metric tracking."""
        
        print("Tracking performance metrics...")
        
        # Simulate performance metrics over time
        metrics = [
            "response_time",
            "throughput", 
            "accuracy",
            "cpu_usage",
            "memory_usage"
        ]
        
        for round_num in range(5):
            print(f"  Round {round_num + 1}/5:")
            
            for metric in metrics:
                # Simulate metric values with trends
                if metric == "response_time":
                    # Gradually improving response time
                    base_value = 1.0 - (round_num * 0.1)
                    value = base_value + random.uniform(-0.1, 0.1)
                elif metric == "throughput":
                    # Gradually improving throughput  
                    base_value = 100 + (round_num * 20)
                    value = base_value + random.uniform(-10, 10)
                elif metric == "accuracy":
                    # Slight improvement in accuracy
                    base_value = 0.8 + (round_num * 0.03)
                    value = base_value + random.uniform(-0.02, 0.02)
                elif metric == "cpu_usage":
                    # Variable CPU usage
                    value = 0.3 + random.uniform(0, 0.4)
                else:  # memory_usage
                    # Gradually increasing memory usage
                    base_value = 0.4 + (round_num * 0.05)
                    value = base_value + random.uniform(-0.05, 0.05)
                
                self.learning_system.record_performance_metric(
                    metric_name=metric,
                    value=value,
                    context={"round": round_num}
                )
            
            print(f"    ✅ Recorded metrics for round {round_num + 1}")
            await asyncio.sleep(1)  # Wait between rounds
        
        # Check trends
        trends_detected = 0
        for metric_name, metrics_data in self.learning_system._performance_metrics.items():
            recent_metric = list(metrics_data)[-1]
            if recent_metric.trend != "stable":
                trends_detected += 1
                print(f"    📈 Trend detected in {metric_name}: {recent_metric.trend}")
        
        print(f"✅ Performance tracking complete - {trends_detected} trends detected")
    
    async def _demonstrate_feedback_learning(self):
        """Demonstrate learning from user feedback."""
        
        print("Learning from user feedback...")
        
        # Get recent events to provide feedback on
        recent_events = list(self.learning_system._learning_events)[-10:]
        
        for i, event in enumerate(recent_events):
            # Simulate user feedback
            if event.event_type == "document_processing":
                # Good performance gets positive feedback
                if event.outcome.get("accuracy", 0) > 0.85:
                    feedback_score = random.uniform(0.7, 1.0)
                else:
                    feedback_score = random.uniform(-0.5, 0.3)
            
            elif event.event_type == "query_execution":
                # Fast queries get positive feedback  
                if event.outcome.get("processing_time", 1.0) < 0.5:
                    feedback_score = random.uniform(0.6, 0.9)
                else:
                    feedback_score = random.uniform(-0.3, 0.4)
            
            else:
                # Random feedback for other events
                feedback_score = random.uniform(-0.5, 0.8)
            
            # Apply feedback
            await self.learning_system.adapt_to_feedback(
                event_id=event.event_id,
                feedback_score=feedback_score,
                feedback_details={"source": "demo_user", "round": i}
            )
            
            print(f"  ✅ Applied feedback {i+1}/10: score={feedback_score:.2f}")
            
            await asyncio.sleep(0.2)
        
        print(f"✅ Feedback learning complete")
        
        # Show adaptation impact
        updated_learning_rate = self.learning_system.learning_rate
        print(f"📊 Learning rate adapted to: {updated_learning_rate:.3f}")
    
    async def _demonstrate_adaptive_optimization(self):
        """Demonstrate autonomous parameter optimization."""
        
        print("Running autonomous parameter optimization...")
        
        # Record initial parameters
        initial_params = self.learning_system._system_parameters.copy()
        print(f"  Initial parameters: {initial_params}")
        
        # Optimize for different objectives
        objectives = [
            OptimizationObjective.PERFORMANCE,
            OptimizationObjective.ACCURACY,
            OptimizationObjective.EFFICIENCY,
            OptimizationObjective.BALANCED
        ]
        
        for i, objective in enumerate(objectives):
            print(f"\n  Optimization {i+1}/4: {objective.value}")
            
            # Run optimization
            result = await self.learning_system.optimize_parameters(objective)
            
            print(f"    ✅ Optimization complete")
            print(f"    📊 Parameters changed: {result['parameters_changed']}")
            print(f"    📈 Performance improvement: {result['performance_improvement']:.2f}%")
            
            await asyncio.sleep(2)  # Wait between optimizations
        
        # Show final parameters
        final_params = self.learning_system._system_parameters.copy()
        print(f"\n  Final parameters: {final_params}")
        
        # Calculate total change
        total_change = sum(abs(final_params[k] - initial_params[k]) 
                          for k in initial_params.keys())
        print(f"✅ Total parameter adaptation: {total_change:.3f}")
    
    async def _demonstrate_pattern_recognition(self):
        """Demonstrate pattern recognition and adaptation."""
        
        print("Testing pattern recognition and adaptation...")
        
        # Create specific patterns
        patterns = [
            ("high_frequency_pattern", 15),  # Frequent event
            ("complex_context_pattern", 8),   # Complex events
            ("error_spike_pattern", 6)        # Error events
        ]
        
        for pattern_name, count in patterns:
            print(f"  Creating pattern: {pattern_name}")
            
            for i in range(count):
                if pattern_name == "high_frequency_pattern":
                    # High frequency events
                    context = {"frequency": "high", "pattern": pattern_name}
                    outcome = {"success": True, "processing_time": 0.1}
                    feedback_score = 0.8
                
                elif pattern_name == "complex_context_pattern":
                    # Complex context events
                    context = {
                        "complexity": "high",
                        "pattern": pattern_name,
                        "param1": random.random(),
                        "param2": random.random(), 
                        "param3": random.random(),
                        "param4": random.random(),
                        "param5": random.random(),
                        "param6": random.random()
                    }
                    outcome = {"success": True, "processing_time": 1.0}
                    feedback_score = 0.7
                
                else:  # error_spike_pattern
                    # Error events
                    context = {"error_type": "simulated", "pattern": pattern_name}
                    outcome = {"success": False, "error": "Simulated error"}
                    feedback_score = -0.8
                
                # Record event
                event_id = self.learning_system.record_event(
                    event_type=pattern_name,
                    context=context,
                    outcome=outcome,
                    feedback_score=feedback_score
                )
                
                await asyncio.sleep(0.1)
            
            print(f"    ✅ Created {count} events for {pattern_name}")
        
        # Wait for pattern processing
        await asyncio.sleep(2)
        
        print("✅ Pattern recognition test complete")
        
        # Check if learning rate was adapted due to error pattern
        current_learning_rate = self.learning_system.learning_rate
        print(f"📊 Learning rate after pattern exposure: {current_learning_rate:.3f}")
    
    async def _demonstrate_self_improvement(self):
        """Demonstrate autonomous self-improvement."""
        
        print("Demonstrating autonomous self-improvement...")
        
        # Add custom adaptation rule
        def custom_improvement_condition(context):
            # Trigger when response time is consistently high
            response_metrics = context.get("performance_metrics", {}).get("response_time", [])
            if response_metrics and len(response_metrics) >= 3:
                recent_times = [m.value for m in list(response_metrics)[-3:]]
                return all(t > 1.0 for t in recent_times)
            return False
        
        def custom_improvement_action(context):
            # Implement self-improvement action
            learning_system = get_adaptive_learning_system()
            learning_system._system_parameters["cache_size"] *= 1.2
            learning_system._system_parameters["worker_threads"] *= 1.1
            return "autonomous_performance_improvement"
        
        self.learning_system.add_adaptation_rule(
            "custom_improvement",
            custom_improvement_condition,
            custom_improvement_action,
            priority=1
        )
        
        print("  ✅ Added custom self-improvement rule")
        
        # Simulate conditions that trigger improvement
        for i in range(5):
            # Record high response times
            self.learning_system.record_performance_metric(
                "response_time",
                1.5 + random.uniform(0, 0.5),  # High response time
                {"trigger_round": i}
            )
            
            await asyncio.sleep(0.5)
        
        # Let adaptation rules process
        await asyncio.sleep(3)
        
        # Check if improvements were made
        improved_cache = self.learning_system._system_parameters.get("cache_size", 1.0)
        improved_workers = self.learning_system._system_parameters.get("worker_threads", 1.0)
        
        print(f"  📈 Cache size improvement: {improved_cache:.2f}x")
        print(f"  📈 Worker threads improvement: {improved_workers:.2f}x")
        
        print("✅ Self-improvement demonstration complete")
    
    async def _show_learning_insights(self):
        """Show comprehensive learning insights."""
        
        insights = self.learning_system.get_learning_insights()
        
        print(f"Total learning events: {insights['total_events']}")
        print(f"Current learning rate: {insights['learning_rate']:.3f}")
        print(f"Active adaptation rules: {insights['active_rules']}")
        
        print("\nPerformance Trends:")
        for metric_name, trend_info in insights["performance_trends"].items():
            print(f"  {metric_name}: {trend_info['current_value']:.3f} ({trend_info['trend']})")
        
        print("\nTop Learning Patterns:")
        for pattern in insights["top_learning_patterns"][:3]:
            print(f"  {pattern['event_type']}: {pattern['count']} events")
        
        print("\nModel Performance:")
        for model_id, performance in insights["model_performance"].items():
            print(f"  {model_id}: {performance['accuracy']:.3f} accuracy")
        
        print(f"\nOptimization History: {len(insights['optimization_history'])} optimizations performed")
        
        print("\n🎉 ADAPTIVE LEARNING DEMONSTRATION COMPLETE!")
        print("🧠 The system has demonstrated autonomous learning and self-improvement")


# Test with decorators
@adaptive_learning("demo_processing", learning_value=1.5)
async def demo_processing_function(complexity: float):
    """Demo function with adaptive learning."""
    
    # Simulate processing time based on complexity
    processing_time = complexity * 0.5 + random.uniform(0.1, 0.3)
    await asyncio.sleep(processing_time)
    
    # Simulate occasional failures
    if random.random() < 0.1:
        raise ValueError("Simulated processing error")
    
    return {
        "result": f"Processed with complexity {complexity:.2f}",
        "processing_time": processing_time
    }


async def test_adaptive_decorators():
    """Test adaptive learning decorators."""
    
    print("\n🔧 Testing Adaptive Learning Decorators")
    print("-" * 40)
    
    learning_system = get_adaptive_learning_system()
    await learning_system.start_learning()
    
    try:
        # Test function with varying complexity
        for i in range(10):
            complexity = random.uniform(0.1, 2.0)
            
            try:
                result = await demo_processing_function(complexity)
                print(f"  ✅ Processed complexity {complexity:.2f}: {result['processing_time']:.3f}s")
            except ValueError as e:
                print(f"  ❌ Error with complexity {complexity:.2f}: {e}")
            
            await asyncio.sleep(0.1)
        
        # Show learning results
        insights = learning_system.get_learning_insights()
        print(f"\n📊 Learning Events: {insights['total_events']}")
        print(f"📈 Performance Trends: {len(insights['performance_trends'])}")
        
    finally:
        await learning_system.stop_learning()


async def main():
    """Main demonstration function."""
    
    # Run main demo
    demo = AdaptiveLearningDemo()
    await demo.run_demo()
    
    # Test decorators
    await test_adaptive_decorators()


if __name__ == "__main__":
    asyncio.run(main())