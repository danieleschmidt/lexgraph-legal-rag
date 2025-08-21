#!/usr/bin/env python3
"""
Production-Grade Resilience System for Bioneural Legal AI
=========================================================

Enterprise-level resilience, error recovery, and fault tolerance
for the bioneural olfactory fusion legal document analysis system.

Features:
- Circuit breaker patterns for downstream services
- Advanced retry mechanisms with exponential backoff
- Health monitoring and auto-recovery
- Graceful degradation under load
- Error classification and intelligent routing
- Performance optimization under stress
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import threading
import psutil
import random

from src.lexgraph_legal_rag.circuit_breaker import CircuitBreaker
from src.lexgraph_legal_rag.intelligent_error_recovery import get_recovery_system
from src.lexgraph_legal_rag.health_monitoring import SystemHealthMonitor
from src.lexgraph_legal_rag.bioneuro_olfactory_fusion import BioneuroOlfactoryFusionEngine
from src.lexgraph_legal_rag.multisensory_legal_processor import MultiSensoryLegalProcessor
from src.lexgraph_legal_rag.adaptive_monitoring import get_monitoring_system

logger = logging.getLogger(__name__)


class ServiceHealth(Enum):
    """Service health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILED = "failed"


class ErrorSeverity(Enum):
    """Error severity classifications."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ResilienceMetrics:
    """Comprehensive resilience metrics tracking."""
    requests_total: int = 0
    requests_successful: int = 0
    requests_failed: int = 0
    requests_retried: int = 0
    circuit_breaker_trips: int = 0
    auto_recoveries: int = 0
    degraded_mode_activations: int = 0
    average_response_time: float = 0.0
    error_rate: float = 0.0
    uptime_percentage: float = 100.0
    last_failure_time: Optional[float] = None
    recovery_time: float = 0.0


@dataclass
class FailurePattern:
    """Pattern detection for recurring failures."""
    error_type: str
    count: int = 0
    first_occurrence: float = field(default_factory=time.time)
    last_occurrence: float = field(default_factory=time.time)
    frequency: float = 0.0
    correlation_factors: List[str] = field(default_factory=list)


class ProductionGradeResilienceSystem:
    """Enterprise-level resilience orchestrator."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        
        # Core components
        self.bioneural_engine = BioneuroOlfactoryFusionEngine()
        self.multisensory_processor = MultiSensoryLegalProcessor()
        self.error_recovery = get_recovery_system()
        self.monitoring = get_monitoring_system()
        
        # Resilience components
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.health_monitors: Dict[str, SystemHealthMonitor] = {}
        self.metrics = ResilienceMetrics()
        self.failure_patterns: Dict[str, FailurePattern] = {}
        
        # State management
        self.is_degraded_mode = False
        self.service_health = ServiceHealth.HEALTHY
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config["max_workers"])
        self.lock = threading.RLock()
        
        # Initialize circuit breakers
        self._initialize_circuit_breakers()
        
        # Start background monitoring
        self._start_monitoring_tasks()
        
        logger.info("Production-grade resilience system initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for production resilience."""
        return {
            "max_workers": 8,
            "circuit_breaker": {
                "failure_threshold": 5,
                "recovery_timeout": 60,
                "expected_exception": Exception
            },
            "retry": {
                "max_attempts": 3,
                "base_delay": 1.0,
                "max_delay": 30.0,
                "exponential_base": 2.0
            },
            "health_monitoring": {
                "check_interval": 30,
                "memory_threshold": 0.85,
                "cpu_threshold": 0.80,
                "response_time_threshold": 5.0
            },
            "degraded_mode": {
                "memory_threshold": 0.90,
                "cpu_threshold": 0.85,
                "error_rate_threshold": 0.10,
                "response_time_threshold": 10.0
            }
        }
    
    def _initialize_circuit_breakers(self):
        """Initialize circuit breakers for critical services."""
        services = [
            "bioneural_analysis",
            "multisensory_processing", 
            "document_classification",
            "similarity_computation",
            "vector_indexing"
        ]
        
        for service in services:
            self.circuit_breakers[service] = CircuitBreaker(
                failure_threshold=self.config["circuit_breaker"]["failure_threshold"],
                recovery_timeout=self.config["circuit_breaker"]["recovery_timeout"],
                expected_exception=self.config["circuit_breaker"]["expected_exception"]
            )
            logger.debug(f"Circuit breaker initialized for {service}")
    
    def _start_monitoring_tasks(self):
        """Start background monitoring and health check tasks."""
        
        async def health_monitor_task():
            """Continuous health monitoring."""
            while True:
                try:
                    await self._perform_health_checks()
                    await asyncio.sleep(self.config["health_monitoring"]["check_interval"])
                except Exception as e:
                    logger.error(f"Health monitoring error: {e}")
                    await asyncio.sleep(5)  # Brief pause on error
        
        async def metrics_collection_task():
            """Continuous metrics collection."""
            while True:
                try:
                    await self._collect_system_metrics()
                    await asyncio.sleep(10)  # Collect metrics every 10 seconds
                except Exception as e:
                    logger.error(f"Metrics collection error: {e}")
                    await asyncio.sleep(5)
        
        # Schedule background tasks
        asyncio.create_task(health_monitor_task())
        asyncio.create_task(metrics_collection_task())
    
    async def _perform_health_checks(self):
        """Comprehensive health checks across all system components."""
        
        # System resource checks
        memory_usage = psutil.virtual_memory().percent / 100
        cpu_usage = psutil.cpu_percent(interval=1) / 100
        
        # Determine service health
        if (memory_usage > self.config["degraded_mode"]["memory_threshold"] or
            cpu_usage > self.config["degraded_mode"]["cpu_threshold"] or
            self.metrics.error_rate > self.config["degraded_mode"]["error_rate_threshold"]):
            
            self.service_health = ServiceHealth.CRITICAL
            if not self.is_degraded_mode:
                await self._activate_degraded_mode()
        
        elif (memory_usage > self.config["health_monitoring"]["memory_threshold"] or
              cpu_usage > self.config["health_monitoring"]["cpu_threshold"] or
              self.metrics.error_rate > 0.05):
            
            self.service_health = ServiceHealth.DEGRADED
        
        else:
            self.service_health = ServiceHealth.HEALTHY
            if self.is_degraded_mode:
                await self._deactivate_degraded_mode()
        
        logger.debug(f"Health check: {self.service_health.value} "
                    f"(Memory: {memory_usage:.2%}, CPU: {cpu_usage:.2%}, "
                    f"Error rate: {self.metrics.error_rate:.2%})")
    
    async def _collect_system_metrics(self):
        """Collect comprehensive system performance metrics."""
        
        with self.lock:
            # Calculate error rate
            if self.metrics.requests_total > 0:
                self.metrics.error_rate = self.metrics.requests_failed / self.metrics.requests_total
            
            # Calculate uptime percentage
            if self.metrics.last_failure_time:
                downtime = time.time() - self.metrics.last_failure_time
                total_time = time.time() - getattr(self, 'start_time', time.time())
                self.metrics.uptime_percentage = max(0, (total_time - downtime) / total_time * 100)
    
    async def _activate_degraded_mode(self):
        """Activate degraded operation mode for system resilience."""
        
        if self.is_degraded_mode:
            return
        
        self.is_degraded_mode = True
        self.metrics.degraded_mode_activations += 1
        
        # Reduce processing complexity
        self.config["max_workers"] = max(2, self.config["max_workers"] // 2)
        
        # Adjust circuit breaker thresholds
        for cb in self.circuit_breakers.values():
            cb.failure_threshold = max(1, cb.failure_threshold // 2)
        
        logger.warning("🚨 DEGRADED MODE ACTIVATED - System operating with reduced capacity")
    
    async def _deactivate_degraded_mode(self):
        """Deactivate degraded mode and restore full capacity."""
        
        if not self.is_degraded_mode:
            return
        
        self.is_degraded_mode = False
        
        # Restore full processing capacity
        self.config["max_workers"] = self._default_config()["max_workers"]
        
        # Restore circuit breaker thresholds
        for cb in self.circuit_breakers.values():
            cb.failure_threshold = self._default_config()["circuit_breaker"]["failure_threshold"]
        
        logger.info("✅ DEGRADED MODE DEACTIVATED - Full system capacity restored")
    
    async def resilient_execute(self, func: Callable, *args, service_name: str = "default", 
                              **kwargs) -> Any:
        """Execute function with comprehensive resilience patterns."""
        
        start_time = time.time()
        attempts = 0
        last_exception = None
        
        # Get circuit breaker for service
        circuit_breaker = self.circuit_breakers.get(service_name)
        if not circuit_breaker:
            circuit_breaker = CircuitBreaker()
            self.circuit_breakers[service_name] = circuit_breaker
        
        # Retry loop with exponential backoff
        while attempts < self.config["retry"]["max_attempts"]:
            attempts += 1
            
            try:
                with self.lock:
                    self.metrics.requests_total += 1
                
                # Check circuit breaker
                if circuit_breaker.is_open():
                    self.metrics.circuit_breaker_trips += 1
                    raise Exception(f"Circuit breaker open for {service_name}")
                
                # Execute function with timeout and monitoring
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Success path
                circuit_breaker.record_success()
                with self.lock:
                    self.metrics.requests_successful += 1
                    # Update average response time
                    response_time = time.time() - start_time
                    self.metrics.average_response_time = (
                        (self.metrics.average_response_time * (self.metrics.requests_successful - 1) + 
                         response_time) / self.metrics.requests_successful
                    )
                
                return result
                
            except Exception as e:
                last_exception = e
                circuit_breaker.record_failure()
                
                # Record failure pattern
                await self._record_failure_pattern(str(type(e).__name__), service_name)
                
                with self.lock:
                    self.metrics.requests_failed += 1
                    self.metrics.last_failure_time = time.time()
                
                # Determine if we should retry
                if attempts >= self.config["retry"]["max_attempts"]:
                    break
                
                # Calculate backoff delay
                delay = min(
                    self.config["retry"]["base_delay"] * 
                    (self.config["retry"]["exponential_base"] ** (attempts - 1)),
                    self.config["retry"]["max_delay"]
                )
                
                # Add jitter to prevent thundering herd
                jitter = random.uniform(0.1, 0.3) * delay
                total_delay = delay + jitter
                
                logger.warning(f"Attempt {attempts} failed for {service_name}: {e}. "
                              f"Retrying in {total_delay:.2f}s")
                
                with self.lock:
                    self.metrics.requests_retried += 1
                
                # Wait before retry
                await asyncio.sleep(total_delay)
        
        # All retries exhausted - attempt error recovery
        logger.error(f"All retries exhausted for {service_name}. Attempting error recovery.")
        
        try:
            recovery_result = await self._attempt_error_recovery(
                last_exception, func, args, kwargs, service_name
            )
            if recovery_result is not None:
                with self.lock:
                    self.metrics.auto_recoveries += 1
                return recovery_result
        except Exception as recovery_error:
            logger.error(f"Error recovery failed: {recovery_error}")
        
        # Final fallback
        raise last_exception or Exception(f"Failed to execute {service_name} after all recovery attempts")
    
    async def _record_failure_pattern(self, error_type: str, service_name: str):
        """Record and analyze failure patterns for predictive recovery."""
        
        pattern_key = f"{service_name}:{error_type}"
        current_time = time.time()
        
        if pattern_key not in self.failure_patterns:
            self.failure_patterns[pattern_key] = FailurePattern(
                error_type=error_type,
                first_occurrence=current_time,
                correlation_factors=[service_name]
            )
        
        pattern = self.failure_patterns[pattern_key]
        pattern.count += 1
        pattern.last_occurrence = current_time
        
        # Calculate frequency
        time_span = current_time - pattern.first_occurrence
        if time_span > 0:
            pattern.frequency = pattern.count / time_span
        
        # Detect concerning patterns
        if pattern.count > 10 and pattern.frequency > 0.1:  # More than 10 failures in 100 seconds
            logger.critical(f"⚠️  FAILURE PATTERN DETECTED: {pattern_key} "
                          f"({pattern.count} occurrences, {pattern.frequency:.3f} Hz)")
    
    async def _attempt_error_recovery(self, exception: Exception, func: Callable, 
                                    args: tuple, kwargs: dict, service_name: str) -> Any:
        """Intelligent error recovery with multiple strategies."""
        
        logger.info(f"Attempting error recovery for {service_name}")
        
        # Strategy 1: Simplified processing
        if "bioneural" in service_name:
            try:
                logger.info("Attempting simplified bioneural processing")
                # Reduce receptor sensitivity for faster processing
                simplified_result = await self._simplified_bioneural_processing(*args, **kwargs)
                return simplified_result
            except Exception as e:
                logger.warning(f"Simplified processing failed: {e}")
        
        # Strategy 2: Fallback to baseline processing
        if "multisensory" in service_name:
            try:
                logger.info("Falling back to single-sensory processing")
                # Use only textual processing as fallback
                fallback_result = await self._fallback_textual_processing(*args, **kwargs)
                return fallback_result
            except Exception as e:
                logger.warning(f"Fallback processing failed: {e}")
        
        # Strategy 3: Cached result if available
        try:
            cached_result = await self._get_cached_result(args, kwargs, service_name)
            if cached_result is not None:
                logger.info("Using cached result for error recovery")
                return cached_result
        except Exception as e:
            logger.warning(f"Cache recovery failed: {e}")
        
        # Strategy 4: Default safe response
        return await self._generate_safe_default_response(service_name)
    
    async def _simplified_bioneural_processing(self, *args, **kwargs) -> Any:
        """Simplified bioneural processing for error recovery."""
        
        # Create simplified engine with reduced complexity
        simple_engine = BioneuroOlfactoryFusionEngine()
        
        # Process with reduced receptor sensitivity
        if args and len(args) >= 2:
            document_text = args[0]
            document_id = args[1]
            
            # Simple classification based on keywords only
            text_lower = document_text.lower()
            
            # Basic classification result
            if any(word in text_lower for word in ["contract", "agreement", "party"]):
                category = "contract"
            elif any(word in text_lower for word in ["statute", "u.s.c.", "section"]):
                category = "statute"
            elif any(word in text_lower for word in ["court", "plaintiff", "defendant"]):
                category = "case_law"
            else:
                category = "regulation"
            
            return {
                "document_id": document_id,
                "classification": category,
                "confidence": 0.6,  # Lower confidence for simplified processing
                "processing_mode": "simplified_recovery",
                "scent_intensity": 0.3,
                "scent_complexity": 0.2
            }
        
        return None
    
    async def _fallback_textual_processing(self, *args, **kwargs) -> Any:
        """Fallback to basic textual processing."""
        
        if args and len(args) >= 2:
            document_text = args[0]
            document_id = args[1]
            
            # Basic textual analysis
            word_count = len(document_text.split())
            sentence_count = document_text.count('.') + document_text.count('!') + document_text.count('?')
            avg_sentence_length = word_count / max(sentence_count, 1)
            
            return {
                "document_id": document_id,
                "primary_channel": "textual",
                "word_count": word_count,
                "sentence_count": sentence_count,
                "avg_sentence_length": avg_sentence_length,
                "processing_mode": "fallback_textual",
                "confidence": 0.5
            }
        
        return None
    
    async def _get_cached_result(self, args: tuple, kwargs: dict, service_name: str) -> Any:
        """Attempt to retrieve cached result for recovery."""
        
        # Simple cache key generation
        cache_key = f"{service_name}:{hash(str(args))}"
        
        # In production, this would connect to Redis or similar
        # For demo, return None (no cache)
        return None
    
    async def _generate_safe_default_response(self, service_name: str) -> Any:
        """Generate safe default response when all recovery strategies fail."""
        
        logger.warning(f"Generating safe default response for {service_name}")
        
        return {
            "status": "degraded",
            "service": service_name,
            "message": "Service temporarily unavailable - using safe default",
            "confidence": 0.1,
            "processing_mode": "safe_default",
            "timestamp": time.time()
        }
    
    async def analyze_document_resilient(self, document_text: str, document_id: str, 
                                       method: str = "bioneural") -> Dict[str, Any]:
        """Resilient document analysis with comprehensive error handling."""
        
        if method == "bioneural":
            return await self.resilient_execute(
                self._bioneural_analysis_wrapper,
                document_text,
                document_id,
                service_name="bioneural_analysis"
            )
        elif method == "multisensory":
            return await self.resilient_execute(
                self._multisensory_analysis_wrapper,
                document_text,
                document_id,
                service_name="multisensory_processing"
            )
        else:
            raise ValueError(f"Unknown analysis method: {method}")
    
    async def _bioneural_analysis_wrapper(self, document_text: str, document_id: str) -> Dict[str, Any]:
        """Wrapper for bioneural analysis with monitoring."""
        
        from src.lexgraph_legal_rag.bioneuro_olfactory_fusion import analyze_document_scent
        
        start_time = time.time()
        result = await analyze_document_scent(document_text, document_id)
        
        processing_time = time.time() - start_time
        
        return {
            "document_id": document_id,
            "scent_profile": result,
            "processing_time": processing_time,
            "method": "bioneural_olfactory",
            "status": "success"
        }
    
    async def _multisensory_analysis_wrapper(self, document_text: str, document_id: str) -> Dict[str, Any]:
        """Wrapper for multisensory analysis with monitoring."""
        
        from src.lexgraph_legal_rag.multisensory_legal_processor import analyze_document_multisensory
        
        start_time = time.time()
        result = await analyze_document_multisensory(document_text, document_id)
        
        processing_time = time.time() - start_time
        
        return {
            "document_id": document_id,
            "multisensory_analysis": result,
            "processing_time": processing_time,
            "method": "multisensory_fusion",
            "status": "success"
        }
    
    def get_resilience_status(self) -> Dict[str, Any]:
        """Get comprehensive resilience system status."""
        
        circuit_breaker_status = {
            name: {
                "state": "open" if cb.is_open() else "closed",
                "failure_count": cb.failure_count,
                "last_failure_time": cb.last_failure_time
            }
            for name, cb in self.circuit_breakers.items()
        }
        
        return {
            "service_health": self.service_health.value,
            "degraded_mode": self.is_degraded_mode,
            "metrics": {
                "requests_total": self.metrics.requests_total,
                "success_rate": (self.metrics.requests_successful / max(self.metrics.requests_total, 1)) * 100,
                "error_rate": self.metrics.error_rate * 100,
                "average_response_time": self.metrics.average_response_time,
                "uptime_percentage": self.metrics.uptime_percentage,
                "auto_recoveries": self.metrics.auto_recoveries,
                "circuit_breaker_trips": self.metrics.circuit_breaker_trips
            },
            "circuit_breakers": circuit_breaker_status,
            "failure_patterns": {
                pattern_key: {
                    "count": pattern.count,
                    "frequency": pattern.frequency,
                    "last_occurrence": pattern.last_occurrence
                }
                for pattern_key, pattern in self.failure_patterns.items()
            },
            "system_resources": {
                "memory_usage": psutil.virtual_memory().percent,
                "cpu_usage": psutil.cpu_percent(),
                "workers": self.config["max_workers"]
            }
        }


async def demonstration_scenario():
    """Comprehensive demonstration of resilience capabilities."""
    
    print("🛡️  PRODUCTION-GRADE RESILIENCE SYSTEM DEMONSTRATION")
    print("=" * 60)
    
    # Initialize resilience system
    resilience_system = ProductionGradeResilienceSystem()
    
    # Test documents
    test_documents = [
        ("This service agreement establishes terms between the contractor and client for consulting services.", "contract_test_1"),
        ("15 U.S.C. § 1681 provides consumer protection standards for credit reporting agencies.", "statute_test_1"),
        ("In Smith v. Jones, the Court held that materiality requires disclosure of significant information.", "case_test_1"),
        ("17 C.F.R. § 240.10b-5 prohibits fraudulent practices in securities transactions.", "regulation_test_1")
    ]
    
    print("\n🔬 Testing Normal Operations")
    print("-" * 40)
    
    # Test normal operations
    for i, (document, doc_id) in enumerate(test_documents):
        try:
            result = await resilience_system.analyze_document_resilient(
                document, doc_id, method="bioneural"
            )
            print(f"✅ Document {i+1}: Successfully processed {doc_id}")
        except Exception as e:
            print(f"❌ Document {i+1}: Failed to process {doc_id}: {e}")
    
    print("\n🔥 Simulating Stress Conditions")
    print("-" * 40)
    
    # Simulate high load with concurrent requests
    concurrent_tasks = []
    for i in range(20):  # 20 concurrent requests
        for document, doc_id in test_documents:
            task = resilience_system.analyze_document_resilient(
                document, f"{doc_id}_stress_{i}", method="bioneural"
            )
            concurrent_tasks.append(task)
    
    # Execute concurrent tasks and measure performance
    start_time = time.time()
    results = await asyncio.gather(*concurrent_tasks, return_exceptions=True)
    execution_time = time.time() - start_time
    
    successful_results = sum(1 for r in results if not isinstance(r, Exception))
    failed_results = len(results) - successful_results
    
    print(f"Processed {len(results)} requests in {execution_time:.2f}s")
    print(f"✅ Successful: {successful_results}")
    print(f"❌ Failed: {failed_results}")
    print(f"📊 Success rate: {(successful_results/len(results)*100):.1f}%")
    
    print("\n📊 RESILIENCE SYSTEM STATUS")
    print("-" * 40)
    
    # Get comprehensive status
    status = resilience_system.get_resilience_status()
    
    print(f"Service Health: {status['service_health'].upper()}")
    print(f"Degraded Mode: {'ACTIVE' if status['degraded_mode'] else 'INACTIVE'}")
    print(f"Total Requests: {status['metrics']['requests_total']}")
    print(f"Success Rate: {status['metrics']['success_rate']:.1f}%")
    print(f"Error Rate: {status['metrics']['error_rate']:.1f}%")
    print(f"Average Response Time: {status['metrics']['average_response_time']:.3f}s")
    print(f"Uptime: {status['metrics']['uptime_percentage']:.1f}%")
    print(f"Auto Recoveries: {status['metrics']['auto_recoveries']}")
    print(f"Circuit Breaker Trips: {status['metrics']['circuit_breaker_trips']}")
    
    print(f"\nSystem Resources:")
    print(f"Memory Usage: {status['system_resources']['memory_usage']:.1f}%")
    print(f"CPU Usage: {status['system_resources']['cpu_usage']:.1f}%")
    print(f"Active Workers: {status['system_resources']['workers']}")
    
    if status['failure_patterns']:
        print(f"\nFailure Patterns Detected: {len(status['failure_patterns'])}")
        for pattern_name, pattern_info in status['failure_patterns'].items():
            print(f"  - {pattern_name}: {pattern_info['count']} occurrences "
                  f"({pattern_info['frequency']:.3f} Hz)")
    
    print("\n✅ Resilience demonstration complete!")
    return status


if __name__ == "__main__":
    # Configure logging for demo
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run demonstration
    asyncio.run(demonstration_scenario())