#!/usr/bin/env python3
"""
🛡️ GENERATION 2: ENHANCED RESILIENCE SYSTEM
Advanced error recovery, circuit breakers, and fault tolerance
"""

import asyncio
import json
import time
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import random
from pathlib import Path

import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.ConsoleRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    context_class=dict,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class HealthStatus(Enum):
    """System health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    DOWN = "down"


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5
    recovery_timeout: int = 60
    success_threshold: int = 3
    request_timeout: float = 30.0


@dataclass
class ResilienceMetrics:
    """Resilience system metrics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    circuit_breaker_trips: int = 0
    average_response_time: float = 0.0
    recovery_attempts: int = 0
    last_failure_time: Optional[datetime] = None


class EnhancedCircuitBreaker:
    """Advanced circuit breaker with intelligent failure detection"""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.request_history: List[Dict[str, Any]] = []
        
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
                logger.info(f"Circuit breaker {self.name} transitioning to HALF_OPEN")
            else:
                raise Exception(f"Circuit breaker {self.name} is OPEN")
        
        start_time = time.time()
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                func(*args, **kwargs), 
                timeout=self.config.request_timeout
            )
            
            # Record success
            duration = time.time() - start_time
            self._record_success(duration)
            return result
            
        except Exception as e:
            # Record failure
            duration = time.time() - start_time
            self._record_failure(str(e), duration)
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        if self.last_failure_time is None:
            return True
        
        return datetime.now() - self.last_failure_time > timedelta(
            seconds=self.config.recovery_timeout
        )
    
    def _record_success(self, duration: float):
        """Record successful request"""
        self.request_history.append({
            'timestamp': datetime.now().isoformat(),
            'success': True,
            'duration': duration
        })
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                logger.info(f"Circuit breaker {self.name} CLOSED - recovered")
        
        # Keep only recent history
        self._trim_history()
    
    def _record_failure(self, error: str, duration: float):
        """Record failed request"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        self.request_history.append({
            'timestamp': self.last_failure_time.isoformat(),
            'success': False,
            'error': error,
            'duration': duration
        })
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            logger.warning(f"Circuit breaker {self.name} OPENED - failure threshold reached")
        
        self._trim_history()
    
    def _trim_history(self):
        """Keep only recent history entries"""
        cutoff_time = datetime.now() - timedelta(minutes=10)
        self.request_history = [
            entry for entry in self.request_history
            if datetime.fromisoformat(entry['timestamp']) > cutoff_time
        ]


class IntelligentRetryManager:
    """Advanced retry logic with exponential backoff and jitter"""
    
    def __init__(self, max_attempts: int = 5, base_delay: float = 1.0):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
    
    async def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with intelligent retry logic"""
        last_exception = None
        
        for attempt in range(self.max_attempts):
            try:
                result = await func(*args, **kwargs)
                if attempt > 0:
                    logger.info(f"Retry succeeded on attempt {attempt + 1}")
                return result
                
            except Exception as e:
                last_exception = e
                
                if attempt == self.max_attempts - 1:
                    logger.error(f"All retry attempts failed: {str(e)}")
                    break
                
                # Calculate delay with exponential backoff and jitter
                delay = self._calculate_delay(attempt)
                logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay:.2f}s: {str(e)}")
                await asyncio.sleep(delay)
        
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and jitter"""
        exponential_delay = self.base_delay * (2 ** attempt)
        jitter = random.uniform(0.1, 0.5) * exponential_delay
        return min(exponential_delay + jitter, 60.0)  # Max 60s delay


class HealthMonitor:
    """Comprehensive health monitoring system"""
    
    def __init__(self):
        self.health_checks: Dict[str, Callable] = {}
        self.health_status = HealthStatus.HEALTHY
        self.last_health_check: Optional[datetime] = None
        self.health_history: List[Dict[str, Any]] = []
    
    def register_health_check(self, name: str, check_func: Callable):
        """Register a health check function"""
        self.health_checks[name] = check_func
        logger.info(f"Registered health check: {name}")
    
    async def perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        start_time = time.time()
        results = {}
        overall_healthy = True
        
        for name, check_func in self.health_checks.items():
            try:
                result = await check_func()
                results[name] = {
                    'status': 'healthy' if result else 'unhealthy',
                    'details': result if isinstance(result, dict) else {'passed': result}
                }
                if not result:
                    overall_healthy = False
                    
            except Exception as e:
                results[name] = {
                    'status': 'error',
                    'error': str(e)
                }
                overall_healthy = False
        
        # Determine overall status
        if overall_healthy:
            self.health_status = HealthStatus.HEALTHY
        elif len([r for r in results.values() if r['status'] == 'healthy']) > len(results) / 2:
            self.health_status = HealthStatus.DEGRADED
        else:
            self.health_status = HealthStatus.CRITICAL
        
        duration = time.time() - start_time
        health_report = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': self.health_status.value,
            'checks': results,
            'check_duration': duration
        }
        
        self.health_history.append(health_report)
        self.last_health_check = datetime.now()
        self._trim_health_history()
        
        return health_report
    
    def _trim_health_history(self):
        """Keep only recent health history"""
        cutoff_time = datetime.now() - timedelta(hours=1)
        self.health_history = [
            entry for entry in self.health_history
            if datetime.fromisoformat(entry['timestamp']) > cutoff_time
        ]


class ResilienceSystem:
    """Main resilience system coordinator"""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, EnhancedCircuitBreaker] = {}
        self.retry_manager = IntelligentRetryManager()
        self.health_monitor = HealthMonitor()
        self.metrics = ResilienceMetrics()
        self.error_patterns: Dict[str, int] = {}
        
        # Setup health checks
        self._setup_health_checks()
    
    def create_circuit_breaker(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> EnhancedCircuitBreaker:
        """Create and register circuit breaker"""
        config = config or CircuitBreakerConfig()
        circuit_breaker = EnhancedCircuitBreaker(name, config)
        self.circuit_breakers[name] = circuit_breaker
        logger.info(f"Created circuit breaker: {name}")
        return circuit_breaker
    
    async def execute_with_resilience(
        self, 
        name: str, 
        func: Callable, 
        *args, 
        use_retry: bool = True,
        **kwargs
    ) -> Any:
        """Execute function with full resilience features"""
        circuit_breaker = self.circuit_breakers.get(name)
        if not circuit_breaker:
            circuit_breaker = self.create_circuit_breaker(name)
        
        async def _execute():
            return await circuit_breaker.call(func, *args, **kwargs)
        
        start_time = time.time()
        try:
            if use_retry:
                result = await self.retry_manager.execute_with_retry(_execute)
            else:
                result = await _execute()
            
            # Update metrics
            self.metrics.total_requests += 1
            self.metrics.successful_requests += 1
            duration = time.time() - start_time
            self._update_average_response_time(duration)
            
            return result
            
        except Exception as e:
            # Update metrics and error patterns
            self.metrics.total_requests += 1
            self.metrics.failed_requests += 1
            self.metrics.last_failure_time = datetime.now()
            
            error_type = type(e).__name__
            self.error_patterns[error_type] = self.error_patterns.get(error_type, 0) + 1
            
            logger.error(f"Resilience execution failed for {name}: {str(e)}")
            raise
    
    def _update_average_response_time(self, duration: float):
        """Update rolling average response time"""
        if self.metrics.average_response_time == 0:
            self.metrics.average_response_time = duration
        else:
            # Exponential moving average
            self.metrics.average_response_time = (
                0.9 * self.metrics.average_response_time + 0.1 * duration
            )
    
    def _setup_health_checks(self):
        """Setup default health checks"""
        
        async def circuit_breaker_health():
            """Check circuit breaker status"""
            open_breakers = [
                name for name, cb in self.circuit_breakers.items()
                if cb.state == CircuitBreakerState.OPEN
            ]
            return {
                'open_circuit_breakers': open_breakers,
                'healthy': len(open_breakers) == 0
            }
        
        async def error_rate_health():
            """Check overall error rate"""
            if self.metrics.total_requests == 0:
                return {'error_rate': 0.0, 'healthy': True}
            
            error_rate = self.metrics.failed_requests / self.metrics.total_requests
            return {
                'error_rate': error_rate,
                'healthy': error_rate < 0.1  # Less than 10% error rate
            }
        
        self.health_monitor.register_health_check('circuit_breakers', circuit_breaker_health)
        self.health_monitor.register_health_check('error_rate', error_rate_health)
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        health_report = await self.health_monitor.perform_health_check()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'health': health_report,
            'metrics': asdict(self.metrics),
            'circuit_breakers': {
                name: {
                    'state': cb.state.value,
                    'failure_count': cb.failure_count,
                    'success_count': cb.success_count
                }
                for name, cb in self.circuit_breakers.items()
            },
            'error_patterns': self.error_patterns
        }


async def demo_resilience_system():
    """Demonstrate resilience system capabilities"""
    print("🛡️ GENERATION 2: ENHANCED RESILIENCE SYSTEM")
    print("=" * 60)
    print("Advanced fault tolerance and error recovery")
    print()
    
    resilience = ResilienceSystem()
    
    # Create some test functions
    failure_count = 0
    
    async def unreliable_service():
        """Simulates an unreliable service"""
        nonlocal failure_count
        failure_count += 1
        
        if failure_count <= 3:
            raise Exception(f"Service failure #{failure_count}")
        return f"Success after {failure_count} attempts"
    
    async def timeout_service():
        """Simulates a service that times out"""
        await asyncio.sleep(35)  # Longer than timeout
        return "Should not reach here"
    
    # Test resilient execution
    print("🔄 Testing resilient execution with retries...")
    try:
        result = await resilience.execute_with_resilience(
            "unreliable_service",
            unreliable_service
        )
        print(f"✅ Success: {result}")
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    print()
    print("⏱️ Testing circuit breaker with timeout...")
    try:
        result = await resilience.execute_with_resilience(
            "timeout_service",
            timeout_service,
            use_retry=False
        )
        print(f"✅ Success: {result}")
    except Exception as e:
        print(f"❌ Expected failure: {e}")
    
    # Generate more failures to trip circuit breaker
    print()
    print("🔌 Testing circuit breaker trips...")
    for i in range(6):
        try:
            await resilience.execute_with_resilience(
                "timeout_service",
                timeout_service,
                use_retry=False
            )
        except Exception:
            pass
    
    # Get system status
    print()
    print("📊 System Status:")
    status = await resilience.get_system_status()
    print(json.dumps(status, indent=2, default=str))
    
    # Save results
    results_file = Path("enhanced_resilience_g2_results.json")
    with open(results_file, 'w') as f:
        json.dump(status, f, indent=2, default=str)
    
    print(f"\n📄 Results saved to {results_file}")
    print("\n🎉 Enhanced resilience system demonstration complete!")


if __name__ == "__main__":
    asyncio.run(demo_resilience_system())