#!/usr/bin/env python3
"""Error recovery and resilience system for bioneural olfactory fusion."""

import sys
import logging
import traceback
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

class ErrorSeverity(Enum):
    """Classification of error severity levels."""
    LOW = "low"           # Minor issues, system can continue
    MEDIUM = "medium"     # Significant issues, degraded functionality
    HIGH = "high"         # Critical issues, system instability
    CRITICAL = "critical" # System failure, immediate intervention needed

class RecoveryStrategy(Enum):
    """Available recovery strategies."""
    RETRY = "retry"                    # Retry the operation
    FALLBACK = "fallback"             # Use fallback implementation
    GRACEFUL_DEGRADATION = "degrade"  # Reduce functionality
    CIRCUIT_BREAKER = "circuit_break" # Stop operations temporarily
    RESTART = "restart"               # Restart component

@dataclass
class ErrorContext:
    """Context information for error analysis."""
    error_type: str
    error_message: str
    severity: ErrorSeverity
    component: str
    timestamp: float
    stack_trace: str
    recovery_strategy: RecoveryStrategy
    metadata: Dict[str, Any]

class BioneuroErrorRecovery:
    """Intelligent error recovery system for bioneural olfactory fusion."""
    
    def __init__(self):
        self.error_history: List[ErrorContext] = []
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        self.fallback_handlers: Dict[str, Callable] = {}
        self.max_retries = 3
        self.retry_delay = 1.0
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Register default fallback handlers
        self._register_default_fallbacks()
    
    def _register_default_fallbacks(self):
        """Register default fallback implementations."""
        
        def simple_text_fallback(text: str) -> Dict[str, Any]:
            """Basic text analysis when bioneural system fails."""
            return {
                'analysis_method': 'basic_fallback',
                'word_count': len(text.split()),
                'character_count': len(text),
                'has_legal_terms': any(term in text.lower() for term in ['contract', 'agreement', 'liable']),
                'confidence': 0.3,  # Low confidence for fallback
                'warning': 'Using fallback analysis due to system error'
            }
        
        def empty_scent_profile() -> Dict[str, Any]:
            """Empty scent profile when olfactory analysis fails."""
            return {
                'primary_scent': 'unknown',
                'intensity': 0.0,
                'receptor_activations': {
                    'legal_complexity': 0.0,
                    'statutory_authority': 0.0,
                    'temporal_freshness': 0.0,
                    'citation_density': 0.0,
                    'risk_profile': 0.0,
                    'semantic_coherence': 0.0
                },
                'document_signature': 'fallback_0000',
                'error_recovery': True
            }
        
        self.fallback_handlers['text_analysis'] = simple_text_fallback
        self.fallback_handlers['scent_profile'] = empty_scent_profile
    
    def classify_error(self, error: Exception, component: str) -> ErrorContext:
        """Classify an error and determine recovery strategy."""
        
        error_type = type(error).__name__
        error_message = str(error)
        stack_trace = traceback.format_exc()
        
        # Determine severity based on error type and component
        severity = ErrorSeverity.LOW
        recovery_strategy = RecoveryStrategy.RETRY
        
        if error_type in ['ImportError', 'ModuleNotFoundError']:
            severity = ErrorSeverity.HIGH
            recovery_strategy = RecoveryStrategy.FALLBACK
        elif error_type in ['MemoryError', 'SystemExit']:
            severity = ErrorSeverity.CRITICAL
            recovery_strategy = RecoveryStrategy.RESTART
        elif error_type in ['TypeError', 'ValueError']:
            severity = ErrorSeverity.MEDIUM
            recovery_strategy = RecoveryStrategy.GRACEFUL_DEGRADATION
        elif error_type in ['ConnectionError', 'TimeoutError']:
            severity = ErrorSeverity.MEDIUM
            recovery_strategy = RecoveryStrategy.CIRCUIT_BREAKER
        
        # Check error frequency for circuit breaking
        recent_errors = [
            e for e in self.error_history
            if e.component == component and time.time() - e.timestamp < 300  # Last 5 minutes
        ]
        
        if len(recent_errors) > 5:  # Too many recent errors
            severity = ErrorSeverity.HIGH
            recovery_strategy = RecoveryStrategy.CIRCUIT_BREAKER
        
        context = ErrorContext(
            error_type=error_type,
            error_message=error_message,
            severity=severity,
            component=component,
            timestamp=time.time(),
            stack_trace=stack_trace,
            recovery_strategy=recovery_strategy,
            metadata={'retry_count': 0}
        )
        
        self.error_history.append(context)
        return context
    
    def execute_with_recovery(self, func: Callable, component: str, *args, **kwargs) -> Any:
        """Execute a function with automatic error recovery."""
        
        # Check circuit breaker
        if self._is_circuit_open(component):
            self.logger.warning(f"Circuit breaker open for {component}, using fallback")
            return self._execute_fallback(component, *args, **kwargs)
        
        retry_count = 0
        last_error = None
        
        while retry_count <= self.max_retries:
            try:
                result = func(*args, **kwargs)
                self._reset_circuit_breaker(component)
                return result
                
            except Exception as error:
                last_error = error
                context = self.classify_error(error, component)
                context.metadata['retry_count'] = retry_count
                
                self.logger.error(f"Error in {component}: {context.error_message}")
                
                if context.recovery_strategy == RecoveryStrategy.RETRY and retry_count < self.max_retries:
                    retry_count += 1
                    time.sleep(self.retry_delay * retry_count)  # Exponential backoff
                    continue
                    
                elif context.recovery_strategy == RecoveryStrategy.FALLBACK:
                    self.logger.info(f"Using fallback for {component}")
                    return self._execute_fallback(component, *args, **kwargs)
                    
                elif context.recovery_strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
                    self.logger.info(f"Graceful degradation for {component}")
                    return self._execute_degraded(component, *args, **kwargs)
                    
                elif context.recovery_strategy == RecoveryStrategy.CIRCUIT_BREAKER:
                    self._open_circuit_breaker(component)
                    return self._execute_fallback(component, *args, **kwargs)
                    
                else:
                    # Re-raise for critical errors
                    raise error
        
        # If all retries failed, use fallback
        self.logger.error(f"All retries failed for {component}, using fallback")
        return self._execute_fallback(component, *args, **kwargs)
    
    def _is_circuit_open(self, component: str) -> bool:
        """Check if circuit breaker is open for a component."""
        if component not in self.circuit_breakers:
            return False
            
        breaker = self.circuit_breakers[component]
        if time.time() - breaker['opened_at'] > breaker['timeout']:
            self._reset_circuit_breaker(component)
            return False
            
        return True
    
    def _open_circuit_breaker(self, component: str):
        """Open circuit breaker for a component."""
        self.circuit_breakers[component] = {
            'opened_at': time.time(),
            'timeout': 60.0,  # 1 minute timeout
            'failure_count': self.circuit_breakers.get(component, {}).get('failure_count', 0) + 1
        }
    
    def _reset_circuit_breaker(self, component: str):
        """Reset circuit breaker for a component."""
        if component in self.circuit_breakers:
            del self.circuit_breakers[component]
    
    def _execute_fallback(self, component: str, *args, **kwargs) -> Any:
        """Execute fallback implementation."""
        if component in self.fallback_handlers:
            return self.fallback_handlers[component](*args, **kwargs)
        else:
            return {'error': f'No fallback available for {component}', 'status': 'failed'}
    
    def _execute_degraded(self, component: str, *args, **kwargs) -> Any:
        """Execute with degraded functionality."""
        # For now, same as fallback
        return self._execute_fallback(component, *args, **kwargs)
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health metrics."""
        
        # Analyze recent errors
        recent_errors = [
            e for e in self.error_history
            if time.time() - e.timestamp < 3600  # Last hour
        ]
        
        error_rate = len(recent_errors) / 60 if recent_errors else 0  # Errors per minute
        
        # Component health
        component_health = {}
        for component in set(e.component for e in recent_errors):
            component_errors = [e for e in recent_errors if e.component == component]
            component_health[component] = {
                'error_count': len(component_errors),
                'circuit_open': self._is_circuit_open(component),
                'last_error': component_errors[-1].error_message if component_errors else None
            }
        
        return {
            'overall_health': 'healthy' if error_rate < 0.1 else 'degraded' if error_rate < 0.5 else 'unhealthy',
            'error_rate': error_rate,
            'active_circuit_breakers': len([c for c in self.circuit_breakers if self._is_circuit_open(c)]),
            'component_health': component_health,
            'total_errors_last_hour': len(recent_errors)
        }

# Global error recovery instance
_error_recovery = BioneuroErrorRecovery()

def get_error_recovery() -> BioneuroErrorRecovery:
    """Get the global error recovery instance."""
    return _error_recovery

def safe_execute(component: str):
    """Decorator for safe execution with error recovery."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            return _error_recovery.execute_with_recovery(func, component, *args, **kwargs)
        return wrapper
    return decorator

def demonstrate_error_recovery():
    """Demonstrate the error recovery system."""
    
    print("🛡️ Bioneural Error Recovery System Demo")
    print("=" * 50)
    
    recovery = get_error_recovery()
    
    # Test normal operation
    @safe_execute('text_analysis')
    def working_function(text: str):
        return {'result': f'Analyzed: {len(text)} characters', 'status': 'success'}
    
    result = working_function("This is a test document")
    print(f"✅ Normal operation: {result}")
    
    # Test error recovery
    @safe_execute('failing_component')  
    def failing_function(text: str):
        raise ValueError("Simulated processing error")
    
    result = failing_function("This will fail")
    print(f"🔄 Error recovery: {result}")
    
    # Test circuit breaker
    for i in range(7):  # Trigger circuit breaker
        result = failing_function(f"Failure attempt {i+1}")
    
    print(f"⚡ Circuit breaker: {result}")
    
    # System health
    health = recovery.get_system_health()
    print(f"\n📊 System Health: {health['overall_health']}")
    print(f"   Error rate: {health['error_rate']:.2f}/min")
    print(f"   Circuit breakers: {health['active_circuit_breakers']}")
    
    return recovery

if __name__ == "__main__":
    demonstrate_error_recovery()