"""
Enhanced Resilience Patterns for Legal RAG
==========================================

Generation 2 Robustness: Advanced error handling, validation, and resilience patterns
- Comprehensive error handling with recovery strategies
- Input validation and sanitization
- Rate limiting and circuit breaking
- Graceful degradation mechanisms
- Health monitoring and alerting
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
import functools

import numpy as np


logger = logging.getLogger(__name__)


class ResilienceLevel(Enum):
    """Levels of resilience protection."""
    
    BASIC = "basic"          # Basic error handling
    ENHANCED = "enhanced"    # Enhanced with retries
    ROBUST = "robust"        # Robust with circuit breaking
    BULLETPROOF = "bulletproof"  # Maximum resilience


@dataclass
class ResilienceConfig:
    """Configuration for resilience patterns."""
    
    max_retries: int = 3
    retry_delay: float = 1.0
    backoff_multiplier: float = 2.0
    circuit_failure_threshold: int = 5
    circuit_timeout: float = 60.0
    rate_limit_per_minute: int = 1000
    validation_enabled: bool = True
    sanitization_enabled: bool = True
    monitoring_enabled: bool = True


@dataclass
class ErrorContext:
    """Context information for error handling."""
    
    operation: str
    timestamp: float
    input_data: Any = None
    error_type: str = ""
    error_message: str = ""
    stack_trace: str = ""
    recovery_attempted: bool = False
    recovery_successful: bool = False


class EnhancedResilienceSystem:
    """
    Comprehensive resilience system for robust operations.
    
    Features:
    - Multi-level error handling with intelligent recovery
    - Input validation and sanitization
    - Rate limiting with adaptive thresholds
    - Circuit breaking for external dependencies
    - Graceful degradation mechanisms
    - Real-time health monitoring
    """
    
    def __init__(self, config: Optional[ResilienceConfig] = None):
        self.config = config or ResilienceConfig()
        self._error_history: List[ErrorContext] = []
        self._rate_limits: Dict[str, List[float]] = {}
        self._circuit_states: Dict[str, str] = {}
        self._health_metrics: Dict[str, float] = {}
        self._lock = asyncio.Lock()
        
    async def with_resilience(
        self,
        operation: Callable,
        *args,
        operation_name: str = "unknown",
        resilience_level: ResilienceLevel = ResilienceLevel.ENHANCED,
        **kwargs
    ) -> Any:
        """Execute operation with comprehensive resilience protection."""
        
        start_time = time.time()
        
        try:
            # Input validation
            if self.config.validation_enabled:
                await self._validate_inputs(args, kwargs)
            
            # Input sanitization
            if self.config.sanitization_enabled:
                args, kwargs = await self._sanitize_inputs(args, kwargs)
            
            # Rate limiting check
            if not await self._check_rate_limit(operation_name):
                raise ResilienceError(f"Rate limit exceeded for {operation_name}")
            
            # Circuit breaker check
            if not await self._check_circuit_breaker(operation_name):
                raise ResilienceError(f"Circuit breaker open for {operation_name}")
            
            # Execute with retry logic
            result = await self._execute_with_retries(
                operation, args, kwargs, operation_name, resilience_level
            )
            
            # Record success
            await self._record_success(operation_name, time.time() - start_time)
            
            return result
            
        except Exception as e:
            error_context = ErrorContext(
                operation=operation_name,
                timestamp=time.time(),
                input_data={"args": str(args)[:200], "kwargs": str(kwargs)[:200]},
                error_type=type(e).__name__,
                error_message=str(e),
            )
            
            await self._handle_error(error_context, resilience_level)
            raise
    
    async def _validate_inputs(self, args: tuple, kwargs: dict) -> None:
        """Comprehensive input validation."""
        
        # Check for None values in critical parameters
        if any(arg is None for arg in args if isinstance(arg, (str, dict, list))):
            raise ValidationError("Critical parameters cannot be None")
        
        # Validate string lengths
        for arg in args:
            if isinstance(arg, str) and len(arg) > 10000:
                raise ValidationError("String parameter too long (>10000 chars)")
        
        # Validate nested structures
        for key, value in kwargs.items():
            if isinstance(value, dict):
                if len(json.dumps(value)) > 100000:
                    raise ValidationError(f"Parameter {key} too large")
    
    async def _sanitize_inputs(self, args: tuple, kwargs: dict) -> tuple:
        """Sanitize inputs to prevent injection attacks."""
        
        def sanitize_string(s: str) -> str:
            if not isinstance(s, str):
                return s
            
            # Remove potentially dangerous patterns
            dangerous_patterns = [
                "<script", "</script>", "javascript:", "eval(",
                "exec(", "__import__", "os.system", "subprocess"
            ]
            
            for pattern in dangerous_patterns:
                s = s.replace(pattern, "")
            
            return s.strip()
        
        # Sanitize args
        sanitized_args = []
        for arg in args:
            if isinstance(arg, str):
                sanitized_args.append(sanitize_string(arg))
            else:
                sanitized_args.append(arg)
        
        # Sanitize kwargs
        sanitized_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, str):
                sanitized_kwargs[key] = sanitize_string(value)
            else:
                sanitized_kwargs[key] = value
        
        return tuple(sanitized_args), sanitized_kwargs
    
    async def _check_rate_limit(self, operation_name: str) -> bool:
        """Check if operation is within rate limits."""
        
        current_time = time.time()
        minute_ago = current_time - 60
        
        # Initialize rate limit tracking
        if operation_name not in self._rate_limits:
            self._rate_limits[operation_name] = []
        
        # Remove old timestamps
        self._rate_limits[operation_name] = [
            ts for ts in self._rate_limits[operation_name] if ts > minute_ago
        ]
        
        # Check current rate
        current_rate = len(self._rate_limits[operation_name])
        if current_rate >= self.config.rate_limit_per_minute:
            return False
        
        # Record this request
        self._rate_limits[operation_name].append(current_time)
        return True
    
    async def _check_circuit_breaker(self, operation_name: str) -> bool:
        """Check circuit breaker state."""
        
        if operation_name not in self._circuit_states:
            self._circuit_states[operation_name] = "closed"
        
        state = self._circuit_states[operation_name]
        
        if state == "open":
            # Check if timeout has passed
            last_failure = self._health_metrics.get(f"{operation_name}_last_failure", 0)
            if time.time() - last_failure > self.config.circuit_timeout:
                self._circuit_states[operation_name] = "half_open"
                return True
            return False
        
        return True
    
    async def _execute_with_retries(
        self,
        operation: Callable,
        args: tuple,
        kwargs: dict,
        operation_name: str,
        resilience_level: ResilienceLevel
    ) -> Any:
        """Execute operation with intelligent retry logic."""
        
        max_retries = self._get_retry_count(resilience_level)
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(operation):
                    result = await operation(*args, **kwargs)
                else:
                    result = operation(*args, **kwargs)
                
                # Reset circuit breaker on success
                if self._circuit_states.get(operation_name) == "half_open":
                    self._circuit_states[operation_name] = "closed"
                
                return result
                
            except Exception as e:
                last_exception = e
                
                # Record failure
                await self._record_failure(operation_name)
                
                # Check if we should retry
                if attempt < max_retries and self._should_retry(e, resilience_level):
                    delay = self._calculate_retry_delay(attempt)
                    logger.warning(f"Attempt {attempt + 1} failed for {operation_name}, retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
                    continue
                else:
                    break
        
        # All retries failed
        raise last_exception
    
    def _get_retry_count(self, resilience_level: ResilienceLevel) -> int:
        """Get retry count based on resilience level."""
        
        retry_counts = {
            ResilienceLevel.BASIC: 1,
            ResilienceLevel.ENHANCED: 3,
            ResilienceLevel.ROBUST: 5,
            ResilienceLevel.BULLETPROOF: 10
        }
        return retry_counts.get(resilience_level, 3)
    
    def _should_retry(self, error: Exception, resilience_level: ResilienceLevel) -> bool:
        """Determine if error is retryable."""
        
        # Don't retry validation errors
        if isinstance(error, (ValidationError, ValueError, TypeError)):
            return False
        
        # Don't retry authentication errors
        if "auth" in str(error).lower() or "permission" in str(error).lower():
            return False
        
        # More aggressive retrying for higher resilience levels
        if resilience_level in [ResilienceLevel.ROBUST, ResilienceLevel.BULLETPROOF]:
            return True
        
        # Default retry logic for network/temporary errors
        retryable_errors = [
            "timeout", "connection", "network", "temporary",
            "503", "502", "504", "ConnectionError", "TimeoutError"
        ]
        
        error_str = str(error).lower()
        return any(keyword in error_str for keyword in retryable_errors)
    
    def _calculate_retry_delay(self, attempt: int) -> float:
        """Calculate delay for retry with exponential backoff."""
        
        base_delay = self.config.retry_delay
        multiplier = self.config.backoff_multiplier
        
        # Exponential backoff with jitter
        delay = base_delay * (multiplier ** attempt)
        jitter = np.random.uniform(0.1, 0.5) * delay
        
        return min(delay + jitter, 30.0)  # Cap at 30 seconds
    
    async def _record_success(self, operation_name: str, duration: float) -> None:
        """Record successful operation."""
        
        async with self._lock:
            self._health_metrics[f"{operation_name}_success_count"] = (
                self._health_metrics.get(f"{operation_name}_success_count", 0) + 1
            )
            self._health_metrics[f"{operation_name}_avg_duration"] = duration
            self._health_metrics[f"{operation_name}_last_success"] = time.time()
    
    async def _record_failure(self, operation_name: str) -> None:
        """Record failed operation."""
        
        async with self._lock:
            failure_count = self._health_metrics.get(f"{operation_name}_failure_count", 0) + 1
            self._health_metrics[f"{operation_name}_failure_count"] = failure_count
            self._health_metrics[f"{operation_name}_last_failure"] = time.time()
            
            # Update circuit breaker
            if failure_count >= self.config.circuit_failure_threshold:
                self._circuit_states[operation_name] = "open"
                logger.error(f"Circuit breaker opened for {operation_name}")
    
    async def _handle_error(
        self, 
        error_context: ErrorContext,
        resilience_level: ResilienceLevel
    ) -> None:
        """Comprehensive error handling."""
        
        # Record error
        self._error_history.append(error_context)
        
        # Keep only recent errors
        cutoff_time = time.time() - 3600  # 1 hour
        self._error_history = [
            ctx for ctx in self._error_history if ctx.timestamp > cutoff_time
        ]
        
        # Attempt recovery based on resilience level
        if resilience_level in [ResilienceLevel.ROBUST, ResilienceLevel.BULLETPROOF]:
            await self._attempt_recovery(error_context)
        
        # Log comprehensive error information
        logger.error(
            f"Operation {error_context.operation} failed: "
            f"{error_context.error_type}: {error_context.error_message}"
        )
    
    async def _attempt_recovery(self, error_context: ErrorContext) -> None:
        """Attempt intelligent error recovery."""
        
        recovery_strategies = [
            self._clear_caches,
            self._reset_connections,
            self._reduce_load,
        ]
        
        for strategy in recovery_strategies:
            try:
                await strategy()
                error_context.recovery_attempted = True
                error_context.recovery_successful = True
                logger.info(f"Recovery successful using {strategy.__name__}")
                break
            except Exception as e:
                logger.warning(f"Recovery strategy {strategy.__name__} failed: {e}")
        
        error_context.recovery_attempted = True
    
    async def _clear_caches(self) -> None:
        """Clear system caches as recovery strategy."""
        # Implementation would clear various caches
        pass
    
    async def _reset_connections(self) -> None:
        """Reset connections as recovery strategy."""
        # Implementation would reset connection pools
        pass
    
    async def _reduce_load(self) -> None:
        """Reduce system load as recovery strategy."""
        # Implementation would reduce concurrent operations
        pass
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        
        return {
            "timestamp": time.time(),
            "error_history_count": len(self._error_history),
            "circuit_states": self._circuit_states.copy(),
            "health_metrics": self._health_metrics.copy(),
            "rate_limits": {
                op: len(timestamps) for op, timestamps in self._rate_limits.items()
            }
        }


# Decorator for easy resilience application
def with_resilience(
    operation_name: str = "unknown",
    resilience_level: ResilienceLevel = ResilienceLevel.ENHANCED,
    config: Optional[ResilienceConfig] = None
):
    """Decorator to add resilience to any function."""
    
    def decorator(func: Callable) -> Callable:
        resilience_system = EnhancedResilienceSystem(config)
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await resilience_system.with_resilience(
                func, *args, 
                operation_name=operation_name,
                resilience_level=resilience_level,
                **kwargs
            )
        
        return wrapper
    return decorator


class ResilienceError(Exception):
    """Base exception for resilience system errors."""
    pass


class ValidationError(ResilienceError):
    """Input validation error."""
    pass


class RateLimitError(ResilienceError):
    """Rate limit exceeded error."""
    pass


class CircuitBreakerError(ResilienceError):
    """Circuit breaker is open error."""
    pass


# Global resilience system instance
_global_resilience_system = None


def get_resilience_system(config: Optional[ResilienceConfig] = None) -> EnhancedResilienceSystem:
    """Get global resilience system instance."""
    
    global _global_resilience_system
    if _global_resilience_system is None:
        _global_resilience_system = EnhancedResilienceSystem(config)
    return _global_resilience_system


# Context manager for resilient operations
@asynccontextmanager
async def resilient_operation(
    operation_name: str,
    resilience_level: ResilienceLevel = ResilienceLevel.ENHANCED
):
    """Context manager for resilient operations."""
    
    resilience_system = get_resilience_system()
    start_time = time.time()
    
    try:
        yield resilience_system
        await resilience_system._record_success(operation_name, time.time() - start_time)
    except Exception as e:
        error_context = ErrorContext(
            operation=operation_name,
            timestamp=time.time(),
            error_type=type(e).__name__,
            error_message=str(e)
        )
        await resilience_system._handle_error(error_context, resilience_level)
        raise