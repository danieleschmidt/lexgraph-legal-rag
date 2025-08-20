"""Advanced resilience and error handling for legal RAG system.

This module provides comprehensive error handling, retry mechanisms,
circuit breakers, and system recovery capabilities.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import random
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any
from typing import Callable


logger = logging.getLogger(__name__)


class ErrorCategory(Enum):
    """Categories of errors for intelligent handling."""

    TRANSIENT = "transient"  # Temporary errors that may resolve
    PERMANENT = "permanent"  # Permanent errors that won't resolve
    TIMEOUT = "timeout"  # Timeout-related errors
    RESOURCE = "resource"  # Resource exhaustion errors
    VALIDATION = "validation"  # Input validation errors
    SECURITY = "security"  # Security-related errors
    EXTERNAL = "external"  # External service errors
    SYSTEM = "system"  # System/infrastructure errors


class RecoveryStrategy(Enum):
    """Recovery strategies for different error types."""

    RETRY = "retry"  # Retry with backoff
    FALLBACK = "fallback"  # Use fallback mechanism
    CIRCUIT_BREAK = "circuit_break"  # Open circuit breaker
    GRACEFUL_DEGRADE = "graceful_degrade"  # Reduce functionality
    FAIL_FAST = "fail_fast"  # Fail immediately
    QUEUE_FOR_RETRY = "queue_for_retry"  # Queue for later retry


@dataclass
class ErrorPattern:
    """Error pattern for classification and handling."""

    error_types: list[type[Exception]]
    keywords: list[str]
    category: ErrorCategory
    strategy: RecoveryStrategy
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    jitter: bool = True

    def matches(self, error: Exception) -> bool:
        """Check if error matches this pattern."""
        # Check exception type
        if type(error) in self.error_types:
            return True

        # Check keywords in error message
        error_msg = str(error).lower()
        return any(keyword.lower() in error_msg for keyword in self.keywords)


@dataclass
class CircuitBreakerState:
    """Circuit breaker state tracking."""

    name: str
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 3

    # Current state
    is_open: bool = False
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float = 0.0
    last_success_time: float = 0.0

    # Statistics
    total_calls: int = 0
    total_failures: int = 0
    total_successes: int = 0

    def should_attempt(self) -> bool:
        """Check if circuit should attempt operation."""
        if not self.is_open:
            return True

        # Check if recovery timeout has passed
        if time.time() - self.last_failure_time >= self.recovery_timeout:
            logger.info(f"Circuit breaker {self.name} attempting recovery")
            return True

        return False

    def record_success(self) -> None:
        """Record successful operation."""
        self.total_calls += 1
        self.total_successes += 1
        self.success_count += 1
        self.last_success_time = time.time()

        # Close circuit if we have enough successes
        if self.is_open and self.success_count >= self.success_threshold:
            self.is_open = False
            self.failure_count = 0
            self.success_count = 0
            logger.info(f"Circuit breaker {self.name} closed after recovery")

    def record_failure(self) -> None:
        """Record failed operation."""
        self.total_calls += 1
        self.total_failures += 1
        self.failure_count += 1
        self.success_count = 0  # Reset success count
        self.last_failure_time = time.time()

        # Open circuit if failure threshold exceeded
        if not self.is_open and self.failure_count >= self.failure_threshold:
            self.is_open = True
            logger.warning(
                f"Circuit breaker {self.name} opened after {self.failure_count} failures"
            )

    def get_stats(self) -> dict[str, Any]:
        """Get circuit breaker statistics."""
        success_rate = self.total_successes / max(self.total_calls, 1)
        return {
            "name": self.name,
            "is_open": self.is_open,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "total_calls": self.total_calls,
            "success_rate": success_rate,
            "last_failure_time": self.last_failure_time,
            "recovery_timeout": self.recovery_timeout,
        }


class ResilienceManager:
    """Advanced resilience management system."""

    def __init__(self):
        self.error_patterns = self._initialize_error_patterns()
        self.circuit_breakers: dict[str, CircuitBreakerState] = {}
        self.error_history: deque = deque(maxlen=1000)
        self.fallback_handlers: dict[str, Callable] = {}
        self._lock = threading.RLock()

        logger.info("Resilience manager initialized")

    def _initialize_error_patterns(self) -> list[ErrorPattern]:
        """Initialize error patterns for classification."""
        return [
            # Transient errors
            ErrorPattern(
                error_types=[ConnectionError, asyncio.TimeoutError],
                keywords=["connection refused", "timeout", "temporary", "unavailable"],
                category=ErrorCategory.TRANSIENT,
                strategy=RecoveryStrategy.RETRY,
                max_retries=5,
                base_delay=1.0,
                max_delay=30.0,
            ),
            # Resource exhaustion
            ErrorPattern(
                error_types=[MemoryError],
                keywords=["memory", "resource", "exhausted", "out of space"],
                category=ErrorCategory.RESOURCE,
                strategy=RecoveryStrategy.GRACEFUL_DEGRADE,
                max_retries=1,
                base_delay=5.0,
            ),
            # Validation errors
            ErrorPattern(
                error_types=[ValueError, TypeError],
                keywords=["invalid", "validation", "format", "type"],
                category=ErrorCategory.VALIDATION,
                strategy=RecoveryStrategy.FAIL_FAST,
                max_retries=0,
            ),
            # Security errors
            ErrorPattern(
                error_types=[
                    PermissionError,
                    (
                        SecurityError
                        if "SecurityError" in dir(__builtins__)
                        else Exception
                    ),
                ],
                keywords=["permission", "unauthorized", "forbidden", "security"],
                category=ErrorCategory.SECURITY,
                strategy=RecoveryStrategy.FAIL_FAST,
                max_retries=0,
            ),
            # External service errors
            ErrorPattern(
                error_types=[],
                keywords=[
                    "api error",
                    "service unavailable",
                    "rate limit",
                    "quota exceeded",
                ],
                category=ErrorCategory.EXTERNAL,
                strategy=RecoveryStrategy.CIRCUIT_BREAK,
                max_retries=3,
                base_delay=2.0,
                max_delay=120.0,
            ),
            # Permanent errors
            ErrorPattern(
                error_types=[FileNotFoundError, ImportError],
                keywords=["not found", "missing", "does not exist"],
                category=ErrorCategory.PERMANENT,
                strategy=RecoveryStrategy.FALLBACK,
                max_retries=1,
            ),
        ]

    def classify_error(self, error: Exception) -> ErrorPattern:
        """Classify error and determine recovery strategy."""
        for pattern in self.error_patterns:
            if pattern.matches(error):
                return pattern

        # Default pattern for unclassified errors
        return ErrorPattern(
            error_types=[Exception],
            keywords=[],
            category=ErrorCategory.SYSTEM,
            strategy=RecoveryStrategy.RETRY,
            max_retries=2,
            base_delay=1.0,
        )

    def get_circuit_breaker(self, name: str, **kwargs) -> CircuitBreakerState:
        """Get or create circuit breaker."""
        with self._lock:
            if name not in self.circuit_breakers:
                self.circuit_breakers[name] = CircuitBreakerState(name=name, **kwargs)
            return self.circuit_breakers[name]

    def register_fallback(self, operation_name: str, fallback_func: Callable) -> None:
        """Register fallback function for operation."""
        self.fallback_handlers[operation_name] = fallback_func
        logger.info(f"Registered fallback handler for {operation_name}")

    async def execute_with_resilience(
        self, func: Callable, operation_name: str, *args, **kwargs
    ) -> Any:
        """Execute function with comprehensive resilience handling."""
        pattern = None
        last_error = None

        # Get or create circuit breaker
        circuit_breaker = self.get_circuit_breaker(operation_name)

        # Check circuit breaker
        if not circuit_breaker.should_attempt():
            logger.warning(
                f"Circuit breaker {operation_name} is open, attempting fallback"
            )
            return await self._attempt_fallback(operation_name, *args, **kwargs)

        for attempt in range(1, 10):  # Maximum 9 attempts
            try:
                # Execute function
                start_time = time.time()

                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)

                execution_time = time.time() - start_time

                # Record success
                circuit_breaker.record_success()
                self._record_success(operation_name, execution_time)

                logger.debug(
                    f"Operation {operation_name} succeeded on attempt {attempt}"
                )
                return result

            except Exception as error:
                last_error = error
                execution_time = time.time() - start_time

                # Classify error if not done yet
                if pattern is None:
                    pattern = self.classify_error(error)

                # Record error
                self._record_error(operation_name, error, attempt, execution_time)

                # Handle based on strategy
                if pattern.strategy == RecoveryStrategy.FAIL_FAST:
                    circuit_breaker.record_failure()
                    logger.error(f"Operation {operation_name} failed fast: {error}")
                    raise error

                elif pattern.strategy == RecoveryStrategy.CIRCUIT_BREAK:
                    circuit_breaker.record_failure()
                    if circuit_breaker.is_open:
                        return await self._attempt_fallback(
                            operation_name, *args, **kwargs
                        )

                # Check if we should retry
                if attempt > pattern.max_retries:
                    if pattern.strategy == RecoveryStrategy.FALLBACK:
                        return await self._attempt_fallback(
                            operation_name, *args, **kwargs
                        )
                    else:
                        circuit_breaker.record_failure()
                        logger.error(
                            f"Operation {operation_name} exhausted retries: {error}"
                        )
                        raise error

                # Calculate delay with exponential backoff and jitter
                delay = min(
                    pattern.base_delay * (2 ** (attempt - 1)), pattern.max_delay
                )
                if pattern.jitter:
                    delay *= 0.5 + random.random() * 0.5  # Add 0-50% jitter

                logger.warning(
                    f"Operation {operation_name} attempt {attempt} failed: {error}. "
                    f"Retrying in {delay:.2f}s"
                )

                await asyncio.sleep(delay)

        # If we get here, all attempts failed
        circuit_breaker.record_failure()
        logger.error(f"Operation {operation_name} failed after all attempts")
        raise last_error

    async def _attempt_fallback(self, operation_name: str, *args, **kwargs) -> Any:
        """Attempt fallback operation."""
        if operation_name in self.fallback_handlers:
            try:
                logger.info(f"Attempting fallback for {operation_name}")
                fallback_func = self.fallback_handlers[operation_name]

                if asyncio.iscoroutinefunction(fallback_func):
                    return await fallback_func(*args, **kwargs)
                else:
                    return fallback_func(*args, **kwargs)

            except Exception as fallback_error:
                logger.error(f"Fallback for {operation_name} failed: {fallback_error}")
                raise fallback_error
        else:
            logger.error(f"No fallback handler registered for {operation_name}")
            raise RuntimeError(
                f"Operation {operation_name} unavailable and no fallback configured"
            )

    def _record_error(
        self, operation_name: str, error: Exception, attempt: int, execution_time: float
    ) -> None:
        """Record error for analysis."""
        with self._lock:
            error_record = {
                "operation": operation_name,
                "error_type": type(error).__name__,
                "error_message": str(error)[:200],  # Truncate long messages
                "attempt": attempt,
                "execution_time": execution_time,
                "timestamp": time.time(),
                "traceback": traceback.format_exc()[
                    -500:
                ],  # Last 500 chars of traceback
            }

            self.error_history.append(error_record)

    def _record_success(self, operation_name: str, execution_time: float) -> None:
        """Record successful operation."""
        # Could be extended to track success patterns
        pass

    def get_resilience_report(self) -> dict[str, Any]:
        """Get comprehensive resilience report."""
        with self._lock:
            # Circuit breaker stats
            circuit_stats = {}
            for name, breaker in self.circuit_breakers.items():
                circuit_stats[name] = breaker.get_stats()

            # Error analysis
            recent_errors = [
                e for e in self.error_history if time.time() - e["timestamp"] < 3600
            ]
            error_types = {}
            operation_errors = {}

            for error in recent_errors:
                error_type = error["error_type"]
                operation = error["operation"]

                error_types[error_type] = error_types.get(error_type, 0) + 1
                operation_errors[operation] = operation_errors.get(operation, 0) + 1

            # Top error patterns
            top_errors = sorted(error_types.items(), key=lambda x: x[1], reverse=True)[
                :10
            ]
            top_failing_operations = sorted(
                operation_errors.items(), key=lambda x: x[1], reverse=True
            )[:10]

            return {
                "circuit_breakers": circuit_stats,
                "recent_errors_count": len(recent_errors),
                "top_error_types": top_errors,
                "top_failing_operations": top_failing_operations,
                "registered_fallbacks": list(self.fallback_handlers.keys()),
                "error_patterns_count": len(self.error_patterns),
            }


# Decorator for easy resilience application
def resilient(operation_name: str | None = None, **resilience_options):
    """Decorator to make functions resilient."""

    def decorator(func):
        nonlocal operation_name
        if operation_name is None:
            operation_name = f"{func.__module__}.{func.__name__}"

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            manager = get_resilience_manager()
            return await manager.execute_with_resilience(
                func, operation_name, *args, **kwargs
            )

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, we create an async wrapper
            async def async_func(*args, **kwargs):
                return func(*args, **kwargs)

            manager = get_resilience_manager()
            return asyncio.run(
                manager.execute_with_resilience(
                    async_func, operation_name, *args, **kwargs
                )
            )

        # Return appropriate wrapper based on function type
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


# Global resilience manager instance
_global_resilience_manager = None


def get_resilience_manager() -> ResilienceManager:
    """Get global resilience manager instance."""
    global _global_resilience_manager
    if _global_resilience_manager is None:
        _global_resilience_manager = ResilienceManager()
    return _global_resilience_manager


def register_fallback(operation_name: str, fallback_func: Callable) -> None:
    """Convenience function to register fallback."""
    manager = get_resilience_manager()
    manager.register_fallback(operation_name, fallback_func)


def get_resilience_report() -> dict[str, Any]:
    """Convenience function to get resilience report."""
    manager = get_resilience_manager()
    return manager.get_resilience_report()


# Example usage and built-in fallbacks
async def search_fallback(query: str, **kwargs) -> str:
    """Fallback for search operations when primary search fails."""
    logger.info("Using search fallback for degraded service")
    return f"Search temporarily unavailable. Please try again later. Query: {query[:100]}..."


async def summary_fallback(text: str, **kwargs) -> str:
    """Fallback for summarization when service fails."""
    logger.info("Using summary fallback for degraded service")
    # Simple truncation fallback
    max_length = kwargs.get("max_length", 200)
    if len(text) <= max_length:
        return text
    return text[:max_length] + "... [Summarization service temporarily unavailable]"


# Register common fallbacks
def setup_default_fallbacks() -> None:
    """Setup default fallback handlers."""
    manager = get_resilience_manager()
    manager.register_fallback("search_operation", search_fallback)
    manager.register_fallback("summarize_operation", summary_fallback)
