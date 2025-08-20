"""
Circuit Breaker Pattern Implementation
Provides resilience for external service calls and resource-intensive operations
"""

import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Union


logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Circuit is open, calls are failing fast
    HALF_OPEN = "half_open"  # Testing if service has recovered


@dataclass
class CircuitBreakerStats:
    """Circuit breaker statistics."""

    total_requests: int = 0
    success_count: int = 0
    failure_count: int = 0
    consecutive_failures: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    state_change_time: float = 0


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""

    pass


class CircuitBreaker:
    """
    Circuit breaker implementation for resilient service calls.

    The circuit breaker pattern prevents cascading failures by monitoring
    the success/failure rate of operations and temporarily failing fast
    when the error rate is too high.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        success_threshold: int = 3,
        timeout: float = 60.0,
        expected_exception: Union[Exception, tuple] = Exception,
        fallback: Optional[Callable] = None,
    ):
        """
        Initialize circuit breaker.

        Args:
            name: Unique name for this circuit breaker
            failure_threshold: Number of consecutive failures before opening
            success_threshold: Number of consecutive successes needed to close from half-open
            timeout: Seconds to wait before trying again (open -> half-open)
            expected_exception: Exception types that count as failures
            fallback: Optional fallback function to call when circuit is open
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        self.fallback = fallback

        self._state = CircuitState.CLOSED
        self._stats = CircuitBreakerStats()
        self._lock = threading.RLock()

        logger.info(
            f"Circuit breaker '{name}' initialized: "
            f"failure_threshold={failure_threshold}, "
            f"success_threshold={success_threshold}, "
            f"timeout={timeout}s"
        )

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            return self._state

    @property
    def stats(self) -> CircuitBreakerStats:
        """Get circuit breaker statistics."""
        with self._lock:
            return CircuitBreakerStats(
                total_requests=self._stats.total_requests,
                success_count=self._stats.success_count,
                failure_count=self._stats.failure_count,
                consecutive_failures=self._stats.consecutive_failures,
                last_failure_time=self._stats.last_failure_time,
                last_success_time=self._stats.last_success_time,
                state_change_time=self._stats.state_change_time,
            )

    def __call__(self, func: Callable) -> Callable:
        """Decorator for protecting functions with circuit breaker."""

        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            *args, **kwargs: Arguments to pass to function

        Returns:
            Function result

        Raises:
            CircuitBreakerError: When circuit is open
            Any exception raised by the function
        """
        with self._lock:
            self._stats.total_requests += 1

            # Check if circuit should transition states
            current_time = time.time()
            self._check_state_transition(current_time)

            # Handle different states
            if self._state == CircuitState.OPEN:
                logger.warning(f"Circuit breaker '{self.name}' is OPEN - failing fast")
                if self.fallback:
                    logger.info(f"Using fallback for '{self.name}'")
                    return self.fallback(*args, **kwargs)
                raise CircuitBreakerError(f"Circuit breaker '{self.name}' is open")

            elif self._state == CircuitState.HALF_OPEN:
                logger.info(
                    f"Circuit breaker '{self.name}' is HALF_OPEN - testing service"
                )

        # Execute the function
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result

        except self.expected_exception as e:
            self._on_failure(e)
            raise

        except Exception as e:
            # Unexpected exceptions don't affect circuit breaker state
            logger.warning(
                f"Unexpected exception in circuit breaker '{self.name}': {e}"
            )
            raise

    def _check_state_transition(self, current_time: float) -> None:
        """Check if circuit breaker should change state."""
        if self._state == CircuitState.OPEN:
            # Check if timeout has elapsed for transition to half-open
            if (
                self._stats.last_failure_time
                and current_time - self._stats.last_failure_time >= self.timeout
            ):
                self._change_state(CircuitState.HALF_OPEN, current_time)

    def _on_success(self) -> None:
        """Handle successful function execution."""
        with self._lock:
            self._stats.success_count += 1
            self._stats.consecutive_failures = 0
            self._stats.last_success_time = time.time()

            logger.debug(
                f"Circuit breaker '{self.name}' recorded success "
                f"(total: {self._stats.success_count})"
            )

            # State transitions on success
            if self._state == CircuitState.HALF_OPEN:
                # Need success_threshold consecutive successes to close
                if self._stats.success_count >= self.success_threshold:
                    self._change_state(CircuitState.CLOSED, time.time())

    def _on_failure(self, exception: Exception) -> None:
        """Handle failed function execution."""
        with self._lock:
            self._stats.failure_count += 1
            self._stats.consecutive_failures += 1
            self._stats.last_failure_time = time.time()

            logger.warning(
                f"Circuit breaker '{self.name}' recorded failure: {exception} "
                f"(consecutive: {self._stats.consecutive_failures})"
            )

            # State transitions on failure
            if (
                self._state in [CircuitState.CLOSED, CircuitState.HALF_OPEN]
                and self._stats.consecutive_failures >= self.failure_threshold
            ):
                self._change_state(CircuitState.OPEN, time.time())

    def _change_state(self, new_state: CircuitState, timestamp: float) -> None:
        """Change circuit breaker state."""
        old_state = self._state
        self._state = new_state
        self._stats.state_change_time = timestamp

        # Reset counters on state change
        if new_state == CircuitState.HALF_OPEN:
            self._stats.success_count = 0
        elif new_state == CircuitState.CLOSED:
            self._stats.consecutive_failures = 0
            self._stats.success_count = 0

        logger.info(
            f"Circuit breaker '{self.name}' state changed: "
            f"{old_state.value} -> {new_state.value}"
        )

    def force_open(self) -> None:
        """Manually force circuit breaker to open state."""
        with self._lock:
            self._change_state(CircuitState.OPEN, time.time())
            logger.warning(f"Circuit breaker '{self.name}' forced to OPEN state")

    def force_close(self) -> None:
        """Manually force circuit breaker to closed state."""
        with self._lock:
            self._change_state(CircuitState.CLOSED, time.time())
            self._stats.consecutive_failures = 0
            logger.info(f"Circuit breaker '{self.name}' forced to CLOSED state")

    def reset(self) -> None:
        """Reset circuit breaker statistics."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._stats = CircuitBreakerStats()
            logger.info(f"Circuit breaker '{self.name}' reset")


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""

    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.RLock()

    def get_or_create(
        self,
        name: str,
        failure_threshold: int = 5,
        success_threshold: int = 3,
        timeout: float = 60.0,
        expected_exception: Union[Exception, tuple] = Exception,
        fallback: Optional[Callable] = None,
    ) -> CircuitBreaker:
        """Get existing circuit breaker or create new one."""
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(
                    name=name,
                    failure_threshold=failure_threshold,
                    success_threshold=success_threshold,
                    timeout=timeout,
                    expected_exception=expected_exception,
                    fallback=fallback,
                )
                logger.info(f"Created new circuit breaker: {name}")

            return self._breakers[name]

    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        with self._lock:
            return self._breakers.get(name)

    def list_breakers(self) -> Dict[str, CircuitBreakerStats]:
        """List all circuit breakers and their stats."""
        with self._lock:
            return {name: breaker.stats for name, breaker in self._breakers.items()}

    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()
            logger.info("All circuit breakers reset")

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all circuit breakers."""
        health_status = {"healthy": [], "degraded": [], "unhealthy": []}

        with self._lock:
            for name, breaker in self._breakers.items():
                stats = breaker.stats
                state = breaker.state

                breaker_info = {
                    "name": name,
                    "state": state.value,
                    "failure_rate": (
                        stats.failure_count / stats.total_requests
                        if stats.total_requests > 0
                        else 0
                    ),
                    "consecutive_failures": stats.consecutive_failures,
                }

                if state == CircuitState.OPEN:
                    health_status["unhealthy"].append(breaker_info)
                elif state == CircuitState.HALF_OPEN or stats.consecutive_failures > 0:
                    health_status["degraded"].append(breaker_info)
                else:
                    health_status["healthy"].append(breaker_info)

        return health_status


# Global circuit breaker registry
_registry = CircuitBreakerRegistry()


def circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    success_threshold: int = 3,
    timeout: float = 60.0,
    expected_exception: Union[Exception, tuple] = Exception,
    fallback: Optional[Callable] = None,
) -> Callable:
    """
    Decorator for adding circuit breaker protection to functions.

    Args:
        name: Unique name for the circuit breaker
        failure_threshold: Number of consecutive failures before opening
        success_threshold: Number of consecutive successes to close from half-open
        timeout: Seconds to wait before trying again
        expected_exception: Exception types that count as failures
        fallback: Optional fallback function

    Returns:
        Decorated function with circuit breaker protection
    """
    breaker = _registry.get_or_create(
        name=name,
        failure_threshold=failure_threshold,
        success_threshold=success_threshold,
        timeout=timeout,
        expected_exception=expected_exception,
        fallback=fallback,
    )

    return breaker


def get_circuit_breaker(name: str) -> Optional[CircuitBreaker]:
    """Get circuit breaker by name from global registry."""
    return _registry.get(name)


def get_all_circuit_breakers() -> Dict[str, CircuitBreakerStats]:
    """Get all circuit breakers and their statistics."""
    return _registry.list_breakers()


def get_circuit_breaker_health() -> Dict[str, Any]:
    """Get health status of all circuit breakers."""
    return _registry.get_health_status()


def reset_all_circuit_breakers() -> None:
    """Reset all circuit breakers in the registry."""
    _registry.reset_all()


# Example usage and fallback functions
def default_search_fallback(*args, **kwargs) -> Dict[str, Any]:
    """Default fallback for search operations."""
    return {
        "results": [],
        "message": "Search service temporarily unavailable",
        "fallback": True,
    }


def default_api_fallback(*args, **kwargs) -> Dict[str, Any]:
    """Default fallback for API calls."""
    return {
        "error": "Service temporarily unavailable",
        "fallback": True,
        "retry_after": 60,
    }


# Pre-configured circuit breakers for common use cases
def create_search_circuit_breaker(name: str = "search_service") -> CircuitBreaker:
    """Create circuit breaker for search operations."""
    return _registry.get_or_create(
        name=name,
        failure_threshold=3,
        success_threshold=2,
        timeout=30.0,
        expected_exception=(ConnectionError, TimeoutError, Exception),
        fallback=default_search_fallback,
    )


def create_api_circuit_breaker(name: str = "external_api") -> CircuitBreaker:
    """Create circuit breaker for external API calls."""
    return _registry.get_or_create(
        name=name,
        failure_threshold=5,
        success_threshold=3,
        timeout=60.0,
        expected_exception=(ConnectionError, TimeoutError, Exception),
        fallback=default_api_fallback,
    )


def create_database_circuit_breaker(name: str = "database") -> CircuitBreaker:
    """Create circuit breaker for database operations."""
    return _registry.get_or_create(
        name=name,
        failure_threshold=3,
        success_threshold=2,
        timeout=120.0,
        expected_exception=(ConnectionError, TimeoutError, Exception),
    )


if __name__ == "__main__":
    # Example usage
    import random

    logging.basicConfig(level=logging.INFO)

    @circuit_breaker(
        name="example_service",
        failure_threshold=3,
        success_threshold=2,
        timeout=5.0,
        fallback=lambda: "Fallback response",
    )
    def unreliable_service():
        """Simulate an unreliable service."""
        if random.random() < 0.7:  # 70% failure rate
            raise ConnectionError("Service unavailable")
        return "Success!"

    # Test the circuit breaker
    print("Testing circuit breaker...")
    for i in range(15):
        try:
            result = unreliable_service()
            print(f"Request {i+1}: {result}")
        except (CircuitBreakerError, ConnectionError) as e:
            print(f"Request {i+1}: Failed - {e}")

        time.sleep(1)

    # Show circuit breaker stats
    print("\nCircuit Breaker Stats:")
    for name, stats in get_all_circuit_breakers().items():
        print(f"{name}: {stats}")

    print("\nHealth Status:")
    health = get_circuit_breaker_health()
    for status, breakers in health.items():
        print(f"{status}: {len(breakers)} breakers")
