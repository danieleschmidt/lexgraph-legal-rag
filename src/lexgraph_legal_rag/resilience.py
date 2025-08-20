"""Resilience patterns for robust legal RAG system operations."""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from contextlib import asynccontextmanager
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Callable
from typing import TypeVar


logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitBreakerState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    expected_exception: type[Exception] = Exception
    name: str = "default"


@dataclass
class CircuitBreakerStats:
    """Circuit breaker statistics."""

    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float | None = None
    last_success_time: float | None = None
    total_requests: int = 0


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""

    pass


class CircuitBreaker:
    """Circuit breaker implementation for resilient operations."""

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.stats = CircuitBreakerStats()
        self._lock = asyncio.Lock()

    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit breaker."""
        if self.stats.state != CircuitBreakerState.OPEN:
            return False

        if self.stats.last_failure_time is None:
            return True

        return time.time() - self.stats.last_failure_time > self.config.recovery_timeout

    def _record_success(self) -> None:
        """Record a successful operation."""
        self.stats.success_count += 1
        self.stats.last_success_time = time.time()
        self.stats.total_requests += 1

        if self.stats.state == CircuitBreakerState.HALF_OPEN:
            # Reset to closed on successful half-open attempt
            self.stats.state = CircuitBreakerState.CLOSED
            self.stats.failure_count = 0
            logger.info(f"Circuit breaker '{self.config.name}' reset to CLOSED")

    def _record_failure(self) -> None:
        """Record a failed operation."""
        self.stats.failure_count += 1
        self.stats.last_failure_time = time.time()
        self.stats.total_requests += 1

        if self.stats.failure_count >= self.config.failure_threshold:
            self.stats.state = CircuitBreakerState.OPEN
            logger.warning(
                f"Circuit breaker '{self.config.name}' opened after {self.stats.failure_count} failures"
            )

    async def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection."""
        async with self._lock:
            # Check if we should attempt reset
            if self._should_attempt_reset():
                self.stats.state = CircuitBreakerState.HALF_OPEN
                logger.info(
                    f"Circuit breaker '{self.config.name}' attempting reset (HALF_OPEN)"
                )

            # Reject if circuit is open
            if self.stats.state == CircuitBreakerState.OPEN:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.config.name}' is OPEN"
                )

        # Execute the function
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            async with self._lock:
                self._record_success()

            return result

        except self.config.expected_exception as e:
            async with self._lock:
                self._record_failure()
            raise e


class RetryConfig:
    """Configuration for retry logic."""

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter


def calculate_retry_delay(attempt: int, config: RetryConfig) -> float:
    """Calculate delay for retry attempt with exponential backoff and jitter."""
    delay = min(
        config.base_delay * (config.exponential_base**attempt), config.max_delay
    )

    if config.jitter:
        import random

        delay *= 0.5 + random.random() * 0.5  # Add 0-50% jitter

    return delay


async def retry_async(
    func: Callable[..., T], config: RetryConfig, *args, **kwargs
) -> T:
    """Retry an async function with exponential backoff."""
    last_exception = None

    for attempt in range(config.max_attempts):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        except Exception as e:
            last_exception = e

            if attempt == config.max_attempts - 1:
                # Last attempt failed
                break

            delay = calculate_retry_delay(attempt, config)
            logger.debug(
                f"Retry attempt {attempt + 1}/{config.max_attempts} failed, "
                f"retrying in {delay:.2f}s: {e}"
            )
            await asyncio.sleep(delay)

    # All attempts failed
    logger.error(f"All {config.max_attempts} retry attempts failed")
    raise last_exception


def retry_sync(func: Callable[..., T], config: RetryConfig, *args, **kwargs) -> T:
    """Retry a sync function with exponential backoff."""
    import time

    last_exception = None

    for attempt in range(config.max_attempts):
        try:
            return func(*args, **kwargs)

        except Exception as e:
            last_exception = e

            if attempt == config.max_attempts - 1:
                # Last attempt failed
                break

            delay = calculate_retry_delay(attempt, config)
            logger.debug(
                f"Retry attempt {attempt + 1}/{config.max_attempts} failed, "
                f"retrying in {delay:.2f}s: {e}"
            )
            time.sleep(delay)

    # All attempts failed
    logger.error(f"All {config.max_attempts} retry attempts failed")
    raise last_exception


class TimeoutError(Exception):
    """Raised when operation times out."""

    pass


@asynccontextmanager
async def timeout(seconds: float):
    """Async context manager for operation timeout."""
    try:
        async with asyncio.timeout(seconds):
            yield
    except asyncio.TimeoutError as e:
        raise TimeoutError(f"Operation timed out after {seconds}s") from e


@contextmanager
def sync_timeout(seconds: float):
    """Sync context manager for operation timeout (simplified)."""
    import signal

    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds}s")

    # Set the signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(seconds))

    try:
        yield
    finally:
        # Restore the old handler and cancel the alarm
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


@dataclass
class BulkheadConfig:
    """Configuration for bulkhead pattern."""

    max_concurrent: int = 10
    queue_size: int = 100
    timeout: float = 30.0


class Bulkhead:
    """Bulkhead pattern implementation for resource isolation."""

    def __init__(self, config: BulkheadConfig):
        self.config = config
        self.semaphore = asyncio.Semaphore(config.max_concurrent)
        self.queue = asyncio.Queue(maxsize=config.queue_size)
        self.active_count = 0

    async def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with bulkhead protection."""
        try:
            # Wait for available slot with timeout
            await asyncio.wait_for(
                self.semaphore.acquire(), timeout=self.config.timeout
            )
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Bulkhead timeout: no available slots after {self.config.timeout}s"
            )

        try:
            self.active_count += 1
            logger.debug(
                f"Bulkhead executing (active: {self.active_count}/{self.config.max_concurrent})"
            )

            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        finally:
            self.active_count -= 1
            self.semaphore.release()


class ResilientOperation:
    """Combines multiple resilience patterns for robust operations."""

    def __init__(
        self,
        circuit_breaker: CircuitBreaker | None = None,
        retry_config: RetryConfig | None = None,
        bulkhead: Bulkhead | None = None,
        operation_timeout: float | None = None,
    ):
        self.circuit_breaker = circuit_breaker
        self.retry_config = retry_config
        self.bulkhead = bulkhead
        self.operation_timeout = operation_timeout

    async def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with all configured resilience patterns."""

        async def _execute_with_patterns():
            # Apply bulkhead if configured
            if self.bulkhead:
                return await self.bulkhead.execute(func, *args, **kwargs)

            # Apply timeout if configured
            if self.operation_timeout:
                async with timeout(self.operation_timeout):
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)

            # Direct execution
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        # Apply circuit breaker if configured
        if self.circuit_breaker:
            if self.retry_config:
                # Combine circuit breaker with retry
                return await self.circuit_breaker.call(
                    retry_async, self.retry_config, _execute_with_patterns
                )
            else:
                return await self.circuit_breaker.call(_execute_with_patterns)

        # Apply retry if configured (without circuit breaker)
        if self.retry_config:
            return await retry_async(_execute_with_patterns, self.retry_config)

        # No resilience patterns, direct execution
        return await _execute_with_patterns()


# Global circuit breakers for common operations
_circuit_breakers: dict[str, CircuitBreaker] = {}


def get_circuit_breaker(
    name: str, config: CircuitBreakerConfig | None = None
) -> CircuitBreaker:
    """Get or create a circuit breaker by name."""
    if name not in _circuit_breakers:
        if config is None:
            config = CircuitBreakerConfig(name=name)
        _circuit_breakers[name] = CircuitBreaker(config)
    return _circuit_breakers[name]


def resilient(
    circuit_breaker_name: str | None = None,
    retry_attempts: int = 3,
    timeout_seconds: float | None = None,
    bulkhead_size: int | None = None,
):
    """Decorator to make functions resilient."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        # Configure resilience patterns
        circuit_breaker = None
        if circuit_breaker_name:
            circuit_breaker = get_circuit_breaker(circuit_breaker_name)

        retry_config = (
            RetryConfig(max_attempts=retry_attempts) if retry_attempts > 1 else None
        )

        bulkhead = None
        if bulkhead_size:
            bulkhead = Bulkhead(BulkheadConfig(max_concurrent=bulkhead_size))

        operation = ResilientOperation(
            circuit_breaker=circuit_breaker,
            retry_config=retry_config,
            bulkhead=bulkhead,
            operation_timeout=timeout_seconds,
        )

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await operation.execute(func, *args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            return asyncio.run(operation.execute(func, *args, **kwargs))

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator
