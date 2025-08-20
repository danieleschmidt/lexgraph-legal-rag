"""Resilient HTTP client with retry logic and circuit breaker pattern."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import Any

import httpx
from httpx import HTTPStatusError
from httpx import RequestError
from httpx import Response


logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_status_codes: tuple[int, ...] = (429, 500, 502, 503, 504)


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""

    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 3  # Successes needed to close circuit


@dataclass
class CircuitBreaker:
    """Circuit breaker implementation for external service protection."""

    config: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    _state: CircuitState = CircuitState.CLOSED
    _failure_count: int = 0
    _success_count: int = 0
    _last_failure_time: float = 0.0

    def can_execute(self) -> bool:
        """Check if requests can be executed."""
        if self._state == CircuitState.CLOSED:
            return True
        elif self._state == CircuitState.OPEN:
            if time.time() - self._last_failure_time >= self.config.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                self._success_count = 0
                logger.info("Circuit breaker moved to HALF_OPEN state")
                return True
            return False
        else:  # HALF_OPEN
            return True

    def record_success(self) -> None:
        """Record a successful request."""
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.config.success_threshold:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                logger.info("Circuit breaker moved to CLOSED state")
        elif self._state == CircuitState.CLOSED:
            self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed request."""
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._state == CircuitState.CLOSED:
            if self._failure_count >= self.config.failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning(
                    f"Circuit breaker opened after {self._failure_count} failures"
                )
        elif self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            logger.warning("Circuit breaker reopened during half-open state")


class ResilientHTTPClient:
    """HTTP client with retry logic and circuit breaker protection."""

    def __init__(
        self,
        base_url: str = "",
        retry_config: RetryConfig | None = None,
        circuit_config: CircuitBreakerConfig | None = None,
        timeout: float = 30.0,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.base_url = base_url
        self.retry_config = retry_config or RetryConfig()
        self.circuit_breaker = CircuitBreaker(circuit_config or CircuitBreakerConfig())
        self.timeout = timeout
        self.default_headers = headers or {}

        self._client = httpx.AsyncClient(
            base_url=base_url, timeout=timeout, headers=self.default_headers
        )

    async def __aenter__(self) -> ResilientHTTPClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self._client.aclose()

    async def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        data: str | bytes | None = None,
        **kwargs: Any,
    ) -> Response:
        """Make HTTP request with retry logic and circuit breaker protection."""

        if not self.circuit_breaker.can_execute():
            # Create a dummy request and response for the error
            request = httpx.Request("GET", url)
            response = httpx.Response(503, request=request)
            raise HTTPStatusError(
                message="Circuit breaker is open", request=request, response=response
            )

        last_exception = None

        for attempt in range(self.retry_config.max_retries + 1):
            try:
                logger.debug(f"Attempting {method} {url} (attempt {attempt + 1})")

                response = await self._client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    params=params,
                    json=json,
                    data=data,
                    **kwargs,
                )

                # Check if response indicates success
                if response.status_code < 400:
                    self.circuit_breaker.record_success()
                    return response

                # Handle retryable status codes
                if response.status_code in self.retry_config.retryable_status_codes:
                    if attempt < self.retry_config.max_retries:
                        delay = self._calculate_delay(attempt)
                        logger.warning(
                            f"Request failed with status {response.status_code}, "
                            f"retrying in {delay:.2f}s (attempt {attempt + 1})"
                        )
                        await self._async_sleep(delay)
                        continue

                # Non-retryable error
                self.circuit_breaker.record_failure()
                response.raise_for_status()

            except RequestError as e:
                last_exception = e

                if attempt < self.retry_config.max_retries:
                    delay = self._calculate_delay(attempt)
                    logger.warning(
                        f"Request failed with {type(e).__name__}: {e}, "
                        f"retrying in {delay:.2f}s (attempt {attempt + 1})"
                    )
                    await self._async_sleep(delay)
                    continue

                self.circuit_breaker.record_failure()
                break

        # All retries exhausted
        if last_exception:
            raise last_exception

        raise RuntimeError("Request failed after all retries")

    async def get(self, url: str, **kwargs: Any) -> Response:
        """Make GET request."""
        return await self.request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs: Any) -> Response:
        """Make POST request."""
        return await self.request("POST", url, **kwargs)

    async def put(self, url: str, **kwargs: Any) -> Response:
        """Make PUT request."""
        return await self.request("PUT", url, **kwargs)

    async def delete(self, url: str, **kwargs: Any) -> Response:
        """Make DELETE request."""
        return await self.request("DELETE", url, **kwargs)

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay with optional jitter."""
        delay = min(
            self.retry_config.initial_delay
            * (self.retry_config.exponential_base**attempt),
            self.retry_config.max_delay,
        )

        if self.retry_config.jitter:
            import secrets

            delay *= (
                0.5 + secrets.randbelow(1000) / 2000.0
            )  # Add 0-50% jitter with cryptographically secure random

        return delay

    async def _async_sleep(self, delay: float) -> None:
        """Async sleep wrapper."""
        import asyncio

        await asyncio.sleep(delay)

    def get_circuit_status(self) -> dict[str, Any]:
        """Get current circuit breaker status."""
        return {
            "state": self.circuit_breaker._state.value,
            "failure_count": self.circuit_breaker._failure_count,
            "success_count": self.circuit_breaker._success_count,
            "last_failure_time": self.circuit_breaker._last_failure_time,
        }


# Convenience function for creating configured clients
def create_legal_api_client(
    base_url: str, api_key: str | None = None, **kwargs: Any
) -> ResilientHTTPClient:
    """Create HTTP client configured for legal API services."""
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        headers["X-API-Key"] = api_key

    return ResilientHTTPClient(base_url=base_url, headers=headers, **kwargs)
