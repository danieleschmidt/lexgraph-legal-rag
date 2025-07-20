"""Tests for resilient HTTP client with retry logic."""

import pytest
from unittest.mock import AsyncMock, patch
import httpx

from lexgraph_legal_rag.http_client import (
    ResilientHTTPClient,
    RetryConfig,
    CircuitBreakerConfig,
    CircuitBreaker,
    CircuitState,
    create_legal_api_client,
)


@pytest.mark.asyncio
async def test_circuit_breaker_basic_functionality():
    """Test basic circuit breaker state transitions."""
    config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=1.0)
    breaker = CircuitBreaker(config)
    
    # Initially closed
    assert breaker._state == CircuitState.CLOSED
    assert breaker.can_execute()
    
    # Record failures
    breaker.record_failure()
    assert breaker._state == CircuitState.CLOSED
    
    breaker.record_failure()
    assert breaker._state == CircuitState.OPEN
    assert not breaker.can_execute()
    
    # Test success resets failure count
    breaker._state = CircuitState.CLOSED
    breaker._failure_count = 1
    breaker.record_success()
    assert breaker._failure_count == 0


@pytest.mark.asyncio
async def test_circuit_breaker_half_open_transition():
    """Test half-open state transitions."""
    config = CircuitBreakerConfig(
        failure_threshold=1,
        recovery_timeout=0.1,  # Short timeout for testing
        success_threshold=2
    )
    breaker = CircuitBreaker(config)
    
    # Force to open state
    breaker.record_failure()
    assert breaker._state == CircuitState.OPEN
    
    # Wait for recovery timeout
    import asyncio
    await asyncio.sleep(0.2)
    
    # Should allow execution and move to half-open
    assert breaker.can_execute()
    assert breaker._state == CircuitState.HALF_OPEN
    
    # Record success
    breaker.record_success()
    assert breaker._state == CircuitState.HALF_OPEN
    
    # Another success should close the circuit
    breaker.record_success()
    assert breaker._state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_retry_config_calculation():
    """Test retry delay calculation."""
    # Create client with jitter disabled for deterministic testing
    retry_config = RetryConfig(jitter=False)
    client = ResilientHTTPClient(retry_config=retry_config)
    
    # Test exponential backoff
    assert client._calculate_delay(0) == 1.0  # initial_delay
    assert client._calculate_delay(1) == 2.0  # initial_delay * base^1
    assert client._calculate_delay(2) == 4.0  # initial_delay * base^2
    
    # Test max delay cap
    config = RetryConfig(initial_delay=10.0, max_delay=15.0, exponential_base=2.0, jitter=False)
    client.retry_config = config
    assert client._calculate_delay(1) == 15.0  # Capped at max_delay


@pytest.mark.asyncio
async def test_successful_request():
    """Test successful HTTP request without retries."""
    with patch('httpx.AsyncClient') as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_client.request.return_value = mock_response
        
        client = ResilientHTTPClient()
        
        response = await client.get("/test")
        
        assert response.status_code == 200
        mock_client.request.assert_called_once()


@pytest.mark.asyncio
async def test_retry_on_retryable_status_code():
    """Test retry behavior on retryable status codes."""
    with patch('httpx.AsyncClient') as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        
        # First call returns 503, second call succeeds
        responses = [
            AsyncMock(status_code=503),
            AsyncMock(status_code=200)
        ]
        mock_client.request.side_effect = responses
        
        client = ResilientHTTPClient()
        client.retry_config.max_retries = 1
        client.retry_config.initial_delay = 0.01  # Fast retry for testing
        
        with patch.object(client, '_async_sleep', new_callable=AsyncMock):
            response = await client.get("/test")
        
        assert response.status_code == 200
        assert mock_client.request.call_count == 2


@pytest.mark.asyncio
async def test_circuit_breaker_blocks_requests():
    """Test that circuit breaker blocks requests when open."""
    config = CircuitBreakerConfig(failure_threshold=1)
    client = ResilientHTTPClient(circuit_config=config)
    
    # Force circuit breaker to open state
    client.circuit_breaker._state = CircuitState.OPEN
    
    with pytest.raises(httpx.HTTPStatusError, match="Circuit breaker is open"):
        await client.get("/test")


@pytest.mark.asyncio
async def test_non_retryable_error():
    """Test that non-retryable errors are not retried."""
    with patch('httpx.AsyncClient') as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        
        mock_response = AsyncMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not found", request=None, response=mock_response
        )
        mock_client.request.return_value = mock_response
        
        client = ResilientHTTPClient()
        
        with pytest.raises(httpx.HTTPStatusError):
            await client.get("/test")
        
        # Should only be called once (no retries for 404)
        mock_client.request.assert_called_once()


@pytest.mark.asyncio
async def test_request_error_retry():
    """Test retry behavior on request errors."""
    with patch('httpx.AsyncClient') as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        
        # First call raises RequestError, second call succeeds
        mock_response = AsyncMock(status_code=200)
        mock_client.request.side_effect = [
            httpx.RequestError("Connection failed"),
            mock_response
        ]
        
        client = ResilientHTTPClient()
        client.retry_config.max_retries = 1
        client.retry_config.initial_delay = 0.01
        
        with patch.object(client, '_async_sleep', new_callable=AsyncMock):
            response = await client.get("/test")
        
        assert response.status_code == 200
        assert mock_client.request.call_count == 2


def test_create_legal_api_client():
    """Test convenience function for creating legal API client."""
    client = create_legal_api_client(
        base_url="https://api.example.com",
        api_key="test-key"
    )
    
    assert client.base_url == "https://api.example.com"
    assert "Authorization" in client.default_headers
    assert "X-API-Key" in client.default_headers
    assert client.default_headers["X-API-Key"] == "test-key"


def test_circuit_status():
    """Test getting circuit breaker status."""
    client = ResilientHTTPClient()
    status = client.get_circuit_status()
    
    assert "state" in status
    assert "failure_count" in status
    assert "success_count" in status
    assert "last_failure_time" in status
    assert status["state"] == "closed"