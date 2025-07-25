"""Comprehensive test coverage for the http_client module."""

import time
import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from dataclasses import dataclass
from typing import Dict, Any

import httpx
from httpx import Response, HTTPStatusError, RequestError

from lexgraph_legal_rag.http_client import (
    CircuitState,
    RetryConfig,
    CircuitBreakerConfig,
    CircuitBreaker,
    ResilientHTTPClient,
    create_legal_api_client
)


class TestRetryConfig:
    """Test cases for RetryConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = RetryConfig()
        
        assert config.max_retries == 3
        assert config.initial_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True
        assert config.retryable_status_codes == (429, 500, 502, 503, 504)
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = RetryConfig(
            max_retries=5,
            initial_delay=0.5,
            max_delay=30.0,
            exponential_base=1.5,
            jitter=False,
            retryable_status_codes=(500, 503)
        )
        
        assert config.max_retries == 5
        assert config.initial_delay == 0.5
        assert config.max_delay == 30.0
        assert config.exponential_base == 1.5
        assert config.jitter is False
        assert config.retryable_status_codes == (500, 503)


class TestCircuitBreakerConfig:
    """Test cases for CircuitBreakerConfig dataclass."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = CircuitBreakerConfig()
        
        assert config.failure_threshold == 5
        assert config.recovery_timeout == 60.0
        assert config.success_threshold == 3
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=30.0,
            success_threshold=2
        )
        
        assert config.failure_threshold == 3
        assert config.recovery_timeout == 30.0
        assert config.success_threshold == 2


class TestCircuitBreaker:
    """Test cases for CircuitBreaker functionality."""
    
    def test_initial_state(self):
        """Test circuit breaker initial state."""
        cb = CircuitBreaker()
        
        assert cb._state == CircuitState.CLOSED
        assert cb._failure_count == 0
        assert cb._success_count == 0
        assert cb._last_failure_time == 0.0
        assert cb.can_execute() is True
    
    def test_closed_state_allows_execution(self):
        """Test that closed state allows execution."""
        cb = CircuitBreaker()
        assert cb.can_execute() is True
    
    def test_failure_recording_in_closed_state(self):
        """Test failure recording in closed state."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker(config)
        
        # Record failures below threshold
        cb.record_failure()
        cb.record_failure()
        assert cb._state == CircuitState.CLOSED
        assert cb._failure_count == 2
        
        # Record failure that triggers opening
        cb.record_failure()
        assert cb._state == CircuitState.OPEN
        assert cb._failure_count == 3
    
    def test_success_recording_resets_failures(self):
        """Test that success recording resets failure count."""
        cb = CircuitBreaker()
        
        cb.record_failure()
        cb.record_failure()
        assert cb._failure_count == 2
        
        cb.record_success()
        assert cb._failure_count == 0
    
    def test_open_state_blocks_execution(self):
        """Test that open state blocks execution."""
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker(config)
        
        cb.record_failure()
        assert cb._state == CircuitState.OPEN
        assert cb.can_execute() is False
    
    def test_open_to_half_open_transition(self):
        """Test transition from open to half-open state."""
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=0.1)
        cb = CircuitBreaker(config)
        
        # Force open state
        cb.record_failure()
        assert cb._state == CircuitState.OPEN
        assert cb.can_execute() is False
        
        # Wait for recovery timeout
        time.sleep(0.2)
        
        # Should transition to half-open
        assert cb.can_execute() is True
        assert cb._state == CircuitState.HALF_OPEN
    
    def test_half_open_success_closes_circuit(self):
        """Test that successes in half-open state close the circuit."""
        config = CircuitBreakerConfig(failure_threshold=1, success_threshold=2)
        cb = CircuitBreaker(config)
        
        # Force half-open state
        cb._state = CircuitState.HALF_OPEN
        
        # Record successes
        cb.record_success()
        assert cb._state == CircuitState.HALF_OPEN
        assert cb._success_count == 1
        
        cb.record_success()
        assert cb._state == CircuitState.CLOSED
        assert cb._failure_count == 0
    
    def test_half_open_failure_reopens_circuit(self):
        """Test that failure in half-open state reopens the circuit."""
        cb = CircuitBreaker()
        cb._state = CircuitState.HALF_OPEN
        
        cb.record_failure()
        assert cb._state == CircuitState.OPEN
    
    def test_half_open_allows_execution(self):
        """Test that half-open state allows execution."""
        cb = CircuitBreaker()
        cb._state = CircuitState.HALF_OPEN
        
        assert cb.can_execute() is True


class TestResilientHTTPClient:
    """Test cases for ResilientHTTPClient."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.base_url = "https://api.example.com"
        self.client = ResilientHTTPClient(base_url=self.base_url)
    
    def test_initialization_defaults(self):
        """Test client initialization with defaults."""
        client = ResilientHTTPClient()
        
        assert client.base_url == ""
        assert isinstance(client.retry_config, RetryConfig)
        assert isinstance(client.circuit_breaker, CircuitBreaker)
        assert client.timeout == 30.0
        assert client.default_headers == {}
    
    def test_initialization_custom_config(self):
        """Test client initialization with custom configuration."""
        retry_config = RetryConfig(max_retries=5)
        circuit_config = CircuitBreakerConfig(failure_threshold=3)
        headers = {"User-Agent": "test-client"}
        
        client = ResilientHTTPClient(
            base_url="https://test.com",
            retry_config=retry_config,
            circuit_config=circuit_config,
            timeout=60.0,
            headers=headers
        )
        
        assert client.base_url == "https://test.com"
        assert client.retry_config.max_retries == 5
        assert client.circuit_breaker.config.failure_threshold == 3
        assert client.timeout == 60.0
        assert client.default_headers == headers
    
    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager functionality."""
        async with ResilientHTTPClient() as client:
            assert isinstance(client, ResilientHTTPClient)
            # Test that client is properly initialized
            assert hasattr(client, '_client')
    
    @pytest.mark.asyncio
    async def test_successful_request(self):
        """Test successful HTTP request."""
        with patch.object(self.client, '_client') as mock_client:
            # Mock successful response
            mock_response = Mock(spec=Response)
            mock_response.status_code = 200
            mock_client.request.return_value = mock_response
            
            response = await self.client.request("GET", "/test")
            
            assert response == mock_response
            mock_client.request.assert_called_once_with(
                method="GET",
                url="/test",
                headers=None,
                params=None,
                json=None,
                data=None
            )
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_blocks_request(self):
        """Test that circuit breaker blocks requests when open."""
        # Force circuit breaker to open state
        self.client.circuit_breaker._state = CircuitState.OPEN
        
        with pytest.raises(HTTPStatusError, match="Circuit breaker is open"):
            await self.client.request("GET", "/test")
    
    @pytest.mark.asyncio
    async def test_retry_on_server_error(self):
        """Test retry logic for server errors."""
        with patch.object(self.client, '_client') as mock_client:
            with patch.object(self.client, '_async_sleep') as mock_sleep:
                # First two calls return server error, third succeeds
                error_response = Mock(spec=Response)
                error_response.status_code = 503
                success_response = Mock(spec=Response)
                success_response.status_code = 200
                
                mock_client.request.side_effect = [error_response, error_response, success_response]
                
                response = await self.client.request("GET", "/test")
                
                assert response == success_response
                assert mock_client.request.call_count == 3
                assert mock_sleep.call_count == 2  # Two retries
    
    @pytest.mark.asyncio
    async def test_no_retry_on_client_error(self):
        """Test no retry for client errors."""
        with patch.object(self.client, '_client') as mock_client:
            # Mock client error response
            error_response = Mock(spec=Response)
            error_response.status_code = 404
            error_response.raise_for_status.side_effect = HTTPStatusError(
                message="Not Found",
                request=Mock(),
                response=error_response
            )
            mock_client.request.return_value = error_response
            
            with pytest.raises(HTTPStatusError):
                await self.client.request("GET", "/test")
            
            # Should only be called once (no retries for 404)
            mock_client.request.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_request_error_retry(self):
        """Test retry on request errors."""
        with patch.object(self.client, '_client') as mock_client:
            with patch.object(self.client, '_async_sleep') as mock_sleep:
                # First calls raise RequestError, last succeeds
                request_error = RequestError("Connection failed")
                success_response = Mock(spec=Response)
                success_response.status_code = 200
                
                mock_client.request.side_effect = [request_error, request_error, success_response]
                
                response = await self.client.request("GET", "/test")
                
                assert response == success_response
                assert mock_client.request.call_count == 3
                assert mock_sleep.call_count == 2
    
    @pytest.mark.asyncio
    async def test_exhaust_retries_raises_exception(self):
        """Test that exhausting retries raises the last exception."""
        with patch.object(self.client, '_client') as mock_client:
            with patch.object(self.client, '_async_sleep'):
                # All calls raise RequestError
                request_error = RequestError("Connection failed")
                mock_client.request.side_effect = request_error
                
                with pytest.raises(RequestError, match="Connection failed"):
                    await self.client.request("GET", "/test")
                
                # Should call max_retries + 1 times
                assert mock_client.request.call_count == 4  # 3 retries + 1 initial
    
    @pytest.mark.asyncio
    async def test_convenience_methods(self):
        """Test convenience methods (get, post, put, delete)."""
        with patch.object(self.client, 'request') as mock_request:
            mock_response = Mock(spec=Response)
            mock_request.return_value = mock_response
            
            # Test GET
            response = await self.client.get("/test", params={"q": "test"})
            assert response == mock_response
            mock_request.assert_called_with("GET", "/test", params={"q": "test"})
            
            # Test POST
            await self.client.post("/test", json={"data": "test"})
            mock_request.assert_called_with("POST", "/test", json={"data": "test"})
            
            # Test PUT
            await self.client.put("/test", data="test data")
            mock_request.assert_called_with("PUT", "/test", data="test data")
            
            # Test DELETE
            await self.client.delete("/test")
            mock_request.assert_called_with("DELETE", "/test")
    
    def test_calculate_delay_exponential_backoff(self):
        """Test exponential backoff delay calculation."""
        config = RetryConfig(
            initial_delay=1.0,
            exponential_base=2.0,
            max_delay=10.0,
            jitter=False
        )
        client = ResilientHTTPClient(retry_config=config)
        
        # Test exponential progression
        assert client._calculate_delay(0) == 1.0  # 1.0 * 2^0
        assert client._calculate_delay(1) == 2.0  # 1.0 * 2^1
        assert client._calculate_delay(2) == 4.0  # 1.0 * 2^2
        
        # Test max delay cap
        assert client._calculate_delay(10) == 10.0  # Capped at max_delay
    
    def test_calculate_delay_with_jitter(self):
        """Test delay calculation with jitter."""
        config = RetryConfig(
            initial_delay=1.0,
            exponential_base=2.0,
            jitter=True
        )
        client = ResilientHTTPClient(retry_config=config)
        
        # With jitter, delay should be between 50% and 150% of base delay
        delay = client._calculate_delay(1)  # Base would be 2.0
        assert 1.0 <= delay <= 3.0
    
    @pytest.mark.asyncio
    async def test_async_sleep(self):
        """Test async sleep functionality."""
        with patch('asyncio.sleep') as mock_sleep:
            await self.client._async_sleep(1.5)
            mock_sleep.assert_called_once_with(1.5)
    
    def test_get_circuit_status(self):
        """Test circuit breaker status reporting."""
        # Set some state on the circuit breaker
        self.client.circuit_breaker._failure_count = 2
        self.client.circuit_breaker._success_count = 1
        self.client.circuit_breaker._last_failure_time = 123456.0
        
        status = self.client.get_circuit_status()
        
        expected = {
            "state": "closed",
            "failure_count": 2,
            "success_count": 1,
            "last_failure_time": 123456.0
        }
        assert status == expected


class TestErrorHandlingEdgeCases:
    """Test cases for error handling edge cases."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.client = ResilientHTTPClient()
    
    @pytest.mark.asyncio
    async def test_zero_retries_configuration(self):
        """Test behavior with zero retries configured."""
        config = RetryConfig(max_retries=0)
        client = ResilientHTTPClient(retry_config=config)
        
        with patch.object(client, '_client') as mock_client:
            error_response = Mock(spec=Response)
            error_response.status_code = 503
            error_response.raise_for_status.side_effect = HTTPStatusError(
                message="Service Unavailable",
                request=Mock(),
                response=error_response
            )
            mock_client.request.return_value = error_response
            
            with pytest.raises(HTTPStatusError):
                await client.request("GET", "/test")
            
            # Should only be called once (no retries)
            mock_client.request.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_success_recording(self):
        """Test that successful requests record success on circuit breaker."""
        with patch.object(self.client, '_client') as mock_client:
            with patch.object(self.client.circuit_breaker, 'record_success') as mock_success:
                success_response = Mock(spec=Response)
                success_response.status_code = 200
                mock_client.request.return_value = success_response
                
                await self.client.request("GET", "/test")
                
                mock_success.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_failure_recording(self):
        """Test that failed requests record failure on circuit breaker."""
        with patch.object(self.client, '_client') as mock_client:
            with patch.object(self.client.circuit_breaker, 'record_failure') as mock_failure:
                # Non-retryable error
                error_response = Mock(spec=Response)
                error_response.status_code = 400
                error_response.raise_for_status.side_effect = HTTPStatusError(
                    message="Bad Request",
                    request=Mock(),
                    response=error_response
                )
                mock_client.request.return_value = error_response
                
                with pytest.raises(HTTPStatusError):
                    await self.client.request("GET", "/test")
                
                mock_failure.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_failure_after_all_retries_exhausted(self):
        """Test failure recording after all retries are exhausted."""
        config = RetryConfig(max_retries=1)
        client = ResilientHTTPClient(retry_config=config)
        
        with patch.object(client, '_client') as mock_client:
            with patch.object(client.circuit_breaker, 'record_failure') as mock_failure:
                with patch.object(client, '_async_sleep'):
                    request_error = RequestError("Connection failed")
                    mock_client.request.side_effect = request_error
                    
                    with pytest.raises(RequestError):
                        await client.request("GET", "/test")
                    
                    mock_failure.assert_called_once()


class TestIntegrationScenarios:
    """Test cases for integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_integration_with_retries(self):
        """Test circuit breaker working with retry logic."""
        config = CircuitBreakerConfig(failure_threshold=2)
        client = ResilientHTTPClient(circuit_config=config)
        
        with patch.object(client, '_client') as mock_client:
            with patch.object(client, '_async_sleep'):
                # All requests fail
                request_error = RequestError("Connection failed")
                mock_client.request.side_effect = request_error
                
                # First request fails after retries
                with pytest.raises(RequestError):
                    await client.request("GET", "/test1")
                
                # Second request fails after retries, should open circuit
                with pytest.raises(RequestError):
                    await client.request("GET", "/test2")
                
                # Third request should be blocked by circuit breaker
                with pytest.raises(HTTPStatusError, match="Circuit breaker is open"):
                    await client.request("GET", "/test3")
    
    @pytest.mark.asyncio
    async def test_recovery_scenario(self):
        """Test service recovery scenario."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout=0.1,
            success_threshold=1
        )
        client = ResilientHTTPClient(circuit_config=config)
        
        with patch.object(client, '_client') as mock_client:
            # First request fails, opens circuit
            request_error = RequestError("Connection failed")
            mock_client.request.side_effect = [request_error]
            
            with pytest.raises(RequestError):
                await client.request("GET", "/test1")
            
            assert client.circuit_breaker._state == CircuitState.OPEN
            
            # Wait for recovery timeout
            time.sleep(0.2)
            
            # Next request should work (half-open state)
            success_response = Mock(spec=Response)
            success_response.status_code = 200
            mock_client.request.side_effect = [success_response]
            
            response = await client.request("GET", "/test2")
            assert response == success_response
            assert client.circuit_breaker._state == CircuitState.CLOSED


class TestCreateLegalAPIClient:
    """Test cases for create_legal_api_client convenience function."""
    
    def test_create_without_api_key(self):
        """Test creating client without API key."""
        client = create_legal_api_client("https://api.legal.com")
        
        assert client.base_url == "https://api.legal.com"
        assert client.default_headers == {}
    
    def test_create_with_api_key(self):
        """Test creating client with API key."""
        api_key = "test-api-key-123"
        client = create_legal_api_client("https://api.legal.com", api_key=api_key)
        
        assert client.base_url == "https://api.legal.com"
        assert client.default_headers["Authorization"] == f"Bearer {api_key}"
        assert client.default_headers["X-API-Key"] == api_key
    
    def test_create_with_custom_config(self):
        """Test creating client with custom configuration."""
        retry_config = RetryConfig(max_retries=5)
        client = create_legal_api_client(
            "https://api.legal.com",
            api_key="test-key",
            retry_config=retry_config
        )
        
        assert client.retry_config.max_retries == 5


class TestCircuitBreakerLogging:
    """Test cases for circuit breaker logging."""
    
    def test_half_open_transition_logging(self):
        """Test logging when transitioning to half-open state."""
        with patch('lexgraph_legal_rag.http_client.logger') as mock_logger:
            config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=0.1)
            cb = CircuitBreaker(config)
            
            # Force open state
            cb.record_failure()
            
            # Wait and trigger half-open transition
            time.sleep(0.2)
            cb.can_execute()
            
            mock_logger.info.assert_called_with("Circuit breaker moved to HALF_OPEN state")
    
    def test_closed_transition_logging(self):
        """Test logging when transitioning to closed state."""
        with patch('lexgraph_legal_rag.http_client.logger') as mock_logger:
            config = CircuitBreakerConfig(success_threshold=1)
            cb = CircuitBreaker(config)
            cb._state = CircuitState.HALF_OPEN
            
            cb.record_success()
            
            mock_logger.info.assert_called_with("Circuit breaker moved to CLOSED state")
    
    def test_open_transition_logging(self):
        """Test logging when transitioning to open state."""
        with patch('lexgraph_legal_rag.http_client.logger') as mock_logger:
            config = CircuitBreakerConfig(failure_threshold=1)
            cb = CircuitBreaker(config)
            
            cb.record_failure()
            
            mock_logger.warning.assert_called_with("Circuit breaker opened after 1 failures")
    
    def test_reopen_from_half_open_logging(self):
        """Test logging when reopening from half-open state."""
        with patch('lexgraph_legal_rag.http_client.logger') as mock_logger:
            cb = CircuitBreaker()
            cb._state = CircuitState.HALF_OPEN
            
            cb.record_failure()
            
            mock_logger.warning.assert_called_with("Circuit breaker reopened during half-open state")


class TestConcurrency:
    """Test cases for concurrent access and thread safety considerations."""
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test multiple concurrent requests."""
        client = ResilientHTTPClient()
        
        with patch.object(client, '_client') as mock_client:
            success_response = Mock(spec=Response)
            success_response.status_code = 200
            mock_client.request.return_value = success_response
            
            # Make multiple concurrent requests
            tasks = [
                client.request("GET", f"/test{i}")
                for i in range(5)
            ]
            
            responses = await asyncio.gather(*tasks)
            
            assert len(responses) == 5
            assert all(r == success_response for r in responses)
            assert mock_client.request.call_count == 5


class TestPerformanceScenarios:
    """Test cases for performance scenarios."""
    
    def test_jitter_randomness(self):
        """Test that jitter produces varied delays."""
        config = RetryConfig(initial_delay=1.0, jitter=True)
        client = ResilientHTTPClient(retry_config=config)
        
        delays = [client._calculate_delay(1) for _ in range(10)]
        
        # All delays should be different due to jitter
        assert len(set(delays)) > 1
        # All delays should be within reasonable bounds
        assert all(1.0 <= d <= 3.0 for d in delays)
    
    def test_max_delay_enforcement(self):
        """Test that max delay is enforced for high retry attempts."""
        config = RetryConfig(
            initial_delay=1.0,
            exponential_base=2.0,
            max_delay=5.0,
            jitter=False
        )
        client = ResilientHTTPClient(retry_config=config)
        
        # High retry attempt should be capped
        delay = client._calculate_delay(10)
        assert delay == 5.0


class TestSecurityAspects:
    """Test cases focusing on security aspects."""
    
    def test_api_key_in_headers(self):
        """Test that API keys are properly set in headers."""
        api_key = "secret-api-key-123"
        client = create_legal_api_client("https://api.legal.com", api_key=api_key)
        
        # Verify headers contain API key
        assert "Authorization" in client.default_headers
        assert "X-API-Key" in client.default_headers
        assert client.default_headers["Authorization"] == f"Bearer {api_key}"
        assert client.default_headers["X-API-Key"] == api_key
    
    def test_secure_jitter_randomness(self):
        """Test that jitter uses cryptographically secure randomness."""
        with patch('secrets.randbelow') as mock_randbelow:
            mock_randbelow.return_value = 500  # 50% jitter
            
            config = RetryConfig(initial_delay=2.0, jitter=True)
            client = ResilientHTTPClient(retry_config=config)
            
            delay = client._calculate_delay(1)  # Base delay would be 4.0
            
            # Should use secrets.randbelow for secure randomness
            mock_randbelow.assert_called_once_with(1000)
            # Delay should be base * (0.5 + 0.25) = 4.0 * 0.75 = 3.0
            assert delay == 3.0