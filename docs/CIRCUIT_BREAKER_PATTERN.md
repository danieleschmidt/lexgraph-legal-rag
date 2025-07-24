# Circuit Breaker Pattern

## Overview

The Circuit Breaker pattern is a stability pattern that prevents cascading failures in distributed systems by monitoring external service calls and temporarily "opening" the circuit when failures exceed a threshold. This pattern is implemented in the `ResilientHTTPClient` class to protect against failing external services.

### Key Benefits
- **Fail Fast**: Prevents wasteful calls to failing services
- **Service Protection**: Reduces load on struggling downstream services  
- **Graceful Degradation**: Allows services to recover naturally
- **Observability**: Provides clear metrics on service health

## Implementation

The circuit breaker is implemented through two main classes:

- `CircuitBreaker`: Core state machine managing circuit states
- `ResilientHTTPClient`: HTTP client with integrated circuit breaker protection

### Circuit States

The circuit breaker operates in three distinct states:

1. **CLOSED** (Normal Operation)
   - All requests are allowed through
   - Failures are counted but don't block requests
   - Transitions to OPEN when failure_threshold is exceeded

2. **OPEN** (Failing State) 
   - All requests are immediately blocked
   - Returns HTTP 503 errors without making actual calls
   - Transitions to HALF_OPEN after recovery_timeout

3. **HALF_OPEN** (Testing Recovery)
   - Limited requests are allowed through to test service health
   - Transitions to CLOSED after success_threshold successful requests
   - Transitions back to OPEN on any failure

## Configuration

### CircuitBreakerConfig Parameters

- **failure_threshold** (int, default: 5): Number of consecutive failures before opening circuit
- **recovery_timeout** (float, default: 60.0): Seconds to wait before testing recovery in HALF_OPEN state
- **success_threshold** (int, default: 3): Number of consecutive successes needed to close circuit

### RetryConfig Parameters

- **max_retries** (int, default: 3): Maximum number of retry attempts
- **initial_delay** (float, default: 1.0): Initial delay between retries in seconds
- **max_delay** (float, default: 60.0): Maximum delay between retries
- **exponential_base** (float, default: 2.0): Base for exponential backoff calculation
- **jitter** (bool, default: True): Add randomization to prevent thundering herd
- **retryable_status_codes** (tuple): HTTP status codes that trigger retries (429, 500, 502, 503, 504)

## States and Transitions

```
    [CLOSED] ──failure_threshold──> [OPEN]
        ^                              │
        │                              │ recovery_timeout
        │                              v
        └──success_threshold── [HALF_OPEN]
                                      │
                                      │ any failure
                                      v
                                   [OPEN]
```

### State Transition Logic

1. **CLOSED → OPEN**: When failure count reaches `failure_threshold`
2. **OPEN → HALF_OPEN**: After `recovery_timeout` seconds have elapsed
3. **HALF_OPEN → CLOSED**: After `success_threshold` consecutive successes
4. **HALF_OPEN → OPEN**: On any failure during testing phase

## Usage Examples

### Basic Usage

```python
from lexgraph_legal_rag.http_client import ResilientHTTPClient, CircuitBreakerConfig, RetryConfig

# Create client with default configuration
async with ResilientHTTPClient("https://api.example.com") as client:
    response = await client.get("/health")
    print(response.status_code)
```

### Custom Configuration

```python
# Configure circuit breaker for aggressive failure detection
circuit_config = CircuitBreakerConfig(
    failure_threshold=3,      # Open after 3 failures
    recovery_timeout=30.0,    # Test recovery every 30 seconds
    success_threshold=2       # Close after 2 successes
)

# Configure retry behavior
retry_config = RetryConfig(
    max_retries=5,
    initial_delay=0.5,
    max_delay=30.0,
    exponential_base=1.5
)

client = ResilientHTTPClient(
    base_url="https://api.legal-service.com",
    circuit_config=circuit_config,
    retry_config=retry_config,
    timeout=10.0
)
```

### Legal API Integration

```python
from lexgraph_legal_rag.http_client import create_legal_api_client

# Convenience function for legal API services
client = create_legal_api_client(
    base_url="https://legal-data-api.com",
    api_key="your-api-key-here"
)

try:
    response = await client.get("/cases/search", params={"query": "contract law"})
    cases = response.json()
except HTTPStatusError as e:
    if e.response.status_code == 503:
        # Circuit breaker is open - service is unavailable
        print("Legal API service is temporarily unavailable")
    else:
        # Handle other HTTP errors
        print(f"API error: {e}")
```

### Monitoring Circuit Status

```python
# Check circuit breaker health
status = client.get_circuit_status()
print(f"Circuit State: {status['state']}")
print(f"Failure Count: {status['failure_count']}")
print(f"Success Count: {status['success_count']}")

# Use status for health checks
if status['state'] == 'open':
    # Alert operations team
    logger.warning("Circuit breaker is open - external service failing")
```

## Monitoring and Observability

### Logging Events

The circuit breaker logs important state transitions:

- Circuit opening: `WARNING: Circuit breaker opened after N failures`
- Half-open transition: `INFO: Circuit breaker moved to HALF_OPEN state`  
- Circuit closing: `INFO: Circuit breaker moved to CLOSED state`
- Half-open failure: `WARNING: Circuit breaker reopened during half-open state`

### Metrics Integration

For production monitoring, integrate circuit breaker status with metrics systems:

```python
from lexgraph_legal_rag.observability import track_error, SYSTEM_HEALTH

# Track circuit breaker state changes
status = client.get_circuit_status()
if status['state'] == 'open':
    # Update system health metric
    SYSTEM_HEALTH.set(0.5)  # Degraded service
    
    # Track the circuit breaker opening
    track_error(
        Exception("Circuit breaker opened"),
        component="http_client",
        severity="warning"
    )
```

## Testing

### Unit Testing Circuit States

```python
from lexgraph_legal_rag.http_client import CircuitBreaker, CircuitBreakerConfig, CircuitState

def test_circuit_breaker_opening():
    config = CircuitBreakerConfig(failure_threshold=2)
    breaker = CircuitBreaker(config)
    
    # Initial state is CLOSED
    assert breaker._state == CircuitState.CLOSED
    assert breaker.can_execute() is True
    
    # Record failures
    breaker.record_failure()
    assert breaker.can_execute() is True  # Still closed
    
    breaker.record_failure() 
    assert breaker._state == CircuitState.OPEN  # Now open
    assert breaker.can_execute() is False
```

### Integration Testing with Mock Services

```python
import pytest
from unittest.mock import Mock, patch
from lexgraph_legal_rag.http_client import ResilientHTTPClient

@pytest.mark.asyncio
async def test_circuit_breaker_with_failing_service():
    client = ResilientHTTPClient("https://failing-service.com")
    
    # Mock httpx to always return 500 errors
    with patch.object(client._client, 'request') as mock_request:
        mock_request.side_effect = HTTPStatusError(
            message="Server Error",
            request=Mock(),
            response=Mock(status_code=500)
        )
        
        # After enough failures, circuit should open
        for _ in range(6):  # > failure_threshold
            with pytest.raises(HTTPStatusError):
                await client.get("/test")
        
        # Circuit should now be open
        status = client.get_circuit_status()
        assert status['state'] == 'open'
```

## Best Practices

### Configuration Guidelines

1. **failure_threshold**: Set based on service SLA
   - Critical services: 3-5 failures
   - Non-critical services: 5-10 failures

2. **recovery_timeout**: Balance between service recovery and user experience
   - Fast services: 30-60 seconds
   - Slow services: 2-5 minutes

3. **success_threshold**: Conservative approach prevents flapping
   - Usually 2-3 consecutive successes
   - Higher for critical services

### Operational Considerations

1. **Monitor Circuit State**: Alert when circuits open frequently
2. **Log Context**: Include request details when circuits trip
3. **Graceful Degradation**: Have fallback mechanisms when circuits open
4. **Testing**: Regularly test circuit breaker behavior in staging

### Common Patterns

#### Fallback Strategy
```python
async def get_legal_precedents(query: str) -> List[Dict]:
    try:
        response = await legal_client.get("/precedents", params={"q": query})
        return response.json()
    except HTTPStatusError as e:
        if e.response.status_code == 503:  # Circuit breaker open
            # Fallback to cached results
            return get_cached_precedents(query)
        raise
```

#### Health Check Integration
```python
async def health_check() -> Dict[str, str]:
    status = legal_client.get_circuit_status()
    return {
        "legal_api": "healthy" if status['state'] == 'closed' else "degraded",
        "circuit_state": status['state']
    }
```

#### Bulk Operations
```python
async def process_legal_documents(documents: List[str]) -> List[Dict]:
    results = []
    for doc in documents:
        try:
            # Process each document independently
            result = await legal_client.post("/analyze", json={"document": doc})
            results.append(result.json())
        except HTTPStatusError as e:
            if e.response.status_code == 503:
                # Circuit open - skip this document and continue
                logger.warning(f"Skipping document due to circuit breaker: {doc[:50]}...")
                continue
            raise
    return results
```

---

*This documentation covers the circuit breaker pattern implementation in the LexGraph Legal RAG system. For implementation details, see `src/lexgraph_legal_rag/http_client.py`.*