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

## Operational Runbook

### Circuit Breaker Events Response Guide

#### Event: Circuit Opens
**Trigger**: `failure_count >= failure_threshold`

**Immediate Actions** (within 2 minutes):
1. **Alert Acknowledgment**: Acknowledge the alert and notify team
2. **Service Health Check**: Verify external service status via:
   ```bash
   curl -I https://external-service.com/health
   ```
3. **Check System Logs**: Review recent error patterns:
   ```bash
   grep "Circuit breaker opened" /var/log/app.log | tail -10
   ```
4. **User Communication**: If critical service, notify users of degraded functionality

**Investigation Steps** (within 5 minutes):
1. **Network Connectivity**: Test basic connectivity
   ```bash
   ping external-service.com
   traceroute external-service.com
   ```
2. **Authentication**: Verify API credentials are valid
3. **Service Status Page**: Check if service provider reports outages
4. **Resource Usage**: Check if local system resources are exhausted

**Recovery Actions**:
- If service is healthy: Review and potentially adjust `failure_threshold`
- If service is down: Wait for service recovery or implement fallback
- If network issues: Engage network operations team

#### Event: Circuit Half-Opens
**Trigger**: `recovery_timeout` elapsed after circuit opened

**Actions**:
1. **Monitor Test Requests**: Watch for successful recovery attempts
2. **Prepare for Failure**: Circuit may reopen if tests fail
3. **Log Analysis**: Review what caused the initial failures

#### Event: Circuit Closes
**Trigger**: `success_count >= success_threshold` in HALF_OPEN state

**Actions**:
1. **Verify Normal Operation**: Confirm requests flowing normally
2. **Clear Alerts**: Resolve related monitoring alerts
3. **Post-Incident Review**: Schedule review if outage was significant

### Troubleshooting Guide

#### Problem: Circuit Opens Too Frequently
**Symptoms**: Circuit repeatedly opens under normal load

**Root Causes & Solutions**:
- **Low failure_threshold**: Increase threshold for the service
- **Network instability**: Work with network team to resolve connectivity issues
- **Service overload**: Coordinate with service provider on capacity
- **Authentication issues**: Verify and refresh API credentials

**Diagnostic Commands**:
```bash
# Check failure patterns
grep -C5 "Circuit breaker opened" /var/log/app.log

# Monitor real-time circuit state
tail -f /var/log/app.log | grep -i circuit

# Test service directly
curl -w "@curl-format.txt" https://service.com/endpoint
```

#### Problem: Circuit Never Opens
**Symptoms**: Service clearly failing but circuit stays closed

**Root Causes & Solutions**:
- **High failure_threshold**: Lower threshold for testing
- **Wrong status codes**: Check if error codes are in retryable list
- **Circuit not integrated**: Verify circuit breaker is actually being used

**Diagnostic Steps**:
1. Enable debug logging: Set log level to DEBUG
2. Check retry configuration: Verify status codes trigger failures
3. Test with lower threshold: Temporarily reduce `failure_threshold`

#### Problem: Circuit Stuck in Half-Open
**Symptoms**: Circuit doesn't transition to CLOSED or OPEN

**Root Causes & Solutions**:
- **Intermittent service issues**: Service failing sporadically during tests
- **High success_threshold**: Reduce threshold for easier recovery
- **Long recovery_timeout**: Adjust timeout for service characteristics

#### Problem: False Positive Circuit Openings
**Symptoms**: Circuit opens for temporary, recoverable issues

**Root Causes & Solutions**:
- **Aggressive retry config**: Increase max_retries and delays
- **Status code misconfiguration**: Review which codes should be retryable
- **Missing graceful degradation**: Implement fallback mechanisms

## Monitoring and Alerting Configuration

### Key Metrics to Track

#### Circuit Breaker State Metrics
```yaml
# Prometheus metrics example
- name: http_circuit_breaker_state
  help: "Current state of circuit breaker (0=CLOSED, 1=OPEN, 2=HALF_OPEN)"
  type: gauge
  labels: [service_name, endpoint]

- name: http_circuit_breaker_failures_total
  help: "Total number of recorded failures"
  type: counter
  labels: [service_name, endpoint]

- name: http_circuit_breaker_successes_total
  help: "Total number of recorded successes"
  type: counter
  labels: [service_name, endpoint]
```

#### Request Performance Metrics
```yaml
- name: http_request_duration_seconds
  help: "HTTP request duration"
  type: histogram
  labels: [service_name, endpoint, status_code]

- name: http_requests_blocked_total
  help: "Requests blocked by circuit breaker"
  type: counter
  labels: [service_name, endpoint]
```

### Alerting Rules

#### Critical Alerts
```yaml
# Circuit Breaker Open Alert
- alert: CircuitBreakerOpen
  expr: http_circuit_breaker_state > 0
  for: 1m
  labels:
    severity: critical
  annotations:
    summary: "Circuit breaker is open for {{ $labels.service_name }}"
    description: "Circuit breaker has been open for {{ $labels.service_name }} service for more than 1 minute"
    runbook_url: "https://docs.company.com/runbooks/circuit-breaker"

# High Request Failure Rate
- alert: HighHTTPFailureRate
  expr: rate(http_circuit_breaker_failures_total[5m]) / rate(http_requests_total[5m]) > 0.1
  for: 2m
  labels:
    severity: warning
  annotations:
    summary: "High failure rate for {{ $labels.service_name }}"
    description: "Failure rate is {{ $value | humanizePercentage }} for {{ $labels.service_name }}"
```

#### Warning Alerts
```yaml
# Circuit Breaker Half-Open
- alert: CircuitBreakerHalfOpen
  expr: http_circuit_breaker_state == 2
  for: 30s
  labels:
    severity: warning
  annotations:
    summary: "Circuit breaker is testing recovery for {{ $labels.service_name }}"
    description: "Circuit breaker is in half-open state, testing service recovery"
```

### Dashboard Configuration

#### Grafana Dashboard Panels
```json
{
  "title": "Circuit Breaker Status",
  "type": "stat",
  "targets": [
    {
      "expr": "http_circuit_breaker_state",
      "legendFormat": "{{ service_name }}"
    }
  ],
  "fieldConfig": {
    "mappings": [
      {"value": 0, "text": "CLOSED", "color": "green"},
      {"value": 1, "text": "OPEN", "color": "red"},
      {"value": 2, "text": "HALF_OPEN", "color": "yellow"}
    ]
  }
}
```

#### Request Success Rate Panel
```json
{
  "title": "Request Success Rate",
  "type": "graph",
  "targets": [
    {
      "expr": "rate(http_circuit_breaker_successes_total[5m]) / (rate(http_circuit_breaker_successes_total[5m]) + rate(http_circuit_breaker_failures_total[5m]))",
      "legendFormat": "{{ service_name }} Success Rate"
    }
  ]
}
```

### Log Monitoring

#### Important Log Patterns to Monitor
```bash
# Circuit state changes
pattern: "Circuit breaker (opened|moved to|reopened)"
action: Generate event for state change tracking

# High frequency of blocked requests
pattern: "Circuit breaker is open"
threshold: > 10 occurrences per minute
action: Alert on service unavailability impact

# Recovery patterns
pattern: "Circuit breaker moved to CLOSED"
action: Clear related alerts and log recovery time
```

#### Log Aggregation Queries
```javascript
// ELK Stack query for circuit breaker events
{
  "query": {
    "bool": {
      "must": [
        {"match": {"message": "circuit breaker"}},
        {"range": {"@timestamp": {"gte": "now-1h"}}}
      ]
    }
  },
  "aggs": {
    "state_changes": {
      "terms": {"field": "service_name"},
      "aggs": {
        "states": {"terms": {"field": "circuit_state"}}
      }
    }
  }
}
```

### Health Check Integration

#### Application Health Endpoint
```python
@app.get("/health")
async def health_check():
    """Application health check including circuit breaker status."""
    circuit_status = legal_client.get_circuit_status()
    
    health_status = {
        "status": "healthy",
        "services": {
            "legal_api": {
                "status": "healthy" if circuit_status['state'] == 'closed' else "degraded",
                "circuit_state": circuit_status['state'],
                "failure_count": circuit_status['failure_count']
            }
        }
    }
    
    # Overall health based on critical services
    if circuit_status['state'] == 'open':
        health_status["status"] = "degraded"
    
    status_code = 200 if health_status["status"] == "healthy" else 503
    return JSONResponse(content=health_status, status_code=status_code)
```

#### Load Balancer Integration
```yaml
# HAProxy health check configuration
backend legal_service
    option httpchk GET /health
    http-check expect status 200
    server app1 app1:8000 check inter 10s fall 3 rise 2
```

## Configuration Parameters Reference

### Environment-Specific Recommendations

#### Development Environment
```python
CircuitBreakerConfig(
    failure_threshold=10,      # Higher tolerance for dev instability
    recovery_timeout=30.0,     # Faster recovery for dev cycles
    success_threshold=2        # Quick recovery
)
```

#### Staging Environment
```python
CircuitBreakerConfig(
    failure_threshold=5,       # Production-like sensitivity
    recovery_timeout=60.0,     # Standard recovery time
    success_threshold=3        # Conservative recovery
)
```

#### Production Environment
```python
CircuitBreakerConfig(
    failure_threshold=3,       # Low tolerance for failures
    recovery_timeout=120.0,    # Allow time for proper recovery
    success_threshold=5        # High confidence in recovery
)
```

### Service-Specific Configuration

#### Critical Legal APIs
```python
# Westlaw, LexisNexis, etc.
CircuitBreakerConfig(
    failure_threshold=2,       # Very sensitive
    recovery_timeout=300.0,    # Allow time for service recovery
    success_threshold=5        # High confidence required
)
```

#### Secondary Data Sources
```python
# Case databases, law journals, etc.
CircuitBreakerConfig(
    failure_threshold=5,       # More tolerant
    recovery_timeout=60.0,     # Standard recovery
    success_threshold=3        # Standard confidence
)
```

#### Internal Services
```python
# Document processing, indexing, etc.
CircuitBreakerConfig(
    failure_threshold=3,       # Moderate sensitivity
    recovery_timeout=30.0,     # Quick recovery expected
    success_threshold=2        # Quick restoration
)
```

---

*This documentation covers the circuit breaker pattern implementation in the LexGraph Legal RAG system. For implementation details, see `src/lexgraph_legal_rag/http_client.py`.*