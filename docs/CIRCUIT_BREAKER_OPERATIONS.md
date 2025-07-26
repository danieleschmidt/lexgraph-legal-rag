# Circuit Breaker Operations Runbook

## Overview
This runbook provides operational guidance for monitoring, troubleshooting, and managing circuit breaker events in the LexGraph Legal RAG system.

## Quick Reference

### Circuit States
- **CLOSED**: ✅ Normal operation, all requests allowed
- **HALF_OPEN**: ⚠️ Testing recovery, limited requests allowed  
- **OPEN**: ❌ Service failing, all requests blocked (returns 503)

### Key Metrics
- **failure_count**: Number of consecutive failures recorded
- **success_count**: Number of consecutive successes (in HALF_OPEN state)
- **last_failure_time**: Timestamp of last recorded failure

## Monitoring and Alerting

### Critical Alerts

#### Circuit Breaker Opened
**Alert Condition**: Circuit breaker state changes to OPEN
**Severity**: WARNING
**Action Required**: Investigate upstream service health within 5 minutes

```python
# Alert query example (for monitoring systems)
circuit_breaker_state == "open"
```

**Response Steps**:
1. Check upstream service health dashboard
2. Verify network connectivity
3. Review recent deployments or configuration changes
4. Escalate to service owner if service is confirmed down

#### Circuit Breaker Flapping
**Alert Condition**: Circuit state changes more than 3 times in 15 minutes
**Severity**: CRITICAL
**Action Required**: Immediate investigation required

```python
# Alert query example
count_by(circuit_state_changes) > 3 IN last(15m)
```

**Response Steps**:
1. Review circuit breaker configuration (thresholds may be too sensitive)
2. Check for intermittent network issues
3. Analyze request patterns for potential DDoS or unusual load
4. Consider temporarily disabling circuit breaker if causing more harm

### Monitoring Dashboards

#### Key Metrics to Track
```python
# Circuit breaker state distribution
circuit_breaker_state_gauge{service="legal-api"}

# Failure rate
circuit_breaker_failures_total{service="legal-api"} / 
circuit_breaker_requests_total{service="legal-api"}

# Time in each state
circuit_breaker_state_duration_seconds{service="legal-api", state="open"}

# Recovery time
circuit_breaker_recovery_duration_seconds{service="legal-api"}
```

#### Recommended Dashboard Panels
1. **Circuit State Timeline**: Shows state transitions over time
2. **Request Success Rate**: Overall success rate including circuit breaker blocks
3. **Recovery Time**: Time spent in OPEN state before successful recovery
4. **Error Rate by Service**: Breakdown of failures by upstream service

## Operational Procedures

### Health Check Integration

```python
async def circuit_breaker_health_check() -> Dict[str, Any]:
    """Health check endpoint that includes circuit breaker status."""
    client = get_legal_api_client()
    status = client.get_circuit_status()
    
    health_status = {
        "service": "legal-api-client",
        "status": "healthy" if status['state'] == 'closed' else "degraded",
        "circuit_breaker": {
            "state": status['state'],
            "failure_count": status['failure_count'],
            "last_failure": status.get('last_failure_time', 0)
        }
    }
    
    # Add warning if circuit has been open for too long
    if status['state'] == 'open':
        open_duration = time.time() - status['last_failure_time']
        if open_duration > 300:  # 5 minutes
            health_status['warnings'] = [
                f"Circuit breaker has been open for {open_duration:.0f} seconds"
            ]
    
    return health_status
```

### Graceful Degradation Strategy

```python
async def get_legal_data_with_fallback(query: str) -> Dict[str, Any]:
    """Example of graceful degradation when circuit is open."""
    try:
        # Primary path through circuit breaker protected client
        response = await legal_api_client.get("/search", params={"q": query})
        return {
            "results": response.json(),
            "source": "api",
            "degraded": False
        }
    except HTTPStatusError as e:
        if e.response.status_code == 503:  # Circuit breaker open
            # Fallback to cached results
            cached_results = await get_cached_legal_data(query)
            return {
                "results": cached_results,
                "source": "cache",
                "degraded": True,
                "message": "Using cached data due to service unavailability"
            }
        raise  # Re-raise other HTTP errors
```

## Troubleshooting Guide

### Common Issues and Solutions

#### Issue: Circuit Breaker Opens Frequently
**Symptoms**: 
- Multiple circuit opening alerts per hour
- High failure rate in upstream service metrics
- Users reporting intermittent service issues

**Possible Causes**:
1. Upstream service is genuinely failing
2. Circuit breaker thresholds are too sensitive
3. Network connectivity issues
4. Authentication/authorization failures

**Troubleshooting Steps**:
```bash
# 1. Check circuit breaker configuration
curl http://localhost:8000/health/circuit-breaker

# 2. Check upstream service directly
curl -H "Authorization: Bearer $API_KEY" https://legal-api.com/health

# 3. Review recent logs for error patterns
grep "Circuit breaker" /var/log/lexgraph/app.log | tail -50

# 4. Check network connectivity
ping legal-api.com
dig legal-api.com
```

**Resolution**:
- If upstream is failing: Contact service provider
- If thresholds too sensitive: Increase `failure_threshold` or `recovery_timeout`
- If network issues: Investigate network configuration

#### Issue: Circuit Breaker Stuck Open
**Symptoms**:
- Circuit remains in OPEN state for extended periods
- Upstream service appears healthy
- No automatic recovery occurring

**Possible Causes**:
1. Recovery timeout is too long
2. Success threshold is too high
3. Upstream service returns unexpected error codes during recovery
4. Bug in circuit breaker state machine

**Troubleshooting Steps**:
```python
# Check circuit breaker internal state
status = client.get_circuit_status()
print(f"State: {status['state']}")
print(f"Last failure: {status['last_failure_time']}")
print(f"Current time: {time.time()}")
print(f"Time since failure: {time.time() - status['last_failure_time']}")

# Manually test upstream service
response = await httpx.get("https://legal-api.com/health")
print(f"Direct test status: {response.status_code}")
```

**Resolution**:
1. Reduce `recovery_timeout` if too conservative
2. Lower `success_threshold` if recovery failing
3. Check for authentication issues in recovery requests
4. Restart service if state machine appears stuck

#### Issue: Circuit Breaker Not Opening When It Should
**Symptoms**:
- Upstream service is clearly failing
- Users experiencing timeouts and errors
- Circuit breaker remains in CLOSED state

**Possible Causes**:
1. Failure threshold is too high
2. Errors are not being properly detected as failures
3. Circuit breaker configuration issue

**Troubleshooting Steps**:
```python
# Check if failures are being recorded
client = get_legal_api_client()
initial_status = client.get_circuit_status()

# Make a request that should fail
try:
    await client.get("/nonexistent-endpoint")
except Exception as e:
    print(f"Expected error: {e}")

# Check if failure was recorded
final_status = client.get_circuit_status()
print(f"Failure count changed: {initial_status['failure_count']} -> {final_status['failure_count']}")
```

**Resolution**:
1. Lower `failure_threshold` for more sensitive detection
2. Verify error types are properly classified as failures
3. Check circuit breaker is properly configured and enabled

## Configuration Management

### Production Configuration
```python
# Recommended production settings
CIRCUIT_BREAKER_CONFIG = {
    "failure_threshold": 5,      # Allow 5 failures before opening
    "recovery_timeout": 60.0,    # Test recovery every minute
    "success_threshold": 3       # Require 3 successes to close
}

RETRY_CONFIG = {
    "max_retries": 3,
    "initial_delay": 1.0,
    "max_delay": 60.0,
    "exponential_base": 2.0,
    "jitter": True
}
```

### Environment-Specific Tuning

#### Development/Testing
```python
# More permissive for testing
DEV_CIRCUIT_CONFIG = {
    "failure_threshold": 10,     # Higher threshold
    "recovery_timeout": 30.0,    # Faster recovery
    "success_threshold": 2       # Easier to close
}
```

#### Production
```python
# Conservative for production stability
PROD_CIRCUIT_CONFIG = {
    "failure_threshold": 5,      # Standard threshold
    "recovery_timeout": 60.0,    # Standard recovery time
    "success_threshold": 3       # Conservative closing
}
```

### Dynamic Configuration Updates

```python
def update_circuit_breaker_config(service: str, config: Dict[str, Any]):
    """Update circuit breaker configuration at runtime."""
    client = get_client_for_service(service)
    
    # Validate configuration
    if config.get('failure_threshold', 0) < 1:
        raise ValueError("failure_threshold must be >= 1")
    
    # Apply new configuration
    new_circuit_config = CircuitBreakerConfig(**config)
    client.circuit_breaker.config = new_circuit_config
    
    logger.info(f"Updated circuit breaker config for {service}: {config}")
```

## Emergency Procedures

### Circuit Breaker Override
For emergency situations where circuit breaker is causing more harm than good:

```python
def emergency_disable_circuit_breaker(service: str, duration_minutes: int = 30):
    """Temporarily disable circuit breaker for emergency recovery."""
    client = get_client_for_service(service)
    
    # Override circuit breaker to always allow requests
    original_can_execute = client.circuit_breaker.can_execute
    client.circuit_breaker.can_execute = lambda: True
    
    # Set timer to re-enable
    def re_enable():
        client.circuit_breaker.can_execute = original_can_execute
        logger.warning(f"Circuit breaker re-enabled for {service}")
    
    # Schedule re-enabling
    threading.Timer(duration_minutes * 60, re_enable).start()
    
    logger.critical(f"Circuit breaker DISABLED for {service} for {duration_minutes} minutes")
```

### Service Restart Procedure
When circuit breaker state becomes inconsistent:

```bash
#!/bin/bash
# emergency_circuit_reset.sh

SERVICE_NAME="lexgraph-legal-rag"
CIRCUIT_RESET_ENDPOINT="http://localhost:8000/admin/circuit-breaker/reset"

echo "Resetting circuit breaker state..."

# Reset circuit breaker state
curl -X POST "$CIRCUIT_RESET_ENDPOINT" \
     -H "Authorization: Bearer $ADMIN_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"action": "reset_all"}'

# Verify reset
sleep 2
curl "$CIRCUIT_RESET_ENDPOINT" | jq '.circuit_breakers[].state'

echo "Circuit breaker reset complete"
```

## Metrics and SLA Monitoring

### Key Performance Indicators
1. **Circuit Availability**: Percentage of time circuit is CLOSED
2. **Mean Time to Recovery**: Average time from OPEN to CLOSED
3. **False Positive Rate**: Circuits opened due to transient issues
4. **Service Resilience**: System availability despite upstream failures

### SLA Targets
- Circuit Availability: > 99.5%
- Mean Time to Recovery: < 2 minutes
- False Positive Rate: < 5%
- Service Resilience: > 99.9% availability with degraded mode

### Reporting
Generate weekly circuit breaker health reports:

```python
def generate_circuit_breaker_report(start_date: datetime, end_date: datetime):
    """Generate weekly circuit breaker performance report."""
    return {
        "period": f"{start_date.date()} to {end_date.date()}",
        "circuits": {
            "legal-api": {
                "total_requests": 15420,
                "failures": 45,
                "circuit_opens": 3,
                "total_downtime_minutes": 12,
                "availability_percent": 99.87,
                "mttr_minutes": 4.2
            }
        },
        "recommendations": [
            "Consider increasing failure_threshold for legal-api (3 opens this week)",
            "Investigate pattern of failures on Tuesday morning"
        ]
    }
```

---

**Emergency Contacts**:
- On-call Engineer: +1-555-0199
- Service Owner: engineering-team@company.com
- Escalation: sre-team@company.com