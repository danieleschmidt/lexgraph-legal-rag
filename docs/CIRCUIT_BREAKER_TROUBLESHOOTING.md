# Circuit Breaker Troubleshooting Guide

## Quick Diagnosis

### Symptom-Based Troubleshooting Table

| Symptom | Likely Cause | Quick Check | Resolution |
|---------|-------------|-------------|-------------|
| 503 errors for all requests | Circuit breaker open | `curl /health/circuit-breaker` | Check upstream service health |
| Intermittent 503 errors | Circuit flapping | Check state transitions in logs | Adjust thresholds or fix upstream |
| No circuit opening despite failures | Threshold too high | Review failure count vs threshold | Lower `failure_threshold` |
| Circuit stuck open | Recovery issues | Check recovery timeout | Verify upstream health, lower timeout |
| High latency | Retry delays | Check retry configuration | Optimize retry parameters |

## Diagnostic Commands

### Basic Health Check
```bash
# Check circuit breaker status
curl -s http://localhost:8000/health/circuit-breaker | jq '.'

# Expected output:
{
  "timestamp": "2025-07-25T12:00:00Z",
  "circuit_breakers": {
    "legal-api": {
      "state": "closed",
      "failure_count": 0,
      "success_count": 0,
      "healthy": true,
      "configuration": {
        "failure_threshold": 5,
        "recovery_timeout": 60.0,
        "success_threshold": 3
      }
    }
  }
}
```

### Testing Circuit Breaker Functionality
```python
# Test script: verify_circuit_breaker.py
import asyncio
import httpx
from lexgraph_legal_rag.http_client import ResilientHTTPClient, CircuitBreakerConfig

async def test_circuit_breaker():
    """Comprehensive circuit breaker functionality test."""
    
    # Configure sensitive circuit breaker for testing
    config = CircuitBreakerConfig(
        failure_threshold=2,
        recovery_timeout=5.0,
        success_threshold=1
    )
    
    client = ResilientHTTPClient(
        base_url="http://httpbin.org",  # Test service
        circuit_config=config
    )
    
    print("Testing circuit breaker functionality...")
    
    # Test 1: Normal operation
    try:
        response = await client.get("/status/200")
        print(f"✅ Normal request: {response.status_code}")
    except Exception as e:
        print(f"❌ Normal request failed: {e}")
    
    # Test 2: Force failures to open circuit
    print("\nForcing failures to test circuit opening...")
    for i in range(3):
        try:
            response = await client.get("/status/500")
        except httpx.HTTPStatusError as e:
            print(f"Expected failure {i+1}: {e.response.status_code}")
    
    # Check if circuit opened
    status = client.get_circuit_status()
    print(f"Circuit state after failures: {status['state']}")
    
    # Test 3: Verify circuit blocks requests
    try:
        response = await client.get("/status/200")
        print("❌ Request should have been blocked!")
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 503:
            print("✅ Circuit breaker correctly blocked request")
        else:
            print(f"❌ Unexpected error: {e}")
    
    # Test 4: Wait for recovery and test
    print(f"\nWaiting {config.recovery_timeout}s for recovery...")
    await asyncio.sleep(config.recovery_timeout + 1)
    
    try:
        response = await client.get("/status/200")
        print(f"✅ Recovery successful: {response.status_code}")
    except Exception as e:
        print(f"❌ Recovery failed: {e}")
    
    # Final status check
    final_status = client.get_circuit_status()
    print(f"Final circuit state: {final_status['state']}")
    
    await client._client.aclose()

# Run test
asyncio.run(test_circuit_breaker())
```

## Common Issues and Solutions

### Issue 1: Circuit Breaker Not Opening

**Problem**: Service is clearly failing but circuit breaker remains closed

**Diagnosis Steps**:
```python
# Check if failures are being recorded
async def diagnose_failure_detection():
    client = get_legal_api_client()
    
    # Record initial state
    initial_status = client.get_circuit_status()
    print(f"Initial failure count: {initial_status['failure_count']}")
    
    # Make request that should fail
    try:
        await client.get("/definitely-not-exists")
    except Exception as e:
        print(f"Request failed with: {type(e).__name__}: {e}")
    
    # Check if failure was recorded
    final_status = client.get_circuit_status()
    print(f"Final failure count: {final_status['failure_count']}")
    
    if final_status['failure_count'] == initial_status['failure_count']:
        print("❌ Failure not recorded - check error classification")
    else:
        print("✅ Failure recorded correctly")
```

**Common Causes**:
1. **Wrong status codes**: Non-retryable errors (4xx) don't count as failures
2. **Exception handling**: Exceptions caught before reaching circuit breaker
3. **Configuration**: `failure_threshold` set too high

**Solutions**:
```python
# Fix 1: Adjust retryable status codes
retry_config = RetryConfig(
    retryable_status_codes=(400, 401, 403, 429, 500, 502, 503, 504)  # Include 4xx errors
)

# Fix 2: Lower failure threshold
circuit_config = CircuitBreakerConfig(
    failure_threshold=3  # Reduced from default 5
)

# Fix 3: Check exception handling
try:
    response = await client.get("/api/endpoint")
except httpx.RequestError as e:
    # These should be counted as failures
    logger.error(f"Network error: {e}")
    raise  # Don't catch and suppress
```

### Issue 2: Circuit Breaker Flapping

**Problem**: Circuit frequently opens and closes, causing instability

**Diagnosis**:
```bash
# Check state transition frequency
grep "Circuit breaker" /var/log/app.log | grep -E "(opened|closed)" | tail -20

# Look for pattern like:
# 2025-07-25 12:01:00 Circuit breaker opened after 5 failures
# 2025-07-25 12:02:30 Circuit breaker moved to CLOSED state  
# 2025-07-25 12:03:45 Circuit breaker opened after 5 failures
```

**Root Causes**:
1. **Thresholds too sensitive**: Circuit opens on minor issues
2. **Upstream instability**: Service having intermittent issues
3. **Network problems**: Transient connectivity issues

**Solutions**:
```python
# Solution 1: Increase thresholds
circuit_config = CircuitBreakerConfig(
    failure_threshold=10,      # More tolerant
    success_threshold=5,       # Require more successes
    recovery_timeout=120.0     # Wait longer before testing
)

# Solution 2: Add jitter to prevent thundering herd
retry_config = RetryConfig(
    jitter=True,               # Randomize delays
    initial_delay=2.0,         # Longer initial delay
    max_delay=120.0            # Longer max delay
)

# Solution 3: Implement backoff for circuit state changes
class StableCircuitBreaker(CircuitBreaker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_state_change = 0
        self._min_state_duration = 30.0  # Minimum time in each state
    
    def _can_change_state(self) -> bool:
        return time.time() - self._last_state_change >= self._min_state_duration
```

### Issue 3: Circuit Stuck Open

**Problem**: Circuit remains open despite upstream service recovery

**Diagnosis Script**:
```python
import time
import httpx

async def diagnose_stuck_circuit():
    """Diagnose why circuit won't close."""
    client = get_legal_api_client()
    status = client.get_circuit_status()
    
    print(f"Current state: {status['state']}")
    print(f"Failure count: {status['failure_count']}")
    print(f"Last failure time: {status.get('last_failure_time', 'N/A')}")
    
    if status['state'] == 'open':
        # Calculate time since last failure
        if 'last_failure_time' in status:
            time_since_failure = time.time() - status['last_failure_time']
            recovery_timeout = client.circuit_breaker.config.recovery_timeout
            
            print(f"Time since failure: {time_since_failure:.1f}s")
            print(f"Recovery timeout: {recovery_timeout}s")
            
            if time_since_failure >= recovery_timeout:
                print("⚠️ Circuit should be testing recovery")
                
                # Test upstream service directly
                try:
                    async with httpx.AsyncClient() as direct_client:
                        response = await direct_client.get(f"{client.base_url}/health")
                        print(f"Direct test result: {response.status_code}")
                        
                        if response.status_code == 200:
                            print("❌ Upstream is healthy but circuit stuck")
                        else:
                            print("✅ Upstream still failing, circuit behavior correct")
                            
                except Exception as e:
                    print(f"Direct test failed: {e}")
            else:
                print("✅ Still in recovery timeout period")
    
    # Test if circuit can execute
    can_execute = client.circuit_breaker.can_execute()
    print(f"Can execute: {can_execute}")
```

**Solutions**:
```python
# Solution 1: Manual circuit reset
def reset_circuit_breaker(client):
    """Manually reset circuit breaker to closed state."""
    client.circuit_breaker._state = CircuitState.CLOSED
    client.circuit_breaker._failure_count = 0
    client.circuit_breaker._success_count = 0
    print("Circuit breaker manually reset to CLOSED")

# Solution 2: Reduce recovery timeout
circuit_config = CircuitBreakerConfig(
    recovery_timeout=30.0  # Reduced from 60s
)

# Solution 3: Force state transition for testing
def force_half_open_state(client):
    """Force circuit to half-open for testing."""
    client.circuit_breaker._state = CircuitState.HALF_OPEN
    client.circuit_breaker._success_count = 0
    print("Forced circuit to HALF_OPEN for testing")
```

### Issue 4: High Latency Due to Retries

**Problem**: Requests taking too long due to excessive retry attempts

**Diagnosis**:
```python
import time

class TimingHTTPClient(ResilientHTTPClient):
    async def request(self, method: str, url: str, **kwargs):
        start_time = time.time()
        try:
            response = await super().request(method, url, **kwargs)
            elapsed = time.time() - start_time
            print(f"Request completed in {elapsed:.2f}s")
            return response
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"Request failed in {elapsed:.2f}s: {e}")
            raise

# Use timing client to identify slow requests
timing_client = TimingHTTPClient("https://slow-api.com")
```

**Solutions**:
```python
# Solution 1: Optimize retry configuration
retry_config = RetryConfig(
    max_retries=2,          # Reduce from 3
    initial_delay=0.5,      # Reduce from 1.0
    max_delay=10.0,         # Reduce from 60.0
    exponential_base=1.5    # Reduce from 2.0
)

# Solution 2: Set aggressive timeouts
client = ResilientHTTPClient(
    base_url="https://api.example.com",
    timeout=5.0,            # Fail fast
    retry_config=retry_config
)

# Solution 3: Implement timeout per retry
class FastFailHTTPClient(ResilientHTTPClient):
    async def request(self, method: str, url: str, **kwargs):
        # Decrease timeout with each retry
        base_timeout = self.timeout
        for attempt in range(self.retry_config.max_retries + 1):
            timeout = base_timeout / (attempt + 1)
            kwargs['timeout'] = timeout
            
            try:
                return await super().request(method, url, **kwargs)
            except httpx.TimeoutException:
                if attempt == self.retry_config.max_retries:
                    raise
                continue
```

### Issue 5: Memory Leaks in Circuit Breaker

**Problem**: Memory usage increases over time with circuit breaker

**Diagnosis**:
```python
import psutil
import gc
from memory_profiler import profile

@profile
def memory_leak_check():
    """Check for memory leaks in circuit breaker."""
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024
    
    # Create many circuit breaker instances
    clients = []
    for i in range(1000):
        client = ResilientHTTPClient(f"https://api-{i}.example.com")
        clients.append(client)
    
    mid_memory = process.memory_info().rss / 1024 / 1024
    print(f"Memory after creation: {mid_memory:.1f} MB (delta: {mid_memory - initial_memory:.1f} MB)")
    
    # Clear references
    clients.clear()
    gc.collect()
    
    final_memory = process.memory_info().rss / 1024 / 1024
    print(f"Memory after cleanup: {final_memory:.1f} MB (delta: {final_memory - initial_memory:.1f} MB)")
    
    if final_memory - initial_memory > 10:  # More than 10MB increase
        print("⚠️ Possible memory leak detected")
```

**Solutions**:
```python
# Solution 1: Proper cleanup
class ManagedHTTPClient(ResilientHTTPClient):
    def __del__(self):
        """Ensure proper cleanup."""
        if hasattr(self, '_client'):
            asyncio.run(self._client.aclose())
    
    async def close(self):
        """Explicit close method."""
        await self._client.aclose()

# Solution 2: Connection pooling
import asyncio
from typing import Dict

class ConnectionPool:
    def __init__(self):
        self._clients: Dict[str, ResilientHTTPClient] = {}
    
    def get_client(self, base_url: str) -> ResilientHTTPClient:
        """Get or create client for base URL."""
        if base_url not in self._clients:
            self._clients[base_url] = ResilientHTTPClient(base_url)
        return self._clients[base_url]
    
    async def close_all(self):
        """Close all clients."""
        for client in self._clients.values():
            await client._client.aclose()
        self._clients.clear()

# Global connection pool
connection_pool = ConnectionPool()
```

## Emergency Procedures

### Complete Circuit Breaker Bypass

```python
# Emergency bypass for production issues
def emergency_bypass_circuit_breaker(client: ResilientHTTPClient):
    """EMERGENCY: Completely bypass circuit breaker."""
    
    print("🚨 EMERGENCY: Bypassing circuit breaker")
    print("⚠️  This should only be used in critical situations")
    
    # Store original method
    original_can_execute = client.circuit_breaker.can_execute
    
    # Override to always return True
    client.circuit_breaker.can_execute = lambda: True
    
    # Log the bypass
    logger.critical(
        "Circuit breaker bypassed",
        service=getattr(client, 'service_name', 'unknown'),
        base_url=client.base_url
    )
    
    return original_can_execute  # Return original for restoration

# Restore circuit breaker
def restore_circuit_breaker(client: ResilientHTTPClient, original_method):
    """Restore circuit breaker functionality."""
    client.circuit_breaker.can_execute = original_method
    logger.info("Circuit breaker functionality restored")
```

### Service Restart Checklist

```bash
#!/bin/bash
# circuit_breaker_restart.sh

echo "🔄 Circuit Breaker Service Restart Checklist"

echo "1. Checking current circuit states..."
curl -s http://localhost:8000/health/circuit-breaker | jq '.circuit_breakers'

echo "2. Backing up current state..."
curl -s http://localhost:8000/health/circuit-breaker > /tmp/circuit_state_backup.json

echo "3. Gracefully shutting down service..."
pkill -SIGTERM lexgraph-service

echo "4. Waiting for graceful shutdown..."
sleep 10

echo "5. Starting service..."
systemctl start lexgraph-service

echo "6. Waiting for startup..."
sleep 30

echo "7. Verifying circuit breaker health..."
curl -f http://localhost:8000/health/circuit-breaker || echo "❌ Health check failed"

echo "8. Checking if circuits reset to expected state..."
curl -s http://localhost:8000/health/circuit-breaker | jq '.circuit_breakers'

echo "✅ Service restart complete"
```

## Performance Tuning

### Circuit Breaker Performance Profile

```python
import cProfile
import pstats
from io import StringIO

def profile_circuit_breaker():
    """Profile circuit breaker performance."""
    
    pr = cProfile.Profile()
    pr.enable()
    
    # Run circuit breaker operations
    client = ResilientHTTPClient("https://httpbin.org")
    
    async def performance_test():
        for _ in range(1000):
            try:
                await client.get("/status/200")
            except:
                pass
    
    asyncio.run(performance_test())
    
    pr.disable()
    
    # Print results
    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats()
    
    print(s.getvalue())
```

### Optimization Techniques

```python
# Optimized circuit breaker with caching
class OptimizedCircuitBreaker(CircuitBreaker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cached_can_execute = None
        self._cache_expiry = 0
        self._cache_duration = 1.0  # Cache for 1 second
    
    def can_execute(self) -> bool:
        """Optimized can_execute with caching."""
        now = time.time()
        
        # Return cached result if still valid
        if self._cached_can_execute is not None and now < self._cache_expiry:
            return self._cached_can_execute
        
        # Calculate fresh result
        result = super().can_execute()
        
        # Cache the result
        self._cached_can_execute = result
        self._cache_expiry = now + self._cache_duration
        
        return result
```

---

This troubleshooting guide provides systematic approaches to diagnose and resolve circuit breaker issues, from basic connectivity problems to complex performance optimizations. The combination of diagnostic tools, common issue patterns, and emergency procedures ensures reliable operation of the circuit breaker system.