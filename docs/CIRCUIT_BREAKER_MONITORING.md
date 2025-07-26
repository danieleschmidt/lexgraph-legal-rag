# Circuit Breaker Monitoring and Alerting

## Overview
This document defines monitoring, metrics, and alerting for circuit breaker patterns in the LexGraph Legal RAG system.

## Metrics Collection

### Prometheus Metrics

```python
# Example metrics collection for circuit breaker
from prometheus_client import Counter, Histogram, Gauge, Info

# Circuit breaker state gauge
CIRCUIT_BREAKER_STATE = Gauge(
    'circuit_breaker_state',
    'Current state of circuit breaker (0=CLOSED, 1=HALF_OPEN, 2=OPEN)',
    ['service', 'endpoint']
)

# Circuit breaker transitions counter
CIRCUIT_BREAKER_TRANSITIONS = Counter(
    'circuit_breaker_transitions_total',
    'Total number of circuit breaker state transitions',
    ['service', 'endpoint', 'from_state', 'to_state']
)

# Request outcome counter
CIRCUIT_BREAKER_REQUESTS = Counter(
    'circuit_breaker_requests_total',
    'Total requests through circuit breaker',
    ['service', 'endpoint', 'outcome']  # outcome: success, failure, blocked
)

# Recovery time histogram
CIRCUIT_BREAKER_RECOVERY_TIME = Histogram(
    'circuit_breaker_recovery_duration_seconds',
    'Time taken for circuit breaker to recover from OPEN to CLOSED',
    ['service', 'endpoint']
)

# Failure count gauge
CIRCUIT_BREAKER_FAILURES = Gauge(
    'circuit_breaker_failures_current',
    'Current consecutive failure count',
    ['service', 'endpoint']
)
```

### Integration with ResilientHTTPClient

```python
class MonitoringResilientHTTPClient(ResilientHTTPClient):
    """Enhanced HTTP client with comprehensive monitoring."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.service_name = kwargs.get('service_name', 'unknown')
        self._last_state = self.circuit_breaker._state
        
    def _record_state_transition(self, from_state: CircuitState, to_state: CircuitState):
        """Record circuit breaker state transitions."""
        CIRCUIT_BREAKER_TRANSITIONS.labels(
            service=self.service_name,
            endpoint=self.base_url,
            from_state=from_state.value,
            to_state=to_state.value
        ).inc()
        
        # Update current state gauge
        state_value = {'closed': 0, 'half_open': 1, 'open': 2}[to_state.value]
        CIRCUIT_BREAKER_STATE.labels(
            service=self.service_name,
            endpoint=self.base_url
        ).set(state_value)
    
    async def request(self, method: str, url: str, **kwargs) -> Response:
        """Enhanced request method with monitoring."""
        # Check for state changes
        current_state = self.circuit_breaker._state
        if current_state != self._last_state:
            self._record_state_transition(self._last_state, current_state)
            self._last_state = current_state
        
        # Record current failure count
        CIRCUIT_BREAKER_FAILURES.labels(
            service=self.service_name,
            endpoint=self.base_url
        ).set(self.circuit_breaker._failure_count)
        
        start_time = time.time()
        
        try:
            response = await super().request(method, url, **kwargs)
            
            # Record successful request
            CIRCUIT_BREAKER_REQUESTS.labels(
                service=self.service_name,
                endpoint=self.base_url,
                outcome='success'
            ).inc()
            
            return response
            
        except HTTPStatusError as e:
            if e.response.status_code == 503:  # Circuit breaker blocked
                CIRCUIT_BREAKER_REQUESTS.labels(
                    service=self.service_name,
                    endpoint=self.base_url,
                    outcome='blocked'
                ).inc()
            else:
                CIRCUIT_BREAKER_REQUESTS.labels(
                    service=self.service_name,
                    endpoint=self.base_url,
                    outcome='failure'
                ).inc()
            
            raise
        
        finally:
            # Record recovery time if circuit just closed
            if (self._last_state == CircuitState.HALF_OPEN and 
                self.circuit_breaker._state == CircuitState.CLOSED):
                recovery_time = time.time() - start_time
                CIRCUIT_BREAKER_RECOVERY_TIME.labels(
                    service=self.service_name,
                    endpoint=self.base_url
                ).observe(recovery_time)
```

### Custom Metrics Endpoint

```python
from fastapi import FastAPI
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

app = FastAPI()

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

@app.get("/health/circuit-breakers")
async def circuit_breaker_health():
    """Detailed circuit breaker health endpoint."""
    clients = get_all_circuit_breaker_clients()
    
    health_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "circuit_breakers": {}
    }
    
    for service_name, client in clients.items():
        status = client.get_circuit_status()
        health_data["circuit_breakers"][service_name] = {
            "state": status["state"],
            "failure_count": status["failure_count"],
            "success_count": status["success_count"],
            "last_failure_time": status.get("last_failure_time"),
            "healthy": status["state"] in ["closed", "half_open"],
            "configuration": {
                "failure_threshold": client.circuit_breaker.config.failure_threshold,
                "recovery_timeout": client.circuit_breaker.config.recovery_timeout,
                "success_threshold": client.circuit_breaker.config.success_threshold
            }
        }
    
    return health_data
```

## Alerting Rules

### Prometheus Alerting Rules

```yaml
# circuit_breaker_alerts.yml
groups:
  - name: circuit_breaker
    rules:
      - alert: CircuitBreakerOpen
        expr: circuit_breaker_state == 2
        for: 1m
        labels:
          severity: warning
          component: circuit_breaker
        annotations:
          summary: "Circuit breaker is open for {{ $labels.service }}"
          description: "Circuit breaker for {{ $labels.service }} ({{ $labels.endpoint }}) has been open for more than 1 minute"
          runbook_url: "https://docs.company.com/runbooks/circuit-breaker"
          
      - alert: CircuitBreakerFlapping
        expr: increase(circuit_breaker_transitions_total[15m]) > 6
        for: 0m
        labels:
          severity: critical
          component: circuit_breaker
        annotations:
          summary: "Circuit breaker flapping for {{ $labels.service }}"
          description: "Circuit breaker for {{ $labels.service }} has changed state {{ $value }} times in the last 15 minutes"
          
      - alert: CircuitBreakerHighFailureRate
        expr: |
          (
            rate(circuit_breaker_requests_total{outcome="failure"}[5m]) /
            rate(circuit_breaker_requests_total[5m])
          ) > 0.1
        for: 2m
        labels:
          severity: warning
          component: circuit_breaker
        annotations:
          summary: "High failure rate for {{ $labels.service }}"
          description: "Failure rate for {{ $labels.service }} is {{ $value | humanizePercentage }} over the last 5 minutes"
          
      - alert: CircuitBreakerStuckOpen
        expr: circuit_breaker_state == 2 and time() - circuit_breaker_last_failure_time > 900
        for: 0m
        labels:
          severity: critical
          component: circuit_breaker
        annotations:
          summary: "Circuit breaker stuck open for {{ $labels.service }}"
          description: "Circuit breaker for {{ $labels.service }} has been open for more than 15 minutes without recovery attempts"
          
      - alert: CircuitBreakerSlowRecovery
        expr: circuit_breaker_recovery_duration_seconds > 300
        for: 0m
        labels:
          severity: warning
          component: circuit_breaker
        annotations:
          summary: "Slow circuit breaker recovery for {{ $labels.service }}"
          description: "Circuit breaker for {{ $labels.service }} took {{ $value }}s to recover"
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "Circuit Breaker Monitoring",
    "panels": [
      {
        "title": "Circuit Breaker States",
        "type": "stat",
        "targets": [
          {
            "expr": "circuit_breaker_state",
            "legendFormat": "{{ service }} - {{ endpoint }}"
          }
        ],
        "fieldOverrides": [
          {
            "matcher": {"id": "byValue", "options": {"value": 0}},
            "properties": [{"id": "color", "value": {"mode": "fixed", "fixedColor": "green"}}]
          },
          {
            "matcher": {"id": "byValue", "options": {"value": 1}},
            "properties": [{"id": "color", "value": {"mode": "fixed", "fixedColor": "yellow"}}]
          },
          {
            "matcher": {"id": "byValue", "options": {"value": 2}},
            "properties": [{"id": "color", "value": {"mode": "fixed", "fixedColor": "red"}}]
          }
        ]
      },
      {
        "title": "Request Success Rate",
        "type": "timeseries",
        "targets": [
          {
            "expr": "rate(circuit_breaker_requests_total{outcome=\"success\"}[1m]) / rate(circuit_breaker_requests_total[1m])",
            "legendFormat": "{{ service }} Success Rate"
          }
        ]
      },
      {
        "title": "Circuit Breaker Transitions",
        "type": "timeseries",
        "targets": [
          {
            "expr": "increase(circuit_breaker_transitions_total[1m])",
            "legendFormat": "{{ service }} {{ from_state }} -> {{ to_state }}"
          }
        ]
      },
      {
        "title": "Current Failure Counts",
        "type": "bargauge",
        "targets": [
          {
            "expr": "circuit_breaker_failures_current",
            "legendFormat": "{{ service }}"
          }
        ]
      },
      {
        "title": "Recovery Time Distribution",
        "type": "histogram",
        "targets": [
          {
            "expr": "circuit_breaker_recovery_duration_seconds_bucket",
            "legendFormat": "{{ service }}"
          }
        ]
      }
    ]
  }
}
```

### Notification Channels

```python
# Slack notifications for circuit breaker events
import requests

class CircuitBreakerNotifier:
    def __init__(self, slack_webhook_url: str):
        self.slack_webhook = slack_webhook_url
    
    def notify_circuit_opened(self, service: str, endpoint: str, failure_count: int):
        """Send notification when circuit breaker opens."""
        message = {
            "text": f"🔴 Circuit Breaker Alert",
            "attachments": [
                {
                    "color": "danger",
                    "fields": [
                        {"title": "Service", "value": service, "short": True},
                        {"title": "Endpoint", "value": endpoint, "short": True},
                        {"title": "Failure Count", "value": str(failure_count), "short": True},
                        {"title": "Status", "value": "OPEN - All requests blocked", "short": True}
                    ],
                    "actions": [
                        {
                            "type": "button",
                            "text": "View Dashboard",
                            "url": f"https://grafana.company.com/d/circuit-breaker?var-service={service}"
                        },
                        {
                            "type": "button",
                            "text": "Runbook",
                            "url": "https://docs.company.com/runbooks/circuit-breaker"
                        }
                    ]
                }
            ]
        }
        
        requests.post(self.slack_webhook, json=message)
    
    def notify_circuit_recovered(self, service: str, endpoint: str, recovery_time: float):
        """Send notification when circuit breaker recovers."""
        message = {
            "text": f"✅ Circuit Breaker Recovered",
            "attachments": [
                {
                    "color": "good",
                    "fields": [
                        {"title": "Service", "value": service, "short": True},
                        {"title": "Endpoint", "value": endpoint, "short": True},
                        {"title": "Recovery Time", "value": f"{recovery_time:.1f}s", "short": True},
                        {"title": "Status", "value": "CLOSED - Normal operation", "short": True}
                    ]
                }
            ]
        }
        
        requests.post(self.slack_webhook, json=message)
```

## Log Analysis

### Log Structured Data

```python
import structlog
from lexgraph_legal_rag.http_client import CircuitState

logger = structlog.get_logger(__name__)

def log_circuit_breaker_event(event_type: str, service: str, **kwargs):
    """Structured logging for circuit breaker events."""
    log_data = {
        "event_type": event_type,
        "service": service,
        "component": "circuit_breaker",
        **kwargs
    }
    
    if event_type == "circuit_opened":
        logger.warning("Circuit breaker opened", **log_data)
    elif event_type == "circuit_closed":
        logger.info("Circuit breaker closed", **log_data)
    elif event_type == "circuit_half_open":
        logger.info("Circuit breaker testing recovery", **log_data)
    else:
        logger.info("Circuit breaker event", **log_data)

# Usage in ResilientHTTPClient
class LoggingResilientHTTPClient(ResilientHTTPClient):
    def _log_state_change(self, old_state: CircuitState, new_state: CircuitState):
        """Log circuit breaker state changes."""
        event_map = {
            (CircuitState.CLOSED, CircuitState.OPEN): "circuit_opened",
            (CircuitState.HALF_OPEN, CircuitState.CLOSED): "circuit_closed", 
            (CircuitState.OPEN, CircuitState.HALF_OPEN): "circuit_half_open",
            (CircuitState.HALF_OPEN, CircuitState.OPEN): "circuit_reopened"
        }
        
        event_type = event_map.get((old_state, new_state), "circuit_state_change")
        
        log_circuit_breaker_event(
            event_type=event_type,
            service=self.service_name,
            endpoint=self.base_url,
            old_state=old_state.value,
            new_state=new_state.value,
            failure_count=self.circuit_breaker._failure_count,
            success_count=self.circuit_breaker._success_count
        )
```

### Log Queries for Analysis

```bash
# ElasticSearch/Kibana queries for circuit breaker analysis

# Find all circuit breaker openings in last 24 hours
{
  "query": {
    "bool": {
      "must": [
        {"term": {"event_type": "circuit_opened"}},
        {"range": {"@timestamp": {"gte": "now-24h"}}}
      ]
    }
  }
}

# Analyze failure patterns before circuit opening
{
  "query": {
    "bool": {
      "must": [
        {"term": {"component": "circuit_breaker"}},
        {"range": {"@timestamp": {"gte": "now-1h"}}}
      ]
    }
  },
  "aggs": {
    "by_service": {
      "terms": {"field": "service"},
      "aggs": {
        "events_over_time": {
          "date_histogram": {
            "field": "@timestamp",
            "interval": "1m"
          }
        }
      }
    }
  }
}
```

## Performance Impact Monitoring

### Resource Usage Metrics

```python
# Monitor performance impact of circuit breaker
import psutil
import time

class CircuitBreakerPerformanceMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.request_count = 0
        self.blocked_count = 0
        
    def record_request(self, blocked: bool = False):
        """Record request for performance analysis."""
        self.request_count += 1
        if blocked:
            self.blocked_count += 1
            
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics."""
        elapsed = time.time() - self.start_time
        return {
            "requests_per_second": self.request_count / elapsed if elapsed > 0 else 0,
            "block_rate": self.blocked_count / self.request_count if self.request_count > 0 else 0,
            "cpu_usage_percent": psutil.cpu_percent(),
            "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024
        }
```

### Load Testing Integration

```python
# Load testing circuit breaker behavior
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor

async def load_test_circuit_breaker(base_url: str, concurrent_requests: int = 100):
    """Load test circuit breaker under various failure scenarios."""
    
    async with aiohttp.ClientSession() as session:
        # Normal operation test
        tasks = []
        for i in range(concurrent_requests):
            task = session.get(f"{base_url}/api/health")
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        success_count = sum(1 for r in responses if not isinstance(r, Exception))
        failure_count = len(responses) - success_count
        
        return {
            "total_requests": len(responses),
            "success_count": success_count,
            "failure_count": failure_count,
            "success_rate": success_count / len(responses)
        }
        
# Usage in CI/CD pipeline
def test_circuit_breaker_under_load():
    """Test circuit breaker behavior under load."""
    results = asyncio.run(load_test_circuit_breaker("http://localhost:8000"))
    
    assert results["success_rate"] > 0.95, f"Success rate too low: {results['success_rate']}"
    print(f"Load test passed: {results['success_count']}/{results['total_requests']} succeeded")
```

---

This comprehensive monitoring setup provides:
- Real-time metrics collection
- Proactive alerting for circuit breaker events  
- Detailed dashboard visualization
- Structured logging for analysis
- Performance impact monitoring
- Load testing capabilities

The monitoring system enables operators to maintain high service reliability while providing visibility into circuit breaker behavior and performance.