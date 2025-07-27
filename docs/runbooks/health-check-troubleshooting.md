# Health Check Troubleshooting

## Overview
This runbook covers troubleshooting health check failures in the LexGraph Legal RAG system.

## Health Check Endpoints

### `/health` - Basic Health Check
Returns 200 OK if the application is running.

**Expected Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "version": "1.0.0"
}
```

### `/ready` - Readiness Check
Returns 200 OK if the application is ready to serve requests.

**Expected Response:**
```json
{
  "status": "ready",
  "dependencies": {
    "database": "healthy",
    "vector_index": "healthy",
    "external_apis": "healthy"
  }
}
```

## Common Issues

### 1. Health Check Returns 503 Service Unavailable

**Symptoms:**
- HTTP 503 response from `/health`
- Application container is running but not responding

**Diagnosis:**
```bash
# Check container status
docker ps | grep lexgraph

# Check container logs
docker logs lexgraph-api --tail 50

# Check if port is accessible
curl -I http://localhost:8000/health
```

**Resolution:**
1. Check application logs for errors
2. Verify environment variables are set correctly
3. Ensure required dependencies are available
4. Restart the container if necessary

### 2. Readiness Check Fails

**Symptoms:**
- `/health` returns 200 but `/ready` returns 503
- Some dependencies are marked as unhealthy

**Diagnosis:**
```bash
# Check specific dependency status
curl http://localhost:8000/ready | jq '.dependencies'

# Check vector index
ls -la ./data/indices/

# Test database connection
# (commands depend on database type)
```

**Resolution:**
1. **Database Issues:**
   - Check database connectivity
   - Verify credentials
   - Check disk space

2. **Vector Index Issues:**
   - Verify index files exist
   - Check file permissions
   - Regenerate index if corrupted

3. **External API Issues:**
   - Check API keys
   - Verify network connectivity
   - Check rate limits

### 3. Long Response Times

**Symptoms:**
- Health checks timeout
- High response latency

**Diagnosis:**
```bash
# Check response time
time curl http://localhost:8000/health

# Check system resources
docker stats lexgraph-api

# Check for I/O wait
top -p $(docker inspect -f '{{.State.Pid}}' lexgraph-api)
```

**Resolution:**
1. Check CPU and memory usage
2. Verify disk I/O is not saturated
3. Check for deadlocks in application logs
4. Scale horizontally if needed

## Monitoring Alerts

### Critical Alerts
- Health check failure for > 5 minutes
- Readiness check failure for > 2 minutes
- Response time > 30 seconds

### Warning Alerts
- Health check failure for > 2 minutes
- Response time > 10 seconds
- High error rate in dependencies

## Prevention

1. **Monitoring**: Set up comprehensive monitoring and alerting
2. **Testing**: Include health checks in CI/CD pipelines
3. **Documentation**: Keep dependency requirements up to date
4. **Automation**: Use orchestration tools for automatic recovery

## Escalation

If health check issues persist after following this runbook:

1. **Check related services** (database, external APIs)
2. **Review recent deployments** for changes
3. **Contact on-call engineer** if critical
4. **Create incident ticket** for tracking