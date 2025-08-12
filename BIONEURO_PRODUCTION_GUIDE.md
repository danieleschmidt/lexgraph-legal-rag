# Bioneural Olfactory Fusion System - Production Deployment Guide

## 🧬 Overview

This guide covers the production deployment of the **Bioneural Olfactory Fusion System**, a novel multi-sensory legal document analysis platform that uses bio-inspired olfactory computing for enhanced legal AI capabilities.

## 🚀 Quick Start

### Docker Deployment (Recommended for Development/Testing)

```bash
cd deployment/bioneuro-production
./deploy.sh docker production
```

### Kubernetes Deployment (Recommended for Production)

```bash
cd deployment/bioneuro-production
./deploy.sh kubernetes production
```

## 📋 Prerequisites

### System Requirements

- **CPU**: 4+ cores
- **Memory**: 8GB+ RAM
- **Storage**: 50GB+ available space
- **Network**: HTTPS/TLS capable

### Software Dependencies

#### For Docker Deployment:
- Docker 20.10+
- Docker Compose 2.0+

#### For Kubernetes Deployment:
- Kubernetes 1.20+
- kubectl configured
- Helm 3.0+ (optional)

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `LOG_LEVEL` | Logging level | `INFO` | No |
| `WORKERS` | Number of worker processes | `4` | No |
| `MAX_CACHE_SIZE` | Maximum cache size | `10000` | No |
| `MEMORY_LIMIT_MB` | Memory limit in MB | `2048` | No |
| `ENABLE_MONITORING` | Enable Prometheus metrics | `true` | No |
| `API_KEY` | API authentication key | - | Yes |

### Optimization Configuration

The system supports advanced optimization settings in `config.json`:

```json
{
  "optimization": {
    "max_cache_size": 10000,
    "cache_strategy": "adaptive",
    "processing_mode": "adaptive",
    "max_workers": 4,
    "batch_size": 32,
    "memory_limit_mb": 2048,
    "enable_profiling": true,
    "auto_scaling": true
  },
  "monitoring": {
    "enable_prometheus": true,
    "metrics_port": 9090,
    "health_check_interval": 30,
    "alert_thresholds": {
      "memory_usage_mb": 1800,
      "cpu_usage_percent": 80,
      "error_rate": 0.05,
      "response_time_p95": 5.0
    }
  }
}
```

## 🐳 Docker Deployment

### Services Included

- **bioneuro-api**: Main API service
- **bioneuro-monitoring**: Prometheus monitoring
- **bioneuro-dashboard**: Grafana dashboard
- **redis**: Caching layer

### Deployment Steps

1. **Build and Deploy**:
   ```bash
   cd deployment/bioneuro-production
   ./deploy.sh docker production
   ```

2. **Access Services**:
   - API: http://localhost:8000
   - Monitoring: http://localhost:9090
   - Dashboard: http://localhost:3000 (admin/admin123)

3. **Health Check**:
   ```bash
   curl http://localhost:8000/health
   ```

### Docker Compose Override

For custom configurations, create `docker-compose.override.yml`:

```yaml
version: '3.8'
services:
  bioneuro-api:
    environment:
      - LOG_LEVEL=DEBUG
      - WORKERS=8
    deploy:
      resources:
        limits:
          memory: 4G
```

## ☸️ Kubernetes Deployment

### Features

- **Auto-scaling**: HPA based on CPU/memory
- **Health checks**: Liveness and readiness probes
- **Persistent storage**: For data and logs
- **Ingress**: HTTPS/TLS termination
- **Monitoring**: Prometheus integration

### Deployment Steps

1. **Deploy to Cluster**:
   ```bash
   cd deployment/bioneuro-production
   ./deploy.sh kubernetes production
   ```

2. **Verify Deployment**:
   ```bash
   kubectl get pods -n bioneuro-system
   kubectl get services -n bioneuro-system
   ```

3. **Access API**:
   ```bash
   kubectl port-forward service/bioneuro-api-service 8000:8000 -n bioneuro-system
   ```

### Scaling

The system includes Horizontal Pod Autoscaler (HPA):

```yaml
spec:
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## 📊 Monitoring & Observability

### Metrics Collected

- **Performance Metrics**:
  - Document processing time
  - Olfactory analysis time
  - Multi-sensory fusion time
  - Memory usage
  - CPU usage
  - Throughput (docs/sec)

- **Quality Metrics**:
  - Signal strength average
  - Signal confidence average
  - Receptor activation rate
  - Scent profile completeness
  - Fusion coherence score

- **System Metrics**:
  - Error rates
  - Cache hit/miss ratios
  - Queue sizes
  - Resource utilization

### Alerts

Pre-configured alerts for:
- High error rates (>10%)
- High memory usage (>1800MB)
- High response times (>5s)
- Low cache hit rates (<70%)
- Service availability
- Receptor anomalies

### Grafana Dashboards

Access Grafana at http://localhost:3000 (Docker) with admin/admin123:

- **System Overview**: High-level system metrics
- **Bioneural Analysis**: Olfactory fusion specific metrics
- **Performance**: Response times and throughput
- **Quality**: Analysis quality metrics

## 🔐 Security

### API Authentication

All API endpoints require authentication via `X-API-Key` header:

```bash
curl -H "X-API-Key: your-api-key" http://localhost:8000/ping
```

### Key Management

The system supports API key rotation:

```bash
curl -X POST \
  -H "X-API-Key: current-key" \
  -H "Content-Type: application/json" \
  -d '{"new_primary_key": "new-key"}' \
  http://localhost:8000/admin/rotate-keys
```

### Security Features

- Non-root container execution
- Resource limits and quotas
- Network policies (Kubernetes)
- Secret management
- TLS/HTTPS support
- CORS configuration

## 🔧 Maintenance

### Log Management

Logs are collected in structured JSON format:

```bash
# Docker
docker-compose logs bioneuro-api

# Kubernetes
kubectl logs -f deployment/bioneuro-api -n bioneuro-system
```

### Database/Cache Maintenance

Redis cache management:

```bash
# Clear cache
redis-cli FLUSHALL

# Monitor cache
redis-cli MONITOR
```

### Backup & Recovery

1. **Data Backup**:
   ```bash
   # Kubernetes
   kubectl exec -n bioneuro-system deployment/bioneuro-api -- \
     tar czf /tmp/backup.tar.gz /app/data
   ```

2. **Configuration Backup**:
   ```bash
   kubectl get configmap bioneuro-config -n bioneuro-system -o yaml > config-backup.yml
   ```

## 🧪 Testing

### Health Checks

```bash
# Basic health
curl http://localhost:8000/health

# Readiness check
curl http://localhost:8000/ready

# API version info
curl http://localhost:8000/version
```

### Load Testing

Use the included performance test:

```bash
python3 test_bioneuro_minimal.py
```

For production load testing:

```bash
# Install dependencies
pip install locust

# Run load test
locust -f load_test.py --host=http://localhost:8000
```

### Integration Testing

```bash
# Run comprehensive tests
python3 run_quality_gates.py

# Run specific bioneural tests
python3 test_bioneuro_minimal.py
```

## 🚨 Troubleshooting

### Common Issues

1. **High Memory Usage**:
   - Check cache size configuration
   - Monitor for memory leaks
   - Increase memory limits

2. **Slow Response Times**:
   - Check processing mode configuration
   - Monitor cache hit rates
   - Scale horizontally

3. **Service Startup Failures**:
   - Check logs for dependency issues
   - Verify environment variables
   - Check resource availability

### Debugging

Enable debug logging:

```bash
export LOG_LEVEL=DEBUG
```

Access container for debugging:

```bash
# Docker
docker exec -it bioneuro-api bash

# Kubernetes
kubectl exec -it deployment/bioneuro-api -n bioneuro-system -- bash
```

## 📈 Performance Tuning

### Optimization Strategies

1. **Cache Optimization**:
   - Increase cache size for better hit rates
   - Use adaptive cache strategy
   - Monitor cache performance

2. **Processing Mode**:
   - Use `concurrent` for medium loads
   - Use `parallel` for high loads
   - Enable `adaptive` for dynamic workloads

3. **Resource Allocation**:
   - Scale CPU based on processing complexity
   - Scale memory based on cache requirements
   - Use HPA for automatic scaling

### Benchmarking

The system includes built-in benchmarking:

```python
from lexgraph_legal_rag.bioneuro_optimization import create_performance_benchmark

# Compare different configurations
results = create_performance_benchmark(documents, configurations)
```

## 🔄 Updates & Upgrades

### Rolling Updates

Kubernetes supports zero-downtime updates:

```bash
kubectl set image deployment/bioneuro-api \
  bioneuro-api=bioneuro-olfactory-fusion:new-version \
  -n bioneuro-system
```

### Configuration Updates

Update configuration without restart:

```bash
kubectl apply -f kubernetes.yml
kubectl rollout restart deployment/bioneuro-api -n bioneuro-system
```

## 📞 Support

### Getting Help

1. **Documentation**: Check this guide and inline code documentation
2. **Logs**: Review application and system logs
3. **Monitoring**: Check Grafana dashboards and Prometheus metrics
4. **Testing**: Run diagnostic tests

### Reporting Issues

When reporting issues, include:
- System configuration
- Error logs
- Performance metrics
- Steps to reproduce

---

## 🎉 Success!

Your Bioneural Olfactory Fusion System is now running in production, delivering novel multi-sensory legal document analysis capabilities using bio-inspired computing!

For advanced configuration and customization, refer to the source code documentation and configuration files.