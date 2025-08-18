# Production Deployment Guide - Bioneural Olfactory Fusion System

## Overview

This guide covers the complete production deployment of the Bioneural Olfactory Fusion System, including Docker containerization, Kubernetes orchestration, monitoring, and CI/CD.

## Prerequisites

- Kubernetes cluster (v1.24+)
- Docker registry access
- kubectl configured
- Helm (optional)
- SSL certificates for HTTPS

## Quick Start

### 1. Build and Push Docker Image

```bash
# Build production image
docker build -f Dockerfile.production -t bioneural-olfactory-fusion:latest .

# Tag and push to registry
docker tag bioneural-olfactory-fusion:latest your-registry/bioneural-olfactory-fusion:latest
docker push your-registry/bioneural-olfactory-fusion:latest
```

### 2. Deploy to Kubernetes

```bash
# Deploy to production
./deployment/scripts/deploy_production.sh

# Verify deployment
kubectl get pods -n bioneural-olfactory-fusion-production
```

### 3. Monitor Deployment

```bash
# Check logs
kubectl logs -f deployment/bioneural-olfactory-fusion-api -n bioneural-olfactory-fusion-production

# Check metrics
kubectl port-forward svc/prometheus 9090:9090
# Open http://localhost:9090
```

## Architecture

### Components

1. **API Service**: FastAPI application serving bioneural analysis
2. **Load Balancer**: NGINX Ingress for traffic routing
3. **Monitoring**: Prometheus + Grafana for observability
4. **Auto-scaling**: HPA for dynamic scaling based on load
5. **Security**: Network policies, RBAC, PSP

### Scaling Strategy

- **Horizontal**: 3-20 replicas based on CPU/memory usage
- **Vertical**: Resource limits adjustable per environment
- **Auto-scaling**: Prometheus metrics-based scaling

## Configuration

### Environment Variables

- `ENVIRONMENT`: deployment environment (production/staging/development)
- `LOG_LEVEL`: logging level (INFO/DEBUG/WARNING/ERROR)
- `CACHE_SIZE`: number of cached analysis results
- `WORKER_THREADS`: number of processing threads

### Secrets Management

All secrets are stored in Kubernetes secrets and should be managed via:
- External secret management (e.g., HashiCorp Vault)
- Cloud provider secret managers (AWS Secrets Manager, Azure Key Vault)
- Sealed Secrets for GitOps workflows

## Monitoring and Alerting

### Metrics

- Request rate and response times
- Error rates and status codes
- Resource utilization (CPU, memory)
- Custom business metrics (analysis accuracy, cache hit rates)

### Alerts

- High error rate (>10% for 5 minutes)
- High response time (>1 second 95th percentile)
- Pod crash looping
- Resource exhaustion

### Dashboards

Access Grafana dashboards at: https://grafana.bioneural-olfactory-fusion.example.com

## Security

### Network Security

- Network policies restrict pod-to-pod communication
- Ingress controller handles SSL termination
- Private container registry with image scanning

### Pod Security

- Non-root user execution
- Read-only root filesystem
- Capability dropping
- Resource limits enforcement

### Secrets Security

- Secrets stored in Kubernetes secrets
- Encryption at rest enabled
- RBAC for secret access
- Regular secret rotation

## Backup and Disaster Recovery

### Application State

- Stateless application design
- Configuration in ConfigMaps/Secrets
- Persistent data in external systems

### Recovery Procedures

1. **Service Disruption**: Auto-scaling and health checks
2. **Node Failure**: Kubernetes reschedules pods
3. **Cluster Failure**: Multi-region deployment strategy
4. **Data Loss**: Backup external dependencies

## Troubleshooting

### Common Issues

1. **Pod Not Starting**
   ```bash
   kubectl describe pod <pod-name> -n bioneural-olfactory-fusion-production
   kubectl logs <pod-name> -n bioneural-olfactory-fusion-production
   ```

2. **High Memory Usage**
   - Check resource limits
   - Review cache settings
   - Monitor for memory leaks

3. **Performance Issues**
   - Check auto-scaling configuration
   - Review resource requests/limits
   - Analyze application metrics

### Rollback Procedure

```bash
# Immediate rollback
./deployment/scripts/rollback_production.sh

# Or manual rollback
kubectl rollout undo deployment/bioneural-olfactory-fusion-api -n bioneural-olfactory-fusion-production
```

## Maintenance

### Regular Tasks

- Monitor resource usage and adjust limits
- Review and update security policies
- Update container images for security patches
- Backup configuration and secrets
- Performance testing and capacity planning

### Upgrade Process

1. Test in staging environment
2. Deploy during maintenance window
3. Monitor for issues
4. Rollback if necessary
5. Update documentation

## Support

For deployment issues, check:
1. Application logs: `kubectl logs`
2. Kubernetes events: `kubectl get events`
3. Monitoring dashboards
4. Application health endpoints

## Contact

- DevOps Team: devops@terragon.dev
- On-call: +1-555-DEVOPS
- Slack: #bioneural-support
