# Deployment Guide

This document provides comprehensive guidance for deploying the LexGraph Legal RAG system across different environments, from local development to production Kubernetes deployments.

## Deployment Overview

The LexGraph Legal RAG system supports multiple deployment patterns:
- **Local Development**: Docker Compose for rapid development
- **Staging Environment**: Docker Compose with production-like configuration
- **Production**: Kubernetes with high availability and scalability
- **Cloud Providers**: AWS EKS, Google GKE, Azure AKS

## Quick Start Deployment

### Prerequisites

```bash
# Required tools
docker --version          # Docker 20.10+
docker-compose --version  # Docker Compose 2.0+
kubectl version          # Kubernetes CLI (for K8s deployments)
```

### Local Development

```bash
# Clone repository
git clone https://github.com/terragon-labs/lexgraph-legal-rag.git
cd lexgraph-legal-rag

# Setup environment
cp .env.example .env
# Edit .env with your API keys

# Start development environment
make docker-up

# Verify deployment
curl http://localhost:8000/health
```

### Production Deployment

```bash
# Build production image
make docker-build

# Deploy to production
docker-compose -f docker-compose.production.yml up -d

# Verify all services
docker-compose ps
```

## Container Architecture

### Multi-Stage Docker Build

The Dockerfile implements a multi-stage build pattern:

#### Base Stage
- Python 3.11 slim base image
- System dependencies and security updates
- Non-root application user
- Common environment variables

#### Development Stage
- Development dependencies (pytest, black, ruff)
- Hot reload capability
- Debug tooling
- Volume mounts for live code editing

#### Production Stage
- Minimal dependencies
- Security hardening
- Health checks
- Optimized for performance

### Image Security

#### Security Features
- Non-root user execution
- Minimal attack surface
- Security updates included
- No secrets in image layers
- Health check endpoints

#### Security Scanning
```bash
# Scan image for vulnerabilities
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image lexgraph-legal-rag:latest

# Generate security report
make security
```

## Docker Compose Configurations

### Development Environment

```yaml
# docker-compose.yml - Development configuration
services:
  lexgraph-dev:
    build:
      target: development
    volumes:
      - .:/app          # Live code reloading
      - ./data:/app/data # Persistent data
    environment:
      - ENVIRONMENT=development
      - DEBUG=true
    profiles:
      - development
```

### Production Environment

```yaml
# docker-compose.production.yml
services:
  lexgraph-api:
    build:
      target: production
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    environment:
      - ENVIRONMENT=production
      - DEBUG=false
```

### Service Dependencies

```yaml
services:
  lexgraph-api:
    depends_on:
      redis:
        condition: service_healthy
      prometheus:
        condition: service_started
```

## Kubernetes Deployment

### Prerequisites

```bash
# Create namespace
kubectl create namespace lexgraph-legal-rag

# Apply configurations
kubectl apply -f k8s/
```

### Core Components

#### Deployment Configuration

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: lexgraph-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: lexgraph-api
  template:
    metadata:
      labels:
        app: lexgraph-api
    spec:
      containers:
      - name: lexgraph-api
        image: lexgraph-legal-rag:latest
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:  
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

#### Service Configuration

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: lexgraph-api-service
spec:
  selector:
    app: lexgraph-api
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
```

#### ConfigMap for Environment Variables

```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: lexgraph-config
data:
  ENVIRONMENT: "production"
  METRICS_PORT: "8001"
  LOG_LEVEL: "INFO"
```

#### Secrets Management

```yaml
# k8s/secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: lexgraph-secrets
type: Opaque
data:
  api-key: <base64-encoded-api-key>
  openai-api-key: <base64-encoded-openai-key>
```

### Horizontal Pod Autoscaler

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: lexgraph-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: lexgraph-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Ingress Configuration

```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: lexgraph-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - api.lexgraph.example.com
    secretName: lexgraph-tls
  rules:
  - host: api.lexgraph.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: lexgraph-api-service
            port:
              number: 80
```

## Cloud Provider Deployments

### AWS EKS Deployment

#### Prerequisites
```bash
# Install AWS CLI and eksctl
aws configure
eksctl create cluster --name lexgraph-cluster --region us-west-2
```

#### EKS-Specific Configuration
```bash
# Update kubeconfig
aws eks update-kubeconfig --region us-west-2 --name lexgraph-cluster

# Apply K8s manifests
kubectl apply -f k8s/
```

### Google GKE Deployment

#### Prerequisites
```bash
# Install gcloud CLI
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Create GKE cluster
gcloud container clusters create lexgraph-cluster \
  --zone us-central1-a \
  --num-nodes 3
```

### Azure AKS Deployment

#### Prerequisites
```bash
# Install Azure CLI
az login
az aks create --resource-group myResourceGroup \
  --name lexgraph-cluster \
  --node-count 3 \
  --enable-addons monitoring
```

## Environment Configuration

### Environment Variables

#### Required Variables
```bash
# API Configuration
API_KEY=secure-production-key-here
ENVIRONMENT=production

# AI Service Keys
OPENAI_API_KEY=sk-your-openai-api-key
PINECONE_API_KEY=your-pinecone-key

# Application Settings
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO
```

#### Optional Variables
```bash
# Performance Tuning
MAX_CONCURRENT_REQUESTS=100
REQUEST_TIMEOUT=30
QUERY_CACHE_TTL=300

# Security Settings
CORS_ALLOWED_ORIGINS=https://your-domain.com
REQUIRE_HTTPS=true
RATE_LIMIT_REQUESTS_PER_MINUTE=60
```

### Configuration Management

#### Docker Compose Environment Files
```bash
# Development
.env.development

# Staging
.env.staging  

# Production
.env.production
```

#### Kubernetes ConfigMaps and Secrets
```bash
# Create from env file
kubectl create configmap lexgraph-config --from-env-file=.env.production

# Create secrets
kubectl create secret generic lexgraph-secrets \
  --from-literal=api-key=your-secret-key \
  --from-literal=openai-api-key=your-openai-key
```

## Monitoring and Observability

### Prometheus Integration

```yaml
# k8s/servicemonitor.yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: lexgraph-metrics
spec:
  selector:
    matchLabels:
      app: lexgraph-api
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics
```

### Grafana Dashboards

Pre-configured dashboards available in `monitoring/grafana/dashboards/`:
- Application Performance Dashboard
- Infrastructure Metrics Dashboard
- Business Metrics Dashboard
- Error Rate and Latency Dashboard

### Log Aggregation

```yaml
# Fluent Bit configuration for log forwarding
apiVersion: v1
kind: ConfigMap
metadata:
  name: fluent-bit-config
data:
  fluent-bit.conf: |
    [SERVICE]
        Flush         1
        Log_Level     info
        Daemon        off
        Parsers_File  parsers.conf

    [INPUT]
        Name              tail
        Path              /var/log/containers/lexgraph-api*.log
        Parser            docker
        Tag               kube.*
        Refresh_Interval  5
```

## SSL/TLS Configuration

### Certificate Management

#### Let's Encrypt with cert-manager
```yaml
# Install cert-manager
kubectl apply -f https://github.com/jetstack/cert-manager/releases/download/v1.8.0/cert-manager.yaml

# Configure cluster issuer
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@example.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
```

### Security Headers

```yaml
# Ingress annotations for security
annotations:
  nginx.ingress.kubernetes.io/configuration-snippet: |
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options DENY;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";
```

## Scaling and Performance

### Horizontal Scaling

#### Application Scaling
```bash
# Scale deployment
kubectl scale deployment lexgraph-api --replicas=5

# Auto-scaling based on metrics
kubectl autoscale deployment lexgraph-api --cpu-percent=70 --min=3 --max=10
```

#### Database Scaling
```bash
# Redis cluster for caching
# Configure Redis Cluster in docker-compose.yml or K8s
```

### Vertical Scaling

#### Resource Limits
```yaml
resources:
  requests:
    memory: "1Gi"
    cpu: "500m"
  limits:
    memory: "2Gi"
    cpu: "1000m"
```

### Performance Optimization

#### Container Optimization
- Multi-stage builds to reduce image size
- Layer caching for faster builds
- Resource limits to prevent resource exhaustion
- Health checks for proper load balancing

#### Application Optimization
- Connection pooling for external APIs
- Caching for frequently accessed data
- Async operations for I/O-bound tasks
- Proper logging levels for production

## Backup and Disaster Recovery

### Data Backup Strategy

#### Vector Index Backup
```bash
# Automated backup script
#!/bin/bash
BACKUP_DIR="/backups/$(date +%Y%m%d-%H%M%S)"
mkdir -p $BACKUP_DIR

# Backup indices
cp -r /app/data/indices $BACKUP_DIR/
tar -czf $BACKUP_DIR/indices.tar.gz -C $BACKUP_DIR indices/

# Upload to cloud storage (AWS S3 example)
aws s3 sync $BACKUP_DIR s3://lexgraph-backups/
```

#### Database Backup (if applicable)
```bash
# Redis backup
redis-cli --rdb backup.rdb
```

### Disaster Recovery Procedures

#### Recovery Steps
1. **Identify Failure Scope**: Application, database, or infrastructure
2. **Restore from Backup**: Latest verified backup
3. **Verify Data Integrity**: Run health checks and validation
4. **Update DNS**: Point traffic to recovered instance
5. **Monitor**: Ensure stable operation

#### Recovery Time Objectives (RTO)
- **Application Recovery**: < 15 minutes
- **Data Recovery**: < 30 minutes  
- **Full System Recovery**: < 1 hour

## Troubleshooting

### Common Issues

#### Application Won't Start
```bash
# Check logs
docker logs lexgraph-api
kubectl logs deployment/lexgraph-api

# Check configuration
kubectl describe pod lexgraph-api-xxx
```

#### High Memory Usage
```bash
# Check resource usage
kubectl top pods
docker stats

# Increase memory limits
# Update deployment configuration
```

#### Network Connectivity Issues
```bash
# Test service connectivity
kubectl exec -it pod-name -- curl http://service-name:port/health

# Check ingress configuration
kubectl describe ingress lexgraph-ingress
```

### Debug Mode

#### Enable Debug Logging
```bash
# Set environment variable
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG
```

#### Access Application Shell
```bash
# Docker
docker exec -it lexgraph-api bash

# Kubernetes
kubectl exec -it deployment/lexgraph-api -- bash
```

## Security Considerations

### Production Security Checklist

- [ ] Use non-root container user
- [ ] Enable resource limits
- [ ] Configure network policies
- [ ] Use secrets for sensitive data
- [ ] Enable SSL/TLS encryption
- [ ] Regular security updates
- [ ] Vulnerability scanning
- [ ] Access logging enabled
- [ ] Rate limiting configured
- [ ] CORS properly configured

### Network Security

```yaml
# Network policy example
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: lexgraph-network-policy
spec:
  podSelector:
    matchLabels:
      app: lexgraph-api
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: nginx-ingress
  egress:
  - to: []
    ports:
    - protocol: TCP
      port: 443  # HTTPS only
```

This deployment guide provides comprehensive coverage of deploying LexGraph Legal RAG across different environments with proper security, monitoring, and scalability considerations.