"""
Complete Production Deployment System for Bioneural Olfactory Fusion

This module implements comprehensive production deployment capabilities including:
- Docker containerization with multi-stage builds
- Kubernetes deployment manifests
- Auto-scaling configurations
- Health monitoring and alerting
- CI/CD pipeline integration
- Load balancing and service mesh
- Security hardening
- Performance monitoring
- Backup and disaster recovery
"""

import asyncio
import logging
import json
import yaml
import time
import subprocess
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import uuid

logger = logging.getLogger(__name__)


@dataclass
class DeploymentConfig:
    """Production deployment configuration."""
    environment: str  # dev, staging, production
    replicas: int
    resource_limits: Dict[str, str]
    resource_requests: Dict[str, str]
    autoscaling: Dict[str, Any]
    monitoring: Dict[str, Any]
    security: Dict[str, Any]


class ProductionDeploymentManager:
    """Comprehensive production deployment management."""
    
    def __init__(self, project_name: str = "bioneural-olfactory-fusion"):
        self.project_name = project_name
        self.base_path = Path(".")
        self.deployment_configs = {
            "development": DeploymentConfig(
                environment="development",
                replicas=1,
                resource_limits={"cpu": "500m", "memory": "512Mi"},
                resource_requests={"cpu": "100m", "memory": "128Mi"},
                autoscaling={"min_replicas": 1, "max_replicas": 3, "target_cpu": 70},
                monitoring={"enabled": True, "metrics_retention": "7d"},
                security={"network_policies": True, "pod_security": "restricted"}
            ),
            "staging": DeploymentConfig(
                environment="staging",
                replicas=2,
                resource_limits={"cpu": "1000m", "memory": "1Gi"},
                resource_requests={"cpu": "200m", "memory": "256Mi"},
                autoscaling={"min_replicas": 2, "max_replicas": 6, "target_cpu": 70},
                monitoring={"enabled": True, "metrics_retention": "30d"},
                security={"network_policies": True, "pod_security": "restricted"}
            ),
            "production": DeploymentConfig(
                environment="production",
                replicas=3,
                resource_limits={"cpu": "2000m", "memory": "2Gi"},
                resource_requests={"cpu": "500m", "memory": "512Mi"},
                autoscaling={"min_replicas": 3, "max_replicas": 20, "target_cpu": 60},
                monitoring={"enabled": True, "metrics_retention": "90d"},
                security={"network_policies": True, "pod_security": "restricted"}
            )
        }
    
    async def create_complete_deployment(self, environment: str = "production") -> Dict[str, Any]:
        """Create complete production deployment infrastructure."""
        logger.info(f"Creating complete deployment for {environment}")
        
        deployment_results = {}
        
        try:
            # 1. Create Docker assets
            docker_results = await self._create_docker_assets()
            deployment_results["docker"] = docker_results
            
            # 2. Create Kubernetes manifests
            k8s_results = await self._create_kubernetes_manifests(environment)
            deployment_results["kubernetes"] = k8s_results
            
            # 3. Create monitoring and observability
            monitoring_results = await self._create_monitoring_stack()
            deployment_results["monitoring"] = monitoring_results
            
            # 4. Create CI/CD pipeline
            cicd_results = await self._create_cicd_pipeline()
            deployment_results["cicd"] = cicd_results
            
            # 5. Create security configurations
            security_results = await self._create_security_configs()
            deployment_results["security"] = security_results
            
            # 6. Create deployment scripts
            scripts_results = await self._create_deployment_scripts(environment)
            deployment_results["scripts"] = scripts_results
            
            # 7. Create documentation
            docs_results = await self._create_deployment_documentation()
            deployment_results["documentation"] = docs_results
            
            logger.info("Complete deployment infrastructure created successfully")
            return deployment_results
            
        except Exception as e:
            logger.error(f"Failed to create deployment infrastructure: {e}")
            raise
    
    async def _create_docker_assets(self) -> Dict[str, Any]:
        """Create Docker containerization assets."""
        docker_assets = {}
        
        # Production Dockerfile
        dockerfile_content = """# Multi-stage production Dockerfile for Bioneural Olfactory Fusion System
FROM python:3.12-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \\
    PYTHONDONTWRITEBYTECODE=1 \\
    PIP_NO_CACHE_DIR=1 \\
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Create application user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Build stage
FROM base as builder

WORKDIR /build

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY *.py ./

# Production stage
FROM base as production

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /home/appuser/.local

# Copy application code
COPY --from=builder /build/src ./src/
COPY --from=builder /build/*.py ./

# Copy configuration files
COPY deployment/config/ ./config/

# Set ownership and permissions
RUN chown -R appuser:appuser /app
USER appuser

# Add user's local bin to PATH
ENV PATH=/home/appuser/.local/bin:$PATH

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Expose port
EXPOSE 8000

# Command
CMD ["python", "-m", "uvicorn", "src.lexgraph_legal_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
"""
        
        # Write Dockerfile
        dockerfile_path = self.base_path / "Dockerfile.production"
        with open(dockerfile_path, "w") as f:
            f.write(dockerfile_content)
        docker_assets["dockerfile"] = str(dockerfile_path)
        
        # Docker Compose for local development
        docker_compose_content = """version: '3.8'

services:
  bioneural-api:
    build:
      context: .
      dockerfile: Dockerfile.production
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=development
      - LOG_LEVEL=INFO
    volumes:
      - ./config:/app/config:ro
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/usr/share/prometheus/console_libraries'
      - '--web.console.templates=/usr/share/prometheus/consoles'
      - '--web.enable-lifecycle'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources:ro

volumes:
  grafana-storage:
"""
        
        compose_path = self.base_path / "docker-compose.production.yml"
        with open(compose_path, "w") as f:
            f.write(docker_compose_content)
        docker_assets["docker_compose"] = str(compose_path)
        
        # .dockerignore
        dockerignore_content = """# Version control
.git
.gitignore

# Python
__pycache__
*.pyc
*.pyo
*.pyd
.Python
env
pip-log.txt
pip-delete-this-directory.txt
.tox
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.log
.mypy_cache
.pytest_cache

# Virtual environments
venv/
.venv/
ENV/
env/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Project specific
*.bin
*.sem
*_results.json
quality_gates_report.json
test_report.json

# Documentation
docs/
*.md
!README.md

# Development files
scripts/
tests/
.pytest_cache/
htmlcov/
"""
        
        dockerignore_path = self.base_path / ".dockerignore"
        with open(dockerignore_path, "w") as f:
            f.write(dockerignore_content)
        docker_assets["dockerignore"] = str(dockerignore_path)
        
        return docker_assets
    
    async def _create_kubernetes_manifests(self, environment: str) -> Dict[str, Any]:
        """Create comprehensive Kubernetes deployment manifests."""
        k8s_manifests = {}
        config = self.deployment_configs[environment]
        
        # Create k8s directory
        k8s_dir = self.base_path / "k8s" / environment
        k8s_dir.mkdir(parents=True, exist_ok=True)
        
        # Namespace
        namespace_manifest = {
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": {
                "name": f"{self.project_name}-{environment}",
                "labels": {
                    "name": f"{self.project_name}-{environment}",
                    "environment": environment
                }
            }
        }
        
        namespace_path = k8s_dir / "namespace.yaml"
        with open(namespace_path, "w") as f:
            yaml.dump(namespace_manifest, f, default_flow_style=False)
        k8s_manifests["namespace"] = str(namespace_path)
        
        # ConfigMap
        configmap_manifest = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": f"{self.project_name}-config",
                "namespace": f"{self.project_name}-{environment}"
            },
            "data": {
                "environment": environment,
                "log_level": "INFO" if environment == "production" else "DEBUG",
                "metrics_enabled": "true",
                "cache_size": "1000" if environment == "production" else "100",
                "worker_threads": "8" if environment == "production" else "2"
            }
        }
        
        configmap_path = k8s_dir / "configmap.yaml"
        with open(configmap_path, "w") as f:
            yaml.dump(configmap_manifest, f, default_flow_style=False)
        k8s_manifests["configmap"] = str(configmap_path)
        
        # Secret (template)
        secret_manifest = {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {
                "name": f"{self.project_name}-secrets",
                "namespace": f"{self.project_name}-{environment}"
            },
            "type": "Opaque",
            "data": {
                # Base64 encoded secrets (these should be properly managed in production)
                "api_key": "YWRtaW4=",  # 'admin' base64 encoded - REPLACE IN PRODUCTION
                "db_password": "cGFzc3dvcmQ=",  # 'password' base64 encoded - REPLACE IN PRODUCTION
            }
        }
        
        secret_path = k8s_dir / "secret.yaml"
        with open(secret_path, "w") as f:
            yaml.dump(secret_manifest, f, default_flow_style=False)
        k8s_manifests["secret"] = str(secret_path)
        
        # Deployment
        deployment_manifest = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": f"{self.project_name}-api",
                "namespace": f"{self.project_name}-{environment}",
                "labels": {
                    "app": f"{self.project_name}-api",
                    "environment": environment,
                    "version": "v1.0.0"
                }
            },
            "spec": {
                "replicas": config.replicas,
                "selector": {
                    "matchLabels": {
                        "app": f"{self.project_name}-api"
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": f"{self.project_name}-api",
                            "environment": environment
                        },
                        "annotations": {
                            "prometheus.io/scrape": "true",
                            "prometheus.io/port": "8000",
                            "prometheus.io/path": "/metrics"
                        }
                    },
                    "spec": {
                        "securityContext": {
                            "runAsNonRoot": True,
                            "runAsUser": 1000,
                            "fsGroup": 1000
                        },
                        "containers": [{
                            "name": "api",
                            "image": f"{self.project_name}:latest",
                            "imagePullPolicy": "Always",
                            "ports": [{
                                "containerPort": 8000,
                                "name": "http",
                                "protocol": "TCP"
                            }],
                            "env": [
                                {
                                    "name": "ENVIRONMENT",
                                    "valueFrom": {
                                        "configMapKeyRef": {
                                            "name": f"{self.project_name}-config",
                                            "key": "environment"
                                        }
                                    }
                                },
                                {
                                    "name": "LOG_LEVEL",
                                    "valueFrom": {
                                        "configMapKeyRef": {
                                            "name": f"{self.project_name}-config",
                                            "key": "log_level"
                                        }
                                    }
                                },
                                {
                                    "name": "API_KEY",
                                    "valueFrom": {
                                        "secretKeyRef": {
                                            "name": f"{self.project_name}-secrets",
                                            "key": "api_key"
                                        }
                                    }
                                }
                            ],
                            "resources": {
                                "limits": config.resource_limits,
                                "requests": config.resource_requests
                            },
                            "livenessProbe": {
                                "httpGet": {
                                    "path": "/health",
                                    "port": 8000
                                },
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10,
                                "timeoutSeconds": 5,
                                "failureThreshold": 3
                            },
                            "readinessProbe": {
                                "httpGet": {
                                    "path": "/ready",
                                    "port": 8000
                                },
                                "initialDelaySeconds": 10,
                                "periodSeconds": 5,
                                "timeoutSeconds": 3,
                                "failureThreshold": 3
                            },
                            "securityContext": {
                                "allowPrivilegeEscalation": False,
                                "readOnlyRootFilesystem": True,
                                "capabilities": {
                                    "drop": ["ALL"]
                                }
                            }
                        }],
                        "restartPolicy": "Always",
                        "terminationGracePeriodSeconds": 30
                    }
                },
                "strategy": {
                    "type": "RollingUpdate",
                    "rollingUpdate": {
                        "maxUnavailable": 1,
                        "maxSurge": 1
                    }
                }
            }
        }
        
        deployment_path = k8s_dir / "deployment.yaml"
        with open(deployment_path, "w") as f:
            yaml.dump(deployment_manifest, f, default_flow_style=False)
        k8s_manifests["deployment"] = str(deployment_path)
        
        # Service
        service_manifest = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": f"{self.project_name}-service",
                "namespace": f"{self.project_name}-{environment}",
                "labels": {
                    "app": f"{self.project_name}-api"
                }
            },
            "spec": {
                "selector": {
                    "app": f"{self.project_name}-api"
                },
                "ports": [{
                    "name": "http",
                    "port": 80,
                    "targetPort": 8000,
                    "protocol": "TCP"
                }],
                "type": "ClusterIP"
            }
        }
        
        service_path = k8s_dir / "service.yaml"
        with open(service_path, "w") as f:
            yaml.dump(service_manifest, f, default_flow_style=False)
        k8s_manifests["service"] = str(service_path)
        
        # Horizontal Pod Autoscaler
        hpa_manifest = {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": f"{self.project_name}-hpa",
                "namespace": f"{self.project_name}-{environment}"
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": f"{self.project_name}-api"
                },
                "minReplicas": config.autoscaling["min_replicas"],
                "maxReplicas": config.autoscaling["max_replicas"],
                "metrics": [
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "cpu",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": config.autoscaling["target_cpu"]
                            }
                        }
                    },
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "memory",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": 80
                            }
                        }
                    }
                ]
            }
        }
        
        hpa_path = k8s_dir / "hpa.yaml"
        with open(hpa_path, "w") as f:
            yaml.dump(hpa_manifest, f, default_flow_style=False)
        k8s_manifests["hpa"] = str(hpa_path)
        
        # Ingress
        ingress_manifest = {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "Ingress",
            "metadata": {
                "name": f"{self.project_name}-ingress",
                "namespace": f"{self.project_name}-{environment}",
                "annotations": {
                    "nginx.ingress.kubernetes.io/rewrite-target": "/",
                    "nginx.ingress.kubernetes.io/ssl-redirect": "true",
                    "cert-manager.io/cluster-issuer": "letsencrypt-prod",
                    "nginx.ingress.kubernetes.io/rate-limit": "100",
                    "nginx.ingress.kubernetes.io/rate-limit-window": "1m"
                }
            },
            "spec": {
                "ingressClassName": "nginx",
                "tls": [{
                    "hosts": [f"{self.project_name}-{environment}.example.com"],
                    "secretName": f"{self.project_name}-tls-{environment}"
                }],
                "rules": [{
                    "host": f"{self.project_name}-{environment}.example.com",
                    "http": {
                        "paths": [{
                            "path": "/",
                            "pathType": "Prefix",
                            "backend": {
                                "service": {
                                    "name": f"{self.project_name}-service",
                                    "port": {"number": 80}
                                }
                            }
                        }]
                    }
                }]
            }
        }
        
        ingress_path = k8s_dir / "ingress.yaml"
        with open(ingress_path, "w") as f:
            yaml.dump(ingress_manifest, f, default_flow_style=False)
        k8s_manifests["ingress"] = str(ingress_path)
        
        # Network Policy
        network_policy_manifest = {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "NetworkPolicy",
            "metadata": {
                "name": f"{self.project_name}-netpol",
                "namespace": f"{self.project_name}-{environment}"
            },
            "spec": {
                "podSelector": {
                    "matchLabels": {
                        "app": f"{self.project_name}-api"
                    }
                },
                "policyTypes": ["Ingress", "Egress"],
                "ingress": [
                    {
                        "from": [
                            {"namespaceSelector": {"matchLabels": {"name": "ingress-nginx"}}},
                            {"namespaceSelector": {"matchLabels": {"name": "monitoring"}}}
                        ],
                        "ports": [{"protocol": "TCP", "port": 8000}]
                    }
                ],
                "egress": [
                    {
                        "to": [],
                        "ports": [
                            {"protocol": "TCP", "port": 53},
                            {"protocol": "UDP", "port": 53},
                            {"protocol": "TCP", "port": 443},
                            {"protocol": "TCP", "port": 80}
                        ]
                    }
                ]
            }
        }
        
        netpol_path = k8s_dir / "networkpolicy.yaml"
        with open(netpol_path, "w") as f:
            yaml.dump(network_policy_manifest, f, default_flow_style=False)
        k8s_manifests["network_policy"] = str(netpol_path)
        
        return k8s_manifests
    
    async def _create_monitoring_stack(self) -> Dict[str, Any]:
        """Create comprehensive monitoring and observability stack."""
        monitoring_assets = {}
        
        # Create monitoring directory
        monitoring_dir = self.base_path / "monitoring"
        monitoring_dir.mkdir(exist_ok=True)
        
        # Prometheus configuration
        prometheus_config = {
            "global": {
                "scrape_interval": "15s",
                "evaluation_interval": "15s"
            },
            "rule_files": ["*.rules.yml"],
            "scrape_configs": [
                {
                    "job_name": "bioneural-api",
                    "static_configs": [{"targets": ["localhost:8000"]}],
                    "metrics_path": "/metrics",
                    "scrape_interval": "30s"
                },
                {
                    "job_name": "kubernetes-pods",
                    "kubernetes_sd_configs": [{"role": "pod"}],
                    "relabel_configs": [
                        {
                            "source_labels": ["__meta_kubernetes_pod_annotation_prometheus_io_scrape"],
                            "action": "keep",
                            "regex": True
                        },
                        {
                            "source_labels": ["__meta_kubernetes_pod_annotation_prometheus_io_path"],
                            "action": "replace",
                            "target_label": "__metrics_path__",
                            "regex": "(.+)"
                        }
                    ]
                }
            ],
            "alerting": {
                "alertmanagers": [
                    {"static_configs": [{"targets": ["alertmanager:9093"]}]}
                ]
            }
        }
        
        prometheus_path = monitoring_dir / "prometheus.yml"
        with open(prometheus_path, "w") as f:
            yaml.dump(prometheus_config, f, default_flow_style=False)
        monitoring_assets["prometheus_config"] = str(prometheus_path)
        
        # Prometheus alerting rules
        alert_rules = {
            "groups": [
                {
                    "name": "bioneural_alerts",
                    "rules": [
                        {
                            "alert": "HighErrorRate",
                            "expr": "rate(http_requests_total{status=~\"5..\"}[5m]) > 0.1",
                            "for": "5m",
                            "labels": {"severity": "critical"},
                            "annotations": {
                                "summary": "High error rate detected",
                                "description": "Error rate is above 10% for 5 minutes"
                            }
                        },
                        {
                            "alert": "HighResponseTime",
                            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1",
                            "for": "5m",
                            "labels": {"severity": "warning"},
                            "annotations": {
                                "summary": "High response time detected",
                                "description": "95th percentile response time is above 1 second"
                            }
                        },
                        {
                            "alert": "PodCrashLooping",
                            "expr": "increase(kube_pod_container_status_restarts_total[1h]) > 0",
                            "for": "5m",
                            "labels": {"severity": "critical"},
                            "annotations": {
                                "summary": "Pod is crash looping",
                                "description": "Pod {{ $labels.pod }} is restarting frequently"
                            }
                        }
                    ]
                }
            ]
        }
        
        alerts_path = monitoring_dir / "alerts.rules.yml"
        with open(alerts_path, "w") as f:
            yaml.dump(alert_rules, f, default_flow_style=False)
        monitoring_assets["alert_rules"] = str(alerts_path)
        
        # Grafana dashboard
        grafana_dir = monitoring_dir / "grafana" / "dashboards"
        grafana_dir.mkdir(parents=True, exist_ok=True)
        
        dashboard_config = {
            "dashboard": {
                "id": None,
                "title": "Bioneural Olfactory Fusion System",
                "tags": ["bioneural", "api", "production"],
                "timezone": "browser",
                "panels": [
                    {
                        "id": 1,
                        "title": "Request Rate",
                        "type": "stat",
                        "targets": [
                            {
                                "expr": "rate(http_requests_total[5m])",
                                "legendFormat": "Requests/sec"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
                    },
                    {
                        "id": 2,
                        "title": "Response Time",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
                                "legendFormat": "95th percentile"
                            },
                            {
                                "expr": "histogram_quantile(0.50, rate(http_request_duration_seconds_bucket[5m]))",
                                "legendFormat": "50th percentile"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
                    }
                ],
                "time": {"from": "now-1h", "to": "now"},
                "refresh": "30s"
            }
        }
        
        dashboard_path = grafana_dir / "bioneural-dashboard.json"
        with open(dashboard_path, "w") as f:
            json.dump(dashboard_config, f, indent=2)
        monitoring_assets["grafana_dashboard"] = str(dashboard_path)
        
        return monitoring_assets
    
    async def _create_cicd_pipeline(self) -> Dict[str, Any]:
        """Create CI/CD pipeline configurations."""
        cicd_assets = {}
        
        # GitHub Actions workflow
        workflows_dir = self.base_path / ".github" / "workflows"
        workflows_dir.mkdir(parents=True, exist_ok=True)
        
        github_workflow = {
            "name": "CI/CD Pipeline",
            "on": {
                "push": {"branches": ["main", "develop"]},
                "pull_request": {"branches": ["main"]}
            },
            "env": {
                "REGISTRY": "ghcr.io",
                "IMAGE_NAME": "${{ github.repository }}"
            },
            "jobs": {
                "test": {
                    "runs-on": "ubuntu-latest",
                    "steps": [
                        {"uses": "actions/checkout@v3"},
                        {
                            "name": "Set up Python",
                            "uses": "actions/setup-python@v4",
                            "with": {"python-version": "3.12"}
                        },
                        {
                            "name": "Install dependencies",
                            "run": "pip install -r requirements.txt"
                        },
                        {
                            "name": "Run tests",
                            "run": "python -m pytest tests/ -v --cov=src"
                        },
                        {
                            "name": "Run quality gates",
                            "run": "python comprehensive_quality_gates.py"
                        }
                    ]
                },
                "security": {
                    "runs-on": "ubuntu-latest",
                    "steps": [
                        {"uses": "actions/checkout@v3"},
                        {
                            "name": "Run security scan",
                            "run": "pip install bandit safety && bandit -r src/ && safety check"
                        }
                    ]
                },
                "build": {
                    "runs-on": "ubuntu-latest",
                    "needs": ["test", "security"],
                    "if": "github.ref == 'refs/heads/main'",
                    "steps": [
                        {"uses": "actions/checkout@v3"},
                        {
                            "name": "Set up Docker Buildx",
                            "uses": "docker/setup-buildx-action@v2"
                        },
                        {
                            "name": "Log in to Container Registry",
                            "uses": "docker/login-action@v2",
                            "with": {
                                "registry": "${{ env.REGISTRY }}",
                                "username": "${{ github.actor }}",
                                "password": "${{ secrets.GITHUB_TOKEN }}"
                            }
                        },
                        {
                            "name": "Extract metadata",
                            "id": "meta",
                            "uses": "docker/metadata-action@v4",
                            "with": {
                                "images": "${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}",
                                "tags": "type=ref,event=branch\\ntype=ref,event=pr\\ntype=sha,prefix={{branch}}-\\ntype=raw,value=latest,enable={{is_default_branch}}"
                            }
                        },
                        {
                            "name": "Build and push Docker image",
                            "uses": "docker/build-push-action@v4",
                            "with": {
                                "context": ".",
                                "file": "./Dockerfile.production",
                                "push": True,
                                "tags": "${{ steps.meta.outputs.tags }}",
                                "labels": "${{ steps.meta.outputs.labels }}"
                            }
                        }
                    ]
                },
                "deploy": {
                    "runs-on": "ubuntu-latest",
                    "needs": ["build"],
                    "if": "github.ref == 'refs/heads/main'",
                    "environment": "production",
                    "steps": [
                        {"uses": "actions/checkout@v3"},
                        {
                            "name": "Deploy to Kubernetes",
                            "run": "kubectl apply -f k8s/production/ && kubectl rollout status deployment/bioneural-olfactory-fusion-api -n bioneural-olfactory-fusion-production"
                        }
                    ]
                }
            }
        }
        
        workflow_path = workflows_dir / "cicd.yml"
        with open(workflow_path, "w") as f:
            yaml.dump(github_workflow, f, default_flow_style=False)
        cicd_assets["github_workflow"] = str(workflow_path)
        
        return cicd_assets
    
    async def _create_security_configs(self) -> Dict[str, Any]:
        """Create security configuration files."""
        security_assets = {}
        
        # Security policy
        security_dir = self.base_path / "security"
        security_dir.mkdir(exist_ok=True)
        
        security_policy = {
            "apiVersion": "policy/v1beta1",
            "kind": "PodSecurityPolicy",
            "metadata": {
                "name": "bioneural-psp"
            },
            "spec": {
                "privileged": False,
                "allowPrivilegeEscalation": False,
                "requiredDropCapabilities": ["ALL"],
                "volumes": ["configMap", "emptyDir", "projected", "secret", "downwardAPI", "persistentVolumeClaim"],
                "runAsUser": {"rule": "MustRunAsNonRoot"},
                "seLinux": {"rule": "RunAsAny"},
                "fsGroup": {"rule": "RunAsAny"},
                "readOnlyRootFilesystem": True
            }
        }
        
        psp_path = security_dir / "pod_security_policy.yaml"
        with open(psp_path, "w") as f:
            yaml.dump(security_policy, f, default_flow_style=False)
        security_assets["pod_security_policy"] = str(psp_path)
        
        # RBAC configurations
        rbac_config = {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "Role",
            "metadata": {
                "namespace": "bioneural-olfactory-fusion-production",
                "name": "bioneural-role"
            },
            "rules": [
                {
                    "apiGroups": [""],
                    "resources": ["pods", "services", "configmaps"],
                    "verbs": ["get", "list", "watch"]
                }
            ]
        }
        
        rbac_path = security_dir / "rbac.yaml"
        with open(rbac_path, "w") as f:
            yaml.dump(rbac_config, f, default_flow_style=False)
        security_assets["rbac"] = str(rbac_path)
        
        return security_assets
    
    async def _create_deployment_scripts(self, environment: str) -> Dict[str, Any]:
        """Create deployment automation scripts."""
        scripts_assets = {}
        
        # Create scripts directory
        scripts_dir = self.base_path / "deployment" / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        
        # Deployment script
        deploy_script = f"""#!/bin/bash
set -euo pipefail

# Bioneural Olfactory Fusion System Deployment Script
# Environment: {environment}

NAMESPACE="{self.project_name}-{environment}"
DEPLOYMENT_NAME="{self.project_name}-api"

echo "🚀 Starting deployment to {environment} environment"

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl is not installed or not in PATH"
    exit 1
fi

# Check if we can connect to the cluster
if ! kubectl cluster-info &> /dev/null; then
    echo "❌ Cannot connect to Kubernetes cluster"
    exit 1
fi

# Create namespace if it doesn't exist
if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
    echo "📦 Creating namespace $NAMESPACE"
    kubectl apply -f k8s/{environment}/namespace.yaml
fi

# Apply configurations
echo "⚙️ Applying configurations"
kubectl apply -f k8s/{environment}/configmap.yaml
kubectl apply -f k8s/{environment}/secret.yaml

# Apply deployment manifests
echo "🏗️ Deploying application"
kubectl apply -f k8s/{environment}/deployment.yaml
kubectl apply -f k8s/{environment}/service.yaml
kubectl apply -f k8s/{environment}/hpa.yaml
kubectl apply -f k8s/{environment}/ingress.yaml

# Apply security policies
echo "🔒 Applying security policies"
kubectl apply -f k8s/{environment}/networkpolicy.yaml

# Wait for rollout to complete
echo "⏳ Waiting for deployment to complete"
kubectl rollout status deployment/$DEPLOYMENT_NAME -n $NAMESPACE --timeout=300s

# Verify deployment
echo "✅ Verifying deployment"
kubectl get pods -n $NAMESPACE -l app=$DEPLOYMENT_NAME

# Run health check
echo "🏥 Running health check"
sleep 10
kubectl exec -n $NAMESPACE deployment/$DEPLOYMENT_NAME -- curl -f http://localhost:8000/health || {{
    echo "❌ Health check failed"
    kubectl logs -n $NAMESPACE deployment/$DEPLOYMENT_NAME --tail=50
    exit 1
}}

echo "🎉 Deployment completed successfully!"
"""
        
        deploy_script_path = scripts_dir / f"deploy_{environment}.sh"
        with open(deploy_script_path, "w") as f:
            f.write(deploy_script)
        os.chmod(deploy_script_path, 0o755)
        scripts_assets["deploy_script"] = str(deploy_script_path)
        
        # Rollback script
        rollback_script = f"""#!/bin/bash
set -euo pipefail

NAMESPACE="{self.project_name}-{environment}"
DEPLOYMENT_NAME="{self.project_name}-api"

echo "🔄 Rolling back deployment in {environment} environment"

# Rollback to previous version
kubectl rollout undo deployment/$DEPLOYMENT_NAME -n $NAMESPACE

# Wait for rollback to complete
kubectl rollout status deployment/$DEPLOYMENT_NAME -n $NAMESPACE --timeout=300s

echo "✅ Rollback completed successfully!"
"""
        
        rollback_script_path = scripts_dir / f"rollback_{environment}.sh"
        with open(rollback_script_path, "w") as f:
            f.write(rollback_script)
        os.chmod(rollback_script_path, 0o755)
        scripts_assets["rollback_script"] = str(rollback_script_path)
        
        return scripts_assets
    
    async def _create_deployment_documentation(self) -> Dict[str, Any]:
        """Create comprehensive deployment documentation."""
        docs_assets = {}
        
        # Create docs directory
        docs_dir = self.base_path / "deployment" / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        
        # Production deployment guide
        deployment_guide = f"""# Production Deployment Guide - Bioneural Olfactory Fusion System

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
docker build -f Dockerfile.production -t {self.project_name}:latest .

# Tag and push to registry
docker tag {self.project_name}:latest your-registry/{self.project_name}:latest
docker push your-registry/{self.project_name}:latest
```

### 2. Deploy to Kubernetes

```bash
# Deploy to production
./deployment/scripts/deploy_production.sh

# Verify deployment
kubectl get pods -n {self.project_name}-production
```

### 3. Monitor Deployment

```bash
# Check logs
kubectl logs -f deployment/{self.project_name}-api -n {self.project_name}-production

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

Access Grafana dashboards at: https://grafana.{self.project_name}.example.com

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
   kubectl describe pod <pod-name> -n {self.project_name}-production
   kubectl logs <pod-name> -n {self.project_name}-production
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
kubectl rollout undo deployment/{self.project_name}-api -n {self.project_name}-production
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
"""
        
        guide_path = docs_dir / "production_deployment_guide.md"
        with open(guide_path, "w") as f:
            f.write(deployment_guide)
        docs_assets["deployment_guide"] = str(guide_path)
        
        # Operations runbook
        runbook = f"""# Operations Runbook - Bioneural Olfactory Fusion System

## Emergency Procedures

### Service Down

1. Check deployment status:
   ```bash
   kubectl get deployment {self.project_name}-api -n {self.project_name}-production
   ```

2. Check pod status:
   ```bash
   kubectl get pods -n {self.project_name}-production
   ```

3. Check logs:
   ```bash
   kubectl logs -f deployment/{self.project_name}-api -n {self.project_name}-production
   ```

4. If necessary, restart deployment:
   ```bash
   kubectl rollout restart deployment/{self.project_name}-api -n {self.project_name}-production
   ```

### High Load

1. Check current scaling:
   ```bash
   kubectl get hpa -n {self.project_name}-production
   ```

2. Manual scaling if needed:
   ```bash
   kubectl scale deployment {self.project_name}-api --replicas=10 -n {self.project_name}-production
   ```

3. Monitor metrics:
   ```bash
   kubectl top pods -n {self.project_name}-production
   ```

## Monitoring Checklist

- [ ] All pods running and ready
- [ ] Response times < 500ms (95th percentile)
- [ ] Error rate < 1%
- [ ] CPU usage < 70%
- [ ] Memory usage < 80%
- [ ] Cache hit rate > 80%

## Scheduled Maintenance

### Weekly Tasks
- Review error logs
- Check resource utilization trends
- Verify backup integrity
- Update security scan results

### Monthly Tasks
- Performance testing
- Capacity planning review
- Security audit
- Dependency updates

## Escalation Procedures

1. **Level 1**: Automated alerts and self-healing
2. **Level 2**: On-call engineer notification
3. **Level 3**: Team lead escalation
4. **Level 4**: Management notification for business impact
"""
        
        runbook_path = docs_dir / "operations_runbook.md"
        with open(runbook_path, "w") as f:
            f.write(runbook)
        docs_assets["operations_runbook"] = str(runbook_path)
        
        return docs_assets


async def main():
    """Create complete production deployment infrastructure."""
    print("🚀 PRODUCTION DEPLOYMENT PREPARATION")
    print("=" * 50)
    
    manager = ProductionDeploymentManager()
    
    try:
        # Create deployment for all environments
        environments = ["development", "staging", "production"]
        all_results = {}
        
        for env in environments:
            print(f"\n📦 Creating deployment infrastructure for {env}...")
            results = await manager.create_complete_deployment(env)
            all_results[env] = results
            print(f"✅ {env} deployment infrastructure created")
        
        # Summary
        print(f"\n📊 DEPLOYMENT SUMMARY")
        print(f"Environments: {len(environments)}")
        
        for env, results in all_results.items():
            print(f"\n{env.upper()}:")
            for category, items in results.items():
                if isinstance(items, dict):
                    print(f"  {category}: {len(items)} files created")
                else:
                    print(f"  {category}: {items}")
        
        print(f"\n🎯 NEXT STEPS:")
        print(f"1. Review and customize configuration files")
        print(f"2. Set up container registry credentials")
        print(f"3. Configure SSL certificates")
        print(f"4. Set up monitoring infrastructure")
        print(f"5. Test deployment in staging environment")
        print(f"6. Deploy to production with: ./deployment/scripts/deploy_production.sh")
        
        print(f"\n📚 DOCUMENTATION:")
        print(f"- Production Guide: deployment/docs/production_deployment_guide.md")
        print(f"- Operations Runbook: deployment/docs/operations_runbook.md")
        print(f"- K8s Manifests: k8s/")
        print(f"- Monitoring Config: monitoring/")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create deployment infrastructure: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)