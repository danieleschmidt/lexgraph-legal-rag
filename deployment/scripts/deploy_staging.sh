#!/bin/bash
set -euo pipefail

# Bioneural Olfactory Fusion System Deployment Script
# Environment: staging

NAMESPACE="bioneural-olfactory-fusion-staging"
DEPLOYMENT_NAME="bioneural-olfactory-fusion-api"

echo "🚀 Starting deployment to staging environment"

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
    kubectl apply -f k8s/staging/namespace.yaml
fi

# Apply configurations
echo "⚙️ Applying configurations"
kubectl apply -f k8s/staging/configmap.yaml
kubectl apply -f k8s/staging/secret.yaml

# Apply deployment manifests
echo "🏗️ Deploying application"
kubectl apply -f k8s/staging/deployment.yaml
kubectl apply -f k8s/staging/service.yaml
kubectl apply -f k8s/staging/hpa.yaml
kubectl apply -f k8s/staging/ingress.yaml

# Apply security policies
echo "🔒 Applying security policies"
kubectl apply -f k8s/staging/networkpolicy.yaml

# Wait for rollout to complete
echo "⏳ Waiting for deployment to complete"
kubectl rollout status deployment/$DEPLOYMENT_NAME -n $NAMESPACE --timeout=300s

# Verify deployment
echo "✅ Verifying deployment"
kubectl get pods -n $NAMESPACE -l app=$DEPLOYMENT_NAME

# Run health check
echo "🏥 Running health check"
sleep 10
kubectl exec -n $NAMESPACE deployment/$DEPLOYMENT_NAME -- curl -f http://localhost:8000/health || {
    echo "❌ Health check failed"
    kubectl logs -n $NAMESPACE deployment/$DEPLOYMENT_NAME --tail=50
    exit 1
}

echo "🎉 Deployment completed successfully!"
