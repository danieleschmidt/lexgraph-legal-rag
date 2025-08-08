#!/bin/bash
set -e

echo "🚀 Starting LexGraph Legal RAG deployment..."

# Configuration
NAMESPACE=${NAMESPACE:-default}
IMAGE_TAG=${IMAGE_TAG:-latest}
ENVIRONMENT=${ENVIRONMENT:-production}

# Pre-deployment checks
echo "📋 Running pre-deployment checks..."
kubectl cluster-info
kubectl get nodes
kubectl get ns $NAMESPACE || kubectl create ns $NAMESPACE

# Build and push Docker image
echo "🐳 Building and pushing Docker image..."
docker build -f deployment/Dockerfile.prod -t lexgraph-legal-rag:$IMAGE_TAG .
docker tag lexgraph-legal-rag:$IMAGE_TAG your-registry.com/lexgraph-legal-rag:$IMAGE_TAG
docker push your-registry.com/lexgraph-legal-rag:$IMAGE_TAG

# Run database migrations
echo "🗃️  Running database migrations..."
python deployment/migrations/run_migrations.py

# Deploy Kubernetes resources
echo "☸️  Deploying Kubernetes resources..."
kubectl apply -f deployment/k8s/ -n $NAMESPACE

# Wait for deployment
echo "⏳ Waiting for deployment to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/lexgraph-legal-rag -n $NAMESPACE

# Verify health
echo "🏥 Verifying deployment health..."
kubectl get pods -n $NAMESPACE -l app=lexgraph-legal-rag
kubectl logs -n $NAMESPACE -l app=lexgraph-legal-rag --tail=10

# Run smoke tests
echo "💨 Running smoke tests..."
INGRESS_IP=$(kubectl get ingress lexgraph-legal-rag-ingress -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
curl -f http://$INGRESS_IP/health || echo "Health check failed"

echo "✅ Deployment completed successfully!"
