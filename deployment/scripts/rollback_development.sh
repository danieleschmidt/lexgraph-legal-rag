#!/bin/bash
set -euo pipefail

NAMESPACE="bioneural-olfactory-fusion-development"
DEPLOYMENT_NAME="bioneural-olfactory-fusion-api"

echo "🔄 Rolling back deployment in development environment"

# Rollback to previous version
kubectl rollout undo deployment/$DEPLOYMENT_NAME -n $NAMESPACE

# Wait for rollback to complete
kubectl rollout status deployment/$DEPLOYMENT_NAME -n $NAMESPACE --timeout=300s

echo "✅ Rollback completed successfully!"
