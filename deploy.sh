#!/bin/bash
set -e

# LexGraph Legal RAG Deployment Script
# This script provides various deployment options for the application

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="lexgraph-legal-rag"
NAMESPACE="lexgraph"
COMPOSE_PROJECT="lexgraph"

print_banner() {
    echo -e "${BLUE}"
    echo "=================================================="
    echo "  LexGraph Legal RAG Deployment Script"
    echo "=================================================="
    echo -e "${NC}"
}

print_usage() {
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  docker-dev      Start development environment with Docker Compose"
    echo "  docker-prod     Start production environment with Docker Compose"
    echo "  k8s-deploy      Deploy to Kubernetes cluster"
    echo "  k8s-delete      Delete from Kubernetes cluster"
    echo "  build           Build Docker images"
    echo "  test            Run tests"
    echo "  benchmark       Run performance benchmarks"
    echo "  clean           Clean up resources"
    echo ""
    echo "Options:"
    echo "  --api-key KEY   Set API key (default: from environment)"
    echo "  --tag TAG       Docker image tag (default: latest)"
    echo "  --namespace NS  Kubernetes namespace (default: lexgraph)"
    echo "  --help          Show this help message"
}

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_dependencies() {
    local deps=("$@")
    local missing=()
    
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            missing+=("$dep")
        fi
    done
    
    if [ ${#missing[@]} -ne 0 ]; then
        log_error "Missing dependencies: ${missing[*]}"
        log_info "Please install the missing dependencies and try again."
        exit 1
    fi
}

wait_for_service() {
    local url="$1"
    local max_attempts="${2:-30}"
    local attempt=1
    
    log_info "Waiting for service at $url..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s "$url" > /dev/null 2>&1; then
            log_info "Service is ready!"
            return 0
        fi
        
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    log_error "Service did not become ready within timeout"
    return 1
}

build_images() {
    local tag="${1:-latest}"
    
    log_info "Building Docker images..."
    
    # Build development image
    log_info "Building development image..."
    docker build --target development -t "${IMAGE_NAME}:dev-${tag}" .
    
    # Build production image
    log_info "Building production image..."
    docker build --target production -t "${IMAGE_NAME}:${tag}" .
    
    log_info "Docker images built successfully"
}

docker_dev() {
    local api_key="${1:-development-key}"
    
    check_dependencies docker docker-compose
    
    log_info "Starting development environment..."
    
    # Set environment variables
    export API_KEY="$api_key"
    export COMPOSE_PROJECT_NAME="$COMPOSE_PROJECT"
    
    # Start development services
    docker-compose --profile development up -d
    
    # Wait for services
    wait_for_service "http://localhost:8080/health"
    
    log_info "Development environment is ready!"
    log_info "API available at: http://localhost:8080"
    log_info "Documentation at: http://localhost:8080/docs"
    log_info "Metrics at: http://localhost:8081"
}

docker_prod() {
    local api_key="${1:-production-key}"
    
    check_dependencies docker docker-compose
    
    log_info "Starting production environment..."
    
    # Set environment variables
    export API_KEY="$api_key"
    export COMPOSE_PROJECT_NAME="$COMPOSE_PROJECT"
    
    # Start production services
    docker-compose up -d lexgraph-api redis prometheus grafana
    
    # Wait for services
    wait_for_service "http://localhost:8000/health"
    
    log_info "Production environment is ready!"
    log_info "API available at: http://localhost:8000"
    log_info "Documentation at: http://localhost:8000/docs"
    log_info "Metrics at: http://localhost:9090 (Prometheus)"
    log_info "Dashboard at: http://localhost:3000 (Grafana)"
}

k8s_deploy() {
    local namespace="${1:-$NAMESPACE}"
    local tag="${2:-latest}"
    
    check_dependencies kubectl
    
    log_info "Deploying to Kubernetes namespace: $namespace"
    
    # Create namespace if it doesn't exist
    kubectl create namespace "$namespace" --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply Kubernetes manifests
    log_info "Applying Kubernetes manifests..."
    
    # Update image tag in deployment
    sed "s|ghcr.io/your-org/lexgraph-legal-rag:latest|ghcr.io/your-org/lexgraph-legal-rag:${tag}|g" \
        k8s/deployment.yaml | kubectl apply -f -
    
    # Apply other manifests
    kubectl apply -f k8s/configmap.yaml
    kubectl apply -f k8s/secret.yaml
    kubectl apply -f k8s/service.yaml
    kubectl apply -f k8s/hpa.yaml
    kubectl apply -f k8s/servicemonitor.yaml
    
    # Wait for deployment
    log_info "Waiting for deployment to be ready..."
    kubectl wait --for=condition=available deployment/lexgraph-api -n "$namespace" --timeout=300s
    
    # Get service information
    kubectl get services -n "$namespace"
    
    log_info "Kubernetes deployment completed successfully!"
}

k8s_delete() {
    local namespace="${1:-$NAMESPACE}"
    
    check_dependencies kubectl
    
    log_info "Deleting Kubernetes resources from namespace: $namespace"
    
    # Delete all resources
    kubectl delete -f k8s/ -n "$namespace" --ignore-not-found=true
    
    # Optionally delete namespace
    read -p "Delete namespace '$namespace'? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        kubectl delete namespace "$namespace"
    fi
    
    log_info "Kubernetes resources deleted"
}

run_tests() {
    check_dependencies python3
    
    log_info "Running test suite..."
    
    # Set test environment
    export API_KEY="test-api-key"
    
    # Run tests
    python3 -m pytest tests/ -v --cov=src/lexgraph_legal_rag --cov-report=term-missing
    
    log_info "Tests completed"
}

run_benchmarks() {
    check_dependencies python3
    
    log_info "Running performance benchmarks..."
    
    export API_KEY="benchmark-key"
    
    # Run Python benchmarks
    python3 tests/performance/benchmark.py
    
    # Run k6 load tests if available
    if command -v k6 &> /dev/null; then
        log_info "Running k6 load tests..."
        
        # Start service for testing
        docker-compose up -d lexgraph-api
        sleep 10
        
        # Run k6 tests
        k6 run tests/performance/load-test.js
        
        # Cleanup
        docker-compose down
    else
        log_warn "k6 not found, skipping load tests"
    fi
    
    log_info "Benchmarks completed"
}

clean_all() {
    log_info "Cleaning up resources..."
    
    # Stop and remove Docker containers
    docker-compose down -v --remove-orphans 2>/dev/null || true
    
    # Remove Docker images
    docker rmi "${IMAGE_NAME}:latest" "${IMAGE_NAME}:dev-latest" 2>/dev/null || true
    
    # Clean up test artifacts
    rm -f performance-*.json performance-*.html
    
    log_info "Cleanup completed"
}

# Parse arguments
API_KEY="${API_KEY:-}"
TAG="latest"
NAMESPACE="lexgraph"

while [[ $# -gt 0 ]]; do
    case $1 in
        --api-key)
            API_KEY="$2"
            shift 2
            ;;
        --tag)
            TAG="$2"
            shift 2
            ;;
        --namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --help)
            print_usage
            exit 0
            ;;
        docker-dev|docker-prod|k8s-deploy|k8s-delete|build|test|benchmark|clean)
            COMMAND="$1"
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Main execution
print_banner

case "${COMMAND:-}" in
    docker-dev)
        docker_dev "$API_KEY"
        ;;
    docker-prod)
        docker_prod "$API_KEY"
        ;;
    k8s-deploy)
        k8s_deploy "$NAMESPACE" "$TAG"
        ;;
    k8s-delete)
        k8s_delete "$NAMESPACE"
        ;;
    build)
        build_images "$TAG"
        ;;
    test)
        run_tests
        ;;
    benchmark)
        run_benchmarks
        ;;
    clean)
        clean_all
        ;;
    "")
        log_error "No command specified"
        print_usage
        exit 1
        ;;
    *)
        log_error "Unknown command: $COMMAND"
        print_usage
        exit 1
        ;;
esac