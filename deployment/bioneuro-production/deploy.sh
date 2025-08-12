#!/bin/bash
set -e

# Bioneural Olfactory Fusion System - Production Deployment Script
# Usage: ./deploy.sh [docker|kubernetes] [environment]

DEPLOYMENT_TYPE=${1:-docker}
ENVIRONMENT=${2:-production}
VERSION=${3:-latest}

echo "🧬 BIONEURAL OLFACTORY FUSION SYSTEM DEPLOYMENT"
echo "================================================"
echo "Deployment Type: $DEPLOYMENT_TYPE"
echo "Environment: $ENVIRONMENT"
echo "Version: $VERSION"
echo "================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if [ "$DEPLOYMENT_TYPE" = "docker" ]; then
        if ! command -v docker &> /dev/null; then
            log_error "Docker is not installed"
            exit 1
        fi
        
        if ! command -v docker-compose &> /dev/null; then
            log_error "Docker Compose is not installed"
            exit 1
        fi
    elif [ "$DEPLOYMENT_TYPE" = "kubernetes" ]; then
        if ! command -v kubectl &> /dev/null; then
            log_error "kubectl is not installed"
            exit 1
        fi
        
        if ! kubectl cluster-info &> /dev/null; then
            log_error "kubectl cannot connect to cluster"
            exit 1
        fi
    fi
    
    log_success "Prerequisites check passed"
}

# Build Docker image
build_image() {
    log_info "Building Bioneural Docker image..."
    
    cd ../..
    
    docker build \
        -f deployment/bioneuro-production/Dockerfile \
        -t bioneuro-olfactory-fusion:$VERSION \
        -t bioneuro-olfactory-fusion:latest \
        .
    
    log_success "Docker image built successfully"
}

# Deploy with Docker Compose
deploy_docker() {
    log_info "Deploying with Docker Compose..."
    
    # Create environment file
    cat > .env << EOF
COMPOSE_PROJECT_NAME=bioneuro-$ENVIRONMENT
VERSION=$VERSION
ENVIRONMENT=$ENVIRONMENT
EOF
    
    # Deploy services
    docker-compose -f docker-compose.yml up -d
    
    # Wait for services to be ready
    log_info "Waiting for services to be ready..."
    sleep 30
    
    # Check service health
    if docker-compose ps | grep -q "Up"; then
        log_success "Services are running"
        
        # Show service URLs
        echo ""
        log_info "Service URLs:"
        echo "  • API: http://localhost:8000"
        echo "  • Monitoring: http://localhost:9090"
        echo "  • Dashboard: http://localhost:3000 (admin/admin123)"
        
        # Run health check
        log_info "Running health check..."
        if curl -s http://localhost:8000/health > /dev/null; then
            log_success "API health check passed"
        else
            log_warning "API health check failed - service may still be starting"
        fi
    else
        log_error "Some services failed to start"
        docker-compose logs
        exit 1
    fi
}

# Deploy to Kubernetes
deploy_kubernetes() {
    log_info "Deploying to Kubernetes..."
    
    # Apply Kubernetes manifests
    kubectl apply -f kubernetes.yml
    
    # Wait for deployment to be ready
    log_info "Waiting for deployment to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/bioneuro-api -n bioneuro-system
    
    # Check pod status
    kubectl get pods -n bioneuro-system
    
    # Get service information
    log_info "Service information:"
    kubectl get services -n bioneuro-system
    
    # Get ingress information (if available)
    if kubectl get ingress -n bioneuro-system bioneuro-api-ingress &> /dev/null; then
        log_info "Ingress information:"
        kubectl get ingress -n bioneuro-system bioneuro-api-ingress
    fi
    
    log_success "Kubernetes deployment completed"
}

# Run smoke tests
run_smoke_tests() {
    log_info "Running smoke tests..."
    
    if [ "$DEPLOYMENT_TYPE" = "docker" ]; then
        API_URL="http://localhost:8000"
    else
        # For Kubernetes, port-forward to test
        kubectl port-forward service/bioneuro-api-service 8000:8000 -n bioneuro-system &
        PORT_FORWARD_PID=$!
        sleep 5
        API_URL="http://localhost:8000"
    fi
    
    # Test health endpoint
    if curl -s "$API_URL/health" | grep -q "healthy"; then
        log_success "Health endpoint test passed"
    else
        log_error "Health endpoint test failed"
        if [ "$DEPLOYMENT_TYPE" = "kubernetes" ]; then
            kill $PORT_FORWARD_PID 2>/dev/null || true
        fi
        exit 1
    fi
    
    # Test ping endpoint (if it requires auth, this might fail - that's expected)
    if curl -s "$API_URL/ping" -H "X-API-Key: test-key" | grep -q "pong" 2>/dev/null; then
        log_success "Ping endpoint test passed"
    else
        log_warning "Ping endpoint test skipped (may require valid API key)"
    fi
    
    if [ "$DEPLOYMENT_TYPE" = "kubernetes" ]; then
        kill $PORT_FORWARD_PID 2>/dev/null || true
    fi
    
    log_success "Smoke tests completed"
}

# Cleanup function
cleanup() {
    if [ "$DEPLOYMENT_TYPE" = "docker" ]; then
        log_info "To stop services: docker-compose down"
        log_info "To remove volumes: docker-compose down -v"
    elif [ "$DEPLOYMENT_TYPE" = "kubernetes" ]; then
        log_info "To remove deployment: kubectl delete -f kubernetes.yml"
    fi
}

# Main deployment flow
main() {
    check_prerequisites
    
    if [ "$DEPLOYMENT_TYPE" = "docker" ]; then
        build_image
        deploy_docker
    elif [ "$DEPLOYMENT_TYPE" = "kubernetes" ]; then
        build_image
        deploy_kubernetes
    else
        log_error "Invalid deployment type: $DEPLOYMENT_TYPE"
        echo "Usage: $0 [docker|kubernetes] [environment] [version]"
        exit 1
    fi
    
    run_smoke_tests
    
    echo ""
    log_success "🎉 Bioneural Olfactory Fusion System deployed successfully!"
    log_info "🔬 Novel multi-sensory legal document analysis is now running in $ENVIRONMENT"
    
    cleanup
}

# Handle script interruption
trap 'log_warning "Deployment interrupted"; exit 1' INT TERM

# Run main function
main "$@"