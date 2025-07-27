# LexGraph Legal RAG - Development Makefile

.PHONY: help install test lint format clean dev docker-up docker-down security docs

# Default target
help:
	@echo "LexGraph Legal RAG Development Commands"
	@echo "======================================"
	@echo ""
	@echo "Setup Commands:"
	@echo "  install     Install project and dependencies"
	@echo "  install-dev Install development dependencies"
	@echo ""
	@echo "Development Commands:"
	@echo "  dev         Start development server"
	@echo "  streamlit   Start Streamlit UI"
	@echo "  ingest      Run document ingestion pipeline"
	@echo ""
	@echo "Code Quality Commands:"
	@echo "  test        Run test suite"
	@echo "  test-cov    Run tests with coverage report"
	@echo "  test-fast   Run fast test suite only"
	@echo "  lint        Run code linting"
	@echo "  format      Format code"
	@echo "  typecheck   Run type checking"
	@echo "  security    Run security scans"
	@echo ""
	@echo "Docker Commands:"
	@echo "  docker-build  Build Docker image"
	@echo "  docker-up     Start services with Docker Compose"
	@echo "  docker-down   Stop Docker Compose services"
	@echo "  docker-logs   View Docker logs"
	@echo ""
	@echo "Documentation Commands:"
	@echo "  docs        Generate documentation"
	@echo "  docs-serve  Serve documentation locally"
	@echo ""
	@echo "Utility Commands:"
	@echo "  clean       Clean temporary files"
	@echo "  clean-all   Clean everything including caches"
	@echo "  requirements Update requirements files"

# Installation commands
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pip install pytest pytest-cov pytest-mock pytest-asyncio black ruff mypy bandit safety pre-commit
	pre-commit install

# Development commands
dev:
	@echo "Starting FastAPI development server..."
	uvicorn src.lexgraph_legal_rag.api:app --host 0.0.0.0 --port 8000 --reload

streamlit:
	@echo "Starting Streamlit UI..."
	streamlit run streamlit_app.py --server.port 3000

ingest:
	@echo "Running document ingestion..."
	python ingest.py --docs ./data/legal_docs --index ./data/indices/faiss.index --semantic

# Testing commands
test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=lexgraph_legal_rag --cov-report=html --cov-report=term-missing

test-fast:
	pytest tests/ -m "not slow" -v

test-integration:
	pytest tests/ -m "integration" -v

test-unit:
	pytest tests/ -m "unit" -v

# Code quality commands
lint:
	ruff check .
	black --check .
	mypy src/

format:
	black .
	ruff check --fix .

typecheck:
	mypy src/

security:
	bandit -r src/
	safety check

# Docker commands
docker-build:
	docker build -t lexgraph-legal-rag:latest .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-clean:
	docker-compose down -v --remove-orphans
	docker system prune -f

# Documentation commands
docs:
	@echo "Generating documentation..."
	@echo "Documentation available in docs/ directory"

docs-serve:
	@echo "Serving documentation locally..."
	python -m http.server 8080 -d docs/

# Utility commands
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/

clean-all: clean
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf build/
	rm -rf dist/
	rm -rf .venv/
	rm -rf venv/

requirements:
	pip-compile requirements.in
	pip-compile requirements-dev.in

# CI/CD simulation commands
ci-test:
	make lint
	make typecheck
	make security
	make test-cov

ci-build:
	make clean
	make install
	make ci-test
	make docker-build

# Monitoring commands
metrics:
	@echo "Fetching metrics from http://localhost:8001/metrics"
	curl -s http://localhost:8001/metrics || echo "Metrics server not running"

health:
	@echo "Checking health endpoint..."
	curl -s http://localhost:8000/health || echo "API server not running"

# Development environment
setup-dev: install-dev
	@echo "Setting up development environment..."
	pre-commit install
	mkdir -p data/legal_docs data/indices logs
	cp .env.example .env
	@echo "Development environment ready!"
	@echo "Please update .env file with your API keys"

# Production deployment helpers
prod-check:
	@echo "Running production readiness checks..."
	make security
	make test
	make lint
	@echo "Production checks complete!"

# Database commands (if applicable)
db-migrate:
	@echo "Running database migrations..."
	# Add database migration commands here

db-reset:
	@echo "Resetting database..."
	# Add database reset commands here

# Backup commands
backup-indices:
	@echo "Backing up vector indices..."
	tar -czf backup-indices-$(shell date +%Y%m%d-%H%M%S).tar.gz data/indices/

restore-indices:
	@echo "Restoring vector indices..."
	@echo "Usage: make restore-indices BACKUP=backup-indices-YYYYMMDD-HHMMSS.tar.gz"
	tar -xzf $(BACKUP)

# Performance testing
perf-test:
	@echo "Running performance tests..."
	pytest tests/performance/ -v

load-test:
	@echo "Running load tests..."
	# Add load testing commands here (e.g., locust, artillery)

# Monitoring stack
monitoring-up:
	docker-compose -f monitoring/docker-compose.yml up -d

monitoring-down:
	docker-compose -f monitoring/docker-compose.yml down