# Multi-stage Docker build for LexGraph Legal RAG
FROM python:3.11-slim as base

# Set build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

# Add image labels for metadata
LABEL org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="lexgraph-legal-rag" \
      org.label-schema.description="LangGraph-powered multi-agent legal document analysis system" \
      org.label-schema.url="https://github.com/terragon-labs/lexgraph-legal-rag" \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url="https://github.com/terragon-labs/lexgraph-legal-rag" \
      org.label-schema.vendor="Terragon Labs" \
      org.label-schema.version=$VERSION \
      org.label-schema.schema-version="1.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src

# Install system dependencies with security updates
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create application user for security
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml requirements.txt ./

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install -e .

# Development stage
FROM base as development

# Install development dependencies
RUN pip install pytest pytest-cov pytest-asyncio black ruff pre-commit

# Copy source code
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Expose development ports
EXPOSE 8000 8001

# Development command
CMD ["python", "-m", "uvicorn", "lexgraph_legal_rag.api:create_api", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Production stage
FROM base as production

# Copy only necessary files
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser *.py ./
COPY --chown=appuser:appuser pyproject.toml ./

# Install the package
RUN pip install -e . --no-deps

# Switch to non-root user
USER appuser

# Create directories for data
RUN mkdir -p /app/data /app/logs

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose application port
EXPOSE 8000

# Production command
CMD ["python", "-m", "uvicorn", "lexgraph_legal_rag.api:create_api", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]