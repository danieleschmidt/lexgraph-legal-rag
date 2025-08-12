# Multi-stage production Dockerfile for Bioneural Olfactory Fusion System
FROM python:3.12-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create application user
RUN useradd --create-home --shell /bin/bash bioneuro
WORKDIR /app

# Install Python dependencies
FROM base as dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM dependencies as production

# Copy application code
COPY src/ src/
COPY bioneuro_olfactory_demo.py .
COPY test_bioneuro_minimal.py .

# Set ownership
RUN chown -R bioneuro:bioneuro /app

# Switch to non-root user
USER bioneuro

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import asyncio; from src.lexgraph_legal_rag.bioneuro_olfactory_fusion import get_fusion_engine; print('OK')" || exit 1

# Expose port for API
EXPOSE 8000

# Default command
CMD ["python3", "bioneuro_olfactory_demo.py"]