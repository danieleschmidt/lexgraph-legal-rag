# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records (ADRs) for the LexGraph Legal RAG project. ADRs document important architectural decisions made during the development process.

## Format

Each ADR follows this format:
- **Title**: Short noun phrase describing the decision
- **Status**: Proposed, Accepted, Deprecated, or Superseded
- **Context**: The situation forcing a decision
- **Decision**: The architectural decision and its rationale
- **Consequences**: What becomes easier or more difficult

## Index

- [ADR-001: Multi-Agent Architecture with LangGraph](001-multi-agent-architecture.md)
- [ADR-002: FAISS for Vector Search](002-faiss-vector-search.md)
- [ADR-003: FastAPI Gateway Pattern](003-fastapi-gateway.md)
- [ADR-004: Prometheus Monitoring Stack](004-prometheus-monitoring.md)

## Guidelines

1. Number ADRs sequentially (001, 002, etc.)
2. Use present tense for the decision
3. Keep ADRs immutable - use superseding ADRs for changes
4. Include date and author information
5. Link to related ADRs when applicable