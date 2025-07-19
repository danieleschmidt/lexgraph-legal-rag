# 🗭 Project Vision

> LexGraph Legal RAG provides a multi-agent retrieval system that indexes legal documents and answers queries with cited passages. It targets legal researchers needing precise references from large corpora.

## Technical Stack
- **Backend**: Python 3.8+, FastAPI, Streamlit
- **Search**: FAISS vector database, semantic embeddings
- **Architecture**: Multi-agent system with LangGraph
- **Monitoring**: Prometheus metrics, structured logging
- **Testing**: pytest with 80% coverage requirement

---

# 📅 12-Week Roadmap

## Increment I1 (Weeks 1–4)
- **Themes:** Security, Developer UX
- **Goals / Epics**
  - Harden API authentication and rate limiting
  - Set up pre-commit with ruff/black and secret scanning
  - Improve CI reliability
- **Definition of Done**
  - API rejects invalid keys and logs attempts
  - Pre-commit hooks run locally and on CI
  - CI pipeline green on main for three consecutive runs

## Increment I2 (Weeks 5–8)
- **Themes:** Performance, Observability
- **Goals / Epics**
  - Optimize document pipeline to avoid N+1 searches
  - Add structured logging and basic metrics dashboard
  - Containerize application for staging
- **Definition of Done**
  - Search latency <200ms for 95th percentile on test data
  - Metrics exposed via Prometheus and visualized in Grafana
  - Docker image builds and runs tests in CI

## Increment I3 (Weeks 9–12)
- **Themes:** Feature Expansion, Documentation
- **Goals / Epics**
  - Enable semantic versioning across APIs
  - Expand test coverage to >90%
  - Publish user guide and contribution docs
- **Definition of Done**
  - Versioned routes documented and validated by tests
  - Coverage report shows >90%
  - Docs deployed and linked from README

---

# ✅ Epic & Task Checklist

### 🔒 Increment 1: Security & Refactoring
- [ ] **[EPIC] Eliminate hardcoded secrets**
  - [x] Load from environment securely
  - [x] Add pre-commit hook for scanning secrets
  - [ ] Implement API key rotation mechanism
  - [ ] Add secret validation in startup checks
- [ ] **[EPIC] Improve CI stability**
  - [ ] Replace flaky integration tests
  - [x] Enable parallel test execution
  - [ ] Add retry logic for external service calls
  - [ ] Implement health check endpoints

### ⚡️ Increment 2: Performance & Observability
- [ ] **[EPIC] Remove N+1 query pattern**
  - [ ] Refactor search pipeline with batch operations
  - [ ] Implement connection pooling for FAISS
  - [ ] Add query result caching layer
  - [ ] Optimize vector similarity calculations
- [ ] **[EPIC] Add metrics dashboard**
  - [ ] Expose Prometheus metrics
  - [ ] Provide Grafana dashboard config
  - [ ] Add request tracing with correlation IDs
  - [ ] Implement alerting for high error rates

### 👀 Increment 3: Feature Expansion
- [ ] **[EPIC] API versioning & docs**
  - [ ] Implement version negotiation middleware
  - [ ] Update API_USAGE_GUIDE.md
  - [ ] Add OpenAPI schema generation
  - [ ] Create interactive API documentation
- [ ] **[EPIC] Increase test coverage**
  - [ ] Add integration tests for CLI and Streamlit UI
  - [ ] Achieve >90% coverage threshold
  - [ ] Add end-to-end testing scenarios
  - [ ] Implement property-based testing for core logic

---

# ⚠️ Risks & Mitigation
- Limited legal dataset access → create small synthetic corpus for testing
- Complex CI dependencies → use Docker to ensure consistent environment
- Performance regressions from semantic search → add benchmarks in CI
- Incomplete security reviews → schedule periodic bandit scans

---

# 🔧 Technical Implementation Details

## Core Architecture Components
1. **Multi-Agent System** (`src/lexgraph_legal_rag/multi_agent.py`)
   - Router Agent: Query classification and routing
   - Retriever Agent: Document search and retrieval
   - Summarizer Agent: Content summarization
   - Citation Agent: Source citation generation

2. **Document Pipeline** (`src/lexgraph_legal_rag/document_pipeline.py`)
   - Text preprocessing and chunking
   - Vector embedding generation
   - FAISS index management

3. **API Layer** (`src/lexgraph_legal_rag/api.py`)
   - FastAPI endpoints with authentication
   - Rate limiting and request validation
   - Structured response formatting

## Key Dependencies
- **faiss-cpu**: Vector similarity search
- **scikit-learn**: Text processing and ML utilities
- **fastapi**: Web API framework
- **streamlit**: Interactive web interface
- **structlog**: Structured logging
- **prometheus-client**: Metrics collection

---

# 📊 KPIs & Metrics
- [ ] >90% test coverage (increased from 85%)
- [ ] <15 min CI pipeline time
- [ ] <5% error rate on core service
- [ ] 100% secrets loaded from vault/env
- [ ] <200ms 95th percentile query latency
- [ ] Zero security vulnerabilities in dependencies

---

# 👥 Ownership & Roles
- **DevOps:** CI/CD, Docker, monitoring, deployment
- **Backend:** API, multi-agent logic, performance optimization
- **QA:** Test automation, coverage, security testing
- **Docs:** Usage guides, API documentation, developer docs
- **Security:** Vulnerability scanning, secret management, compliance
