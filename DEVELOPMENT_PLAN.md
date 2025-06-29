# 🗭 Project Vision

> LexGraph Legal RAG provides a multi-agent retrieval system that indexes legal documents and answers queries with cited passages. It targets legal researchers needing precise references from large corpora.

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
- [ ] **[EPIC] Improve CI stability**
  - [ ] Replace flaky integration tests
  - [x] Enable parallel test execution

### ⚡️ Increment 2: Performance & Observability
- [ ] **[EPIC] Remove N+1 query pattern**
  - [ ] Refactor search pipeline with batch operations
- [ ] **[EPIC] Add metrics dashboard**
  - [ ] Expose Prometheus metrics
  - [ ] Provide Grafana dashboard config

### 👀 Increment 3: Feature Expansion
- [ ] **[EPIC] API versioning & docs**
  - [ ] Implement version negotiation middleware
  - [ ] Update API_USAGE_GUIDE.md
- [ ] **[EPIC] Increase test coverage**
  - [ ] Add integration tests for CLI and Streamlit UI
  - [ ] Achieve >90% coverage threshold

---

# ⚠️ Risks & Mitigation
- Limited legal dataset access → create small synthetic corpus for testing
- Complex CI dependencies → use Docker to ensure consistent environment
- Performance regressions from semantic search → add benchmarks in CI
- Incomplete security reviews → schedule periodic bandit scans

---

# 📊 KPIs & Metrics
- [ ] >85% test coverage
- [ ] <15 min CI pipeline time
- [ ] <5% error rate on core service
- [ ] 100% secrets loaded from vault/env

---

# 👥 Ownership & Roles
- **DevOps:** CI/CD, Docker, monitoring
- **Backend:** API, multi-agent logic
- **QA:** Test automation and coverage
- **Docs:** Usage guides and developer docs
