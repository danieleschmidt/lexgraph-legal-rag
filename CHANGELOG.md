# Release v1.0.1 (2025-07-19)

## 🧪 Testing
- feat(test): comprehensive test coverage improvements (+8% overall coverage)
- fix(test): FastAPI import compatibility for newer versions
- feat(config): test-friendly configuration system with test_mode parameter
- feat(test): added test suites for models, sample, config, and metrics modules

## 📊 Coverage Improvements
- models.py: 100% coverage
- sample.py: 100% coverage (was 67%)
- config.py: 95% coverage (was 27%)
- metrics.py: 88% coverage (was 46%)

## 🔧 Technical Debt
- docs(debt): created TECH_DEBT.md tracking coverage status and priorities
- docs(backlog): updated BACKLOG.md with WSJF prioritization and progress

# Release v1.0.0

## ✨ Features
- feat(api): API key authentication and basic rate limiting
- feat(ui): simple Streamlit interface for querying

# Release v0.2.0

## ✨ Features
- feat(search): add semantic search pipeline
- feat(search): persist semantic search indices
- remove joblib dependency; use JSON persistence
- feat(index): add FAISS-based vector index for scalability
- feat(observability): add Prometheus metrics and structured logging
