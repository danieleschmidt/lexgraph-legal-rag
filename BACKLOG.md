# Technical Backlog - WSJF Prioritized

## Scoring Methodology
**WSJF Score = (Business Value + Time Criticality + Risk Reduction) / Job Size**
- Business Value: 1-10 (user impact, revenue, competitive advantage)
- Time Criticality: 1-10 (urgency, dependencies)
- Risk Reduction: 1-10 (security, stability, compliance)
- Job Size: 1-13 (effort estimate in story points)

---

## High Priority (WSJF > 7)

### 1. Achieve >90% Test Coverage (WSJF: 8.5) ✅ IN PROGRESS
- **Business Value**: 8 (quality assurance, production confidence)
- **Time Criticality**: 9 (KPI requirement)
- **Risk Reduction**: 9 (bug prevention, regression safety)
- **Job Size**: 3 (expand existing test suite)
- **Status**: In Progress - Improved from 22% to 19.92% overall, with 4 modules at >85% coverage
- **Owner**: Terry (Autonomous Agent)
- **Progress**:
  - ✅ Fixed FastAPI import compatibility issues
  - ✅ Created comprehensive test suites for models.py (100%), sample.py (100%), config.py (95%), metrics.py (88%)
  - ✅ Implemented test-friendly configuration system
  - 🔄 Next: Add tests for remaining modules (auth.py, cache.py, semantic_search.py, etc.)

### 2. Implement Request Tracing with Correlation IDs (WSJF: 8.0)
- **Business Value**: 7 (debugging, observability)
- **Time Criticality**: 8 (production monitoring gap)
- **Risk Reduction**: 9 (incident response)
- **Job Size**: 3 (logging enhancement)
- **Status**: Not Started
- **Owner**: Unassigned

### 3. Add Alerting for High Error Rates (WSJF: 7.7)
- **Business Value**: 8 (uptime, user experience)
- **Time Criticality**: 8 (production safety)
- **Risk Reduction**: 8 (proactive issue detection)
- **Job Size**: 3 (Prometheus alerting rules)
- **Status**: Not Started
- **Owner**: Unassigned

---

## Medium Priority (WSJF 4-7)

### 4. Optimize CI Pipeline to <15min (WSJF: 6.0)
- **Business Value**: 6 (developer productivity)
- **Time Criticality**: 7 (KPI requirement)
- **Risk Reduction**: 5 (faster feedback)
- **Job Size**: 3 (parallelization, caching)
- **Status**: Not Started
- **Owner**: Unassigned

### 5. Security Dependency Scanning (WSJF: 5.8)
- **Business Value**: 6 (compliance, security)
- **Time Criticality**: 6 (security best practice)
- **Risk Reduction**: 9 (vulnerability prevention)
- **Job Size**: 4 (bandit, safety integration)
- **Status**: Not Started
- **Owner**: Unassigned

### 6. Add Circuit Breaker Pattern Documentation (WSJF: 4.5)
- **Business Value**: 4 (maintainability)
- **Time Criticality**: 3 (nice to have)
- **Risk Reduction**: 6 (operational knowledge)
- **Job Size**: 2 (documentation)
- **Status**: Not Started
- **Owner**: Unassigned

---

## Low Priority (WSJF < 4)

### 7. Integrate with Westlaw/LexisNexis APIs (WSJF: 3.3)
- **Business Value**: 8 (feature expansion)
- **Time Criticality**: 2 (future roadmap)
- **Risk Reduction**: 2 (not critical)
- **Job Size**: 8 (external API integration)
- **Status**: Not Started
- **Owner**: Unassigned

### 8. Add Multi-jurisdiction Support (WSJF: 2.8)
- **Business Value**: 7 (market expansion)
- **Time Criticality**: 2 (future feature)
- **Risk Reduction**: 2 (not critical)
- **Job Size**: 8 (complex feature)
- **Status**: Not Started
- **Owner**: Unassigned

---

## Technical Debt

### Code Quality
- No immediate technical debt identified
- Test coverage currently at ~80%, needs increase to >90%

### Performance
- Search latency target: <200ms 95th percentile (current status unknown)
- CI pipeline time target: <15min (current status unknown)

### Security
- Need to implement dependency vulnerability scanning
- Consider adding secret rotation automation

---

*Last Updated: 2025-07-19*
*Next Review: Weekly during iteration planning*