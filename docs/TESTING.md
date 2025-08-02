# Testing Guide

This document provides comprehensive guidance on testing practices, infrastructure, and procedures for the LexGraph Legal RAG project.

## Testing Philosophy

Our testing strategy emphasizes:
- **Quality Assurance**: High test coverage with meaningful assertions
- **Test Pyramid**: Unit tests (fast), integration tests (medium), e2e tests (slow)
- **Continuous Testing**: Automated testing on every commit and pull request
- **Performance Validation**: Load testing and benchmarking for production readiness
- **Security Testing**: Vulnerability scanning and penetration testing

## Test Structure

### Directory Organization
```
tests/
├── conftest.py                 # Shared pytest fixtures and configuration
├── fixtures/                   # Test data and mock objects
│   └── documents/             # Sample legal documents for testing
├── performance/               # Performance and load testing
│   ├── benchmark.py          # Performance benchmarks
│   ├── load-test.js          # k6 load testing scripts
│   └── stress-test.js        # k6 stress testing scripts
├── unit/                     # Fast unit tests (implied by test_*.py files)
├── integration/              # Integration tests
└── e2e/                      # End-to-end tests
```

### Test Categories

#### Unit Tests (`test_*.py`)
- **Purpose**: Test individual functions and classes in isolation
- **Speed**: Fast execution (< 100ms per test)
- **Scope**: Single module or function
- **Mocking**: Heavy use of mocks for external dependencies
- **Coverage**: Aim for 90%+ coverage on core business logic

#### Integration Tests (`@pytest.mark.integration`)
- **Purpose**: Test component interactions and external integrations
- **Speed**: Medium execution (100ms - 2s per test)
- **Scope**: Multiple modules working together
- **Dependencies**: Real databases, APIs (with sandboxing)
- **Coverage**: Critical integration paths

#### End-to-End Tests (`@pytest.mark.e2e`)
- **Purpose**: Test complete user workflows and system behavior
- **Speed**: Slow execution (2s+ per test)
- **Scope**: Full system functionality
- **Environment**: Production-like environment
- **Coverage**: User-critical scenarios

#### Performance Tests (`@pytest.mark.performance`)
- **Purpose**: Validate system performance under load
- **Tools**: k6 for load testing, pytest-benchmark for microbenchmarks
- **Metrics**: Response time, throughput, resource utilization
- **Thresholds**: Sub-second response times, 100+ concurrent users

## Testing Infrastructure

### Pytest Configuration

#### Main Configuration (`pyproject.toml`)
```toml
[tool.pytest.ini_options]
addopts = "-ra --cov=lexgraph_legal_rag --cov-branch --cov-fail-under=80"
testpaths = ["tests"]
markers = [
    "unit: Unit tests",
    "integration: Integration tests", 
    "e2e: End-to-end tests",
    "slow: Slow running tests",
    "performance: Performance tests",
    "security: Security tests",
    "smoke: Smoke tests",
]
```

#### CI Configuration (`pytest.ci.ini`)
- Optimized for fast parallel execution
- Coverage reporting with HTML and XML output
- Fail-fast configuration for quick feedback

#### Mutation Testing (`pytest-mutation.ini`)
- Comprehensive test quality assessment
- Validates test effectiveness through code mutation
- Identifies weak test assertions and missing edge cases

### Test Fixtures

#### Shared Fixtures (`conftest.py`)
```python
@pytest.fixture
def api_client():
    """FastAPI test client with authentication."""
    
@pytest.fixture 
def sample_documents():
    """Sample legal documents for testing."""
    
@pytest.fixture
def mock_vector_index():
    """Mocked FAISS vector index."""
```

#### Test Data
- **Sample Documents**: Legal contracts, statutes, regulations
- **Mock Responses**: API responses, embedding vectors
- **Configuration**: Test-specific environment variables

## Running Tests

### Quick Commands

```bash
# Run all tests
npm test

# Run specific test categories
npm run test:unit       # Unit tests only
npm run test:integration # Integration tests only
npm run test:e2e        # End-to-end tests only

# Run tests with coverage
npm run test:coverage

# Run performance tests
npm run test:performance
npm run test:stress

# Run mutation tests
npm run test:mutation
```

### Advanced Testing

#### Parallel Execution
```bash
# Run tests in parallel (auto-detect CPU cores)
pytest tests/ -n auto

# Run tests with specific worker count
pytest tests/ -n 4
```

#### Coverage Analysis
```bash
# Generate HTML coverage report
pytest tests/ --cov=lexgraph_legal_rag --cov-report=html

# Coverage with branch analysis
pytest tests/ --cov=lexgraph_legal_rag --cov-branch

# Fail if coverage below threshold
pytest tests/ --cov-fail-under=80
```

#### Test Selection
```bash
# Run tests by marker
pytest -m unit                    # Unit tests only
pytest -m "not slow"             # Skip slow tests
pytest -m "integration and not slow" # Fast integration tests

# Run tests by pattern
pytest tests/test_api*.py         # API tests only
pytest -k "search"               # Tests with 'search' in name
```

## Performance Testing

### Load Testing with k6

#### Basic Load Test
```javascript
// tests/performance/load-test.js
import http from 'k6/http';
import { check } from 'k6';

export let options = {
  stages: [
    { duration: '5m', target: 100 }, // Ramp up
    { duration: '10m', target: 100 }, // Stay at 100 users
    { duration: '5m', target: 0 }, // Ramp down
  ],
};

export default function() {
  let response = http.post('http://localhost:8000/api/v1/search', {
    query: 'What constitutes indemnification?',
    api_key: 'test-key'
  });
  
  check(response, {
    'status is 200': (r) => r.status === 200,
    'response time < 2s': (r) => r.timings.duration < 2000,
  });
}
```

#### Stress Testing
```bash
# Run stress test
k6 run tests/performance/stress-test.js

# Load test with custom parameters
k6 run --vus 50 --duration 10m tests/performance/load-test.js
```

### Benchmarking with pytest-benchmark

```python
def test_search_performance(benchmark):
    """Benchmark search operation performance."""
    result = benchmark(search_function, "test query")
    assert result is not None
```

## Security Testing

### Automated Security Scanning

#### Dependency Scanning
```bash
# Check for known vulnerabilities
safety check

# Audit Python packages
pip-audit

# Check for secrets in code
detect-secrets scan --all-files
```

#### Static Security Analysis
```bash
# Run Bandit security linter
bandit -r src/

# Generate security report
bandit -r src/ -f json -o security-report.json
```

### Security Test Cases

#### Authentication Tests
- Invalid API key handling
- Rate limiting enforcement
- CORS policy validation
- Input sanitization

#### Data Protection Tests
- No sensitive data in logs
- Secure error handling
- PII data handling

## Continuous Integration

### GitHub Actions Integration

Tests are automatically executed on:
- **Pull Requests**: Full test suite with coverage reporting
- **Main Branch**: Tests + security scanning + performance validation
- **Release**: Complete test suite + mutation testing

### Test Reporting

#### Coverage Reports
- HTML coverage reports generated in `htmlcov/`
- XML coverage for CI integration
- Coverage badges in README

#### Test Results
- JUnit XML output for CI integration
- Test timing and performance metrics
- Failure analysis and debugging information

## Quality Gates

### Required Test Coverage
- **Overall Coverage**: 80% minimum
- **Branch Coverage**: 70% minimum
- **Critical Modules**: 90% minimum (api.py, multi_agent.py)

### Performance Thresholds
- **API Response Time**: 95th percentile < 1 second
- **Search Latency**: < 500ms for vector similarity search
- **Concurrent Users**: Support 100+ simultaneous users

### Security Requirements
- **Vulnerability Scanning**: Zero high/critical vulnerabilities
- **Secret Detection**: No hardcoded secrets or keys
- **Dependency Audit**: All dependencies up-to-date

## Best Practices

### Writing Effective Tests

#### Test Naming
```python
def test_search_returns_relevant_results_for_legal_query():
    """Test names should be descriptive and behavior-focused."""
    
def test_api_returns_401_when_invalid_api_key_provided():
    """Include expected behavior and conditions."""
```

#### Test Structure (AAA Pattern)
```python
def test_vector_search_with_empty_query():
    # Arrange
    search_engine = VectorSearchEngine()
    empty_query = ""
    
    # Act
    result = search_engine.search(empty_query)
    
    # Assert
    assert result.status == "error"
    assert "query cannot be empty" in result.message
```

#### Mocking External Dependencies
```python
@pytest.fixture
def mock_openai_client():
    with patch('openai.Client') as mock:
        mock.return_value.embeddings.create.return_value = mock_embedding_response
        yield mock

def test_embedding_generation_with_mock(mock_openai_client):
    # Test uses mocked OpenAI client
    embeddings = generate_embeddings("test text")
    assert len(embeddings) == 768
```

### Test Maintenance

#### Regular Review
- Review test effectiveness through mutation testing
- Update tests when requirements change
- Remove obsolete or redundant tests
- Refactor tests to improve maintainability

#### Documentation
- Document complex test scenarios
- Maintain test data documentation
- Update testing guides with new practices

## Debugging Tests

### Common Issues

#### Flaky Tests
- Use deterministic test data
- Avoid time-dependent assertions
- Mock external services properly
- Use appropriate timeouts

#### Slow Tests
- Profile test execution times
- Optimize database operations
- Use appropriate test markers
- Consider parallel execution

#### Coverage Gaps
- Identify untested code paths
- Add tests for edge cases
- Test error conditions
- Validate integration points

### Debugging Tools

```bash
# Run tests with verbose output
pytest tests/ -v -s

# Drop into debugger on failure
pytest tests/ --pdb

# Show test execution times
pytest tests/ --durations=10

# Run specific failing test
pytest tests/test_api.py::test_specific_function -v -s
```

## Contributing to Tests

### Test Requirements for PRs
- All new code must have corresponding tests
- Test coverage must not decrease
- All tests must pass before merging
- Performance tests for performance-critical changes

### Test Review Checklist
- [ ] Tests are comprehensive and cover edge cases
- [ ] Test names are descriptive and clear
- [ ] Mocks are used appropriately for external dependencies
- [ ] Tests are fast and don't introduce flakiness
- [ ] Security implications are tested where applicable

This testing guide ensures comprehensive quality assurance for the LexGraph Legal RAG system through systematic testing practices and infrastructure.