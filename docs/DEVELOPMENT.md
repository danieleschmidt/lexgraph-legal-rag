# LexGraph Legal RAG - Development Guide

This guide provides comprehensive instructions for setting up and contributing to the LexGraph Legal RAG project.

## Table of Contents

- [Quick Start](#quick-start)
- [Development Environment](#development-environment)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Testing](#testing)
- [Code Quality](#code-quality)
- [Documentation](#documentation)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)

## Quick Start

### Prerequisites

- **Python 3.8+** (recommended: 3.11)
- **Git**
- **Docker** (optional, for containerized development)
- **VS Code** (recommended) with Dev Containers extension

### Option 1: DevContainer (Recommended)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/terragon-labs/lexgraph-legal-rag.git
   cd lexgraph-legal-rag
   ```

2. **Open in VS Code:**
   ```bash
   code .
   ```

3. **Reopen in Container:**
   - Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
   - Select "Dev Containers: Reopen in Container"
   - Wait for the container to build and start

4. **Start development:**
   ```bash
   make dev
   ```

### Option 2: Local Development

1. **Clone and setup:**
   ```bash
   git clone https://github.com/terragon-labs/lexgraph-legal-rag.git
   cd lexgraph-legal-rag
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   make setup-dev
   ```

4. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

5. **Start development server:**
   ```bash
   make dev
   ```

## Development Environment

### Environment Variables

Create a `.env` file with the following variables:

```env
# Required
API_KEY=your-development-api-key
OPENAI_API_KEY=sk-your-openai-key-here

# Optional
ENVIRONMENT=development
DEBUG=true
METRICS_PORT=8001
LOG_LEVEL=INFO
```

### IDE Configuration

The project includes comprehensive IDE configuration:

- **VS Code**: Settings in `.vscode/`
- **Pre-commit hooks**: Automated code quality checks
- **Debugger configurations**: Ready-to-use debug setups

### Available Commands

```bash
# Development
make dev              # Start development server
make streamlit        # Start Streamlit UI
make test             # Run test suite
make lint             # Run code quality checks
make format           # Format code

# Docker
make docker-up        # Start with Docker Compose
make docker-down      # Stop Docker services
make docker-build     # Build Docker image

# Utilities
make clean            # Clean temporary files
make requirements     # Update requirements files
```

## Project Structure

```
lexgraph-legal-rag/
├── .devcontainer/          # DevContainer configuration
├── .github/               # GitHub workflows and templates
├── .vscode/               # VS Code settings
├── docs/                  # Documentation
│   ├── adr/              # Architecture Decision Records
│   └── runbooks/         # Operational runbooks
├── k8s/                  # Kubernetes configurations
├── monitoring/           # Monitoring and observability
├── src/                  # Source code
│   └── lexgraph_legal_rag/
├── tests/                # Test suite
│   ├── fixtures/         # Test data
│   └── performance/      # Performance tests
├── ARCHITECTURE.md       # System architecture
├── ROADMAP.md           # Product roadmap
├── Makefile             # Development commands
└── pyproject.toml       # Python project configuration
```

## Development Workflow

### 1. Feature Development

1. **Create feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Implement feature:**
   - Write code following project conventions
   - Add tests for new functionality
   - Update documentation if needed

3. **Test locally:**
   ```bash
   make test
   make lint
   ```

4. **Commit changes:**
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

5. **Push and create PR:**
   ```bash
   git push origin feature/your-feature-name
   # Create PR through GitHub UI
   ```

### 2. Code Review Process

- **All PRs require review** from at least one maintainer
- **CI checks must pass** before merging
- **Test coverage** should not decrease
- **Documentation** should be updated for user-facing changes

### 3. Merge Process

- Use **squash and merge** for clean history
- **Delete feature branch** after merging
- **Semantic commit messages** trigger automated releases

## Testing

### Test Categories

- **Unit Tests**: Fast, isolated component tests
- **Integration Tests**: Component interaction tests
- **Performance Tests**: Load and benchmark tests
- **Security Tests**: Vulnerability and penetration tests

### Running Tests

```bash
# All tests
make test

# Specific test types
pytest tests/ -m "unit"
pytest tests/ -m "integration"
pytest tests/ -m "performance"

# With coverage
make test-cov

# Fast tests only (skip slow tests)
make test-fast
```

### Writing Tests

- Place tests in `tests/` directory
- Use descriptive test names
- Follow AAA pattern (Arrange, Act, Assert)
- Mock external dependencies
- Use fixtures for test data

Example:
```python
def test_document_processing_success():
    # Arrange
    processor = DocumentProcessor()
    sample_document = "legal contract text..."
    
    # Act
    result = processor.process(sample_document)
    
    # Assert
    assert result.status == "success"
    assert len(result.chunks) > 0
```

## Code Quality

### Standards

- **Python Style**: Black formatter (88 character line length)
- **Linting**: Ruff for fast Python linting
- **Type Checking**: mypy for static type analysis
- **Security**: Bandit for security issue detection

### Pre-commit Hooks

Automatically run on each commit:
- Code formatting (Black)
- Linting (Ruff)
- Secret detection
- Import sorting

### Manual Quality Checks

```bash
make lint      # Run all quality checks
make format    # Auto-format code
make security  # Security scan
```

## Documentation

### Types of Documentation

1. **Code Documentation**: Docstrings and inline comments
2. **API Documentation**: OpenAPI/Swagger specs
3. **Architecture Documentation**: System design and ADRs
4. **User Documentation**: README and guides
5. **Operational Documentation**: Runbooks and troubleshooting

### Writing Guidelines

- **Clear and concise** language
- **Code examples** for complex concepts
- **Up-to-date** with current implementation
- **Searchable** and well-organized

## Deployment

### Staging Deployment

```bash
# Deploy to staging (automatic on main branch)
git push origin main
```

### Production Deployment

```bash
# Create release (triggers production deployment)
git tag v1.0.0
git push origin v1.0.0
```

### Manual Deployment

```bash
# Build and deploy manually
make docker-build
docker tag lexgraph-legal-rag:latest your-registry/lexgraph-legal-rag:v1.0.0
docker push your-registry/lexgraph-legal-rag:v1.0.0
```

## Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Ensure PYTHONPATH is set
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

#### 2. Docker Issues
```bash
# Clean Docker cache
make docker-clean
docker system prune -f
```

#### 3. Test Failures
```bash
# Run specific failing test
pytest tests/test_specific.py::test_function_name -v -s
```

#### 4. Pre-commit Hook Failures
```bash
# Run pre-commit manually
pre-commit run --all-files
```

### Getting Help

1. **Check documentation** in `docs/` directory
2. **Search existing issues** on GitHub
3. **Ask in discussions** for general questions
4. **Create an issue** for bugs or feature requests

### Performance Debugging

```bash
# Profile application performance
python -m cProfile -o profile.stats your_script.py
python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"
```

### Memory Debugging

```bash
# Monitor memory usage
python -m memory_profiler your_script.py
```

## Contributing Guidelines

1. **Follow the code style** and conventions
2. **Write tests** for new functionality
3. **Update documentation** when needed
4. **Use semantic commit messages**
5. **Keep PRs focused** and atomic
6. **Respond to review feedback** promptly

### Commit Message Format

```
type(scope): description

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Example:
```
feat(api): add support for document batch processing

Add new endpoint /api/v1/documents/batch that allows
uploading multiple documents for processing.

Closes #123
```

## Development Best Practices

- **Test-driven development** when possible
- **Small, frequent commits** over large changes
- **Meaningful commit messages** for project history
- **Code reviews** to maintain quality
- **Continuous integration** to catch issues early
- **Documentation** as part of development process

## Resources

- [Architecture Documentation](ARCHITECTURE.md)
- [API Documentation](API_USAGE_GUIDE.md)
- [Contributing Guidelines](CONTRIBUTING.md)
- [Security Policy](SECURITY.md)
- [Troubleshooting Runbooks](docs/runbooks/)

---

For additional help or questions, please:
- 📖 Check the [documentation](docs/)
- 💬 Start a [discussion](https://github.com/terragon-labs/lexgraph-legal-rag/discussions)
- 🐛 Report [issues](https://github.com/terragon-labs/lexgraph-legal-rag/issues)
- 📧 Email [support@terragon.ai](mailto:support@terragon.ai)