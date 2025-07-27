"""
Global pytest configuration for LexGraph Legal RAG
"""
import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient

# Set test environment variables
os.environ["ENVIRONMENT"] = "test"
os.environ["DEBUG"] = "false"
os.environ["API_KEY"] = "test-api-key"


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Provide path to test data directory."""
    return Path(__file__).parent / "tests" / "fixtures"


@pytest.fixture(scope="session")
def temp_index_dir() -> Generator[Path, None, None]:
    """Provide temporary directory for test indices."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def test_documents(test_data_dir: Path) -> list[str]:
    """Provide sample legal documents for testing."""
    return [
        "Sample commercial contract with indemnification clause.",
        "Employment agreement with non-compete provisions.",
        "Privacy policy with data protection requirements.",
        "Terms of service with limitation of liability.",
    ]


@pytest.fixture
def api_client() -> TestClient:
    """Provide FastAPI test client."""
    from src.lexgraph_legal_rag.api import app
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment variables after each test."""
    yield
    # Reset any environment variables that might have been changed during tests
    os.environ["ENVIRONMENT"] = "test"


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on file paths."""
    for item in items:
        # Add integration marker for integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Add unit marker for unit tests
        elif "unit" in str(item.fspath) or "test_" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        
        # Add e2e marker for end-to-end tests
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        
        # Add slow marker for performance tests
        if "performance" in str(item.fspath) or "load" in str(item.fspath):
            item.add_marker(pytest.mark.slow)