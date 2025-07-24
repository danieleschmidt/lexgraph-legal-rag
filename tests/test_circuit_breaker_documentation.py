"""Tests for circuit breaker pattern documentation requirements."""

import os
from pathlib import Path


def test_circuit_breaker_documentation_exists():
    """Test that circuit breaker pattern documentation exists."""
    doc_path = Path("docs/CIRCUIT_BREAKER_PATTERN.md")
    assert doc_path.exists(), "Circuit breaker pattern documentation must exist"


def test_circuit_breaker_documentation_completeness():
    """Test that circuit breaker documentation covers all required topics."""
    doc_path = Path("docs/CIRCUIT_BREAKER_PATTERN.md")
    
    if not doc_path.exists():
        assert False, "Documentation file does not exist"
    
    content = doc_path.read_text()
    
    # Required sections
    required_sections = [
        "# Circuit Breaker Pattern",
        "## Overview",
        "## Implementation",
        "## Configuration",
        "## States and Transitions",
        "## Usage Examples",
        "## Monitoring and Observability",
        "## Testing",
        "## Best Practices"
    ]
    
    for section in required_sections:
        assert section in content, f"Required section '{section}' missing from documentation"


def test_circuit_breaker_code_examples():
    """Test that documentation includes practical code examples."""
    doc_path = Path("docs/CIRCUIT_BREAKER_PATTERN.md")
    
    if not doc_path.exists():
        assert False, "Documentation file does not exist"
    
    content = doc_path.read_text()
    
    # Should include code examples
    assert "```python" in content, "Documentation should include Python code examples"
    assert "ResilientHTTPClient" in content, "Should reference the actual implementation"
    assert "CircuitState" in content, "Should document circuit states"


def test_circuit_breaker_configuration_documented():
    """Test that configuration options are properly documented."""
    doc_path = Path("docs/CIRCUIT_BREAKER_PATTERN.md")
    
    if not doc_path.exists():
        assert False, "Documentation file does not exist"
    
    content = doc_path.read_text()
    
    # Configuration parameters should be documented
    config_params = [
        "failure_threshold",
        "recovery_timeout", 
        "success_threshold",
        "max_retries",
        "exponential_base"
    ]
    
    for param in config_params:
        assert param in content, f"Configuration parameter '{param}' should be documented"