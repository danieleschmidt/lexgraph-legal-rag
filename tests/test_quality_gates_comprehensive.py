"""Comprehensive tests for quality_gates.py module.

This module provides extensive test coverage for the QualityGateValidator class
and all quality gate validation functionality to improve overall test coverage.
"""

import asyncio
import json
import os
import subprocess
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from typing import Dict, Any

from lexgraph_legal_rag.quality_gates import (
    QualityGateValidator,
    QualityGateResult,
    QualityGateStatus
)


class TestQualityGateValidator:
    """Test suite for QualityGateValidator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.src_path = self.temp_dir / "src"
        self.tests_path = self.temp_dir / "tests"
        self.src_path.mkdir()
        self.tests_path.mkdir()
        
        # Create validator instance
        self.validator = QualityGateValidator(self.temp_dir)
        
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init_with_valid_project_root(self):
        """Test QualityGateValidator initialization with valid project root."""
        validator = QualityGateValidator(self.temp_dir)
        assert validator.project_root == self.temp_dir
        assert validator.src_path == self.temp_dir / "src"
        assert validator.tests_path == self.temp_dir / "tests"
        
    def test_init_with_invalid_project_root(self):
        """Test QualityGateValidator initialization with invalid project root."""
        # The validator initializes even with invalid paths
        # This tests that it doesn't crash on initialization
        validator = QualityGateValidator(Path("/nonexistent/path"))
        assert validator.project_root == Path("/nonexistent/path")
    
    @pytest.mark.asyncio
    async def test_run_command_success(self):
        """Test successful command execution."""
        with patch('subprocess.run') as mock_run:
            # Mock successful subprocess execution
            mock_run.return_value = Mock(
                returncode=0,
                stdout="success output",
                stderr=""
            )
            
            result = await self.validator._run_command(["echo", "test"])
            
            assert result.returncode == 0
            assert result.stdout == "success output"
            mock_run.assert_called_once()
        
    @pytest.mark.asyncio
    async def test_run_command_failure(self):
        """Test failed command execution."""
        with patch('subprocess.run') as mock_run:
            # Mock failed subprocess execution
            mock_run.return_value = Mock(
                returncode=1,
                stdout="",
                stderr="command failed"
            )
            
            result = await self.validator._run_command(["false"])
            
            assert result.returncode == 1
            assert result.stderr == "command failed"
        
    @pytest.mark.asyncio
    async def test_run_command_timeout(self):
        """Test command execution timeout."""
        with patch('subprocess.run') as mock_run:
            # Mock subprocess timeout
            mock_run.side_effect = subprocess.TimeoutExpired(
                cmd=["sleep", "10"], 
                timeout=5
            )
            
            result = await self.validator._run_command(["sleep", "10"])
            
            # The implementation should handle timeouts gracefully
            assert result.returncode != 0
        
    @pytest.mark.asyncio
    async def test_check_code_format_success(self):
        """Test successful code format validation."""
        # Create test Python file
        test_file = self.src_path / "test_module.py"
        test_file.write_text('def hello():\n    return "world"\n')
        
        with patch('subprocess.run') as mock_run:
            # Mock successful black and ruff execution
            mock_run.return_value = Mock(returncode=0, stdout="All done! ✨ 🍰 ✨", stderr="")
            
            result = await self.validator._check_code_format()
            
            assert result.status == QualityGateStatus.PASS
            assert "format" in result.message.lower() or "passed" in result.message.lower()
        
    @patch('subprocess.run')
    def test_check_code_format_issues(self, mock_run):
        """Test code format validation with formatting issues."""
        # Create test Python file
        test_file = self.src_path / "test_module.py"
        test_file.write_text('def hello( ):\n  return"world"')
        
        # Mock black execution with formatting issues
        mock_run.return_value = Mock(
            returncode=1, 
            stdout="would reformat test_module.py", 
            stderr=""
        )
        
        result = self.validator._check_code_format()
        
        assert result.status == QualityGateStatus.FAIL
        assert "formatting issues" in result.message.lower()
        
    @patch('subprocess.run')
    def test_check_linting_success(self, mock_run):
        """Test successful linting validation."""
        # Create test Python file
        test_file = self.src_path / "test_module.py"
        test_file.write_text('"""Test module."""\n\ndef hello() -> str:\n    """Return greeting."""\n    return "world"\n')
        
        # Mock successful ruff execution
        mock_run.return_value = Mock(returncode=0, stdout="All checks passed!", stderr="")
        
        result = self.validator._check_linting()
        
        assert result.status == QualityGateStatus.PASS
        assert "linting" in result.message.lower()
        
    @patch('subprocess.run')
    def test_check_linting_issues(self, mock_run):
        """Test linting validation with lint issues."""
        # Create test Python file with lint issues
        test_file = self.src_path / "test_module.py"
        test_file.write_text('import unused_module\ndef hello():\n    pass')
        
        # Mock ruff execution with lint issues
        mock_run.return_value = Mock(
            returncode=1,
            stdout="test_module.py:1:1: F401 'unused_module' imported but unused",
            stderr=""
        )
        
        result = self.validator._check_linting()
        
        assert result.status == QualityGateStatus.FAIL
        assert "linting issues found" in result.message.lower()
        
    @patch('subprocess.run')
    def test_check_type_checking_success(self, mock_run):
        """Test successful type checking validation."""
        # Create test Python file with proper typing
        test_file = self.src_path / "test_module.py"
        test_file.write_text('def add(a: int, b: int) -> int:\n    return a + b\n')
        
        # Mock successful mypy execution
        mock_run.return_value = Mock(returncode=0, stdout="Success: no issues found", stderr="")
        
        result = self.validator._check_type_checking()
        
        assert result.status == QualityGateStatus.PASS
        assert "type checking" in result.message.lower()
        
    @patch('subprocess.run')
    def test_check_type_checking_issues(self, mock_run):
        """Test type checking validation with type errors."""
        # Create test Python file with type issues
        test_file = self.src_path / "test_module.py"
        test_file.write_text('def add(a, b):\n    return a + "invalid"\n')
        
        # Mock mypy execution with type errors
        mock_run.return_value = Mock(
            returncode=1,
            stdout="test_module.py:2: error: Unsupported operand types",
            stderr=""
        )
        
        result = self.validator._check_type_checking()
        
        assert result.status == QualityGateStatus.FAIL
        assert "type checking issues" in result.message.lower()
        
    @patch('subprocess.run')
    def test_check_security_scan_success(self, mock_run):
        """Test successful security scan validation."""
        # Create test Python file
        test_file = self.src_path / "test_module.py"
        test_file.write_text('import hashlib\n\ndef secure_hash(data: str) -> str:\n    return hashlib.sha256(data.encode()).hexdigest()\n')
        
        # Mock successful bandit execution
        mock_run.return_value = Mock(
            returncode=0,
            stdout='{"results": [], "metrics": {"_totals": {"nosec": 0, "skipped_tests": 0}}}',
            stderr=""
        )
        
        result = self.validator._check_security()
        
        assert result.status == QualityGateStatus.PASS
        assert "security scan" in result.message.lower()
        
    @patch('subprocess.run')
    def test_check_security_scan_issues(self, mock_run):
        """Test security scan validation with security issues."""
        # Create test Python file with security issues
        test_file = self.src_path / "test_module.py"
        test_file.write_text('import subprocess\n\ndef run_command(cmd):\n    subprocess.call(cmd, shell=True)\n')
        
        # Mock bandit execution with security issues
        security_report = {
            "results": [
                {
                    "filename": "test_module.py",
                    "issue_severity": "HIGH",
                    "issue_confidence": "HIGH",
                    "test_id": "B602",
                    "test_name": "subprocess_popen_with_shell_equals_true"
                }
            ],
            "metrics": {"_totals": {"nosec": 0, "skipped_tests": 0}}
        }
        
        mock_run.return_value = Mock(
            returncode=1,
            stdout=json.dumps(security_report),
            stderr=""
        )
        
        result = self.validator._check_security()
        
        assert result.status == QualityGateStatus.FAIL
        assert "security issues found" in result.message.lower()
        
    @patch('subprocess.run')
    def test_check_test_coverage_success(self, mock_run):
        """Test successful test coverage validation."""
        # Create test files
        test_file = self.tests_path / "test_module.py"
        test_file.write_text('import pytest\n\ndef test_example():\n    assert True\n')
        
        # Mock successful pytest coverage execution
        mock_run.return_value = Mock(
            returncode=0,
            stdout="TOTAL 100 0 100%",
            stderr=""
        )
        
        result = self.validator._check_test_coverage()
        
        assert result.status == QualityGateStatus.PASS
        assert "coverage" in result.message.lower()
        
    @patch('subprocess.run')
    def test_check_test_coverage_low(self, mock_run):
        """Test test coverage validation with low coverage."""
        # Create test files
        test_file = self.tests_path / "test_module.py"
        test_file.write_text('def test_example():\n    pass\n')
        
        # Mock pytest coverage execution with low coverage
        mock_run.return_value = Mock(
            returncode=2,
            stdout="TOTAL 100 60 40%",
            stderr="Coverage failure: total coverage 40% is less than 80%"
        )
        
        result = self.validator._check_test_coverage()
        
        assert result.status == QualityGateStatus.FAIL
        assert "coverage" in result.message.lower()
        assert "40%" in result.message
        
    def test_check_file_structure_success(self):
        """Test successful file structure validation."""
        # Create required files
        (self.temp_dir / "README.md").touch()
        (self.temp_dir / "pyproject.toml").touch()
        (self.temp_dir / "requirements.txt").touch()
        (self.src_path / "__init__.py").touch()
        (self.tests_path / "__init__.py").touch()
        
        result = self.validator._check_file_structure()
        
        assert result.status == QualityGateStatus.PASS
        assert "file structure" in result.message.lower()
        
    def test_check_file_structure_missing_files(self):
        """Test file structure validation with missing files."""
        # Don't create required files
        
        result = self.validator._check_file_structure()
        
        assert result.status == QualityGateStatus.FAIL
        assert "missing files" in result.message.lower()
        
    @pytest.mark.asyncio
    async def test_run_all_gates_success(self):
        """Test running all quality gates successfully."""
        # Setup required files
        (self.temp_dir / "README.md").touch()
        (self.temp_dir / "pyproject.toml").touch()
        (self.temp_dir / "requirements.txt").touch()
        (self.src_path / "__init__.py").touch()
        (self.tests_path / "__init__.py").touch()
        
        # Create a simple Python file
        test_file = self.src_path / "test_module.py"
        test_file.write_text('"""Test module."""\n\ndef hello() -> str:\n    """Return greeting."""\n    return "world"\n')
        
        with patch('subprocess.run') as mock_run:
            # Mock all subprocess calls to succeed
            mock_run.return_value = Mock(
                returncode=0,
                stdout='{"results": [], "metrics": {"_totals": {"nosec": 0}}}',
                stderr=""
            )
            
            results = await self.validator.run_all_gates()
            
            assert len(results) > 0
            # At least file structure should pass
            assert any(r.status == QualityGateStatus.PASS for r in results.values())
            
    @pytest.mark.asyncio
    async def test_run_all_gates_parallel_execution(self):
        """Test that quality gates run in parallel."""
        # Setup minimal requirements
        (self.temp_dir / "README.md").touch()
        (self.src_path / "__init__.py").touch()
        
        with patch('subprocess.run') as mock_run:
            # Mock subprocess calls with delays to test parallelism
            def slow_subprocess(*args, **kwargs):
                import time
                time.sleep(0.1)  # Small delay
                return Mock(returncode=0, stdout="success", stderr="")
            
            mock_run.side_effect = slow_subprocess
            
            import time
            start_time = time.time()
            results = await self.validator.run_all_gates()
            end_time = time.time()
            
            # Should complete faster than sequential execution would take
            # (This is a rough test - in practice parallel execution should be faster)
            assert end_time - start_time < 2.0  # Generous threshold
            assert len(results) > 0
            
    def test_quality_gate_result_creation(self):
        """Test QualityGateResult creation and attributes."""
        result = QualityGateResult(
            name="test_gate",
            status=QualityGateStatus.PASS,
            message="Test passed",
            details={"key": "value"}
        )
        
        assert result.name == "test_gate"
        assert result.status == QualityGateStatus.PASS
        assert result.message == "Test passed"
        assert result.details == {"key": "value"}
        assert isinstance(result.duration_ms, float)
        assert isinstance(result.timestamp, float)
        
    def test_quality_gate_result_to_dict(self):
        """Test QualityGateResult dictionary conversion."""
        result = QualityGateResult(
            name="test_gate",
            status=QualityGateStatus.FAIL,
            message="Test failed",
            details={"error": "Something went wrong"}
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["name"] == "test_gate"
        assert result_dict["status"] == "FAIL"
        assert result_dict["message"] == "Test failed"
        assert result_dict["details"] == {"error": "Something went wrong"}
        assert "duration_ms" in result_dict
        assert "timestamp" in result_dict
        
    def test_validation_error_creation(self):
        """Test ValidationError exception creation."""
        error = ValidationError("Test validation error")
        assert str(error) == "Test validation error"
        assert isinstance(error, Exception)


class TestQualityGateIntegration:
    """Integration tests for quality gate system."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.validator = QualityGateValidator(self.temp_dir)
        
    def teardown_method(self):
        """Clean up integration test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    @pytest.mark.asyncio
    async def test_end_to_end_quality_validation(self):
        """Test complete quality validation workflow."""
        # Setup a realistic project structure
        src_path = self.temp_dir / "src" / "myproject"
        tests_path = self.temp_dir / "tests"
        src_path.mkdir(parents=True)
        tests_path.mkdir(parents=True)
        
        # Create project files
        (self.temp_dir / "README.md").write_text("# My Project\n\nA test project.")
        (self.temp_dir / "pyproject.toml").write_text("[build-system]\nrequires = ['setuptools']\n")
        (self.temp_dir / "requirements.txt").write_text("requests==2.28.0\n")
        
        # Create source code
        (src_path / "__init__.py").write_text('"""My project package."""\n__version__ = "1.0.0"\n')
        (src_path / "main.py").write_text('''
"""Main module."""

def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"

def main() -> None:
    """Main entry point."""
    print(greet("World"))

if __name__ == "__main__":
    main()
''')
        
        # Create tests
        (tests_path / "__init__.py").touch()
        (tests_path / "test_main.py").write_text('''
"""Tests for main module."""

import pytest
from src.myproject.main import greet

def test_greet():
    """Test greet function."""
    assert greet("Alice") == "Hello, Alice!"

def test_greet_empty():
    """Test greet with empty string."""
    assert greet("") == "Hello, !"
''')
        
        # Mock all external tools to pass
        with patch('subprocess.run') as mock_run:
            def mock_subprocess(cmd, *args, **kwargs):
                cmd_str = ' '.join(cmd) if isinstance(cmd, list) else str(cmd)
                
                if 'black' in cmd_str:
                    return Mock(returncode=0, stdout="All done! ✨ 🍰 ✨", stderr="")
                elif 'ruff' in cmd_str:
                    return Mock(returncode=0, stdout="All checks passed!", stderr="")
                elif 'mypy' in cmd_str:
                    return Mock(returncode=0, stdout="Success: no issues found", stderr="")
                elif 'bandit' in cmd_str:
                    return Mock(returncode=0, stdout='{"results": [], "metrics": {"_totals": {"nosec": 0}}}', stderr="")
                elif 'pytest' in cmd_str and '--cov' in cmd_str:
                    return Mock(returncode=0, stdout="TOTAL 100 10 90%", stderr="")
                else:
                    return Mock(returncode=0, stdout="success", stderr="")
            
            mock_run.side_effect = mock_subprocess
            
            # Run quality gates
            results = await self.validator.run_all_gates()
            
            # Verify results
            assert len(results) >= 5  # At least 5 quality gates
            
            # Check that file structure passes
            assert "file_structure" in results
            assert results["file_structure"].status == QualityGateStatus.PASS
            
            # Verify other gates were attempted
            expected_gates = ["code_format", "linting", "type_checking", "security", "test_coverage"]
            for gate in expected_gates:
                if gate in results:
                    # If the gate ran, it should have a result
                    assert isinstance(results[gate], QualityGateResult)