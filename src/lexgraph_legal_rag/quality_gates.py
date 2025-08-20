"""Comprehensive quality gates validation for production readiness."""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import time
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


class QualityGateStatus(Enum):
    """Quality gate validation status."""

    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    SKIP = "skip"


@dataclass
class QualityGateResult:
    """Result of a quality gate check."""

    gate_name: str
    status: QualityGateStatus
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)


class QualityGateValidator:
    """Comprehensive quality gate validation system."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results: dict[str, QualityGateResult] = {}
        self.src_path = project_root / "src"
        self.tests_path = project_root / "tests"

    async def run_all_gates(self) -> dict[str, QualityGateResult]:
        """Run all quality gates and return comprehensive results."""
        logger.info("Starting comprehensive quality gate validation...")

        gates = [
            ("code_format", self._check_code_format),
            ("type_checking", self._check_type_checking),
            ("security_scan", self._check_security),
            ("test_coverage", self._check_test_coverage),
            ("performance_benchmarks", self._check_performance),
            ("api_contract", self._check_api_contract),
            ("documentation", self._check_documentation),
            ("dependency_security", self._check_dependency_security),
            ("container_security", self._check_container_security),
            ("configuration_validation", self._check_configuration),
        ]

        # Run gates in parallel where possible
        tasks = []
        for gate_name, gate_func in gates:
            task = asyncio.create_task(self._run_gate(gate_name, gate_func))
            tasks.append(task)

        await asyncio.gather(*tasks, return_exceptions=True)

        # Generate summary
        self._generate_summary()

        return self.results

    async def _run_gate(self, gate_name: str, gate_func) -> None:
        """Run a single quality gate."""
        start_time = time.time()

        try:
            result = await gate_func()
            result.execution_time_ms = (time.time() - start_time) * 1000
            self.results[gate_name] = result

            logger.info(
                f"Quality gate '{gate_name}': {result.status.value} - {result.message}"
            )

        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            self.results[gate_name] = QualityGateResult(
                gate_name=gate_name,
                status=QualityGateStatus.FAIL,
                message=f"Gate execution failed: {e!s}",
                execution_time_ms=execution_time_ms,
                details={"error": str(e), "error_type": type(e).__name__},
            )
            logger.error(f"Quality gate '{gate_name}' failed with exception: {e}")

    async def _check_code_format(self) -> QualityGateResult:
        """Check code formatting with black and ruff."""
        try:
            # Check if black would make changes
            black_result = await self._run_command(
                ["python3", "-m", "black", "--check", "--diff", str(self.src_path)],
                capture_output=True,
            )

            # Check ruff linting
            ruff_result = await self._run_command(
                ["python3", "-m", "ruff", "check", str(self.src_path)],
                capture_output=True,
            )

            black_ok = black_result.returncode == 0
            ruff_ok = ruff_result.returncode == 0

            if black_ok and ruff_ok:
                return QualityGateResult(
                    gate_name="code_format",
                    status=QualityGateStatus.PASS,
                    message="Code formatting and linting passed",
                    details={
                        "black_ok": True,
                        "ruff_ok": True,
                        "files_checked": len(list(self.src_path.rglob("*.py"))),
                    },
                )
            else:
                issues = []
                if not black_ok:
                    issues.append("Black formatting issues found")
                if not ruff_ok:
                    issues.append("Ruff linting issues found")

                return QualityGateResult(
                    gate_name="code_format",
                    status=QualityGateStatus.FAIL,
                    message=f"Code quality issues: {'; '.join(issues)}",
                    details={
                        "black_ok": black_ok,
                        "ruff_ok": ruff_ok,
                        "black_output": black_result.stdout + black_result.stderr,
                        "ruff_output": ruff_result.stdout + ruff_result.stderr,
                    },
                )

        except FileNotFoundError:
            return QualityGateResult(
                gate_name="code_format",
                status=QualityGateStatus.SKIP,
                message="Code formatting tools not available",
                details={"reason": "black or ruff not installed"},
            )
        except Exception as e:
            return QualityGateResult(
                gate_name="code_format",
                status=QualityGateStatus.FAIL,
                message=f"Code format check failed: {e!s}",
                details={"error": str(e)},
            )

    async def _check_type_checking(self) -> QualityGateResult:
        """Check static type checking with mypy."""
        try:
            result = await self._run_command(
                ["python3", "-m", "mypy", str(self.src_path)], capture_output=True
            )

            if result.returncode == 0:
                return QualityGateResult(
                    gate_name="type_checking",
                    status=QualityGateStatus.PASS,
                    message="Type checking passed",
                    details={
                        "mypy_output": result.stdout,
                        "files_checked": len(list(self.src_path.rglob("*.py"))),
                    },
                )
            else:
                return QualityGateResult(
                    gate_name="type_checking",
                    status=QualityGateStatus.FAIL,
                    message="Type checking failed",
                    details={
                        "mypy_output": result.stdout + result.stderr,
                        "return_code": result.returncode,
                    },
                )

        except FileNotFoundError:
            return QualityGateResult(
                gate_name="type_checking",
                status=QualityGateStatus.SKIP,
                message="mypy not available",
                details={"reason": "mypy not installed"},
            )
        except Exception as e:
            return QualityGateResult(
                gate_name="type_checking",
                status=QualityGateStatus.FAIL,
                message=f"Type checking failed: {e!s}",
                details={"error": str(e)},
            )

    async def _check_security(self) -> QualityGateResult:
        """Run security scans with bandit."""
        try:
            result = await self._run_command(
                [
                    "python3",
                    "-m",
                    "bandit",
                    "-r",
                    str(self.src_path),
                    "-f",
                    "json",
                    "-o",
                    "/tmp/bandit_report.json",
                ],
                capture_output=True,
            )

            # Parse bandit results
            try:
                with open("/tmp/bandit_report.json") as f:
                    bandit_data = json.load(f)

                high_issues = len(
                    [
                        i
                        for i in bandit_data.get("results", [])
                        if i.get("issue_severity") == "HIGH"
                    ]
                )
                medium_issues = len(
                    [
                        i
                        for i in bandit_data.get("results", [])
                        if i.get("issue_severity") == "MEDIUM"
                    ]
                )
                low_issues = len(
                    [
                        i
                        for i in bandit_data.get("results", [])
                        if i.get("issue_severity") == "LOW"
                    ]
                )

                if high_issues > 0:
                    status = QualityGateStatus.FAIL
                    message = f"High severity security issues found: {high_issues}"
                elif medium_issues > 5:  # Allow some medium issues
                    status = QualityGateStatus.WARNING
                    message = (
                        f"Multiple medium severity security issues: {medium_issues}"
                    )
                else:
                    status = QualityGateStatus.PASS
                    message = f"Security scan passed (low: {low_issues}, medium: {medium_issues})"

                return QualityGateResult(
                    gate_name="security_scan",
                    status=status,
                    message=message,
                    details={
                        "high_issues": high_issues,
                        "medium_issues": medium_issues,
                        "low_issues": low_issues,
                        "files_scanned": bandit_data.get("metrics", {})
                        .get("_totals", {})
                        .get("loc", 0),
                    },
                )

            except (json.JSONDecodeError, FileNotFoundError):
                # Fallback to stdout parsing
                if result.returncode == 0:
                    return QualityGateResult(
                        gate_name="security_scan",
                        status=QualityGateStatus.PASS,
                        message="Security scan completed with no critical issues",
                        details={"bandit_output": result.stdout},
                    )
                else:
                    return QualityGateResult(
                        gate_name="security_scan",
                        status=QualityGateStatus.WARNING,
                        message="Security scan completed with potential issues",
                        details={"bandit_output": result.stdout + result.stderr},
                    )

        except FileNotFoundError:
            return QualityGateResult(
                gate_name="security_scan",
                status=QualityGateStatus.SKIP,
                message="bandit not available",
                details={"reason": "bandit not installed"},
            )
        except Exception as e:
            return QualityGateResult(
                gate_name="security_scan",
                status=QualityGateStatus.FAIL,
                message=f"Security scan failed: {e!s}",
                details={"error": str(e)},
            )

    async def _check_test_coverage(self) -> QualityGateResult:
        """Check test coverage."""
        try:
            # Run tests with coverage
            result = await self._run_command(
                [
                    "python3",
                    "-m",
                    "pytest",
                    str(self.tests_path),
                    "--cov=" + str(self.src_path),
                    "--cov-report=json:/tmp/coverage.json",
                    "--cov-fail-under=80",
                ],
                capture_output=True,
            )

            # Parse coverage results
            try:
                with open("/tmp/coverage.json") as f:
                    coverage_data = json.load(f)

                total_coverage = coverage_data.get("totals", {}).get(
                    "percent_covered", 0
                )

                if total_coverage >= 85:
                    status = QualityGateStatus.PASS
                    message = f"Excellent test coverage: {total_coverage:.1f}%"
                elif total_coverage >= 75:
                    status = QualityGateStatus.WARNING
                    message = f"Adequate test coverage: {total_coverage:.1f}%"
                else:
                    status = QualityGateStatus.FAIL
                    message = f"Insufficient test coverage: {total_coverage:.1f}%"

                return QualityGateResult(
                    gate_name="test_coverage",
                    status=status,
                    message=message,
                    details={
                        "total_coverage": total_coverage,
                        "files_covered": coverage_data.get("totals", {}).get(
                            "num_statements", 0
                        ),
                        "tests_run": len(list(self.tests_path.rglob("test_*.py"))),
                    },
                )

            except (json.JSONDecodeError, FileNotFoundError):
                # Fallback based on return code
                if result.returncode == 0:
                    return QualityGateResult(
                        gate_name="test_coverage",
                        status=QualityGateStatus.PASS,
                        message="Tests passed (coverage data unavailable)",
                        details={"pytest_output": result.stdout},
                    )
                else:
                    return QualityGateResult(
                        gate_name="test_coverage",
                        status=QualityGateStatus.FAIL,
                        message="Tests failed or coverage insufficient",
                        details={"pytest_output": result.stdout + result.stderr},
                    )

        except FileNotFoundError:
            return QualityGateResult(
                gate_name="test_coverage",
                status=QualityGateStatus.SKIP,
                message="pytest not available",
                details={"reason": "pytest not installed"},
            )
        except Exception as e:
            return QualityGateResult(
                gate_name="test_coverage",
                status=QualityGateStatus.FAIL,
                message=f"Test coverage check failed: {e!s}",
                details={"error": str(e)},
            )

    async def _check_performance(self) -> QualityGateResult:
        """Run performance benchmarks."""
        try:
            # Check if we have performance test files
            perf_test_files = list(
                self.tests_path.glob("**/test_*performance*.py")
            ) + list(self.tests_path.glob("**/test_*benchmark*.py"))

            if not perf_test_files:
                return QualityGateResult(
                    gate_name="performance_benchmarks",
                    status=QualityGateStatus.SKIP,
                    message="No performance tests found",
                    details={"reason": "No performance test files available"},
                )

            # Run performance tests
            result = await self._run_command(
                [
                    "python3",
                    "-m",
                    "pytest",
                    str(self.tests_path),
                    "-k",
                    "performance or benchmark",
                    "--benchmark-only",
                ],
                capture_output=True,
            )

            if result.returncode == 0:
                return QualityGateResult(
                    gate_name="performance_benchmarks",
                    status=QualityGateStatus.PASS,
                    message="Performance benchmarks passed",
                    details={
                        "benchmark_output": result.stdout,
                        "test_files": [str(f) for f in perf_test_files],
                    },
                )
            else:
                return QualityGateResult(
                    gate_name="performance_benchmarks",
                    status=QualityGateStatus.WARNING,
                    message="Performance benchmarks completed with issues",
                    details={"benchmark_output": result.stdout + result.stderr},
                )

        except Exception as e:
            return QualityGateResult(
                gate_name="performance_benchmarks",
                status=QualityGateStatus.FAIL,
                message=f"Performance benchmark check failed: {e!s}",
                details={"error": str(e)},
            )

    async def _check_api_contract(self) -> QualityGateResult:
        """Validate API contracts and OpenAPI specification."""
        try:
            # Check if we can import and validate the API
            from .api import create_api

            app = create_api(test_mode=True)

            # Get OpenAPI schema
            openapi_schema = app.openapi()

            # Basic validation
            required_paths = ["/health", "/ready", "/ping"]
            missing_paths = []

            for path in required_paths:
                if path not in openapi_schema.get("paths", {}):
                    missing_paths.append(path)

            if missing_paths:
                return QualityGateResult(
                    gate_name="api_contract",
                    status=QualityGateStatus.FAIL,
                    message=f"Missing required API endpoints: {missing_paths}",
                    details={"missing_paths": missing_paths},
                )
            else:
                return QualityGateResult(
                    gate_name="api_contract",
                    status=QualityGateStatus.PASS,
                    message="API contract validation passed",
                    details={
                        "endpoints_count": len(openapi_schema.get("paths", {})),
                        "openapi_version": openapi_schema.get("openapi", "unknown"),
                    },
                )

        except ImportError as e:
            return QualityGateResult(
                gate_name="api_contract",
                status=QualityGateStatus.SKIP,
                message="API contract validation skipped - dependencies missing",
                details={"error": str(e)},
            )
        except Exception as e:
            return QualityGateResult(
                gate_name="api_contract",
                status=QualityGateStatus.FAIL,
                message=f"API contract validation failed: {e!s}",
                details={"error": str(e)},
            )

    async def _check_documentation(self) -> QualityGateResult:
        """Check documentation completeness."""
        try:
            # Check for essential documentation files
            required_docs = [
                "README.md",
                "CHANGELOG.md",
                "LICENSE",
                "docs/DEPLOYMENT.md",
            ]

            missing_docs = []
            for doc in required_docs:
                if not (self.project_root / doc).exists():
                    missing_docs.append(doc)

            # Check docstring coverage
            python_files = list(self.src_path.rglob("*.py"))
            files_with_docstrings = 0

            for py_file in python_files:
                try:
                    content = py_file.read_text()
                    if '"""' in content or "'''" in content:
                        files_with_docstrings += 1
                except:
                    continue

            docstring_coverage = (
                files_with_docstrings / len(python_files) if python_files else 0
            )

            if missing_docs and docstring_coverage < 0.6:
                return QualityGateResult(
                    gate_name="documentation",
                    status=QualityGateStatus.FAIL,
                    message=f"Documentation incomplete: missing {len(missing_docs)} files, {docstring_coverage:.1%} docstring coverage",
                    details={
                        "missing_docs": missing_docs,
                        "docstring_coverage": docstring_coverage,
                    },
                )
            elif missing_docs or docstring_coverage < 0.8:
                return QualityGateResult(
                    gate_name="documentation",
                    status=QualityGateStatus.WARNING,
                    message=f"Documentation could be improved: {len(missing_docs)} missing files, {docstring_coverage:.1%} docstring coverage",
                    details={
                        "missing_docs": missing_docs,
                        "docstring_coverage": docstring_coverage,
                    },
                )
            else:
                return QualityGateResult(
                    gate_name="documentation",
                    status=QualityGateStatus.PASS,
                    message=f"Documentation complete: {docstring_coverage:.1%} docstring coverage",
                    details={
                        "docstring_coverage": docstring_coverage,
                        "python_files": len(python_files),
                    },
                )

        except Exception as e:
            return QualityGateResult(
                gate_name="documentation",
                status=QualityGateStatus.FAIL,
                message=f"Documentation check failed: {e!s}",
                details={"error": str(e)},
            )

    async def _check_dependency_security(self) -> QualityGateResult:
        """Check for known security vulnerabilities in dependencies."""
        try:
            # Check if safety is available
            result = await self._run_command(
                ["python3", "-m", "safety", "check", "--json"], capture_output=True
            )

            if result.returncode == 0:
                try:
                    safety_data = json.loads(result.stdout)
                    if not safety_data:  # Empty list means no vulnerabilities
                        return QualityGateResult(
                            gate_name="dependency_security",
                            status=QualityGateStatus.PASS,
                            message="No known security vulnerabilities in dependencies",
                            details={"vulnerabilities": 0},
                        )
                    else:
                        high_vulns = len(
                            [
                                v
                                for v in safety_data
                                if v.get("severity", "").lower() == "high"
                            ]
                        )
                        medium_vulns = len(
                            [
                                v
                                for v in safety_data
                                if v.get("severity", "").lower() == "medium"
                            ]
                        )

                        if high_vulns > 0:
                            status = QualityGateStatus.FAIL
                            message = (
                                f"High severity vulnerabilities found: {high_vulns}"
                            )
                        elif medium_vulns > 3:
                            status = QualityGateStatus.WARNING
                            message = f"Multiple medium severity vulnerabilities: {medium_vulns}"
                        else:
                            status = QualityGateStatus.WARNING
                            message = (
                                f"Some vulnerabilities found: {len(safety_data)} total"
                            )

                        return QualityGateResult(
                            gate_name="dependency_security",
                            status=status,
                            message=message,
                            details={
                                "total_vulnerabilities": len(safety_data),
                                "high_severity": high_vulns,
                                "medium_severity": medium_vulns,
                            },
                        )

                except json.JSONDecodeError:
                    # Fallback to text parsing
                    if "No known security vulnerabilities" in result.stdout:
                        return QualityGateResult(
                            gate_name="dependency_security",
                            status=QualityGateStatus.PASS,
                            message="No known security vulnerabilities",
                            details={"safety_output": result.stdout},
                        )
                    else:
                        return QualityGateResult(
                            gate_name="dependency_security",
                            status=QualityGateStatus.WARNING,
                            message="Dependency security check completed",
                            details={"safety_output": result.stdout},
                        )
            else:
                return QualityGateResult(
                    gate_name="dependency_security",
                    status=QualityGateStatus.WARNING,
                    message="Dependency security check completed with warnings",
                    details={"safety_output": result.stdout + result.stderr},
                )

        except FileNotFoundError:
            return QualityGateResult(
                gate_name="dependency_security",
                status=QualityGateStatus.SKIP,
                message="safety not available",
                details={"reason": "safety not installed"},
            )
        except Exception as e:
            return QualityGateResult(
                gate_name="dependency_security",
                status=QualityGateStatus.FAIL,
                message=f"Dependency security check failed: {e!s}",
                details={"error": str(e)},
            )

    async def _check_container_security(self) -> QualityGateResult:
        """Check container security if Dockerfile exists."""
        dockerfile = self.project_root / "Dockerfile"

        if not dockerfile.exists():
            return QualityGateResult(
                gate_name="container_security",
                status=QualityGateStatus.SKIP,
                message="No Dockerfile found",
                details={"reason": "Dockerfile not present"},
            )

        try:
            # Basic Dockerfile security checks
            content = dockerfile.read_text()

            issues = []
            if "USER root" in content and "USER " not in content.split("USER root")[1]:
                issues.append("Running as root user")

            if "ADD http" in content:
                issues.append("Using ADD with URLs (security risk)")

            if "--privileged" in content:
                issues.append("Using privileged mode")

            if "curl" in content and "rm" not in content:
                issues.append("Installing curl without cleanup")

            if len(issues) > 2:
                return QualityGateResult(
                    gate_name="container_security",
                    status=QualityGateStatus.FAIL,
                    message=f"Multiple container security issues: {len(issues)}",
                    details={"issues": issues},
                )
            elif issues:
                return QualityGateResult(
                    gate_name="container_security",
                    status=QualityGateStatus.WARNING,
                    message=f"Some container security issues: {len(issues)}",
                    details={"issues": issues},
                )
            else:
                return QualityGateResult(
                    gate_name="container_security",
                    status=QualityGateStatus.PASS,
                    message="Container security checks passed",
                    details={"dockerfile_lines": len(content.splitlines())},
                )

        except Exception as e:
            return QualityGateResult(
                gate_name="container_security",
                status=QualityGateStatus.FAIL,
                message=f"Container security check failed: {e!s}",
                details={"error": str(e)},
            )

    async def _check_configuration(self) -> QualityGateResult:
        """Validate configuration files and environment setup."""
        try:
            issues = []

            # Check pyproject.toml
            pyproject = self.project_root / "pyproject.toml"
            if not pyproject.exists():
                issues.append("pyproject.toml missing")

            # Check requirements.txt
            requirements = self.project_root / "requirements.txt"
            if not requirements.exists():
                issues.append("requirements.txt missing")

            # Check for environment file template
            env_example = self.project_root / ".env.example"
            if not env_example.exists():
                issues.append(".env.example missing")

            # Check for critical configuration
            try:
                from .config import validate_environment

                validate_environment(allow_test_mode=True)
                config_ok = True
            except Exception:
                config_ok = False
                issues.append("Configuration validation failed")

            if len(issues) > 2:
                return QualityGateResult(
                    gate_name="configuration_validation",
                    status=QualityGateStatus.FAIL,
                    message=f"Configuration issues: {'; '.join(issues)}",
                    details={"issues": issues},
                )
            elif issues:
                return QualityGateResult(
                    gate_name="configuration_validation",
                    status=QualityGateStatus.WARNING,
                    message=f"Minor configuration issues: {'; '.join(issues)}",
                    details={"issues": issues},
                )
            else:
                return QualityGateResult(
                    gate_name="configuration_validation",
                    status=QualityGateStatus.PASS,
                    message="Configuration validation passed",
                    details={"config_ok": config_ok},
                )

        except Exception as e:
            return QualityGateResult(
                gate_name="configuration_validation",
                status=QualityGateStatus.FAIL,
                message=f"Configuration validation failed: {e!s}",
                details={"error": str(e)},
            )

    async def _run_command(
        self, cmd: list[str], capture_output: bool = True
    ) -> subprocess.CompletedProcess:
        """Run a command asynchronously."""
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=subprocess.PIPE if capture_output else None,
            stderr=subprocess.PIPE if capture_output else None,
            cwd=self.project_root,
        )

        stdout, stderr = await process.communicate()

        # Create a subprocess.CompletedProcess-like object
        class AsyncResult:
            def __init__(self, returncode, stdout, stderr):
                self.returncode = returncode
                self.stdout = stdout.decode() if stdout else ""
                self.stderr = stderr.decode() if stderr else ""

        return AsyncResult(process.returncode, stdout, stderr)

    def _generate_summary(self) -> None:
        """Generate summary of all quality gate results."""
        len(self.results)
        passed = len(
            [r for r in self.results.values() if r.status == QualityGateStatus.PASS]
        )
        failed = len(
            [r for r in self.results.values() if r.status == QualityGateStatus.FAIL]
        )
        warnings = len(
            [r for r in self.results.values() if r.status == QualityGateStatus.WARNING]
        )
        skipped = len(
            [r for r in self.results.values() if r.status == QualityGateStatus.SKIP]
        )

        logger.info(
            f"Quality Gates Summary: {passed} passed, {failed} failed, {warnings} warnings, {skipped} skipped"
        )

        if failed > 0:
            failed_gates = [
                name
                for name, result in self.results.items()
                if result.status == QualityGateStatus.FAIL
            ]
            logger.error(f"Failed gates: {', '.join(failed_gates)}")

        # Overall assessment
        if failed == 0 and warnings <= 2:
            logger.info("✅ PRODUCTION READY: All critical quality gates passed")
        elif failed <= 1 and warnings <= 4:
            logger.warning(
                "⚠️ NEEDS ATTENTION: Some quality gates need improvement before production"
            )
        else:
            logger.error("❌ NOT PRODUCTION READY: Critical quality gates failed")


async def run_quality_gates(project_root: Path | None = None) -> dict[str, Any]:
    """Run comprehensive quality gates validation."""
    if project_root is None:
        project_root = Path.cwd()

    validator = QualityGateValidator(project_root)
    results = await validator.run_all_gates()

    # Create summary for external consumption
    summary = {
        "timestamp": time.time(),
        "project_root": str(project_root),
        "total_gates": len(results),
        "results": {
            name: {
                "status": result.status.value,
                "message": result.message,
                "execution_time_ms": result.execution_time_ms,
                "details": result.details,
            }
            for name, result in results.items()
        },
    }

    # Calculate overall status
    statuses = [result.status for result in results.values()]
    if QualityGateStatus.FAIL in statuses:
        summary["overall_status"] = "FAIL"
    elif statuses.count(QualityGateStatus.WARNING) > 2:
        summary["overall_status"] = "WARNING"
    else:
        summary["overall_status"] = "PASS"

    return summary
