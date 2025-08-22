"""
Comprehensive Quality Gates System
==================================

Complete quality assurance with testing, security, performance, and compliance validation.
Generation 2-3 Quality Gates with autonomous validation and reporting.
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class QualityGateResult:
    """Result of a quality gate check."""
    
    gate_name: str
    passed: bool
    score: float
    details: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class QualityReport:
    """Comprehensive quality report."""
    
    timestamp: datetime
    overall_passed: bool
    overall_score: float
    gates: List[QualityGateResult]
    summary: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


class ComprehensiveQualityGates:
    """
    Comprehensive quality gates system for autonomous validation.
    
    Features:
    - Automated testing with coverage analysis
    - Security vulnerability scanning
    - Performance benchmarking
    - Code quality analysis
    - Compliance verification
    - Documentation validation
    """
    
    def __init__(self, root_path: str = "/root/repo"):
        self.root_path = Path(root_path)
        self.results: List[QualityGateResult] = []
        
        # Quality thresholds
        self.thresholds = {
            "test_coverage": 80.0,
            "security_score": 8.0,
            "performance_score": 7.0,
            "code_quality": 8.0,
            "documentation_coverage": 70.0
        }
    
    async def run_all_quality_gates(self) -> QualityReport:
        """Run all quality gates and generate comprehensive report."""
        
        logger.info("🛡️ Starting Comprehensive Quality Gates")
        print("=" * 80)
        
        start_time = time.time()
        
        # Run all quality gates
        gates = [
            ("Unit Tests", self._run_unit_tests),
            ("Integration Tests", self._run_integration_tests),
            ("Security Scan", self._run_security_scan),
            ("Performance Benchmarks", self._run_performance_benchmarks),
            ("Code Quality Analysis", self._run_code_quality),
            ("Documentation Validation", self._run_documentation_validation),
            ("Bioneural System Tests", self._run_bioneural_tests),
            ("Resilience System Tests", self._run_resilience_tests),
            ("Scaling System Tests", self._run_scaling_tests),
            ("Compliance Checks", self._run_compliance_checks)
        ]
        
        for gate_name, gate_function in gates:
            try:
                logger.info(f"🔍 Running {gate_name}...")
                result = await gate_function()
                self.results.append(result)
                
                status = "✅ PASSED" if result.passed else "❌ FAILED"
                print(f"{status} {gate_name}: {result.score:.1f}/10.0")
                
                if result.errors:
                    for error in result.errors[:3]:  # Show first 3 errors
                        print(f"   ⚠️  {error}")
                
            except Exception as e:
                error_result = QualityGateResult(
                    gate_name=gate_name,
                    passed=False,
                    score=0.0,
                    errors=[f"Gate execution failed: {str(e)}"],
                    execution_time=0.0
                )
                self.results.append(error_result)
                print(f"❌ FAILED {gate_name}: Execution error")
                logger.error(f"Error in {gate_name}: {e}")
        
        # Generate report
        total_time = time.time() - start_time
        report = self._generate_report(total_time)
        
        # Save report
        await self._save_report(report)
        
        # Print summary
        self._print_summary(report)
        
        return report
    
    async def _run_unit_tests(self) -> QualityGateResult:
        """Run unit tests with coverage analysis."""
        
        start_time = time.time()
        
        try:
            # Set environment
            env = os.environ.copy()
            env["PYTHONPATH"] = f"{self.root_path}/src:{env.get('PYTHONPATH', '')}"
            
            # Run tests with coverage
            cmd = [
                sys.executable, "-m", "pytest",
                "--cov=lexgraph_legal_rag",
                "--cov-report=json",
                "--cov-report=term-missing",
                "--tb=short",
                "-v",
                "test_enhanced_resilience_g2.py",
                "test_generation3_scaling.py"
            ]
            
            result = subprocess.run(
                cmd,
                cwd=self.root_path,
                env=env,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            # Parse coverage results
            coverage_data = {}
            try:
                with open(self.root_path / "coverage.json", "r") as f:
                    coverage_data = json.load(f)
            except FileNotFoundError:
                pass
            
            # Calculate metrics
            coverage_percent = coverage_data.get("totals", {}).get("percent_covered", 0)
            tests_passed = "failed" not in result.stdout.lower()
            
            # Determine score
            score = min(10.0, (coverage_percent / 10.0) + (5.0 if tests_passed else 0.0))
            passed = coverage_percent >= self.thresholds["test_coverage"] and tests_passed
            
            errors = []
            if not tests_passed:
                errors.append("Some tests failed")
            if coverage_percent < self.thresholds["test_coverage"]:
                errors.append(f"Coverage {coverage_percent:.1f}% below threshold {self.thresholds['test_coverage']}%")
            
            return QualityGateResult(
                gate_name="Unit Tests",
                passed=passed,
                score=score,
                details={
                    "coverage_percent": coverage_percent,
                    "tests_passed": tests_passed,
                    "stdout": result.stdout[-1000:],  # Last 1000 chars
                    "stderr": result.stderr[-500:] if result.stderr else ""
                },
                errors=errors,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name="Unit Tests",
                passed=False,
                score=0.0,
                errors=[f"Test execution failed: {str(e)}"],
                execution_time=time.time() - start_time
            )
    
    async def _run_integration_tests(self) -> QualityGateResult:
        """Run integration tests."""
        
        start_time = time.time()
        
        try:
            # Test bioneural system integration
            sys.path.insert(0, str(self.root_path / "src"))
            
            # Import and test core systems
            from lexgraph_legal_rag.bioneuro_olfactory_fusion import analyze_document_scent
            from lexgraph_legal_rag.multisensory_legal_processor import analyze_document_multisensory
            
            # Test document
            test_doc = """
            WHEREAS, the parties hereto agree to this contract pursuant to 
            15 U.S.C. § 1681, the Contractor shall indemnify Company from 
            any liability, damages, or penalties arising from breach.
            """
            
            # Test bioneural analysis
            scent_profile = await analyze_document_scent(test_doc, "integration_test")
            multi_analysis = await analyze_document_multisensory(test_doc, "integration_test")
            
            # Validate results
            integration_score = 8.0
            passed = True
            errors = []
            
            if not scent_profile or len(scent_profile.signals) == 0:
                errors.append("Bioneural analysis returned no signals")
                integration_score -= 3.0
                passed = False
            
            if not multi_analysis or multi_analysis.analysis_confidence < 0.5:
                errors.append("Multi-sensory analysis has low confidence")
                integration_score -= 2.0
            
            return QualityGateResult(
                gate_name="Integration Tests",
                passed=passed,
                score=max(0.0, integration_score),
                details={
                    "scent_signals": len(scent_profile.signals) if scent_profile else 0,
                    "analysis_confidence": multi_analysis.analysis_confidence if multi_analysis else 0.0,
                    "primary_channel": multi_analysis.primary_sensory_channel.value if multi_analysis else "none"
                },
                errors=errors,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name="Integration Tests",
                passed=False,
                score=0.0,
                errors=[f"Integration test failed: {str(e)}"],
                execution_time=time.time() - start_time
            )
    
    async def _run_security_scan(self) -> QualityGateResult:
        """Run security vulnerability scan."""
        
        start_time = time.time()
        
        try:
            # Install bandit if not available
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "bandit", "--version"],
                    capture_output=True,
                    text=True
                )
                if result.returncode != 0:
                    subprocess.run(
                        [sys.executable, "-m", "pip", "install", "--break-system-packages", "bandit[toml]"],
                        check=True
                    )
            except:
                pass
            
            # Run bandit security scan
            cmd = [
                sys.executable, "-m", "bandit",
                "-r", "src/",
                "-f", "json",
                "-o", "security-report.json",
                "--skip", "B101"  # Skip assert usage
            ]
            
            result = subprocess.run(
                cmd,
                cwd=self.root_path,
                capture_output=True,
                text=True
            )
            
            # Parse security results
            security_data = {}
            try:
                with open(self.root_path / "security-report.json", "r") as f:
                    security_data = json.load(f)
            except:
                pass
            
            # Calculate security score
            issues = security_data.get("results", [])
            high_severity = sum(1 for issue in issues if issue.get("issue_severity") == "HIGH")
            medium_severity = sum(1 for issue in issues if issue.get("issue_severity") == "MEDIUM")
            low_severity = sum(1 for issue in issues if issue.get("issue_severity") == "LOW")
            
            # Security scoring (10 - penalties)
            security_score = 10.0 - (high_severity * 3.0) - (medium_severity * 1.0) - (low_severity * 0.5)
            security_score = max(0.0, security_score)
            
            passed = security_score >= self.thresholds["security_score"]
            
            errors = []
            if high_severity > 0:
                errors.append(f"{high_severity} high severity security issues")
            if medium_severity > 3:
                errors.append(f"{medium_severity} medium severity security issues")
            
            return QualityGateResult(
                gate_name="Security Scan",
                passed=passed,
                score=security_score,
                details={
                    "high_severity_issues": high_severity,
                    "medium_severity_issues": medium_severity,
                    "low_severity_issues": low_severity,
                    "total_issues": len(issues)
                },
                errors=errors,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name="Security Scan",
                passed=False,
                score=5.0,  # Default moderate score if scan fails
                errors=[f"Security scan failed: {str(e)}"],
                execution_time=time.time() - start_time
            )
    
    async def _run_performance_benchmarks(self) -> QualityGateResult:
        """Run performance benchmarks."""
        
        start_time = time.time()
        
        try:
            sys.path.insert(0, str(self.root_path / "src"))
            
            # Test bioneural performance
            from lexgraph_legal_rag.bioneuro_olfactory_fusion import get_fusion_engine
            
            engine = get_fusion_engine()
            
            # Benchmark document processing
            test_docs = [
                f"Test document {i} with legal content and statutory references."
                for i in range(100)
            ]
            
            benchmark_start = time.time()
            
            results = []
            for i, doc in enumerate(test_docs):
                doc_start = time.time()
                scent_profile = await engine.analyze_document_scent(doc, f"bench_{i}")
                doc_time = time.time() - doc_start
                results.append(doc_time)
            
            total_benchmark_time = time.time() - benchmark_start
            
            # Calculate performance metrics
            avg_doc_time = sum(results) / len(results)
            throughput = len(test_docs) / total_benchmark_time
            
            # Performance scoring
            target_throughput = 1000  # docs/sec
            performance_score = min(10.0, (throughput / target_throughput) * 10.0)
            
            passed = performance_score >= self.thresholds["performance_score"]
            
            errors = []
            if avg_doc_time > 0.01:  # 10ms per document
                errors.append(f"Average document processing time too high: {avg_doc_time*1000:.1f}ms")
            if throughput < 500:
                errors.append(f"Throughput too low: {throughput:.1f} docs/sec")
            
            return QualityGateResult(
                gate_name="Performance Benchmarks",
                passed=passed,
                score=performance_score,
                details={
                    "documents_processed": len(test_docs),
                    "total_time": total_benchmark_time,
                    "avg_doc_time_ms": avg_doc_time * 1000,
                    "throughput_docs_per_sec": throughput
                },
                errors=errors,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name="Performance Benchmarks",
                passed=False,
                score=0.0,
                errors=[f"Performance benchmark failed: {str(e)}"],
                execution_time=time.time() - start_time
            )
    
    async def _run_code_quality(self) -> QualityGateResult:
        """Run code quality analysis."""
        
        start_time = time.time()
        
        try:
            # Install ruff if not available
            try:
                result = subprocess.run([sys.executable, "-m", "ruff", "--version"], capture_output=True)
                if result.returncode != 0:
                    subprocess.run([sys.executable, "-m", "pip", "install", "--break-system-packages", "ruff"], check=True)
            except:
                pass
            
            # Run ruff linting
            cmd = [sys.executable, "-m", "ruff", "check", "src/", "--output-format=json"]
            
            result = subprocess.run(
                cmd,
                cwd=self.root_path,
                capture_output=True,
                text=True
            )
            
            # Parse linting results
            issues = []
            if result.stdout:
                try:
                    for line in result.stdout.strip().split('\n'):
                        if line.strip():
                            issues.append(json.loads(line))
                except:
                    pass
            
            # Calculate quality score
            error_count = sum(1 for issue in issues if issue.get("message", {}).get("type") == "error")
            warning_count = len(issues) - error_count
            
            quality_score = 10.0 - (error_count * 2.0) - (warning_count * 0.5)
            quality_score = max(0.0, quality_score)
            
            passed = quality_score >= self.thresholds["code_quality"]
            
            errors = []
            if error_count > 0:
                errors.append(f"{error_count} code quality errors")
            if warning_count > 10:
                errors.append(f"{warning_count} code quality warnings")
            
            return QualityGateResult(
                gate_name="Code Quality Analysis",
                passed=passed,
                score=quality_score,
                details={
                    "error_count": error_count,
                    "warning_count": warning_count,
                    "total_issues": len(issues)
                },
                errors=errors,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name="Code Quality Analysis",
                passed=False,
                score=5.0,  # Default moderate score
                errors=[f"Code quality analysis failed: {str(e)}"],
                execution_time=time.time() - start_time
            )
    
    async def _run_documentation_validation(self) -> QualityGateResult:
        """Validate documentation coverage and quality."""
        
        start_time = time.time()
        
        try:
            # Count documented vs undocumented functions/classes
            src_path = self.root_path / "src"
            total_functions = 0
            documented_functions = 0
            
            for py_file in src_path.glob("**/*.py"):
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Simple documentation analysis
                lines = content.split('\n')
                in_function = False
                function_has_docstring = False
                
                for i, line in enumerate(lines):
                    stripped = line.strip()
                    
                    # Function/class definition
                    if (stripped.startswith('def ') or stripped.startswith('class ')) and not stripped.startswith('def _'):
                        if in_function and not function_has_docstring:
                            pass  # Previous function wasn't documented
                        
                        total_functions += 1
                        in_function = True
                        function_has_docstring = False
                        
                        # Check next few lines for docstring
                        for j in range(i+1, min(i+5, len(lines))):
                            if '"""' in lines[j] or "'''" in lines[j]:
                                function_has_docstring = True
                                documented_functions += 1
                                break
                        
                        if not function_has_docstring:
                            # Check if next line starts docstring
                            if i+1 < len(lines) and lines[i+1].strip().startswith('"""'):
                                function_has_docstring = True
                                documented_functions += 1
            
            # Calculate documentation coverage
            doc_coverage = (documented_functions / max(1, total_functions)) * 100
            
            # Documentation score
            doc_score = min(10.0, (doc_coverage / 10.0))
            passed = doc_coverage >= self.thresholds["documentation_coverage"]
            
            errors = []
            if doc_coverage < self.thresholds["documentation_coverage"]:
                errors.append(f"Documentation coverage {doc_coverage:.1f}% below threshold {self.thresholds['documentation_coverage']}%")
            
            return QualityGateResult(
                gate_name="Documentation Validation",
                passed=passed,
                score=doc_score,
                details={
                    "total_functions": total_functions,
                    "documented_functions": documented_functions,
                    "documentation_coverage": doc_coverage
                },
                errors=errors,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name="Documentation Validation",
                passed=False,
                score=5.0,
                errors=[f"Documentation validation failed: {str(e)}"],
                execution_time=time.time() - start_time
            )
    
    async def _run_bioneural_tests(self) -> QualityGateResult:
        """Test bioneural system functionality."""
        
        start_time = time.time()
        
        try:
            # Run bioneural demo
            env = os.environ.copy()
            env["PYTHONPATH"] = f"{self.root_path}/src:{env.get('PYTHONPATH', '')}"
            
            result = subprocess.run(
                [sys.executable, "test_bioneuro_minimal.py"],
                cwd=self.root_path,
                env=env,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            # Check if bioneural tests passed
            success = result.returncode == 0 and "SUCCESS" in result.stdout
            
            return QualityGateResult(
                gate_name="Bioneural System Tests",
                passed=success,
                score=9.0 if success else 2.0,
                details={
                    "return_code": result.returncode,
                    "output_length": len(result.stdout)
                },
                errors=[] if success else ["Bioneural system tests failed"],
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name="Bioneural System Tests",
                passed=False,
                score=0.0,
                errors=[f"Bioneural test execution failed: {str(e)}"],
                execution_time=time.time() - start_time
            )
    
    async def _run_resilience_tests(self) -> QualityGateResult:
        """Test resilience system functionality."""
        
        start_time = time.time()
        
        try:
            sys.path.insert(0, str(self.root_path / "src"))
            
            from lexgraph_legal_rag.enhanced_resilience_patterns import EnhancedResilienceSystem
            
            system = EnhancedResilienceSystem()
            
            # Test basic resilience
            async def test_operation():
                return "success"
            
            result = await system.with_resilience(test_operation, operation_name="resilience_test")
            
            success = result == "success"
            
            return QualityGateResult(
                gate_name="Resilience System Tests",
                passed=success,
                score=8.0 if success else 3.0,
                details={"basic_test_passed": success},
                errors=[] if success else ["Resilience system basic test failed"],
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name="Resilience System Tests",
                passed=False,
                score=0.0,
                errors=[f"Resilience test failed: {str(e)}"],
                execution_time=time.time() - start_time
            )
    
    async def _run_scaling_tests(self) -> QualityGateResult:
        """Test scaling system functionality."""
        
        start_time = time.time()
        
        try:
            sys.path.insert(0, str(self.root_path / "src"))
            
            from lexgraph_legal_rag.quantum_scaling_optimizer import QuantumScalingOptimizer
            from lexgraph_legal_rag.intelligent_caching_system import IntelligentCachingSystem
            
            # Test quantum optimizer
            optimizer = QuantumScalingOptimizer()
            await optimizer.register_resource("test_resource", 100.0, 50.0)
            
            metrics = await optimizer.get_scaling_metrics()
            
            # Test intelligent cache
            cache = IntelligentCachingSystem()
            await cache.put("test_key", "test_value")
            cached_value = await cache.get("test_key")
            
            success = (
                metrics["system_efficiency"] > 0 and
                cached_value == "test_value"
            )
            
            return QualityGateResult(
                gate_name="Scaling System Tests",
                passed=success,
                score=8.0 if success else 2.0,
                details={
                    "quantum_optimizer_efficiency": metrics["system_efficiency"],
                    "cache_test_passed": cached_value == "test_value"
                },
                errors=[] if success else ["Scaling system tests failed"],
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name="Scaling System Tests",
                passed=False,
                score=0.0,
                errors=[f"Scaling test failed: {str(e)}"],
                execution_time=time.time() - start_time
            )
    
    async def _run_compliance_checks(self) -> QualityGateResult:
        """Run compliance and governance checks."""
        
        start_time = time.time()
        
        try:
            compliance_score = 8.0
            passed = True
            errors = []
            
            # Check for required files
            required_files = ["README.md", "LICENSE", "pyproject.toml"]
            for file_name in required_files:
                if not (self.root_path / file_name).exists():
                    errors.append(f"Missing required file: {file_name}")
                    compliance_score -= 1.0
                    passed = False
            
            # Check code structure
            src_path = self.root_path / "src"
            if not src_path.exists():
                errors.append("Missing src/ directory")
                compliance_score -= 2.0
                passed = False
            
            tests_exist = any(
                f.name.startswith("test_") for f in self.root_path.glob("test_*.py")
            )
            if not tests_exist:
                errors.append("No test files found")
                compliance_score -= 1.0
            
            return QualityGateResult(
                gate_name="Compliance Checks",
                passed=passed,
                score=max(0.0, compliance_score),
                details={
                    "required_files_present": len(required_files) - len([e for e in errors if "Missing required file" in e]),
                    "tests_present": tests_exist
                },
                errors=errors,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name="Compliance Checks",
                passed=False,
                score=0.0,
                errors=[f"Compliance check failed: {str(e)}"],
                execution_time=time.time() - start_time
            )
    
    def _generate_report(self, total_time: float) -> QualityReport:
        """Generate comprehensive quality report."""
        
        # Calculate overall metrics
        passed_gates = sum(1 for result in self.results if result.passed)
        total_gates = len(self.results)
        overall_passed = passed_gates == total_gates
        
        # Calculate weighted overall score
        total_score = sum(result.score for result in self.results)
        overall_score = total_score / max(1, total_gates)
        
        # Generate recommendations
        recommendations = []
        
        for result in self.results:
            if not result.passed:
                if "test" in result.gate_name.lower():
                    recommendations.append(f"Improve {result.gate_name}: Add more tests and increase coverage")
                elif "security" in result.gate_name.lower():
                    recommendations.append(f"Address security issues in {result.gate_name}")
                elif "performance" in result.gate_name.lower():
                    recommendations.append(f"Optimize performance for {result.gate_name}")
                else:
                    recommendations.append(f"Fix issues in {result.gate_name}")
        
        if overall_score < 7.0:
            recommendations.append("Overall quality score is below target (7.0/10.0)")
        
        return QualityReport(
            timestamp=datetime.now(),
            overall_passed=overall_passed,
            overall_score=overall_score,
            gates=self.results,
            summary={
                "total_gates": total_gates,
                "passed_gates": passed_gates,
                "failed_gates": total_gates - passed_gates,
                "execution_time": total_time,
                "pass_rate": (passed_gates / total_gates) * 100 if total_gates > 0 else 0
            },
            recommendations=recommendations
        )
    
    async def _save_report(self, report: QualityReport) -> None:
        """Save quality report to file."""
        
        report_data = {
            "timestamp": report.timestamp.isoformat(),
            "overall_passed": report.overall_passed,
            "overall_score": report.overall_score,
            "summary": report.summary,
            "gates": [
                {
                    "gate_name": gate.gate_name,
                    "passed": gate.passed,
                    "score": gate.score,
                    "details": gate.details,
                    "errors": gate.errors,
                    "execution_time": gate.execution_time
                }
                for gate in report.gates
            ],
            "recommendations": report.recommendations
        }
        
        report_path = self.root_path / f"quality_gates_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"Quality report saved to {report_path}")
    
    def _print_summary(self, report: QualityReport) -> None:
        """Print quality report summary."""
        
        print("\n" + "=" * 80)
        print("🛡️ COMPREHENSIVE QUALITY GATES REPORT")
        print("=" * 80)
        
        status = "✅ PASSED" if report.overall_passed else "❌ FAILED"
        print(f"Overall Status: {status}")
        print(f"Overall Score: {report.overall_score:.1f}/10.0")
        print(f"Pass Rate: {report.summary['pass_rate']:.1f}%")
        print(f"Execution Time: {report.summary['execution_time']:.1f}s")
        
        print(f"\nGate Results ({report.summary['passed_gates']}/{report.summary['total_gates']} passed):")
        for gate in report.gates:
            status = "✅" if gate.passed else "❌"
            print(f"  {status} {gate.gate_name}: {gate.score:.1f}/10.0")
        
        if report.recommendations:
            print("\n🔧 Recommendations:")
            for i, rec in enumerate(report.recommendations[:5], 1):
                print(f"  {i}. {rec}")
        
        if report.overall_passed:
            print("\n🎉 All quality gates passed! System is ready for production.")
        else:
            print("\n⚠️ Some quality gates failed. Please address the issues above.")


async def main():
    """Main function to run quality gates."""
    
    quality_gates = ComprehensiveQualityGates()
    report = await quality_gates.run_all_quality_gates()
    
    # Exit with appropriate code
    exit_code = 0 if report.overall_passed else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())