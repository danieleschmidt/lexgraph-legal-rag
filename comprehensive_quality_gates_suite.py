#!/usr/bin/env python3
"""
Comprehensive Quality Gates Testing Suite
Autonomous SDLC with integrated testing framework for all 3 generations

This suite validates all progressive quality gate implementations:
- Generation 1: Simple implementation testing
- Generation 2: Robust implementation with error handling
- Generation 3: Quantum-optimized implementation with ML/AI
- Integration testing across all generations
- Performance benchmarking and validation
- Security vulnerability testing
"""

import asyncio
import json
import logging
import time
import traceback
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

# Import all three generations
try:
    from progressive_quality_gates_g1 import ProgressiveQualityGateEngine as G1Engine
    from progressive_quality_gates_g1 import ProgressiveGateConfig as G1Config
    from progressive_quality_gates_g2 import RobustQualityGateEngine as G2Engine
    from progressive_quality_gates_g2 import ProgressiveGateConfig as G2Config
    from progressive_quality_gates_g3 import QuantumOptimizedQualityGateEngine as G3Engine
    from progressive_quality_gates_g3 import ProgressiveGateConfig as G3Config
except ImportError as e:
    logging.error(f"Failed to import quality gate engines: {e}")
    raise


class TestResult(Enum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class TestCase:
    """Individual test case definition"""
    name: str
    description: str
    test_func: str
    expected_result: TestResult = TestResult.PASSED
    timeout: float = 30.0
    retry_count: int = 0
    prerequisites: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestExecution:
    """Test execution results"""
    test_case: TestCase
    result: TestResult
    execution_time: float
    error_message: Optional[str] = None
    output: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    generation: str = "unknown"
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestSuiteResults:
    """Comprehensive test suite results"""
    suite_name: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    error_tests: int
    skipped_tests: int
    total_execution_time: float
    test_executions: List[TestExecution]
    timestamp: datetime = field(default_factory=datetime.now)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)


class ComprehensiveQualityGatesTestSuite:
    """Comprehensive testing suite for all quality gate generations"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.test_results: List[TestExecution] = []
        self.performance_benchmarks: Dict[str, List[float]] = {
            "g1_execution_times": [],
            "g2_execution_times": [],
            "g3_execution_times": [],
            "g1_scores": [],
            "g2_scores": [],
            "g3_scores": []
        }
        
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive test logging"""
        logger = logging.getLogger(f"{__name__}.TestSuite")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - TEST - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
            # Test log file
            try:
                log_dir = Path("logs")
                log_dir.mkdir(exist_ok=True)
                file_handler = logging.FileHandler(log_dir / "comprehensive_quality_gates_tests.log")
                file_handler.setFormatter(console_formatter)
                logger.addHandler(file_handler)
            except Exception as e:
                logger.warning(f"Could not setup file logging: {e}")
                
        return logger
    
    async def test_generation_1_basic_functionality(self) -> TestExecution:
        """Test Generation 1 basic functionality"""
        test_case = TestCase(
            name="test_g1_basic_functionality",
            description="Test Generation 1 Progressive Quality Gates basic execution",
            test_func="test_generation_1_basic_functionality"
        )
        
        start_time = time.time()
        
        try:
            self.logger.info("Testing Generation 1 - Basic Functionality")
            
            config = G1Config(
                enable_adaptive_thresholds=True,
                baseline_threshold=0.7
            )
            
            engine = G1Engine(config)
            results = await engine.run_all_gates()
            
            # Validation checks
            assert len(results) > 0, "No gates were executed"
            
            passed_gates = [r for r in results if r.status.value == "passed"]
            assert len(passed_gates) >= len(results) * 0.5, "Less than 50% of gates passed"
            
            # Performance validation
            total_execution_time = sum(r.execution_time for r in results)
            assert total_execution_time < 30.0, f"Execution too slow: {total_execution_time}s"
            
            # Record benchmarks
            self.performance_benchmarks["g1_execution_times"].append(total_execution_time)
            avg_score = sum(r.score for r in results) / len(results)
            self.performance_benchmarks["g1_scores"].append(avg_score)
            
            execution_time = time.time() - start_time
            
            return TestExecution(
                test_case=test_case,
                result=TestResult.PASSED,
                execution_time=execution_time,
                generation="G1",
                metrics={
                    "gates_executed": len(results),
                    "gates_passed": len(passed_gates),
                    "average_score": avg_score,
                    "total_gate_time": total_execution_time
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestExecution(
                test_case=test_case,
                result=TestResult.ERROR,
                execution_time=execution_time,
                error_message=str(e),
                generation="G1"
            )
    
    async def test_generation_2_robust_features(self) -> TestExecution:
        """Test Generation 2 robust features"""
        test_case = TestCase(
            name="test_g2_robust_features",
            description="Test Generation 2 robust error handling and resilience patterns",
            test_func="test_generation_2_robust_features"
        )
        
        start_time = time.time()
        
        try:
            self.logger.info("Testing Generation 2 - Robust Features")
            
            config = G2Config(
                enable_adaptive_thresholds=True,
                enable_circuit_breakers=True,
                enable_retries=True,
                enable_health_checks=True,
                parallel_execution=True
            )
            
            engine = G2Engine(config)
            results = await engine.run_all_gates()
            
            # Validation checks
            assert len(results) > 0, "No gates were executed"
            
            # Check for Generation 2 specific features
            dependency_gates = [r for r in results if r.gate_type.value == "dependency"]
            assert len(dependency_gates) > 0, "Dependency gates not executed in G2"
            
            # Validate health checks
            health_status = engine.get_health_status()
            assert "healthy" in health_status, "Health status not available"
            
            # Circuit breaker validation
            assert len(engine.circuit_breakers) > 0, "Circuit breakers not initialized"
            
            # Performance validation
            total_execution_time = sum(r.execution_time for r in results)
            self.performance_benchmarks["g2_execution_times"].append(total_execution_time)
            avg_score = sum(r.score for r in results) / len(results)
            self.performance_benchmarks["g2_scores"].append(avg_score)
            
            execution_time = time.time() - start_time
            
            return TestExecution(
                test_case=test_case,
                result=TestResult.PASSED,
                execution_time=execution_time,
                generation="G2",
                metrics={
                    "gates_executed": len(results),
                    "dependency_gates": len(dependency_gates),
                    "circuit_breakers": len(engine.circuit_breakers),
                    "health_checks_enabled": config.enable_health_checks,
                    "average_score": avg_score
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestExecution(
                test_case=test_case,
                result=TestResult.ERROR,
                execution_time=execution_time,
                error_message=str(e),
                generation="G2"
            )
    
    async def test_generation_3_quantum_optimization(self) -> TestExecution:
        """Test Generation 3 quantum optimization features"""
        test_case = TestCase(
            name="test_g3_quantum_optimization",
            description="Test Generation 3 quantum optimization and ML features",
            test_func="test_generation_3_quantum_optimization"
        )
        
        start_time = time.time()
        
        try:
            self.logger.info("Testing Generation 3 - Quantum Optimization")
            
            config = G3Config()
            config.optimization.enable_intelligent_caching = True
            config.optimization.enable_predictive_analysis = True
            config.optimization.enable_auto_scaling = True
            config.optimization.enable_quantum_optimization = True
            config.optimization.enable_anomaly_detection = True
            
            engine = G3Engine(config)
            results = await engine.run_all_gates()
            
            # Validation checks
            assert len(results) > 0, "No gates were executed"
            
            # Check for Generation 3 specific features
            quantum_optimized = [r for r in results if hasattr(r, 'optimization_level') and 
                               r.optimization_level.value == "quantum"]
            assert len(quantum_optimized) > 0, "No quantum-optimized gates found"
            
            # Validate caching system
            cache_hit_ratio = engine.cache.get_hit_ratio()
            assert cache_hit_ratio >= 0.0, "Cache hit ratio not available"
            
            # Validate resource pool
            resource_load = engine.resource_pool.get_load()
            assert 0.0 <= resource_load <= 1.0, f"Invalid resource load: {resource_load}"
            
            # Validate ML model
            assert engine.predictive_model.model.accuracy > 0.5, "ML model accuracy too low"
            
            # Performance validation
            total_execution_time = sum(r.execution_time for r in results)
            self.performance_benchmarks["g3_execution_times"].append(total_execution_time)
            avg_score = sum(r.score for r in results) / len(results)
            self.performance_benchmarks["g3_scores"].append(avg_score)
            
            execution_time = time.time() - start_time
            
            return TestExecution(
                test_case=test_case,
                result=TestResult.PASSED,
                execution_time=execution_time,
                generation="G3",
                metrics={
                    "gates_executed": len(results),
                    "quantum_optimized": len(quantum_optimized),
                    "cache_hit_ratio": cache_hit_ratio,
                    "ml_accuracy": engine.predictive_model.model.accuracy,
                    "resource_load": resource_load,
                    "average_score": avg_score
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestExecution(
                test_case=test_case,
                result=TestResult.ERROR,
                execution_time=execution_time,
                error_message=str(e),
                generation="G3"
            )
    
    async def test_performance_comparison(self) -> TestExecution:
        """Test performance comparison across all generations"""
        test_case = TestCase(
            name="test_performance_comparison",
            description="Compare performance across all three generations",
            test_func="test_performance_comparison"
        )
        
        start_time = time.time()
        
        try:
            self.logger.info("Testing Performance Comparison")
            
            # Ensure we have benchmark data
            if (not self.performance_benchmarks["g1_execution_times"] or
                not self.performance_benchmarks["g2_execution_times"] or
                not self.performance_benchmarks["g3_execution_times"]):
                raise Exception("Missing performance benchmark data. Run individual generation tests first.")
            
            # Calculate performance statistics
            g1_avg_time = statistics.mean(self.performance_benchmarks["g1_execution_times"])
            g2_avg_time = statistics.mean(self.performance_benchmarks["g2_execution_times"])
            g3_avg_time = statistics.mean(self.performance_benchmarks["g3_execution_times"])
            
            g1_avg_score = statistics.mean(self.performance_benchmarks["g1_scores"])
            g2_avg_score = statistics.mean(self.performance_benchmarks["g2_scores"])
            g3_avg_score = statistics.mean(self.performance_benchmarks["g3_scores"])
            
            # Performance assertions
            # G2 should have more features than G1 but similar performance
            assert g2_avg_time <= g1_avg_time * 2.0, f"G2 too slow compared to G1: {g2_avg_time} vs {g1_avg_time}"
            
            # G3 should be optimized and potentially faster despite more features
            assert g3_avg_time <= g2_avg_time * 1.5, f"G3 too slow compared to G2: {g3_avg_time} vs {g2_avg_time}"
            
            # Quality scores should improve or maintain across generations
            assert g2_avg_score >= g1_avg_score * 0.9, f"G2 quality regression: {g2_avg_score} vs {g1_avg_score}"
            assert g3_avg_score >= g2_avg_score * 0.9, f"G3 quality regression: {g3_avg_score} vs {g2_avg_score}"
            
            execution_time = time.time() - start_time
            
            return TestExecution(
                test_case=test_case,
                result=TestResult.PASSED,
                execution_time=execution_time,
                generation="ALL",
                metrics={
                    "g1_avg_time": g1_avg_time,
                    "g2_avg_time": g2_avg_time,
                    "g3_avg_time": g3_avg_time,
                    "g1_avg_score": g1_avg_score,
                    "g2_avg_score": g2_avg_score,
                    "g3_avg_score": g3_avg_score,
                    "g2_vs_g1_time_ratio": g2_avg_time / g1_avg_time,
                    "g3_vs_g2_time_ratio": g3_avg_time / g2_avg_time,
                    "quality_progression": [g1_avg_score, g2_avg_score, g3_avg_score]
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestExecution(
                test_case=test_case,
                result=TestResult.ERROR,
                execution_time=execution_time,
                error_message=str(e),
                generation="ALL"
            )
    
    async def test_security_validation(self) -> TestExecution:
        """Test security validation across all generations"""
        test_case = TestCase(
            name="test_security_validation",
            description="Validate security gate implementations across generations",
            test_func="test_security_validation"
        )
        
        start_time = time.time()
        
        try:
            self.logger.info("Testing Security Validation")
            
            security_results = {"G1": None, "G2": None, "G3": None}
            
            # Test G1 security
            g1_config = G1Config()
            g1_engine = G1Engine(g1_config)
            g1_result = await g1_engine.run_security_gate()
            security_results["G1"] = g1_result
            
            # Test G2 security
            g2_config = G2Config(enable_security_scanning=True)
            g2_engine = G2Engine(g2_config)
            g2_result = await g2_engine.run_security_gate()
            security_results["G2"] = g2_result
            
            # Test G3 security
            g3_config = G3Config()
            g3_engine = G3Engine(g3_config)
            g3_result = await g3_engine.run_quantum_security_gate()
            security_results["G3"] = g3_result
            
            # Validate security improvements across generations
            g1_score = security_results["G1"].score
            g2_score = security_results["G2"].score
            g3_score = security_results["G3"].score
            
            # G2 should have enhanced security features
            g2_metadata = security_results["G2"].metadata
            assert "security_scanning_enabled" in g2_metadata, "G2 security scanning not enabled"
            
            # G3 should have quantum optimizations
            g3_metadata = security_results["G3"].metadata
            assert g3_metadata.get("quantum_optimized", False), "G3 quantum optimization not applied"
            
            execution_time = time.time() - start_time
            
            return TestExecution(
                test_case=test_case,
                result=TestResult.PASSED,
                execution_time=execution_time,
                generation="ALL",
                metrics={
                    "g1_security_score": g1_score,
                    "g2_security_score": g2_score,
                    "g3_security_score": g3_score,
                    "security_progression": [g1_score, g2_score, g3_score],
                    "g2_enhanced_features": len(g2_metadata.get("security_issues", [])),
                    "g3_quantum_features": g3_metadata.get("quantum_optimized", False)
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestExecution(
                test_case=test_case,
                result=TestResult.ERROR,
                execution_time=execution_time,
                error_message=str(e),
                generation="ALL"
            )
    
    async def test_integration_compatibility(self) -> TestExecution:
        """Test integration and compatibility between generations"""
        test_case = TestCase(
            name="test_integration_compatibility",
            description="Test integration and compatibility between all generations",
            test_func="test_integration_compatibility"
        )
        
        start_time = time.time()
        
        try:
            self.logger.info("Testing Integration Compatibility")
            
            # Test that all generations can be instantiated together
            g1_config = G1Config()
            g2_config = G2Config() 
            g3_config = G3Config()
            
            g1_engine = G1Engine(g1_config)
            g2_engine = G2Engine(g2_config)
            g3_engine = G3Engine(g3_config)
            
            # Test concurrent execution
            tasks = [
                g1_engine.run_code_quality_gate("integration_test_g1"),
                g2_engine.run_code_quality_gate("integration_test_g2"),
                g3_engine.run_quantum_code_quality_gate("integration_test_g3")
            ]
            
            concurrent_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Validate concurrent execution
            for i, result in enumerate(concurrent_results):
                if isinstance(result, Exception):
                    raise Exception(f"Generation {i+1} failed during concurrent execution: {result}")
            
            # Validate result consistency
            scores = [r.score for r in concurrent_results if hasattr(r, 'score')]
            assert len(scores) == 3, f"Expected 3 scores, got {len(scores)}"
            
            # Check that scores are within reasonable range
            for i, score in enumerate(scores):
                assert 0.0 <= score <= 3.0, f"Generation {i+1} score out of range: {score}"
            
            execution_time = time.time() - start_time
            
            return TestExecution(
                test_case=test_case,
                result=TestResult.PASSED,
                execution_time=execution_time,
                generation="ALL",
                metrics={
                    "concurrent_execution_success": True,
                    "g1_score": scores[0] if len(scores) > 0 else 0,
                    "g2_score": scores[1] if len(scores) > 1 else 0,
                    "g3_score": scores[2] if len(scores) > 2 else 0,
                    "score_variance": statistics.variance(scores) if len(scores) > 1 else 0,
                    "all_generations_compatible": True
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestExecution(
                test_case=test_case,
                result=TestResult.ERROR,
                execution_time=execution_time,
                error_message=str(e),
                generation="ALL"
            )
    
    async def run_comprehensive_test_suite(self) -> TestSuiteResults:
        """Run the comprehensive test suite"""
        self.logger.info("🧪 Starting Comprehensive Quality Gates Test Suite")
        
        suite_start_time = time.time()
        
        # Define test sequence
        test_methods = [
            self.test_generation_1_basic_functionality,
            self.test_generation_2_robust_features,
            self.test_generation_3_quantum_optimization,
            self.test_performance_comparison,
            self.test_security_validation,
            self.test_integration_compatibility
        ]
        
        test_executions = []
        
        for test_method in test_methods:
            try:
                self.logger.info(f"Running test: {test_method.__name__}")
                execution = await test_method()
                test_executions.append(execution)
                
                if execution.result == TestResult.PASSED:
                    self.logger.info(f"✅ {execution.test_case.name} PASSED")
                elif execution.result == TestResult.ERROR:
                    self.logger.error(f"❌ {execution.test_case.name} ERROR: {execution.error_message}")
                else:
                    self.logger.warning(f"⚠️  {execution.test_case.name} {execution.result.value}")
                    
            except Exception as e:
                self.logger.error(f"Test method {test_method.__name__} failed: {e}")
                # Create error execution
                error_execution = TestExecution(
                    test_case=TestCase(name=test_method.__name__, description="Test method execution", test_func=test_method.__name__),
                    result=TestResult.ERROR,
                    execution_time=0.0,
                    error_message=str(e)
                )
                test_executions.append(error_execution)
        
        # Calculate results
        total_execution_time = time.time() - suite_start_time
        passed_tests = len([t for t in test_executions if t.result == TestResult.PASSED])
        failed_tests = len([t for t in test_executions if t.result == TestResult.FAILED])
        error_tests = len([t for t in test_executions if t.result == TestResult.ERROR])
        skipped_tests = len([t for t in test_executions if t.result == TestResult.SKIPPED])
        
        # Generate performance metrics
        performance_metrics = {}
        if self.performance_benchmarks["g1_execution_times"]:
            performance_metrics.update({
                "g1_avg_execution_time": statistics.mean(self.performance_benchmarks["g1_execution_times"]),
                "g1_avg_score": statistics.mean(self.performance_benchmarks["g1_scores"]),
                "g2_avg_execution_time": statistics.mean(self.performance_benchmarks["g2_execution_times"]),
                "g2_avg_score": statistics.mean(self.performance_benchmarks["g2_scores"]),
                "g3_avg_execution_time": statistics.mean(self.performance_benchmarks["g3_execution_times"]),
                "g3_avg_score": statistics.mean(self.performance_benchmarks["g3_scores"])
            })
        
        results = TestSuiteResults(
            suite_name="Comprehensive Quality Gates Test Suite",
            total_tests=len(test_executions),
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            error_tests=error_tests,
            skipped_tests=skipped_tests,
            total_execution_time=total_execution_time,
            test_executions=test_executions,
            performance_metrics=performance_metrics
        )
        
        self.test_results.extend(test_executions)
        
        return results
    
    def export_test_results(self, results: TestSuiteResults, output_path: str = "comprehensive_quality_gates_test_results.json") -> None:
        """Export comprehensive test results"""
        export_data = {
            "suite_name": results.suite_name,
            "timestamp": results.timestamp.isoformat(),
            "summary": {
                "total_tests": results.total_tests,
                "passed_tests": results.passed_tests,
                "failed_tests": results.failed_tests,
                "error_tests": results.error_tests,
                "skipped_tests": results.skipped_tests,
                "success_rate": (results.passed_tests / results.total_tests) * 100 if results.total_tests > 0 else 0,
                "total_execution_time": results.total_execution_time
            },
            "performance_metrics": results.performance_metrics,
            "test_executions": [
                {
                    "test_name": exec.test_case.name,
                    "description": exec.test_case.description,
                    "result": exec.result.value,
                    "execution_time": exec.execution_time,
                    "generation": exec.generation,
                    "error_message": exec.error_message,
                    "metrics": exec.metrics,
                    "timestamp": exec.timestamp.isoformat()
                }
                for exec in results.test_executions
            ],
            "benchmark_data": self.performance_benchmarks
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Test results exported to {output_path}")


async def main():
    """Main comprehensive test execution"""
    print("🧪 TERRAGON AUTONOMOUS SDLC - Comprehensive Quality Gates Test Suite")
    print("=" * 80)
    
    test_suite = ComprehensiveQualityGatesTestSuite()
    
    try:
        # Run comprehensive tests
        results = await test_suite.run_comprehensive_test_suite()
        
        # Export results
        test_suite.export_test_results(results)
        
        # Display comprehensive summary
        print("\n🧪 COMPREHENSIVE TEST RESULTS SUMMARY")
        print("-" * 50)
        
        print(f"📊 Total Tests: {results.total_tests}")
        print(f"✅ Passed: {results.passed_tests}")
        print(f"❌ Failed: {results.failed_tests}")
        print(f"🚨 Errors: {results.error_tests}")
        print(f"⏭️  Skipped: {results.skipped_tests}")
        print(f"🎯 Success Rate: {(results.passed_tests/results.total_tests)*100:.1f}%" if results.total_tests > 0 else "N/A")
        print(f"⏱️  Total Execution Time: {results.total_execution_time:.2f}s")
        
        if results.performance_metrics:
            print("\n⚡ PERFORMANCE BENCHMARKS:")
            print(f"Generation 1 Avg Time: {results.performance_metrics.get('g1_avg_execution_time', 0):.3f}s")
            print(f"Generation 2 Avg Time: {results.performance_metrics.get('g2_avg_execution_time', 0):.3f}s") 
            print(f"Generation 3 Avg Time: {results.performance_metrics.get('g3_avg_execution_time', 0):.3f}s")
            print(f"Generation 1 Avg Score: {results.performance_metrics.get('g1_avg_score', 0):.3f}")
            print(f"Generation 2 Avg Score: {results.performance_metrics.get('g2_avg_score', 0):.3f}")
            print(f"Generation 3 Avg Score: {results.performance_metrics.get('g3_avg_score', 0):.3f}")
        
        print("\n🎯 DETAILED TEST RESULTS:")
        for execution in results.test_executions:
            status_symbol = {
                TestResult.PASSED: "✅",
                TestResult.FAILED: "❌", 
                TestResult.ERROR: "🚨",
                TestResult.SKIPPED: "⏭️"
            }.get(execution.result, "❓")
            
            print(f"{status_symbol} {execution.test_case.name} ({execution.generation}) - {execution.execution_time:.3f}s")
            if execution.error_message:
                print(f"    Error: {execution.error_message}")
            
            if execution.metrics:
                key_metrics = {k: v for k, v in execution.metrics.items() if k in ['gates_executed', 'gates_passed', 'average_score']}
                if key_metrics:
                    print(f"    Metrics: {key_metrics}")
        
        # Final status
        if results.passed_tests == results.total_tests:
            print(f"\n🎉 ALL TESTS PASSED! Comprehensive Quality Gates Implementation Validated ✅")
        elif results.error_tests == 0 and results.failed_tests == 0:
            print(f"\n✅ All tests executed successfully with some skipped")
        else:
            print(f"\n⚠️  Some tests failed or had errors. Review results for details.")
        
        print("\n🏆 AUTONOMOUS SDLC QUALITY GATES - TESTING COMPLETE")
        
        return results
        
    except Exception as e:
        print(f"❌ Test suite execution failed: {e}")
        print(f"Error details: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    asyncio.run(main())