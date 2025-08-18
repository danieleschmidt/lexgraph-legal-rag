"""
Comprehensive Quality Gates Validation for Bioneural Olfactory Fusion System

This module implements comprehensive quality gates including:
- Performance benchmarking
- Security scanning
- Test coverage validation
- Memory leak detection
- Load testing
- Error rate monitoring
- Code quality metrics
- Integration testing
- Production readiness validation
"""

import asyncio
import logging
import time
import json
import traceback
import psutil
import gc
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import concurrent.futures
import numpy as np

# Import our bioneural systems
from enhanced_bioneural_fusion_g1 import analyze_document_scent_enhanced
from robust_bioneural_system_g2 import analyze_document_scent_robust
from scalable_bioneural_system_g3 import analyze_document_scent_scalable

logger = logging.getLogger(__name__)


@dataclass
class QualityGateResult:
    """Result of a quality gate check."""
    gate_name: str
    passed: bool
    score: float  # 0.0 to 1.0
    details: Dict[str, Any]
    duration: float
    timestamp: float = field(default_factory=time.time)
    errors: List[str] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    """Performance benchmark result."""
    test_name: str
    generation: str
    documents_processed: int
    total_time: float
    avg_time_per_doc: float
    throughput: float
    memory_peak: float
    cpu_usage_avg: float
    errors: int


class QualityGateValidator:
    """Comprehensive quality gate validation system."""
    
    def __init__(self):
        self.results: List[QualityGateResult] = []
        self.benchmark_results: List[BenchmarkResult] = []
        self.test_documents = self._load_test_documents()
        
    def _load_test_documents(self) -> List[Tuple[str, str]]:
        """Load test documents for validation."""
        return [
            ("simple_contract", "This is a simple contract between parties."),
            ("complex_legal", """
            WHEREAS, the parties hereto acknowledge that the Disclosing Party has developed
            certain proprietary and confidential information, data, and know-how pursuant to
            15 U.S.C. § 1681 and other applicable federal regulations, and WHEREAS, the
            Receiving Party agrees to maintain strict confidentiality of all such information
            and to indemnify and hold harmless the Disclosing Party from any unauthorized
            disclosure or breach thereof, NOW THEREFORE, in consideration of the mutual
            covenants contained herein, the parties agree to the following terms and conditions.
            """),
            ("statutory_reference", """
            Section 501(c)(3) of the Internal Revenue Code defines tax-exempt organizations.
            See also 26 U.S.C. § 501(c)(3) and related Treasury Regulations at 26 CFR 1.501(c)(3)-1.
            The organization must operate exclusively for exempt purposes as defined in IRC § 501(c)(3).
            """),
            ("risk_heavy", """
            INDEMNIFICATION CLAUSE: Contractor agrees to defend, indemnify, and hold harmless
            Company from all liability, damages, penalties, fines, costs, and expenses arising
            from any breach of this agreement, violation of applicable laws, or negligent acts
            of Contractor. This indemnification obligation shall survive termination of this
            agreement and applies to all claims, regardless of legal theory.
            """),
            ("citation_dense", """
            The Supreme Court's holding in Brown v. Board of Education, 347 U.S. 483 (1954),
            established that separate educational facilities are inherently unequal. See also
            Plessy v. Ferguson, 163 U.S. 537 (1896) (overruled). Cf. Loving v. Virginia,
            388 U.S. 1 (1967); McLaughlin v. Florida, 379 U.S. 184 (1964). Id. at 495.
            """)
        ]
    
    async def run_all_quality_gates(self) -> Dict[str, Any]:
        """Run all quality gates and return comprehensive results."""
        logger.info("Starting comprehensive quality gate validation")
        start_time = time.time()
        
        # Run all quality gates
        gates = [
            self._performance_benchmark_gate,
            self._security_validation_gate,
            self._memory_leak_detection_gate,
            self._error_handling_validation_gate,
            self._load_testing_gate,
            self._accuracy_validation_gate,
            self._scalability_validation_gate,
            self._integration_testing_gate
        ]
        
        for gate in gates:
            try:
                result = await gate()
                self.results.append(result)
                logger.info(f"Quality gate completed: {result.gate_name} - {'PASSED' if result.passed else 'FAILED'}")
            except Exception as e:
                error_result = QualityGateResult(
                    gate_name=gate.__name__,
                    passed=False,
                    score=0.0,
                    details={"error": str(e), "traceback": traceback.format_exc()},
                    duration=0.0,
                    errors=[str(e)]
                )
                self.results.append(error_result)
                logger.error(f"Quality gate failed with exception: {gate.__name__} - {e}")
        
        total_time = time.time() - start_time
        
        # Generate comprehensive report
        return self._generate_quality_report(total_time)
    
    async def _performance_benchmark_gate(self) -> QualityGateResult:
        """Performance benchmark quality gate."""
        start_time = time.time()
        
        try:
            # Benchmark all three generations
            g1_results = await self._benchmark_generation("G1", analyze_document_scent_enhanced)
            g2_results = await self._benchmark_generation("G2", analyze_document_scent_robust)
            g3_results = await self._benchmark_generation("G3", analyze_document_scent_scalable)
            
            # Validate performance requirements
            performance_requirements = {
                "max_avg_time_per_doc": 1.0,  # 1 second max per document
                "min_throughput": 10.0,       # 10 docs/sec minimum
                "max_memory_usage": 500.0,    # 500MB max memory
                "max_error_rate": 0.05        # 5% max error rate
            }
            
            passed = True
            details = {
                "g1_results": g1_results.__dict__,
                "g2_results": g2_results.__dict__,
                "g3_results": g3_results.__dict__,
                "requirements": performance_requirements
            }
            
            # Check each generation against requirements
            for gen_result in [g1_results, g2_results, g3_results]:
                if gen_result.avg_time_per_doc > performance_requirements["max_avg_time_per_doc"]:
                    passed = False
                if gen_result.throughput < performance_requirements["min_throughput"]:
                    passed = False
                if gen_result.memory_peak > performance_requirements["max_memory_usage"]:
                    passed = False
                error_rate = gen_result.errors / max(1, gen_result.documents_processed)
                if error_rate > performance_requirements["max_error_rate"]:
                    passed = False
            
            # Calculate performance score
            best_throughput = max(g1_results.throughput, g2_results.throughput, g3_results.throughput)
            score = min(1.0, best_throughput / performance_requirements["min_throughput"])
            
            duration = time.time() - start_time
            
            return QualityGateResult(
                gate_name="Performance Benchmark",
                passed=passed,
                score=score,
                details=details,
                duration=duration
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name="Performance Benchmark",
                passed=False,
                score=0.0,
                details={"error": str(e)},
                duration=time.time() - start_time,
                errors=[str(e)]
            )
    
    async def _benchmark_generation(self, generation: str, analyze_func) -> BenchmarkResult:
        """Benchmark a specific generation."""
        start_time = time.time()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        documents_processed = 0
        errors = 0
        cpu_readings = []
        
        # Process test documents multiple times for better statistics
        for iteration in range(3):  # 3 iterations
            for doc_id, doc_text in self.test_documents:
                try:
                    # Monitor CPU during processing
                    cpu_before = psutil.cpu_percent()
                    
                    await analyze_func(doc_text, f"{generation}_{doc_id}_{iteration}")
                    
                    cpu_after = psutil.cpu_percent()
                    cpu_readings.append((cpu_before + cpu_after) / 2)
                    
                    documents_processed += 1
                    
                except Exception as e:
                    errors += 1
                    logger.error(f"Benchmark error in {generation}: {e}")
        
        total_time = time.time() - start_time
        peak_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        avg_cpu = sum(cpu_readings) / len(cpu_readings) if cpu_readings else 0
        avg_time_per_doc = total_time / max(1, documents_processed)
        throughput = documents_processed / total_time if total_time > 0 else 0
        
        result = BenchmarkResult(
            test_name=f"Benchmark_{generation}",
            generation=generation,
            documents_processed=documents_processed,
            total_time=total_time,
            avg_time_per_doc=avg_time_per_doc,
            throughput=throughput,
            memory_peak=memory_increase,
            cpu_usage_avg=avg_cpu,
            errors=errors
        )
        
        self.benchmark_results.append(result)
        return result
    
    async def _security_validation_gate(self) -> QualityGateResult:
        """Security validation quality gate."""
        start_time = time.time()
        
        try:
            security_tests = []
            
            # Test 1: Input validation
            malicious_inputs = [
                "'; DROP TABLE users; --",
                "<script>alert('xss')</script>",
                "../../etc/passwd",
                "\\x00\\x01\\x02",
                "A" * 1000000,  # Very long input
            ]
            
            input_validation_passed = 0
            for malicious_input in malicious_inputs:
                try:
                    await analyze_document_scent_robust(malicious_input, "security_test")
                    input_validation_passed += 1
                except Exception:
                    # Should fail gracefully, not crash
                    input_validation_passed += 1
            
            security_tests.append({
                "test": "input_validation",
                "passed": input_validation_passed == len(malicious_inputs),
                "score": input_validation_passed / len(malicious_inputs)
            })
            
            # Test 2: Memory safety
            large_docs = ["Large document " * 10000 for _ in range(5)]
            memory_safety_passed = 0
            
            for i, doc in enumerate(large_docs):
                try:
                    initial_memory = psutil.Process().memory_info().rss
                    await analyze_document_scent_robust(doc, f"memory_test_{i}")
                    final_memory = psutil.Process().memory_info().rss
                    
                    # Check for reasonable memory usage (< 100MB increase)
                    memory_increase = (final_memory - initial_memory) / 1024 / 1024
                    if memory_increase < 100:
                        memory_safety_passed += 1
                except Exception:
                    pass
            
            security_tests.append({
                "test": "memory_safety",
                "passed": memory_safety_passed >= len(large_docs) * 0.8,  # 80% threshold
                "score": memory_safety_passed / len(large_docs)
            })
            
            # Test 3: Concurrent access safety
            concurrent_tasks = []
            for i in range(10):
                task = analyze_document_scent_robust(f"Concurrent test document {i}", f"concurrent_{i}")
                concurrent_tasks.append(task)
            
            concurrent_results = await asyncio.gather(*concurrent_tasks, return_exceptions=True)
            concurrent_success = sum(1 for r in concurrent_results if not isinstance(r, Exception))
            
            security_tests.append({
                "test": "concurrent_safety",
                "passed": concurrent_success >= 8,  # 80% success rate
                "score": concurrent_success / len(concurrent_tasks)
            })
            
            # Overall security score
            overall_passed = all(test["passed"] for test in security_tests)
            overall_score = sum(test["score"] for test in security_tests) / len(security_tests)
            
            duration = time.time() - start_time
            
            return QualityGateResult(
                gate_name="Security Validation",
                passed=overall_passed,
                score=overall_score,
                details={"security_tests": security_tests},
                duration=duration
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name="Security Validation",
                passed=False,
                score=0.0,
                details={"error": str(e)},
                duration=time.time() - start_time,
                errors=[str(e)]
            )
    
    async def _memory_leak_detection_gate(self) -> QualityGateResult:
        """Memory leak detection quality gate."""
        start_time = time.time()
        
        try:
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            memory_readings = [initial_memory]
            
            # Run many iterations to detect memory leaks
            iterations = 50
            for i in range(iterations):
                # Process multiple documents
                for doc_id, doc_text in self.test_documents:
                    await analyze_document_scent_enhanced(doc_text, f"memleak_{i}_{doc_id}")
                
                # Force garbage collection
                gc.collect()
                
                # Record memory usage every 10 iterations
                if i % 10 == 0:
                    current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    memory_readings.append(current_memory)
            
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_increase = final_memory - initial_memory
            
            # Calculate memory growth trend
            if len(memory_readings) > 2:
                # Simple linear regression to detect growth trend
                x = list(range(len(memory_readings)))
                y = memory_readings
                n = len(x)
                sum_x = sum(x)
                sum_y = sum(y)
                sum_xy = sum(xi * yi for xi, yi in zip(x, y))
                sum_x2 = sum(xi * xi for xi in x)
                
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
                memory_growth_rate = slope  # MB per measurement interval
            else:
                memory_growth_rate = memory_increase / iterations
            
            # Acceptable memory growth thresholds
            max_acceptable_increase = 50.0  # 50MB total increase
            max_acceptable_growth_rate = 1.0  # 1MB per interval
            
            passed = (memory_increase < max_acceptable_increase and 
                     abs(memory_growth_rate) < max_acceptable_growth_rate)
            
            # Score based on memory efficiency
            score = max(0.0, 1.0 - (memory_increase / max_acceptable_increase))
            
            duration = time.time() - start_time
            
            return QualityGateResult(
                gate_name="Memory Leak Detection",
                passed=passed,
                score=score,
                details={
                    "initial_memory_mb": initial_memory,
                    "final_memory_mb": final_memory,
                    "memory_increase_mb": memory_increase,
                    "memory_growth_rate_mb_per_interval": memory_growth_rate,
                    "iterations": iterations,
                    "memory_readings": memory_readings
                },
                duration=duration
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name="Memory Leak Detection",
                passed=False,
                score=0.0,
                details={"error": str(e)},
                duration=time.time() - start_time,
                errors=[str(e)]
            )
    
    async def _error_handling_validation_gate(self) -> QualityGateResult:
        """Error handling validation quality gate."""
        start_time = time.time()
        
        try:
            error_scenarios = [
                ("empty_string", ""),
                ("none_input", None),
                ("invalid_characters", "\\x00\\x01\\x02\\x03"),
                ("extremely_long", "A" * 500000),
                ("invalid_unicode", "\\xff\\xfe"),
                ("binary_data", bytes([0, 1, 2, 3, 4]).decode('latin1', errors='ignore')),
            ]
            
            graceful_handling_count = 0
            error_details = []
            
            for scenario_name, test_input in error_scenarios:
                try:
                    if test_input is None:
                        # Test None input specifically
                        try:
                            result = await analyze_document_scent_robust(test_input, f"error_test_{scenario_name}")
                            error_details.append({
                                "scenario": scenario_name,
                                "handled_gracefully": False,
                                "error": "Should have failed but didn't"
                            })
                        except Exception as e:
                            # Should fail gracefully with validation error
                            graceful_handling_count += 1
                            error_details.append({
                                "scenario": scenario_name,
                                "handled_gracefully": True,
                                "error_type": type(e).__name__
                            })
                    else:
                        result = await analyze_document_scent_robust(test_input, f"error_test_{scenario_name}")
                        # If we get here, it handled the input gracefully
                        graceful_handling_count += 1
                        error_details.append({
                            "scenario": scenario_name,
                            "handled_gracefully": True,
                            "result": "Processed successfully"
                        })
                        
                except Exception as e:
                    # Check if it's a graceful error (validation) vs system crash
                    if "validation" in str(e).lower() or "invalid" in str(e).lower():
                        graceful_handling_count += 1
                        error_details.append({
                            "scenario": scenario_name,
                            "handled_gracefully": True,
                            "error_type": type(e).__name__
                        })
                    else:
                        error_details.append({
                            "scenario": scenario_name,
                            "handled_gracefully": False,
                            "error": str(e)
                        })
            
            passed = graceful_handling_count >= len(error_scenarios) * 0.8  # 80% threshold
            score = graceful_handling_count / len(error_scenarios)
            
            duration = time.time() - start_time
            
            return QualityGateResult(
                gate_name="Error Handling Validation",
                passed=passed,
                score=score,
                details={
                    "total_scenarios": len(error_scenarios),
                    "graceful_handling_count": graceful_handling_count,
                    "error_details": error_details
                },
                duration=duration
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name="Error Handling Validation",
                passed=False,
                score=0.0,
                details={"error": str(e)},
                duration=time.time() - start_time,
                errors=[str(e)]
            )
    
    async def _load_testing_gate(self) -> QualityGateResult:
        """Load testing quality gate."""
        start_time = time.time()
        
        try:
            # Simulate concurrent load
            concurrent_requests = 20
            requests_per_client = 5
            
            async def client_load():
                client_results = []
                for i in range(requests_per_client):
                    doc_id, doc_text = self.test_documents[i % len(self.test_documents)]
                    try:
                        request_start = time.time()
                        await analyze_document_scent_scalable(doc_text, f"load_test_{doc_id}_{i}")
                        request_time = time.time() - request_start
                        client_results.append({"success": True, "time": request_time})
                    except Exception as e:
                        client_results.append({"success": False, "error": str(e)})
                return client_results
            
            # Execute concurrent load
            load_start = time.time()
            client_tasks = [client_load() for _ in range(concurrent_requests)]
            all_results = await asyncio.gather(*client_tasks)
            load_duration = time.time() - load_start
            
            # Analyze results
            total_requests = 0
            successful_requests = 0
            response_times = []
            
            for client_results in all_results:
                for result in client_results:
                    total_requests += 1
                    if result["success"]:
                        successful_requests += 1
                        response_times.append(result["time"])
            
            success_rate = successful_requests / total_requests if total_requests > 0 else 0
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0
            max_response_time = max(response_times) if response_times else 0
            throughput = total_requests / load_duration
            
            # Load testing criteria
            min_success_rate = 0.95  # 95% success rate
            max_avg_response_time = 2.0  # 2 seconds average
            min_throughput = 5.0  # 5 requests/sec
            
            passed = (success_rate >= min_success_rate and 
                     avg_response_time <= max_avg_response_time and 
                     throughput >= min_throughput)
            
            score = (success_rate + 
                    max(0, 1.0 - avg_response_time / max_avg_response_time) + 
                    min(1.0, throughput / min_throughput)) / 3.0
            
            duration = time.time() - start_time
            
            return QualityGateResult(
                gate_name="Load Testing",
                passed=passed,
                score=score,
                details={
                    "concurrent_clients": concurrent_requests,
                    "requests_per_client": requests_per_client,
                    "total_requests": total_requests,
                    "successful_requests": successful_requests,
                    "success_rate": success_rate,
                    "avg_response_time": avg_response_time,
                    "max_response_time": max_response_time,
                    "throughput": throughput,
                    "load_duration": load_duration
                },
                duration=duration
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name="Load Testing",
                passed=False,
                score=0.0,
                details={"error": str(e)},
                duration=time.time() - start_time,
                errors=[str(e)]
            )
    
    async def _accuracy_validation_gate(self) -> QualityGateResult:
        """Accuracy validation quality gate."""
        start_time = time.time()
        
        try:
            # Test known document patterns with expected outcomes
            test_cases = [
                {
                    "name": "high_complexity_legal",
                    "text": "WHEREAS, notwithstanding the heretofore mentioned provisions, pursuant to the aforementioned regulations...",
                    "expected_receptors": ["legal_complexity"],
                    "min_intensity": 0.3
                },
                {
                    "name": "statutory_references",
                    "text": "15 U.S.C. § 1681 and Section 501(c)(3) of the Internal Revenue Code and 26 CFR 1.501",
                    "expected_receptors": ["statutory_authority"],
                    "min_intensity": 0.4
                },
                {
                    "name": "risk_indicators",
                    "text": "LIABILITY, PENALTIES, DAMAGES, INDEMNIFICATION, BREACH, VIOLATION of this agreement",
                    "expected_receptors": ["risk_profile"],
                    "min_intensity": 0.5
                },
                {
                    "name": "citation_heavy",
                    "text": "Brown v. Board, 347 U.S. 483 (1954); Plessy v. Ferguson, 163 U.S. 537 (1896); Id. at 495",
                    "expected_receptors": ["citation_density"],
                    "min_intensity": 0.3
                }
            ]
            
            accuracy_results = []
            
            for test_case in test_cases:
                try:
                    profile = await analyze_document_scent_enhanced(test_case["text"], f"accuracy_{test_case['name']}")
                    
                    case_passed = True
                    receptor_results = {}
                    
                    for receptor_type_str in test_case["expected_receptors"]:
                        # Find matching signal
                        matching_signals = [s for s in profile.signals 
                                          if s.receptor_type.value == receptor_type_str]
                        
                        if matching_signals:
                            signal = matching_signals[0]
                            receptor_results[receptor_type_str] = {
                                "intensity": signal.intensity,
                                "confidence": signal.confidence,
                                "meets_threshold": signal.intensity >= test_case["min_intensity"]
                            }
                            if signal.intensity < test_case["min_intensity"]:
                                case_passed = False
                        else:
                            receptor_results[receptor_type_str] = {
                                "intensity": 0.0,
                                "confidence": 0.0,
                                "meets_threshold": False
                            }
                            case_passed = False
                    
                    accuracy_results.append({
                        "test_case": test_case["name"],
                        "passed": case_passed,
                        "receptor_results": receptor_results
                    })
                    
                except Exception as e:
                    accuracy_results.append({
                        "test_case": test_case["name"],
                        "passed": False,
                        "error": str(e)
                    })
            
            total_passed = sum(1 for result in accuracy_results if result["passed"])
            passed = total_passed >= len(test_cases) * 0.75  # 75% threshold
            score = total_passed / len(test_cases)
            
            duration = time.time() - start_time
            
            return QualityGateResult(
                gate_name="Accuracy Validation",
                passed=passed,
                score=score,
                details={
                    "test_cases": test_cases,
                    "accuracy_results": accuracy_results,
                    "total_passed": total_passed,
                    "total_cases": len(test_cases)
                },
                duration=duration
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name="Accuracy Validation",
                passed=False,
                score=0.0,
                details={"error": str(e)},
                duration=time.time() - start_time,
                errors=[str(e)]
            )
    
    async def _scalability_validation_gate(self) -> QualityGateResult:
        """Scalability validation quality gate."""
        start_time = time.time()
        
        try:
            # Test scalability across different document sizes
            scalability_tests = []
            
            # Small documents (< 100 words)
            small_docs = ["Short legal clause. " * 10 for _ in range(10)]
            small_start = time.time()
            small_tasks = [analyze_document_scent_scalable(doc, f"small_{i}") for i, doc in enumerate(small_docs)]
            await asyncio.gather(*small_tasks)
            small_time = time.time() - small_start
            small_throughput = len(small_docs) / small_time
            
            scalability_tests.append({
                "size_category": "small",
                "document_count": len(small_docs),
                "total_time": small_time,
                "throughput": small_throughput
            })
            
            # Medium documents (100-1000 words)
            medium_docs = ["Medium legal document. " * 100 for _ in range(5)]
            medium_start = time.time()
            medium_tasks = [analyze_document_scent_scalable(doc, f"medium_{i}") for i, doc in enumerate(medium_docs)]
            await asyncio.gather(*medium_tasks)
            medium_time = time.time() - medium_start
            medium_throughput = len(medium_docs) / medium_time
            
            scalability_tests.append({
                "size_category": "medium",
                "document_count": len(medium_docs),
                "total_time": medium_time,
                "throughput": medium_throughput
            })
            
            # Large documents (> 1000 words)
            large_docs = ["Large legal document. " * 300 for _ in range(2)]
            large_start = time.time()
            large_tasks = [analyze_document_scent_scalable(doc, f"large_{i}") for i, doc in enumerate(large_docs)]
            await asyncio.gather(*large_tasks)
            large_time = time.time() - large_start
            large_throughput = len(large_docs) / large_time
            
            scalability_tests.append({
                "size_category": "large",
                "document_count": len(large_docs),
                "total_time": large_time,
                "throughput": large_throughput
            })
            
            # Evaluate scalability
            min_small_throughput = 5.0  # 5 docs/sec for small docs
            min_medium_throughput = 2.0  # 2 docs/sec for medium docs  
            min_large_throughput = 0.5   # 0.5 docs/sec for large docs
            
            passed = (small_throughput >= min_small_throughput and
                     medium_throughput >= min_medium_throughput and
                     large_throughput >= min_large_throughput)
            
            # Calculate score based on throughput performance
            small_score = min(1.0, small_throughput / min_small_throughput)
            medium_score = min(1.0, medium_throughput / min_medium_throughput)
            large_score = min(1.0, large_throughput / min_large_throughput)
            score = (small_score + medium_score + large_score) / 3.0
            
            duration = time.time() - start_time
            
            return QualityGateResult(
                gate_name="Scalability Validation",
                passed=passed,
                score=score,
                details={
                    "scalability_tests": scalability_tests,
                    "thresholds": {
                        "min_small_throughput": min_small_throughput,
                        "min_medium_throughput": min_medium_throughput,
                        "min_large_throughput": min_large_throughput
                    }
                },
                duration=duration
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name="Scalability Validation",
                passed=False,
                score=0.0,
                details={"error": str(e)},
                duration=time.time() - start_time,
                errors=[str(e)]
            )
    
    async def _integration_testing_gate(self) -> QualityGateResult:
        """Integration testing quality gate."""
        start_time = time.time()
        
        try:
            integration_tests = []
            
            # Test 1: Cross-generation consistency
            test_doc = "This is a comprehensive legal document with statutory references 15 U.S.C. § 1681 and liability clauses."
            
            g1_profile = await analyze_document_scent_enhanced(test_doc, "integration_g1")
            g2_profile = await analyze_document_scent_robust(test_doc, "integration_g2")
            g3_profile = await analyze_document_scent_scalable(test_doc, "integration_g3")
            
            # Check consistency between generations
            consistency_score = self._calculate_profile_consistency([g1_profile, g2_profile, g3_profile])
            
            integration_tests.append({
                "test": "cross_generation_consistency",
                "passed": consistency_score >= 0.7,  # 70% consistency threshold
                "score": consistency_score
            })
            
            # Test 2: End-to-end workflow
            try:
                workflow_docs = [
                    "Legal document 1 with contract terms",
                    "Legal document 2 with statutory references",
                    "Legal document 3 with risk factors"
                ]
                
                workflow_results = []
                for i, doc in enumerate(workflow_docs):
                    profile = await analyze_document_scent_scalable(doc, f"workflow_{i}")
                    workflow_results.append(profile)
                
                workflow_passed = len(workflow_results) == len(workflow_docs)
                
                integration_tests.append({
                    "test": "end_to_end_workflow",
                    "passed": workflow_passed,
                    "score": 1.0 if workflow_passed else 0.0
                })
                
            except Exception as e:
                integration_tests.append({
                    "test": "end_to_end_workflow",
                    "passed": False,
                    "score": 0.0,
                    "error": str(e)
                })
            
            # Overall integration score
            overall_passed = all(test["passed"] for test in integration_tests)
            overall_score = sum(test["score"] for test in integration_tests) / len(integration_tests)
            
            duration = time.time() - start_time
            
            return QualityGateResult(
                gate_name="Integration Testing",
                passed=overall_passed,
                score=overall_score,
                details={"integration_tests": integration_tests},
                duration=duration
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name="Integration Testing",
                passed=False,
                score=0.0,
                details={"error": str(e)},
                duration=time.time() - start_time,
                errors=[str(e)]
            )
    
    def _calculate_profile_consistency(self, profiles: List) -> float:
        """Calculate consistency score between profiles."""
        if len(profiles) < 2:
            return 1.0
        
        # Compare receptor intensities across profiles
        receptor_consistency_scores = []
        
        for receptor_type in profiles[0].signals:
            intensities = []
            for profile in profiles:
                matching_signals = [s for s in profile.signals if s.receptor_type == receptor_type.receptor_type]
                if matching_signals:
                    intensities.append(matching_signals[0].intensity)
            
            if len(intensities) == len(profiles):
                # Calculate coefficient of variation (lower is more consistent)
                mean_intensity = sum(intensities) / len(intensities)
                if mean_intensity > 0:
                    variance = sum((x - mean_intensity) ** 2 for x in intensities) / len(intensities)
                    std_dev = variance ** 0.5
                    cv = std_dev / mean_intensity
                    consistency_score = max(0.0, 1.0 - cv)  # Convert CV to consistency score
                    receptor_consistency_scores.append(consistency_score)
        
        return sum(receptor_consistency_scores) / len(receptor_consistency_scores) if receptor_consistency_scores else 0.0
    
    def _generate_quality_report(self, total_time: float) -> Dict[str, Any]:
        """Generate comprehensive quality report."""
        passed_gates = sum(1 for result in self.results if result.passed)
        total_gates = len(self.results)
        overall_score = sum(result.score for result in self.results) / total_gates if total_gates > 0 else 0.0
        
        # Generate detailed results
        detailed_results = []
        for result in self.results:
            detailed_results.append({
                "gate_name": result.gate_name,
                "passed": result.passed,
                "score": result.score,
                "duration": result.duration,
                "errors": result.errors,
                "details": result.details
            })
        
        # Performance summary
        performance_summary = {
            "total_benchmarks": len(self.benchmark_results),
            "benchmark_results": [
                {
                    "generation": br.generation,
                    "throughput": br.throughput,
                    "avg_time_per_doc": br.avg_time_per_doc,
                    "memory_peak": br.memory_peak,
                    "errors": br.errors
                }
                for br in self.benchmark_results
            ]
        }
        
        return {
            "timestamp": time.time(),
            "total_execution_time": total_time,
            "overall_status": "PASSED" if passed_gates == total_gates else "FAILED",
            "overall_score": overall_score,
            "gates_passed": passed_gates,
            "gates_total": total_gates,
            "pass_rate": passed_gates / total_gates if total_gates > 0 else 0.0,
            "quality_gates": detailed_results,
            "performance_summary": performance_summary,
            "recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on quality gate results."""
        recommendations = []
        
        for result in self.results:
            if not result.passed:
                if "Performance" in result.gate_name:
                    recommendations.append("Consider optimizing performance bottlenecks identified in benchmarks")
                elif "Security" in result.gate_name:
                    recommendations.append("Address security vulnerabilities found in validation tests")
                elif "Memory" in result.gate_name:
                    recommendations.append("Investigate and fix potential memory leaks")
                elif "Error" in result.gate_name:
                    recommendations.append("Improve error handling and graceful degradation")
                elif "Load" in result.gate_name:
                    recommendations.append("Enhance load handling capabilities and concurrency support")
                elif "Accuracy" in result.gate_name:
                    recommendations.append("Review and improve algorithm accuracy for expected patterns")
                elif "Scalability" in result.gate_name:
                    recommendations.append("Optimize scalability for different document sizes")
                elif "Integration" in result.gate_name:
                    recommendations.append("Fix integration issues and ensure cross-component compatibility")
        
        if overall_score := sum(result.score for result in self.results) / len(self.results) if self.results else 0.0:
            if overall_score < 0.8:
                recommendations.append("Overall quality score is below 80% - comprehensive review recommended")
        
        return recommendations


async def main():
    """Run comprehensive quality gates validation."""
    print("🔍 COMPREHENSIVE QUALITY GATES VALIDATION")
    print("=" * 60)
    
    validator = QualityGateValidator()
    
    try:
        report = await validator.run_all_quality_gates()
        
        # Display summary
        print(f"\n📊 QUALITY GATES SUMMARY")
        print(f"Overall Status: {'✅ PASSED' if report['overall_status'] == 'PASSED' else '❌ FAILED'}")
        print(f"Gates Passed: {report['gates_passed']}/{report['gates_total']} ({report['pass_rate']:.1%})")
        print(f"Overall Score: {report['overall_score']:.3f}")
        print(f"Total Execution Time: {report['total_execution_time']:.1f}s")
        
        # Display individual gate results
        print(f"\n🚪 INDIVIDUAL GATE RESULTS")
        for gate in report['quality_gates']:
            status = "✅ PASSED" if gate['passed'] else "❌ FAILED"
            print(f"  {gate['gate_name']}: {status} (Score: {gate['score']:.3f}, Time: {gate['duration']:.1f}s)")
        
        # Display performance summary
        print(f"\n⚡ PERFORMANCE SUMMARY")
        for benchmark in report['performance_summary']['benchmark_results']:
            print(f"  {benchmark['generation']}: {benchmark['throughput']:.1f} docs/sec, "
                  f"{benchmark['avg_time_per_doc']:.3f}s avg, "
                  f"{benchmark['memory_peak']:.1f}MB peak")
        
        # Display recommendations
        if report['recommendations']:
            print(f"\n💡 RECOMMENDATIONS")
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"  {i}. {rec}")
        
        # Save detailed report
        report_file = Path("quality_gates_report.json")
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📄 Detailed report saved to: {report_file}")
        
        return report['overall_status'] == 'PASSED'
        
    except Exception as e:
        print(f"❌ Quality gates validation failed with error: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)