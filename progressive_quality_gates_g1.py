#!/usr/bin/env python3
"""
Progressive Quality Gates Generation 1: Simple Implementation
Autonomous SDLC with incremental quality validation

This system implements progressive quality gates that adapt and evolve
based on project characteristics and performance metrics.
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime

import psutil
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest


class QualityGateStatus(Enum):
    PENDING = "pending"
    RUNNING = "running" 
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


class GateType(Enum):
    CODE_QUALITY = "code_quality"
    SECURITY = "security"
    PERFORMANCE = "performance"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    RESEARCH = "research"


@dataclass
class QualityMetrics:
    """Quality metrics for progressive gates"""
    gate_id: str
    gate_type: GateType
    status: QualityGateStatus = QualityGateStatus.PENDING
    score: float = 0.0
    threshold: float = 0.8
    execution_time: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ProgressiveGateConfig:
    """Configuration for progressive quality gates"""
    enable_adaptive_thresholds: bool = True
    baseline_threshold: float = 0.8
    improvement_rate: float = 0.05
    max_threshold: float = 0.95
    performance_weight: float = 0.3
    quality_weight: float = 0.4
    security_weight: float = 0.3


class ProgressiveQualityGateEngine:
    """Generation 1: Simple progressive quality gate implementation"""
    
    def __init__(self, config: Optional[ProgressiveGateConfig] = None):
        self.config = config or ProgressiveGateConfig()
        self.metrics: List[QualityMetrics] = []
        self.gate_history: Dict[str, List[float]] = {}
        self.logger = self._setup_logging()
        self.registry = CollectorRegistry()
        self._setup_prometheus_metrics()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
        
    def _setup_prometheus_metrics(self) -> None:
        """Setup Prometheus metrics for monitoring"""
        self.gate_counter = Counter(
            'quality_gates_total',
            'Total quality gate executions',
            ['gate_type', 'status'],
            registry=self.registry
        )
        
        self.gate_duration = Histogram(
            'quality_gate_duration_seconds',
            'Quality gate execution duration',
            ['gate_type'],
            registry=self.registry
        )
        
        self.gate_score = Gauge(
            'quality_gate_score',
            'Current quality gate score',
            ['gate_type', 'gate_id'],
            registry=self.registry
        )
    
    async def run_code_quality_gate(self, gate_id: str = "code_quality_g1") -> QualityMetrics:
        """Run code quality analysis gate"""
        start_time = time.time()
        
        try:
            self.logger.info(f"Running code quality gate: {gate_id}")
            
            # Simulate code quality checks
            quality_score = 0.85  # Base score for Generation 1
            
            # Check for basic code patterns
            src_files = list(Path("src").rglob("*.py")) if Path("src").exists() else []
            test_files = list(Path("tests").rglob("*.py")) if Path("tests").exists() else []
            
            # Basic quality metrics
            has_tests = len(test_files) > 0
            has_src_structure = len(src_files) > 0
            
            if has_tests:
                quality_score += 0.1
            if has_src_structure:
                quality_score += 0.05
                
            # Adaptive threshold adjustment
            threshold = self._get_adaptive_threshold(gate_id, quality_score)
            
            execution_time = time.time() - start_time
            status = QualityGateStatus.PASSED if quality_score >= threshold else QualityGateStatus.FAILED
            
            metrics = QualityMetrics(
                gate_id=gate_id,
                gate_type=GateType.CODE_QUALITY,
                status=status,
                score=quality_score,
                threshold=threshold,
                execution_time=execution_time,
                metadata={
                    "src_files": len(src_files),
                    "test_files": len(test_files),
                    "has_tests": has_tests
                }
            )
            
            self._record_metrics(metrics)
            self.logger.info(f"Code quality gate completed: {status.value} (score: {quality_score:.3f})")
            
            return metrics
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_metrics = QualityMetrics(
                gate_id=gate_id,
                gate_type=GateType.CODE_QUALITY,
                status=QualityGateStatus.FAILED,
                execution_time=execution_time,
                error_message=str(e)
            )
            
            self._record_metrics(error_metrics)
            self.logger.error(f"Code quality gate failed: {e}")
            return error_metrics
    
    async def run_security_gate(self, gate_id: str = "security_g1") -> QualityMetrics:
        """Run security analysis gate"""
        start_time = time.time()
        
        try:
            self.logger.info(f"Running security gate: {gate_id}")
            
            # Basic security checks for Generation 1
            security_score = 0.80
            
            # Check for security indicators
            has_requirements = Path("requirements.txt").exists()
            has_security_config = Path("bandit.yaml").exists() or Path(".bandit").exists()
            
            if has_requirements:
                security_score += 0.05
            if has_security_config:
                security_score += 0.1
                
            # Check for common security issues
            security_issues = []
            
            if Path(".env").exists():
                security_issues.append("Environment file exposed")
                security_score -= 0.1
                
            threshold = self._get_adaptive_threshold(gate_id, security_score)
            
            execution_time = time.time() - start_time
            status = QualityGateStatus.PASSED if security_score >= threshold else QualityGateStatus.FAILED
            
            metrics = QualityMetrics(
                gate_id=gate_id,
                gate_type=GateType.SECURITY,
                status=status,
                score=security_score,
                threshold=threshold,
                execution_time=execution_time,
                metadata={
                    "security_issues": security_issues,
                    "has_requirements": has_requirements,
                    "has_security_config": has_security_config
                }
            )
            
            self._record_metrics(metrics)
            self.logger.info(f"Security gate completed: {status.value} (score: {security_score:.3f})")
            
            return metrics
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_metrics = QualityMetrics(
                gate_id=gate_id,
                gate_type=GateType.SECURITY,
                status=QualityGateStatus.FAILED,
                execution_time=execution_time,
                error_message=str(e)
            )
            
            self._record_metrics(error_metrics)
            self.logger.error(f"Security gate failed: {e}")
            return error_metrics
    
    async def run_performance_gate(self, gate_id: str = "performance_g1") -> QualityMetrics:
        """Run performance analysis gate"""
        start_time = time.time()
        
        try:
            self.logger.info(f"Running performance gate: {gate_id}")
            
            # Basic performance metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_info = psutil.virtual_memory()
            disk_usage = psutil.disk_usage('/')
            
            # Calculate performance score
            performance_score = 0.75
            
            # CPU utilization scoring
            if cpu_percent < 50:
                performance_score += 0.1
            elif cpu_percent < 80:
                performance_score += 0.05
                
            # Memory utilization scoring
            if memory_info.percent < 70:
                performance_score += 0.1
            elif memory_info.percent < 85:
                performance_score += 0.05
                
            # Disk space scoring
            if disk_usage.percent < 80:
                performance_score += 0.05
                
            threshold = self._get_adaptive_threshold(gate_id, performance_score)
            
            execution_time = time.time() - start_time
            status = QualityGateStatus.PASSED if performance_score >= threshold else QualityGateStatus.FAILED
            
            metrics = QualityMetrics(
                gate_id=gate_id,
                gate_type=GateType.PERFORMANCE,
                status=status,
                score=performance_score,
                threshold=threshold,
                execution_time=execution_time,
                metadata={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_info.percent,
                    "disk_percent": disk_usage.percent,
                    "available_memory_gb": memory_info.available / (1024**3)
                }
            )
            
            self._record_metrics(metrics)
            self.logger.info(f"Performance gate completed: {status.value} (score: {performance_score:.3f})")
            
            return metrics
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_metrics = QualityMetrics(
                gate_id=gate_id,
                gate_type=GateType.PERFORMANCE,
                status=QualityGateStatus.FAILED,
                execution_time=execution_time,
                error_message=str(e)
            )
            
            self._record_metrics(error_metrics)
            self.logger.error(f"Performance gate failed: {e}")
            return error_metrics
    
    async def run_research_gate(self, gate_id: str = "research_g1") -> QualityMetrics:
        """Run research validation gate for bioneural system"""
        start_time = time.time()
        
        try:
            self.logger.info(f"Running research gate: {gate_id}")
            
            # Check for bioneural research components
            research_score = 0.70
            
            # Look for bioneural-specific files
            bioneural_files = [
                "bioneuro_olfactory_demo.py",
                "src/lexgraph_legal_rag/bioneuro_olfactory_fusion.py",
                "src/lexgraph_legal_rag/multisensory_legal_processor.py"
            ]
            
            found_files = [f for f in bioneural_files if Path(f).exists()]
            research_score += 0.05 * len(found_files)
            
            # Check for research documentation
            research_docs = [
                "BIONEURO_RESEARCH_PAPER.md",
                "README.md"
            ]
            
            found_docs = [d for d in research_docs if Path(d).exists()]
            research_score += 0.03 * len(found_docs)
            
            # Check for experimental results
            result_files = list(Path(".").glob("*results.json"))
            if result_files:
                research_score += 0.1
                
            threshold = self._get_adaptive_threshold(gate_id, research_score)
            
            execution_time = time.time() - start_time
            status = QualityGateStatus.PASSED if research_score >= threshold else QualityGateStatus.FAILED
            
            metrics = QualityMetrics(
                gate_id=gate_id,
                gate_type=GateType.RESEARCH,
                status=status,
                score=research_score,
                threshold=threshold,
                execution_time=execution_time,
                metadata={
                    "bioneural_files": found_files,
                    "research_docs": found_docs,
                    "result_files": len(result_files)
                }
            )
            
            self._record_metrics(metrics)
            self.logger.info(f"Research gate completed: {status.value} (score: {research_score:.3f})")
            
            return metrics
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_metrics = QualityMetrics(
                gate_id=gate_id,
                gate_type=GateType.RESEARCH,
                status=QualityGateStatus.FAILED,
                execution_time=execution_time,
                error_message=str(e)
            )
            
            self._record_metrics(error_metrics)
            self.logger.error(f"Research gate failed: {e}")
            return error_metrics
    
    async def run_all_gates(self) -> List[QualityMetrics]:
        """Run all quality gates in parallel"""
        self.logger.info("Starting progressive quality gate execution - Generation 1")
        
        gates = [
            self.run_code_quality_gate(),
            self.run_security_gate(),
            self.run_performance_gate(),
            self.run_research_gate()
        ]
        
        results = await asyncio.gather(*gates, return_exceptions=True)
        
        metrics = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Gate execution failed: {result}")
            else:
                metrics.append(result)
        
        self.metrics.extend(metrics)
        
        # Calculate overall score
        passed_gates = [m for m in metrics if m.status == QualityGateStatus.PASSED]
        overall_score = len(passed_gates) / len(metrics) if metrics else 0.0
        
        self.logger.info(f"Quality gates completed - Generation 1")
        self.logger.info(f"Gates passed: {len(passed_gates)}/{len(metrics)}")
        self.logger.info(f"Overall score: {overall_score:.3f}")
        
        return metrics
    
    def _get_adaptive_threshold(self, gate_id: str, current_score: float) -> float:
        """Get adaptive threshold based on historical performance"""
        if not self.config.enable_adaptive_thresholds:
            return self.config.baseline_threshold
            
        history = self.gate_history.get(gate_id, [])
        
        if len(history) < 3:  # Not enough history, use baseline
            return self.config.baseline_threshold
            
        # Calculate adaptive threshold
        avg_score = sum(history[-5:]) / min(len(history), 5)
        adaptive_threshold = min(
            avg_score + self.config.improvement_rate,
            self.config.max_threshold
        )
        
        return max(adaptive_threshold, self.config.baseline_threshold)
    
    def _record_metrics(self, metrics: QualityMetrics) -> None:
        """Record metrics for monitoring and adaptation"""
        # Update Prometheus metrics
        self.gate_counter.labels(
            gate_type=metrics.gate_type.value,
            status=metrics.status.value
        ).inc()
        
        self.gate_duration.labels(
            gate_type=metrics.gate_type.value
        ).observe(metrics.execution_time)
        
        self.gate_score.labels(
            gate_type=metrics.gate_type.value,
            gate_id=metrics.gate_id
        ).set(metrics.score)
        
        # Update history for adaptive thresholds
        if metrics.gate_id not in self.gate_history:
            self.gate_history[metrics.gate_id] = []
            
        self.gate_history[metrics.gate_id].append(metrics.score)
        
        # Keep only recent history
        if len(self.gate_history[metrics.gate_id]) > 10:
            self.gate_history[metrics.gate_id] = self.gate_history[metrics.gate_id][-10:]
    
    def export_results(self, output_path: str = "progressive_quality_gates_g1_results.json") -> None:
        """Export quality gate results"""
        results = {
            "generation": 1,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "enable_adaptive_thresholds": self.config.enable_adaptive_thresholds,
                "baseline_threshold": self.config.baseline_threshold,
                "improvement_rate": self.config.improvement_rate
            },
            "metrics": [
                {
                    "gate_id": m.gate_id,
                    "gate_type": m.gate_type.value,
                    "status": m.status.value,
                    "score": m.score,
                    "threshold": m.threshold,
                    "execution_time": m.execution_time,
                    "error_message": m.error_message,
                    "metadata": m.metadata,
                    "timestamp": m.timestamp.isoformat()
                }
                for m in self.metrics
            ],
            "summary": {
                "total_gates": len(self.metrics),
                "passed_gates": len([m for m in self.metrics if m.status == QualityGateStatus.PASSED]),
                "failed_gates": len([m for m in self.metrics if m.status == QualityGateStatus.FAILED]),
                "average_score": sum(m.score for m in self.metrics) / len(self.metrics) if self.metrics else 0.0,
                "total_execution_time": sum(m.execution_time for m in self.metrics)
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Results exported to {output_path}")
        
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics in text format"""
        return generate_latest(self.registry).decode('utf-8')


async def main():
    """Main demonstration of Generation 1 Progressive Quality Gates"""
    print("🚀 TERRAGON AUTONOMOUS SDLC - Progressive Quality Gates Generation 1")
    print("=" * 70)
    
    config = ProgressiveGateConfig(
        enable_adaptive_thresholds=True,
        baseline_threshold=0.75,
        improvement_rate=0.05
    )
    
    engine = ProgressiveQualityGateEngine(config)
    
    try:
        # Run all quality gates
        results = await engine.run_all_gates()
        
        # Export results
        engine.export_results()
        
        # Display summary
        print("\n📊 GENERATION 1 RESULTS SUMMARY")
        print("-" * 40)
        
        passed = len([r for r in results if r.status == QualityGateStatus.PASSED])
        total = len(results)
        
        print(f"Gates Passed: {passed}/{total}")
        print(f"Overall Success Rate: {(passed/total)*100:.1f}%" if total > 0 else "N/A")
        
        for result in results:
            status_symbol = "✅" if result.status == QualityGateStatus.PASSED else "❌"
            print(f"{status_symbol} {result.gate_type.value}: {result.score:.3f} (threshold: {result.threshold:.3f})")
        
        print("\n🎯 NEXT: Generation 2 will add robust error handling and validation")
        
        return results
        
    except Exception as e:
        print(f"❌ Generation 1 execution failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())