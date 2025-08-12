"""
Advanced Monitoring and Observability for Bioneural Olfactory Fusion System

Provides comprehensive monitoring, alerting, and performance tracking for
the multi-sensory legal document analysis pipeline.

Features:
- Real-time performance metrics
- Bioneural signal quality monitoring
- Anomaly detection for scent profiles
- Advanced alerting for system health
- Research metrics for academic validation
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from collections import defaultdict, deque
import statistics
import json
from datetime import datetime, timedelta
import threading

import numpy as np
from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics tracked."""
    PERFORMANCE = "performance"
    QUALITY = "quality"
    ACCURACY = "accuracy"
    SYSTEM_HEALTH = "system_health"
    RESEARCH = "research"


@dataclass
class BioneuroAlert:
    """Represents an alert for bioneural system monitoring."""
    timestamp: datetime
    severity: AlertSeverity
    component: str
    metric: str
    value: float
    threshold: float
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Performance metrics for bioneural processing."""
    document_processing_time: float
    olfactory_analysis_time: float
    multisensory_fusion_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    throughput_docs_per_second: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            "document_processing_time": self.document_processing_time,
            "olfactory_analysis_time": self.olfactory_analysis_time,
            "multisensory_fusion_time": self.multisensory_fusion_time,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_usage_percent": self.cpu_usage_percent,
            "throughput_docs_per_second": self.throughput_docs_per_second
        }


@dataclass
class QualityMetrics:
    """Quality metrics for bioneural analysis."""
    signal_strength_avg: float
    signal_confidence_avg: float
    receptor_activation_rate: float
    scent_profile_completeness: float
    fusion_coherence_score: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            "signal_strength_avg": self.signal_strength_avg,
            "signal_confidence_avg": self.signal_confidence_avg,
            "receptor_activation_rate": self.receptor_activation_rate,
            "scent_profile_completeness": self.scent_profile_completeness,
            "fusion_coherence_score": self.fusion_coherence_score
        }


class BioneuroMetricsCollector:
    """Collects and aggregates metrics from bioneural processing."""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """Initialize metrics collector."""
        self.registry = registry or CollectorRegistry()
        self.processing_times: deque = deque(maxlen=1000)
        self.quality_scores: deque = deque(maxlen=1000)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.alert_handlers: List[Callable[[BioneuroAlert], None]] = []
        
        # Prometheus metrics
        self._setup_prometheus_metrics()
        
        # Alert thresholds
        self.alert_thresholds = {
            "processing_time_p95": 5.0,  # seconds
            "memory_usage_mb": 1000.0,   # MB
            "cpu_usage_percent": 80.0,   # percent
            "signal_strength_min": 0.1,  # minimum acceptable strength
            "confidence_min": 0.3,       # minimum acceptable confidence
            "error_rate": 0.05           # maximum error rate (5%)
        }
        
        self.logger = logging.getLogger(__name__)
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics for monitoring."""
        self.prom_document_processing_time = Histogram(
            'bioneuro_document_processing_seconds',
            'Time spent processing documents through bioneural pipeline',
            registry=self.registry
        )
        
        self.prom_olfactory_analysis_time = Histogram(
            'bioneuro_olfactory_analysis_seconds',
            'Time spent on olfactory fusion analysis',
            registry=self.registry
        )
        
        self.prom_multisensory_fusion_time = Histogram(
            'bioneuro_multisensory_fusion_seconds',
            'Time spent on multi-sensory fusion',
            registry=self.registry
        )
        
        self.prom_signal_strength = Gauge(
            'bioneuro_signal_strength_average',
            'Average signal strength across all receptors',
            registry=self.registry
        )
        
        self.prom_signal_confidence = Gauge(
            'bioneuro_signal_confidence_average',
            'Average signal confidence across all receptors',
            registry=self.registry
        )
        
        self.prom_receptor_activations = Counter(
            'bioneuro_receptor_activations_total',
            'Total number of receptor activations by type',
            ['receptor_type'],
            registry=self.registry
        )
        
        self.prom_documents_processed = Counter(
            'bioneuro_documents_processed_total',
            'Total number of documents processed',
            ['status'],
            registry=self.registry
        )
        
        self.prom_errors = Counter(
            'bioneuro_errors_total',
            'Total number of errors by component',
            ['component', 'error_type'],
            registry=self.registry
        )
        
        self.prom_memory_usage = Gauge(
            'bioneuro_memory_usage_bytes',
            'Memory usage of bioneural components',
            registry=self.registry
        )
        
        self.prom_scent_similarity = Summary(
            'bioneuro_scent_similarity_score',
            'Distribution of scent similarity scores',
            registry=self.registry
        )
    
    def record_document_processing(self, processing_time: float, success: bool = True):
        """Record document processing metrics."""
        self.processing_times.append(processing_time)
        self.prom_document_processing_time.observe(processing_time)
        
        status = "success" if success else "error"
        self.prom_documents_processed.labels(status=status).inc()
        
        # Check for performance alerts
        if processing_time > self.alert_thresholds["processing_time_p95"]:
            self._trigger_alert(
                AlertSeverity.WARNING,
                "performance",
                "processing_time",
                processing_time,
                self.alert_thresholds["processing_time_p95"],
                f"Document processing time exceeded threshold: {processing_time:.2f}s"
            )
    
    def record_olfactory_analysis(self, analysis_time: float, signal_strength: float, 
                                 confidence: float, receptor_activations: Dict[str, int]):
        """Record olfactory analysis metrics."""
        self.prom_olfactory_analysis_time.observe(analysis_time)
        self.prom_signal_strength.set(signal_strength)
        self.prom_signal_confidence.set(confidence)
        
        # Record receptor activations
        for receptor_type, count in receptor_activations.items():
            self.prom_receptor_activations.labels(receptor_type=receptor_type).inc(count)
        
        # Quality checks
        if signal_strength < self.alert_thresholds["signal_strength_min"]:
            self._trigger_alert(
                AlertSeverity.WARNING,
                "quality",
                "signal_strength",
                signal_strength,
                self.alert_thresholds["signal_strength_min"],
                f"Low signal strength detected: {signal_strength:.3f}"
            )
        
        if confidence < self.alert_thresholds["confidence_min"]:
            self._trigger_alert(
                AlertSeverity.WARNING,
                "quality", 
                "confidence",
                confidence,
                self.alert_thresholds["confidence_min"],
                f"Low confidence detected: {confidence:.3f}"
            )
    
    def record_multisensory_fusion(self, fusion_time: float, coherence_score: float):
        """Record multi-sensory fusion metrics."""
        self.prom_multisensory_fusion_time.observe(fusion_time)
        
        # Track fusion quality
        self.quality_scores.append(coherence_score)
    
    def record_scent_similarity(self, similarity_score: float):
        """Record scent similarity metrics."""
        self.prom_scent_similarity.observe(similarity_score)
    
    def record_error(self, component: str, error_type: str, error_message: str):
        """Record error occurrence."""
        self.error_counts[f"{component}:{error_type}"] += 1
        self.prom_errors.labels(component=component, error_type=error_type).inc()
        
        # Calculate error rate
        total_errors = sum(self.error_counts.values())
        total_processed = len(self.processing_times)
        
        if total_processed > 0:
            error_rate = total_errors / total_processed
            
            if error_rate > self.alert_thresholds["error_rate"]:
                self._trigger_alert(
                    AlertSeverity.ERROR,
                    component,
                    "error_rate",
                    error_rate,
                    self.alert_thresholds["error_rate"],
                    f"High error rate detected: {error_rate:.1%}"
                )
    
    def record_system_resources(self, memory_mb: float, cpu_percent: float):
        """Record system resource usage."""
        self.prom_memory_usage.set(memory_mb * 1024 * 1024)  # Convert to bytes
        
        # Resource usage alerts
        if memory_mb > self.alert_thresholds["memory_usage_mb"]:
            self._trigger_alert(
                AlertSeverity.WARNING,
                "system",
                "memory_usage",
                memory_mb,
                self.alert_thresholds["memory_usage_mb"],
                f"High memory usage: {memory_mb:.1f}MB"
            )
        
        if cpu_percent > self.alert_thresholds["cpu_usage_percent"]:
            self._trigger_alert(
                AlertSeverity.WARNING,
                "system",
                "cpu_usage",
                cpu_percent,
                self.alert_thresholds["cpu_usage_percent"],
                f"High CPU usage: {cpu_percent:.1f}%"
            )
    
    def get_performance_summary(self) -> PerformanceMetrics:
        """Get current performance metrics summary."""
        if not self.processing_times:
            return PerformanceMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        avg_processing_time = statistics.mean(self.processing_times)
        
        # Get current resource usage
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        cpu_percent = process.cpu_percent()
        
        # Calculate throughput
        recent_times = list(self.processing_times)[-100:]  # Last 100 documents
        if len(recent_times) > 1:
            throughput = len(recent_times) / sum(recent_times)
        else:
            throughput = 0.0
        
        return PerformanceMetrics(
            document_processing_time=avg_processing_time,
            olfactory_analysis_time=0.0,  # Would be calculated separately
            multisensory_fusion_time=0.0,  # Would be calculated separately
            memory_usage_mb=memory_mb,
            cpu_usage_percent=cpu_percent,
            throughput_docs_per_second=throughput
        )
    
    def get_quality_summary(self) -> QualityMetrics:
        """Get current quality metrics summary."""
        if not self.quality_scores:
            return QualityMetrics(0.0, 0.0, 0.0, 0.0, 0.0)
        
        avg_coherence = statistics.mean(self.quality_scores)
        
        # These would be calculated from collected data
        return QualityMetrics(
            signal_strength_avg=0.0,  # Would be tracked separately
            signal_confidence_avg=0.0,  # Would be tracked separately
            receptor_activation_rate=0.0,  # Would be calculated
            scent_profile_completeness=0.0,  # Would be calculated
            fusion_coherence_score=avg_coherence
        )
    
    def add_alert_handler(self, handler: Callable[[BioneuroAlert], None]):
        """Add an alert handler function."""
        self.alert_handlers.append(handler)
    
    def _trigger_alert(self, severity: AlertSeverity, component: str, metric: str,
                      value: float, threshold: float, message: str):
        """Trigger an alert."""
        alert = BioneuroAlert(
            timestamp=datetime.now(),
            severity=severity,
            component=component,
            metric=metric,
            value=value,
            threshold=threshold,
            message=message
        )
        
        # Log the alert
        log_level = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.ERROR: logging.ERROR,
            AlertSeverity.CRITICAL: logging.CRITICAL
        }[severity]
        
        self.logger.log(log_level, f"BIONEURAL ALERT [{severity.value.upper()}] {component}: {message}")
        
        # Notify alert handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"Alert handler failed: {e}")


class BioneuroAnomalyDetector:
    """Detects anomalies in bioneural processing patterns."""
    
    def __init__(self, window_size: int = 100):
        """Initialize anomaly detector."""
        self.window_size = window_size
        self.signal_patterns: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.baseline_statistics: Dict[str, Dict[str, float]] = {}
        self.anomaly_threshold = 2.0  # Number of standard deviations
        
        self.logger = logging.getLogger(__name__)
    
    def update_signal_pattern(self, receptor_type: str, intensity: float, confidence: float):
        """Update signal patterns for anomaly detection."""
        combined_signal = intensity * confidence  # Combined signal strength
        self.signal_patterns[receptor_type].append(combined_signal)
        
        # Update baseline statistics if we have enough data
        if len(self.signal_patterns[receptor_type]) >= 20:
            signals = list(self.signal_patterns[receptor_type])
            self.baseline_statistics[receptor_type] = {
                "mean": statistics.mean(signals),
                "stdev": statistics.stdev(signals) if len(signals) > 1 else 0.0,
                "min": min(signals),
                "max": max(signals)
            }
    
    def detect_anomaly(self, receptor_type: str, intensity: float, confidence: float) -> Optional[Dict[str, Any]]:
        """Detect if current signal represents an anomaly."""
        if receptor_type not in self.baseline_statistics:
            return None  # Not enough baseline data
        
        combined_signal = intensity * confidence
        baseline = self.baseline_statistics[receptor_type]
        
        # Z-score anomaly detection
        if baseline["stdev"] > 0:
            z_score = abs(combined_signal - baseline["mean"]) / baseline["stdev"]
            
            if z_score > self.anomaly_threshold:
                return {
                    "type": "signal_anomaly",
                    "receptor_type": receptor_type,
                    "current_signal": combined_signal,
                    "expected_mean": baseline["mean"],
                    "z_score": z_score,
                    "severity": "high" if z_score > 3.0 else "medium"
                }
        
        return None
    
    def detect_pattern_anomalies(self) -> List[Dict[str, Any]]:
        """Detect pattern-based anomalies across all receptors."""
        anomalies = []
        
        for receptor_type, signals in self.signal_patterns.items():
            if len(signals) < 10:
                continue
            
            recent_signals = list(signals)[-10:]  # Last 10 signals
            older_signals = list(signals)[:-10] if len(signals) > 10 else []
            
            if not older_signals:
                continue
            
            # Compare recent vs historical patterns
            recent_mean = statistics.mean(recent_signals)
            historical_mean = statistics.mean(older_signals)
            
            # Significant change in pattern
            if abs(recent_mean - historical_mean) > historical_mean * 0.5:  # 50% change
                anomalies.append({
                    "type": "pattern_shift",
                    "receptor_type": receptor_type,
                    "recent_mean": recent_mean,
                    "historical_mean": historical_mean,
                    "change_percent": abs(recent_mean - historical_mean) / historical_mean * 100
                })
        
        return anomalies


class BioneuroHealthChecker:
    """Monitors overall health of bioneural system components."""
    
    def __init__(self, metrics_collector: BioneuroMetricsCollector):
        """Initialize health checker."""
        self.metrics_collector = metrics_collector
        self.health_checks: Dict[str, Callable[[], bool]] = {}
        self.last_check_time = time.time()
        self.check_interval = 60.0  # Check every minute
        
        self.logger = logging.getLogger(__name__)
        
        # Register default health checks
        self._register_default_checks()
    
    def _register_default_checks(self):
        """Register default health check functions."""
        self.health_checks["memory_usage"] = self._check_memory_usage
        self.health_checks["processing_performance"] = self._check_processing_performance
        self.health_checks["error_rate"] = self._check_error_rate
        self.health_checks["signal_quality"] = self._check_signal_quality
    
    def _check_memory_usage(self) -> bool:
        """Check if memory usage is within acceptable limits."""
        try:
            import psutil
            memory_percent = psutil.virtual_memory().percent
            return memory_percent < 90.0  # Less than 90% memory usage
        except Exception:
            return True  # Assume healthy if can't check
    
    def _check_processing_performance(self) -> bool:
        """Check if processing performance is acceptable."""
        if not self.metrics_collector.processing_times:
            return True  # No data yet, assume healthy
        
        recent_times = list(self.metrics_collector.processing_times)[-10:]
        avg_time = statistics.mean(recent_times)
        
        return avg_time < 10.0  # Less than 10 seconds average
    
    def _check_error_rate(self) -> bool:
        """Check if error rate is acceptable."""
        total_errors = sum(self.metrics_collector.error_counts.values())
        total_processed = len(self.metrics_collector.processing_times)
        
        if total_processed == 0:
            return True  # No processing yet
        
        error_rate = total_errors / total_processed
        return error_rate < 0.1  # Less than 10% error rate
    
    def _check_signal_quality(self) -> bool:
        """Check if signal quality is acceptable."""
        if not self.metrics_collector.quality_scores:
            return True  # No quality data yet
        
        recent_scores = list(self.metrics_collector.quality_scores)[-10:]
        avg_quality = statistics.mean(recent_scores)
        
        return avg_quality > 0.3  # Quality score above 0.3
    
    def register_health_check(self, name: str, check_func: Callable[[], bool]):
        """Register a custom health check function."""
        self.health_checks[name] = check_func
    
    def run_health_checks(self) -> Dict[str, Any]:
        """Run all health checks and return results."""
        results = {
            "timestamp": datetime.now().isoformat(),
            "overall_healthy": True,
            "checks": {}
        }
        
        for check_name, check_func in self.health_checks.items():
            try:
                check_result = check_func()
                results["checks"][check_name] = {
                    "status": "pass" if check_result else "fail",
                    "healthy": check_result
                }
                
                if not check_result:
                    results["overall_healthy"] = False
                    
            except Exception as e:
                self.logger.error(f"Health check '{check_name}' failed with error: {e}")
                results["checks"][check_name] = {
                    "status": "error",
                    "healthy": False,
                    "error": str(e)
                }
                results["overall_healthy"] = False
        
        return results
    
    def should_run_checks(self) -> bool:
        """Check if it's time to run health checks."""
        return time.time() - self.last_check_time >= self.check_interval
    
    def run_periodic_checks(self):
        """Run health checks if it's time."""
        if self.should_run_checks():
            health_status = self.run_health_checks()
            self.last_check_time = time.time()
            
            if not health_status["overall_healthy"]:
                self.logger.warning("Bioneural system health check failed", extra=health_status)
            
            return health_status
        
        return None


class BioneuroResearchMetrics:
    """Specialized metrics collection for research validation and publication."""
    
    def __init__(self):
        """Initialize research metrics collector."""
        self.experiment_data: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.baseline_comparisons: Dict[str, Dict[str, float]] = {}
        self.statistical_tests: Dict[str, Dict[str, Any]] = {}
        
        self.logger = logging.getLogger(__name__)
    
    def record_experiment_result(self, experiment_name: str, method: str, 
                               accuracy: float, precision: float, recall: float,
                               f1_score: float, processing_time: float,
                               metadata: Optional[Dict[str, Any]] = None):
        """Record experimental results for academic validation."""
        result = {
            "timestamp": datetime.now().isoformat(),
            "method": method,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "processing_time": processing_time,
            "metadata": metadata or {}
        }
        
        self.experiment_data[experiment_name].append(result)
        
        self.logger.info(f"Recorded research result for {experiment_name}: "
                        f"method={method}, accuracy={accuracy:.3f}, f1={f1_score:.3f}")
    
    def set_baseline_performance(self, experiment_name: str, baseline_metrics: Dict[str, float]):
        """Set baseline performance metrics for comparison."""
        self.baseline_comparisons[experiment_name] = baseline_metrics
        
        self.logger.info(f"Set baseline for {experiment_name}: {baseline_metrics}")
    
    def calculate_statistical_significance(self, experiment_name: str, 
                                        method_a: str, method_b: str) -> Dict[str, Any]:
        """Calculate statistical significance between two methods."""
        if experiment_name not in self.experiment_data:
            return {"error": "No experimental data found"}
        
        data = self.experiment_data[experiment_name]
        
        method_a_results = [r for r in data if r["method"] == method_a]
        method_b_results = [r for r in data if r["method"] == method_b]
        
        if len(method_a_results) < 3 or len(method_b_results) < 3:
            return {"error": "Insufficient data for statistical testing"}
        
        # Extract accuracy scores for comparison
        a_scores = [r["accuracy"] for r in method_a_results]
        b_scores = [r["accuracy"] for r in method_b_results]
        
        # Perform t-test (simplified)
        try:
            from scipy import stats
            t_stat, p_value = stats.ttest_ind(a_scores, b_scores)
            
            result = {
                "method_a": method_a,
                "method_b": method_b,
                "method_a_mean": statistics.mean(a_scores),
                "method_b_mean": statistics.mean(b_scores),
                "method_a_std": statistics.stdev(a_scores),
                "method_b_std": statistics.stdev(b_scores),
                "t_statistic": t_stat,
                "p_value": p_value,
                "significant": p_value < 0.05,
                "effect_size": abs(statistics.mean(a_scores) - statistics.mean(b_scores)) / 
                             statistics.stdev(a_scores + b_scores)
            }
            
            self.statistical_tests[f"{experiment_name}_{method_a}_vs_{method_b}"] = result
            return result
            
        except ImportError:
            # Fallback to simple comparison if scipy not available
            a_mean = statistics.mean(a_scores)
            b_mean = statistics.mean(b_scores)
            
            return {
                "method_a": method_a,
                "method_b": method_b,
                "method_a_mean": a_mean,
                "method_b_mean": b_mean,
                "improvement": (b_mean - a_mean) / a_mean * 100,
                "note": "Statistical testing requires scipy for full analysis"
            }
    
    def generate_research_report(self, experiment_name: str) -> Dict[str, Any]:
        """Generate comprehensive research report for publication."""
        if experiment_name not in self.experiment_data:
            return {"error": "No experimental data found"}
        
        data = self.experiment_data[experiment_name]
        
        # Aggregate results by method
        method_results = defaultdict(list)
        for result in data:
            method_results[result["method"]].append(result)
        
        # Calculate summary statistics for each method
        method_summaries = {}
        for method, results in method_results.items():
            accuracies = [r["accuracy"] for r in results]
            f1_scores = [r["f1_score"] for r in results]
            times = [r["processing_time"] for r in results]
            
            method_summaries[method] = {
                "n_samples": len(results),
                "accuracy_mean": statistics.mean(accuracies),
                "accuracy_std": statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0,
                "accuracy_min": min(accuracies),
                "accuracy_max": max(accuracies),
                "f1_score_mean": statistics.mean(f1_scores),
                "f1_score_std": statistics.stdev(f1_scores) if len(f1_scores) > 1 else 0.0,
                "processing_time_mean": statistics.mean(times),
                "processing_time_std": statistics.stdev(times) if len(times) > 1 else 0.0
            }
        
        # Compare with baseline if available
        baseline_comparison = None
        if experiment_name in self.baseline_comparisons:
            baseline = self.baseline_comparisons[experiment_name]
            
            # Find best performing method
            best_method = max(method_summaries.items(), 
                            key=lambda x: x[1]["accuracy_mean"])
            
            baseline_comparison = {
                "baseline_accuracy": baseline.get("accuracy", 0.0),
                "best_method": best_method[0],
                "best_accuracy": best_method[1]["accuracy_mean"],
                "improvement": (best_method[1]["accuracy_mean"] - baseline.get("accuracy", 0.0)) / baseline.get("accuracy", 1.0) * 100
            }
        
        report = {
            "experiment_name": experiment_name,
            "total_samples": len(data),
            "methods_tested": list(method_summaries.keys()),
            "method_summaries": method_summaries,
            "baseline_comparison": baseline_comparison,
            "statistical_tests": {k: v for k, v in self.statistical_tests.items() if experiment_name in k},
            "generated_at": datetime.now().isoformat()
        }
        
        return report
    
    def export_for_publication(self, experiment_name: str, format: str = "json") -> str:
        """Export research data in publication-ready format."""
        report = self.generate_research_report(experiment_name)
        
        if format == "json":
            return json.dumps(report, indent=2)
        elif format == "csv":
            # Convert to CSV format for statistical analysis
            import io
            import csv
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow(["method", "accuracy", "precision", "recall", "f1_score", "processing_time"])
            
            # Write data
            for result in self.experiment_data[experiment_name]:
                writer.writerow([
                    result["method"],
                    result["accuracy"],
                    result["precision"],
                    result["recall"],
                    result["f1_score"],
                    result["processing_time"]
                ])
            
            return output.getvalue()
        else:
            raise ValueError(f"Unsupported export format: {format}")


# Global monitoring instances
_metrics_collector: Optional[BioneuroMetricsCollector] = None
_anomaly_detector: Optional[BioneuroAnomalyDetector] = None
_health_checker: Optional[BioneuroHealthChecker] = None
_research_metrics: Optional[BioneuroResearchMetrics] = None


def get_metrics_collector() -> BioneuroMetricsCollector:
    """Get or create global metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = BioneuroMetricsCollector()
    return _metrics_collector


def get_anomaly_detector() -> BioneuroAnomalyDetector:
    """Get or create global anomaly detector."""
    global _anomaly_detector
    if _anomaly_detector is None:
        _anomaly_detector = BioneuroAnomalyDetector()
    return _anomaly_detector


def get_health_checker() -> BioneuroHealthChecker:
    """Get or create global health checker."""
    global _health_checker
    if _health_checker is None:
        _health_checker = BioneuroHealthChecker(get_metrics_collector())
    return _health_checker


def get_research_metrics() -> BioneuroResearchMetrics:
    """Get or create global research metrics collector."""
    global _research_metrics
    if _research_metrics is None:
        _research_metrics = BioneuroResearchMetrics()
    return _research_metrics


def setup_monitoring_dashboard():
    """Setup monitoring dashboard with default configuration."""
    metrics_collector = get_metrics_collector()
    
    # Add default alert handlers
    def log_alert_handler(alert: BioneuroAlert):
        logger.warning(f"BIONEURAL ALERT: {alert.message}")
    
    def console_alert_handler(alert: BioneuroAlert):
        print(f"🚨 [{alert.severity.value.upper()}] {alert.component}: {alert.message}")
    
    metrics_collector.add_alert_handler(log_alert_handler)
    metrics_collector.add_alert_handler(console_alert_handler)
    
    logger.info("Bioneural monitoring dashboard initialized")