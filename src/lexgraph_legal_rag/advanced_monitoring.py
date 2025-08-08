"""
Advanced Monitoring and Observability System
Comprehensive system monitoring with predictive analytics and automated alerting
"""

import os
import json
import time
import logging
import threading
import statistics
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from pathlib import Path
import psutil
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """Represents a single metric measurement."""
    name: str
    value: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)
    unit: str = ""


@dataclass
class Alert:
    """Represents a monitoring alert."""
    id: str
    name: str
    description: str
    severity: str  # info, warning, critical
    metric_name: str
    threshold: float
    condition: str  # gt, lt, eq
    triggered_at: datetime
    resolved_at: Optional[datetime] = None
    notification_sent: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthCheck:
    """Represents a health check configuration."""
    name: str
    check_function: Callable[[], Dict[str, Any]]
    interval_seconds: float
    timeout_seconds: float
    failure_threshold: int
    success_threshold: int
    current_failures: int = 0
    current_successes: int = 0
    last_check: Optional[datetime] = None
    status: str = "unknown"  # healthy, degraded, unhealthy


@dataclass
class PerformanceBaseline:
    """Performance baseline for anomaly detection."""
    metric_name: str
    mean: float
    std_dev: float
    min_value: float
    max_value: float
    sample_count: int
    calculated_at: datetime
    confidence_interval: Tuple[float, float]


class MetricCollector:
    """Collects and stores time-series metrics."""
    
    def __init__(self, retention_hours: int = 24):
        self.metrics = defaultdict(lambda: deque(maxlen=retention_hours * 3600))  # 1 second resolution
        self.retention_hours = retention_hours
        self.lock = threading.RLock()
        
        # Built-in system metrics
        self.system_collectors = {
            'cpu_percent': lambda: psutil.cpu_percent(interval=0.1),
            'memory_percent': lambda: psutil.virtual_memory().percent,
            'disk_percent': lambda: psutil.disk_usage('/').percent,
            'network_bytes_sent': lambda: psutil.net_io_counters().bytes_sent if hasattr(psutil, 'net_io_counters') else 0,
            'network_bytes_recv': lambda: psutil.net_io_counters().bytes_recv if hasattr(psutil, 'net_io_counters') else 0,
            'process_count': lambda: len(psutil.pids()),
            'load_average': lambda: os.getloadavg()[0] if hasattr(os, 'getloadavg') else 0
        }
        
        # Custom application metrics
        self.app_metrics = {
            'api_requests_total': 0,
            'api_request_duration_sum': 0,
            'api_errors_total': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'search_queries_total': 0,
            'documents_indexed_total': 0
        }
        
        # Start collection thread
        self.collecting = False
        self.collection_thread = None
    
    def start_collection(self, interval: float = 1.0) -> None:
        """Start automated metric collection."""
        if not self.collecting:
            self.collecting = True
            self.collection_thread = threading.Thread(
                target=self._collection_loop,
                args=(interval,),
                daemon=True
            )
            self.collection_thread.start()
            logger.info("Metric collection started")
    
    def stop_collection(self) -> None:
        """Stop automated metric collection."""
        self.collecting = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        logger.info("Metric collection stopped")
    
    def _collection_loop(self, interval: float) -> None:
        """Main metric collection loop."""
        while self.collecting:
            try:
                # Collect system metrics
                for metric_name, collector_func in self.system_collectors.items():
                    try:
                        value = collector_func()
                        self.record_metric(metric_name, value)
                    except Exception as e:
                        logger.warning(f"Failed to collect {metric_name}: {e}")
                
                # Collect application metrics
                for metric_name, value in self.app_metrics.items():
                    self.record_metric(f"app_{metric_name}", value)
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Metric collection error: {e}")
                time.sleep(5)  # Back off on error
    
    def record_metric(self, name: str, value: float, labels: Optional[Dict[str, str]] = None, 
                     unit: str = "") -> None:
        """Record a metric value."""
        with self.lock:
            metric_point = MetricPoint(
                name=name,
                value=value,
                timestamp=time.time(),
                labels=labels or {},
                unit=unit
            )
            self.metrics[name].append(metric_point)
    
    def increment_counter(self, name: str, increment: float = 1.0) -> None:
        """Increment a counter metric."""
        if name in self.app_metrics:
            self.app_metrics[name] += increment
        else:
            self.app_metrics[name] = increment
    
    def get_metric_values(self, name: str, duration_seconds: int = 3600) -> List[float]:
        """Get metric values for the specified duration."""
        with self.lock:
            if name not in self.metrics:
                return []
            
            cutoff_time = time.time() - duration_seconds
            values = []
            
            for metric_point in self.metrics[name]:
                if metric_point.timestamp >= cutoff_time:
                    values.append(metric_point.value)
            
            return values
    
    def get_metric_stats(self, name: str, duration_seconds: int = 3600) -> Dict[str, float]:
        """Get statistical summary of metric values."""
        values = self.get_metric_values(name, duration_seconds)
        
        if not values:
            return {"count": 0}
        
        return {
            "count": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
            "latest": values[-1] if values else 0
        }
    
    def get_all_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all collected metrics."""
        summary = {}
        
        with self.lock:
            for metric_name in self.metrics:
                stats = self.get_metric_stats(metric_name, duration_seconds=300)  # Last 5 minutes
                if stats["count"] > 0:
                    summary[metric_name] = stats
        
        return summary


class AnomalyDetector:
    """Detects anomalies in metric data using statistical methods."""
    
    def __init__(self, sensitivity: float = 2.0):
        self.sensitivity = sensitivity  # Standard deviations for anomaly threshold
        self.baselines = {}
        self.anomalies_detected = deque(maxlen=1000)
        self.lock = threading.RLock()
    
    def calculate_baseline(self, metric_name: str, values: List[float]) -> PerformanceBaseline:
        """Calculate performance baseline for anomaly detection."""
        if len(values) < 10:
            raise ValueError("Need at least 10 data points to calculate baseline")
        
        mean_val = statistics.mean(values)
        std_dev = statistics.stdev(values)
        min_val = min(values)
        max_val = max(values)
        
        # Calculate confidence interval (95%)
        confidence_margin = 1.96 * std_dev / np.sqrt(len(values))
        confidence_interval = (mean_val - confidence_margin, mean_val + confidence_margin)
        
        baseline = PerformanceBaseline(
            metric_name=metric_name,
            mean=mean_val,
            std_dev=std_dev,
            min_value=min_val,
            max_value=max_val,
            sample_count=len(values),
            calculated_at=datetime.now(),
            confidence_interval=confidence_interval
        )
        
        with self.lock:
            self.baselines[metric_name] = baseline
        
        logger.info(f"Baseline calculated for {metric_name}: "
                   f"mean={mean_val:.2f}, std={std_dev:.2f}")
        
        return baseline
    
    def detect_anomaly(self, metric_name: str, current_value: float) -> Optional[Dict[str, Any]]:
        """Detect if current value is anomalous compared to baseline."""
        with self.lock:
            if metric_name not in self.baselines:
                return None
            
            baseline = self.baselines[metric_name]
        
        # Calculate z-score
        z_score = abs(current_value - baseline.mean) / baseline.std_dev if baseline.std_dev > 0 else 0
        
        # Check if anomalous
        is_anomaly = z_score > self.sensitivity
        
        if is_anomaly:
            anomaly = {
                'metric_name': metric_name,
                'current_value': current_value,
                'baseline_mean': baseline.mean,
                'z_score': z_score,
                'severity': 'high' if z_score > 3.0 else 'medium',
                'detected_at': datetime.now().isoformat()
            }
            
            self.anomalies_detected.append(anomaly)
            
            logger.warning(f"Anomaly detected in {metric_name}: "
                          f"value={current_value:.2f}, z-score={z_score:.2f}")
            
            return anomaly
        
        return None
    
    def get_anomaly_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get summary of detected anomalies."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_anomalies = [
            a for a in self.anomalies_detected
            if datetime.fromisoformat(a['detected_at']) > cutoff_time
        ]
        
        if not recent_anomalies:
            return {"anomalies_detected": 0, "time_period_hours": hours}
        
        anomalies_by_metric = defaultdict(int)
        anomalies_by_severity = defaultdict(int)
        
        for anomaly in recent_anomalies:
            anomalies_by_metric[anomaly['metric_name']] += 1
            anomalies_by_severity[anomaly['severity']] += 1
        
        return {
            "anomalies_detected": len(recent_anomalies),
            "time_period_hours": hours,
            "by_metric": dict(anomalies_by_metric),
            "by_severity": dict(anomalies_by_severity),
            "baselines_available": len(self.baselines)
        }


class PredictiveAnalytics:
    """Predictive analytics for capacity planning and trend analysis."""
    
    def __init__(self):
        self.models = {}
        self.predictions_cache = {}
        self.lock = threading.RLock()
    
    def train_prediction_model(self, metric_name: str, values: List[float], 
                             timestamps: List[float]) -> Dict[str, Any]:
        """Train predictive model for a metric."""
        if len(values) < 20:
            raise ValueError("Need at least 20 data points for prediction model")
        
        # Prepare data for training
        X = np.array(timestamps).reshape(-1, 1)
        y = np.array(values)
        
        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train linear regression model
        model = LinearRegression()
        model.fit(X_scaled, y)
        
        # Calculate model performance
        score = model.score(X_scaled, y)
        
        model_info = {
            'model': model,
            'scaler': scaler,
            'score': score,
            'trained_at': datetime.now(),
            'sample_count': len(values),
            'feature_range': (min(timestamps), max(timestamps))
        }
        
        with self.lock:
            self.models[metric_name] = model_info
        
        logger.info(f"Prediction model trained for {metric_name}: "
                   f"R² score={score:.3f}, samples={len(values)}")
        
        return {
            'metric_name': metric_name,
            'r2_score': score,
            'sample_count': len(values),
            'model_accuracy': 'good' if score > 0.7 else 'fair' if score > 0.5 else 'poor'
        }
    
    def predict_future_values(self, metric_name: str, future_minutes: int = 60) -> Optional[Dict[str, Any]]:
        """Predict future values for a metric."""
        with self.lock:
            if metric_name not in self.models:
                return None
            
            model_info = self.models[metric_name]
        
        # Generate future timestamps
        current_time = time.time()
        future_timestamps = [
            current_time + (i * 60)  # Every minute
            for i in range(1, future_minutes + 1)
        ]
        
        # Make predictions
        X_future = np.array(future_timestamps).reshape(-1, 1)
        X_future_scaled = model_info['scaler'].transform(X_future)
        
        predictions = model_info['model'].predict(X_future_scaled)
        
        # Calculate confidence intervals (simplified)
        prediction_std = np.std(predictions)
        confidence_intervals = [
            (pred - 1.96 * prediction_std, pred + 1.96 * prediction_std)
            for pred in predictions
        ]
        
        prediction_result = {
            'metric_name': metric_name,
            'predictions': [
                {
                    'timestamp': ts,
                    'predicted_value': pred,
                    'confidence_interval': ci,
                    'minutes_ahead': i + 1
                }
                for i, (ts, pred, ci) in enumerate(
                    zip(future_timestamps, predictions, confidence_intervals)
                )
            ],
            'model_score': model_info['score'],
            'prediction_horizon_minutes': future_minutes,
            'predicted_at': datetime.now().isoformat()
        }
        
        # Cache prediction
        cache_key = f"{metric_name}_{future_minutes}min"
        with self.lock:
            self.predictions_cache[cache_key] = prediction_result
        
        return prediction_result
    
    def detect_capacity_issues(self, metric_name: str, threshold_percent: float = 80.0) -> Optional[Dict[str, Any]]:
        """Detect potential capacity issues based on predictions."""
        prediction = self.predict_future_values(metric_name, future_minutes=120)  # 2 hours ahead
        
        if not prediction:
            return None
        
        # Check if any predictions exceed threshold
        capacity_alerts = []
        
        for pred_point in prediction['predictions']:
            if pred_point['predicted_value'] > threshold_percent:
                capacity_alerts.append({
                    'timestamp': pred_point['timestamp'],
                    'predicted_value': pred_point['predicted_value'],
                    'threshold': threshold_percent,
                    'minutes_ahead': pred_point['minutes_ahead'],
                    'severity': 'high' if pred_point['predicted_value'] > 90 else 'medium'
                })
        
        if capacity_alerts:
            return {
                'metric_name': metric_name,
                'threshold_percent': threshold_percent,
                'capacity_alerts': capacity_alerts,
                'earliest_alert_minutes': min(alert['minutes_ahead'] for alert in capacity_alerts),
                'detected_at': datetime.now().isoformat()
            }
        
        return None


class AlertManager:
    """Manages monitoring alerts and notifications."""
    
    def __init__(self):
        self.alerts = {}
        self.alert_rules = []
        self.notification_channels = []
        self.lock = threading.RLock()
        
        # Default alert rules
        self._setup_default_alert_rules()
    
    def _setup_default_alert_rules(self) -> None:
        """Setup default alert rules for common metrics."""
        default_rules = [
            {
                'name': 'high_cpu_usage',
                'metric_name': 'cpu_percent',
                'condition': 'gt',
                'threshold': 85.0,
                'severity': 'warning',
                'description': 'CPU usage is above 85%'
            },
            {
                'name': 'high_memory_usage',
                'metric_name': 'memory_percent',
                'condition': 'gt',
                'threshold': 80.0,
                'severity': 'warning',
                'description': 'Memory usage is above 80%'
            },
            {
                'name': 'disk_space_low',
                'metric_name': 'disk_percent',
                'condition': 'gt',
                'threshold': 90.0,
                'severity': 'critical',
                'description': 'Disk usage is above 90%'
            },
            {
                'name': 'high_api_error_rate',
                'metric_name': 'app_api_errors_total',
                'condition': 'gt',
                'threshold': 10.0,
                'severity': 'critical',
                'description': 'API error rate is too high'
            }
        ]
        
        for rule in default_rules:
            self.add_alert_rule(**rule)
    
    def add_alert_rule(self, name: str, metric_name: str, condition: str, 
                      threshold: float, severity: str, description: str) -> None:
        """Add a new alert rule."""
        rule = {
            'name': name,
            'metric_name': metric_name,
            'condition': condition,
            'threshold': threshold,
            'severity': severity,
            'description': description,
            'enabled': True
        }
        
        with self.lock:
            self.alert_rules.append(rule)
        
        logger.info(f"Alert rule added: {name}")
    
    def evaluate_alerts(self, metrics: Dict[str, Any]) -> List[Alert]:
        """Evaluate alert rules against current metrics."""
        triggered_alerts = []
        
        with self.lock:
            for rule in self.alert_rules:
                if not rule['enabled']:
                    continue
                
                metric_name = rule['metric_name']
                if metric_name not in metrics:
                    continue
                
                metric_stats = metrics[metric_name]
                current_value = metric_stats.get('latest', 0)
                
                # Evaluate condition
                condition_met = False
                if rule['condition'] == 'gt' and current_value > rule['threshold']:
                    condition_met = True
                elif rule['condition'] == 'lt' and current_value < rule['threshold']:
                    condition_met = True
                elif rule['condition'] == 'eq' and current_value == rule['threshold']:
                    condition_met = True
                
                if condition_met:
                    alert_id = f"{rule['name']}_{int(time.time())}"
                    
                    alert = Alert(
                        id=alert_id,
                        name=rule['name'],
                        description=rule['description'],
                        severity=rule['severity'],
                        metric_name=metric_name,
                        threshold=rule['threshold'],
                        condition=rule['condition'],
                        triggered_at=datetime.now(),
                        metadata={
                            'current_value': current_value,
                            'rule': rule.copy()
                        }
                    )
                    
                    triggered_alerts.append(alert)
                    self.alerts[alert_id] = alert
                    
                    logger.warning(f"Alert triggered: {rule['name']} - "
                                 f"{metric_name}={current_value} {rule['condition']} {rule['threshold']}")
        
        return triggered_alerts
    
    def get_active_alerts(self) -> List[Alert]:
        """Get currently active (unresolved) alerts."""
        with self.lock:
            return [alert for alert in self.alerts.values() if not alert.resolved_at]
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        with self.lock:
            if alert_id in self.alerts:
                self.alerts[alert_id].resolved_at = datetime.now()
                logger.info(f"Alert resolved: {alert_id}")
                return True
        return False


class AdvancedMonitoringSystem:
    """Main advanced monitoring and observability system."""
    
    def __init__(self, repo_path: str = "/root/repo"):
        self.repo_path = Path(repo_path)
        self.monitoring_dir = self.repo_path / "monitoring"
        self.monitoring_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.metric_collector = MetricCollector()
        self.anomaly_detector = AnomalyDetector()
        self.predictive_analytics = PredictiveAnalytics()
        self.alert_manager = AlertManager()
        
        # Health checks
        self.health_checks = {}
        self._setup_default_health_checks()
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread = None
    
    def _setup_default_health_checks(self) -> None:
        """Setup default health checks."""
        health_checks = [
            HealthCheck(
                name="api_health",
                check_function=self._check_api_health,
                interval_seconds=30,
                timeout_seconds=5,
                failure_threshold=3,
                success_threshold=2
            ),
            HealthCheck(
                name="database_health",
                check_function=self._check_database_health,
                interval_seconds=60,
                timeout_seconds=10,
                failure_threshold=2,
                success_threshold=1
            ),
            HealthCheck(
                name="disk_space",
                check_function=self._check_disk_space,
                interval_seconds=300,  # 5 minutes
                timeout_seconds=5,
                failure_threshold=1,
                success_threshold=1
            )
        ]
        
        for check in health_checks:
            self.health_checks[check.name] = check
    
    def _check_api_health(self) -> Dict[str, Any]:
        """Check API health."""
        # Simulate API health check
        import random
        response_time = random.uniform(0.1, 0.5)
        
        return {
            'status': 'healthy' if response_time < 2.0 else 'unhealthy',
            'response_time': response_time,
            'details': {'endpoint': '/health', 'method': 'GET'}
        }
    
    def _check_database_health(self) -> Dict[str, Any]:
        """Check database health."""
        # Simulate database health check
        return {
            'status': 'healthy',
            'connection_pool': {'active': 5, 'idle': 10, 'max': 20},
            'query_performance': {'avg_response_ms': 45}
        }
    
    def _check_disk_space(self) -> Dict[str, Any]:
        """Check disk space."""
        disk_usage = psutil.disk_usage('/')
        free_percent = (disk_usage.free / disk_usage.total) * 100
        
        return {
            'status': 'healthy' if free_percent > 10 else 'unhealthy',
            'free_percent': free_percent,
            'total_gb': disk_usage.total / (1024**3),
            'used_gb': disk_usage.used / (1024**3)
        }
    
    def start_monitoring(self) -> None:
        """Start comprehensive monitoring system."""
        if self.monitoring_active:
            logger.warning("Monitoring is already active")
            return
        
        # Start metric collection
        self.metric_collector.start_collection(interval=5.0)
        
        # Start monitoring loop
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Advanced monitoring system started")
    
    def stop_monitoring(self) -> None:
        """Stop monitoring system."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        self.metric_collector.stop_collection()
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)
        
        logger.info("Advanced monitoring system stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        baseline_calculation_interval = 300  # 5 minutes
        last_baseline_calculation = 0
        
        while self.monitoring_active:
            try:
                current_time = time.time()
                
                # Get current metrics
                metrics_summary = self.metric_collector.get_all_metrics_summary()
                
                # Calculate baselines periodically
                if current_time - last_baseline_calculation > baseline_calculation_interval:
                    self._update_baselines(metrics_summary)
                    last_baseline_calculation = current_time
                
                # Anomaly detection
                self._check_anomalies(metrics_summary)
                
                # Evaluate alerts
                triggered_alerts = self.alert_manager.evaluate_alerts(metrics_summary)
                
                # Run health checks
                self._run_health_checks()
                
                # Capacity planning (every 10 minutes)
                if int(current_time) % 600 == 0:
                    self._check_capacity_issues()
                
                time.sleep(30)  # Main loop interval
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(60)  # Back off on error
    
    def _update_baselines(self, metrics_summary: Dict[str, Any]) -> None:
        """Update performance baselines for anomaly detection."""
        for metric_name, stats in metrics_summary.items():
            if stats['count'] >= 10:  # Need minimum data points
                values = self.metric_collector.get_metric_values(metric_name, duration_seconds=3600)
                try:
                    self.anomaly_detector.calculate_baseline(metric_name, values)
                except Exception as e:
                    logger.warning(f"Failed to calculate baseline for {metric_name}: {e}")
    
    def _check_anomalies(self, metrics_summary: Dict[str, Any]) -> None:
        """Check for anomalies in current metrics."""
        for metric_name, stats in metrics_summary.items():
            current_value = stats.get('latest', 0)
            anomaly = self.anomaly_detector.detect_anomaly(metric_name, current_value)
            
            if anomaly:
                # Could trigger special alerts or notifications here
                pass
    
    def _run_health_checks(self) -> None:
        """Run all configured health checks."""
        current_time = datetime.now()
        
        for check_name, health_check in self.health_checks.items():
            # Check if it's time to run this health check
            if (health_check.last_check is None or
                (current_time - health_check.last_check).total_seconds() >= health_check.interval_seconds):
                
                try:
                    result = health_check.check_function()
                    health_check.last_check = current_time
                    
                    if result.get('status') == 'healthy':
                        health_check.current_successes += 1
                        health_check.current_failures = 0
                        
                        if health_check.current_successes >= health_check.success_threshold:
                            health_check.status = 'healthy'
                    else:
                        health_check.current_failures += 1
                        health_check.current_successes = 0
                        
                        if health_check.current_failures >= health_check.failure_threshold:
                            health_check.status = 'unhealthy'
                        else:
                            health_check.status = 'degraded'
                    
                except Exception as e:
                    logger.error(f"Health check {check_name} failed: {e}")
                    health_check.status = 'error'
    
    def _check_capacity_issues(self) -> None:
        """Check for potential capacity issues using predictive analytics."""
        key_metrics = ['cpu_percent', 'memory_percent', 'disk_percent']
        
        for metric_name in key_metrics:
            values = self.metric_collector.get_metric_values(metric_name, duration_seconds=7200)  # 2 hours
            timestamps = [time.time() - (len(values) - i - 1) * 5 for i in range(len(values))]
            
            if len(values) >= 20:
                try:
                    # Train prediction model
                    self.predictive_analytics.train_prediction_model(
                        metric_name, values, timestamps
                    )
                    
                    # Check for capacity issues
                    capacity_alert = self.predictive_analytics.detect_capacity_issues(
                        metric_name, threshold_percent=85.0
                    )
                    
                    if capacity_alert:
                        logger.warning(f"Capacity issue predicted for {metric_name}: "
                                     f"{capacity_alert['earliest_alert_minutes']} minutes ahead")
                
                except Exception as e:
                    logger.warning(f"Capacity planning failed for {metric_name}: {e}")
    
    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive monitoring dashboard data."""
        metrics_summary = self.metric_collector.get_all_metrics_summary()
        anomaly_summary = self.anomaly_detector.get_anomaly_summary(hours=1)
        active_alerts = self.alert_manager.get_active_alerts()
        
        # Health check summary
        health_summary = {}
        for name, check in self.health_checks.items():
            health_summary[name] = {
                'status': check.status,
                'last_check': check.last_check.isoformat() if check.last_check else None,
                'failures': check.current_failures,
                'successes': check.current_successes
            }
        
        # System overview
        system_health = 'healthy'
        if any(alert.severity == 'critical' for alert in active_alerts):
            system_health = 'critical'
        elif any(alert.severity == 'warning' for alert in active_alerts) or anomaly_summary['anomalies_detected'] > 0:
            system_health = 'warning'
        
        return {
            'system_health': system_health,
            'timestamp': datetime.now().isoformat(),
            'metrics_summary': metrics_summary,
            'anomaly_summary': anomaly_summary,
            'active_alerts': [asdict(alert) for alert in active_alerts],
            'health_checks': health_summary,
            'monitoring_status': {
                'active': self.monitoring_active,
                'uptime_seconds': time.time() - self.metric_collector.start_time if hasattr(self.metric_collector, 'start_time') else 0
            }
        }
    
    def save_monitoring_report(self) -> str:
        """Save comprehensive monitoring report."""
        dashboard_data = self.get_monitoring_dashboard()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.monitoring_dir / f"monitoring_report_{timestamp}.json"
        
        with open(report_file, "w") as f:
            json.dump(dashboard_data, f, indent=2, default=str)
        
        logger.info(f"Monitoring report saved: {report_file}")
        return str(report_file)


def main():
    """Main entry point for advanced monitoring system."""
    logging.basicConfig(level=logging.INFO)
    
    monitoring_system = AdvancedMonitoringSystem()
    
    try:
        # Start monitoring
        monitoring_system.start_monitoring()
        print("📊 ADVANCED MONITORING SYSTEM STARTED")
        
        # Simulate some metric activity
        print("🔧 Simulating metric activity...")
        for i in range(30):
            # Simulate API requests
            monitoring_system.metric_collector.increment_counter('api_requests_total', 1)
            monitoring_system.metric_collector.increment_counter('api_request_duration_sum', 0.5)
            
            # Occasionally simulate errors
            if i % 10 == 0:
                monitoring_system.metric_collector.increment_counter('api_errors_total', 1)
            
            time.sleep(2)
        
        # Get dashboard data
        dashboard = monitoring_system.get_monitoring_dashboard()
        print(f"📋 System Health: {dashboard['system_health']}")
        print(f"📈 Metrics Collected: {len(dashboard['metrics_summary'])}")
        print(f"🚨 Active Alerts: {len(dashboard['active_alerts'])}")
        print(f"❗ Anomalies Detected: {dashboard['anomaly_summary']['anomalies_detected']}")
        
        # Save monitoring report
        report_file = monitoring_system.save_monitoring_report()
        print(f"📊 Monitoring report saved: {report_file}")
        
        print("✅ ADVANCED MONITORING DEMONSTRATION COMPLETED")
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping monitoring system...")
    finally:
        monitoring_system.stop_monitoring()


if __name__ == "__main__":
    main()