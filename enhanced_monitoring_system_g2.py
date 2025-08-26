#!/usr/bin/env python3
"""
📊 GENERATION 2: ENHANCED MONITORING SYSTEM
Advanced observability, metrics, and real-time alerting
"""

import asyncio
import json
import time
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import statistics
from pathlib import Path
import threading
from collections import defaultdict, deque

import structlog
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.ConsoleRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    context_class=dict,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class Alert:
    """Alert data structure"""
    id: str
    severity: AlertSeverity
    title: str
    description: str
    metric_name: str
    threshold: float
    actual_value: float
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None


@dataclass
class MetricPoint:
    """Single metric data point"""
    timestamp: datetime
    value: float
    labels: Dict[str, str]


class EnhancedMetricsCollector:
    """Advanced metrics collection and analysis"""
    
    def __init__(self):
        self.registry = CollectorRegistry()
        self.counters: Dict[str, Counter] = {}
        self.gauges: Dict[str, Gauge] = {}
        self.histograms: Dict[str, Histogram] = {}
        
        self.custom_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.metric_metadata: Dict[str, Dict] = {}
        
        # Initialize core metrics
        self._init_core_metrics()
    
    def _init_core_metrics(self):
        """Initialize core system metrics"""
        self.counters['requests_total'] = Counter(
            'requests_total', 
            'Total requests processed',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.histograms['request_duration'] = Histogram(
            'request_duration_seconds',
            'Request processing time',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        self.gauges['active_connections'] = Gauge(
            'active_connections',
            'Current active connections',
            registry=self.registry
        )
        
        self.gauges['memory_usage'] = Gauge(
            'memory_usage_bytes',
            'Current memory usage',
            registry=self.registry
        )
        
        self.counters['bioneural_analyses'] = Counter(
            'bioneural_analyses_total',
            'Total bioneural analyses performed',
            ['document_type', 'receptor_type'],
            registry=self.registry
        )
    
    def increment_counter(self, name: str, value: float = 1, labels: Optional[Dict[str, str]] = None):
        """Increment a counter metric"""
        if name in self.counters:
            if labels:
                self.counters[name].labels(**labels).inc(value)
            else:
                self.counters[name].inc(value)
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set a gauge metric value"""
        if name in self.gauges:
            if labels:
                self.gauges[name].labels(**labels).set(value)
            else:
                self.gauges[name].set(value)
    
    def observe_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a histogram observation"""
        if name in self.histograms:
            if labels:
                self.histograms[name].labels(**labels).observe(value)
            else:
                self.histograms[name].observe(value)
    
    def record_custom_metric(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a custom metric point"""
        point = MetricPoint(
            timestamp=datetime.now(),
            value=value,
            labels=labels or {}
        )
        self.custom_metrics[name].append(point)
    
    def get_metric_stats(self, name: str, window_minutes: int = 5) -> Dict[str, Any]:
        """Get statistical analysis of a custom metric"""
        if name not in self.custom_metrics:
            return {}
        
        # Filter to window
        cutoff = datetime.now() - timedelta(minutes=window_minutes)
        recent_points = [
            p for p in self.custom_metrics[name] 
            if p.timestamp > cutoff
        ]
        
        if not recent_points:
            return {}
        
        values = [p.value for p in recent_points]
        
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'std_dev': statistics.stdev(values) if len(values) > 1 else 0,
            'percentile_95': statistics.quantiles(values, n=20)[18] if len(values) > 1 else values[0],
            'percentile_99': statistics.quantiles(values, n=100)[98] if len(values) > 1 else values[0]
        }
    
    def export_prometheus_metrics(self) -> bytes:
        """Export metrics in Prometheus format"""
        return generate_latest(self.registry)


class AlertManager:
    """Intelligent alerting system"""
    
    def __init__(self, metrics_collector: EnhancedMetricsCollector):
        self.metrics = metrics_collector
        self.alerts: Dict[str, Alert] = {}
        self.alert_rules: List[Dict] = []
        self.notification_handlers: List[Callable] = []
        
        # Setup default alert rules
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default alerting rules"""
        self.alert_rules.extend([
            {
                'name': 'high_error_rate',
                'metric': 'error_rate',
                'condition': 'greater_than',
                'threshold': 0.1,
                'severity': AlertSeverity.WARNING,
                'description': 'Error rate exceeds 10%'
            },
            {
                'name': 'response_time_high',
                'metric': 'response_time_p95',
                'condition': 'greater_than',
                'threshold': 2.0,
                'severity': AlertSeverity.WARNING,
                'description': '95th percentile response time exceeds 2s'
            },
            {
                'name': 'memory_usage_critical',
                'metric': 'memory_usage_percent',
                'condition': 'greater_than',
                'threshold': 90.0,
                'severity': AlertSeverity.CRITICAL,
                'description': 'Memory usage exceeds 90%'
            },
            {
                'name': 'bioneural_failure_rate',
                'metric': 'bioneural_failure_rate',
                'condition': 'greater_than',
                'threshold': 0.05,
                'severity': AlertSeverity.ERROR,
                'description': 'Bioneural analysis failure rate exceeds 5%'
            }
        ])
    
    def add_alert_rule(self, name: str, metric: str, condition: str, threshold: float, 
                       severity: AlertSeverity, description: str):
        """Add custom alert rule"""
        rule = {
            'name': name,
            'metric': metric,
            'condition': condition,
            'threshold': threshold,
            'severity': severity,
            'description': description
        }
        self.alert_rules.append(rule)
        logger.info(f"Added alert rule: {name}")
    
    def add_notification_handler(self, handler: Callable):
        """Add alert notification handler"""
        self.notification_handlers.append(handler)
    
    async def check_alerts(self) -> List[Alert]:
        """Check all alert rules and trigger alerts"""
        triggered_alerts = []
        
        for rule in self.alert_rules:
            try:
                alert = await self._evaluate_rule(rule)
                if alert:
                    triggered_alerts.append(alert)
            except Exception as e:
                logger.error(f"Error evaluating alert rule {rule['name']}: {e}")
        
        return triggered_alerts
    
    async def _evaluate_rule(self, rule: Dict) -> Optional[Alert]:
        """Evaluate a single alert rule"""
        metric_name = rule['metric']
        
        # Get current metric value
        current_value = await self._get_current_metric_value(metric_name)
        if current_value is None:
            return None
        
        # Check condition
        threshold = rule['threshold']
        condition = rule['condition']
        triggered = False
        
        if condition == 'greater_than' and current_value > threshold:
            triggered = True
        elif condition == 'less_than' and current_value < threshold:
            triggered = True
        elif condition == 'equals' and abs(current_value - threshold) < 0.001:
            triggered = True
        
        if triggered:
            alert_id = f"{rule['name']}_{int(time.time())}"
            alert = Alert(
                id=alert_id,
                severity=rule['severity'],
                title=f"Alert: {rule['name']}",
                description=rule['description'],
                metric_name=metric_name,
                threshold=threshold,
                actual_value=current_value,
                timestamp=datetime.now()
            )
            
            # Store and notify
            self.alerts[alert_id] = alert
            await self._send_notifications(alert)
            
            logger.warning(f"Alert triggered: {alert.title} (value: {current_value}, threshold: {threshold})")
            return alert
        
        return None
    
    async def _get_current_metric_value(self, metric_name: str) -> Optional[float]:
        """Get current value for a metric"""
        if metric_name == 'error_rate':
            # Calculate from custom metrics
            stats = self.metrics.get_metric_stats('request_errors')
            total_stats = self.metrics.get_metric_stats('total_requests')
            if stats.get('count', 0) > 0 and total_stats.get('count', 0) > 0:
                return stats['count'] / total_stats['count']
            return 0.0
        
        elif metric_name == 'response_time_p95':
            stats = self.metrics.get_metric_stats('response_time')
            return stats.get('percentile_95', 0.0)
        
        elif metric_name == 'memory_usage_percent':
            # Simulated memory usage
            import psutil
            return psutil.virtual_memory().percent
        
        elif metric_name == 'bioneural_failure_rate':
            failure_stats = self.metrics.get_metric_stats('bioneural_failures')
            success_stats = self.metrics.get_metric_stats('bioneural_successes')
            total_attempts = failure_stats.get('count', 0) + success_stats.get('count', 0)
            if total_attempts > 0:
                return failure_stats.get('count', 0) / total_attempts
            return 0.0
        
        return None
    
    async def _send_notifications(self, alert: Alert):
        """Send alert notifications"""
        for handler in self.notification_handlers:
            try:
                await handler(alert)
            except Exception as e:
                logger.error(f"Error sending alert notification: {e}")


class RealTimeMonitor:
    """Real-time monitoring and dashboard data"""
    
    def __init__(self, metrics_collector: EnhancedMetricsCollector):
        self.metrics = metrics_collector
        self.dashboard_data: Dict[str, Any] = {}
        self.monitoring_tasks: List[asyncio.Task] = []
        self.running = False
    
    async def start_monitoring(self):
        """Start real-time monitoring tasks"""
        self.running = True
        
        # Start monitoring tasks
        self.monitoring_tasks.append(
            asyncio.create_task(self._monitor_system_metrics())
        )
        self.monitoring_tasks.append(
            asyncio.create_task(self._monitor_bioneural_performance())
        )
        self.monitoring_tasks.append(
            asyncio.create_task(self._update_dashboard())
        )
        
        logger.info("Real-time monitoring started")
    
    async def stop_monitoring(self):
        """Stop monitoring tasks"""
        self.running = False
        for task in self.monitoring_tasks:
            task.cancel()
        
        await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        logger.info("Real-time monitoring stopped")
    
    async def _monitor_system_metrics(self):
        """Monitor system-level metrics"""
        while self.running:
            try:
                import psutil
                
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.metrics.set_gauge('cpu_usage', cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.metrics.set_gauge('memory_usage', memory.used)
                self.metrics.record_custom_metric('memory_usage_percent', memory.percent)
                
                # Disk usage
                disk = psutil.disk_usage('/')
                self.metrics.set_gauge('disk_usage', disk.used)
                
                await asyncio.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring system metrics: {e}")
                await asyncio.sleep(10)
    
    async def _monitor_bioneural_performance(self):
        """Monitor bioneural system performance"""
        while self.running:
            try:
                # Simulate bioneural metrics (in real system, get from actual components)
                import random
                
                # Olfactory receptor performance
                for receptor in ['legal_complexity', 'statutory_authority', 'temporal_freshness']:
                    accuracy = 0.85 + random.uniform(-0.1, 0.1)
                    response_time = 0.05 + random.uniform(-0.02, 0.02)
                    
                    self.metrics.record_custom_metric(
                        f'receptor_accuracy', 
                        accuracy, 
                        {'receptor': receptor}
                    )
                    self.metrics.record_custom_metric(
                        f'receptor_response_time', 
                        response_time, 
                        {'receptor': receptor}
                    )
                
                # Overall bioneural performance
                overall_accuracy = 0.92 + random.uniform(-0.05, 0.05)
                processing_time = 0.15 + random.uniform(-0.05, 0.05)
                
                self.metrics.record_custom_metric('bioneural_accuracy', overall_accuracy)
                self.metrics.record_custom_metric('bioneural_processing_time', processing_time)
                
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring bioneural performance: {e}")
                await asyncio.sleep(5)
    
    async def _update_dashboard(self):
        """Update dashboard data"""
        while self.running:
            try:
                self.dashboard_data = {
                    'timestamp': datetime.now().isoformat(),
                    'system': {
                        'memory_stats': self.metrics.get_metric_stats('memory_usage_percent', 5),
                        'cpu_usage': self.metrics.get_metric_stats('cpu_usage', 5),
                    },
                    'bioneural': {
                        'accuracy_stats': self.metrics.get_metric_stats('bioneural_accuracy', 10),
                        'processing_time_stats': self.metrics.get_metric_stats('bioneural_processing_time', 10),
                        'receptor_performance': {
                            receptor: self.metrics.get_metric_stats('receptor_accuracy', 10)
                            for receptor in ['legal_complexity', 'statutory_authority', 'temporal_freshness']
                        }
                    },
                    'requests': {
                        'total_requests': self.metrics.get_metric_stats('total_requests', 30),
                        'error_rate': self.metrics.get_metric_stats('request_errors', 30),
                        'response_time': self.metrics.get_metric_stats('response_time', 30)
                    }
                }
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"Error updating dashboard: {e}")
                await asyncio.sleep(30)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data"""
        return self.dashboard_data.copy()


class EnhancedMonitoringSystem:
    """Complete monitoring system with metrics, alerts, and dashboards"""
    
    def __init__(self):
        self.metrics_collector = EnhancedMetricsCollector()
        self.alert_manager = AlertManager(self.metrics_collector)
        self.real_time_monitor = RealTimeMonitor(self.metrics_collector)
        
        # Setup notification handlers
        self._setup_notifications()
    
    def _setup_notifications(self):
        """Setup alert notification handlers"""
        
        async def console_handler(alert: Alert):
            """Console notification handler"""
            severity_emoji = {
                AlertSeverity.INFO: "ℹ️",
                AlertSeverity.WARNING: "⚠️",
                AlertSeverity.ERROR: "❌",
                AlertSeverity.CRITICAL: "🚨"
            }
            
            print(f"\n{severity_emoji[alert.severity]} ALERT: {alert.title}")
            print(f"Description: {alert.description}")
            print(f"Metric: {alert.metric_name}")
            print(f"Threshold: {alert.threshold}")
            print(f"Actual: {alert.actual_value}")
            print(f"Time: {alert.timestamp}")
            print("-" * 50)
        
        self.alert_manager.add_notification_handler(console_handler)
    
    async def start(self):
        """Start the monitoring system"""
        await self.real_time_monitor.start_monitoring()
        logger.info("Enhanced monitoring system started")
    
    async def stop(self):
        """Stop the monitoring system"""
        await self.real_time_monitor.stop_monitoring()
        logger.info("Enhanced monitoring system stopped")
    
    def record_request(self, method: str, endpoint: str, status: int, duration: float):
        """Record API request metrics"""
        self.metrics_collector.increment_counter(
            'requests_total', 
            labels={'method': method, 'endpoint': endpoint, 'status': str(status)}
        )
        
        self.metrics_collector.observe_histogram(
            'request_duration',
            duration,
            labels={'method': method, 'endpoint': endpoint}
        )
        
        self.metrics_collector.record_custom_metric('total_requests', 1)
        self.metrics_collector.record_custom_metric('response_time', duration)
        
        if status >= 400:
            self.metrics_collector.record_custom_metric('request_errors', 1)
    
    def record_bioneural_analysis(self, document_type: str, receptor_type: str, 
                                 success: bool, duration: float):
        """Record bioneural analysis metrics"""
        self.metrics_collector.increment_counter(
            'bioneural_analyses',
            labels={'document_type': document_type, 'receptor_type': receptor_type}
        )
        
        if success:
            self.metrics_collector.record_custom_metric('bioneural_successes', 1)
        else:
            self.metrics_collector.record_custom_metric('bioneural_failures', 1)
        
        self.metrics_collector.record_custom_metric('bioneural_processing_time', duration)
    
    async def run_health_check(self) -> Dict[str, Any]:
        """Run comprehensive health check"""
        # Check for alerts
        alerts = await self.alert_manager.check_alerts()
        
        # Get dashboard data
        dashboard_data = self.real_time_monitor.get_dashboard_data()
        
        # System status
        status = "healthy"
        if any(alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.ERROR] for alert in alerts):
            status = "critical"
        elif any(alert.severity == AlertSeverity.WARNING for alert in alerts):
            status = "degraded"
        
        return {
            'status': status,
            'timestamp': datetime.now().isoformat(),
            'active_alerts': [asdict(alert) for alert in alerts],
            'metrics_summary': dashboard_data,
            'system_info': {
                'total_metrics': len(self.metrics_collector.custom_metrics),
                'alert_rules': len(self.alert_manager.alert_rules),
                'monitoring_active': self.real_time_monitor.running
            }
        }


async def demo_monitoring_system():
    """Demonstrate enhanced monitoring system"""
    print("📊 GENERATION 2: ENHANCED MONITORING SYSTEM")
    print("=" * 60)
    print("Advanced observability and real-time alerting")
    print()
    
    monitoring = EnhancedMonitoringSystem()
    
    # Start monitoring
    await monitoring.start()
    
    print("🚀 Monitoring system started")
    
    # Simulate some activity
    print("\n📈 Simulating system activity...")
    
    # Simulate API requests
    for i in range(10):
        method = "GET" if i % 2 == 0 else "POST"
        endpoint = "/api/analyze" if i % 3 == 0 else "/api/health"
        status = 200 if i % 7 != 0 else 500  # Some errors
        duration = 0.1 + (i * 0.02)
        
        monitoring.record_request(method, endpoint, status, duration)
        
        # Simulate bioneural analysis
        monitoring.record_bioneural_analysis(
            document_type="contract",
            receptor_type="legal_complexity",
            success=i % 8 != 0,  # Some failures
            duration=0.05 + (i * 0.01)
        )
        
        await asyncio.sleep(0.1)
    
    # Wait for monitoring to collect data
    print("⏱️ Collecting metrics data...")
    await asyncio.sleep(5)
    
    # Run health check
    print("\n🏥 Running health check...")
    health = await monitoring.run_health_check()
    
    print(f"System Status: {health['status']}")
    print(f"Active Alerts: {len(health['active_alerts'])}")
    
    if health['active_alerts']:
        print("\n🚨 Active Alerts:")
        for alert in health['active_alerts']:
            print(f"  - {alert['title']}: {alert['description']}")
    
    # Get Prometheus metrics
    print("\n📊 Prometheus Metrics Sample:")
    metrics_data = monitoring.metrics_collector.export_prometheus_metrics()
    print(metrics_data.decode()[:500] + "...")
    
    # Save results
    results_file = Path("enhanced_monitoring_g2_results.json")
    with open(results_file, 'w') as f:
        json.dump(health, f, indent=2, default=str)
    
    print(f"\n📄 Results saved to {results_file}")
    
    # Stop monitoring
    await monitoring.stop()
    print("\n🎉 Enhanced monitoring system demonstration complete!")


if __name__ == "__main__":
    asyncio.run(demo_monitoring_system())