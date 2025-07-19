"""Alerting system for high error rates and production monitoring."""

from __future__ import annotations

import os
import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


@dataclass
class AlertRule:
    """Represents a single Prometheus alerting rule."""
    
    name: str
    description: str
    expression: str
    duration: str
    severity: str
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert rule to Prometheus rule format."""
        rule = {
            "alert": self.name,
            "expr": self.expression,
            "for": self.duration,
            "labels": {"severity": self.severity, **self.labels},
            "annotations": {"description": self.description, **self.annotations}
        }
        return rule


class BaseAlert(ABC):
    """Base class for alert definitions."""
    
    def __init__(self, name: str, severity: str = "warning", duration: str = "5m"):
        self.name = name
        self.severity = severity
        self.duration = duration
    
    @abstractmethod
    def to_alert_rule(self) -> AlertRule:
        """Convert alert to AlertRule format."""
        pass


class ErrorRateAlert(BaseAlert):
    """Alert for high error rates."""
    
    def __init__(
        self,
        threshold: float = 0.05,  # 5% error rate
        duration: str = "5m",
        severity: str = "critical",
        window: str = "5m"
    ):
        super().__init__("HighErrorRate", severity, duration)
        self.threshold = threshold
        self.window = window
    
    def to_alert_rule(self) -> AlertRule:
        """Generate Prometheus rule for error rate monitoring."""
        expression = (
            f"rate(http_requests_errors_total[{self.window}]) / "
            f"rate(http_requests_total[{self.window}]) > {self.threshold}"
        )
        
        description = (
            f"Error rate is above {self.threshold*100:.1f}% "
            f"for more than {self.duration}"
        )
        
        annotations = {
            "summary": f"High error rate detected: {{{{ $value | humanizePercentage }}}}",
            "description": description,
            "runbook_url": "https://docs.company.com/runbooks/high-error-rate"
        }
        
        return AlertRule(
            name=self.name,
            description=description,
            expression=expression,
            duration=self.duration,
            severity=self.severity,
            annotations=annotations
        )


class LatencyAlert(BaseAlert):
    """Alert for high response latency."""
    
    def __init__(
        self,
        threshold_seconds: float = 2.0,
        percentile: str = "95",
        duration: str = "5m",
        severity: str = "warning",
        window: str = "5m"
    ):
        super().__init__("HighLatency", severity, duration)
        self.threshold_seconds = threshold_seconds
        self.percentile = percentile
        self.window = window
    
    def to_alert_rule(self) -> AlertRule:
        """Generate Prometheus rule for latency monitoring."""
        percentile_value = float(self.percentile) / 100
        expression = (
            f"histogram_quantile({percentile_value}, "
            f"rate(http_request_duration_seconds_bucket[{self.window}])) "
            f"> {self.threshold_seconds}"
        )
        
        description = (
            f"{self.percentile}th percentile latency is above "
            f"{self.threshold_seconds}s for more than {self.duration}"
        )
        
        annotations = {
            "summary": f"High latency detected: {{{{ $value }}}}s",
            "description": description,
            "runbook_url": "https://docs.company.com/runbooks/high-latency"
        }
        
        return AlertRule(
            name=self.name,
            description=description,
            expression=expression,
            duration=self.duration,
            severity=self.severity,
            annotations=annotations
        )


class ServiceDownAlert(BaseAlert):
    """Alert for service availability."""
    
    def __init__(
        self,
        duration: str = "1m",
        severity: str = "critical"
    ):
        super().__init__("ServiceDown", severity, duration)
    
    def to_alert_rule(self) -> AlertRule:
        """Generate Prometheus rule for service availability."""
        expression = "up{job=\"lexgraph-api\"} == 0"
        description = f"Service has been down for more than {self.duration}"
        
        annotations = {
            "summary": "LexGraph service is down",
            "description": description,
            "runbook_url": "https://docs.company.com/runbooks/service-down"
        }
        
        return AlertRule(
            name=self.name,
            description=description,
            expression=expression,
            duration=self.duration,
            severity=self.severity,
            annotations=annotations
        )


@dataclass
class AlertingConfig:
    """Configuration for alerting system."""
    
    enabled: bool = True
    error_rate_threshold: float = 0.05  # 5%
    latency_threshold_seconds: float = 2.0  # 2 seconds
    latency_percentile: str = "95"
    check_interval: str = "5m"
    alert_duration: str = "5m"
    prometheus_rules_path: Optional[str] = None


class AlertManager:
    """Manages alert rules and Prometheus configuration."""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.alerts: List[BaseAlert] = []
    
    def add_alert(self, alert: BaseAlert) -> None:
        """Add an alert to the manager."""
        self.alerts.append(alert)
        logger.debug(f"Added alert: {alert.name}")
    
    def remove_alert(self, alert_name: str) -> bool:
        """Remove an alert by name."""
        initial_count = len(self.alerts)
        self.alerts = [alert for alert in self.alerts if alert.name != alert_name]
        removed = len(self.alerts) < initial_count
        if removed:
            logger.debug(f"Removed alert: {alert_name}")
        return removed
    
    def generate_prometheus_rules(self) -> Dict[str, Any]:
        """Generate Prometheus alerting rules configuration."""
        if not self.enabled or not self.alerts:
            return {"groups": []}
        
        rules = []
        for alert in self.alerts:
            rule = alert.to_alert_rule()
            rules.append(rule.to_dict())
        
        prometheus_config = {
            "groups": [
                {
                    "name": "lexgraph_alerts",
                    "interval": "30s",
                    "rules": rules
                }
            ]
        }
        
        return prometheus_config
    
    def save_prometheus_rules(self, file_path: str) -> None:
        """Save Prometheus rules to YAML file."""
        rules = self.generate_prometheus_rules()
        
        try:
            with open(file_path, 'w') as f:
                yaml.dump(rules, f, default_flow_style=False, sort_keys=False)
            logger.info(f"Saved Prometheus rules to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save Prometheus rules: {e}")
            raise
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of configured alerts."""
        return {
            "enabled": self.enabled,
            "total_alerts": len(self.alerts),
            "alerts": [
                {
                    "name": alert.name,
                    "severity": alert.severity,
                    "duration": alert.duration
                }
                for alert in self.alerts
            ]
        }


def load_alerting_config() -> AlertingConfig:
    """Load alerting configuration from environment variables."""
    enabled = os.getenv("ALERTING_ENABLED", "true").lower() == "true"
    
    error_rate_threshold = float(os.getenv("ERROR_RATE_THRESHOLD", "0.05"))
    latency_threshold = float(os.getenv("LATENCY_THRESHOLD_SECONDS", "2.0"))
    latency_percentile = os.getenv("LATENCY_PERCENTILE", "95")
    check_interval = os.getenv("ALERT_CHECK_INTERVAL", "5m")
    alert_duration = os.getenv("ALERT_DURATION", "5m")
    rules_path = os.getenv("PROMETHEUS_RULES_PATH")
    
    return AlertingConfig(
        enabled=enabled,
        error_rate_threshold=error_rate_threshold,
        latency_threshold_seconds=latency_threshold,
        latency_percentile=latency_percentile,
        check_interval=check_interval,
        alert_duration=alert_duration,
        prometheus_rules_path=rules_path
    )


def create_default_alerts(config: AlertingConfig) -> AlertManager:
    """Create default alert set based on configuration."""
    manager = AlertManager(enabled=config.enabled)
    
    if not config.enabled:
        return manager
    
    # High error rate alert
    error_alert = ErrorRateAlert(
        threshold=config.error_rate_threshold,
        duration=config.alert_duration,
        severity="critical"
    )
    manager.add_alert(error_alert)
    
    # High latency alert
    latency_alert = LatencyAlert(
        threshold_seconds=config.latency_threshold_seconds,
        percentile=config.latency_percentile,
        duration=config.alert_duration,
        severity="warning"
    )
    manager.add_alert(latency_alert)
    
    # Service down alert
    service_alert = ServiceDownAlert(
        duration="1m",
        severity="critical"
    )
    manager.add_alert(service_alert)
    
    return manager


def create_prometheus_alerts(config: AlertingConfig) -> Optional[str]:
    """Create Prometheus alerting rules YAML configuration."""
    if not config.enabled:
        return None
    
    manager = create_default_alerts(config)
    rules = manager.generate_prometheus_rules()
    
    try:
        return yaml.dump(rules, default_flow_style=False, sort_keys=False)
    except Exception as e:
        logger.error(f"Failed to generate Prometheus alerts YAML: {e}")
        return None


def setup_alerting() -> AlertManager:
    """Setup alerting system with default configuration."""
    config = load_alerting_config()
    manager = create_default_alerts(config)
    
    # Save rules to file if path is configured
    if config.prometheus_rules_path and config.enabled:
        try:
            manager.save_prometheus_rules(config.prometheus_rules_path)
        except Exception as e:
            logger.warning(f"Could not save Prometheus rules: {e}")
    
    logger.info(f"Alerting setup complete. Enabled: {config.enabled}, "
                f"Alerts: {len(manager.alerts)}")
    
    return manager