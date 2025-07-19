"""Test alerting system for high error rates and production monitoring."""

import pytest
from unittest.mock import patch, MagicMock, mock_open
import json
import tempfile
import os
from pathlib import Path

from lexgraph_legal_rag.alerting import (
    AlertingConfig,
    AlertRule,
    AlertManager,
    ErrorRateAlert,
    LatencyAlert,
    create_prometheus_alerts,
    load_alerting_config,
)


class TestAlertRule:
    """Test alert rule configuration and validation."""
    
    def test_alert_rule_creation(self):
        """Test creating a basic alert rule."""
        rule = AlertRule(
            name="high_error_rate",
            description="Alert on high error rates",
            expression="rate(http_requests_errors_total[5m]) > 0.05",
            duration="2m",
            severity="critical"
        )
        
        assert rule.name == "high_error_rate"
        assert rule.description == "Alert on high error rates"
        assert rule.expression == "rate(http_requests_errors_total[5m]) > 0.05"
        assert rule.duration == "2m"
        assert rule.severity == "critical"
    
    def test_alert_rule_to_dict(self):
        """Test converting alert rule to dictionary format."""
        rule = AlertRule(
            name="test_alert",
            description="Test alert",
            expression="up == 0",
            duration="1m",
            severity="warning"
        )
        
        expected = {
            "alert": "test_alert",
            "expr": "up == 0",
            "for": "1m",
            "labels": {"severity": "warning"},
            "annotations": {"description": "Test alert"}
        }
        
        assert rule.to_dict() == expected


class TestErrorRateAlert:
    """Test error rate alert functionality."""
    
    def test_error_rate_alert_creation(self):
        """Test creating error rate alert with default thresholds."""
        alert = ErrorRateAlert()
        
        assert alert.name == "HighErrorRate"
        assert alert.threshold == 0.05  # 5% default
        assert alert.duration == "5m"
        assert alert.severity == "critical"
    
    def test_error_rate_alert_custom_threshold(self):
        """Test creating error rate alert with custom threshold."""
        alert = ErrorRateAlert(threshold=0.10, duration="2m", severity="warning")
        
        assert alert.threshold == 0.10
        assert alert.duration == "2m"
        assert alert.severity == "warning"
    
    def test_error_rate_alert_prometheus_expression(self):
        """Test generating correct Prometheus expression."""
        alert = ErrorRateAlert(threshold=0.03, duration="3m")
        rule = alert.to_alert_rule()
        
        expected_expr = "rate(http_requests_errors_total[5m]) / rate(http_requests_total[5m]) > 0.03"
        assert rule.expression == expected_expr
        assert rule.duration == "3m"


class TestLatencyAlert:
    """Test latency alert functionality."""
    
    def test_latency_alert_creation(self):
        """Test creating latency alert with default values."""
        alert = LatencyAlert()
        
        assert alert.name == "HighLatency"
        assert alert.threshold_seconds == 2.0
        assert alert.percentile == "95"
        assert alert.duration == "5m"
    
    def test_latency_alert_prometheus_expression(self):
        """Test generating correct Prometheus expression for latency."""
        alert = LatencyAlert(threshold_seconds=1.5, percentile="99", duration="3m")
        rule = alert.to_alert_rule()
        
        expected_expr = "histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m])) > 1.5"
        assert rule.expression == expected_expr


class TestAlertManager:
    """Test alert manager functionality."""
    
    def test_alert_manager_initialization(self):
        """Test alert manager creation."""
        manager = AlertManager()
        
        assert len(manager.alerts) == 0
        assert manager.enabled is True
    
    def test_add_alert_to_manager(self):
        """Test adding alerts to manager."""
        manager = AlertManager()
        error_alert = ErrorRateAlert()
        latency_alert = LatencyAlert()
        
        manager.add_alert(error_alert)
        manager.add_alert(latency_alert)
        
        assert len(manager.alerts) == 2
        assert any(alert.name == "HighErrorRate" for alert in manager.alerts)
        assert any(alert.name == "HighLatency" for alert in manager.alerts)
    
    def test_generate_prometheus_rules(self):
        """Test generating Prometheus rules configuration."""
        manager = AlertManager()
        manager.add_alert(ErrorRateAlert(threshold=0.05))
        manager.add_alert(LatencyAlert(threshold_seconds=2.0))
        
        rules = manager.generate_prometheus_rules()
        
        assert "groups" in rules
        assert len(rules["groups"]) == 1
        
        group = rules["groups"][0]
        assert group["name"] == "lexgraph_alerts"
        assert len(group["rules"]) == 2
    
    def test_save_prometheus_rules(self):
        """Test saving Prometheus rules to file."""
        manager = AlertManager()
        manager.add_alert(ErrorRateAlert())
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            temp_path = f.name
        
        try:
            manager.save_prometheus_rules(temp_path)
            
            # Verify file was created and contains expected content
            assert os.path.exists(temp_path)
            
            with open(temp_path, 'r') as f:
                content = f.read()
                assert "lexgraph_alerts" in content
                assert "HighErrorRate" in content
        
        finally:
            os.unlink(temp_path)


class TestAlertingConfig:
    """Test alerting configuration management."""
    
    def test_alerting_config_creation(self):
        """Test creating alerting configuration."""
        config = AlertingConfig(
            enabled=True,
            error_rate_threshold=0.04,
            latency_threshold_seconds=1.5,
            check_interval="30s"
        )
        
        assert config.enabled is True
        assert config.error_rate_threshold == 0.04
        assert config.latency_threshold_seconds == 1.5
        assert config.check_interval == "30s"
    
    def test_load_alerting_config_from_env(self):
        """Test loading alerting config from environment variables."""
        with patch.dict(os.environ, {
            'ALERTING_ENABLED': 'true',
            'ERROR_RATE_THRESHOLD': '0.03',
            'LATENCY_THRESHOLD_SECONDS': '2.5',
            'ALERT_CHECK_INTERVAL': '60s'
        }):
            config = load_alerting_config()
            
            assert config.enabled is True
            assert config.error_rate_threshold == 0.03
            assert config.latency_threshold_seconds == 2.5
            assert config.check_interval == "60s"
    
    def test_load_alerting_config_defaults(self):
        """Test loading alerting config with default values."""
        with patch.dict(os.environ, {}, clear=True):
            config = load_alerting_config()
            
            assert config.enabled is True  # Default
            assert config.error_rate_threshold == 0.05  # Default 5%
            assert config.latency_threshold_seconds == 2.0  # Default 2s
            assert config.check_interval == "5m"  # Default


class TestPrometheusIntegration:
    """Test Prometheus alerting rules integration."""
    
    def test_create_prometheus_alerts_function(self):
        """Test creating complete Prometheus alerts configuration."""
        config = AlertingConfig(
            error_rate_threshold=0.04,
            latency_threshold_seconds=1.8
        )
        
        alerts_yaml = create_prometheus_alerts(config)
        
        # Should return valid YAML string
        assert isinstance(alerts_yaml, str)
        assert "groups:" in alerts_yaml
        assert "HighErrorRate" in alerts_yaml
        assert "HighLatency" in alerts_yaml
        assert "0.04" in alerts_yaml  # Custom threshold
        assert "1.8" in alerts_yaml   # Custom threshold
    
    def test_prometheus_alerts_disabled(self):
        """Test behavior when alerting is disabled."""
        config = AlertingConfig(enabled=False)
        
        alerts_yaml = create_prometheus_alerts(config)
        
        # Should return empty or minimal configuration
        assert alerts_yaml is None or "groups: []" in alerts_yaml