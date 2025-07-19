#!/usr/bin/env python3
"""Manual test for alerting functionality."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Test the individual components
def test_alert_rule():
    from lexgraph_legal_rag.alerting import AlertRule
    
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
    print("✅ AlertRule test passed")

def test_error_rate_alert():
    from lexgraph_legal_rag.alerting import ErrorRateAlert
    
    alert = ErrorRateAlert(threshold=0.03, duration="3m")
    rule = alert.to_alert_rule()
    
    expected_expr = "rate(http_requests_errors_total[5m]) / rate(http_requests_total[5m]) > 0.03"
    assert rule.expression == expected_expr
    assert rule.duration == "3m"
    print("✅ ErrorRateAlert test passed")

def test_latency_alert():
    from lexgraph_legal_rag.alerting import LatencyAlert
    
    alert = LatencyAlert(threshold_seconds=1.5, percentile="99", duration="3m")
    rule = alert.to_alert_rule()
    
    expected_expr = "histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m])) > 1.5"
    assert rule.expression == expected_expr
    print("✅ LatencyAlert test passed")

def test_alert_manager():
    from lexgraph_legal_rag.alerting import AlertManager, ErrorRateAlert, LatencyAlert
    
    manager = AlertManager()
    manager.add_alert(ErrorRateAlert(threshold=0.05))
    manager.add_alert(LatencyAlert(threshold_seconds=2.0))
    
    rules = manager.generate_prometheus_rules()
    
    assert "groups" in rules
    assert len(rules["groups"]) == 1
    
    group = rules["groups"][0]
    assert group["name"] == "lexgraph_alerts"
    assert len(group["rules"]) == 2
    print("✅ AlertManager test passed")

def test_prometheus_integration():
    from lexgraph_legal_rag.alerting import create_prometheus_alerts, AlertingConfig
    
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
    print("✅ Prometheus integration test passed")

if __name__ == "__main__":
    print("Running manual alerting tests...")
    test_alert_rule()
    test_error_rate_alert()
    test_latency_alert()
    test_alert_manager()
    test_prometheus_integration()
    print("✅ All manual alerting tests passed!")