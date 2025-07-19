#!/usr/bin/env python3
"""Generate Prometheus alerting rules for LexGraph API."""

import sys
import os
import importlib.util

# Load alerting module directly
spec = importlib.util.spec_from_file_location(
    "alerting", 
    os.path.join(os.path.dirname(__file__), 'src', 'lexgraph_legal_rag', 'alerting.py')
)
alerting = importlib.util.module_from_spec(spec)
sys.modules['alerting'] = alerting
spec.loader.exec_module(alerting)

def main():
    """Generate and save Prometheus alerting rules."""
    # Create configuration with production-ready defaults
    config = alerting.AlertingConfig(
        enabled=True,
        error_rate_threshold=0.05,  # 5% error rate
        latency_threshold_seconds=2.0,  # 2 seconds
        latency_percentile="95",
        check_interval="5m",
        alert_duration="5m"
    )
    
    # Generate alerts YAML
    alerts_yaml = alerting.create_prometheus_alerts(config)
    
    # Save to monitoring directory
    rules_path = os.path.join("monitoring", "lexgraph_alerts.yml")
    with open(rules_path, 'w') as f:
        f.write(alerts_yaml)
    
    print(f"✅ Generated Prometheus alerting rules: {rules_path}")
    
    # Also create a configuration file example
    env_example_path = os.path.join("monitoring", "alerting.env.example")
    with open(env_example_path, 'w') as f:
        f.write("""# Alerting Configuration Environment Variables
# Copy to .env or set in your deployment environment

# Enable/disable alerting system
ALERTING_ENABLED=true

# Error rate threshold (0.05 = 5%)
ERROR_RATE_THRESHOLD=0.05

# Latency threshold in seconds
LATENCY_THRESHOLD_SECONDS=2.0

# Latency percentile to monitor (95 = 95th percentile)  
LATENCY_PERCENTILE=95

# How often to check alerts
ALERT_CHECK_INTERVAL=5m

# How long condition must persist before firing
ALERT_DURATION=5m

# Optional: Path to save Prometheus rules
PROMETHEUS_RULES_PATH=./monitoring/lexgraph_alerts.yml
""")
    
    print(f"✅ Created alerting configuration example: {env_example_path}")
    
    # Display the generated rules
    print("\n" + "="*60)
    print("GENERATED PROMETHEUS ALERTING RULES:")
    print("="*60)
    print(alerts_yaml)

if __name__ == "__main__":
    main()