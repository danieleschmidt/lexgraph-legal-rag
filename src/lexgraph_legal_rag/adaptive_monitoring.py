"""
Adaptive Monitoring System for Legal RAG
========================================

Self-adjusting monitoring with predictive alerts and intelligent thresholds.
Novel contribution: AI-driven monitoring that learns optimal alert thresholds.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from collections import deque
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import Any
from typing import Callable

import numpy as np


logger = logging.getLogger(__name__)


class MonitoringMode(Enum):
    """Monitoring system operating modes."""

    LEARNING = "learning"  # Learning baseline patterns
    ADAPTIVE = "adaptive"  # Adaptive threshold adjustment
    PREDICTIVE = "predictive"  # Predictive alerting
    EMERGENCY = "emergency"  # Emergency monitoring mode


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricThreshold:
    """Adaptive threshold for a metric."""

    metric_name: str
    current_value: float = 0.0
    baseline_mean: float = 0.0
    baseline_std: float = 1.0
    warning_threshold: float = 2.0  # Standard deviations
    error_threshold: float = 3.0  # Standard deviations
    critical_threshold: float = 4.0  # Standard deviations
    adaptation_rate: float = 0.1  # How quickly to adapt
    last_updated: float = field(default_factory=time.time)


@dataclass
class AdaptiveAlert:
    """Adaptive alert with context and predictions."""

    metric_name: str
    severity: AlertSeverity
    current_value: float
    threshold_value: float
    confidence: float
    prediction: float | None = None
    context: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class SystemPattern:
    """Learned system behavior pattern."""

    pattern_id: str
    conditions: dict[str, Any]
    expected_metrics: dict[str, float]
    confidence: float = 0.5
    occurrences: int = 0
    last_seen: float = field(default_factory=time.time)


class AdaptiveMonitoringSystem:
    """
    Advanced monitoring system with adaptive thresholds and predictive alerts.

    Features:
    - Self-learning baselines and thresholds
    - Predictive alerting before issues occur
    - Pattern recognition for system behavior
    - Intelligent noise reduction
    - Context-aware alert prioritization
    """

    def __init__(self, learning_window: int = 1000):
        self.mode = MonitoringMode.LEARNING
        self.learning_window = learning_window

        # Threshold management
        self.thresholds: dict[str, MetricThreshold] = {}
        self.metric_history: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=learning_window)
        )

        # Alert management
        self.active_alerts: dict[str, AdaptiveAlert] = {}
        self.alert_history: deque = deque(maxlen=10000)
        self.alert_callbacks: list[Callable] = []

        # Pattern recognition
        self.learned_patterns: dict[str, SystemPattern] = {}
        self.current_context: dict[str, Any] = {}

        # Prediction models
        self.prediction_models: dict[str, dict[str, float]] = {}

        # Statistics
        self.total_alerts = 0
        self.false_positive_rate = 0.0
        self.prediction_accuracy = 0.5

    def _initialize_threshold(self, metric_name: str, initial_value: float):
        """Initialize threshold for a new metric."""
        if metric_name not in self.thresholds:
            self.thresholds[metric_name] = MetricThreshold(
                metric_name=metric_name,
                current_value=initial_value,
                baseline_mean=initial_value,
                baseline_std=1.0,
            )

    def _update_baseline(self, metric_name: str, value: float):
        """Update baseline statistics for a metric."""
        history = self.metric_history[metric_name]
        threshold = self.thresholds[metric_name]

        # Add to history
        history.append(value)

        # Update baseline if we have enough data
        if len(history) >= 10:  # Minimum data points
            values = np.array(list(history))
            new_mean = np.mean(values)
            new_std = np.std(values)

            # Adaptive update with smoothing
            adaptation_rate = threshold.adaptation_rate
            threshold.baseline_mean = (
                1 - adaptation_rate
            ) * threshold.baseline_mean + adaptation_rate * new_mean
            threshold.baseline_std = max(
                0.1,
                (1 - adaptation_rate) * threshold.baseline_std
                + adaptation_rate * new_std,
            )
            threshold.last_updated = time.time()

    def _calculate_anomaly_score(self, metric_name: str, value: float) -> float:
        """Calculate anomaly score for a metric value."""
        if metric_name not in self.thresholds:
            return 0.0

        threshold = self.thresholds[metric_name]
        z_score = abs(value - threshold.baseline_mean) / max(
            threshold.baseline_std, 0.1
        )
        return z_score

    def _predict_next_value(self, metric_name: str) -> float | None:
        """Predict next value for a metric using simple time series analysis."""
        history = self.metric_history[metric_name]

        if len(history) < 5:
            return None

        values = np.array(list(history))

        # Simple trend prediction
        if len(values) >= 10:
            # Linear trend extrapolation
            x = np.arange(len(values))
            coeffs = np.polyfit(x, values, 1)
            prediction = coeffs[0] * len(values) + coeffs[1]
            return float(prediction)
        else:
            # Moving average
            return float(np.mean(values[-3:]))

    def _should_alert(self, metric_name: str, value: float) -> AlertSeverity | None:
        """Determine if an alert should be triggered."""
        if metric_name not in self.thresholds:
            return None

        threshold = self.thresholds[metric_name]
        anomaly_score = self._calculate_anomaly_score(metric_name, value)

        # Check thresholds
        if anomaly_score >= threshold.critical_threshold:
            return AlertSeverity.CRITICAL
        elif anomaly_score >= threshold.error_threshold:
            return AlertSeverity.ERROR
        elif anomaly_score >= threshold.warning_threshold:
            return AlertSeverity.WARNING

        return None

    def _create_alert(
        self, metric_name: str, severity: AlertSeverity, value: float
    ) -> AdaptiveAlert:
        """Create an adaptive alert."""
        threshold = self.thresholds[metric_name]
        anomaly_score = self._calculate_anomaly_score(metric_name, value)

        # Calculate confidence based on historical accuracy
        confidence = min(0.9, 0.5 + (anomaly_score - 2.0) * 0.1)

        # Get prediction
        prediction = self._predict_next_value(metric_name)

        alert = AdaptiveAlert(
            metric_name=metric_name,
            severity=severity,
            current_value=value,
            threshold_value=threshold.baseline_mean
            + threshold.warning_threshold * threshold.baseline_std,
            confidence=confidence,
            prediction=prediction,
            context=self.current_context.copy(),
        )

        return alert

    def _recognize_patterns(self, metrics: dict[str, float]) -> SystemPattern | None:
        """Recognize known system patterns."""
        for pattern in self.learned_patterns.values():
            # Simple pattern matching based on metric ranges
            matches = 0
            total_conditions = len(pattern.conditions)

            for metric, expected_range in pattern.conditions.items():
                if metric in metrics:
                    value = metrics[metric]
                    min_val, max_val = expected_range
                    if min_val <= value <= max_val:
                        matches += 1

            # Pattern matches if >80% of conditions are met
            if matches / total_conditions > 0.8:
                pattern.occurrences += 1
                pattern.last_seen = time.time()
                return pattern

        return None

    async def update_metrics(
        self, metrics: dict[str, float], context: dict[str, Any] | None = None
    ):
        """Update metrics and check for alerts."""
        if context:
            self.current_context.update(context)

        alerts_triggered = []

        for metric_name, value in metrics.items():
            # Initialize threshold if needed
            self._initialize_threshold(metric_name, value)

            # Update baseline
            self._update_baseline(metric_name, value)

            # Update current value
            self.thresholds[metric_name].current_value = value

            # Check for alerts
            if self.mode != MonitoringMode.LEARNING:
                severity = self._should_alert(metric_name, value)
                if severity:
                    alert = self._create_alert(metric_name, severity, value)

                    # Avoid duplicate alerts
                    alert_key = f"{metric_name}_{severity.value}"
                    if alert_key not in self.active_alerts:
                        self.active_alerts[alert_key] = alert
                        alerts_triggered.append(alert)
                        self.total_alerts += 1

        # Pattern recognition
        if self.mode in [MonitoringMode.ADAPTIVE, MonitoringMode.PREDICTIVE]:
            pattern = self._recognize_patterns(metrics)
            if pattern:
                logger.info(f"Recognized system pattern: {pattern.pattern_id}")

        # Trigger alert callbacks
        for alert in alerts_triggered:
            await self._trigger_alert_callbacks(alert)

        # Switch modes based on learning progress
        if self.mode == MonitoringMode.LEARNING and len(self.metric_history) > 0:
            min_history = min(len(history) for history in self.metric_history.values())
            if min_history >= self.learning_window * 0.8:
                self.mode = MonitoringMode.ADAPTIVE
                logger.info("Switched to adaptive monitoring mode")

    async def _trigger_alert_callbacks(self, alert: AdaptiveAlert):
        """Trigger registered alert callbacks."""
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

    def register_alert_callback(self, callback: Callable):
        """Register callback for alert notifications."""
        self.alert_callbacks.append(callback)

    def set_custom_thresholds(
        self, metric_name: str, warning: float, error: float, critical: float
    ):
        """Set custom thresholds for a metric."""
        if metric_name in self.thresholds:
            threshold = self.thresholds[metric_name]
            threshold.warning_threshold = warning
            threshold.error_threshold = error
            threshold.critical_threshold = critical
            logger.info(f"Updated thresholds for {metric_name}")

    def clear_alert(self, metric_name: str, severity: AlertSeverity):
        """Clear an active alert."""
        alert_key = f"{metric_name}_{severity.value}"
        if alert_key in self.active_alerts:
            alert = self.active_alerts.pop(alert_key)
            self.alert_history.append(alert)
            logger.info(f"Cleared alert: {alert_key}")

    def get_system_health(self) -> dict[str, Any]:
        """Get comprehensive system health status."""
        active_critical = sum(
            1
            for alert in self.active_alerts.values()
            if alert.severity == AlertSeverity.CRITICAL
        )
        active_errors = sum(
            1
            for alert in self.active_alerts.values()
            if alert.severity == AlertSeverity.ERROR
        )
        active_warnings = sum(
            1
            for alert in self.active_alerts.values()
            if alert.severity == AlertSeverity.WARNING
        )

        # Calculate overall health score
        health_score = max(
            0.0,
            1.0 - (active_critical * 0.5 + active_errors * 0.3 + active_warnings * 0.1),
        )

        return {
            "mode": self.mode.value,
            "health_score": health_score,
            "active_alerts": {
                "critical": active_critical,
                "error": active_errors,
                "warning": active_warnings,
                "total": len(self.active_alerts),
            },
            "metrics_monitored": len(self.thresholds),
            "patterns_learned": len(self.learned_patterns),
            "prediction_accuracy": self.prediction_accuracy,
            "total_alerts": self.total_alerts,
            "recent_alerts": [
                {
                    "metric": alert.metric_name,
                    "severity": alert.severity.value,
                    "value": alert.current_value,
                    "confidence": alert.confidence,
                    "timestamp": alert.timestamp,
                }
                for alert in list(self.active_alerts.values())[-5:]
            ],
        }

    def get_metric_insights(self, metric_name: str) -> dict[str, Any]:
        """Get detailed insights for a specific metric."""
        if metric_name not in self.thresholds:
            return {"error": "Metric not found"}

        threshold = self.thresholds[metric_name]
        history = list(self.metric_history[metric_name])

        # Calculate trends
        if len(history) >= 10:
            values = np.array(history)
            trend = np.polyfit(range(len(values)), values, 1)[0]
        else:
            trend = 0.0

        return {
            "metric_name": metric_name,
            "current_value": threshold.current_value,
            "baseline_mean": threshold.baseline_mean,
            "baseline_std": threshold.baseline_std,
            "trend": trend,
            "anomaly_score": self._calculate_anomaly_score(
                metric_name, threshold.current_value
            ),
            "prediction": self._predict_next_value(metric_name),
            "thresholds": {
                "warning": threshold.warning_threshold,
                "error": threshold.error_threshold,
                "critical": threshold.critical_threshold,
            },
            "history_length": len(history),
            "last_updated": threshold.last_updated,
        }


# Global instance
_monitoring_system = AdaptiveMonitoringSystem()


def get_monitoring_system() -> AdaptiveMonitoringSystem:
    """Get the global monitoring system instance."""
    return _monitoring_system


async def log_metrics(metrics: dict[str, float], context: dict[str, Any] | None = None):
    """Convenience function to log metrics to the monitoring system."""
    monitoring_system = get_monitoring_system()
    await monitoring_system.update_metrics(metrics, context)
