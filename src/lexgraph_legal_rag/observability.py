"""Enhanced observability with OpenTelemetry distributed tracing and metrics."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any

from opentelemetry import metrics

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.semconv.resource import ResourceAttributes
from opentelemetry.trace import Status
from opentelemetry.trace import StatusCode
from prometheus_client import Counter
from prometheus_client import Gauge
from prometheus_client import Histogram

# Prometheus client
from prometheus_client import start_http_server


logger = logging.getLogger(__name__)

# Service information
SERVICE_NAME = "lexgraph-legal-rag"
SERVICE_VERSION = "1.0.0"
SERVICE_ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

# Global tracer and meter
tracer: trace.Tracer | None = None
meter: metrics.Meter | None = None

# Enhanced metrics for observability
AGENT_OPERATIONS = Counter(
    "agent_operations_total",
    "Total number of agent operations",
    ["agent_type", "operation", "status"],
)

AGENT_OPERATION_DURATION = Histogram(
    "agent_operation_duration_seconds",
    "Duration of agent operations",
    ["agent_type", "operation"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
)

DOCUMENT_PROCESSING = Counter(
    "document_processing_total",
    "Total number of documents processed",
    ["pipeline_stage", "status"],
)

RAG_QUERIES = Counter(
    "rag_queries_total",
    "Total number of RAG queries",
    ["query_type", "agent", "status"],
)

RAG_QUERY_LATENCY = Histogram(
    "rag_query_latency_seconds",
    "RAG query processing latency",
    ["query_type", "agent"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

SYSTEM_HEALTH = Gauge("system_health_score", "Overall system health score (0-1)")

ACTIVE_CONNECTIONS = Gauge("active_connections_count", "Number of active connections")

ERROR_TRACKING = Counter(
    "errors_total",
    "Total number of errors by type and component",
    ["error_type", "component", "severity"],
)


def initialize_observability(
    enable_tracing: bool = True,
    enable_metrics: bool = True,
    prometheus_port: int = 8001,
) -> None:
    """Initialize OpenTelemetry observability stack."""
    global tracer, meter

    logger.info(f"Initializing observability for {SERVICE_NAME} v{SERVICE_VERSION}")

    # Create resource with service information
    resource = Resource.create(
        {
            ResourceAttributes.SERVICE_NAME: SERVICE_NAME,
            ResourceAttributes.SERVICE_VERSION: SERVICE_VERSION,
            ResourceAttributes.SERVICE_NAMESPACE: "lexgraph",
            ResourceAttributes.DEPLOYMENT_ENVIRONMENT: SERVICE_ENVIRONMENT,
        }
    )

    if enable_tracing:
        # Initialize tracing
        trace_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(trace_provider)

        # Add console exporter for development
        if SERVICE_ENVIRONMENT == "development":
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter

            console_exporter = ConsoleSpanExporter()
            trace_provider.add_span_processor(BatchSpanProcessor(console_exporter))

        tracer = trace.get_tracer(__name__)
        logger.info("✅ Distributed tracing initialized")

    if enable_metrics:
        # Initialize metrics with Prometheus exporter
        prometheus_reader = PrometheusMetricReader()
        metric_provider = MeterProvider(
            resource=resource, metric_readers=[prometheus_reader]
        )
        metrics.set_meter_provider(metric_provider)
        meter = metrics.get_meter(__name__)

        # Start Prometheus metrics server
        try:
            start_http_server(prometheus_port)
            logger.info(
                f"✅ Prometheus metrics server started on port {prometheus_port}"
            )
        except Exception as e:
            logger.warning(f"Could not start Prometheus server: {e}")

    # Auto-instrument FastAPI and HTTPX
    try:
        FastAPIInstrumentor.instrument()
        HTTPXClientInstrumentor.instrument()
        logger.info("✅ Auto-instrumentation enabled for FastAPI and HTTPX")
    except Exception as e:
        logger.warning(f"Auto-instrumentation failed: {e}")


@contextmanager
def trace_operation(
    operation_name: str,
    component: str = "unknown",
    attributes: dict[str, Any] | None = None,
):
    """Context manager for tracing operations with error handling."""
    if not tracer:
        # If tracing not initialized, just yield
        yield None
        return

    with tracer.start_as_current_span(operation_name) as span:
        try:
            # Add basic attributes
            span.set_attribute("component", component)
            span.set_attribute("service.name", SERVICE_NAME)
            span.set_attribute("environment", SERVICE_ENVIRONMENT)

            # Add custom attributes
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, str(value))

            yield span

            # Mark as successful
            span.set_status(Status(StatusCode.OK))

        except Exception as e:
            # Record error in span
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)

            # Track error in metrics
            ERROR_TRACKING.labels(
                error_type=type(e).__name__, component=component, severity="error"
            ).inc()

            raise


def track_agent_operation(
    agent_type: str,
    operation: str,
    duration_seconds: float,
    success: bool = True,
    attributes: dict[str, Any] | None = None,
) -> None:
    """Track agent operation metrics."""
    status = "success" if success else "failure"

    # Record operation count
    AGENT_OPERATIONS.labels(
        agent_type=agent_type, operation=operation, status=status
    ).inc()

    # Record operation duration
    AGENT_OPERATION_DURATION.labels(agent_type=agent_type, operation=operation).observe(
        duration_seconds
    )

    # Add to trace if available
    if tracer:
        with tracer.start_as_current_span(f"agent.{operation}") as span:
            span.set_attribute("agent.type", agent_type)
            span.set_attribute("agent.operation", operation)
            span.set_attribute("agent.duration", duration_seconds)
            span.set_attribute("agent.success", success)

            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(f"agent.{key}", str(value))


def track_rag_query(
    query_type: str,
    agent: str,
    latency_seconds: float,
    success: bool = True,
    result_count: int = 0,
) -> None:
    """Track RAG query metrics with detailed context."""
    status = "success" if success else "failure"

    # Record query count
    RAG_QUERIES.labels(query_type=query_type, agent=agent, status=status).inc()

    # Record query latency
    RAG_QUERY_LATENCY.labels(query_type=query_type, agent=agent).observe(
        latency_seconds
    )

    # Add trace context
    if tracer:
        with tracer.start_as_current_span(f"rag.query.{query_type}") as span:
            span.set_attribute("rag.query_type", query_type)
            span.set_attribute("rag.agent", agent)
            span.set_attribute("rag.latency", latency_seconds)
            span.set_attribute("rag.success", success)
            span.set_attribute("rag.result_count", result_count)


def track_document_processing(
    pipeline_stage: str, success: bool = True, document_count: int = 1
) -> None:
    """Track document processing pipeline metrics."""
    status = "success" if success else "failure"

    DOCUMENT_PROCESSING.labels(pipeline_stage=pipeline_stage, status=status).inc(
        document_count
    )


def update_system_health(health_score: float) -> None:
    """Update overall system health score (0.0 to 1.0)."""
    SYSTEM_HEALTH.set(max(0.0, min(1.0, health_score)))


def track_active_connections(count: int) -> None:
    """Track number of active connections."""
    ACTIVE_CONNECTIONS.set(count)


def track_error(error: Exception, component: str, severity: str = "error") -> None:
    """Track error with classification."""
    ERROR_TRACKING.labels(
        error_type=type(error).__name__, component=component, severity=severity
    ).inc()

    # Add to current trace if available
    span = trace.get_current_span()
    if span and span.is_recording():
        span.record_exception(error)
        span.set_status(Status(StatusCode.ERROR, str(error)))


class ObservabilityMixin:
    """Mixin class to add observability to any component."""

    def __init__(self, component_name: str):
        self.component_name = component_name

    @contextmanager
    def trace(self, operation: str, **attributes):
        """Trace an operation with automatic error handling."""
        with trace_operation(
            f"{self.component_name}.{operation}",
            component=self.component_name,
            attributes=attributes,
        ) as span:
            yield span

    def track_operation(
        self, operation: str, duration: float, success: bool = True, **attributes
    ):
        """Track operation metrics."""
        track_agent_operation(
            agent_type=self.component_name,
            operation=operation,
            duration_seconds=duration,
            success=success,
            attributes=attributes,
        )

    def track_error(self, error: Exception, severity: str = "error"):
        """Track error with component context."""
        track_error(error, self.component_name, severity)


def get_observability_info() -> dict[str, Any]:
    """Get current observability configuration and status."""
    return {
        "service": {
            "name": SERVICE_NAME,
            "version": SERVICE_VERSION,
            "environment": SERVICE_ENVIRONMENT,
        },
        "tracing": {
            "enabled": tracer is not None,
            "provider": str(trace.get_tracer_provider()) if tracer else None,
        },
        "metrics": {
            "enabled": meter is not None,
            "provider": str(metrics.get_meter_provider()) if meter else None,
        },
    }
