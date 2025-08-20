"""Comprehensive health check system for LexGraph Legal RAG."""

from __future__ import annotations

import logging
import os
import time
from enum import Enum
from typing import Any

import httpx
import psutil
from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel


logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health check status enumeration."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


class ComponentHealth(BaseModel):
    """Health status of an individual component."""

    name: str
    status: HealthStatus
    message: str | None = None
    response_time_ms: float | None = None
    last_checked: float
    details: dict[str, Any] = {}


class SystemHealth(BaseModel):
    """Overall system health status."""

    status: HealthStatus
    timestamp: float
    version: str
    uptime_seconds: float
    checks: list[ComponentHealth]
    summary: dict[str, Any]


class HealthChecker:
    """Comprehensive health check system."""

    def __init__(self):
        self.start_time = time.time()
        self.version = os.getenv("VERSION", "1.0.0")
        self.checks = []
        self._register_default_checks()

    def _register_default_checks(self):
        """Register default health checks."""
        self.checks = [
            self._check_system_resources,
            self._check_disk_space,
            self._check_memory_usage,
            self._check_api_endpoints,
            self._check_dependencies,
            self._check_database_connection,
            self._check_cache_connection,
            self._check_external_apis,
        ]

    async def get_health(self, detailed: bool = False) -> SystemHealth:
        """Get comprehensive system health status."""
        start_time = time.time()
        check_results = []

        # Run all health checks
        for check in self.checks:
            try:
                result = await check()
                check_results.append(result)
            except Exception as e:
                logger.error(f"Health check failed: {check.__name__}: {e}")
                check_results.append(
                    ComponentHealth(
                        name=check.__name__.replace("_check_", ""),
                        status=HealthStatus.UNHEALTHY,
                        message=f"Check failed: {e!s}",
                        last_checked=time.time(),
                    )
                )

        # Determine overall status
        overall_status = self._determine_overall_status(check_results)

        # Create summary
        summary = self._create_summary(check_results)
        summary["total_checks"] = len(check_results)
        summary["health_check_duration_ms"] = (time.time() - start_time) * 1000

        return SystemHealth(
            status=overall_status,
            timestamp=time.time(),
            version=self.version,
            uptime_seconds=time.time() - self.start_time,
            checks=check_results if detailed else [],
            summary=summary,
        )

    def _determine_overall_status(self, checks: list[ComponentHealth]) -> HealthStatus:
        """Determine overall system health from component checks."""
        if not checks:
            return HealthStatus.UNKNOWN

        statuses = [check.status for check in checks]

        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        elif all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN

    def _create_summary(self, checks: list[ComponentHealth]) -> dict[str, Any]:
        """Create health check summary."""
        summary = {
            "healthy": sum(
                1 for check in checks if check.status == HealthStatus.HEALTHY
            ),
            "unhealthy": sum(
                1 for check in checks if check.status == HealthStatus.UNHEALTHY
            ),
            "degraded": sum(
                1 for check in checks if check.status == HealthStatus.DEGRADED
            ),
            "unknown": sum(
                1 for check in checks if check.status == HealthStatus.UNKNOWN
            ),
        }

        response_times = [
            check.response_time_ms for check in checks if check.response_time_ms
        ]
        if response_times:
            summary["avg_response_time_ms"] = sum(response_times) / len(response_times)
            summary["max_response_time_ms"] = max(response_times)

        return summary

    async def _check_system_resources(self) -> ComponentHealth:
        """Check system CPU and memory resources."""
        start_time = time.time()

        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()

            status = HealthStatus.HEALTHY
            message = "System resources normal"

            if cpu_percent > 90:
                status = HealthStatus.UNHEALTHY
                message = f"High CPU usage: {cpu_percent}%"
            elif cpu_percent > 75:
                status = HealthStatus.DEGRADED
                message = f"Elevated CPU usage: {cpu_percent}%"

            if memory.percent > 90:
                status = HealthStatus.UNHEALTHY
                message = f"High memory usage: {memory.percent}%"
            elif memory.percent > 75 and status == HealthStatus.HEALTHY:
                status = HealthStatus.DEGRADED
                message = f"Elevated memory usage: {memory.percent}%"

            return ComponentHealth(
                name="system_resources",
                status=status,
                message=message,
                response_time_ms=(time.time() - start_time) * 1000,
                last_checked=time.time(),
                details={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available_gb": memory.available / (1024**3),
                },
            )
        except Exception as e:
            return ComponentHealth(
                name="system_resources",
                status=HealthStatus.UNHEALTHY,
                message=f"Failed to check system resources: {e!s}",
                last_checked=time.time(),
            )

    async def _check_disk_space(self) -> ComponentHealth:
        """Check available disk space."""
        start_time = time.time()

        try:
            disk = psutil.disk_usage("/")
            usage_percent = (disk.used / disk.total) * 100

            status = HealthStatus.HEALTHY
            message = "Disk space sufficient"

            if usage_percent > 95:
                status = HealthStatus.UNHEALTHY
                message = f"Critical disk space: {usage_percent:.1f}% used"
            elif usage_percent > 85:
                status = HealthStatus.DEGRADED
                message = f"Low disk space: {usage_percent:.1f}% used"

            return ComponentHealth(
                name="disk_space",
                status=status,
                message=message,
                response_time_ms=(time.time() - start_time) * 1000,
                last_checked=time.time(),
                details={
                    "usage_percent": usage_percent,
                    "free_gb": disk.free / (1024**3),
                    "total_gb": disk.total / (1024**3),
                },
            )
        except Exception as e:
            return ComponentHealth(
                name="disk_space",
                status=HealthStatus.UNHEALTHY,
                message=f"Failed to check disk space: {e!s}",
                last_checked=time.time(),
            )

    async def _check_memory_usage(self) -> ComponentHealth:
        """Check application memory usage."""
        start_time = time.time()

        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)

            status = HealthStatus.HEALTHY
            message = f"Memory usage normal: {memory_mb:.1f}MB"

            # Configurable thresholds
            warning_threshold = float(os.getenv("MEMORY_WARNING_MB", "1000"))
            critical_threshold = float(os.getenv("MEMORY_CRITICAL_MB", "2000"))

            if memory_mb > critical_threshold:
                status = HealthStatus.UNHEALTHY
                message = f"High memory usage: {memory_mb:.1f}MB"
            elif memory_mb > warning_threshold:
                status = HealthStatus.DEGRADED
                message = f"Elevated memory usage: {memory_mb:.1f}MB"

            return ComponentHealth(
                name="memory_usage",
                status=status,
                message=message,
                response_time_ms=(time.time() - start_time) * 1000,
                last_checked=time.time(),
                details={
                    "memory_mb": memory_mb,
                    "memory_percent": process.memory_percent(),
                },
            )
        except Exception as e:
            return ComponentHealth(
                name="memory_usage",
                status=HealthStatus.UNHEALTHY,
                message=f"Failed to check memory usage: {e!s}",
                last_checked=time.time(),
            )

    async def _check_api_endpoints(self) -> ComponentHealth:
        """Check critical API endpoints."""
        start_time = time.time()

        try:
            # Test internal endpoints
            base_url = os.getenv("BASE_URL", "http://localhost:8000")
            timeout = float(os.getenv("HEALTH_CHECK_TIMEOUT", "5.0"))

            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(f"{base_url}/health")

            if response.status_code == 200:
                status = HealthStatus.HEALTHY
                message = "API endpoints responding"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"API endpoint returned {response.status_code}"

            return ComponentHealth(
                name="api_endpoints",
                status=status,
                message=message,
                response_time_ms=(time.time() - start_time) * 1000,
                last_checked=time.time(),
                details={"status_code": response.status_code, "base_url": base_url},
            )
        except Exception as e:
            return ComponentHealth(
                name="api_endpoints",
                status=HealthStatus.UNHEALTHY,
                message=f"API endpoint check failed: {e!s}",
                last_checked=time.time(),
            )

    async def _check_dependencies(self) -> ComponentHealth:
        """Check required dependencies and imports."""
        start_time = time.time()

        try:
            # Test critical imports
            import fastapi
            import prometheus_client
            import structlog
            import uvicorn

            status = HealthStatus.HEALTHY
            message = "All dependencies available"

            return ComponentHealth(
                name="dependencies",
                status=status,
                message=message,
                response_time_ms=(time.time() - start_time) * 1000,
                last_checked=time.time(),
                details={
                    "fastapi_version": fastapi.__version__,
                    "uvicorn_version": uvicorn.__version__,
                },
            )
        except ImportError as e:
            return ComponentHealth(
                name="dependencies",
                status=HealthStatus.UNHEALTHY,
                message=f"Missing dependency: {e!s}",
                last_checked=time.time(),
            )
        except Exception as e:
            return ComponentHealth(
                name="dependencies",
                status=HealthStatus.UNHEALTHY,
                message=f"Dependency check failed: {e!s}",
                last_checked=time.time(),
            )

    async def _check_database_connection(self) -> ComponentHealth:
        """Check database connectivity (if applicable)."""
        start_time = time.time()

        # For this application, we might not have a traditional database
        # This is a placeholder for future database connections
        try:
            status = HealthStatus.HEALTHY
            message = "No database configured"

            # If database is configured, add actual connection test here
            db_url = os.getenv("DATABASE_URL")
            if db_url:
                # Add actual database connection test
                message = "Database connection not tested (placeholder)"
                status = HealthStatus.UNKNOWN

            return ComponentHealth(
                name="database",
                status=status,
                message=message,
                response_time_ms=(time.time() - start_time) * 1000,
                last_checked=time.time(),
                details={"database_url_configured": bool(db_url)},
            )
        except Exception as e:
            return ComponentHealth(
                name="database",
                status=HealthStatus.UNHEALTHY,
                message=f"Database check failed: {e!s}",
                last_checked=time.time(),
            )

    async def _check_cache_connection(self) -> ComponentHealth:
        """Check cache connectivity (Redis, etc.)."""
        start_time = time.time()

        try:
            status = HealthStatus.HEALTHY
            message = "No cache configured"

            # If Redis or other cache is configured, add connection test
            cache_url = os.getenv("REDIS_URL")
            if cache_url:
                # Add actual cache connection test
                message = "Cache connection not tested (placeholder)"
                status = HealthStatus.UNKNOWN

            return ComponentHealth(
                name="cache",
                status=status,
                message=message,
                response_time_ms=(time.time() - start_time) * 1000,
                last_checked=time.time(),
                details={"cache_url_configured": bool(cache_url)},
            )
        except Exception as e:
            return ComponentHealth(
                name="cache",
                status=HealthStatus.UNHEALTHY,
                message=f"Cache check failed: {e!s}",
                last_checked=time.time(),
            )

    async def _check_external_apis(self) -> ComponentHealth:
        """Check external API dependencies."""
        start_time = time.time()

        try:
            status = HealthStatus.HEALTHY
            message = "External APIs not configured"

            # Check OpenAI API if configured
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                # Add actual OpenAI API test here
                message = "External API checks not implemented (placeholder)"
                status = HealthStatus.UNKNOWN

            return ComponentHealth(
                name="external_apis",
                status=status,
                message=message,
                response_time_ms=(time.time() - start_time) * 1000,
                last_checked=time.time(),
                details={"apis_configured": bool(openai_key)},
            )
        except Exception as e:
            return ComponentHealth(
                name="external_apis",
                status=HealthStatus.UNHEALTHY,
                message=f"External API check failed: {e!s}",
                last_checked=time.time(),
            )


# Global health checker instance
health_checker = HealthChecker()


def add_health_endpoints(app: FastAPI):
    """Add health check endpoints to FastAPI app."""

    @app.get("/health", response_model=SystemHealth, tags=["Health"])
    async def health_check():
        """Basic health check endpoint."""
        return await health_checker.get_health(detailed=False)

    @app.get("/health/detailed", response_model=SystemHealth, tags=["Health"])
    async def detailed_health_check():
        """Detailed health check with all component statuses."""
        return await health_checker.get_health(detailed=True)

    @app.get("/ready", tags=["Health"])
    async def readiness_check():
        """Kubernetes readiness probe endpoint."""
        health = await health_checker.get_health(detailed=False)
        if health.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]:
            return {"status": "ready"}
        else:
            raise HTTPException(status_code=503, detail="Service not ready")

    @app.get("/live", tags=["Health"])
    async def liveness_check():
        """Kubernetes liveness probe endpoint."""
        # Simple liveness check - just verify the process is running
        return {"status": "alive", "timestamp": time.time()}

    @app.get("/metrics/health", tags=["Health"])
    async def health_metrics():
        """Health metrics in Prometheus format."""
        health = await health_checker.get_health(detailed=True)

        metrics = []
        metrics.append(
            f'health_status{{service="lexgraph"}} {1 if health.status == HealthStatus.HEALTHY else 0}'
        )
        metrics.append(
            f'health_uptime_seconds{{service="lexgraph"}} {health.uptime_seconds}'
        )

        for check in health.checks:
            status_value = 1 if check.status == HealthStatus.HEALTHY else 0
            metrics.append(
                f'health_component_status{{component="{check.name}"}} {status_value}'
            )

            if check.response_time_ms:
                metrics.append(
                    f'health_component_response_time_ms{{component="{check.name}"}} {check.response_time_ms}'
                )

        return "\n".join(metrics)
