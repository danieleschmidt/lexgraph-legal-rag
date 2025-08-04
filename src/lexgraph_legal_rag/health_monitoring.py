"""Comprehensive health monitoring and system diagnostics."""

from __future__ import annotations

import asyncio
import logging
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
import psutil

from .resilience import CircuitBreaker, get_circuit_breaker

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str
    duration_ms: float
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class SystemMetrics:
    """System-wide metrics snapshot."""
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_percent: float
    load_average: List[float]
    active_connections: int
    timestamp: float = field(default_factory=time.time)


class HealthChecker:
    """Performs various system health checks."""
    
    def __init__(self):
        self.checks: Dict[str, Callable[[], HealthCheckResult]] = {}
        self.last_results: Dict[str, HealthCheckResult] = {}
        self._lock = threading.Lock()
    
    def register_check(self, name: str, check_func: Callable[[], HealthCheckResult]) -> None:
        """Register a health check function."""
        with self._lock:
            self.checks[name] = check_func
            logger.debug(f"Registered health check: {name}")
    
    def run_check(self, name: str) -> HealthCheckResult:
        """Run a specific health check."""
        if name not in self.checks:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNKNOWN,
                message=f"Health check '{name}' not found",
                duration_ms=0.0
            )
        
        start_time = time.time()
        try:
            result = self.checks[name]()
            result.duration_ms = (time.time() - start_time) * 1000
            
            with self._lock:
                self.last_results[name] = result
            
            return result
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            result = HealthCheckResult(
                name=name,
                status=HealthStatus.CRITICAL,
                message=f"Health check failed: {str(e)}",
                duration_ms=duration_ms,
                details={"error": str(e), "type": type(e).__name__}
            )
            
            with self._lock:
                self.last_results[name] = result
            
            logger.error(f"Health check '{name}' failed: {e}")
            return result
    
    def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks."""
        results = {}
        
        for name in self.checks:
            results[name] = self.run_check(name)
        
        return results
    
    def get_overall_status(self) -> HealthStatus:
        """Get overall system health status."""
        if not self.last_results:
            return HealthStatus.UNKNOWN
        
        statuses = [result.status for result in self.last_results.values()]
        
        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.WARNING in statuses:
            return HealthStatus.WARNING
        elif all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN


class SystemMonitor:
    """Monitors system resources and performance."""
    
    def __init__(self):
        self.metrics_history: List[SystemMetrics] = []
        self.max_history = 1000  # Keep last 1000 metrics snapshots
        self._lock = threading.Lock()
    
    def collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_gb = memory.available / (1024**3)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # Load average (Unix-like systems)
            try:
                load_average = list(psutil.getloadavg())
            except AttributeError:
                # Windows doesn't have load average
                load_average = [0.0, 0.0, 0.0]
            
            # Network connections
            connections = psutil.net_connections()
            active_connections = len([c for c in connections if c.status == 'ESTABLISHED'])
            
            metrics = SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_available_gb=memory_available_gb,
                disk_percent=disk_percent,
                load_average=load_average,
                active_connections=active_connections
            )
            
            # Store in history
            with self._lock:
                self.metrics_history.append(metrics)
                if len(self.metrics_history) > self.max_history:
                    self.metrics_history.pop(0)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            raise
    
    def get_metrics_summary(self, minutes: int = 5) -> Dict[str, Any]:
        """Get metrics summary for the last N minutes."""
        cutoff_time = time.time() - (minutes * 60)
        
        with self._lock:
            recent_metrics = [
                m for m in self.metrics_history 
                if m.timestamp >= cutoff_time
            ]
        
        if not recent_metrics:
            return {}
        
        # Calculate averages and extremes
        cpu_values = [m.cpu_percent for m in recent_metrics]
        memory_values = [m.memory_percent for m in recent_metrics]
        
        return {
            "timespan_minutes": minutes,
            "sample_count": len(recent_metrics),
            "cpu": {
                "avg": sum(cpu_values) / len(cpu_values),
                "min": min(cpu_values),
                "max": max(cpu_values),
                "current": recent_metrics[-1].cpu_percent
            },
            "memory": {
                "avg": sum(memory_values) / len(memory_values),
                "min": min(memory_values),
                "max": max(memory_values),
                "current": recent_metrics[-1].memory_percent,
                "available_gb": recent_metrics[-1].memory_available_gb
            },
            "disk": {
                "current": recent_metrics[-1].disk_percent
            },
            "load_average": recent_metrics[-1].load_average,
            "connections": recent_metrics[-1].active_connections
        }


class LegalRAGHealthChecker(HealthChecker):
    """Specialized health checker for Legal RAG system."""
    
    def __init__(self, system_monitor: Optional[SystemMonitor] = None):
        super().__init__()
        self.system_monitor = system_monitor or SystemMonitor()
        self._setup_default_checks()
    
    def _setup_default_checks(self) -> None:
        """Setup default health checks for Legal RAG system."""
        self.register_check("system_resources", self._check_system_resources)
        self.register_check("circuit_breakers", self._check_circuit_breakers)
        self.register_check("memory_usage", self._check_memory_usage)
        self.register_check("disk_space", self._check_disk_space)
        self.register_check("api_dependencies", self._check_api_dependencies)
    
    def _check_system_resources(self) -> HealthCheckResult:
        """Check overall system resource health."""
        try:
            metrics = self.system_monitor.collect_metrics()
            
            # Determine status based on resource usage
            if metrics.cpu_percent > 90 or metrics.memory_percent > 90:
                status = HealthStatus.CRITICAL
                message = f"High resource usage: CPU {metrics.cpu_percent:.1f}%, Memory {metrics.memory_percent:.1f}%"
            elif metrics.cpu_percent > 70 or metrics.memory_percent > 70:
                status = HealthStatus.WARNING
                message = f"Moderate resource usage: CPU {metrics.cpu_percent:.1f}%, Memory {metrics.memory_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Resource usage normal: CPU {metrics.cpu_percent:.1f}%, Memory {metrics.memory_percent:.1f}%"
            
            return HealthCheckResult(
                name="system_resources",
                status=status,
                message=message,
                duration_ms=0.0,
                details={
                    "cpu_percent": metrics.cpu_percent,
                    "memory_percent": metrics.memory_percent,
                    "memory_available_gb": metrics.memory_available_gb,
                    "disk_percent": metrics.disk_percent,
                    "load_average": metrics.load_average
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="system_resources",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check system resources: {e}",
                duration_ms=0.0
            )
    
    def _check_circuit_breakers(self) -> HealthCheckResult:
        """Check status of circuit breakers."""
        try:
            from .resilience import _circuit_breakers
            
            if not _circuit_breakers:
                return HealthCheckResult(
                    name="circuit_breakers",
                    status=HealthStatus.HEALTHY,
                    message="No circuit breakers configured",
                    duration_ms=0.0
                )
            
            open_breakers = []
            half_open_breakers = []
            
            for name, breaker in _circuit_breakers.items():
                if breaker.stats.state.value == "open":
                    open_breakers.append(name)
                elif breaker.stats.state.value == "half_open":
                    half_open_breakers.append(name)
            
            if open_breakers:
                status = HealthStatus.CRITICAL
                message = f"Circuit breakers open: {', '.join(open_breakers)}"
            elif half_open_breakers:
                status = HealthStatus.WARNING
                message = f"Circuit breakers half-open: {', '.join(half_open_breakers)}"
            else:
                status = HealthStatus.HEALTHY
                message = f"All {len(_circuit_breakers)} circuit breakers healthy"
            
            details = {}
            for name, breaker in _circuit_breakers.items():
                details[name] = {
                    "state": breaker.stats.state.value,
                    "failure_count": breaker.stats.failure_count,
                    "success_count": breaker.stats.success_count,
                    "total_requests": breaker.stats.total_requests
                }
            
            return HealthCheckResult(
                name="circuit_breakers",
                status=status,
                message=message,
                duration_ms=0.0,
                details=details
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="circuit_breakers",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check circuit breakers: {e}",
                duration_ms=0.0
            )
    
    def _check_memory_usage(self) -> HealthCheckResult:
        """Check memory usage patterns."""
        try:
            import gc
            
            # Force garbage collection
            collected = gc.collect()
            
            # Get memory info
            memory = psutil.virtual_memory()
            process = psutil.Process()
            process_memory = process.memory_info()
            
            # Calculate memory efficiency
            process_memory_mb = process_memory.rss / (1024**2)
            
            if process_memory_mb > 1024:  # > 1GB
                status = HealthStatus.WARNING
                message = f"High process memory usage: {process_memory_mb:.1f} MB"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory usage normal: {process_memory_mb:.1f} MB"
            
            return HealthCheckResult(
                name="memory_usage",
                status=status,
                message=message,
                duration_ms=0.0,
                details={
                    "process_memory_mb": process_memory_mb,
                    "system_memory_percent": memory.percent,
                    "gc_collected": collected,
                    "gc_counts": gc.get_count()
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="memory_usage",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check memory usage: {e}",
                duration_ms=0.0
            )
    
    def _check_disk_space(self) -> HealthCheckResult:
        """Check available disk space."""
        try:
            disk = psutil.disk_usage('/')
            free_gb = disk.free / (1024**3)
            
            if disk.percent > 95:
                status = HealthStatus.CRITICAL
                message = f"Very low disk space: {disk.percent:.1f}% used, {free_gb:.1f} GB free"
            elif disk.percent > 85:
                status = HealthStatus.WARNING
                message = f"Low disk space: {disk.percent:.1f}% used, {free_gb:.1f} GB free"
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk space adequate: {disk.percent:.1f}% used, {free_gb:.1f} GB free"
            
            return HealthCheckResult(
                name="disk_space",
                status=status,
                message=message,
                duration_ms=0.0,
                details={
                    "total_gb": disk.total / (1024**3),
                    "used_gb": disk.used / (1024**3),
                    "free_gb": free_gb,
                    "percent_used": disk.percent
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="disk_space",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check disk space: {e}",
                duration_ms=0.0
            )
    
    def _check_api_dependencies(self) -> HealthCheckResult:
        """Check API and service dependencies."""
        try:
            # Check if required environment variables are set
            import os
            required_vars = ["API_KEY"]
            missing_vars = []
            
            for var in required_vars:
                if not os.environ.get(var):
                    missing_vars.append(var)
            
            # Check cache availability
            cache_available = True
            try:
                from .cache import get_query_cache
                cache = get_query_cache()
                cache.get("__health_check__", top_k=1, semantic=False)
            except Exception:
                cache_available = False
            
            # Determine status
            if missing_vars:
                status = HealthStatus.CRITICAL
                message = f"Missing required environment variables: {', '.join(missing_vars)}"
            elif not cache_available:
                status = HealthStatus.WARNING
                message = "Cache system not available"
            else:
                status = HealthStatus.HEALTHY
                message = "All dependencies available"
            
            return HealthCheckResult(
                name="api_dependencies",
                status=status,
                message=message,
                duration_ms=0.0,
                details={
                    "missing_env_vars": missing_vars,
                    "cache_available": cache_available
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                name="api_dependencies",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check dependencies: {e}",
                duration_ms=0.0
            )


# Global health checker instance
_health_checker: Optional[LegalRAGHealthChecker] = None


def get_health_checker() -> LegalRAGHealthChecker:
    """Get the global health checker instance."""
    global _health_checker
    if _health_checker is None:
        _health_checker = LegalRAGHealthChecker()
    return _health_checker


def run_health_checks() -> Dict[str, Any]:
    """Run all health checks and return comprehensive status."""
    checker = get_health_checker()
    results = checker.run_all_checks()
    overall_status = checker.get_overall_status()
    
    # Get system metrics summary
    metrics_summary = checker.system_monitor.get_metrics_summary(5)
    
    return {
        "overall_status": overall_status.value,
        "checks": {name: {
            "status": result.status.value,
            "message": result.message,
            "duration_ms": result.duration_ms,
            "details": result.details,
            "timestamp": result.timestamp
        } for name, result in results.items()},
        "system_metrics": metrics_summary,
        "timestamp": time.time()
    }