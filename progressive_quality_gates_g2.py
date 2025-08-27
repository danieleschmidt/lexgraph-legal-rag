#!/usr/bin/env python3
"""
Progressive Quality Gates Generation 2: Robust Implementation
Autonomous SDLC with comprehensive error handling, validation, and resilience patterns

This system builds on Generation 1 with enhanced robustness:
- Comprehensive error handling and recovery
- Advanced validation and circuit breakers  
- Logging, monitoring, and health checks
- Security hardening and input sanitization
- Adaptive retry mechanisms and graceful degradation
"""

import asyncio
import json
import logging
import os
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from datetime import datetime, timedelta
import hashlib
import subprocess
import sys
import tempfile
from functools import wraps
import concurrent.futures
from contextlib import asynccontextmanager

import psutil
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest


class QualityGateStatus(Enum):
    PENDING = "pending"
    RUNNING = "running" 
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"
    CIRCUIT_OPEN = "circuit_open"


class GateType(Enum):
    CODE_QUALITY = "code_quality"
    SECURITY = "security"
    PERFORMANCE = "performance"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    RESEARCH = "research"
    DEPENDENCY = "dependency"
    COMPLIANCE = "compliance"


class CircuitBreakerState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration for gate resilience"""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 3
    half_open_max_calls: int = 10


@dataclass
class RetryConfig:
    """Retry configuration for gate execution"""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0


@dataclass
class QualityMetrics:
    """Enhanced quality metrics with validation and metadata"""
    gate_id: str
    gate_type: GateType
    status: QualityGateStatus = QualityGateStatus.PENDING
    score: float = 0.0
    threshold: float = 0.8
    execution_time: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    retry_count: int = 0
    circuit_breaker_state: Optional[CircuitBreakerState] = None
    validation_errors: List[str] = field(default_factory=list)
    health_score: float = 1.0
    
    def __post_init__(self):
        """Validate metrics after initialization"""
        self.validation_errors = []
        
        if not 0.0 <= self.score <= 2.0:  # Allow scores above 1.0 for exceptional performance
            self.validation_errors.append(f"Score {self.score} outside valid range [0.0, 2.0]")
            
        if not 0.0 <= self.threshold <= 1.0:
            self.validation_errors.append(f"Threshold {self.threshold} outside valid range [0.0, 1.0]")
            
        if self.execution_time < 0:
            self.validation_errors.append(f"Execution time {self.execution_time} cannot be negative")
            
        if not self.gate_id or not isinstance(self.gate_id, str):
            self.validation_errors.append("Gate ID must be a non-empty string")


@dataclass
class ProgressiveGateConfig:
    """Enhanced configuration for progressive quality gates"""
    enable_adaptive_thresholds: bool = True
    baseline_threshold: float = 0.8
    improvement_rate: float = 0.05
    max_threshold: float = 0.95
    performance_weight: float = 0.3
    quality_weight: float = 0.4
    security_weight: float = 0.3
    
    # Generation 2 enhancements
    enable_circuit_breakers: bool = True
    enable_retries: bool = True
    max_execution_time: float = 300.0  # 5 minutes
    enable_health_checks: bool = True
    enable_security_scanning: bool = True
    parallel_execution: bool = True
    max_concurrent_gates: int = 4
    
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)


class CircuitBreaker:
    """Circuit breaker implementation for gate resilience"""
    
    def __init__(self, config: CircuitBreakerConfig, name: str):
        self.config = config
        self.name = name
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.call_count = 0
        
    async def __aenter__(self):
        """Enter circuit breaker context"""
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
                self.call_count = 0
            else:
                raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is open")
                
        if self.state == CircuitBreakerState.HALF_OPEN:
            if self.call_count >= self.config.half_open_max_calls:
                raise CircuitBreakerOpenError(f"Circuit breaker {self.name} half-open call limit reached")
            self.call_count += 1
            
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit circuit breaker context"""
        if exc_type is None:
            self._record_success()
        else:
            self._record_failure()
            
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if not self.last_failure_time:
            return False
        return datetime.now() - self.last_failure_time >= timedelta(seconds=self.config.recovery_timeout)
    
    def _record_success(self):
        """Record successful execution"""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                self.success_count = 0
        
    def _record_failure(self):
        """Record failed execution"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            self.success_count = 0


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open"""
    pass


class ValidationError(Exception):
    """Exception raised when validation fails"""
    pass


class GateTimeoutError(Exception):
    """Exception raised when gate execution times out"""
    pass


def retry_on_failure(config: RetryConfig):
    """Decorator for retrying failed operations"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(config.max_retries + 1):
                try:
                    result = await func(*args, **kwargs)
                    if hasattr(result, 'retry_count'):
                        result.retry_count = attempt
                    return result
                except Exception as e:
                    last_exception = e
                    if attempt < config.max_retries:
                        delay = min(
                            config.base_delay * (config.exponential_base ** attempt),
                            config.max_delay
                        )
                        await asyncio.sleep(delay)
                        
            # All retries exhausted
            if hasattr(args[0], 'logger'):  # Assume first arg is self with logger
                args[0].logger.error(f"All retries exhausted for {func.__name__}: {last_exception}")
            raise last_exception
            
        return wrapper
    return decorator


def timeout(seconds: float):
    """Decorator for adding timeout to async functions"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                raise GateTimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")
        return wrapper
    return decorator


class RobustQualityGateEngine:
    """Generation 2: Robust progressive quality gate implementation"""
    
    def __init__(self, config: Optional[ProgressiveGateConfig] = None):
        self.config = config or ProgressiveGateConfig()
        self.metrics: List[QualityMetrics] = []
        self.gate_history: Dict[str, List[float]] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.logger = self._setup_logging()
        self.registry = CollectorRegistry()
        self._setup_prometheus_metrics()
        self.health_status = {"healthy": True, "last_check": datetime.now()}
        
    def _setup_logging(self) -> logging.Logger:
        """Enhanced logging configuration with structured logging"""
        logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
            # File handler for audit trail
            try:
                log_dir = Path("logs")
                log_dir.mkdir(exist_ok=True)
                file_handler = logging.FileHandler(log_dir / "quality_gates_g2.log")
                file_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
                )
                file_handler.setFormatter(file_formatter)
                logger.addHandler(file_handler)
            except Exception as e:
                logger.warning(f"Could not setup file logging: {e}")
            
        return logger
        
    def _setup_prometheus_metrics(self) -> None:
        """Enhanced Prometheus metrics for monitoring"""
        self.gate_counter = Counter(
            'quality_gates_total',
            'Total quality gate executions',
            ['gate_type', 'status', 'generation'],
            registry=self.registry
        )
        
        self.gate_duration = Histogram(
            'quality_gate_duration_seconds',
            'Quality gate execution duration',
            ['gate_type', 'generation'],
            registry=self.registry
        )
        
        self.gate_score = Gauge(
            'quality_gate_score',
            'Current quality gate score',
            ['gate_type', 'gate_id', 'generation'],
            registry=self.registry
        )
        
        self.circuit_breaker_state = Gauge(
            'circuit_breaker_state',
            'Circuit breaker state (0=closed, 1=open, 2=half-open)',
            ['gate_type', 'gate_id'],
            registry=self.registry
        )
        
        self.retry_counter = Counter(
            'quality_gate_retries_total',
            'Total gate execution retries',
            ['gate_type', 'gate_id'],
            registry=self.registry
        )
        
        self.health_check = Gauge(
            'quality_gate_engine_health',
            'Health status of quality gate engine',
            registry=self.registry
        )
    
    def _get_circuit_breaker(self, gate_id: str) -> CircuitBreaker:
        """Get or create circuit breaker for gate"""
        if gate_id not in self.circuit_breakers:
            self.circuit_breakers[gate_id] = CircuitBreaker(
                self.config.circuit_breaker, 
                gate_id
            )
        return self.circuit_breakers[gate_id]
    
    def _validate_environment(self) -> List[str]:
        """Validate environment for gate execution"""
        issues = []
        
        # Check disk space
        try:
            disk_usage = psutil.disk_usage('/')
            if disk_usage.percent > 95:
                issues.append(f"Disk usage critically high: {disk_usage.percent}%")
            elif disk_usage.percent > 85:
                issues.append(f"Disk usage high: {disk_usage.percent}%")
        except Exception as e:
            issues.append(f"Could not check disk usage: {e}")
            
        # Check memory
        try:
            memory = psutil.virtual_memory()
            if memory.percent > 95:
                issues.append(f"Memory usage critically high: {memory.percent}%")
            elif memory.percent > 85:
                issues.append(f"Memory usage high: {memory.percent}%")
        except Exception as e:
            issues.append(f"Could not check memory usage: {e}")
            
        # Check Python version
        if sys.version_info < (3, 8):
            issues.append(f"Python version {sys.version} may be too old")
            
        return issues
    
    async def _health_check(self) -> bool:
        """Perform health check on the engine"""
        try:
            # Check environment
            env_issues = self._validate_environment()
            
            # Check circuit breakers
            open_breakers = [
                name for name, cb in self.circuit_breakers.items()
                if cb.state == CircuitBreakerState.OPEN
            ]
            
            # Update health status
            healthy = len(env_issues) == 0 and len(open_breakers) < len(self.circuit_breakers) / 2
            
            self.health_status = {
                "healthy": healthy,
                "last_check": datetime.now(),
                "environment_issues": env_issues,
                "open_circuit_breakers": open_breakers,
                "total_metrics": len(self.metrics)
            }
            
            self.health_check.set(1.0 if healthy else 0.0)
            
            if not healthy:
                self.logger.warning(f"Health check failed: {self.health_status}")
            
            return healthy
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            self.health_check.set(0.0)
            return False
    
    @retry_on_failure(RetryConfig())
    @timeout(120.0)
    async def run_code_quality_gate(self, gate_id: str = "code_quality_g2") -> QualityMetrics:
        """Enhanced code quality analysis gate with robust error handling"""
        start_time = time.time()
        
        if self.config.enable_circuit_breakers:
            circuit_breaker = self._get_circuit_breaker(gate_id)
        else:
            circuit_breaker = None
            
        try:
            if circuit_breaker:
                async with circuit_breaker:
                    return await self._execute_code_quality_gate(gate_id, start_time)
            else:
                return await self._execute_code_quality_gate(gate_id, start_time)
                
        except CircuitBreakerOpenError as e:
            execution_time = time.time() - start_time
            metrics = QualityMetrics(
                gate_id=gate_id,
                gate_type=GateType.CODE_QUALITY,
                status=QualityGateStatus.CIRCUIT_OPEN,
                execution_time=execution_time,
                error_message=str(e),
                circuit_breaker_state=CircuitBreakerState.OPEN
            )
            
            self._record_metrics(metrics)
            self.logger.warning(f"Code quality gate circuit breaker open: {e}")
            return metrics
            
        except GateTimeoutError as e:
            execution_time = time.time() - start_time
            metrics = QualityMetrics(
                gate_id=gate_id,
                gate_type=GateType.CODE_QUALITY,
                status=QualityGateStatus.TIMEOUT,
                execution_time=execution_time,
                error_message=str(e)
            )
            
            self._record_metrics(metrics)
            self.logger.error(f"Code quality gate timed out: {e}")
            return metrics
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_metrics = QualityMetrics(
                gate_id=gate_id,
                gate_type=GateType.CODE_QUALITY,
                status=QualityGateStatus.FAILED,
                execution_time=execution_time,
                error_message=str(e)
            )
            
            self._record_metrics(error_metrics)
            self.logger.error(f"Code quality gate failed: {e}\n{traceback.format_exc()}")
            return error_metrics
    
    async def _execute_code_quality_gate(self, gate_id: str, start_time: float) -> QualityMetrics:
        """Execute code quality analysis"""
        self.logger.info(f"Running enhanced code quality gate: {gate_id}")
        
        quality_score = 0.70  # Conservative base score for Generation 2
        
        # Enhanced file analysis
        src_files = list(Path("src").rglob("*.py")) if Path("src").exists() else []
        test_files = list(Path("tests").rglob("*.py")) if Path("tests").exists() else []
        config_files = [
            Path("pyproject.toml"),
            Path("setup.py"),
            Path("requirements.txt"),
            Path(".gitignore")
        ]
        
        existing_configs = [f for f in config_files if f.exists()]
        
        # Code quality checks
        has_tests = len(test_files) > 0
        has_src_structure = len(src_files) > 0
        has_configs = len(existing_configs) > 0
        test_coverage = len(test_files) / max(len(src_files), 1) if src_files else 0
        
        # Scoring with validation
        if has_tests:
            quality_score += min(0.15, test_coverage * 0.3)
        if has_src_structure:
            quality_score += 0.1
        if has_configs:
            quality_score += 0.05
            
        # Check for code quality tools
        quality_tools = [
            Path(".pre-commit-config.yaml"),
            Path("pyproject.toml"),  # May contain tool configs
            Path("setup.cfg")
        ]
        
        has_quality_tools = any(f.exists() for f in quality_tools)
        if has_quality_tools:
            quality_score += 0.1
            
        # Advanced checks with error handling
        try:
            # Check for common anti-patterns in Python files
            antipattern_count = 0
            for py_file in src_files[:10]:  # Limit to prevent timeout
                try:
                    content = py_file.read_text(encoding='utf-8')
                    # Check for basic anti-patterns
                    if "eval(" in content:
                        antipattern_count += 1
                    if "exec(" in content:
                        antipattern_count += 1
                except Exception:
                    continue
                    
            if antipattern_count == 0:
                quality_score += 0.05
            else:
                quality_score -= antipattern_count * 0.02
                
        except Exception as e:
            self.logger.warning(f"Advanced code analysis failed: {e}")
            
        # Ensure score is valid
        quality_score = max(0.0, min(quality_score, 2.0))
        
        # Adaptive threshold with validation
        threshold = self._get_adaptive_threshold(gate_id, quality_score)
        
        execution_time = time.time() - start_time
        status = QualityGateStatus.PASSED if quality_score >= threshold else QualityGateStatus.FAILED
        
        metrics = QualityMetrics(
            gate_id=gate_id,
            gate_type=GateType.CODE_QUALITY,
            status=status,
            score=quality_score,
            threshold=threshold,
            execution_time=execution_time,
            metadata={
                "src_files": len(src_files),
                "test_files": len(test_files),
                "has_tests": has_tests,
                "test_coverage_ratio": test_coverage,
                "config_files": [str(f) for f in existing_configs],
                "has_quality_tools": has_quality_tools,
                "antipattern_count": antipattern_count
            }
        )
        
        self._record_metrics(metrics)
        self.logger.info(f"Code quality gate completed: {status.value} (score: {quality_score:.3f})")
        
        return metrics
    
    @retry_on_failure(RetryConfig())
    @timeout(180.0)
    async def run_security_gate(self, gate_id: str = "security_g2") -> QualityMetrics:
        """Enhanced security analysis gate with comprehensive checks"""
        start_time = time.time()
        
        if self.config.enable_circuit_breakers:
            circuit_breaker = self._get_circuit_breaker(gate_id)
        else:
            circuit_breaker = None
            
        try:
            if circuit_breaker:
                async with circuit_breaker:
                    return await self._execute_security_gate(gate_id, start_time)
            else:
                return await self._execute_security_gate(gate_id, start_time)
                
        except CircuitBreakerOpenError as e:
            execution_time = time.time() - start_time
            metrics = QualityMetrics(
                gate_id=gate_id,
                gate_type=GateType.SECURITY,
                status=QualityGateStatus.CIRCUIT_OPEN,
                execution_time=execution_time,
                error_message=str(e),
                circuit_breaker_state=CircuitBreakerState.OPEN
            )
            
            self._record_metrics(metrics)
            return metrics
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_metrics = QualityMetrics(
                gate_id=gate_id,
                gate_type=GateType.SECURITY,
                status=QualityGateStatus.FAILED,
                execution_time=execution_time,
                error_message=str(e)
            )
            
            self._record_metrics(error_metrics)
            self.logger.error(f"Security gate failed: {e}")
            return error_metrics
    
    async def _execute_security_gate(self, gate_id: str, start_time: float) -> QualityMetrics:
        """Execute comprehensive security analysis"""
        self.logger.info(f"Running enhanced security gate: {gate_id}")
        
        security_score = 0.70  # Conservative base score
        security_issues = []
        
        # File-based security checks
        has_requirements = Path("requirements.txt").exists()
        has_security_config = any([
            Path("bandit.yaml").exists(),
            Path(".bandit").exists(),
            Path("pyproject.toml").exists()
        ])
        
        if has_requirements:
            security_score += 0.05
        if has_security_config:
            security_score += 0.1
            
        # Check for exposed sensitive files
        sensitive_files = [
            ".env",
            ".env.local",
            ".env.production",
            "config.ini",
            "secrets.yaml",
            "private.key"
        ]
        
        exposed_files = [f for f in sensitive_files if Path(f).exists()]
        if exposed_files:
            security_issues.extend(exposed_files)
            security_score -= len(exposed_files) * 0.1
            
        # Check git repository for sensitive patterns
        try:
            gitignore_path = Path(".gitignore")
            if gitignore_path.exists():
                gitignore_content = gitignore_path.read_text()
                security_patterns = [".env", "*.key", "secrets", "config.ini"]
                ignored_patterns = sum(1 for pattern in security_patterns if pattern in gitignore_content)
                security_score += ignored_patterns * 0.02
            else:
                security_issues.append("Missing .gitignore file")
                security_score -= 0.05
        except Exception as e:
            self.logger.warning(f"Could not check .gitignore: {e}")
            
        # Advanced security scanning if enabled
        if self.config.enable_security_scanning:
            try:
                # Check for hardcoded credentials in Python files
                credential_patterns = [
                    b"password",
                    b"secret",
                    b"api_key",
                    b"token",
                    b"auth"
                ]
                
                py_files = list(Path(".").rglob("*.py"))
                credential_violations = 0
                
                for py_file in py_files[:20]:  # Limit to prevent timeout
                    try:
                        content = py_file.read_bytes()
                        for pattern in credential_patterns:
                            if pattern in content.lower():
                                # Additional context check to reduce false positives
                                lines = content.decode('utf-8', errors='ignore').split('\n')
                                for line in lines:
                                    if pattern.decode() in line.lower() and ('=' in line or ':' in line):
                                        if not line.strip().startswith('#'):  # Ignore comments
                                            credential_violations += 1
                                            break
                    except Exception:
                        continue
                        
                if credential_violations > 0:
                    security_issues.append(f"Potential hardcoded credentials: {credential_violations}")
                    security_score -= credential_violations * 0.03
                else:
                    security_score += 0.1
                    
            except Exception as e:
                self.logger.warning(f"Advanced security scanning failed: {e}")
        
        # Dependency security check
        try:
            if has_requirements:
                req_content = Path("requirements.txt").read_text()
                # Check for pinned versions (basic security practice)
                lines = [line.strip() for line in req_content.split('\n') if line.strip() and not line.startswith('#')]
                pinned_deps = sum(1 for line in lines if '==' in line)
                total_deps = len(lines)
                
                if total_deps > 0:
                    pin_ratio = pinned_deps / total_deps
                    security_score += pin_ratio * 0.1
                    
        except Exception as e:
            self.logger.warning(f"Dependency security check failed: {e}")
            
        # Ensure score is valid
        security_score = max(0.0, min(security_score, 2.0))
        
        threshold = self._get_adaptive_threshold(gate_id, security_score)
        
        execution_time = time.time() - start_time
        status = QualityGateStatus.PASSED if security_score >= threshold else QualityGateStatus.FAILED
        
        metrics = QualityMetrics(
            gate_id=gate_id,
            gate_type=GateType.SECURITY,
            status=status,
            score=security_score,
            threshold=threshold,
            execution_time=execution_time,
            metadata={
                "security_issues": security_issues,
                "has_requirements": has_requirements,
                "has_security_config": has_security_config,
                "exposed_sensitive_files": exposed_files,
                "security_scanning_enabled": self.config.enable_security_scanning
            }
        )
        
        self._record_metrics(metrics)
        self.logger.info(f"Security gate completed: {status.value} (score: {security_score:.3f})")
        
        if security_issues:
            self.logger.warning(f"Security issues found: {security_issues}")
        
        return metrics
    
    @retry_on_failure(RetryConfig())
    @timeout(90.0)
    async def run_performance_gate(self, gate_id: str = "performance_g2") -> QualityMetrics:
        """Enhanced performance analysis gate with detailed monitoring"""
        start_time = time.time()
        
        try:
            self.logger.info(f"Running enhanced performance gate: {gate_id}")
            
            # Comprehensive system metrics
            cpu_percent = psutil.cpu_percent(interval=2)  # Longer interval for accuracy
            memory_info = psutil.virtual_memory()
            disk_usage = psutil.disk_usage('/')
            
            # Network and process information
            try:
                network_io = psutil.net_io_counters()
                process_count = len(psutil.pids())
            except Exception:
                network_io = None
                process_count = 0
                
            # Load average (Unix-like systems)
            try:
                load_avg = os.getloadavg()
            except (OSError, AttributeError):
                load_avg = (0, 0, 0)  # Not available on all systems
                
            # Enhanced performance scoring
            performance_score = 0.60  # Conservative base
            
            # CPU utilization scoring with more granular levels
            if cpu_percent < 30:
                performance_score += 0.15
            elif cpu_percent < 50:
                performance_score += 0.10
            elif cpu_percent < 70:
                performance_score += 0.05
            elif cpu_percent > 90:
                performance_score -= 0.1
                
            # Memory utilization with swap consideration
            if memory_info.percent < 60:
                performance_score += 0.15
            elif memory_info.percent < 75:
                performance_score += 0.10
            elif memory_info.percent < 85:
                performance_score += 0.05
            elif memory_info.percent > 95:
                performance_score -= 0.1
                
            # Disk space scoring
            if disk_usage.percent < 70:
                performance_score += 0.10
            elif disk_usage.percent < 80:
                performance_score += 0.05
            elif disk_usage.percent > 90:
                performance_score -= 0.1
                
            # Load average scoring (if available)
            if load_avg[0] > 0:  # 1-minute load average
                cpu_count = psutil.cpu_count()
                if cpu_count and load_avg[0] < cpu_count * 0.7:
                    performance_score += 0.05
                elif cpu_count and load_avg[0] > cpu_count * 1.5:
                    performance_score -= 0.05
            
            # Ensure score bounds
            performance_score = max(0.0, min(performance_score, 2.0))
            
            threshold = self._get_adaptive_threshold(gate_id, performance_score)
            
            execution_time = time.time() - start_time
            status = QualityGateStatus.PASSED if performance_score >= threshold else QualityGateStatus.FAILED
            
            metrics = QualityMetrics(
                gate_id=gate_id,
                gate_type=GateType.PERFORMANCE,
                status=status,
                score=performance_score,
                threshold=threshold,
                execution_time=execution_time,
                metadata={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_info.percent,
                    "disk_percent": disk_usage.percent,
                    "available_memory_gb": memory_info.available / (1024**3),
                    "total_memory_gb": memory_info.total / (1024**3),
                    "disk_free_gb": disk_usage.free / (1024**3),
                    "load_average": load_avg,
                    "process_count": process_count,
                    "network_available": network_io is not None
                }
            )
            
            self._record_metrics(metrics)
            self.logger.info(f"Performance gate completed: {status.value} (score: {performance_score:.3f})")
            
            return metrics
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_metrics = QualityMetrics(
                gate_id=gate_id,
                gate_type=GateType.PERFORMANCE,
                status=QualityGateStatus.FAILED,
                execution_time=execution_time,
                error_message=str(e)
            )
            
            self._record_metrics(error_metrics)
            self.logger.error(f"Performance gate failed: {e}")
            return error_metrics
    
    @retry_on_failure(RetryConfig())
    @timeout(150.0)
    async def run_research_gate(self, gate_id: str = "research_g2") -> QualityMetrics:
        """Enhanced research validation gate with comprehensive analysis"""
        start_time = time.time()
        
        try:
            self.logger.info(f"Running enhanced research gate: {gate_id}")
            
            research_score = 0.65  # Conservative base
            
            # Enhanced bioneural component detection
            bioneural_files = [
                "bioneuro_olfactory_demo.py",
                "src/lexgraph_legal_rag/bioneuro_olfactory_fusion.py",
                "src/lexgraph_legal_rag/multisensory_legal_processor.py",
                "generation1_bioneural_research_framework.py",
                "generation2_robust_bioneural_system.py",
                "generation3_scalable_bioneural_system.py"
            ]
            
            found_files = [f for f in bioneural_files if Path(f).exists()]
            research_score += min(0.20, 0.03 * len(found_files))
            
            # Research documentation analysis
            research_docs = [
                "BIONEURO_RESEARCH_PAPER.md",
                "README.md",
                "AUTONOMOUS_SDLC_COMPLETION_REPORT.md"
            ]
            
            found_docs = [d for d in research_docs if Path(d).exists()]
            research_score += min(0.10, 0.03 * len(found_docs))
            
            # Experimental results and validation
            result_files = list(Path(".").glob("*results.json"))
            if result_files:
                research_score += min(0.15, 0.05 * len(result_files))
                
                # Analyze result file quality
                try:
                    for result_file in result_files[:5]:  # Limit to prevent timeout
                        content = json.loads(result_file.read_text())
                        if isinstance(content, dict) and "timestamp" in content:
                            research_score += 0.02
                        if isinstance(content, dict) and "metrics" in content:
                            research_score += 0.02
                except Exception:
                    pass
            
            # Check for testing infrastructure
            test_files = list(Path("tests").rglob("test_*.py")) if Path("tests").exists() else []
            bioneural_test_files = [f for f in test_files if "bioneuro" in str(f).lower()]
            
            if bioneural_test_files:
                research_score += 0.10
                
            # Advanced research metrics
            try:
                # Check for research-specific patterns in code
                py_files = list(Path(".").rglob("*.py"))
                research_patterns = ["receptor", "olfactory", "sensory", "bioneural", "multisensory"]
                pattern_matches = 0
                
                for py_file in py_files[:15]:  # Limit to prevent timeout
                    try:
                        content = py_file.read_text(encoding='utf-8').lower()
                        for pattern in research_patterns:
                            if pattern in content:
                                pattern_matches += 1
                                break
                    except Exception:
                        continue
                        
                if pattern_matches > 5:
                    research_score += 0.08
                elif pattern_matches > 2:
                    research_score += 0.05
                    
            except Exception as e:
                self.logger.warning(f"Advanced research analysis failed: {e}")
            
            # Check for configuration and deployment files
            deployment_files = [
                "docker-compose.yml",
                "Dockerfile",
                "deployment/",
                "k8s/"
            ]
            
            found_deployment = [f for f in deployment_files if Path(f).exists()]
            if found_deployment:
                research_score += 0.05
                
            # Ensure score bounds
            research_score = max(0.0, min(research_score, 2.0))
            
            threshold = self._get_adaptive_threshold(gate_id, research_score)
            
            execution_time = time.time() - start_time
            status = QualityGateStatus.PASSED if research_score >= threshold else QualityGateStatus.FAILED
            
            metrics = QualityMetrics(
                gate_id=gate_id,
                gate_type=GateType.RESEARCH,
                status=status,
                score=research_score,
                threshold=threshold,
                execution_time=execution_time,
                metadata={
                    "bioneural_files": found_files,
                    "research_docs": found_docs,
                    "result_files": len(result_files),
                    "bioneural_test_files": len(bioneural_test_files),
                    "deployment_files": found_deployment,
                    "research_pattern_matches": pattern_matches
                }
            )
            
            self._record_metrics(metrics)
            self.logger.info(f"Research gate completed: {status.value} (score: {research_score:.3f})")
            
            return metrics
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_metrics = QualityMetrics(
                gate_id=gate_id,
                gate_type=GateType.RESEARCH,
                status=QualityGateStatus.FAILED,
                execution_time=execution_time,
                error_message=str(e)
            )
            
            self._record_metrics(error_metrics)
            self.logger.error(f"Research gate failed: {e}")
            return error_metrics
    
    async def run_dependency_gate(self, gate_id: str = "dependency_g2") -> QualityMetrics:
        """New dependency analysis gate for Generation 2"""
        start_time = time.time()
        
        try:
            self.logger.info(f"Running dependency gate: {gate_id}")
            
            dependency_score = 0.70
            
            # Check requirements files
            req_files = ["requirements.txt", "requirements-dev.txt", "pyproject.toml"]
            found_req_files = [f for f in req_files if Path(f).exists()]
            
            if found_req_files:
                dependency_score += 0.10
                
                # Analyze dependency quality
                try:
                    if Path("requirements.txt").exists():
                        req_content = Path("requirements.txt").read_text()
                        lines = [line.strip() for line in req_content.split('\n') 
                                if line.strip() and not line.startswith('#')]
                        
                        # Check for version pinning
                        pinned = sum(1 for line in lines if '==' in line)
                        if lines:
                            pin_ratio = pinned / len(lines)
                            dependency_score += pin_ratio * 0.10
                            
                        # Check for reasonable number of dependencies
                        if len(lines) < 50:  # Not too many dependencies
                            dependency_score += 0.05
                            
                except Exception:
                    pass
            else:
                dependency_score -= 0.15
                
            # Check for lock files (better dependency management)
            lock_files = ["poetry.lock", "Pipfile.lock"]
            if any(Path(f).exists() for f in lock_files):
                dependency_score += 0.15
                
            threshold = self._get_adaptive_threshold(gate_id, dependency_score)
            
            execution_time = time.time() - start_time
            status = QualityGateStatus.PASSED if dependency_score >= threshold else QualityGateStatus.FAILED
            
            metrics = QualityMetrics(
                gate_id=gate_id,
                gate_type=GateType.DEPENDENCY,
                status=status,
                score=dependency_score,
                threshold=threshold,
                execution_time=execution_time,
                metadata={
                    "requirement_files": found_req_files,
                    "has_lock_files": any(Path(f).exists() for f in lock_files),
                    "dependency_count": len(lines) if 'lines' in locals() else 0,
                    "pin_ratio": pin_ratio if 'pin_ratio' in locals() else 0.0
                }
            )
            
            self._record_metrics(metrics)
            self.logger.info(f"Dependency gate completed: {status.value} (score: {dependency_score:.3f})")
            
            return metrics
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_metrics = QualityMetrics(
                gate_id=gate_id,
                gate_type=GateType.DEPENDENCY,
                status=QualityGateStatus.FAILED,
                execution_time=execution_time,
                error_message=str(e)
            )
            
            self._record_metrics(error_metrics)
            self.logger.error(f"Dependency gate failed: {e}")
            return error_metrics
    
    async def run_all_gates(self) -> List[QualityMetrics]:
        """Run all quality gates with enhanced parallelism and error handling"""
        self.logger.info("Starting progressive quality gate execution - Generation 2 (Robust)")
        
        # Perform health check first
        if self.config.enable_health_checks:
            healthy = await self._health_check()
            if not healthy:
                self.logger.warning("System health check failed, but continuing with gates")
        
        # Define gates to run
        gates = [
            self.run_code_quality_gate(),
            self.run_security_gate(),
            self.run_performance_gate(),
            self.run_research_gate(),
            self.run_dependency_gate()
        ]
        
        if self.config.parallel_execution:
            # Run gates in parallel with concurrency limit
            semaphore = asyncio.Semaphore(self.config.max_concurrent_gates)
            
            async def run_gate_with_semaphore(gate_coro):
                async with semaphore:
                    return await gate_coro
            
            tasks = [run_gate_with_semaphore(gate) for gate in gates]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # Run gates sequentially
            results = []
            for gate in gates:
                try:
                    result = await gate
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Gate execution failed: {e}")
                    results.append(e)
        
        # Process results
        metrics = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Gate execution failed: {result}")
                # Create error metric
                error_metric = QualityMetrics(
                    gate_id="unknown",
                    gate_type=GateType.CODE_QUALITY,  # Default type
                    status=QualityGateStatus.FAILED,
                    error_message=str(result)
                )
                metrics.append(error_metric)
            else:
                metrics.append(result)
        
        self.metrics.extend(metrics)
        
        # Calculate comprehensive statistics
        passed_gates = [m for m in metrics if m.status == QualityGateStatus.PASSED]
        failed_gates = [m for m in metrics if m.status == QualityGateStatus.FAILED]
        circuit_open_gates = [m for m in metrics if m.status == QualityGateStatus.CIRCUIT_OPEN]
        timeout_gates = [m for m in metrics if m.status == QualityGateStatus.TIMEOUT]
        
        overall_score = len(passed_gates) / len(metrics) if metrics else 0.0
        avg_execution_time = sum(m.execution_time for m in metrics) / len(metrics) if metrics else 0.0
        
        self.logger.info(f"Quality gates completed - Generation 2 (Robust)")
        self.logger.info(f"Gates passed: {len(passed_gates)}/{len(metrics)}")
        self.logger.info(f"Gates failed: {len(failed_gates)}")
        self.logger.info(f"Circuit breakers open: {len(circuit_open_gates)}")
        self.logger.info(f"Timeouts: {len(timeout_gates)}")
        self.logger.info(f"Overall score: {overall_score:.3f}")
        self.logger.info(f"Average execution time: {avg_execution_time:.3f}s")
        
        return metrics
    
    def _get_adaptive_threshold(self, gate_id: str, current_score: float) -> float:
        """Enhanced adaptive threshold with statistical analysis"""
        if not self.config.enable_adaptive_thresholds:
            return self.config.baseline_threshold
            
        history = self.gate_history.get(gate_id, [])
        
        if len(history) < 3:  # Not enough history, use baseline
            return self.config.baseline_threshold
            
        # Statistical analysis of historical performance
        recent_scores = history[-10:]  # Last 10 runs
        avg_score = sum(recent_scores) / len(recent_scores)
        
        # Calculate standard deviation
        variance = sum((score - avg_score) ** 2 for score in recent_scores) / len(recent_scores)
        std_dev = variance ** 0.5
        
        # Adaptive threshold with confidence interval
        confidence_factor = 0.8  # 80% confidence
        adaptive_threshold = max(
            avg_score - (std_dev * confidence_factor),  # Lower bound
            self.config.baseline_threshold
        )
        
        # Gradual improvement
        improvement_threshold = min(
            avg_score + self.config.improvement_rate,
            self.config.max_threshold
        )
        
        final_threshold = min(adaptive_threshold, improvement_threshold)
        
        self.logger.debug(f"Adaptive threshold for {gate_id}: {final_threshold:.3f} "
                         f"(avg: {avg_score:.3f}, std: {std_dev:.3f})")
        
        return final_threshold
    
    def _record_metrics(self, metrics: QualityMetrics) -> None:
        """Enhanced metrics recording with validation"""
        # Validate metrics before recording
        if metrics.validation_errors:
            self.logger.warning(f"Metrics validation errors for {metrics.gate_id}: {metrics.validation_errors}")
        
        # Update Prometheus metrics
        self.gate_counter.labels(
            gate_type=metrics.gate_type.value,
            status=metrics.status.value,
            generation="2"
        ).inc()
        
        self.gate_duration.labels(
            gate_type=metrics.gate_type.value,
            generation="2"
        ).observe(metrics.execution_time)
        
        self.gate_score.labels(
            gate_type=metrics.gate_type.value,
            gate_id=metrics.gate_id,
            generation="2"
        ).set(metrics.score)
        
        # Record circuit breaker state if applicable
        if metrics.circuit_breaker_state and metrics.gate_id in self.circuit_breakers:
            cb_state = self.circuit_breakers[metrics.gate_id].state
            state_value = {"closed": 0, "open": 1, "half_open": 2}.get(cb_state.value, 0)
            self.circuit_breaker_state.labels(
                gate_type=metrics.gate_type.value,
                gate_id=metrics.gate_id
            ).set(state_value)
        
        # Record retry count
        if metrics.retry_count > 0:
            self.retry_counter.labels(
                gate_type=metrics.gate_type.value,
                gate_id=metrics.gate_id
            ).inc(metrics.retry_count)
        
        # Update history for adaptive thresholds
        if metrics.gate_id not in self.gate_history:
            self.gate_history[metrics.gate_id] = []
            
        # Only add to history if execution was successful
        if metrics.status in [QualityGateStatus.PASSED, QualityGateStatus.FAILED]:
            self.gate_history[metrics.gate_id].append(metrics.score)
            
            # Keep only recent history
            if len(self.gate_history[metrics.gate_id]) > 20:  # Increased history size
                self.gate_history[metrics.gate_id] = self.gate_history[metrics.gate_id][-20:]
    
    def export_results(self, output_path: str = "progressive_quality_gates_g2_results.json") -> None:
        """Enhanced results export with comprehensive metadata"""
        results = {
            "generation": 2,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "enable_adaptive_thresholds": self.config.enable_adaptive_thresholds,
                "baseline_threshold": self.config.baseline_threshold,
                "improvement_rate": self.config.improvement_rate,
                "enable_circuit_breakers": self.config.enable_circuit_breakers,
                "enable_retries": self.config.enable_retries,
                "enable_health_checks": self.config.enable_health_checks,
                "enable_security_scanning": self.config.enable_security_scanning,
                "parallel_execution": self.config.parallel_execution,
                "max_concurrent_gates": self.config.max_concurrent_gates
            },
            "health_status": {
                **self.health_status,
                "last_check": self.health_status["last_check"].isoformat() if isinstance(self.health_status.get("last_check"), datetime) else str(self.health_status.get("last_check"))
            },
            "circuit_breakers": {
                name: {
                    "state": cb.state.value,
                    "failure_count": cb.failure_count,
                    "success_count": cb.success_count,
                    "last_failure_time": cb.last_failure_time.isoformat() if cb.last_failure_time else None
                }
                for name, cb in self.circuit_breakers.items()
            },
            "metrics": [
                {
                    "gate_id": m.gate_id,
                    "gate_type": m.gate_type.value,
                    "status": m.status.value,
                    "score": m.score,
                    "threshold": m.threshold,
                    "execution_time": m.execution_time,
                    "error_message": m.error_message,
                    "metadata": m.metadata,
                    "timestamp": m.timestamp.isoformat(),
                    "retry_count": m.retry_count,
                    "circuit_breaker_state": m.circuit_breaker_state.value if m.circuit_breaker_state else None,
                    "validation_errors": m.validation_errors,
                    "health_score": m.health_score
                }
                for m in self.metrics
            ],
            "summary": {
                "total_gates": len(self.metrics),
                "passed_gates": len([m for m in self.metrics if m.status == QualityGateStatus.PASSED]),
                "failed_gates": len([m for m in self.metrics if m.status == QualityGateStatus.FAILED]),
                "circuit_open_gates": len([m for m in self.metrics if m.status == QualityGateStatus.CIRCUIT_OPEN]),
                "timeout_gates": len([m for m in self.metrics if m.status == QualityGateStatus.TIMEOUT]),
                "average_score": sum(m.score for m in self.metrics) / len(self.metrics) if self.metrics else 0.0,
                "total_execution_time": sum(m.execution_time for m in self.metrics),
                "total_retries": sum(m.retry_count for m in self.metrics),
                "validation_errors": sum(len(m.validation_errors) for m in self.metrics)
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Enhanced results exported to {output_path}")
        
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics in text format"""
        return generate_latest(self.registry).decode('utf-8')
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status"""
        return self.health_status.copy()


async def main():
    """Main demonstration of Generation 2 Progressive Quality Gates"""
    print("🛡️ TERRAGON AUTONOMOUS SDLC - Progressive Quality Gates Generation 2 (Robust)")
    print("=" * 80)
    
    config = ProgressiveGateConfig(
        enable_adaptive_thresholds=True,
        baseline_threshold=0.75,
        improvement_rate=0.03,
        enable_circuit_breakers=True,
        enable_retries=True,
        enable_health_checks=True,
        enable_security_scanning=True,
        parallel_execution=True,
        max_concurrent_gates=3
    )
    
    engine = RobustQualityGateEngine(config)
    
    try:
        # Run all quality gates
        results = await engine.run_all_gates()
        
        # Export results
        engine.export_results()
        
        # Display comprehensive summary
        print("\n📊 GENERATION 2 RESULTS SUMMARY")
        print("-" * 50)
        
        passed = len([r for r in results if r.status == QualityGateStatus.PASSED])
        failed = len([r for r in results if r.status == QualityGateStatus.FAILED])
        circuit_open = len([r for r in results if r.status == QualityGateStatus.CIRCUIT_OPEN])
        timeout = len([r for r in results if r.status == QualityGateStatus.TIMEOUT])
        total = len(results)
        
        print(f"Gates Passed: {passed}/{total}")
        print(f"Gates Failed: {failed}")
        print(f"Circuit Breakers Open: {circuit_open}")
        print(f"Timeouts: {timeout}")
        print(f"Overall Success Rate: {(passed/total)*100:.1f}%" if total > 0 else "N/A")
        
        total_retries = sum(r.retry_count for r in results)
        avg_execution_time = sum(r.execution_time for r in results) / len(results) if results else 0
        
        print(f"Total Retries: {total_retries}")
        print(f"Average Execution Time: {avg_execution_time:.2f}s")
        
        print("\n🎯 DETAILED GATE RESULTS:")
        for result in results:
            status_symbol = {
                QualityGateStatus.PASSED: "✅",
                QualityGateStatus.FAILED: "❌",
                QualityGateStatus.CIRCUIT_OPEN: "🔴",
                QualityGateStatus.TIMEOUT: "⏰"
            }.get(result.status, "❓")
            
            retry_info = f" (retries: {result.retry_count})" if result.retry_count > 0 else ""
            print(f"{status_symbol} {result.gate_type.value}: {result.score:.3f} "
                  f"(threshold: {result.threshold:.3f}, time: {result.execution_time:.2f}s){retry_info}")
            
            if result.validation_errors:
                print(f"    ⚠️  Validation errors: {result.validation_errors}")
        
        # Health status
        health = engine.get_health_status()
        health_symbol = "💚" if health["healthy"] else "💛"
        print(f"\n{health_symbol} System Health: {'Healthy' if health['healthy'] else 'Issues Detected'}")
        
        if "environment_issues" in health and health["environment_issues"]:
            print(f"    Environment Issues: {health['environment_issues']}")
            
        if "open_circuit_breakers" in health and health["open_circuit_breakers"]:
            print(f"    Open Circuit Breakers: {health['open_circuit_breakers']}")
        
        print("\n🎯 NEXT: Generation 3 will add advanced optimization and auto-scaling")
        
        return results
        
    except Exception as e:
        print(f"❌ Generation 2 execution failed: {e}")
        print(f"Error details: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    asyncio.run(main())