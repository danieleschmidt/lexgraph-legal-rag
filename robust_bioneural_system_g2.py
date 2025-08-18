"""
Robust Bioneural Olfactory Fusion - Generation 2: Make It Robust (Reliable)

This module implements comprehensive reliability features for the bioneural
olfactory fusion system including error handling, validation, monitoring,
logging, security, and health checks.

Generation 2 Enhancements:
- Comprehensive error handling and circuit breakers
- Input validation and sanitization  
- Structured logging with correlation IDs
- Health monitoring and alerting
- Security hardening and audit trails
- Data integrity validation
- Graceful degradation strategies
- Recovery mechanisms
"""

from __future__ import annotations

import asyncio
import logging
import numpy as np
import hashlib
import json
import time
import uuid
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from collections import defaultdict, deque
import functools
from contextlib import asynccontextmanager
import structlog

# Import enhanced Generation 1 components
from enhanced_bioneural_fusion_g1 import (
    EnhancedBioneuroCache,
    EnhancedBioneuroReceptor,
    EnhancedBioneuroFusionEngine
)

from src.lexgraph_legal_rag.bioneuro_olfactory_fusion import (
    OlfactoryReceptorType, 
    OlfactorySignal, 
    DocumentScentProfile
)

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


class BioneuroError(Exception):
    """Base exception for bioneural system errors."""
    
    def __init__(self, message: str, error_code: str = None, context: Dict[str, Any] = None):
        super().__init__(message)
        self.error_code = error_code or "BIONEURO_ERROR"
        self.context = context or {}
        self.timestamp = time.time()
        self.correlation_id = str(uuid.uuid4())


class ValidationError(BioneuroError):
    """Validation-specific error."""
    
    def __init__(self, message: str, field: str = None, value: Any = None):
        super().__init__(message, "VALIDATION_ERROR", {
            "field": field,
            "value": str(value) if value is not None else None
        })


class ProcessingError(BioneuroError):
    """Processing-specific error."""
    
    def __init__(self, message: str, operation: str = None, receptor_type: str = None):
        super().__init__(message, "PROCESSING_ERROR", {
            "operation": operation,
            "receptor_type": receptor_type
        })


class SystemHealthError(BioneuroError):
    """System health-specific error."""
    
    def __init__(self, message: str, component: str = None, severity: str = "medium"):
        super().__init__(message, "HEALTH_ERROR", {
            "component": component,
            "severity": severity
        })


@dataclass
class HealthStatus:
    """System health status."""
    is_healthy: bool
    components: Dict[str, bool]
    metrics: Dict[str, float]
    alerts: List[str]
    last_check: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityContext:
    """Security context for operations."""
    correlation_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    permissions: List[str] = field(default_factory=list)


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60, timeout: int = 30):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
    def is_available(self) -> bool:
        """Check if circuit breaker allows operations."""
        if self.state == "CLOSED":
            return True
        elif self.state == "OPEN":
            if self.last_failure_time and time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                return True
            return False
        elif self.state == "HALF_OPEN":
            return True
        return False
    
    def record_success(self):
        """Record successful operation."""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def record_failure(self):
        """Record failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"


class InputValidator:
    """Comprehensive input validation and sanitization."""
    
    @staticmethod
    def validate_document_text(text: str, max_length: int = 1000000) -> str:
        """Validate and sanitize document text."""
        if not isinstance(text, str):
            raise ValidationError("Document text must be a string", "text", type(text))
        
        if len(text) == 0:
            raise ValidationError("Document text cannot be empty", "text", len(text))
        
        if len(text) > max_length:
            raise ValidationError(f"Document text exceeds maximum length of {max_length}", "text", len(text))
        
        # Remove potential malicious content
        sanitized = re.sub(r'[\\x00-\\x08\\x0B\\x0C\\x0E-\\x1F\\x7F]', '', text)
        
        # Basic SQL injection protection
        suspicious_patterns = [
            r'(?i)(union\s+select)',
            r'(?i)(drop\s+table)',
            r'(?i)(delete\s+from)',
            r'(?i)(insert\s+into)',
            r'(?i)(update\s+.+set)',
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, sanitized):
                raise ValidationError("Potentially malicious content detected", "text", "SQL_INJECTION_SUSPECTED")
        
        return sanitized
    
    @staticmethod
    def validate_document_id(doc_id: str, max_length: int = 255) -> str:
        """Validate document ID."""
        if not isinstance(doc_id, str):
            raise ValidationError("Document ID must be a string", "document_id", type(doc_id))
        
        if len(doc_id) == 0:
            raise ValidationError("Document ID cannot be empty", "document_id", len(doc_id))
        
        if len(doc_id) > max_length:
            raise ValidationError(f"Document ID exceeds maximum length of {max_length}", "document_id", len(doc_id))
        
        # Allow only alphanumeric, hyphens, underscores
        if not re.match(r'^[a-zA-Z0-9_-]+$', doc_id):
            raise ValidationError("Document ID contains invalid characters", "document_id", doc_id)
        
        return doc_id
    
    @staticmethod
    def validate_metadata(metadata: Dict[str, Any], max_keys: int = 50) -> Dict[str, Any]:
        """Validate metadata dictionary."""
        if not isinstance(metadata, dict):
            raise ValidationError("Metadata must be a dictionary", "metadata", type(metadata))
        
        if len(metadata) > max_keys:
            raise ValidationError(f"Metadata has too many keys (max: {max_keys})", "metadata", len(metadata))
        
        # Validate keys and values
        validated = {}
        for key, value in metadata.items():
            if not isinstance(key, str):
                raise ValidationError("Metadata keys must be strings", f"metadata.{key}", type(key))
            
            if len(key) > 100:
                raise ValidationError("Metadata key too long", f"metadata.{key}", len(key))
            
            # Serialize complex values to strings
            if isinstance(value, (dict, list)):
                validated[key] = json.dumps(value)
            elif isinstance(value, (str, int, float, bool, type(None))):
                validated[key] = value
            else:
                validated[key] = str(value)
        
        return validated


class SecurityManager:
    """Security management and audit logging."""
    
    def __init__(self):
        self.audit_log = deque(maxlen=10000)  # Keep last 10k audit entries
        
    def create_security_context(self, **kwargs) -> SecurityContext:
        """Create security context for operations."""
        return SecurityContext(
            correlation_id=str(uuid.uuid4()),
            **kwargs
        )
    
    def audit_operation(self, operation: str, context: SecurityContext, 
                       result: str = "SUCCESS", details: Dict[str, Any] = None):
        """Log audit trail for operations."""
        audit_entry = {
            "timestamp": time.time(),
            "operation": operation,
            "correlation_id": context.correlation_id,
            "user_id": context.user_id,
            "result": result,
            "details": details or {}
        }
        
        self.audit_log.append(audit_entry)
        
        logger.info("Operation audited", 
                   op=operation,
                   correlation_id=context.correlation_id,
                   result=result,
                   user_id=context.user_id)
    
    def get_audit_trail(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent audit trail entries."""
        return list(self.audit_log)[-limit:]


class HealthMonitor:
    """Comprehensive health monitoring."""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.alerts = []
        self.thresholds = {
            "error_rate": 0.05,      # 5% error rate threshold
            "avg_response_time": 1.0, # 1 second response time threshold
            "memory_usage": 0.8,      # 80% memory usage threshold
            "cache_hit_rate": 0.7     # 70% cache hit rate threshold
        }
    
    async def check_health(self) -> HealthStatus:
        """Perform comprehensive health check."""
        components = {}
        metrics = {}
        alerts = []
        
        # Check system components
        try:
            # Memory usage check
            import psutil
            memory = psutil.virtual_memory()
            metrics["memory_usage"] = memory.percent / 100.0
            components["memory"] = memory.percent < 90.0
            
            if memory.percent > 85.0:
                alerts.append(f"High memory usage: {memory.percent:.1f}%")
                
        except Exception as e:
            components["memory"] = False
            alerts.append(f"Memory check failed: {e}")
        
        # Check error rates
        error_count = len([m for m in self.metrics.get("errors", []) if time.time() - m["timestamp"] < 300])
        total_count = len([m for m in self.metrics.get("operations", []) if time.time() - m["timestamp"] < 300])
        
        if total_count > 0:
            error_rate = error_count / total_count
            metrics["error_rate"] = error_rate
            components["error_rate"] = error_rate < self.thresholds["error_rate"]
            
            if error_rate > self.thresholds["error_rate"]:
                alerts.append(f"High error rate: {error_rate:.2%}")
        else:
            components["error_rate"] = True
            metrics["error_rate"] = 0.0
        
        # Check response times
        recent_times = [m["duration"] for m in self.metrics.get("response_times", []) 
                       if time.time() - m["timestamp"] < 300]
        
        if recent_times:
            avg_time = sum(recent_times) / len(recent_times)
            metrics["avg_response_time"] = avg_time
            components["response_time"] = avg_time < self.thresholds["avg_response_time"]
            
            if avg_time > self.thresholds["avg_response_time"]:
                alerts.append(f"High response time: {avg_time:.3f}s")
        else:
            components["response_time"] = True
            metrics["avg_response_time"] = 0.0
        
        # Overall health
        is_healthy = all(components.values()) and len(alerts) == 0
        
        return HealthStatus(
            is_healthy=is_healthy,
            components=components,
            metrics=metrics,
            alerts=alerts,
            last_check=time.time()
        )
    
    def record_metric(self, metric_type: str, value: float, metadata: Dict[str, Any] = None):
        """Record a metric for monitoring."""
        entry = {
            "timestamp": time.time(),
            "value": value,
            "metadata": metadata or {}
        }
        self.metrics[metric_type].append(entry)
        
        # Keep only recent metrics (last hour)
        cutoff = time.time() - 3600
        self.metrics[metric_type] = [m for m in self.metrics[metric_type] if m["timestamp"] > cutoff]


class RobustBioneuroReceptor(EnhancedBioneuroReceptor):
    """Robust receptor with comprehensive error handling and monitoring."""
    
    def __init__(self, receptor_type: OlfactoryReceptorType, sensitivity: float = 0.5):
        super().__init__(receptor_type, sensitivity)
        self.circuit_breaker = CircuitBreaker()
        self.security_manager = SecurityManager()
        self.health_monitor = HealthMonitor()
        self.validator = InputValidator()
        
    async def activate(self, document_text: str, metadata: Dict[str, Any] = None, 
                      security_context: SecurityContext = None) -> OlfactorySignal:
        """Robust activation with comprehensive error handling."""
        correlation_id = security_context.correlation_id if security_context else str(uuid.uuid4())
        start_time = time.time()
        
        # Create security context if not provided
        if not security_context:
            security_context = self.security_manager.create_security_context()
        
        logger.info("Receptor activation started",
                   receptor_type=self.receptor_type.value,
                   correlation_id=correlation_id)
        
        try:
            # Check circuit breaker
            if not self.circuit_breaker.is_available():
                raise ProcessingError("Circuit breaker is open", "activation", self.receptor_type.value)
            
            # Validate inputs
            validated_text = self.validator.validate_document_text(document_text)
            validated_metadata = self.validator.validate_metadata(metadata or {})
            
            # Audit operation start
            self.security_manager.audit_operation("receptor_activation_start", security_context, 
                                                 details={"receptor_type": self.receptor_type.value})
            
            # Perform activation with timeout
            try:
                signal = await asyncio.wait_for(
                    super().activate(validated_text, validated_metadata),
                    timeout=30.0  # 30 second timeout
                )
                
                # Record success
                self.circuit_breaker.record_success()
                processing_time = time.time() - start_time
                
                # Record metrics
                self.health_monitor.record_metric("operations", 1.0, {
                    "receptor_type": self.receptor_type.value,
                    "success": True
                })
                self.health_monitor.record_metric("response_times", processing_time, {
                    "receptor_type": self.receptor_type.value
                })
                
                # Validate output
                self._validate_signal(signal)
                
                # Audit successful completion
                self.security_manager.audit_operation("receptor_activation_complete", security_context,
                                                     details={
                                                         "receptor_type": self.receptor_type.value,
                                                         "intensity": signal.intensity,
                                                         "confidence": signal.confidence,
                                                         "processing_time": processing_time
                                                     })
                
                logger.info("Receptor activation completed successfully",
                           receptor_type=self.receptor_type.value,
                           correlation_id=correlation_id,
                           intensity=signal.intensity,
                           confidence=signal.confidence,
                           processing_time=processing_time)
                
                return signal
                
            except asyncio.TimeoutError:
                raise ProcessingError("Receptor activation timeout", "activation", self.receptor_type.value)
            
        except Exception as e:
            # Record failure
            self.circuit_breaker.record_failure()
            processing_time = time.time() - start_time
            
            # Record error metrics
            self.health_monitor.record_metric("errors", 1.0, {
                "receptor_type": self.receptor_type.value,
                "error_type": type(e).__name__
            })
            self.health_monitor.record_metric("operations", 1.0, {
                "receptor_type": self.receptor_type.value,
                "success": False
            })
            
            # Audit failure
            self.security_manager.audit_operation("receptor_activation_failed", security_context,
                                                 result="FAILURE",
                                                 details={
                                                     "receptor_type": self.receptor_type.value,
                                                     "error": str(e),
                                                     "error_type": type(e).__name__,
                                                     "processing_time": processing_time
                                                 })
            
            logger.error("Receptor activation failed",
                        receptor_type=self.receptor_type.value,
                        correlation_id=correlation_id,
                        error=str(e),
                        error_type=type(e).__name__,
                        processing_time=processing_time)
            
            # Return graceful fallback signal
            return self._create_fallback_signal(e)
    
    def _validate_signal(self, signal: OlfactorySignal):
        """Validate output signal integrity."""
        if not isinstance(signal, OlfactorySignal):
            raise ValidationError("Invalid signal type", "signal", type(signal))
        
        if not isinstance(signal.intensity, (int, float)) or signal.intensity < 0 or signal.intensity > 1:
            raise ValidationError("Invalid intensity value", "intensity", signal.intensity)
        
        if not isinstance(signal.confidence, (int, float)) or signal.confidence < 0 or signal.confidence > 1:
            raise ValidationError("Invalid confidence value", "confidence", signal.confidence)
        
        if signal.receptor_type != self.receptor_type:
            raise ValidationError("Signal receptor type mismatch", "receptor_type", signal.receptor_type)
    
    def _create_fallback_signal(self, error: Exception) -> OlfactorySignal:
        """Create graceful fallback signal for errors."""
        return OlfactorySignal(
            receptor_type=self.receptor_type,
            intensity=0.05,  # Minimal fallback intensity
            confidence=0.1,   # Low confidence
            metadata={
                "fallback": True,
                "error": str(error),
                "error_type": type(error).__name__,
                "generation": "G2_robust_fallback"
            }
        )
    
    async def get_health_status(self) -> HealthStatus:
        """Get receptor health status."""
        return await self.health_monitor.check_health()


class RobustBioneuroFusionEngine(EnhancedBioneuroFusionEngine):
    """Robust fusion engine with comprehensive reliability features."""
    
    def __init__(self, receptor_sensitivities: Optional[Dict[str, float]] = None):
        super().__init__(receptor_sensitivities)
        
        # Replace receptors with robust versions
        self.receptors = {
            receptor_type: RobustBioneuroReceptor(
                receptor_type, 
                self.receptor_sensitivities.get(receptor_type.value, 0.5)
            )
            for receptor_type in OlfactoryReceptorType
        }
        
        self.security_manager = SecurityManager()
        self.health_monitor = HealthMonitor()
        self.validator = InputValidator()
        self.circuit_breaker = CircuitBreaker(failure_threshold=10, recovery_timeout=120)
        
    async def analyze_document_scent(self, document_text: str, document_id: str, 
                                   metadata: Optional[Dict[str, Any]] = None,
                                   security_context: Optional[SecurityContext] = None) -> DocumentScentProfile:
        """Robust document scent analysis with comprehensive error handling."""
        start_time = time.time()
        
        # Create security context if not provided
        if not security_context:
            security_context = self.security_manager.create_security_context()
        
        correlation_id = security_context.correlation_id
        
        logger.info("Document analysis started",
                   document_id=document_id,
                   correlation_id=correlation_id)
        
        try:
            # Check circuit breaker
            if not self.circuit_breaker.is_available():
                raise SystemHealthError("System temporarily unavailable", "fusion_engine", "high")
            
            # Validate inputs
            validated_text = self.validator.validate_document_text(document_text)
            validated_id = self.validator.validate_document_id(document_id)
            validated_metadata = self.validator.validate_metadata(metadata or {})
            
            # Audit operation start
            self.security_manager.audit_operation("document_analysis_start", security_context,
                                                 details={
                                                     "document_id": validated_id,
                                                     "text_length": len(validated_text),
                                                     "metadata_keys": list(validated_metadata.keys())
                                                 })
            
            # Check cache with security context
            cache_key = f"analysis:{hashlib.md5(validated_text.encode('utf-8'), usedforsecurity=False).hexdigest()}"
            cached_profile = self.analysis_cache.get(cache_key)
            
            if cached_profile:
                # Audit cache hit
                self.security_manager.audit_operation("analysis_cache_hit", security_context,
                                                     details={"document_id": validated_id})
                
                processing_time = time.time() - start_time
                logger.info("Document analysis completed (cached)",
                           document_id=validated_id,
                           correlation_id=correlation_id,
                           processing_time=processing_time)
                
                return cached_profile
            
            # Perform parallel receptor activation with robust error handling
            activation_tasks = []
            for receptor in self.receptors.values():
                task = receptor.activate(validated_text, validated_metadata, security_context)
                activation_tasks.append(task)
            
            # Execute with timeout and gather results
            try:
                signals = await asyncio.wait_for(
                    asyncio.gather(*activation_tasks, return_exceptions=True),
                    timeout=60.0  # 60 second total timeout
                )
            except asyncio.TimeoutError:
                raise ProcessingError("Document analysis timeout", "parallel_activation")
            
            # Process signals and handle errors
            valid_signals = []
            error_count = 0
            
            for i, signal in enumerate(signals):
                if isinstance(signal, Exception):
                    error_count += 1
                    logger.warning("Receptor activation failed",
                                 receptor_type=list(self.receptors.keys())[i].value,
                                 error=str(signal),
                                 correlation_id=correlation_id)
                    
                    # Create fallback signal
                    fallback_signal = OlfactorySignal(
                        receptor_type=list(self.receptors.keys())[i],
                        intensity=0.05,
                        confidence=0.1,
                        metadata={"fallback": True, "error": str(signal)}
                    )
                    valid_signals.append(fallback_signal)
                else:
                    valid_signals.append(signal)
            
            # Check if too many errors occurred
            if error_count > len(self.receptors) / 2:
                raise ProcessingError(f"Too many receptor failures: {error_count}/{len(self.receptors)}")
            
            # Generate composite scent with validation
            try:
                composite_scent = self._generate_enhanced_composite_scent(valid_signals)
                self._validate_composite_scent(composite_scent)
            except Exception as e:
                raise ProcessingError("Composite scent generation failed", "scent_generation") from e
            
            # Create similarity hash with integrity check
            signal_data = [{"type": s.receptor_type.value, "intensity": s.intensity, "confidence": s.confidence} 
                          for s in valid_signals]
            similarity_hash = hashlib.md5(
                json.dumps(signal_data, sort_keys=True).encode('utf-8'),
                usedforsecurity=False
            ).hexdigest()[:16]
            
            # Create validated profile
            profile = DocumentScentProfile(
                document_id=validated_id,
                signals=valid_signals,
                composite_scent=composite_scent,
                similarity_hash=similarity_hash
            )
            
            # Validate profile integrity
            self._validate_profile(profile)
            
            # Cache the result with metadata
            profile_metadata = {
                "creation_time": time.time(),
                "correlation_id": correlation_id,
                "error_count": error_count
            }
            enhanced_profile = profile  # Add metadata if needed
            self.analysis_cache.set(cache_key, enhanced_profile)
            
            # Record success metrics
            self.circuit_breaker.record_success()
            processing_time = time.time() - start_time
            
            self.health_monitor.record_metric("operations", 1.0, {
                "operation": "document_analysis",
                "success": True,
                "error_count": error_count
            })
            self.health_monitor.record_metric("response_times", processing_time, {
                "operation": "document_analysis"
            })
            
            # Update performance statistics
            self.performance_stats["documents_analyzed"] += 1
            self.performance_stats["total_analysis_time"] += processing_time
            
            # Audit successful completion
            self.security_manager.audit_operation("document_analysis_complete", security_context,
                                                 details={
                                                     "document_id": validated_id,
                                                     "signals_count": len(valid_signals),
                                                     "error_count": error_count,
                                                     "processing_time": processing_time,
                                                     "similarity_hash": similarity_hash
                                                 })
            
            logger.info("Document analysis completed successfully",
                       document_id=validated_id,
                       correlation_id=correlation_id,
                       signals_count=len(valid_signals),
                       error_count=error_count,
                       processing_time=processing_time)
            
            return profile
            
        except Exception as e:
            # Record failure
            self.circuit_breaker.record_failure()
            processing_time = time.time() - start_time
            
            # Record error metrics
            self.health_monitor.record_metric("errors", 1.0, {
                "operation": "document_analysis",
                "error_type": type(e).__name__
            })
            self.health_monitor.record_metric("operations", 1.0, {
                "operation": "document_analysis",
                "success": False
            })
            
            # Audit failure
            self.security_manager.audit_operation("document_analysis_failed", security_context,
                                                 result="FAILURE",
                                                 details={
                                                     "document_id": document_id,
                                                     "error": str(e),
                                                     "error_type": type(e).__name__,
                                                     "processing_time": processing_time
                                                 })
            
            logger.error("Document analysis failed",
                        document_id=document_id,
                        correlation_id=correlation_id,
                        error=str(e),
                        error_type=type(e).__name__,
                        processing_time=processing_time)
            
            # Return graceful fallback profile
            return self._create_fallback_profile(document_id, e)
    
    def _validate_composite_scent(self, scent: np.ndarray):
        """Validate composite scent vector integrity."""
        if not isinstance(scent, np.ndarray):
            raise ValidationError("Composite scent must be numpy array", "composite_scent", type(scent))
        
        if scent.dtype != np.float32:
            raise ValidationError("Composite scent must be float32", "composite_scent", scent.dtype)
        
        if len(scent) == 0:
            raise ValidationError("Composite scent cannot be empty", "composite_scent", len(scent))
        
        if np.any(np.isnan(scent)) or np.any(np.isinf(scent)):
            raise ValidationError("Composite scent contains invalid values", "composite_scent", "NaN/Inf")
    
    def _validate_profile(self, profile: DocumentScentProfile):
        """Validate document scent profile integrity."""
        if not isinstance(profile, DocumentScentProfile):
            raise ValidationError("Invalid profile type", "profile", type(profile))
        
        if not profile.document_id:
            raise ValidationError("Profile missing document ID", "document_id", None)
        
        if not profile.signals:
            raise ValidationError("Profile missing signals", "signals", len(profile.signals))
        
        if len(profile.signals) != len(OlfactoryReceptorType):
            raise ValidationError("Incomplete signal set", "signals", len(profile.signals))
        
        if not profile.similarity_hash:
            raise ValidationError("Profile missing similarity hash", "similarity_hash", None)
    
    def _create_fallback_profile(self, document_id: str, error: Exception) -> DocumentScentProfile:
        """Create graceful fallback profile for errors."""
        fallback_signals = [
            OlfactorySignal(
                receptor_type=receptor_type,
                intensity=0.05,
                confidence=0.1,
                metadata={
                    "fallback": True,
                    "error": str(error),
                    "error_type": type(error).__name__,
                    "generation": "G2_robust_fallback"
                }
            )
            for receptor_type in OlfactoryReceptorType
        ]
        
        fallback_scent = np.array([0.05] * (len(OlfactoryReceptorType) * 2 + 3), dtype=np.float32)
        
        return DocumentScentProfile(
            document_id=document_id,
            signals=fallback_signals,
            composite_scent=fallback_scent,
            similarity_hash="fallback_error"
        )
    
    async def get_system_health(self) -> HealthStatus:
        """Get comprehensive system health status."""
        # Collect health from all components
        engine_health = await self.health_monitor.check_health()
        
        # Check receptor health
        receptor_health = {}
        for receptor_type, receptor in self.receptors.items():
            try:
                receptor_status = await receptor.get_health_status()
                receptor_health[receptor_type.value] = receptor_status.is_healthy
            except Exception as e:
                receptor_health[receptor_type.value] = False
                engine_health.alerts.append(f"Receptor {receptor_type.value} health check failed: {e}")
        
        # Update component status
        engine_health.components.update(receptor_health)
        engine_health.components["circuit_breaker"] = self.circuit_breaker.state == "CLOSED"
        engine_health.components["cache"] = len(self.analysis_cache.cache) < self.analysis_cache.max_size
        
        # Overall health is all components healthy
        engine_health.is_healthy = all(engine_health.components.values()) and len(engine_health.alerts) == 0
        
        return engine_health
    
    def get_security_audit_trail(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get security audit trail."""
        return self.security_manager.get_audit_trail(limit)


# Enhanced convenience functions with security
async def analyze_document_scent_robust(document_text: str, document_id: str, 
                                      metadata: Optional[Dict[str, Any]] = None,
                                      user_id: Optional[str] = None) -> DocumentScentProfile:
    """Robust document scent analysis with Generation 2 reliability features."""
    engine = RobustBioneuroFusionEngine()
    
    # Create security context
    security_context = engine.security_manager.create_security_context(user_id=user_id)
    
    return await engine.analyze_document_scent(document_text, document_id, metadata, security_context)


if __name__ == "__main__":
    # Example usage and testing
    async def demo_robust_bioneural():
        print("🛡️ ROBUST BIONEURAL OLFACTORY FUSION - GENERATION 2")
        print("=" * 65)
        
        engine = RobustBioneuroFusionEngine()
        
        sample_doc = """
        WHEREAS, the parties hereto agree pursuant to 15 U.S.C. § 1681,
        the Contractor shall indemnify Company from any liability, damages,
        or penalties arising from breach of this agreement. This contract
        shall be governed by the laws of Delaware and any disputes shall
        be resolved through binding arbitration.
        """
        
        print("📄 Testing robust document analysis...")
        
        # Test normal operation
        try:
            start = time.time()
            profile = await analyze_document_scent_robust(sample_doc, "robust_test_001", 
                                                        {"test": True}, "demo_user")
            analysis_time = time.time() - start
            
            print(f"✅ Analysis completed in {analysis_time:.3f}s")
            print(f"🔬 Signals detected: {len(profile.signals)}")
            print(f"📊 Composite scent dimensions: {len(profile.composite_scent)}")
            
            for signal in profile.signals:
                if signal.intensity > 0.1:
                    print(f"   {signal.receptor_type.value}: {signal.intensity:.3f} (conf: {signal.confidence:.3f})")
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
        
        # Test system health
        print("\n🏥 Checking system health...")
        health = await engine.get_system_health()
        print(f"Overall health: {'✅ Healthy' if health.is_healthy else '❌ Unhealthy'}")
        
        for component, status in health.components.items():
            status_icon = "✅" if status else "❌"
            print(f"   {component}: {status_icon}")
        
        if health.alerts:
            print("⚠️ Alerts:")
            for alert in health.alerts:
                print(f"   {alert}")
        
        # Test input validation
        print("\n🔍 Testing input validation...")
        try:
            await analyze_document_scent_robust("", "empty_doc", {}, "demo_user")
        except ValidationError as e:
            print(f"✅ Validation caught empty document: {e.error_code}")
        
        try:
            await analyze_document_scent_robust("Valid text", "", {}, "demo_user")  
        except ValidationError as e:
            print(f"✅ Validation caught empty ID: {e.error_code}")
        
        # Show audit trail
        print("\n📋 Security audit trail (last 5 entries):")
        audit_trail = engine.get_security_audit_trail(5)
        for entry in audit_trail:
            print(f"   {entry['operation']} - {entry['result']} - {entry['correlation_id'][:8]}")
        
        print("\n🛡️ Generation 2 robustness features active:")
        print("   ✓ Comprehensive error handling")
        print("   ✓ Input validation and sanitization") 
        print("   ✓ Circuit breaker pattern")
        print("   ✓ Structured logging with correlation IDs")
        print("   ✓ Health monitoring and alerting")
        print("   ✓ Security audit trails")
        print("   ✓ Graceful degradation")
        print("   ✓ Data integrity validation")
    
    # Run demo
    asyncio.run(demo_robust_bioneural())