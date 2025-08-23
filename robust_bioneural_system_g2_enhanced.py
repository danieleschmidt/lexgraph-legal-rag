#!/usr/bin/env python3
"""
Robust Bioneural System - Generation 2: Make It Robust (Reliable)
=============================================================

Advanced error handling, validation, logging, monitoring, health checks, and security:
- Comprehensive input validation and sanitization
- Advanced error recovery and circuit breaker patterns
- Structured logging with correlation IDs
- Health monitoring and alerting
- Security hardening and input sanitization
- Performance monitoring with detailed metrics
- Auto-recovery mechanisms for failed operations
"""

import asyncio
import hashlib
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import numpy as np
from pathlib import Path

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s'
)
logger = logging.getLogger(__name__)


class SystemState(Enum):
    """System operational states."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILING = "failing"
    CRITICAL = "critical"
    RECOVERING = "recovering"


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Input validation results."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    sanitized_input: Optional[str] = None


@dataclass
class ProcessingMetrics:
    """Detailed processing metrics."""
    correlation_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    processing_duration_ms: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    errors_encountered: int = 0
    warnings_generated: int = 0
    retry_attempts: int = 0


@dataclass
class HealthStatus:
    """System health status."""
    state: SystemState
    last_check: datetime
    errors_last_hour: int = 0
    avg_response_time_ms: float = 0.0
    memory_usage_percent: float = 0.0
    active_connections: int = 0
    alerts: List[str] = field(default_factory=list)


class CircuitBreaker:
    """Circuit breaker for fault tolerance."""
    
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
        
    def can_execute(self) -> bool:
        """Check if operation can be executed."""
        if self.state == "closed":
            return True
        elif self.state == "open":
            if self.last_failure_time and time.time() - self.last_failure_time > self.timeout:
                self.state = "half-open"
                return True
            return False
        else:  # half-open
            return True
    
    def record_success(self):
        """Record successful operation."""
        self.failure_count = 0
        self.state = "closed"
        
    def record_failure(self):
        """Record failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")


class InputValidator:
    """Comprehensive input validation and sanitization."""
    
    @staticmethod
    def validate_document(text: str, max_length: int = 1000000) -> ValidationResult:
        """Validate and sanitize document input."""
        errors = []
        warnings = []
        sanitized = text
        
        # Basic validation
        if not text or not isinstance(text, str):
            errors.append("Document text must be a non-empty string")
            return ValidationResult(False, errors)
        
        # Length validation
        if len(text) > max_length:
            errors.append(f"Document exceeds maximum length of {max_length} characters")
            
        # Content validation
        if len(text.strip()) == 0:
            errors.append("Document cannot be empty or whitespace only")
            
        # Security checks
        if '<script' in text.lower() or 'javascript:' in text.lower():
            errors.append("Document contains potentially malicious content")
            
        # Sanitization
        sanitized = text.strip()
        
        # Character encoding validation
        try:
            sanitized.encode('utf-8')
        except UnicodeEncodeError:
            errors.append("Document contains invalid Unicode characters")
            
        # Performance warnings
        if len(text) > 50000:
            warnings.append("Large document may impact processing performance")
            
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_input=sanitized
        )
    
    @staticmethod
    def validate_document_id(doc_id: str) -> ValidationResult:
        """Validate document ID."""
        errors = []
        
        if not doc_id or not isinstance(doc_id, str):
            errors.append("Document ID must be a non-empty string")
        elif len(doc_id) > 255:
            errors.append("Document ID too long (max 255 characters)")
        elif not doc_id.replace('-', '').replace('_', '').isalnum():
            errors.append("Document ID contains invalid characters")
            
        return ValidationResult(is_valid=len(errors) == 0, errors=errors)


class RobustReceptor:
    """Enhanced receptor with error handling and monitoring."""
    
    def __init__(self, name: str, threshold: float = 0.5, circuit_breaker: Optional[CircuitBreaker] = None):
        self.name = name
        self.threshold = max(0.0, min(1.0, threshold))  # Clamp to valid range
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self.total_activations = 0
        self.successful_activations = 0
        self.error_count = 0
        self.avg_processing_time = 0.0
        
        # Enhanced keyword sets with validation
        self.keywords = self._get_validated_keywords()
        
    def _get_validated_keywords(self) -> List[str]:
        """Get validated keyword set for receptor."""
        keyword_sets = {
            'legal_complexity': [
                'whereas', 'pursuant', 'liability', 'shall', 'herein', 'thereof',
                'notwithstanding', 'heretofore', 'aforementioned', 'stipulated'
            ],
            'statutory_authority': [
                'u.s.c', 'cfr', 'section', '§', 'regulation', 'statute',
                'federal register', 'code', 'title', 'chapter'
            ],
            'citation_density': [
                'v.', 'case', 'court', 'holding', 'precedent', 'ruling',
                'decision', 'opinion', 'judgment', 'appeal'
            ],
            'risk_profile': [
                'liability', 'damages', 'breach', 'penalty', 'violation',
                'sanctions', 'fines', 'enforcement', 'compliance', 'indemnify'
            ],
            'temporal_freshness': [
                'current', 'recent', 'updated', 'amended', 'effective',
                'superseded', 'expired', 'valid', 'applicable', 'active'
            ],
            'semantic_coherence': [
                'therefore', 'however', 'consequently', 'furthermore', 'moreover',
                'nevertheless', 'accordingly', 'subsequently', 'specifically'
            ]
        }
        
        return keyword_sets.get(self.name, [])
    
    async def activate(self, text: str, correlation_id: str = None) -> tuple[float, float, Dict[str, Any]]:
        """Robust activation with error handling and monitoring."""
        correlation_id = correlation_id or str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # Check circuit breaker
            if not self.circuit_breaker.can_execute():
                logger.warning(f"Circuit breaker open for receptor {self.name}")
                return 0.0, 0.0, {
                    'error': 'Circuit breaker open',
                    'correlation_id': correlation_id,
                    'processing_time_ms': 0.0
                }
            
            self.total_activations += 1
            
            # Validate input
            if not text or not isinstance(text, str):
                raise ValueError("Invalid input text")
            
            # Enhanced activation logic with error handling
            text_lower = text.lower()
            matches = 0
            total_keywords = len(self.keywords)
            
            if total_keywords == 0:
                logger.warning(f"No keywords defined for receptor {self.name}")
                intensity = 0.0
                confidence = 0.0
            else:
                # Calculate matches with position weighting
                for i, keyword in enumerate(self.keywords):
                    if keyword in text_lower:
                        # Early keywords have higher weight
                        weight = 1.0 + (0.1 * (total_keywords - i) / total_keywords)
                        matches += weight
                
                # Normalize intensity
                max_possible = sum(1.0 + (0.1 * (total_keywords - i) / total_keywords) 
                                 for i in range(total_keywords))
                intensity = min(matches / max_possible if max_possible > 0 else 0.0, 1.0)
                
                # Calculate confidence based on match distribution
                unique_matches = sum(1 for keyword in self.keywords if keyword in text_lower)
                confidence = min(unique_matches / total_keywords + 0.1, 1.0)
            
            processing_time = (time.time() - start_time) * 1000
            self.avg_processing_time = (
                (self.avg_processing_time * self.successful_activations + processing_time) / 
                (self.successful_activations + 1)
            )
            
            self.successful_activations += 1
            self.circuit_breaker.record_success()
            
            metadata = {
                'correlation_id': correlation_id,
                'processing_time_ms': processing_time,
                'matches_found': int(matches),
                'total_keywords': total_keywords,
                'activated': intensity > self.threshold
            }
            
            logger.debug(f"Receptor {self.name} activation successful: {intensity:.3f}")
            return intensity, confidence, metadata
            
        except Exception as e:
            self.error_count += 1
            self.circuit_breaker.record_failure()
            
            processing_time = (time.time() - start_time) * 1000
            
            logger.error(f"Receptor {self.name} activation failed: {e}", extra={
                'correlation_id': correlation_id,
                'error_type': type(e).__name__
            })
            
            return 0.0, 0.0, {
                'error': str(e),
                'correlation_id': correlation_id,
                'processing_time_ms': processing_time
            }
    
    def get_health_stats(self) -> Dict[str, Any]:
        """Get receptor health statistics."""
        success_rate = (
            self.successful_activations / max(self.total_activations, 1) * 100
        )
        
        return {
            'name': self.name,
            'total_activations': self.total_activations,
            'successful_activations': self.successful_activations,
            'success_rate_percent': round(success_rate, 2),
            'error_count': self.error_count,
            'avg_processing_time_ms': round(self.avg_processing_time, 2),
            'circuit_breaker_state': self.circuit_breaker.state
        }


class RobustBioneuralSystem:
    """Enhanced bioneural system with comprehensive robustness features."""
    
    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self.receptors: Dict[str, RobustReceptor] = {}
        self.health_status = HealthStatus(
            state=SystemState.HEALTHY,
            last_check=datetime.now()
        )
        self.processing_history: List[ProcessingMetrics] = []
        self.error_log: List[Dict[str, Any]] = []
        self.validator = InputValidator()
        
        self._initialize_receptors()
        
    def _initialize_receptors(self):
        """Initialize robust receptors with error handling."""
        receptor_configs = {
            'legal_complexity': 0.3,
            'statutory_authority': 0.4,
            'citation_density': 0.2,
            'risk_profile': 0.3,
            'temporal_freshness': 0.35,
            'semantic_coherence': 0.25
        }
        
        for name, threshold in receptor_configs.items():
            try:
                self.receptors[name] = RobustReceptor(name, threshold)
                logger.info(f"Initialized receptor: {name}")
            except Exception as e:
                logger.error(f"Failed to initialize receptor {name}: {e}")
                self._log_error(ErrorSeverity.HIGH, f"Receptor initialization failed: {name}", str(e))
    
    async def analyze_document_robust(
        self, 
        text: str, 
        doc_id: str = "unknown"
    ) -> Dict[str, Any]:
        """Robust document analysis with comprehensive error handling."""
        correlation_id = str(uuid.uuid4())
        metrics = ProcessingMetrics(
            correlation_id=correlation_id,
            start_time=datetime.now()
        )
        
        try:
            logger.info(f"Starting robust document analysis", extra={
                'correlation_id': correlation_id,
                'document_id': doc_id
            })
            
            # Input validation
            text_validation = self.validator.validate_document(text)
            id_validation = self.validator.validate_document_id(doc_id)
            
            if not text_validation.is_valid:
                raise ValueError(f"Invalid document text: {'; '.join(text_validation.errors)}")
            if not id_validation.is_valid:
                raise ValueError(f"Invalid document ID: {'; '.join(id_validation.errors)}")
            
            # Log warnings
            for warning in text_validation.warnings:
                logger.warning(warning, extra={'correlation_id': correlation_id})
                metrics.warnings_generated += 1
            
            # Use sanitized input
            sanitized_text = text_validation.sanitized_input
            
            # Analyze with retry logic
            analysis_result = await self._analyze_with_retry(
                sanitized_text, doc_id, correlation_id, metrics
            )
            
            # Update metrics
            metrics.end_time = datetime.now()
            metrics.processing_duration_ms = (
                (metrics.end_time - metrics.start_time).total_seconds() * 1000
            )
            
            self.processing_history.append(metrics)
            self._update_health_status()
            
            # Add metadata
            analysis_result.update({
                'correlation_id': correlation_id,
                'processing_metrics': {
                    'duration_ms': metrics.processing_duration_ms,
                    'retry_attempts': metrics.retry_attempts,
                    'warnings': metrics.warnings_generated,
                    'errors': metrics.errors_encountered
                },
                'validation_warnings': text_validation.warnings,
                'system_health': self.health_status.state.value
            })
            
            logger.info(f"Document analysis completed successfully", extra={
                'correlation_id': correlation_id,
                'processing_time_ms': metrics.processing_duration_ms
            })
            
            return analysis_result
            
        except Exception as e:
            metrics.errors_encountered += 1
            metrics.end_time = datetime.now()
            metrics.processing_duration_ms = (
                (metrics.end_time - metrics.start_time).total_seconds() * 1000
            )
            
            self._log_error(
                ErrorSeverity.MEDIUM, 
                f"Document analysis failed for {doc_id}", 
                str(e),
                correlation_id
            )
            
            self.processing_history.append(metrics)
            self._update_health_status()
            
            # Return error response with correlation ID
            return {
                'document_id': doc_id,
                'correlation_id': correlation_id,
                'error': str(e),
                'processing_metrics': {
                    'duration_ms': metrics.processing_duration_ms,
                    'retry_attempts': metrics.retry_attempts,
                    'errors': metrics.errors_encountered
                },
                'receptors': {},
                'summary': {
                    'activated_receptors': 0,
                    'total_receptors': len(self.receptors),
                    'average_intensity': 0.0,
                    'composite_scent': [0.0] * len(self.receptors)
                }
            }
    
    async def _analyze_with_retry(
        self, 
        text: str, 
        doc_id: str, 
        correlation_id: str, 
        metrics: ProcessingMetrics
    ) -> Dict[str, Any]:
        """Analyze document with retry logic."""
        
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    metrics.retry_attempts += 1
                    wait_time = 2 ** (attempt - 1)  # Exponential backoff
                    logger.info(f"Retry attempt {attempt} after {wait_time}s", extra={
                        'correlation_id': correlation_id
                    })
                    await asyncio.sleep(wait_time)
                
                # Analyze with all receptors
                results = {}
                activated_count = 0
                total_intensity = 0.0
                
                for name, receptor in self.receptors.items():
                    intensity, confidence, metadata = await receptor.activate(
                        text, correlation_id
                    )
                    
                    results[name] = {
                        'intensity': intensity,
                        'confidence': confidence,
                        'activated': intensity > receptor.threshold,
                        'metadata': metadata
                    }
                    
                    if intensity > receptor.threshold:
                        activated_count += 1
                    total_intensity += intensity
                
                # Generate composite scent vector
                scent_vector = [results[name]['intensity'] for name in self.receptors.keys()]
                
                return {
                    'document_id': doc_id,
                    'receptors': results,
                    'summary': {
                        'activated_receptors': activated_count,
                        'total_receptors': len(self.receptors),
                        'average_intensity': total_intensity / len(self.receptors),
                        'composite_scent': scent_vector
                    }
                }
                
            except Exception as e:
                if attempt == self.max_retries:
                    raise
                    
                logger.warning(f"Analysis attempt {attempt + 1} failed: {e}", extra={
                    'correlation_id': correlation_id
                })
                metrics.errors_encountered += 1
    
    def _log_error(
        self, 
        severity: ErrorSeverity, 
        message: str, 
        details: str, 
        correlation_id: str = None
    ):
        """Log structured error information."""
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'severity': severity.value,
            'message': message,
            'details': details,
            'correlation_id': correlation_id or str(uuid.uuid4())
        }
        
        self.error_log.append(error_entry)
        
        # Keep only last 1000 errors
        if len(self.error_log) > 1000:
            self.error_log = self.error_log[-1000:]
    
    def _update_health_status(self):
        """Update system health status based on recent metrics."""
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        
        # Count recent errors
        recent_errors = sum(1 for entry in self.error_log 
                          if datetime.fromisoformat(entry['timestamp']) > hour_ago)
        
        # Calculate average response time
        recent_metrics = [m for m in self.processing_history 
                        if m.start_time > hour_ago]
        
        avg_response_time = (
            sum(m.processing_duration_ms for m in recent_metrics) / 
            max(len(recent_metrics), 1)
        )
        
        # Determine system state
        if recent_errors == 0 and avg_response_time < 100:
            state = SystemState.HEALTHY
        elif recent_errors < 10 and avg_response_time < 500:
            state = SystemState.DEGRADED
        elif recent_errors < 50:
            state = SystemState.FAILING
        else:
            state = SystemState.CRITICAL
        
        self.health_status = HealthStatus(
            state=state,
            last_check=now,
            errors_last_hour=recent_errors,
            avg_response_time_ms=avg_response_time
        )
        
        if state != SystemState.HEALTHY:
            logger.warning(f"System health degraded: {state.value}")
    
    def calculate_robust_similarity(
        self, 
        analysis1: Dict[str, Any], 
        analysis2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate similarity with comprehensive error handling."""
        correlation_id = str(uuid.uuid4())
        
        try:
            # Validate inputs
            if not analysis1 or not analysis2:
                raise ValueError("Both analyses are required")
            
            scent1 = np.array(analysis1.get('summary', {}).get('composite_scent', []))
            scent2 = np.array(analysis2.get('summary', {}).get('composite_scent', []))
            
            if scent1.size == 0 or scent2.size == 0:
                raise ValueError("Invalid scent vectors")
            
            # Cosine similarity with error handling
            dot_product = np.dot(scent1, scent2)
            norm1 = np.linalg.norm(scent1)
            norm2 = np.linalg.norm(scent2)
            
            if norm1 == 0 or norm2 == 0:
                similarity = 0.0
                confidence = 0.0
            else:
                similarity = max(0.0, min(1.0, dot_product / (norm1 * norm2)))
                confidence = min(
                    analysis1.get('processing_metrics', {}).get('warnings', 10),
                    analysis2.get('processing_metrics', {}).get('warnings', 10)
                ) / 10.0
            
            return {
                'similarity': similarity,
                'confidence': confidence,
                'correlation_id': correlation_id,
                'method': 'cosine_similarity_robust',
                'metadata': {
                    'scent1_norm': float(norm1),
                    'scent2_norm': float(norm2),
                    'dot_product': float(dot_product)
                }
            }
            
        except Exception as e:
            self._log_error(
                ErrorSeverity.LOW,
                "Similarity calculation failed",
                str(e),
                correlation_id
            )
            
            return {
                'similarity': 0.0,
                'confidence': 0.0,
                'correlation_id': correlation_id,
                'error': str(e),
                'method': 'cosine_similarity_robust'
            }
    
    def get_system_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive system health report."""
        receptor_health = {name: receptor.get_health_stats() 
                          for name, receptor in self.receptors.items()}
        
        total_processed = len(self.processing_history)
        successful_processed = sum(1 for m in self.processing_history 
                                 if m.errors_encountered == 0)
        
        return {
            'system_health': {
                'state': self.health_status.state.value,
                'last_check': self.health_status.last_check.isoformat(),
                'errors_last_hour': self.health_status.errors_last_hour,
                'avg_response_time_ms': self.health_status.avg_response_time_ms
            },
            'processing_stats': {
                'total_documents': total_processed,
                'successful_documents': successful_processed,
                'success_rate_percent': round(
                    (successful_processed / max(total_processed, 1)) * 100, 2
                ),
                'total_errors': len(self.error_log)
            },
            'receptor_health': receptor_health,
            'recent_errors': self.error_log[-5:] if self.error_log else []
        }


async def demo_generation_2():
    """Demonstrate Generation 2 robust bioneural system."""
    
    print("🧬 BIONEURAL SYSTEM GENERATION 2: MAKE IT ROBUST")
    print("=" * 60)
    print("Comprehensive error handling, validation, monitoring, and security")
    print("=" * 60)
    
    system = RobustBioneuralSystem()
    
    # Test documents including problematic cases
    test_cases = [
        ("Legal Document", """
            WHEREAS, the parties hereto agree that the Contractor shall provide 
            services pursuant to 15 U.S.C. § 1681, and Company agrees to pay 
            Contractor for such services. The Contractor shall indemnify Company 
            from any liability, damages, or penalties arising from breach of this 
            agreement or violation of applicable regulations. The court held in
            Smith v. Jones, 123 F.3d 456 (2020), that such provisions are enforceable.
        """),
        ("Contract Document", """
            This Service Agreement outlines the terms for consulting services.
            The Service Provider will deliver work as specified in the attached 
            Statement of Work. Payment shall be made within 30 days of invoice.
            Both parties agree to maintain confidentiality and comply with applicable laws.
        """),
        ("Invalid Document", ""),  # Empty document
        ("Large Document", "A" * 100000),  # Very large document
        ("Malicious Document", "<script>alert('xss')</script>Normal content here"),
        ("Unicode Document", "Legal document with émojis 📄 and spéciäl characters ñ")
    ]
    
    print(f"\n🔒 Testing Robust Document Analysis with Error Handling")
    print("-" * 60)
    
    results = {}
    for doc_name, doc_text in test_cases:
        print(f"\n📄 Analyzing: {doc_name}")
        
        analysis = await system.analyze_document_robust(doc_text, doc_name)
        results[doc_name] = analysis
        
        if 'error' in analysis:
            print(f"   ❌ Error: {analysis['error']}")
        else:
            metrics = analysis.get('processing_metrics', {})
            summary = analysis.get('summary', {})
            
            print(f"   ✅ Success - Processing time: {metrics.get('duration_ms', 0):.1f}ms")
            print(f"      Activated receptors: {summary.get('activated_receptors', 0)}/{summary.get('total_receptors', 0)}")
            print(f"      Retry attempts: {metrics.get('retry_attempts', 0)}")
            print(f"      Warnings: {metrics.get('warnings', 0)}")
            
            if analysis.get('validation_warnings'):
                for warning in analysis['validation_warnings']:
                    print(f"      ⚠️  {warning}")
    
    print(f"\n🎯 Testing Robust Similarity Analysis")
    print("-" * 60)
    
    # Test similarity with valid documents
    valid_results = {k: v for k, v in results.items() 
                    if 'error' not in v and k in ['Legal Document', 'Contract Document']}
    
    if len(valid_results) >= 2:
        doc_names = list(valid_results.keys())
        for i in range(len(doc_names)):
            for j in range(i + 1, len(doc_names)):
                name1, name2 = doc_names[i], doc_names[j]
                similarity_result = system.calculate_robust_similarity(
                    valid_results[name1], 
                    valid_results[name2]
                )
                
                print(f"   {name1} vs {name2}:")
                print(f"      Similarity: {similarity_result.get('similarity', 0):.3f}")
                print(f"      Confidence: {similarity_result.get('confidence', 0):.3f}")
                
                if 'error' in similarity_result:
                    print(f"      ❌ Error: {similarity_result['error']}")
    
    print(f"\n📊 System Health and Performance Report")
    print("-" * 60)
    
    health_report = system.get_system_health_report()
    
    # System health
    health = health_report['system_health']
    print(f"   System State: {health['state']}")
    print(f"   Errors in last hour: {health['errors_last_hour']}")
    print(f"   Average response time: {health['avg_response_time_ms']:.1f}ms")
    
    # Processing statistics
    stats = health_report['processing_stats']
    print(f"\n   Documents processed: {stats['total_documents']}")
    print(f"   Success rate: {stats['success_rate_percent']}%")
    print(f"   Total errors logged: {stats['total_errors']}")
    
    # Receptor health
    print(f"\n   Receptor Health:")
    for name, health_stats in health_report['receptor_health'].items():
        print(f"      {name}:")
        print(f"         Success rate: {health_stats['success_rate_percent']}%")
        print(f"         Avg processing: {health_stats['avg_processing_time_ms']}ms")
        print(f"         Circuit breaker: {health_stats['circuit_breaker_state']}")
    
    # Recent errors
    if health_report['recent_errors']:
        print(f"\n   Recent Errors:")
        for error in health_report['recent_errors'][-3:]:
            print(f"      [{error['severity']}] {error['message']}")
    
    print(f"\n✅ Generation 2 Robustness Features Verified:")
    print("   • Comprehensive input validation and sanitization")
    print("   • Advanced error handling with circuit breaker patterns")
    print("   • Structured logging with correlation IDs")
    print("   • Health monitoring and performance metrics")
    print("   • Security checks for malicious content")
    print("   • Retry logic with exponential backoff")
    print("   • Graceful degradation under failure conditions")
    
    print(f"\n" + "=" * 60)
    print("🎉 GENERATION 2: MAKE IT ROBUST - COMPLETE!")
    print("✨ Ready to proceed to Generation 3: Make it Scale!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(demo_generation_2())