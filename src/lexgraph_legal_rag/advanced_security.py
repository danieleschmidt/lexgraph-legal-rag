"""Advanced security and input validation for legal RAG system.

This module provides comprehensive security measures including:
- Advanced input sanitization and validation
- Query injection prevention
- Rate limiting with intelligent throttling
- Anomaly detection for malicious queries
- Data privacy and PII protection
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
from collections import defaultdict
from collections import deque
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import Any


logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Security threat levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityViolationType(Enum):
    """Types of security violations."""

    MALICIOUS_QUERY = "malicious_query"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_PATTERN = "suspicious_pattern"
    PII_DETECTED = "pii_detected"
    INJECTION_ATTEMPT = "injection_attempt"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"


@dataclass
class SecurityEvent:
    """Security event record."""

    event_type: SecurityViolationType
    threat_level: ThreatLevel
    source_ip: str | None
    user_id: str | None
    query: str
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)
    blocked: bool = False

    def get_severity_score(self) -> float:
        """Get numeric severity score."""
        scores = {
            ThreatLevel.LOW: 1.0,
            ThreatLevel.MEDIUM: 3.0,
            ThreatLevel.HIGH: 7.0,
            ThreatLevel.CRITICAL: 10.0,
        }
        return scores.get(self.threat_level, 1.0)


class InputValidator:
    """Advanced input validation and sanitization."""

    def __init__(self):
        # Patterns for detecting malicious input
        self.malicious_patterns = [
            # SQL injection patterns
            r"(union\s+select|drop\s+table|delete\s+from)",
            r"(\'\s*;\s*--|\/\*.*\*\/)",
            # NoSQL injection patterns
            r"(\$where|\$ne|\$gt|\$lt|\$regex)",
            # Script injection patterns
            r"(<script|javascript:|vbscript:|onload=)",
            # Path traversal
            r"(\.\./|\.\.\\|\.\./\.\./)",
            # Command injection
            r"(;\s*cat\s+|;\s*ls\s+|;\s*rm\s+|;\s*chmod\s+)",
            # Legal system specific threats
            r"(drop\s+database|truncate\s+|exec\s+|execute\s+)",
            # Excessive pattern repetition (potential DoS)
            r"(.)\1{50,}",  # 50+ repeated characters
        ]

        # PII patterns
        self.pii_patterns = [
            # Social Security Numbers
            (r"\b\d{3}-\d{2}-\d{4}\b", "SSN"),
            (r"\b\d{9}\b", "SSN"),
            # Credit card numbers
            (r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", "Credit Card"),
            # Email addresses
            (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "Email"),
            # Phone numbers
            (r"\b\d{3}-\d{3}-\d{4}\b", "Phone"),
            (r"\(\d{3}\)\s?\d{3}-\d{4}", "Phone"),
            # Names (simple pattern - would be more sophisticated in production)
            (r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b", "Name"),
        ]

        # Legal query constraints
        self.max_query_length = 2000
        self.max_legal_terms = 20
        self.forbidden_terms = {
            "confidential_client_data",
            "attorney_work_product",
            "privileged_communication",
            "sealed_document",
        }

    async def validate_query(
        self, query: str, source_ip: str | None = None
    ) -> tuple[bool, list[str]]:
        """Validate query for security threats.

        Returns:
            Tuple of (is_valid, list_of_violations)
        """
        violations = []

        if not query or not isinstance(query, str):
            violations.append("Invalid query format")
            return False, violations

        # Length validation
        if len(query) > self.max_query_length:
            violations.append(f"Query exceeds maximum length ({self.max_query_length})")

        # Check for malicious patterns
        query_lower = query.lower()
        for pattern in self.malicious_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                violations.append(f"Malicious pattern detected: {pattern[:30]}...")

        # Check for excessive legal terms (potential enumeration attack)
        legal_term_count = len(
            re.findall(
                r"\b(contract|liability|breach|statute|law|legal|court|judge)\b",
                query_lower,
            )
        )
        if legal_term_count > self.max_legal_terms:
            violations.append(f"Excessive legal terms ({legal_term_count})")

        # Check for forbidden terms
        for forbidden in self.forbidden_terms:
            if forbidden in query_lower:
                violations.append(f"Forbidden term: {forbidden}")

        # PII detection
        pii_found = self._detect_pii(query)
        if pii_found:
            violations.extend(
                [f"PII detected: {pii_type}" for _, pii_type in pii_found]
            )

        return len(violations) == 0, violations

    def _detect_pii(self, text: str) -> list[tuple[str, str]]:
        """Detect personally identifiable information."""
        pii_found = []

        for pattern, pii_type in self.pii_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                pii_found.append((match, pii_type))

        return pii_found

    def sanitize_query(self, query: str) -> str:
        """Sanitize query by removing potentially harmful content."""
        sanitized = query.strip()

        # Remove HTML/XML tags
        sanitized = re.sub(r"<[^>]+>", "", sanitized)

        # Remove excessive whitespace
        sanitized = re.sub(r"\s+", " ", sanitized)

        # Remove control characters
        sanitized = "".join(char for char in sanitized if ord(char) >= 32)

        # Truncate if too long
        if len(sanitized) > self.max_query_length:
            sanitized = sanitized[: self.max_query_length]

        return sanitized


class IntelligentRateLimiter:
    """Intelligent rate limiting with adaptive throttling."""

    def __init__(self):
        self.client_requests: dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.client_violations: dict[str, int] = defaultdict(int)
        self.client_reputation: dict[str, float] = defaultdict(
            lambda: 1.0
        )  # 1.0 = good, 0.0 = blocked
        self._lock = threading.RLock()

        # Base rate limits (requests per minute)
        self.base_limits = {"default": 60, "premium": 300, "internal": 1000}

        # Reputation-based multipliers
        self.reputation_multipliers = {
            "high": 2.0,  # Above 0.8 reputation
            "medium": 1.0,  # 0.3 - 0.8 reputation
            "low": 0.5,  # 0.1 - 0.3 reputation
            "blocked": 0.0,  # Below 0.1 reputation
        }

    def _get_client_id(self, source_ip: str, user_id: str | None = None) -> str:
        """Generate client identifier."""
        if user_id:
            return f"user:{user_id}"
        return f"ip:{source_ip}"

    def _get_reputation_tier(self, reputation: float) -> str:
        """Get reputation tier for client."""
        if reputation >= 0.8:
            return "high"
        elif reputation >= 0.3:
            return "medium"
        elif reputation >= 0.1:
            return "low"
        else:
            return "blocked"

    async def check_rate_limit(
        self, source_ip: str, user_id: str | None = None, client_tier: str = "default"
    ) -> tuple[bool, dict[str, Any]]:
        """Check if request should be rate limited.

        Returns:
            Tuple of (allowed, rate_limit_info)
        """
        with self._lock:
            client_id = self._get_client_id(source_ip, user_id)
            current_time = time.time()

            # Clean old requests (beyond 1 minute)
            client_requests = self.client_requests[client_id]
            while client_requests and current_time - client_requests[0] > 60:
                client_requests.popleft()

            # Get effective rate limit based on reputation
            reputation = self.client_reputation[client_id]
            reputation_tier = self._get_reputation_tier(reputation)

            base_limit = self.base_limits.get(client_tier, self.base_limits["default"])
            multiplier = self.reputation_multipliers[reputation_tier]
            effective_limit = int(base_limit * multiplier)

            current_requests = len(client_requests)
            allowed = current_requests < effective_limit

            if allowed:
                client_requests.append(current_time)
            else:
                # Record violation
                self.client_violations[client_id] += 1
                # Reduce reputation
                self._update_reputation(client_id, -0.1)

            rate_limit_info = {
                "limit": effective_limit,
                "remaining": max(0, effective_limit - current_requests),
                "reset_time": current_time + 60,
                "reputation": reputation,
                "reputation_tier": reputation_tier,
                "violations": self.client_violations[client_id],
            }

            return allowed, rate_limit_info

    def _update_reputation(self, client_id: str, change: float) -> None:
        """Update client reputation score."""
        current = self.client_reputation[client_id]
        new_reputation = max(0.0, min(1.0, current + change))
        self.client_reputation[client_id] = new_reputation

        # Log significant reputation changes
        if abs(change) >= 0.1:
            logger.info(
                f"Client {client_id} reputation: {current:.2f} -> {new_reputation:.2f}"
            )

    def record_successful_request(
        self, source_ip: str, user_id: str | None = None
    ) -> None:
        """Record successful request to improve reputation."""
        with self._lock:
            client_id = self._get_client_id(source_ip, user_id)
            self._update_reputation(client_id, 0.01)  # Small positive increment

    def record_security_violation(
        self, source_ip: str, violation_severity: float, user_id: str | None = None
    ) -> None:
        """Record security violation and penalize reputation."""
        with self._lock:
            client_id = self._get_client_id(source_ip, user_id)
            penalty = min(0.3, violation_severity * 0.1)  # Cap penalty at 0.3
            self._update_reputation(client_id, -penalty)
            self.client_violations[client_id] += 1


class AnomalyDetector:
    """Detects anomalous query patterns and behaviors."""

    def __init__(self):
        self.query_patterns: dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.client_behavior: dict[str, dict] = defaultdict(dict)
        self._lock = threading.RLock()

        # Anomaly thresholds
        self.thresholds = {
            "rapid_identical_queries": 5,  # Same query 5+ times in 1 minute
            "pattern_repetition": 10,  # Similar patterns 10+ times
            "unusual_query_length": 1500,  # Queries over 1500 chars
            "high_frequency_client": 100,  # 100+ queries in 5 minutes
        }

    async def analyze_query(
        self, query: str, source_ip: str, user_id: str | None = None
    ) -> list[tuple[str, float]]:
        """Analyze query for anomalous patterns.

        Returns:
            List of (anomaly_type, severity_score) tuples
        """
        anomalies = []

        with self._lock:
            client_id = f"{source_ip}:{user_id or 'anonymous'}"
            current_time = time.time()

            # Initialize client behavior tracking
            if client_id not in self.client_behavior:
                self.client_behavior[client_id] = {
                    "query_times": deque(maxlen=200),
                    "query_hashes": deque(maxlen=50),
                    "total_queries": 0,
                    "first_seen": current_time,
                }

            behavior = self.client_behavior[client_id]
            behavior["query_times"].append(current_time)
            behavior["total_queries"] += 1

            # Generate query hash for pattern detection
            query_hash = hashlib.md5(
                query.lower().encode(), usedforsecurity=False
            ).hexdigest()
            behavior["query_hashes"].append(query_hash)

            # Check for rapid identical queries
            recent_hashes = list(behavior["query_hashes"])[-10:]  # Last 10 queries
            identical_count = recent_hashes.count(query_hash)
            if identical_count >= self.thresholds["rapid_identical_queries"]:
                anomalies.append(("rapid_identical_queries", identical_count / 10.0))

            # Check for high frequency behavior
            recent_queries = [
                t for t in behavior["query_times"] if current_time - t <= 300
            ]  # 5 minutes
            if len(recent_queries) >= self.thresholds["high_frequency_client"]:
                anomalies.append(("high_frequency_client", len(recent_queries) / 100.0))

            # Check for unusual query characteristics
            if len(query) >= self.thresholds["unusual_query_length"]:
                anomalies.append(("unusual_query_length", len(query) / 2000.0))

            # Pattern repetition analysis (simplified)
            if len(recent_hashes) >= 10:
                unique_patterns = len(set(recent_hashes))
                repetition_score = 1.0 - (unique_patterns / len(recent_hashes))
                if repetition_score > 0.7:  # 70% repetition
                    anomalies.append(("pattern_repetition", repetition_score))

        return anomalies


class SecurityManager:
    """Central security management system."""

    def __init__(self):
        self.input_validator = InputValidator()
        self.rate_limiter = IntelligentRateLimiter()
        self.anomaly_detector = AnomalyDetector()
        self.security_events: list[SecurityEvent] = []
        self._lock = threading.RLock()

        # Security policies
        self.block_on_critical = True
        self.log_all_violations = True
        self.auto_ban_threshold = 10  # Auto-ban after 10 violations

        logger.info("Security manager initialized")

    async def validate_request(
        self,
        query: str,
        source_ip: str,
        user_id: str | None = None,
        client_tier: str = "default",
    ) -> tuple[bool, dict[str, Any]]:
        """Comprehensive request validation.

        Returns:
            Tuple of (allowed, security_info)
        """
        start_time = time.time()
        security_info = {
            "validation_time": 0.0,
            "violations": [],
            "threat_level": ThreatLevel.LOW,
            "rate_limit_info": {},
            "anomalies": [],
            "blocked_reason": None,
        }

        try:
            # Input validation
            is_valid, violations = await self.input_validator.validate_query(
                query, source_ip
            )
            security_info["violations"] = violations

            if violations:
                max_threat = (
                    ThreatLevel.HIGH
                    if any(
                        "malicious" in v.lower() or "forbidden" in v.lower()
                        for v in violations
                    )
                    else ThreatLevel.MEDIUM
                )
                security_info["threat_level"] = max_threat

            # Rate limiting check
            rate_allowed, rate_info = await self.rate_limiter.check_rate_limit(
                source_ip, user_id, client_tier
            )
            security_info["rate_limit_info"] = rate_info

            if not rate_allowed:
                security_info["blocked_reason"] = "rate_limit_exceeded"
                self._record_security_event(
                    SecurityViolationType.RATE_LIMIT_EXCEEDED,
                    ThreatLevel.MEDIUM,
                    source_ip,
                    user_id,
                    query,
                    blocked=True,
                )

            # Anomaly detection
            anomalies = await self.anomaly_detector.analyze_query(
                query, source_ip, user_id
            )
            security_info["anomalies"] = anomalies

            if anomalies:
                # Calculate overall anomaly severity
                max_severity = max([severity for _, severity in anomalies], default=0)
                if max_severity > 0.8:
                    security_info["threat_level"] = ThreatLevel.HIGH
                elif max_severity > 0.5:
                    security_info["threat_level"] = max(
                        security_info["threat_level"], ThreatLevel.MEDIUM
                    )

            # Make final decision
            allowed = True

            # Block on critical threats
            if (
                self.block_on_critical
                and security_info["threat_level"] == ThreatLevel.CRITICAL
            ):
                allowed = False
                security_info["blocked_reason"] = "critical_threat_detected"

            # Block if input validation failed severely
            if not is_valid and any(
                "malicious" in v.lower() or "forbidden" in v.lower() for v in violations
            ):
                allowed = False
                security_info["blocked_reason"] = "malicious_input_detected"

            # Block if rate limited
            if not rate_allowed:
                allowed = False

            # Record security events
            if violations or anomalies or not allowed:
                event_type = self._determine_event_type(violations, anomalies)
                self._record_security_event(
                    event_type,
                    security_info["threat_level"],
                    source_ip,
                    user_id,
                    query,
                    blocked=not allowed,
                )
            else:
                # Record successful request for reputation
                self.rate_limiter.record_successful_request(source_ip, user_id)

            security_info["validation_time"] = time.time() - start_time
            return allowed, security_info

        except Exception as e:
            logger.error(f"Security validation error: {e}")
            # Fail secure - block on error
            security_info["blocked_reason"] = "security_system_error"
            security_info["validation_time"] = time.time() - start_time
            return False, security_info

    def _determine_event_type(
        self, violations: list[str], anomalies: list[tuple[str, float]]
    ) -> SecurityViolationType:
        """Determine the primary security event type."""
        if any("malicious" in v.lower() for v in violations):
            return SecurityViolationType.MALICIOUS_QUERY
        if any("pii" in v.lower() for v in violations):
            return SecurityViolationType.PII_DETECTED
        if any("forbidden" in v.lower() for v in violations):
            return SecurityViolationType.INJECTION_ATTEMPT
        if anomalies:
            return SecurityViolationType.ANOMALOUS_BEHAVIOR
        return SecurityViolationType.SUSPICIOUS_PATTERN

    def _record_security_event(
        self,
        event_type: SecurityViolationType,
        threat_level: ThreatLevel,
        source_ip: str | None,
        user_id: str | None,
        query: str,
        blocked: bool = False,
    ) -> None:
        """Record a security event."""
        with self._lock:
            event = SecurityEvent(
                event_type=event_type,
                threat_level=threat_level,
                source_ip=source_ip,
                user_id=user_id,
                query=query[:200]
                + ("..." if len(query) > 200 else ""),  # Truncate for logs
                timestamp=time.time(),
                blocked=blocked,
            )

            self.security_events.append(event)

            # Keep only recent events (last 24 hours)
            cutoff_time = time.time() - 86400
            self.security_events = [
                e for e in self.security_events if e.timestamp >= cutoff_time
            ]

            # Update client reputation based on event
            if source_ip:
                severity_score = event.get_severity_score()
                self.rate_limiter.record_security_violation(
                    source_ip, severity_score, user_id
                )

            if self.log_all_violations:
                logger.warning(
                    f"Security event: {event_type.value} ({threat_level.value}) "
                    f"from {source_ip or 'unknown'} - {'BLOCKED' if blocked else 'ALLOWED'}"
                )

    def get_security_report(self, hours: int = 24) -> dict[str, Any]:
        """Get comprehensive security report."""
        with self._lock:
            cutoff_time = time.time() - (hours * 3600)
            recent_events = [
                e for e in self.security_events if e.timestamp >= cutoff_time
            ]

            # Event type distribution
            event_counts = {}
            for event in recent_events:
                event_counts[event.event_type.value] = (
                    event_counts.get(event.event_type.value, 0) + 1
                )

            # Threat level distribution
            threat_counts = {}
            for event in recent_events:
                threat_counts[event.threat_level.value] = (
                    threat_counts.get(event.threat_level.value, 0) + 1
                )

            # Top threat sources
            source_counts = {}
            for event in recent_events:
                if event.source_ip:
                    source_counts[event.source_ip] = (
                        source_counts.get(event.source_ip, 0) + 1
                    )

            top_sources = sorted(
                source_counts.items(), key=lambda x: x[1], reverse=True
            )[:10]

            return {
                "report_period_hours": hours,
                "total_events": len(recent_events),
                "blocked_requests": sum(1 for e in recent_events if e.blocked),
                "event_types": event_counts,
                "threat_levels": threat_counts,
                "top_threat_sources": top_sources,
                "security_policies": {
                    "block_on_critical": self.block_on_critical,
                    "log_all_violations": self.log_all_violations,
                    "auto_ban_threshold": self.auto_ban_threshold,
                },
            }


# Global security manager instance
_global_security_manager = None


def get_security_manager() -> SecurityManager:
    """Get global security manager instance."""
    global _global_security_manager
    if _global_security_manager is None:
        _global_security_manager = SecurityManager()
    return _global_security_manager


async def validate_request(
    query: str, source_ip: str, user_id: str | None = None, client_tier: str = "default"
) -> tuple[bool, dict[str, Any]]:
    """Convenience function for request validation."""
    manager = get_security_manager()
    return await manager.validate_request(query, source_ip, user_id, client_tier)


def get_security_report(hours: int = 24) -> dict[str, Any]:
    """Convenience function to get security report."""
    manager = get_security_manager()
    return manager.get_security_report(hours)
