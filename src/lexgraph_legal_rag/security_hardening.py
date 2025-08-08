"""
Advanced Security Hardening Module
Comprehensive security features including input validation, rate limiting, and threat detection
"""

import os
import re
import json
import time
import hashlib
import logging
import secrets
import threading
from typing import Dict, List, Any, Optional, Union, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from ipaddress import ip_address, ip_network, AddressValueError
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SecurityEvent:
    """Represents a security event for monitoring and alerting."""
    event_type: str
    source_ip: str
    user_agent: Optional[str]
    endpoint: str
    payload_hash: Optional[str]
    timestamp: datetime
    severity: str  # low, medium, high, critical
    blocked: bool
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ThreatSignature:
    """Represents a threat detection signature."""
    name: str
    pattern: str
    description: str
    severity: str
    action: str  # log, block, alert
    enabled: bool = True


class InputSanitizer:
    """Advanced input sanitization and validation."""
    
    # Dangerous patterns that should be blocked
    DANGEROUS_PATTERNS = [
        # SQL Injection patterns
        r"(?i)(union\s+select|insert\s+into|delete\s+from|drop\s+table|exec\s*\()",
        r"(?i)(script\s*:|javascript\s*:|vbscript\s*:)",
        r"(?i)(<script[^>]*>.*?</script>|<iframe[^>]*>.*?</iframe>)",
        
        # XSS patterns
        r"(?i)(<script|<object|<embed|<applet|<meta|<link)",
        r"(?i)(on\w+\s*=|href\s*=\s*[\"']?\s*javascript:|src\s*=\s*[\"']?\s*javascript:)",
        
        # Command injection patterns
        r"(?i)(;\s*rm\s|;\s*cat\s|;\s*ls\s|;\s*ps\s|;\s*kill\s)",
        r"(?i)(`|\\$\\(|\\.\\./|/etc/passwd|/etc/shadow)",
        
        # Path traversal patterns
        r"(?i)(\\.\\./|\\.\\.\\\|%2e%2e%2f|%252e%252e%252f)",
        
        # LDAP injection patterns
        r"(?i)(\\*\\)|\\*\\(|\\)\\(|\\(\\*)",
        
        # NoSQL injection patterns
        r"(?i)(\\$where|\\$regex|\\$gt|\\$lt|\\$ne|\\$in)"
    ]
    
    # File extension blacklist
    DANGEROUS_EXTENSIONS = {
        '.exe', '.bat', '.cmd', '.com', '.pif', '.scr', '.vbs', '.vbe',
        '.js', '.jar', '.jsp', '.php', '.asp', '.aspx', '.sh', '.py',
        '.rb', '.pl', '.ps1', '.psm1'
    }
    
    def __init__(self):
        self.compiled_patterns = [re.compile(pattern) for pattern in self.DANGEROUS_PATTERNS]
        self._threat_cache = {}
        self._validation_stats = defaultdict(int)
    
    def sanitize_string(self, input_str: str, max_length: int = 10000) -> str:
        """
        Sanitize string input by removing/escaping dangerous content.
        
        Args:
            input_str: String to sanitize
            max_length: Maximum allowed string length
            
        Returns:
            Sanitized string
            
        Raises:
            ValueError: If input contains dangerous patterns
        """
        if not isinstance(input_str, str):
            raise ValueError("Input must be a string")
        
        # Length check
        if len(input_str) > max_length:
            raise ValueError(f"Input exceeds maximum length of {max_length}")
        
        self._validation_stats['total_validations'] += 1
        
        # Check for dangerous patterns
        threats_found = []
        for i, pattern in enumerate(self.compiled_patterns):
            if pattern.search(input_str):
                threats_found.append(self.DANGEROUS_PATTERNS[i])
                self._validation_stats['threats_detected'] += 1
        
        if threats_found:
            logger.warning(f"Dangerous patterns detected in input: {threats_found}")
            raise ValueError(f"Input contains dangerous patterns: {len(threats_found)} threats detected")
        
        # Basic HTML escaping
        sanitized = (input_str.replace('&', '&amp;')
                              .replace('<', '&lt;')
                              .replace('>', '&gt;')
                              .replace('"', '&quot;')
                              .replace("'", '&#x27;'))
        
        self._validation_stats['successful_sanitizations'] += 1
        return sanitized
    
    def validate_filename(self, filename: str) -> str:
        """
        Validate and sanitize filename.
        
        Args:
            filename: Filename to validate
            
        Returns:
            Sanitized filename
            
        Raises:
            ValueError: If filename is invalid or dangerous
        """
        if not filename or len(filename) > 255:
            raise ValueError("Invalid filename length")
        
        # Check for path traversal
        if '..' in filename or filename.startswith('/') or ':' in filename:
            raise ValueError("Filename contains path traversal patterns")
        
        # Check extension
        file_path = Path(filename)
        if file_path.suffix.lower() in self.DANGEROUS_EXTENSIONS:
            raise ValueError(f"File extension {file_path.suffix} is not allowed")
        
        # Remove dangerous characters
        safe_filename = re.sub(r'[^\w\-_\.]', '', filename)
        
        if not safe_filename:
            raise ValueError("Filename contains only invalid characters")
        
        return safe_filename
    
    def validate_json_payload(self, payload: str, max_depth: int = 10, max_keys: int = 1000) -> Dict[str, Any]:
        """
        Validate and parse JSON payload with safety checks.
        
        Args:
            payload: JSON string to validate
            max_depth: Maximum nesting depth allowed
            max_keys: Maximum number of keys allowed
            
        Returns:
            Parsed JSON object
            
        Raises:
            ValueError: If JSON is invalid or unsafe
        """
        if len(payload) > 1_000_000:  # 1MB limit
            raise ValueError("JSON payload exceeds size limit")
        
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")
        
        # Check depth and key count
        if self._get_json_depth(data) > max_depth:
            raise ValueError(f"JSON depth exceeds limit of {max_depth}")
        
        if self._count_json_keys(data) > max_keys:
            raise ValueError(f"JSON key count exceeds limit of {max_keys}")
        
        return data
    
    def _get_json_depth(self, obj: Any, current_depth: int = 0) -> int:
        """Calculate maximum depth of JSON object."""
        if isinstance(obj, dict):
            if not obj:
                return current_depth
            return max(self._get_json_depth(v, current_depth + 1) for v in obj.values())
        elif isinstance(obj, list):
            if not obj:
                return current_depth
            return max(self._get_json_depth(item, current_depth + 1) for item in obj)
        else:
            return current_depth
    
    def _count_json_keys(self, obj: Any) -> int:
        """Count total number of keys in JSON object."""
        if isinstance(obj, dict):
            count = len(obj)
            for value in obj.values():
                count += self._count_json_keys(value)
            return count
        elif isinstance(obj, list):
            return sum(self._count_json_keys(item) for item in obj)
        else:
            return 0
    
    def get_validation_stats(self) -> Dict[str, int]:
        """Get input validation statistics."""
        return dict(self._validation_stats)


class RateLimiter:
    """Advanced rate limiting with multiple algorithms and IP tracking."""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client  # Optional Redis for distributed rate limiting
        self.local_buckets = {}
        self.ip_tracking = defaultdict(lambda: {'requests': deque(), 'blocked_until': None})
        self.lock = threading.RLock()
        
        # Rate limiting rules
        self.rules = {
            'default': {'requests': 100, 'window': 3600},  # 100 per hour
            'search': {'requests': 1000, 'window': 3600},  # 1000 search requests per hour
            'auth': {'requests': 10, 'window': 300},       # 10 auth attempts per 5 minutes
            'upload': {'requests': 20, 'window': 3600},    # 20 uploads per hour
        }
        
        # Burst protection
        self.burst_rules = {
            'default': {'requests': 20, 'window': 60},     # 20 per minute burst
            'search': {'requests': 50, 'window': 60},      # 50 search per minute
            'auth': {'requests': 3, 'window': 60},         # 3 auth per minute
        }
    
    def is_allowed(self, identifier: str, rule_name: str = 'default', 
                   check_burst: bool = True) -> tuple[bool, Dict[str, Any]]:
        """
        Check if request is allowed under rate limiting rules.
        
        Args:
            identifier: IP address or user identifier
            rule_name: Name of rate limiting rule to apply
            check_burst: Whether to also check burst protection
            
        Returns:
            Tuple of (allowed, metadata)
        """
        with self.lock:
            current_time = time.time()
            
            # Clean up old tracking data
            self._cleanup_old_data(current_time)
            
            # Check if IP is currently blocked
            ip_data = self.ip_tracking[identifier]
            if ip_data['blocked_until'] and current_time < ip_data['blocked_until']:
                return False, {
                    'rule': rule_name,
                    'blocked_until': ip_data['blocked_until'],
                    'reason': 'rate_limit_exceeded'
                }
            
            # Get rate limiting rule
            rule = self.rules.get(rule_name, self.rules['default'])
            burst_rule = self.burst_rules.get(rule_name, self.burst_rules['default'])
            
            # Check main rate limit
            allowed_main = self._check_sliding_window(identifier, rule['requests'], rule['window'])
            allowed_burst = True
            
            if check_burst:
                allowed_burst = self._check_sliding_window(
                    f"{identifier}:burst", burst_rule['requests'], burst_rule['window']
                )
            
            allowed = allowed_main and allowed_burst
            
            if not allowed:
                # Block IP for progressive duration
                block_duration = self._calculate_block_duration(identifier)
                ip_data['blocked_until'] = current_time + block_duration
                
                logger.warning(f"Rate limit exceeded for {identifier}: "
                              f"rule={rule_name}, blocked_for={block_duration}s")
            
            # Record request
            ip_data['requests'].append(current_time)
            
            return allowed, {
                'rule': rule_name,
                'remaining': max(0, rule['requests'] - len([
                    t for t in ip_data['requests'] 
                    if current_time - t <= rule['window']
                ])),
                'reset_time': current_time + rule['window'],
                'blocked': not allowed
            }
    
    def _check_sliding_window(self, key: str, limit: int, window: int) -> bool:
        """Check sliding window rate limit."""
        current_time = time.time()
        
        if self.redis_client:
            # Use Redis for distributed rate limiting
            return self._redis_sliding_window(key, limit, window, current_time)
        else:
            # Use local sliding window
            if key not in self.local_buckets:
                self.local_buckets[key] = deque()
            
            bucket = self.local_buckets[key]
            
            # Remove expired requests
            while bucket and current_time - bucket[0] > window:
                bucket.popleft()
            
            return len(bucket) < limit
    
    def _redis_sliding_window(self, key: str, limit: int, window: int, current_time: float) -> bool:
        """Redis-based sliding window rate limiting."""
        # This would implement Redis-based distributed rate limiting
        # For now, fall back to local implementation
        return self._check_sliding_window(key, limit, window)
    
    def _calculate_block_duration(self, identifier: str) -> float:
        """Calculate progressive block duration based on violation history."""
        ip_data = self.ip_tracking[identifier]
        recent_blocks = len([
            t for t in ip_data['requests'][-100:] 
            if time.time() - t <= 3600  # Last hour
        ])
        
        # Progressive blocking: 60s, 300s, 900s, 3600s
        durations = [60, 300, 900, 3600]
        return durations[min(recent_blocks // 10, len(durations) - 1)]
    
    def _cleanup_old_data(self, current_time: float) -> None:
        """Clean up old tracking data to prevent memory leaks."""
        cutoff_time = current_time - 86400  # 24 hours
        
        for identifier, data in list(self.ip_tracking.items()):
            # Remove old requests
            while data['requests'] and data['requests'][0] < cutoff_time:
                data['requests'].popleft()
            
            # Remove empty tracking records
            if not data['requests'] and (not data['blocked_until'] or data['blocked_until'] < current_time):
                del self.ip_tracking[identifier]
    
    def get_rate_limit_stats(self) -> Dict[str, Any]:
        """Get rate limiting statistics."""
        with self.lock:
            current_time = time.time()
            
            active_ips = len(self.ip_tracking)
            blocked_ips = len([
                data for data in self.ip_tracking.values()
                if data['blocked_until'] and current_time < data['blocked_until']
            ])
            
            total_requests = sum(len(data['requests']) for data in self.ip_tracking.values())
            
            return {
                'active_ips': active_ips,
                'blocked_ips': blocked_ips,
                'total_requests_tracked': total_requests,
                'rules': self.rules.copy(),
                'burst_rules': self.burst_rules.copy()
            }


class ThreatDetector:
    """Advanced threat detection and anomaly detection system."""
    
    def __init__(self):
        self.security_events = deque(maxlen=10000)
        self.threat_signatures = self._load_threat_signatures()
        self.anomaly_baselines = defaultdict(lambda: {'mean': 0, 'std': 0, 'samples': []})
        self.lock = threading.RLock()
    
    def _load_threat_signatures(self) -> List[ThreatSignature]:
        """Load threat detection signatures."""
        signatures = [
            ThreatSignature(
                name="sql_injection_attempt",
                pattern=r"(?i)(union\s+select|insert\s+into|delete\s+from)",
                description="SQL injection attempt detected",
                severity="high",
                action="block"
            ),
            ThreatSignature(
                name="xss_attempt",
                pattern=r"(?i)(<script|javascript:|on\w+\s*=)",
                description="Cross-site scripting attempt detected",
                severity="high",
                action="block"
            ),
            ThreatSignature(
                name="path_traversal",
                pattern=r"(?i)(\.\./|%2e%2e%2f|/etc/passwd)",
                description="Path traversal attack detected",
                severity="high",
                action="block"
            ),
            ThreatSignature(
                name="command_injection",
                pattern=r"(?i)(;\s*rm\s|;\s*cat\s|`|\$\()",
                description="Command injection attempt detected",
                severity="critical",
                action="block"
            ),
            ThreatSignature(
                name="suspicious_user_agent",
                pattern=r"(?i)(sqlmap|nikto|burp|nmap|curl|wget|python-requests)",
                description="Suspicious user agent detected",
                severity="medium",
                action="log"
            ),
            ThreatSignature(
                name="bruteforce_pattern",
                pattern=r"",  # Detected via behavioral analysis
                description="Brute force attack pattern detected",
                severity="high",
                action="block"
            )
        ]
        return signatures
    
    def analyze_request(self, request_data: Dict[str, Any]) -> List[SecurityEvent]:
        """
        Analyze request for security threats.
        
        Args:
            request_data: Dictionary containing request information
                - ip: source IP address
                - user_agent: user agent string
                - endpoint: requested endpoint
                - payload: request payload
                - headers: request headers
        
        Returns:
            List of detected security events
        """
        events = []
        current_time = datetime.now()
        
        # Extract request components
        source_ip = request_data.get('ip', 'unknown')
        user_agent = request_data.get('user_agent', '')
        endpoint = request_data.get('endpoint', '')
        payload = request_data.get('payload', '')
        headers = request_data.get('headers', {})
        
        # Pattern-based detection
        for signature in self.threat_signatures:
            if not signature.enabled:
                continue
            
            threat_detected = False
            details = {}
            
            # Check different request components
            if signature.name == "suspicious_user_agent" and user_agent:
                if re.search(signature.pattern, user_agent):
                    threat_detected = True
                    details['user_agent'] = user_agent
            
            elif signature.pattern and payload:
                if re.search(signature.pattern, payload):
                    threat_detected = True
                    details['payload_match'] = True
            
            elif signature.pattern and endpoint:
                if re.search(signature.pattern, endpoint):
                    threat_detected = True
                    details['endpoint_match'] = True
            
            # Behavioral detection for brute force
            elif signature.name == "bruteforce_pattern":
                if self._detect_bruteforce(source_ip, endpoint):
                    threat_detected = True
                    details['pattern'] = 'rapid_requests'
            
            if threat_detected:
                event = SecurityEvent(
                    event_type=signature.name,
                    source_ip=source_ip,
                    user_agent=user_agent,
                    endpoint=endpoint,
                    payload_hash=hashlib.sha256(payload.encode()).hexdigest() if payload else None,
                    timestamp=current_time,
                    severity=signature.severity,
                    blocked=(signature.action == "block"),
                    details=details
                )
                events.append(event)
                
                logger.warning(f"Security threat detected: {signature.name} from {source_ip}")
        
        # Anomaly detection
        anomaly_events = self._detect_anomalies(request_data)
        events.extend(anomaly_events)
        
        # Store events
        with self.lock:
            self.security_events.extend(events)
        
        return events
    
    def _detect_bruteforce(self, source_ip: str, endpoint: str) -> bool:
        """Detect brute force attack patterns."""
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(minutes=10)
        
        # Count recent requests from this IP to this endpoint
        recent_requests = [
            event for event in self.security_events
            if (event.source_ip == source_ip and 
                event.endpoint == endpoint and 
                event.timestamp > cutoff_time)
        ]
        
        # Threshold for brute force detection
        if len(recent_requests) > 20:  # 20 requests in 10 minutes
            return True
        
        return False
    
    def _detect_anomalies(self, request_data: Dict[str, Any]) -> List[SecurityEvent]:
        """Detect anomalous request patterns."""
        events = []
        
        # Payload size anomaly
        payload = request_data.get('payload', '')
        if payload and len(payload) > 100000:  # 100KB threshold
            event = SecurityEvent(
                event_type="large_payload_anomaly",
                source_ip=request_data.get('ip', 'unknown'),
                user_agent=request_data.get('user_agent'),
                endpoint=request_data.get('endpoint', ''),
                payload_hash=hashlib.sha256(payload.encode()).hexdigest(),
                timestamp=datetime.now(),
                severity="medium",
                blocked=False,
                details={'payload_size': len(payload)}
            )
            events.append(event)
        
        return events
    
    def get_security_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get security summary for the specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self.lock:
            recent_events = [
                event for event in self.security_events
                if event.timestamp > cutoff_time
            ]
        
        if not recent_events:
            return {"message": "No security events in the specified period"}
        
        # Aggregate statistics
        events_by_type = defaultdict(int)
        events_by_severity = defaultdict(int)
        events_by_ip = defaultdict(int)
        blocked_events = 0
        
        for event in recent_events:
            events_by_type[event.event_type] += 1
            events_by_severity[event.severity] += 1
            events_by_ip[event.source_ip] += 1
            if event.blocked:
                blocked_events += 1
        
        return {
            "time_period_hours": hours,
            "total_events": len(recent_events),
            "blocked_events": blocked_events,
            "events_by_type": dict(events_by_type),
            "events_by_severity": dict(events_by_severity),
            "top_source_ips": dict(sorted(events_by_ip.items(), key=lambda x: x[1], reverse=True)[:10]),
            "threat_signatures_active": len([s for s in self.threat_signatures if s.enabled])
        }


class SecurityHardeningManager:
    """Main security hardening manager coordinating all security features."""
    
    def __init__(self, repo_path: str = "/root/repo"):
        self.repo_path = Path(repo_path)
        self.security_dir = self.repo_path / "security"
        self.security_dir.mkdir(exist_ok=True)
        
        # Initialize security components
        self.input_sanitizer = InputSanitizer()
        self.rate_limiter = RateLimiter()
        self.threat_detector = ThreatDetector()
        
        # Security configuration
        self.security_config = self._load_security_config()
        
        # Trusted IP networks (for allowlisting)
        self.trusted_networks = self._load_trusted_networks()
    
    def _load_security_config(self) -> Dict[str, Any]:
        """Load security configuration."""
        config_file = self.security_dir / "security_config.json"
        default_config = {
            "rate_limiting_enabled": True,
            "threat_detection_enabled": True,
            "input_validation_enabled": True,
            "security_headers_enabled": True,
            "audit_logging_enabled": True,
            "max_request_size": 10485760,  # 10MB
            "session_timeout_minutes": 30,
            "password_policy": {
                "min_length": 12,
                "require_uppercase": True,
                "require_lowercase": True,
                "require_numbers": True,
                "require_special": True
            }
        }
        
        try:
            if config_file.exists():
                with open(config_file) as f:
                    user_config = json.load(f)
                default_config.update(user_config)
        except Exception as e:
            logger.warning(f"Could not load security config: {e}")
        
        return default_config
    
    def _load_trusted_networks(self) -> List[str]:
        """Load trusted IP networks for allowlisting."""
        return [
            "127.0.0.0/8",    # Localhost
            "10.0.0.0/8",     # Private network
            "172.16.0.0/12",  # Private network
            "192.168.0.0/16"  # Private network
        ]
    
    def is_trusted_ip(self, ip_address: str) -> bool:
        """Check if IP address is in trusted networks."""
        try:
            ip = ip_address(ip_address)
            for network in self.trusted_networks:
                if ip in ip_network(network):
                    return True
            return False
        except (AddressValueError, ValueError):
            return False
    
    def validate_and_secure_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive request validation and security checking.
        
        Args:
            request_data: Request data to validate
            
        Returns:
            Dictionary with validation results and sanitized data
        """
        source_ip = request_data.get('ip', 'unknown')
        endpoint = request_data.get('endpoint', '')
        
        # Check if IP is trusted (skip some checks)
        is_trusted = self.is_trusted_ip(source_ip)
        
        validation_results = {
            'allowed': True,
            'reasons': [],
            'sanitized_data': {},
            'security_events': [],
            'rate_limit_info': {}
        }
        
        try:
            # Rate limiting (unless trusted IP)
            if not is_trusted and self.security_config.get('rate_limiting_enabled', True):
                rule_name = self._determine_rate_limit_rule(endpoint)
                allowed, rate_info = self.rate_limiter.is_allowed(source_ip, rule_name)
                
                validation_results['rate_limit_info'] = rate_info
                
                if not allowed:
                    validation_results['allowed'] = False
                    validation_results['reasons'].append('rate_limit_exceeded')
            
            # Threat detection
            if self.security_config.get('threat_detection_enabled', True):
                security_events = self.threat_detector.analyze_request(request_data)
                validation_results['security_events'] = [
                    {
                        'type': event.event_type,
                        'severity': event.severity,
                        'blocked': event.blocked,
                        'details': event.details
                    }
                    for event in security_events
                ]
                
                # Block request if high severity threats detected
                high_severity_events = [
                    e for e in security_events 
                    if e.severity in ['high', 'critical'] and e.blocked
                ]
                
                if high_severity_events:
                    validation_results['allowed'] = False
                    validation_results['reasons'].append('security_threat_detected')
            
            # Input validation and sanitization
            if self.security_config.get('input_validation_enabled', True):
                sanitized_data = self._sanitize_request_data(request_data)
                validation_results['sanitized_data'] = sanitized_data
            
        except Exception as e:
            logger.error(f"Security validation error: {e}")
            validation_results['allowed'] = False
            validation_results['reasons'].append('validation_error')
        
        return validation_results
    
    def _determine_rate_limit_rule(self, endpoint: str) -> str:
        """Determine which rate limiting rule to apply based on endpoint."""
        if '/auth' in endpoint or '/login' in endpoint:
            return 'auth'
        elif '/search' in endpoint:
            return 'search'
        elif '/upload' in endpoint:
            return 'upload'
        else:
            return 'default'
    
    def _sanitize_request_data(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize all string data in request."""
        sanitized = {}
        
        for key, value in request_data.items():
            if isinstance(value, str):
                sanitized[key] = self.input_sanitizer.sanitize_string(value)
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_request_data(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    self.input_sanitizer.sanitize_string(item) if isinstance(item, str) else item
                    for item in value
                ]
            else:
                sanitized[key] = value
        
        return sanitized
    
    def generate_security_headers(self) -> Dict[str, str]:
        """Generate security headers for HTTP responses."""
        if not self.security_config.get('security_headers_enabled', True):
            return {}
        
        return {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
            'Referrer-Policy': 'strict-origin-when-cross-origin',
            'Permissions-Policy': 'camera=(), microphone=(), geolocation=()',
            'X-Correlation-ID': secrets.token_urlsafe(16)
        }
    
    def audit_log_security_event(self, event: SecurityEvent) -> None:
        """Log security event for audit purposes."""
        if not self.security_config.get('audit_logging_enabled', True):
            return
        
        audit_entry = {
            'timestamp': event.timestamp.isoformat(),
            'event_type': event.event_type,
            'source_ip': event.source_ip,
            'endpoint': event.endpoint,
            'severity': event.severity,
            'blocked': event.blocked,
            'details': event.details
        }
        
        # In production, this would write to a secure audit log
        logger.info(f"SECURITY_AUDIT: {json.dumps(audit_entry)}")
    
    def get_security_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive security dashboard data."""
        return {
            'threat_summary': self.threat_detector.get_security_summary(),
            'rate_limit_stats': self.rate_limiter.get_rate_limit_stats(),
            'input_validation_stats': self.input_sanitizer.get_validation_stats(),
            'security_config': self.security_config.copy(),
            'trusted_networks': self.trusted_networks.copy(),
            'last_updated': datetime.now().isoformat()
        }
    
    def save_security_report(self) -> str:
        """Save comprehensive security report."""
        dashboard_data = self.get_security_dashboard_data()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.security_dir / f"security_report_{timestamp}.json"
        
        with open(report_file, "w") as f:
            json.dump(dashboard_data, f, indent=2)
        
        logger.info(f"Security report saved: {report_file}")
        return str(report_file)


def main():
    """Main entry point for security hardening system."""
    logging.basicConfig(level=logging.INFO)
    
    security_manager = SecurityHardeningManager()
    
    # Example usage
    print("🔒 SECURITY HARDENING SYSTEM INITIALIZED")
    
    # Test request validation
    test_request = {
        'ip': '192.168.1.100',
        'user_agent': 'Mozilla/5.0 (compatible; test)',
        'endpoint': '/api/search',
        'payload': '{"query": "test search"}',
        'headers': {}
    }
    
    result = security_manager.validate_and_secure_request(test_request)
    print(f"Request validation result: {result['allowed']}")
    
    # Generate security headers
    headers = security_manager.generate_security_headers()
    print(f"Security headers generated: {len(headers)} headers")
    
    # Save security report
    report_file = security_manager.save_security_report()
    print(f"Security report saved: {report_file}")
    
    print("✅ SECURITY HARDENING DEMONSTRATION COMPLETED")


if __name__ == "__main__":
    main()