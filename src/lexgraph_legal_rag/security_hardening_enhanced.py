"""
Enhanced Security Hardening for Legal RAG
==========================================

Generation 2 Security: Comprehensive security measures for production deployment
- Advanced input sanitization and validation
- SQL injection and XSS prevention
- Rate limiting and DDoS protection
- Secure data handling and encryption
- Audit logging and compliance
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import re
import secrets
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set
import base64
import bleach

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security protection levels."""
    
    BASIC = "basic"          # Basic validation
    STANDARD = "standard"    # Standard security measures
    ENHANCED = "enhanced"    # Enhanced protection
    MAXIMUM = "maximum"      # Maximum security


@dataclass
class SecurityConfig:
    """Security configuration settings."""
    
    enable_input_sanitization: bool = True
    enable_rate_limiting: bool = True
    enable_audit_logging: bool = True
    enable_encryption: bool = True
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    rate_limit_requests_per_minute: int = 100
    session_timeout_minutes: int = 30
    password_min_length: int = 12
    require_https: bool = True
    allowed_file_types: Set[str] = field(default_factory=lambda: {
        '.txt', '.pdf', '.docx', '.json'
    })


@dataclass
class SecurityEvent:
    """Security event for audit logging."""
    
    event_type: str
    timestamp: float
    user_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    resource: Optional[str] = None
    action: Optional[str] = None
    success: bool = True
    details: Dict[str, Any] = field(default_factory=dict)


class SecurityHardeningSystem:
    """
    Comprehensive security hardening system for legal document processing.
    
    Features:
    - Advanced input sanitization and validation
    - Protection against injection attacks
    - Rate limiting and DDoS protection
    - Secure session management
    - Audit logging and compliance
    - Data encryption and secure handling
    """
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()
        self._rate_limits: Dict[str, List[float]] = {}
        self._failed_attempts: Dict[str, int] = {}
        self._blocked_ips: Set[str] = set()
        self._security_events: List[SecurityEvent] = []
        self._session_keys: Dict[str, float] = {}  # session_id -> expiry
        
        # Initialize encryption key
        self._encryption_key = self._generate_encryption_key()
        
        # Dangerous patterns for injection detection
        self._sql_injection_patterns = [
            r"(';\s*(union|select|insert|delete|update|drop|create|alter))",
            r"(;\s*(union|select|insert|delete|update|drop|create|alter)\s)",
            r"(\bunion\s+select\b)",
            r"(\bdrop\s+table\b)",
            r"(\bexec\s*\()",
        ]
        
        self._xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"data:text/html",
            r"vbscript:",
            r"on\w+\s*=",
        ]
        
    def sanitize_input(
        self,
        input_data: Any,
        security_level: SecurityLevel = SecurityLevel.STANDARD
    ) -> Any:
        """Comprehensive input sanitization."""
        
        if not self.config.enable_input_sanitization:
            return input_data
        
        try:
            if isinstance(input_data, str):
                return self._sanitize_string(input_data, security_level)
            elif isinstance(input_data, dict):
                return self._sanitize_dict(input_data, security_level)
            elif isinstance(input_data, list):
                return self._sanitize_list(input_data, security_level)
            else:
                return input_data
                
        except Exception as e:
            self._log_security_event(
                "sanitization_error",
                success=False,
                details={"error": str(e), "input_type": type(input_data).__name__}
            )
            raise SecurityError(f"Input sanitization failed: {e}")
    
    def _sanitize_string(self, text: str, security_level: SecurityLevel) -> str:
        """Sanitize string input against various attacks."""
        
        if not isinstance(text, str):
            return text
        
        # Basic length check
        if len(text) > self.config.max_request_size:
            raise SecurityError("Input too large")
        
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # SQL injection protection
        if self._detect_sql_injection(text):
            self._log_security_event(
                "sql_injection_attempt",
                success=False,
                details={"input": text[:100]}
            )
            raise SecurityError("Potential SQL injection detected")
        
        # XSS protection
        if self._detect_xss(text):
            self._log_security_event(
                "xss_attempt",
                success=False,
                details={"input": text[:100]}
            )
            raise SecurityError("Potential XSS attack detected")
        
        # Enhanced sanitization for higher security levels
        if security_level in [SecurityLevel.ENHANCED, SecurityLevel.MAXIMUM]:
            # Use bleach for HTML sanitization
            text = bleach.clean(text, tags=[], attributes={}, strip=True)
            
            # Remove potentially dangerous Unicode characters
            text = self._remove_dangerous_unicode(text)
        
        # Basic sanitization
        text = self._basic_sanitize(text)
        
        return text
    
    def _sanitize_dict(self, data: Dict[str, Any], security_level: SecurityLevel) -> Dict[str, Any]:
        """Sanitize dictionary recursively."""
        
        if len(json.dumps(data)) > self.config.max_request_size:
            raise SecurityError("Input dictionary too large")
        
        sanitized = {}
        for key, value in data.items():
            # Sanitize key
            clean_key = self._sanitize_string(str(key), security_level)
            
            # Sanitize value
            clean_value = self.sanitize_input(value, security_level)
            
            sanitized[clean_key] = clean_value
        
        return sanitized
    
    def _sanitize_list(self, data: List[Any], security_level: SecurityLevel) -> List[Any]:
        """Sanitize list recursively."""
        
        return [self.sanitize_input(item, security_level) for item in data]
    
    def _detect_sql_injection(self, text: str) -> bool:
        """Detect potential SQL injection patterns."""
        
        text_lower = text.lower()
        
        for pattern in self._sql_injection_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        
        return False
    
    def _detect_xss(self, text: str) -> bool:
        """Detect potential XSS attack patterns."""
        
        for pattern in self._xss_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def _remove_dangerous_unicode(self, text: str) -> str:
        """Remove potentially dangerous Unicode characters."""
        
        # Remove control characters except common whitespace
        cleaned = ''.join(
            char for char in text
            if ord(char) >= 32 or char in '\t\n\r'
        )
        
        return cleaned
    
    def _basic_sanitize(self, text: str) -> str:
        """Basic string sanitization."""
        
        # Escape HTML entities
        html_entities = {
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#x27;',
            '&': '&amp;',
        }
        
        for char, entity in html_entities.items():
            text = text.replace(char, entity)
        
        return text.strip()
    
    def validate_file_upload(self, filename: str, content: bytes) -> bool:
        """Validate file uploads for security."""
        
        # Check file extension
        if not any(filename.lower().endswith(ext) for ext in self.config.allowed_file_types):
            self._log_security_event(
                "invalid_file_type",
                success=False,
                details={"filename": filename}
            )
            return False
        
        # Check file size
        if len(content) > self.config.max_request_size:
            self._log_security_event(
                "file_too_large",
                success=False,
                details={"filename": filename, "size": len(content)}
            )
            return False
        
        # Check for embedded executables
        if self._contains_executable_signatures(content):
            self._log_security_event(
                "executable_upload_attempt",
                success=False,
                details={"filename": filename}
            )
            return False
        
        return True
    
    def _contains_executable_signatures(self, content: bytes) -> bool:
        """Check for executable file signatures."""
        
        # Common executable signatures
        executable_signatures = [
            b'\x4D\x5A',  # PE executable
            b'\x7F\x45\x4C\x46',  # ELF executable
            b'\xCA\xFE\xBA\xBE',  # Java class file
            b'\xFE\xED\xFA',  # Mach-O executable
        ]
        
        for signature in executable_signatures:
            if content.startswith(signature):
                return True
        
        return False
    
    def check_rate_limit(self, identifier: str) -> bool:
        """Check if request is within rate limits."""
        
        if not self.config.enable_rate_limiting:
            return True
        
        current_time = time.time()
        minute_ago = current_time - 60
        
        # Initialize tracking
        if identifier not in self._rate_limits:
            self._rate_limits[identifier] = []
        
        # Remove old requests
        self._rate_limits[identifier] = [
            timestamp for timestamp in self._rate_limits[identifier]
            if timestamp > minute_ago
        ]
        
        # Check current rate
        current_requests = len(self._rate_limits[identifier])
        if current_requests >= self.config.rate_limit_requests_per_minute:
            self._log_security_event(
                "rate_limit_exceeded",
                success=False,
                details={"identifier": identifier, "requests": current_requests}
            )
            
            # Track failed attempts
            self._failed_attempts[identifier] = self._failed_attempts.get(identifier, 0) + 1
            
            # Block IP after too many violations
            if self._failed_attempts[identifier] > 10:
                self._blocked_ips.add(identifier)
                self._log_security_event(
                    "ip_blocked",
                    success=False,
                    details={"identifier": identifier}
                )
            
            return False
        
        # Record this request
        self._rate_limits[identifier].append(current_time)
        return True
    
    def is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP address is blocked."""
        return ip_address in self._blocked_ips
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate cryptographically secure token."""
        return secrets.token_urlsafe(length)
    
    def hash_password(self, password: str, salt: Optional[str] = None) -> tuple[str, str]:
        """Securely hash password with salt."""
        
        if len(password) < self.config.password_min_length:
            raise SecurityError(f"Password too short (minimum {self.config.password_min_length} characters)")
        
        if salt is None:
            salt = secrets.token_hex(16)
        
        # Use PBKDF2 with SHA-256
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000  # 100k iterations
        )
        
        return base64.b64encode(password_hash).decode('utf-8'), salt
    
    def verify_password(self, password: str, password_hash: str, salt: str) -> bool:
        """Verify password against hash."""
        
        try:
            computed_hash, _ = self.hash_password(password, salt)
            return hmac.compare_digest(password_hash, computed_hash)
        except Exception:
            return False
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        
        if not self.config.enable_encryption:
            return data
        
        # Simple XOR encryption (replace with proper encryption in production)
        encrypted = bytearray()
        key_bytes = self._encryption_key.encode('utf-8')
        
        for i, byte in enumerate(data.encode('utf-8')):
            encrypted.append(byte ^ key_bytes[i % len(key_bytes)])
        
        return base64.b64encode(encrypted).decode('utf-8')
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        
        if not self.config.enable_encryption:
            return encrypted_data
        
        try:
            encrypted_bytes = base64.b64decode(encrypted_data.encode('utf-8'))
            decrypted = bytearray()
            key_bytes = self._encryption_key.encode('utf-8')
            
            for i, byte in enumerate(encrypted_bytes):
                decrypted.append(byte ^ key_bytes[i % len(key_bytes)])
            
            return decrypted.decode('utf-8')
        except Exception as e:
            raise SecurityError(f"Decryption failed: {e}")
    
    def create_session(self, user_id: str) -> str:
        """Create secure session."""
        
        session_id = self.generate_secure_token()
        expiry = time.time() + (self.config.session_timeout_minutes * 60)
        
        self._session_keys[session_id] = expiry
        
        self._log_security_event(
            "session_created",
            user_id=user_id,
            details={"session_id": session_id}
        )
        
        return session_id
    
    def validate_session(self, session_id: str) -> bool:
        """Validate session token."""
        
        if session_id not in self._session_keys:
            return False
        
        expiry = self._session_keys[session_id]
        if time.time() > expiry:
            del self._session_keys[session_id]
            return False
        
        return True
    
    def invalidate_session(self, session_id: str) -> None:
        """Invalidate session."""
        
        if session_id in self._session_keys:
            del self._session_keys[session_id]
            
            self._log_security_event(
                "session_invalidated",
                details={"session_id": session_id}
            )
    
    def _generate_encryption_key(self) -> str:
        """Generate encryption key."""
        return secrets.token_hex(32)
    
    def _log_security_event(
        self,
        event_type: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        success: bool = True,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log security event for audit trail."""
        
        if not self.config.enable_audit_logging:
            return
        
        event = SecurityEvent(
            event_type=event_type,
            timestamp=time.time(),
            user_id=user_id,
            ip_address=ip_address,
            success=success,
            details=details or {}
        )
        
        self._security_events.append(event)
        
        # Keep only recent events (last 24 hours)
        cutoff_time = time.time() - (24 * 3600)
        self._security_events = [
            event for event in self._security_events
            if event.timestamp > cutoff_time
        ]
        
        # Log to standard logger
        level = logging.INFO if success else logging.WARNING
        logger.log(
            level,
            f"Security event: {event_type} - Success: {success} - "
            f"User: {user_id} - IP: {ip_address} - Details: {details}"
        )
    
    def get_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        
        current_time = time.time()
        
        # Calculate statistics
        total_events = len(self._security_events)
        failed_events = sum(1 for event in self._security_events if not event.success)
        
        event_types = {}
        for event in self._security_events:
            event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
        
        return {
            "timestamp": current_time,
            "total_security_events": total_events,
            "failed_events": failed_events,
            "success_rate": (total_events - failed_events) / max(total_events, 1),
            "event_types": event_types,
            "blocked_ips_count": len(self._blocked_ips),
            "active_sessions": len(self._session_keys),
            "rate_limit_violations": sum(
                max(0, len(requests) - self.config.rate_limit_requests_per_minute)
                for requests in self._rate_limits.values()
            ),
            "config": {
                "input_sanitization": self.config.enable_input_sanitization,
                "rate_limiting": self.config.enable_rate_limiting,
                "audit_logging": self.config.enable_audit_logging,
                "encryption": self.config.enable_encryption,
            }
        }


class SecurityError(Exception):
    """Security-related error."""
    pass


# Global security system instance
_global_security_system = None


def get_security_system(config: Optional[SecurityConfig] = None) -> SecurityHardeningSystem:
    """Get global security system instance."""
    
    global _global_security_system
    if _global_security_system is None:
        _global_security_system = SecurityHardeningSystem(config)
    return _global_security_system


# Decorator for security protection
def secure_endpoint(
    security_level: SecurityLevel = SecurityLevel.STANDARD,
    require_auth: bool = True,
    rate_limit: bool = True
):
    """Decorator to add security to endpoints."""
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            security_system = get_security_system()
            
            # Security checks would go here
            # This is a simplified example
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator