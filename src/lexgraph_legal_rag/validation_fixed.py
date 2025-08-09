"""Input validation and sanitization for LexGraph Legal RAG system."""

from __future__ import annotations

import re
import logging
from typing import Any, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when input validation fails."""
    pass


class SecurityLevel(Enum):
    """Security validation levels."""
    BASIC = "basic"
    STRICT = "strict"
    PARANOID = "paranoid"


@dataclass
class ValidationResult:
    """Result of input validation."""
    is_valid: bool
    sanitized_input: Any
    warnings: List[str] = None
    errors: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.errors is None:
            self.errors = []


class QueryValidator:
    """Validates and sanitizes legal query inputs."""
    
    # Potentially dangerous patterns that should be rejected
    DANGEROUS_PATTERNS = [
        r'<script[^>]*>.*?</script>',  # Script tags
        r'javascript:',                # JavaScript protocols
        r'data:',                     # Data URLs
        r'vbscript:',                 # VBScript
        r'on\w+\s*=',                 # Event handlers
        r'<iframe[^>]*>',             # Iframes
        r'<object[^>]*>',             # Objects
        r'<embed[^>]*>',              # Embeds
        r'<link[^>]*>',               # Links
        r'<meta[^>]*>',               # Meta tags
        r'<!--.*?-->',                # HTML comments
        r'expression\s*\(',           # CSS expressions
        r'url\s*\(',                  # CSS url()
        r'@import',                   # CSS imports
    ]
    
    # SQL injection patterns
    SQL_INJECTION_PATTERNS = [
        r'(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)',
        r'(--|#|/\*|\*/)',  # SQL comments
        r'(\bOR\b|\bAND\b).*?=.*?=',  # Boolean-based injection
        r';\s*(DROP|DELETE|INSERT|UPDATE)',  # Command injection
        r'\bUNION\b.*?\bSELECT\b',  # Union-based injection
    ]
    
    # Command injection patterns
    COMMAND_INJECTION_PATTERNS = [
        r'[;&|`$(){}<>]',  # Shell metacharacters
        r'\b(cat|ls|pwd|id|whoami|uname|ps|netstat|ifconfig)\b',  # Common commands
        r'\.\./',  # Directory traversal
        r'/etc/passwd',  # System files
        r'/proc/',  # Process information
    ]
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.STRICT):
        self.security_level = security_level
        self.max_query_length = self._get_max_length()
        self.min_query_length = 3
    
    def _get_max_length(self) -> int:
        """Get maximum query length based on security level."""
        if self.security_level == SecurityLevel.BASIC:
            return 10000
        elif self.security_level == SecurityLevel.STRICT:
            return 5000
        else:  # PARANOID
            return 2000
    
    def validate_query(self, query: str) -> ValidationResult:
        """Validate and sanitize a legal query."""
        if not isinstance(query, str):
            return ValidationResult(
                is_valid=False,
                sanitized_input="",
                errors=["Query must be a string"]
            )
        
        warnings = []
        errors = []
        sanitized = query.strip()
        
        # Length validation
        if len(sanitized) < self.min_query_length:
            errors.append(f"Query too short (minimum {self.min_query_length} characters)")
        
        if len(sanitized) > self.max_query_length:
            errors.append(f"Query too long (maximum {self.max_query_length} characters)")
            # Truncate if not too excessive
            if len(sanitized) < self.max_query_length * 2:
                sanitized = sanitized[:self.max_query_length]
                warnings.append("Query was truncated")
            else:
                return ValidationResult(
                    is_valid=False,
                    sanitized_input="",
                    errors=errors + ["Query excessively long"]
                )
        
        # Security validation
        security_issues = self._check_security_patterns(sanitized)
        if security_issues:
            if self.security_level == SecurityLevel.PARANOID:
                errors.extend(security_issues)
            else:
                warnings.extend(security_issues)
                # Attempt sanitization
                sanitized = self._sanitize_query(sanitized)
        
        # Content validation
        content_issues = self._validate_content(sanitized)
        warnings.extend(content_issues)
        
        # Final sanitization
        sanitized = self._final_sanitize(sanitized)
        
        is_valid = len(errors) == 0 and len(sanitized.strip()) >= self.min_query_length
        
        return ValidationResult(
            is_valid=is_valid,
            sanitized_input=sanitized,
            warnings=warnings,
            errors=errors
        )
    
    def _check_security_patterns(self, text: str) -> List[str]:
        """Check for potentially dangerous patterns."""
        issues = []
        text_lower = text.lower()
        
        # Check for XSS patterns
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE | re.DOTALL):
                issues.append("Potentially dangerous pattern detected: XSS/HTML")
                break
        
        # Check for SQL injection
        for pattern in self.SQL_INJECTION_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                issues.append("Potentially dangerous pattern detected: SQL injection")
                break
        
        # Check for command injection
        if self.security_level in [SecurityLevel.STRICT, SecurityLevel.PARANOID]:
            for pattern in self.COMMAND_INJECTION_PATTERNS:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    issues.append("Potentially dangerous pattern detected: Command injection")
                    break
        
        return issues
    
    def _sanitize_query(self, query: str) -> str:
        """Sanitize query by removing dangerous patterns."""
        sanitized = query
        
        # Remove HTML/XML tags
        sanitized = re.sub(r'<[^>]*>', '', sanitized)
        
        # Remove script content
        sanitized = re.sub(r'<script[^>]*>.*?</script>', '', sanitized, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove potentially dangerous protocols
        sanitized = re.sub(r'javascript:', '', sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r'vbscript:', '', sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r'data:', '', sanitized, flags=re.IGNORECASE)
        
        # Remove SQL injection patterns (basic)
        sanitized = re.sub(r'(--|#|/\*|\*/)', '', sanitized)
        
        # Remove excessive whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized)
        
        return sanitized.strip()
    
    def _validate_content(self, query: str) -> List[str]:
        """Validate query content for legal domain appropriateness."""
        warnings = []
        
        # Check for common non-legal terms that might indicate misuse
        non_legal_indicators = [
            r'\b(password|login|admin|root)\b',
            r'\b(download|upload|file|directory)\b',
            r'\b(database|table|select|insert)\b',
        ]
        
        for pattern in non_legal_indicators:
            if re.search(pattern, query, re.IGNORECASE):
                warnings.append("Query contains terms not typically associated with legal searches")
                break
        
        return warnings
    
    def _final_sanitize(self, query: str) -> str:
        """Final sanitization pass."""
        # Ensure single spaces and proper trimming
        sanitized = re.sub(r'\s+', ' ', query.strip())
        
        # Remove any remaining potentially problematic characters
        sanitized = re.sub(r'[<>]', '', sanitized)
        
        return sanitized


def validate_query_input(query: str, security_level: SecurityLevel = SecurityLevel.STRICT) -> ValidationResult:
    """Validate query input with specified security level."""
    validator = QueryValidator(security_level)
    return validator.validate_query(query)


def validate_document_content(content: str, source_path: str = "") -> ValidationResult:
    """Validate document content for ingestion."""
    warnings = []
    errors = []
    
    # Basic content checks
    if not isinstance(content, str):
        return ValidationResult(
            is_valid=False,
            sanitized_input="",
            errors=["Content must be a string"]
        )
    
    content = content.strip()
    
    if len(content) < 10:
        errors.append(f"Document content too short: {len(content)} characters")
    
    if len(content) > 1_000_000:  # 1MB limit
        warnings.append(f"Large document: {len(content)} characters")
    
    # Check for valid text encoding
    try:
        content.encode('utf-8')
    except UnicodeEncodeError as e:
        errors.append(f"Invalid UTF-8 encoding: {e}")
    
    # Check for suspicious content patterns
    suspicious_patterns = [
        (r'<script', "Contains script tags"),
        (r'javascript:', "Contains JavaScript"),
        (r'data:image/', "Contains data URLs"),
        (r'<iframe', "Contains iframes"),
    ]
    
    for pattern, message in suspicious_patterns:
        if re.search(pattern, content, re.IGNORECASE):
            warnings.append(f"{message} in {source_path}")
    
    # Legal document content validation
    legal_indicators = [
        r'\b(contract|agreement|terms|conditions)\b',
        r'\b(liability|damages|indemnif|warrant)\b', 
        r'\b(section|clause|article|provision)\b',
        r'\b(shall|hereby|whereas|therefore)\b'
    ]
    
    has_legal_content = any(
        re.search(pattern, content, re.IGNORECASE) 
        for pattern in legal_indicators
    )
    
    if not has_legal_content and len(content) > 100:
        warnings.append(f"Document may not contain legal content: {source_path}")
    
    is_valid = len(errors) == 0
    
    return ValidationResult(
        is_valid=is_valid,
        sanitized_input=content,
        warnings=warnings,
        errors=errors
    )