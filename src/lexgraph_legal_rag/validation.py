"""Input validation and sanitization for LexGraph Legal RAG system."""

from __future__ import annotations

import re
import logging
from typing import Any, Optional, Dict, List
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
                issues.append(f"Potentially dangerous pattern detected: XSS/HTML")
                break
        
        # Check for SQL injection
        for pattern in self.SQL_INJECTION_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                issues.append(f"Potentially dangerous pattern detected: SQL injection")
                break
        
        # Check for command injection
        if self.security_level in [SecurityLevel.STRICT, SecurityLevel.PARANOID]:
            for pattern in self.COMMAND_INJECTION_PATTERNS:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    issues.append(f"Potentially dangerous pattern detected: Command injection")
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
        
        # Check for minimum meaningful content
        words = query.split()
        if len(words) < 2:
            warnings.append("Query appears to have minimal content")
        
        # Check for excessive repetition
        if len(set(words)) < len(words) * 0.5 and len(words) > 10:
            warnings.append("Query contains excessive repetition")
        
        # Check for legal terminology appropriateness
        legal_terms = [
            'contract', 'agreement', 'clause', 'liability', 'breach', 'damages',
            'warranty', 'indemnify', 'jurisdiction', 'legal', 'law', 'statute',
            'regulation', 'court', 'case', 'precedent', 'ruling', 'judgment'
        ]
        
        query_lower = query.lower()
        has_legal_terms = any(term in query_lower for term in legal_terms)
        
        if not has_legal_terms and len(words) > 5:
            warnings.append("Query may not be legal domain related")
        
        return warnings
    
    def _final_sanitize(self, query: str) -> str:
        """Final sanitization pass."""
        # Normalize whitespace
        query = re.sub(r'\s+', ' ', query)
        
        # Remove leading/trailing quotes if they're unmatched
        if query.startswith('"') and not query.endswith('"'):
            query = query[1:]
        elif query.endswith('"') and not query.startswith('"'):
            query = query[:-1]
        
        return query.strip()


class ParameterValidator:
    """Validates API parameters."""
    
    @staticmethod
    def validate_top_k(top_k: Any) -> ValidationResult:
        """Validate top_k parameter."""
        if not isinstance(top_k, int):
            try:
                top_k = int(top_k)
            except (ValueError, TypeError):
                return ValidationResult(
                    is_valid=False,
                    sanitized_input=5,
                    errors=["top_k must be an integer"]
                )
        
        if top_k < 1:
            return ValidationResult(
                is_valid=False,
                sanitized_input=1,
                errors=["top_k must be at least 1"]
            )
        
        if top_k > 100:
            return ValidationResult(
                is_valid=True,
                sanitized_input=100,
                warnings=["top_k capped at 100 for performance"]
            )
        
        return ValidationResult(is_valid=True, sanitized_input=top_k)
    
    @staticmethod
    def validate_hops(hops: Any) -> ValidationResult:
        """Validate hops parameter for context reasoning."""
        if not isinstance(hops, int):
            try:
                hops = int(hops)
            except (ValueError, TypeError):
                return ValidationResult(
                    is_valid=False,
                    sanitized_input=3,
                    errors=["hops must be an integer"]
                )
        
        if hops < 1:
            return ValidationResult(
                is_valid=False,
                sanitized_input=1,
                errors=["hops must be at least 1"]
            )
        
        if hops > 10:
            return ValidationResult(
                is_valid=True,
                sanitized_input=10,
                warnings=["hops capped at 10 for performance"]
            )
        
        return ValidationResult(is_valid=True, sanitized_input=hops)
    
    @staticmethod
    def validate_chunk_size(chunk_size: Any) -> ValidationResult:
        """Validate chunk_size parameter."""
        if not isinstance(chunk_size, int):
            try:
                chunk_size = int(chunk_size)
            except (ValueError, TypeError):
                return ValidationResult(
                    is_valid=False,
                    sanitized_input=512,
                    errors=["chunk_size must be an integer"]
                )
        
        if chunk_size < 100:
            return ValidationResult(
                is_valid=False,
                sanitized_input=100,
                errors=["chunk_size must be at least 100"]
            )
        
        if chunk_size > 5000:
            return ValidationResult(
                is_valid=True,
                sanitized_input=5000,
                warnings=["chunk_size capped at 5000 for performance"]
            )
        
        return ValidationResult(is_valid=True, sanitized_input=chunk_size)


def validate_api_input(
    query: Optional[str] = None,
    top_k: Optional[int] = None,
    hops: Optional[int] = None,
    chunk_size: Optional[int] = None,
    security_level: SecurityLevel = SecurityLevel.STRICT
) -> Dict[str, ValidationResult]:
    """Validate all API inputs comprehensively."""
    results = {}
    
    if query is not None:
        validator = QueryValidator(security_level)
        results['query'] = validator.validate_query(query)
    
    if top_k is not None:
        results['top_k'] = ParameterValidator.validate_top_k(top_k)
    
    if hops is not None:
        results['hops'] = ParameterValidator.validate_hops(hops)
    
    if chunk_size is not None:
        results['chunk_size'] = ParameterValidator.validate_chunk_size(chunk_size)
    
    return results


def log_validation_results(results: Dict[str, ValidationResult]) -> None:
    """Log validation results for monitoring."""
    for param_name, result in results.items():
        if not result.is_valid:
            logger.warning(
                f"Validation failed for {param_name}: {result.errors}",
                extra={
                    "parameter": param_name,
                    "validation_errors": result.errors,
                    "validation_warnings": result.warnings
                }
            )
        elif result.warnings:
            logger.info(
                f"Validation warnings for {param_name}: {result.warnings}",
                extra={
                    "parameter": param_name,
                    "validation_warnings": result.warnings
                }
            )
        else:
            logger.debug(f"Validation passed for {param_name}")