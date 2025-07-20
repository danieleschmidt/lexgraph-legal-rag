"""Custom exception hierarchy for LexGraph Legal RAG system."""

from typing import Any, Dict, Optional, Type, TypeVar


T = TypeVar('T', bound='LexGraphError')


class LexGraphError(Exception):
    """Base exception for all LexGraph Legal RAG errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.error_code = error_code
        self.details = details
        self.context = context or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error": self.__class__.__name__,
            "message": str(self),
            "error_code": self.error_code,
            "details": self.details,
            "context": self.context
        }
    
    @classmethod
    def from_dict(cls: Type[T], error_dict: Dict[str, Any]) -> T:
        """Create exception from dictionary."""
        error_type = error_dict.get("error", "LexGraphError")
        message = error_dict.get("message", "Unknown error")
        error_code = error_dict.get("error_code")
        details = error_dict.get("details")
        context = error_dict.get("context", {})
        
        # Map error type to appropriate class
        error_classes = {
            "ConfigurationError": ConfigurationError,
            "AuthenticationError": AuthenticationError,
            "AuthorizationError": AuthorizationError,
            "APIKeyError": APIKeyError,
            "RateLimitError": RateLimitError,
            "DocumentError": DocumentError,
            "DocumentNotFoundError": DocumentNotFoundError,
            "DocumentParsingError": DocumentParsingError,
            "IndexError": IndexError,
            "IndexCorruptedError": IndexCorruptedError,
            "IndexNotFoundError": IndexNotFoundError,
            "SearchError": SearchError,
            "QueryError": QueryError,
            "ValidationError": ValidationError,
            "ExternalServiceError": ExternalServiceError,
        }
        
        error_class = error_classes.get(error_type, LexGraphError)
        return error_class(message, error_code, details, context)


class ConfigurationError(LexGraphError):
    """Raised when there are configuration-related errors."""
    pass


class AuthenticationError(LexGraphError):
    """Raised when authentication fails."""
    pass


class AuthorizationError(LexGraphError):
    """Raised when authorization fails."""
    pass


class APIKeyError(AuthenticationError):
    """Raised when API key is invalid or missing."""
    pass


class RateLimitError(AuthorizationError):
    """Raised when rate limits are exceeded."""
    pass


class DocumentError(LexGraphError):
    """Base exception for document-related errors."""
    pass


class DocumentNotFoundError(DocumentError):
    """Raised when a document cannot be found."""
    pass


class DocumentParsingError(DocumentError):
    """Raised when document parsing fails."""
    pass


class IndexError(LexGraphError):
    """Base exception for index-related errors."""
    pass


class IndexCorruptedError(IndexError):
    """Raised when index corruption is detected."""
    pass


class IndexNotFoundError(IndexError):
    """Raised when an index cannot be found."""
    pass


class SearchError(LexGraphError):
    """Base exception for search-related errors."""
    pass


class QueryError(SearchError):
    """Raised when query parsing or execution fails."""
    pass


class ValidationError(LexGraphError):
    """Raised when input validation fails."""
    pass


class ExternalServiceError(LexGraphError):
    """Raised when external service calls fail."""
    pass