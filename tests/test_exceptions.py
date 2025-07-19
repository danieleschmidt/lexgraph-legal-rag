"""Tests for custom exception hierarchy."""

import pytest
from lexgraph_legal_rag.exceptions import (
    LexGraphError,
    ConfigurationError,
    AuthenticationError,
    AuthorizationError,
    DocumentError,
    DocumentNotFoundError,
    DocumentParsingError,
    IndexError,
    IndexCorruptedError,
    IndexNotFoundError,
    SearchError,
    QueryError,
    ValidationError,
    ExternalServiceError,
    APIKeyError,
    RateLimitError,
)


class TestExceptionHierarchy:
    """Test the exception hierarchy structure."""
    
    def test_base_exception_hierarchy(self):
        """Test that all custom exceptions inherit from LexGraphError."""
        # Configuration exceptions
        assert issubclass(ConfigurationError, LexGraphError)
        
        # Authentication/Authorization exceptions
        assert issubclass(AuthenticationError, LexGraphError)
        assert issubclass(AuthorizationError, LexGraphError)
        assert issubclass(APIKeyError, AuthenticationError)
        assert issubclass(RateLimitError, AuthorizationError)
        
        # Document exceptions
        assert issubclass(DocumentError, LexGraphError)
        assert issubclass(DocumentNotFoundError, DocumentError)
        assert issubclass(DocumentParsingError, DocumentError)
        
        # Index exceptions
        assert issubclass(IndexError, LexGraphError)
        assert issubclass(IndexCorruptedError, IndexError)
        assert issubclass(IndexNotFoundError, IndexError)
        
        # Search exceptions
        assert issubclass(SearchError, LexGraphError)
        assert issubclass(QueryError, SearchError)
        
        # Validation exceptions
        assert issubclass(ValidationError, LexGraphError)
        
        # External service exceptions
        assert issubclass(ExternalServiceError, LexGraphError)
    
    def test_base_exception_creation(self):
        """Test creating base exception with message and details."""
        error = LexGraphError("Test error", error_code="TEST_001")
        
        assert str(error) == "Test error"
        assert error.error_code == "TEST_001"
        assert error.details is None
        assert error.context == {}
    
    def test_exception_with_details(self):
        """Test exception with additional details."""
        details = {"field": "value", "count": 42}
        context = {"request_id": "req-123", "user_id": "user-456"}
        
        error = DocumentNotFoundError(
            "Document not found",
            error_code="DOC_404",
            details=details,
            context=context
        )
        
        assert str(error) == "Document not found"
        assert error.error_code == "DOC_404"
        assert error.details == details
        assert error.context == context
    
    def test_configuration_error(self):
        """Test configuration-specific error."""
        error = ConfigurationError("Invalid API key", error_code="CFG_001")
        
        assert isinstance(error, LexGraphError)
        assert error.error_code == "CFG_001"
    
    def test_authentication_error(self):
        """Test authentication-specific error."""
        error = AuthenticationError("Invalid credentials", error_code="AUTH_001")
        
        assert isinstance(error, LexGraphError)
        assert error.error_code == "AUTH_001"
    
    def test_api_key_error(self):
        """Test API key specific error."""
        error = APIKeyError("API key expired", error_code="KEY_002")
        
        assert isinstance(error, AuthenticationError)
        assert isinstance(error, LexGraphError)
        assert error.error_code == "KEY_002"
    
    def test_document_error(self):
        """Test document-specific error."""
        error = DocumentNotFoundError(
            "Document doc-123 not found",
            error_code="DOC_404",
            details={"document_id": "doc-123"}
        )
        
        assert isinstance(error, DocumentError)
        assert isinstance(error, LexGraphError)
        assert error.details["document_id"] == "doc-123"
    
    def test_index_error(self):
        """Test index-specific error."""
        error = IndexCorruptedError(
            "Index corruption detected",
            error_code="IDX_500",
            details={"index_path": "/path/to/index", "corruption_type": "checksum_mismatch"}
        )
        
        assert isinstance(error, IndexError)
        assert isinstance(error, LexGraphError)
        assert error.details["corruption_type"] == "checksum_mismatch"
    
    def test_search_error(self):
        """Test search-specific error."""
        error = QueryError(
            "Invalid query syntax",
            error_code="QRY_400",
            details={"query": "invalid[syntax", "position": 8}
        )
        
        assert isinstance(error, SearchError)
        assert isinstance(error, LexGraphError)
        assert error.details["position"] == 8
    
    def test_external_service_error(self):
        """Test external service error."""
        error = ExternalServiceError(
            "OpenAI API unavailable",
            error_code="EXT_503",
            details={"service": "openai", "status_code": 503}
        )
        
        assert isinstance(error, LexGraphError)
        assert error.details["service"] == "openai"
    
    def test_rate_limit_error(self):
        """Test rate limit specific error."""
        error = RateLimitError(
            "Rate limit exceeded",
            error_code="RATE_429",
            details={"limit": 100, "reset_time": "2024-01-01T00:00:00Z"}
        )
        
        assert isinstance(error, AuthorizationError)
        assert isinstance(error, LexGraphError)
        assert error.details["limit"] == 100


class TestExceptionUtilities:
    """Test exception utility functions."""
    
    def test_error_to_dict(self):
        """Test converting exception to dictionary for API responses."""
        error = DocumentNotFoundError(
            "Document not found",
            error_code="DOC_404",
            details={"document_id": "doc-123"},
            context={"request_id": "req-456"}
        )
        
        error_dict = error.to_dict()
        
        expected = {
            "error": "DocumentNotFoundError",
            "message": "Document not found",
            "error_code": "DOC_404",
            "details": {"document_id": "doc-123"},
            "context": {"request_id": "req-456"}
        }
        
        assert error_dict == expected
    
    def test_error_to_dict_minimal(self):
        """Test converting minimal exception to dictionary."""
        error = LexGraphError("Simple error")
        
        error_dict = error.to_dict()
        
        expected = {
            "error": "LexGraphError",
            "message": "Simple error",
            "error_code": None,
            "details": None,
            "context": {}
        }
        
        assert error_dict == expected
    
    def test_error_from_dict(self):
        """Test creating exception from dictionary."""
        error_dict = {
            "error": "DocumentNotFoundError",
            "message": "Document not found",
            "error_code": "DOC_404",
            "details": {"document_id": "doc-123"},
            "context": {"request_id": "req-456"}
        }
        
        error = LexGraphError.from_dict(error_dict)
        
        assert isinstance(error, DocumentNotFoundError)
        assert str(error) == "Document not found"
        assert error.error_code == "DOC_404"
        assert error.details == {"document_id": "doc-123"}
        assert error.context == {"request_id": "req-456"}
    
    def test_error_chaining(self):
        """Test exception chaining for root cause analysis."""
        try:
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise DocumentParsingError(
                    "Failed to parse document",
                    error_code="DOC_PARSE_001"
                ) from e
        except DocumentParsingError as error:
            assert error.__cause__ is not None
            assert isinstance(error.__cause__, ValueError)
            assert str(error.__cause__) == "Original error"


class TestExceptionIntegration:
    """Test integration with existing error handling."""
    
    def test_fastapi_exception_handler_format(self):
        """Test that exceptions format correctly for FastAPI."""
        error = ValidationError(
            "Invalid input data",
            error_code="VAL_400",
            details={"field": "email", "value": "invalid-email"}
        )
        
        # Should be serializable to JSON
        import json
        error_dict = error.to_dict()
        json_str = json.dumps(error_dict)
        
        # Should round-trip correctly
        parsed = json.loads(json_str)
        assert parsed["message"] == "Invalid input data"
        assert parsed["details"]["field"] == "email"
    
    def test_logging_integration(self):
        """Test that exceptions work well with structured logging."""
        error = IndexCorruptedError(
            "Index corruption detected",
            error_code="IDX_500",
            details={"path": "/data/index.bin", "size": 1024},
            context={"operation": "load_index", "timestamp": "2024-01-01T00:00:00Z"}
        )
        
        # Should have all necessary fields for structured logging
        log_data = {
            "level": "ERROR",
            "message": str(error),
            "error_code": error.error_code,
            "error_type": error.__class__.__name__,
            "details": error.details,
            "context": error.context
        }
        
        assert log_data["error_code"] == "IDX_500"
        assert log_data["error_type"] == "IndexCorruptedError"
        assert log_data["details"]["path"] == "/data/index.bin"