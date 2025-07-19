"""Test correlation ID functionality for request tracing."""

import pytest
import uuid
from unittest.mock import patch, MagicMock
from fastapi import FastAPI, Request, Response
from fastapi.testclient import TestClient

from lexgraph_legal_rag.correlation import (
    CorrelationIdMiddleware,
    get_correlation_id,
    set_correlation_id,
    correlation_context,
    CorrelationIdProcessor,
    CORRELATION_ID_HEADER,
)


class TestCorrelationIdMiddleware:
    """Test correlation ID middleware functionality."""
    
    def test_generates_correlation_id_when_missing(self):
        """Test that middleware generates correlation ID when not provided."""
        app = FastAPI()
        app.add_middleware(CorrelationIdMiddleware)
        
        @app.get("/test")
        async def test_endpoint():
            correlation_id = get_correlation_id()
            return {"correlation_id": correlation_id}
        
        client = TestClient(app)
        response = client.get("/test")
        
        assert response.status_code == 200
        data = response.json()
        # Should have generated a correlation ID
        assert data["correlation_id"] is not None
        assert len(data["correlation_id"]) == 36  # UUID format
        # Response should include correlation ID header
        assert CORRELATION_ID_HEADER in response.headers
        assert response.headers[CORRELATION_ID_HEADER] == data["correlation_id"]
    
    def test_uses_provided_correlation_id(self):
        """Test that middleware uses correlation ID from request header."""
        app = FastAPI()
        app.add_middleware(CorrelationIdMiddleware)
        
        @app.get("/test")
        async def test_endpoint():
            correlation_id = get_correlation_id()
            return {"correlation_id": correlation_id}
        
        client = TestClient(app)
        test_id = 'test-correlation-id-123'
        response = client.get("/test", headers={CORRELATION_ID_HEADER: test_id})
        
        assert response.status_code == 200
        data = response.json()
        # Should use provided correlation ID
        assert data["correlation_id"] == test_id
        # Response should include same correlation ID header
        assert response.headers[CORRELATION_ID_HEADER] == test_id
    
    def test_adds_correlation_id_to_response_headers(self):
        """Test that middleware adds correlation ID to response headers."""
        app = FastAPI()
        app.add_middleware(CorrelationIdMiddleware)
        
        @app.get("/test")
        async def test_endpoint():
            return {"message": "test"}
        
        client = TestClient(app)
        response = client.get("/test")
        
        assert response.status_code == 200
        # Response should always include correlation ID header
        assert CORRELATION_ID_HEADER in response.headers
        correlation_id = response.headers[CORRELATION_ID_HEADER]
        assert correlation_id is not None
        assert len(correlation_id) == 36  # UUID format


class TestCorrelationIdContext:
    """Test correlation ID context management."""
    
    def setup_method(self):
        """Clear correlation ID context before each test."""
        # Reset any existing correlation ID
        from lexgraph_legal_rag.correlation import _correlation_id
        _correlation_id.set(None)
    
    def test_get_correlation_id_returns_none_when_not_set(self):
        """Test get_correlation_id returns None when no ID is set."""
        # No correlation ID set in context
        correlation_id = get_correlation_id()
        assert correlation_id is None
    
    def test_set_and_get_correlation_id(self):
        """Test setting and getting correlation ID."""
        test_id = "test-correlation-id-456"
        set_correlation_id(test_id)
        
        correlation_id = get_correlation_id()
        assert correlation_id == test_id
    
    def test_correlation_context_manager(self):
        """Test correlation ID context manager."""
        test_id = "context-test-id-789"
        
        # Before context
        assert get_correlation_id() is None
        
        # Inside context
        with correlation_context(test_id):
            assert get_correlation_id() == test_id
        
        # After context - should be reset
        assert get_correlation_id() is None
    
    def test_nested_correlation_contexts(self):
        """Test nested correlation ID contexts."""
        outer_id = "outer-context-id"
        inner_id = "inner-context-id"
        
        with correlation_context(outer_id):
            assert get_correlation_id() == outer_id
            
            with correlation_context(inner_id):
                assert get_correlation_id() == inner_id
            
            # Should restore outer context
            assert get_correlation_id() == outer_id


class TestCorrelationIdLogging:
    """Test correlation ID integration with logging."""
    
    def setup_method(self):
        """Clear correlation ID context before each test."""
        # Reset any existing correlation ID
        from lexgraph_legal_rag.correlation import _correlation_id
        _correlation_id.set(None)
    
    def test_correlation_id_processor_adds_id_to_logs(self):
        """Test that CorrelationIdProcessor adds correlation ID to log events."""
        processor = CorrelationIdProcessor()
        test_id = "logging-test-id-999"
        
        # Set correlation ID in context
        with correlation_context(test_id):
            # Mock log event
            event_dict = {"message": "test log message", "level": "info"}
            
            # Process the event
            processed = processor(None, "info", event_dict)
            
            # Should include correlation ID
            assert "correlation_id" in processed
            assert processed["correlation_id"] == test_id
    
    def test_correlation_id_processor_no_id_set(self):
        """Test processor behavior when no correlation ID is set."""
        processor = CorrelationIdProcessor()
        
        # Mock log event without correlation ID context
        event_dict = {"message": "test log message", "level": "info"}
        
        # Process the event
        processed = processor(None, "info", event_dict)
        
        # Should not include correlation ID when none is set
        assert "correlation_id" not in processed


class TestCorrelationIdAPIIntegration:
    """Test correlation ID integration with API endpoints."""
    
    def test_api_endpoint_returns_correlation_id_header(self):
        """Test that API endpoints return correlation ID in headers."""
        # Create a simple test app
        app = FastAPI()
        app.add_middleware(CorrelationIdMiddleware)
        
        @app.get("/test")
        async def test_endpoint():
            return {"message": "test"}
        
        client = TestClient(app)
        response = client.get("/test")
        
        assert response.status_code == 200
        assert CORRELATION_ID_HEADER in response.headers
        correlation_id = response.headers[CORRELATION_ID_HEADER]
        assert correlation_id is not None
        assert len(correlation_id) == 36  # UUID format
    
    def test_correlation_id_preserved_across_requests(self):
        """Test that correlation ID is preserved when provided."""
        app = FastAPI()
        app.add_middleware(CorrelationIdMiddleware)
        
        @app.get("/test")
        async def test_endpoint():
            return {"correlation_id": get_correlation_id()}
        
        client = TestClient(app)
        test_id = "persistent-test-id-123"
        
        # Send request with specific correlation ID
        response = client.get("/test", headers={CORRELATION_ID_HEADER: test_id})
        
        assert response.status_code == 200
        # Response header should match
        assert response.headers[CORRELATION_ID_HEADER] == test_id
        # Response body should also match
        assert response.json()["correlation_id"] == test_id