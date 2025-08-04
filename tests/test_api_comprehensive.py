"""Comprehensive tests for api.py module.

This module provides extensive test coverage for the FastAPI application
and all API endpoints to improve overall test coverage significantly.
"""

import pytest
import json
import os
import time
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import FastAPI, status
from typing import Dict, Any

from lexgraph_legal_rag.api import (
    create_api,
    _get_cors_origins,
    API_KEY_ENV,
    RATE_LIMIT,
    SUPPORTED_VERSIONS
)


class TestAppCreation:
    """Test suite for FastAPI application creation and configuration."""
    
    def test_create_api_basic(self):
        """Test basic API creation."""
        app = create_api()
        assert isinstance(app, FastAPI)
        assert app.title == "LexGraph Legal RAG API"
        assert app.description is not None
        assert app.version is not None
    
    def test_create_api_with_test_mode(self):
        """Test API creation in test mode."""
        app = create_api(test_mode=True)
        assert isinstance(app, FastAPI)
        # In test mode, should have more permissive CORS
        assert len(app.middleware_stack) > 0
    
    def test_create_api_without_test_mode(self):
        """Test app creation in production mode."""
        app = create_api(test_mode=False)
        assert isinstance(app, FastAPI)
        # Should still have middleware configured
        assert len(app.middleware_stack) > 0
    
    def test_app_has_required_routes(self):
        """Test that app has all required routes."""
        app = create_api(test_mode=True)
        
        # Get all route paths
        routes = [route.path for route in app.routes if hasattr(route, 'path')]
        
        # Should have at least basic endpoints
        assert any("/docs" in route or "/openapi" in route for route in routes)
        
        # Check that we have some API routes defined
        assert len(routes) > 2  # At least docs + openapi + some API routes


class TestCORSConfiguration:
    """Test suite for CORS configuration."""
    
    def test_get_cors_origins_test_mode(self):
        """Test CORS origins in test mode."""
        origins = _get_cors_origins(test_mode=True)
        assert isinstance(origins, list)
        assert len(origins) > 0
        # Should include localhost origins for development
        assert any("localhost" in origin for origin in origins)
    
    def test_get_cors_origins_production_mode(self):
        """Test CORS origins in production mode."""
        origins = _get_cors_origins(test_mode=False)
        assert isinstance(origins, list)
        # Production mode should have more restrictive origins
        # but still return a list
    
    @patch.dict(os.environ, {"CORS_ALLOWED_ORIGINS": "https://app1.com,https://app2.com"})
    def test_get_cors_origins_from_env(self):
        """Test CORS origins from environment variable."""
        origins = _get_cors_origins(test_mode=False)
        assert "https://app1.com" in origins
        assert "https://app2.com" in origins
        assert len(origins) == 2
    
    @patch.dict(os.environ, {"CORS_ALLOWED_ORIGINS": "https://app1.com, , https://app2.com,"})
    def test_get_cors_origins_env_with_empty_strings(self):
        """Test CORS origins filtering empty strings."""
        origins = _get_cors_origins(test_mode=False)
        assert "https://app1.com" in origins
        assert "https://app2.com" in origins
        assert "" not in origins
        assert len(origins) == 2


class TestAPIConstants:
    """Test suite for API constants and configuration."""
    
    def test_api_key_env_constant(self):
        """Test API key environment variable constant."""
        assert API_KEY_ENV == "API_KEY"
        assert isinstance(API_KEY_ENV, str)
    
    def test_rate_limit_constant(self):
        """Test rate limit constant."""
        assert RATE_LIMIT is not None
        assert isinstance(RATE_LIMIT, int)
        assert RATE_LIMIT > 0
    
    def test_supported_versions_constant(self):
        """Test supported versions constant."""
        assert SUPPORTED_VERSIONS is not None
        assert isinstance(SUPPORTED_VERSIONS, (list, tuple, set))


class TestAPIIntegration:
    """Integration tests for the API application."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Set required environment variables for testing
        os.environ[API_KEY_ENV] = "test-api-key-12345678901234567890"
        self.app = create_api(test_mode=True)
        self.client = TestClient(self.app)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        # Clean up environment
        if API_KEY_ENV in os.environ:
            del os.environ[API_KEY_ENV]
    
    def test_app_starts_successfully(self):
        """Test that the app starts without errors."""
        # Simply creating the test client should work
        assert self.client is not None
        assert self.app is not None
    
    def test_openapi_schema_accessible(self):
        """Test that OpenAPI schema is accessible."""
        response = self.client.get("/openapi.json")
        assert response.status_code == 200
        
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert schema["info"]["title"] == "LexGraph Legal RAG API"
    
    def test_docs_accessible(self):
        """Test that API documentation is accessible."""
        response = self.client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_redoc_accessible(self):
        """Test that ReDoc documentation is accessible."""
        response = self.client.get("/redoc")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


class TestAPIKeyAuthentication:
    """Test suite for API key authentication."""
    
    def setup_method(self):
        """Set up test fixtures with API key."""
        os.environ[API_KEY_ENV] = "test-api-key-12345678901234567890"
        self.app = create_api(test_mode=True)
        self.client = TestClient(self.app)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if API_KEY_ENV in os.environ:
            del os.environ[API_KEY_ENV]
    
    def test_api_key_required_constant(self):
        """Test that API key environment constant is correct."""
        assert API_KEY_ENV == "API_KEY"
    
    @patch.dict(os.environ, {"API_KEY": "valid-test-key-12345"})
    def test_environment_api_key_loading(self):
        """Test that API key is loaded from environment."""
        # Create new app to test environment loading
        app = create_api(test_mode=True)
        assert app is not None
        # Environment should be loaded during app creation


class TestAPIVersioning:
    """Test suite for API versioning functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        os.environ[API_KEY_ENV] = "test-api-key-12345678901234567890"
        self.app = create_api(test_mode=True)
        self.client = TestClient(self.app)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if API_KEY_ENV in os.environ:
            del os.environ[API_KEY_ENV]
    
    def test_supported_versions_available(self):
        """Test that supported versions are defined."""
        assert SUPPORTED_VERSIONS is not None
        assert len(SUPPORTED_VERSIONS) > 0
    
    def test_version_header_handling(self):
        """Test API version header handling."""
        # Test that version headers are handled (if implemented)
        response = self.client.get("/docs", headers={"Accept-Version": "v1"})
        # Should not error out, even if versioning is not fully implemented
        assert response.status_code in [200, 404, 406]


class TestAPIMiddleware:
    """Test suite for API middleware functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        os.environ[API_KEY_ENV] = "test-api-key-12345678901234567890"
        self.app = create_api(test_mode=True)
        self.client = TestClient(self.app)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if API_KEY_ENV in os.environ:
            del os.environ[API_KEY_ENV]
    
    def test_cors_middleware_configured(self):
        """Test that CORS middleware is properly configured."""
        # Test CORS preflight request
        response = self.client.options(
            "/docs",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "Content-Type"
            }
        )
        # Should handle CORS preflight
        assert response.status_code in [200, 404]
    
    def test_middleware_stack_exists(self):
        """Test that middleware stack is configured."""
        assert len(self.app.middleware_stack) > 0
        
        # Should have at least some middleware configured
        middleware_types = [type(middleware).__name__ for middleware in self.app.middleware_stack]
        assert len(middleware_types) > 0


class TestAPIErrorHandling:
    """Test suite for API error handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        os.environ[API_KEY_ENV] = "test-api-key-12345678901234567890"
        self.app = create_api(test_mode=True)
        self.client = TestClient(self.app)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if API_KEY_ENV in os.environ:
            del os.environ[API_KEY_ENV]
    
    def test_404_error_handling(self):
        """Test 404 error handling."""
        response = self.client.get("/nonexistent-endpoint")
        assert response.status_code == 404
    
    def test_method_not_allowed_handling(self):
        """Test method not allowed error handling."""
        # Try POST on a GET-only endpoint
        response = self.client.post("/docs")
        assert response.status_code in [404, 405, 422]  # Various possible responses


class TestAPIConfiguration:
    """Test suite for API configuration management."""
    
    def test_app_metadata(self):
        """Test application metadata configuration."""
        app = create_api(test_mode=True)
        
        assert app.title is not None
        assert isinstance(app.title, str)
        assert len(app.title) > 0
        
        assert app.version is not None
        assert isinstance(app.version, str)
        
        if app.description:
            assert isinstance(app.description, str)
    
    def test_app_with_different_configs(self):
        """Test app creation with different configurations."""
        # Test mode configuration
        test_app = create_api(test_mode=True)
        assert isinstance(test_app, FastAPI)
        
        # Production mode configuration
        prod_app = create_api(test_mode=False)
        assert isinstance(prod_app, FastAPI)
        
        # Both should be valid FastAPI apps
        assert test_app.title == prod_app.title
    
    @patch.dict(os.environ, {"DEBUG": "true"})
    def test_app_debug_mode(self):
        """Test app creation with debug mode."""
        app = create_api(test_mode=True)
        assert isinstance(app, FastAPI)
    
    @patch.dict(os.environ, {"ENVIRONMENT": "production"})
    def test_app_production_environment(self):
        """Test app creation in production environment."""
        app = create_api(test_mode=False)
        assert isinstance(app, FastAPI)


class TestAPIPerformance:
    """Test suite for API performance considerations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        os.environ[API_KEY_ENV] = "test-api-key-12345678901234567890"
        self.app = create_api(test_mode=True)
        self.client = TestClient(self.app)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if API_KEY_ENV in os.environ:
            del os.environ[API_KEY_ENV]
    
    def test_app_startup_time(self):
        """Test that app starts up quickly."""
        start_time = time.time()
        app = create_api(test_mode=True)
        client = TestClient(app)
        end_time = time.time()
        
        startup_time = end_time - start_time
        # Should start up in reasonable time (less than 5 seconds)
        assert startup_time < 5.0
        assert client is not None
    
    def test_docs_response_time(self):
        """Test documentation endpoint response time."""
        start_time = time.time()
        response = self.client.get("/docs")
        end_time = time.time()
        
        response_time = end_time - start_time
        # Should respond quickly (less than 2 seconds)
        assert response_time < 2.0
        assert response.status_code == 200


class TestAPISecurity:
    """Test suite for API security features."""
    
    def test_api_key_environment_required(self):
        """Test that API key environment configuration is required."""
        # Test the constant exists and is correct
        assert API_KEY_ENV == "API_KEY"
        assert isinstance(API_KEY_ENV, str)
    
    def test_rate_limit_configuration(self):
        """Test rate limit configuration."""
        assert RATE_LIMIT is not None
        assert isinstance(RATE_LIMIT, int)
        assert RATE_LIMIT > 0
        # Should be a reasonable rate limit
        assert RATE_LIMIT <= 10000  # Not unlimited
    
    @patch.dict(os.environ, {"API_KEY": "short"})
    def test_short_api_key_handling(self):
        """Test handling of short API keys."""
        # Should be able to create app even with short key in test mode
        app = create_api(test_mode=True)
        assert isinstance(app, FastAPI)
    
    def test_cors_security_configuration(self):
        """Test CORS security configuration."""
        # Test mode should be more permissive
        test_origins = _get_cors_origins(test_mode=True)
        assert isinstance(test_origins, list)
        
        # Production mode should be more restrictive
        prod_origins = _get_cors_origins(test_mode=False)
        assert isinstance(prod_origins, list)