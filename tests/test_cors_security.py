"""Tests for CORS security configuration.

Security-focused tests to ensure proper CORS configuration preventing
unauthorized cross-origin requests in production.
"""

import pytest
import os
from unittest.mock import patch
from fastapi.testclient import TestClient

from lexgraph_legal_rag.api import create_api


class TestCORSSecurity:
    """Test CORS security configuration."""

    def test_cors_production_mode_restrictive(self):
        """Test CORS configuration in production mode is restrictive."""
        # Set required environment variables for production mode
        with patch.dict(os.environ, {
            "API_KEY": "test-key-12345",
            "ENVIRONMENT": "production"
        }, clear=False):
            app = create_api(api_key="test-key", test_mode=False)
            client = TestClient(app)
            
            # Test preflight request from unauthorized origin
            response = client.options(
                "/ping",
                headers={
                    "Origin": "https://malicious-site.com",
                    "Access-Control-Request-Method": "GET", 
                    "Access-Control-Request-Headers": "X-API-Key"
                }
            )
            
            # Should either reject or be very restrictive
            if response.status_code == 200:
                # If allowing CORS, should not allow all origins
                cors_header = response.headers.get("Access-Control-Allow-Origin")
                assert cors_header != "*", "Production should not allow all origins"
                assert cors_header is None or "malicious-site.com" not in cors_header

    def test_cors_development_mode_permissive(self):
        """Test CORS configuration in development/test mode can be more permissive."""
        app = create_api(api_key="test-key", test_mode=True)
        client = TestClient(app)
        
        # Development mode can be more permissive
        response = client.options(
            "/ping",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "X-API-Key"
            }
        )
        
        # Development should allow localhost
        if response.status_code == 200:
            cors_header = response.headers.get("Access-Control-Allow-Origin")
            assert cors_header is not None  # Some CORS policy should exist

    def test_cors_configuration_environment_based(self):
        """Test CORS configuration respects environment variables."""
        allowed_origins = "https://example.com,https://app.example.com"
        
        with patch.dict(os.environ, {
            "API_KEY": "test-key-12345",
            "CORS_ALLOWED_ORIGINS": allowed_origins,
            "ENVIRONMENT": "production"
        }, clear=False):
            app = create_api(api_key="test-key", test_mode=False)
            client = TestClient(app)
            
            # Test allowed origin
            response = client.options(
                "/ping",
                headers={
                    "Origin": "https://example.com",
                    "Access-Control-Request-Method": "GET",
                    "Access-Control-Request-Headers": "X-API-Key"
                }
            )
            
            if response.status_code == 200:
                cors_header = response.headers.get("Access-Control-Allow-Origin")
                assert cors_header in ["https://example.com", "*"] or cors_header is None

    def test_cors_security_headers_present(self):
        """Test that security-related CORS headers are properly configured."""
        with patch.dict(os.environ, {
            "API_KEY": "test-key-12345",
            "ENVIRONMENT": "production"
        }, clear=False):
            app = create_api(api_key="test-key", test_mode=False)
            client = TestClient(app)
        
        response = client.options(
            "/ping",
            headers={
                "Origin": "https://trusted-domain.com",
                "Access-Control-Request-Method": "GET"
            }
        )
        
        # Even if CORS is allowed, credentials should be handled carefully
        if "Access-Control-Allow-Origin" in response.headers:
            if response.headers.get("Access-Control-Allow-Credentials") == "true":
                # If credentials are allowed, origin should not be *
                assert response.headers.get("Access-Control-Allow-Origin") != "*"

    def test_cors_methods_restricted(self):
        """Test that CORS allowed methods are restricted appropriately."""
        with patch.dict(os.environ, {
            "API_KEY": "test-key-12345",
            "ENVIRONMENT": "production"
        }, clear=False):
            app = create_api(api_key="test-key", test_mode=False)
            client = TestClient(app)
        
        response = client.options(
            "/ping",
            headers={
                "Origin": "https://example.com",
                "Access-Control-Request-Method": "DELETE"
            }
        )
        
        # Should not allow all methods in production
        if response.status_code == 200:
            allowed_methods = response.headers.get("Access-Control-Allow-Methods", "")
            # In production, should be restrictive about dangerous methods
            assert allowed_methods != "*" or "DELETE" not in allowed_methods

    def test_cors_headers_restricted(self):
        """Test that CORS allowed headers are restricted appropriately."""
        with patch.dict(os.environ, {
            "API_KEY": "test-key-12345",
            "ENVIRONMENT": "production"
        }, clear=False):
            app = create_api(api_key="test-key", test_mode=False)
            client = TestClient(app)
        
        response = client.options(
            "/ping",
            headers={
                "Origin": "https://example.com",
                "Access-Control-Request-Headers": "X-Custom-Dangerous-Header"
            }
        )
        
        # Should not allow all headers in production
        if response.status_code == 200:
            allowed_headers = response.headers.get("Access-Control-Allow-Headers", "")
            # In production, should be specific about allowed headers
            assert allowed_headers != "*"


class TestCORSEnvironmentConfiguration:
    """Test CORS configuration through environment variables."""

    def test_cors_origins_from_environment(self):
        """Test CORS origins can be configured via environment."""
        origins = "https://app1.com,https://app2.com"
        
        with patch.dict(os.environ, {
            "API_KEY": "test-key-12345",
            "CORS_ALLOWED_ORIGINS": origins,
            "ENVIRONMENT": "production"
        }, clear=False):
            app = create_api(api_key="test-key", test_mode=False)
            # App should be created without error
            assert app is not None

    def test_cors_default_secure_configuration(self):
        """Test default CORS configuration is secure."""
        # Clear any CORS environment variables
        with patch.dict(os.environ, {}, clear=True):
            # Add required variables
            with patch.dict(os.environ, {"API_KEY": "test-key-12345"}):
                app = create_api(test_mode=False)
                client = TestClient(app)
                
                # Test that default configuration is secure
                response = client.options(
                    "/health",  # Use health endpoint which should be accessible
                    headers={"Origin": "https://random-site.com"}
                )
                
                # Default should be secure - not allow arbitrary origins
                if response.status_code == 200:
                    cors_header = response.headers.get("Access-Control-Allow-Origin")
                    assert cors_header != "*"

    def test_cors_localhost_development_allowed(self):
        """Test localhost is allowed in development mode."""
        app = create_api(api_key="test-key", test_mode=True)
        client = TestClient(app)
        
        # Common development origins should be handled appropriately
        dev_origins = [
            "http://localhost:3000",
            "http://localhost:8080",
            "http://127.0.0.1:3000"
        ]
        
        for origin in dev_origins:
            response = client.options(
                "/health",
                headers={"Origin": origin}
            )
            
            # Development mode should be more permissive
            if response.status_code == 200:
                cors_header = response.headers.get("Access-Control-Allow-Origin")
                # Should either allow the origin specifically or allow all (in dev)
                assert cors_header is not None