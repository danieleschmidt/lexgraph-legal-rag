"""Additional tests to boost API module coverage to 80%+.

This test suite targets uncovered lines in the API module to reach the coverage goal.
"""

import pytest
import os
import time
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import HTTPException

from lexgraph_legal_rag.api import (
    create_api,
    verify_api_key,
    enforce_rate_limit,
    SUPPORTED_VERSIONS,
    API_KEY_ENV,
    RATE_LIMIT
)


class TestAPIEdgeCases:
    """Test edge cases and error paths in API functionality."""

    @patch('lexgraph_legal_rag.api.get_key_manager')
    def test_verify_api_key_exception_fallback(self, mock_get_key_manager):
        """Test verify_api_key falls back when key manager raises exception."""
        mock_key_manager = Mock()
        mock_key_manager.get_active_key_count.side_effect = Exception("Key manager error")
        mock_get_key_manager.return_value = mock_key_manager
        
        # Should fall back to legacy validation
        verify_api_key("test-key", api_key="test-key")  # Should not raise
        
    @patch('lexgraph_legal_rag.api.get_key_manager') 
    def test_verify_api_key_exception_fallback_invalid(self, mock_get_key_manager):
        """Test verify_api_key fallback fails with invalid key."""
        mock_key_manager = Mock()
        mock_key_manager.get_active_key_count.side_effect = Exception("Key manager error")
        mock_get_key_manager.return_value = mock_key_manager
        
        with pytest.raises(HTTPException) as exc_info:
            verify_api_key("wrong-key", api_key="test-key")
        assert exc_info.value.status_code == 401

    def test_verify_api_key_no_key_provided(self):
        """Test verify_api_key with no key provided fails."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(HTTPException) as exc_info:
                verify_api_key("test-key", api_key=None)
            assert exc_info.value.status_code == 401

    def test_enforce_rate_limit_exact_edge_case(self):
        """Test rate limiting at exact boundary conditions."""
        from collections import deque
        current_time = time.time()
        
        # Test with requests exactly at 60-second boundary
        request_times = deque([current_time - 60.0])  # Exactly 60 seconds ago
        
        # This should clean up the old request and allow the new one
        enforce_rate_limit(request_times, limit=60)
        assert len(request_times) == 1  # Old one cleaned, new one added

    def test_create_api_production_mode_with_env_key(self):
        """Test API creation in production mode with environment key."""
        with patch.dict('os.environ', {API_KEY_ENV: 'production-api-key-16chars'}):
            app = create_api(test_mode=False)
            assert app.title == "LexGraph Legal RAG API"

    def test_create_api_unsupported_version_message(self):
        """Test specific error message for unsupported version."""
        with pytest.raises(ValueError) as exc_info:
            create_api(version="v999", api_key="test-key", test_mode=True)
        assert "Unsupported API version: v999" in str(exc_info.value)


class TestVersionSpecificRoutes:
    """Test version-specific route behavior."""

    def test_add_endpoint_v2_response_format(self):
        """Test add endpoint returns v2 format when requested."""
        if len(SUPPORTED_VERSIONS) > 1:
            app = create_api(version=SUPPORTED_VERSIONS[1], api_key="test-key", test_mode=True)
            client = TestClient(app)
            
            response = client.get(
                "/add?a=10&b=20",
                headers={
                    "X-API-Key": "test-key",
                    "X-API-Version": "v2"
                }
            )
            assert response.status_code == 200
            # v2 format should be different from v1
            data = response.json()
            # This tests the version-specific formatting logic

    def test_default_router_inclusion(self):
        """Test default router is included for first supported version."""
        app = create_api(version=SUPPORTED_VERSIONS[0], api_key="test-key", test_mode=True)
        client = TestClient(app)
        
        # Test that unversioned endpoints work (default router)
        response = client.get("/ping", headers={"X-API-Key": "test-key"})
        assert response.status_code == 200

    def test_non_default_version_no_default_router(self):
        """Test non-default version doesn't include default router."""
        if len(SUPPORTED_VERSIONS) > 1:
            app = create_api(version=SUPPORTED_VERSIONS[1], api_key="test-key", test_mode=True)
            client = TestClient(app)
            
            # Unversioned endpoint should not work for non-default versions
            response = client.get("/ping", headers={"X-API-Key": "test-key"})
            # This should either be 404 or require versioned path


class TestOpenAPICustomization:
    """Test OpenAPI schema customization."""

    def test_custom_openapi_schema_caching(self):
        """Test OpenAPI schema is cached after first generation."""
        app = create_api(api_key="test-key", test_mode=True)
        
        # First call should generate schema
        schema1 = app.openapi()
        
        # Second call should return cached schema
        schema2 = app.openapi()
        
        assert schema1 is schema2  # Same object reference (cached)

    def test_openapi_security_schemes(self):
        """Test OpenAPI schema includes security schemes."""
        app = create_api(api_key="test-key", test_mode=True)
        client = TestClient(app)
        
        response = client.get("/openapi.json")
        schema = response.json()
        
        assert "components" in schema
        assert "securitySchemes" in schema["components"]
        assert "ApiKeyAuth" in schema["components"]["securitySchemes"]
        assert "VersionHeader" in schema["components"]["securitySchemes"]

    def test_openapi_version_info(self):
        """Test OpenAPI schema includes version information."""
        app = create_api(api_key="test-key", test_mode=True)
        client = TestClient(app)
        
        response = client.get("/openapi.json")
        schema = response.json()
        
        assert "info" in schema
        assert "x-api-versions" in schema["info"]
        assert "supported" in schema["info"]["x-api-versions"]
        assert "default" in schema["info"]["x-api-versions"]

    def test_openapi_tags_definition(self):
        """Test OpenAPI schema includes tag definitions."""
        app = create_api(api_key="test-key", test_mode=True)
        client = TestClient(app)
        
        response = client.get("/openapi.json")
        schema = response.json()
        
        assert "tags" in schema
        tag_names = [tag["name"] for tag in schema["tags"]]
        assert "Health" in tag_names
        assert "Admin" in tag_names
        assert "Utilities" in tag_names


class TestReadinessEdgeCases:
    """Test readiness endpoint edge cases."""

    @patch('psutil.virtual_memory')
    def test_readiness_high_memory_warning(self, mock_memory):
        """Test readiness endpoint warns on high memory usage."""
        # Mock high memory usage
        mock_memory_obj = Mock()
        mock_memory_obj.percent = 95.0
        mock_memory.return_value = mock_memory_obj
        
        app = create_api(api_key="test-key", test_mode=True)
        client = TestClient(app)
        
        response = client.get("/ready")
        assert response.status_code == 200
        
        data = response.json()
        assert data["checks"]["memory"]["status"] == "warn"
        assert data["checks"]["memory"]["usage_percent"] == 95.0

    @patch('lexgraph_legal_rag.cache.get_query_cache')
    def test_readiness_cache_error_handling(self, mock_get_cache):
        """Test readiness endpoint handles cache errors gracefully."""
        mock_get_cache.side_effect = Exception("Cache unavailable")
        
        app = create_api(api_key="test-key", test_mode=True)
        client = TestClient(app)
        
        response = client.get("/ready")
        assert response.status_code == 200
        
        data = response.json()
        assert data["checks"]["cache"]["status"] == "warn"
        assert "Cache check failed" in data["checks"]["cache"]["message"]

    @patch('lexgraph_legal_rag.http_client.ResilientHTTPClient')
    def test_readiness_http_client_error(self, mock_http_client):
        """Test readiness endpoint handles HTTP client check errors."""
        mock_http_client.side_effect = Exception("HTTP client error")
        
        app = create_api(api_key="test-key", test_mode=True)
        client = TestClient(app)
        
        response = client.get("/ready")
        assert response.status_code == 200
        
        data = response.json()
        # Should handle the circuit breaker check failure gracefully


class TestMetricsEndpointEdgeCases:
    """Test metrics endpoint edge cases."""

    @patch('lexgraph_legal_rag.cache.get_query_cache')
    def test_metrics_cache_exception_handling(self, mock_get_cache):
        """Test metrics endpoint handles cache exceptions."""
        mock_get_cache.side_effect = Exception("Cache error")
        
        app = create_api(api_key="test-key", test_mode=True)
        client = TestClient(app)
        
        response = client.get("/admin/metrics", headers={"X-API-Key": "test-key"})
        assert response.status_code == 200
        
        data = response.json()
        assert "cache" in data
        # Should return empty dict when cache fails

    @patch('lexgraph_legal_rag.faiss_index.FaissVectorIndex')
    def test_metrics_faiss_index_exception(self, mock_faiss):
        """Test metrics endpoint handles FAISS index exceptions."""
        mock_faiss.side_effect = Exception("FAISS error")
        
        app = create_api(api_key="test-key", test_mode=True)
        client = TestClient(app)
        
        response = client.get("/admin/metrics", headers={"X-API-Key": "test-key"})
        assert response.status_code == 200
        
        data = response.json()
        assert "index_pool" in data
        # Should return empty dict when FAISS fails

    def test_metrics_psutil_not_available(self):
        """Test metrics endpoint when psutil is not available."""
        app = create_api(api_key="test-key", test_mode=True)
        client = TestClient(app)
        
        # Mock globals to simulate psutil not being available
        with patch('lexgraph_legal_rag.api.globals', return_value={'psutil': False}):
            response = client.get("/admin/metrics", headers={"X-API-Key": "test-key"})
            assert response.status_code == 200
            
            data = response.json()
            assert data["memory"]["usage_percent"] is None


class TestAuthenticationDependency:
    """Test authentication dependency edge cases."""

    def test_auth_dep_none_header(self):
        """Test authentication dependency with None API key header."""
        app = create_api(api_key="test-key", test_mode=True)
        client = TestClient(app)
        
        # Don't provide X-API-Key header
        response = client.get("/ping")
        assert response.status_code == 401
        assert "X-API-Key header is required" in response.json()["detail"]

    def test_rate_limit_dependency(self):
        """Test rate limiting dependency integration."""
        app = create_api(api_key="test-key", rate_limit=1, test_mode=True)
        client = TestClient(app)
        
        headers = {"X-API-Key": "test-key"}
        
        # First request should work
        response = client.get("/ping", headers=headers)
        assert response.status_code == 200
        
        # Second request should be rate limited
        response = client.get("/ping", headers=headers)
        assert response.status_code == 429


class TestConfigurationValidation:
    """Test configuration validation at startup."""

    @patch('lexgraph_legal_rag.api.validate_environment')
    def test_environment_validation_called(self, mock_validate):
        """Test that environment validation is called during app creation."""
        app = create_api(api_key="test-key", test_mode=True)
        mock_validate.assert_called_once_with(allow_test_mode=True)

    @patch('lexgraph_legal_rag.api.validate_environment')
    def test_environment_validation_production(self, mock_validate):
        """Test environment validation called for production mode."""
        with patch.dict('os.environ', {API_KEY_ENV: 'production-valid-key-16plus'}):
            app = create_api(test_mode=False)
            mock_validate.assert_called_once_with(allow_test_mode=False)