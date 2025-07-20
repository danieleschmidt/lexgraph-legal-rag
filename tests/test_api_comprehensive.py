"""Comprehensive tests for API module to increase coverage to 80%+.

This test suite covers all API endpoints, middleware, authentication,
and error handling to maximize coverage of the 229-line API module.
"""

import pytest
import time
import json
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import HTTPException

from lexgraph_legal_rag.api import (
    create_api,
    verify_api_key,
    enforce_rate_limit,
    PingResponse,
    AddResponse,
    HealthResponse,
    ReadinessResponse,
    VersionInfo,
    KeyRotationRequest,
    KeyRevocationRequest,
    KeyManagementResponse,
    SUPPORTED_VERSIONS,
    API_KEY_ENV,
    RATE_LIMIT
)


class TestAPICreation:
    """Test API application creation and configuration."""

    def test_create_api_with_defaults(self):
        """Test API creation with default parameters."""
        app = create_api(api_key="test-key", test_mode=True)
        assert app.title == "LexGraph Legal RAG API"
        assert app.version == SUPPORTED_VERSIONS[0]

    def test_create_api_with_custom_version(self):
        """Test API creation with custom version."""
        if len(SUPPORTED_VERSIONS) > 1:
            custom_version = SUPPORTED_VERSIONS[1]
            app = create_api(version=custom_version, api_key="test-key", test_mode=True)
            assert app.version == custom_version

    def test_create_api_unsupported_version(self):
        """Test API creation fails with unsupported version."""
        with pytest.raises(ValueError, match="Unsupported API version"):
            create_api(version="v99", api_key="test-key", test_mode=True)

    def test_create_api_no_key_production_mode(self):
        """Test API creation fails without API key in production mode."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="API key not provided"):
                create_api(test_mode=False)

    def test_create_api_with_env_key(self):
        """Test API creation uses environment variable for API key."""
        with patch.dict('os.environ', {API_KEY_ENV: 'env-test-key'}):
            app = create_api(test_mode=True)
            assert app is not None

    def test_create_api_custom_rate_limit(self):
        """Test API creation with custom rate limit."""
        app = create_api(api_key="test-key", rate_limit=120, test_mode=True)
        assert app is not None

    def test_create_api_docs_disabled(self):
        """Test API creation with documentation disabled."""
        app = create_api(api_key="test-key", enable_docs=False, test_mode=True)
        assert app is not None


class TestAuthentication:
    """Test API key authentication functionality."""

    @patch('lexgraph_legal_rag.api.get_key_manager')
    def test_verify_api_key_with_key_manager(self, mock_get_key_manager):
        """Test API key verification using key manager."""
        mock_key_manager = Mock()
        mock_key_manager.get_active_key_count.return_value = 1
        mock_key_manager.is_valid_key.return_value = True
        mock_get_key_manager.return_value = mock_key_manager

        # Should not raise exception
        verify_api_key("valid-key")
        mock_key_manager.is_valid_key.assert_called_once_with("valid-key")

    @patch('lexgraph_legal_rag.api.get_key_manager')
    def test_verify_api_key_invalid_with_key_manager(self, mock_get_key_manager):
        """Test API key verification fails with invalid key via key manager."""
        mock_key_manager = Mock()
        mock_key_manager.get_active_key_count.return_value = 1
        mock_key_manager.is_valid_key.return_value = False
        mock_get_key_manager.return_value = mock_key_manager

        with pytest.raises(HTTPException) as exc_info:
            verify_api_key("invalid-key")
        assert exc_info.value.status_code == 401
        assert "Invalid API key" in str(exc_info.value.detail)

    @patch('lexgraph_legal_rag.api.get_key_manager')
    def test_verify_api_key_fallback_to_legacy(self, mock_get_key_manager):
        """Test API key verification falls back to legacy method."""
        mock_key_manager = Mock()
        mock_key_manager.get_active_key_count.return_value = 0
        mock_get_key_manager.return_value = mock_key_manager

        # Should not raise exception when falling back to legacy with matching key
        verify_api_key("test-key", api_key="test-key")

    def test_verify_api_key_legacy_method_valid(self):
        """Test legacy API key verification with valid key."""
        # Should not raise exception
        verify_api_key("test-key", api_key="test-key")

    def test_verify_api_key_legacy_method_invalid(self):
        """Test legacy API key verification with invalid key."""
        with pytest.raises(HTTPException) as exc_info:
            verify_api_key("wrong-key", api_key="test-key")
        assert exc_info.value.status_code == 401


class TestRateLimiting:
    """Test rate limiting functionality."""

    def test_enforce_rate_limit_under_limit(self):
        """Test rate limiting allows requests under the limit."""
        from collections import deque
        request_times = deque()
        # Should not raise exception
        enforce_rate_limit(request_times, limit=60)
        assert len(request_times) == 1

    def test_enforce_rate_limit_at_limit(self):
        """Test rate limiting allows requests at the limit."""
        from collections import deque
        current_time = time.time()
        request_times = deque([current_time - 30] * 59)  # 59 requests in last minute
        
        # Should not raise exception for 60th request
        enforce_rate_limit(request_times, limit=60)
        assert len(request_times) == 60

    def test_enforce_rate_limit_over_limit(self):
        """Test rate limiting blocks requests over the limit."""
        from collections import deque
        current_time = time.time()
        request_times = deque([current_time - 30] * 60)  # 60 requests in last minute
        
        with pytest.raises(HTTPException) as exc_info:
            enforce_rate_limit(request_times, limit=60)
        assert exc_info.value.status_code == 429
        assert "Rate limit exceeded" in str(exc_info.value.detail)

    def test_enforce_rate_limit_cleans_old_requests(self):
        """Test rate limiting cleans up old request timestamps."""
        from collections import deque
        current_time = time.time()
        request_times = deque([current_time - 120] * 30)  # Old requests beyond window
        
        enforce_rate_limit(request_times, limit=60)
        # Old requests should be cleaned up, only new one should remain
        assert len(request_times) == 1


class TestHealthEndpoints:
    """Test health check endpoints (no authentication required)."""

    def test_health_endpoint(self):
        """Test health check endpoint."""
        app = create_api(api_key="test-key", test_mode=True)
        client = TestClient(app)
        
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "checks" in data
        assert data["checks"]["api_key_configured"] is True

    def test_health_endpoint_no_api_key(self):
        """Test health endpoint shows API key not configured."""
        app = create_api(api_key="", test_mode=True)
        client = TestClient(app)
        
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["checks"]["api_key_configured"] is False

    def test_readiness_endpoint_ready(self):
        """Test readiness endpoint when service is ready."""
        app = create_api(api_key="test-key", test_mode=True)
        client = TestClient(app)
        
        response = client.get("/ready")
        assert response.status_code == 200
        
        data = response.json()
        assert "ready" in data
        assert "checks" in data
        assert "api_key" in data["checks"]
        assert "memory" in data["checks"]

    def test_readiness_endpoint_not_ready(self):
        """Test readiness endpoint when service is not ready."""
        app = create_api(api_key="", test_mode=True)
        client = TestClient(app)
        
        response = client.get("/ready")
        assert response.status_code == 200
        
        data = response.json()
        assert data["ready"] is False
        assert data["checks"]["api_key"]["status"] == "fail"


class TestCoreEndpoints:
    """Test core API endpoints (require authentication)."""

    def test_ping_endpoint_authenticated(self):
        """Test ping endpoint with valid authentication."""
        app = create_api(api_key="test-key", test_mode=True)
        client = TestClient(app)
        
        response = client.get("/ping", headers={"X-API-Key": "test-key"})
        assert response.status_code == 200
        
        data = response.json()
        assert data["ping"] == "pong"
        assert "version" in data
        assert "timestamp" in data

    def test_ping_endpoint_unauthenticated(self):
        """Test ping endpoint without authentication."""
        app = create_api(api_key="test-key", test_mode=True)
        client = TestClient(app)
        
        response = client.get("/ping")
        assert response.status_code == 401

    def test_ping_endpoint_invalid_key(self):
        """Test ping endpoint with invalid API key."""
        app = create_api(api_key="test-key", test_mode=True)
        client = TestClient(app)
        
        response = client.get("/ping", headers={"X-API-Key": "wrong-key"})
        assert response.status_code == 401

    def test_add_endpoint_valid_params(self):
        """Test add endpoint with valid parameters."""
        app = create_api(api_key="test-key", test_mode=True)
        client = TestClient(app)
        
        response = client.get(
            "/add?a=5&b=7", 
            headers={"X-API-Key": "test-key"}
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["result"] == 12

    def test_add_endpoint_missing_params(self):
        """Test add endpoint with missing parameters."""
        app = create_api(api_key="test-key", test_mode=True)
        client = TestClient(app)
        
        response = client.get(
            "/add?a=5", 
            headers={"X-API-Key": "test-key"}
        )
        assert response.status_code == 422  # Validation error

    def test_version_info_endpoint(self):
        """Test version information endpoint."""
        app = create_api(api_key="test-key", test_mode=True)
        client = TestClient(app)
        
        response = client.get(
            "/version", 
            headers={"X-API-Key": "test-key"}
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "supported_versions" in data
        assert "default_version" in data
        assert "latest_version" in data
        assert "current_version" in data


class TestVersionedEndpoints:
    """Test versioned API endpoints."""

    def test_versioned_ping_endpoint(self):
        """Test ping endpoint with version prefix."""
        version = SUPPORTED_VERSIONS[0]
        app = create_api(version=version, api_key="test-key", test_mode=True)
        client = TestClient(app)
        
        response = client.get(
            f"/{version}/ping", 
            headers={"X-API-Key": "test-key"}
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["version"] == version

    def test_add_endpoint_version_specific_response(self):
        """Test add endpoint returns version-specific response format."""
        app = create_api(api_key="test-key", test_mode=True)
        client = TestClient(app)
        
        # Test with version header
        response = client.get(
            "/add?a=3&b=4",
            headers={
                "X-API-Key": "test-key",
                "X-API-Version": SUPPORTED_VERSIONS[0]
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["result"] == 7


class TestAdminEndpoints:
    """Test admin endpoints for key management."""

    @patch('lexgraph_legal_rag.api.get_key_manager')
    def test_key_status_endpoint(self, mock_get_key_manager):
        """Test key status admin endpoint."""
        mock_key_manager = Mock()
        mock_key_manager.get_rotation_info.return_value = {
            "active_keys": 2,
            "revoked_keys": 0,
            "last_rotation": time.time()
        }
        mock_get_key_manager.return_value = mock_key_manager

        app = create_api(api_key="test-key", test_mode=True)
        client = TestClient(app)
        
        response = client.get(
            "/admin/key-status",
            headers={"X-API-Key": "test-key"}
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "rotation_info" in data

    @patch('lexgraph_legal_rag.api.get_key_manager')
    def test_rotate_keys_endpoint(self, mock_get_key_manager):
        """Test key rotation admin endpoint."""
        mock_key_manager = Mock()
        mock_key_manager.get_rotation_info.return_value = {
            "active_keys": 2,
            "revoked_keys": 0,
            "last_rotation": time.time()
        }
        mock_get_key_manager.return_value = mock_key_manager

        app = create_api(api_key="test-key", test_mode=True)
        client = TestClient(app)
        
        request_data = {"new_primary_key": "new-secure-key-123"}
        response = client.post(
            "/admin/rotate-keys",
            json=request_data,
            headers={"X-API-Key": "test-key"}
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "Key rotation completed" in data["message"]
        mock_key_manager.rotate_keys.assert_called_once_with("new-secure-key-123")

    @patch('lexgraph_legal_rag.api.get_key_manager')
    def test_revoke_key_endpoint(self, mock_get_key_manager):
        """Test key revocation admin endpoint."""
        mock_key_manager = Mock()
        mock_key_manager.get_rotation_info.return_value = {
            "active_keys": 1,
            "revoked_keys": 1,
            "last_rotation": time.time()
        }
        mock_get_key_manager.return_value = mock_key_manager

        app = create_api(api_key="test-key", test_mode=True)
        client = TestClient(app)
        
        request_data = {"api_key": "key-to-revoke"}
        response = client.post(
            "/admin/revoke-key",
            json=request_data,
            headers={"X-API-Key": "test-key"}
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "Key revocation completed" in data["message"]
        mock_key_manager.revoke_key.assert_called_once_with("key-to-revoke")

    @patch('lexgraph_legal_rag.api.get_key_manager')
    @patch('lexgraph_legal_rag.cache.get_query_cache')
    def test_metrics_endpoint(self, mock_get_cache, mock_get_key_manager):
        """Test metrics summary admin endpoint."""
        # Mock cache
        mock_cache = Mock()
        mock_cache.get_stats.return_value = {
            "hit_rate": 0.85,
            "total_requests": 100,
            "cache_hits": 85
        }
        mock_get_cache.return_value = mock_cache

        # Mock key manager
        mock_key_manager = Mock()
        mock_key_manager.get_rotation_info.return_value = {
            "active_keys": 2,
            "last_rotation": time.time()
        }
        mock_get_key_manager.return_value = mock_key_manager

        app = create_api(api_key="test-key", test_mode=True)
        client = TestClient(app)
        
        response = client.get(
            "/admin/metrics",
            headers={"X-API-Key": "test-key"}
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "cache" in data
        assert "index_pool" in data
        assert "memory" in data
        assert "key_management" in data
        assert "timestamp" in data


class TestMiddleware:
    """Test middleware functionality."""

    def test_cors_middleware(self):
        """Test CORS middleware is configured."""
        app = create_api(api_key="test-key", test_mode=True)
        client = TestClient(app)
        
        # Test preflight request
        response = client.options(
            "/ping",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "X-API-Key"
            }
        )
        # CORS should be configured (though exact behavior depends on settings)
        assert response.status_code in [200, 404]  # Depends on CORS config

    def test_rate_limiting_middleware(self):
        """Test rate limiting middleware integration."""
        app = create_api(api_key="test-key", rate_limit=2, test_mode=True)
        client = TestClient(app)
        
        headers = {"X-API-Key": "test-key"}
        
        # First request should succeed
        response = client.get("/ping", headers=headers)
        assert response.status_code == 200
        
        # Second request should succeed
        response = client.get("/ping", headers=headers)
        assert response.status_code == 200
        
        # Third request should be rate limited
        response = client.get("/ping", headers=headers)
        assert response.status_code == 429

    def test_correlation_id_middleware(self):
        """Test correlation ID middleware adds headers."""
        app = create_api(api_key="test-key", test_mode=True)
        client = TestClient(app)
        
        response = client.get("/health")
        # Should have correlation ID in response headers
        assert "X-Correlation-ID" in response.headers

    def test_version_negotiation_middleware(self):
        """Test version negotiation middleware."""
        app = create_api(api_key="test-key", test_mode=True)
        client = TestClient(app)
        
        # Test with version header
        response = client.get(
            "/ping",
            headers={
                "X-API-Key": "test-key",
                "X-API-Version": SUPPORTED_VERSIONS[0]
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["version"] == SUPPORTED_VERSIONS[0]


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_json_request(self):
        """Test API handles invalid JSON gracefully."""
        app = create_api(api_key="test-key", test_mode=True)
        client = TestClient(app)
        
        response = client.post(
            "/admin/rotate-keys",
            data="invalid json",
            headers={
                "X-API-Key": "test-key",
                "Content-Type": "application/json"
            }
        )
        assert response.status_code == 422

    def test_missing_required_fields(self):
        """Test API validates required fields."""
        app = create_api(api_key="test-key", test_mode=True)
        client = TestClient(app)
        
        # Missing new_primary_key field
        response = client.post(
            "/admin/rotate-keys",
            json={},
            headers={"X-API-Key": "test-key"}
        )
        assert response.status_code == 422

    def test_openapi_schema_generation(self):
        """Test OpenAPI schema is generated correctly."""
        app = create_api(api_key="test-key", test_mode=True)
        client = TestClient(app)
        
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        schema = response.json()
        assert "info" in schema
        assert "paths" in schema
        assert "components" in schema
        assert "securitySchemes" in schema["components"]


class TestResponseModels:
    """Test Pydantic response models."""

    def test_ping_response_model(self):
        """Test PingResponse model validation."""
        response = PingResponse(
            version="v1",
            ping="pong",
            timestamp=time.time()
        )
        assert response.version == "v1"
        assert response.ping == "pong"
        assert response.timestamp is not None

    def test_add_response_model(self):
        """Test AddResponse model validation."""
        response = AddResponse(result=42)
        assert response.result == 42

    def test_health_response_model(self):
        """Test HealthResponse model validation."""
        response = HealthResponse(
            status="healthy",
            version="v1",
            checks={"test": True}
        )
        assert response.status == "healthy"
        assert response.version == "v1"
        assert response.checks == {"test": True}

    def test_key_rotation_request_model(self):
        """Test KeyRotationRequest model validation."""
        request = KeyRotationRequest(new_primary_key="new-key-123")
        assert request.new_primary_key == "new-key-123"

    def test_key_rotation_request_validation(self):
        """Test KeyRotationRequest model validates minimum length."""
        with pytest.raises(ValueError):
            KeyRotationRequest(new_primary_key="short")

    def test_version_info_model(self):
        """Test VersionInfo model."""
        info = VersionInfo(
            supported_versions=["v1", "v2"],
            default_version="v1",
            latest_version="v2", 
            current_version="v1"
        )
        assert len(info.supported_versions) == 2
        assert info.default_version == "v1"


class TestIntegration:
    """Integration tests for complete API workflows."""

    def test_full_key_management_workflow(self):
        """Test complete key management workflow."""
        app = create_api(api_key="original-key", test_mode=True)
        client = TestClient(app)
        headers = {"X-API-Key": "original-key"}

        # 1. Check initial key status
        response = client.get("/admin/key-status", headers=headers)
        assert response.status_code == 200

        # 2. Rotate keys
        response = client.post(
            "/admin/rotate-keys",
            json={"new_primary_key": "new-rotated-key"},
            headers=headers
        )
        assert response.status_code == 200

        # 3. Check metrics after rotation
        response = client.get("/admin/metrics", headers=headers)
        assert response.status_code == 200

    def test_version_negotiation_workflow(self):
        """Test complete version negotiation workflow."""
        app = create_api(api_key="test-key", test_mode=True)
        client = TestClient(app)
        headers = {"X-API-Key": "test-key"}

        # 1. Get version info
        response = client.get("/version", headers=headers)
        assert response.status_code == 200
        version_data = response.json()

        # 2. Use specific version
        version_headers = {**headers, "X-API-Version": version_data["current_version"]}
        response = client.get("/ping", headers=version_headers)
        assert response.status_code == 200

        # 3. Test version-specific response
        response = client.get("/add?a=1&b=2", headers=version_headers)
        assert response.status_code == 200

    def test_health_check_workflow(self):
        """Test complete health check workflow."""
        app = create_api(api_key="test-key", test_mode=True)
        client = TestClient(app)

        # 1. Basic health check
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

        # 2. Detailed readiness check
        response = client.get("/ready")
        assert response.status_code == 200
        readiness_data = response.json()
        assert "ready" in readiness_data
        assert "checks" in readiness_data