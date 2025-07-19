"""Integration tests for API endpoints with versioning."""

import pytest
from fastapi.testclient import TestClient

from lexgraph_legal_rag.api import create_api


class TestAPIVersioning:
    """Test API versioning functionality."""
    
    def setup_method(self):
        """Set up test client."""
        self.app = create_api(api_key="test-secret-key")
        self.client = TestClient(self.app)
        self.headers = {"X-API-Key": "test-secret-key"}
    
    def test_version_negotiation_via_url(self):
        """Test version negotiation via URL path."""
        # Test v1 URL
        resp = self.client.get("/v1/ping", headers=self.headers)
        assert resp.status_code == 200
        assert resp.json()["version"] == "v1"
        assert resp.headers["X-API-Version"] == "v1"
    
    def test_version_negotiation_via_header(self):
        """Test version negotiation via Accept header."""
        headers = {
            **self.headers,
            "Accept": "application/vnd.lexgraph.v1+json"
        }
        resp = self.client.get("/ping", headers=headers)
        assert resp.status_code == 200
        assert resp.json()["version"] == "v1"
        assert resp.headers["X-API-Version"] == "v1"
    
    def test_version_negotiation_via_custom_header(self):
        """Test version negotiation via X-API-Version header."""
        headers = {
            **self.headers,
            "X-API-Version": "v1"
        }
        resp = self.client.get("/ping", headers=headers)
        assert resp.status_code == 200
        assert resp.json()["version"] == "v1"
    
    def test_version_negotiation_via_query(self):
        """Test version negotiation via query parameter."""
        resp = self.client.get("/ping?version=v1", headers=self.headers)
        assert resp.status_code == 200
        assert resp.json()["version"] == "v1"
    
    def test_default_version_fallback(self):
        """Test default version when none specified."""
        resp = self.client.get("/ping", headers=self.headers)
        assert resp.status_code == 200
        assert resp.json()["version"] == "v1"  # Default version
    
    def test_unsupported_version_fallback(self):
        """Test fallback for unsupported versions."""
        headers = {
            **self.headers,
            "X-API-Version": "v99"
        }
        resp = self.client.get("/ping", headers=headers)
        assert resp.status_code == 200
        # Should fall back to supported version
        assert resp.json()["version"] in ["v1", "v2"]
    
    def test_version_info_endpoint(self):
        """Test the version information endpoint."""
        resp = self.client.get("/version", headers=self.headers)
        assert resp.status_code == 200
        
        data = resp.json()
        assert "supported_versions" in data
        assert "default_version" in data
        assert "latest_version" in data
        assert "current_version" in data
        assert isinstance(data["supported_versions"], list)


class TestAPIAuthentication:
    """Test API authentication and security."""
    
    def setup_method(self):
        """Set up test client."""
        self.app = create_api(api_key="secure-test-key")
        self.client = TestClient(self.app)
    
    def test_authenticated_request(self):
        """Test successful authenticated request."""
        headers = {"X-API-Key": "secure-test-key"}
        resp = self.client.get("/v1/ping", headers=headers)
        assert resp.status_code == 200
    
    def test_missing_api_key(self):
        """Test request without API key."""
        resp = self.client.get("/v1/ping")
        assert resp.status_code == 401
    
    def test_invalid_api_key(self):
        """Test request with invalid API key."""
        headers = {"X-API-Key": "invalid-key"}
        resp = self.client.get("/v1/ping", headers=headers)
        assert resp.status_code == 401
    
    def test_health_endpoints_no_auth(self):
        """Test that health endpoints don't require authentication."""
        # Health check
        resp = self.client.get("/health")
        assert resp.status_code == 200
        
        # Readiness check
        resp = self.client.get("/ready")
        assert resp.status_code == 200


class TestAPIEndpoints:
    """Test core API endpoint functionality."""
    
    def setup_method(self):
        """Set up test client."""
        self.app = create_api(api_key="test-api-key")
        self.client = TestClient(self.app)
        self.headers = {"X-API-Key": "test-api-key"}
    
    def test_ping_endpoint(self):
        """Test ping endpoint functionality."""
        resp = self.client.get("/v1/ping", headers=self.headers)
        assert resp.status_code == 200
        
        data = resp.json()
        assert data["ping"] == "pong"
        assert "version" in data
        assert "timestamp" in data
    
    def test_add_endpoint(self):
        """Test addition endpoint."""
        resp = self.client.get("/v1/add?a=5&b=7", headers=self.headers)
        assert resp.status_code == 200
        
        data = resp.json()
        assert data["result"] == 12
    
    def test_add_endpoint_validation(self):
        """Test addition endpoint parameter validation."""
        # Missing parameters
        resp = self.client.get("/v1/add", headers=self.headers)
        assert resp.status_code == 422  # Validation error
        
        # Invalid parameter types
        resp = self.client.get("/v1/add?a=not_a_number&b=7", headers=self.headers)
        assert resp.status_code == 422
    
    def test_health_endpoint_structure(self):
        """Test health endpoint response structure."""
        resp = self.client.get("/health")
        assert resp.status_code == 200
        
        data = resp.json()
        assert "status" in data
        assert "version" in data
        assert "checks" in data
        assert isinstance(data["checks"], dict)
    
    def test_readiness_endpoint_structure(self):
        """Test readiness endpoint response structure."""
        resp = self.client.get("/ready")
        assert resp.status_code == 200
        
        data = resp.json()
        assert "ready" in data
        assert "checks" in data
        assert isinstance(data["checks"], dict)
        assert isinstance(data["ready"], bool)


class TestAPIKeyManagement:
    """Test API key management endpoints."""
    
    def setup_method(self):
        """Set up test client."""
        self.app = create_api(api_key="admin-key")
        self.client = TestClient(self.app)
        self.headers = {"X-API-Key": "admin-key"}
    
    def test_key_status_endpoint(self):
        """Test key status endpoint."""
        resp = self.client.get("/v1/admin/key-status", headers=self.headers)
        assert resp.status_code == 200
        
        data = resp.json()
        assert "message" in data
        assert "rotation_info" in data
        assert isinstance(data["rotation_info"], dict)
    
    def test_key_rotation_endpoint(self):
        """Test key rotation endpoint."""
        payload = {"new_primary_key": "new-secure-key-123"}
        resp = self.client.post("/v1/admin/rotate-keys", json=payload, headers=self.headers)
        assert resp.status_code == 200
        
        data = resp.json()
        assert "message" in data
        assert "rotation_info" in data
        assert "successfully" in data["message"].lower()
    
    def test_key_revocation_endpoint(self):
        """Test key revocation endpoint."""
        payload = {"api_key": "key-to-revoke"}
        resp = self.client.post("/v1/admin/revoke-key", json=payload, headers=self.headers)
        assert resp.status_code == 200
        
        data = resp.json()
        assert "message" in data
        assert "rotation_info" in data
    
    def test_key_management_validation(self):
        """Test key management endpoint validation."""
        # Invalid payload for rotation
        resp = self.client.post("/v1/admin/rotate-keys", json={}, headers=self.headers)
        assert resp.status_code == 422
        
        # Short key
        payload = {"new_primary_key": "short"}
        resp = self.client.post("/v1/admin/rotate-keys", json=payload, headers=self.headers)
        assert resp.status_code == 422
    
    def test_admin_endpoints_require_auth(self):
        """Test that admin endpoints require authentication."""
        # No auth header
        resp = self.client.get("/v1/admin/key-status")
        assert resp.status_code == 401
        
        # Wrong auth header
        headers = {"X-API-Key": "wrong-key"}
        resp = self.client.get("/v1/admin/key-status", headers=headers)
        assert resp.status_code == 401


class TestAPIMetrics:
    """Test metrics and monitoring endpoints."""
    
    def setup_method(self):
        """Set up test client."""
        self.app = create_api(api_key="metrics-key")
        self.client = TestClient(self.app)
        self.headers = {"X-API-Key": "metrics-key"}
    
    def test_metrics_endpoint(self):
        """Test metrics summary endpoint."""
        resp = self.client.get("/v1/admin/metrics", headers=self.headers)
        assert resp.status_code == 200
        
        data = resp.json()
        assert isinstance(data, dict)
        assert "timestamp" in data
        
        # Should contain various metric categories
        expected_keys = ["cache", "index_pool", "memory", "key_management"]
        for key in expected_keys:
            assert key in data


class TestAPIRateLimit:
    """Test rate limiting functionality."""
    
    def setup_method(self):
        """Set up test client with low rate limit."""
        self.app = create_api(api_key="rate-test-key", rate_limit=2)
        self.client = TestClient(self.app)
        self.headers = {"X-API-Key": "rate-test-key"}
    
    def test_rate_limiting(self):
        """Test that rate limiting works."""
        # First two requests should succeed
        resp1 = self.client.get("/v1/ping", headers=self.headers)
        assert resp1.status_code == 200
        
        resp2 = self.client.get("/v1/ping", headers=self.headers)
        assert resp2.status_code == 200
        
        # Third request should be rate limited
        resp3 = self.client.get("/v1/ping", headers=self.headers)
        assert resp3.status_code == 429


class TestOpenAPIDocumentation:
    """Test OpenAPI documentation generation."""
    
    def setup_method(self):
        """Set up test client."""
        self.app = create_api(api_key="docs-key")
        self.client = TestClient(self.app)
    
    def test_openapi_schema(self):
        """Test OpenAPI schema generation."""
        resp = self.client.get("/openapi.json")
        assert resp.status_code == 200
        
        schema = resp.json()
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema
        assert "components" in schema
        
        # Check for security schemes
        assert "securitySchemes" in schema["components"]
        assert "ApiKeyAuth" in schema["components"]["securitySchemes"]
    
    def test_swagger_ui(self):
        """Test Swagger UI accessibility."""
        resp = self.client.get("/docs")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
    
    def test_redoc_ui(self):
        """Test ReDoc UI accessibility."""
        resp = self.client.get("/redoc")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]


@pytest.mark.integration
class TestFullAPIWorkflow:
    """Test complete API workflows."""
    
    def setup_method(self):
        """Set up test client."""
        self.app = create_api(api_key="workflow-key")
        self.client = TestClient(self.app)
        self.headers = {"X-API-Key": "workflow-key"}
    
    def test_complete_api_workflow(self):
        """Test a complete API usage workflow."""
        # 1. Check health
        health_resp = self.client.get("/health")
        assert health_resp.status_code == 200
        
        # 2. Check readiness
        ready_resp = self.client.get("/ready")
        assert ready_resp.status_code == 200
        
        # 3. Get version info
        version_resp = self.client.get("/version", headers=self.headers)
        assert version_resp.status_code == 200
        
        # 4. Test basic functionality
        ping_resp = self.client.get("/v1/ping", headers=self.headers)
        assert ping_resp.status_code == 200
        
        # 5. Test computation
        add_resp = self.client.get("/v1/add?a=10&b=20", headers=self.headers)
        assert add_resp.status_code == 200
        assert add_resp.json()["result"] == 30
        
        # 6. Check admin functionality
        status_resp = self.client.get("/v1/admin/key-status", headers=self.headers)
        assert status_resp.status_code == 200
        
        # 7. Check metrics
        metrics_resp = self.client.get("/v1/admin/metrics", headers=self.headers)
        assert metrics_resp.status_code == 200