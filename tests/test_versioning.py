"""Tests for API versioning system."""

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from unittest.mock import Mock

from lexgraph_legal_rag.versioning import (
    APIVersion,
    VersionNegotiationMiddleware,
    get_request_version,
    require_version,
    version_deprecated,
    VersionedResponse,
    create_version_info_endpoint,
    SUPPORTED_VERSIONS,
    DEFAULT_VERSION,
)


class TestAPIVersion:
    """Test APIVersion class functionality."""
    
    def test_version_creation(self):
        """Test APIVersion object creation."""
        v1 = APIVersion("v1")
        assert str(v1) == "v1"
        assert v1.major == 1
        
        v2 = APIVersion("v2")
        assert str(v2) == "v2"
        assert v2.major == 2
    
    def test_version_comparison(self):
        """Test version comparison operators."""
        v1 = APIVersion("v1")
        v2 = APIVersion("v2")
        
        assert v1 < v2
        assert v1 <= v2
        assert v2 > v1
        assert v1 == APIVersion("v1")
        assert v1 != v2
    
    def test_version_support_check(self):
        """Test version support checking."""
        v1 = APIVersion("v1")
        v99 = APIVersion("v99")
        
        assert v1.is_supported()
        assert not v99.is_supported()


class TestVersionNegotiationMiddleware:
    """Test version negotiation middleware."""
    
    def setup_method(self):
        """Set up test app with middleware."""
        self.app = FastAPI()
        self.app.add_middleware(VersionNegotiationMiddleware)
        
        @self.app.get("/test")
        async def test_endpoint(request: Request):
            version = get_request_version(request)
            return {"version": str(version)}
        
        self.client = TestClient(self.app)
    
    def test_url_version_negotiation(self):
        """Test version negotiation via URL path."""
        resp = self.client.get("/v1/test")
        # Middleware should extract v1 from URL
        # Note: This test verifies the middleware processes the request
        assert resp.status_code == 200
    
    def test_header_version_negotiation(self):
        """Test version negotiation via Accept header."""
        headers = {"Accept": "application/vnd.lexgraph.v1+json"}
        resp = self.client.get("/test", headers=headers)
        assert resp.status_code == 200
        assert "X-API-Version" in resp.headers
    
    def test_custom_header_negotiation(self):
        """Test version negotiation via X-API-Version header."""
        headers = {"X-API-Version": "v1"}
        resp = self.client.get("/test", headers=headers)
        assert resp.status_code == 200
        assert "X-API-Version" in resp.headers
    
    def test_query_parameter_negotiation(self):
        """Test version negotiation via query parameter."""
        resp = self.client.get("/test?version=v1")
        assert resp.status_code == 200
        assert "X-API-Version" in resp.headers
    
    def test_default_version_fallback(self):
        """Test default version when no version specified."""
        resp = self.client.get("/test")
        assert resp.status_code == 200
        assert resp.headers["X-API-Version"] == DEFAULT_VERSION
    
    def test_skip_versioning_paths(self):
        """Test that certain paths skip version negotiation."""
        # Health endpoints should skip versioning
        @self.app.get("/health")
        async def health():
            return {"status": "healthy"}
        
        resp = self.client.get("/health")
        assert resp.status_code == 200
        # Should not have version headers for skipped paths


class TestVersionDecorators:
    """Test version requirement and deprecation decorators."""
    
    def test_require_version_decorator(self):
        """Test require_version decorator."""
        @require_version("v2")
        async def test_endpoint(request: Request):
            return {"message": "success"}
        
        # Mock request with v1
        request = Mock()
        request.state.api_version = APIVersion("v1")
        
        # Should raise HTTPException for insufficient version
        with pytest.raises(Exception):  # HTTPException or similar
            import asyncio
            asyncio.run(test_endpoint(request))
    
    def test_version_deprecated_decorator(self):
        """Test version_deprecated decorator."""
        @version_deprecated("v1", "v3", "use /new-endpoint instead")
        async def deprecated_endpoint(request: Request):
            return {"message": "deprecated"}
        
        # Mock request
        request = Mock()
        request.state.api_version = APIVersion("v1")
        
        # Should execute but log deprecation warning
        import asyncio
        result = asyncio.run(deprecated_endpoint(request))
        assert result["message"] == "deprecated"


class TestVersionedResponse:
    """Test versioned response formatting."""
    
    def test_v1_response_format(self):
        """Test v1 response format (unchanged)."""
        data = {"result": 42}
        version = APIVersion("v1")
        
        formatted = VersionedResponse.format_response(data, version)
        assert formatted == data  # Should be unchanged for v1
    
    def test_v2_success_response_format(self):
        """Test v2 success response format."""
        data = {"result": 42}
        version = APIVersion("v2")
        
        formatted = VersionedResponse.format_response(data, version)
        
        assert formatted["success"] is True
        assert formatted["data"] == data
        assert "metadata" in formatted
        assert formatted["metadata"]["version"] == "v2"
    
    def test_v2_error_response_format(self):
        """Test v2 error response format."""
        data = {"error": {"detail": "Something went wrong"}}
        version = APIVersion("v2")
        
        formatted = VersionedResponse.format_response(data, version)
        
        assert formatted["success"] is False
        assert formatted["error"] == data["error"]
        assert "metadata" in formatted
    
    def test_unsupported_version_fallback(self):
        """Test fallback for unsupported versions."""
        data = {"result": 42}
        version = APIVersion("v99")
        
        formatted = VersionedResponse.format_response(data, version)
        assert formatted == data  # Should fall back to v1 format


class TestVersionInfoEndpoint:
    """Test version information endpoint."""
    
    def test_version_info_structure(self):
        """Test version info endpoint structure."""
        endpoint = create_version_info_endpoint()
        
        import asyncio
        info = asyncio.run(endpoint())
        
        assert "supported_versions" in info
        assert "default_version" in info
        assert "latest_version" in info
        assert "version_negotiation" in info
        assert "deprecations" in info
        
        assert isinstance(info["supported_versions"], (list, tuple))
        assert info["default_version"] in info["supported_versions"]
    
    def test_version_negotiation_methods(self):
        """Test that version negotiation methods are documented."""
        endpoint = create_version_info_endpoint()
        
        import asyncio
        info = asyncio.run(endpoint())
        
        methods = info["version_negotiation"]["methods"]
        assert any("Accept header" in method for method in methods)
        assert any("URL path" in method for method in methods)
        assert any("Query parameter" in method for method in methods)
        assert any("Custom header" in method for method in methods)


class TestVersionNegotiationPatterns:
    """Test version negotiation pattern matching."""
    
    def test_header_pattern_matching(self):
        """Test Accept header pattern matching."""
        from lexgraph_legal_rag.versioning import VERSION_PATTERNS
        
        pattern = VERSION_PATTERNS["header"]
        
        # Valid patterns
        assert pattern.search("application/vnd.lexgraph.v1+json")
        assert pattern.search("application/vnd.lexgraph.v2+json")
        
        # Invalid patterns
        assert not pattern.search("application/json")
        assert not pattern.search("text/plain")
    
    def test_url_pattern_matching(self):
        """Test URL pattern matching."""
        from lexgraph_legal_rag.versioning import VERSION_PATTERNS
        
        pattern = VERSION_PATTERNS["url"]
        
        # Valid patterns
        assert pattern.search("/v1/endpoint")
        assert pattern.search("/v2/ping")
        
        # Invalid patterns
        assert not pattern.search("/endpoint")
        assert not pattern.search("/version1/endpoint")
    
    def test_query_pattern_matching(self):
        """Test query parameter pattern matching."""
        from lexgraph_legal_rag.versioning import VERSION_PATTERNS
        
        pattern = VERSION_PATTERNS["query"]
        
        # Valid patterns
        assert pattern.search("version=v1")
        assert pattern.search("version=1")
        assert pattern.search("other=value&version=v2")
        
        # Invalid patterns
        assert not pattern.search("ver=v1")
        assert not pattern.search("version=latest")


class TestVersionMigration:
    """Test version migration scenarios."""
    
    def test_version_compatibility_check(self):
        """Test version compatibility checking."""
        v1 = APIVersion("v1")
        v2 = APIVersion("v2")
        
        # Test that v1 requests can be handled
        assert v1.is_supported()
        
        # Test that newer versions are handled appropriately
        assert v2.is_supported() or v1.is_supported()  # Fallback logic
    
    def test_backward_compatibility(self):
        """Test backward compatibility handling."""
        # Simulate request for older version
        older_version = APIVersion("v1")
        newer_version = APIVersion("v2")
        
        # Should be able to handle older version requests
        assert older_version.is_supported()
        
        # Newer version should be compatible with older
        assert newer_version >= older_version


@pytest.mark.integration
class TestVersioningIntegration:
    """Integration tests for versioning system."""
    
    def test_complete_versioning_workflow(self):
        """Test complete version negotiation workflow."""
        app = FastAPI()
        app.add_middleware(VersionNegotiationMiddleware)
        
        @app.get("/api/test")
        async def test_endpoint(request: Request):
            version = get_request_version(request)
            data = {"message": "success", "value": 42}
            return VersionedResponse.format_response(data, version)
        
        client = TestClient(app)
        
        # Test different version negotiation methods
        methods = [
            {"headers": {"Accept": "application/vnd.lexgraph.v1+json"}},
            {"url": "/v1/api/test"},
            {"params": "?version=v1"},
            {"headers": {"X-API-Version": "v1"}},
        ]
        
        for method in methods:
            if "url" in method:
                resp = client.get(method["url"])
            elif "params" in method:
                resp = client.get("/api/test" + method["params"])
            else:
                resp = client.get("/api/test", headers=method.get("headers", {}))
            
            assert resp.status_code == 200
            assert "X-API-Version" in resp.headers