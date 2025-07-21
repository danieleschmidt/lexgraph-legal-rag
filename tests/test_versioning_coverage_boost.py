"""Comprehensive versioning module coverage boost test suite.

This test suite targets comprehensive coverage of the versioning module,
focusing on edge cases, error paths, and integration scenarios to boost
coverage from 46% to 80%+.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from fastapi import FastAPI, Request, HTTPException
from fastapi.testclient import TestClient
from starlette.responses import JSONResponse

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
    LATEST_VERSION,
    VERSION_PATTERNS,
)


class TestAPIVersionComprehensive:
    """Comprehensive APIVersion testing for full coverage."""

    def test_version_creation_edge_cases(self):
        """Test APIVersion creation with various edge cases."""
        # Test with 'v' prefix
        v1 = APIVersion("v1")
        assert v1.version == "v1"
        assert v1.major == 1
        
        # Test without 'v' prefix
        v2_no_v = APIVersion("2")
        assert v2_no_v.version == "2"
        assert v2_no_v.major == 2
        
        # Test with non-numeric version
        v_alpha = APIVersion("alpha")
        assert v_alpha.version == "alpha"
        assert v_alpha.major == 1  # Default fallback
        
        # Test empty version
        v_empty = APIVersion("")
        assert v_empty.version == ""
        assert v_empty.major == 1  # Default fallback

    def test_version_string_repr(self):
        """Test string representation of APIVersion."""
        v1 = APIVersion("v1")
        assert str(v1) == "v1"
        assert f"{v1}" == "v1"

    def test_version_comparison_comprehensive(self):
        """Test all comparison scenarios."""
        v1 = APIVersion("v1")
        v2 = APIVersion("v2")
        v1_copy = APIVersion("v1")
        
        # Equality tests
        assert v1 == v1_copy
        assert v1 == "v1"
        assert not (v1 == v2)
        assert not (v1 == "v2")
        
        # Less than tests
        assert v1 < v2
        assert v1 < "v2"
        assert not (v2 < v1)
        assert not (v1 < "v1")
        
        # Less than or equal tests
        assert v1 <= v2
        assert v1 <= v1_copy
        assert v1 <= "v1"
        assert v1 <= "v2"
        
        # Greater than (implied)
        assert v2 > v1
        assert not (v1 > v2)
        
    def test_version_support_checking(self):
        """Test version support validation."""
        for supported_version in SUPPORTED_VERSIONS:
            version = APIVersion(supported_version)
            assert version.is_supported()
        
        unsupported = APIVersion("v999")
        assert not unsupported.is_supported()


class TestVersionNegotiationMiddlewareComprehensive:
    """Comprehensive middleware testing for full coverage."""

    def setup_method(self):
        """Set up test fixtures."""
        self.middleware = VersionNegotiationMiddleware(None, DEFAULT_VERSION)

    def test_middleware_initialization(self):
        """Test middleware initialization with custom default."""
        custom_middleware = VersionNegotiationMiddleware(None, "v2")
        assert custom_middleware.default_version == "v2"

    def test_skip_versioning_paths_comprehensive(self):
        """Test all skip path scenarios."""
        skip_paths = [
            "/health",
            "/ready", 
            "/metrics",
            "/docs",
            "/openapi.json",
            "/health/detailed",  # Subpaths should also be skipped
            "/docs/swagger",
        ]
        
        for path in skip_paths:
            assert self.middleware._should_skip_versioning(path)
        
        # Non-skip paths
        normal_paths = ["/api/test", "/v1/ping", "/admin", "/"]
        for path in normal_paths:
            assert not self.middleware._should_skip_versioning(path)

    def test_version_extraction_accept_header(self):
        """Test version extraction from Accept header."""
        request = Mock()
        request.url.path = "/test"
        request.url.query = ""
        request.headers.get.side_effect = lambda key, default="": {
            "accept": "application/vnd.lexgraph.v2+json",
            "X-API-Version": None
        }.get(key, default)
        
        version = self.middleware._extract_version(request)
        assert version == "v2"

    def test_version_extraction_url_path(self):
        """Test version extraction from URL path."""
        request = Mock()
        request.url.path = "/v2/endpoint"
        request.url.query = ""
        request.headers.get.side_effect = lambda key, default="": ""
        
        version = self.middleware._extract_version(request)
        assert version == "v2"

    def test_version_extraction_query_parameter(self):
        """Test version extraction from query parameter."""
        request = Mock()
        request.url.path = "/test"
        request.url.query = "version=v2&other=param"
        request.headers.get.side_effect = lambda key, default="": ""
        
        version = self.middleware._extract_version(request)
        assert version == "v2"

    def test_version_extraction_query_parameter_no_v(self):
        """Test version extraction from query parameter without 'v' prefix."""
        request = Mock()
        request.url.path = "/test"
        request.url.query = "version=2"
        request.headers.get.side_effect = lambda key, default="": ""
        
        version = self.middleware._extract_version(request)
        assert version == "v2"

    def test_version_extraction_custom_header(self):
        """Test version extraction from X-API-Version header."""
        request = Mock()
        request.url.path = "/test"
        request.url.query = ""
        request.headers.get.side_effect = lambda key, default="": {
            "accept": "",
            "X-API-Version": "v2"
        }.get(key, default)
        
        version = self.middleware._extract_version(request)
        assert version == "v2"

    def test_version_extraction_custom_header_no_v(self):
        """Test version extraction from X-API-Version header without 'v' prefix."""
        request = Mock()
        request.url.path = "/test"
        request.url.query = ""
        request.headers.get.side_effect = lambda key, default="": {
            "accept": "",
            "X-API-Version": "2"
        }.get(key, default)
        
        version = self.middleware._extract_version(request)
        assert version == "v2"

    def test_version_extraction_precedence(self):
        """Test version extraction precedence (Accept header wins)."""
        request = Mock()
        request.url.path = "/v1/test"  # URL has v1
        request.url.query = "version=v1"  # Query has v1
        request.headers.get.side_effect = lambda key, default="": {
            "accept": "application/vnd.lexgraph.v2+json",  # Accept header has v2
            "X-API-Version": "v1"  # Custom header has v1
        }.get(key, default)
        
        version = self.middleware._extract_version(request)
        assert version == "v2"  # Accept header should win

    def test_version_extraction_no_version(self):
        """Test version extraction when no version specified."""
        request = Mock()
        request.url.path = "/test"
        request.url.query = ""
        request.headers.get.side_effect = lambda key, default="": ""
        
        version = self.middleware._extract_version(request)
        assert version is None

    def test_version_negotiation_exact_match(self):
        """Test version negotiation for exact supported version."""
        for supported_version in SUPPORTED_VERSIONS:
            negotiated = self.middleware._negotiate_version(supported_version)
            assert negotiated.version == supported_version
            assert negotiated.is_supported()

    def test_version_negotiation_no_version(self):
        """Test version negotiation when no version requested."""
        negotiated = self.middleware._negotiate_version(None)
        assert str(negotiated) == DEFAULT_VERSION

    def test_version_negotiation_higher_than_latest(self):
        """Test version negotiation for version higher than latest."""
        negotiated = self.middleware._negotiate_version("v999")
        assert str(negotiated) == LATEST_VERSION

    def test_version_negotiation_unsupported_lower(self):
        """Test version negotiation for unsupported lower version."""
        # Mock a scenario with v0 (lower than supported)
        negotiated = self.middleware._negotiate_version("v0")
        # Should fall back to default since no compatible version found
        assert str(negotiated) == DEFAULT_VERSION

    def test_version_negotiation_compatible_fallback(self):
        """Test version negotiation finds compatible version."""
        # This tests the compatible version logic path
        negotiated = self.middleware._negotiate_version("v1")
        assert negotiated.is_supported()

    @pytest.mark.asyncio
    async def test_middleware_dispatch_full_cycle(self):
        """Test complete middleware dispatch cycle."""
        app = Mock()
        middleware = VersionNegotiationMiddleware(app, DEFAULT_VERSION)
        
        # Mock request
        request = Mock()
        request.url.path = "/test"
        request.url.query = "version=v1"
        request.headers.get.side_effect = lambda key, default="": ""
        request.state = Mock()
        
        # Mock response
        response = Mock()
        response.headers = {}
        
        # Mock call_next
        async def mock_call_next(req):
            # Verify request state was set
            assert hasattr(req.state, 'api_version')
            assert str(req.state.api_version) == "v1"
            return response
        
        # Execute middleware
        result = await middleware.dispatch(request, mock_call_next)
        
        # Verify response headers were added
        assert "X-API-Version" in result.headers
        assert "X-Supported-Versions" in result.headers
        assert result.headers["X-Supported-Versions"] == ",".join(SUPPORTED_VERSIONS)

    @pytest.mark.asyncio
    async def test_middleware_dispatch_skip_path(self):
        """Test middleware skips processing for certain paths."""
        app = Mock()
        middleware = VersionNegotiationMiddleware(app, DEFAULT_VERSION)
        
        request = Mock()
        request.url.path = "/health"
        
        response = Mock()
        call_next_called = False
        
        async def mock_call_next(req):
            nonlocal call_next_called
            call_next_called = True
            # For skip paths, state should not be modified
            return response
        
        result = await middleware.dispatch(request, mock_call_next)
        
        assert call_next_called
        assert result is response

    @pytest.mark.asyncio  
    async def test_middleware_dispatch_response_without_headers(self):
        """Test middleware handles response without headers attribute."""
        app = Mock()
        middleware = VersionNegotiationMiddleware(app, DEFAULT_VERSION)
        
        request = Mock()
        request.url.path = "/test"
        request.headers.get.side_effect = lambda key, default="": ""
        request.url.query = ""
        request.state = Mock()
        
        # Response without headers attribute
        response = Mock(spec=[])  # Empty spec = no attributes
        
        async def mock_call_next(req):
            return response
        
        # Should not raise exception
        result = await middleware.dispatch(request, mock_call_next)
        assert result is response


class TestVersionDecoratorsComprehensive:
    """Comprehensive testing of version decorators."""

    @pytest.mark.asyncio
    async def test_require_version_success_cases(self):
        """Test require_version decorator success scenarios."""
        @require_version("v1")
        async def endpoint_v1(request):
            return {"version": "v1_endpoint"}
        
        @require_version("v2")
        async def endpoint_v2(request):
            return {"version": "v2_endpoint"}
        
        # Test v1 endpoint with v1 request
        request = Mock()
        request.state.api_version = APIVersion("v1")
        result = await endpoint_v1(request)
        assert result["version"] == "v1_endpoint"
        
        # Test v1 endpoint with v2 request (should work - higher version)
        request.state.api_version = APIVersion("v2")
        result = await endpoint_v1(request)
        assert result["version"] == "v1_endpoint"
        
        # Test v2 endpoint with v2 request
        request.state.api_version = APIVersion("v2")
        result = await endpoint_v2(request)
        assert result["version"] == "v2_endpoint"

    @pytest.mark.asyncio
    async def test_require_version_failure_cases(self):
        """Test require_version decorator failure scenarios."""
        @require_version("v2")
        async def endpoint_requires_v2(request):
            return {"success": True}
        
        # Test v2 endpoint with v1 request (should fail)
        request = Mock()
        request.state.api_version = APIVersion("v1")
        
        with pytest.raises(HTTPException) as exc_info:
            await endpoint_requires_v2(request)
        
        assert exc_info.value.status_code == 400
        assert "requires API version v2" in exc_info.value.detail
        assert "X-Required-Version" in exc_info.value.headers
        assert "X-Current-Version" in exc_info.value.headers

    @pytest.mark.asyncio
    async def test_version_deprecated_all_scenarios(self):
        """Test version_deprecated decorator in all scenarios."""
        
        # Deprecated without removal or alternative
        @version_deprecated("v1", None, None)
        async def deprecated_simple(request):
            return {"type": "simple"}
        
        # Deprecated with removal but no alternative
        @version_deprecated("v1", "v3", None) 
        async def deprecated_with_removal(request):
            return {"type": "with_removal"}
        
        # Deprecated with removal and alternative
        @version_deprecated("v1", "v3", "/new-endpoint")
        async def deprecated_complete(request):
            return {"type": "complete"}
        
        # Test with version that triggers deprecation
        request = Mock()
        request.state.api_version = APIVersion("v1")
        
        with patch('lexgraph_legal_rag.versioning.logger') as mock_logger:
            # Test simple deprecation
            result = await deprecated_simple(request)
            assert result["type"] == "simple"
            mock_logger.warning.assert_called_with(
                "This endpoint is deprecated as of version v1"
            )
            
            # Test deprecation with removal
            result = await deprecated_with_removal(request)
            assert result["type"] == "with_removal"
            expected_msg = "This endpoint is deprecated as of version v1 and will be removed in version v3"
            mock_logger.warning.assert_called_with(expected_msg)
            
            # Test complete deprecation info
            result = await deprecated_complete(request)
            assert result["type"] == "complete"
            expected_msg = "This endpoint is deprecated as of version v1 and will be removed in version v3. Use /new-endpoint instead"
            mock_logger.warning.assert_called_with(expected_msg)

    @pytest.mark.asyncio
    async def test_version_deprecated_not_triggered(self):
        """Test version_deprecated decorator when deprecation not triggered."""
        @version_deprecated("v2", "v3", None)  # Deprecated in v2
        async def not_yet_deprecated(request):
            return {"status": "active"}
        
        # Test with v1 request (below deprecation threshold)
        request = Mock()
        request.state.api_version = APIVersion("v1")
        
        with patch('lexgraph_legal_rag.versioning.logger') as mock_logger:
            result = await not_yet_deprecated(request)
            assert result["status"] == "active"
            # Logger should not be called
            mock_logger.warning.assert_not_called()


class TestVersionedResponseComprehensive:
    """Comprehensive testing of versioned response formatting."""

    def test_v1_response_passthrough(self):
        """Test v1 responses pass through unchanged."""
        test_data = [
            {"simple": "value"},
            {"complex": {"nested": True, "list": [1, 2, 3]}},
            {"with_error": "not_an_error"},
            {"timestamp": 1234567890, "result": 42}
        ]
        
        v1 = APIVersion("v1")
        for data in test_data:
            result = VersionedResponse.format_response(data, v1)
            assert result == data  # Should be unchanged

    def test_v2_success_response_formatting(self):
        """Test v2 success response formatting."""
        test_cases = [
            {"result": 42},
            {"data": [1, 2, 3], "count": 3},
            {"timestamp": 1234567890, "value": "test"}
        ]
        
        v2 = APIVersion("v2")
        for data in test_cases:
            result = VersionedResponse.format_response(data, v2)
            
            assert result["success"] is True
            assert result["data"] == data
            assert "metadata" in result
            assert result["metadata"]["version"] == "v2"
            
            # Timestamp should be preserved if present
            if "timestamp" in data:
                assert result["metadata"]["timestamp"] == data["timestamp"]

    def test_v2_error_response_formatting(self):
        """Test v2 error response formatting."""
        test_cases = [
            {"error": {"detail": "Something went wrong"}},
            {"error": {"code": 500, "message": "Internal error"}, "timestamp": 1234567890}
        ]
        
        v2 = APIVersion("v2")
        for data in test_cases:
            result = VersionedResponse.format_response(data, v2)
            
            assert result["success"] is False
            assert result["error"] == data["error"]
            assert "metadata" in result
            assert result["metadata"]["version"] == "v2"
            
            # Timestamp should be preserved if present
            if "timestamp" in data:
                assert result["metadata"]["timestamp"] == data["timestamp"]

    def test_unsupported_version_fallback(self):
        """Test fallback for unsupported version numbers."""
        data = {"test": "data"}
        
        unsupported_versions = [
            APIVersion("v3"),
            APIVersion("v99"),
            APIVersion("v0"),
            APIVersion("alpha")
        ]
        
        for version in unsupported_versions:
            result = VersionedResponse.format_response(data, version)
            assert result == data  # Should fallback to v1 format (unchanged)


class TestUtilityFunctions:
    """Test utility functions comprehensively."""

    def test_get_request_version_with_version(self):
        """Test get_request_version when version exists."""
        request = Mock()
        request.state.api_version = APIVersion("v2")
        
        version = get_request_version(request)
        assert str(version) == "v2"

    def test_get_request_version_no_version(self):
        """Test get_request_version when no version exists."""
        request = Mock()
        request.state = Mock(spec=[])  # No api_version attribute
        
        version = get_request_version(request)
        assert str(version) == DEFAULT_VERSION

    def test_get_request_version_no_state(self):
        """Test get_request_version when no state exists."""
        request = Mock(spec=[])  # No state attribute
        
        version = get_request_version(request)
        assert str(version) == DEFAULT_VERSION


class TestVersionInfoEndpointComprehensive:
    """Comprehensive testing of version info endpoint."""

    @pytest.mark.asyncio
    async def test_version_info_complete_structure(self):
        """Test complete version info endpoint structure."""
        endpoint = create_version_info_endpoint()
        info = await endpoint()
        
        # Test required fields
        assert "supported_versions" in info
        assert "default_version" in info
        assert "latest_version" in info
        assert "version_negotiation" in info
        assert "deprecations" in info
        
        # Test values
        assert info["supported_versions"] == SUPPORTED_VERSIONS
        assert info["default_version"] == DEFAULT_VERSION
        assert info["latest_version"] == LATEST_VERSION
        
        # Test version negotiation methods
        methods = info["version_negotiation"]["methods"]
        assert len(methods) == 4
        assert any("Accept header" in method for method in methods)
        assert any("URL path" in method for method in methods)
        assert any("Query parameter" in method for method in methods)
        assert any("Custom header" in method for method in methods)
        
        # Test deprecation info
        assert "v1" in info["deprecations"]
        v1_info = info["deprecations"]["v1"]
        assert "deprecated_in" in v1_info
        assert "removed_in" in v1_info
        assert "status" in v1_info
        assert v1_info["status"] == "stable"


class TestVersionPatternsComprehensive:
    """Comprehensive testing of version pattern matching."""

    def test_header_pattern_comprehensive(self):
        """Test header pattern matching comprehensively."""
        pattern = VERSION_PATTERNS["header"]
        
        # Valid patterns
        valid_cases = [
            "application/vnd.lexgraph.v1+json",
            "application/vnd.lexgraph.v2+json",
            "application/vnd.lexgraph.v10+json",
            "text/html, application/vnd.lexgraph.v1+json, */*"  # With other types
        ]
        
        for case in valid_cases:
            match = pattern.search(case)
            assert match is not None, f"Should match: {case}"
            
        # Invalid patterns
        invalid_cases = [
            "application/json",
            "application/vnd.lexgraph+json",  # Missing version
            "application/vnd.lexgraph.+json",  # Empty version
            "text/plain"
        ]
        
        for case in invalid_cases:
            match = pattern.search(case)
            assert match is None, f"Should not match: {case}"

    def test_url_pattern_comprehensive(self):
        """Test URL pattern matching comprehensively."""
        pattern = VERSION_PATTERNS["url"]
        
        # Valid patterns
        valid_cases = [
            "/v1/endpoint",
            "/v2/ping",
            "/v10/test",
            "/api/v1/resource",
            "/v1/",
        ]
        
        for case in valid_cases:
            match = pattern.search(case)
            assert match is not None, f"Should match: {case}"
            
        # Invalid patterns
        invalid_cases = [
            "/endpoint",
            "/version1/endpoint",
            "/v/endpoint",  # No number
            "/V1/endpoint",  # Capital V
            "/1/endpoint",   # No v prefix
        ]
        
        for case in invalid_cases:
            match = pattern.search(case)
            assert match is None, f"Should not match: {case}"

    def test_query_pattern_comprehensive(self):
        """Test query parameter pattern matching comprehensively."""
        pattern = VERSION_PATTERNS["query"]
        
        # Valid patterns
        valid_cases = [
            "version=v1",
            "version=1",
            "version=v2",
            "other=value&version=v1",
            "version=v1&other=value",
            "version=10",
        ]
        
        for case in valid_cases:
            match = pattern.search(case)
            assert match is not None, f"Should match: {case}"
            
        # Invalid patterns
        invalid_cases = [
            "ver=v1",
            "version=latest",
            "version=",
            "api_version=v1",
            "version=v",  # No number after v
        ]
        
        for case in invalid_cases:
            match = pattern.search(case)
            assert match is None, f"Should not match: {case}"


class TestVersioningIntegration:
    """Integration tests for complete versioning workflow."""

    def test_complete_api_versioning_workflow(self):
        """Test complete versioning workflow with real FastAPI app."""
        app = FastAPI()
        app.add_middleware(VersionNegotiationMiddleware, default_version=DEFAULT_VERSION)
        
        @app.get("/test")
        async def test_endpoint(request: Request):
            version = get_request_version(request)
            data = {"message": "success", "requested_version": str(version)}
            return VersionedResponse.format_response(data, version)
        
        @app.get("/v1/specific")
        async def v1_specific(request: Request):
            version = get_request_version(request)
            return {"endpoint": "v1_specific", "version": str(version)}
        
        @app.get("/deprecated")
        @version_deprecated("v1", "v2", "/new-endpoint")
        async def deprecated_endpoint(request: Request):
            return {"status": "deprecated"}
        
        @app.get("/requires-v2")
        @require_version("v2") 
        async def requires_v2_endpoint(request: Request):
            return {"message": "v2 required endpoint"}
        
        client = TestClient(app)
        
        # Test 1: Default version negotiation
        response = client.get("/test")
        assert response.status_code == 200
        assert "X-API-Version" in response.headers
        assert response.headers["X-API-Version"] == DEFAULT_VERSION
        
        # Test 2: Accept header negotiation
        response = client.get("/test", headers={"Accept": "application/vnd.lexgraph.v2+json"})
        assert response.status_code == 200
        assert response.headers["X-API-Version"] == "v2"
        data = response.json()
        # Should be v2 formatted (wrapped)
        assert "success" in data
        assert "data" in data
        
        # Test 3: URL path negotiation
        response = client.get("/v1/specific")
        assert response.status_code == 200
        assert response.headers["X-API-Version"] == "v1"
        
        # Test 4: Query parameter negotiation
        response = client.get("/test?version=v1")
        assert response.status_code == 200
        assert response.headers["X-API-Version"] == "v1"
        
        # Test 5: Custom header negotiation
        response = client.get("/test", headers={"X-API-Version": "v2"})
        assert response.status_code == 200
        assert response.headers["X-API-Version"] == "v2"
        
        # Test 6: Version requirement success
        if "v2" in SUPPORTED_VERSIONS:
            response = client.get("/requires-v2", headers={"X-API-Version": "v2"})
            assert response.status_code == 200
        
        # Test 7: Version requirement failure
        response = client.get("/requires-v2", headers={"X-API-Version": "v1"})
        assert response.status_code == 400
        
        # Test 8: Deprecated endpoint still works
        response = client.get("/deprecated")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "deprecated"

    def test_version_info_endpoint_integration(self):
        """Test version info endpoint integration."""
        app = FastAPI()
        version_endpoint = create_version_info_endpoint()
        app.get("/version")(version_endpoint)
        
        client = TestClient(app)
        response = client.get("/version")
        
        assert response.status_code == 200
        data = response.json()
        
        assert set(data["supported_versions"]) == set(SUPPORTED_VERSIONS)
        assert data["default_version"] == DEFAULT_VERSION
        assert data["latest_version"] == LATEST_VERSION

    def test_middleware_with_different_response_types(self):
        """Test middleware works with different response types."""
        app = FastAPI()
        app.add_middleware(VersionNegotiationMiddleware)
        
        @app.get("/json")
        async def json_response():
            return {"type": "json"}
        
        @app.get("/plain")
        async def plain_response():
            from starlette.responses import PlainTextResponse
            return PlainTextResponse("plain text")
        
        @app.get("/custom")
        async def custom_response():
            return JSONResponse({"type": "custom"})
        
        client = TestClient(app)
        
        # All should work and include version headers
        for endpoint in ["/json", "/plain", "/custom"]:
            response = client.get(endpoint)
            assert response.status_code == 200
            assert "X-API-Version" in response.headers
            assert "X-Supported-Versions" in response.headers


if __name__ == "__main__":
    pytest.main([__file__])