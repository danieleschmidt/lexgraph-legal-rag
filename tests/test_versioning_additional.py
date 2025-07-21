"""Additional tests for versioning module to achieve 100% coverage.

This test suite targets the remaining uncovered lines in the versioning module.
"""

import pytest
from unittest.mock import Mock, AsyncMock
from fastapi import HTTPException

from lexgraph_legal_rag.versioning import (
    APIVersion,
    VersionNegotiationMiddleware,
    get_request_version,
    require_version,
    version_deprecated,
    VersionedResponse,
    DEFAULT_VERSION,
    SUPPORTED_VERSIONS,
)


class TestAPIVersionEdgeCases:
    """Test edge cases in APIVersion class."""

    def test_version_string_comparison(self):
        """Test version comparison with strings."""
        v1 = APIVersion("v1")
        
        # Test equality with string
        assert v1 == "v1"
        assert v1 != "v2"
        
        # Test less than with string
        assert v1 < "v2"
        assert not (v1 < "v1")

    def test_version_less_than_equal(self):
        """Test <= operator."""
        v1 = APIVersion("v1")
        v2 = APIVersion("v2")
        
        assert v1 <= v2
        assert v1 <= v1  # Equal case
        
    def test_version_from_non_digit(self):
        """Test version creation from non-digit strings."""
        v_alpha = APIVersion("v-alpha")
        assert v_alpha.major == 1  # Should default to 1 for non-digit


class TestMiddlewareEdgeCases:
    """Test edge cases in version negotiation middleware."""

    def test_middleware_skip_paths(self):
        """Test all skip paths in middleware."""
        middleware = VersionNegotiationMiddleware(None)
        
        skip_paths = ["/health", "/ready", "/metrics", "/docs", "/openapi.json"]
        for path in skip_paths:
            assert middleware._should_skip_versioning(path)
            
        # Test non-skip path
        assert not middleware._should_skip_versioning("/api/test")

    def test_extract_version_header_normalization(self):
        """Test header version normalization."""
        middleware = VersionNegotiationMiddleware(None)
        
        # Mock request with custom header without 'v' prefix
        request = Mock()
        request.headers.get.side_effect = lambda key, default="": {
            "accept": "",
            "X-API-Version": "2"  # No 'v' prefix
        }.get(key, default)
        request.url.path = "/test"
        request.url.query = ""
        
        version = middleware._extract_version(request)
        assert version == "v2"  # Should be normalized to v2

    def test_negotiate_version_unsupported_higher(self):
        """Test version negotiation for unsupported higher version."""
        middleware = VersionNegotiationMiddleware(None)
        
        # Request version higher than latest supported
        negotiated = middleware._negotiate_version("v99")
        
        from lexgraph_legal_rag.versioning import LATEST_VERSION
        assert str(negotiated) == LATEST_VERSION

    def test_negotiate_version_compatible_fallback(self):
        """Test compatible version fallback logic."""
        middleware = VersionNegotiationMiddleware(None)
        
        # Mock scenario where we need compatible version fallback
        # Request v1.5 (doesn't exist) should fall back to v1
        negotiated = middleware._negotiate_version("v1")
        assert negotiated.is_supported()

    def test_negotiate_version_no_compatible(self):
        """Test fallback when no compatible version found."""
        middleware = VersionNegotiationMiddleware(None)
        
        # Mock a scenario where version comparison would fail
        # This is difficult to trigger naturally, but tests the fallback path
        with pytest.MonkeyPatch().context() as mp:
            # Temporarily modify supported versions to test edge case
            mp.setattr("lexgraph_legal_rag.versioning.SUPPORTED_VERSIONS", ["v3"])
            negotiated = middleware._negotiate_version("v1")
            # Should fall back to default version
            assert str(negotiated) == DEFAULT_VERSION


class TestDecoratorEdgeCases:
    """Test decorator edge cases and error paths."""

    def test_require_version_success(self):
        """Test require_version decorator when version is sufficient."""
        @require_version("v1")
        async def test_endpoint(request):
            return {"success": True}
        
        # Mock request with sufficient version
        request = Mock()
        request.state.api_version = APIVersion("v2")  # Higher than required v1
        
        import asyncio
        result = asyncio.run(test_endpoint(request))
        assert result["success"] is True

    def test_require_version_exact_match(self):
        """Test require_version decorator with exact version match."""
        @require_version("v1")
        async def test_endpoint(request):
            return {"success": True}
        
        request = Mock()
        request.state.api_version = APIVersion("v1")  # Exact match
        
        import asyncio
        result = asyncio.run(test_endpoint(request))
        assert result["success"] is True

    def test_version_deprecated_no_removal(self):
        """Test version_deprecated decorator without removal version."""
        @version_deprecated("v1", None, None)
        async def deprecated_endpoint(request):
            return {"deprecated": True}
        
        request = Mock()
        request.state.api_version = APIVersion("v1")
        
        import asyncio
        result = asyncio.run(deprecated_endpoint(request))
        assert result["deprecated"] is True

    def test_version_deprecated_with_alternative(self):
        """Test version_deprecated decorator with alternative specified."""
        @version_deprecated("v1", "v3", "/new-endpoint")
        async def deprecated_endpoint(request):
            return {"deprecated": True}
        
        request = Mock()
        request.state.api_version = APIVersion("v1")
        
        import asyncio
        result = asyncio.run(deprecated_endpoint(request))
        assert result["deprecated"] is True

    def test_version_deprecated_below_threshold(self):
        """Test deprecated decorator when version is below deprecation threshold."""
        @version_deprecated("v2", "v3", None)  # Deprecated in v2
        async def not_deprecated_yet(request):
            return {"ok": True}
        
        request = Mock()
        request.state.api_version = APIVersion("v1")  # Below v2, so not deprecated
        
        import asyncio
        result = asyncio.run(not_deprecated_yet(request))
        assert result["ok"] is True


class TestVersionedResponseEdgeCases:
    """Test edge cases in versioned response formatting."""

    def test_versioned_response_v2_with_timestamp(self):
        """Test v2 response format preserves timestamp."""
        data = {"result": 42, "timestamp": 1234567890}
        version = APIVersion("v2")
        
        formatted = VersionedResponse.format_response(data, version)
        
        assert formatted["success"] is True
        assert formatted["metadata"]["timestamp"] == 1234567890

    def test_versioned_response_v2_error_with_timestamp(self):
        """Test v2 error response format preserves timestamp."""
        data = {"error": {"detail": "Error"}, "timestamp": 1234567890}
        version = APIVersion("v2")
        
        formatted = VersionedResponse.format_response(data, version)
        
        assert formatted["success"] is False
        assert formatted["metadata"]["timestamp"] == 1234567890

    def test_versioned_response_unknown_version(self):
        """Test versioned response falls back for unknown versions."""
        data = {"result": 42}
        version = APIVersion("v99")
        
        formatted = VersionedResponse.format_response(data, version)
        
        # Should fall back to v1 format (unchanged data)
        assert formatted == data


class TestGetRequestVersion:
    """Test get_request_version utility function."""

    def test_get_request_version_no_state(self):
        """Test get_request_version when no version in request state."""
        request = Mock()
        # Mock request without api_version attribute
        request.state = Mock(spec=[])  # Empty spec means no attributes
        
        version = get_request_version(request)
        assert str(version) == DEFAULT_VERSION

    def test_get_request_version_with_state(self):
        """Test get_request_version when version exists in request state."""
        request = Mock()
        request.state.api_version = APIVersion("v2")
        
        version = get_request_version(request)
        assert str(version) == "v2"


class TestVersionNegotiationHeaders:
    """Test version negotiation through different header mechanisms."""

    def test_middleware_response_headers(self):
        """Test middleware adds version headers to response."""
        import asyncio
        
        middleware = VersionNegotiationMiddleware(None)
        
        # Mock request and response
        request = Mock()
        request.url.path = "/test"
        request.headers.get.side_effect = lambda key, default="": ""
        request.url.query = ""
        request.state = Mock()
        
        response = Mock()
        response.headers = {}
        
        async def mock_call_next(req):
            return response
        
        # Run the middleware dispatch
        result = asyncio.run(middleware.dispatch(request, mock_call_next))
        
        # Should have version headers
        assert "X-API-Version" in result.headers
        assert "X-Supported-Versions" in result.headers

    def test_middleware_response_without_headers_attribute(self):
        """Test middleware handles responses without headers attribute."""
        import asyncio
        
        middleware = VersionNegotiationMiddleware(None)
        
        request = Mock()
        request.url.path = "/test"
        request.headers.get.side_effect = lambda key, default="": ""
        request.url.query = ""
        request.state = Mock()
        
        # Response without headers attribute
        response = Mock(spec=[])  # No headers attribute
        
        async def mock_call_next(req):
            return response
        
        # Should not raise exception
        result = asyncio.run(middleware.dispatch(request, mock_call_next))
        assert result is response