"""API versioning middleware and utilities."""

from __future__ import annotations

import re
from typing import Optional, Tuple
from fastapi import Request, HTTPException, status
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.responses import Response
import logging

logger = logging.getLogger(__name__)

# Supported API versions
SUPPORTED_VERSIONS = ("v1", "v2")
DEFAULT_VERSION = "v1"
LATEST_VERSION = "v1"  # Keep v1 as latest for now until v2 is fully implemented

# Version patterns
VERSION_PATTERNS = {
    "header": re.compile(r"application/vnd\.lexgraph\.([^+]+)\+json"),
    "url": re.compile(r"/v(\d+)/"),
    "query": re.compile(r"version=v?(\d+)"),
}


class APIVersion:
    """Represents an API version with comparison capabilities."""
    
    def __init__(self, version: str) -> None:
        self.version = version
        self.major = int(version.lstrip('v')) if version.lstrip('v').isdigit() else 1
    
    def __str__(self) -> str:
        return self.version
    
    def __eq__(self, other) -> bool:
        if isinstance(other, APIVersion):
            return self.version == other.version
        return self.version == str(other)
    
    def __lt__(self, other) -> bool:
        if isinstance(other, APIVersion):
            return self.major < other.major
        return self.major < int(str(other).lstrip('v'))
    
    def __le__(self, other) -> bool:
        return self == other or self < other
    
    def is_supported(self) -> bool:
        """Check if this version is supported."""
        return self.version in SUPPORTED_VERSIONS


class VersionNegotiationMiddleware(BaseHTTPMiddleware):
    """Middleware for API version negotiation."""
    
    def __init__(self, app, default_version: str = DEFAULT_VERSION):
        super().__init__(app)
        self.default_version = default_version
    
    async def dispatch(self, request: Request, call_next):
        """Process request and negotiate API version."""
        
        # Skip version negotiation for certain paths
        if self._should_skip_versioning(request.url.path):
            return await call_next(request)
        
        # Determine requested version
        requested_version = self._extract_version(request)
        
        # Validate and set version
        version = self._negotiate_version(requested_version)
        
        # Add version info to request state
        request.state.api_version = version
        
        # Add version headers to response
        response = await call_next(request)
        if hasattr(response, 'headers'):
            response.headers["X-API-Version"] = str(version)
            response.headers["X-Supported-Versions"] = ",".join(SUPPORTED_VERSIONS)
        
        return response
    
    def _should_skip_versioning(self, path: str) -> bool:
        """Determine if path should skip version negotiation."""
        skip_paths = ["/health", "/ready", "/metrics", "/docs", "/openapi.json"]
        return any(path.startswith(skip_path) for skip_path in skip_paths)
    
    def _extract_version(self, request: Request) -> Optional[str]:
        """Extract version from request headers, URL, or query parameters."""
        
        # 1. Check Accept header (preferred method)
        accept_header = request.headers.get("accept", "")
        match = VERSION_PATTERNS["header"].search(accept_header)
        if match:
            return f"v{match.group(1)}"
        
        # 2. Check URL path
        match = VERSION_PATTERNS["url"].search(str(request.url.path))
        if match:
            return f"v{match.group(1)}"
        
        # 3. Check query parameter
        query_string = str(request.url.query)
        match = VERSION_PATTERNS["query"].search(query_string)
        if match:
            return f"v{match.group(1)}"
        
        # 4. Check custom header
        version_header = request.headers.get("X-API-Version")
        if version_header:
            # Normalize version format
            if not version_header.startswith('v'):
                version_header = f"v{version_header}"
            return version_header
        
        return None
    
    def _negotiate_version(self, requested_version: Optional[str]) -> APIVersion:
        """Negotiate the API version to use."""
        
        if not requested_version:
            logger.debug("No version specified, using default: %s", self.default_version)
            return APIVersion(self.default_version)
        
        requested = APIVersion(requested_version)
        
        # Check if exact version is supported
        if requested.is_supported():
            logger.debug("Using requested version: %s", requested_version)
            return requested
        
        # Find closest supported version
        supported_versions = [APIVersion(v) for v in SUPPORTED_VERSIONS]
        
        # If requested version is higher than latest, use latest
        latest = APIVersion(LATEST_VERSION)
        if requested > latest:
            logger.warning(
                "Requested version %s not supported, using latest: %s",
                requested_version, LATEST_VERSION
            )
            return latest
        
        # Find highest supported version that's <= requested
        compatible_versions = [v for v in supported_versions if v <= requested]
        if compatible_versions:
            chosen = max(compatible_versions)
            if chosen != requested:
                logger.warning(
                    "Requested version %s not supported, using compatible: %s",
                    requested_version, chosen.version
                )
            return chosen
        
        # Fallback to default
        logger.warning(
            "No compatible version found for %s, using default: %s",
            requested_version, self.default_version
        )
        return APIVersion(self.default_version)


def get_request_version(request: Request) -> APIVersion:
    """Get the negotiated API version from request state."""
    if hasattr(request.state, 'api_version'):
        return request.state.api_version
    return APIVersion(DEFAULT_VERSION)


def require_version(min_version: str):
    """Decorator to require minimum API version for endpoint."""
    def decorator(func):
        async def wrapper(request: Request, *args, **kwargs):
            current_version = get_request_version(request)
            min_ver = APIVersion(min_version)
            
            if current_version < min_ver:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"This endpoint requires API version {min_version} or higher. "
                           f"Current version: {current_version}",
                    headers={
                        "X-Required-Version": min_version,
                        "X-Current-Version": str(current_version),
                    }
                )
            
            return await func(request, *args, **kwargs)
        return wrapper
    return decorator


def version_deprecated(deprecated_in: str, removed_in: str, alternative: str = None):
    """Decorator to mark endpoint as deprecated in specific version."""
    def decorator(func):
        async def wrapper(request: Request, *args, **kwargs):
            current_version = get_request_version(request)
            deprecated_ver = APIVersion(deprecated_in)
            
            if current_version >= deprecated_ver:
                warning_msg = f"This endpoint is deprecated as of version {deprecated_in}"
                if removed_in:
                    warning_msg += f" and will be removed in version {removed_in}"
                if alternative:
                    warning_msg += f". Use {alternative} instead"
                
                logger.warning(warning_msg)
                # Add deprecation warning to response headers would be done in middleware
            
            return await func(request, *args, **kwargs)
        return wrapper
    return decorator


class VersionedResponse:
    """Utility for creating version-specific response formats."""
    
    @staticmethod
    def format_response(data: dict, version: APIVersion) -> dict:
        """Format response data based on API version."""
        
        if version.version == "v1":
            return data
        
        elif version.version == "v2":
            # V2 adds metadata wrapper and standardized error format
            if "error" in data:
                return {
                    "success": False,
                    "error": data,
                    "metadata": {
                        "version": str(version),
                        "timestamp": data.get("timestamp"),
                    }
                }
            else:
                return {
                    "success": True,
                    "data": data,
                    "metadata": {
                        "version": str(version),
                        "timestamp": data.get("timestamp"),
                    }
                }
        
        # Fallback to v1 format
        return data


def create_version_info_endpoint():
    """Create endpoint that returns version information."""
    
    async def version_info():
        """Get API version information."""
        return {
            "supported_versions": SUPPORTED_VERSIONS,
            "default_version": DEFAULT_VERSION,
            "latest_version": LATEST_VERSION,
            "version_negotiation": {
                "methods": [
                    "Accept header: application/vnd.lexgraph.v1+json",
                    "URL path: /v1/endpoint",
                    "Query parameter: ?version=v1",
                    "Custom header: X-API-Version: v1"
                ]
            },
            "deprecations": {
                "v1": {
                    "deprecated_in": None,
                    "removed_in": None,
                    "status": "stable"
                }
            }
        }
    
    return version_info