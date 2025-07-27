from __future__ import annotations

from fastapi import APIRouter, FastAPI, Depends, Header, HTTPException, status, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.utils import get_openapi
import os
import time
from collections import deque
from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Any, Optional

import logging

from .sample import add as add_numbers
from .config import validate_environment
from .auth import get_key_manager, setup_test_key_manager
from .metrics import update_memory_metrics, update_cache_metrics, update_index_metrics
from .versioning import (
    VersionNegotiationMiddleware, 
    get_request_version, 
    require_version,
    version_deprecated,
    VersionedResponse,
    create_version_info_endpoint,
    SUPPORTED_VERSIONS as VERSIONING_SUPPORTED_VERSIONS,
    DEFAULT_VERSION
)
from .correlation import CorrelationIdMiddleware, get_correlation_id
from .monitoring import HTTPMonitoringMiddleware

"""FastAPI application with API key auth and rate limiting."""

SUPPORTED_VERSIONS = VERSIONING_SUPPORTED_VERSIONS

API_KEY_ENV = "API_KEY"  # pragma: allowlist secret
RATE_LIMIT = 60  # requests per minute


def _get_cors_origins(test_mode: bool) -> list[str]:
    """Get CORS allowed origins based on environment and mode."""
    # Check for explicit configuration first
    cors_origins = os.environ.get("CORS_ALLOWED_ORIGINS")
    if cors_origins:
        return [origin.strip() for origin in cors_origins.split(",")]
    
    # Default secure configuration based on mode
    if test_mode:
        # Development/test mode: allow common development origins
        return [
            "http://localhost:3000",
            "http://localhost:8080", 
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8080"
        ]
    else:
        # Production mode: no origins allowed by default (must be explicitly configured)
        return []


def _get_cors_methods(test_mode: bool) -> list[str]:
    """Get CORS allowed methods based on environment and mode."""
    cors_methods = os.environ.get("CORS_ALLOWED_METHODS")
    if cors_methods:
        return [method.strip() for method in cors_methods.split(",")]
    
    # Default secure methods
    if test_mode:
        return ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    else:
        return ["GET", "POST", "OPTIONS"]  # Minimal set for production


def _get_cors_headers(test_mode: bool) -> list[str]:
    """Get CORS allowed headers based on environment and mode."""
    cors_headers = os.environ.get("CORS_ALLOWED_HEADERS")
    if cors_headers:
        return [header.strip() for header in cors_headers.split(",")]
    
    # Default secure headers
    return [
        "Accept",
        "Accept-Language", 
        "Content-Language",
        "Content-Type",
        "X-API-Key",
        "X-API-Version",
        "X-Correlation-ID"
    ]


def verify_api_key(x_api_key: str, api_key: str | None = None) -> None:
    # Use the key manager for validation if available, fallback to legacy method
    try:
        key_manager = get_key_manager()
        
        if key_manager.get_active_key_count() > 0:
            # Use key manager for validation with rate limiting
            if not key_manager.is_valid_key(x_api_key):
                logger.warning("invalid API key attempt via key manager", extra={"provided": x_api_key[:8] + "..."})
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key"
                )
            
            # Check per-key rate limit
            if not key_manager.check_rate_limit(x_api_key):
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded for this API key"
                )
        else:
            # Fallback to legacy validation
            if api_key is None:
                api_key = os.environ.get(API_KEY_ENV)
            if not api_key or x_api_key != api_key:
                logger.warning("invalid API key attempt via legacy", extra={"provided": x_api_key[:8] + "..."})
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key"
                )
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # In case of any auth system failure, fall back to direct comparison
        if api_key is None:
            api_key = os.environ.get(API_KEY_ENV)
        if not api_key or x_api_key != api_key:
            logger.warning("invalid API key attempt via fallback", extra={"provided": x_api_key[:8] + "...", "error": str(e)})
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key"
            )


def enforce_rate_limit(request_times: deque[float], limit: int = RATE_LIMIT) -> None:
    now = time.time()
    while request_times and now - request_times[0] > 60:
        request_times.popleft()
    if len(request_times) >= limit:
        logger.warning("rate limit exceeded")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
        )
    request_times.append(now)


logger = logging.getLogger(__name__)


class PingResponse(BaseModel):
    """Response model for ping endpoint."""
    version: str = Field(..., description="API version", examples=["v1"])
    ping: str = Field(..., description="Ping response", examples=["pong"])
    timestamp: Optional[float] = Field(None, description="Response timestamp")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{
                "version": "v1",
                "ping": "pong",
                "timestamp": 1234567890.123
            }]
        }
    )


class AddResponse(BaseModel):
    """Response model for addition endpoint."""
    result: int = Field(..., description="Sum of the two integers", examples=[42])

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"result": 42}]
        }
    )


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str = Field(..., description="Health status", examples=["healthy"])
    version: str = Field(..., description="API version", examples=["v1"])
    checks: Dict[str, Any] = Field(..., description="Individual health checks")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{
                "status": "healthy",
                "version": "v1",
                "checks": {
                    "api_key_configured": True,
                    "supported_versions": ["v1", "v2"],
                    "timestamp": 1234567890.123
                }
            }]
        }
    )


class ReadinessResponse(BaseModel):
    """Response model for readiness check endpoint."""
    ready: bool = Field(..., description="Whether the service is ready to handle requests")
    checks: Dict[str, Any] = Field(..., description="Readiness check results")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{
                "ready": True,
                "checks": {
                    "api_key": {"status": "pass"},
                    "memory": {"status": "pass", "usage_percent": 45.2},
                    "cache": {"status": "pass", "hit_rate": 0.85},
                    "external_services": {"status": "pass", "message": "All services available"}
                }
            }]
        }
    )


class KeyRotationRequest(BaseModel):
    """Request model for API key rotation."""
    new_primary_key: str = Field(
        ..., 
        min_length=8,
        description="New API key to add to the active set",
        examples=["new-secure-api-key-123"]
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"new_primary_key": "new-secure-api-key-123"}]
        }
    )


class KeyRevocationRequest(BaseModel):
    """Request model for API key revocation."""
    api_key: str = Field(
        ...,
        min_length=8, 
        description="API key to revoke",
        examples=["old-api-key-to-revoke"]
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"api_key": "old-api-key-to-revoke"}]
        }
    )


class KeyManagementResponse(BaseModel):
    """Response model for key management operations."""
    message: str = Field(..., description="Operation result message")
    rotation_info: Dict[str, Any] = Field(..., description="Key rotation statistics")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{
                "message": "Key rotation completed successfully",
                "rotation_info": {
                    "active_keys": 3,
                    "revoked_keys": 1,
                    "last_rotation": 1234567890.123,
                    "days_since_rotation": 0.001
                }
            }]
        }
    )


class VersionInfo(BaseModel):
    """Response model for version information."""
    supported_versions: list[str] = Field(..., description="List of supported API versions")
    default_version: str = Field(..., description="Default API version")
    latest_version: str = Field(..., description="Latest API version")
    current_version: str = Field(..., description="Currently negotiated version")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{
                "supported_versions": ["v1", "v2"],
                "default_version": "v1",
                "latest_version": "v2",
                "current_version": "v1"
            }]
        }
    )


def create_api(
    version: str = SUPPORTED_VERSIONS[0],
    api_key: str | None = None,
    rate_limit: int = RATE_LIMIT,
    enable_docs: bool = True,
    test_mode: bool = False,
    config = None,
) -> FastAPI:
    """Return a FastAPI app configured for the given API ``version``."""
    if version not in SUPPORTED_VERSIONS:
        raise ValueError(f"Unsupported API version: {version}")

    if api_key is None:
        api_key = os.environ.get(API_KEY_ENV)
    if not api_key and not test_mode:
        raise ValueError("API key not provided")
    
    # Get or validate configuration
    if config is None:
        config = validate_environment(allow_test_mode=test_mode)
    
    # Setup test mode key management if needed
    if test_mode and api_key:
        setup_test_key_manager(api_key)

    # Custom OpenAPI configuration
    app = FastAPI(
        title="LexGraph Legal RAG API",
        version=version,
        description="""
        Multi-agent retrieval system for legal document analysis with cited passages.
        
        ## Features
        
        * **Multi-Agent Architecture**: Specialized tools for retrieval, summarization, and explanation
        * **Legal Document Search**: Vector-based and semantic search capabilities
        * **Citation Generation**: Precise source references for all responses
        * **Performance Monitoring**: Built-in metrics and health checks
        * **API Key Management**: Secure authentication with key rotation support
        
        ## Version Negotiation
        
        This API supports multiple versions. You can specify the desired version using:
        
        * **Accept Header**: `Accept: application/vnd.lexgraph.v1+json`
        * **URL Path**: `/v1/endpoint` or `/v2/endpoint`
        * **Query Parameter**: `?version=v1`
        * **Custom Header**: `X-API-Version: v1`
        
        ## Authentication
        
        All endpoints require an API key passed in the `X-API-Key` header.
        """,
        contact={
            "name": "LexGraph Legal RAG Support",
            "email": "support@lexgraph.com",
        },
        license_info={
            "name": "MIT License",
            "url": "https://opensource.org/licenses/MIT",
        },
        docs_url="/docs" if enable_docs else None,
        redoc_url="/redoc" if enable_docs else None,
        openapi_url="/openapi.json" if enable_docs else None,
    )

    # Add CORS middleware with security-focused configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.allowed_origins,
        allow_credentials=True,
        allow_methods=_get_cors_methods(test_mode),
        allow_headers=_get_cors_headers(test_mode),
        expose_headers=["X-Correlation-ID", "X-API-Version"],
        max_age=600,  # Cache preflight for 10 minutes
    )
    
    # Add HTTP monitoring middleware for alerting
    app.add_middleware(HTTPMonitoringMiddleware)
    
    # Add correlation ID middleware for request tracing
    app.add_middleware(CorrelationIdMiddleware)
    
    # Add version negotiation middleware
    app.add_middleware(VersionNegotiationMiddleware, default_version=DEFAULT_VERSION)

    request_times: deque[float] = deque()

    def register_routes(router: APIRouter) -> None:
        @router.get("/ping", response_model=PingResponse, tags=["Health"])
        async def ping(request: Request) -> PingResponse:
            """Basic connectivity test endpoint."""
            correlation_id = get_correlation_id()
            logger.info("/ping endpoint called", extra={
                "endpoint": "/ping",
                "correlation_id": correlation_id,
                "method": "GET"
            })
            current_version = get_request_version(request)
            response_data = {
                "version": str(current_version),
                "ping": "pong",
                "timestamp": time.time()
            }
            logger.debug("/ping response prepared", extra={
                "response_version": str(current_version),
                "correlation_id": correlation_id
            })
            return PingResponse(**response_data)

        @router.get("/add", response_model=AddResponse, tags=["Utilities"])
        async def add(
            request: Request,
            a: int = Query(..., description="First integer", examples=[5]),
            b: int = Query(..., description="Second integer", examples=[7])
        ) -> AddResponse:
            """Add two integers together.
            
            This is a simple arithmetic endpoint for testing API functionality.
            """
            logger.debug("/add called with a=%s b=%s", a, b)
            current_version = get_request_version(request)
            result = add_numbers(a, b)
            
            # Version-specific response formatting
            response_data = {"result": result}
            formatted_response = VersionedResponse.format_response(response_data, current_version)
            
            if current_version.version == "v2":
                # For v2, we need to extract the result from the formatted response 
                # to match the AddResponse model structure
                return AddResponse(result=formatted_response.get("data", {}).get("result", result))
            else:
                return AddResponse(result=result)
        
        @router.get("/version", response_model=VersionInfo, tags=["API Info"])
        async def version_info(request: Request) -> VersionInfo:
            """Get API version information and negotiation details."""
            current_version = get_request_version(request)
            version_endpoint = create_version_info_endpoint()
            info = await version_endpoint()
            
            return VersionInfo(
                supported_versions=info["supported_versions"],
                default_version=info["default_version"],
                latest_version=info["latest_version"],
                current_version=str(current_version)
            )

    def register_health_routes(app: FastAPI) -> None:
        """Register health check endpoints without authentication."""
        
        @app.get("/health", response_model=HealthResponse, tags=["Health"])
        async def health() -> HealthResponse:
            """Health check endpoint - returns basic service status."""
            checks = {
                "api_key_configured": bool(api_key),
                "supported_versions": SUPPORTED_VERSIONS,
                "timestamp": time.time()
            }
            return HealthResponse(
                status="healthy",
                version=version,
                checks=checks
            )

        @app.get("/ready", response_model=ReadinessResponse, tags=["Health"])
        async def readiness() -> ReadinessResponse:
            """Readiness check endpoint - checks if service is ready to handle requests."""
            checks = {}
            ready = True
            
            # Check API key
            if not api_key:
                checks["api_key"] = {"status": "fail", "message": "API key not configured"}
                ready = False
            else:
                checks["api_key"] = {"status": "pass"}
            
            # Check memory usage (basic check)
            import psutil
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > 90:
                checks["memory"] = {"status": "warn", "usage_percent": memory_percent}
            else:
                checks["memory"] = {"status": "pass", "usage_percent": memory_percent}
            
            # Update memory metrics
            update_memory_metrics()
            
            # Check cache status
            try:
                from .cache import get_query_cache
                cache = get_query_cache()
                cache_stats = cache.get_stats()
                update_cache_metrics(cache_stats)
                checks["cache"] = {"status": "pass", "hit_rate": cache_stats.get("hit_rate", 0.0)}
            except Exception as e:
                checks["cache"] = {"status": "warn", "message": f"Cache check failed: {e}"}
            
            # Check external service circuit breakers (if any configured)
            try:
                from .http_client import ResilientHTTPClient
                # This would be expanded when actual external services are configured
                checks["external_services"] = {"status": "pass", "message": "No external services configured"}
            except Exception as e:
                checks["external_services"] = {"status": "warn", "message": f"Circuit breaker check failed: {e}"}
            
            return ReadinessResponse(ready=ready, checks=checks)

    def register_admin_routes(router: APIRouter) -> None:
        """Register admin endpoints for key management."""
        
        @router.post("/admin/rotate-keys", response_model=KeyManagementResponse, tags=["Admin"])
        async def rotate_keys(request: KeyRotationRequest) -> KeyManagementResponse:
            """Rotate API keys - requires existing valid API key."""
            key_manager = get_key_manager()
            key_manager.rotate_keys(request.new_primary_key)
            rotation_info = key_manager.get_rotation_info()
            
            logger.info("API key rotation completed")
            return KeyManagementResponse(
                message="Key rotation completed successfully",
                rotation_info=rotation_info
            )

        @router.post("/admin/revoke-key", response_model=KeyManagementResponse, tags=["Admin"])
        async def revoke_key(request: KeyRevocationRequest) -> KeyManagementResponse:
            """Revoke a specific API key."""
            key_manager = get_key_manager()
            key_manager.revoke_key(request.api_key)
            rotation_info = key_manager.get_rotation_info()
            
            logger.info("API key revocation completed")
            return KeyManagementResponse(
                message="Key revocation completed successfully",
                rotation_info=rotation_info
            )

        @router.get("/admin/key-status", response_model=KeyManagementResponse, tags=["Admin"])
        async def key_status() -> KeyManagementResponse:
            """Get current key management status."""
            key_manager = get_key_manager()
            rotation_info = key_manager.get_rotation_info()
            
            return KeyManagementResponse(
                message="Key status retrieved successfully",
                rotation_info=rotation_info
            )

        @router.get("/admin/metrics", tags=["Admin", "Monitoring"])
        async def get_metrics_summary() -> Dict[str, Any]:
            """Get summary of application metrics and performance statistics."""
            # Update metrics before returning
            update_memory_metrics()
            
            # Collect cache metrics
            try:
                from .cache import get_query_cache
                cache = get_query_cache()
                cache_stats = cache.get_stats()
                update_cache_metrics(cache_stats)
            except Exception:
                cache_stats = {}
            
            # Collect index pool stats if available
            try:
                from .faiss_index import FaissVectorIndex
                # This would be more sophisticated in a real implementation
                pool_stats = {"available": 0, "in_use": 0, "has_master": False}
            except Exception:
                pool_stats = {}
            
            return {
                "cache": cache_stats,
                "index_pool": pool_stats,
                "memory": {
                    "usage_percent": psutil.virtual_memory().percent if 'psutil' in globals() else None
                },
                "key_management": get_key_manager().get_rotation_info(),
                "timestamp": time.time(),
            }

    def auth_dep(x_api_key: str | None = Header(None)) -> None:
        if x_api_key is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, 
                detail="X-API-Key header is required"
            )
        verify_api_key(x_api_key, api_key)

    def rate_dep() -> None:
        enforce_rate_limit(request_times, rate_limit)

    # Register health endpoints without authentication
    register_health_routes(app)
    
    dependencies = [Depends(auth_dep), Depends(rate_dep)]
    versioned_router = APIRouter(prefix=f"/{version}", dependencies=dependencies)
    register_routes(versioned_router)
    register_admin_routes(versioned_router)  # Admin routes with authentication
    app.include_router(versioned_router)

    # Always include default routes (without version prefix) for backward compatibility
    # and to support the expected test behavior
    default_router = APIRouter(dependencies=dependencies)
    register_routes(default_router)
    register_admin_routes(default_router)  # Admin routes with authentication
    app.include_router(default_router)

    # Validate configuration at startup
    validate_environment(allow_test_mode=test_mode)

    # Custom OpenAPI schema
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        
        openapi_schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        
        # Add security schemes
        openapi_schema["components"]["securitySchemes"] = {
            "ApiKeyAuth": {
                "type": "apiKey",
                "in": "header",
                "name": "X-API-Key",
                "description": "API key for authentication"
            },
            "VersionHeader": {
                "type": "apiKey",
                "in": "header", 
                "name": "X-API-Version",
                "description": "API version specification"
            }
        }
        
        # Add global security requirement
        openapi_schema["security"] = [{"ApiKeyAuth": []}]
        
        # Add tags
        openapi_schema["tags"] = [
            {
                "name": "Health",
                "description": "Health check and status endpoints"
            },
            {
                "name": "Utilities", 
                "description": "Utility endpoints for testing"
            },
            {
                "name": "API Info",
                "description": "API version and information endpoints"
            },
            {
                "name": "Admin",
                "description": "Administrative endpoints for key management"
            },
            {
                "name": "Monitoring",
                "description": "Monitoring and metrics endpoints"
            }
        ]
        
        # Add version info to schema
        openapi_schema["info"]["x-api-versions"] = {
            "supported": list(SUPPORTED_VERSIONS),
            "default": DEFAULT_VERSION,
            "latest": version
        }
        
        app.openapi_schema = openapi_schema
        return app.openapi_schema

    app.openapi = custom_openapi

    return app
