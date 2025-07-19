from __future__ import annotations

from fastapi import APIRouter, FastAPI, Depends, Header, HTTPException, status
import os
import time
from collections import deque
from pydantic import BaseModel
from typing import Dict, Any

import logging

from .sample import add as add_numbers
from .config import validate_environment
from .auth import get_key_manager
from .metrics import update_memory_metrics, update_cache_metrics, update_index_metrics

"""FastAPI application with API key auth and rate limiting."""

SUPPORTED_VERSIONS = ("v1",)

API_KEY_ENV = "API_KEY"  # pragma: allowlist secret
RATE_LIMIT = 60  # requests per minute


def verify_api_key(x_api_key: str = Header(...), api_key: str | None = None) -> None:
    # Use the key manager for validation if available, fallback to legacy method
    key_manager = get_key_manager()
    
    if key_manager.get_active_key_count() > 0:
        # Use key manager for validation
        if not key_manager.is_valid_key(x_api_key):
            logger.warning("invalid API key attempt via key manager", extra={"provided": x_api_key[:8] + "..."})
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key"
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
    version: str
    ping: str


class AddResponse(BaseModel):
    result: int


class HealthResponse(BaseModel):
    status: str
    version: str
    checks: Dict[str, Any]


class ReadinessResponse(BaseModel):
    ready: bool
    checks: Dict[str, Any]


class KeyRotationRequest(BaseModel):
    new_primary_key: str


class KeyRevocationRequest(BaseModel):
    api_key: str


class KeyManagementResponse(BaseModel):
    message: str
    rotation_info: Dict[str, Any]


def create_api(
    version: str = SUPPORTED_VERSIONS[0],
    api_key: str | None = None,
    rate_limit: int = RATE_LIMIT,
) -> FastAPI:
    """Return a FastAPI app configured for the given API ``version``."""
    if version not in SUPPORTED_VERSIONS:
        raise ValueError(f"Unsupported API version: {version}")

    if api_key is None:
        api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        raise ValueError("API key not provided")

    app = FastAPI(title="LexGraph Legal RAG", version=version)

    request_times: deque[float] = deque()

    def register_routes(router: APIRouter) -> None:
        @router.get("/ping", response_model=PingResponse)
        async def ping() -> PingResponse:
            logger.debug("/ping called")
            return PingResponse(version=version, ping="pong")

        @router.get("/add", response_model=AddResponse)
        async def add(a: int, b: int) -> AddResponse:
            """Return the sum of two integers."""
            logger.debug("/add called with a=%s b=%s", a, b)
            return AddResponse(result=add_numbers(a, b))

    def register_health_routes(app: FastAPI) -> None:
        """Register health check endpoints without authentication."""
        
        @app.get("/health", response_model=HealthResponse)
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

        @app.get("/ready", response_model=ReadinessResponse)
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
        
        @router.post("/admin/rotate-keys", response_model=KeyManagementResponse)
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

        @router.post("/admin/revoke-key", response_model=KeyManagementResponse)
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

        @router.get("/admin/key-status", response_model=KeyManagementResponse)
        async def key_status() -> KeyManagementResponse:
            """Get current key management status."""
            key_manager = get_key_manager()
            rotation_info = key_manager.get_rotation_info()
            
            return KeyManagementResponse(
                message="Key status retrieved successfully",
                rotation_info=rotation_info
            )

        @router.get("/admin/metrics")
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

    if version == SUPPORTED_VERSIONS[0]:
        default_router = APIRouter(dependencies=dependencies)
        register_routes(default_router)
        register_admin_routes(default_router)  # Admin routes with authentication
        app.include_router(default_router)

    # Validate configuration at startup
    validate_environment()

    return app
