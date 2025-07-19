"""HTTP monitoring middleware for request tracking and alerting."""

from __future__ import annotations

import time
import logging
from typing import Any
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request, Response

from .metrics import record_http_request
from .correlation import get_correlation_id

logger = logging.getLogger(__name__)


class HTTPMonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware to track HTTP requests for alerting and monitoring.
    
    This middleware:
    1. Records HTTP request metrics (total, errors, duration)
    2. Includes correlation ID in monitoring context
    3. Logs request details for debugging
    4. Provides data for alerting on error rates and latency
    """
    
    def __init__(self, app: Any):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: Any) -> Response:
        """Process request with monitoring."""
        start_time = time.time()
        method = request.method
        path = request.url.path
        correlation_id = get_correlation_id()
        
        # Log request start
        logger.info("HTTP request started", extra={
            "method": method,
            "path": path,
            "correlation_id": correlation_id,
            "user_agent": request.headers.get("user-agent", "unknown")
        })
        
        try:
            # Process the request
            response = await call_next(request)
            status_code = response.status_code
            
        except Exception as e:
            # Handle unexpected errors
            duration = time.time() - start_time
            status_code = 500
            
            # Record the error
            record_http_request(method, path, status_code, duration)
            
            logger.error("HTTP request failed with exception", extra={
                "method": method,
                "path": path,
                "status_code": status_code,
                "duration": duration,
                "correlation_id": correlation_id,
                "error": str(e)
            })
            
            # Re-raise the exception to let FastAPI handle it
            raise
        
        else:
            # Calculate duration and record metrics
            duration = time.time() - start_time
            record_http_request(method, path, status_code, duration)
            
            # Log request completion
            log_level = logging.WARNING if status_code >= 400 else logging.INFO
            logger.log(log_level, "HTTP request completed", extra={
                "method": method,
                "path": path,
                "status_code": status_code,
                "duration": duration,
                "correlation_id": correlation_id
            })
            
            return response
    
    def _extract_endpoint_name(self, path: str) -> str:
        """Extract normalized endpoint name from path for metrics."""
        # Normalize paths to avoid high cardinality in metrics
        # Example: /api/v1/documents/123 -> /api/v1/documents/{id}
        
        # Simple normalization - replace numeric IDs
        import re
        normalized = re.sub(r'/\d+', '/{id}', path)
        
        # Replace UUIDs
        normalized = re.sub(
            r'/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', 
            '/{uuid}', 
            normalized
        )
        
        return normalized