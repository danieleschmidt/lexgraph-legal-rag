"""Correlation ID middleware and utilities for request tracing."""

from __future__ import annotations

import logging
import uuid
from contextvars import ContextVar
from typing import TYPE_CHECKING
from typing import Any

import structlog
from starlette.middleware.base import BaseHTTPMiddleware


if TYPE_CHECKING:
    from fastapi import Request
    from fastapi import Response


logger = structlog.get_logger(__name__)

# Context variable to store correlation ID for the current request
_correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)

# Default header name for correlation ID
CORRELATION_ID_HEADER = "X-Correlation-ID"
CORRELATION_ID_RESPONSE_HEADER = "X-Correlation-ID"


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Middleware to handle correlation ID for request tracing.

    This middleware:
    1. Extracts correlation ID from incoming request headers
    2. Generates a new correlation ID if none is provided
    3. Sets the correlation ID in the request context
    4. Adds correlation ID to response headers
    5. Ensures all logs include the correlation ID
    """

    def __init__(
        self,
        app: Any,
        header_name: str = CORRELATION_ID_HEADER,
        response_header_name: str = CORRELATION_ID_RESPONSE_HEADER,
    ):
        super().__init__(app)
        self.header_name = header_name
        self.response_header_name = response_header_name

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        """Process request with correlation ID handling."""
        # Extract or generate correlation ID
        correlation_id = request.headers.get(self.header_name)
        if not correlation_id:
            correlation_id = str(uuid.uuid4())

        # Set correlation ID in context
        correlation_token = _correlation_id.set(correlation_id)

        try:
            # Add correlation ID to request state for access in endpoints
            request.state.correlation_id = correlation_id

            # Process the request with correlation ID context
            with correlation_context(correlation_id):
                response = await call_next(request)

            # Add correlation ID to response headers
            response.headers[self.response_header_name] = correlation_id

            return response

        finally:
            # Reset correlation ID context
            _correlation_id.reset(correlation_token)


def get_correlation_id() -> str | None:
    """Get the current correlation ID from context."""
    return _correlation_id.get()


def set_correlation_id(correlation_id: str) -> None:
    """Set the correlation ID in the current context."""
    _correlation_id.set(correlation_id)


class correlation_context:
    """Context manager for correlation ID."""

    def __init__(self, correlation_id: str):
        self.correlation_id = correlation_id
        self.token = None

    def __enter__(self):
        self.token = _correlation_id.set(self.correlation_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.token:
            _correlation_id.reset(self.token)


class CorrelationIdProcessor:
    """Structlog processor to add correlation ID to log records."""

    def __call__(
        self, logger: Any, method_name: str, event_dict: dict[str, Any]
    ) -> dict[str, Any]:
        """Add correlation ID to log event."""
        correlation_id = get_correlation_id()
        if correlation_id:
            event_dict["correlation_id"] = correlation_id
        return event_dict


def configure_correlation_logging() -> None:
    """Configure structlog to include correlation IDs in all log messages."""
    structlog.configure(
        processors=[
            CorrelationIdProcessor(),
            structlog.processors.TimeStamper(fmt="ISO"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.PrintLoggerFactory(),
        context_class=dict,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> Any:
    """Get a logger that includes correlation ID in all messages."""
    return structlog.get_logger(name)
