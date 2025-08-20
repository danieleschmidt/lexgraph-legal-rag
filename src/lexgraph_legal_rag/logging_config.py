"""Structured logging configuration using structlog."""

from __future__ import annotations

import logging

import structlog

from .correlation import CorrelationIdProcessor


def configure_logging(level: int = logging.INFO) -> None:
    """Configure structlog with JSON rendering and correlation ID support."""

    logging.basicConfig(level=level, format="%(message)s")
    structlog.configure(
        processors=[
            CorrelationIdProcessor(),
            structlog.processors.TimeStamper(fmt="ISO"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        context_class=dict,
        cache_logger_on_first_use=True,
    )
