"""Structured logging configuration using structlog."""

from __future__ import annotations

import logging

import structlog


def configure_logging(level: int = logging.INFO) -> None:
    """Configure structlog with JSON rendering."""

    logging.basicConfig(level=level, format="%(message)s")
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(level),
        processors=[structlog.processors.JSONRenderer()],
    )
