"""Shared data models for legal RAG."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any


@dataclass
class LegalDocument:
    """Representation of a single legal document."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
