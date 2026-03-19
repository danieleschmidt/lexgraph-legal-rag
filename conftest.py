"""Global pytest configuration for LexGraph Legal RAG."""

import os
import pathlib
import sys

import pytest

# Ensure src/ is on sys.path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Test environment
os.environ.setdefault("ENVIRONMENT", "test")


@pytest.fixture
def sample_legal_texts():
    """Return a dict of {name: text} for use in tests."""
    return {
        "NDA": (
            "Section 1. Confidentiality.\n"
            "The Receiving Party shall maintain strict confidentiality.\n\n"
            "Section 2. Term.\n"
            "This Agreement remains in effect for three years."
        ),
        "SLA": (
            "Article 1. License.\n"
            "Licensor grants a non-exclusive license to use the Software.\n\n"
            "Article 2. Warranty Disclaimer.\n"
            "THE SOFTWARE IS PROVIDED AS IS WITHOUT WARRANTY."
        ),
    }
