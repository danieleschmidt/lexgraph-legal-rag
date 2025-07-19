"""Test coverage for sample module."""

import pytest
from lexgraph_legal_rag.sample import add


def test_add_positive_numbers():
    """Test adding positive numbers."""
    assert add(2, 3) == 5
    assert add(10, 20) == 30


def test_add_negative_numbers():
    """Test adding negative numbers."""
    assert add(-2, -3) == -5
    assert add(-10, 5) == -5


def test_add_zero():
    """Test adding with zero."""
    assert add(0, 0) == 0
    assert add(5, 0) == 5
    assert add(0, -5) == -5


def test_add_large_numbers():
    """Test adding large numbers."""
    assert add(1000000, 2000000) == 3000000
    assert add(-1000000, 500000) == -500000