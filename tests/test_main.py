"""Test module for the main functionality."""

import pytest


def test_dummy_calculation():
    """Dummy test to demonstrate basic testing patterns."""
    # Test basic arithmetic
    assert 1 + 1 == 2


def test_dummy_exception_handling():
    """Dummy test to demonstrate exception testing."""
    with pytest.raises(ZeroDivisionError):
        _ = 1 / 0

    with pytest.raises(TypeError):
        _ = "string" + 5
