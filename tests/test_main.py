"""Test module for the main functionality."""

import pytest
from unittest.mock import patch
from exploration_sequential_testing import main


def test_main_prints_greeting():
    """Test that main function prints the expected greeting message."""
    with patch("builtins.print") as mock_print:
        main()
        mock_print.assert_called_once_with("Hello from exploration-sequential-testing!")


def test_main_returns_none():
    """Test that main function returns None."""
    result = main()
    assert result is None


def test_dummy_calculation():
    """Dummy test to demonstrate basic testing patterns."""
    # Test basic arithmetic
    assert 1 + 1 == 2
    assert 5 * 3 == 15
    
    # Test list operations
    test_list = [1, 2, 3]
    assert len(test_list) == 3
    assert test_list[0] == 1
    
    # Test string operations
    test_string = "Hello World"
    assert test_string.lower() == "hello world"
    assert "World" in test_string


def test_dummy_exception_handling():
    """Dummy test to demonstrate exception testing."""
    with pytest.raises(ZeroDivisionError):
        _ = 1 / 0
    
    with pytest.raises(TypeError):
        _ = "string" + 5