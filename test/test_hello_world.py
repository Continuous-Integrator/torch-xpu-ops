"""Simple test to verify conftest.py Hello World functionality."""


def test_hello_world_print():
    """Test that conftest prints Hello World before this test."""
    assert True


def test_another_hello_world():
    """Second test to verify Hello World is printed before each test."""
    assert 1 + 1 == 2
