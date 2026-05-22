# conftest.py - Print Hello World before each test execution
import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
    """Print Hello World before each test execution."""
    print("Hello World")
    # Return None to let pytest continue with the default protocol
    return None
