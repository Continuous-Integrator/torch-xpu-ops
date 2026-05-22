import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    """Print 'Hello World' before each test execution."""
    print("Hello World")
