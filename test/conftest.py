import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
    print("Hello World")
    return None
