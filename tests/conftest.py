import os

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "network: test requires external network access")


def pytest_collection_modifyitems(config, items):
    ci = os.environ.get("CI", "").lower()
    if ci in ("true", "1", "yes", "on"):
        skip_network = pytest.mark.skip(reason="network test skipped in CI")
        for item in items:
            if "network" in item.keywords:
                item.add_marker(skip_network)
