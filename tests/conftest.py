import os

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "network: test requires external network access")


def pytest_collection_modifyitems(config, items):
    if os.environ.get("CI") == "true":
        skip_network = pytest.mark.skip(reason="network test skipped in CI")
        for item in items:
            if "network" in item.keywords:
                item.add_marker(skip_network)
