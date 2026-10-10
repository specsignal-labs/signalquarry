# SPDX-License-Identifier: Apache-2.0
"""Register optional, expensive statistical acceptance tests.

Tests marked ``slow`` (the 200-panel null test of ADR 0014 takes about two minutes) run only
when ``SIGNALQUARRY_SLOW_TESTS=1`` is set, so the default suite and its coverage run stay fast.
"""

import os

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: statistical acceptance tests; set SIGNALQUARRY_SLOW_TESTS=1")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.environ.get("SIGNALQUARRY_SLOW_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="slow statistical test; set SIGNALQUARRY_SLOW_TESTS=1 to run it")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
