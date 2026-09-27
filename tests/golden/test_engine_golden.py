# SPDX-License-Identifier: Apache-2.0
"""Pinned engine outputs on the seeded synthetic dataset.

A change here is a behaviour change of the engine or the synthetic generator.
Explain it in the pull request before updating the pinned values.
"""

from __future__ import annotations

import time
from datetime import date
from decimal import Decimal

import pytest

from signalquarry._internal.data.synthetic import synthetic_dataset
from signalquarry._internal.engine.backtest import run_backtest
from signalquarry.sdk import definition_of
from tests.helpers import spec
from tests.unit.test_engine import SmaP, sma_trend

DATASET_IDENTITY = "sha256:b4887398f63653054056330f8f57e0d31d5b3e8714fd8204ef3c1e3bd0bc5ebe"
EXPECTED = {
    50: ("sha256:b66949f2992536b9cf261201f9204de1a6e1196f77677f0a96ecb0e6b1dcc292", Decimal("10415.190000")),
    200: ("sha256:55c1500cc47eceb31f4a449c1f384c15c025142988ac511ffdfb71098ffc75c0", Decimal("11089.950000")),
}


@pytest.fixture(scope="module")
def data():
    return synthetic_dataset(date(2014, 1, 2), date(2023, 12, 29))


def test_synthetic_dataset_is_deterministic(data) -> None:
    assert data.identity() == DATASET_IDENTITY


@pytest.mark.parametrize("period", sorted(EXPECTED))
def test_sma_ledgers_are_pinned(data, period: int) -> None:
    result = run_backtest(spec(("SYNA",)), definition_of(sma_trend), SmaP(period=period), data)
    assert (result.ledger_hash, result.equity[-1]) == EXPECTED[period]


def test_ten_year_single_symbol_backtest_is_fast(data) -> None:
    started = time.perf_counter()
    run_backtest(spec(("SYNA",)), definition_of(sma_trend), SmaP(period=200), data)
    assert time.perf_counter() - started < 2.0  # target < 1 s; fail at 2x
