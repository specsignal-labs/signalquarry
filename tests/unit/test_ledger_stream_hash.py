# SPDX-License-Identifier: Apache-2.0
"""The bounded ledger hash must equal the existing canonical document hash."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from signalquarry._internal.canonical import canonical_hash
from signalquarry._internal.engine.backtest import BacktestResult, Fill


def test_streamed_ledger_hash_matches_legacy_across_chunk_boundaries() -> None:
    first, second = date(2024, 1, 2), date(2024, 1, 3)
    result = BacktestResult(
        [first, second],
        [Decimal("10000.00"), Decimal("10001.25")],
        [Decimal("5000"), Decimal("4999.75")],
        [
            {"session": first.isoformat(), "action": "target", "weights": {"SYNA": Decimal(i) / 100}}
            for i in range(35)
        ],
        [
            Fill(first, "SYNA", "buy", Decimal(1), Decimal("42.25"), Decimal("0.01"), None)
            for _ in range(4101)
        ],
        {"SYNB": Decimal(-1), "SYNA": Decimal("2.5")},
        [],
        "sha256:" + "a" * 64,
    )
    assert result.compute_ledger_hash() == canonical_hash(result.ledger_document())
    assert result.compute_ledger_hash() == result.compute_ledger_hash()


def test_empty_ledger_and_nonfinite_decision_match_canonical_rules() -> None:
    result = BacktestResult([], [], [], [], [], {}, [], "sha256:" + "b" * 64)
    assert result.compute_ledger_hash() == canonical_hash(result.ledger_document())
    result.decisions.append({"bad": float("nan")})
    with pytest.raises(ValueError, match="CANONICAL_FLOAT_NOT_FINITE"):
        result.compute_ledger_hash()
