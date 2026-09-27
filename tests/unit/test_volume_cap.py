# SPDX-License-Identifier: Apache-2.0
"""The volume_cap fill model limits fills to a share of session volume and retries the rest."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from signalquarry._internal.contracts.spec import StrategySpecV1
from signalquarry._internal.data.synthetic import synthetic_dataset
from signalquarry._internal.engine.backtest import run_backtest
from signalquarry.sdk import Ctx, Decision, Params, definition_of, strategy

DATA = synthetic_dataset(date(2020, 1, 2), date(2021, 12, 31), symbols=("SYNA",))


class P(Params):
    weight: Decimal = Decimal("0.9")


@strategy(params=P, lookback=lambda p: 5)
def hold(ctx: Ctx, p: P) -> Decision:
    return Decision.target({"SYNA": p.weight}, "GO")


def _spec(fill=None, cash: str = "100000") -> StrategySpecV1:
    execution = {"costs": {"bps": "0", "per_share": "0"}}
    if fill is not None:
        execution["fill"] = fill
    return StrategySpecV1.model_validate(
        {
            "schema": "signalquarry.strategy/v1",
            "id": "cap",
            "family": "cap",
            "version": "1",
            "hypothesis": {"statement": "Capacity test.", "falsification": "Not a hypothesis."},
            "data": {"symbols": ["SYNA"], "feed": "synthetic"},
            "account": {"initial_cash": cash},
            "execution": execution,
            "reason_codes": {"GO": "hold"},
        }
    )


def test_absent_fill_model_keeps_the_configuration_identity() -> None:
    base = _spec().outcome_document()
    assert "fill" not in base["execution"]
    assert "fill" in _spec({"model": "volume_cap"}).outcome_document()["execution"]


def test_orders_fill_over_several_sessions_under_the_cap() -> None:
    uncapped = run_backtest(_spec(cash="50000000"), definition_of(hold), P(), DATA)
    fraction = Decimal("0.0001")
    capped = run_backtest(
        _spec({"model": "volume_cap", "max_volume_fraction": str(fraction)}, cash="50000000"),
        definition_of(hold),
        P(),
        DATA,
    )
    first_buys = [f for f in uncapped.fills if f.side == "buy"]
    assert len(first_buys) == 1  # without a cap the whole target fills at once
    buys = [f for f in capped.fills if f.side == "buy"]
    assert len(buys) > 1 and any(w.startswith("VOLUME_CAPPED:") for w in capped.warnings)
    for fill in buys:
        volume = Decimal(str(float(DATA.series["SYNA"].volume[DATA.index_of(fill.session)])))
        assert fill.quantity <= volume * fraction
    assert sum(f.quantity for f in buys) <= first_buys[0].quantity
