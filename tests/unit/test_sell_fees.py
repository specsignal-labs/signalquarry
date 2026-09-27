# SPDX-License-Identifier: Apache-2.0
"""Sell-side regulatory fees: configurable, sells only, capped per order, hash-neutral when unset."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from signalquarry._internal.contracts.spec import CostSpec, StrategySpecV1
from signalquarry._internal.data.synthetic import synthetic_dataset
from signalquarry._internal.engine.backtest import run_backtest
from signalquarry.sdk import Ctx, Decision, Params, definition_of, strategy

DATA = synthetic_dataset(date(2020, 1, 2), date(2021, 12, 31), symbols=("SYNA",))


class P(Params):
    period: int = 10


@strategy(params=P, lookback=lambda p: p.period)
def flip(ctx: Ctx, p: P) -> Decision:
    close = ctx.bars("SYNA").close
    return Decision.target({"SYNA": "0.9"} if close[-1] > close.mean() else {}, "GO")


def _spec(**costs: str) -> StrategySpecV1:
    return StrategySpecV1.model_validate(
        {
            "schema": "signalquarry.strategy/v1",
            "id": "fees",
            "family": "fees",
            "version": "1",
            "hypothesis": {"statement": "Sell-side fee test.", "falsification": "Not a hypothesis."},
            "data": {"symbols": ["SYNA"], "feed": "synthetic"},
            "execution": {"costs": {"bps": "0", "per_share": "0", **costs}},
            "reason_codes": {"GO": "flip"},
        }
    )


def test_unset_fees_keep_the_configuration_identity() -> None:
    assert set(_spec().outcome_document()["execution"]["costs"]) == {"bps", "per_share"}
    assert "sell_bps" in _spec(sell_bps="0.278").outcome_document()["execution"]["costs"]


def test_fee_formula_and_cap() -> None:
    costs = CostSpec(
        sell_bps=Decimal("0.278"), sell_per_share=Decimal("0.000166"), sell_per_order_max=Decimal("8.30")
    )
    assert costs.sell_fees(Decimal(1_000_000), Decimal(100)) == Decimal("27.8") + Decimal("0.0166")
    assert costs.sell_fees(Decimal(1_000_000), Decimal(1_000_000)) == Decimal("27.8") + Decimal("8.30")


def test_only_sells_pay_them() -> None:
    free = run_backtest(_spec(), definition_of(flip), P(), DATA)
    charged = run_backtest(_spec(sell_bps="5", sell_per_share="0.001"), definition_of(flip), P(), DATA)
    sells = [f for f in charged.fills if f.side == "sell"]
    assert sells and all(f.fee > 0 for f in sells)
    assert all(f.fee == 0 for f in charged.fills if f.side == "buy")
    assert all(f.fee == 0 for f in free.fills)
    assert charged.equity[-1] < free.equity[-1]
