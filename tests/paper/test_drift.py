# SPDX-License-Identifier: Apache-2.0
"""sqy paper drift: replay parity, fill slippage, shadow costs and dividends, gate G5."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from decimal import Decimal
from pathlib import Path

from signalquarry._internal.data.synthetic import synthetic_dataset
from signalquarry._internal.paper.brokers.fake import FakeBroker
from signalquarry._internal.paper.runner import PaperKernel
from signalquarry._internal.validation.conformance import perturb_after
from tests.paper.harness import Momentum, hold_syna, paper_spec, rig


def _run(tmp_path: Path, sessions: int = 25, **kwargs):
    paper = rig(tmp_path, decide=hold_syna, **kwargs)
    days = list(paper.dataset.sessions[60 : 60 + sessions])
    for number, session in enumerate(days):
        if number % 20 == 0:
            paper.arm(session)
        paper.session(session)
    return paper, days


def test_clean_forward_record_passes_g5_with_shadow_costs(tmp_path: Path) -> None:
    data = replace(synthetic_dataset(date(2023, 1, 2), date(2024, 12, 31), symbols=("SYNA", "SYNB")))
    paper, _ = _run(
        tmp_path, dataset=data, the_spec=paper_spec(execution={"costs": {"bps": "10", "per_share": "0"}})
    )
    outcome = paper.kernel.drift()
    g5 = outcome.data["g5"]
    assert g5["ok"] is True and g5["clean_sessions"] >= 20 and g5["parity_mismatches"] == 0
    assert g5["mean_adverse_slippage_bps"] == 0.0 and outcome.data["fills_measured"] >= 1
    shadow = outcome.data["shadow"]
    assert shadow["model_costs"] > 0
    held_through_dividend = any(
        d.symbol == "SYNA" and data.sessions[60] < d.ex_date <= data.sessions[84] for d in data.dividends
    )
    assert (shadow["uncredited_dividends"] > 0) == held_through_dividend
    assert (
        shadow["shadow_equity"]
        == shadow["broker_equity"] - shadow["model_costs"] + shadow["uncredited_dividends"]
    )


def test_changed_parameters_break_parity(tmp_path: Path) -> None:
    paper, _ = _run(tmp_path, sessions=5)
    changed = PaperKernel(
        replace(paper.deployment, params=Momentum(weight=Decimal("0.3"))),
        paper.broker,
        paper.loader,
        now=lambda: paper.broker.now,
    )
    outcome = changed.drift()
    assert outcome.reason_codes == ["PAPER_PARITY_MISMATCH"] and outcome.data["g5"]["ok"] is False


def test_revised_provider_data_is_unverifiable(tmp_path: Path) -> None:
    paper, days = _run(tmp_path, sessions=3)
    revised = PaperKernel(
        paper.deployment,
        paper.broker,
        lambda session: perturb_after(paper.loader(session), 5, 1.01, 1.01),
        now=lambda: paper.broker.now,
    )
    rows = revised.drift().data["sessions"]
    assert {row["parity"] for row in rows} == {"unverifiable"} and rows[0]["reason"] == "PROVIDER_REVISION"


class SlippingBroker(FakeBroker):
    def open_session(self, session: date) -> None:
        super().open_session(session)
        for order_id, order in list(self.orders.items()):
            if (
                order.status == "filled"
                and order.filled_average_price is not None
                and not getattr(order, "_slipped", False)
            ):
                worse = order.filled_average_price * (
                    Decimal("1.01") if order.side == "buy" else Decimal("0.99")
                )
                self.orders[order_id] = replace(order, filled_average_price=worse.quantize(Decimal("0.01")))


def test_adverse_fills_fail_the_slippage_tolerance(tmp_path: Path) -> None:
    paper, _ = _run(tmp_path, broker_class=SlippingBroker)
    g5 = paper.kernel.drift().data["g5"]
    assert g5["mean_adverse_slippage_bps"] > 25 and g5["ok"] is False
