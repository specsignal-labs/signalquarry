# SPDX-License-Identifier: Apache-2.0
"""Property tests (Hypothesis) for the invariants the evidence depends on."""

from __future__ import annotations

import json
from datetime import date
from decimal import Decimal
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from signalquarry._internal.canonical import canonical_hash, canonical_json, to_canonical
from signalquarry._internal.engine.backtest import run_backtest
from signalquarry._internal.paper.journal import Journal
from signalquarry._internal.paper.models import PaperError
from signalquarry.sdk import Ctx, Decision, Params, definition_of, strategy
from tests.helpers import Split, dataset, spec, weekdays

json_values = st.recursive(
    st.none()
    | st.booleans()
    | st.integers(-(10**12), 10**12)
    | st.text(max_size=8)
    | st.decimals(allow_nan=False, allow_infinity=False, places=4, min_value=-(10**6), max_value=10**6),
    lambda children: (
        st.lists(children, max_size=4) | st.dictionaries(st.text(max_size=5), children, max_size=4)
    ),
    max_leaves=12,
)


@given(json_values)
def test_canonical_json_round_trips_and_is_order_independent(value) -> None:
    text = canonical_json(value)
    assert json.loads(text) == to_canonical(value)
    if isinstance(value, dict):
        assert canonical_hash(dict(reversed(list(value.items())))) == canonical_hash(value)


@given(st.decimals(allow_nan=False, allow_infinity=False, places=6, min_value=-(10**6), max_value=10**6))
def test_equal_decimals_hash_equally(value: Decimal) -> None:
    assert canonical_hash(value) == canonical_hash(value.quantize(Decimal("0.000001")) * 1)
    assert canonical_hash(value) == canonical_hash(Decimal(str(value)) + Decimal("0.000"))


class W(Params):
    pass


def _strategy(targets: list[float]):
    @strategy(params=W, lookback=lambda p: 1)
    def decide(ctx: Ctx, p: W) -> Decision:
        index = len(ctx.state.get("seen", []))
        weight = targets[index % len(targets)]
        return Decision.target(
            {"AAA": weight} if weight else {}, "GO", state={"seen": [*ctx.state.get("seen", []), 1][-50:]}
        )

    return definition_of(decide)


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    prices=st.lists(st.floats(min_value=5, max_value=500), min_size=8, max_size=40),
    targets=st.lists(st.sampled_from([0.0, 0.25, 0.5, 0.9]), min_size=1, max_size=6),
    split_at=st.one_of(st.none(), st.integers(min_value=2, max_value=6)),
    model=st.sampled_from(["cash", "margin"]),
)
def test_accounting_invariants(
    prices: list[float], targets: list[float], split_at: int | None, model: str
) -> None:
    days = weekdays(date(2024, 1, 2), len(prices))
    opens = [round(p, 2) for p in prices]
    closes = [round(p * 1.01, 2) for p in prices]
    splits = []
    if split_at is not None:
        splits = [Split("AAA", days[split_at], Decimal(2))]
        opens = opens[:split_at] + [round(x / 2, 2) for x in opens[split_at:]]
        closes = closes[:split_at] + [round(x / 2, 2) for x in closes[split_at:]]
    data = dataset(days, {"AAA": {"open": opens, "close": closes}}, splits=splits)
    result = run_backtest(spec(("AAA",), account={"model": model}), _strategy(targets), W(), data)
    held = result.positions.get("AAA", Decimal(0))
    last_close = data.price("AAA", "close", len(days) - 1)
    assert held >= 0
    assert all(cash >= 0 for cash in result.cash), "settled cash never negative"
    # equity = all cash (settled + pending) + positions at the close
    pending_and_settled = result.equity[-1] - held * last_close
    assert pending_and_settled >= 0

    def adjusted(fill) -> Decimal:  # quantities in today's share basis
        return fill.quantity * (2 if split_at is not None and fill.session < days[split_at] else 1)

    bought = sum((adjusted(f) for f in result.fills if f.side == "buy"), Decimal(0))
    sold = sum((adjusted(f) for f in result.fills if f.side == "sell"), Decimal(0))
    assert held == bought - sold, "positions reconcile with fills and splits"


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    bodies=st.lists(
        st.dictionaries(st.sampled_from("abc"), st.integers(0, 9), max_size=3), min_size=1, max_size=6
    ),
    flip=st.integers(0, 10**6),
)
def test_any_journal_edit_breaks_verification(tmp_path_factory, bodies, flip) -> None:
    path = Path(tmp_path_factory.mktemp("journal")) / "journal.jsonl"
    journal = Journal.open(path)
    for body in bodies:
        journal.append("note", {"body": body})
    assert len(Journal.open(path).entries) == len(bodies)
    lines = path.read_text().splitlines()
    index = flip % len(lines)
    entry = json.loads(lines[index])
    entry["body"] = {"tampered": flip}
    lines[index] = json.dumps(entry, separators=(",", ":"), sort_keys=True)
    path.write_text("\n".join(lines) + "\n")
    try:
        Journal.open(path)
    except PaperError as exc:
        assert exc.code == "PAPER_JOURNAL_CORRUPT"
    else:
        raise AssertionError("tampered journal verified")
