# SPDX-License-Identifier: Apache-2.0
"""Options core: OCC identities, quote rules, the wheel state machine and the resolver."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from signalquarry._internal.contracts.spec import StrategySpecV1
from signalquarry._internal.options.contracts import (
    OptionsError,
    Quote,
    contract,
    occ_symbol,
    parse_occ,
    quote_violations,
)
from signalquarry._internal.options.resolver import Candidate, QuoteRules, Selection, plan_invariants, resolve
from signalquarry._internal.options.wheel import EVENTS, TRANSITIONS, Leg, Wheel, WheelState, transition
from signalquarry.sdk.options import OD, Strike, sell_call, sell_put

NOW = datetime(2026, 9, 28, 14, 0, tzinfo=UTC)
TODAY = date(2026, 9, 28)


@given(
    root=st.text("ABCDEFGHIJKLMNOPQRSTUVWXYZ", min_size=1, max_size=6),
    days=st.integers(0, 3000),
    right=st.sampled_from(["PUT", "CALL"]),
    strike=st.integers(1, 99_999_999).map(lambda k: Decimal(k).scaleb(-3)),
)
def test_occ_round_trip(root: str, days: int, right: str, strike: Decimal) -> None:
    expiration = date(2024, 1, 1) + timedelta(days=days)
    symbol = occ_symbol(root, expiration, right, strike)  # type: ignore[arg-type]
    parsed = parse_occ(symbol)
    assert (parsed.underlying, parsed.expiration, parsed.right, parsed.strike) == (
        root,
        expiration,
        right,
        strike,
    )
    assert parsed.symbol == symbol


def test_occ_rejects_garbage() -> None:
    with pytest.raises(OptionsError):
        parse_occ("QQQ-261002-P")
    with pytest.raises(OptionsError):
        occ_symbol("QQQ", TODAY, "PUT", Decimal("0.0001"))


def test_quote_rules() -> None:
    good = Quote(Decimal("1.00"), Decimal("1.10"), NOW - timedelta(seconds=5))
    rules = dict(now=NOW, max_age_seconds=15, min_bid=Decimal("0.05"), max_relative_spread=Decimal("0.25"))
    assert quote_violations(good, **rules) == ()
    assert quote_violations(None, **rules) == ("QUOTE_MISSING",)
    assert "QUOTE_STALE" in quote_violations(
        Quote(Decimal(1), Decimal("1.1"), NOW - timedelta(seconds=60)), **rules
    )
    assert "QUOTE_SPREAD_TOO_WIDE" in quote_violations(Quote(Decimal(1), Decimal(2), NOW), **rules)
    assert "QUOTE_BID_TOO_LOW" in quote_violations(Quote(Decimal("0.01"), Decimal("0.012"), NOW), **rules)
    assert "QUOTE_FROM_FUTURE" in quote_violations(
        Quote(Decimal(1), Decimal("1.1"), NOW + timedelta(seconds=30)), **rules
    )


def _leg(right: str) -> Leg:
    return Leg(
        contract("QQQ", TODAY + timedelta(days=10), right, Decimal(480)), Decimal("1.20"), TODAY, "sq-x"
    )  # type: ignore[arg-type]


@settings(max_examples=200)
@given(st.lists(st.sampled_from(sorted(EVENTS)), max_size=40))
def test_wheel_invariants_hold_for_any_legal_path(events: list[str]) -> None:
    wheel = Wheel("QQQ")
    for event in events:
        leg = _leg("PUT" if event == "put_opened" else "CALL") if event.endswith("_opened") else None
        if (wheel.state, event) not in TRANSITIONS:
            with pytest.raises(OptionsError):
                transition(wheel, event, leg=leg)
            continue
        wheel = transition(wheel, event, leg=leg)
        assert wheel.shares in (0, 100)
        assert (wheel.leg is not None) == (wheel.state in (WheelState.SHORT_PUT, WheelState.COVERED_CALL))
        assert (wheel.shares == 100) == (wheel.state in (WheelState.LONG_SHARES, WheelState.COVERED_CALL))


def test_assignment_sets_the_cost_basis() -> None:
    wheel = transition(Wheel("QQQ"), "put_opened", leg=_leg("PUT"))
    wheel = transition(wheel, "put_assigned")
    assert (wheel.state, wheel.shares, wheel.share_cost_basis) == (WheelState.LONG_SHARES, 100, Decimal(480))
    with pytest.raises(OptionsError):
        transition(wheel, "call_opened", leg=_leg("PUT"))


def _chain(spot: Decimal) -> list[Candidate]:
    out = []
    for weeks in (1, 2, 3):
        expiry = TODAY + timedelta(days=4 + 7 * (weeks - 1))
        for k in range(-20, 21):
            strike = (spot + k * 2).quantize(Decimal(1))
            for right in ("PUT", "CALL"):
                intrinsic = max(Decimal(0), (strike - spot) if right == "PUT" else (spot - strike))
                bid = intrinsic + Decimal("0.80") + Decimal(weeks) / 10
                out.append(
                    Candidate(
                        contract("QQQ", expiry, right, strike),
                        Quote(bid, bid + Decimal("0.05"), NOW - timedelta(seconds=3)),
                    )
                )  # type: ignore[arg-type]
    return out


RULES = QuoteRules(min_bid=Decimal("0.05"), max_relative_spread=Decimal("0.25"), max_age_seconds=15)


@settings(max_examples=150, deadline=None)
@given(
    spot=st.integers(300, 600).map(Decimal),
    otm=st.sampled_from([Decimal("0.01"), Decimal("0.03"), Decimal("0.05")]),
    right=st.sampled_from(["PUT", "CALL"]),
    window=st.sampled_from([(3, 10), (7, 14), (1, 30)]),
)
def test_every_resolved_plan_satisfies_the_invariants(
    spot: Decimal, otm: Decimal, right: str, window: tuple[int, int]
) -> None:
    selection = Selection(right, window[0], window[1], otm)
    plan = resolve(
        selection, spot=spot, today=TODAY, now=NOW, candidates=_chain(spot), rules=RULES, tick=Decimal("0.01")
    )
    if isinstance(plan, tuple):
        assert plan == ("OPTIONS_NO_ELIGIBLE_CONTRACT",)
        return
    assert plan_invariants(plan, selection, today=TODAY, tick=Decimal("0.01")) == []


def test_resolver_prefers_strike_then_earliest_expiry_and_skips_bad_quotes() -> None:
    chain = _chain(Decimal(500))
    plan = resolve(
        Selection("PUT", 3, 20, Decimal("0.01")),
        spot=Decimal(500),
        today=TODAY,
        now=NOW,
        candidates=chain,
        rules=RULES,
        tick=Decimal("0.01"),
    )
    assert plan.contract.strike == Decimal(494) and plan.contract.expiration == TODAY + timedelta(days=4)  # type: ignore[union-attr]
    stale = [
        Candidate(c.contract, Quote(c.quote.bid, c.quote.ask, NOW - timedelta(minutes=5))) for c in chain
    ]
    assert resolve(
        Selection("PUT", 3, 20, Decimal("0.01")),
        spot=Decimal(500),
        today=TODAY,
        now=NOW,
        candidates=stale,
        rules=RULES,
        tick=Decimal("0.01"),
    ) == ("OPTIONS_NO_ELIGIBLE_CONTRACT",)


def test_sdk_selectors_and_decisions() -> None:
    selector = sell_put("qqq", dte=(7, 14), strike=Strike.otm("0.01"))
    assert (selector.underlying, selector.right, selector.min_dte, selector.max_dte) == ("QQQ", "PUT", 7, 14)
    assert sell_call("QQQ", dte=(7, 14), strike=Strike.otm(0.03)).right == "CALL"
    assert (
        OD.open(selector, "SELL_PUT").underlying == "QQQ"
        and OD.close("qqq", "TAKE_PROFIT").underlying == "QQQ"
    )
    with pytest.raises(ValueError):
        Strike.otm("0.6")
    with pytest.raises(ValueError):
        sell_put("QQQ", dte=(14, 7), strike=Strike.otm("0.01"))


def _spec(**extra):
    body = {
        "schema": "signalquarry.strategy/v1",
        "id": "wheel",
        "family": "wheel",
        "version": "1",
        "hypothesis": {
            "statement": "Selling weekly puts harvests premium.",
            "falsification": "It loses money after costs.",
        },
        "data": {"symbols": ["QQQ"], "feed": "synthetic"},
    }
    return StrategySpecV1.model_validate({**body, **extra})


def test_spec_options_section() -> None:
    assert _spec(kind="options_single_leg", options={"underlyings": ["QQQ"]}).options.underlyings == ("QQQ",)  # type: ignore[union-attr]
    with pytest.raises(ValueError, match="SPEC_OPTIONS_SECTION_REQUIRED"):
        _spec(kind="options_single_leg")
    with pytest.raises(ValueError, match="SPEC_OPTIONS_SECTION_REQUIRED"):
        _spec(options={"underlyings": ["QQQ"]})
    with pytest.raises(ValueError, match="SPEC_OPTIONS_UNDERLYING_NOT_IN_DATA_SYMBOLS"):
        _spec(kind="options_single_leg", options={"underlyings": ["SPY"]})


def test_each_plan_invariant_detects_its_violation() -> None:
    from dataclasses import replace

    from signalquarry._internal.options.resolver import floor_to_tick

    selection = Selection("PUT", 3, 20, Decimal("0.01"))
    plan = resolve(
        selection,
        spot=Decimal(500),
        today=TODAY,
        now=NOW,
        candidates=_chain(Decimal(500)),
        rules=RULES,
        tick=Decimal("0.01"),
    )
    assert (
        not isinstance(plan, tuple)
        and plan_invariants(plan, selection, today=TODAY, tick=Decimal("0.01")) == []
    )
    far = contract("QQQ", TODAY + timedelta(days=60), "PUT", plan.contract.strike)
    call = contract("QQQ", plan.contract.expiration, "CALL", plan.contract.strike)
    above = contract("QQQ", plan.contract.expiration, "PUT", Decimal(600))
    cases = {
        "PLAN_DTE_OUT_OF_WINDOW": replace(plan, contract=far),
        "PLAN_RIGHT_MISMATCH": replace(plan, contract=call),
        "PLAN_STRIKE_WRONG_SIDE": replace(
            plan,
            contract=above,
            collateral=Decimal(60000),
            maximum_loss=Decimal(60000) - plan.limit_price * 100,
        ),
        "PLAN_LIMIT_NOT_FLOOR_OF_BID": replace(plan, limit_price=plan.limit_price + Decimal("0.05")),
        "PLAN_COLLATERAL_INCONSISTENT": replace(plan, collateral=Decimal(1)),
    }
    for code, bad in cases.items():
        assert code in plan_invariants(bad, selection, today=TODAY, tick=Decimal("0.01")), code
    call_selection = Selection("CALL", 3, 20, Decimal("0.01"))
    call_plan = resolve(
        call_selection,
        spot=Decimal(500),
        today=TODAY,
        now=NOW,
        candidates=_chain(Decimal(500)),
        rules=RULES,
        tick=Decimal("0.01"),
    )
    below = contract("QQQ", call_plan.contract.expiration, "CALL", Decimal(400))  # type: ignore[union-attr]
    assert "PLAN_STRIKE_WRONG_SIDE" in plan_invariants(
        replace(call_plan, contract=below), call_selection, today=TODAY, tick=Decimal("0.01")
    )  # type: ignore[arg-type]
    assert floor_to_tick(Decimal("1.239"), Decimal("0.05")) == Decimal("1.20")
    zero_bid = [
        Candidate(c.contract, Quote(Decimal("0.004"), Decimal("0.006"), NOW)) for c in _chain(Decimal(500))
    ]
    loose = QuoteRules(min_bid=Decimal("0.001"), max_relative_spread=Decimal("1"), max_age_seconds=15)
    assert resolve(
        selection,
        spot=Decimal(500),
        today=TODAY,
        now=NOW,
        candidates=zero_bid,
        rules=loose,
        tick=Decimal("0.01"),
    ) == ("OPTIONS_NO_ELIGIBLE_CONTRACT",)
