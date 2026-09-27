# SPDX-License-Identifier: Apache-2.0
"""Choose a contract for a selector and price the order. One resolver for simulation and paper.

Selection (the reference wheel's rule): among contracts of the selector's right whose
expiration is inside the DTE window and whose quote is usable, the strike must be on the
out-of-the-money side of the target (``spot × (1 - otm)`` for puts, ``× (1 + otm)`` for
calls). Puts take the highest such strike, calls the lowest; ties go to the earliest
expiration, then the symbol. Sell orders are limited at the bid floored to the tick;
buy-to-close orders at the ask raised to the tick.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal

from signalquarry._internal.options.contracts import OptionContract, Quote, quote_violations


@dataclass(frozen=True)
class QuoteRules:
    min_bid: Decimal
    max_relative_spread: Decimal
    max_age_seconds: int


@dataclass(frozen=True)
class Candidate:
    contract: OptionContract
    quote: Quote


@dataclass(frozen=True)
class Selection:
    right: str  # "PUT" | "CALL"
    min_dte: int
    max_dte: int
    otm_fraction: Decimal


@dataclass(frozen=True)
class EntryPlan:
    contract: OptionContract
    limit_price: Decimal
    quote: Quote
    spot: Decimal
    target_strike: Decimal
    collateral: Decimal  # cash for a put (strike × 100); shares' value for a call
    maximum_loss: Decimal


def floor_to_tick(price: Decimal, tick: Decimal) -> Decimal:
    return (price / tick).to_integral_value(rounding=ROUND_FLOOR) * tick


def ceil_to_tick(price: Decimal, tick: Decimal) -> Decimal:
    return (price / tick).to_integral_value(rounding=ROUND_CEILING) * tick


def target_strike(selection: Selection, spot: Decimal) -> Decimal:
    return (
        spot * (1 - selection.otm_fraction)
        if selection.right == "PUT"
        else spot * (1 + selection.otm_fraction)
    )


def resolve(
    selection: Selection,
    *,
    spot: Decimal,
    today: date,
    now: datetime,
    candidates: list[Candidate],
    rules: QuoteRules,
    tick: Decimal,
) -> EntryPlan | tuple[str, ...]:
    target = target_strike(selection, spot)
    first, last = today + timedelta(days=selection.min_dte), today + timedelta(days=selection.max_dte)
    usable = [
        c
        for c in candidates
        if c.contract.right == selection.right
        and first <= c.contract.expiration <= last
        and not quote_violations(
            c.quote,
            now=now,
            max_age_seconds=rules.max_age_seconds,
            min_bid=rules.min_bid,
            max_relative_spread=rules.max_relative_spread,
        )
    ]
    if selection.right == "PUT":
        eligible = [c for c in usable if c.contract.strike <= target]
        chosen = max(
            eligible,
            key=lambda c: (c.contract.strike, -c.contract.expiration.toordinal(), c.contract.symbol),
            default=None,
        )
    else:
        eligible = [c for c in usable if c.contract.strike >= target]
        chosen = min(
            eligible,
            key=lambda c: (c.contract.strike, c.contract.expiration, c.contract.symbol),
            default=None,
        )
    if chosen is None:
        return ("OPTIONS_NO_ELIGIBLE_CONTRACT",)
    limit = floor_to_tick(chosen.quote.bid, tick)
    if limit <= 0:
        return ("OPTIONS_NO_ELIGIBLE_CONTRACT",)
    collateral = chosen.contract.strike * 100 if selection.right == "PUT" else spot * 100
    return EntryPlan(
        contract=chosen.contract,
        limit_price=limit,
        quote=chosen.quote,
        spot=spot,
        target_strike=target,
        collateral=collateral,
        maximum_loss=max(Decimal("0.01"), collateral - limit * 100),
    )


def plan_invariants(plan: EntryPlan, selection: Selection, *, today: date, tick: Decimal) -> list[str]:
    """What must hold for every entry plan (checked in tests and before every paper submission)."""
    problems = []
    dte = plan.contract.dte(today)
    if not selection.min_dte <= dte <= selection.max_dte:
        problems.append("PLAN_DTE_OUT_OF_WINDOW")
    if plan.contract.right != selection.right:
        problems.append("PLAN_RIGHT_MISMATCH")
    if selection.right == "PUT" and plan.contract.strike > plan.target_strike:
        problems.append("PLAN_STRIKE_WRONG_SIDE")
    if selection.right == "CALL" and plan.contract.strike < plan.target_strike:
        problems.append("PLAN_STRIKE_WRONG_SIDE")
    if plan.limit_price != floor_to_tick(plan.quote.bid, tick) or plan.limit_price > plan.quote.bid:
        problems.append("PLAN_LIMIT_NOT_FLOOR_OF_BID")
    expected_collateral = plan.contract.strike * 100 if selection.right == "PUT" else plan.spot * 100
    if plan.collateral != expected_collateral or plan.maximum_loss != max(
        Decimal("0.01"), plan.collateral - plan.limit_price * 100
    ):
        problems.append("PLAN_COLLATERAL_INCONSISTENT")
    return problems
