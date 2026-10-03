# SPDX-License-Identifier: Apache-2.0
"""Hand-worked synthetic lifecycle vectors; no provider data or broker orders."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime
from decimal import Decimal
from operator import setitem

import pytest

from signalquarry._internal.data.identity import AssetKey
from signalquarry._internal.data.lifecycle import LifecycleError, LifecycleEvent
from signalquarry._internal.engine.lifecycle import (
    CashReceivable,
    LifecycleState,
    apply_lifecycle_event,
    apply_lifecycle_events,
    settle_lifecycle_receivables,
    value_lifecycle_state,
)

OLD = AssetKey("synthetic", "asset-old")
NEW = AssetKey("synthetic", "asset-new")
OTHER = AssetKey("synthetic", "asset-other")
DAY = date(2024, 1, 10)
PAY = date(2024, 1, 15)


def event(kind: str, **terms) -> LifecycleEvent:
    base = {
        "event_id": f"synthetic-{kind}",
        "kind": kind,
        "source": OLD,
        "source_symbol": "SYNA",
        "effective_date": DAY,
        "process_date": date(2024, 1, 12),
        "observed_at": datetime(2024, 1, 13, tzinfo=UTC),
        "page_hashes": ("a" * 64,),
        "normalization_version": 1,
        "sequence": 0,
        "fee_per_old_share": Decimal(0),
    }
    return LifecycleEvent(**(base | terms))


def holding(shares: str = "5") -> LifecycleState:
    return LifecycleState({OLD: Decimal(shares)}, {OLD: "SYNA"})


def test_rename_changes_only_alias_and_preserves_asset_and_shares() -> None:
    before = holding()
    action = event("rename", target_symbol="SYNB")
    after = apply_lifecycle_event(before, action)
    assert dict(after.positions) == {OLD: Decimal(5)}
    assert dict(after.symbols) == {OLD: "SYNB"}
    assert after.receivables == ()
    assert dict(before.symbols) == {OLD: "SYNA"}
    assert after.applied_event_ids == frozenset({action.event_id})


def test_split_with_symbol_change_updates_shares_and_alias_atomically() -> None:
    action = event("split", target_symbol="SYNB", share_ratio=Decimal("0.5"), fraction_policy="retain")
    after = apply_lifecycle_event(holding(), action)
    assert dict(after.positions) == {OLD: Decimal("2.5")}
    assert dict(after.symbols) == {OLD: "SYNB"}
    with pytest.raises(LifecycleError, match="fractional shares"):
        apply_lifecycle_event(holding(), replace(action, fraction_policy="reject_noninteger"))


def test_worthless_removal_is_explicit_zero_payoff_not_a_missing_bar() -> None:
    action = event("worthless_removal", cash_per_old_share=Decimal(0))
    after = apply_lifecycle_event(holding(), action)
    assert dict(after.positions) == {} and dict(after.symbols) == {}
    assert after.receivables == ()


def test_cash_merger_removes_old_asset_and_records_payable_consideration() -> None:
    action = event("cash_merger", cash_per_old_share=Decimal("12.34"), cash_pay_date=PAY)
    after = apply_lifecycle_event(holding(), action)
    assert dict(after.positions) == {} and dict(after.symbols) == {}
    assert after.receivables == (CashReceivable(action.event_id, PAY, Decimal("61.70")),)


def test_stock_merger_uses_distinct_successor_asset_even_if_ticker_is_reused() -> None:
    action = event(
        "stock_merger",
        target=NEW,
        target_symbol="SYNA",
        share_ratio=Decimal("0.4"),
        fraction_policy="reject_noninteger",
    )
    after = apply_lifecycle_event(holding(), action)
    assert dict(after.positions) == {NEW: Decimal(2)}
    assert dict(after.symbols) == {NEW: "SYNA"}
    assert OLD not in after.positions


def test_mixed_merger_combines_existing_successor_holding_and_cash() -> None:
    before = LifecycleState({OLD: Decimal(5), NEW: Decimal(3)}, {OLD: "SYNA", NEW: "SYNB"})
    action = event(
        "mixed_merger",
        target=NEW,
        target_symbol="SYNB",
        share_ratio=Decimal("0.4"),
        fraction_policy="reject_noninteger",
        cash_per_old_share=Decimal("3.25"),
        cash_pay_date=PAY,
    )
    after = apply_lifecycle_event(before, action)
    assert dict(after.positions) == {NEW: Decimal(5)}
    assert dict(after.symbols) == {NEW: "SYNB"}
    assert after.receivables == (CashReceivable(action.event_id, PAY, Decimal("16.25")),)
    assert dict(before.positions) == {OLD: Decimal(5), NEW: Decimal(3)}


def test_cash_merger_receivable_is_valued_then_credited_once() -> None:
    action = event("cash_merger", cash_per_old_share=Decimal("12.34"), cash_pay_date=PAY)
    before = apply_lifecycle_event(holding(), action)
    assert value_lifecycle_state(before, {}, Decimal(100), DAY) == Decimal("161.70")
    pending, cash = settle_lifecycle_receivables(before, Decimal(100), date(2024, 1, 14))
    assert pending is before and cash == Decimal(100)
    settled, cash = settle_lifecycle_receivables(pending, cash, PAY)
    assert cash == Decimal("161.70") and settled.receivables == ()
    assert value_lifecycle_state(settled, {}, cash, PAY) == Decimal("161.70")
    again, cash_again = settle_lifecycle_receivables(settled, cash, PAY)
    assert again is settled and cash_again == cash
    with pytest.raises(LifecycleError, match="duplicate or out-of-order"):
        apply_lifecycle_event(settled, action)


def test_stock_merger_values_successor_by_asset_key_not_reused_ticker() -> None:
    action = event(
        "stock_merger",
        target=NEW,
        target_symbol="SYNA",
        share_ratio=Decimal("0.4"),
        fraction_policy="reject_noninteger",
    )
    after = apply_lifecycle_event(holding(), action)
    assert value_lifecycle_state(after, {NEW: Decimal("24.68")}, Decimal(10), DAY) == Decimal("59.36")
    with pytest.raises(LifecycleError, match="missing or invalid mark"):
        value_lifecycle_state(after, {OLD: Decimal("24.68")}, Decimal(10), DAY)


@pytest.mark.parametrize("bad_mark", [None, Decimal(0), Decimal(-1), Decimal("NaN")])
def test_held_asset_requires_valid_mark(bad_mark: Decimal | None) -> None:
    marks = {} if bad_mark is None else {OLD: bad_mark}
    with pytest.raises(LifecycleError, match="missing or invalid mark"):
        value_lifecycle_state(holding(), marks, Decimal(100), DAY)


def test_valuation_rejects_invalid_cash_and_time_travel() -> None:
    after = apply_lifecycle_event(holding(), event("rename", target_symbol="SYNB"))
    with pytest.raises(LifecycleError, match="invalid settled cash"):
        value_lifecycle_state(after, {OLD: Decimal(1)}, Decimal("NaN"), DAY)
    with pytest.raises(LifecycleError, match="valuation before applied event"):
        value_lifecycle_state(after, {OLD: Decimal(1)}, Decimal(0), date(2024, 1, 9))
    with pytest.raises(LifecycleError, match="valuation before applied event"):
        settle_lifecycle_receivables(after, Decimal(0), date(2024, 1, 9))


def test_zero_holding_still_updates_identity_without_creating_consideration() -> None:
    before = LifecycleState({}, {OLD: "SYNA"})
    action = event("cash_merger", cash_per_old_share=Decimal("12.34"), cash_pay_date=PAY)
    after = apply_lifecycle_event(before, action)
    assert dict(after.positions) == {} and dict(after.symbols) == {}
    assert after.receivables == ()
    assert value_lifecycle_state(after, {}, Decimal(100), DAY) == Decimal(100)


def test_mixed_merger_valuation_counts_successor_and_pending_cash() -> None:
    before = LifecycleState({OLD: Decimal(5), NEW: Decimal(3)}, {OLD: "SYNA", NEW: "SYNB"})
    action = event(
        "mixed_merger",
        target=NEW,
        target_symbol="SYNB",
        share_ratio=Decimal("0.4"),
        fraction_policy="reject_noninteger",
        cash_per_old_share=Decimal("3.25"),
        cash_pay_date=PAY,
    )
    after = apply_lifecycle_event(before, action)
    # Five successor shares at 20.00 plus 16.25 pending cash consideration.
    assert value_lifecycle_state(after, {NEW: Decimal(20)}, Decimal(100), DAY) == Decimal("216.25")
    settled, cash = settle_lifecycle_receivables(after, Decimal(100), PAY)
    assert value_lifecycle_state(settled, {NEW: Decimal(20)}, cash, PAY) == Decimal("216.25")


def test_same_day_actions_require_explicit_sequence_and_replay_once() -> None:
    split = event("split", share_ratio=Decimal(2), fraction_policy="reject_noninteger")
    merger = event(
        "cash_merger",
        event_id="synthetic-following-merger",
        sequence=1,
        cash_per_old_share=Decimal(5),
        cash_pay_date=PAY,
    )
    after = apply_lifecycle_events(holding(), (merger, split))
    assert after.receivables == (CashReceivable(merger.event_id, PAY, Decimal(50)),)
    assert after.last_event_key == (DAY, 1)
    with pytest.raises(LifecycleError, match="duplicate or out-of-order"):
        apply_lifecycle_event(after, split)
    with pytest.raises(LifecycleError, match="duplicate or out-of-order"):
        apply_lifecycle_events(holding(), (split, replace(merger, sequence=0)))


def test_ambiguous_successor_and_source_aliases_are_rejected() -> None:
    action = event(
        "stock_merger", target=NEW, target_symbol="SYNB", share_ratio=Decimal(1), fraction_policy="retain"
    )
    before = LifecycleState({OLD: Decimal(5)}, {OLD: "SYNA", OTHER: "SYNB"})
    with pytest.raises(LifecycleError, match="ambiguous successor symbol"):
        apply_lifecycle_event(before, action)
    with pytest.raises(LifecycleError, match="source asset alias mismatch"):
        apply_lifecycle_event(holding(), replace(action, source_symbol="SYNZ"))
    existing = LifecycleState({OLD: Decimal(5), NEW: Decimal(3)}, {OLD: "SYNA", NEW: "SYNB"})
    with pytest.raises(LifecycleError, match="successor asset alias mismatch"):
        apply_lifecycle_event(existing, replace(action, target_symbol="SYNZ"))


@pytest.mark.parametrize(
    "change",
    [
        {"event_id": ""},
        {"observed_at": datetime(2024, 1, 13)},
        {"page_hashes": ()},
        {"page_hashes": ("not-a-hash",)},
        {"normalization_version": 0},
        {"sequence": -1},
    ],
)
def test_event_provenance_and_order_must_be_explicit(change: dict) -> None:
    with pytest.raises(LifecycleError) as info:
        event("rename", target_symbol="SYNB", **change)
    assert info.value.code == "CORPORATE_ACTION_UNSUPPORTED"


@pytest.mark.parametrize(
    "positions, symbols",
    [
        ({OLD: Decimal(-1)}, {OLD: "SYNA"}),
        ({OLD: Decimal("NaN")}, {OLD: "SYNA"}),
        ({OLD: Decimal(1)}, {}),
        ({}, {OLD: "SYNA", OTHER: "SYNA"}),
    ],
)
def test_invalid_initial_state_is_rejected(positions: dict, symbols: dict) -> None:
    with pytest.raises(LifecycleError) as info:
        LifecycleState(positions, symbols)
    assert info.value.code == "CORPORATE_ACTION_UNSUPPORTED"


def test_invalid_cash_receivable_is_rejected() -> None:
    with pytest.raises(LifecycleError, match="invalid cash receivable"):
        LifecycleState({}, {}, (CashReceivable("event", PAY, Decimal("NaN")),))


def test_zero_quantity_does_not_require_a_price_mark() -> None:
    state = LifecycleState({OLD: Decimal(0)}, {OLD: "SYNA"})
    assert value_lifecycle_state(state, {}, Decimal(100), DAY) == Decimal(100)


def test_state_does_not_expose_mutable_position_or_alias_maps() -> None:
    before = holding()
    with pytest.raises(TypeError):
        setitem(before.positions, OLD, Decimal(9))
    with pytest.raises(TypeError):
        setitem(before.symbols, OLD, "SYNB")


@pytest.mark.parametrize(
    "kind, terms",
    [
        ("rename", {}),
        ("split", {"share_ratio": Decimal(2)}),
        ("split", {"share_ratio": Decimal("NaN"), "fraction_policy": "retain"}),
        ("split", {"share_ratio": Decimal(2), "fraction_policy": "guess"}),
        ("worthless_removal", {}),
        ("cash_merger", {"cash_per_old_share": Decimal(4)}),
        ("cash_merger", {"cash_per_old_share": Decimal(-1), "cash_pay_date": PAY}),
        ("cash_merger", {"cash_per_old_share": Decimal(1), "cash_pay_date": date(2024, 1, 9)}),
        (
            "stock_merger",
            {"target": OLD, "target_symbol": "SYNB", "share_ratio": Decimal(1), "fraction_policy": "retain"},
        ),
        (
            "stock_merger",
            {
                "target": AssetKey("other", "asset-new"),
                "target_symbol": "SYNB",
                "share_ratio": Decimal(1),
                "fraction_policy": "retain",
            },
        ),
        (
            "mixed_merger",
            {"target": NEW, "target_symbol": "SYNB", "share_ratio": Decimal(1), "fraction_policy": "retain"},
        ),
        ("rename", {"target_symbol": "SYNB", "fee_per_old_share": Decimal("0.01")}),
        ("rename", {"target_symbol": "SYNB", "fee_per_old_share": None}),
    ],
)
def test_incomplete_or_ambiguous_terms_fail_closed(kind: str, terms: dict) -> None:
    with pytest.raises(LifecycleError) as info:
        event(kind, **terms)
    assert info.value.code == "CORPORATE_ACTION_UNSUPPORTED"
