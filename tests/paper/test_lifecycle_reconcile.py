# SPDX-License-Identifier: Apache-2.0
"""Synthetic paper activity proof for fully resolved corporate actions."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from signalquarry._internal.data.identity import AssetKey
from signalquarry._internal.data.lifecycle import LifecycleEvent
from signalquarry._internal.engine.lifecycle import LifecycleState
from signalquarry._internal.paper.lifecycle import (
    BrokerAssetPosition,
    CashActivity,
    PositionActivity,
    reconcile_cash_payment,
    reconcile_position_transition,
)
from signalquarry._internal.paper.models import PaperError

OLD = AssetKey("synthetic", "asset-old")
NEW = AssetKey("synthetic", "asset-new")
DAY = date(2024, 1, 10)
PAY = date(2024, 1, 15)


def action(kind: str, **terms) -> LifecycleEvent:
    return LifecycleEvent(
        **{
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
        | terms
    )


def before() -> LifecycleState:
    return LifecycleState({OLD: Decimal(5)}, {OLD: "SYNA"})


def position(asset: AssetKey, symbol: str, quantity: str) -> BrokerAssetPosition:
    return BrokerAssetPosition(asset, symbol, Decimal(quantity))


def activity(
    event: LifecycleEvent,
    asset: AssetKey,
    old_symbol: str | None,
    new_symbol: str | None,
    delta: str,
    *,
    activity_id: str = "activity-1",
) -> PositionActivity:
    return PositionActivity(activity_id, event.event_id, DAY, asset, old_symbol, new_symbol, Decimal(delta))


def code(fn) -> tuple[str, str]:
    with pytest.raises(PaperError) as info:
        fn()
    return info.value.code, info.value.status


def test_rename_requires_verified_activity_and_new_alias() -> None:
    event = action("rename", target_symbol="SYNB")
    proof = activity(event, OLD, "SYNA", "SYNB", "0")
    after = reconcile_position_transition(before(), event, [proof], [position(OLD, "SYNB", "5")])
    assert dict(after.positions) == {OLD: Decimal(5)}
    assert dict(after.symbols) == {OLD: "SYNB"}
    assert code(lambda: reconcile_position_transition(before(), event, [], [position(OLD, "SYNA", "5")])) == (
        "PAPER_CORPORATE_ACTION_PENDING",
        "busy",
    )
    assert code(lambda: reconcile_position_transition(before(), event, [], [position(OLD, "SYNB", "5")])) == (
        "PAPER_CORPORATE_ACTION_UNRECONCILED",
        "blocked",
    )
    assert code(
        lambda: reconcile_position_transition(before(), event, [proof], [position(NEW, "SYNB", "5")])
    ) == ("PAPER_CORPORATE_ACTION_UNRECONCILED", "blocked")


def test_reverse_split_requires_exact_fraction_and_asset_mapping() -> None:
    event = action("split", target_symbol="SYNB", share_ratio=Decimal("0.5"), fraction_policy="retain")
    proof = activity(event, OLD, "SYNA", "SYNB", "-2.5")
    after = reconcile_position_transition(before(), event, [proof], [position(OLD, "SYNB", "2.5")])
    assert dict(after.positions) == {OLD: Decimal("2.5")}
    assert code(
        lambda: reconcile_position_transition(
            before(), event, [replace(proof, quantity_delta=Decimal(-2))], [position(OLD, "SYNB", "2.5")]
        )
    ) == ("PAPER_CORPORATE_ACTION_UNRECONCILED", "blocked")
    assert code(
        lambda: reconcile_position_transition(
            before(), event, [replace(proof, symbol_after="SYNA")], [position(OLD, "SYNB", "2.5")]
        )
    ) == ("PAPER_CORPORATE_ACTION_UNRECONCILED", "blocked")


def test_duplicate_or_invalid_broker_positions_are_refused() -> None:
    event = action("rename", target_symbol="SYNB")
    proof = activity(event, OLD, "SYNA", "SYNB", "0")
    for observed in (
        [position(OLD, "SYNB", "5"), position(OLD, "SYNB", "5")],
        [position(OLD, "SYNB", "0")],
        [position(OLD, "SYNB", "NaN")],
    ):
        assert code(
            lambda observed=observed: reconcile_position_transition(before(), event, [proof], observed)
        ) == (
            "PAPER_CORPORATE_ACTION_UNRECONCILED",
            "blocked",
        )


def test_stock_merger_proves_both_asset_deltas_and_reused_symbol() -> None:
    event = action(
        "stock_merger",
        target=NEW,
        target_symbol="SYNA",
        share_ratio=Decimal("0.4"),
        fraction_policy="reject_noninteger",
    )
    source = activity(event, OLD, "SYNA", None, "-5", activity_id="source")
    target = activity(event, NEW, None, "SYNA", "2", activity_id="target")
    after = reconcile_position_transition(before(), event, [target, source], [position(NEW, "SYNA", "2")])
    assert dict(after.positions) == {NEW: Decimal(2)}
    assert code(
        lambda: reconcile_position_transition(before(), event, [source], [position(NEW, "SYNA", "2")])
    ) == ("PAPER_CORPORATE_ACTION_UNRECONCILED", "blocked")
    assert code(
        lambda: reconcile_position_transition(
            before(), event, [source, replace(target, activity_id="source")], [position(NEW, "SYNA", "2")]
        )
    ) == ("PAPER_CORPORATE_ACTION_UNRECONCILED", "blocked")


def test_cash_merger_requires_share_removal_then_exact_dated_cash_credit() -> None:
    event = action("cash_merger", cash_per_old_share=Decimal("12.34"), cash_pay_date=PAY)
    share_removal = activity(event, OLD, "SYNA", None, "-5")
    after = reconcile_position_transition(before(), event, [share_removal], [])
    assert len(after.receivables) == 1 and after.receivables[0].amount == Decimal("61.70")
    assert code(
        lambda: reconcile_cash_payment(
            after, event.event_id, None, session=PAY, cash_before=Decimal(100), cash_after=Decimal(100)
        )
    ) == ("PAPER_CORPORATE_ACTION_PENDING", "busy")
    payment = CashActivity("cash-1", event.event_id, PAY, Decimal("61.70"))
    assert code(
        lambda: reconcile_cash_payment(
            after,
            event.event_id,
            None,
            session=DAY,
            cash_before=Decimal(100),
            cash_after=Decimal(100),
        )
    ) == ("PAPER_CORPORATE_ACTION_PENDING", "busy")
    assert code(
        lambda: reconcile_cash_payment(
            after,
            event.event_id,
            payment,
            session=DAY,
            cash_before=Decimal(100),
            cash_after=Decimal("161.70"),
        )
    ) == ("PAPER_CORPORATE_ACTION_UNRECONCILED", "blocked")
    assert code(
        lambda: reconcile_cash_payment(
            after,
            event.event_id,
            replace(payment, amount=Decimal("61.71")),
            session=PAY,
            cash_before=Decimal(100),
            cash_after=Decimal("161.70"),
        )
    ) == ("PAPER_CORPORATE_ACTION_UNRECONCILED", "blocked")
    assert code(
        lambda: reconcile_cash_payment(
            after,
            event.event_id,
            payment,
            session=PAY,
            cash_before=Decimal("NaN"),
            cash_after=Decimal("161.70"),
        )
    ) == ("PAPER_CORPORATE_ACTION_UNRECONCILED", "blocked")
    assert code(
        lambda: reconcile_cash_payment(
            after,
            event.event_id,
            payment,
            session=PAY,
            cash_before=Decimal(100),
            cash_after=Decimal("161.69"),
        )
    ) == ("PAPER_CORPORATE_ACTION_UNRECONCILED", "blocked")
    settled, cash = reconcile_cash_payment(
        after, event.event_id, payment, session=PAY, cash_before=Decimal(100), cash_after=Decimal("161.70")
    )
    assert settled.receivables == () and cash == Decimal("161.70")
    _, debit_cash = reconcile_cash_payment(
        after, event.event_id, payment, session=PAY, cash_before=Decimal(-100), cash_after=Decimal("-38.30")
    )
    assert debit_cash == Decimal("-38.30")
    assert code(
        lambda: reconcile_cash_payment(
            settled, event.event_id, payment, session=PAY, cash_before=cash, cash_after=cash
        )
    ) == ("PAPER_CORPORATE_ACTION_UNRECONCILED", "blocked")


def test_mixed_merger_preserves_other_holding_and_separate_receivable() -> None:
    event = action(
        "mixed_merger",
        target=NEW,
        target_symbol="SYNB",
        share_ratio=Decimal("0.4"),
        fraction_policy="reject_noninteger",
        cash_per_old_share=Decimal("3.25"),
        cash_pay_date=PAY,
    )
    prior = LifecycleState({OLD: Decimal(5), NEW: Decimal(3)}, {OLD: "SYNA", NEW: "SYNB"})
    after = reconcile_position_transition(
        prior,
        event,
        [
            activity(event, OLD, "SYNA", None, "-5", activity_id="source"),
            activity(event, NEW, "SYNB", "SYNB", "2", activity_id="target"),
        ],
        [position(NEW, "SYNB", "5")],
    )
    assert dict(after.positions) == {NEW: Decimal(5)}
    assert after.receivables[0].amount == Decimal("16.25")


def test_zero_holding_needs_no_broker_activity() -> None:
    event = action("worthless_removal", cash_per_old_share=Decimal(0))
    empty = LifecycleState({}, {OLD: "SYNA"})
    after = reconcile_position_transition(empty, event, [], [])
    assert dict(after.positions) == {} and after.receivables == ()


def test_worthless_removal_requires_broker_evidence_for_held_shares() -> None:
    event = action("worthless_removal", cash_per_old_share=Decimal(0))
    proof = activity(event, OLD, "SYNA", None, "-5")
    after = reconcile_position_transition(before(), event, [proof], [])
    assert dict(after.positions) == {} and after.receivables == ()
    assert code(lambda: reconcile_position_transition(before(), event, [], [])) == (
        "PAPER_CORPORATE_ACTION_UNRECONCILED",
        "blocked",
    )


@pytest.mark.parametrize(
    "change",
    [
        {"event_id": "different"},
        {"occurred": date(2024, 1, 11)},
        {"quantity_delta": Decimal("NaN")},
        {"activity_id": ""},
    ],
)
def test_unrelated_or_invalid_activity_is_refused(change: dict) -> None:
    event = action("worthless_removal", cash_per_old_share=Decimal(0))
    proof = activity(event, OLD, "SYNA", None, "-5")
    assert code(lambda: reconcile_position_transition(before(), event, [replace(proof, **change)], [])) == (
        "PAPER_CORPORATE_ACTION_UNRECONCILED",
        "blocked",
    )
