# SPDX-License-Identifier: Apache-2.0
"""Synthetic paper activity proof for fully resolved corporate actions."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import cast

import pytest

from signalquarry._internal.data.identity import AssetKey
from signalquarry._internal.data.lifecycle import LifecycleEvent
from signalquarry._internal.engine.lifecycle import LifecycleState
from signalquarry._internal.paper.lifecycle import (
    BrokerAssetPosition,
    CashActivity,
    PositionActivity,
    _positions,
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


def test_broker_position_rows_require_decimal_positive_quantities_and_unique_aliases() -> None:
    for invalid_quantity in (Decimal(0), cast(Decimal, 5.0), cast(Decimal, "5")):
        assert code(
            lambda invalid_quantity=invalid_quantity: _positions(
                [BrokerAssetPosition(OLD, "SYNA", invalid_quantity)]
            )
        ) == (
            "PAPER_CORPORATE_ACTION_UNRECONCILED",
            "blocked",
        )

    with pytest.raises(PaperError) as duplicate_alias:
        _positions([position(OLD, "SYNA", "5"), position(NEW, "SYNA", "2")])
    assert duplicate_alias.value.code == "PAPER_CORPORATE_ACTION_UNRECONCILED"


def test_duplicate_asset_rows_are_rejected_even_when_the_last_row_matches() -> None:
    event = action("rename", target_symbol="SYNB")
    proof = activity(event, OLD, "SYNA", "SYNB", "0")
    assert code(
        lambda: reconcile_position_transition(
            before(),
            event,
            [proof],
            [position(OLD, "SYNA", "5"), position(OLD, "SYNB", "5")],
        )
    ) == ("PAPER_CORPORATE_ACTION_UNRECONCILED", "blocked")


def test_single_share_holding_survives_a_verified_rename() -> None:
    event = action("rename", target_symbol="SYNB")
    one_share = LifecycleState({OLD: Decimal(1)}, {OLD: "SYNA"})
    proof = activity(event, OLD, "SYNA", "SYNB", "0")
    after = reconcile_position_transition(one_share, event, [proof], [position(OLD, "SYNB", "1")])
    assert dict(after.positions) == {OLD: Decimal(1)}


def test_multiple_position_activities_for_one_asset_are_summed() -> None:
    event = action("split", share_ratio=Decimal("0.6"), fraction_policy="retain")
    proofs = [
        activity(event, OLD, "SYNA", "SYNA", "-1", activity_id="activity-a"),
        activity(event, OLD, "SYNA", "SYNA", "-1", activity_id="activity-b"),
    ]
    after = reconcile_position_transition(before(), event, proofs, [position(OLD, "SYNA", "3")])
    assert dict(after.positions) == {OLD: Decimal(3)}


def test_one_to_one_split_with_unchanged_alias_needs_no_position_activity() -> None:
    event = action("split", share_ratio=Decimal(1), fraction_policy="retain")
    after = reconcile_position_transition(before(), event, [], [position(OLD, "SYNA", "5")])
    assert dict(after.positions) == {OLD: Decimal(5)}


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
    assert settled.applied_event_ids == after.applied_event_ids
    assert settled.last_event_key == after.last_event_key
    _, debit_cash = reconcile_cash_payment(
        after, event.event_id, payment, session=PAY, cash_before=Decimal(-100), cash_after=Decimal("-38.30")
    )
    assert debit_cash == Decimal("-38.30")
    assert code(
        lambda: reconcile_cash_payment(
            settled, event.event_id, payment, session=PAY, cash_before=cash, cash_after=cash
        )
    ) == ("PAPER_CORPORATE_ACTION_UNRECONCILED", "blocked")


def cash_receivable_state() -> tuple[LifecycleEvent, LifecycleState, CashActivity]:
    event = action("cash_merger", cash_per_old_share=Decimal("12.34"), cash_pay_date=PAY)
    after = reconcile_position_transition(before(), event, [activity(event, OLD, "SYNA", None, "-5")], [])
    payment = CashActivity("cash-1", event.event_id, PAY, Decimal("61.70"))
    return event, after, payment


def test_cash_payment_rejects_non_decimal_balances_and_activity_amounts() -> None:
    event, after, payment = cash_receivable_state()
    for invalid_balance in (cast(Decimal, 100.0), cast(Decimal, "100")):
        assert code(
            lambda invalid_balance=invalid_balance: reconcile_cash_payment(
                after,
                event.event_id,
                payment,
                session=PAY,
                cash_before=invalid_balance,
                cash_after=Decimal("161.70"),
            )
        ) == ("PAPER_CORPORATE_ACTION_UNRECONCILED", "blocked")
    assert code(
        lambda: reconcile_cash_payment(
            after,
            event.event_id,
            replace(payment, amount=cast(Decimal, 61.70)),
            session=PAY,
            cash_before=Decimal(100),
            cash_after=Decimal("161.70"),
        )
    ) == ("PAPER_CORPORATE_ACTION_UNRECONCILED", "blocked")


def test_cash_payment_requires_activity_identity_and_exact_credit_timing() -> None:
    event, after, payment = cash_receivable_state()
    for invalid in (
        replace(payment, activity_id=""),
        replace(payment, event_id="different"),
        replace(payment, occurred=DAY),
    ):
        assert code(
            lambda invalid=invalid: reconcile_cash_payment(
                after,
                event.event_id,
                invalid,
                session=PAY,
                cash_before=Decimal(100),
                cash_after=Decimal("161.70"),
            )
        ) == ("PAPER_CORPORATE_ACTION_UNRECONCILED", "blocked")


def test_cash_payment_distinguishes_pending_from_unreconciled_cash() -> None:
    event, after, payment = cash_receivable_state()
    assert code(
        lambda: reconcile_cash_payment(
            after,
            event.event_id,
            payment,
            session=PAY,
            cash_before=Decimal(100),
            cash_after=Decimal(100),
        )
    ) == ("PAPER_CORPORATE_ACTION_UNRECONCILED", "blocked")
    assert code(
        lambda: reconcile_cash_payment(
            after,
            event.event_id,
            None,
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
            session=DAY,
            cash_before=Decimal(100),
            cash_after=Decimal(100),
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


def test_zero_holding_rename_needs_no_broker_activity() -> None:
    event = action("rename", target_symbol="SYNB")
    empty = LifecycleState({OLD: Decimal(0)}, {OLD: "SYNA"})
    after = reconcile_position_transition(empty, event, [], [])
    assert dict(after.positions) == {OLD: Decimal(0)}
    assert dict(after.symbols) == {OLD: "SYNB"}


def test_explicit_zero_holding_can_be_removed_without_broker_activity() -> None:
    event = action("worthless_removal", cash_per_old_share=Decimal(0))
    zero_holding = LifecycleState({OLD: Decimal(0)}, {OLD: "SYNA"})
    after = reconcile_position_transition(zero_holding, event, [], [])
    assert dict(after.positions) == {}
    assert after.receivables == ()


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
        {"quantity_delta": cast(Decimal, 1.0)},
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
