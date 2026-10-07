# SPDX-License-Identifier: Apache-2.0
"""Both sides of every cutoff, age, price and rank limit in the point-in-time universe build."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

import numpy as np
import pytest

from signalquarry._internal.canonical import hash_without
from signalquarry._internal.data.library import LibraryError
from signalquarry._internal.data.universe_build import (
    DAILY_BAR_SAFE_AFTER,
    NEW_YORK,
    UniverseBuildPolicy,
    make_universe_manifest,
)

from .test_universe_build import DECISION_SESSION, KNOWN_AT, OBSERVED_AT, _inputs

_POLICY = UniverseBuildPolicy(Decimal("10"), 30, Decimal("50"))


def _rehash(manifest: dict[str, Any]) -> dict[str, Any]:
    manifest["manifest_hash"] = hash_without(manifest, "manifest_hash")
    return manifest


def _call(
    *,
    known_at: datetime = KNOWN_AT,
    decision_session: date = DECISION_SESSION,
    policy: UniverseBuildPolicy = _POLICY,
    edit: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    assets, asset_manifest, classification, dataset_manifest, panel = _inputs()
    parts = {
        "assets": assets,
        "asset_manifest": asset_manifest,
        "classification_snapshot": classification,
        "dataset_manifest": dataset_manifest,
        "panel": panel,
    }
    if edit is not None:
        edit(parts)
    return make_universe_manifest(
        **parts, decision_session=decision_session, known_at=known_at, policy=policy
    )


def _asset_observed(moment: datetime) -> Callable[[dict[str, Any]], None]:
    def edit(parts: dict[str, Any]) -> None:
        parts["asset_manifest"]["observed_at"] = moment.isoformat().replace("+00:00", "Z")
        _rehash(parts["asset_manifest"])

    return edit


def _dataset(**changes: object) -> Callable[[dict[str, Any]], None]:
    def edit(parts: dict[str, Any]) -> None:
        parts["dataset_manifest"].update(changes)
        _rehash(parts["dataset_manifest"])

    return edit


def _stamp(moment: datetime) -> str:
    return moment.isoformat().replace("+00:00", "Z")


def _last_bar_available() -> datetime:
    session = _inputs()[4].sessions[-1]
    return datetime.combine(session, DAILY_BAR_SAFE_AFTER, tzinfo=NEW_YORK).astimezone(UTC)


def test_the_cutoff_must_be_timezone_aware_and_fall_before_the_session_date_in_utc() -> None:
    with pytest.raises(LibraryError, match="^DATA_MANIFEST_INVALID:decision cutoff must precede session$"):
        _call(known_at=datetime(2026, 9, 28, 21))
    with pytest.raises(LibraryError, match="^DATA_MANIFEST_INVALID:decision cutoff must precede session$"):
        _call(known_at=datetime(2026, 9, 29, 0, 0, tzinfo=UTC))
    # 22:00 at UTC-5 on 9/28 is 03:00 UTC on 9/29: the UTC date is the session date.
    with pytest.raises(LibraryError, match="^DATA_MANIFEST_INVALID:decision cutoff must precede session$"):
        _call(known_at=datetime(2026, 9, 28, 22, tzinfo=timezone(timedelta(hours=-5))))
    # 23:59:59 UTC on the previous date is still allowed (the inputs were all gathered earlier)
    ok = _call(known_at=datetime(2026, 9, 28, 23, 59, 59, tzinfo=UTC))
    assert ok["known_at"] == "2026-09-28T23:59:59Z"
    # a non-UTC cutoff is recorded in UTC
    zoned = _call(known_at=datetime(2026, 9, 28, 17, tzinfo=timezone(timedelta(hours=-4))))
    assert zoned["known_at"] == "2026-09-28T21:00:00Z"


def test_the_asset_snapshot_may_be_observed_up_to_the_cutoff_and_at_most_31_days_earlier() -> None:
    detail = "^UNIVERSE_INPUT_UNAVAILABLE:asset snapshot outside cutoff$"
    assert _call(edit=_asset_observed(KNOWN_AT))
    with pytest.raises(LibraryError, match=detail):
        _call(edit=_asset_observed(KNOWN_AT + timedelta(seconds=1)))
    assert _call(edit=_asset_observed(KNOWN_AT - timedelta(days=31)))
    with pytest.raises(LibraryError, match=detail):
        _call(edit=_asset_observed(KNOWN_AT - timedelta(days=31, seconds=1)))


def test_a_malformed_or_tampered_asset_snapshot_is_rejected_with_its_detail() -> None:
    def tampered(parts: dict[str, Any]) -> None:
        parts["asset_manifest"]["asset_count"] += 1

    def no_hash(parts: dict[str, Any]) -> None:
        parts["asset_manifest"]["manifest_hash"] = 7

    def short_hash(parts: dict[str, Any]) -> None:
        parts["asset_manifest"]["manifest_hash"] = "sha256:abc"

    for edit in (tampered, no_hash, short_hash):
        with pytest.raises(LibraryError, match="^DATA_MANIFEST_INVALID:asset snapshot identity$"):
            _call(edit=edit)
    with pytest.raises(LibraryError, match="^DATA_MANIFEST_INVALID"):
        _call(edit=lambda parts: parts["asset_manifest"].update(observed_at="not a time"))


def test_the_dataset_must_be_fetched_by_the_cutoff_and_after_the_last_daily_bar_is_final() -> None:
    assert _call(edit=_dataset(fetched_at=_stamp(KNOWN_AT)))
    with pytest.raises(LibraryError, match="^UNIVERSE_INPUT_UNAVAILABLE:dataset fetched after cutoff$"):
        _call(edit=_dataset(fetched_at=_stamp(KNOWN_AT + timedelta(seconds=1))))
    ready = _last_bar_available()
    assert ready == datetime(2026, 9, 28, 20, 15, tzinfo=UTC)
    assert _call(edit=_dataset(fetched_at=_stamp(ready)))
    with pytest.raises(
        LibraryError, match="^UNIVERSE_INPUT_UNAVAILABLE:latest daily bar fetched before session completion$"
    ):
        _call(edit=_dataset(fetched_at=_stamp(ready - timedelta(seconds=1))))
    with pytest.raises(
        LibraryError, match="^DATA_MANIFEST_INVALID:dataset fetched_at$|^DATA_MANIFEST_INVALID"
    ):
        _call(edit=_dataset(fetched_at="never"))


def test_the_dataset_must_end_before_the_session_and_by_the_cutoff_date() -> None:
    detail = "^UNIVERSE_INPUT_UNAVAILABLE:dataset includes sessions after cutoff$"
    with pytest.raises(LibraryError, match=detail):
        _call(edit=_dataset(end="2026-09-29"))
    with pytest.raises(LibraryError, match=detail):
        _call(edit=_dataset(end="2026-09-30"))
    # ends on the cutoff's own date: allowed
    assert _call(edit=_dataset(end="2026-09-28"))
    # a cutoff earlier than the dataset end, with the session still later than both
    early = datetime(2026, 9, 27, 10, tzinfo=UTC)

    def earlier_inputs(parts: dict[str, Any]) -> None:
        _asset_observed(early)(parts)
        parts["classification_snapshot"]["observed_at"] = _stamp(early)
        _dataset(fetched_at="2026-09-27T12:00:00Z", end="2026-09-28")(parts)

    with pytest.raises(LibraryError, match=detail):
        _call(known_at=datetime(2026, 9, 27, 21, tzinfo=UTC), edit=earlier_inputs)
    for bad in ("2026-13-45", None, 5):
        with pytest.raises(LibraryError, match="^DATA_MANIFEST_INVALID:dataset end$"):
            _call(edit=_dataset(end=bad))


@pytest.mark.parametrize(
    "changes",
    [
        {"schema": "other"},
        {"provider": "other"},
        {"adjustment": "split"},
        {"symbols": "AAA"},
        {"symbols": ["AAA", ""]},
        {"symbols": ["AAA", 3]},
        {"symbols": ["BBB", "AAA"]},
        {"symbols": ["AAA", "AAA"]},
        {"dataset_identity": 5},
        {"dataset_identity": "sha256:xyz"},
        {"dataset_identity": "sha256:" + "b" * 64},
    ],
)
def test_dataset_manifest_identity_checks_reject_each_field(changes: dict[str, object]) -> None:
    with pytest.raises(LibraryError, match="^DATA_MANIFEST_INVALID:dataset identity$"):
        _call(edit=_dataset(**changes))


def test_the_dataset_manifest_hash_must_be_a_matching_sha256() -> None:
    def broken(manifest_hash: object) -> Callable[[dict[str, Any]], None]:
        def edit(parts: dict[str, Any]) -> None:
            parts["dataset_manifest"]["manifest_hash"] = manifest_hash

        return edit

    for value in (5, "sha256:abc", "sha256:" + "0" * 64):
        with pytest.raises(LibraryError, match="^DATA_MANIFEST_INVALID:dataset identity$"):
            _call(edit=broken(value))


def _panel_with(parts: dict[str, Any], **changes: object) -> None:
    parts["panel"] = replace(parts["panel"], **changes)


def test_the_panel_must_hold_twenty_complete_sessions_ending_before_the_decision() -> None:
    detail = "^UNIVERSE_INPUT_UNAVAILABLE:20 complete prior sessions are required$"

    def sessions(values: tuple[date, ...]) -> Callable[[dict[str, Any]], None]:
        return lambda parts: _panel_with(parts, sessions=values)

    base = _inputs()[4].sessions
    with pytest.raises(LibraryError, match=detail):
        _call(edit=sessions(base[:-1]))
    with pytest.raises(LibraryError, match=detail):
        _call(edit=sessions((*base[:-1], DECISION_SESSION)))
    with pytest.raises(LibraryError, match=detail):
        _call(edit=sessions((*base[:-1], DECISION_SESSION + timedelta(days=1))))


def test_panel_shapes_and_symbols_are_checked_against_the_lookback() -> None:
    detail = "^DATA_PANEL_INVALID:presence shape$"
    panel = _inputs()[4]

    def volume(parts: dict[str, Any]) -> None:
        _panel_with(parts, volume=panel.volume[:, :-1])

    def present(parts: dict[str, Any]) -> None:
        _panel_with(parts, present=panel.present[:, :-1])

    def close(parts: dict[str, Any]) -> None:
        _panel_with(parts, micro={"close": panel.micro["close"][:, :-1]})

    def duplicate(parts: dict[str, Any]) -> None:
        _panel_with(parts, symbols=(panel.symbols[0], *panel.symbols[:-1]))

    for edit in (volume, present, close, duplicate):
        with pytest.raises(LibraryError, match=detail):
            _call(edit=edit)
    with pytest.raises(LibraryError, match="^DATA_PANEL_INVALID:panel symbols differ from dataset manifest$"):
        _call(edit=lambda parts: _panel_with(parts, symbols=(*panel.symbols[:-1], "OTHER")))


def test_listing_age_is_inclusive_at_the_minimum_and_a_future_listing_is_an_error() -> None:
    def listed(days_before: int) -> Callable[[dict[str, Any]], None]:
        def edit(parts: dict[str, Any]) -> None:
            for record in parts["classification_snapshot"]["records"]:
                if record["listed_at"] == "2026-09-20":
                    record["listed_at"] = (DECISION_SESSION - timedelta(days=days_before)).isoformat()

        return edit

    young = _call(edit=listed(29))
    assert young["counts"]["excluded_listing_age"] == 1
    exactly = _call(edit=listed(30))
    assert exactly["counts"]["excluded_listing_age"] == 0
    assert exactly["counts"]["dollar_volume_ranked"] == 3
    same_day = _call(edit=listed(0))
    assert same_day["counts"]["excluded_listing_age"] == 1
    with pytest.raises(LibraryError, match="^UNIVERSE_CLASSIFICATION_INVALID:listing date after decision$"):
        _call(edit=listed(-1))


def _priced(prices: dict[str, int]) -> Callable[[dict[str, Any]], None]:
    def edit(parts: dict[str, Any]) -> None:
        panel = parts["panel"]
        close = panel.micro["close"].copy()
        for symbol, micro in prices.items():
            close[:, panel.symbols.index(symbol)] = micro
        parts["panel"] = replace(panel, micro={"close": close})

    return edit


def test_the_price_floor_is_inclusive_and_uses_the_latest_close() -> None:
    at_floor = _call(edit=_priced({"CHEAP": 10_000_000}))
    assert at_floor["counts"]["excluded_price"] == 0
    below = _call(edit=_priced({"CHEAP": 9_999_999}))
    assert below["counts"]["excluded_price"] == 1

    def drop_last_close(parts: dict[str, Any]) -> None:
        panel = parts["panel"]
        close = panel.micro["close"].copy()
        close[-1, panel.symbols.index("AAA")] = 1_000_000  # only the latest close is below the floor
        parts["panel"] = replace(panel, micro={"close": close})

    assert _call(edit=drop_last_close)["counts"]["excluded_price"] == 2

    # an earlier low close does not matter, only that every close is positive
    def early_low(parts: dict[str, Any]) -> None:
        panel = parts["panel"]
        close = panel.micro["close"].copy()
        close[0, panel.symbols.index("AAA")] = 1_000_000
        parts["panel"] = replace(panel, micro={"close": close})

    assert _call(edit=early_low)["counts"]["excluded_price"] == 1


def test_a_nonpositive_close_or_gap_counts_as_missing_history_not_price() -> None:
    def zero_close(parts: dict[str, Any]) -> None:
        panel = parts["panel"]
        close = panel.micro["close"].copy()
        close[5, panel.symbols.index("AAA")] = 0
        parts["panel"] = replace(panel, micro={"close": close})

    result = _call(edit=zero_close)
    assert result["counts"]["excluded_history"] == 1
    assert "AAA" not in result["member_symbols"]

    def smallest_valid(parts: dict[str, Any]) -> None:
        panel = parts["panel"]
        close = panel.micro["close"].copy()
        close[5, panel.symbols.index("AAA")] = 1
        parts["panel"] = replace(panel, micro={"close": close})

    assert _call(edit=smallest_valid)["counts"]["excluded_history"] == 0

    def gap(parts: dict[str, Any]) -> None:
        panel = parts["panel"]
        present = panel.present.copy()
        present[3, panel.symbols.index("BBB")] = False
        parts["panel"] = replace(panel, present=present)

    assert _call(edit=gap)["counts"]["excluded_history"] == 1

    def absent(parts: dict[str, Any]) -> None:
        panel = parts["panel"]
        keep = [i for i, symbol in enumerate(panel.symbols) if symbol != "BBB"]
        parts["panel"] = replace(
            panel,
            symbols=tuple(panel.symbols[i] for i in keep),
            micro={"close": panel.micro["close"][:, keep]},
            volume=panel.volume[:, keep],
            present=panel.present[:, keep],
        )
        parts["dataset_manifest"]["symbols"] = [panel.symbols[i] for i in keep]
        _rehash(parts["dataset_manifest"])

    assert _call(edit=absent)["counts"]["excluded_history"] == 1


def _volumes(
    volumes: dict[str, float], prices: dict[str, int] | None = None
) -> Callable[[dict[str, Any]], None]:
    def edit(parts: dict[str, Any]) -> None:
        panel = parts["panel"]
        volume = panel.volume.copy()
        for symbol, value in volumes.items():
            volume[:, panel.symbols.index(symbol)] = value
        close = panel.micro["close"].copy()
        for symbol, micro in (prices or {}).items():
            close[:, panel.symbols.index(symbol)] = micro
        parts["panel"] = replace(panel, volume=volume, micro={"close": close})
        for record in parts["classification_snapshot"]["records"]:
            record["listed_at"] = record["listed_at"] and "2020-01-01"

    return edit


def test_percentiles_use_average_ranks_over_the_price_and_history_cohort() -> None:
    # Three eligible common stocks (AAA, BBB, CHEAP once priced) with distinct median dollar volume.
    edit = _volumes({"AAA": 10.0, "CHEAP": 20.0, "BBB": 20.0, "NEW": 20.0}, {"CHEAP": 20_000_000})
    ranked = [
        _call(policy=replace(_POLICY, minimum_dollar_volume_percentile=Decimal(level)), edit=edit)
        for level in (0, 25, 50, 75, 100)
    ]
    # median dollar volumes 200 < 400 < 600 < 800: percentiles 0, 33.3, 66.7, 100
    assert [item["counts"]["dollar_volume_ranked"] for item in ranked] == [4] * 5
    assert [item["counts"]["members"] for item in ranked] == [4, 3, 2, 1, 1]
    assert ranked[4]["member_symbols"] == ["NEW"]
    assert ranked[2]["member_symbols"] == ["BBB", "NEW"]
    assert ranked[0]["counts"]["excluded_dollar_volume_rank"] == 0
    assert ranked[3]["counts"]["excluded_dollar_volume_rank"] == 3


def test_ties_share_the_average_percentile_and_the_threshold_is_inclusive() -> None:
    # AAA and CHEAP tie at 400; BBB (600) and NEW (800) are above. The tied pair sits at (0 + 0.5) * 100 / 3
    edit = _volumes({"AAA": 20.0, "CHEAP": 20.0, "BBB": 20.0, "NEW": 20.0}, {"CHEAP": 20_000_000})
    threshold = (Decimal(100) * Decimal("0.5")) / Decimal(3)
    at = _call(policy=replace(_POLICY, minimum_dollar_volume_percentile=threshold), edit=edit)
    assert at["member_symbols"] == ["AAA", "BBB", "CHEAP", "NEW"]
    above = _call(
        policy=replace(_POLICY, minimum_dollar_volume_percentile=threshold + Decimal("0.0001")), edit=edit
    )
    assert above["member_symbols"] == ["BBB", "NEW"]


def test_a_single_ranked_name_is_the_hundredth_percentile() -> None:
    result = _call(policy=replace(_POLICY, minimum_dollar_volume_percentile=Decimal(100)))
    assert result["member_symbols"] == ["AAA"]
    assert result["counts"]["dollar_volume_ranked"] == 2  # AAA and BBB reach the ranking
    only_one = _call(
        policy=replace(_POLICY, minimum_dollar_volume_percentile=Decimal(100)),
        edit=_priced({"BBB": 5_000_000}),
    )
    assert only_one["counts"]["dollar_volume_ranked"] == 1
    assert only_one["member_symbols"] == ["AAA"]


def test_the_manifest_records_its_inputs_policy_and_hashes() -> None:
    result = _call()
    assets, asset_manifest, classification, dataset_manifest, panel = _inputs()
    assert result["schema"] == "signalquarry.point-in-time-universe/v1"
    assert result["decision_session"] == "2026-09-29"
    assert result["asset_snapshot_id"] == asset_manifest["snapshot_id"]
    assert result["asset_snapshot_hash"] == asset_manifest["manifest_hash"]
    assert result["dataset_id"] == dataset_manifest["dataset_id"]
    assert result["dataset_identity"] == dataset_manifest["dataset_identity"]
    assert result["dataset_manifest_hash"] == dataset_manifest["manifest_hash"]
    assert result["lookback_sessions"] == 20
    assert result["policy"] == {
        "minimum_price": "10",
        "minimum_listing_age_days": 30,
        "minimum_dollar_volume_percentile": "50",
    }
    assert result["redistributable"] is False
    assert result["classification_snapshot_hash"].startswith("sha256:")
    assert result["members_hash"].startswith("sha256:")
    assert OBSERVED_AT < KNOWN_AT
    assert np.isfinite(panel.volume).all() and len(assets) == 5 and classification["records"]
