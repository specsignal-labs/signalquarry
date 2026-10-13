# SPDX-License-Identifier: Apache-2.0
"""Fail-closed contracts for the pure data evidence used by universe research."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from decimal import Decimal

import numpy as np
import pytest

from signalquarry._internal.data.asset_dataset import AssetDatasetError, AssetSeries
from signalquarry._internal.data.dataset import Dataset, truncated, with_pending_session
from signalquarry._internal.data.identity import AliasBook, AssetKey, IdentityError
from signalquarry._internal.data.lifecycle import LifecycleError

from .test_asset_dataset import CUTOFFS, NEW, OBSERVATION_CUTOFF, OLD, SESSIONS, dataset, merger
from .test_dataset_prefix import _inputs
from .test_lifecycle import event


def test_legacy_dataset_rejects_reordered_calendar_misaligned_bars_and_past_pending_session():
    earlier, _ = _inputs()
    with pytest.raises(ValueError, match="SESSIONS_NOT_STRICTLY_INCREASING"):
        Dataset(earlier.sessions[::-1], earlier.series)
    item = earlier.series["AAA"]
    with pytest.raises(ValueError, match="SERIES_LENGTH_MISMATCH"):
        Dataset(earlier.sessions, {"AAA": replace(item, present=item.present[:-1])})
    with pytest.raises(ValueError, match="SERIES_LENGTH_MISMATCH"):
        Dataset(
            earlier.sessions, {"AAA": replace(item, micro={**item.micro, "close": item.micro["close"][:-1]})}
        )
    with pytest.raises(ValueError, match="PENDING_SESSION_NOT_AFTER_END"):
        with_pending_session(earlier, earlier.sessions[-1])


@pytest.mark.parametrize("provider,asset_id", [("UPPER", "test"), ("synthetic", ""), ("synthetic", " test ")])
def test_invalid_asset_identifiers_are_refused(provider, asset_id):
    with pytest.raises(IdentityError, match="asset key"):
        AssetKey(provider, asset_id)


def test_alias_symbol_and_query_cutoff_must_be_valid():
    frozen = dataset()
    with pytest.raises(IdentityError, match="alias symbol"):
        replace(frozen.aliases.observations[0], symbol="lowercase")
    with pytest.raises(IdentityError, match="alias cutoff time"):
        frozen.aliases.as_of(SESSIONS[0], datetime(2024, 1, 2))


@pytest.mark.parametrize(
    "damage",
    ["fields", "float-price", "negative-price", "overflow", "volume", "presence", "length", "zero-price"],
)
def test_raw_asset_bars_reject_invalid_fields_units_and_missingness(damage):
    frozen = dataset().series[OLD]
    changes = {}
    if damage == "fields":
        changes["micro"] = {key: value for key, value in frozen.micro.items() if key != "open"}
    elif damage in ("float-price", "negative-price", "overflow", "zero-price"):
        values = {
            "float-price": np.array([1.0, 0, 0]),
            "negative-price": np.array([-1, 0, 0], dtype=np.int64),
            "overflow": np.array([2**63, 0, 0], dtype=np.uint64),
            "zero-price": np.array([0, 0, 0], dtype=np.int64),
        }[damage]
        changes["micro"] = {**frozen.micro, "close": values}
    elif damage == "volume":
        changes["volume"] = np.array([np.nan, 0, 0])
    elif damage == "presence":
        changes["present"] = np.array([1, 0, 0])
    else:
        changes["volume"] = np.array([1000.0])
    with pytest.raises(AssetDatasetError):
        replace(frozen, **changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"sessions": ()},
        {"sessions": SESSIONS[::-1]},
        {"source": " "},
        {"aliases": ()},
        {"decision_cutoffs": CUTOFFS[:1]},
        {"decision_cutoffs": (datetime(2024, 1, 2), *CUTOFFS[1:])},
        {"series": {}},
        {"series": {"SYNA": "raw"}},
        {"series": {OLD: "raw"}},
    ],
)
def test_asset_manifest_rejects_ambiguous_structure_and_cutoffs(changes):
    with pytest.raises(AssetDatasetError):
        replace(dataset(), **changes)


def test_asset_manifest_rejects_conflicting_alias_and_event_provenance():
    frozen = dataset()
    first = frozen.aliases.observations[0]
    for observations, message in (
        ((replace(first, asset=AssetKey("synthetic", "missing")),), "without raw series"),
        ((first, replace(first, asset=NEW)), "changed asset"),
        ((first, replace(first, symbol="SYNB")), "conflicting alias observations"),
    ):
        with pytest.raises(AssetDatasetError, match=message):
            replace(frozen, aliases=AliasBook(observations))
    with pytest.raises(AssetDatasetError, match="event observation exceeds"):
        replace(frozen, events=(replace(merger(), observed_at=OBSERVATION_CUTOFF.replace(day=5)),))
    with pytest.raises(AssetDatasetError, match="conflicting event versions"):
        replace(frozen, events=(merger(), replace(merger(), share_ratio=Decimal("0.4"))))


def test_asset_alias_revisions_are_order_independent_and_queries_enforce_frozen_cutoff():
    frozen = dataset()
    first = frozen.aliases.observations[0]
    revision = replace(first, symbol="SYNB", observed_at=OBSERVATION_CUTOFF)
    forward = replace(frozen, aliases=AliasBook((first, revision)))
    reversed_ = replace(frozen, aliases=AliasBook((revision, first)))
    assert forward.identity() == reversed_.identity()
    assert reversed_.aliases_as_of(SESSIONS[0], CUTOFFS[0]) == {"SYNA": OLD}
    assert reversed_.aliases_as_of(SESSIONS[0], OBSERVATION_CUTOFF) == {"SYNB": OLD}
    with pytest.raises(AssetDatasetError, match="unknown asset"):
        frozen.price(AssetKey("synthetic", "missing"), "close", 0)
    with pytest.raises(KeyError, match="FIELD_UNKNOWN"):
        frozen.series[OLD].price("adjusted", 0)
    for cutoff, message in (
        (datetime(2024, 1, 2), "timezone-aware"),
        (OBSERVATION_CUTOFF.replace(day=5), "exceeds manifest cutoff"),
    ):
        with pytest.raises(AssetDatasetError, match=message):
            frozen.events_as_of(cutoff)
        with pytest.raises(AssetDatasetError, match=message):
            frozen.aliases_as_of(SESSIONS[0], cutoff)


@pytest.mark.parametrize(
    "kind,terms",
    [
        ("unknown", {}),
        ("rename", {"target_symbol": "SYNB", "source_symbol": "lowercase"}),
        ("rename", {"target_symbol": "lowercase"}),
        ("rename", {"target_symbol": "SYNB", "target": NEW}),
        ("worthless_removal", {"cash_per_old_share": Decimal(0), "target": NEW}),
        ("stock_merger", {}),
        ("rename", {"target_symbol": "SYNB", "fraction_policy": "retain"}),
        ("rename", {"target_symbol": "SYNB", "cash_per_old_share": Decimal(1)}),
    ],
)
def test_incomplete_lifecycle_identity_and_consideration_fail_closed(kind, terms):
    with pytest.raises(LifecycleError):
        event(kind, **terms)


def test_asset_series_length_must_match_the_frozen_calendar():
    frozen = dataset()
    short = AssetSeries(
        {name: values[:1] for name, values in frozen.series[OLD].micro.items()},
        frozen.series[OLD].volume[:1],
        frozen.series[OLD].present[:1],
    )
    with pytest.raises(AssetDatasetError, match="series length"):
        replace(frozen, series={**frozen.series, OLD: short})
    # A truncated legacy dataset remains a valid, independent copy.
    earlier, _ = _inputs()
    assert truncated(earlier, 2).sessions == earlier.sessions[:2]
